// SPDX-License-Identifier: Apache-2.0

package llmproxy

import (
	"encoding/json"
	"strconv"
	"strings"
	"time"

	"github.com/cespare/xxhash/v2"

	"github.com/rostamlabs/rostam/semcache"
)

// chatRequest is the subset of the OpenAI chat-completions request body the
// proxy needs to judge cacheability and derive a cache key. Raw holds the
// original body so it can be forwarded verbatim on a miss or passthrough —
// the proxy never needs to re-encode a request it didn't answer itself.
type chatRequest struct {
	Model       string        `json:"model"`
	Messages    []chatMessage `json:"messages"`
	Temperature *float64      `json:"temperature,omitempty"` // nil => 1.0 (OpenAI default)
	MaxTokens   int           `json:"max_tokens,omitempty"`
	N           int           `json:"n,omitempty"`
	Stream      bool          `json:"stream,omitempty"`
	// StreamOptions (e.g. {"include_usage":true}) changes the SHAPE of the
	// stream the client expects, which a cache replay cannot reproduce: the
	// replay emits a fixed three-chunk stream with no usage chunk. Present
	// (and non-null) => uncacheable passthrough.
	StreamOptions json.RawMessage `json:"stream_options,omitempty"`
	Tools         []chatTool      `json:"tools,omitempty"`
	Raw           json.RawMessage `json:"-"` // original body, forwarded verbatim on miss/passthrough
}

// chatMessage holds Content undecoded because OpenAI allows either a plain
// string or an array of typed parts (text/image/etc). Only the string form is
// cacheable — an array means we cannot safely reduce the message to text.
type chatMessage struct {
	Role    string          `json:"role"`
	Content json.RawMessage `json:"content"` // string OR array; array => uncacheable
}

// chatTool is the minimal shape needed to pull a tool name out of the
// request's tool list; everything else in the tool definition is irrelevant
// to cache scoping.
type chatTool struct {
	Function struct {
		Name string `json:"name"`
	} `json:"function"`
}

// chatResponse is decode-tolerant on purpose: it captures just enough of an
// upstream response to decide whether the call is cacheable and to store the
// answer, without choking on fields OpenAI adds over time.
type chatResponse struct {
	Choices []struct {
		Message struct {
			Content   string          `json:"content"`
			ToolCalls json.RawMessage `json:"tool_calls,omitempty"`
		} `json:"message"`
		FinishReason string `json:"finish_reason"`
	} `json:"choices"`
	Usage struct {
		CompletionTokens int `json:"completion_tokens"`
	} `json:"usage"`
}

// parseChatRequest decodes a chat-completions request body and stashes the
// original bytes in Raw for verbatim forwarding.
func parseChatRequest(body []byte) (*chatRequest, error) {
	var req chatRequest
	if err := json.Unmarshal(body, &req); err != nil {
		return nil, err
	}
	req.Raw = body
	return &req, nil
}

// temperature returns the effective sampling temperature, applying OpenAI's
// default of 1.0 when the client omitted the field. A caller cannot
// distinguish "not sent" from "sent as 0" without the pointer, and the two
// mean different things for cache scoping.
func (r *chatRequest) temperature() float64 {
	if r.Temperature == nil {
		return 1.0
	}
	return *r.Temperature
}

// contentString returns the message content as a string, and false if the
// content is not a JSON string (i.e. it's the array-of-parts form).
func (m chatMessage) contentString() (string, bool) {
	var s string
	if err := json.Unmarshal(m.Content, &s); err != nil {
		return "", false
	}
	return s, true
}

// cacheIdentity reduces a request to the prompt text and scope semcache needs
// to look up or store an answer. It returns ok=false when the request shape
// isn't safely reducible to text: any message with array-form content, n > 1
// (multiple choices can't be represented by a single cached answer), or
// stream_options (whose stream shape a replay can't reproduce).
func (r *chatRequest) cacheIdentity(tenant string) (prompt string, scope semcache.Scope, ok bool) {
	if r.N > 1 {
		return "", semcache.Scope{}, false
	}
	if rawFieldPresent(r.StreamOptions) {
		return "", semcache.Scope{}, false
	}

	var promptBuf strings.Builder
	var systemParts []string

	for _, msg := range r.Messages {
		content, isString := msg.contentString()
		if !isString {
			return "", semcache.Scope{}, false
		}

		if msg.Role == "system" {
			systemParts = append(systemParts, content)
			continue
		}

		promptBuf.WriteString(msg.Role)
		promptBuf.WriteByte('\x00')
		promptBuf.WriteString(content)
		promptBuf.WriteByte('\x1e')
	}

	tools := make([]string, 0, len(r.Tools))
	for _, t := range r.Tools {
		tools = append(tools, t.Function.Name)
	}

	scope = semcache.Scope{
		Model:       r.Model,
		System:      strings.Join(systemParts, "\n"),
		Tools:       tools,
		Temperature: r.temperature(),
		MaxTokens:   r.MaxTokens,
		Tenant:      tenant,
		Extra:       extraDiscriminator(r.Raw),
	}
	return promptBuf.String(), scope, true
}

// extraDiscriminator hashes everything in the request body that isn't already
// part of the prompt or the stream decision, so any request knob the proxy
// doesn't model by name still partitions the cache. Without it, a
// response_format:json_object request happily received prose cached from the
// same messages, and seed / top_p / stop / frequency_penalty all silently
// shared one entry.
//
// messages is removed because it IS the prompt; stream and stream_options
// because a cached answer is deliberately shared between the streaming and
// non-streaming forms of the same call. Everything else — including fields
// already in the scope, harmlessly counted twice — is hashed from the
// unmarshaled map, so two byte-different but semantically equal bodies (key
// order, whitespace) still agree: Go marshals map keys in sorted order.
func extraDiscriminator(raw json.RawMessage) string {
	if len(raw) == 0 {
		return ""
	}
	var m map[string]any
	if err := json.Unmarshal(raw, &m); err != nil {
		// Unreachable for a body that parsed as a chat request (that needs a
		// JSON object), but if it ever happens, partition on the raw bytes
		// rather than let two unrelated requests share a cache entry.
		return strconv.FormatUint(xxhash.Sum64String(string(raw)), 16)
	}
	delete(m, "messages")
	delete(m, "stream")
	delete(m, "stream_options")
	if len(m) == 0 {
		return ""
	}
	canonical, err := json.Marshal(m)
	if err != nil {
		return strconv.FormatUint(xxhash.Sum64String(string(raw)), 16)
	}
	return strconv.FormatUint(xxhash.Sum64String(string(canonical)), 16)
}

// synthesizeResponse builds a non-streaming chat-completions response body
// for a cache hit, in the exact shape a real OpenAI response would have so
// clients can't tell the difference. prompt_tokens is always 0 because the
// prompt was never sent upstream on a hit. created is the synthesis time
// (Unix seconds), not the original completion's — OpenAI's SDKs treat a
// missing created as an error-shaped response, so a synthesized reply needs
// *a* timestamp even though the real one was never recorded.
func synthesizeResponse(model, answer string, outTokens int) []byte {
	resp := map[string]any{
		"id":      "rostam-cache",
		"object":  "chat.completion",
		"created": time.Now().Unix(),
		"model":   model,
		"choices": []map[string]any{
			{
				"index": 0,
				"message": map[string]any{
					"role":    "assistant",
					"content": answer,
				},
				"finish_reason": "stop",
			},
		},
		"usage": map[string]any{
			"prompt_tokens":     0,
			"completion_tokens": outTokens,
			"total_tokens":      outTokens,
		},
	}
	// The shape above is fixed and always marshals cleanly; an error here
	// would mean a bug in this function, not bad input.
	b, err := json.Marshal(resp)
	if err != nil {
		panic("llmproxy: synthesizeResponse: " + err.Error())
	}
	return b
}

// tenantOf derives a cache-scope tenant identity from a request's credential
// headers, so one client's cached answers are never served to another.
//
// All three headers are covered because an OpenAI-compatible surface is not
// only OpenAI: Azure OpenAI authenticates with "api-key" and Anthropic with
// "x-api-key", and hashing Authorization alone collapsed every such caller
// into the single empty "no auth" tenant — every Azure user on one proxy
// sharing one cache. The values are hashed, never stored raw: the tenant ends
// up in cache metadata, which is no place for a credential.
func tenantOf(authHeader, apiKeyHeader, xAPIKeyHeader string) string {
	if authHeader == "" && apiKeyHeader == "" && xAPIKeyHeader == "" {
		return ""
	}
	return strconv.FormatUint(xxhash.Sum64String(authHeader+"\x00"+apiKeyHeader+"\x00"+xAPIKeyHeader), 16)
}
