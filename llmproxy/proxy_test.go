// SPDX-License-Identifier: Apache-2.0

package llmproxy

import (
	"encoding/json"
	"io"
	"net/http"
	"testing"
)

const chatPath = "/v1/chat/completions"

// (a) miss -> store -> hit: identical body twice. First request is a miss
// that reaches upstream and gets stored; the second is a hit answered
// entirely from the cache, without touching upstream again.
func TestProxy_MissThenHit(t *testing.T) {
	upstream := newFakeUpstream(t)
	upstream.script(chatPath, scriptedResponse{
		body: chatCompletionJSON(t, "the answer", "stop", 7),
	})
	proxy := newProxy(t, upstream, nil)

	body := `{"model":"gpt-4","messages":[{"role":"user","content":"what is 2+2"}]}`

	resp1, body1 := postChat(t, proxy.URL, body, nil)
	if resp1.StatusCode != http.StatusOK {
		t.Fatalf("first request status = %d, want 200", resp1.StatusCode)
	}
	if got := resp1.Header.Get("x-rostam-cache"); got != "miss" {
		t.Fatalf("first request x-rostam-cache = %q, want miss", got)
	}
	if n := upstream.requestCount(chatPath); n != 1 {
		t.Fatalf("upstream requests after first call = %d, want 1", n)
	}

	resp2, body2 := postChat(t, proxy.URL, body, nil)
	if resp2.StatusCode != http.StatusOK {
		t.Fatalf("second request status = %d, want 200", resp2.StatusCode)
	}
	if got := resp2.Header.Get("x-rostam-cache"); got != "hit" {
		t.Fatalf("second request x-rostam-cache = %q, want hit", got)
	}
	if n := upstream.requestCount(chatPath); n != 1 {
		t.Fatalf("upstream requests after second call = %d, want still 1 (cache hit must not call upstream)", n)
	}

	var decoded chatResponse
	if err := json.Unmarshal(body2, &decoded); err != nil {
		t.Fatalf("unmarshal cached response: %v", err)
	}
	if len(decoded.Choices) != 1 || decoded.Choices[0].Message.Content != "the answer" {
		t.Fatalf("cached response content = %+v, want %q", decoded.Choices, "the answer")
	}
	_ = body1
}

// (b) a one-character-different body must miss under the exact-mode stub
// embedder (threshold 0.999) — near-identical text is not close enough.
func TestProxy_OneCharDifferentBodyMisses(t *testing.T) {
	upstream := newFakeUpstream(t)
	upstream.script(chatPath, scriptedResponse{
		body: chatCompletionJSON(t, "answer", "stop", 3),
	})
	proxy := newProxy(t, upstream, nil)

	body1 := `{"model":"gpt-4","messages":[{"role":"user","content":"what is 2+2a"}]}`
	body2 := `{"model":"gpt-4","messages":[{"role":"user","content":"what is 2+2b"}]}`

	resp1, _ := postChat(t, proxy.URL, body1, nil)
	if got := resp1.Header.Get("x-rostam-cache"); got != "miss" {
		t.Fatalf("first request x-rostam-cache = %q, want miss", got)
	}

	resp2, _ := postChat(t, proxy.URL, body2, nil)
	if got := resp2.Header.Get("x-rostam-cache"); got != "miss" {
		t.Fatalf("second (one-char-different) request x-rostam-cache = %q, want miss", got)
	}
	if n := upstream.requestCount(chatPath); n != 2 {
		t.Fatalf("upstream requests = %d, want 2 (both should miss)", n)
	}
}

// (c) tenant isolation: the same prompt under two different Authorization
// headers must miss both times — one tenant never sees another's cache.
func TestProxy_TenantIsolation(t *testing.T) {
	upstream := newFakeUpstream(t)
	upstream.script(chatPath, scriptedResponse{
		body: chatCompletionJSON(t, "answer", "stop", 3),
	})
	proxy := newProxy(t, upstream, nil)

	body := `{"model":"gpt-4","messages":[{"role":"user","content":"shared prompt"}]}`

	resp1, _ := postChat(t, proxy.URL, body, map[string]string{"Authorization": "Bearer tenant-a"})
	if got := resp1.Header.Get("x-rostam-cache"); got != "miss" {
		t.Fatalf("tenant-a request x-rostam-cache = %q, want miss", got)
	}

	resp2, _ := postChat(t, proxy.URL, body, map[string]string{"Authorization": "Bearer tenant-b"})
	if got := resp2.Header.Get("x-rostam-cache"); got != "miss" {
		t.Fatalf("tenant-b request x-rostam-cache = %q, want miss (different tenant must not see tenant-a's cache)", got)
	}
	if n := upstream.requestCount(chatPath); n != 2 {
		t.Fatalf("upstream requests = %d, want 2", n)
	}
}

// (d) temperature above MaxTemp is uncacheable: the request passes through
// to upstream and the uncacheable counter grows, but nothing is stored or
// looked up.
func TestProxy_HighTemperaturePassesThroughUncacheable(t *testing.T) {
	upstream := newFakeUpstream(t)
	upstream.script(chatPath, scriptedResponse{
		body: chatCompletionJSON(t, "answer", "stop", 3),
	})
	proxy := newProxy(t, upstream, nil)

	body := `{"model":"gpt-4","temperature":1.5,"messages":[{"role":"user","content":"hi"}]}`

	resp, _ := postChat(t, proxy.URL, body, nil)
	if resp.StatusCode != http.StatusOK {
		t.Fatalf("status = %d, want 200", resp.StatusCode)
	}
	if n := upstream.requestCount(chatPath); n != 1 {
		t.Fatalf("upstream requests = %d, want 1", n)
	}

	statsResp, statsBody := getStats(t, proxy.URL)
	if statsResp.StatusCode != http.StatusOK {
		t.Fatalf("/stats status = %d, want 200", statsResp.StatusCode)
	}
	var st map[string]any
	if err := json.Unmarshal(statsBody, &st); err != nil {
		t.Fatalf("unmarshal /stats: %v", err)
	}
	if got := st["uncacheable"]; got != float64(1) {
		t.Fatalf("/stats uncacheable = %v, want 1", got)
	}
}

// (e) an upstream response carrying tool_calls is relayed to the client but
// must not be stored: a repeat of the same request misses again.
func TestProxy_ToolCallsResponseNotStored(t *testing.T) {
	upstream := newFakeUpstream(t)
	toolResp := map[string]any{
		"id":     "chatcmpl-fake",
		"object": "chat.completion",
		"model":  "gpt-4",
		"choices": []map[string]any{
			{
				"index": 0,
				"message": map[string]any{
					"role": "assistant",
					"tool_calls": []map[string]any{
						{"id": "call_1", "type": "function", "function": map[string]any{"name": "get_weather", "arguments": "{}"}},
					},
				},
				"finish_reason": "tool_calls",
			},
		},
		"usage": map[string]any{"completion_tokens": 5},
	}
	toolBody, err := json.Marshal(toolResp)
	if err != nil {
		t.Fatalf("marshal: %v", err)
	}
	upstream.script(chatPath, scriptedResponse{body: toolBody})
	proxy := newProxy(t, upstream, nil)

	body := `{"model":"gpt-4","messages":[{"role":"user","content":"what's the weather"}],
		"tools":[{"function":{"name":"get_weather"}}]}`

	resp1, _ := postChat(t, proxy.URL, body, nil)
	if resp1.StatusCode != http.StatusOK {
		t.Fatalf("status = %d, want 200", resp1.StatusCode)
	}
	if got := resp1.Header.Get("x-rostam-cache"); got != "miss" {
		t.Fatalf("x-rostam-cache = %q, want miss", got)
	}

	resp2, _ := postChat(t, proxy.URL, body, nil)
	if got := resp2.Header.Get("x-rostam-cache"); got != "miss" {
		t.Fatalf("repeat request x-rostam-cache = %q, want miss (tool_calls responses must not be stored)", got)
	}
	if n := upstream.requestCount(chatPath); n != 2 {
		t.Fatalf("upstream requests = %d, want 2 (both should reach upstream)", n)
	}
}

// (f) a non-chat-completions path (e.g. GET /v1/models) passes through
// verbatim, preserving path and the Authorization header at upstream.
func TestProxy_PassthroughPreservesPathAndAuth(t *testing.T) {
	upstream := newFakeUpstream(t)
	upstream.script("/v1/models", scriptedResponse{
		body: []byte(`{"object":"list","data":[]}`),
	})
	proxy := newProxy(t, upstream, nil)

	req, err := http.NewRequest(http.MethodGet, proxy.URL+"/v1/models", nil)
	if err != nil {
		t.Fatalf("NewRequest: %v", err)
	}
	req.Header.Set("Authorization", "Bearer sk-passthrough")

	resp, err := http.DefaultClient.Do(req)
	if err != nil {
		t.Fatalf("Do: %v", err)
	}
	defer resp.Body.Close()
	if resp.StatusCode != http.StatusOK {
		t.Fatalf("status = %d, want 200", resp.StatusCode)
	}

	captured := upstream.lastRequest("/v1/models")
	if captured == nil {
		t.Fatalf("upstream never saw /v1/models")
	}
	if captured.Method != http.MethodGet {
		t.Fatalf("upstream method = %q, want GET", captured.Method)
	}
	if got := captured.Headers.Get("Authorization"); got != "Bearer sk-passthrough" {
		t.Fatalf("upstream Authorization = %q, want %q", got, "Bearer sk-passthrough")
	}
}

// A passthrough body must be streamed through to upstream rather than
// buffered in the proxy: a 20 MiB body (well over the 16 MiB chat-completions
// cap, which does not apply to passthrough routes) must reach upstream
// intact and get a normal 200, never a 413. The request body is sourced from
// an io.Reader rather than a pre-built []byte so the test itself can't
// accidentally hide a proxy-side buffering bug behind its own buffering.
func TestProxy_PassthroughStreamsLargeBody(t *testing.T) {
	upstream := newFakeUpstream(t)
	upstream.script("/v1/embeddings", scriptedResponse{body: []byte(`{"ok":true}`)})
	proxy := newProxy(t, upstream, nil)

	const size = 20 << 20 // 20 MiB

	req, err := http.NewRequest(http.MethodPost, proxy.URL+"/v1/embeddings", io.LimitReader(zeroReader{}, size))
	if err != nil {
		t.Fatalf("NewRequest: %v", err)
	}
	req.ContentLength = size

	resp, err := http.DefaultClient.Do(req)
	if err != nil {
		t.Fatalf("Do: %v", err)
	}
	defer resp.Body.Close()
	if resp.StatusCode != http.StatusOK {
		t.Fatalf("status = %d, want 200 (not 413 — passthrough has no chat-completions body cap)", resp.StatusCode)
	}

	captured := upstream.lastRequest("/v1/embeddings")
	if captured == nil {
		t.Fatalf("upstream never saw /v1/embeddings")
	}
	if len(captured.Body) != size {
		t.Fatalf("upstream received body length = %d, want %d (body must reach upstream intact)", len(captured.Body), size)
	}
}

// zeroReader is an endless source of zero bytes, used with io.LimitReader to
// produce a large test request body without ever holding it as a single
// in-memory []byte.
type zeroReader struct{}

func (zeroReader) Read(p []byte) (int, error) {
	for i := range p {
		p[i] = 0
	}
	return len(p), nil
}

// (g) malformed JSON on the cache path returns an OpenAI-shaped 400 error.
func TestProxy_MalformedJSONReturns400(t *testing.T) {
	upstream := newFakeUpstream(t)
	proxy := newProxy(t, upstream, nil)

	resp, body := postChat(t, proxy.URL, `{not valid json`, nil)
	if resp.StatusCode != http.StatusBadRequest {
		t.Fatalf("status = %d, want 400", resp.StatusCode)
	}

	var errBody struct {
		Error struct {
			Message string `json:"message"`
			Type    string `json:"type"`
		} `json:"error"`
	}
	if err := json.Unmarshal(body, &errBody); err != nil {
		t.Fatalf("unmarshal error body: %v; body=%s", err, body)
	}
	if errBody.Error.Type != "invalid_request_error" {
		t.Fatalf("error.type = %q, want invalid_request_error", errBody.Error.Type)
	}
	if errBody.Error.Message == "" {
		t.Fatalf("error.message is empty")
	}
}

// (h) an upstream 500 is relayed untouched to the client and not cached.
func TestProxy_Upstream500RelayedNotCached(t *testing.T) {
	upstream := newFakeUpstream(t)
	upstream.script(chatPath, scriptedResponse{
		status: http.StatusInternalServerError,
		body:   []byte(`{"error":{"message":"boom","type":"server_error"}}`),
	})
	proxy := newProxy(t, upstream, nil)

	body := `{"model":"gpt-4","messages":[{"role":"user","content":"trigger a 500"}]}`

	resp1, body1 := postChat(t, proxy.URL, body, nil)
	if resp1.StatusCode != http.StatusInternalServerError {
		t.Fatalf("status = %d, want 500", resp1.StatusCode)
	}
	if got := string(body1); got != `{"error":{"message":"boom","type":"server_error"}}` {
		t.Fatalf("relayed body = %q, want upstream body verbatim", got)
	}

	// A repeat of the same request must miss again: a 500 must never be
	// stored as a cached answer.
	resp2, _ := postChat(t, proxy.URL, body, nil)
	if resp2.StatusCode != http.StatusInternalServerError {
		t.Fatalf("second status = %d, want 500 again (nothing cached)", resp2.StatusCode)
	}
	if n := upstream.requestCount(chatPath); n != 2 {
		t.Fatalf("upstream requests = %d, want 2", n)
	}
}

// (i) /stats reflects hits, misses, stored, uncacheable and tokens_saved
// across a mixed sequence of the scenarios above, plus the configured mode.
func TestProxy_StatsAggregatesAcrossRequests(t *testing.T) {
	upstream := newFakeUpstream(t)
	upstream.script(chatPath, scriptedResponse{
		body: chatCompletionJSON(t, "cacheable answer", "stop", 11),
	})
	proxy := newProxy(t, upstream, nil)

	cacheableBody := `{"model":"gpt-4","messages":[{"role":"user","content":"cacheable prompt"}]}`
	uncacheableBody := `{"model":"gpt-4","temperature":2.0,"messages":[{"role":"user","content":"loud prompt"}]}`

	// miss, then store
	postChat(t, proxy.URL, cacheableBody, nil)
	// hit
	postChat(t, proxy.URL, cacheableBody, nil)
	// uncacheable passthrough
	postChat(t, proxy.URL, uncacheableBody, nil)

	statsResp, statsBody := getStats(t, proxy.URL)
	if statsResp.StatusCode != http.StatusOK {
		t.Fatalf("/stats status = %d, want 200", statsResp.StatusCode)
	}

	var st struct {
		Hits        float64 `json:"hits"`
		Misses      float64 `json:"misses"`
		Stored      float64 `json:"stored"`
		Uncacheable float64 `json:"uncacheable"`
		TokensSaved float64 `json:"tokens_saved"`
		Mode        string  `json:"mode"`
	}
	if err := json.Unmarshal(statsBody, &st); err != nil {
		t.Fatalf("unmarshal /stats: %v; body=%s", err, statsBody)
	}
	if st.Hits != 1 {
		t.Fatalf("hits = %v, want 1", st.Hits)
	}
	if st.Misses != 1 {
		t.Fatalf("misses = %v, want 1", st.Misses)
	}
	if st.Stored != 1 {
		t.Fatalf("stored = %v, want 1", st.Stored)
	}
	if st.Uncacheable != 1 {
		t.Fatalf("uncacheable = %v, want 1", st.Uncacheable)
	}
	if st.TokensSaved != 11 {
		t.Fatalf("tokens_saved = %v, want 11", st.TokensSaved)
	}
	if st.Mode != "exact" {
		t.Fatalf("mode = %q, want exact", st.Mode)
	}
}

// getStats GETs /stats on the proxy and returns the response plus its body.
func getStats(t *testing.T, proxyURL string) (*http.Response, []byte) {
	t.Helper()
	resp, err := http.Get(proxyURL + "/stats")
	if err != nil {
		t.Fatalf("GET /stats: %v", err)
	}
	defer resp.Body.Close()
	body, err := io.ReadAll(resp.Body)
	if err != nil {
		t.Fatalf("ReadAll: %v", err)
	}
	return resp, body
}
