// SPDX-License-Identifier: Apache-2.0

package llmproxy

import (
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"log/slog"
	"net/http"
	"strconv"
	"strings"
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

// fnv32a is the 32-bit hash the stub embedder used to seed itself with,
// reproduced here to manufacture the prompt pair that used to cross-hit.
func fnv32a(s string) uint32 {
	h := uint32(2166136261)
	for _, b := range []byte(s) {
		h = (h ^ uint32(b)) * 16777619
	}
	return h
}

// colliding32BitPair finds two distinct user-message contents whose SERIALIZED
// prompts (what cacheIdentity actually hands the embedder, role and separators
// included) share a 32-bit FNV-1a hash, by birthday search over a 2^32 space.
// Colliding the bare content instead would prove nothing: the embedder never
// sees it.
func colliding32BitPair(t *testing.T) (string, string) {
	t.Helper()
	const maxCandidates = 500_000 // P(no collision) < 1e-6

	seen := make(map[uint32]string, maxCandidates)
	for i := 0; i < maxCandidates; i++ {
		s := "collide-" + strconv.Itoa(i)
		h := fnv32a("user\x00" + s + "\x1e")
		if prev, ok := seen[h]; ok {
			return prev, s
		}
		seen[h] = s
	}
	t.Skip("no 32-bit FNV-1a collision found in the candidate budget")
	return "", ""
}

// (b2) two prompts that collide in 32-bit FNV-1a must still miss each other
// end to end. The stub embedder used to derive every dimension from that
// 32-bit hash, so this pair embedded identically and the second prompt was
// answered with the first prompt's completion — a wrong answer, served with
// x-rostam-cache: hit.
func TestProxy_ThirtyTwoBitCollidingPromptsDoNotCrossHit(t *testing.T) {
	promptA, promptB := colliding32BitPair(t)

	upstream := newFakeUpstream(t)
	upstream.script(chatPath, scriptedResponse{
		body: chatCompletionJSON(t, "answer for A", "stop", 5),
	})
	proxy := newProxy(t, upstream, nil)

	bodyA := `{"model":"gpt-4","messages":[{"role":"user","content":"` + promptA + `"}]}`
	bodyB := `{"model":"gpt-4","messages":[{"role":"user","content":"` + promptB + `"}]}`

	resp1, _ := postChat(t, proxy.URL, bodyA, nil)
	if got := resp1.Header.Get("x-rostam-cache"); got != "miss" {
		t.Fatalf("first request x-rostam-cache = %q, want miss", got)
	}

	resp2, _ := postChat(t, proxy.URL, bodyB, nil)
	if got := resp2.Header.Get("x-rostam-cache"); got != "miss" {
		t.Fatalf("32-bit-colliding prompt %q x-rostam-cache = %q, want miss (it must not be answered with %q's cached completion)",
			promptB, got, promptA)
	}
	if n := upstream.requestCount(chatPath); n != 2 {
		t.Fatalf("upstream requests = %d, want 2 (both prompts must reach upstream)", n)
	}
}

// A client that advertises gzip must still get its answers cached. The proxy
// used to forward the client's Accept-Encoding, which suppressed Go's
// transparent decompression, so maybeStore was handed raw gzip bytes and the
// JSON decode failed on every single response — and openai-python sends
// "Accept-Encoding: gzip, deflate" by default, so that was the common case.
func TestProxy_GzipAcceptingClientStillGetsCached(t *testing.T) {
	upstream := newFakeUpstream(t)
	upstream.script(chatPath, scriptedResponse{
		body:           chatCompletionJSON(t, "compressed answer", "stop", 6),
		gzipIfAccepted: true,
	})
	proxy := newProxy(t, upstream, nil)

	body := `{"model":"gpt-4","messages":[{"role":"user","content":"compress me"}]}`
	headers := map[string]string{"Accept-Encoding": "gzip, deflate"}

	resp1, body1 := postChat(t, proxy.URL, body, headers)
	if resp1.StatusCode != http.StatusOK {
		t.Fatalf("first request status = %d, want 200", resp1.StatusCode)
	}
	if got := resp1.Header.Get("x-rostam-cache"); got != "miss" {
		t.Fatalf("first request x-rostam-cache = %q, want miss", got)
	}
	// The client must receive decodable JSON, not the gzip frame.
	var decoded1 chatResponse
	if err := json.Unmarshal(body1, &decoded1); err != nil {
		t.Fatalf("client could not decode the relayed body: %v; body=%q", err, body1)
	}

	// The upstream must have been asked with the transport's OWN gzip header,
	// not the client's — that is what buys transparent decompression.
	captured := upstream.lastRequest(chatPath)
	if captured == nil {
		t.Fatalf("upstream never saw %s", chatPath)
	}
	if got := captured.Headers.Get("Accept-Encoding"); got != "gzip" {
		t.Fatalf("upstream Accept-Encoding = %q, want %q (the client's header must not be forwarded on a cached path)", got, "gzip")
	}

	resp2, body2 := postChat(t, proxy.URL, body, headers)
	if got := resp2.Header.Get("x-rostam-cache"); got != "hit" {
		t.Fatalf("second request x-rostam-cache = %q, want hit (a gzip-accepting client's response must still be stored)", got)
	}
	if n := upstream.requestCount(chatPath); n != 1 {
		t.Fatalf("upstream requests = %d, want 1 (the second request must be served from the cache)", n)
	}
	var decoded2 chatResponse
	if err := json.Unmarshal(body2, &decoded2); err != nil {
		t.Fatalf("unmarshal cached response: %v", err)
	}
	if len(decoded2.Choices) != 1 || decoded2.Choices[0].Message.Content != "compressed answer" {
		t.Fatalf("cached response content = %+v, want %q", decoded2.Choices, "compressed answer")
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

// Same messages, different sampling/formatting surface: the proxy used to
// scope only on temperature and max_tokens, so a response_format:json_object
// request was answered with prose cached from the identical messages. Each
// variation must reach upstream on its own.
func TestProxy_SamplingParamsPartitionTheCache(t *testing.T) {
	upstream := newFakeUpstream(t)
	upstream.script(chatPath, scriptedResponse{
		body: chatCompletionJSON(t, "an answer", "stop", 4),
	})
	proxy := newProxy(t, upstream, nil)

	const messages = `"model":"gpt-4","messages":[{"role":"user","content":"list three colors"}]`
	bodies := []string{
		`{` + messages + `}`,
		`{` + messages + `,"response_format":{"type":"json_object"}}`,
		`{` + messages + `,"seed":7}`,
		`{` + messages + `,"top_p":0.1}`,
		`{` + messages + `,"stop":["\n"]}`,
		`{` + messages + `,"frequency_penalty":1.5}`,
	}

	for i, body := range bodies {
		resp, _ := postChat(t, proxy.URL, body, nil)
		if got := resp.Header.Get("x-rostam-cache"); got != "miss" {
			t.Fatalf("body %d (%s) x-rostam-cache = %q, want miss (it must not be served another variant's answer)", i, body, got)
		}
	}
	if n := upstream.requestCount(chatPath); n != len(bodies) {
		t.Fatalf("upstream requests = %d, want %d (every variant must reach upstream)", n, len(bodies))
	}

	// A repeat of a variant still hits: partitioning must not break caching.
	resp, _ := postChat(t, proxy.URL, bodies[1], nil)
	if got := resp.Header.Get("x-rostam-cache"); got != "hit" {
		t.Fatalf("repeat of the json_object variant x-rostam-cache = %q, want hit", got)
	}

	// Same request surface, written differently (key order, whitespace) —
	// still the same cache entry.
	reordered := `{"response_format":{"type":"json_object"}, "messages":[{"role":"user","content":"list three colors"}] , "model":"gpt-4"}`
	resp, _ = postChat(t, proxy.URL, reordered, nil)
	if got := resp.Header.Get("x-rostam-cache"); got != "hit" {
		t.Fatalf("key-reordered json_object variant x-rostam-cache = %q, want hit (key order is not request surface)", got)
	}
	if n := upstream.requestCount(chatPath); n != len(bodies) {
		t.Fatalf("upstream requests = %d, want still %d", n, len(bodies))
	}
}

// stream_options asks for a stream shape a cache replay cannot produce (an
// extra usage chunk), so such a request is uncacheable passthrough: counted
// uncacheable, and never stored — a later request without it still misses.
func TestProxy_StreamOptionsPassesThroughUncacheable(t *testing.T) {
	upstream := newFakeUpstream(t)
	upstream.script(chatPath, scriptedResponse{sse: true, lines: basicStreamLines})
	proxy := newProxy(t, upstream, nil)

	withOptions := `{"model":"gpt-4","stream":true,"stream_options":{"include_usage":true},` +
		`"messages":[{"role":"user","content":"count my tokens"}]}`

	resp1, body1 := postChat(t, proxy.URL, withOptions, nil)
	if resp1.StatusCode != http.StatusOK {
		t.Fatalf("status = %d, want 200", resp1.StatusCode)
	}
	if got := resp1.Header.Get("x-rostam-cache"); got != "uncacheable" {
		t.Fatalf("x-rostam-cache = %q, want uncacheable", got)
	}
	if got := string(body1); got != strings.Join(basicStreamLines, "") {
		t.Fatalf("relayed body = %q, want the upstream stream verbatim", got)
	}

	// Repeat it: still uncacheable, still reaching upstream.
	postChat(t, proxy.URL, withOptions, nil)
	if n := upstream.requestCount(chatPath); n != 2 {
		t.Fatalf("upstream requests = %d, want 2", n)
	}

	statsResp, statsBody := getStats(t, proxy.URL)
	if statsResp.StatusCode != http.StatusOK {
		t.Fatalf("/stats status = %d, want 200", statsResp.StatusCode)
	}
	var st struct {
		Uncacheable float64 `json:"uncacheable"`
		Stored      float64 `json:"stored"`
		Misses      float64 `json:"misses"`
	}
	if err := json.Unmarshal(statsBody, &st); err != nil {
		t.Fatalf("unmarshal /stats: %v; body=%s", err, statsBody)
	}
	if st.Uncacheable != 2 {
		t.Fatalf("uncacheable = %v, want 2", st.Uncacheable)
	}
	if st.Stored != 0 {
		t.Fatalf("stored = %v, want 0 (a stream_options request must never be stored)", st.Stored)
	}
	if st.Misses != 0 {
		t.Fatalf("misses = %v, want 0 (an uncacheable request is not a cache miss)", st.Misses)
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

// flushCounter is a writer that records how many times it was flushed, so a
// relay can be checked for incremental delivery without timing anything.
type flushCounter struct {
	bytes.Buffer
	flushes int
}

func (f *flushCounter) Flush() { f.flushes++ }

// io.Copy hands the destination 32 KiB at a time and never flushes, so a
// relay built on it alone buffers a "stream" until the copy finishes. The
// wrapper must flush after every write instead.
func TestFlushWriter_FlushesAfterEveryWrite(t *testing.T) {
	fc := &flushCounter{}
	fw := newFlushWriter(fc)

	chunks := []string{"data: one\n\n", "data: two\n\n", "data: [DONE]\n\n"}
	for _, c := range chunks {
		if _, err := io.WriteString(fw, c); err != nil {
			t.Fatalf("Write: %v", err)
		}
	}

	if fc.flushes != len(chunks) {
		t.Fatalf("flushes = %d, want %d (one per write)", fc.flushes, len(chunks))
	}
	if got, want := fc.String(), strings.Join(chunks, ""); got != want {
		t.Fatalf("relayed bytes = %q, want %q", got, want)
	}
}

// A destination that can't flush must still be written to, not rejected:
// these relays have to work on any writer.
func TestFlushWriter_NonFlusherStillCopies(t *testing.T) {
	var buf bytes.Buffer
	if _, err := io.WriteString(newFlushWriter(&buf), "payload"); err != nil {
		t.Fatalf("Write: %v", err)
	}
	if buf.String() != "payload" {
		t.Fatalf("wrote %q, want %q", buf.String(), "payload")
	}
}

// An uncacheable STREAMING request (temperature above the ceiling) takes the
// passthrough relay, which must still deliver a real SSE stream — flushed
// chunk by chunk — and label the response uncacheable.
func TestProxy_UncacheableStreamingRelaysSSE(t *testing.T) {
	upstream := newFakeUpstream(t)
	upstream.script(chatPath, scriptedResponse{sse: true, lines: basicStreamLines})
	proxy := newProxy(t, upstream, nil)

	body := `{"model":"gpt-4","stream":true,"temperature":1.5,` +
		`"messages":[{"role":"user","content":"be creative"}]}`

	resp, respBody := postChat(t, proxy.URL, body, nil)
	if resp.StatusCode != http.StatusOK {
		t.Fatalf("status = %d, want 200", resp.StatusCode)
	}
	if got := resp.Header.Get("x-rostam-cache"); got != "uncacheable" {
		t.Fatalf("x-rostam-cache = %q, want uncacheable", got)
	}
	if got := resp.Header.Get("Content-Type"); got != "text/event-stream" {
		t.Fatalf("Content-Type = %q, want text/event-stream", got)
	}
	if got := string(respBody); got != strings.Join(basicStreamLines, "") {
		t.Fatalf("relayed body = %q, want the upstream stream verbatim", got)
	}
	if got := assembleSSEContent(t, string(respBody)); got != "Hello, world" {
		t.Fatalf("assembled answer = %q, want %q", got, "Hello, world")
	}
}

// A client hanging up mid-response (context.Canceled, however wrapped) is
// normal traffic on a streaming proxy, so it must not be logged as an ERROR;
// anything else still is.
func TestLogRelayError_ClientCancelLogsAtDebug(t *testing.T) {
	var buf bytes.Buffer
	s := &Server{log: slog.New(slog.NewTextHandler(&buf, &slog.HandlerOptions{Level: slog.LevelDebug}))}

	for _, err := range []error{context.Canceled, fmt.Errorf("llmproxy: relay: %w", context.Canceled)} {
		buf.Reset()
		s.logRelayError("llmproxy: relaying", err)
		if got := buf.String(); !strings.Contains(got, "level=DEBUG") || strings.Contains(got, "level=ERROR") {
			t.Fatalf("logRelayError(%v) logged %q, want DEBUG", err, got)
		}
	}

	buf.Reset()
	s.logRelayError("llmproxy: relaying", errors.New("connection reset by peer"))
	if got := buf.String(); !strings.Contains(got, "level=ERROR") {
		t.Fatalf("logRelayError(real failure) logged %q, want ERROR", got)
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
