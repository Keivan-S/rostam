// SPDX-License-Identifier: Apache-2.0

package llmproxy

import (
	"bufio"
	"encoding/json"
	"net/http"
	"strings"
	"testing"
)

// assembleSSEContent decodes an SSE chat-completions-chunk body (as produced
// by either relayAndCapture's live relay or replayAsSSE's cache replay) and
// concatenates every delta.content field in order, the way a real client
// would reconstruct the answer.
func assembleSSEContent(t *testing.T, body string) string {
	t.Helper()
	var sb strings.Builder
	sc := bufio.NewScanner(strings.NewReader(body))
	for sc.Scan() {
		data, ok := strings.CutPrefix(sc.Text(), "data: ")
		if !ok || data == "[DONE]" {
			continue
		}
		var c sseChunk
		if err := json.Unmarshal([]byte(data), &c); err != nil {
			t.Fatalf("unmarshal chunk: %v; data=%s", err, data)
		}
		for _, ch := range c.Choices {
			sb.WriteString(ch.Delta.Content)
		}
	}
	if err := sc.Err(); err != nil {
		t.Fatalf("scan SSE body: %v", err)
	}
	return sb.String()
}

// basicStreamLines is a realistic upstream chat-completions SSE script: role
// chunk, two content deltas, a finish chunk, a usage chunk, then [DONE].
var basicStreamLines = []string{
	"data: {\"choices\":[{\"index\":0,\"delta\":{\"role\":\"assistant\"},\"finish_reason\":null}]}\n\n",
	"data: {\"choices\":[{\"index\":0,\"delta\":{\"content\":\"Hello, \"},\"finish_reason\":null}]}\n\n",
	"data: {\"choices\":[{\"index\":0,\"delta\":{\"content\":\"world\"},\"finish_reason\":null}]}\n\n",
	"data: {\"choices\":[{\"index\":0,\"delta\":{},\"finish_reason\":\"stop\"}]}\n\n",
	"data: {\"choices\":[],\"usage\":{\"completion_tokens\":9}}\n\n",
	"data: [DONE]\n\n",
}

// (a) streaming miss: scripted upstream SSE relayed to the client byte-for-
// byte. Then an identical streaming request is served from the cache as a
// valid SSE stream without a second upstream call, and the assembled answer
// from the cached replay matches the deltas assembled from the live relay.
func TestProxy_StreamingMissThenHit(t *testing.T) {
	upstream := newFakeUpstream(t)
	upstream.script(chatPath, scriptedResponse{sse: true, lines: basicStreamLines})
	proxy := newProxy(t, upstream, nil)

	body := `{"model":"gpt-4","stream":true,"messages":[{"role":"user","content":"say hello"}]}`

	resp1, body1 := postChat(t, proxy.URL, body, nil)
	if resp1.StatusCode != http.StatusOK {
		t.Fatalf("first request status = %d, want 200", resp1.StatusCode)
	}
	if got := resp1.Header.Get("x-rostam-cache"); got != "miss" {
		t.Fatalf("first request x-rostam-cache = %q, want miss", got)
	}
	wantBody := strings.Join(basicStreamLines, "")
	if got := string(body1); got != wantBody {
		t.Fatalf("relayed body = %q, want %q (byte-for-byte)", got, wantBody)
	}
	if n := upstream.requestCount(chatPath); n != 1 {
		t.Fatalf("upstream requests after first call = %d, want 1", n)
	}
	firstAnswer := assembleSSEContent(t, string(body1))
	if firstAnswer != "Hello, world" {
		t.Fatalf("assembled answer from relay = %q, want %q", firstAnswer, "Hello, world")
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
	if got := resp2.Header.Get("Content-Type"); got != "text/event-stream" {
		t.Fatalf("second request Content-Type = %q, want text/event-stream", got)
	}
	secondAnswer := assembleSSEContent(t, string(body2))
	if secondAnswer != firstAnswer {
		t.Fatalf("assembled answer from cache replay = %q, want %q (concatenated deltas from the miss)", secondAnswer, firstAnswer)
	}
}

// (b1) cross-mode hit: cache populated via a NON-streaming request, then a
// streaming request for the same identity replays from the cache.
func TestProxy_NonStreamingPopulatesThenStreamingHits(t *testing.T) {
	upstream := newFakeUpstream(t)
	upstream.script(chatPath, scriptedResponse{
		body: chatCompletionJSON(t, "the answer", "stop", 7),
	})
	proxy := newProxy(t, upstream, nil)

	nonStreamBody := `{"model":"gpt-4","messages":[{"role":"user","content":"cross mode prompt"}]}`
	streamBody := `{"model":"gpt-4","stream":true,"messages":[{"role":"user","content":"cross mode prompt"}]}`

	resp1, _ := postChat(t, proxy.URL, nonStreamBody, nil)
	if got := resp1.Header.Get("x-rostam-cache"); got != "miss" {
		t.Fatalf("non-streaming request x-rostam-cache = %q, want miss", got)
	}
	if n := upstream.requestCount(chatPath); n != 1 {
		t.Fatalf("upstream requests after non-streaming miss = %d, want 1", n)
	}

	resp2, body2 := postChat(t, proxy.URL, streamBody, nil)
	if resp2.StatusCode != http.StatusOK {
		t.Fatalf("streaming request status = %d, want 200", resp2.StatusCode)
	}
	if got := resp2.Header.Get("x-rostam-cache"); got != "hit" {
		t.Fatalf("streaming request x-rostam-cache = %q, want hit", got)
	}
	if n := upstream.requestCount(chatPath); n != 1 {
		t.Fatalf("upstream requests after streaming hit = %d, want still 1", n)
	}
	if got := assembleSSEContent(t, string(body2)); got != "the answer" {
		t.Fatalf("streaming hit assembled answer = %q, want %q", got, "the answer")
	}
}

// (b2) the reverse cross-mode direction: cache populated via a STREAMING
// request, then a non-streaming request for the same identity serves a hit.
func TestProxy_StreamingPopulatesThenNonStreamingHits(t *testing.T) {
	upstream := newFakeUpstream(t)
	upstream.script(chatPath, scriptedResponse{sse: true, lines: basicStreamLines})
	proxy := newProxy(t, upstream, nil)

	streamBody := `{"model":"gpt-4","stream":true,"messages":[{"role":"user","content":"reverse cross mode"}]}`
	nonStreamBody := `{"model":"gpt-4","messages":[{"role":"user","content":"reverse cross mode"}]}`

	resp1, _ := postChat(t, proxy.URL, streamBody, nil)
	if got := resp1.Header.Get("x-rostam-cache"); got != "miss" {
		t.Fatalf("streaming request x-rostam-cache = %q, want miss", got)
	}
	if n := upstream.requestCount(chatPath); n != 1 {
		t.Fatalf("upstream requests after streaming miss = %d, want 1", n)
	}

	resp2, body2 := postChat(t, proxy.URL, nonStreamBody, nil)
	if resp2.StatusCode != http.StatusOK {
		t.Fatalf("non-streaming request status = %d, want 200", resp2.StatusCode)
	}
	if got := resp2.Header.Get("x-rostam-cache"); got != "hit" {
		t.Fatalf("non-streaming request x-rostam-cache = %q, want hit", got)
	}
	if n := upstream.requestCount(chatPath); n != 1 {
		t.Fatalf("upstream requests after non-streaming hit = %d, want still 1", n)
	}

	var decoded chatResponse
	if err := json.Unmarshal(body2, &decoded); err != nil {
		t.Fatalf("unmarshal cached response: %v", err)
	}
	if len(decoded.Choices) != 1 || decoded.Choices[0].Message.Content != "Hello, world" {
		t.Fatalf("cached response content = %+v, want %q", decoded.Choices, "Hello, world")
	}
}

// (c) mid-stream abort: upstream cuts the connection before writing [DONE].
// The client still sees whatever was relayed before the drop, but nothing is
// stored, so a repeat of the same request misses again.
func TestProxy_StreamingMidStreamAbortNotStored(t *testing.T) {
	partialLines := []string{
		"data: {\"choices\":[{\"index\":0,\"delta\":{\"role\":\"assistant\"},\"finish_reason\":null}]}\n\n",
		"data: {\"choices\":[{\"index\":0,\"delta\":{\"content\":\"partial\"},\"finish_reason\":null}]}\n\n",
	}
	upstream := newFakeUpstream(t)
	upstream.script(chatPath, scriptedResponse{sse: true, lines: partialLines, abort: true})
	proxy := newProxy(t, upstream, nil)

	body := `{"model":"gpt-4","stream":true,"messages":[{"role":"user","content":"abort mid stream"}]}`

	resp1, body1 := postChat(t, proxy.URL, body, nil)
	if resp1.StatusCode != http.StatusOK {
		t.Fatalf("first request status = %d, want 200", resp1.StatusCode)
	}
	if got := resp1.Header.Get("x-rostam-cache"); got != "miss" {
		t.Fatalf("first request x-rostam-cache = %q, want miss", got)
	}
	wantBody := strings.Join(partialLines, "")
	if got := string(body1); got != wantBody {
		t.Fatalf("relayed (truncated) body = %q, want %q", got, wantBody)
	}
	if strings.Contains(string(body1), "[DONE]") {
		t.Fatalf("relayed body contains [DONE], want a truncated stream")
	}

	resp2, _ := postChat(t, proxy.URL, body, nil)
	if got := resp2.Header.Get("x-rostam-cache"); got != "miss" {
		t.Fatalf("repeat request x-rostam-cache = %q, want miss (aborted stream must not be stored)", got)
	}
	if n := upstream.requestCount(chatPath); n != 2 {
		t.Fatalf("upstream requests = %d, want 2 (both should reach upstream)", n)
	}
}

// (d) a streamed response carrying tool_calls is relayed to the client but
// must not be stored: a repeat of the same request misses again.
func TestProxy_StreamingToolCallsNotStored(t *testing.T) {
	lines := []string{
		"data: {\"choices\":[{\"index\":0,\"delta\":{\"tool_calls\":[{\"index\":0,\"id\":\"call_1\"}]},\"finish_reason\":null}]}\n\n",
		"data: {\"choices\":[{\"index\":0,\"delta\":{},\"finish_reason\":\"tool_calls\"}]}\n\n",
		"data: [DONE]\n\n",
	}
	upstream := newFakeUpstream(t)
	upstream.script(chatPath, scriptedResponse{sse: true, lines: lines})
	proxy := newProxy(t, upstream, nil)

	body := `{"model":"gpt-4","stream":true,"messages":[{"role":"user","content":"what's the weather"}],
		"tools":[{"function":{"name":"get_weather"}}]}`

	resp1, body1 := postChat(t, proxy.URL, body, nil)
	if resp1.StatusCode != http.StatusOK {
		t.Fatalf("first request status = %d, want 200", resp1.StatusCode)
	}
	if got := resp1.Header.Get("x-rostam-cache"); got != "miss" {
		t.Fatalf("first request x-rostam-cache = %q, want miss", got)
	}
	wantBody := strings.Join(lines, "")
	if got := string(body1); got != wantBody {
		t.Fatalf("relayed body = %q, want %q (byte-for-byte)", got, wantBody)
	}

	resp2, _ := postChat(t, proxy.URL, body, nil)
	if got := resp2.Header.Get("x-rostam-cache"); got != "miss" {
		t.Fatalf("repeat request x-rostam-cache = %q, want miss (tool_calls responses must not be stored)", got)
	}
	if n := upstream.requestCount(chatPath); n != 2 {
		t.Fatalf("upstream requests = %d, want 2 (both should reach upstream)", n)
	}
}

// (e) usage-chunk completion_tokens land in tokens_saved once the answer is
// served from a subsequent cache hit, and /stats counts streaming hits and
// misses exactly like non-streaming ones (a gap Task 5 left open).
func TestProxy_StreamingStatsAndTokensSaved(t *testing.T) {
	upstream := newFakeUpstream(t)
	upstream.script(chatPath, scriptedResponse{sse: true, lines: basicStreamLines})
	proxy := newProxy(t, upstream, nil)

	body := `{"model":"gpt-4","stream":true,"messages":[{"role":"user","content":"count my tokens"}]}`

	// miss, then store
	postChat(t, proxy.URL, body, nil)
	// hit
	postChat(t, proxy.URL, body, nil)

	statsResp, statsBody := getStats(t, proxy.URL)
	if statsResp.StatusCode != http.StatusOK {
		t.Fatalf("/stats status = %d, want 200", statsResp.StatusCode)
	}
	var st struct {
		Hits        float64 `json:"hits"`
		Misses      float64 `json:"misses"`
		Stored      float64 `json:"stored"`
		TokensSaved float64 `json:"tokens_saved"`
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
	// basicStreamLines' usage chunk carries completion_tokens: 9, captured by
	// relayAndCapture and stored; the subsequent hit credits it as saved.
	if st.TokensSaved != 9 {
		t.Fatalf("tokens_saved = %v, want 9", st.TokensSaved)
	}
}
