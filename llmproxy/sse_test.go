// SPDX-License-Identifier: Apache-2.0

package llmproxy

import (
	"bufio"
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"
)

// nonFlushingWriter satisfies http.ResponseWriter but deliberately does not
// implement http.Flusher, so it exercises newSSEWriter's rejection path.
type nonFlushingWriter struct {
	header http.Header
}

func (w *nonFlushingWriter) Header() http.Header         { return w.header }
func (w *nonFlushingWriter) Write(b []byte) (int, error) { return len(b), nil }
func (w *nonFlushingWriter) WriteHeader(int)             {}

func TestNewSSEWriter_RejectsNonFlusher(t *testing.T) {
	w := &nonFlushingWriter{header: http.Header{}}
	_, err := newSSEWriter(w)
	if err == nil {
		t.Fatalf("newSSEWriter err = nil, want error for non-Flusher writer")
	}
}

func TestNewSSEWriter_SetsHeaders(t *testing.T) {
	rec := httptest.NewRecorder()
	if _, err := newSSEWriter(rec); err != nil {
		t.Fatalf("newSSEWriter: %v", err)
	}
	if got := rec.Header().Get("Content-Type"); got != "text/event-stream" {
		t.Fatalf("Content-Type = %q, want text/event-stream", got)
	}
	if got := rec.Header().Get("Cache-Control"); got != "no-cache" {
		t.Fatalf("Cache-Control = %q, want no-cache", got)
	}
}

// dataEvents splits an SSE body into the payloads of its "data: " lines, in
// order, matching how a real client would parse the stream.
func dataEvents(t *testing.T, body string) []string {
	t.Helper()
	var events []string
	sc := bufio.NewScanner(strings.NewReader(body))
	for sc.Scan() {
		line := sc.Text()
		if data, ok := strings.CutPrefix(line, "data: "); ok {
			events = append(events, data)
		}
	}
	if err := sc.Err(); err != nil {
		t.Fatalf("scan body: %v", err)
	}
	return events
}

type chunkChoice struct {
	Index        int            `json:"index"`
	Delta        map[string]any `json:"delta"`
	FinishReason string         `json:"finish_reason"`
}

type chunk struct {
	ID      string        `json:"id"`
	Object  string        `json:"object"`
	Model   string        `json:"model"`
	Choices []chunkChoice `json:"choices"`
}

func TestReplayAsSSE_ParsesAsThreeChunksThenDone(t *testing.T) {
	rec := httptest.NewRecorder()
	if err := replayAsSSE(rec, "gpt-4", "hello world"); err != nil {
		t.Fatalf("replayAsSSE: %v", err)
	}

	events := dataEvents(t, rec.Body.String())
	if len(events) != 4 {
		t.Fatalf("len(events) = %d, want 4 (3 chunks + [DONE])", len(events))
	}
	if events[3] != "[DONE]" {
		t.Fatalf("last event = %q, want [DONE]", events[3])
	}

	var roleChunk chunk
	if err := json.Unmarshal([]byte(events[0]), &roleChunk); err != nil {
		t.Fatalf("unmarshal chunk 0: %v", err)
	}
	if roleChunk.ID != "rostam-cache" {
		t.Fatalf("chunk 0 id = %q, want rostam-cache", roleChunk.ID)
	}
	if roleChunk.Object != "chat.completion.chunk" {
		t.Fatalf("chunk 0 object = %q, want chat.completion.chunk", roleChunk.Object)
	}
	if roleChunk.Model != "gpt-4" {
		t.Fatalf("chunk 0 model = %q, want gpt-4", roleChunk.Model)
	}
	if role, _ := roleChunk.Choices[0].Delta["role"].(string); role != "assistant" {
		t.Fatalf("chunk 0 delta.role = %q, want assistant", role)
	}

	var contentChunk chunk
	if err := json.Unmarshal([]byte(events[1]), &contentChunk); err != nil {
		t.Fatalf("unmarshal chunk 1: %v", err)
	}
	if content, _ := contentChunk.Choices[0].Delta["content"].(string); content != "hello world" {
		t.Fatalf("chunk 1 delta.content = %q, want hello world", content)
	}

	var finishChunk chunk
	if err := json.Unmarshal([]byte(events[2]), &finishChunk); err != nil {
		t.Fatalf("unmarshal chunk 2: %v", err)
	}
	if finishChunk.Choices[0].FinishReason != "stop" {
		t.Fatalf("chunk 2 finish_reason = %q, want stop", finishChunk.Choices[0].FinishReason)
	}
}

// scriptedStream is a realistic upstream chat-completions SSE body: a role
// chunk, three content deltas, a finish chunk, a usage chunk, and [DONE],
// with a comment line and an event: line mixed in to exercise pass-through
// of non-data SSE fields.
const scriptedStream = ": ping\n" +
	"data: {\"choices\":[{\"index\":0,\"delta\":{\"role\":\"assistant\"},\"finish_reason\":null}]}\n\n" +
	"event: message\n" +
	"data: {\"choices\":[{\"index\":0,\"delta\":{\"content\":\"Hel\"},\"finish_reason\":null}]}\n\n" +
	"data: {\"choices\":[{\"index\":0,\"delta\":{\"content\":\"lo, \"},\"finish_reason\":null}]}\n\n" +
	"data: {\"choices\":[{\"index\":0,\"delta\":{\"content\":\"world\"},\"finish_reason\":null}]}\n\n" +
	"data: {\"choices\":[{\"index\":0,\"delta\":{},\"finish_reason\":\"stop\"}]}\n\n" +
	"data: {\"choices\":[],\"usage\":{\"completion_tokens\":42}}\n\n" +
	"data: [DONE]\n\n"

func TestRelayAndCapture_ForwardsByteIdenticalAndAssembles(t *testing.T) {
	rec := httptest.NewRecorder()
	result, err := relayAndCapture(rec, strings.NewReader(scriptedStream))
	if err != nil {
		t.Fatalf("relayAndCapture: %v", err)
	}

	if got := rec.Body.String(); got != scriptedStream {
		t.Fatalf("relayed body = %q, want %q", got, scriptedStream)
	}
	if got := result.content.String(); got != "Hello, world" {
		t.Fatalf("content = %q, want %q", got, "Hello, world")
	}
	if result.finishReason != "stop" {
		t.Fatalf("finishReason = %q, want stop", result.finishReason)
	}
	if result.completionTokens != 42 {
		t.Fatalf("completionTokens = %d, want 42", result.completionTokens)
	}
	if !result.clean {
		t.Fatalf("clean = false, want true ([DONE] was present)")
	}
	if result.sawToolCalls {
		t.Fatalf("sawToolCalls = true, want false")
	}
}

func TestRelayAndCapture_ToolCallsSetsFlag(t *testing.T) {
	stream := "data: {\"choices\":[{\"index\":0,\"delta\":{\"tool_calls\":[{\"index\":0,\"id\":\"call_1\"}]},\"finish_reason\":null}]}\n\n" +
		"data: {\"choices\":[{\"index\":0,\"delta\":{},\"finish_reason\":\"tool_calls\"}]}\n\n" +
		"data: [DONE]\n\n"

	rec := httptest.NewRecorder()
	result, err := relayAndCapture(rec, strings.NewReader(stream))
	if err != nil {
		t.Fatalf("relayAndCapture: %v", err)
	}
	if !result.sawToolCalls {
		t.Fatalf("sawToolCalls = false, want true")
	}
	if result.finishReason != "tool_calls" {
		t.Fatalf("finishReason = %q, want tool_calls", result.finishReason)
	}
	if !result.clean {
		t.Fatalf("clean = false, want true")
	}
}

func TestRelayAndCapture_NullToolCallsNotFlagged(t *testing.T) {
	// vLLM/Azure and other OpenAI-compatible backends emit "tool_calls":
	// null on ordinary content deltas; that must not be misread as tool
	// calls present.
	stream := "data: {\"choices\":[{\"index\":0,\"delta\":{\"content\":\"hi\",\"tool_calls\":null},\"finish_reason\":null}]}\n\n" +
		"data: [DONE]\n\n"

	rec := httptest.NewRecorder()
	result, err := relayAndCapture(rec, strings.NewReader(stream))
	if err != nil {
		t.Fatalf("relayAndCapture: %v", err)
	}
	if result.sawToolCalls {
		t.Fatalf("sawToolCalls = true, want false (tool_calls: null is not tool calls)")
	}
	if result.content.String() != "hi" {
		t.Fatalf("content = %q, want hi", result.content.String())
	}
}

func TestRelayAndCapture_TruncatedStreamNotClean(t *testing.T) {
	stream := "data: {\"choices\":[{\"index\":0,\"delta\":{\"content\":\"partial\"},\"finish_reason\":null}]}\n\n"

	rec := httptest.NewRecorder()
	result, err := relayAndCapture(rec, strings.NewReader(stream))
	if err != nil {
		t.Fatalf("relayAndCapture: %v", err)
	}
	if result.clean {
		t.Fatalf("clean = true, want false (no [DONE] in stream)")
	}
	if result.content.String() != "partial" {
		t.Fatalf("content = %q, want partial", result.content.String())
	}
}

func TestRelayAndCapture_NonDataLinesForwardedUntouched(t *testing.T) {
	stream := ": keep-alive\n" +
		"event: ping\n" +
		"id: 123\n\n" +
		"data: [DONE]\n\n"

	rec := httptest.NewRecorder()
	_, err := relayAndCapture(rec, strings.NewReader(stream))
	if err != nil {
		t.Fatalf("relayAndCapture: %v", err)
	}
	if got := rec.Body.String(); got != stream {
		t.Fatalf("relayed body = %q, want %q", got, stream)
	}
}

func TestRelayAndCapture_CRLFForwardedByteIdentical(t *testing.T) {
	stream := "data: {\"choices\":[{\"index\":0,\"delta\":{\"content\":\"hi\"},\"finish_reason\":null}]}\r\n\r\n" +
		"data: [DONE]\r\n\r\n"

	rec := httptest.NewRecorder()
	result, err := relayAndCapture(rec, strings.NewReader(stream))
	if err != nil {
		t.Fatalf("relayAndCapture: %v", err)
	}
	if got := rec.Body.String(); got != stream {
		t.Fatalf("relayed body = %q, want %q (CRLF terminators must be preserved)", got, stream)
	}
	if result.content.String() != "hi" {
		t.Fatalf("content = %q, want hi", result.content.String())
	}
	if !result.clean {
		t.Fatalf("clean = false, want true")
	}
}

func TestRelayAndCapture_OversizedLineErrors(t *testing.T) {
	huge := strings.Repeat("a", sseMaxLine+1) // one line, no newline, past the scanner bound

	rec := httptest.NewRecorder()
	_, err := relayAndCapture(rec, strings.NewReader(huge))
	if err == nil {
		t.Fatalf("relayAndCapture err = nil, want error for an oversized line")
	}
}
