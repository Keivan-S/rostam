// SPDX-License-Identifier: Apache-2.0

package llmproxy

import (
	"bufio"
	"bytes"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"net/http"
	"strings"
	"time"
)

// sseMaxLine bounds a single upstream SSE line, matching mcp/jsonrpc.go's
// maxLine so an upstream that never terminates a line cannot make the proxy
// buffer unboundedly.
const sseMaxLine = 16 << 20

// sseWriter emits Server-Sent Events to an http.ResponseWriter, flushing
// after every event so the client sees each chunk as it's produced rather
// than the whole response buffered until the handler returns.
type sseWriter struct {
	w  http.ResponseWriter
	fl http.Flusher
}

// newSSEWriter sets the SSE response headers and returns a writer. It errors
// if w cannot flush — without that, a "stream" would just buffer behind
// whatever the transport does by default and defeat the point of SSE.
func newSSEWriter(w http.ResponseWriter) (*sseWriter, error) {
	fl, ok := w.(http.Flusher)
	if !ok {
		return nil, errors.New("llmproxy: response writer does not support flushing")
	}
	w.Header().Set("Content-Type", "text/event-stream")
	w.Header().Set("Cache-Control", "no-cache")
	return &sseWriter{w: w, fl: fl}, nil
}

// event writes one SSE data frame and flushes it to the client immediately.
func (s *sseWriter) event(data []byte) error {
	if _, err := s.w.Write([]byte("data: ")); err != nil {
		return err
	}
	if _, err := s.w.Write(data); err != nil {
		return err
	}
	if _, err := s.w.Write([]byte("\n\n")); err != nil {
		return err
	}
	s.fl.Flush()
	return nil
}

// done writes the terminal [DONE] sentinel OpenAI-compatible clients expect
// to see before they stop reading the stream.
func (s *sseWriter) done() error {
	if _, err := s.w.Write([]byte("data: [DONE]\n\n")); err != nil {
		return err
	}
	s.fl.Flush()
	return nil
}

// replayAsSSE writes a cached answer back as a minimal valid chat-completions
// stream, so a client that requested stream=true can't tell a cache hit from
// a real upstream stream. Three chunks (role, content, finish) mirror the
// smallest shape a real OpenAI stream ever produces.
func replayAsSSE(w http.ResponseWriter, model, answer string) error {
	sw, err := newSSEWriter(w)
	if err != nil {
		return err
	}

	// One timestamp shared across every chunk in the stream, the way a real
	// upstream stream carries the same created value from its first chunk to
	// its last — one call to time.Now, not one per chunk.
	created := time.Now().Unix()

	chunks := []map[string]any{
		{
			"id":      "rostam-cache",
			"object":  "chat.completion.chunk",
			"created": created,
			"model":   model,
			"choices": []map[string]any{
				{"index": 0, "delta": map[string]any{"role": "assistant"}},
			},
		},
		{
			"id":      "rostam-cache",
			"object":  "chat.completion.chunk",
			"created": created,
			"model":   model,
			"choices": []map[string]any{
				{"index": 0, "delta": map[string]any{"content": answer}},
			},
		},
		{
			"id":      "rostam-cache",
			"object":  "chat.completion.chunk",
			"created": created,
			"model":   model,
			"choices": []map[string]any{
				{"index": 0, "delta": map[string]any{}, "finish_reason": "stop"},
			},
		},
	}

	for _, c := range chunks {
		// The shapes above are fixed and always marshal cleanly; an error
		// here would mean a bug in this function, not bad input.
		b, err := json.Marshal(c)
		if err != nil {
			panic("llmproxy: replayAsSSE: " + err.Error())
		}
		if err := sw.event(b); err != nil {
			return err
		}
	}
	return sw.done()
}

// sseChunk is the tolerant decode shape for one upstream chat-completions
// stream chunk: just enough to update streamResult without choking on fields
// OpenAI adds over time.
type sseChunk struct {
	Choices []struct {
		Delta struct {
			Content   string          `json:"content"`
			ToolCalls json.RawMessage `json:"tool_calls,omitempty"`
		} `json:"delta"`
		FinishReason string `json:"finish_reason"`
	} `json:"choices"`
	Usage struct {
		CompletionTokens int `json:"completion_tokens"`
	} `json:"usage"`
}

// rawFieldPresent reports whether a captured json.RawMessage field carries a
// value. Absent (len 0) and an explicit null both count as absent: several
// OpenAI-compatible backends (vLLM, Azure) send "field": null where OpenAI
// omits the field, and json.RawMessage captures that literal, so a bare
// len(raw) > 0 check would read every such request or chunk as carrying
// something it doesn't.
func rawFieldPresent(raw json.RawMessage) bool {
	if len(raw) == 0 {
		return false
	}
	return string(bytes.TrimSpace(raw)) != "null"
}

// toolCallsPresent reports whether a decoded delta.tool_calls field actually
// carries tool calls — see rawFieldPresent for why an explicit null matters
// here: misreading it would wrongly mark plain-text answers uncacheable.
func toolCallsPresent(raw json.RawMessage) bool {
	return rawFieldPresent(raw)
}

// scanRawLines is like bufio.ScanLines but keeps each line's terminator
// (LF or CRLF) attached to the returned token instead of stripping it. A
// relay that re-appends its own "\n" after ScanLines strips \r silently
// turns CRLF upstream input into LF-only output; returning the terminator
// verbatim lets the caller forward exactly the bytes it read.
func scanRawLines(data []byte, atEOF bool) (advance int, token []byte, err error) {
	if atEOF && len(data) == 0 {
		return 0, nil, nil
	}
	if i := bytes.IndexByte(data, '\n'); i >= 0 {
		return i + 1, data[0 : i+1], nil
	}
	if atEOF {
		return len(data), data, nil
	}
	// Request more data.
	return 0, nil, nil
}

// streamResult accumulates what the cache needs from an upstream SSE stream
// as it's relayed to the client.
type streamResult struct {
	content          strings.Builder
	finishReason     string
	sawToolCalls     bool
	completionTokens int  // from a usage-bearing chunk when present; else 0
	clean            bool // saw [DONE] with no relay error
}

// relayAndCapture relays an upstream SSE body to the client line-for-line
// while accumulating the assembled answer, so a streamed response can still
// be cached without buffering the whole thing in memory before forwarding
// it. Every raw line is forwarded verbatim — comments, event: lines, and
// blank separators included — the proxy doesn't get to decide which SSE
// fields the client cares about; only data: lines are decoded, and
// tolerantly, for capture.
func relayAndCapture(dst http.ResponseWriter, src io.Reader) (streamResult, error) {
	var result streamResult

	fl, ok := dst.(http.Flusher)
	if !ok {
		return result, errors.New("llmproxy: response writer does not support flushing")
	}

	sc := bufio.NewScanner(src)
	sc.Buffer(make([]byte, 64<<10), sseMaxLine)
	sc.Split(scanRawLines)

	for sc.Scan() {
		raw := sc.Bytes() // includes its original terminator (LF or CRLF), or none at EOF
		if _, err := dst.Write(raw); err != nil {
			return result, err
		}
		fl.Flush()

		line := bytes.TrimRight(raw, "\r\n")
		data, isData := strings.CutPrefix(string(line), "data:")
		if !isData {
			continue
		}
		// Per the SSE spec a single space after the colon is optional
		// padding, not part of the value. OpenAI sends it; other
		// OpenAI-compatible servers write "data:{...}" and used to be
		// captured as nothing at all — the client still got the stream
		// relayed verbatim above, but the answer was never cached.
		data = strings.TrimPrefix(data, " ")
		if data == "[DONE]" {
			result.clean = true
			continue
		}

		var chunk sseChunk
		if err := json.Unmarshal([]byte(data), &chunk); err != nil {
			// Capture is best-effort: the client already got the raw bytes
			// above, so a chunk shape we don't recognize just doesn't
			// contribute to the accumulated result.
			continue
		}
		for _, ch := range chunk.Choices {
			result.content.WriteString(ch.Delta.Content)
			if toolCallsPresent(ch.Delta.ToolCalls) {
				result.sawToolCalls = true
			}
			if ch.FinishReason != "" {
				result.finishReason = ch.FinishReason
			}
		}
		if chunk.Usage.CompletionTokens > 0 {
			result.completionTokens = chunk.Usage.CompletionTokens
		}
	}
	if err := sc.Err(); err != nil {
		return result, fmt.Errorf("llmproxy: relay: %w", err)
	}
	return result, nil
}
