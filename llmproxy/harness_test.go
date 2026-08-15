// SPDX-License-Identifier: Apache-2.0

package llmproxy

import (
	"compress/gzip"
	"context"
	"encoding/json"
	"io"
	"net/http"
	"net/http/httptest"
	"net/url"
	"strings"
	"sync"
	"testing"

	"github.com/rostamlabs/rostam"
	"github.com/rostamlabs/rostam/ops"
	"github.com/rostamlabs/rostam/semcache"
)

// capturedRequest records what fakeUpstream saw for a single request, so
// tests can assert on routing (path, method, headers) without the fake
// server needing to understand OpenAI's wire format itself.
type capturedRequest struct {
	Method  string
	Path    string
	Headers http.Header
	Body    []byte
}

// scriptedResponse is one canned reply fakeUpstream serves for a path: either
// a plain JSON/status response, or a scripted SSE body written line-by-line
// with a flush after each line (abort stops mid-script, simulating a
// connection drop).
type scriptedResponse struct {
	status  int
	body    []byte // used when sseLines is nil
	headers map[string]string
	sse     bool
	lines   []string // raw SSE lines (including "\n"), written+flushed one at a time
	abort   bool     // stop after writing lines without a clean close (Task 6 use)
	// gzipIfAccepted makes the fake behave like every real OpenAI-compatible
	// endpoint: compress body whenever the request says it accepts gzip. Only
	// a fake that does this can catch a proxy that reads a compressed body as
	// if it were JSON.
	gzipIfAccepted bool
}

// fakeUpstream is a scriptable stand-in for the real OpenAI-compatible
// upstream: tests register a canned response per path, then assert on what
// the proxy forwarded (method, headers, body) via Requests.
type fakeUpstream struct {
	srv *httptest.Server

	mu       sync.Mutex
	scripts  map[string]scriptedResponse
	requests []capturedRequest
}

// newFakeUpstream starts the fake server. Callers script responses with
// script() before pointing a proxy at it.
func newFakeUpstream(t *testing.T) *fakeUpstream {
	t.Helper()
	fu := &fakeUpstream{scripts: make(map[string]scriptedResponse)}
	fu.srv = httptest.NewServer(http.HandlerFunc(fu.handle))
	t.Cleanup(fu.srv.Close)
	return fu
}

// script registers the canned response fakeUpstream serves for path,
// overwriting any previous script for that path.
func (fu *fakeUpstream) script(path string, resp scriptedResponse) {
	fu.mu.Lock()
	defer fu.mu.Unlock()
	fu.scripts[path] = resp
}

// requestCount returns how many requests fakeUpstream has received for path,
// so tests can assert a cache hit never touched upstream.
func (fu *fakeUpstream) requestCount(path string) int {
	fu.mu.Lock()
	defer fu.mu.Unlock()
	n := 0
	for _, r := range fu.requests {
		if r.Path == path {
			n++
		}
	}
	return n
}

// lastRequest returns the most recently captured request for path, or nil.
func (fu *fakeUpstream) lastRequest(path string) *capturedRequest {
	fu.mu.Lock()
	defer fu.mu.Unlock()
	for i := len(fu.requests) - 1; i >= 0; i-- {
		if fu.requests[i].Path == path {
			r := fu.requests[i]
			return &r
		}
	}
	return nil
}

func (fu *fakeUpstream) handle(w http.ResponseWriter, r *http.Request) {
	body, _ := io.ReadAll(r.Body)

	fu.mu.Lock()
	fu.requests = append(fu.requests, capturedRequest{
		Method:  r.Method,
		Path:    r.URL.Path,
		Headers: r.Header.Clone(),
		Body:    body,
	})
	resp, ok := fu.scripts[r.URL.Path]
	fu.mu.Unlock()

	if !ok {
		w.WriteHeader(http.StatusNotFound)
		return
	}

	for k, v := range resp.headers {
		w.Header().Set(k, v)
	}

	if resp.sse {
		w.Header().Set("Content-Type", "text/event-stream")
		status := resp.status
		if status == 0 {
			status = http.StatusOK
		}
		w.WriteHeader(status)
		fl, _ := w.(http.Flusher)
		for _, line := range resp.lines {
			_, _ = io.WriteString(w, line)
			if fl != nil {
				fl.Flush()
			}
		}
		if resp.abort {
			// Simulate a dropped connection: close the underlying conn
			// without writing a clean terminator.
			if hj, ok := w.(http.Hijacker); ok {
				conn, _, err := hj.Hijack()
				if err == nil {
					_ = conn.Close()
				}
			}
		}
		return
	}

	status := resp.status
	if status == 0 {
		status = http.StatusOK
	}

	if resp.gzipIfAccepted && strings.Contains(r.Header.Get("Accept-Encoding"), "gzip") {
		w.Header().Set("Content-Encoding", "gzip")
		w.WriteHeader(status)
		zw := gzip.NewWriter(w)
		_, _ = zw.Write(resp.body)
		_ = zw.Close()
		return
	}

	w.WriteHeader(status)
	_, _ = w.Write(resp.body)
}

// newProxy builds a proxy.Server backed by a heap Direct store and a
// semcache.Cache pointed at upstream, and returns it as an httptest.Server
// along with the fakeUpstream it targets. embedder nil selects the
// deterministic stub embedder ("exact" mode, dim 64) used by default across
// Tasks 5-6's tests.
func newProxy(t *testing.T, upstream *fakeUpstream, embedder semcache.Embedder) *httptest.Server {
	t.Helper()
	proxy := httptest.NewServer(newProxyServer(t, upstream, embedder).Handler())
	t.Cleanup(proxy.Close)
	return proxy
}

// newProxyServer is newProxy without the http listener, for a test that needs
// to drive Handler() with its own ResponseWriter (to observe flushes, say)
// rather than talk to it over a real socket.
func newProxyServer(t *testing.T, upstream *fakeUpstream, embedder semcache.Embedder) *Server {
	t.Helper()

	if embedder == nil {
		embedder = semcache.NewStubEmbedder("exact", 64)
	}

	reg := ops.NewRegistry()
	if err := ops.RegisterBuiltins(reg); err != nil {
		t.Fatalf("RegisterBuiltins: %v", err)
	}
	store, err := rostam.NewDirect(rostam.DirectConfig{Ops: reg})
	if err != nil {
		t.Fatalf("NewDirect: %v", err)
	}
	t.Cleanup(func() { _ = store.Close() })

	cache, err := semcache.New(context.Background(), semcache.Config{
		Store:      store,
		Embedder:   embedder,
		Collection: "llm-cache",
		Threshold:  0.999,
		MaxTemp:    1.0,
	})
	if err != nil {
		t.Fatalf("semcache.New: %v", err)
	}

	upstreamURL, err := url.Parse(upstream.srv.URL)
	if err != nil {
		t.Fatalf("parse upstream URL: %v", err)
	}

	srv, err := NewServer(Config{
		Cache:    cache,
		Upstream: upstreamURL,
		Mode:     "exact",
	})
	if err != nil {
		t.Fatalf("NewServer: %v", err)
	}
	return srv
}

// postChat POSTs body to proxyURL+"/v1/chat/completions" with headers merged
// on top of a default Content-Type, and returns the response plus its fully
// read body.
func postChat(t *testing.T, proxyURL, body string, headers map[string]string) (*http.Response, []byte) {
	t.Helper()

	req, err := http.NewRequest(http.MethodPost, proxyURL+"/v1/chat/completions", strings.NewReader(body))
	if err != nil {
		t.Fatalf("NewRequest: %v", err)
	}
	req.Header.Set("Content-Type", "application/json")
	for k, v := range headers {
		req.Header.Set(k, v)
	}

	resp, err := http.DefaultClient.Do(req)
	if err != nil {
		t.Fatalf("Do: %v", err)
	}
	defer resp.Body.Close()

	respBody, err := io.ReadAll(resp.Body)
	if err != nil {
		t.Fatalf("ReadAll: %v", err)
	}
	return resp, respBody
}

// chatCompletionJSON builds a canned non-streaming chat-completions upstream
// response body with the given content, finish reason and completion-token
// count, for use with fakeUpstream.script.
func chatCompletionJSON(t *testing.T, content, finishReason string, completionTokens int) []byte {
	t.Helper()
	resp := map[string]any{
		"id":     "chatcmpl-fake",
		"object": "chat.completion",
		"model":  "gpt-4",
		"choices": []map[string]any{
			{
				"index": 0,
				"message": map[string]any{
					"role":    "assistant",
					"content": content,
				},
				"finish_reason": finishReason,
			},
		},
		"usage": map[string]any{
			"completion_tokens": completionTokens,
		},
	}
	b, err := json.Marshal(resp)
	if err != nil {
		t.Fatalf("marshal: %v", err)
	}
	return b
}
