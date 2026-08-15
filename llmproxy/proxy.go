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
	"net/url"
	"strings"
	"sync/atomic"
	"time"

	"github.com/rostamlabs/rostam/semcache"
)

// maxBodyBytes bounds a chat-completions request body the proxy will parse
// for cache scoping. 16 MiB is generous for any real chat prompt; without a
// cap a client could make the proxy buffer an unbounded body in memory
// before it ever gets to json.Unmarshal.
const maxBodyBytes = 16 << 20

// hopByHopHeaders are stripped when relaying a request or response between
// the proxy and upstream/client, per RFC 7230 §6.1 — they describe THIS hop's
// connection, not the message, and forwarding them verbatim would leak or
// corrupt the next hop's own connection handling.
var hopByHopHeaders = []string{
	"Connection",
	"Keep-Alive",
	"Proxy-Authenticate",
	"Proxy-Authorization",
	"TE",
	"Trailer",
	"Transfer-Encoding",
	"Upgrade",
}

// Config configures a Server.
type Config struct {
	Cache    *semcache.Cache // required
	Upstream *url.URL        // required
	MaxTemp  float64         // cacheability ceiling (proxy-level; also set on the semcache Config by the caller)
	Mode     string          // "exact" | "semantic" — surfaced verbatim in /stats, not otherwise used here
	Logger   *slog.Logger
	Client   *http.Client // nil => &http.Client{Timeout: 5 * time.Minute}
}

// stats holds the proxy's request counters, exposed as JSON at GET /stats.
// Fields are atomic because requests are served concurrently.
type stats struct {
	hits        atomic.Int64
	misses      atomic.Int64
	stored      atomic.Int64
	uncacheable atomic.Int64
	tokensSaved atomic.Int64
}

// Server is the OpenAI-compatible caching reverse proxy. Safe for concurrent
// use (http.Handler is called concurrently by net/http already).
type Server struct {
	cache    *semcache.Cache
	upstream *url.URL
	maxTemp  float64
	mode     string
	log      *slog.Logger
	client   *http.Client
	stats    stats
}

// NewServer validates cfg and builds a Server.
func NewServer(cfg Config) (*Server, error) {
	if cfg.Cache == nil {
		return nil, errors.New("llmproxy: Config.Cache is required")
	}
	if cfg.Upstream == nil {
		return nil, errors.New("llmproxy: Config.Upstream is required")
	}

	log := cfg.Logger
	if log == nil {
		log = slog.Default()
	}
	client := cfg.Client
	if client == nil {
		client = &http.Client{Timeout: 5 * time.Minute}
	}

	return &Server{
		cache:    cfg.Cache,
		upstream: cfg.Upstream,
		maxTemp:  cfg.MaxTemp,
		mode:     cfg.Mode,
		log:      log,
		client:   client,
	}, nil
}

// Handler returns the proxy's http.Handler. Routing: POST
// /v1/chat/completions goes through the cache path; GET /stats returns
// counters; everything else is passed through to Upstream verbatim.
func (s *Server) Handler() http.Handler {
	mux := http.NewServeMux()
	mux.HandleFunc("POST /v1/chat/completions", s.handleChatCompletions)
	mux.HandleFunc("GET /stats", s.handleStats)
	mux.HandleFunc("/", s.handlePassthrough)
	return mux
}

// handleStats serves the current counters as JSON.
func (s *Server) handleStats(w http.ResponseWriter, r *http.Request) {
	body := map[string]any{
		"hits":         s.stats.hits.Load(),
		"misses":       s.stats.misses.Load(),
		"stored":       s.stats.stored.Load(),
		"uncacheable":  s.stats.uncacheable.Load(),
		"tokens_saved": s.stats.tokensSaved.Load(),
		"mode":         s.mode,
	}
	w.Header().Set("Content-Type", "application/json")
	if err := json.NewEncoder(w).Encode(body); err != nil {
		s.log.Error("llmproxy: encode /stats response", "err", err)
	}
}

// writeOpenAIError writes an OpenAI-shaped error body: clients of an
// OpenAI-compatible API expect {"error":{"message":...,"type":...}} even
// from proxy-local failures, not a bare status code or plain text.
func writeOpenAIError(w http.ResponseWriter, status int, message, errType string) {
	w.Header().Set("Content-Type", "application/json")
	w.WriteHeader(status)
	body := map[string]any{
		"error": map[string]any{
			"message": message,
			"type":    errType,
		},
	}
	// The shape above is fixed and always marshals cleanly.
	b, err := json.Marshal(body)
	if err != nil {
		panic("llmproxy: writeOpenAIError: " + err.Error())
	}
	_, _ = w.Write(b)
}

// handleChatCompletions implements the cache path for POST
// /v1/chat/completions: parse, check cacheability, look up, answer from
// cache or forward and store. Stream:true requests take the passthrough
// (uncached) branch in this task — a later task replaces that branch with
// real streaming-cache support; see the SEAM comment below.
func (s *Server) handleChatCompletions(w http.ResponseWriter, r *http.Request) {
	r.Body = http.MaxBytesReader(w, r.Body, maxBodyBytes)
	body, err := io.ReadAll(r.Body)
	if err != nil {
		if isMaxBytesError(err) {
			writeOpenAIError(w, http.StatusRequestEntityTooLarge,
				"request body exceeds the maximum allowed size", "invalid_request_error")
			return
		}
		writeOpenAIError(w, http.StatusBadRequest, err.Error(), "invalid_request_error")
		return
	}

	req, err := parseChatRequest(body)
	if err != nil {
		writeOpenAIError(w, http.StatusBadRequest, "invalid JSON: "+err.Error(), "invalid_request_error")
		return
	}

	tenant := tenantOf(r.Header.Get("Authorization"))
	prompt, scope, ok := req.cacheIdentity(tenant)
	if !ok || !s.cache.Cacheable(req.temperature()) {
		s.stats.uncacheable.Add(1)
		s.passthroughRequest(w, r, body)
		return
	}

	// SEAM: Stream:true requests are uncached passthrough here; a later task
	// replaces this branch with a caching-aware SSE relay (cache hit -> replay
	// as SSE, miss -> relay + capture via relayAndCapture).
	if req.Stream {
		s.passthroughRequest(w, r, body)
		return
	}

	hit, found, err := s.cache.Lookup(r.Context(), prompt, scope)
	switch {
	case err != nil:
		// Lookup errored: treat as a miss but tell the client the cache was
		// bypassed rather than claiming a clean miss it didn't get.
		s.log.Error("llmproxy: cache lookup failed", "err", err)
		s.forwardAndMaybeStore(w, r, req, prompt, scope, "bypass")
	case found:
		s.stats.hits.Add(1)
		s.stats.tokensSaved.Add(int64(hit.OutTokens))
		w.Header().Set("x-rostam-cache", "hit")
		w.Header().Set("Content-Type", "application/json")
		w.WriteHeader(http.StatusOK)
		_, _ = w.Write(synthesizeResponse(req.Model, hit.Answer, hit.OutTokens))
	default:
		s.forwardAndMaybeStore(w, r, req, prompt, scope, "miss")
	}
}

// isMaxBytesError reports whether err came from http.MaxBytesReader's limit,
// across Go versions that either expose *http.MaxBytesError or a plain
// "http: request body too large" message.
func isMaxBytesError(err error) bool {
	var mbErr *http.MaxBytesError
	if errors.As(err, &mbErr) {
		return true
	}
	return strings.Contains(err.Error(), "request body too large")
}

// forwardAndMaybeStore forwards req.Raw upstream, relays the response
// verbatim to the client with the given cache-status header, and — for a
// cacheable-shaped success — stores the answer for next time.
func (s *Server) forwardAndMaybeStore(w http.ResponseWriter, r *http.Request, req *chatRequest,
	prompt string, scope semcache.Scope, cacheStatus string) {
	s.stats.misses.Add(1)

	upstreamReq, err := s.newUpstreamRequest(r, req.Raw)
	if err != nil {
		writeOpenAIError(w, http.StatusBadGateway, "building upstream request: "+err.Error(), "upstream_error")
		return
	}

	resp, err := s.client.Do(upstreamReq)
	if err != nil {
		s.log.Error("llmproxy: upstream request failed", "err", err)
		writeOpenAIError(w, http.StatusBadGateway, "upstream request failed: "+err.Error(), "upstream_error")
		return
	}
	defer resp.Body.Close()

	respBody, err := io.ReadAll(resp.Body)
	if err != nil {
		s.log.Error("llmproxy: reading upstream response", "err", err)
		writeOpenAIError(w, http.StatusBadGateway, "reading upstream response: "+err.Error(), "upstream_error")
		return
	}

	copyHeadersExceptHopByHop(w.Header(), resp.Header)
	w.Header().Set("x-rostam-cache", cacheStatus)
	w.WriteHeader(resp.StatusCode)
	_, _ = w.Write(respBody)

	if resp.StatusCode == http.StatusOK {
		s.maybeStore(r.Context(), req.Model, prompt, scope, respBody)
	}
}

// maybeStore decodes an upstream chat-completions response and stores it in
// the cache if it has the shape a cached answer can safely stand in for:
// exactly one choice, finish_reason "stop", no tool calls, non-empty
// content. Anything else (multiple choices already excluded upstream by
// cacheIdentity's n>1 check, but a defensively-checked finish reason or a
// tool call) means the response isn't a plain completion a synthesized reply
// could faithfully replay.
func (s *Server) maybeStore(ctx context.Context, model, prompt string, scope semcache.Scope, respBody []byte) {
	var resp chatResponse
	if err := json.Unmarshal(respBody, &resp); err != nil {
		s.log.Error("llmproxy: decode upstream response for cache store", "err", err)
		return
	}
	if len(resp.Choices) != 1 {
		return
	}
	choice := resp.Choices[0]
	if choice.FinishReason != "stop" {
		return
	}
	if toolCallsPresent(choice.Message.ToolCalls) {
		return
	}
	if choice.Message.Content == "" {
		return
	}

	if err := s.cache.Store(ctx, prompt, scope, choice.Message.Content, resp.Usage.CompletionTokens); err != nil {
		s.log.Error("llmproxy: cache store failed", "err", err)
		return
	}
	s.stats.stored.Add(1)
}

// passthroughRequest relays r to Upstream verbatim, streaming the response
// body back to the client. body is the already-read request body (the
// caller has consumed r.Body via io.ReadAll for size-capping/parsing, so it
// can't be read from r again).
func (s *Server) passthroughRequest(w http.ResponseWriter, r *http.Request, body []byte) {
	upstreamReq, err := s.newUpstreamRequest(r, body)
	if err != nil {
		writeOpenAIError(w, http.StatusBadGateway, "building upstream request: "+err.Error(), "upstream_error")
		return
	}

	resp, err := s.client.Do(upstreamReq)
	if err != nil {
		s.log.Error("llmproxy: upstream request failed", "err", err)
		writeOpenAIError(w, http.StatusBadGateway, "upstream request failed: "+err.Error(), "upstream_error")
		return
	}
	defer resp.Body.Close()

	copyHeadersExceptHopByHop(w.Header(), resp.Header)
	w.WriteHeader(resp.StatusCode)
	// Stream rather than buffer: passthrough also carries SSE bodies (a
	// stream:true request that took this branch), and buffering the whole
	// response would defeat the point of a stream.
	if _, err := io.Copy(w, resp.Body); err != nil {
		s.log.Error("llmproxy: relaying passthrough response body", "err", err)
	}
}

// handlePassthrough relays any request not matched by a more specific route
// to Upstream verbatim: method, path, query, headers (minus hop-by-hop) and
// body all preserved, response streamed back unmodified.
func (s *Server) handlePassthrough(w http.ResponseWriter, r *http.Request) {
	body, err := io.ReadAll(r.Body)
	if err != nil {
		writeOpenAIError(w, http.StatusBadRequest, err.Error(), "invalid_request_error")
		return
	}
	s.passthroughRequest(w, r, body)
}

// newUpstreamRequest builds the request the proxy sends to Upstream for an
// incoming client request r, rebuilding the URL on Upstream's scheme/host
// while keeping r's path and query, and copying method/headers (minus
// hop-by-hop) with the given body.
func (s *Server) newUpstreamRequest(r *http.Request, body []byte) (*http.Request, error) {
	u := *s.upstream
	u.Path = singleJoiningSlash(s.upstream.Path, r.URL.Path)
	u.RawQuery = r.URL.RawQuery

	upstreamReq, err := http.NewRequestWithContext(r.Context(), r.Method, u.String(), bytes.NewReader(body))
	if err != nil {
		return nil, fmt.Errorf("llmproxy: new upstream request: %w", err)
	}
	copyHeadersExceptHopByHop(upstreamReq.Header, r.Header)
	upstreamReq.Host = s.upstream.Host
	upstreamReq.ContentLength = int64(len(body))
	return upstreamReq, nil
}

// singleJoiningSlash joins a base path and a request path with exactly one
// slash between them, mirroring net/http/httputil.ReverseProxy's helper —
// Upstream is typically bare ("https://api.openai.com", empty Path) but
// mustn't produce a doubled or missing slash if it ever carries one.
func singleJoiningSlash(base, ref string) string {
	baseSlash := strings.HasSuffix(base, "/")
	refSlash := strings.HasPrefix(ref, "/")
	switch {
	case baseSlash && refSlash:
		return base + ref[1:]
	case !baseSlash && !refSlash:
		return base + "/" + ref
	default:
		return base + ref
	}
}

// copyHeadersExceptHopByHop copies every header from src to dst except the
// hop-by-hop set, which describes the connection between two specific peers
// and must not be forwarded to the next hop.
func copyHeadersExceptHopByHop(dst, src http.Header) {
	for k, vv := range src {
		if isHopByHop(k) {
			continue
		}
		for _, v := range vv {
			dst.Add(k, v)
		}
	}
}

func isHopByHop(header string) bool {
	for _, h := range hopByHopHeaders {
		if strings.EqualFold(h, header) {
			return true
		}
	}
	return false
}
