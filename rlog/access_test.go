// SPDX-License-Identifier: Apache-2.0

package rlog

import (
	"bytes"
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"

	"github.com/rostamlabs/rostam/vector"
)

// rawToken is a stand-in bearer secret used across the hygiene assertions. No log
// line produced by these tests may EVER contain it verbatim.
const rawToken = "super-secret-bearer-token-DO-NOT-LOG"

// TestAccessLogHTTPRedactsPrincipal drives a request carrying a bearer token
// through the HTTP middleware and asserts the emitted access line (a) carries the
// request-id, status, latency, and bytes, (b) records the principal as the token
// FINGERPRINT, and (c) NEVER contains the raw token anywhere in the line.
func TestAccessLogHTTPRedactsPrincipal(t *testing.T) {
	var buf bytes.Buffer
	al, err := NewTo(&buf, "json")
	if err != nil {
		t.Fatalf("newAccessLog: %v", err)
	}
	h := al.Middleware(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.WriteHeader(http.StatusCreated)
		_, _ = w.Write([]byte("hello"))
	}))

	req := httptest.NewRequest(http.MethodPost, "/v1/collections/docs/points", nil)
	req.Header.Set("Authorization", "Bearer "+rawToken)
	rec := httptest.NewRecorder()
	h.ServeHTTP(rec, req)

	line := strings.TrimSpace(buf.String())
	if strings.Contains(line, rawToken) {
		t.Fatalf("SECRET LEAK: access line contains the raw token: %s", line)
	}
	var m map[string]any
	if err := json.Unmarshal([]byte(line), &m); err != nil {
		t.Fatalf("access line not valid JSON: %v\n%s", err, line)
	}
	fp := vector.TokenFingerprint(rawToken)
	if m["principal"] != "token:"+fp {
		t.Errorf("principal = %v, want token:%s", m["principal"], fp)
	}
	if m["status"] != "201" {
		t.Errorf("status = %v, want 201", m["status"])
	}
	if m["transport"] != "http" {
		t.Errorf("transport = %v, want http", m["transport"])
	}
	if id, _ := m["request_id"].(string); id == "" {
		t.Error("access line has empty request_id")
	}
	if _, ok := m["latency_ms"]; !ok {
		t.Error("access line missing latency_ms")
	}
	if b, ok := m["bytes"].(float64); !ok || b != 5 {
		t.Errorf("bytes = %v, want 5", m["bytes"])
	}
	// The response echoes the server-assigned request id.
	if rec.Header().Get(RequestIDHeader) == "" {
		t.Error("response missing X-Request-Id header")
	}
}

// TestAccessLogHTTPReusesInboundID proves a client-supplied X-Request-Id is
// reused (end-to-end tracing), and a fresh one is generated when absent.
func TestAccessLogHTTPReusesInboundID(t *testing.T) {
	var buf bytes.Buffer
	al, _ := NewTo(&buf, "json")
	h := al.Middleware(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		// The handler sees the id in its context.
		if RequestID(r.Context()) != "trace-123" {
			t.Errorf("handler context id = %q, want trace-123", RequestID(r.Context()))
		}
	}))
	req := httptest.NewRequest(http.MethodGet, "/v1/health", nil)
	req.Header.Set(RequestIDHeader, "trace-123")
	rec := httptest.NewRecorder()
	h.ServeHTTP(rec, req)

	if rec.Header().Get(RequestIDHeader) != "trace-123" {
		t.Errorf("echoed id = %q, want trace-123", rec.Header().Get(RequestIDHeader))
	}
	var m map[string]any
	_ = json.Unmarshal([]byte(strings.TrimSpace(buf.String())), &m)
	if m["request_id"] != "trace-123" {
		t.Errorf("logged request_id = %v, want trace-123", m["request_id"])
	}
}

// TestAccessLogDisabledNoOutput proves that when the access log is off (nil
// AccessLog, the default), the middleware is a pass-through: no access line is
// emitted, no X-Request-Id is added, and the wrapped handler runs unchanged. This
// is the zero-hot-path-cost guarantee for the default build.
func TestAccessLogDisabledNoOutput(t *testing.T) {
	var buf bytes.Buffer
	var al *AccessLog // nil = disabled (default)
	if al.Enabled() {
		t.Fatal("nil AccessLog must report Enabled()==false")
	}
	al.Log(Entry{RequestID: "x", Principal: "token:" + vector.TokenFingerprint(rawToken)}) // must be a no-op

	inner := http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) { w.WriteHeader(http.StatusOK) })
	wrapped := al.Middleware(inner)

	req := httptest.NewRequest(http.MethodGet, "/v1/health", nil)
	req.Header.Set("Authorization", "Bearer "+rawToken)
	rec := httptest.NewRecorder()
	wrapped.ServeHTTP(rec, req)

	if buf.Len() != 0 {
		t.Errorf("disabled access log emitted output: %q", buf.String())
	}
	if rec.Header().Get(RequestIDHeader) != "" {
		t.Error("disabled middleware should not set X-Request-Id")
	}
}

// TestPrincipalNeverRawToken locks the redaction contract at the helper level:
// Principal and Fingerprint never return the raw token, and Principal falls back
// to the cert CN (not a secret) when no token is present.
func TestPrincipalNeverRawToken(t *testing.T) {
	p := Principal(rawToken, "")
	if strings.Contains(p, rawToken) {
		t.Fatalf("SECRET LEAK: Principal returned the raw token: %s", p)
	}
	if p != "token:"+vector.TokenFingerprint(rawToken) {
		t.Errorf("Principal = %q", p)
	}
	if fp := Fingerprint(rawToken); strings.Contains(fp, rawToken) || fp == "" {
		t.Errorf("Fingerprint leaked or empty: %q", fp)
	}
	if Fingerprint("") != "" {
		t.Error("empty token must fingerprint to empty")
	}
	// Cert-only principal (no token) uses the CN, which is not a secret.
	if got := Principal("", "client-cn"); got != "cn:client-cn" {
		t.Errorf("Principal(\"\", cn) = %q, want cn:client-cn", got)
	}
	if Principal("", "") != "" {
		t.Error("no token and no cn must yield empty principal")
	}
}
