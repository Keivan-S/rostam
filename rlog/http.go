// SPDX-License-Identifier: Apache-2.0

package rlog

import (
	"net/http"
	"strconv"
	"strings"
	"time"
)

// RequestIDHeader is the HTTP header carrying a request correlation id, both
// inbound (reused when a client supplies it) and outbound (echoed on every
// response so a caller can record the id the server used).
const RequestIDHeader = "X-Request-Id"

// statusRecorder wraps an http.ResponseWriter to capture the status code and the
// number of body bytes written, for the access line. It defaults to 200 (Go's
// implicit status when a handler writes a body without calling WriteHeader).
type statusRecorder struct {
	http.ResponseWriter
	status int
	bytes  int
}

func (r *statusRecorder) WriteHeader(code int) {
	r.status = code
	r.ResponseWriter.WriteHeader(code)
}

func (r *statusRecorder) Write(b []byte) (int, error) {
	n, err := r.ResponseWriter.Write(b)
	r.bytes += n
	return n, err
}

// httpBearer extracts the bearer token from an Authorization header (stripping a
// "Bearer " prefix), matching the httpapi transport. Used ONLY to derive the
// redacted principal fingerprint — the token itself never leaves this function.
func httpBearer(r *http.Request) string {
	h := r.Header.Get("Authorization")
	if after, ok := strings.CutPrefix(h, "Bearer "); ok {
		return after
	}
	return h
}

// httpClientCN returns the VERIFIED mTLS client-cert CommonName, or "" — the same
// verified-chain source the httpapi transport uses (never a spoofable header).
func httpClientCN(r *http.Request) string {
	if r.TLS == nil || len(r.TLS.VerifiedChains) == 0 || len(r.TLS.PeerCertificates) == 0 {
		return ""
	}
	return r.TLS.PeerCertificates[0].Subject.CommonName
}

// Middleware wraps next with request-id + access logging. It:
//   - reuses an inbound X-Request-Id header or generates one, puts it in the
//     request context (so handlers/error logs correlate), and echoes it on the
//     response;
//   - records the status and response bytes, and emits one access line per
//     request with a REDACTED principal (token fingerprint or cert CN, never the
//     raw token).
//
// It is only installed when -access-log is on (a nil/disabled AccessLog returns
// next unchanged), so the default build wraps nothing and pays zero cost.
func (a *AccessLog) Middleware(next http.Handler) http.Handler {
	if !a.Enabled() {
		return next
	}
	return http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		start := time.Now()
		ctx, id := EnsureRequestID(r.Context(), r.Header.Get(RequestIDHeader))
		w.Header().Set(RequestIDHeader, id)
		rec := &statusRecorder{ResponseWriter: w, status: http.StatusOK}
		next.ServeHTTP(rec, r.WithContext(ctx))
		a.Log(Entry{
			RequestID: id,
			Transport: "http",
			Op:        r.Method + " " + r.URL.Path,
			Status:    strconv.Itoa(rec.status),
			Latency:   time.Since(start),
			Principal: Principal(httpBearer(r), httpClientCN(r)),
			Bytes:     rec.bytes,
		})
	})
}
