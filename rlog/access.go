// SPDX-License-Identifier: Apache-2.0

package rlog

import (
	"io"
	"log/slog"
	"os"
	"time"

	"github.com/rostamlabs/rostam/vector"
)

// Fingerprint returns the same short, NON-reversible token fingerprint the authz
// audit log uses (vector.TokenFingerprint: first 16 hex of sha256, "" for an
// empty token). It is the ONLY representation of a caller's bearer token that may
// appear in an access-log line.
//
// SECURITY: the raw bearer/internal token is a secret and MUST NEVER be logged.
// The access log records only this fingerprint (or a cert CN, which is not a
// secret) as the principal. Reusing the audit definition keeps exactly one
// fingerprint scheme across the codebase so operators can correlate an access
// line with an audit record.
func Fingerprint(token string) string {
	return vector.TokenFingerprint(token)
}

// Principal derives the redacted principal for an access line from the request's
// credentials, mirroring the authz audit record's precedence: the bearer token's
// fingerprint when a token was presented, else the verified mTLS client-cert CN
// (not a secret), else "". It NEVER returns the raw token.
func Principal(token, clientCN string) string {
	if token != "" {
		return "token:" + Fingerprint(token)
	}
	if clientCN != "" {
		return "cn:" + clientCN
	}
	return ""
}

// Entry is one request's access-log record. Status is a transport-appropriate
// short string (an HTTP code, a gRPC code name, or a TCP wire-status name).
// Principal is ALREADY redacted (see Principal / Fingerprint) — a raw token must
// never be placed here.
type Entry struct {
	RequestID string
	Transport string // "http" | "grpc" | "tcp"
	Op        string // op name / RPC method / HTTP "METHOD path"
	Status    string
	Latency   time.Duration
	Principal string
	Bytes     int // response payload bytes
}

// AccessLog is the OPT-IN per-request access logger. A nil *AccessLog means the
// access log is DISABLED: every method is nil-safe and Enabled reports false, so
// callers guard on it and pay zero cost on the hot path when -access-log is off.
// When enabled it emits one structured slog line per request through its own
// info-level logger (independent of -log-level, so an operator who asks for the
// access log always gets its lines).
type AccessLog struct {
	logger *slog.Logger
}

// New builds an enabled AccessLog writing one line per request to os.Stderr in
// the given format ("text"|"json"), always at info level regardless of the
// process -log-level (an operator who set -access-log wants the lines). A bad
// format is an error (fail loud at startup).
func New(format string) (*AccessLog, error) {
	return NewTo(os.Stderr, format)
}

// NewTo builds an enabled AccessLog writing each line to w in the given format —
// the injectable core of New (which pins os.Stderr). Embedders that want the
// access log redirected (a file, a rotating writer, an in-memory sink) build it
// here; tests use it to capture and assert on emitted lines.
func NewTo(w io.Writer, format string) (*AccessLog, error) {
	h, err := NewHandler(w, format, slog.LevelInfo)
	if err != nil {
		return nil, err
	}
	return &AccessLog{logger: slog.New(h)}, nil
}

// Enabled reports whether access logging is on. It is nil-safe: the zero/nil
// AccessLog is disabled, which is how callers avoid all per-request cost when
// -access-log is off.
func (a *AccessLog) Enabled() bool { return a != nil && a.logger != nil }

// Log emits one access line for e. It is a no-op on a nil/disabled AccessLog, so
// callers may always call it once they have decided (via Enabled) to build an
// Entry. The line carries the request id, transport, op, status, latency (ms),
// redacted principal, and response bytes.
func (a *AccessLog) Log(e Entry) {
	if !a.Enabled() {
		return
	}
	a.logger.Info("access",
		"request_id", e.RequestID,
		"transport", e.Transport,
		"op", e.Op,
		"status", e.Status,
		"latency_ms", float64(e.Latency.Microseconds())/1000.0,
		"principal", e.Principal,
		"bytes", e.Bytes,
	)
}
