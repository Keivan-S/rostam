// SPDX-License-Identifier: Apache-2.0

package rlog

import (
	"context"
	"crypto/rand"
	"encoding/hex"
	"strconv"
	"sync/atomic"
)

// idPrefix is a short random per-process tag prepended to every generated
// request id, so ids from different server processes never collide even though
// each process's counter restarts at 1. It is drawn once from crypto/rand at
// package init (never on the hot path). A rand failure falls back to a fixed
// tag — a request id is a correlation aid, not a security token, so degraded
// uniqueness is acceptable and must never panic the server.
var idPrefix = func() string {
	var b [4]byte
	if _, err := rand.Read(b[:]); err != nil {
		return "0000"
	}
	return hex.EncodeToString(b[:])
}()

// idCounter is the per-process monotonic request counter. Add is a single atomic
// increment — the whole cost of NewID beyond a base-36 format and a short concat.
var idCounter atomic.Uint64

// NewID returns a fresh, cheap, per-process-unique request id (the random
// process prefix plus a base-36 monotonic counter, e.g. "a1b2c3d4-2f"). It does
// NO allocation-heavy formatting and no syscall, so it is safe to call at each
// transport ingress. Ids are for correlation only — they are not secrets and are
// not guaranteed globally unique across processes beyond the random prefix.
func NewID() string {
	n := idCounter.Add(1)
	return idPrefix + "-" + strconv.FormatUint(n, 36)
}

// ctxKey is the unexported context key type for the request id (avoids
// collisions with any other package's context values).
type ctxKey struct{}

// WithRequestID returns a child context carrying id. Transport ingress points
// call this after generating or reusing an id so downstream handlers and
// error/warn logs can correlate to the same request.
func WithRequestID(ctx context.Context, id string) context.Context {
	return context.WithValue(ctx, ctxKey{}, id)
}

// RequestID returns the request id carried in ctx, or "" if none was set. A "" id
// is logged as an omitted field, never as a spurious value.
func RequestID(ctx context.Context) string {
	if ctx == nil {
		return ""
	}
	if id, ok := ctx.Value(ctxKey{}).(string); ok {
		return id
	}
	return ""
}

// maxInboundReqID bounds a reused client-supplied request id. The value is
// attacker-controlled and logged verbatim on every request, so an unbounded or
// control-char-laden id would mean log bloat / amplification and a tainted value
// reflected in the response header. Beyond this we ignore it and generate a fresh
// id. 128 is generous for any real trace/correlation id (UUID, W3C traceparent).
const maxInboundReqID = 128

// EnsureRequestID returns an id for this request and a context carrying it: it
// reuses inbound (a client-supplied X-Request-Id header / gRPC metadata value)
// when it is present AND safe to log, else generates a fresh one. Reusing the
// client's id lets a request be traced end-to-end across a caller and this
// server. An inbound id that is empty, over maxInboundReqID, or contains a
// non-printable/space character is rejected (fresh id generated) — it is
// attacker-controlled and echoed into logs and the response header.
func EnsureRequestID(ctx context.Context, inbound string) (context.Context, string) {
	id := inbound
	if id == "" || len(id) > maxInboundReqID || !safeReqID(id) {
		id = NewID()
	}
	return WithRequestID(ctx, id), id
}

// safeReqID reports whether s is safe to log/echo verbatim: printable ASCII
// only (no controls, no newline that could forge a log line, no non-ASCII).
func safeReqID(s string) bool {
	for i := 0; i < len(s); i++ {
		if s[i] < 0x20 || s[i] > 0x7e {
			return false
		}
	}
	return true
}
