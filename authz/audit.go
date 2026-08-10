// SPDX-License-Identifier: Apache-2.0

package authz

import (
	"encoding/json"
	"os"
	"sync"
	"time"

	"github.com/rostamlabs/rostam/vector"
)

// AuditRecord is one structured authorization-decision audit entry. It records
// WHO (a REDACTED principal — never the raw bearer token), WHAT (Action on
// Resource via Op), and the VERDICT (Decision allow|deny + the Reason branch),
// with a timestamp, so operators get a who-did-what-and-was-it-allowed trail.
//
// SECURITY: this record MUST NEVER carry the raw bearer/internal token (a
// secret). The caller's credential is represented ONLY by a redacted Principal
// (cert CN, tenant, or "unknown") plus TokenFP — a NON-reversible fingerprint
// (see tokenFingerprint). The internal-service-token grant records TokenFP=""
// (the internal secret is never even fingerprinted).
type AuditRecord struct {
	Time      time.Time `json:"time"`
	Principal string    `json:"principal"`
	TokenFP   string    `json:"token_fp,omitempty"`
	Tenant    string    `json:"tenant,omitempty"`
	Action    string    `json:"action,omitempty"`
	Resource  string    `json:"resource,omitempty"`
	Op        string    `json:"op,omitempty"`
	Decision  string    `json:"decision"` // "allow" | "deny"
	// Reason names the deciding branch of the authorizer, one of:
	//   "internal-token"   — internal-service superuser grant (allow)
	//   "no-principal"     — no key store, or no token and no cert CN (deny)
	//   "token-not-found"  — bearer token not in the registry (deny)
	//   "cert-not-found"   — verified cert CN not bound to a key (deny)
	//   "scope-match"      — a key scope covered (action, resource) (allow)
	//   "scope-miss"       — no key scope covered (action, resource) (deny)
	//   "tenant-mismatch"  — tenant isolation ON: the scope GRANTED but the
	//                        resource's tenant != key.Tenant (and the key is not
	//                        the "*" cross-tenant marker), or a tenant-scoped key
	//                        targeted a cluster resource — the scope-grant is
	//                        downgraded to a deny (fail-closed; opt-in via
	//                        -tenant-isolation).
	//   "jwt-invalid"      — JWT path (opt-in -jwt-public-key): a JWT-looking
	//                        bearer was NOT a registry token AND failed
	//                        verification (bad signature/alg/exp/nbf/iss/aud or
	//                        missing tenant/scopes) — fail-closed deny, never a
	//                        fallthrough to grant. The raw JWT is NEVER recorded
	//                        (only its fingerprint).
	//   "jwt-scope-match"  — a verified JWT principal's scope covered (action,
	//                        resource) (allow).
	//   "jwt-scope-miss"   — a verified JWT principal had no scope covering
	//                        (action, resource) (deny).
	//   "jwt-tenant-mismatch" — tenant isolation ON: a verified JWT's scope
	//                        GRANTED but the resource's tenant != the JWT's tenant
	//                        claim — downgraded to a deny (fail-closed).
	Reason string `json:"reason"`
}

// AuditLogger receives one AuditRecord per authorization decision. It is an
// interface so the authorizer is decoupled from the sink — production wires the
// jsonStderrAuditLogger; tests inject a capturing fake. A nil AuditLogger means
// auditing is DISABLED (the authorizer takes a zero-cost fast path).
//
// Implementations MUST be safe for concurrent calls: the authorizer runs per-op
// across all transports (HTTP/gRPC/TCP) and may invoke Audit from many
// goroutines simultaneously.
type AuditLogger interface {
	Audit(AuditRecord)
}

// jsonStderrAuditLogger writes one JSON-encoded AuditRecord per line to stderr.
// A sync.Mutex guards the encoder so concurrent records never interleave on the
// wire. This is the default sink wired by -audit-log.
type jsonStderrAuditLogger struct {
	mu  sync.Mutex
	enc *json.Encoder
}

// NewJSONStderrAuditLogger builds the default audit sink: one JSON line per
// record to os.Stderr, mutex-guarded against interleaving.
func NewJSONStderrAuditLogger() AuditLogger {
	return &jsonStderrAuditLogger{enc: json.NewEncoder(os.Stderr)}
}

// Audit writes rec as a single JSON line to stderr. Encode failures are dropped
// (an audit sink must never break the request path); the mutex holds only the
// write, never anything in the authorize verdict.
func (l *jsonStderrAuditLogger) Audit(rec AuditRecord) {
	l.mu.Lock()
	_ = l.enc.Encode(rec) // json.Encoder.Encode appends a newline per call.
	l.mu.Unlock()
}

// tokenFingerprint returns a short, NON-reversible identifier for a token: the
// first 16 hex chars (8 bytes) of sha256(token). Empty token -> "" (no
// fingerprint). This lets operators correlate repeated requests/probes from the
// same credential WITHOUT ever logging the secret itself.
//
// It delegates to vector.TokenFingerprint so there is exactly ONE fingerprint
// scheme across the codebase (the online key-admin list redaction uses the same
// definition); this thin wrapper keeps the existing authz call sites unchanged.
//
// SECURITY: NEVER log the raw token. Only this fingerprint (and a redacted
// principal) may appear in an AuditRecord. sha256 is one-way; the truncation to
// 16 hex chars keeps records compact while remaining collision-resistant enough
// for correlation.
func tokenFingerprint(token string) string {
	return vector.TokenFingerprint(token)
}
