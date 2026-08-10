// SPDX-License-Identifier: Apache-2.0

package authz

import (
	"encoding/json"
	"strings"
	"sync"
	"testing"

	"github.com/rostamlabs/rostam/vector"
)

// captureLogger is a test AuditLogger that collects every emitted record. It is
// mutex-guarded so it is safe to share across concurrent authorize calls (the
// -race test relies on this).
type captureLogger struct {
	mu      sync.Mutex
	records []AuditRecord
}

func (c *captureLogger) Audit(rec AuditRecord) {
	c.mu.Lock()
	c.records = append(c.records, rec)
	c.mu.Unlock()
}

func (c *captureLogger) snapshot() []AuditRecord {
	c.mu.Lock()
	defer c.mu.Unlock()
	out := make([]AuditRecord, len(c.records))
	copy(out, c.records)
	return out
}

// sentinelToken is a distinctive raw bearer token used by the teeth test: it
// must NEVER appear in any emitted record's marshaled JSON.
const sentinelToken = "SECRET-SENTINEL-TOKEN-123"

// --- tokenFingerprint ---

func TestTokenFingerprint(t *testing.T) {
	if got := tokenFingerprint(""); got != "" {
		t.Errorf("tokenFingerprint(\"\")=%q want \"\"", got)
	}
	fp := tokenFingerprint(sentinelToken)
	if fp == "" {
		t.Fatal("tokenFingerprint of a non-empty token must be non-empty")
	}
	if len(fp) != 16 {
		t.Errorf("fingerprint len=%d want 16 hex chars", len(fp))
	}
	if fp == sentinelToken {
		t.Error("fingerprint must NOT equal the input token (non-reversible)")
	}
	if strings.Contains(fp, sentinelToken) {
		t.Error("fingerprint must not contain the raw token")
	}
	// Deterministic.
	if tokenFingerprint(sentinelToken) != fp {
		t.Error("tokenFingerprint must be deterministic")
	}
	// Distinct inputs -> distinct fingerprints.
	if tokenFingerprint("other-token") == fp {
		t.Error("different tokens should fingerprint differently")
	}
}

// --- per-branch emit: exactly one record with correct fields ---

func TestAuditEmitsOneRecordPerBranch(t *testing.T) {
	opsReg := newOpsReg(t)
	reg := newRegWithKeys(t,
		vector.APIKey{Token: sentinelToken, Tenant: "acme", Scopes: []string{"read:default/docs"}},
		vector.APIKey{Token: "certguy", Tenant: "acme", CertCN: "cn-svc", Scopes: []string{"read:default/docs"}},
	)

	cases := []struct {
		desc      string
		req       AuthRequest
		wantAllow bool
		decision  string
		reason    string
		principal string
		tenant    string
		wantFP    bool // fingerprint should be non-empty
	}{
		{
			desc:      "internal-token allow",
			req:       AuthRequest{Token: "INTERNAL-SVC-TOKEN", Op: "vector_create_collection", Args: at1("docs")},
			wantAllow: true, decision: "allow", reason: "internal-token",
			principal: "internal", wantFP: false,
		},
		{
			desc:      "valid token scope-match allow",
			req:       AuthRequest{Token: sentinelToken, Op: "vector_search", Args: at2("docs")},
			wantAllow: true, decision: "allow", reason: "scope-match",
			principal: "token:" + tokenFingerprint(sentinelToken), tenant: "acme", wantFP: true,
		},
		{
			desc:      "valid token scope-MISS deny",
			req:       AuthRequest{Token: sentinelToken, Op: "vector_insert", Args: at2("docs")},
			wantAllow: false, decision: "deny", reason: "scope-miss",
			principal: "token:" + tokenFingerprint(sentinelToken), tenant: "acme", wantFP: true,
		},
		{
			desc:      "unknown token deny",
			req:       AuthRequest{Token: "nope-unknown", Op: "vector_search", Args: at2("docs")},
			wantAllow: false, decision: "deny", reason: "token-not-found",
			principal: "unknown", wantFP: true,
		},
		{
			desc:      "cert-CN principal allow",
			req:       AuthRequest{ClientCN: "cn-svc", Op: "vector_search", Args: at2("docs")},
			wantAllow: true, decision: "allow", reason: "scope-match",
			principal: "cn-svc", tenant: "acme", wantFP: false,
		},
		{
			desc:      "cert-CN principal deny (scope-miss)",
			req:       AuthRequest{ClientCN: "cn-svc", Op: "vector_insert", Args: at2("docs")},
			wantAllow: false, decision: "deny", reason: "scope-miss",
			principal: "cn-svc", tenant: "acme", wantFP: false,
		},
		{
			desc:      "unknown cert-CN deny",
			req:       AuthRequest{ClientCN: "unknown-cn", Op: "vector_search", Args: at2("docs")},
			wantAllow: false, decision: "deny", reason: "cert-not-found",
			principal: "unknown", wantFP: false,
		},
		{
			desc:      "no principal deny",
			req:       AuthRequest{Op: "vector_search", Args: at2("docs")},
			wantAllow: false, decision: "deny", reason: "no-principal",
			principal: "unknown", wantFP: false,
		},
	}

	for _, c := range cases {
		t.Run(c.desc, func(t *testing.T) {
			cap := &captureLogger{}
			auth := NewRBACAuthenticatorWithAudit(reg, opsReg, "INTERNAL-SVC-TOKEN", cap)
			got := auth(c.req)
			if got != c.wantAllow {
				t.Errorf("verdict=%v want %v", got, c.wantAllow)
			}
			recs := cap.snapshot()
			if len(recs) != 1 {
				t.Fatalf("got %d records, want exactly 1", len(recs))
			}
			r := recs[0]
			if r.Decision != c.decision {
				t.Errorf("Decision=%q want %q", r.Decision, c.decision)
			}
			if r.Reason != c.reason {
				t.Errorf("Reason=%q want %q", r.Reason, c.reason)
			}
			if r.Principal != c.principal {
				t.Errorf("Principal=%q want %q", r.Principal, c.principal)
			}
			if r.Tenant != c.tenant {
				t.Errorf("Tenant=%q want %q", r.Tenant, c.tenant)
			}
			if r.Op != c.req.Op {
				t.Errorf("Op=%q want %q", r.Op, c.req.Op)
			}
			if c.wantFP && r.TokenFP == "" {
				t.Error("expected a non-empty TokenFP")
			}
			if !c.wantFP && r.TokenFP != "" {
				t.Errorf("expected empty TokenFP, got %q", r.TokenFP)
			}
			if r.Time.IsZero() {
				t.Error("Time must be set")
			}
		})
	}
}

// TestAuditNeverLeaksRawToken is the #1 correctness criterion (the teeth test):
// the raw bearer token must NEVER appear in any emitted record, while its
// fingerprint MUST be present. Checked for both an allow and a deny.
func TestAuditNeverLeaksRawToken(t *testing.T) {
	opsReg := newOpsReg(t)
	reg := newRegWithKeys(t,
		vector.APIKey{Token: sentinelToken, Tenant: "acme", Scopes: []string{"read:default/docs"}},
	)
	cap := &captureLogger{}
	auth := NewRBACAuthenticatorWithAudit(reg, opsReg, "INTERNAL-SVC-TOKEN", cap)

	// allow (scope-match), deny (scope-miss), deny (unknown token — still uses
	// the raw token to derive a fingerprint, so it's the riskiest leak path).
	auth(AuthRequest{Token: sentinelToken, Op: "vector_search", Args: at2("docs")})
	auth(AuthRequest{Token: sentinelToken, Op: "vector_insert", Args: at2("docs")})
	auth(AuthRequest{Token: sentinelToken + "-probe", Op: "vector_search", Args: at2("docs")})

	wantFP := tokenFingerprint(sentinelToken)
	recs := cap.snapshot()
	if len(recs) != 3 {
		t.Fatalf("got %d records, want 3", len(recs))
	}
	for i, r := range recs {
		blob, err := json.Marshal(r)
		if err != nil {
			t.Fatalf("marshal record %d: %v", i, err)
		}
		s := string(blob)
		if strings.Contains(s, sentinelToken) {
			t.Errorf("record %d LEAKS the raw token: %s", i, s)
		}
		// The first two records (known key) must carry the fingerprint.
		if i < 2 {
			if !strings.Contains(s, wantFP) {
				t.Errorf("record %d missing the expected fingerprint %q: %s", i, wantFP, s)
			}
			if r.TokenFP != wantFP {
				t.Errorf("record %d TokenFP=%q want %q", i, r.TokenFP, wantFP)
			}
		}
	}
}

// TestAuditNilLoggerZeroRecordsAndIdenticalVerdict asserts the OFF path: a nil
// logger emits ZERO records AND the verdict is byte-identical to the with-logger
// verdict for the same requests (audit is pure observation, never changes a
// decision).
func TestAuditNilLoggerZeroRecordsAndIdenticalVerdict(t *testing.T) {
	opsReg := newOpsReg(t)
	reg := newRegWithKeys(t,
		vector.APIKey{Token: sentinelToken, Tenant: "acme", Scopes: []string{"read:default/docs"}},
		vector.APIKey{Token: "certguy", Tenant: "acme", CertCN: "cn-svc", Scopes: []string{"read:default/docs"}},
	)
	authNil := NewRBACAuthenticatorWithAudit(reg, opsReg, "INTERNAL-SVC-TOKEN", nil)
	cap := &captureLogger{}
	authLog := NewRBACAuthenticatorWithAudit(reg, opsReg, "INTERNAL-SVC-TOKEN", cap)

	reqs := []AuthRequest{
		{Token: "INTERNAL-SVC-TOKEN", Op: "vector_create_collection", Args: at1("docs")},
		{Token: sentinelToken, Op: "vector_search", Args: at2("docs")},
		{Token: sentinelToken, Op: "vector_insert", Args: at2("docs")},
		{Token: "unknown", Op: "vector_search", Args: at2("docs")},
		{ClientCN: "cn-svc", Op: "vector_search", Args: at2("docs")},
		{ClientCN: "cn-svc", Op: "vector_insert", Args: at2("docs")},
		{ClientCN: "unknown-cn", Op: "vector_search", Args: at2("docs")},
		{Op: "vector_search", Args: at2("docs")},
	}
	for _, req := range reqs {
		if vNil, vLog := authNil(req), authLog(req); vNil != vLog {
			t.Errorf("verdict differs nil=%v log=%v for %+v", vNil, vLog, req)
		}
	}
	// nil-logger authenticator must have emitted nothing into cap (it shares no
	// logger); cap only saw the with-logger calls — one per request.
	if n := len(cap.snapshot()); n != len(reqs) {
		t.Errorf("with-logger emitted %d records, want %d (one per request)", n, len(reqs))
	}
}

// TestAuditConcurrent runs many authorize calls sharing one logger to exercise
// the logger mutex under -race.
func TestAuditConcurrent(t *testing.T) {
	opsReg := newOpsReg(t)
	reg := newRegWithKeys(t,
		vector.APIKey{Token: sentinelToken, Tenant: "acme", Scopes: []string{"read:default/docs"}},
	)
	cap := &captureLogger{}
	auth := NewRBACAuthenticatorWithAudit(reg, opsReg, "INTERNAL-SVC-TOKEN", cap)

	const goroutines, perG = 16, 50
	var wg sync.WaitGroup
	wg.Add(goroutines)
	for g := 0; g < goroutines; g++ {
		go func() {
			defer wg.Done()
			for i := 0; i < perG; i++ {
				auth(AuthRequest{Token: sentinelToken, Op: "vector_search", Args: at2("docs")})
			}
		}()
	}
	wg.Wait()
	if n := len(cap.snapshot()); n != goroutines*perG {
		t.Errorf("emitted %d records, want %d", n, goroutines*perG)
	}
}
