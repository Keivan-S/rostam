// SPDX-License-Identifier: Apache-2.0

package authz

import (
	"encoding/json"
	"strings"
	"testing"

	"github.com/rostamlabs/rostam/vector"
)

// internalTok is the configured inter-node service token used across these tests.
const internalTok = "INTERNAL-SVC-TOKEN"

// TestNodeCNAllowlistServerGate exercises the OPT-IN server-side CN gate on the
// internal-token superuser grant: with an allowlist configured, the internal
// token alone is not enough — the caller's verified ClientCN must also be
// allowlisted (defense-in-depth against a leaked token).
func TestNodeCNAllowlistServerGate(t *testing.T) {
	opsReg := newOpsReg(t)
	reg := newRegWithKeys(t,
		vector.APIKey{Token: "reader", Tenant: "acme", Scopes: []string{"read:default/docs"}},
		vector.APIKey{Token: "certguy", Tenant: "acme", CertCN: "cn-svc", Scopes: []string{"read:default/docs"}},
	)
	allow := map[string]bool{"n1": true, "n2": true}
	auth := NewRBACAuthenticatorOpts(reg, opsReg, internalTok, RBACOptions{NodeCNAllowlist: allow})

	cases := []struct {
		req  AuthRequest
		want bool
		desc string
	}{
		// internal token + allowlisted ClientCN -> superuser grant.
		{AuthRequest{Token: internalTok, ClientCN: "n1", Op: "vector_create_collection", Args: at1("docs")}, true, "internal token + allowlisted CN grants"},
		{AuthRequest{Token: internalTok, ClientCN: "n2", Op: "totally_unknown_op", Args: nil}, true, "internal token + allowlisted CN does anything"},
		// internal token + non-allowlisted ClientCN -> DENY (the security property).
		{AuthRequest{Token: internalTok, ClientCN: "rogue", Op: "vector_create_collection", Args: at1("docs")}, false, "internal token + non-allowlisted CN denied"},
		// internal token + ABSENT ClientCN (a leaked bare token) -> DENY.
		{AuthRequest{Token: internalTok, ClientCN: "", Op: "vector_create_collection", Args: at1("docs")}, false, "internal token + absent CN denied"},
		// Non-internal-token paths must be byte-identical (allowlist does NOT touch them):
		{AuthRequest{Token: "reader", Op: "vector_search", Args: at2("default/docs")}, true, "registry token path unaffected (grant)"},
		{AuthRequest{Token: "reader", Op: "vector_insert", Args: at2("default/docs")}, false, "registry token path unaffected (scope-miss deny)"},
		// cert-CN registry path: a node CN that is allowlisted but NOT a registry key
		// is still denied on the non-internal path (the allowlist is not a principal store).
		{AuthRequest{ClientCN: "n1", Op: "vector_search", Args: at2("default/docs")}, false, "allowlisted CN is not a registry principal (cert path deny)"},
		{AuthRequest{ClientCN: "cn-svc", Op: "vector_search", Args: at2("default/docs")}, true, "registry cert-CN path unaffected (grant)"},
		{AuthRequest{ClientCN: "unknown-cn", Op: "vector_search", Args: at2("default/docs")}, false, "unknown cert-CN path unaffected (deny)"},
		{AuthRequest{Op: "vector_search", Args: at2("default/docs")}, false, "no principal unaffected (deny)"},
	}
	for _, c := range cases {
		if got := auth(c.req); got != c.want {
			t.Errorf("auth(%+v)=%v want %v — %s", c.req, got, c.want, c.desc)
		}
	}
}

// TestNodeCNAllowlistOffByteIdentical proves the OFF case (empty/nil allowlist):
// the internal-token grant is byte-identical to the historical engine — the
// internal token alone grants superuser regardless of ClientCN, and every other
// path is unchanged. This is the #1 no-break property.
func TestNodeCNAllowlistOffByteIdentical(t *testing.T) {
	opsReg := newOpsReg(t)
	reg := newRegWithKeys(t,
		vector.APIKey{Token: "reader", Tenant: "acme", Scopes: []string{"read:default/docs"}},
	)
	// nil allowlist (the zero value) and the un-optioned constructor must agree.
	authOpts := NewRBACAuthenticatorOpts(reg, opsReg, internalTok, RBACOptions{NodeCNAllowlist: nil})
	authPlain := NewRBACAuthenticator(reg, opsReg, internalTok)

	cases := []AuthRequest{
		// internal token with NO ClientCN still grants (byte-identical to today).
		{Token: internalTok, ClientCN: "", Op: "vector_create_collection", Args: at1("docs")},
		// internal token with ANY ClientCN still grants (the allowlist is off).
		{Token: internalTok, ClientCN: "whatever", Op: "vector_create_collection", Args: at1("docs")},
		{Token: internalTok, ClientCN: "rogue", Op: "totally_unknown_op", Args: nil},
		// registry + cert + no-principal paths.
		{Token: "reader", Op: "vector_search", Args: at2("default/docs")},
		{Token: "reader", Op: "vector_insert", Args: at2("default/docs")},
		{Token: "nope", Op: "vector_search", Args: at2("default/docs")},
		{Op: "vector_search", Args: at2("default/docs")},
	}
	for _, req := range cases {
		want := authPlain(req)
		if got := authOpts(req); got != want {
			t.Errorf("OFF allowlist must be byte-identical: auth(%+v)=%v want %v (plain)", req, got, want)
		}
		// And the internal-token + any-CN cases must specifically be ALLOWED.
		if req.Token == internalTok && !want {
			t.Errorf("internal token must grant superuser with allowlist OFF: %+v", req)
		}
	}
}

// TestNodeCNAllowlistAuditReason verifies the deny on a non-allowlisted internal
// caller emits Reason="peer-cn-unlisted" with a principal naming the (non-secret)
// peer CN and NEVER the raw internal token.
func TestNodeCNAllowlistAuditReason(t *testing.T) {
	opsReg := newOpsReg(t)
	reg := newRegWithKeys(t)
	log := &captureLogger{}
	allow := map[string]bool{"n1": true}
	auth := NewRBACAuthenticatorOpts(reg, opsReg, internalTok, RBACOptions{Audit: log, NodeCNAllowlist: allow})

	// Non-allowlisted CN with the (secret) internal token.
	if auth(AuthRequest{Token: internalTok, ClientCN: "rogue", Op: "vector_create_collection", Args: at1("docs")}) {
		t.Fatal("non-allowlisted internal caller must be denied")
	}
	// Absent CN.
	if auth(AuthRequest{Token: internalTok, ClientCN: "", Op: "vector_create_collection", Args: at1("docs")}) {
		t.Fatal("absent-CN internal caller must be denied")
	}
	recs := log.snapshot()
	if len(recs) != 2 {
		t.Fatalf("want 2 audit records, got %d", len(recs))
	}
	for i, rec := range recs {
		if rec.Decision != "deny" {
			t.Errorf("rec[%d].Decision=%q want deny", i, rec.Decision)
		}
		if rec.Reason != "peer-cn-unlisted" {
			t.Errorf("rec[%d].Reason=%q want peer-cn-unlisted", i, rec.Reason)
		}
		if rec.TokenFP != "" {
			t.Errorf("rec[%d].TokenFP=%q want empty (internal token must NEVER be fingerprinted)", i, rec.TokenFP)
		}
		// The marshaled record must never leak the raw internal token.
		blob, err := json.Marshal(rec)
		if err != nil {
			t.Fatal(err)
		}
		if strings.Contains(string(blob), internalTok) {
			t.Errorf("rec[%d] leaked the raw internal token: %s", i, blob)
		}
	}
	// Principal labels: "internal:"+CN for a present CN, "internal:unlisted" for absent.
	if recs[0].Principal != "internal:rogue" {
		t.Errorf("rec[0].Principal=%q want internal:rogue", recs[0].Principal)
	}
	if recs[1].Principal != "internal:unlisted" {
		t.Errorf("rec[1].Principal=%q want internal:unlisted", recs[1].Principal)
	}
}
