// SPDX-License-Identifier: Apache-2.0

package authz

import (
	"encoding/json"
	"strings"
	"testing"
	"time"

	"github.com/rostamlabs/rostam/vector"
)

// These tests cover the COMPOSITION of the Task-1 JWT verifier with the RBAC
// authorizer (the Task-2 branch): a verified JWT yields a synthetic principal
// that runs through the SAME scope-match + tenant-isolation guard + audit as a
// registry key, while a verify failure is fail-closed and the no-verifier path
// stays byte-identical to the pre-JWT engine.

// --- a valid JWT is granted/denied purely by its scopes ---------------------

// TestJWTScopeGrantAndMiss: a verified RS256 JWT whose scopes cover the op is
// ALLOWED; the same valid JWT whose scopes miss the op is DENIED. The JWT never
// touches the registry — it is resolved entirely through the verifier branch.
func TestJWTScopeGrantAndMiss(t *testing.T) {
	opsReg := newOpsReg(t)
	// An EMPTY registry: the JWT is never a registry token, so this proves the
	// grant comes solely from the JWT branch.
	reg := newRegWithKeys(t)
	key := mustRSAKey(t)
	v := newRS256Verifier(t, key, "", "")
	auth := NewRBACAuthenticatorOpts(reg, opsReg, "INT", RBACOptions{JWTVerifier: v})

	// scopes grant read on default/docs -> ALLOW a vector_search there.
	grantTok := mintRS256(t, key, map[string]any{
		"sub": "alice", "tenant": "acme", "scopes": "read:default/docs",
		"exp": expSoon(),
	})
	if !auth(AuthRequest{Token: grantTok, Op: "vector_search", Args: at2("default/docs")}) {
		t.Error("JWT with read:default/docs must be GRANTED on vector_search default/docs")
	}

	// scopes do NOT grant write -> DENY a vector_insert (write).
	if auth(AuthRequest{Token: grantTok, Op: "vector_insert", Args: at2("default/docs")}) {
		t.Error("JWT with only read scope must be DENIED a write op")
	}

	// scopes do not cover a different collection -> DENY.
	if auth(AuthRequest{Token: grantTok, Op: "vector_search", Args: at2("default/other")}) {
		t.Error("JWT scoped to default/docs must be DENIED on default/other")
	}
}

// TestJWTScopeGrantES256 confirms the ES256 alg path composes identically.
func TestJWTScopeGrantES256(t *testing.T) {
	opsReg := newOpsReg(t)
	reg := newRegWithKeys(t)
	key := mustECKey(t)
	v := newES256Verifier(t, key, "", "")
	auth := NewRBACAuthenticatorOpts(reg, opsReg, "INT", RBACOptions{JWTVerifier: v})

	tok := mintES256(t, key, map[string]any{
		"sub": "bob", "tenant": "acme", "scopes": []any{"write:default/*"},
		"exp": expSoon(),
	})
	if !auth(AuthRequest{Token: tok, Op: "vector_insert", Args: at2("default/docs")}) {
		t.Error("ES256 JWT with write:default/* must be GRANTED a write op")
	}
}

// --- JWT honors tenant-isolation IDENTICALLY --------------------------------

// TestJWTHonorsTenantIsolation: with -tenant-isolation ON, a JWT whose tenant
// claim is "acme" but whose scope is mis-scoped into "default/*" is DENIED on a
// default/... resource (the tenant guard downgrades the scope-grant), and is
// ALLOWED on an acme/... resource it is scoped for. This proves the JWT's tenant
// claim drives the SAME guard as APIKey.Tenant.
func TestJWTHonorsTenantIsolation(t *testing.T) {
	opsReg := newOpsReg(t)
	reg := newRegWithKeys(t)
	key := mustRSAKey(t)
	v := newRS256Verifier(t, key, "", "")
	auth := NewRBACAuthenticatorOpts(reg, opsReg, "INT", RBACOptions{
		JWTVerifier:     v,
		TenantIsolation: true,
	})

	// tenant=acme but scoped cross-tenant into default (a mis-scoped JWT).
	tok := mintRS256(t, key, map[string]any{
		"sub": "carol", "tenant": "acme",
		"scopes": "read:default/* read:acme/*",
		"exp":    expSoon(),
	})

	// On default/docs: scope grants, but tenant guard (acme != default) -> DENY.
	if auth(AuthRequest{Token: tok, Op: "vector_search", Args: at2("default/docs")}) {
		t.Error("JWT tenant=acme must be DENIED cross-tenant on default/docs under isolation ON")
	}
	// On acme/docs: scope grants AND tenant matches -> ALLOW.
	if !auth(AuthRequest{Token: tok, Op: "vector_search", Args: at2("acme/docs")}) {
		t.Error("JWT tenant=acme scoped acme/* must be ALLOWED on acme/docs")
	}
}

// TestJWTTenantIsolationAudit checks the tenant-mismatch DENY is audited with
// the jwt-prefixed reason and the redacted principal/fingerprint.
func TestJWTTenantIsolationAudit(t *testing.T) {
	opsReg := newOpsReg(t)
	reg := newRegWithKeys(t)
	key := mustRSAKey(t)
	v := newRS256Verifier(t, key, "", "")
	cap := &captureLogger{}
	auth := NewRBACAuthenticatorOpts(reg, opsReg, "INT", RBACOptions{
		JWTVerifier:     v,
		TenantIsolation: true,
		Audit:           cap,
	})
	tok := mintRS256(t, key, map[string]any{
		"sub": "carol", "tenant": "acme", "scopes": "read:default/*",
		"exp": expSoon(),
	})
	if auth(AuthRequest{Token: tok, Op: "vector_search", Args: at2("default/docs")}) {
		t.Fatal("expected tenant-mismatch deny")
	}
	recs := cap.snapshot()
	if len(recs) != 1 {
		t.Fatalf("want 1 record, got %d", len(recs))
	}
	r := recs[0]
	if r.Decision != "deny" || r.Reason != "jwt-tenant-mismatch" {
		t.Errorf("got decision=%q reason=%q want deny/jwt-tenant-mismatch", r.Decision, r.Reason)
	}
	if r.Principal != "carol" {
		t.Errorf("principal=%q want sub claim \"carol\"", r.Principal)
	}
	if r.Tenant != "acme" {
		t.Errorf("tenant=%q want \"acme\"", r.Tenant)
	}
}

// --- valid-JWT audit reason + principal --------------------------------------

// TestJWTAuditReasons covers the jwt-scope-match (allow) and jwt-scope-miss
// (deny) audit reasons, and that the principal is the sub claim.
func TestJWTAuditReasons(t *testing.T) {
	opsReg := newOpsReg(t)
	reg := newRegWithKeys(t)
	key := mustRSAKey(t)
	v := newRS256Verifier(t, key, "", "")
	cap := &captureLogger{}
	auth := NewRBACAuthenticatorOpts(reg, opsReg, "INT", RBACOptions{JWTVerifier: v, Audit: cap})

	tok := mintRS256(t, key, map[string]any{
		"sub": "dave", "tenant": "acme", "scopes": "read:default/docs",
		"exp": expSoon(),
	})
	// allow
	if !auth(AuthRequest{Token: tok, Op: "vector_search", Args: at2("default/docs")}) {
		t.Fatal("expected allow")
	}
	// deny (write not scoped)
	if auth(AuthRequest{Token: tok, Op: "vector_insert", Args: at2("default/docs")}) {
		t.Fatal("expected deny")
	}
	recs := cap.snapshot()
	if len(recs) != 2 {
		t.Fatalf("want 2 records, got %d", len(recs))
	}
	if recs[0].Decision != "allow" || recs[0].Reason != "jwt-scope-match" {
		t.Errorf("rec0 = %q/%q want allow/jwt-scope-match", recs[0].Decision, recs[0].Reason)
	}
	if recs[1].Decision != "deny" || recs[1].Reason != "jwt-scope-miss" {
		t.Errorf("rec1 = %q/%q want deny/jwt-scope-miss", recs[1].Decision, recs[1].Reason)
	}
	for i, r := range recs {
		if r.Principal != "dave" {
			t.Errorf("rec%d principal=%q want \"dave\"", i, r.Principal)
		}
		if r.TokenFP == "" {
			t.Errorf("rec%d TokenFP must be set (fingerprint of the JWT)", i)
		}
	}
}

// TestJWTPrincipalFallbackWhenNoSub: a verified JWT with no "sub" claim still
// requires tenant+scopes; the principal label falls back to "jwt:"+fingerprint.
func TestJWTPrincipalFallbackWhenNoSub(t *testing.T) {
	opsReg := newOpsReg(t)
	reg := newRegWithKeys(t)
	key := mustRSAKey(t)
	v := newRS256Verifier(t, key, "", "")
	cap := &captureLogger{}
	auth := NewRBACAuthenticatorOpts(reg, opsReg, "INT", RBACOptions{JWTVerifier: v, Audit: cap})

	tok := mintRS256(t, key, map[string]any{
		"tenant": "acme", "scopes": "read:default/docs", "exp": expSoon(),
	})
	if !auth(AuthRequest{Token: tok, Op: "vector_search", Args: at2("default/docs")}) {
		t.Fatal("expected allow")
	}
	r := cap.snapshot()[0]
	if !strings.HasPrefix(r.Principal, "jwt:") {
		t.Errorf("principal=%q want \"jwt:\"+fp fallback", r.Principal)
	}
	if r.Principal != "jwt:"+r.TokenFP {
		t.Errorf("principal=%q want jwt:%s", r.Principal, r.TokenFP)
	}
}

// --- an invalid / expired JWT -> deny + audit "jwt-invalid", NO fallthrough --

// TestJWTInvalidExpiredDeny: an EXPIRED JWT (verifier configured) is DENIED with
// audit Reason="jwt-invalid", and the verdict does NOT fall through to any other
// grant path.
func TestJWTInvalidExpiredDeny(t *testing.T) {
	opsReg := newOpsReg(t)
	reg := newRegWithKeys(t)
	key := mustRSAKey(t)
	v := newRS256Verifier(t, key, "", "")
	cap := &captureLogger{}
	auth := NewRBACAuthenticatorOpts(reg, opsReg, "INT", RBACOptions{JWTVerifier: v, Audit: cap})

	expired := mintRS256(t, key, map[string]any{
		"sub": "eve", "tenant": "acme", "scopes": "*:*",
		"exp": expPast(),
	})
	if auth(AuthRequest{Token: expired, Op: "vector_search", Args: at2("default/docs")}) {
		t.Fatal("expired JWT must be DENIED")
	}
	recs := cap.snapshot()
	if len(recs) != 1 {
		t.Fatalf("want 1 record, got %d", len(recs))
	}
	if recs[0].Decision != "deny" || recs[0].Reason != "jwt-invalid" {
		t.Errorf("got %q/%q want deny/jwt-invalid", recs[0].Decision, recs[0].Reason)
	}
	if recs[0].TokenFP == "" {
		t.Error("jwt-invalid record must carry the fingerprint")
	}
}

// TestJWTWrongKeyDeny: a JWT signed with a DIFFERENT key fails signature
// verification -> jwt-invalid deny (no fallthrough to a registry/cert grant).
func TestJWTWrongKeyDeny(t *testing.T) {
	opsReg := newOpsReg(t)
	reg := newRegWithKeys(t)
	signKey := mustRSAKey(t)
	verifyKey := mustRSAKey(t) // different key
	v := newRS256Verifier(t, verifyKey, "", "")
	auth := NewRBACAuthenticatorOpts(reg, opsReg, "INT", RBACOptions{JWTVerifier: v})

	tok := mintRS256(t, signKey, map[string]any{
		"sub": "mallory", "tenant": "acme", "scopes": "*:*", "exp": expSoon(),
	})
	if auth(AuthRequest{Token: tok, Op: "vector_search", Args: at2("default/docs")}) {
		t.Fatal("JWT signed with the wrong key must be DENIED")
	}
}

// TestJWTInvalidDoesNotTryCertCN: a failed JWT must NOT fall through to the
// cert-CN path. We bind a cert CN to a superuser key, then send a (failing) JWT
// together with that ClientCN — the JWT failure must DENY, never reach the cert
// grant.
func TestJWTInvalidDoesNotTryCertCN(t *testing.T) {
	opsReg := newOpsReg(t)
	// A cert-bound superuser key exists in the registry.
	reg := newRegWithKeys(t,
		vector.APIKey{Token: "certkey", Tenant: "*", CertCN: "trusted-cn", Scopes: []string{"*:*"}},
	)
	key := mustRSAKey(t)
	v := newRS256Verifier(t, key, "", "")
	auth := NewRBACAuthenticatorOpts(reg, opsReg, "INT", RBACOptions{JWTVerifier: v})

	expired := mintRS256(t, key, map[string]any{
		"sub": "x", "tenant": "acme", "scopes": "*:*", "exp": expPast(),
	})
	// Both a (failing) JWT token AND a valid trusted ClientCN are present. The
	// JWT branch must win and DENY — a fallthrough to the cert grant would (wrongly)
	// ALLOW via the *:* cert key.
	if auth(AuthRequest{Token: expired, ClientCN: "trusted-cn", Op: "vector_search", Args: at2("default/docs")}) {
		t.Fatal("failed JWT must DENY and NEVER fall through to the cert-CN grant")
	}
}

// --- NO-VERIFIER configured -> a JWT-looking token -> deny (byte-identical) ---

// TestNoVerifierJWTLooksLikeTokenNotFound: with NO verifier configured, a
// JWT-looking token is just an unknown registry token -> token-not-found deny,
// byte-identical to today. We also assert that adding a verifier does NOT change
// the verdict for a NON-JWT unknown token.
func TestNoVerifierJWTLooksLikeTokenNotFound(t *testing.T) {
	opsReg := newOpsReg(t)
	reg := newRegWithKeys(t)
	key := mustRSAKey(t)
	jwtTok := mintRS256(t, key, map[string]any{
		"sub": "z", "tenant": "acme", "scopes": "*:*", "exp": expSoon(),
	})

	// No verifier: the JWT-looking token is denied as token-not-found.
	cap := &captureLogger{}
	noVerif := NewRBACAuthenticatorOpts(reg, opsReg, "INT", RBACOptions{Audit: cap})
	if noVerif(AuthRequest{Token: jwtTok, Op: "vector_search", Args: at2("default/docs")}) {
		t.Fatal("no-verifier: a JWT-looking token must be DENIED")
	}
	rec := cap.snapshot()
	if len(rec) != 1 || rec[0].Reason != "token-not-found" {
		t.Fatalf("no-verifier JWT-looking token must audit token-not-found, got %+v", rec)
	}

	// Byte-identical verdict for a NON-JWT unknown token with and without a verifier.
	withVerif := NewRBACAuthenticatorOpts(reg, opsReg, "INT", RBACOptions{JWTVerifier: newRS256Verifier(t, key, "", "")})
	const nonJWT = "not-a-jwt-token"
	if noVerif(AuthRequest{Token: nonJWT, Op: "vector_search", Args: at2("default/docs")}) !=
		withVerif(AuthRequest{Token: nonJWT, Op: "vector_search", Args: at2("default/docs")}) {
		t.Error("a non-JWT unknown token must yield the SAME verdict with and without a verifier")
	}
	// And both deny.
	if withVerif(AuthRequest{Token: nonJWT, Op: "vector_search", Args: at2("default/docs")}) {
		t.Error("a non-JWT unknown token must be DENIED even with a verifier configured")
	}
}

// TestNoVerifierAuditByteIdentical: for a JWT-looking token, the FULL audit
// record (sans Time) emitted with no verifier matches the record emitted by the
// pre-JWT engine — proving the no-verifier path is byte-identical. We approximate
// "pre-JWT" by the zero-options engine, which never enters the JWT branch.
func TestNoVerifierAuditByteIdentical(t *testing.T) {
	opsReg := newOpsReg(t)
	reg := newRegWithKeys(t)
	key := mustRSAKey(t)
	jwtTok := mintRS256(t, key, map[string]any{
		"sub": "z", "tenant": "acme", "scopes": "*:*", "exp": expSoon(),
	})
	req := AuthRequest{Token: jwtTok, Op: "vector_search", Args: at2("default/docs")}

	capA := &captureLogger{}
	authA := NewRBACAuthenticatorOpts(reg, opsReg, "INT", RBACOptions{Audit: capA})
	authA(req)

	capB := &captureLogger{}
	authB := NewRBACAuthenticatorOpts(reg, opsReg, "INT", RBACOptions{Audit: capB})
	authB(req)

	a, b := capA.snapshot(), capB.snapshot()
	if len(a) != 1 || len(b) != 1 {
		t.Fatalf("want 1 record each, got %d / %d", len(a), len(b))
	}
	a[0].Time, b[0].Time = time.Time{}, time.Time{} // ignore Time
	if a[0] != b[0] {
		t.Errorf("no-verifier records differ:\n a=%+v\n b=%+v", a[0], b[0])
	}
}

// --- AUDIT TEETH: the raw JWT is ABSENT from every emitted record ------------

// TestJWTAuditTeethNoRawJWT mints a JWT carrying a distinctive sentinel substring
// in its claims and asserts the raw JWT (and the sentinel) NEVER appears in any
// marshaled audit record — only the fingerprint. Covers allow, scope-miss, and
// jwt-invalid records.
func TestJWTAuditTeethNoRawJWT(t *testing.T) {
	opsReg := newOpsReg(t)
	reg := newRegWithKeys(t)
	key := mustRSAKey(t)
	v := newRS256Verifier(t, key, "", "")
	cap := &captureLogger{}
	auth := NewRBACAuthenticatorOpts(reg, opsReg, "INT", RBACOptions{JWTVerifier: v, Audit: cap})

	const sentinel = "SENTINEL-JWT-SECRET-XYZ"
	// A valid JWT whose tenant carries the sentinel as a SECRET-bearing claim that
	// is NOT the principal id (sub is "y"). The sentinel rides ONLY inside the raw
	// JWT bytes; it must never surface in a record unless the raw token leaked.
	// (Note: the `sub` claim IS deliberately the redacted principal, so a sentinel
	// there would correctly appear — we test a NON-principal claim here.)
	validTok := mintRS256(t, key, map[string]any{
		"sub": "redacted-principal", "tenant": "acme",
		"scopes": "read:default/docs " + sentinel + ":nope", "exp": expSoon(),
	})
	auth(AuthRequest{Token: validTok, Op: "vector_search", Args: at2("default/docs")}) // allow
	auth(AuthRequest{Token: validTok, Op: "vector_insert", Args: at2("default/docs")}) // scope-miss

	// An expired JWT -> jwt-invalid record.
	expiredTok := mintRS256(t, key, map[string]any{
		"sub": "y", "tenant": "acme", "scopes": "*:*", "exp": expPast(),
	})
	auth(AuthRequest{Token: expiredTok, Op: "vector_search", Args: at2("default/docs")})

	for i, r := range cap.snapshot() {
		blob, err := json.Marshal(r)
		if err != nil {
			t.Fatalf("marshal record %d: %v", i, err)
		}
		s := string(blob)
		if strings.Contains(s, validTok) || strings.Contains(s, expiredTok) {
			t.Errorf("record %d leaks the raw JWT: %s", i, s)
		}
		if strings.Contains(s, sentinel) {
			t.Errorf("record %d leaks the sentinel claim: %s", i, s)
		}
		// the fingerprint MUST be present (correlation without the secret).
		if !strings.Contains(s, "token_fp") {
			t.Errorf("record %d missing the fingerprint: %s", i, s)
		}
	}
}

// --- REGRESSION: registry/cert/internal paths unchanged with a verifier ON ---

// TestRegistryPathsUnchangedWithVerifier: with a JWT verifier configured, the
// internal-token, registry-hit (allow + deny), and cert-CN paths produce the
// SAME decisions as without it — the JWT branch is entered ONLY on a registry
// miss, so a real registry token / cert never touches it.
func TestRegistryPathsUnchangedWithVerifier(t *testing.T) {
	opsReg := newOpsReg(t)
	reg := newRegWithKeys(t,
		vector.APIKey{Token: "reader", Tenant: "acme", Scopes: []string{"read:default/docs"}},
		vector.APIKey{Token: "certguy", Tenant: "acme", CertCN: "cn-svc", Scopes: []string{"read:default/docs"}},
	)
	key := mustRSAKey(t)
	v := newRS256Verifier(t, key, "", "")
	auth := NewRBACAuthenticatorOpts(reg, opsReg, "INT", RBACOptions{JWTVerifier: v})

	cases := []struct {
		req  AuthRequest
		want bool
		desc string
	}{
		{AuthRequest{Token: "INT", Op: "vector_search", Args: at2("anything/x")}, true, "internal-token superuser"},
		{AuthRequest{Token: "reader", Op: "vector_search", Args: at2("default/docs")}, true, "registry-hit allow"},
		{AuthRequest{Token: "reader", Op: "vector_insert", Args: at2("default/docs")}, false, "registry-hit deny (no write scope)"},
		{AuthRequest{ClientCN: "cn-svc", Op: "vector_search", Args: at2("default/docs")}, true, "cert-CN allow"},
		{AuthRequest{ClientCN: "cn-svc", Op: "vector_insert", Args: at2("default/docs")}, false, "cert-CN deny (no write scope)"},
		{AuthRequest{ClientCN: "unknown-cn", Op: "vector_search", Args: at2("default/docs")}, false, "cert-CN not found deny"},
	}
	for _, c := range cases {
		if got := auth(c.req); got != c.want {
			t.Errorf("%s: got %v want %v", c.desc, got, c.want)
		}
	}
}

// --- helpers -----------------------------------------------------------------

func expSoon() int64 { return validClaims()["exp"].(int64) }
func expPast() int64 { return validClaims()["exp"].(int64) - 7200 }
