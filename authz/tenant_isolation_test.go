// SPDX-License-Identifier: Apache-2.0

package authz

import (
	"encoding/json"
	"strings"
	"testing"

	"github.com/rostamlabs/rostam/vector"
)

// --- resourceTenant: single source of truth with the storage-layout parse ---

// TestResourceTenant pins the tenant-parse rule the guard uses: the substring
// before the FIRST '/', or "" for the empty/cluster resource or a name with no
// tenant prefix. It delegates to vector.TenantOf, so this is also the contract
// that must match how a collection is actually keyed on disk (splitTenant).
func TestResourceTenant(t *testing.T) {
	cases := []struct {
		resource, want string
	}{
		{"acme/docs", "acme"},
		{"default/docs", "default"},
		{"acme/docs/extra", "acme"}, // first '/' wins (defensive; canonical names have one)
		{"docs", ""},                // no prefix -> no tenant
		{"", ""},                    // cluster resource -> no tenant
		{"/docs", ""},               // leading '/' (empty tenant) -> no tenant
	}
	for _, c := range cases {
		if got := resourceTenant(c.resource); got != c.want {
			t.Errorf("resourceTenant(%q)=%q want %q", c.resource, got, c.want)
		}
	}
}

// TestResourceTenantMatchesResourceFor confirms the guard parses the SAME tenant
// out of a canonical resourceFor output that the storage layout would: for any
// wire name resourceFor canonicalizes, resourceTenant(resourceFor(...)) is the
// leading path component. This is the "a malformed resource cannot fool the
// guard" invariant — the parse is shared, never reinvented.
func TestResourceTenantMatchesResourceFor(t *testing.T) {
	cases := []struct {
		op, col, want string
	}{
		{"vector_search", "acme/docs", "acme"},
		{"vector_search", "docs", "default"},       // bare -> canonical default/docs
		{"vector_insert", "acme/docs#3", "acme"},   // physical suffix stripped first
		{"vector_search", "acme/docs@1#0", "acme"}, // gen+part suffix stripped first
	}
	for _, c := range cases {
		res := resourceFor(c.op, at2(c.col))
		if got := resourceTenant(res); got != c.want {
			t.Errorf("resourceTenant(resourceFor(%q,%q))=%q (res=%q) want %q",
				c.op, c.col, got, res, c.want)
		}
	}
}

// --- OFF (default) is byte/behaviour-identical to today ---

// TestTenantIsolationOffIsScopeOnly is the #1 no-break criterion: with isolation
// OFF (the default), a key whose Tenant is "default" but whose scope grants
// "read:acme/*" STILL reads acme/docs — exactly today's scope-only behavior. The
// guard must be entirely skipped when off.
func TestTenantIsolationOffIsScopeOnly(t *testing.T) {
	opsReg := newOpsReg(t)
	reg := newRegWithKeys(t,
		// Tenant=default but scoped cross-tenant into acme (a mis-scoped key).
		vector.APIKey{Token: "k", Tenant: "default", Scopes: []string{"read:default/docs", "read:acme/*"}},
	)

	// All three un-optioned constructors must behave identically (off by default).
	auths := map[string]Authenticator{
		"NewRBACAuthenticator":          NewRBACAuthenticator(reg, opsReg, "INT"),
		"NewRBACAuthenticatorWithAudit": NewRBACAuthenticatorWithAudit(reg, opsReg, "INT", nil),
		"NewRBACAuthenticatorOpts-zero": NewRBACAuthenticatorOpts(reg, opsReg, "INT", RBACOptions{}),
	}
	for name, auth := range auths {
		if !auth(AuthRequest{Token: "k", Op: "vector_search", Args: at2("acme/docs")}) {
			t.Errorf("%s: OFF must grant cross-tenant scope (scope-only behavior) for acme/docs", name)
		}
		if !auth(AuthRequest{Token: "k", Op: "vector_search", Args: at2("default/docs")}) {
			t.Errorf("%s: OFF must grant own-tenant scoped read default/docs", name)
		}
	}
}

// --- ON enforces the tenant boundary ---

// TestTenantIsolationOnEnforces: the SAME mis-scoped key (Tenant=default, scope
// read:acme/*) is DENIED on acme/docs under isolation ON (tenant-mismatch), but
// its own-tenant scoped read (default/docs) still succeeds.
func TestTenantIsolationOnEnforces(t *testing.T) {
	opsReg := newOpsReg(t)
	reg := newRegWithKeys(t,
		vector.APIKey{Token: "k", Tenant: "default", Scopes: []string{"read:default/docs", "read:acme/*"}},
	)
	auth := NewRBACAuthenticatorOpts(reg, opsReg, "INT", RBACOptions{TenantIsolation: true})

	if auth(AuthRequest{Token: "k", Op: "vector_search", Args: at2("acme/docs")}) {
		t.Error("ON must DENY a cross-tenant resource even though the scope grants it (tenant-mismatch)")
	}
	if !auth(AuthRequest{Token: "k", Op: "vector_search", Args: at2("default/docs")}) {
		t.Error("ON must still ALLOW an own-tenant scoped read (default/docs)")
	}
}

// TestTenantIsolationSingleTenant: a single-tenant key (Tenant=acme, scope
// read:acme/*) is allowed on acme/* and denied on other/* under ON.
func TestTenantIsolationSingleTenant(t *testing.T) {
	opsReg := newOpsReg(t)
	reg := newRegWithKeys(t,
		vector.APIKey{Token: "acmekey", Tenant: "acme", Scopes: []string{"read:acme/*"}},
		// Deliberately broad scope but tenant-bound: scope grants other/* too, but
		// the guard must clamp it to the key's tenant.
		vector.APIKey{Token: "broad", Tenant: "acme", Scopes: []string{"read:*"}},
	)
	auth := NewRBACAuthenticatorOpts(reg, opsReg, "INT", RBACOptions{TenantIsolation: true})

	if !auth(AuthRequest{Token: "acmekey", Op: "vector_search", Args: at2("acme/x")}) {
		t.Error("acme key must read acme/x")
	}
	if auth(AuthRequest{Token: "acmekey", Op: "vector_search", Args: at2("other/x")}) {
		t.Error("acme key must NOT read other/x under ON")
	}
	// read:* grants any resource by scope, but the guard clamps to tenant=acme.
	if !auth(AuthRequest{Token: "broad", Op: "vector_search", Args: at2("acme/y")}) {
		t.Error("read:* acme key must read acme/y")
	}
	if auth(AuthRequest{Token: "broad", Op: "vector_search", Args: at2("other/y")}) {
		t.Error("read:* acme key must NOT read other/y under ON (guard clamps to tenant)")
	}
}

// --- Exemptions under ON ---

// TestTenantIsolationStarTenantExempt: a Tenant="*" key is the cross-tenant/admin
// marker and is exempt — any tenant its scopes grant is allowed under ON.
func TestTenantIsolationStarTenantExempt(t *testing.T) {
	opsReg := newOpsReg(t)
	reg := newRegWithKeys(t,
		vector.APIKey{Token: "cross", Tenant: "*", Scopes: []string{"read:*"}},
	)
	auth := NewRBACAuthenticatorOpts(reg, opsReg, "INT", RBACOptions{TenantIsolation: true})

	for _, res := range []string{"acme/docs", "default/docs", "other/x"} {
		if !auth(AuthRequest{Token: "cross", Op: "vector_search", Args: at2(res)}) {
			t.Errorf("Tenant=* key must be exempt and allowed on %q", res)
		}
	}
}

// TestTenantIsolationInternalTokenUntouched: the internal-service token is still
// superuser under ON (its early-return is before key resolution — untouched).
func TestTenantIsolationInternalTokenUntouched(t *testing.T) {
	opsReg := newOpsReg(t)
	reg := newRegWithKeys(t) // empty registry
	auth := NewRBACAuthenticatorOpts(reg, opsReg, "INTERNAL-SVC", RBACOptions{TenantIsolation: true})

	if !auth(AuthRequest{Token: "INTERNAL-SVC", Op: "vector_search", Args: at2("acme/docs")}) {
		t.Error("internal token must remain superuser under ON")
	}
	if !auth(AuthRequest{Token: "INTERNAL-SVC", Op: "vector_create_collection", Args: at1("docs")}) {
		t.Error("internal token must remain superuser for cluster ops under ON")
	}
}

// --- The guard only ever REMOVES access (never grants) ---

// TestTenantIsolationGuardNeverGrants: a resource the SCOPE denies stays denied
// regardless of tenant, ON or OFF. The guard runs only after a scope-grant; it
// can never turn a scope-miss into an allow.
func TestTenantIsolationGuardNeverGrants(t *testing.T) {
	opsReg := newOpsReg(t)
	reg := newRegWithKeys(t,
		// Scope grants ONLY read on acme/docs; no write, no other collection.
		vector.APIKey{Token: "k", Tenant: "acme", Scopes: []string{"read:acme/docs"}},
	)
	off := NewRBACAuthenticatorOpts(reg, opsReg, "INT", RBACOptions{})
	on := NewRBACAuthenticatorOpts(reg, opsReg, "INT", RBACOptions{TenantIsolation: true})

	// A write the scope denies stays denied (own tenant) under BOTH off and on.
	for _, tc := range []struct {
		auth Authenticator
		name string
	}{{off, "OFF"}, {on, "ON"}} {
		if tc.auth(AuthRequest{Token: "k", Op: "vector_insert", Args: at2("acme/docs")}) {
			t.Errorf("%s: scope denies write on acme/docs -> must stay denied (guard never grants)", tc.name)
		}
		// A different collection the scope never granted stays denied.
		if tc.auth(AuthRequest{Token: "k", Op: "vector_search", Args: at2("acme/other")}) {
			t.Errorf("%s: scope denies acme/other -> must stay denied (guard never grants)", tc.name)
		}
	}
}

// --- Cluster-resource case under ON ---

// TestTenantIsolationClusterResourceDenied: a tenant-scoped key that the SCOPE
// would grant on a cluster resource (resource=="") is denied under ON — a
// tenant-scoped key has no cluster-admin business. A Tenant="*" key is allowed.
func TestTenantIsolationClusterResourceDenied(t *testing.T) {
	opsReg := newOpsReg(t)
	reg := newRegWithKeys(t,
		// *:* scope grants admin on the empty cluster resource by scope.
		vector.APIKey{Token: "tenantsuper", Tenant: "acme", Scopes: []string{"*:*"}},
		vector.APIKey{Token: "crosssuper", Tenant: "*", Scopes: []string{"*:*"}},
	)
	auth := NewRBACAuthenticatorOpts(reg, opsReg, "INT", RBACOptions{TenantIsolation: true})

	// vector_reshard is a cluster op (resource==""); *:* grants it by scope.
	if auth(AuthRequest{Token: "tenantsuper", Op: "vector_reshard", Args: at1("docs")}) {
		t.Error("tenant-scoped key must be DENIED on a cluster resource under ON")
	}
	// Sanity: OFF, the same key IS allowed (scope-only) — proving the deny is the guard.
	authOff := NewRBACAuthenticatorOpts(reg, opsReg, "INT", RBACOptions{})
	if !authOff(AuthRequest{Token: "tenantsuper", Op: "vector_reshard", Args: at1("docs")}) {
		t.Error("OFF: *:* tenant key must be allowed on a cluster op (scope-only)")
	}
	// Tenant=* cross key is exempt -> allowed on the cluster resource under ON.
	if !auth(AuthRequest{Token: "crosssuper", Op: "vector_reshard", Args: at1("docs")}) {
		t.Error("Tenant=* key must be allowed on a cluster resource under ON (exempt)")
	}
}

// --- Audit composition with feature 1 ---

// TestTenantIsolationAuditEmitsTenantMismatch: with a capturing logger and
// isolation ON, a tenant-mismatch deny emits EXACTLY ONE record with
// Decision="deny", Reason="tenant-mismatch" (never an allow-then-deny pair), and
// NO raw token leaks (feature-1 invariant).
func TestTenantIsolationAuditEmitsTenantMismatch(t *testing.T) {
	opsReg := newOpsReg(t)
	const rawTok = "SECRET-TENANT-TOK-999"
	reg := newRegWithKeys(t,
		vector.APIKey{Token: rawTok, Tenant: "default", Scopes: []string{"read:acme/*"}},
	)
	cap := &captureLogger{}
	auth := NewRBACAuthenticatorOpts(reg, opsReg, "INT", RBACOptions{Audit: cap, TenantIsolation: true})

	if auth(AuthRequest{Token: rawTok, Op: "vector_search", Args: at2("acme/docs")}) {
		t.Fatal("expected tenant-mismatch DENY")
	}
	recs := cap.snapshot()
	if len(recs) != 1 {
		t.Fatalf("expected exactly 1 audit record, got %d: %+v", len(recs), recs)
	}
	r := recs[0]
	if r.Decision != "deny" || r.Reason != "tenant-mismatch" {
		t.Errorf("got Decision=%q Reason=%q, want deny/tenant-mismatch", r.Decision, r.Reason)
	}
	if r.Resource != "acme/docs" || r.Tenant != "default" || r.Action != "read" {
		t.Errorf("unexpected record fields: %+v", r)
	}
	// Feature-1 invariant: the raw token must never appear in the marshaled record.
	blob, err := json.Marshal(r)
	if err != nil {
		t.Fatal(err)
	}
	if strings.Contains(string(blob), rawTok) {
		t.Errorf("raw token leaked into audit record: %s", blob)
	}
}

// TestTenantIsolationAuditAllowRecordOnSuccess: an own-tenant allowed request
// under ON still emits the scope-match allow record (the guard doesn't fire).
func TestTenantIsolationAuditAllowRecordOnSuccess(t *testing.T) {
	opsReg := newOpsReg(t)
	reg := newRegWithKeys(t,
		vector.APIKey{Token: "k", Tenant: "acme", Scopes: []string{"read:acme/*"}},
	)
	cap := &captureLogger{}
	auth := NewRBACAuthenticatorOpts(reg, opsReg, "INT", RBACOptions{Audit: cap, TenantIsolation: true})

	if !auth(AuthRequest{Token: "k", Op: "vector_search", Args: at2("acme/docs")}) {
		t.Fatal("expected own-tenant ALLOW")
	}
	recs := cap.snapshot()
	if len(recs) != 1 {
		t.Fatalf("expected exactly 1 record, got %d", len(recs))
	}
	if recs[0].Decision != "allow" || recs[0].Reason != "scope-match" {
		t.Errorf("got %q/%q, want allow/scope-match", recs[0].Decision, recs[0].Reason)
	}
}
