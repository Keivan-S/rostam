// SPDX-License-Identifier: Apache-2.0

package authz

import (
	"encoding/json"
	"os"
	"path/filepath"
	"testing"

	"github.com/rostamlabs/rostam/ops"
	"github.com/rostamlabs/rostam/vector"
)

// --- wire-layout helpers (mirror vectorKeyColAt1/At2) ---

// at1 builds [colLen:u8][col][rest...] (the At1 op layout).
func at1(col string, rest ...byte) []byte {
	out := append([]byte{byte(len(col))}, col...)
	return append(out, rest...)
}

// at2 builds [flags:u8][colLen:u8][col][rest...] (the At2 op layout).
func at2(col string, rest ...byte) []byte {
	out := append([]byte{0x00, byte(len(col))}, col...)
	return append(out, rest...)
}

func newOpsReg(t *testing.T) *ops.Registry {
	t.Helper()
	r := ops.NewRegistry()
	if err := ops.RegisterBuiltins(r); err != nil {
		t.Fatalf("RegisterBuiltins: %v", err)
	}
	return r
}

// --- Step 2: scopeGrants matrix ---

func TestScopeGrantsMatrix(t *testing.T) {
	cases := []struct {
		scope, action, resource string
		want                    bool
		desc                    string
	}{
		// exact match
		{"read:default/docs", "read", "default/docs", true, "exact read grants read on same coll"},
		{"read:default/docs", "write", "default/docs", false, "read scope does not grant write"},
		{"read:default/docs", "read", "default/other", false, "exact scope does not grant other coll"},
		// namespace / prefix glob
		{"read:default/*", "read", "default/x", true, "namespace glob grants any in namespace"},
		{"read:default/*", "read", "default/docs", true, "namespace glob grants docs"},
		{"read:default/*", "read", "other/x", false, "namespace glob denies other namespace"},
		{"read:default/docs*", "read", "default/docs", true, "prefix glob grants exact prefix"},
		{"read:default/docs*", "read", "default/docsX", true, "prefix glob grants extended"},
		{"read:default/docs*", "read", "default/do", false, "prefix glob denies shorter"},
		// superuser
		{"*:*", "admin", "anything", true, "*:* grants admin on anything"},
		{"*:*", "write", "", true, "*:* grants write on empty resource"},
		{"*:*", "read", "default/docs", true, "*:* grants read on a coll"},
		// wildcard pattern, specific action
		{"read:*", "read", "", true, "read:* grants read on empty resource"},
		{"read:*", "read", "default/anycoll", true, "read:* grants read on any coll"},
		{"read:*", "write", "default/anycoll", false, "read:* does not grant write"},
		{"read:*", "write", "", false, "read:* does not grant write on empty"},
		// wildcard action, specific pattern
		{"*:default/docs", "read", "default/docs", true, "*:pattern grants any action on pattern"},
		{"*:default/docs", "admin", "default/docs", true, "*:pattern grants admin on pattern"},
		{"*:default/docs", "read", "default/other", false, "*:pattern denies other coll"},
		// empty-resource rule: only matched by "*"
		{"read:default/docs", "read", "", false, "specific pattern never matches empty resource"},
		{"read:default/*", "read", "", false, "namespace glob never matches empty resource"},
		{"admin:*", "admin", "", true, "admin:* matches empty resource"},
		// malformed / empty
		{"readdefault/docs", "read", "default/docs", false, "malformed scope (no colon) denies"},
		{"read:", "read", "default/docs", false, "empty pattern denies"},
		{"read:", "read", "", false, "empty pattern denies even empty resource"},
		{"", "read", "default/docs", false, "empty scope denies"},
		// action mismatch
		{"write:default/docs", "read", "default/docs", false, "write scope does not grant read action"},
	}
	for _, c := range cases {
		if got := scopeGrants(c.scope, c.action, c.resource); got != c.want {
			t.Errorf("scopeGrants(%q,%q,%q)=%v want %v — %s",
				c.scope, c.action, c.resource, got, c.want, c.desc)
		}
	}
}

func TestKeyGrantsEmptyAndAny(t *testing.T) {
	if keyGrants(nil, "read", "default/docs") {
		t.Error("nil scopes must deny")
	}
	if keyGrants([]string{}, "read", "default/docs") {
		t.Error("empty scopes must deny")
	}
	if !keyGrants([]string{"write:default/*", "read:default/docs"}, "read", "default/docs") {
		t.Error("any matching scope must grant")
	}
	if keyGrants([]string{"write:default/*", "read:default/other"}, "read", "default/docs") {
		t.Error("no matching scope must deny")
	}
}

// --- Step 3: actionFor ---

func TestActionFor(t *testing.T) {
	reg := newOpsReg(t)
	cases := []struct {
		op   string
		want string
	}{
		{"vector_search", ActionRead},
		{"vector_get", ActionRead},
		{"vector_scroll", ActionRead},
		{"vector_search_docs", ActionRead},
		{"vector_insert", ActionWrite},
		{"vector_delete", ActionWrite},
		{"vector_set_payload", ActionWrite},
		{"vector_upsert", ActionWrite},
		{"vector_create_collection", ActionAdmin},
		{"vector_drop_collection", ActionAdmin},
		{"vector_mv_create_collection", ActionAdmin},
		{"vector_named_drop_collection", ActionAdmin},
		{"vector_reshard", ActionAdmin},
		{"vector_resplit", ActionAdmin},
		{"vector_mv_reshard_abort", ActionAdmin},
		{"vector_resplit_cleanup", ActionAdmin},
		{"alias_batch", ActionAdmin},
		{"alias_list", ActionAdmin},
		{"__rb_status__", ActionAdmin},
		{"__rb_add_owner__", ActionAdmin},
		{"__ping__", ActionRead},
		{"__topology__", ActionRead},
		{"totally_unknown_op", ActionAdmin}, // deny-by-default
		{"get", ActionRead},                 // KV read
		{"put", ActionWrite},                // KV write
	}
	for _, c := range cases {
		if got := actionFor(c.op, reg); got != c.want {
			t.Errorf("actionFor(%q)=%q want %q", c.op, got, c.want)
		}
	}
}

func TestActionForNilRegistryFailsClosed(t *testing.T) {
	// With a nil registry, a registry-only op (vector_search) cannot be classified
	// and must fall through to admin (highest privilege). Admin-set and read-set
	// ops are still classified from the hardcoded sets.
	if got := actionFor("vector_search", nil); got != ActionAdmin {
		t.Errorf("actionFor(vector_search,nil)=%q want admin (fail-closed)", got)
	}
	if got := actionFor("vector_create_collection", nil); got != ActionAdmin {
		t.Errorf("actionFor admin op with nil reg = %q want admin", got)
	}
	if got := actionFor("__ping__", nil); got != ActionRead {
		t.Errorf("actionFor(__ping__,nil)=%q want read", got)
	}
}

// --- Step 3: resourceFor ---

func TestResourceFor(t *testing.T) {
	cases := []struct {
		op   string
		args []byte
		want string
		desc string
	}{
		{"vector_search", at2("docs"), "default/docs", "bare name canonicalized"},
		{"vector_search", at2("default/docs"), "default/docs", "qualified name passes through"},
		{"vector_insert", at2("default/docs#3"), "default/docs", "partitioned physical name stripped at #"},
		{"vector_search", at2("default/docs@1#0"), "default/docs", "generation+partition stripped at @"},
		{"vector_drop_collection", at1("docs"), "default/docs", "At1 op canonicalized"},
		{"vector_bulk_stage", at1("docs"), "default/docs", "bulk-stage is collection-keyed (per-collection write scope applies)"},
		{"vector_bulk_build", at1("docs"), "default/docs", "bulk-build is collection-keyed"},
		// NOTE: dense vector_create_collection is NOT in CollectionNameFor's switch
		// (only drop + named/mv-drop + named-create are), so its resource is ""
		// (cluster). It is an admin action, matched by admin:* / *:* (pattern "*").
		// A per-collection admin scope (admin:default/*) would DENY it — fail-closed.
		{"vector_create_collection", at1("docs"), "", "dense create is not collection-keyed in CollectionNameFor -> empty resource"},
		{"vector_named_create_collection", at1("docs"), "default/docs", "named create IS collection-keyed"},
		{"vector_reshard", at1("docs"), "", "reshard is not collection-keyed -> empty resource"},
		{"alias_batch", []byte{0x01, 0x02}, "", "alias op -> empty resource"},
		{"get", []byte{0x00, 0x03, 'a', 'b', 'c'}, "", "KV op -> empty resource"},
		{"__ping__", nil, "", "ping -> empty resource"},
	}
	for _, c := range cases {
		if got := resourceFor(c.op, c.args); got != c.want {
			t.Errorf("resourceFor(%q,..)=%q want %q — %s", c.op, got, c.want, c.desc)
		}
	}
}

// --- Step 4/5: end-to-end NewRBACAuthenticator ---

func newRegWithKeys(t *testing.T, keys ...vector.APIKey) *vector.KeyRegistry {
	t.Helper()
	dir := t.TempDir()
	reg, err := vector.OpenKeyRegistry(filepath.Join(dir, "keys.json"))
	if err != nil {
		t.Fatalf("OpenKeyRegistry: %v", err)
	}
	for _, k := range keys {
		if err := reg.AddKey(k); err != nil {
			t.Fatalf("AddKey %q: %v", k.Token, err)
		}
	}
	return reg
}

func TestRBACEndToEnd(t *testing.T) {
	opsReg := newOpsReg(t)
	reg := newRegWithKeys(t,
		vector.APIKey{Token: "reader", Tenant: "acme", Scopes: []string{"read:default/docs"}},
		vector.APIKey{Token: "admin", Tenant: "acme", Scopes: []string{"admin:*"}},
		vector.APIKey{Token: "super", Tenant: "acme", Scopes: []string{"*:*"}},
		vector.APIKey{Token: "certguy", Tenant: "acme", CertCN: "cn-svc", Scopes: []string{"read:default/docs"}},
	)
	auth := NewRBACAuthenticator(reg, opsReg, "INTERNAL-SVC-TOKEN")

	cases := []struct {
		req  AuthRequest
		want bool
		desc string
	}{
		// read:default/docs key
		{AuthRequest{Token: "reader", Op: "vector_search", Args: at2("docs")}, true, "reader can search docs"},
		{AuthRequest{Token: "reader", Op: "vector_insert", Args: at2("docs")}, false, "reader cannot insert docs"},
		{AuthRequest{Token: "reader", Op: "vector_search", Args: at2("other")}, false, "reader cannot search other"},
		{AuthRequest{Token: "reader", Op: "vector_create_collection", Args: at1("docs")}, false, "reader cannot create"},
		// admin:* key (admin action on any resource; but NOT read/write on a coll — admin action only)
		{AuthRequest{Token: "admin", Op: "vector_create_collection", Args: at1("docs")}, true, "admin can create"},
		{AuthRequest{Token: "admin", Op: "vector_drop_collection", Args: at1("docs")}, true, "admin can drop"},
		{AuthRequest{Token: "admin", Op: "vector_reshard", Args: at1("docs")}, true, "admin can reshard"},
		{AuthRequest{Token: "admin", Op: "vector_search", Args: at2("docs")}, false, "admin:* does not grant read action"},
		// superuser
		{AuthRequest{Token: "super", Op: "vector_search", Args: at2("docs")}, true, "super reads"},
		{AuthRequest{Token: "super", Op: "vector_insert", Args: at2("docs")}, true, "super writes"},
		{AuthRequest{Token: "super", Op: "vector_create_collection", Args: at1("docs")}, true, "super creates"},
		{AuthRequest{Token: "super", Op: "totally_unknown_op", Args: nil}, true, "super does even unknown(admin) ops"},
		// internal token = superuser regardless of registry
		{AuthRequest{Token: "INTERNAL-SVC-TOKEN", Op: "vector_create_collection", Args: at1("docs")}, true, "internal token creates"},
		{AuthRequest{Token: "INTERNAL-SVC-TOKEN", Op: "totally_unknown_op", Args: nil}, true, "internal token does anything"},
		// no principal
		{AuthRequest{Op: "vector_search", Args: at2("docs")}, false, "no token + no CN denied"},
		{AuthRequest{Token: "nope", Op: "vector_search", Args: at2("docs")}, false, "unknown token denied"},
		// cert-CN principal (no token)
		{AuthRequest{ClientCN: "cn-svc", Op: "vector_search", Args: at2("docs")}, true, "cert CN authorizes by its scopes"},
		{AuthRequest{ClientCN: "cn-svc", Op: "vector_insert", Args: at2("docs")}, false, "cert CN denied write"},
		{AuthRequest{ClientCN: "unknown-cn", Op: "vector_search", Args: at2("docs")}, false, "unknown CN denied"},
		{AuthRequest{ClientCN: "", Op: "vector_search", Args: at2("docs")}, false, "empty CN denied"},
		// token takes precedence over CN
		{AuthRequest{Token: "reader", ClientCN: "cn-svc", Op: "vector_search", Args: at2("other")}, false, "token wins: reader cannot read other even with cert CN set"},
	}
	for _, c := range cases {
		if got := auth(c.req); got != c.want {
			t.Errorf("auth(%+v)=%v want %v — %s", c.req, got, c.want, c.desc)
		}
	}
}

// Internal token disabled (empty) means the literal empty-string token never
// short-circuits; a request with an empty token resolves no principal -> deny.
func TestRBACInternalTokenEmptyNeverGrants(t *testing.T) {
	opsReg := newOpsReg(t)
	reg := newRegWithKeys(t)
	auth := NewRBACAuthenticator(reg, opsReg, "")
	if auth(AuthRequest{Token: "", Op: "vector_search", Args: at2("docs")}) {
		t.Error("empty internal token must not let an empty-token request through")
	}
}

// A nil key registry denies everything except the internal-token superuser.
func TestRBACNilRegistryFailsClosed(t *testing.T) {
	opsReg := newOpsReg(t)
	auth := NewRBACAuthenticator(nil, opsReg, "INT")
	if auth(AuthRequest{Token: "anything", Op: "vector_search", Args: at2("docs")}) {
		t.Error("nil registry must deny a normal token")
	}
	if !auth(AuthRequest{Token: "INT", Op: "vector_create_collection", Args: at1("docs")}) {
		t.Error("internal token must still be superuser with a nil registry")
	}
}

// Backward-compat: an OLD Permissions-only key FILE (no Scopes) loads with
// synthesized scopes (read->read:*, write->read:*+write:*, admin->*:*) and
// authorizes identically to an explicit-scope key. Synthesis happens on load
// (OpenKeyRegistry), so this test writes a legacy JSON file and reopens it
// rather than using AddKey (which stores verbatim).
func TestRBACBackwardCompatPermissionsKey(t *testing.T) {
	opsReg := newOpsReg(t)
	dir := t.TempDir()
	path := filepath.Join(dir, "keys.json")
	legacy := []vector.APIKey{
		{Token: "oldread", Tenant: "acme", Permissions: []string{vector.PermRead}},
		{Token: "oldwrite", Tenant: "acme", Permissions: []string{vector.PermWrite}},
		{Token: "oldadmin", Tenant: "acme", Permissions: []string{vector.PermAdmin}},
	}
	data, err := json.MarshalIndent(legacy, "", "  ")
	if err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(path, data, 0o600); err != nil {
		t.Fatal(err)
	}
	reg, err := vector.OpenKeyRegistry(path)
	if err != nil {
		t.Fatalf("OpenKeyRegistry: %v", err)
	}
	auth := NewRBACAuthenticator(reg, opsReg, "")

	// read perm -> read:* : can search any coll, cannot write.
	if !auth(AuthRequest{Token: "oldread", Op: "vector_search", Args: at2("docs")}) {
		t.Error("old read key should authorize a search")
	}
	if auth(AuthRequest{Token: "oldread", Op: "vector_insert", Args: at2("docs")}) {
		t.Error("old read key should NOT authorize a write")
	}
	// write perm -> read:*,write:* : can read and write, cannot admin.
	if !auth(AuthRequest{Token: "oldwrite", Op: "vector_insert", Args: at2("docs")}) {
		t.Error("old write key should authorize a write")
	}
	if !auth(AuthRequest{Token: "oldwrite", Op: "vector_search", Args: at2("docs")}) {
		t.Error("old write key should authorize a read too")
	}
	if auth(AuthRequest{Token: "oldwrite", Op: "vector_create_collection", Args: at1("docs")}) {
		t.Error("old write key should NOT authorize admin (create)")
	}
	// admin perm -> *:* : superuser.
	if !auth(AuthRequest{Token: "oldadmin", Op: "vector_create_collection", Args: at1("docs")}) {
		t.Error("old admin key should authorize create")
	}
}
