// SPDX-License-Identifier: Apache-2.0

package vector

import (
	"encoding/json"
	"os"
	"path/filepath"
	"reflect"
	"testing"
)

func TestKeyRegistryEmpty(t *testing.T) {
	dir := t.TempDir()
	r, err := OpenKeyRegistry(filepath.Join(dir, "auth", "keys.json"))
	if err != nil {
		t.Fatalf("OpenKeyRegistry on empty dir: %v", err)
	}
	if got := r.ListKeys(); len(got) != 0 {
		t.Errorf("empty registry ListKeys = %v, want empty", got)
	}
	if _, ok := r.Lookup("nonexistent"); ok {
		t.Error("Lookup(nonexistent) = ok, want false")
	}
}

func TestKeyRegistryAddLookupRevoke(t *testing.T) {
	dir := t.TempDir()
	path := filepath.Join(dir, "auth", "keys.json")
	r, err := OpenKeyRegistry(path)
	if err != nil {
		t.Fatal(err)
	}
	key := APIKey{
		Token:       "k_test123",
		Tenant:      "acme",
		Permissions: []string{PermRead, PermWrite},
	}
	if err := r.AddKey(key); err != nil {
		t.Fatalf("AddKey: %v", err)
	}
	if got, ok := r.Lookup("k_test123"); !ok || got.Tenant != "acme" {
		t.Errorf("Lookup after Add = (%+v, %v), want acme key", got, ok)
	}
	if err := r.AddKey(key); err != ErrAPIKeyExists {
		t.Errorf("duplicate AddKey err = %v, want ErrAPIKeyExists", err)
	}
	if err := r.RevokeKey("k_test123"); err != nil {
		t.Fatalf("RevokeKey: %v", err)
	}
	if _, ok := r.Lookup("k_test123"); ok {
		t.Error("Lookup after Revoke = ok, want false")
	}
	if err := r.RevokeKey("k_test123"); err != ErrAPIKeyNotFound {
		t.Errorf("RevokeKey of missing = %v, want ErrAPIKeyNotFound", err)
	}
}

func TestKeyRegistryPersistsAcrossReopen(t *testing.T) {
	dir := t.TempDir()
	path := filepath.Join(dir, "auth", "keys.json")

	r1, _ := OpenKeyRegistry(path)
	_ = r1.AddKey(APIKey{Token: "k_a", Tenant: "acme", Permissions: []string{PermRead}})
	_ = r1.AddKey(APIKey{Token: "k_b", Tenant: "globex", Permissions: []string{PermRead, PermWrite, PermAdmin}})

	r2, err := OpenKeyRegistry(path)
	if err != nil {
		t.Fatal(err)
	}
	if got := len(r2.ListKeys()); got != 2 {
		t.Errorf("reopened ListKeys size = %d, want 2", got)
	}
	a, ok := r2.Lookup("k_a")
	if !ok || a.Tenant != "acme" {
		t.Errorf("Lookup(k_a) = (%+v, %v)", a, ok)
	}
	b, ok := r2.Lookup("k_b")
	if !ok || !b.Has(PermAdmin) {
		t.Errorf("Lookup(k_b) = (%+v, %v), want admin permission", b, ok)
	}
}

func TestAPIKeyHas(t *testing.T) {
	k := APIKey{Permissions: []string{PermRead, PermWrite}}
	if !k.Has(PermRead) {
		t.Error("Has(read) = false, want true")
	}
	if !k.Has(PermWrite) {
		t.Error("Has(write) = false, want true")
	}
	if k.Has(PermAdmin) {
		t.Error("Has(admin) = true, want false (admin not in perm list)")
	}

	// Admin implies everything.
	adminKey := APIKey{Permissions: []string{PermAdmin}}
	if !adminKey.Has(PermRead) || !adminKey.Has(PermWrite) || !adminKey.Has(PermAdmin) {
		t.Error("admin permission should imply read+write+admin")
	}
}

func TestKeyRegistryInvalidPermission(t *testing.T) {
	dir := t.TempDir()
	r, _ := OpenKeyRegistry(filepath.Join(dir, "auth", "keys.json"))
	err := r.AddKey(APIKey{Token: "k_x", Tenant: "acme", Permissions: []string{"superuser"}})
	if err == nil {
		t.Fatal("AddKey with bogus permission = nil err")
	}
}

func TestKeyRegistryRejectEmptyTokenOrTenant(t *testing.T) {
	dir := t.TempDir()
	r, _ := OpenKeyRegistry(filepath.Join(dir, "auth", "keys.json"))
	if err := r.AddKey(APIKey{Token: "", Tenant: "acme", Permissions: []string{PermRead}}); err == nil {
		t.Error("empty token allowed")
	}
	if err := r.AddKey(APIKey{Token: "k_x", Tenant: "", Permissions: []string{PermRead}}); err == nil {
		t.Error("empty tenant allowed")
	}
}

// TestScopesForPermissionsMapping verifies the legacy Permissions -> Scopes
// synthesis mapping documented on OpenKeyRegistry: read->["read:*"],
// write->["read:*","write:*"], admin (alone or with others)->["*:*"], empty->nil.
func TestScopesForPermissionsMapping(t *testing.T) {
	cases := []struct {
		perms []string
		want  []string
	}{
		{[]string{PermRead}, []string{"read:*"}},
		{[]string{PermWrite}, []string{"read:*", "write:*"}},
		{[]string{PermRead, PermWrite}, []string{"read:*", "write:*"}},
		{[]string{PermAdmin}, []string{"*:*"}},
		{[]string{PermRead, PermWrite, PermAdmin}, []string{"*:*"}},
		{[]string{PermRead, PermAdmin}, []string{"*:*"}},
		{nil, nil},
		{[]string{}, nil},
	}
	for _, c := range cases {
		got := scopesForPermissions(c.perms)
		if !reflect.DeepEqual(got, c.want) {
			t.Errorf("scopesForPermissions(%v)=%v want %v", c.perms, got, c.want)
		}
	}
}

// TestOpenKeyRegistrySynthesizesScopesFromLegacyFile proves that an OLD key file
// carrying only Permissions (no Scopes) loads with synthesized scopes, so it
// authorizes identically under the scope engine. A file that already declares
// Scopes is taken verbatim (no override).
func TestOpenKeyRegistrySynthesizesScopesFromLegacyFile(t *testing.T) {
	dir := t.TempDir()
	path := filepath.Join(dir, "keys.json")
	legacy := []APIKey{
		{Token: "r", Tenant: "acme", Permissions: []string{PermRead}},
		{Token: "w", Tenant: "acme", Permissions: []string{PermWrite}},
		{Token: "a", Tenant: "acme", Permissions: []string{PermAdmin}},
		// An entry that already has explicit scopes must NOT be overwritten.
		{Token: "explicit", Tenant: "acme", Permissions: []string{PermAdmin}, Scopes: []string{"read:default/docs"}},
	}
	data, err := json.MarshalIndent(legacy, "", "  ")
	if err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(path, data, 0o600); err != nil {
		t.Fatal(err)
	}

	r, err := OpenKeyRegistry(path)
	if err != nil {
		t.Fatalf("OpenKeyRegistry: %v", err)
	}
	want := map[string][]string{
		"r":        {"read:*"},
		"w":        {"read:*", "write:*"},
		"a":        {"*:*"},
		"explicit": {"read:default/docs"},
	}
	for tok, ws := range want {
		k, ok := r.Lookup(tok)
		if !ok {
			t.Fatalf("Lookup(%q) missing", tok)
		}
		if !reflect.DeepEqual(k.Scopes, ws) {
			t.Errorf("key %q scopes=%v want %v", tok, k.Scopes, ws)
		}
	}
}

// TestLookupByCN verifies CertCN indexing: an entry is found by its CN, an empty
// CN never matches, and AddKey keeps the index current.
func TestLookupByCN(t *testing.T) {
	dir := t.TempDir()
	r, err := OpenKeyRegistry(filepath.Join(dir, "keys.json"))
	if err != nil {
		t.Fatal(err)
	}
	if err := r.AddKey(APIKey{Token: "svc", Tenant: "acme", CertCN: "svc.acme", Scopes: []string{"*:*"}}); err != nil {
		t.Fatal(err)
	}
	if err := r.AddKey(APIKey{Token: "nocn", Tenant: "acme", Scopes: []string{"read:*"}}); err != nil {
		t.Fatal(err)
	}

	k, ok := r.LookupByCN("svc.acme")
	if !ok || k.Token != "svc" {
		t.Errorf("LookupByCN(svc.acme) = (%+v,%v), want svc/true", k, ok)
	}
	if _, ok := r.LookupByCN(""); ok {
		t.Error("LookupByCN(\"\") must never match")
	}
	if _, ok := r.LookupByCN("unknown"); ok {
		t.Error("LookupByCN(unknown) must not match")
	}

	// Revoke removes the CN index entry too.
	if err := r.RevokeKey("svc"); err != nil {
		t.Fatal(err)
	}
	if _, ok := r.LookupByCN("svc.acme"); ok {
		t.Error("LookupByCN after revoke must not match")
	}
}

// TestLookupByCNSurvivesReopen proves the CN index is rebuilt on load.
func TestLookupByCNSurvivesReopen(t *testing.T) {
	dir := t.TempDir()
	path := filepath.Join(dir, "keys.json")
	r1, err := OpenKeyRegistry(path)
	if err != nil {
		t.Fatal(err)
	}
	if err := r1.AddKey(APIKey{Token: "svc", Tenant: "acme", CertCN: "svc.acme", Permissions: []string{PermRead}}); err != nil {
		t.Fatal(err)
	}
	r2, err := OpenKeyRegistry(path)
	if err != nil {
		t.Fatal(err)
	}
	k, ok := r2.LookupByCN("svc.acme")
	if !ok || k.Token != "svc" {
		t.Errorf("after reopen LookupByCN = (%+v,%v) want svc/true", k, ok)
	}
	// And the legacy Permissions were synthesized on reload.
	if !reflect.DeepEqual(k.Scopes, []string{"read:*"}) {
		t.Errorf("reopened key scopes=%v want [read:*]", k.Scopes)
	}
}
