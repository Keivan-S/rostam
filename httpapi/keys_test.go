// SPDX-License-Identifier: Apache-2.0

package httpapi

import (
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"path/filepath"
	"strings"
	"testing"

	"github.com/rostamlabs/rostam/authz"
	"github.com/rostamlabs/rostam/cache"
	"github.com/rostamlabs/rostam/ops"
	"github.com/rostamlabs/rostam/vector"
)

// keysTestDispatcher mirrors the production keysDispatcher (in the root rostam
// package, un-importable here without a cycle): it intercepts the three keys
// coordinator virtual-ops and applies them to a live *vector.KeyRegistry, passing
// every other op through to the ops-registry-backed inner dispatcher. This lets
// the HTTP keys endpoints round-trip against a real registry over the real
// transport without the import cycle.
type keysTestDispatcher struct {
	inner *testDispatcher
	reg   *vector.KeyRegistry
}

func (d *keysTestDispatcher) Call(name string, args []byte) ([]byte, error) {
	switch name {
	case ops.OpKeysAdd:
		a, err := ops.DecodeKeysAddArgs(args)
		if err != nil {
			return nil, err
		}
		if err := d.reg.AddKey(vector.APIKey{Token: a.Token, Tenant: a.Tenant, Scopes: a.Scopes, CertCN: a.CertCN}); err != nil {
			return nil, err
		}
		return nil, nil
	case ops.OpKeysRevoke:
		tok, err := ops.DecodeKeysRevokeArgs(args)
		if err != nil {
			return nil, err
		}
		if err := d.reg.RevokeKey(tok); err != nil {
			return nil, err
		}
		return nil, nil
	case ops.OpKeysList:
		redacted := d.reg.ListRedacted()
		entries := make([]ops.RedactedKeyEntry, len(redacted))
		for i, rk := range redacted {
			entries[i] = ops.RedactedKeyEntry{Fingerprint: rk.Fingerprint, Tenant: rk.Tenant, Scopes: rk.Scopes, CertCN: rk.CertCN}
		}
		return ops.EncodeKeysListResult(entries), nil
	default:
		return d.inner.Call(name, args)
	}
}

func (d *keysTestDispatcher) LeaderAddr() string { return "" }

// newKeysTestAPI builds an HTTP handler whose dispatcher routes the keys ops to a
// live registry and whose authorizer is RBAC-backed by that SAME registry — so an
// add takes effect on the registry the authenticator reads (live, no restart).
func newKeysTestAPI(t *testing.T) (http.Handler, *vector.KeyRegistry, func()) {
	t.Helper()
	reg := ops.NewRegistry()
	if err := ops.RegisterBuiltins(reg); err != nil {
		t.Fatal(err)
	}
	c, _ := cache.New(cache.DefaultConfig())
	vstore, err := vector.OpenCollectionStore(t.TempDir())
	if err != nil {
		t.Fatal(err)
	}
	keyReg, err := vector.OpenKeyRegistry(filepath.Join(t.TempDir(), "keys.json"))
	if err != nil {
		t.Fatal(err)
	}
	inner := &testDispatcher{reg: reg, tx: ops.NewTxContextWithVectors(c, vstore)}
	disp := &keysTestDispatcher{inner: inner, reg: keyReg}
	h := Handler(disp, Options{Authenticator: authz.NewRBACAuthenticator(keyReg, reg, "")})
	cleanup := func() { _ = vstore.Close(); c.Close() }
	return h, keyReg, cleanup
}

// keysReq issues an authenticated request and returns the status code + raw body.
func keysReq(t *testing.T, h http.Handler, token, method, path, body string) (int, string) {
	t.Helper()
	var r *http.Request
	if body == "" {
		r = httptest.NewRequest(method, path, nil)
	} else {
		r = httptest.NewRequest(method, path, strings.NewReader(body))
	}
	if token != "" {
		r.Header.Set("Authorization", "Bearer "+token)
	}
	rec := httptest.NewRecorder()
	h.ServeHTTP(rec, r)
	return rec.Code, rec.Body.String()
}

// TestHTTPKeysAddAuthenticatesThenRevokeDenies is the headline round-trip teeth
// test over the real HTTP transport: an admin POST /v1/admin/keys makes a new
// token authenticate against the LIVE registry with no restart; a DELETE
// /v1/admin/keys (token in the BODY) then makes that token fail auth.
func TestHTTPKeysAddAuthenticatesThenRevokeDenies(t *testing.T) {
	h, keyReg, cleanup := newKeysTestAPI(t)
	defer cleanup()
	mustAddKey(t, keyReg, vector.APIKey{Token: "k_admin", Tenant: "t", Scopes: []string{"*:*"}})

	const tok = "LIVE-HTTP-TOKEN"
	// Before add: the new token cannot list (not authenticated as admin).
	if code, _ := keysReq(t, h, tok, "GET", "/v1/admin/keys", ""); code != http.StatusUnauthorized {
		t.Fatalf("pre-add list by new token = %d, want 401", code)
	}
	// Admin adds the new key with admin scope.
	if code, _ := keysReq(t, h, "k_admin", "POST", "/v1/admin/keys",
		`{"token":"`+tok+`","tenant":"acme","scopes":["*:*"],"cert_cn":""}`); code != http.StatusCreated {
		t.Fatalf("admin add = %d, want 201", code)
	}
	// After add: the new token authenticates immediately (live registry, no restart).
	if code, _ := keysReq(t, h, tok, "GET", "/v1/admin/keys", ""); code != http.StatusOK {
		t.Fatalf("post-add list by new token = %d, want 200", code)
	}
	// Revoke the new token via the BODY (not the path).
	if code, _ := keysReq(t, h, "k_admin", "DELETE", "/v1/admin/keys", `{"token":"`+tok+`"}`); code != http.StatusOK {
		t.Fatalf("admin revoke = %d, want 200", code)
	}
	// After revoke: the token no longer authenticates.
	if code, _ := keysReq(t, h, tok, "GET", "/v1/admin/keys", ""); code != http.StatusUnauthorized {
		t.Fatalf("post-revoke list by revoked token = %d, want 401", code)
	}
}

// TestHTTPKeysListRedacted is the security teeth test: the GET /v1/admin/keys JSON
// carries the fingerprint + tenant + scopes + cert_cn but NEVER the raw token.
func TestHTTPKeysListRedacted(t *testing.T) {
	h, keyReg, cleanup := newKeysTestAPI(t)
	defer cleanup()
	mustAddKey(t, keyReg, vector.APIKey{Token: "k_admin", Tenant: "t", Scopes: []string{"*:*"}})

	const sentinel = "SUPER-SECRET-SENTINEL-TOKEN"
	if code, _ := keysReq(t, h, "k_admin", "POST", "/v1/admin/keys",
		`{"token":"`+sentinel+`","tenant":"acme","scopes":["read:default/docs"],"cert_cn":"cn1"}`); code != http.StatusCreated {
		t.Fatalf("add sentinel key = %d, want 201", code)
	}

	code, raw := keysReq(t, h, "k_admin", "GET", "/v1/admin/keys", "")
	if code != http.StatusOK {
		t.Fatalf("list = %d, want 200", code)
	}
	// TEETH: the raw token must NOT appear anywhere in the marshaled JSON.
	if strings.Contains(raw, sentinel) {
		t.Fatalf("list response leaked the raw token: %s", raw)
	}
	// The fingerprint MUST be present (redacted identity).
	fp := vector.TokenFingerprint(sentinel)
	if !strings.Contains(raw, fp) {
		t.Fatalf("list response missing fingerprint %q: %s", fp, raw)
	}
	// Decode and assert the redacted fields are intact (and there is no token field).
	var resp struct {
		Keys []map[string]any `json:"keys"`
	}
	if err := json.Unmarshal([]byte(raw), &resp); err != nil {
		t.Fatalf("decode list: %v", err)
	}
	var found bool
	for _, k := range resp.Keys {
		if k["fingerprint"] == fp {
			found = true
			if k["tenant"] != "acme" {
				t.Errorf("tenant = %v, want acme", k["tenant"])
			}
			if _, ok := k["token"]; ok {
				t.Errorf("redacted key must have NO token field: %v", k)
			}
		}
	}
	if !found {
		t.Fatalf("sentinel key not found in list by fingerprint: %s", raw)
	}
}

// TestHTTPKeysNonAdminDenied proves a non-admin key is denied at the authorize
// gate (401) for all three keys endpoints and the registry is never mutated.
func TestHTTPKeysNonAdminDenied(t *testing.T) {
	h, keyReg, cleanup := newKeysTestAPI(t)
	defer cleanup()
	mustAddKey(t, keyReg, vector.APIKey{Token: "k_read", Tenant: "t", Scopes: []string{"read:default/docs"}})

	cases := []struct {
		method, path, body string
	}{
		{"POST", "/v1/admin/keys", `{"token":"x","tenant":"t","scopes":["read:default/docs"]}`},
		{"DELETE", "/v1/admin/keys", `{"token":"x"}`},
		{"GET", "/v1/admin/keys", ""},
	}
	for _, tc := range cases {
		if code, _ := keysReq(t, h, "k_read", tc.method, tc.path, tc.body); code != http.StatusUnauthorized {
			t.Errorf("%s %s by non-admin = %d, want 401", tc.method, tc.path, code)
		}
	}
	// The denied add never reached the registry.
	if len(keyReg.ListRedacted()) != 1 {
		t.Fatalf("registry mutated by a denied non-admin add: %d keys", len(keyReg.ListRedacted()))
	}
}

// TestHTTPKeysBadInput proves edge validation: missing token/tenant → 400 before
// dispatch.
func TestHTTPKeysBadInput(t *testing.T) {
	h, keyReg, cleanup := newKeysTestAPI(t)
	defer cleanup()
	mustAddKey(t, keyReg, vector.APIKey{Token: "k_admin", Tenant: "t", Scopes: []string{"*:*"}})

	if code, _ := keysReq(t, h, "k_admin", "POST", "/v1/admin/keys", `{"token":"","tenant":"t"}`); code != http.StatusBadRequest {
		t.Errorf("add empty token = %d, want 400", code)
	}
	if code, _ := keysReq(t, h, "k_admin", "POST", "/v1/admin/keys", `{"token":"x","tenant":""}`); code != http.StatusBadRequest {
		t.Errorf("add empty tenant = %d, want 400", code)
	}
	if code, _ := keysReq(t, h, "k_admin", "DELETE", "/v1/admin/keys", `{"token":""}`); code != http.StatusBadRequest {
		t.Errorf("revoke empty token = %d, want 400", code)
	}
	// Revoke of an unknown token → 404.
	if code, _ := keysReq(t, h, "k_admin", "DELETE", "/v1/admin/keys", `{"token":"nope"}`); code != http.StatusNotFound {
		t.Errorf("revoke unknown token = %d, want 404", code)
	}
}
