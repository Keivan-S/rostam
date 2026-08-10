// SPDX-License-Identifier: Apache-2.0

package main

import (
	"crypto/tls"
	"path/filepath"
	"strings"
	"testing"

	"github.com/rostamlabs/rostam/authz"
	"github.com/rostamlabs/rostam/ops"
	"github.com/rostamlabs/rostam/tlsutil/testcerts"
	"github.com/rostamlabs/rostam/vector"
)

// --- TLS-flag wiring (buildServerTLS) ---------------------------------------

// TestBuildServerTLSValid: cert+key (no CA) builds a server-auth-only config.
func TestBuildServerTLSValid(t *testing.T) {
	ca := testcerts.GenCA(t)
	cert, key := ca.ServerCert(t, "localhost")

	cfg, err := buildServerTLS(cert, key, "", false)
	if err != nil {
		t.Fatalf("buildServerTLS: %v", err)
	}
	if cfg == nil {
		t.Fatal("nil config")
	}
	if cfg.MinVersion != tls.VersionTLS12 {
		t.Errorf("MinVersion = %x, want TLS1.2", cfg.MinVersion)
	}
	if len(cfg.Certificates) != 1 {
		t.Errorf("Certificates len = %d, want 1", len(cfg.Certificates))
	}
	if cfg.ClientAuth != tls.NoClientCert {
		t.Errorf("ClientAuth = %v, want NoClientCert (no CA)", cfg.ClientAuth)
	}
}

// TestBuildServerTLSRequireClientCert: require-client-cert + CA ⇒ strict mTLS.
func TestBuildServerTLSRequireClientCert(t *testing.T) {
	ca := testcerts.GenCA(t)
	cert, key := ca.ServerCert(t, "localhost")

	cfg, err := buildServerTLS(cert, key, ca.CAFile, true)
	if err != nil {
		t.Fatalf("buildServerTLS(mTLS): %v", err)
	}
	if cfg.ClientAuth != tls.RequireAndVerifyClientCert {
		t.Errorf("ClientAuth = %v, want RequireAndVerifyClientCert", cfg.ClientAuth)
	}
	if cfg.ClientCAs == nil {
		t.Error("ClientCAs must be set when CA given")
	}
}

// TestBuildServerTLSFailClosed: the fail-closed misconfigurations all return an
// error (and a nil config) WITHOUT calling log.Fatalf — we test the
// error-returning helper layer, not main()'s fatal.
func TestBuildServerTLSFailClosed(t *testing.T) {
	ca := testcerts.GenCA(t)
	cert, key := ca.ServerCert(t, "localhost")

	t.Run("cert without key", func(t *testing.T) {
		cfg, err := buildServerTLS(cert, "", "", false)
		if err == nil {
			t.Fatal("cert-without-key must be an error")
		}
		if cfg != nil {
			t.Error("config must be nil on error (no plaintext fallback)")
		}
	})

	t.Run("key without cert", func(t *testing.T) {
		if _, err := buildServerTLS("", key, "", false); err == nil {
			t.Fatal("key-without-cert must be an error")
		}
	})

	t.Run("require client cert without ca", func(t *testing.T) {
		cfg, err := buildServerTLS(cert, key, "", true)
		if err == nil {
			t.Fatal("require-client-cert without CA must be an error")
		}
		if cfg != nil {
			t.Error("config must be nil on error")
		}
	})

	t.Run("missing cert file", func(t *testing.T) {
		if _, err := buildServerTLS(filepath.Join(t.TempDir(), "nope.pem"), key, "", false); err == nil {
			t.Fatal("missing cert file must be an error")
		}
	})

	t.Run("missing ca file", func(t *testing.T) {
		if _, err := buildServerTLS(cert, key, filepath.Join(t.TempDir(), "nope-ca.pem"), true); err == nil {
			t.Fatal("missing CA file must be an error")
		}
	})
}

// --- key-admin CLI ----------------------------------------------------------

// authzReg builds an ops registry with the builtins, for the RBAC round-trip.
func authzReg(t *testing.T) *ops.Registry {
	t.Helper()
	reg := ops.NewRegistry()
	if err := ops.RegisterBuiltins(reg); err != nil {
		t.Fatalf("register builtins: %v", err)
	}
	return reg
}

// TestKeysAddRoundTripAndAuthorize: add a scoped key to a fresh file, reopen,
// confirm scopes+cert-cn round-trip AND that NewRBACAuthenticator authorizes
// correctly (read docs ✓, write docs ✗, read other ✗).
func TestKeysAddRoundTripAndAuthorize(t *testing.T) {
	file := filepath.Join(t.TempDir(), "keys.json")

	if err := keysAdd([]string{
		"-file", file,
		"-token", "tok-reader",
		"-tenant", "acme",
		"-scopes", "read:default/docs",
		"-cert-cn", "reader-cn",
	}); err != nil {
		t.Fatalf("keysAdd: %v", err)
	}

	// Reopen from disk — proves the atomic flush persisted the entry.
	reg, err := vector.OpenKeyRegistry(file)
	if err != nil {
		t.Fatalf("reopen: %v", err)
	}
	k, ok := reg.Lookup("tok-reader")
	if !ok {
		t.Fatal("key not persisted")
	}
	if k.Tenant != "acme" || k.CertCN != "reader-cn" {
		t.Errorf("round-trip mismatch: %+v", k)
	}
	if len(k.Scopes) != 1 || k.Scopes[0] != "read:default/docs" {
		t.Errorf("scopes round-trip = %v", k.Scopes)
	}

	auth := authz.NewRBACAuthenticator(reg, authzReg(t), "")
	if !auth(authz.AuthRequest{Token: "tok-reader", Op: "vector_search", Args: ops.EncodeVectorSearchArgs("docs", 3, []float32{1, 2})}) {
		t.Error("read:default/docs should allow vector_search docs")
	}
	if auth(authz.AuthRequest{Token: "tok-reader", Op: "vector_insert", Args: ops.EncodeVectorInsertArgs("docs", 1, []float32{1, 2})}) {
		t.Error("read:default/docs must NOT allow vector_insert docs")
	}
	if auth(authz.AuthRequest{Token: "tok-reader", Op: "vector_search", Args: ops.EncodeVectorSearchArgs("other", 3, []float32{1, 2})}) {
		t.Error("read:default/docs must NOT allow searching other")
	}
}

// TestKeysRevoke: revoke removes the key from the file.
func TestKeysRevoke(t *testing.T) {
	file := filepath.Join(t.TempDir(), "keys.json")
	if err := keysAdd([]string{"-file", file, "-token", "t1", "-tenant", "acme", "-scopes", "read:*"}); err != nil {
		t.Fatalf("add: %v", err)
	}
	if err := keysRevoke([]string{"-file", file, "-token", "t1"}); err != nil {
		t.Fatalf("revoke: %v", err)
	}
	reg, err := vector.OpenKeyRegistry(file)
	if err != nil {
		t.Fatalf("reopen: %v", err)
	}
	if _, ok := reg.Lookup("t1"); ok {
		t.Error("key still present after revoke")
	}
	// Revoking an absent token surfaces an error.
	if err := keysRevoke([]string{"-file", file, "-token", "t1"}); err == nil {
		t.Error("revoking absent token should error")
	}
}

// TestKeysListMasksTokens: list output masks the token (prefix + stars), never
// echoing the full secret.
func TestKeysListMasksTokens(t *testing.T) {
	if got := maskToken("supersecrettoken"); strings.Contains(got, "secret") || !strings.HasPrefix(got, "supe") {
		t.Errorf("maskToken leaked or wrong: %q", got)
	}
	if got := maskToken("abcd"); got != "****" {
		t.Errorf("short token not fully masked: %q", got)
	}
	if got := maskToken(""); got != "(empty)" {
		t.Errorf("empty token = %q", got)
	}
	// keysList itself must run without error on a populated file.
	file := filepath.Join(t.TempDir(), "keys.json")
	if err := keysAdd([]string{"-file", file, "-token", "tok-12345678", "-tenant", "acme", "-scopes", "read:*"}); err != nil {
		t.Fatalf("add: %v", err)
	}
	if err := keysList([]string{"-file", file}); err != nil {
		t.Fatalf("list: %v", err)
	}
}

// TestKeysAddRejectsBadScope: a scope without ':' (and other malformed forms) is
// REJECTED fail-closed and nothing is written.
func TestKeysAddRejectsBadScope(t *testing.T) {
	file := filepath.Join(t.TempDir(), "keys.json")
	for _, bad := range []string{"readdocs", "read:", "bogus:docs", ":docs"} {
		err := keysAdd([]string{"-file", file, "-token", "t", "-tenant", "acme", "-scopes", bad})
		if err == nil {
			t.Errorf("scope %q should be rejected", bad)
		}
	}
	// Nothing persisted for that token.
	reg, err := vector.OpenKeyRegistry(file)
	if err != nil {
		t.Fatalf("reopen: %v", err)
	}
	if _, ok := reg.Lookup("t"); ok {
		t.Error("a key with a bad scope must not be persisted")
	}
}

// TestKeysAddRejectsEmptyTenant: AddKey enforces non-empty token AND tenant; the
// CLI surfaces that as an error.
func TestKeysAddRejectsEmptyTenant(t *testing.T) {
	file := filepath.Join(t.TempDir(), "keys.json")
	if err := keysAdd([]string{"-file", file, "-token", "t", "-tenant", "", "-scopes", "read:*"}); err == nil {
		t.Error("empty tenant must be rejected")
	}
	if err := keysAdd([]string{"-file", file, "-token", "", "-tenant", "acme", "-scopes", "read:*"}); err == nil {
		t.Error("empty token must be rejected")
	}
}

// TestKeysAddCertCNAuthorizes: a cert-bound key authorizes a cert-only client
// (no token, verified ClientCN) via LookupByCN under the RBAC engine.
func TestKeysAddCertCNAuthorizes(t *testing.T) {
	file := filepath.Join(t.TempDir(), "keys.json")
	if err := keysAdd([]string{"-file", file, "-token", "cn-key", "-tenant", "acme", "-scopes", "read:default/docs", "-cert-cn", "client-cn"}); err != nil {
		t.Fatalf("add: %v", err)
	}
	reg, err := vector.OpenKeyRegistry(file)
	if err != nil {
		t.Fatalf("reopen: %v", err)
	}
	auth := authz.NewRBACAuthenticator(reg, authzReg(t), "")
	if !auth(authz.AuthRequest{ClientCN: "client-cn", Op: "vector_search", Args: ops.EncodeVectorSearchArgs("docs", 3, []float32{1, 2})}) {
		t.Error("cert-CN client should authorize via its CN scopes")
	}
	if auth(authz.AuthRequest{ClientCN: "client-cn", Op: "vector_insert", Args: ops.EncodeVectorInsertArgs("docs", 1, []float32{1, 2})}) {
		t.Error("cert-CN read scope must NOT allow insert")
	}
}
