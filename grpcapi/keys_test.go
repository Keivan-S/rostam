// SPDX-License-Identifier: Apache-2.0

package grpcapi

import (
	"context"
	"path/filepath"
	"strings"
	"testing"

	"google.golang.org/grpc/codes"
	"google.golang.org/grpc/metadata"
	"google.golang.org/grpc/status"

	"github.com/rostamlabs/rostam/authz"
	"github.com/rostamlabs/rostam/ops"
	"github.com/rostamlabs/rostam/sdk/pb"
	"github.com/rostamlabs/rostam/vector"
)

// keysStubDispatcher mirrors the production keysDispatcher (in the root rostam
// package, un-importable here without a cycle): it intercepts the three keys
// coordinator virtual-ops and applies them to a live *vector.KeyRegistry. Every
// other op is a no-op ack (the keys RPCs never dispatch a non-keys op). This lets
// the gRPC keys handlers round-trip against a real registry over the real handler
// path without the import cycle.
type keysStubDispatcher struct {
	reg *vector.KeyRegistry
}

func (d *keysStubDispatcher) Call(name string, args []byte) ([]byte, error) {
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
		return nil, nil
	}
}

func (d *keysStubDispatcher) LeaderAddr() string { return "" }

// newKeysGRPCServer builds a gRPC Server whose dispatcher routes the keys ops to a
// live registry and whose authorizer is RBAC-backed by that SAME registry.
func newKeysGRPCServer(t *testing.T) (*Server, *vector.KeyRegistry) {
	t.Helper()
	opsReg := ops.NewRegistry()
	if err := ops.RegisterBuiltins(opsReg); err != nil {
		t.Fatal(err)
	}
	keyReg, err := vector.OpenKeyRegistry(filepath.Join(t.TempDir(), "keys.json"))
	if err != nil {
		t.Fatal(err)
	}
	disp := &keysStubDispatcher{reg: keyReg}
	s := NewServer(disp, authz.NewRBACAuthenticator(keyReg, opsReg, ""))
	return s, keyReg
}

// authCtx returns a context carrying the bearer token in gRPC metadata (the
// server reads md["authorization"]).
func authCtx(token string) context.Context {
	return metadata.NewIncomingContext(context.Background(), metadata.Pairs("authorization", "Bearer "+token))
}

// TestGRPCKeysAddAuthenticatesThenRevokeDenies is the headline round-trip teeth
// test over the real gRPC handlers: KeysAdd makes a new token authenticate
// against the LIVE registry (no restart); KeysRevoke then makes it fail auth.
func TestGRPCKeysAddAuthenticatesThenRevokeDenies(t *testing.T) {
	s, keyReg := newKeysGRPCServer(t)
	mustAddKey := func(k vector.APIKey) {
		if err := keyReg.AddKey(k); err != nil {
			t.Fatalf("AddKey(%q): %v", k.Token, err)
		}
	}
	mustAddKey(vector.APIKey{Token: "k_admin", Tenant: "t", Scopes: []string{"*:*"}})

	const tok = "LIVE-GRPC-TOKEN"
	// Before add: the new token cannot list (not admin).
	if _, err := s.KeysList(authCtx(tok), &pb.KeysListRequest{}); status.Code(err) != codes.Unauthenticated {
		t.Fatalf("pre-add list code = %v, want Unauthenticated", status.Code(err))
	}
	// Admin adds the new key with admin scope.
	if _, err := s.KeysAdd(authCtx("k_admin"), &pb.KeysAddRequest{Token: tok, Tenant: "acme", Scopes: []string{"*:*"}}); err != nil {
		t.Fatalf("KeysAdd: %v", err)
	}
	// After add: the new token authenticates immediately.
	if _, err := s.KeysList(authCtx(tok), &pb.KeysListRequest{}); err != nil {
		t.Fatalf("post-add list: %v", err)
	}
	// Revoke the new token.
	if _, err := s.KeysRevoke(authCtx("k_admin"), &pb.KeysRevokeRequest{Token: tok}); err != nil {
		t.Fatalf("KeysRevoke: %v", err)
	}
	// After revoke: the token no longer authenticates.
	if _, err := s.KeysList(authCtx(tok), &pb.KeysListRequest{}); status.Code(err) != codes.Unauthenticated {
		t.Fatalf("post-revoke list code = %v, want Unauthenticated", status.Code(err))
	}
}

// TestGRPCKeysListRedacted is the security teeth test: the KeysList proto response
// carries fingerprint + tenant + scopes + cert_cn but NEVER the raw token. The
// RedactedKey message has no token field, so this checks the serialized bytes too.
func TestGRPCKeysListRedacted(t *testing.T) {
	s, keyReg := newKeysGRPCServer(t)
	if err := keyReg.AddKey(vector.APIKey{Token: "k_admin", Tenant: "t", Scopes: []string{"*:*"}}); err != nil {
		t.Fatal(err)
	}
	const sentinel = "SUPER-SECRET-GRPC-SENTINEL"
	if _, err := s.KeysAdd(authCtx("k_admin"), &pb.KeysAddRequest{Token: sentinel, Tenant: "acme", Scopes: []string{"read:default/docs"}, CertCn: "cn1"}); err != nil {
		t.Fatalf("KeysAdd: %v", err)
	}

	resp, err := s.KeysList(authCtx("k_admin"), &pb.KeysListRequest{})
	if err != nil {
		t.Fatalf("KeysList: %v", err)
	}
	fp := vector.TokenFingerprint(sentinel)
	var found bool
	for _, k := range resp.GetKeys() {
		// TEETH: no field on the redacted message may carry the raw token.
		if strings.Contains(k.GetFingerprint(), sentinel) || strings.Contains(k.GetTenant(), sentinel) ||
			strings.Contains(k.GetCertCn(), sentinel) || strings.Contains(strings.Join(k.GetScopes(), ","), sentinel) {
			t.Fatalf("redacted key leaked the raw token: %v", k)
		}
		if k.GetFingerprint() == fp {
			found = true
			if k.GetTenant() != "acme" {
				t.Errorf("tenant = %q, want acme", k.GetTenant())
			}
		}
	}
	if !found {
		t.Fatalf("sentinel key not found by fingerprint %q in %v", fp, resp.GetKeys())
	}
}

// TestGRPCKeysNonAdminDenied proves a non-admin caller is denied at the authorize
// gate (Unauthenticated) for all three keys RPCs and the registry is untouched.
func TestGRPCKeysNonAdminDenied(t *testing.T) {
	s, keyReg := newKeysGRPCServer(t)
	if err := keyReg.AddKey(vector.APIKey{Token: "k_read", Tenant: "t", Scopes: []string{"read:default/docs"}}); err != nil {
		t.Fatal(err)
	}
	if _, err := s.KeysAdd(authCtx("k_read"), &pb.KeysAddRequest{Token: "x", Tenant: "t", Scopes: []string{"read:default/docs"}}); status.Code(err) != codes.Unauthenticated {
		t.Errorf("non-admin KeysAdd code = %v, want Unauthenticated", status.Code(err))
	}
	if _, err := s.KeysRevoke(authCtx("k_read"), &pb.KeysRevokeRequest{Token: "x"}); status.Code(err) != codes.Unauthenticated {
		t.Errorf("non-admin KeysRevoke code = %v, want Unauthenticated", status.Code(err))
	}
	if _, err := s.KeysList(authCtx("k_read"), &pb.KeysListRequest{}); status.Code(err) != codes.Unauthenticated {
		t.Errorf("non-admin KeysList code = %v, want Unauthenticated", status.Code(err))
	}
	if len(keyReg.ListRedacted()) != 1 {
		t.Fatalf("registry mutated by a denied non-admin add: %d keys", len(keyReg.ListRedacted()))
	}
}

// TestGRPCKeysErrorMapping proves the keys-specific errors map to the right gRPC
// codes: dup token → AlreadyExists, unknown revoke → NotFound, bad input →
// InvalidArgument.
func TestGRPCKeysErrorMapping(t *testing.T) {
	s, keyReg := newKeysGRPCServer(t)
	if err := keyReg.AddKey(vector.APIKey{Token: "k_admin", Tenant: "t", Scopes: []string{"*:*"}}); err != nil {
		t.Fatal(err)
	}
	// Add then re-add the same token → AlreadyExists.
	if _, err := s.KeysAdd(authCtx("k_admin"), &pb.KeysAddRequest{Token: "dup", Tenant: "t", Scopes: []string{"read:default/docs"}}); err != nil {
		t.Fatalf("first add: %v", err)
	}
	if _, err := s.KeysAdd(authCtx("k_admin"), &pb.KeysAddRequest{Token: "dup", Tenant: "t", Scopes: []string{"read:default/docs"}}); status.Code(err) != codes.AlreadyExists {
		t.Errorf("dup add code = %v, want AlreadyExists", status.Code(err))
	}
	// Revoke unknown token → NotFound.
	if _, err := s.KeysRevoke(authCtx("k_admin"), &pb.KeysRevokeRequest{Token: "nope"}); status.Code(err) != codes.NotFound {
		t.Errorf("unknown revoke code = %v, want NotFound", status.Code(err))
	}
	// Bad input (empty token) → InvalidArgument, before dispatch.
	if _, err := s.KeysAdd(authCtx("k_admin"), &pb.KeysAddRequest{Token: "", Tenant: "t"}); status.Code(err) != codes.InvalidArgument {
		t.Errorf("empty-token add code = %v, want InvalidArgument", status.Code(err))
	}
	if _, err := s.KeysRevoke(authCtx("k_admin"), &pb.KeysRevokeRequest{Token: ""}); status.Code(err) != codes.InvalidArgument {
		t.Errorf("empty-token revoke code = %v, want InvalidArgument", status.Code(err))
	}
}
