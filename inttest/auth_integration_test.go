// SPDX-License-Identifier: Apache-2.0

package inttest

import (
	"context"
	"errors"
	"path/filepath"
	"testing"

	"github.com/rostamlabs/rostam"
	"github.com/rostamlabs/rostam/authz"
	"github.com/rostamlabs/rostam/client"
	"github.com/rostamlabs/rostam/ops"
	"github.com/rostamlabs/rostam/server"
	"github.com/rostamlabs/rostam/vector"
)

// authFromRegistry returns a server.Authenticator (now authz.Authenticator) that
// validates the bearer token against a vector.KeyRegistry by the coarse
// Permissions tier. It is the legacy/regression helper: read ops require
// PermRead; write/create ops require PermWrite; admin/unknown ops require
// PermAdmin. Adapted to the unified authz.AuthRequest signature (it reads
// req.Token + req.Op; per-collection scope enforcement is exercised by the
// dedicated RBAC tests, not this coarse helper).
func authFromRegistry(reg *vector.KeyRegistry) server.Authenticator {
	return func(req authz.AuthRequest) bool {
		if req.Token == "" {
			return false
		}
		k, ok := reg.Lookup(req.Token)
		if !ok {
			return false
		}
		switch req.Op {
		case "vector_search", "get", "__ping__", "__topology__":
			return k.Has(vector.PermRead)
		case "vector_insert", "vector_delete", "vector_create_collection",
			"vector_drop_collection", "put", "del", "expire", "incr":
			return k.Has(vector.PermWrite)
		default:
			// Unknown ops require admin to be safe by default.
			return k.Has(vector.PermAdmin)
		}
	}
}

func TestAuthIntegrationServerRejectsMissingToken(t *testing.T) {
	dir := t.TempDir()
	keyReg, err := vector.OpenKeyRegistry(filepath.Join(dir, "auth", "keys.json"))
	if err != nil {
		t.Fatal(err)
	}
	_ = keyReg.AddKey(vector.APIKey{
		Token:       "k_valid",
		Tenant:      "acme",
		Permissions: []string{vector.PermRead, vector.PermWrite},
	})

	reg := ops.NewRegistry()
	if err := ops.RegisterBuiltins(reg); err != nil {
		t.Fatal(err)
	}
	srv, err := rostam.NewDirectServer("127.0.0.1:0", rostam.DirectConfig{
		DataDir:       dir,
		Ops:           reg,
		Cache:         rostam.CacheConfig{NumShardsPerNode: 4},
		Authenticator: authFromRegistry(keyReg),
	})
	if err != nil {
		t.Fatal(err)
	}
	defer srv.Close()

	// Client without an AuthToken — server rejects.
	noAuthClient, err := rostam.NewClient(rostam.ClientConfig{Servers: []string{srv.Addr()}})
	if err != nil {
		t.Fatal(err)
	}
	defer noAuthClient.Close()

	err = noAuthClient.Put(context.Background(), []byte("k"), []byte("v"), 0)
	if !errors.Is(err, client.ErrUnauthorized) {
		t.Errorf("Put without token: err = %v, want client.ErrUnauthorized", err)
	}
}

func TestAuthIntegrationServerAcceptsValidToken(t *testing.T) {
	dir := t.TempDir()
	keyReg, err := vector.OpenKeyRegistry(filepath.Join(dir, "auth", "keys.json"))
	if err != nil {
		t.Fatal(err)
	}
	_ = keyReg.AddKey(vector.APIKey{
		Token:       "k_valid",
		Tenant:      "acme",
		Permissions: []string{vector.PermRead, vector.PermWrite},
	})

	reg := ops.NewRegistry()
	if err := ops.RegisterBuiltins(reg); err != nil {
		t.Fatal(err)
	}
	srv, err := rostam.NewDirectServer("127.0.0.1:0", rostam.DirectConfig{
		DataDir:       dir,
		Ops:           reg,
		Cache:         rostam.CacheConfig{NumShardsPerNode: 4},
		Authenticator: authFromRegistry(keyReg),
	})
	if err != nil {
		t.Fatal(err)
	}
	defer srv.Close()

	authedClient, err := rostam.NewClient(rostam.ClientConfig{
		Servers:   []string{srv.Addr()},
		AuthToken: "k_valid",
	})
	if err != nil {
		t.Fatal(err)
	}
	defer authedClient.Close()

	ctx := context.Background()
	if err := authedClient.Put(ctx, []byte("greeting"), []byte("hello"), 0); err != nil {
		t.Fatalf("Put with valid token: %v", err)
	}
	got, err := authedClient.Get(ctx, []byte("greeting"))
	if err != nil {
		t.Fatalf("Get with valid token: %v", err)
	}
	if string(got) != "hello" {
		t.Errorf("Get returned %q, want hello", got)
	}
}

func TestAuthIntegrationInvalidTokenRejected(t *testing.T) {
	dir := t.TempDir()
	keyReg, err := vector.OpenKeyRegistry(filepath.Join(dir, "auth", "keys.json"))
	if err != nil {
		t.Fatal(err)
	}
	_ = keyReg.AddKey(vector.APIKey{
		Token:       "k_real",
		Tenant:      "acme",
		Permissions: []string{vector.PermRead, vector.PermWrite},
	})

	reg := ops.NewRegistry()
	if err := ops.RegisterBuiltins(reg); err != nil {
		t.Fatal(err)
	}
	srv, err := rostam.NewDirectServer("127.0.0.1:0", rostam.DirectConfig{
		DataDir:       dir,
		Ops:           reg,
		Cache:         rostam.CacheConfig{NumShardsPerNode: 4},
		Authenticator: authFromRegistry(keyReg),
	})
	if err != nil {
		t.Fatal(err)
	}
	defer srv.Close()

	badClient, err := rostam.NewClient(rostam.ClientConfig{
		Servers:   []string{srv.Addr()},
		AuthToken: "k_bogus",
	})
	if err != nil {
		t.Fatal(err)
	}
	defer badClient.Close()

	err = badClient.Put(context.Background(), []byte("k"), []byte("v"), 0)
	if !errors.Is(err, client.ErrUnauthorized) {
		t.Errorf("Put with bogus token: err = %v, want client.ErrUnauthorized", err)
	}
}

func TestAuthIntegrationLegacyClientStillWorksWhenNoAuthRequired(t *testing.T) {
	// Authenticator nil = no-auth mode; v1 frame from a no-token client works.
	dir := t.TempDir()
	reg := ops.NewRegistry()
	if err := ops.RegisterBuiltins(reg); err != nil {
		t.Fatal(err)
	}
	srv, err := rostam.NewDirectServer("127.0.0.1:0", rostam.DirectConfig{
		DataDir: dir,
		Ops:     reg,
		Cache:   rostam.CacheConfig{NumShardsPerNode: 4},
	})
	if err != nil {
		t.Fatal(err)
	}
	defer srv.Close()

	legacy, err := rostam.NewClient(rostam.ClientConfig{Servers: []string{srv.Addr()}})
	if err != nil {
		t.Fatal(err)
	}
	defer legacy.Close()

	ctx := context.Background()
	if err := legacy.Put(ctx, []byte("hi"), []byte("there"), 0); err != nil {
		t.Fatalf("legacy Put: %v", err)
	}
	got, err := legacy.Get(ctx, []byte("hi"))
	if err != nil {
		t.Fatalf("legacy Get: %v", err)
	}
	if string(got) != "there" {
		t.Errorf("Get returned %q, want there", got)
	}
}
