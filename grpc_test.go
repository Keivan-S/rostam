// SPDX-License-Identifier: Apache-2.0

package rostam

import (
	"context"
	"path/filepath"
	"testing"
	"time"

	"google.golang.org/grpc"
	"google.golang.org/grpc/codes"
	"google.golang.org/grpc/credentials/insecure"
	"google.golang.org/grpc/metadata"
	"google.golang.org/grpc/status"

	"github.com/rostamlabs/rostam/authz"
	"github.com/rostamlabs/rostam/grpcapi/grpcsvc"
	"github.com/rostamlabs/rostam/grpcapi/pb"
	"github.com/rostamlabs/rostam/ops"
	"github.com/rostamlabs/rostam/vector"
)

func dialGRPC(t *testing.T, addr string) (grpcsvc.VectorServiceClient, func()) {
	t.Helper()
	conn, err := grpc.NewClient(addr, grpc.WithTransportCredentials(insecure.NewCredentials()))
	if err != nil {
		t.Fatalf("dial: %v", err)
	}
	return grpcsvc.NewVectorServiceClient(conn), func() { _ = conn.Close() }
}

// TestGRPCServerEndToEnd drives the RAG surface over a real gRPC connection:
// create → upsert chunks across documents → group-by-document search.
func TestGRPCServerEndToEnd(t *testing.T) {
	reg := ops.NewRegistry()
	if err := ops.RegisterBuiltins(reg); err != nil {
		t.Fatal(err)
	}
	srv, err := NewGRPCServer("127.0.0.1:0", DirectConfig{
		DataDir: t.TempDir(),
		Ops:     reg,
		Cache:   CacheConfig{NumShardsPerNode: 4},
	})
	if err != nil {
		t.Fatal(err)
	}
	defer srv.Close()

	cli, closeConn := dialGRPC(t, srv.Addr())
	defer closeConn()
	ctx, cancel := context.WithTimeout(context.Background(), 10*time.Second)
	defer cancel()

	if _, err := cli.Health(ctx, &pb.HealthRequest{}); err != nil {
		t.Fatalf("health: %v", err)
	}

	if _, err := cli.CreateCollection(ctx, &pb.CreateCollectionRequest{
		Name:   "docs",
		Config: &pb.Config{Dim: 3, Metric: "l2"},
	}); err != nil {
		t.Fatalf("create: %v", err)
	}

	for i := 1; i <= 6; i++ {
		doc := (i + 1) / 2
		if _, err := cli.Upsert(ctx, &pb.UpsertRequest{
			Collection:   "docs",
			Id:           uint64(i),
			Vector:       []float32{float32(i), 0, 0},
			Content:      "chunk",
			Upsert:       true,
			MetadataJson: `{"doc":{"kind":"int","int":` + itoa(doc) + `}}`,
		}); err != nil {
			t.Fatalf("upsert %d: %v", i, err)
		}
	}

	resp, err := cli.SearchGroups(ctx, &pb.SearchGroupsRequest{
		Collection: "docs", Query: []float32{0, 0, 0}, K: 2, GroupBy: "doc", GroupSize: 1,
	})
	if err != nil {
		t.Fatalf("search_groups: %v", err)
	}
	if len(resp.GetGroups()) != 2 {
		t.Fatalf("got %d groups, want 2", len(resp.GetGroups()))
	}
	g0 := resp.GetGroups()[0]
	if g0.GetKeyJson() != `{"kind":"int","int":1}` {
		t.Errorf("group0 key = %s, want doc 1", g0.GetKeyJson())
	}
	if len(g0.GetHits()) != 1 || g0.GetHits()[0].GetId() != 1 || g0.GetHits()[0].GetContent() != "chunk" {
		t.Errorf("group0 hits = %+v", g0.GetHits())
	}

	// search returns the nearest ids ascending by distance.
	sr, err := cli.Search(ctx, &pb.SearchRequest{Collection: "docs", Query: []float32{0, 0, 0}, K: 3})
	if err != nil {
		t.Fatalf("search: %v", err)
	}
	if len(sr.GetResults()) != 3 || sr.GetResults()[0].GetId() != 1 {
		t.Errorf("search = %+v", sr.GetResults())
	}

	// Unknown collection maps to a NotFound status.
	if _, err := cli.Search(ctx, &pb.SearchRequest{Collection: "nope", Query: []float32{0, 0, 0}, K: 1}); err == nil {
		t.Error("expected error for unknown collection")
	}
}

func TestGRPCAuth(t *testing.T) {
	reg := ops.NewRegistry()
	if err := ops.RegisterBuiltins(reg); err != nil {
		t.Fatal(err)
	}
	srv, err := NewGRPCServer("127.0.0.1:0", DirectConfig{
		DataDir:       t.TempDir(),
		Ops:           reg,
		Cache:         CacheConfig{NumShardsPerNode: 4},
		Authenticator: func(req authz.AuthRequest) bool { return req.Token == "secret" },
	})
	if err != nil {
		t.Fatal(err)
	}
	defer srv.Close()

	cli, closeConn := dialGRPC(t, srv.Addr())
	defer closeConn()
	ctx, cancel := context.WithTimeout(context.Background(), 10*time.Second)
	defer cancel()

	// No token → rejected.
	if _, err := cli.CreateCollection(ctx, &pb.CreateCollectionRequest{Name: "d", Config: &pb.Config{Dim: 3}}); err == nil {
		t.Error("expected unauthenticated error without token")
	}

	// Correct token via metadata → allowed.
	authCtx := metadata.AppendToOutgoingContext(ctx, "authorization", "Bearer secret")
	if _, err := cli.CreateCollection(authCtx, &pb.CreateCollectionRequest{Name: "d", Config: &pb.Config{Dim: 3}}); err != nil {
		t.Errorf("create with token: %v", err)
	}

	// Health bypasses auth.
	if _, err := cli.Health(ctx, &pb.HealthRequest{}); err != nil {
		t.Errorf("health w/ auth: %v", err)
	}
}

// TestGRPCRBACScopes exercises the granular per-collection RBAC matrix over real
// gRPC: a read:default/docs key can Search docs but NOT Upsert docs and NOT
// Search a different collection; an admin:* key can create/drop; no token is
// denied. A denied op returns codes.Unauthenticated and never reaches the engine.
func TestGRPCRBACScopes(t *testing.T) {
	reg := ops.NewRegistry()
	if err := ops.RegisterBuiltins(reg); err != nil {
		t.Fatal(err)
	}
	keyReg, err := vector.OpenKeyRegistry(filepath.Join(t.TempDir(), "keys.json"))
	if err != nil {
		t.Fatal(err)
	}
	for _, k := range []vector.APIKey{
		{Token: "k_read", Tenant: "t", Scopes: []string{"read:default/docs"}},
		{Token: "k_write", Tenant: "t", Scopes: []string{"read:default/docs", "write:default/docs"}},
		{Token: "k_admin", Tenant: "t", Scopes: []string{"*:*"}},
	} {
		if err := keyReg.AddKey(k); err != nil {
			t.Fatalf("AddKey(%q): %v", k.Token, err)
		}
	}
	srv, err := NewGRPCServer("127.0.0.1:0", DirectConfig{
		DataDir:       t.TempDir(),
		Ops:           reg,
		Cache:         CacheConfig{NumShardsPerNode: 4},
		Authenticator: authz.NewRBACAuthenticator(keyReg, reg, ""),
	})
	if err != nil {
		t.Fatal(err)
	}
	defer srv.Close()

	cli, closeConn := dialGRPC(t, srv.Addr())
	defer closeConn()
	base, cancel := context.WithTimeout(context.Background(), 10*time.Second)
	defer cancel()
	ctxFor := func(tok string) context.Context {
		if tok == "" {
			return base
		}
		return metadata.AppendToOutgoingContext(base, "authorization", "Bearer "+tok)
	}
	codeOf := func(err error) codes.Code { return status.Code(err) }

	// admin creates docs; a read key cannot (create is admin).
	if _, err := cli.CreateCollection(ctxFor("k_read"), &pb.CreateCollectionRequest{Name: "docs", Config: &pb.Config{Dim: 3}}); codeOf(err) != codes.Unauthenticated {
		t.Errorf("read create err = %v, want Unauthenticated", err)
	}
	if _, err := cli.CreateCollection(ctxFor("k_admin"), &pb.CreateCollectionRequest{Name: "docs", Config: &pb.Config{Dim: 3}}); err != nil {
		t.Fatalf("admin create: %v", err)
	}

	// reader CAN search docs.
	if _, err := cli.Search(ctxFor("k_read"), &pb.SearchRequest{Collection: "docs", Query: []float32{1, 2, 3}, K: 1}); err != nil {
		t.Errorf("read search docs: %v", err)
	}
	// reader CANNOT upsert docs.
	if _, err := cli.Upsert(ctxFor("k_read"), &pb.UpsertRequest{Collection: "docs", Id: 1, Vector: []float32{1, 2, 3}}); codeOf(err) != codes.Unauthenticated {
		t.Errorf("read upsert docs err = %v, want Unauthenticated", err)
	}
	// reader CANNOT search a different collection.
	if _, err := cli.Search(ctxFor("k_read"), &pb.SearchRequest{Collection: "other", Query: []float32{1, 2, 3}, K: 1}); codeOf(err) != codes.Unauthenticated {
		t.Errorf("read search other err = %v, want Unauthenticated", err)
	}
	// writer CAN upsert docs.
	if _, err := cli.Upsert(ctxFor("k_write"), &pb.UpsertRequest{Collection: "docs", Id: 1, Vector: []float32{1, 2, 3}}); err != nil {
		t.Errorf("write upsert docs: %v", err)
	}
	// no token → Unauthenticated.
	if _, err := cli.Search(ctxFor(""), &pb.SearchRequest{Collection: "docs", Query: []float32{1, 2, 3}, K: 1}); codeOf(err) != codes.Unauthenticated {
		t.Errorf("no-token search err = %v, want Unauthenticated", err)
	}
	// admin can drop.
	if _, err := cli.DropCollection(ctxFor("k_admin"), &pb.DropCollectionRequest{Name: "docs"}); err != nil {
		t.Errorf("admin drop: %v", err)
	}
}
