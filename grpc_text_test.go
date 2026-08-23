// SPDX-License-Identifier: Apache-2.0

package rostam

import (
	"context"
	"testing"
	"time"

	"github.com/rostamlabs/rostam/sdk/pb"
	"github.com/rostamlabs/rostam/ops"
)

// TestGRPCFullTextEndToEnd drives TextSearch + HybridTextSearch over a real gRPC
// connection: create a full-text collection (Config.FullText set) → upsert
// content docs → TextSearch (rare term ranks its doc first) → HybridTextSearch
// (dense + BM25 fused). Also asserts a full-text RPC on a NON-full-text collection
// maps to InvalidArgument (ErrFullTextDisabled).
func TestGRPCFullTextEndToEnd(t *testing.T) {
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

	if _, err := cli.CreateCollection(ctx, &pb.CreateCollectionRequest{
		Name: "ft",
		Config: &pb.Config{
			Dim: 4, Metric: "l2",
			FullText: &pb.FullTextConfig{Analyzer: "english"},
		},
	}); err != nil {
		t.Fatalf("create full-text: %v", err)
	}

	for id, content := range ftDocs {
		if _, err := cli.Upsert(ctx, &pb.UpsertRequest{
			Collection: "ft",
			Id:         id,
			Vector:     denseFor(id),
			Content:    content,
			Upsert:     true,
		}); err != nil {
			t.Fatalf("upsert %d: %v", id, err)
		}
	}

	// TextSearch: the rare term "fox" ranks its (only) doc id 1 first, content rides.
	ts, err := cli.TextSearch(ctx, &pb.TextSearchRequest{Collection: "ft", Text: "fox", K: 5})
	if err != nil {
		t.Fatalf("TextSearch: %v", err)
	}
	if len(ts.GetDocuments()) == 0 || ts.GetDocuments()[0].GetId() != 1 {
		t.Fatalf("TextSearch: docs = %+v, want id 1 first", ts.GetDocuments())
	}
	if ts.GetDocuments()[0].GetContent() != ftDocs[1] {
		t.Fatalf("TextSearch: content = %q, want %q", ts.GetDocuments()[0].GetContent(), ftDocs[1])
	}

	// HybridTextSearch: dense(id1) + "fox" → id 1 first.
	hts, err := cli.HybridTextSearch(ctx, &pb.HybridTextRequest{
		Collection: "ft", Dense: denseFor(1), Text: "fox", K: 5, Method: "rrf",
	})
	if err != nil {
		t.Fatalf("HybridTextSearch: %v", err)
	}
	if len(hts.GetResults()) == 0 || hts.GetResults()[0].GetId() != 1 {
		t.Fatalf("HybridTextSearch: results = %+v, want id 1 first", hts.GetResults())
	}

	// A full-text RPC on a NON-full-text collection is InvalidArgument.
	if _, err := cli.CreateCollection(ctx, &pb.CreateCollectionRequest{
		Name: "plain", Config: &pb.Config{Dim: 4, Metric: "l2"},
	}); err != nil {
		t.Fatalf("create plain: %v", err)
	}
	if _, err := cli.Upsert(ctx, &pb.UpsertRequest{
		Collection: "plain", Id: 1, Vector: denseFor(1), Content: "hello", Upsert: true,
	}); err != nil {
		t.Fatalf("upsert plain: %v", err)
	}
	if _, err := cli.TextSearch(ctx, &pb.TextSearchRequest{Collection: "plain", Text: "hello", K: 5}); err == nil {
		t.Fatal("TextSearch on plain collection: want error, got nil")
	}
}
