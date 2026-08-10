// SPDX-License-Identifier: Apache-2.0

package inttest

import (
	"context"
	"testing"

	"github.com/rostamlabs/rostam"
	"github.com/rostamlabs/rostam/ops"
	"github.com/rostamlabs/rostam/vector"
)

func TestVectorHybridSearchEndToEndDirect(t *testing.T) {
	reg := ops.NewRegistry()
	if err := ops.RegisterBuiltins(reg); err != nil {
		t.Fatal(err)
	}
	srv, err := rostam.NewDirectServer("127.0.0.1:0", rostam.DirectConfig{
		DataDir: t.TempDir(),
		Ops:     reg,
		Cache:   rostam.CacheConfig{NumShardsPerNode: 4},
	})
	if err != nil {
		t.Fatal(err)
	}
	defer srv.Close()

	store, err := rostam.NewClient(rostam.ClientConfig{Servers: []string{srv.Addr()}})
	if err != nil {
		t.Fatal(err)
	}
	defer store.Close()

	ctx := context.Background()
	cfg := rostam.VectorConfig{Dim: 4, M: 8, EfConstruction: 50, EfSearch: 32, Seed: 1, Metric: vector.L2}
	if err := store.CreateCollection(ctx, "docs", cfg); err != nil {
		t.Fatal(err)
	}

	// Docs 1-5 cluster near the dense query; each carries a weak shared sparse
	// term. Doc 100 is far in dense space but owns the exact sparse term 42.
	for i := uint64(1); i <= 5; i++ {
		opts := rostam.VectorInsertOpts{Sparse: rostam.VectorSparse{Indices: []uint32{1}, Values: []float32{0.1}}}
		if err := store.VectorInsertExt(ctx, "docs", i, []float32{float32(i) * 0.01, 0, 0, 0}, opts); err != nil {
			t.Fatalf("insert %d: %v", i, err)
		}
	}
	if err := store.VectorInsertExt(ctx, "docs", 100, []float32{9, 9, 9, 9},
		rostam.VectorInsertOpts{Sparse: rostam.VectorSparse{Indices: []uint32{42}, Values: []float32{10}}}); err != nil {
		t.Fatalf("insert 100: %v", err)
	}

	dense := []float32{0, 0, 0, 0}

	// Pure dense (VectorSearch) must NOT surface doc 100 (farthest vector).
	pure, err := store.VectorSearch(ctx, "docs", dense, 3)
	if err != nil {
		t.Fatal(err)
	}
	for _, r := range pure {
		if r.ID == 100 {
			t.Fatal("doc 100 must not appear in pure dense top-3")
		}
	}

	// Hybrid with the exact sparse term should pull doc 100 in.
	got, _, err := store.VectorHybridSearch(ctx, "docs", dense, 3, rostam.VectorHybridOpts{
		Sparse: rostam.VectorSparse{Indices: []uint32{42}, Values: []float32{5}},
	})
	if err != nil {
		t.Fatal(err)
	}
	found := false
	for _, r := range got {
		if r.ID == 100 {
			found = true
		}
		if r.Score == 0 {
			t.Errorf("hybrid result id %d has zero fusion score", r.ID)
		}
	}
	if !found {
		t.Errorf("hybrid search should surface doc 100 via sparse term 42; got %+v", got)
	}
}

func TestVectorHybridSearchFilterEndToEnd(t *testing.T) {
	reg := ops.NewRegistry()
	if err := ops.RegisterBuiltins(reg); err != nil {
		t.Fatal(err)
	}
	srv, err := rostam.NewDirectServer("127.0.0.1:0", rostam.DirectConfig{
		DataDir: t.TempDir(),
		Ops:     reg,
		Cache:   rostam.CacheConfig{NumShardsPerNode: 4},
	})
	if err != nil {
		t.Fatal(err)
	}
	defer srv.Close()

	store, err := rostam.NewClient(rostam.ClientConfig{Servers: []string{srv.Addr()}})
	if err != nil {
		t.Fatal(err)
	}
	defer store.Close()

	ctx := context.Background()
	cfg := rostam.VectorConfig{Dim: 4, M: 8, EfConstruction: 50, EfSearch: 32, Seed: 1, Metric: vector.L2}
	if err := store.CreateCollection(ctx, "docs", cfg); err != nil {
		t.Fatal(err)
	}

	for i := uint64(1); i <= 10; i++ {
		tenant := "acme"
		if i%2 == 0 {
			tenant = "globex"
		}
		opts := rostam.VectorInsertOpts{
			Metadata: rostam.VectorMetadata{"tenant": vector.NewString(tenant)},
			Sparse:   rostam.VectorSparse{Indices: []uint32{7}, Values: []float32{1.0}},
		}
		if err := store.VectorInsertExt(ctx, "docs", i, []float32{float32(i) * 0.01, 0, 0, 0}, opts); err != nil {
			t.Fatalf("insert %d: %v", i, err)
		}
	}

	got, _, err := store.VectorHybridSearch(ctx, "docs", []float32{0, 0, 0, 0}, 10, rostam.VectorHybridOpts{
		Sparse: rostam.VectorSparse{Indices: []uint32{7}, Values: []float32{1.0}},
		Filter: rostam.VectorFilter{Op: vector.FilterEq, Field: "tenant", Value: vector.NewString("acme")},
	})
	if err != nil {
		t.Fatal(err)
	}
	if len(got) == 0 {
		t.Fatal("filtered hybrid returned nothing")
	}
	for _, r := range got {
		if r.ID%2 == 0 {
			t.Errorf("result id %d is globex (even); filter should exclude it", r.ID)
		}
	}
}
