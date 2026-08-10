// SPDX-License-Identifier: Apache-2.0

package inttest

import (
	"context"
	"testing"
	"time"

	"github.com/rostamlabs/rostam"
	"github.com/rostamlabs/rostam/ops"
	"github.com/rostamlabs/rostam/vector"
)

func TestVectorMetadataEndToEndDirect(t *testing.T) {
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

	// Insert vectors tagged by tenant, all near the origin.
	for i := uint64(1); i <= 20; i++ {
		tenant := "acme"
		if i%2 == 0 {
			tenant = "globex"
		}
		v := []float32{float32(i) * 0.01, 0, 0, 0}
		opts := rostam.VectorInsertOpts{Metadata: rostam.VectorMetadata{"tenant": vector.NewString(tenant)}}
		if err := store.VectorInsertExt(ctx, "docs", i, v, opts); err != nil {
			t.Fatalf("VectorInsertExt %d: %v", i, err)
		}
	}

	// Filtered search: only acme (odd ids).
	filter := rostam.VectorFilter{Op: vector.FilterEq, Field: "tenant", Value: vector.NewString("acme")}
	results, _, err := store.VectorSearchExt(ctx, "docs", []float32{0, 0, 0, 0}, 10, rostam.VectorSearchOpts{Filter: filter})
	if err != nil {
		t.Fatal(err)
	}
	if len(results) == 0 {
		t.Fatal("filtered search returned nothing")
	}
	for _, r := range results {
		if r.ID%2 == 0 {
			t.Errorf("result id %d is even (globex); filter should exclude it", r.ID)
		}
	}

	// Unfiltered search still sees everything.
	all, _, err := store.VectorSearchExt(ctx, "docs", []float32{0, 0, 0, 0}, 20, rostam.VectorSearchOpts{})
	if err != nil {
		t.Fatal(err)
	}
	if len(all) <= len(results) {
		t.Errorf("unfiltered search (%d) should return more than filtered (%d)", len(all), len(results))
	}
}

func TestVectorTTLEndToEndDirect(t *testing.T) {
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

	// TTL is carried over the wire; a 1-hour TTL must not expire immediately.
	if err := store.VectorInsertExt(ctx, "docs", 1, []float32{1, 0, 0, 0}, rostam.VectorInsertOpts{TTL: time.Hour}); err != nil {
		t.Fatalf("VectorInsertExt with ttl: %v", err)
	}
	res, _, err := store.VectorSearchExt(ctx, "docs", []float32{1, 0, 0, 0}, 5, rostam.VectorSearchOpts{})
	if err != nil {
		t.Fatal(err)
	}
	if len(res) != 1 || res[0].ID != 1 {
		t.Errorf("search after ttl insert = %+v, want [id=1]", res)
	}
}
