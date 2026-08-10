// SPDX-License-Identifier: Apache-2.0

package inttest

import (
	"context"
	"math/rand"
	"testing"

	"github.com/rostamlabs/rostam"
	"github.com/rostamlabs/rostam/ops"
	"github.com/rostamlabs/rostam/vector"
)

func TestVectorEndToEndDirect(t *testing.T) {
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

	// Create a collection.
	cfg := rostam.VectorConfig{Dim: 4, M: 8, EfConstruction: 50, EfSearch: 32, Seed: 1, Metric: vector.L2}
	if err := store.CreateCollection(ctx, "docs", cfg); err != nil {
		t.Fatal(err)
	}

	// Insert 100 random vectors.
	rng := rand.New(rand.NewSource(42))
	for i := uint64(1); i <= 100; i++ {
		v := []float32{
			float32(rng.NormFloat64()),
			float32(rng.NormFloat64()),
			float32(rng.NormFloat64()),
			float32(rng.NormFloat64()),
		}
		if err := store.VectorInsert(ctx, "docs", i, v); err != nil {
			t.Fatalf("insert %d: %v", i, err)
		}
	}

	// Search.
	results, err := store.VectorSearch(ctx, "docs", []float32{0, 0, 0, 0}, 5)
	if err != nil {
		t.Fatal(err)
	}
	if len(results) != 5 {
		t.Errorf("got %d results, want 5", len(results))
	}

	// Delete one and verify search hides it.
	if _, err := store.VectorDelete(ctx, "docs", results[0].ID); err != nil {
		t.Fatal(err)
	}
	results2, _ := store.VectorSearch(ctx, "docs", []float32{0, 0, 0, 0}, 5)
	for _, r := range results2 {
		if r.ID == results[0].ID {
			t.Errorf("deleted id %d still present in results: %+v", results[0].ID, results2)
		}
	}
}
