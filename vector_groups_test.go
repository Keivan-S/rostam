// SPDX-License-Identifier: Apache-2.0

package rostam

import (
	"context"
	"testing"

	"github.com/rostamlabs/rostam/ops"
	"github.com/rostamlabs/rostam/vector"
)

// TestVectorSearchGroupsEndToEnd drives group-by-document search over the full
// networked path (client → TCP → server → ops handler): upsert chunks across
// documents, then retrieve the top-k distinct documents with their best chunk.
func TestVectorSearchGroupsEndToEnd(t *testing.T) {
	reg := ops.NewRegistry()
	if err := ops.RegisterBuiltins(reg); err != nil {
		t.Fatal(err)
	}
	srv, err := NewDirectServer("127.0.0.1:0", DirectConfig{
		DataDir: t.TempDir(),
		Ops:     reg,
		Cache:   CacheConfig{NumShardsPerNode: 4},
	})
	if err != nil {
		t.Fatal(err)
	}
	defer srv.Close()

	store, err := NewClient(ClientConfig{Servers: []string{srv.Addr()}})
	if err != nil {
		t.Fatal(err)
	}
	defer store.Close()

	ctx := context.Background()
	cfg := VectorConfig{Dim: 3, M: 8, EfConstruction: 50, EfSearch: 32, Seed: 1, Metric: vector.L2}
	if err := store.CreateCollection(ctx, "docs", cfg); err != nil {
		t.Fatal(err)
	}

	// 6 chunks, two per document, increasing distance from the origin.
	for i := 1; i <= 6; i++ {
		doc := int64((i + 1) / 2)
		opts := VectorInsertOpts{Metadata: VectorMetadata{"doc": vector.NewInt(doc)}}
		if err := store.VectorUpsert(ctx, "docs", uint64(i), []float32{float32(i), 0, 0}, "chunk", opts); err != nil {
			t.Fatalf("upsert %d: %v", i, err)
		}
	}

	groups, _, err := store.VectorSearchGroups(ctx, "docs", []float32{0, 0, 0}, 2,
		VectorGroupOpts{GroupBy: "doc", GroupSize: 2})
	if err != nil {
		t.Fatal(err)
	}
	if len(groups) != 2 {
		t.Fatalf("got %d groups, want 2", len(groups))
	}
	if groups[0].Key.Int != 1 || groups[1].Key.Int != 2 {
		t.Errorf("group keys = %d,%d, want 1,2", groups[0].Key.Int, groups[1].Key.Int)
	}
	want := [][]uint64{{1, 2}, {3, 4}}
	for gi, g := range groups {
		if len(g.Hits) != 2 {
			t.Fatalf("group %d has %d hits, want 2", gi, len(g.Hits))
		}
		for hi, hit := range g.Hits {
			if hit.ID != want[gi][hi] {
				t.Errorf("group %d hit %d = id%d, want id%d", gi, hi, hit.ID, want[gi][hi])
			}
			if hit.Content != "chunk" {
				t.Errorf("group %d hit %d lost content: %q", gi, hi, hit.Content)
			}
		}
	}
}
