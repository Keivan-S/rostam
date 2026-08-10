// SPDX-License-Identifier: Apache-2.0

package shard

import (
	"bytes"
	"io"
	"math"
	"testing"

	"github.com/rostamlabs/rostam/cache"
	"github.com/rostamlabs/rostam/vector"
)

// TestFSMSnapshotRebuildsFullText proves the Raft shard durability path for the
// BM25 full-text lane: a v2 FSM snapshot persists the FullText config + each
// point's $content, and restoring into a fresh store (a recovering/new node)
// rebuilds the bm25 lane so SearchText / HybridText work after restore with the
// SAME ranking as before the snapshot. The bm25Index itself is never serialized;
// RestoreAll -> NewCollection allocates the lane + analyzer (from the persisted
// config) and Collection.Restore re-derives the postings from $content.
func TestFSMSnapshotRebuildsFullText(t *testing.T) {
	c, _ := cache.New(cache.DefaultConfig())
	defer c.Close()

	src, err := vector.OpenCollectionStore(t.TempDir())
	if err != nil {
		t.Fatal(err)
	}
	defer src.Close()

	cfg := vector.Config{
		Dim: 4, Metric: vector.L2, M: 8, EfConstruction: 50, EfSearch: 32, Seed: 1,
		FullText: &vector.FullTextConfig{Analyzer: "english", K1: 1.3, B: 0.6},
	}
	if err := src.CreateCollection("docs", cfg); err != nil {
		t.Fatal(err)
	}
	corpus := []struct {
		id      uint64
		vec     []float32
		content string
	}{
		{1, []float32{1, 0, 0, 0}, "the quick brown fox jumps over the lazy dog"},
		{2, []float32{0, 1, 0, 0}, "quick foxes are clever and quick and quick"},
		{3, []float32{0, 0, 1, 0}, "a lazy dog sleeps in the warm sun"},
		{4, []float32{0, 0, 0, 1}, "brown bears and brown foxes roam the forest"},
	}
	for _, d := range corpus {
		if err := src.Upsert("docs", d.id, d.vec, d.content, 0, nil, nil); err != nil {
			t.Fatal(err)
		}
	}
	const query = "quick brown fox"
	before, err := src.SearchText("docs", query, 10, vector.Filter{})
	if err != nil {
		t.Fatal(err)
	}
	if len(before) == 0 {
		t.Fatal("pre-snapshot SearchText returned nothing; test can't distinguish a rebuild")
	}

	// Persist the FSM snapshot (cache + vectors).
	var sink fakeSink
	sink.Buffer = &bytes.Buffer{}
	data, err := serializeSnapshot(c, src, 0, nil)
	if err != nil {
		t.Fatalf("serialize: %v", err)
	}
	snap := &fsmSnapshot{data: data}
	if err := snap.Persist(sink); err != nil {
		t.Fatalf("persist: %v", err)
	}

	// Restore into a fresh store (recovering node).
	c2, _ := cache.New(cache.DefaultConfig())
	defer c2.Close()
	dst, err := vector.OpenCollectionStore(t.TempDir())
	if err != nil {
		t.Fatal(err)
	}
	defer dst.Close()
	rc := io.NopCloser(bytes.NewReader(sink.Bytes()))
	if _, err := restoreSnapshot(c2, dst, nil, rc); err != nil {
		t.Fatalf("restore: %v", err)
	}

	// SearchText works after restore with the identical ranking.
	after, err := dst.SearchText("docs", query, 10, vector.Filter{})
	if err != nil {
		t.Fatalf("SearchText after restore: %v", err)
	}
	if len(after) != len(before) {
		t.Fatalf("restored ranking size = %d, want %d", len(after), len(before))
	}
	for i := range after {
		if after[i].ID != before[i].ID {
			t.Fatalf("restored rank %d: id got %d want %d", i, after[i].ID, before[i].ID)
		}
		if math.Abs(float64(after[i].Score-before[i].Score)) > 1e-5 {
			t.Fatalf("restored rank %d (id %d): score got %v want %v", i, after[i].ID, after[i].Score, before[i].Score)
		}
	}

	// HybridText (dense + text lanes fused) must also work post-restore.
	if _, err := dst.HybridText("docs", []float32{1, 0, 0, 0}, query, 5, vector.HybridOpts{}); err != nil {
		t.Fatalf("HybridText after restore: %v", err)
	}

	// The FullText config (analyzer + k1/b) survived the snapshot round-trip.
	col, ok := dst.Get("docs")
	if !ok {
		t.Fatal("collection missing after restore")
	}
	ft := col.Config().FullText
	if ft == nil {
		t.Fatal("FullText config lost across the FSM snapshot")
	}
	if ft.Analyzer != "english" || ft.K1 != 1.3 || ft.B != 0.6 {
		t.Errorf("restored FullText config = %+v, want {english 1.3 0.6}", *ft)
	}
}
