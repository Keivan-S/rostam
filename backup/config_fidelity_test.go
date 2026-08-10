// SPDX-License-Identifier: Apache-2.0

package backup

import (
	"context"
	"testing"
	"time"

	"github.com/rostamlabs/rostam/objstore"
	"github.com/rostamlabs/rostam/vector"
)

// siftLike builds a deterministic pseudo-random corpus + queries, mirroring the
// vector package's test corpus shape (this package can't import that test helper).
func siftLike(n, dim int, seed int64) ([]uint64, [][]float32) {
	ids := make([]uint64, n)
	vecs := make([][]float32, n)
	st := uint64(seed*2862933555777941757 + 3037000493)
	next := func() float32 {
		st = st*6364136223846793005 + 1442695040888963407
		return float32((st>>33)&0xFFFFFF)/float32(0x1000000)*2 - 1
	}
	for i := 0; i < n; i++ {
		ids[i] = uint64(i + 1)
		v := make([]float32, dim)
		for d := range v {
			v[d] = next()
		}
		vecs[i] = v
	}
	return ids, vecs
}

func resultIDs(res []vector.Result) []uint64 {
	out := make([]uint64, len(res))
	for i, r := range res {
		out[i] = r.ID
	}
	return out
}

func eqIDs(a, b []uint64) bool {
	if len(a) != len(b) {
		return false
	}
	for i := range a {
		if a[i] != b[i] {
			return false
		}
	}
	return true
}

// buildAndSearch creates a collection with cfg on store, bulk-builds the corpus,
// and returns the per-query result-id lists (the search "fingerprint").
func buildAndSearch(t *testing.T, store *vector.CollectionStore, name string, cfg vector.Config,
	ids []uint64, vecs [][]float32, queries [][]float32, k int) [][]uint64 {
	t.Helper()
	if err := store.CreateCollection(name, cfg); err != nil {
		t.Fatalf("create %q: %v", name, err)
	}
	if err := store.StageBulk(name, ids, vecs); err != nil {
		t.Fatalf("stage %q: %v", name, err)
	}
	if err := store.BuildStaged(name, 4); err != nil {
		t.Fatalf("build %q: %v", name, err)
	}
	c, ok := store.Acquire(name)
	if !ok {
		t.Fatalf("acquire %q", name)
	}
	defer c.Release()
	out := make([][]uint64, len(queries))
	for i, q := range queries {
		res, err := c.Search(q, k)
		if err != nil {
			t.Fatalf("search %q: %v", name, err)
		}
		out[i] = resultIDs(res)
	}
	return out
}

// TestRestoreConfigFaithfulQuantSQ is the Part-A gap closure for a QUANTIZED
// (QuantSQ, non-default SQBits=4) collection: backup → fresh store → RestoreLatest
// must reconstruct the collection with the SAME quantizer geometry and return the
// SAME search results. With the OLD config-LESS RestoreCollection the collection
// would rebuild as a plain HNSW (Quant=QuantNone), so Config().Quant would be wrong
// and the results would diverge — this test fails without the .cfg.json sibling.
func TestRestoreConfigFaithfulQuantSQ(t *testing.T) {
	ctx := context.Background()
	const (
		n   = 1500
		dim = 64
		k   = 10
	)
	ids, vecs := siftLike(n, dim, 31)
	_, queries := siftLike(20, dim, 99)

	cfg := vector.Config{
		Dim: dim, Metric: vector.L2, M: 16, EfConstruction: 200, EfSearch: 64, Seed: 31,
		Quant: vector.QuantSQ, SQBits: 4,
	}

	src := newStore(t)
	before := buildAndSearch(t, src, "sq", cfg, ids, vecs, queries, k)

	obj := objstore.NewMemStore()
	ts := time.Date(2026, 6, 23, 10, 0, 0, 0, time.UTC)
	results, err := Backup(ctx, src, obj, BackupOpts{Tenant: "acme", Timestamp: ts})
	if err != nil {
		t.Fatalf("backup: %v", err)
	}
	if len(results) != 1 || results[0].Err != nil {
		t.Fatalf("backup results = %+v", results)
	}

	// Restore onto a FRESH store via the config-faithful path.
	dst := newStore(t)
	if err := RestoreLatest(ctx, dst, obj, "acme", "default/sq"); err != nil {
		t.Fatalf("restore latest: %v", err)
	}

	rc, ok := dst.Get("sq")
	if !ok {
		t.Fatal("restored collection missing")
	}
	if got := rc.Config().Quant; got != vector.QuantSQ {
		t.Fatalf("restored Quant = %v, want QuantSQ (config-less restore lost the quantizer)", got)
	}
	if got := rc.Config().SQBits; got != 4 {
		t.Fatalf("restored SQBits = %d, want 4 (config-less restore lost the geometry)", got)
	}
	for i, q := range queries {
		res, serr := rc.Search(q, k)
		if serr != nil {
			t.Fatal(serr)
		}
		if !eqIDs(resultIDs(res), before[i]) {
			t.Fatalf("query %d: restored %v != original %v", i, resultIDs(res), before[i])
		}
	}
}

// TestRestoreConfigFaithfulVamana is the Part-A gap closure for a VAMANA
// (non-default R/L/alpha) collection: the snapshot stream alone does NOT carry
// IndexType/VamanaR, so a config-less restore rebuilds it as a plain HNSW with the
// wrong slab stride and scrambled graph. The .cfg.json sibling lets RestoreLatest
// re-create it as IndexVamana with the right geometry and return identical results.
func TestRestoreConfigFaithfulVamana(t *testing.T) {
	ctx := context.Background()
	const (
		n   = 1200
		dim = 48
		k   = 10
	)
	ids, vecs := siftLike(n, dim, 23)
	_, queries := siftLike(20, dim, 77)

	cfg := vector.Config{
		Dim: dim, Metric: vector.Cosine, IndexType: vector.IndexVamana,
		VamanaR: 40, VamanaL: 90, VamanaAlpha: 1.4, Seed: 23,
	}

	src := newStore(t)
	before := buildAndSearch(t, src, "vam", cfg, ids, vecs, queries, k)

	obj := objstore.NewMemStore()
	ts := time.Date(2026, 6, 23, 11, 0, 0, 0, time.UTC)
	if _, err := Backup(ctx, src, obj, BackupOpts{Tenant: "acme", Timestamp: ts}); err != nil {
		t.Fatalf("backup: %v", err)
	}

	dst := newStore(t)
	if err := RestoreLatest(ctx, dst, obj, "acme", "default/vam"); err != nil {
		t.Fatalf("restore latest: %v", err)
	}

	rc, ok := dst.Get("vam")
	if !ok {
		t.Fatal("restored collection missing")
	}
	if got := rc.Config().IndexType; got != vector.IndexVamana {
		t.Fatalf("restored IndexType = %d, want IndexVamana (config-less restore lost it)", got)
	}
	if rc.Config().VamanaR != 40 || rc.Config().VamanaL != 90 || rc.Config().VamanaAlpha != 1.4 {
		t.Fatalf("restored Vamana geometry lost: R=%d L=%d alpha=%v",
			rc.Config().VamanaR, rc.Config().VamanaL, rc.Config().VamanaAlpha)
	}
	for i, q := range queries {
		res, serr := rc.Search(q, k)
		if serr != nil {
			t.Fatal(serr)
		}
		if !eqIDs(resultIDs(res), before[i]) {
			t.Fatalf("query %d: restored %v != original %v", i, resultIDs(res), before[i])
		}
	}
}
