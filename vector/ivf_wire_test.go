// SPDX-License-Identifier: Apache-2.0

package vector

import (
	"bytes"
	"math/rand"
	"testing"
)

// TestNewIndexDispatch verifies newIndex/openIndex pick the index implementation
// from cfg.IndexType: IndexIVF -> *ivf, default (IndexHNSW) -> *hnsw.
func TestNewIndexDispatch(t *testing.T) {
	base := Config{Dim: 8, Metric: Cosine, M: 16, EfConstruction: 200, EfSearch: 64, Seed: 1}

	hi, err := newIndex(base)
	if err != nil {
		t.Fatalf("newIndex(HNSW): %v", err)
	}
	if _, ok := hi.(*hnsw); !ok {
		t.Fatalf("default IndexType: got %T, want *hnsw", hi)
	}
	_ = hi.Close()

	ivfCfg := base
	ivfCfg.IndexType = IndexIVF
	ivfCfg.IVFNlist = 4
	ivfCfg.IVFNprobe = 3
	ii, err := newIndex(ivfCfg)
	if err != nil {
		t.Fatalf("newIndex(IVF): %v", err)
	}
	ix, ok := ii.(*ivf)
	if !ok {
		t.Fatalf("IndexIVF: got %T, want *ivf", ii)
	}
	if ix.nprobe != 3 {
		t.Errorf("IVFNprobe not threaded: nprobe=%d, want 3", ix.nprobe)
	}
	_ = ii.Close()

	// openIndex mirrors the dispatch (IVF is snapshot-only — openIVF returns a
	// fresh empty index the caller then Restores).
	oi, err := openIndex(ivfCfg, "")
	if err != nil {
		t.Fatalf("openIndex(IVF): %v", err)
	}
	if _, ok := oi.(*ivf); !ok {
		t.Fatalf("openIndex IndexIVF: got %T, want *ivf", oi)
	}
	_ = oi.Close()
}

// TestIVFCollectionEndToEnd creates an IVF collection through the store, inserts,
// builds (trains), searches, then snapshots + restores it and confirms it is
// STILL an IVF collection and search still returns the nearest neighbor.
func TestIVFCollectionEndToEnd(t *testing.T) {
	src, err := OpenCollectionStore(t.TempDir())
	if err != nil {
		t.Fatal(err)
	}
	defer src.Close()

	cfg := Config{
		Dim: 16, Metric: L2, M: 16, EfConstruction: 200, EfSearch: 64, Seed: 1,
		IndexType: IndexIVF, IVFNlist: 8, IVFNprobe: 4,
	}
	if err := src.CreateCollection("ivfcol", cfg); err != nil {
		t.Fatalf("create IVF collection: %v", err)
	}

	// Confirm the collection is backed by an *ivf.
	c, ok := src.Get("ivfcol")
	if !ok {
		t.Fatal("collection missing after create")
	}
	if _, isIVF := c.idx.(*ivf); !isIVF {
		t.Fatalf("created collection idx = %T, want *ivf", c.idx)
	}
	if c.Config().IndexType != IndexIVF {
		t.Fatalf("collection Config().IndexType = %d, want IndexIVF", c.Config().IndexType)
	}

	// Insert 200 random vectors; bulk-build so the IVF trains its coarse quantizer.
	rng := rand.New(rand.NewSource(7))
	const n = 200
	ids := make([]uint64, n)
	vecs := make([][]float32, n)
	for i := 0; i < n; i++ {
		v := make([]float32, cfg.Dim)
		for d := range v {
			v[d] = rng.Float32()
		}
		ids[i] = uint64(i + 1)
		vecs[i] = v
	}
	if err := src.StageBulk("ivfcol", ids, vecs); err != nil {
		t.Fatalf("stage bulk: %v", err)
	}
	if err := src.BuildStaged("ivfcol", 0); err != nil {
		t.Fatalf("build staged: %v", err)
	}

	// The query is vector #1 itself, so its nearest neighbor must be id 1.
	res, err := src.SearchFiltered("ivfcol", vecs[0], 5, Filter{})
	if err != nil {
		t.Fatalf("search: %v", err)
	}
	if len(res) == 0 {
		t.Fatal("search returned no results")
	}
	if res[0].ID != 1 {
		t.Fatalf("nearest neighbor = id %d, want 1", res[0].ID)
	}

	// Snapshot + restore into a fresh store.
	var buf bytes.Buffer
	if err := src.SnapshotAll(&buf); err != nil {
		t.Fatalf("snapshot: %v", err)
	}
	dst, err := OpenCollectionStore(t.TempDir())
	if err != nil {
		t.Fatal(err)
	}
	defer dst.Close()
	if err := dst.RestoreAll(bytes.NewReader(buf.Bytes())); err != nil {
		t.Fatalf("restore: %v", err)
	}

	// It must restore AS IVF (the persisted IndexType drove openIndex/NewCollection).
	rc, ok := dst.Get("ivfcol")
	if !ok {
		t.Fatal("collection missing after restore")
	}
	if _, isIVF := rc.idx.(*ivf); !isIVF {
		t.Fatalf("restored collection idx = %T, want *ivf (IndexType not persisted/dispatched)", rc.idx)
	}
	if rc.Config().IndexType != IndexIVF {
		t.Fatalf("restored Config().IndexType = %d, want IndexIVF", rc.Config().IndexType)
	}

	// Search still works post-restore: nearest neighbor of vec #1 is id 1.
	rres, err := dst.SearchFiltered("ivfcol", vecs[0], 5, Filter{})
	if err != nil {
		t.Fatalf("search after restore: %v", err)
	}
	if len(rres) == 0 || rres[0].ID != 1 {
		t.Fatalf("post-restore nearest = %+v, want id 1", rres)
	}
}

// TestDefaultCreateStaysHNSW confirms a default create (no IndexType) is still an
// HNSW collection — the IVF dispatch does not regress the default path.
func TestDefaultCreateStaysHNSW(t *testing.T) {
	src, err := OpenCollectionStore(t.TempDir())
	if err != nil {
		t.Fatal(err)
	}
	defer src.Close()
	if err := src.CreateCollection("hcol", Config{Dim: 4, Metric: L2, M: 8, EfConstruction: 50, EfSearch: 32, Seed: 1}); err != nil {
		t.Fatal(err)
	}
	c, ok := src.Get("hcol")
	if !ok {
		t.Fatal("collection missing")
	}
	if _, isHNSW := c.idx.(*hnsw); !isHNSW {
		t.Fatalf("default create idx = %T, want *hnsw", c.idx)
	}
}
