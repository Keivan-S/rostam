// SPDX-License-Identifier: Apache-2.0

package vector

import (
	"os"
	"path/filepath"
	"runtime"
	"testing"
)

// TestEffectiveConfigIVFMmapPath: a Persistent IVF collection gets MmapPath set to
// the .vecs file (so newIVF/openPersistIVF back the float arena with mmap) but does
// NOT get QuantStorage=QuantMmap (IVF-Flat is QuantNone; the QuantStorage gate would
// reject it) nor a GraphMmapPath (IVF has no level-0 graph slab). A Persistent HNSW
// collection still gets all three (regression).
func TestEffectiveConfigIVFMmapPath(t *testing.T) {
	dir := t.TempDir()
	cs, err := OpenCollectionStore(dir)
	if err != nil {
		t.Fatal(err)
	}
	defer func() { _ = cs.Close() }()

	ivfCfg := Config{
		Dim: 8, Metric: L2, Seed: 1, Persistent: true, M: 16, EfConstruction: 200, EfSearch: 64,
		IndexType: IndexIVF, IVFNlist: 8, IVFNprobe: 4,
	}
	eff := cs.effectiveConfig("default/ivf", ivfCfg)
	wantVecs, _, _, _ := cs.persistPaths("default/ivf")
	if eff.MmapPath != wantVecs {
		t.Errorf("IVF MmapPath = %q, want %q", eff.MmapPath, wantVecs)
	}
	if eff.QuantStorage == QuantMmap {
		t.Errorf("IVF QuantStorage = QuantMmap; want unset (gate requires Quant != None)")
	}
	if eff.GraphMmapPath != "" {
		t.Errorf("IVF GraphMmapPath = %q; want empty (IVF has no graph slab)", eff.GraphMmapPath)
	}
	// The effective config must round-trip through Validate (the gate must not reject
	// an IVF-Flat Persistent config now that we only set MmapPath).
	if verr := eff.Validate(); verr != nil {
		t.Errorf("effective IVF config Validate = %v, want nil", verr)
	}

	// HNSW regression: full mmap backing still set.
	hnswCfg := Config{
		Dim: 8, Metric: Cosine, M: 8, EfConstruction: 50, EfSearch: 50,
		Quant: QuantSQ8, RescoreFactor: 2, Persistent: true,
	}
	heff := cs.effectiveConfig("default/hnsw", hnswCfg)
	if heff.QuantStorage != QuantMmap || heff.MmapPath == "" || heff.GraphMmapPath == "" {
		t.Errorf("HNSW effective cfg lost mmap backing: %+v", heff)
	}
}

// TestEffectiveClusterConfigIVFForcesNonPersistent: a clustered IVF collection must
// stay on the snapshot/Raft path — effectiveClusterConfig forces Persistent=false and
// writes NO sidecar (no MmapPath for a QuantNone IVF).
func TestEffectiveClusterConfigIVFForcesNonPersistent(t *testing.T) {
	dir := t.TempDir()
	cs, err := OpenCollectionStore(dir)
	if err != nil {
		t.Fatal(err)
	}
	defer func() { _ = cs.Close() }()

	ivfCfg := Config{
		Dim: 8, Metric: L2, Seed: 1, Persistent: true, M: 16, EfConstruction: 200, EfSearch: 64,
		IndexType: IndexIVF, IVFNlist: 8, IVFNprobe: 4,
	}
	eff := cs.effectiveClusterConfig("default/ivf", 1, ivfCfg)
	if eff.Persistent {
		t.Errorf("cluster IVF Persistent = true; want false (snapshot/Raft-durable)")
	}
	if eff.WAL {
		t.Errorf("cluster IVF WAL = true; want false")
	}
	// QuantNone IVF: no vecs mmap (no sidecar) on the cluster path.
	if eff.MmapPath != "" {
		t.Errorf("cluster IVF MmapPath = %q; want empty (no sidecar on cluster path)", eff.MmapPath)
	}
}

// TestCollectionStoreIVFFlatPersistentRestart: end-to-end single-node instant restart
// for a Persistent IVF-Flat collection — create, insert past auto-train, query, Flush,
// Close the store, reopen the same DataDir, run the SAME queries → IDENTICAL results
// (the sidecar + .vecs mmap restored the index, no rebuild-from-snapshot). Also asserts
// the sidecar (.meta) + .vecs files were written and NO .graph file (IVF has no graph).
func TestCollectionStoreIVFFlatPersistentRestart(t *testing.T) {
	const n, dim, k = 1500, 16, 10
	dir := t.TempDir()

	ids, vecs := siftLikeCorpus(n, dim, 7)
	_, queries := siftLikeCorpus(40, dim, 8)

	cs, err := OpenCollectionStore(dir)
	if err != nil {
		t.Fatal(err)
	}
	cfg := Config{
		Dim: dim, Metric: L2, Seed: 1, Persistent: true, M: 16, EfConstruction: 200, EfSearch: 64,
		IndexType: IndexIVF, IVFNlist: 16, IVFNprobe: 8,
		IVFTrainThreshold: 500, // small so the inserts cross auto-train in-test
	}
	if err := cs.CreateCollection("docs", cfg); err != nil {
		t.Fatalf("create: %v", err)
	}
	coll, ok := cs.Get("docs")
	if !ok {
		t.Fatal("collection missing after create")
	}
	for i, id := range ids {
		if ierr := coll.Insert(id, vecs[i], 0, nil, nil); ierr != nil {
			t.Fatalf("insert %d: %v", id, ierr)
		}
	}

	before := make([][]uint64, len(queries))
	for i, q := range queries {
		res, serr := coll.Search(q, k)
		if serr != nil {
			t.Fatal(serr)
		}
		before[i] = resultIDs(res)
	}

	if err := cs.Flush("docs"); err != nil {
		t.Fatalf("flush: %v", err)
	}
	if err := cs.Close(); err != nil {
		t.Fatalf("close store: %v", err)
	}

	// The sidecar + vecs mmap must exist; IVF has no graph file.
	for _, suffix := range []string{".json", ".vecs", ".meta"} {
		p := filepath.Join(dir, "vectors", "default", "docs"+suffix)
		if _, serr := os.Stat(p); serr != nil {
			t.Errorf("expected persisted file %s: %v", p, serr)
		}
	}
	if _, serr := os.Stat(filepath.Join(dir, "vectors", "default", "docs.graph")); serr == nil {
		t.Errorf("IVF wrote a .graph file; want none")
	}

	cs2, err := OpenCollectionStore(dir)
	if err != nil {
		t.Fatalf("reopen store: %v", err)
	}
	defer func() { _ = cs2.Close() }()
	coll2, ok := cs2.Get("docs")
	if !ok {
		t.Fatal("collection missing after reopen")
	}
	for i, q := range queries {
		res, serr := coll2.Search(q, k)
		if serr != nil {
			t.Fatal(serr)
		}
		if !eqUint64(resultIDs(res), before[i]) {
			t.Fatalf("query %d after instant-restart: %v != %v", i, resultIDs(res), before[i])
		}
	}
}

// TestCollectionStoreIVFPQPersistentRestart: same instant-restart contract for a
// Persistent IVF-PQ (IVFRerank: vecs present, mmap-backed) collection. The PQ codes +
// codebooks + the rerank floats all survive the close/reopen via the sidecar + vecs mmap.
func TestCollectionStoreIVFPQPersistentRestart(t *testing.T) {
	const n, dim, k = 1500, 16, 10
	dir := t.TempDir()

	ids, vecs := siftLikeCorpus(n, dim, 11)
	_, queries := siftLikeCorpus(40, dim, 12)

	cs, err := OpenCollectionStore(dir)
	if err != nil {
		t.Fatal(err)
	}
	cfg := Config{
		Dim: dim, Metric: L2, Seed: 1, Persistent: true, M: 16, EfConstruction: 200, EfSearch: 64,
		IndexType: IndexIVF, IVFNlist: 16, IVFNprobe: 8,
		IVFPQ: true, IVFPQM: 8, IVFRerank: true,
		IVFTrainThreshold: 500,
	}
	if err := cs.CreateCollection("docs", cfg); err != nil {
		t.Fatalf("create: %v", err)
	}
	coll, ok := cs.Get("docs")
	if !ok {
		t.Fatal("collection missing after create")
	}
	if berr := coll.BuildConcurrent(ids, vecs, runtime.GOMAXPROCS(0)); berr != nil {
		t.Fatalf("build: %v", berr)
	}

	before := make([][]uint64, len(queries))
	for i, q := range queries {
		res, serr := coll.Search(q, k)
		if serr != nil {
			t.Fatal(serr)
		}
		before[i] = resultIDs(res)
	}

	if err := cs.Flush("docs"); err != nil {
		t.Fatalf("flush: %v", err)
	}
	if err := cs.Close(); err != nil {
		t.Fatalf("close store: %v", err)
	}

	cs2, err := OpenCollectionStore(dir)
	if err != nil {
		t.Fatalf("reopen store: %v", err)
	}
	defer func() { _ = cs2.Close() }()
	coll2, ok := cs2.Get("docs")
	if !ok {
		t.Fatal("collection missing after reopen")
	}
	for i, q := range queries {
		res, serr := coll2.Search(q, k)
		if serr != nil {
			t.Fatal(serr)
		}
		if !eqUint64(resultIDs(res), before[i]) {
			t.Fatalf("IVF-PQ query %d after instant-restart: %v != %v", i, resultIDs(res), before[i])
		}
	}
}
