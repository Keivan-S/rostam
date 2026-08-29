// SPDX-License-Identifier: Apache-2.0

package vector

import (
	"errors"
	"fmt"
	"os"
	"path/filepath"
	"runtime"
	"testing"
	"time"
)

// buildPersistableSIFT builds a normalized cosine SQ8 index backed by mmap files
// (vectors + graph) under dir, ready for SavePersist.
func buildPersistableSIFT(t *testing.T, dir string, n, dim int, seed int64) (*hnsw, Config, [][]float32) {
	t.Helper()
	ids, vecs := siftLikeCorpus(n, dim, seed)
	for _, v := range vecs {
		normalize(v)
	}
	cfg := Config{
		Dim: dim, Metric: Cosine, M: 16, EfConstruction: 200, EfSearch: 64, Seed: 1,
		Quant: QuantSQ8, QuantStorage: QuantMmap,
		MmapPath:      filepath.Join(dir, "vecs.dat"),
		RescoreFactor: 3,
		GraphMmapPath: filepath.Join(dir, "graph.dat"),
	}
	h, err := newHNSW(cfg)
	if err != nil {
		t.Fatal(err)
	}
	if err := h.BuildConcurrent(ids, vecs, runtime.GOMAXPROCS(0)); err != nil {
		t.Fatalf("build: %v", err)
	}
	return h, cfg, vecs
}

// TestPersistInstantRestart is the core check: an mmap+graph-mmap index is
// saved, closed, and reopened by MAPPING its files (no rebuild) — and returns
// byte-identical search results to the original. If OpenPersist were rebuilding
// or losing edges, results would differ.
func TestPersistInstantRestart(t *testing.T) {
	const n, dim, k = 4000, 32, 10
	dir := t.TempDir()
	h, cfg, vecs := buildPersistableSIFT(t, dir, n, dim, 3)
	_, queries := siftLikeCorpus(120, dim, 9)
	for _, q := range queries {
		normalize(q)
	}

	// Capture results before saving.
	before := make([][]uint64, len(queries))
	for i, q := range queries {
		res, err := h.Search(q, k)
		if err != nil {
			t.Fatal(err)
		}
		before[i] = resultIDs(res)
	}

	metaPath := filepath.Join(dir, "meta.bin")
	if err := h.SavePersist(metaPath); err != nil {
		t.Fatalf("SavePersist: %v", err)
	}
	if err := h.Close(); err != nil {
		t.Fatalf("close: %v", err)
	}

	// Reopen by mapping the files — no BuildConcurrent / no Insert loop.
	h2, err := openPersist(cfg, metaPath)
	if err != nil {
		t.Fatalf("openPersist: %v", err)
	}
	defer func() { _ = h2.Close() }()

	if got := h2.arena.Size(); got != n {
		t.Errorf("restored size = %d, want %d", got, n)
	}
	for i, q := range queries {
		res, err := h2.Search(q, k)
		if err != nil {
			t.Fatal(err)
		}
		if !eqUint64(resultIDs(res), before[i]) {
			t.Errorf("query %d: restored results %v != original %v", i, resultIDs(res), before[i])
			break
		}
	}

	// A restored index must still accept new inserts (exercises mmap growth past
	// the restored size on both the vector and graph slabs).
	nv := make([]float32, dim)
	for j := range nv {
		nv[j] = 0.1 * float32(j+1)
	}
	normalize(nv)
	if _, _, err := h2.Insert(uint64(n+1), nv, 0, nil, nil, nil, CASCond{}); err != nil {
		t.Errorf("insert after restore: %v", err)
	}
	if got := h2.arena.Size(); got != n+1 {
		t.Errorf("size after post-restore insert = %d, want %d", got, n+1)
	}
	_ = vecs
}

// TestPersistOpenLatencySIFT quantifies "instant restart" on SIFT-1M: it times
// the concurrent build, saves+closes, then times OpenPersist (map files + O(n)
// re-encode/idMap, no graph linking) and reports the speedup. Opt-in:
// ROSTAM_SIFT1M=1.
func TestPersistOpenLatencySIFT(t *testing.T) {
	if os.Getenv("ROSTAM_SIFT1M") != "1" {
		t.Skip("set ROSTAM_SIFT1M=1 with dataset at /tmp/rostam-sift1m/sift/ to run")
	}
	dir := "/tmp/rostam-sift1m/sift"
	if d := os.Getenv("ROSTAM_SIFT_DIR"); d != "" {
		dir = d
	}
	base, err := readFvecs(filepath.Join(dir, "sift_base.fvecs"))
	if err != nil {
		t.Fatal(err)
	}
	for _, v := range base {
		normalize(v)
	}
	ids := make([]uint64, len(base))
	for i := range ids {
		ids[i] = uint64(i + 1)
	}
	work := t.TempDir()
	cfg := Config{
		Dim: len(base[0]), Metric: Cosine, M: 16, EfConstruction: 200, EfSearch: 64, Seed: 1,
		Quant: QuantSQ8, QuantStorage: QuantMmap, MmapPath: filepath.Join(work, "vecs.dat"),
		RescoreFactor: 3, GraphMmapPath: filepath.Join(work, "graph.dat"),
	}
	h, err := newHNSW(cfg)
	if err != nil {
		t.Fatal(err)
	}
	t0 := time.Now()
	if err := h.BuildConcurrent(ids, base, runtime.GOMAXPROCS(0)); err != nil {
		t.Fatal(err)
	}
	buildDur := time.Since(t0)
	metaPath := filepath.Join(work, "meta.bin")
	if err := h.SavePersist(metaPath); err != nil {
		t.Fatal(err)
	}
	_ = h.Close()

	t1 := time.Now()
	h2, err := openPersist(cfg, metaPath)
	if err != nil {
		t.Fatalf("openPersist: %v", err)
	}
	openDur := time.Since(t1)
	defer func() { _ = h2.Close() }()

	fmt.Fprintf(os.Stderr, "[persist] build=%v  open(instant)=%v  speedup=%.0fx  (n=%d)\n",
		buildDur.Round(time.Millisecond), openDur.Round(time.Millisecond),
		buildDur.Seconds()/openDur.Seconds(), len(base))
	if got := h2.arena.Size(); got != len(base) {
		t.Errorf("restored size = %d, want %d", got, len(base))
	}
}

// TestPersistRejectsUnsupported checks SavePersist refuses (rather than silently
// dropping state) for a non-mmap index, which has no files to flush.
func TestPersistRejectsUnsupported(t *testing.T) {
	plain, err := newHNSW(Config{Dim: 4, Metric: L2, M: 8, EfConstruction: 50, EfSearch: 50, Seed: 1})
	if err != nil {
		t.Fatal(err)
	}
	_, _, _ = plain.Insert(1, []float32{1, 0, 0, 0}, 0, nil, nil, nil, CASCond{})
	if err := plain.SavePersist(filepath.Join(t.TempDir(), "m.bin")); !errors.Is(err, ErrPersistUnsupported) {
		t.Errorf("non-mmap SavePersist = %v, want ErrPersistUnsupported", err)
	}
}

// TestPersistDeleteTolerant checks that a Persistent (mmap) index carrying
// tombstones (lazy Delete) and reclaimed holes (Reclaim) can be saved and
// reopened: the survivors round-trip and the deleted ids stay gone.
func TestPersistDeleteTolerant(t *testing.T) {
	const dim = 8
	dir := t.TempDir()
	cfg := Config{
		Dim: dim, Metric: Cosine, M: 16, EfConstruction: 100, EfSearch: 64, Seed: 1,
		Quant: QuantSQ8, QuantStorage: QuantMmap, MmapPath: filepath.Join(dir, "v.dat"),
		RescoreFactor: 3, GraphMmapPath: filepath.Join(dir, "g.dat"),
	}
	h, err := newHNSW(cfg)
	if err != nil {
		t.Fatal(err)
	}
	_, vecs := siftLikeCorpus(200, dim, 8)
	for i, v := range vecs {
		if _, _, err := h.Insert(uint64(i+1), v, 0, Metadata{"i": NewInt(int64(i + 1))}, nil, nil, CASCond{}); err != nil {
			t.Fatal(err)
		}
	}

	// Delete a tombstone-only set, and a reclaimed (hole) set.
	deleted := map[uint64]bool{}
	for _, id := range []uint64{5, 17, 42, 99, 150} { // stay tombstoned
		h.Delete(id, CASCond{})
		deleted[id] = true
	}
	for _, id := range []uint64{3, 60, 120} { // will become holes after Reclaim
		h.Delete(id, CASCond{})
		deleted[id] = true
	}
	h.Reclaim() // turns the most-recent tombstones into free holes
	wantSize := h.arena.Size()

	if err := h.SavePersist(filepath.Join(dir, "m.bin")); err != nil {
		t.Fatalf("SavePersist with deletes: %v", err)
	}
	_ = h.Close()

	h2, err := openPersist(cfg, filepath.Join(dir, "m.bin"))
	if err != nil {
		t.Fatalf("openPersist: %v", err)
	}
	defer func() { _ = h2.Close() }()

	if got := h2.arena.Size(); got != wantSize {
		t.Errorf("reopened size = %d, want %d", got, wantSize)
	}
	// Every deleted id must be absent; a sample of survivors must be present.
	for id := range deleted {
		if _, ok := h2.arena.idMap[id]; ok {
			t.Errorf("deleted id %d still present after reopen", id)
		}
	}
	for _, id := range []uint64{1, 2, 4, 100, 200} {
		if _, ok := h2.arena.idMap[id]; !ok {
			t.Errorf("survivor id %d missing after reopen", id)
		}
	}
	// Search still works and never returns a deleted id.
	res, err := h2.Search(vecs[0], 10)
	if err != nil {
		t.Fatal(err)
	}
	for _, r := range res {
		if deleted[r.ID] {
			t.Errorf("search returned deleted id %d", r.ID)
		}
	}
}

// TestPersistSparseRoundtrip checks dense+sparse (hybrid) vectors survive
// save+restart and that hybrid search returns identical results (the sparse
// inverted index is rebuilt on open).
func TestPersistSparseRoundtrip(t *testing.T) {
	const dim, k = 16, 10
	dir := t.TempDir()
	cfg := Config{
		Dim: dim, Metric: Cosine, M: 16, EfConstruction: 100, EfSearch: 64, Seed: 1,
		Quant: QuantSQ8, QuantStorage: QuantMmap, MmapPath: filepath.Join(dir, "v.dat"),
		RescoreFactor: 3, GraphMmapPath: filepath.Join(dir, "g.dat"),
	}
	h, err := newHNSW(cfg)
	if err != nil {
		t.Fatal(err)
	}
	_, vecs := siftLikeCorpus(300, dim, 8)
	for i, v := range vecs {
		normalize(v)
		// A small sparse vector with sorted, unique indices.
		sv := &SparseVector{
			Indices: []uint32{uint32(i % 7), uint32(7 + i%5)},
			Values:  []float32{1, 0.5},
		}
		if _, _, err := h.Insert(uint64(i+1), v, 0, nil, sv, nil, CASCond{}); err != nil {
			t.Fatal(err)
		}
	}
	qd := vecs[0]
	qs := SparseVector{Indices: []uint32{0, 8}, Values: []float32{1, 1}}
	before, err := h.HybridSearch(qd, qs, k, HybridOpts{})
	if err != nil {
		t.Fatal(err)
	}
	if len(before) == 0 {
		t.Fatal("pre-save hybrid search returned nothing")
	}

	metaPath := filepath.Join(dir, "meta.bin")
	if err := h.SavePersist(metaPath); err != nil {
		t.Fatalf("SavePersist with sparse: %v", err)
	}
	_ = h.Close()

	h2, err := openPersist(cfg, metaPath)
	if err != nil {
		t.Fatalf("openPersist: %v", err)
	}
	defer func() { _ = h2.Close() }()
	after, err := h2.HybridSearch(qd, qs, k, HybridOpts{})
	if err != nil {
		t.Fatal(err)
	}
	if !eqUint64(resultIDs(before), resultIDs(after)) {
		t.Errorf("hybrid search after restart: %v != %v", resultIDs(after), resultIDs(before))
	}
}

// TestPersistMetadataRoundtrip checks that per-slot metadata and TTL survive
// save+restart and that filtered search works against the reopened index (the
// payload index is rebuilt on open).
func TestPersistMetadataRoundtrip(t *testing.T) {
	const dim, k = 16, 10
	dir := t.TempDir()
	cfg := Config{
		Dim: dim, Metric: Cosine, M: 16, EfConstruction: 100, EfSearch: 64, Seed: 1,
		Quant: QuantSQ8, QuantStorage: QuantMmap, MmapPath: filepath.Join(dir, "v.dat"),
		RescoreFactor: 3, GraphMmapPath: filepath.Join(dir, "g.dat"),
	}
	h, err := newHNSW(cfg)
	if err != nil {
		t.Fatal(err)
	}
	// Insert (not BuildConcurrent — metadata travels through Insert).
	_, vecs := siftLikeCorpus(400, dim, 8)
	for i, v := range vecs {
		normalize(v)
		meta := Metadata{
			"cat":   NewInt(int64(i % 5)),
			"score": NewFloat(float64(i) / 400),
			// Geo rides the same writeValue/readValue codec as snapshot/WAL, so a
			// geo field here proves geo metadata is durable through persist too.
			"loc": NewGeo(float64(i%90), float64(i%180)),
		}
		if _, _, err := h.Insert(uint64(i+1), v, 0, meta, nil, nil, CASCond{}); err != nil {
			t.Fatal(err)
		}
	}
	q := vecs[0]
	flt := Filter{Op: FilterAnd, And: []Filter{
		{Op: FilterEq, Field: "cat", Value: NewInt(2)},
		{Op: FilterGte, Field: "score", Value: NewFloat(0.5)},
	}}
	before, err := h.SearchFiltered(q, k, flt)
	if err != nil {
		t.Fatal(err)
	}
	if len(before) == 0 {
		t.Fatal("pre-save filtered search returned nothing; test can't distinguish")
	}

	metaPath := filepath.Join(dir, "meta.bin")
	if err := h.SavePersist(metaPath); err != nil {
		t.Fatalf("SavePersist with metadata: %v", err)
	}
	_ = h.Close()

	h2, err := openPersist(cfg, metaPath)
	if err != nil {
		t.Fatalf("openPersist: %v", err)
	}
	defer func() { _ = h2.Close() }()

	after, err := h2.SearchFiltered(q, k, flt)
	if err != nil {
		t.Fatal(err)
	}
	if !eqUint64(resultIDs(before), resultIDs(after)) {
		t.Errorf("filtered search after restart: %v != %v", resultIDs(after), resultIDs(before))
	}

	// Geo metadata must survive persist exactly (durable through writeValue/readValue).
	slot, ok := h2.arena.Slot(1)
	if !ok {
		t.Fatal("id 1 missing after persist reopen")
	}
	gv := h2.arena.Metadata(slot)["loc"]
	if gv.Kind != ValueGeo || gv.Lat != 0 || gv.Lon != 0 {
		t.Errorf("reopened geo for id 1 = %+v, want kind=geo lat=0 lon=0", gv)
	}
}

// TestPersistMmapGrowReopen exercises the full Windows-critical round-trip on the
// VECTOR and GRAPH mmap slabs together: build an mmap-backed collection with
// serial inserts that cross the initial 1024-slot reserve on BOTH slabs (forcing
// at least one growVecMmap each — the unmap+truncate+remap grow path that only the
// windows CI lane now actually runs), sync+close via SavePersist, then reopen from
// the same paths and assert the vectors and graph survived (ids present, count
// matches, search results identical). Small dim + fixed seed keep it deterministic
// and fast; the two file-size asserts prove a grow really happened (else the test
// would be a no-op for the grow path it exists to cover).
func TestPersistMmapGrowReopen(t *testing.T) {
	const dim, k = 16, 10
	// > mmapInitVectors (1024) so BOTH the vector slab (initial 1024*dim floats) and
	// the level-0 graph slab (initial 1024*m0 u32) grow at least once via growVecMmap.
	// At these sizes newBytes stays well under slabReserveThreshold, so growth takes
	// the mmap truncate+remap path (not a reservation) — exactly the Windows code.
	const n = 1500
	dir := t.TempDir()
	cfg := Config{
		Dim: dim, Metric: Cosine, M: 16, EfConstruction: 100, EfSearch: 64, Seed: 1,
		Quant: QuantSQ8, QuantStorage: QuantMmap, MmapPath: filepath.Join(dir, "vecs.dat"),
		RescoreFactor: 3, GraphMmapPath: filepath.Join(dir, "graph.dat"),
	}
	h, err := newHNSW(cfg)
	if err != nil {
		t.Fatal(err)
	}
	ids, vecs := siftLikeCorpus(n, dim, 4)
	for _, v := range vecs { // SQ8 is cosine-scope
		normalize(v)
	}
	// Serial Insert exercises the incremental remap-on-grow path (each insert can
	// extend both mmap regions), unlike BuildConcurrent's single pre-sizing grow.
	for i := range vecs {
		if _, _, err := h.Insert(ids[i], vecs[i], 0, nil, nil, nil, CASCond{}); err != nil {
			t.Fatalf("insert %d: %v", i, err)
		}
	}

	// Both slabs must have grown past their initial reserve — otherwise this test
	// never engaged growVecMmap and would silently stop covering it.
	initVec := int64(mmapInitVectors * dim * 4)
	initGraph := int64(mmapInitVectors * h.m0 * 4)
	if fi, err := os.Stat(cfg.MmapPath); err != nil {
		t.Fatalf("stat vec file: %v", err)
	} else if fi.Size() <= initVec {
		t.Fatalf("vec file = %d bytes, want > initial reserve %d (no growVecMmap happened)", fi.Size(), initVec)
	}
	if fi, err := os.Stat(cfg.GraphMmapPath); err != nil {
		t.Fatalf("stat graph file: %v", err)
	} else if fi.Size() <= initGraph {
		t.Fatalf("graph file = %d bytes, want > initial reserve %d (no growVecMmap happened)", fi.Size(), initGraph)
	}

	_, queries := siftLikeCorpus(40, dim, 9)
	for _, q := range queries {
		normalize(q)
	}
	before := make([][]uint64, len(queries))
	for i, q := range queries {
		res, err := h.Search(q, k)
		if err != nil {
			t.Fatal(err)
		}
		before[i] = resultIDs(res)
	}

	metaPath := filepath.Join(dir, "meta.bin")
	if err := h.SavePersist(metaPath); err != nil { // sync mmaps + write sidecar
		t.Fatalf("SavePersist: %v", err)
	}
	if err := h.Close(); err != nil {
		t.Fatalf("close: %v", err)
	}

	// Reopen by MAPPING the same files (no rebuild / no Insert loop).
	h2, err := openPersist(cfg, metaPath)
	if err != nil {
		t.Fatalf("openPersist: %v", err)
	}
	defer func() { _ = h2.Close() }()

	if got := h2.arena.Size(); got != n {
		t.Errorf("reopened size = %d, want %d", got, n)
	}
	for _, id := range ids {
		if _, ok := h2.arena.idMap[id]; !ok {
			t.Errorf("id %d missing after reopen", id)
			break
		}
	}
	for i, q := range queries {
		res, err := h2.Search(q, k)
		if err != nil {
			t.Fatal(err)
		}
		if !eqUint64(resultIDs(res), before[i]) {
			t.Errorf("query %d: reopened results %v != original %v", i, resultIDs(res), before[i])
			break
		}
	}
}

// TestPersistConfigMismatch checks OpenPersist rejects a Config whose
// dim/metric/M/quant differ from the saved index.
func TestPersistConfigMismatch(t *testing.T) {
	const n, dim = 500, 16
	dir := t.TempDir()
	h, cfg, _ := buildPersistableSIFT(t, dir, n, dim, 5)
	metaPath := filepath.Join(dir, "meta.bin")
	if err := h.SavePersist(metaPath); err != nil {
		t.Fatal(err)
	}
	_ = h.Close()

	bad := cfg
	bad.M = 32 // differs from saved M=16
	if _, err := openPersist(bad, metaPath); !errors.Is(err, ErrPersistMismatch) {
		t.Errorf("mismatched M openPersist = %v, want ErrPersistMismatch", err)
	}
}
