// SPDX-License-Identifier: Apache-2.0

package vector

import (
	"bytes"
	"errors"
	"os"
	"path/filepath"
	"runtime"
	"testing"
)

// TestGraphMmapMatchesHeapSerial builds the same corpus serially into a
// heap-backed graph and an mmap-backed graph (Config.GraphMmapPath) and asserts
// recall is IDENTICAL — the algorithm is unchanged, only where the level-0 slab
// lives differs. Serial Insert also exercises the incremental remap-on-grow path
// (each insert can extend the mmap region).
func TestGraphMmapMatchesHeapSerial(t *testing.T) {
	const (
		n, dim, k = 2000, 16, 10
	)
	ids, vecs := siftLikeCorpus(n, dim, 3)
	_, queries := siftLikeCorpus(100, dim, 4)
	cfg := Config{Dim: dim, Metric: L2, M: 16, EfConstruction: 100, EfSearch: 64, Seed: 42}

	heap, err := newHNSW(cfg)
	if err != nil {
		t.Fatal(err)
	}
	for i := range vecs {
		if _, _, err := heap.Insert(ids[i], vecs[i], 0, nil, nil, nil, CASCond{}); err != nil {
			t.Fatal(err)
		}
	}

	mcfg := cfg
	path := filepath.Join(t.TempDir(), "graph.dat")
	mcfg.GraphMmapPath = path
	mm, err := newHNSW(mcfg)
	if err != nil {
		t.Fatal(err)
	}
	defer func() { _ = mm.Close() }()
	for i := range vecs {
		if _, _, err := mm.Insert(ids[i], vecs[i], 0, nil, nil, nil, CASCond{}); err != nil {
			t.Fatalf("mmap insert %d: %v", i, err)
		}
	}

	if fi, err := os.Stat(path); err != nil {
		t.Fatalf("stat graph file: %v", err)
	} else if fi.Size() < int64(n*mcfg.M*2*4) { // >= n * m0 * 4 bytes (m0 = 2*M)
		t.Errorf("graph file = %d bytes, want >= %d", fi.Size(), n*mcfg.M*2*4)
	}

	rh := recallOf(t, heap, vecs, queries, k)
	rm := recallOf(t, mm, vecs, queries, k)
	t.Logf("recall@%d heap=%.4f graph-mmap=%.4f", k, rh, rm)
	if rh != rm {
		t.Errorf("graph-mmap recall %.4f != heap %.4f — must be identical (only storage differs)", rm, rh)
	}
}

// TestGraphMmapConcurrentBuild checks the bulk-build path with an mmap-backed
// graph (pre-sized slab, parallel link phase) is race-clean and matches a
// heap-backed concurrent build within the usual parallel-build noise.
func TestGraphMmapConcurrentBuild(t *testing.T) {
	const n, dim, k = 4000, 24, 10
	ids, vecs := siftLikeCorpus(n, dim, 9)
	_, queries := siftLikeCorpus(120, dim, 5)
	cfg := Config{Dim: dim, Metric: L2, M: 16, EfConstruction: 200, EfSearch: 64, Seed: 1}

	heap, err := newHNSW(cfg)
	if err != nil {
		t.Fatal(err)
	}
	if err := heap.BuildConcurrent(ids, vecs, runtime.GOMAXPROCS(0)); err != nil {
		t.Fatal(err)
	}

	mcfg := cfg
	mcfg.GraphMmapPath = filepath.Join(t.TempDir(), "graph.dat")
	mm, err := newHNSW(mcfg)
	if err != nil {
		t.Fatal(err)
	}
	defer func() { _ = mm.Close() }()
	if err := mm.BuildConcurrent(ids, vecs, runtime.GOMAXPROCS(0)); err != nil {
		t.Fatalf("BuildConcurrent (graph-mmap): %v", err)
	}

	rh := recallOf(t, heap, vecs, queries, k)
	rm := recallOf(t, mm, vecs, queries, k)
	t.Logf("recall@%d heap=%.4f graph-mmap=%.4f", k, rh, rm)
	if diff := rh - rm; diff > 0.03 || diff < -0.03 {
		t.Errorf("graph-mmap concurrent recall %.4f differs from heap %.4f by >0.03", rm, rh)
	}
}

// TestGraphMmapSnapshotRoundtrip verifies an mmap-backed graph snapshots and
// restores into another mmap-backed graph, with search results preserved.
func TestGraphMmapSnapshotRoundtrip(t *testing.T) {
	const n, dim, k = 1500, 12, 10
	ids, vecs := siftLikeCorpus(n, dim, 2)
	_, queries := siftLikeCorpus(50, dim, 6)
	cfg := Config{Dim: dim, Metric: L2, M: 16, EfConstruction: 100, EfSearch: 64, Seed: 7}

	src, err := newHNSW(withGraphMmap(cfg, filepath.Join(t.TempDir(), "src.dat")))
	if err != nil {
		t.Fatal(err)
	}
	defer func() { _ = src.Close() }()
	for i := range vecs {
		if _, _, err := src.Insert(ids[i], vecs[i], 0, nil, nil, nil, CASCond{}); err != nil {
			t.Fatal(err)
		}
	}
	var buf bytes.Buffer
	if err := src.Snapshot(&buf); err != nil {
		t.Fatal(err)
	}

	dst, err := newHNSW(withGraphMmap(cfg, filepath.Join(t.TempDir(), "dst.dat")))
	if err != nil {
		t.Fatal(err)
	}
	defer func() { _ = dst.Close() }()
	if err := dst.Restore(&buf); err != nil {
		t.Fatalf("restore into graph-mmap: %v", err)
	}

	for _, q := range queries {
		a, _ := src.Search(q, k)
		b, _ := dst.Search(q, k)
		if !eqUint64(resultIDs(a), resultIDs(b)) {
			t.Errorf("restored graph-mmap results differ: src=%v dst=%v", resultIDs(a), resultIDs(b))
			break
		}
	}
}

// TestGraphMmapWithQuant exercises graph-mmap alongside QuantSQ8+QuantMmap — the
// vectors and the graph's level-0 slab BOTH live in (separate) mmap files, only
// the int8 codes + small structures on the heap.
func TestGraphMmapWithQuant(t *testing.T) {
	const n, dim, k = 2000, 32, 10
	ids, vecs := siftLikeCorpus(n, dim, 11)
	_, queries := siftLikeCorpus(60, dim, 12)
	for _, v := range vecs { // SQ8 is cosine-scope
		normalize(v)
	}
	for _, q := range queries {
		normalize(q)
	}
	dir := t.TempDir()
	cfg := Config{
		Dim: dim, Metric: Cosine, M: 16, EfConstruction: 200, EfSearch: 64, Seed: 3,
		Quant: QuantSQ8, QuantStorage: QuantMmap, MmapPath: filepath.Join(dir, "vecs.dat"),
		RescoreFactor: 3, GraphMmapPath: filepath.Join(dir, "graph.dat"),
	}
	h, err := newHNSW(cfg)
	if err != nil {
		t.Fatal(err)
	}
	defer func() { _ = h.Close() }()
	if err := h.BuildConcurrent(ids, vecs, runtime.GOMAXPROCS(0)); err != nil {
		t.Fatalf("build (quant+graph mmap): %v", err)
	}
	for _, p := range []string{cfg.MmapPath, cfg.GraphMmapPath} {
		if fi, err := os.Stat(p); err != nil || fi.Size() == 0 {
			t.Errorf("backing file %s missing/empty (err=%v)", p, err)
		}
	}
	res, err := h.Search(queries[0], k)
	if err != nil {
		t.Fatal(err)
	}
	if len(res) == 0 {
		t.Error("search over quant+graph-mmap returned no results")
	}
}

// TestGraphMmapValidation rejects a GraphMmapPath equal to MmapPath (they must
// be distinct files).
func TestGraphMmapValidation(t *testing.T) {
	_, err := newHNSW(Config{
		Dim: 4, Metric: L2, M: 8, EfConstruction: 10, EfSearch: 10,
		Quant: QuantSQ8, QuantStorage: QuantMmap, MmapPath: "/tmp/x.dat", GraphMmapPath: "/tmp/x.dat",
	})
	if !errors.Is(err, ErrInvalidGraphMmap) {
		t.Errorf("want ErrInvalidGraphMmap for equal paths, got %v", err)
	}
}

func withGraphMmap(cfg Config, path string) Config {
	cfg.GraphMmapPath = path
	return cfg
}
