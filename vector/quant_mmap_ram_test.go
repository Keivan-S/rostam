// SPDX-License-Identifier: Apache-2.0

package vector

import (
	"fmt"
	"math/rand"
	"os"
	"path/filepath"
	"runtime"
	"runtime/debug"
	"strconv"
	"testing"
)

// heapInuseMB returns Go heap in-use after a GC, in MiB.
func heapInuseMB() float64 {
	runtime.GC()
	var m runtime.MemStats
	runtime.ReadMemStats(&m)
	return float64(m.HeapInuse) / (1 << 20)
}

// rssMB reads resident set size from /proc/self/statm (Linux), in MiB. Returns
// -1 if unavailable.
func rssMB() float64 {
	data, err := os.ReadFile("/proc/self/statm")
	if err != nil {
		return -1
	}
	var size, resident int64
	if _, err := fmt.Sscanf(string(data), "%d %d", &size, &resident); err != nil {
		return -1
	}
	return float64(resident*int64(os.Getpagesize())) / (1 << 20)
}

// TestSQ8MmapRAM measures the resident-memory win of QuantMmap: with the float32
// vectors moved to a memory-mapped file, only the int8 codes stay on the Go
// heap. Compares QuantNone (float32 in heap) vs QuantSQ8+QuantMmap on the same
// corpus, reporting Go heap in-use, process RSS, and the backing-file size.
// Opt-in: ROSTAM_RAMTEST=1. Corpus size via ROSTAM_RAMTEST_N (default 500k;
// ~1 GB heap at 500k, ~2 GB at 1M). Both indexes build concurrently.
func TestSQ8MmapRAM(t *testing.T) {
	if os.Getenv("ROSTAM_RAMTEST") != "1" {
		t.Skip("set ROSTAM_RAMTEST=1 to run")
	}
	const dim = 128
	n := 500_000 // override with ROSTAM_RAMTEST_N (e.g. 1000000)
	if v := os.Getenv("ROSTAM_RAMTEST_N"); v != "" {
		if parsed, err := strconv.Atoi(v); err == nil && parsed > 0 {
			n = parsed
		}
	}
	rng := rand.New(rand.NewSource(1))
	ids := make([]uint64, n)
	vecs := make([][]float32, n)
	for i := range vecs {
		v := make([]float32, dim)
		for j := range v {
			v[j] = float32(rng.NormFloat64())
		}
		normalize(v)
		ids[i] = uint64(i + 1)
		vecs[i] = v
	}
	floatBytesMB := float64(n*dim*4) / (1 << 20)
	codeBytesMB := float64(n*dim) / (1 << 20)
	fmt.Fprintf(os.Stderr, "[ram] %d vecs x %d dim: float32=%.0fMB, sq8 codes=%.0fMB\n", n, dim, floatBytesMB, codeBytesMB)

	// --- QuantNone: float32 lives on the Go heap ---
	none, err := newHNSW(Config{Dim: dim, Metric: Cosine, M: 16, EfConstruction: 200, EfSearch: 64, Seed: 1})
	if err != nil {
		t.Fatal(err)
	}
	if err := none.BuildConcurrent(ids, vecs, runtime.GOMAXPROCS(0)); err != nil {
		t.Fatal(err)
	}
	noneHeap, noneRSS := heapInuseMB(), rssMB()
	fmt.Fprintf(os.Stderr, "[ram] QuantNone (f32 in heap):       heap=%.0fMB  rss=%.0fMB\n", noneHeap, noneRSS)
	_ = none.Close()
	none = nil
	runtime.GC()

	// --- QuantSQ8 + QuantMmap: float32 in an mmap file, codes in heap ---
	path := filepath.Join(t.TempDir(), "vecs.dat")
	mm, err := newHNSW(Config{
		Dim: dim, Metric: Cosine, M: 16, EfConstruction: 200, EfSearch: 64, Seed: 1,
		Quant: QuantSQ8, QuantStorage: QuantMmap, MmapPath: path, RescoreFactor: 3,
	})
	if err != nil {
		t.Fatal(err)
	}
	if err := mm.BuildConcurrent(ids, vecs, runtime.GOMAXPROCS(0)); err != nil {
		t.Fatalf("mmap build: %v", err)
	}
	mmHeap, mmRSS := heapInuseMB(), rssMB()
	fi, _ := os.Stat(path)
	fileMB := float64(fi.Size()) / (1 << 20)
	fmt.Fprintf(os.Stderr, "[ram] QuantMmap+SQ8 (codes in heap):  heap=%.0fMB  rss=%.0fMB  mmap_file=%.0fMB\n", mmHeap, mmRSS, fileMB)
	fmt.Fprintf(os.Stderr, "[ram] => Go heap saved: %.0fMB (%.1fx less for vector storage); float32 (%.0fMB) is on disk, OS-reclaimable\n",
		noneHeap-mmHeap, noneHeap/mmHeap, fileMB)
	_ = mm.Close()

	// The float32 vectors must be off the Go heap, so QuantMmap's heap is well
	// below QuantNone's (which holds the full float32 set).
	if mmHeap >= noneHeap {
		t.Errorf("QuantMmap heap %.0fMB not below QuantNone heap %.0fMB — float32 not off-heap?", mmHeap, noneHeap)
	}
}

// TestMemCompareSIFT measures Rostam's index-resident memory on the real SIFT-1M
// corpus the same way bench/sift1m/hnswlib_bench.py does — build the index, drop
// the input vectors (the index keeps its own copy), force the runtime to return
// free pages to the OS, then read heap-in-use and process RSS. This makes the
// number apples-to-apples with the hnswlib "index RSS" figure (TestSQ8MmapRAM,
// by contrast, holds the input array resident because it reuses it for a second
// build, so its RSS includes ~512 MB of input that hnswlib frees).
//
// Reports two Rostam configs against the same corpus: QuantNone (float32 in
// heap, L2 — matches hnswlib's index) and QuantSQ8+QuantMmap (int8 codes in
// heap, float32 in an mmap file, Cosine). Opt-in: ROSTAM_SIFT1M=1 with the
// dataset at /tmp/rostam-sift1m/sift/ (override dir via ROSTAM_SIFT_DIR).
func TestMemCompareSIFT(t *testing.T) {
	if os.Getenv("ROSTAM_SIFT1M") != "1" {
		t.Skip("set ROSTAM_SIFT1M=1 with dataset at /tmp/rostam-sift1m/sift/ to run")
	}
	dir := "/tmp/rostam-sift1m/sift"
	if d := os.Getenv("ROSTAM_SIFT_DIR"); d != "" {
		dir = d
	}
	basePath := filepath.Join(dir, "sift_base.fvecs")

	// rssBaseline: Go runtime + test binary, before any corpus is loaded. We
	// report index RSS = rss - baseline so it lines up with hnswlib's harness,
	// which subtracts its interpreter+arrays baseline.
	debug.FreeOSMemory()
	rssBaseline := rssMB()

	measure := func(label string, cfg Config) {
		base, err := readFvecs(basePath)
		if err != nil {
			t.Fatalf("%s: read base: %v", label, err)
		}
		n := len(base)
		dim := len(base[0])
		ids := make([]uint64, n)
		for i := range ids {
			ids[i] = uint64(i + 1)
		}
		h, err := newHNSW(cfg)
		if err != nil {
			t.Fatalf("%s: newHNSW: %v", label, err)
		}
		if err := h.BuildConcurrent(ids, base, runtime.GOMAXPROCS(0)); err != nil {
			t.Fatalf("%s: build: %v", label, err)
		}
		// Drop the inputs (the index holds its own arena copy), then return free
		// pages to the OS so RSS reflects only what the index keeps resident.
		base = nil //nolint:ineffassign // drop reference so FreeOSMemory can reclaim the inputs before we measure RSS
		ids = nil  //nolint:ineffassign // drop reference so FreeOSMemory can reclaim the inputs before we measure RSS
		debug.FreeOSMemory()
		heap := heapInuseMB()
		rss := rssMB()
		fileMB := 0.0
		if cfg.MmapPath != "" {
			if fi, err := os.Stat(cfg.MmapPath); err == nil {
				fileMB = float64(fi.Size()) / (1 << 20)
			}
		}
		fmt.Fprintf(os.Stderr,
			"[memcmp] %-22s n=%d dim=%d  heap=%.0fMB  rss=%.0fMB  index_rss=%.0fMB  mmap_file=%.0fMB\n",
			label, n, dim, heap, rss, rss-rssBaseline, fileMB)
		_ = h.Close()
		h = nil
		debug.FreeOSMemory()
	}

	measure("QuantNone (f32, L2)", Config{
		Dim: 128, Metric: L2, M: 16, EfConstruction: 200, EfSearch: 64, Seed: 1,
	})
	measure("QuantSQ8+Mmap (Cosine)", Config{
		Dim: 128, Metric: Cosine, M: 16, EfConstruction: 200, EfSearch: 64, Seed: 1,
		Quant: QuantSQ8, QuantStorage: QuantMmap,
		MmapPath: filepath.Join(t.TempDir(), "sift.dat"), RescoreFactor: 3,
	})
	measure("QuantSQ8+Mmap+GraphMmap", Config{
		Dim: 128, Metric: Cosine, M: 16, EfConstruction: 200, EfSearch: 64, Seed: 1,
		Quant: QuantSQ8, QuantStorage: QuantMmap,
		MmapPath:      filepath.Join(t.TempDir(), "sift2.dat"),
		RescoreFactor: 3,
		GraphMmapPath: filepath.Join(t.TempDir(), "graph.dat"),
	})
	fmt.Fprintf(os.Stderr, "[memcmp] (rss baseline = %.0fMB; index_rss subtracts it to match hnswlib harness)\n", rssBaseline)
}

// TestMemBreakdownSIFT attributes Rostam's float32 (QuantNone) heap to each
// structure by nulling them one at a time and watching HeapInuse drop after a
// full GC + scavenge. This is a *measured* breakdown (not an estimate) of why
// the index is heavier than hnswlib's packed layout. Opt-in: ROSTAM_SIFT1M=1.
func TestMemBreakdownSIFT(t *testing.T) {
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
	n := len(base)
	ids := make([]uint64, n)
	for i := range ids {
		ids[i] = uint64(i + 1)
	}
	h, err := newHNSW(Config{Dim: 128, Metric: L2, M: 16, EfConstruction: 200, EfSearch: 64, Seed: 1})
	if err != nil {
		t.Fatal(err)
	}
	if err := h.BuildConcurrent(ids, base, runtime.GOMAXPROCS(0)); err != nil {
		t.Fatal(err)
	}
	base = nil //nolint:ineffassign // drop reference so the inputs can be reclaimed before we measure memory
	ids = nil  //nolint:ineffassign // drop reference so the inputs can be reclaimed before we measure memory

	// Count graph edges before we free anything: level-0 edges live in the flat
	// slab, upper levels in per-node slices.
	var nodes, level0Edges, upperEdges, upperSlices int
	for _, nd := range h.nodes {
		if nd == nil {
			continue
		}
		nodes++
		level0Edges += h.nbrLen(nd, 0)
		for lc := 1; lc <= nd.level; lc++ {
			upperSlices++
			upperEdges += len(nd.upper[lc-1])
		}
	}

	drop := func(label string, free func()) float64 {
		before := heapInuseMB()
		free()
		debug.FreeOSMemory()
		after := heapInuseMB()
		d := before - after
		fmt.Fprintf(os.Stderr, "[breakdown] %-26s -%.0fMB  (heap now %.0fMB)\n", label, d, after)
		return d
	}

	debug.FreeOSMemory()
	total := heapInuseMB()
	fmt.Fprintf(os.Stderr, "[breakdown] === QuantNone float32, n=%d, M=16 (CSR level-0 slab) ===\n", n)
	fmt.Fprintf(os.Stderr, "[breakdown] total heap=%.0fMB; nodes=%d level0_edges=%d upper_edges=%d upper_slices=%d\n",
		total, nodes, level0Edges, upperEdges, upperSlices)
	fmt.Fprintf(os.Stderr, "[breakdown] level0 slab (n*2M*4B)=%.0fMB; level0Len (n*2B)=%.0fMB; node structs (nodes*48B)=%.0fMB\n",
		float64(len(h.level0)*4)/(1<<20), float64(len(h.level0Len)*2)/(1<<20), float64(nodes*48)/(1<<20))

	drop("arena.vecs (float32)", func() { h.arena.vecs = nil })
	drop("idMap", func() { h.arena.idMap = nil })
	drop("arena side-arrays", func() {
		h.arena.ids, h.arena.expires, h.arena.metadata, h.arena.sparse, h.arena.free = nil, nil, nil, nil, nil
	})
	drop("graph level-0 slab", func() { h.level0, h.level0Len = nil, nil })
	drop("graph nodes+upper", func() { h.nodes = nil })
	runtime.KeepAlive(h)
	debug.FreeOSMemory()
	fmt.Fprintf(os.Stderr, "[breakdown] remainder (runtime+misc) heap=%.0fMB\n", heapInuseMB())
}
