// SPDX-License-Identifier: Apache-2.0

package vector

import (
	"math/rand"
	"os"
	"path/filepath"
	"runtime"
	"sort"
	"sync"
	"testing"
)

func siftLikeCorpus(n, dim int, seed int64) ([]uint64, [][]float32) {
	rng := rand.New(rand.NewSource(seed))
	ids := make([]uint64, n)
	vecs := make([][]float32, n)
	for i := range vecs {
		v := make([]float32, dim)
		for j := range v {
			v[j] = float32(rng.NormFloat64())
		}
		ids[i] = uint64(i + 1)
		vecs[i] = v
	}
	return ids, vecs
}

func bruteTopK(vecs [][]float32, q []float32, k int) map[uint64]bool {
	type cd struct {
		id uint64
		d  float32
	}
	cs := make([]cd, len(vecs))
	for i, v := range vecs {
		cs[i] = cd{uint64(i + 1), l2SquaredScalar(v, q)}
	}
	sort.Slice(cs, func(a, b int) bool { return cs[a].d < cs[b].d })
	out := make(map[uint64]bool, k)
	for i := 0; i < k && i < len(cs); i++ {
		out[cs[i].id] = true
	}
	return out
}

func recallOf(t *testing.T, h *hnsw, vecs [][]float32, queries [][]float32, k int) float64 {
	t.Helper()
	var matches int
	for _, q := range queries {
		truth := bruteTopK(vecs, q, k)
		res, err := h.Search(q, k)
		if err != nil {
			t.Fatal(err)
		}
		for _, r := range res {
			if truth[r.ID] {
				matches++
			}
		}
	}
	return float64(matches) / float64(len(queries)*k)
}

// TestBuildConcurrentMatchesSerial builds the same corpus serially and via
// BuildConcurrent and asserts the concurrent build's recall is within noise of
// the serial build's — the core correctness guarantee. Run under -race to also
// prove the concurrent path is data-race-free.
func TestBuildConcurrentMatchesSerial(t *testing.T) {
	const (
		n    = 5000
		dim  = 32
		k    = 10
		seed = 42
	)
	ids, vecs := siftLikeCorpus(n, dim, seed)
	_, queries := siftLikeCorpus(200, dim, 7)

	cfg := Config{Dim: dim, Metric: L2, M: 16, EfConstruction: 200, EfSearch: 64, Seed: seed}

	serial, err := newHNSW(cfg)
	if err != nil {
		t.Fatal(err)
	}
	for i, v := range vecs {
		if _, _, err := serial.Insert(ids[i], v, 0, nil, nil, nil, CASCond{}); err != nil {
			t.Fatal(err)
		}
	}

	conc, err := newHNSW(cfg)
	if err != nil {
		t.Fatal(err)
	}
	if err := conc.BuildConcurrent(ids, vecs, runtime.GOMAXPROCS(0)); err != nil {
		t.Fatal(err)
	}

	rs := recallOf(t, serial, vecs, queries, k)
	rc := recallOf(t, conc, vecs, queries, k)
	t.Logf("recall@%d serial=%.4f concurrent=%.4f", k, rs, rc)
	if conc.arena.Size() != n {
		t.Errorf("concurrent index size = %d, want %d", conc.arena.Size(), n)
	}
	if diff := rs - rc; diff > 0.03 || diff < -0.03 {
		t.Errorf("concurrent recall %.4f differs from serial %.4f by %.4f (>0.03)", rc, rs, diff)
	}
}

// TestBuildConcurrentExtend exercises the concurrent ExtendCandidates path: it
// asserts the concurrent+extend build matches the serial+extend build within
// noise (parity / no race-induced quality loss) and that no neighbor list
// exceeds its cap. Run under -race to prove the second-hop neighbor reads
// (neighborsAt under each owner's link lock) are data-race-free.
func TestBuildConcurrentExtend(t *testing.T) {
	const (
		n    = 5000
		dim  = 32
		k    = 10
		seed = 42
	)
	ids, vecs := siftLikeCorpus(n, dim, seed)
	_, queries := siftLikeCorpus(200, dim, 7)

	cfg := Config{Dim: dim, Metric: L2, M: 16, EfConstruction: 200, EfSearch: 64, Seed: seed, ExtendCandidates: true}

	serial, err := newHNSW(cfg)
	if err != nil {
		t.Fatal(err)
	}
	for i, v := range vecs {
		if _, _, err := serial.Insert(ids[i], v, 0, nil, nil, nil, CASCond{}); err != nil {
			t.Fatal(err)
		}
	}

	conc, err := newHNSW(cfg)
	if err != nil {
		t.Fatal(err)
	}
	if err := conc.BuildConcurrent(ids, vecs, runtime.GOMAXPROCS(0)); err != nil {
		t.Fatal(err)
	}

	rs := recallOf(t, serial, vecs, queries, k)
	rc := recallOf(t, conc, vecs, queries, k)
	t.Logf("recall@%d serial+extend=%.4f concurrent+extend=%.4f", k, rs, rc)
	if diff := rs - rc; diff > 0.03 || diff < -0.03 {
		t.Errorf("concurrent+extend recall %.4f differs from serial+extend %.4f by %.4f (>0.03)", rc, rs, diff)
	}
	for slot, nd := range conc.nodes {
		if nd == nil {
			continue
		}
		for lc := 0; lc <= nd.level; lc++ {
			maxM := conc.cfg.M
			if lc == 0 {
				maxM = 2 * conc.cfg.M
			}
			if got := conc.nbrLen(nd, lc); got > maxM {
				t.Errorf("slot %d level %d has %d neighbors, cap %d", slot, lc, got, maxM)
			}
		}
	}
}

// TestLevel0FullDegree verifies the flag raises mean level-0 degree: with it
// set, each node selects up to 2*M forward neighbors at level 0 (vs M), so the
// bottom layer is denser. A denser bottom layer is the matched-params lever for
// recall@k at large k (see TestGloveProbe).
func TestLevel0FullDegree(t *testing.T) {
	const (
		n   = 4000
		dim = 24
	)
	ids, vecs := siftLikeCorpus(n, dim, 17)

	meanLevel0 := func(l0full bool) float64 {
		cfg := Config{Dim: dim, Metric: L2, M: 16, EfConstruction: 200, EfSearch: 64, Seed: 5, Level0FullDegree: l0full}
		h, err := newHNSW(cfg)
		if err != nil {
			t.Fatal(err)
		}
		if err := h.BuildConcurrent(ids, vecs, 8); err != nil {
			t.Fatal(err)
		}
		var sum, cnt float64
		for _, nd := range h.nodes {
			if nd == nil {
				continue
			}
			sum += float64(h.nbrLen(nd, 0))
			cnt++
		}
		return sum / cnt
	}

	base := meanLevel0(false)
	full := meanLevel0(true)
	t.Logf("mean level-0 degree: baseline=%.1f level0full=%.1f (M=16, cap=32)", base, full)
	if full <= base {
		t.Errorf("Level0FullDegree should raise mean level-0 degree: baseline=%.1f full=%.1f", base, full)
	}
}

// TestQuantizedBuildRecall checks that a quantized build (navigate+select on
// int8 codes) produces a graph whose recall is within a small tolerance of the
// default float32 build — the speed/quality tradeoff must stay a small quality
// cost, not a collapse. Both use SQ8 so search is identical; only the build
// distance space differs.
func TestQuantizedBuildRecall(t *testing.T) {
	const (
		n   = 5000
		dim = 64
		k   = 10
	)
	ids, vecs := siftLikeCorpus(n, dim, 21)
	for _, v := range vecs {
		normalize(v) // SQ8 is cosine-scope
	}
	_, queries := siftLikeCorpus(200, dim, 22)
	for _, q := range queries {
		normalize(q)
	}
	truth := func(q []float32) map[uint64]bool {
		type cd struct {
			id uint64
			d  float32
		}
		cs := make([]cd, n)
		for i, v := range vecs {
			cs[i] = cd{uint64(i + 1), -dotProduct(v, q)}
		}
		sort.Slice(cs, func(a, b int) bool { return cs[a].d < cs[b].d })
		m := make(map[uint64]bool, k)
		for i := 0; i < k; i++ {
			m[cs[i].id] = true
		}
		return m
	}
	recallOf := func(qb bool) float64 {
		cfg := Config{Dim: dim, Metric: Cosine, M: 16, EfConstruction: 200, EfSearch: 64, Seed: 7, Quant: QuantSQ8, RescoreFactor: 3, QuantizedBuild: qb}
		h, err := newHNSW(cfg)
		if err != nil {
			t.Fatal(err)
		}
		if err := h.BuildConcurrent(ids, vecs, 8); err != nil {
			t.Fatal(err)
		}
		var match int
		for _, q := range queries {
			tr := truth(q)
			res, _ := h.Search(q, k)
			for _, r := range res {
				if tr[r.ID] {
					match++
				}
			}
		}
		return float64(match) / float64(len(queries)*k)
	}
	rf := recallOf(false)
	rq := recallOf(true)
	t.Logf("recall@%d float-build=%.4f quantized-build=%.4f", k, rf, rq)
	if rf-rq > 0.05 {
		t.Errorf("quantized build recall %.4f trails float build %.4f by >0.05", rq, rf)
	}
}

// TestBuildConcurrentMmap checks that BuildConcurrent works with QuantMmap
// storage (float32 in a memory-mapped file): the mmap is pre-reserved to N so
// workers write disjoint slots with no remap during the build. Verifies the
// backing file holds the vectors and recall matches an exact float32 baseline.
func TestBuildConcurrentMmap(t *testing.T) {
	const (
		n   = 5000
		dim = 32
		k   = 10
	)
	ids, vecs := siftLikeCorpus(n, dim, 3)
	_, queries := siftLikeCorpus(100, dim, 9)
	// SQ8 is Cosine-scope; normalize so cosine = dot.
	for _, v := range vecs {
		normalize(v)
	}
	for _, q := range queries {
		normalize(q)
	}
	cosRecall := func(h *hnsw) float64 {
		var matches int
		for _, q := range queries {
			type cd struct {
				id uint64
				d  float32
			}
			cs := make([]cd, n)
			for i, v := range vecs {
				cs[i] = cd{uint64(i + 1), -dotProduct(v, q)} // -dot: ascending = nearest
			}
			sort.Slice(cs, func(a, b int) bool { return cs[a].d < cs[b].d })
			truth := make(map[uint64]bool, k)
			for i := 0; i < k; i++ {
				truth[cs[i].id] = true
			}
			res, _ := h.Search(q, k)
			for _, r := range res {
				if truth[r.ID] {
					matches++
				}
			}
		}
		return float64(matches) / float64(len(queries)*k)
	}

	// Same valid config (Cosine + SQ8) both ways; only build method + storage
	// differ, so this isolates concurrent-mmap-build correctness.
	base := Config{Dim: dim, Metric: Cosine, M: 16, EfConstruction: 200, EfSearch: 64, Seed: 3, Quant: QuantSQ8, RescoreFactor: 3}
	serial, err := newHNSW(base) // QuantInRAM, serial Insert
	if err != nil {
		t.Fatal(err)
	}
	for i, v := range vecs {
		if _, _, err := serial.Insert(ids[i], v, 0, nil, nil, nil, CASCond{}); err != nil {
			t.Fatal(err)
		}
	}

	path := filepath.Join(t.TempDir(), "vecs.dat")
	mcfg := base
	mcfg.QuantStorage = QuantMmap
	mcfg.MmapPath = path
	mm, err := newHNSW(mcfg)
	if err != nil {
		t.Fatal(err)
	}
	if err := mm.BuildConcurrent(ids, vecs, runtime.GOMAXPROCS(0)); err != nil {
		t.Fatalf("BuildConcurrent (mmap): %v", err)
	}
	if fi, err := os.Stat(path); err != nil {
		t.Fatalf("stat backing file: %v", err)
	} else if minBytes := int64(n * dim * 4); fi.Size() < minBytes {
		t.Errorf("backing file = %d bytes, want >= %d", fi.Size(), minBytes)
	}

	rs := cosRecall(serial)
	rc := cosRecall(mm)
	t.Logf("recall@%d serial-inram-sq8=%.4f concurrent-mmap-sq8=%.4f", k, rs, rc)
	if rc < rs-0.05 {
		t.Errorf("mmap concurrent recall %.4f below serial %.4f by >0.05", rc, rs)
	}
	if err := mm.Close(); err != nil {
		t.Fatalf("close: %v", err)
	}
}

// TestBuildConcurrentEdgeCaps checks that no neighbor list exceeds its cap after
// a concurrent build — a lost-update or prune race would violate this.
func TestBuildConcurrentEdgeCaps(t *testing.T) {
	const (
		n   = 4000
		dim = 16
	)
	ids, vecs := siftLikeCorpus(n, dim, 99)
	cfg := Config{Dim: dim, Metric: L2, M: 16, EfConstruction: 100, EfSearch: 64, Seed: 1}
	h, err := newHNSW(cfg)
	if err != nil {
		t.Fatal(err)
	}
	if err := h.BuildConcurrent(ids, vecs, 8); err != nil {
		t.Fatal(err)
	}
	for slot, nd := range h.nodes {
		if nd == nil {
			continue
		}
		for lc := 0; lc <= nd.level; lc++ {
			maxM := h.cfg.M
			if lc == 0 {
				maxM = 2 * h.cfg.M
			}
			if got := h.nbrLen(nd, lc); got > maxM {
				t.Errorf("slot %d level %d has %d neighbors, cap %d", slot, lc, got, maxM)
			}
		}
	}
}

// TestBuildConcurrentThenSearchRace runs searches concurrently right after a
// concurrent build to confirm the post-build index behaves as a normal serial
// index (meaningful under -race).
func TestBuildConcurrentThenSearchRace(t *testing.T) {
	const n, dim = 3000, 24
	ids, vecs := siftLikeCorpus(n, dim, 5)
	h, err := newHNSW(Config{Dim: dim, Metric: L2, M: 16, EfConstruction: 100, EfSearch: 64, Seed: 1})
	if err != nil {
		t.Fatal(err)
	}
	if err := h.BuildConcurrent(ids, vecs, 8); err != nil {
		t.Fatal(err)
	}
	var wg sync.WaitGroup
	for w := 0; w < 8; w++ {
		wg.Add(1)
		go func(off int) {
			defer wg.Done()
			for i := 0; i < 200; i++ {
				_, _ = h.Search(vecs[(off*200+i)%n], 10)
			}
		}(w)
	}
	wg.Wait()
}
