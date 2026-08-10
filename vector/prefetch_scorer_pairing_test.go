// SPDX-License-Identifier: Apache-2.0

package vector

import (
	"math/rand"
	"testing"
	"unsafe"
)

// prefetchTarget names the arena array a slotPrefetcher warms, so a test can
// assert the pairing without reaching into raw pointers at every call site.
type prefetchTarget string

const (
	targetNone prefetchTarget = "none"
	targetVecs prefetchTarget = "vecs"
	targetCode prefetchTarget = "codes"
)

// classify reports which arena array p points at.
func classify(h *hnsw, p slotPrefetcher) prefetchTarget {
	a := h.arena
	if p.maxSlot == 0 || p.base == nil {
		return targetNone
	}
	if len(a.vecs) > 0 && p.base == unsafe.Pointer(&a.vecs[0]) {
		return targetVecs
	}
	if len(a.codes) > 0 && p.base == unsafe.Pointer(&a.codes[0]) {
		return targetCode
	}
	return targetNone
}

// TestScorerAndPrefetcherAgreeOnPQ is the regression pin for a prefetcher that
// warmed the wrong array. The prefetch target used to be chosen by
// `h.quant != nil`, but nothing on the build path reads codes for a PQ index:
// quantizedBuild() returns false for every PQ and PRQ quantizer, so the link
// scorer reads raw float32 vectors. The prefetcher therefore pulled in a single
// line of PQ codes that were never touched, on every neighbor, while evicting
// vector lines the distance kernel was about to read.
//
// The two are now produced together by layerScorer, and this test states the
// contract each path must satisfy: SEARCH on a trained PQ index reads codes,
// BUILD on the same index reads vectors.
func TestScorerAndPrefetcherAgreeOnPQ(t *testing.T) {
	const dim, n, threshold = 32, 400, 300
	h, err := newHNSW(Config{
		Dim: dim, Metric: L2, M: 8, EfConstruction: 50, EfSearch: 32, Seed: 1,
		Quant: QuantPQ, RescoreFactor: 3, IVFTrainThreshold: threshold,
	})
	if err != nil {
		t.Fatal(err)
	}
	defer h.Close()

	rng := rand.New(rand.NewSource(9))
	vecs := make([][]float32, n)
	for i := range vecs {
		v := make([]float32, dim)
		// Cluster the corpus so PQ training produces meaningful codebooks.
		c := float32(i % 8)
		for j := range v {
			v[j] = c + float32(rng.NormFloat64())*0.1
		}
		vecs[i] = v
		if _, _, err := h.Insert(uint64(i+1), v, 0, nil, nil, nil, CASCond{}); err != nil {
			t.Fatal(err)
		}
	}

	if h.quant == nil {
		t.Fatal("fixture: expected a PQ quantizer to be configured")
	}
	if h.pqUntrained() {
		t.Fatalf("fixture: PQ index did not auto-train after %d inserts (threshold %d)", n, threshold)
	}
	// The premise of the bug: quant is non-nil, yet the build path is exact.
	if h.quantizedBuild() {
		t.Fatal("fixture: quantizedBuild() must be false for a PQ index")
	}

	s := &layerScratch{}

	// SEARCH on a trained PQ index navigates on ADC over codes.
	if got := classify(h, h.searchScorer(vecs[0]).pf); got != targetCode {
		t.Errorf("searchScorer prefetch target = %s, want %s "+
			"(trained PQ search reads arena.Code)", got, targetCode)
	}
	// BUILD on the SAME index scores raw floats — the case the old
	// `h.quant != nil` predicate got wrong.
	if got := classify(h, h.buildScorer(s, vecs[0]).pf); got != targetVecs {
		t.Errorf("buildScorer prefetch target = %s, want %s "+
			"(quantizedBuild() is false for PQ, so the link scorer reads arena.Vec)", got, targetVecs)
	}
	// And the stride must match the record actually being read, not just the array.
	if pf := h.buildScorer(s, vecs[0]).pf; pf.stride != uintptr(dim*4) {
		t.Errorf("buildScorer prefetch stride = %d, want %d (dim*4 bytes)", pf.stride, dim*4)
	}
	if pf := h.searchScorer(vecs[0]).pf; pf.stride != uintptr(h.arena.codeLen) {
		t.Errorf("searchScorer prefetch stride = %d, want codeLen %d", pf.stride, h.arena.codeLen)
	}
}

// TestScorerAndPrefetcherAgreeUnquantized pins the plain path: with no
// quantizer at all, both build and search score raw vectors.
func TestScorerAndPrefetcherAgreeUnquantized(t *testing.T) {
	const dim, n = 16, 100
	h, err := newHNSW(Config{Dim: dim, Metric: L2, M: 8, EfConstruction: 50, EfSearch: 32, Seed: 1})
	if err != nil {
		t.Fatal(err)
	}
	defer h.Close()

	rng := rand.New(rand.NewSource(4))
	var first []float32
	for i := 0; i < n; i++ {
		v := make([]float32, dim)
		for j := range v {
			v[j] = float32(rng.NormFloat64())
		}
		if first == nil {
			first = v
		}
		if _, _, err := h.Insert(uint64(i+1), v, 0, nil, nil, nil, CASCond{}); err != nil {
			t.Fatal(err)
		}
	}

	s := &layerScratch{}
	if got := classify(h, h.searchScorer(first).pf); got != targetVecs {
		t.Errorf("unquantized searchScorer prefetch target = %s, want %s", got, targetVecs)
	}
	if got := classify(h, h.buildScorer(s, first).pf); got != targetVecs {
		t.Errorf("unquantized buildScorer prefetch target = %s, want %s", got, targetVecs)
	}
}

// TestUntrainedQuantScorerPrefetchesVectors covers the other divergence the old
// predicate produced: an UNTRAINED quantizer has non-nil h.quant but placeholder
// codes, so searchScorer falls back to exact float32 — and the prefetcher must
// follow it to the vectors rather than warming meaningless code bytes.
func TestUntrainedQuantScorerPrefetchesVectors(t *testing.T) {
	const dim, n, threshold = 32, 50, 10_000 // threshold far above n: never trains
	h, err := newHNSW(Config{
		Dim: dim, Metric: L2, M: 8, EfConstruction: 50, EfSearch: 32, Seed: 1,
		Quant: QuantPQ, RescoreFactor: 3, IVFTrainThreshold: threshold,
	})
	if err != nil {
		t.Fatal(err)
	}
	defer h.Close()

	rng := rand.New(rand.NewSource(6))
	var first []float32
	for i := 0; i < n; i++ {
		v := make([]float32, dim)
		for j := range v {
			v[j] = float32(rng.NormFloat64())
		}
		if first == nil {
			first = v
		}
		if _, _, err := h.Insert(uint64(i+1), v, 0, nil, nil, nil, CASCond{}); err != nil {
			t.Fatal(err)
		}
	}
	if !h.pqUntrained() {
		t.Fatalf("fixture: PQ index trained unexpectedly at n=%d (threshold %d)", n, threshold)
	}
	if h.quant == nil {
		t.Fatal("fixture: expected a (non-nil) untrained PQ quantizer")
	}
	if got := classify(h, h.searchScorer(first).pf); got != targetVecs {
		t.Errorf("untrained-PQ searchScorer prefetch target = %s, want %s "+
			"(untrained codes are placeholders, so the scorer reads arena.Vec)", got, targetVecs)
	}
}
