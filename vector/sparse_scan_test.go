// SPDX-License-Identifier: Apache-2.0

package vector

import (
	"fmt"
	"testing"
)

// sameTopK asserts two ranked sparse results are identical in length, order,
// slot, and score. The pooled scan (searchTopK) accumulates each slot's
// contributions in the same order as the exhaustive map scan, so this is an
// exact comparison — no score epsilon.
func sameTopK(t *testing.T, ctx string, got, want []slotScore) {
	t.Helper()
	if len(got) != len(want) {
		t.Fatalf("%s: len got %d want %d\n got=%v\nwant=%v", ctx, len(got), len(want), got, want)
	}
	for i := range want {
		if got[i].slot != want[i].slot || got[i].score != want[i].score {
			t.Fatalf("%s: rank %d: got {slot %d score %v} want {slot %d score %v}",
				ctx, i, got[i].slot, got[i].score, want[i].slot, want[i].score)
		}
	}
}

// TestSearchTopKMatchesExhaustive is the core property test: over random
// corpora and queries, the pooled scan (searchTopK) must return exactly the
// ranking the exhaustive map scan (topKSparse(search(...))) returns, for every k.
func TestSearchTopKMatchesExhaustive(t *testing.T) {
	const n, nnz, vocab, nq = 2000, 20, 4000, 40
	for _, seed := range []int64{1, 7, 42, 99, 1234} {
		corpus := makeSparseCorpus(n, nnz, vocab, seed)
		queries := makeSparseCorpus(nq, nnz, vocab, seed*31+5)
		si := newSparseIndex()
		for slot, sv := range corpus {
			si.add(uint32(slot), sv)
		}
		var s layerScratch
		for _, k := range []int{1, 3, 10, 50, 200, n + 100} {
			for qi, q := range queries {
				want := topKSparse(si.search(q, nil), k)
				got := si.searchTopK(&s, n, q, k, nil)
				sameTopK(t, fmt.Sprintf("seed=%d k=%d q=%d", seed, k, qi), got, want)
			}
		}
	}
}

// TestSearchTopKAdmitGating verifies the pooled scan honors the admit predicate
// (tombstone / TTL / filter gating) and still matches the exhaustive scan.
func TestSearchTopKAdmitGating(t *testing.T) {
	const n, nnz, vocab = 1500, 20, 3000
	corpus := makeSparseCorpus(n, nnz, vocab, 3)
	si := newSparseIndex()
	for slot, sv := range corpus {
		si.add(uint32(slot), sv)
	}
	admit := func(slot uint32) bool { return slot%3 == 0 } // keep every third slot
	queries := makeSparseCorpus(20, nnz, vocab, 77)
	var s layerScratch
	for _, k := range []int{1, 10, 50} {
		for qi, q := range queries {
			want := topKSparse(si.search(q, admit), k)
			got := si.searchTopK(&s, n, q, k, admit)
			sameTopK(t, fmt.Sprintf("admit k=%d q=%d", k, qi), got, want)
		}
	}
}

// TestSearchTopKUnorderedAdds checks that slots added out of order (free-list
// reuse: a deleted slot reused by a later insert) still scan correctly — the
// pooled accumulator is order-independent.
func TestSearchTopKUnorderedAdds(t *testing.T) {
	si := newSparseIndex()
	si.add(5, SparseVector{Indices: []uint32{1, 2}, Values: []float32{1, 1}})
	si.add(2, SparseVector{Indices: []uint32{1, 2}, Values: []float32{3, 1}})
	si.add(9, SparseVector{Indices: []uint32{1}, Values: []float32{2}})
	si.add(1, SparseVector{Indices: []uint32{2}, Values: []float32{5}})
	si.add(7, SparseVector{Indices: []uint32{1, 2}, Values: []float32{0.5, 4}})

	q := SparseVector{Indices: []uint32{1, 2}, Values: []float32{1, 1}}
	var s layerScratch
	for _, k := range []int{1, 2, 5, 100} {
		want := topKSparse(si.search(q, nil), k)
		got := si.searchTopK(&s, 10, q, k, nil)
		sameTopK(t, fmt.Sprintf("unordered k=%d", k), got, want)
	}
}

// TestSearchTopKNegativeWeights verifies the pooled scan handles signed weights
// natively (no upper-bound assumption to break) and still matches the oracle.
func TestSearchTopKNegativeWeights(t *testing.T) {
	si := newSparseIndex()
	si.add(0, SparseVector{Indices: []uint32{1, 2}, Values: []float32{1, -2}})
	si.add(1, SparseVector{Indices: []uint32{1, 2}, Values: []float32{-1, 3}})
	si.add(2, SparseVector{Indices: []uint32{2}, Values: []float32{4}})
	var s layerScratch

	q := SparseVector{Indices: []uint32{1, 2}, Values: []float32{2, -1}}
	for _, k := range []int{1, 2, 3} {
		want := topKSparse(si.search(q, nil), k)
		got := si.searchTopK(&s, 3, q, k, nil)
		sameTopK(t, fmt.Sprintf("neg k=%d", k), got, want)
	}
}

// TestSearchTopKEdge covers degenerate inputs.
func TestSearchTopKEdge(t *testing.T) {
	si := newSparseIndex()
	si.add(0, SparseVector{Indices: []uint32{1}, Values: []float32{1}})
	var s layerScratch

	if got := si.searchTopK(&s, 1, SparseVector{Indices: []uint32{1}, Values: []float32{1}}, 0, nil); got != nil {
		t.Errorf("k=0 should be nil, got %v", got)
	}
	if got := si.searchTopK(&s, 1, SparseVector{}, 5, nil); got != nil {
		t.Errorf("empty query should be nil, got %v", got)
	}
	if got := si.searchTopK(&s, 1, SparseVector{Indices: []uint32{999}, Values: []float32{1}}, 5, nil); len(got) != 0 {
		t.Errorf("absent term should yield no results, got %v", got)
	}
}
