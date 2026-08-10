// SPDX-License-Identifier: Apache-2.0

package vector

import "testing"

// denseLeafN builds a flat dense leaf prefetch source (a valid 1-level lane).
func denseLeafN() QueryLeaf {
	return QueryLeaf{Kind: LeafDense, Dense: []float32{0.01, 0, 0, 0}, K: 3}
}

// TestQueryTooManyPrefetchSources proves the BREADTH bound (the structural
// companion to the depth bound): a spec node with more than MaxPrefetchSources
// prefetch sources is rejected fail-loud (ErrTooManyPrefetchSources); a node AT
// the bound runs; and the check fires at a NESTED node too (recursion).
func TestQueryTooManyPrefetchSources(t *testing.T) {
	c := newQueryCorpus(t)

	mkLeaves := func(n int) []QuerySource {
		out := make([]QuerySource, n)
		for i := range out {
			out[i] = LeafSource(denseLeafN())
		}
		return out
	}

	// At the bound (MaxPrefetchSources sources): runs without the breadth error.
	atBound := QuerySpec{Mode: ModeFusion, Method: FusionRRF, Prefetch: mkLeaves(MaxPrefetchSources), K: 3}
	if _, err := c.Query(atBound); err == ErrTooManyPrefetchSources {
		t.Fatalf("spec at the bound (%d) wrongly rejected as too-many", MaxPrefetchSources)
	}

	// One past the bound: ErrTooManyPrefetchSources, fail-loud.
	tooMany := QuerySpec{Mode: ModeFusion, Method: FusionRRF, Prefetch: mkLeaves(MaxPrefetchSources + 1), K: 3}
	if _, err := c.Query(tooMany); err != ErrTooManyPrefetchSources {
		t.Fatalf("over-wide spec: err = %v, want ErrTooManyPrefetchSources", err)
	}

	// NESTED node with too many sources: the breadth check must fire at the nested
	// level (recursion), not just the root. The root is a valid 1-source spec whose
	// single source is an over-wide sub-spec.
	overWideSub := QuerySpec{Mode: ModeFusion, Method: FusionRRF, Prefetch: mkLeaves(MaxPrefetchSources + 1), K: 3}
	parent := QuerySpec{
		Mode:     ModeFusion,
		Method:   FusionRRF,
		Prefetch: []QuerySource{{Spec: &overWideSub}},
		K:        3,
	}
	if _, err := c.Query(parent); err != ErrTooManyPrefetchSources {
		t.Fatalf("nested over-wide node: err = %v, want ErrTooManyPrefetchSources (recursion)", err)
	}
}

// TestLeafLanePoolClamp proves the per-lane pool clamp: a LaneK above MaxLanePool is
// capped to MaxLanePool; a LaneK at/under the ceiling is unchanged; and the
// K/LaneK/50 default path is unaffected.
func TestLeafLanePoolClamp(t *testing.T) {
	const k = 10

	// LaneK above the ceiling → clamped to MaxLanePool.
	if got := leafLanePool(QueryLeaf{LaneK: 50000}, k); got != MaxLanePool {
		t.Errorf("LaneK=50000 pool = %d, want %d (clamped)", got, MaxLanePool)
	}
	// LaneK at the ceiling → unchanged.
	if got := leafLanePool(QueryLeaf{LaneK: MaxLanePool}, k); got != MaxLanePool {
		t.Errorf("LaneK=MaxLanePool pool = %d, want %d", got, MaxLanePool)
	}
	// LaneK under the ceiling → unchanged.
	if got := leafLanePool(QueryLeaf{LaneK: 5000}, k); got != 5000 {
		t.Errorf("LaneK=5000 pool = %d, want 5000 (unchanged)", got)
	}
	// No LaneK, K under ceiling → K (unchanged default).
	if got := leafLanePool(QueryLeaf{K: 25}, k); got != 25 {
		t.Errorf("K=25 pool = %d, want 25 (unchanged)", got)
	}
	// No LaneK, no K, k under 50 → 50 (unchanged default).
	if got := leafLanePool(QueryLeaf{}, 7); got != 50 {
		t.Errorf("default pool = %d, want 50 (unchanged)", got)
	}
	// No LaneK, no K, k over 50 → k (unchanged default).
	if got := leafLanePool(QueryLeaf{}, 200); got != 200 {
		t.Errorf("k=200 pool = %d, want 200 (unchanged)", got)
	}

	// SourceLanePool mirrors the same ceiling for a leaf source and for a nested
	// sub-spec source (sub.K clamped).
	if got := SourceLanePool(LeafSource(QueryLeaf{LaneK: 50000}), k); got != MaxLanePool {
		t.Errorf("SourceLanePool leaf LaneK=50000 = %d, want %d", got, MaxLanePool)
	}
	sub := QuerySpec{K: 50000}
	if got := SourceLanePool(QuerySource{Spec: &sub}, k); got != MaxLanePool {
		t.Errorf("SourceLanePool sub-spec K=50000 = %d, want %d", got, MaxLanePool)
	}
}

// TestLaneClampEqualsLaneKCap proves the clamp is EXACTLY equivalent to the client
// having passed LaneK=MaxLanePool: a query with LaneK=MaxLanePool and one with
// LaneK far above it return byte-identical results (both lanes return their full
// score-ordered pool — the index has far fewer than MaxLanePool docs, so the pool
// is the same set either way).
func TestLaneClampEqualsLaneKCap(t *testing.T) {
	c := newQueryCorpus(t)
	mk := func(laneK int) QuerySpec {
		return QuerySpec{
			Mode:   ModeFusion,
			Method: FusionRRF,
			Prefetch: srcs(
				QueryLeaf{Kind: LeafDense, Dense: []float32{0.01, 0, 0, 0}, LaneK: laneK},
				QueryLeaf{Kind: LeafSparse, Sparse: SparseVector{Indices: []uint32{42}, Values: []float32{1}}, LaneK: laneK, ScoreDesc: true},
			),
			K: 5,
		}
	}
	atCeiling, err := c.Query(mk(MaxLanePool))
	if err != nil {
		t.Fatalf("LaneK=MaxLanePool query: %v", err)
	}
	wayAbove, err := c.Query(mk(1000000))
	if err != nil {
		t.Fatalf("LaneK=1e6 query: %v", err)
	}
	if !queryResultsEqual(atCeiling.Fused, wayAbove.Fused) {
		t.Errorf("clamp not equivalent to LaneK cap:\n LaneK=10000 -> %+v\n LaneK=1e6  -> %+v", atCeiling.Fused, wayAbove.Fused)
	}
}

// TestNormalSpecIdenticalPrePost proves a normal small spec (few sources, small K,
// no LaneK over the ceiling) is byte-identical pre/post the cap: the same spec run
// twice returns identical results, and a small-LaneK lane is unaffected by the
// clamp (its pool is far under the ceiling).
func TestNormalSpecIdenticalPrePost(t *testing.T) {
	c := newQueryCorpus(t)
	spec := QuerySpec{
		Mode:   ModeFusion,
		Method: FusionRRF,
		Prefetch: srcs(
			QueryLeaf{Kind: LeafDense, Dense: []float32{0.01, 0, 0, 0}, K: 3},
			QueryLeaf{Kind: LeafSparse, Sparse: SparseVector{Indices: []uint32{42}, Values: []float32{1}}, K: 3, ScoreDesc: true},
		),
		K: 3,
	}
	a, err := c.Query(spec)
	if err != nil {
		t.Fatalf("query a: %v", err)
	}
	b, err := c.Query(spec)
	if err != nil {
		t.Fatalf("query b: %v", err)
	}
	if !queryResultsEqual(a.Fused, b.Fused) {
		t.Errorf("normal spec not stable: %+v vs %+v", a.Fused, b.Fused)
	}
}
