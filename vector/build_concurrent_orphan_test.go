// SPDX-License-Identifier: Apache-2.0

package vector

import (
	"math"
	"testing"
)

// Regression guard for the concurrent-build orphan bug: linkOneNode's per-level
// writeNbrsFromDist used to OVERWRITE a node's neighbor list, destroying
// back-edges another worker had appended before it wrote its own forward edges
// (frontier carry-down — a multi-level node reachable at level lc+1 is used as a
// direct searchLayer entry point at lc before it writes there). That silently
// left nodes with zero in-edges (unreachable → unfindable), only under parallel
// scheduling. The fix preserves and re-applies those back-edges, so a concurrent
// build must have the SAME connectivity guarantee as serial: no node is ever
// orphaned. TestBuildConcurrentMatchesSerial's 3% general-recall tolerance is too
// loose to catch this; this test asserts the exact invariant.

// denseArcCorpus places n vectors on a quarter-circle arc — adjacent points only
// ~1.28° apart. This near-collinear packing has thin in-degree, so a single lost
// back-edge orphans a node; it made the bug reproduce reliably.
func denseArcCorpus(n int) ([]uint64, [][]float32) {
	ids := make([]uint64, n)
	vecs := make([][]float32, n)
	for i := 0; i < n; i++ {
		theta := float64(i) * (math.Pi / 2 / 40)
		ids[i] = uint64(i)
		vecs[i] = []float32{float32(math.Cos(theta)), float32(math.Sin(theta)), 0, 0}
	}
	return ids, vecs
}

// arcInDegreeAndReach returns each slot's level-0 in-degree and whether it is
// reachable from the entry point via level-0 out-edges (BFS).
func arcInDegreeAndReach(h *hnsw) (inDeg []int, reachable []bool) {
	n := len(h.nodes)
	inDeg = make([]int, n)
	reachable = make([]bool, n)
	for slot := 0; slot < n; slot++ {
		nd := h.nodes[slot]
		if nd == nil {
			continue
		}
		for _, nb := range h.nbrsAt(nd, 0) {
			inDeg[nb]++
		}
	}
	stack := []uint32{h.entryPoint}
	reachable[h.entryPoint] = true
	for len(stack) > 0 {
		s := stack[len(stack)-1]
		stack = stack[:len(stack)-1]
		nd := h.nodes[s]
		if nd == nil {
			continue
		}
		for _, nb := range h.nbrsAt(nd, 0) {
			if !reachable[nb] {
				reachable[nb] = true
				stack = append(stack, nb)
			}
		}
	}
	return inDeg, reachable
}

// TestBuildConcurrentNoOrphans asserts BuildConcurrent never leaves a node with
// zero in-edges or unreachable, at every worker count, matching serial. Before
// the linkOneNode preserve-back-edges fix this failed at workers >= 2 (dozens of
// zero-in-degree nodes over these builds); serial and workers=1 always passed.
func TestBuildConcurrentNoOrphans(t *testing.T) {
	const (
		n     = 40
		iters = 200
	)
	ids, vecs := denseArcCorpus(n)
	for _, w := range []int{1, 2, 4, 8} {
		orphans, unreach := 0, 0
		for it := 0; it < iters; it++ {
			h, err := newHNSW(Config{Dim: 4, Metric: Cosine, M: 16, EfConstruction: 200, EfSearch: 64, Seed: int64(it + 1)})
			if err != nil {
				t.Fatal(err)
			}
			if err := h.BuildConcurrent(ids, vecs, w); err != nil {
				t.Fatal(err)
			}
			inDeg, reach := arcInDegreeAndReach(h)
			for slot, d := range inDeg {
				if h.nodes[slot] == nil {
					continue
				}
				if d == 0 {
					orphans++
				}
				if !reach[slot] {
					unreach++
				}
			}
		}
		if orphans != 0 || unreach != 0 {
			t.Errorf("workers=%d: concurrent build orphaned nodes over %d builds: zero-in-degree=%d unreachable=%d (want 0/0, serial guarantee)", w, iters, orphans, unreach)
		}
	}
}
