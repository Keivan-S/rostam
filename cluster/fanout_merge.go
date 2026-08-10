// SPDX-License-Identifier: Apache-2.0

package cluster

import "slices"

// Scored is the minimal merge unit shared by the exact-merge search modes: an id
// plus its distance (dense/filtered) and/or score (multivector). The caller's
// "less" comparator decides ordering, so the same merger serves all three.
type Scored struct {
	ID    uint64
	Dist  float32
	Score float32
}

// MergeTopK merges per-partition result slices into the global top-k under less.
// Each input slice may be in any order; the result is sorted by less and capped
// at k (k<0 means "no cap"). Exact for partition-local scores: the global top-k
// is a subset of the union of per-partition results, since each partition
// returns its own top-k (>= k results or all it has). Ties under less are
// broken arbitrarily (the sort is not stable).
func MergeTopK(parts [][]Scored, k int, less func(a, b Scored) bool) []Scored {
	cmp := func(a, b Scored) int {
		if less(a, b) {
			return -1
		}
		if less(b, a) {
			return 1
		}
		return 0
	}
	if k < 0 {
		// No cap: concat everything and full-sort (rare admin path).
		var all []Scored
		for _, p := range parts {
			all = append(all, p...)
		}
		slices.SortFunc(all, cmp)
		return all
	}
	if k == 0 {
		return nil
	}
	// Bounded top-k via a max-heap of size <= k (root = the largest kept so far):
	// scan every candidate once, keeping the k smallest under less. This is
	// O(N log k) versus the old concat + full-sort O(N log N) — N grows with the
	// partition count P (~P*k), so the merge no longer scales super-linearly in P.
	// Ties at the k-boundary are broken arbitrarily, exactly as the prior unstable
	// sort documented.
	h := make([]Scored, 0, k)
	for _, p := range parts {
		for _, s := range p {
			if len(h) < k {
				h = append(h, s)
				maxHeapUp(h, len(h)-1, less)
			} else if less(s, h[0]) { // s is smaller than the current largest kept
				h[0] = s
				maxHeapDown(h, less)
			}
		}
	}
	slices.SortFunc(h, cmp) // ascending under less for the returned order
	return h
}

// maxHeapUp / maxHeapDown maintain a binary MAX-heap over h under less (the root
// is the element no other is less than). Used by MergeTopK to keep the k smallest
// candidates: the root is the largest of those k, so a new candidate smaller than
// the root evicts it.
func maxHeapUp(h []Scored, i int, less func(a, b Scored) bool) {
	for i > 0 {
		parent := (i - 1) / 2
		if !less(h[parent], h[i]) { // parent already >= child
			break
		}
		h[parent], h[i] = h[i], h[parent]
		i = parent
	}
}

func maxHeapDown(h []Scored, less func(a, b Scored) bool) {
	n, i := len(h), 0
	for {
		largest, l, r := i, 2*i+1, 2*i+2
		if l < n && less(h[largest], h[l]) {
			largest = l
		}
		if r < n && less(h[largest], h[r]) {
			largest = r
		}
		if largest == i {
			break
		}
		h[i], h[largest] = h[largest], h[i]
		i = largest
	}
}
