// SPDX-License-Identifier: Apache-2.0

package vector

import "slices"

// sparseAccumulator is pooled scratch for the sparse-lane top-k scan. It
// replaces the per-query map[uint32]float32 with an epoch-stamped dense score
// array (the same trick as the dense visited-set): stamp[slot]==cur marks a
// slot scored this query, scores[slot] holds its running dot-product, and
// touched lists the candidate slots so top-k iterates only real candidates
// instead of the whole array. Reused across queries via layerScratch, so steady
// state allocates nothing here (array indexing also beats map hashing on the
// hot path). Owned by one goroutine for the duration of a search.
type sparseAccumulator struct {
	stamp   []uint32
	scores  []float32
	cur     uint32
	touched []uint32
	top     []slotScore
}

// prepare sizes the accumulator for n slots and starts a fresh epoch — clearing
// the previous query's scores in O(1) without touching memory, except on the
// 2^32 epoch wrap which pays one O(n) zeroing.
func (a *sparseAccumulator) prepare(n int) {
	if cap(a.stamp) < n {
		a.stamp = make([]uint32, n)
		a.scores = make([]float32, n)
		a.cur = 1
	} else {
		a.stamp = a.stamp[:n]
		a.scores = a.scores[:n]
		a.cur++
		if a.cur == 0 {
			for i := range a.stamp {
				a.stamp[i] = 0
			}
			a.cur = 1
		}
	}
	a.touched = a.touched[:0]
}

// add accumulates contrib into slot's score, recording slot as a candidate the
// first time it is scored this epoch.
func (a *sparseAccumulator) add(slot uint32, contrib float32) {
	if a.stamp[slot] != a.cur {
		a.stamp[slot] = a.cur
		a.scores[slot] = contrib
		a.touched = append(a.touched, slot)
	} else {
		a.scores[slot] += contrib
	}
}

// topK returns the k highest-scoring candidates, descending, ties broken by
// lower slot — identical to topKSparse over the equivalent score map. The
// returned slice aliases pooled scratch and is valid only until the next sparse
// scan on the same layerScratch; the caller consumes it immediately.
func (a *sparseAccumulator) topK(k int) []slotScore {
	if k <= 0 || len(a.touched) == 0 {
		return nil
	}
	buf := a.top[:0]
	for _, slot := range a.touched {
		buf = append(buf, slotScore{slot: slot, score: a.scores[slot]})
	}
	slices.SortFunc(buf, func(x, y slotScore) int {
		if x.score != y.score {
			if x.score > y.score {
				return -1
			}
			return 1
		}
		return cmpUint32(x.slot, y.slot)
	})
	a.top = buf
	if len(buf) > k {
		buf = buf[:k]
	}
	return buf
}

func cmpUint32(a, b uint32) int {
	switch {
	case a < b:
		return -1
	case a > b:
		return 1
	default:
		return 0
	}
}

// searchTopK accumulates sparse dot-product scores into pooled scratch and
// returns the top-k slots, gated by admit (nil = admit all). n is the arena
// slot capacity (sizes the score buffer). It is the allocation-free hot-path
// equivalent of topKSparse(si.search(query, admit), k), summing each slot's
// contributions in the same order so results are identical.
func (si *sparseIndex) searchTopK(s *layerScratch, n int, query SparseVector, k int, admit func(slot uint32) bool) []slotScore {
	if k <= 0 || len(query.Indices) == 0 {
		return nil
	}
	acc := &s.sparseAcc
	acc.prepare(n)
	for i, dim := range query.Indices {
		qv := query.Values[i]
		for _, p := range si.postings[dim] {
			if admit != nil && !admit(p.slot) {
				continue
			}
			acc.add(p.slot, qv*p.value)
		}
	}
	return acc.topK(k)
}
