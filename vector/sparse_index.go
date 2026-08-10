// SPDX-License-Identifier: Apache-2.0

package vector

import "sort"

// posting is one entry in an inverted-index postings list: the slot that holds
// a sparse vector and the weight that vector assigns to the list's dimension.
type posting struct {
	slot  uint32
	value float32
}

// sparseIndex is an inverted index over sparse-vector dimensions. postings[dim]
// holds every (slot, value) that references dim. It is append-only on insert;
// deletions are lazy (the search path skips tombstoned/expired/filtered slots
// via an admit callback) and reclaimed by rebuild().
//
// Not safe for concurrent use; the embedding hnsw guards it with its RWMutex
// (write lock for add/rebuild, read lock for search).
type sparseIndex struct {
	postings map[uint32][]posting
}

func newSparseIndex() *sparseIndex {
	return &sparseIndex{postings: make(map[uint32][]posting)}
}

// add records sv's terms under slot. The sparse vector must be validated by
// the caller (sorted, unique, equal lengths).
func (si *sparseIndex) add(slot uint32, sv SparseVector) {
	for i, dim := range sv.Indices {
		si.postings[dim] = append(si.postings[dim], posting{slot: slot, value: sv.Values[i]})
	}
}

// remove deletes slot's postings for the dimensions of sv (its prior sparse
// vector). Used when a slot is hard-removed for replacement (Upsert) so stale
// postings don't point at the reused slot. O(nnz · avg postings per dim).
func (si *sparseIndex) remove(slot uint32, sv SparseVector) {
	for _, dim := range sv.Indices {
		lst := si.postings[dim]
		w := 0
		for _, p := range lst {
			if p.slot != slot {
				lst[w] = p
				w++
			}
		}
		if w == 0 {
			delete(si.postings, dim)
		} else {
			si.postings[dim] = lst[:w]
		}
	}
}

// search accumulates sparse dot-product scores for every slot whose sparse
// vector shares a dimension with query. admit (when non-nil) gates which slots
// may contribute — slots it rejects (tombstoned, expired, filtered) are
// skipped. Returns a slot→score map; the caller top-K's it.
func (si *sparseIndex) search(query SparseVector, admit func(slot uint32) bool) map[uint32]float32 {
	scores := make(map[uint32]float32)
	for i, dim := range query.Indices {
		qv := query.Values[i]
		for _, p := range si.postings[dim] {
			if admit != nil && !admit(p.slot) {
				continue
			}
			scores[p.slot] += qv * p.value
		}
	}
	return scores
}

// rebuild clears the index and re-adds every live (non-tombstoned) slot's
// sparse vector from the arena. Called on snapshot restore and after Reclaim,
// where stale postings would otherwise reference removed slots.
func (si *sparseIndex) rebuild(a *arena, tombstoned map[uint32]bool) {
	si.postings = make(map[uint32][]posting)
	for _, slot := range a.idMap {
		if tombstoned[slot] {
			continue
		}
		if sv := a.sparse[slot]; sv != nil {
			si.add(slot, *sv)
		}
	}
}

// topK reduces a slot→score map to the k highest-scoring slots, descending.
// Ties break by lower slot for determinism.
func topKSparse(scores map[uint32]float32, k int) []slotScore {
	if k <= 0 || len(scores) == 0 {
		return nil
	}
	all := make([]slotScore, 0, len(scores))
	for slot, sc := range scores {
		all = append(all, slotScore{slot: slot, score: sc})
	}
	sort.Slice(all, func(i, j int) bool {
		if all[i].score != all[j].score {
			return all[i].score > all[j].score
		}
		return all[i].slot < all[j].slot
	})
	if len(all) > k {
		all = all[:k]
	}
	return all
}

// slotScore pairs a slot with a sparse relevance score (higher = better).
type slotScore struct {
	slot  uint32
	score float32
}
