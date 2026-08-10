// SPDX-License-Identifier: Apache-2.0

package vector

import "sort"

// sparsePostingID is one entry in the id-keyed inverted index: the POINT ID that
// holds a sparse vector and the weight that vector assigns to the list's
// dimension. It is the id-keyed mirror of the slot-keyed posting (sparse_index.go)
// — the named family is id-keyed (no arena/slots), so postings reference point ids
// directly.
type sparsePostingID struct {
	id    uint64
	value float32
}

// sparseIndexID is an inverted index over sparse-vector dimensions, keyed by
// POINT ID (the id-keyed mirror of the dense slot-keyed sparseIndex in
// sparse_index.go). postings[dim] holds every (id, value) that references dim.
// add/remove keep it exact on upsert/delete (the named family has no tombstones —
// removal is eager). searchTopK accumulates dot-product scores over the shared
// dimensions, gated by an admit callback (TTL/filter), and returns the top-k.
//
// Not safe for concurrent use; the embedding NamedCollection guards it with nc.mu
// (write lock for add/remove/rebuild, read lock for searchTopK). The dense
// slot-keyed sparse index is UNTOUCHED — this is a standalone parallel structure.
type sparseIndexID struct {
	postings map[uint32][]sparsePostingID
}

func newSparseIndexID() *sparseIndexID {
	return &sparseIndexID{postings: make(map[uint32][]sparsePostingID)}
}

// add records v's terms under id. The sparse vector must be validated by the
// caller (sorted, unique, equal lengths). A nil/zero vector adds nothing.
func (si *sparseIndexID) add(id uint64, v *SparseVector) {
	if v == nil {
		return
	}
	for i, dim := range v.Indices {
		si.postings[dim] = append(si.postings[dim], sparsePostingID{id: id, value: v.Values[i]})
	}
}

// remove deletes every posting that references id, across all dimensions. Unlike
// the dense slot-keyed remove (which is handed the prior vector to know which
// dims to touch), the id-keyed remove does not require the prior vector: callers
// (delete / upsert) may not hold it, so it sweeps all postings lists. O(total
// postings); acceptable for the eager-removal named path. An empty list deletes
// the dim entry so a rebuild-free index stays compact.
func (si *sparseIndexID) remove(id uint64) {
	for dim, lst := range si.postings {
		w := 0
		for _, p := range lst {
			if p.id != id {
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

// searchTopK accumulates sparse dot-product scores for every id whose sparse
// vector shares a dimension with query, gated by admit (when non-nil; ids it
// rejects — expired/filtered — are skipped), and returns the k highest-scoring
// ids descending (ties broken by lower id for determinism). Score equals
// sparseDot(query, stored) over the shared dims. Returns nil for k<=0 or an empty
// query. This is the engine surface SearchNamedSparse and
// NamedHybrid call.
func (si *sparseIndexID) searchTopK(query *SparseVector, k int, admit func(id uint64) bool) []Result {
	if k <= 0 || query == nil || len(query.Indices) == 0 {
		return nil
	}
	scores := make(map[uint64]float32)
	for i, dim := range query.Indices {
		qv := query.Values[i]
		for _, p := range si.postings[dim] {
			if admit != nil && !admit(p.id) {
				continue
			}
			scores[p.id] += qv * p.value
		}
	}
	if len(scores) == 0 {
		return nil
	}
	all := make([]Result, 0, len(scores))
	for id, sc := range scores {
		all = append(all, Result{ID: id, Score: sc})
	}
	sort.Slice(all, func(i, j int) bool {
		if all[i].Score != all[j].Score {
			return all[i].Score > all[j].Score
		}
		return all[i].ID < all[j].ID
	})
	if len(all) > k {
		all = all[:k]
	}
	return all
}

// rebuild clears the index and re-adds every id's sparse vector from vecs. Called
// on snapshot restore (the index is never serialized — rebuilt from the stored
// vecs map, mirroring the dense sparseIndex.rebuild from the arena).
func (si *sparseIndexID) rebuild(vecs map[uint64]*SparseVector) {
	si.postings = make(map[uint32][]sparsePostingID)
	for id, v := range vecs {
		si.add(id, v)
	}
}
