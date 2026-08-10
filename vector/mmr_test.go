// SPDX-License-Identifier: Apache-2.0

package vector

import "testing"

// TestSearchMMR checks the two ends of the lambda knob on a corpus with two
// near-duplicate vectors (ids 1, 2) close to the query and one diverse vector
// (id 3) further away:
//   - lambda=1 is pure relevance, so the top-2 are the two near-duplicates.
//   - lambda=0.3 weights diversity enough to drop the redundant near-duplicate
//     (2) in favor of the diverse vector (3). (The crossover is near lambda~0.49
//     for this corpus, since vector 3 is still ~0.71 similar to vector 1.)
func TestSearchMMR(t *testing.T) {
	h, err := newHNSW(Config{Dim: 2, M: 8, EfConstruction: 50, EfSearch: 50, Seed: 1, Metric: Cosine})
	if err != nil {
		t.Fatal(err)
	}
	insert := func(id uint64, v []float32) {
		if _, _, err := h.Insert(id, v, 0, nil, nil, nil, CASCond{}); err != nil {
			t.Fatal(err)
		}
	}
	insert(1, []float32{1, 0.01})  // closest to query
	insert(2, []float32{1, 0.02})  // near-duplicate of 1
	insert(3, []float32{0.7, 0.7}) // diverse, less relevant
	query := []float32{1, 0}

	rel, err := h.SearchMMR(query, 2, MMROpts{Lambda: 1, FetchK: 3})
	if err != nil {
		t.Fatal(err)
	}
	if len(rel) != 2 || rel[0].ID != 1 || rel[1].ID != 2 {
		t.Errorf("lambda=1: got ids [%d %d], want [1 2]", idAt(rel, 0), idAt(rel, 1))
	}

	div, err := h.SearchMMR(query, 2, MMROpts{Lambda: 0.3, FetchK: 3})
	if err != nil {
		t.Fatal(err)
	}
	if len(div) != 2 || div[0].ID != 1 || div[1].ID != 3 {
		t.Errorf("lambda=0.3: got ids [%d %d], want [1 3]", idAt(div, 0), idAt(div, 1))
	}
}

func idAt(rs []Result, i int) uint64 {
	if i < len(rs) {
		return rs[i].ID
	}
	return 0
}
