// SPDX-License-Identifier: Apache-2.0

package vector

import (
	"sort"
	"testing"
)

// TestDiscover checks that context pairs constrain the result region. The
// target [0.7,0.7] sits between cluster A ([1,0]-ish: ids 1,2,3) and cluster B
// ([0,1]-ish: ids 4,5,6), so a plain search would mix both. A single context
// pair (positive in A, negative in B) steers the results to cluster A.
func TestDiscover(t *testing.T) {
	h, err := newHNSW(Config{Dim: 2, M: 8, EfConstruction: 50, EfSearch: 50, Seed: 1, Metric: Cosine})
	if err != nil {
		t.Fatal(err)
	}
	insert := func(id uint64, v []float32) {
		if _, _, err := h.Insert(id, v, 0, nil, nil, nil, CASCond{}); err != nil {
			t.Fatal(err)
		}
	}
	insert(1, []float32{1, 0.02})
	insert(2, []float32{1, -0.02})
	insert(3, []float32{0.98, 0.05})
	insert(4, []float32{0.02, 1})
	insert(5, []float32{-0.02, 1})
	insert(6, []float32{0.05, 0.98})

	got, err := h.Discover(3, DiscoverOpts{
		Target:  []float32{0.7, 0.7},
		Context: []ContextPair{{Positive: 1, Negative: 4}},
	})
	if err != nil {
		t.Fatal(err)
	}
	ids := make([]uint64, len(got))
	for i, r := range got {
		ids[i] = r.ID
	}
	sort.Slice(ids, func(a, b int) bool { return ids[a] < ids[b] })
	if len(ids) != 3 || ids[0] != 1 || ids[1] != 2 || ids[2] != 3 {
		t.Errorf("Discover = %v, want cluster-A ids [1 2 3]", ids)
	}
}

// TestDiscoverRequiresContext checks that discovery with no context pairs is
// rejected.
func TestDiscoverRequiresContext(t *testing.T) {
	h, err := newHNSW(Config{Dim: 2, M: 8, EfConstruction: 50, EfSearch: 50, Seed: 1, Metric: Cosine})
	if err != nil {
		t.Fatal(err)
	}
	_, _, _ = h.Insert(1, []float32{1, 0}, 0, nil, nil, nil, CASCond{})
	if _, err := h.Discover(3, DiscoverOpts{Target: []float32{1, 0}}); err == nil {
		t.Error("Discover with no context pairs should return an error")
	}
}
