// SPDX-License-Identifier: Apache-2.0

package vector

import (
	"sort"
	"testing"
)

// TestRecommend checks that recommendation steers toward the positive example's
// region and away from the negative's: with a positive in cluster A ([1,0]-ish)
// and a negative in cluster B ([0,1]-ish), the top-2 are the other cluster-A
// vectors, and the example ids themselves are excluded from the results.
func TestRecommend(t *testing.T) {
	h, err := newHNSW(Config{Dim: 2, M: 8, EfConstruction: 50, EfSearch: 50, Seed: 1, Metric: Cosine})
	if err != nil {
		t.Fatal(err)
	}
	insert := func(id uint64, v []float32) {
		if _, _, err := h.Insert(id, v, 0, nil, nil, nil, CASCond{}); err != nil {
			t.Fatal(err)
		}
	}
	insert(1, []float32{1, 0.05})   // cluster A
	insert(2, []float32{1, -0.05})  // cluster A
	insert(3, []float32{0.99, 0.1}) // cluster A
	insert(4, []float32{0.05, 1})   // cluster B
	insert(5, []float32{-0.05, 1})  // cluster B

	got, err := h.Recommend(2, RecommendOpts{Positive: []uint64{1}, Negative: []uint64{4}})
	if err != nil {
		t.Fatal(err)
	}
	ids := make([]uint64, len(got))
	for i, r := range got {
		ids[i] = r.ID
	}
	sort.Slice(ids, func(a, b int) bool { return ids[a] < ids[b] })
	if len(ids) != 2 || ids[0] != 2 || ids[1] != 3 {
		t.Errorf("Recommend = %v, want cluster-A ids [2 3] (example ids excluded)", ids)
	}
}

// TestRecommendRequiresPositive checks that recommendation with no positive
// examples is rejected.
func TestRecommendRequiresPositive(t *testing.T) {
	h, err := newHNSW(Config{Dim: 2, M: 8, EfConstruction: 50, EfSearch: 50, Seed: 1, Metric: Cosine})
	if err != nil {
		t.Fatal(err)
	}
	_, _, _ = h.Insert(1, []float32{1, 0}, 0, nil, nil, nil, CASCond{})
	if _, err := h.Recommend(2, RecommendOpts{Negative: []uint64{1}}); err == nil {
		t.Error("Recommend with no positive examples should return an error")
	}
}
