// SPDX-License-Identifier: Apache-2.0

package vector

import "testing"

func TestStatsFilterRejects(t *testing.T) {
	// FilterFirstThreshold:1 forces the equality filter onto the graph-traversal
	// path (its candidate set exceeds 1), where FilterRejects is metered. With
	// the default threshold this selective eq filter would take the exact
	// filter-first path, which matches every candidate and rejects none.
	h, err := newHNSW(Config{Dim: 4, Metric: L2, M: 16, EfConstruction: 200, EfSearch: 64, Seed: 1, FilterFirstThreshold: 1})
	if err != nil {
		t.Fatalf("newHNSW: %v", err)
	}
	// Insert 50 vectors, only 5 with bucket="hit".
	for i := 1; i <= 50; i++ {
		v := []float32{float32(i), float32(i % 7), 0, 0}
		var meta Metadata
		if i%10 == 0 {
			meta = Metadata{"bucket": NewString("hit")}
		} else {
			meta = Metadata{"bucket": NewString("miss")}
		}
		if _, _, err := h.Insert(uint64(i), v, 0, meta, nil, nil, CASCond{}); err != nil {
			t.Fatalf("Insert %d: %v", i, err)
		}
	}

	before := h.Stats().FilterRejects
	filter := Filter{Op: FilterEq, Field: "bucket", Value: NewString("hit")}
	if _, err := h.SearchFiltered([]float32{1, 0, 0, 0}, 5, filter); err != nil {
		t.Fatalf("SearchFiltered: %v", err)
	}
	after := h.Stats().FilterRejects
	if after <= before {
		t.Errorf("FilterRejects did not increase: before=%d after=%d (a selective filter should reject candidates)", before, after)
	}

	// An unfiltered search must not bump FilterRejects.
	mid := h.Stats().FilterRejects
	if _, err := h.Search([]float32{1, 0, 0, 0}, 5); err != nil {
		t.Fatalf("Search: %v", err)
	}
	if h.Stats().FilterRejects != mid {
		t.Errorf("unfiltered Search changed FilterRejects %d -> %d", mid, h.Stats().FilterRejects)
	}
}
