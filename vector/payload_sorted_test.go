// SPDX-License-Identifier: Apache-2.0

package vector

import "testing"

// TestRangeSortedCacheUpdates exercises the sorted-range cache's invalidation:
// a new distinct value triggers a rebuild, a new slot on an EXISTING value is
// reflected live (no rebuild), and reclaim removes a value. Uses candidateIDs
// (payload_range_test.go) to inspect the narrowed set directly.
func TestRangeSortedCacheUpdates(t *testing.T) {
	h, err := newHNSW(Config{Dim: 3, Metric: L2, M: 8, EfConstruction: 50, EfSearch: 50, Seed: 1})
	if err != nil {
		t.Fatal(err)
	}
	const limit = 10_000
	ins := func(id uint64, v int) {
		if _, _, err := h.Insert(id, []float32{float32(id), 0, 0}, 0, Metadata{"v": NewInt(int64(v))}, nil, nil, CASCond{}); err != nil {
			t.Fatalf("insert %d: %v", id, err)
		}
	}
	gt := func(field string, v int64) Filter { return Filter{Op: FilterGt, Field: field, Value: NewInt(v)} }
	gte := func(field string, v int64) Filter { return Filter{Op: FilterGte, Field: field, Value: NewInt(v)} }
	check := func(name string, f Filter, want []uint64) {
		got, ok := candidateIDs(t, h, f, limit)
		if !ok {
			t.Errorf("%s: not narrowed (ok=false)", name)
			return
		}
		if !eqUint64(got, want) {
			t.Errorf("%s: candidates %v, want %v", name, got, want)
		}
	}

	for i := 1; i <= 10; i++ {
		ins(uint64(i), i) // v = 1..10, distinct
	}
	check("v>5 (builds cache)", gt("v", 5), []uint64{6, 7, 8, 9, 10})

	// (a) new distinct value -> cache invalidated, must rebuild and include it.
	ins(11, 11)
	check("v>5 after new key 11", gt("v", 5), []uint64{6, 7, 8, 9, 10, 11})

	// (b) new slot on an EXISTING value (v=10) -> reflected live, no rebuild.
	ins(12, 10)
	check("v>=10 after existing-key insert", gte("v", 10), []uint64{10, 11, 12})

	// (c) delete + reclaim removes value 11 -> excluded afterward.
	h.Delete(11, CASCond{})
	h.Reclaim()
	check("v>9 after reclaim of 11", gt("v", 9), []uint64{10, 12})
}

// TestRangeSortedCacheStrings checks the string sorted-range path (lexicographic
// bounds) against expected sets, including a rebuild after a new key.
func TestRangeSortedCacheStrings(t *testing.T) {
	h, err := newHNSW(Config{Dim: 2, Metric: L2, M: 8, EfConstruction: 50, EfSearch: 50, Seed: 1})
	if err != nil {
		t.Fatal(err)
	}
	const limit = 10_000
	ins := func(id uint64, s string) {
		if _, _, err := h.Insert(id, []float32{float32(id), 0}, 0, Metadata{"tag": NewString(s)}, nil, nil, CASCond{}); err != nil {
			t.Fatalf("insert %d: %v", id, err)
		}
	}
	ins(1, "apple")
	ins(2, "banana")
	ins(3, "cherry")
	got, ok := candidateIDs(t, h, Filter{Op: FilterGte, Field: "tag", Value: NewString("banana")}, limit)
	if !ok || !eqUint64(got, []uint64{2, 3}) {
		t.Errorf("tag>=banana = %v (ok=%v)", got, ok)
	}
	ins(4, "avocado") // new key between apple and banana -> rebuild
	got, _ = candidateIDs(t, h, Filter{Op: FilterLt, Field: "tag", Value: NewString("banana")}, limit)
	if !eqUint64(got, []uint64{1, 4}) {
		t.Errorf("tag<banana after new key = %v, want [1 4]", got)
	}
}
