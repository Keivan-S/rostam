// SPDX-License-Identifier: Apache-2.0

package vector

import (
	"math/rand"
	"testing"
)

// candidateIDs maps a payload-index candidate slot set to sorted user ids, for
// comparing the index's narrowing against an expected set.
func candidateIDs(t *testing.T, h *hnsw, f Filter, limit int) ([]uint64, bool) {
	t.Helper()
	slots, ok := h.payloadIdx.candidates(f, limit)
	if !ok {
		return nil, false
	}
	ids := make([]uint64, 0, len(slots))
	for _, s := range slots {
		if id := h.slotID(s); id != 0 {
			ids = append(ids, id)
		}
	}
	return sortedIDs(ids), true
}

// TestRangeCandidatesNarrows checks that range / In filters with no equality
// conjunct are narrowed by the payload index (not punted to graph search), and
// that the narrowed set is exactly the slots satisfying the leaf — int and
// float values interoperating, per the predicate's numeric semantics.
func TestRangeCandidatesNarrows(t *testing.T) {
	h, err := newHNSW(Config{Dim: 3, Metric: L2, M: 8, EfConstruction: 50, EfSearch: 50, Seed: 1})
	if err != nil {
		t.Fatal(err)
	}
	// ids 1..20: price = id (mix int/float), year = 2000 + id%5.
	for i := 1; i <= 20; i++ {
		price := Value(NewInt(int64(i)))
		if i%2 == 0 {
			price = NewFloat(float64(i)) // exercise float keys in the same field
		}
		meta := Metadata{"price": price, "year": NewInt(int64(2000 + i%5))}
		if _, _, err := h.Insert(uint64(i), []float32{float32(i), 0, 0}, 0, meta, nil, nil, CASCond{}); err != nil {
			t.Fatal(err)
		}
	}
	const limit = 10_000

	expect := func(pred func(i int) bool) []uint64 {
		var out []uint64
		for i := 1; i <= 20; i++ {
			if pred(i) {
				out = append(out, uint64(i))
			}
		}
		return sortedIDs(out)
	}
	check := func(name string, f Filter, pred func(i int) bool) {
		got, ok := candidateIDs(t, h, f, limit)
		if !ok {
			t.Errorf("%s: candidates returned ok=false (fell back to graph)", name)
			return
		}
		want := expect(pred)
		if !eqUint64(got, want) {
			t.Errorf("%s: candidate ids %v != expected %v", name, got, want)
		}
	}

	check("price > 15", Filter{Op: FilterGt, Field: "price", Value: NewFloat(15)}, func(i int) bool { return i > 15 })
	check("price >= 15", Filter{Op: FilterGte, Field: "price", Value: NewInt(15)}, func(i int) bool { return i >= 15 })
	check("price < 5", Filter{Op: FilterLt, Field: "price", Value: NewInt(5)}, func(i int) bool { return i < 5 })
	check("price <= 5", Filter{Op: FilterLte, Field: "price", Value: NewFloat(5)}, func(i int) bool { return i <= 5 })
	// And of two ranges (no eq term): intersection.
	check("10 < price <= 15",
		Filter{Op: FilterAnd, And: []Filter{
			{Op: FilterGt, Field: "price", Value: NewInt(10)},
			{Op: FilterLte, Field: "price", Value: NewInt(15)},
		}},
		func(i int) bool { return i > 10 && i <= 15 })
	// In over a high-and-low set.
	check("year in {2002, 2004}",
		Filter{Op: FilterIn, Field: "year", Value: NewInts([]int64{2002, 2004})},
		func(i int) bool { return (2000+i%5) == 2002 || (2000+i%5) == 2004 })
}

// TestRangeCandidatesOverflowFallsBack verifies a non-selective range exceeding
// the limit reports ok=false (so the planner uses graph search) rather than
// materializing a near-full-corpus candidate set.
func TestRangeCandidatesOverflowFallsBack(t *testing.T) {
	h, err := newHNSW(Config{Dim: 3, Metric: L2, M: 8, EfConstruction: 50, EfSearch: 50, Seed: 1})
	if err != nil {
		t.Fatal(err)
	}
	for i := 1; i <= 50; i++ {
		if _, _, err := h.Insert(uint64(i), []float32{float32(i), 0, 0}, 0, Metadata{"v": NewInt(int64(i))}, nil, nil, CASCond{}); err != nil {
			t.Fatal(err)
		}
	}
	// "v > 0" matches all 50; with limit 10 the union overflows -> ok=false.
	if _, ok := h.payloadIdx.candidates(Filter{Op: FilterGt, Field: "v", Value: NewInt(0)}, 10); ok {
		t.Error("expected overflow (ok=false) for non-selective range under small limit")
	}
	// A selective range under the same limit still narrows.
	if _, ok := h.payloadIdx.candidates(Filter{Op: FilterGt, Field: "v", Value: NewInt(45)}, 10); !ok {
		t.Error("expected selective range to narrow under limit")
	}
}

// TestFilterFirstRangeExact is the end-to-end check: a range-only filtered
// search returns exactly the brute-force ground truth (the index narrows, the
// predicate re-check keeps it exact).
func TestFilterFirstRangeExact(t *testing.T) {
	const (
		n   = 800
		dim = 8
		k   = 10
	)
	rng := rand.New(rand.NewSource(7))
	h, err := newHNSW(Config{Dim: dim, Metric: L2, M: 16, EfConstruction: 200, EfSearch: 64, Seed: 1})
	if err != nil {
		t.Fatal(err)
	}
	corpus := make(map[uint64][]float32, n)
	metas := make(map[uint64]Metadata, n)
	for i := 1; i <= n; i++ {
		v := make([]float32, dim)
		for j := range v {
			v[j] = float32(rng.NormFloat64())
		}
		id := uint64(i)
		meta := Metadata{"score": NewFloat(rng.Float64()), "rank": NewInt(int64(i % 50))}
		corpus[id] = v
		metas[id] = meta
		if _, _, err := h.Insert(id, v, 0, meta, nil, nil, CASCond{}); err != nil {
			t.Fatalf("insert %d: %v", i, err)
		}
	}
	q := make([]float32, dim)
	for j := range q {
		q[j] = float32(rng.NormFloat64())
	}

	cases := []struct {
		name string
		f    Filter
		pred func(m Metadata) bool
	}{
		{"score >= 0.9", Filter{Op: FilterGt, Field: "score", Value: NewFloat(0.9)},
			func(m Metadata) bool { return m["score"].Flt > 0.9 }},
		{"rank < 5", Filter{Op: FilterLt, Field: "rank", Value: NewInt(5)},
			func(m Metadata) bool { return m["rank"].Int < 5 }},
		{"0.4 <= score <= 0.6", Filter{Op: FilterAnd, And: []Filter{
			{Op: FilterGte, Field: "score", Value: NewFloat(0.4)},
			{Op: FilterLte, Field: "score", Value: NewFloat(0.6)},
		}}, func(m Metadata) bool { return m["score"].Flt >= 0.4 && m["score"].Flt <= 0.6 }},
		{"rank in {1,2,3}", Filter{Op: FilterIn, Field: "rank", Value: NewInts([]int64{1, 2, 3})},
			func(m Metadata) bool { r := m["rank"].Int; return r == 1 || r == 2 || r == 3 }},
	}
	for _, tc := range cases {
		got, err := h.SearchFiltered(q, k, tc.f)
		if err != nil {
			t.Fatalf("%s: %v", tc.name, err)
		}
		want := bruteForceFiltered(corpus, metas, q, k, tc.pred)
		if !eqUint64(resultIDs(got), want) {
			t.Errorf("%s: filter-first ids %v != brute-force %v", tc.name, resultIDs(got), want)
		}
	}
}
