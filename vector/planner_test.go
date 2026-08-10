// SPDX-License-Identifier: Apache-2.0

package vector

import (
	"math/rand"
	"testing"
)

// TestPreferFilterFirstCostModel checks the cost-based planner's decision across
// selectivity regimes: selective filters take the exact filter-first path,
// non-selective ones fall to graph traversal (cheaper, recall-safe). M and
// EfSearch are small so the crossover is reachable at a test-sized corpus.
func TestPreferFilterFirstCostModel(t *testing.T) {
	h, err := newHNSW(Config{Dim: 4, Metric: L2, M: 4, EfConstruction: 50, EfSearch: 10, Seed: 1})
	if err != nil {
		t.Fatal(err)
	}
	for i := 1; i <= 2000; i++ {
		if _, _, err := h.Insert(uint64(i), []float32{float32(i), 0, 0, 0}, 0, nil, nil, nil, CASCond{}); err != nil {
			t.Fatal(err)
		}
	}
	if h.arena.Size() != 2000 {
		t.Fatalf("size = %d, want 2000", h.arena.Size())
	}
	const k = 10
	cases := []struct {
		name   string
		ncand  int
		wantFF bool
	}{
		{"empty candidate set", 0, true},
		{"very selective (s=0.025)", 50, true},
		{"moderately selective (s=0.1)", 200, true},
		{"non-selective (s=0.95)", 1900, false},
		{"matches all (s=1)", 2000, false},
	}
	for _, c := range cases {
		if got := h.preferFilterFirst(c.ncand, k); got != c.wantFF {
			t.Errorf("%s: preferFilterFirst(%d) = %v, want %v", c.name, c.ncand, got, c.wantFF)
		}
	}
}

// TestPlannerSelectiveExactNonSelectiveRecall is the end-to-end check: a
// selective filter routes to exact filter-first (results match brute force), a
// non-selective filter routes to graph traversal and still returns high recall.
func TestPlannerSelectiveExactNonSelectiveRecall(t *testing.T) {
	const n, dim, k = 5000, 16, 10
	rng := rand.New(rand.NewSource(3))
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
		g := int64(1) // ~98% common (non-selective)
		if i%50 == 0 {
			g = 0 // 2% rare (selective)
		}
		meta := Metadata{"g": NewInt(g)}
		corpus[uint64(i)] = v
		metas[uint64(i)] = meta
		if _, _, err := h.Insert(uint64(i), v, 0, meta, nil, nil, CASCond{}); err != nil {
			t.Fatal(err)
		}
	}
	q := make([]float32, dim)
	for j := range q {
		q[j] = float32(rng.NormFloat64())
	}

	// Selective g==0 -> filter-first -> exact.
	sel := Filter{Op: FilterEq, Field: "g", Value: NewInt(0)}
	got := mustSearch(t, h, q, k, sel)
	want := bruteForceFiltered(corpus, metas, q, k, func(m Metadata) bool { return m["g"].Int == 0 })
	if !eqUint64(resultIDs(got), want) {
		t.Errorf("selective filter not exact: %v != %v", resultIDs(got), want)
	}

	// Non-selective g==1 -> graph traversal -> high recall vs exact truth.
	nonsel := Filter{Op: FilterEq, Field: "g", Value: NewInt(1)}
	got2 := mustSearch(t, h, q, k, nonsel)
	truth := make(map[uint64]bool)
	for _, id := range bruteForceFiltered(corpus, metas, q, k, func(m Metadata) bool { return m["g"].Int == 1 }) {
		truth[id] = true
	}
	matches := 0
	for _, r := range got2 {
		if truth[r.ID] {
			matches++
		}
	}
	if recall := float64(matches) / float64(k); recall < 0.8 {
		t.Errorf("non-selective (graph) recall = %.2f, want >= 0.8", recall)
	}
}

func mustSearch(t *testing.T, h *hnsw, q []float32, k int, f Filter) []Result {
	t.Helper()
	res, err := h.SearchFiltered(q, k, f)
	if err != nil {
		t.Fatal(err)
	}
	return res
}
