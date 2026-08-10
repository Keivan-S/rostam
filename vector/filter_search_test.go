// SPDX-License-Identifier: Apache-2.0

package vector

import (
	"math/rand"
	"testing"
)

// buildFilteredCorpus inserts n random vectors, assigning each a "bucket"
// string metadata so that roughly matchFrac of them carry bucket="hit".
// Returns the index, the set of matching ids, and the 1-indexed vectors.
func buildFilteredCorpus(t *testing.T, n, dim int, matchFrac float64, seed int64) (*hnsw, map[uint64]bool, [][]float32) {
	t.Helper()
	h, err := newHNSW(Config{Dim: dim, M: 16, EfConstruction: 200, EfSearch: 64, Seed: seed, Metric: L2})
	if err != nil {
		t.Fatalf("newHNSW: %v", err)
	}
	rng := rand.New(rand.NewSource(seed))
	matches := make(map[uint64]bool, n)
	vecs := make([][]float32, n+1) // 1-indexed by id
	for i := 1; i <= n; i++ {
		v := make([]float32, dim)
		for j := range v {
			v[j] = float32(rng.NormFloat64())
		}
		vecs[i] = v
		var meta Metadata
		if rng.Float64() < matchFrac {
			meta = Metadata{"bucket": NewString("hit")}
			matches[uint64(i)] = true
		} else {
			meta = Metadata{"bucket": NewString("miss")}
		}
		if _, _, err := h.Insert(uint64(i), v, 0, meta, nil, nil, CASCond{}); err != nil {
			t.Fatalf("Insert %d: %v", i, err)
		}
	}
	return h, matches, vecs
}

func bruteTopKAmong(query []float32, vecs [][]float32, matches map[uint64]bool, k int) []uint64 {
	type sd struct {
		id uint64
		d  float32
	}
	var all []sd
	for id := uint64(1); int(id) < len(vecs); id++ {
		if !matches[id] {
			continue
		}
		all = append(all, sd{id, l2SquaredScalar(vecs[id], query)})
	}
	for i := 1; i < len(all); i++ {
		for j := i; j > 0 && all[j-1].d > all[j].d; j-- {
			all[j-1], all[j] = all[j], all[j-1]
		}
	}
	out := make([]uint64, 0, k)
	for i := 0; i < len(all) && i < k; i++ {
		out = append(out, all[i].id)
	}
	return out
}

func TestSearchFilteredCorrectness(t *testing.T) {
	h, matches, _ := buildFilteredCorpus(t, 1000, 16, 0.5, 1)
	filter := Filter{Op: FilterEq, Field: "bucket", Value: NewString("hit")}

	query := make([]float32, 16) // origin
	results, err := h.SearchFiltered(query, 20, filter)
	if err != nil {
		t.Fatalf("SearchFiltered: %v", err)
	}
	if len(results) == 0 {
		t.Fatal("filtered search returned no results")
	}
	// Every returned id MUST be a match — this is the hard correctness bar.
	for _, r := range results {
		if !matches[r.ID] {
			t.Errorf("result id %d does not satisfy filter (bucket!=hit)", r.ID)
		}
	}
}

func TestSearchFilteredRecall(t *testing.T) {
	const n, dim, k = 2000, 32, 10
	h, matches, vecs := buildFilteredCorpus(t, n, dim, 0.5, 7)
	filter := Filter{Op: FilterEq, Field: "bucket", Value: NewString("hit")}

	rng := rand.New(rand.NewSource(99))
	const queries = 20
	var totalRecall float64
	for q := 0; q < queries; q++ {
		query := make([]float32, dim)
		for j := range query {
			query[j] = float32(rng.NormFloat64())
		}
		truth := bruteTopKAmong(query, vecs, matches, k)
		got, err := h.SearchFiltered(query, k, filter)
		if err != nil {
			t.Fatalf("SearchFiltered: %v", err)
		}
		gotSet := make(map[uint64]bool, len(got))
		for _, r := range got {
			gotSet[r.ID] = true
			if !matches[r.ID] {
				t.Fatalf("non-matching id %d in filtered results", r.ID)
			}
		}
		hits := 0
		for _, id := range truth {
			if gotSet[id] {
				hits++
			}
		}
		if len(truth) > 0 {
			totalRecall += float64(hits) / float64(len(truth))
		}
	}
	avg := totalRecall / queries
	// Conservative threshold for pure random Gaussian (harder than real
	// embeddings). Asserts the mechanism (ef widening recovers matching
	// neighbors at 50% selectivity), not the spec's real-embedding target.
	if avg < 0.60 {
		t.Errorf("filtered recall@%d = %.2f, want >= 0.60", k, avg)
	}
	t.Logf("filtered recall@%d (50%% selectivity) = %.3f", k, avg)
}

func TestSearchFilteredEfWidening(t *testing.T) {
	// With a highly selective filter and a tiny MaxEfSearch, the search can't
	// widen enough to find many matches. A large MaxEfSearch recovers more.
	const n, dim, k = 2000, 16, 10
	mkConfig := func(maxEf int) Config {
		return Config{Dim: dim, M: 16, EfConstruction: 200, EfSearch: 16, Seed: 3, Metric: L2, MaxEfSearch: maxEf}
	}

	build := func(cfg Config) *hnsw {
		h, err := newHNSW(cfg)
		if err != nil {
			t.Fatalf("newHNSW: %v", err)
		}
		rng := rand.New(rand.NewSource(3))
		for i := 1; i <= n; i++ {
			v := make([]float32, dim)
			for j := range v {
				v[j] = float32(rng.NormFloat64())
			}
			var meta Metadata
			if rng.Float64() < 0.02 { // 2% selectivity
				meta = Metadata{"bucket": NewString("hit")}
			} else {
				meta = Metadata{"bucket": NewString("miss")}
			}
			if _, _, err := h.Insert(uint64(i), v, 0, meta, nil, nil, CASCond{}); err != nil {
				t.Fatalf("Insert: %v", err)
			}
		}
		return h
	}

	filter := Filter{Op: FilterEq, Field: "bucket", Value: NewString("hit")}
	query := make([]float32, dim)

	hSmall := build(mkConfig(16)) // no widening room
	hLarge := build(mkConfig(2048))

	small, err := hSmall.SearchFiltered(query, k, filter)
	if err != nil {
		t.Fatal(err)
	}
	large, err := hLarge.SearchFiltered(query, k, filter)
	if err != nil {
		t.Fatal(err)
	}
	// Larger ef budget should find at least as many matches as the tiny one.
	if len(large) < len(small) {
		t.Errorf("large MaxEfSearch found fewer results (%d) than small (%d)", len(large), len(small))
	}
	t.Logf("ef-widening: small=%d large=%d matches found", len(small), len(large))
}

func TestSearchNoFilterMatchesSearch(t *testing.T) {
	h, _, _ := buildFilteredCorpus(t, 500, 16, 0.5, 5)
	query := make([]float32, 16)
	a, err := h.Search(query, 10)
	if err != nil {
		t.Fatal(err)
	}
	b, err := h.SearchFiltered(query, 10, Filter{})
	if err != nil {
		t.Fatal(err)
	}
	if len(a) != len(b) {
		t.Fatalf("Search returned %d, SearchFiltered(zero) returned %d", len(a), len(b))
	}
	for i := range a {
		if a[i].ID != b[i].ID || a[i].Distance != b[i].Distance {
			t.Errorf("result %d differs: Search=%+v SearchFiltered=%+v", i, a[i], b[i])
		}
	}
}
