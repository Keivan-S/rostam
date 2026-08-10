// SPDX-License-Identifier: Apache-2.0

package vector

import (
	"math"
	"math/rand"
	"sort"
	"testing"
)

// mvFilterCorpus builds an MV index of n docs (ids 1..n), each with a single
// random token vector, and tags each doc with a representative payload spread:
//
//	bucket: "hit" for even ids, "miss" for odd ids        (eq / in)
//	rank:   the id as an int                              (gt / lt range)
//	tags:   ["red","blue"] for ids divisible by 3         (match)
//	note:   explicit null for ids divisible by 5          (is_null)
//	geo.city: "NYC" for ids divisible by 7                (nested/dotted key)
//
// Returns the index and the per-id metadata so tests can compute references.
func mvFilterCorpus(t *testing.T, n, dim int, seed int64) (*MultiVectorIndex, map[uint64]Metadata) {
	t.Helper()
	m, err := NewMultiVectorIndex(MultiVectorConfig{Dim: dim, M: 16, EfConstruction: 200, EfSearch: 128, Seed: seed})
	if err != nil {
		t.Fatalf("NewMultiVectorIndex: %v", err)
	}
	rng := rand.New(rand.NewSource(seed))
	metas := make(map[uint64]Metadata, n)
	for id := uint64(1); id <= uint64(n); id++ {
		toks := randTokens(rng, 2+rng.Intn(4), dim)
		meta := Metadata{"rank": NewInt(int64(id))}
		if id%2 == 0 {
			meta["bucket"] = NewString("hit")
		} else {
			meta["bucket"] = NewString("miss")
		}
		if id%3 == 0 {
			meta["tags"] = NewStrings([]string{"red", "blue"})
		}
		if id%5 == 0 {
			meta["note"] = Value{Kind: ValueNone}
		}
		if id%7 == 0 {
			meta["geo.city"] = NewString("NYC")
		}
		if err := m.Add(id, toks, meta); err != nil {
			t.Fatalf("Add %d: %v", id, err)
		}
		metas[id] = meta
	}
	return m, metas
}

// TestMultiVectorFilterOperators exercises a representative operator spread and
// asserts every returned doc satisfies the filter (the hard correctness bar).
func TestMultiVectorFilterOperators(t *testing.T) {
	const n, dim = 120, 12
	m, metas := mvFilterCorpus(t, n, dim, 11)
	defer m.Close()

	rng := rand.New(rand.NewSource(3))
	query := randTokens(rng, 4, dim)

	cases := []struct {
		name   string
		filter Filter
		want   func(Metadata) bool
	}{
		{
			name:   "eq",
			filter: Filter{Op: FilterEq, Field: "bucket", Value: NewString("hit")},
			want:   func(md Metadata) bool { return md["bucket"].Str == "hit" },
		},
		{
			name:   "in",
			filter: Filter{Op: FilterIn, Field: "bucket", Value: NewStrings([]string{"hit", "other"})},
			want:   func(md Metadata) bool { return md["bucket"].Str == "hit" },
		},
		{
			name: "range gt/lt",
			filter: Filter{Op: FilterAnd, And: []Filter{
				{Op: FilterGt, Field: "rank", Value: NewInt(40)},
				{Op: FilterLt, Field: "rank", Value: NewInt(60)},
			}},
			want: func(md Metadata) bool { return md["rank"].Int > 40 && md["rank"].Int < 60 },
		},
		{
			name:   "match",
			filter: Filter{Op: FilterMatch, Field: "tags", Value: NewString("red")},
			want:   func(md Metadata) bool { return md["tags"].Kind == ValueStrings },
		},
		{
			name:   "is_null",
			filter: Filter{Op: FilterIsNull, Field: "note"},
			want:   func(md Metadata) bool { return md["note"].Kind == ValueNone },
		},
		{
			name:   "nested dotted key eq",
			filter: Filter{Op: FilterEq, Field: "geo.city", Value: NewString("NYC")},
			want:   func(md Metadata) bool { return md["geo.city"].Str == "NYC" },
		},
	}

	for _, c := range cases {
		t.Run(c.name, func(t *testing.T) {
			res, err := m.Search(query, 10, MultiSearchOpts{Filter: c.filter})
			if err != nil {
				t.Fatalf("Search: %v", err)
			}
			if len(res) == 0 {
				t.Fatalf("filter %q returned no results", c.name)
			}
			// Count the corpus matches; the result count must equal min(k, matches).
			matches := 0
			for _, md := range metas {
				if c.want(md) {
					matches++
				}
			}
			wantN := matches
			if wantN > 10 {
				wantN = 10
			}
			if len(res) != wantN {
				t.Errorf("filter %q: got %d results, want %d (corpus matches=%d, k=10)",
					c.name, len(res), wantN, matches)
			}
			for _, r := range res {
				if !c.want(metas[r.ID]) {
					t.Errorf("filter %q: result id %d does not satisfy filter (meta=%v)",
						c.name, r.ID, metas[r.ID])
				}
			}
		})
	}
}

// TestMultiVectorNoFilterUnchanged asserts the no-filter path is
// byte/behaviour-identical: a search with a zero Filter returns the exact same
// ids, scores, and order as a search with no opts at all.
func TestMultiVectorNoFilterUnchanged(t *testing.T) {
	const n, dim = 80, 16
	m, _ := mvFilterCorpus(t, n, dim, 23)
	defer m.Close()

	rng := rand.New(rand.NewSource(99))
	query := randTokens(rng, 5, dim)

	base, err := m.Search(query, 12, MultiSearchOpts{CandidatesPerToken: 64})
	if err != nil {
		t.Fatalf("baseline Search: %v", err)
	}
	withZero, err := m.Search(query, 12, MultiSearchOpts{CandidatesPerToken: 64, Filter: Filter{}})
	if err != nil {
		t.Fatalf("zero-filter Search: %v", err)
	}
	if len(base) != len(withZero) {
		t.Fatalf("len mismatch: baseline=%d zero-filter=%d", len(base), len(withZero))
	}
	for i := range base {
		if base[i].ID != withZero[i].ID || base[i].Score != withZero[i].Score {
			t.Errorf("rank %d differs: baseline=(%d,%.6f) zero-filter=(%d,%.6f)",
				i, base[i].ID, base[i].Score, withZero[i].ID, withZero[i].Score)
		}
	}
}

// TestMultiVectorFilterAdaptiveWiden proves the adaptive over-fetch fills k for a
// SELECTIVE filter the un-widened candidate pool would under-fill.
//
// Construction (deterministic, no HNSW-recall guesswork): 100 single-token docs
// laid on a fine angular ramp from +e0 (doc 1, closest to the query) toward +e1
// (doc 100, farthest), so similarity rank == doc id. The ONLY matching docs are
// ids 56..65 — i.e. the matches sit at ranks 56-65, strictly OUTSIDE the top-50.
//
// Sizing is tied to the implementation's widen math (k=5): the un-widened DEFAULT
// per-token budget is max(4*k,50)=50, whose pool is exactly docs 1..50 — ZERO
// matches. The widen floor is max(8*k,100)=100, whose pool reaches docs 1..100 —
// all 10 matches. So:
//   - The un-widened pool (asserted directly below) contains NONE of the matches,
//     so without the widen the filtered Search would return 0 (the test FAILS).
//   - The default-opts filtered Search auto-widens to 100 and fills k.
func TestMultiVectorFilterAdaptiveWiden(t *testing.T) {
	const dim, k = 8, 5
	const matchLo, matchHi = 56, 65 // matches sit at ranks 56..65 (outside top-50)
	m, err := NewMultiVectorIndex(MultiVectorConfig{Dim: dim, M: 16, EfConstruction: 200, EfSearch: 200, Seed: 5})
	if err != nil {
		t.Fatalf("NewMultiVectorIndex: %v", err)
	}
	defer m.Close()

	query := [][]float32{unit(dim, 0, 1)}
	matchIDs := map[uint64]bool{}
	for i := 1; i <= 100; i++ {
		// theta grows with i => cos(theta)=dot with +e0 query shrinks => rank == i.
		theta := float64(i) * 0.012
		v := make([]float32, dim)
		v[0] = float32(math.Cos(theta))
		v[1] = float32(math.Sin(theta))
		normalize(v)
		keep := i >= matchLo && i <= matchHi
		if keep {
			matchIDs[uint64(i)] = true
		}
		if err := m.Add(uint64(i), [][]float32{v}, Metadata{"keep": NewBool(keep)}); err != nil {
			t.Fatalf("Add %d: %v", i, err)
		}
	}

	// Establish selectivity DIRECTLY: the un-widened default pool (per-token
	// budget 50) is exactly the top-50 docs and contains NONE of the matches.
	// An unfiltered search (no predicate => no widen) at budget 50 exposes that
	// pool. If the filtered Search used this pool it would return 0.
	unwidened, err := m.Search(query, 50, MultiSearchOpts{CandidatesPerToken: 50})
	if err != nil {
		t.Fatalf("un-widened pool Search: %v", err)
	}
	for _, r := range unwidened {
		if matchIDs[r.ID] {
			t.Fatalf("un-widened pool already contains match id %d — corpus not selective enough to prove the widen", r.ID)
		}
	}

	// Adaptive path: default opts auto-widen the per-token budget to
	// max(8*k,100)=100, whose pool reaches the rank-56..65 matches and fills k.
	// Remove the widen block and this Search falls back to the budget-50 pool
	// above (zero matches) and returns 0 — so this assertion is the widen's proof.
	filter := Filter{Op: FilterEq, Field: "keep", Value: NewBool(true)}
	got, err := m.Search(query, k, MultiSearchOpts{Filter: filter})
	if err != nil {
		t.Fatalf("adaptive Search: %v", err)
	}
	if len(got) != k {
		t.Fatalf("adaptive widen failed to fill k: got %d, want %d (un-widened pool held 0 matches)", len(got), k)
	}
	for _, r := range got {
		if !matchIDs[r.ID] {
			t.Errorf("adaptive result id %d is not a matching doc (ranks %d..%d)", r.ID, matchLo, matchHi)
		}
	}
}

// TestMultiVectorFilterEmptyAndError covers the zero-result and fail-loud paths.
func TestMultiVectorFilterEmptyAndError(t *testing.T) {
	const n, dim = 40, 8
	m, _ := mvFilterCorpus(t, n, dim, 13)
	defer m.Close()

	rng := rand.New(rand.NewSource(1))
	query := randTokens(rng, 3, dim)

	// Filter matching nothing => 0 results, NOT an error.
	none, err := m.Search(query, 10, MultiSearchOpts{
		Filter: Filter{Op: FilterEq, Field: "bucket", Value: NewString("nonexistent")},
	})
	if err != nil {
		t.Fatalf("no-match Search returned error: %v", err)
	}
	if len(none) != 0 {
		t.Fatalf("no-match Search returned %d results, want 0", len(none))
	}

	// Malformed filter (invalid regex) => compile error surfaced from Search.
	_, err = m.Search(query, 10, MultiSearchOpts{
		Filter: Filter{Op: FilterRegex, Field: "bucket", Value: NewString("(")},
	})
	if err == nil {
		t.Fatal("malformed regex filter should surface a Compile error from Search")
	}
}

// TestMultiVectorFilterPreservesOrder asserts the filter removes docs but never
// reorders survivors: the filtered ranking is the unfiltered ranking with
// non-matching docs deleted.
func TestMultiVectorFilterPreservesOrder(t *testing.T) {
	const n, dim, k = 100, 16, 15
	m, metas := mvFilterCorpus(t, n, dim, 31)
	defer m.Close()

	rng := rand.New(rand.NewSource(77))
	query := randTokens(rng, 5, dim)

	// Unfiltered top-(k*4): generous so the filtered survivors are all present.
	full, err := m.Search(query, k*4, MultiSearchOpts{CandidatesPerToken: 256})
	if err != nil {
		t.Fatalf("unfiltered Search: %v", err)
	}
	// Expected filtered order = unfiltered order with non-"hit" docs removed.
	keep := func(id uint64) bool { return metas[id]["bucket"].Str == "hit" }
	var wantOrder []uint64
	for _, r := range full {
		if keep(r.ID) {
			wantOrder = append(wantOrder, r.ID)
			if len(wantOrder) == k {
				break
			}
		}
	}

	filtered, err := m.Search(query, k, MultiSearchOpts{
		CandidatesPerToken: 256,
		Filter:             Filter{Op: FilterEq, Field: "bucket", Value: NewString("hit")},
	})
	if err != nil {
		t.Fatalf("filtered Search: %v", err)
	}
	if len(filtered) != len(wantOrder) {
		t.Fatalf("filtered len = %d, want %d", len(filtered), len(wantOrder))
	}
	for i := range filtered {
		if filtered[i].ID != wantOrder[i] {
			t.Errorf("rank %d: filtered id %d, want %d (order must match unfiltered survivors)",
				i, filtered[i].ID, wantOrder[i])
		}
	}
	// Sanity: scores are still descending (filter never reorders survivors).
	if !sort.SliceIsSorted(filtered, func(i, j int) bool {
		if filtered[i].Score != filtered[j].Score {
			return filtered[i].Score > filtered[j].Score
		}
		return filtered[i].ID < filtered[j].ID
	}) {
		t.Error("filtered results are not in descending MaxSim order")
	}
}

// unit returns a dim-length vector that is `val` along axis `ax`, else 0.
func unit(dim, ax int, val float32) []float32 {
	v := make([]float32, dim)
	v[ax] = val
	return v
}
