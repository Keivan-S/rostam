// SPDX-License-Identifier: Apache-2.0

package vector

import (
	"math/rand"
	"testing"
	"time"
)

// rfc3339ms parses an RFC3339 string into unix-ms (test helper mirroring the
// engine's datetime convention). Fails the test on a bad literal.
func rfc3339ms(t *testing.T, s string) int64 {
	t.Helper()
	tm, err := time.Parse(time.RFC3339, s)
	if err != nil {
		t.Fatalf("rfc3339ms(%q): %v", s, err)
	}
	return tm.UnixMilli()
}

// buildRichCorpus inserts n vectors with string, array, and int64-ms datetime
// metadata so the rich operators have something to bite on. Returns the index,
// the per-id metadata (for brute-force reference), and the per-id vectors.
func buildRichCorpus(t *testing.T, n, dim int, seed int64) (*hnsw, map[uint64]Metadata, map[uint64][]float32) {
	t.Helper()
	h, err := newHNSW(Config{Dim: dim, Metric: L2, M: 16, EfConstruction: 200, EfSearch: 64, Seed: seed})
	if err != nil {
		t.Fatalf("newHNSW: %v", err)
	}
	rng := rand.New(rand.NewSource(seed))
	metas := make(map[uint64]Metadata, n)
	corpus := make(map[uint64][]float32, n)
	// A spread of datetimes across ~30 days so dt ranges are selective.
	base := rfc3339ms(t, "2024-01-01T00:00:00Z")
	const dayMS = int64(24 * 60 * 60 * 1000)
	titles := []string{
		"the quick brown fox",
		"lazy dog sleeps",
		"quick silver fox",
		"brown bear runs",
		"the lazy fox jumps",
	}
	tags := [][]string{
		{"red", "urgent"},
		{"blue"},
		{"green", "urgent"},
		{},
		{"red"},
	}
	for i := 1; i <= n; i++ {
		v := make([]float32, dim)
		for j := range v {
			v[j] = float32(rng.NormFloat64())
		}
		id := uint64(i)
		corpus[id] = v
		meta := Metadata{
			"title": NewString(titles[i%len(titles)]),
			"tags":  NewStrings(append([]string(nil), tags[i%len(tags)]...)),
			"sku":   NewString(skuFor(i)),
			"ts":    NewInt(base + int64(i%30)*dayMS),
		}
		metas[id] = meta
		if _, _, err := h.Insert(id, v, 0, meta, nil, nil, CASCond{}); err != nil {
			t.Fatalf("Insert %d: %v", i, err)
		}
	}
	return h, metas, corpus
}

// skuFor builds a regex-friendly sku like "AB-0007" / "XY-0042".
func skuFor(i int) string {
	letters := []string{"AB", "XY", "ZZ"}
	return letters[i%len(letters)] + "-" + pad4(i)
}

func pad4(i int) string {
	s := []byte("0000")
	for p := 3; p >= 0 && i > 0; p-- {
		s[p] = byte('0' + i%10)
		i /= 10
	}
	return string(s)
}

// TestRichFilterSearchCorrectness runs match / regex / is_empty / is_null / dt
// filters through SearchFiltered and asserts EVERY returned result satisfies the
// compiled predicate (the hard correctness bar) plus sane recall on a selective
// dt range.
func TestRichFilterSearchCorrectness(t *testing.T) {
	const n, dim, k = 1200, 16, 20
	h, metas, corpus := buildRichCorpus(t, n, dim, 11)

	query := make([]float32, dim) // origin

	cases := []struct {
		name string
		f    Filter
	}{
		{"match title=fox", Filter{Op: FilterMatch, Field: "title", Value: NewString("fox")}},
		{"match title=quick fox", Filter{Op: FilterMatch, Field: "title", Value: NewString("quick fox")}},
		{"regex sku=^AB-", Filter{Op: FilterRegex, Field: "sku", Value: NewString("^AB-")}},
		{"regex tags=urgent", Filter{Op: FilterRegex, Field: "tags", Value: NewString("urgent")}},
		{"is_empty tags", Filter{Op: FilterIsEmpty, Field: "tags"}},
		{"is_empty missing", Filter{Op: FilterIsEmpty, Field: "nope"}},
		{"is_null ts (all false)", Filter{Op: FilterIsNull, Field: "ts"}},
		{"dt_gte", Filter{Op: FilterDtGte, Field: "ts", Value: NewString("2024-01-15T00:00:00Z")}},
		{"dt range [10,20)", Filter{Op: FilterAnd, And: []Filter{
			{Op: FilterDtGte, Field: "ts", Value: NewString("2024-01-11T00:00:00Z")},
			{Op: FilterDtLt, Field: "ts", Value: NewString("2024-01-21T00:00:00Z")},
		}}},
	}

	for _, tc := range cases {
		pred, err := tc.f.Compile()
		if err != nil {
			t.Fatalf("%s: compile: %v", tc.name, err)
		}
		got, err := h.SearchFiltered(query, k, tc.f)
		if err != nil {
			t.Fatalf("%s: search: %v", tc.name, err)
		}
		for _, r := range got {
			if pred != nil && !pred(metas[r.ID]) {
				t.Errorf("%s: result id %d does not satisfy predicate (meta=%v)", tc.name, r.ID, metas[r.ID])
			}
		}
	}

	// Recall sanity on a selective dt range: filter-first must return exactly the
	// brute-force ground truth (it is exact when the index narrows).
	dtRange := Filter{Op: FilterAnd, And: []Filter{
		{Op: FilterDtGte, Field: "ts", Value: NewString("2024-01-11T00:00:00Z")},
		{Op: FilterDtLt, Field: "ts", Value: NewString("2024-01-13T00:00:00Z")},
	}}
	lo := rfc3339ms(t, "2024-01-11T00:00:00Z")
	hi := rfc3339ms(t, "2024-01-13T00:00:00Z")
	want := bruteForceFiltered(corpus, metas, query, k, func(m Metadata) bool {
		ts := m["ts"].Int
		return ts >= lo && ts < hi
	})
	got := mustSearch(t, h, query, k, dtRange)
	if !eqUint64(resultIDs(got), want) {
		t.Errorf("selective dt range not exact: %v != %v", resultIDs(got), want)
	}
}

// TestDatetimeRangeUsesIndex asserts a dt_gte+dt_lt And is NARROWED by the
// payload index's sorted-range path (ok==true) to exactly the matching slots —
// the same acceleration a plain numeric range gets. Mirrors
// TestRangeCandidatesNarrows.
func TestDatetimeRangeUsesIndex(t *testing.T) {
	h, err := newHNSW(Config{Dim: 3, Metric: L2, M: 8, EfConstruction: 50, EfSearch: 50, Seed: 1})
	if err != nil {
		t.Fatal(err)
	}
	base := rfc3339ms(t, "2024-01-01T00:00:00Z")
	const dayMS = int64(24 * 60 * 60 * 1000)
	// ids 1..30: ts = base + (id-1) days, one distinct day each.
	for i := 1; i <= 30; i++ {
		ts := base + int64(i-1)*dayMS
		meta := Metadata{"ts": NewInt(ts)}
		if _, _, err := h.Insert(uint64(i), []float32{float32(i), 0, 0}, 0, meta, nil, nil, CASCond{}); err != nil {
			t.Fatal(err)
		}
	}
	const limit = 10_000

	// dt_gte 2024-01-11 (id 11) AND dt_lt 2024-01-21 (id 21) -> ids 11..20.
	f := Filter{Op: FilterAnd, And: []Filter{
		{Op: FilterDtGte, Field: "ts", Value: NewString("2024-01-11T00:00:00Z")},
		{Op: FilterDtLt, Field: "ts", Value: NewString("2024-01-21T00:00:00Z")},
	}}
	got, ok := candidateIDs(t, h, f, limit)
	if !ok {
		t.Fatal("dt range: candidates returned ok=false (expected index narrowing)")
	}
	var want []uint64
	for i := 11; i <= 20; i++ {
		want = append(want, uint64(i))
	}
	if !eqUint64(got, sortedIDs(want)) {
		t.Errorf("dt range candidate ids %v != expected %v", got, sortedIDs(want))
	}

	// A single dt_lte also narrows.
	if _, ok := h.payloadIdx.candidates(Filter{Op: FilterDtLte, Field: "ts", Value: NewString("2024-01-05T00:00:00Z")}, limit); !ok {
		t.Error("single dt_lte: expected index narrowing (ok=true)")
	}

	// An invalid RFC3339 bound must DECLINE (ok=false), never panic — the compile
	// error is the authoritative rejection elsewhere.
	if _, ok := h.payloadIdx.candidates(Filter{Op: FilterDtGt, Field: "ts", Value: NewString("not-a-date")}, limit); ok {
		t.Error("invalid dt bound: expected candidates ok=false (decline), got ok=true")
	}
}

// TestNonIndexableRichFiltersFallBackToGraph asserts match/regex/is_empty/is_null
// DECLINE indexing (candidates ok=false) so they fall back to filtered graph
// traversal — and that graph traversal still returns correct results, exercising
// ef-widening on a selective filter.
func TestNonIndexableRichFiltersFallBackToGraph(t *testing.T) {
	h, metas, corpus := buildRichCorpus(t, 1500, 16, 23)
	const limit = 10_000

	// match is now token-index accelerated (selective token narrows to a superset);
	// assert it narrows AND stays a superset of the predicate matches.
	matchF := Filter{Op: FilterMatch, Field: "title", Value: NewString("fox")}
	mcands, ok := h.payloadIdx.candidates(matchF, limit)
	if !ok {
		t.Error("match: expected token-index narrowing (ok=true)")
	}
	mPred, err := matchF.Compile()
	if err != nil {
		t.Fatal(err)
	}
	mCandIDs := make(map[uint64]bool, len(mcands))
	h.mu.RLock()
	for _, s := range mcands {
		mCandIDs[h.arena.ID(s)] = true
	}
	h.mu.RUnlock()
	for id, m := range metas {
		if mPred(m) && !mCandIDs[id] {
			t.Errorf("match candidate set MISSING true match id %d (not a superset)", id)
		}
	}

	nonIndexable := []struct {
		name string
		f    Filter
	}{
		{"regex", Filter{Op: FilterRegex, Field: "sku", Value: NewString("^AB-")}},
		{"is_empty", Filter{Op: FilterIsEmpty, Field: "tags"}},
		{"is_null", Filter{Op: FilterIsNull, Field: "ts"}},
	}
	for _, tc := range nonIndexable {
		if _, ok := h.payloadIdx.candidates(tc.f, limit); ok {
			t.Errorf("%s: candidates returned ok=true, expected decline (ok=false)", tc.name)
		}
		// An And mixing a non-indexable op with an indexable dt op still narrows on
		// the dt op only (the non-indexable conjunct is re-checked by the predicate).
		mixed := Filter{Op: FilterAnd, And: []Filter{
			tc.f,
			{Op: FilterDtGte, Field: "ts", Value: NewString("2024-01-15T00:00:00Z")},
		}}
		if _, ok := h.payloadIdx.candidates(mixed, limit); !ok {
			t.Errorf("%s: And(non-indexable, dt) should still narrow via the dt conjunct", tc.name)
		}
	}

	// Correctness through graph traversal: a regex selective on ~1/3 of the corpus.
	query := make([]float32, 16)
	for j := range query {
		query[j] = 0.1 * float32(j)
	}
	f := Filter{Op: FilterRegex, Field: "sku", Value: NewString("^AB-")}
	pred, err := f.Compile()
	if err != nil {
		t.Fatal(err)
	}
	const k = 10
	got := mustSearch(t, h, query, k, f)
	for _, r := range got {
		if !pred(metas[r.ID]) {
			t.Errorf("graph-path regex result id %d violates predicate", r.ID)
		}
	}
	// Recall vs brute-force ground truth: ef-widening should recover most of the
	// true nearest matching neighbors at this selectivity.
	truth := make(map[uint64]bool)
	for _, id := range bruteForceFiltered(corpus, metas, query, k, func(m Metadata) bool { return pred(m) }) {
		truth[id] = true
	}
	matches := 0
	for _, r := range got {
		if truth[r.ID] {
			matches++
		}
	}
	if recall := float64(matches) / float64(k); recall < 0.7 {
		t.Errorf("graph-path regex recall = %.2f, want >= 0.7", recall)
	}
}

// TestRichFilterBothSearchPaths forces filter-first (selective dt range) and
// graph traversal (non-selective match), asserting correctness on both — the
// predicate gate (admits) is predicate-agnostic, so both paths must agree with
// brute force / the compiled predicate. Mirrors planner_test.go.
func TestRichFilterBothSearchPaths(t *testing.T) {
	const n, dim, k = 4000, 16, 10
	rng := rand.New(rand.NewSource(5))
	h, err := newHNSW(Config{Dim: dim, Metric: L2, M: 16, EfConstruction: 200, EfSearch: 64, Seed: 1})
	if err != nil {
		t.Fatal(err)
	}
	base := rfc3339ms(t, "2024-01-01T00:00:00Z")
	const dayMS = int64(24 * 60 * 60 * 1000)
	corpus := make(map[uint64][]float32, n)
	metas := make(map[uint64]Metadata, n)
	for i := 1; i <= n; i++ {
		v := make([]float32, dim)
		for j := range v {
			v[j] = float32(rng.NormFloat64())
		}
		id := uint64(i)
		// ts: 2% of points land in a narrow 1-day window (selective filter-first),
		// the rest spread over 300 days. title is "common fox" on ~98% of points
		// (non-selective match -> graph).
		var ts int64
		if i%50 == 0 {
			ts = base // the rare selective bucket
		} else {
			ts = base + int64(1+rng.Intn(300))*dayMS
		}
		title := "common fox text"
		if i%50 == 0 {
			title = "rare zebra text"
		}
		meta := Metadata{"ts": NewInt(ts), "title": NewString(title)}
		corpus[id] = v
		metas[id] = meta
		if _, _, err := h.Insert(id, v, 0, meta, nil, nil, CASCond{}); err != nil {
			t.Fatal(err)
		}
	}
	q := make([]float32, dim)
	for j := range q {
		q[j] = float32(rng.NormFloat64())
	}

	// Selective dt range -> filter-first -> EXACT vs brute force (catches any
	// index-narrowing bug: a too-narrow index would drop valid results).
	loS := "2024-01-01T00:00:00Z"
	hiS := "2024-01-01T12:00:00Z"
	dtSel := Filter{Op: FilterAnd, And: []Filter{
		{Op: FilterDtGte, Field: "ts", Value: NewString(loS)},
		{Op: FilterDtLt, Field: "ts", Value: NewString(hiS)},
	}}
	// Confirm the planner actually takes the index path for this selective filter.
	if _, ok := h.payloadIdx.candidates(dtSel, h.filterFirstThreshold()); !ok {
		t.Fatal("selective dt range: expected payload-index narrowing (filter-first)")
	}
	lo := rfc3339ms(t, loS)
	hi := rfc3339ms(t, hiS)
	got := mustSearch(t, h, q, k, dtSel)
	want := bruteForceFiltered(corpus, metas, q, k, func(m Metadata) bool {
		ts := m["ts"].Int
		return ts >= lo && ts < hi
	})
	if !eqUint64(resultIDs(got), want) {
		t.Errorf("filter-first dt range != brute force: %v != %v", resultIDs(got), want)
	}

	// Match is now token-index accelerated: "fox" is on ~98% of docs but still
	// under the filter-first cap (n < threshold), so candidates narrows it to a
	// superset (ok=true). The candidate set must be a SUPERSET of the predicate
	// matches, and the end-to-end result must be EXACT vs brute force regardless
	// of which path the planner picks.
	matchF := Filter{Op: FilterMatch, Field: "title", Value: NewString("fox")}
	pred, err := matchF.Compile()
	if err != nil {
		t.Fatal(err)
	}
	cands, ok := h.payloadIdx.candidates(matchF, h.filterFirstThreshold())
	if !ok {
		t.Fatal("match filter: expected token-index narrowing (ok=true)")
	}
	candIDs := make(map[uint64]bool, len(cands))
	h.mu.RLock()
	for _, s := range cands {
		candIDs[h.arena.ID(s)] = true
	}
	h.mu.RUnlock()
	for id, m := range metas {
		if pred(m) && !candIDs[id] {
			t.Errorf("match candidate set MISSING true predicate match id %d (not a superset)", id)
		}
	}
	got2 := mustSearch(t, h, q, k, matchF)
	for _, r := range got2 {
		if !pred(metas[r.ID]) {
			t.Errorf("match result id %d violates predicate", r.ID)
		}
	}
	want2 := bruteForceFiltered(corpus, metas, q, k, func(m Metadata) bool { return pred(m) })
	if !eqUint64(resultIDs(got2), want2) {
		t.Errorf("match filter-first != brute force: %v != %v", resultIDs(got2), want2)
	}
}
