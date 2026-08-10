// SPDX-License-Identifier: Apache-2.0

package vector

import (
	"bytes"
	"math/rand"
	"sort"
	"testing"
	"time"
)

// Filter-first payload-index integration tests for MultiVectorIndex (late-
// interaction MaxSim).
//
// The central proof is that the filter-first path (an index-narrowed candidate
// doc superset brute-force-reranked with the Stage-2 live-meta predicate
// re-check) returns the SAME top-k — identical doc ids AND identical MaxSim
// scores — as the adaptive-over-fetch Stage-1 + Stage-2 post-filter fallback.
// Both paths are exercised by toggling m.payloadIdx: with the index present the
// selective filters take filter-first; nilling it forces the existing token-HNSW
// gather + post-filter path VERBATIM (mvFilterFirstCands returns ok=false).

// forcePredicateEval disables the payload index so a search takes the token-HNSW
// gather + post-filter fallback, returning a restore func. Mirrors named's helper.
func (m *MultiVectorIndex) forcePredicateEval(disable bool) (restore func()) {
	m.mu.Lock()
	saved := m.payloadIdx
	if disable {
		m.payloadIdx = nil
	}
	m.mu.Unlock()
	return func() {
		m.mu.Lock()
		m.payloadIdx = saved
		m.mu.Unlock()
	}
}

func sortedMultiIDs(res []MultiResult) []uint64 {
	ids := make([]uint64, len(res))
	for i, r := range res {
		ids[i] = r.ID
	}
	sort.Slice(ids, func(a, b int) bool { return ids[a] < ids[b] })
	return ids
}

// assertMVFilterFirstMatchesFallback runs the same MaxSim Search twice — once
// with the index (filter-first) and once with it disabled (post-filter fallback)
// — and asserts identical result id sets AND identical per-id MaxSim scores.
func assertMVFilterFirstMatchesFallback(t *testing.T, m *MultiVectorIndex, q [][]float32, k int, f Filter) []MultiResult {
	t.Helper()

	ffRes, err := m.Search(q, k, MultiSearchOpts{Filter: f})
	if err != nil {
		t.Fatalf("filter-first search: %v", err)
	}

	restore := m.forcePredicateEval(true)
	fbRes, err := m.Search(q, k, MultiSearchOpts{Filter: f})
	restore()
	if err != nil {
		t.Fatalf("fallback search: %v", err)
	}

	if !equalU64(sortedMultiIDs(ffRes), sortedMultiIDs(fbRes)) {
		t.Fatalf("filter-first ids %v != fallback ids %v", sortedMultiIDs(ffRes), sortedMultiIDs(fbRes))
	}
	score := func(rs []MultiResult) map[uint64]float32 {
		s := make(map[uint64]float32, len(rs))
		for _, r := range rs {
			s[r.ID] = r.Score
		}
		return s
	}
	sff, sfb := score(ffRes), score(fbRes)
	for id, sc := range sff {
		if sfb[id] != sc {
			t.Fatalf("id %d score filter-first %v != fallback %v", id, sc, sfb[id])
		}
	}
	return ffRes
}

// TestMVFilterFirstSameAsFallback covers the accelerated clause families: eq,
// numeric range, numeric band (And), datetime range, In, geo, and And(eq,range).
// Each must yield the SAME top-k (ids + MaxSim scores) as the post-filter
// fallback.
func TestMVFilterFirstSameAsFallback(t *testing.T) {
	const n, dim = 60, 12
	m, err := NewMultiVectorIndex(MultiVectorConfig{Dim: dim, M: 16, EfConstruction: 200, EfSearch: 128, Seed: 7})
	if err != nil {
		t.Fatalf("NewMultiVectorIndex: %v", err)
	}
	defer m.Close()

	rng := rand.New(rand.NewSource(7))
	// Datetime stored as int64 unix-ms (the dt_* ops lower their RFC3339 bound to
	// ms and compare numerically). base is the reference instant.
	base := time.Date(2024, 1, 1, 0, 0, 0, 0, time.UTC)
	baseMs := base.UnixMilli()
	for id := uint64(1); id <= uint64(n); id++ {
		cat := "even"
		if id%2 == 1 {
			cat = "odd"
		}
		meta := Metadata{
			"category": NewString(cat),
			"score":    NewInt(int64(id)),
			"ts":       NewInt(baseMs + int64(id)*3_600_000),
			"tag":      NewString("t" + string(rune('a'+id%5))),
			"loc":      NewGeo(40.0+float64(id)*0.01, -74.0+float64(id)*0.01),
		}
		toks := randTokens(rng, 2+rng.Intn(4), dim)
		if err := m.Add(id, toks, meta); err != nil {
			t.Fatalf("Add %d: %v", id, err)
		}
	}
	q := randTokens(rng, 4, dim)

	cases := []struct {
		name string
		f    Filter
	}{
		{"eq", Filter{Op: FilterEq, Field: "category", Value: NewString("odd")}},
		{"numeric-range", Filter{Op: FilterGte, Field: "score", Value: NewInt(45)}},
		{"numeric-band", Filter{Op: FilterAnd, And: []Filter{
			{Op: FilterGte, Field: "score", Value: NewInt(10)},
			{Op: FilterLte, Field: "score", Value: NewInt(20)},
		}}},
		{"datetime-range", Filter{Op: FilterDtGte, Field: "ts", Value: NewString(base.Add(45 * time.Hour).Format(time.RFC3339))}},
		{"in", Filter{Op: FilterIn, Field: "tag", Value: NewStrings([]string{"ta", "tc"})}},
		{"geo-box", Filter{Op: FilterGeoBox, Field: "loc", Geo: &GeoCondition{
			MinLat: 40.0, MinLon: -74.0, MaxLat: 40.15, MaxLon: -73.85,
		}}},
		{"and-eq-range", Filter{Op: FilterAnd, And: []Filter{
			{Op: FilterEq, Field: "category", Value: NewString("even")},
			{Op: FilterGte, Field: "score", Value: NewInt(30)},
		}}},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			res := assertMVFilterFirstMatchesFallback(t, m, q, 8, tc.f)
			if len(res) == 0 {
				t.Fatalf("%s: expected non-empty result (filter should match some docs)", tc.name)
			}
		})
	}
}

// TestMVFilterFirstMatch proves a `match` filter on a MaxSim search is now
// filter-first (the id-keyed token index narrows it through candidates) and
// returns the SAME top-k (ids + MaxSim scores) as the post-filter fallback —
// single token, match-ALL multi-token, ValueStrings multi-value, And(eq, match).
func TestMVFilterFirstMatch(t *testing.T) {
	const n, dim = 80, 12
	m, err := NewMultiVectorIndex(MultiVectorConfig{Dim: dim, M: 16, EfConstruction: 200, EfSearch: 128, Seed: 11})
	if err != nil {
		t.Fatalf("NewMultiVectorIndex: %v", err)
	}
	defer m.Close()
	rng := rand.New(rand.NewSource(11))
	vocab := []string{"alpha", "beta", "gamma", "delta", "epsilon", "zeta"}
	for id := uint64(1); id <= uint64(n); id++ {
		body := vocab[rng.Intn(len(vocab))] + " " + vocab[rng.Intn(len(vocab))] + " " + vocab[rng.Intn(len(vocab))]
		cat := "even"
		if id%2 == 1 {
			cat = "odd"
		}
		meta := Metadata{
			"body": NewString(body),
			"tags": NewStrings([]string{vocab[rng.Intn(len(vocab))], vocab[rng.Intn(len(vocab))]}),
			"cat":  NewString(cat),
		}
		if err := m.Add(id, randTokens(rng, 2+rng.Intn(4), dim), meta); err != nil {
			t.Fatalf("Add %d: %v", id, err)
		}
	}
	q := randTokens(rng, 4, dim)

	cases := []struct {
		name string
		f    Filter
	}{
		{"single-token", Filter{Op: FilterMatch, Field: "body", Value: NewString("alpha")}},
		{"multi-token", Filter{Op: FilterMatch, Field: "body", Value: NewString("alpha beta")}},
		{"value-strings", Filter{Op: FilterMatch, Field: "tags", Value: NewString("gamma")}},
		{"and-eq-match", Filter{Op: FilterAnd, And: []Filter{
			{Op: FilterEq, Field: "cat", Value: NewString("odd")},
			{Op: FilterMatch, Field: "body", Value: NewString("alpha")},
		}}},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			assertMVFilterFirstMatchesFallback(t, m, q, 8, tc.f)
		})
	}
}

// TestMVFilterFirstNonAccelerableFallsBack confirms a non-index-narrowable filter
// (regex / or / is-null) still returns correct results via the fallback
// (candidates() returns ok=false ⇒ filter-first declines).
func TestMVFilterFirstNonAccelerableFallsBack(t *testing.T) {
	const dim = 8
	m, err := NewMultiVectorIndex(MultiVectorConfig{Dim: dim, M: 16, EfConstruction: 200, EfSearch: 128, Seed: 3})
	if err != nil {
		t.Fatalf("NewMultiVectorIndex: %v", err)
	}
	defer m.Close()
	rng := rand.New(rand.NewSource(3))
	for id := uint64(1); id <= 10; id++ {
		cat := "alpha"
		if id%2 == 0 {
			cat = "beta"
		}
		if err := m.Add(id, randTokens(rng, 3, dim), Metadata{"category": NewString(cat)}); err != nil {
			t.Fatalf("Add %d: %v", id, err)
		}
	}
	q := randTokens(rng, 3, dim)

	// Regex (not accelerated): the filter-first path declines (ok=false) and the
	// post-filter fallback runs. The result must still be correct.
	regex := Filter{Op: FilterMatch, Field: "category", Value: NewString("^alpha$")}
	got := assertMVFilterFirstMatchesFallback(t, m, q, 10, regex)
	for _, r := range got {
		if r.ID%2 == 0 {
			t.Fatalf("regex ^alpha$ matched even id %d (category beta)", r.ID)
		}
	}

	// Or (not accelerated): same — declines, falls back, correct.
	or := Filter{Op: FilterOr, Or: []Filter{
		{Op: FilterEq, Field: "category", Value: NewString("alpha")},
		{Op: FilterEq, Field: "category", Value: NewString("beta")},
	}}
	all := assertMVFilterFirstMatchesFallback(t, m, q, 10, or)
	if len(all) != 10 {
		t.Fatalf("or(alpha,beta) returned %d, want all 10", len(all))
	}
}

// TestMVFilterFirstTTLReCheck proves the Stage-2 live-meta re-check drops a doc
// whose per-key TTL on the FILTERED key has expired — the index over-covers it
// (it has no notion of TTL) but the re-check rejects it (no false match).
func TestMVFilterFirstTTLReCheck(t *testing.T) {
	const dim = 8
	m, err := NewMultiVectorIndex(MultiVectorConfig{Dim: dim, M: 16, EfConstruction: 200, EfSearch: 128, Seed: 9})
	if err != nil {
		t.Fatalf("NewMultiVectorIndex: %v", err)
	}
	defer m.Close()
	var fakeNow int64 = 1_000_000
	m.now = func() int64 { return fakeNow }
	rng := rand.New(rand.NewSource(9))

	// ids 1,2: live, no per-key TTL. id 3: per-key TTL on the filtered key
	// "category" that will expire.
	if err := m.Add(1, randTokens(rng, 3, dim), Metadata{"category": NewString("x")}); err != nil {
		t.Fatalf("add 1: %v", err)
	}
	if err := m.Add(2, randTokens(rng, 3, dim), Metadata{"category": NewString("x")}); err != nil {
		t.Fatalf("add 2: %v", err)
	}
	if _, err := m.AddCASKeyTTL(3, randTokens(rng, 3, dim), Metadata{"category": NewString("x")},
		map[string]int64{"category": 50}, CASCond{}); err != nil {
		t.Fatalf("add 3: %v", err)
	}
	q := randTokens(rng, 3, dim)
	f := Filter{Op: FilterEq, Field: "category", Value: NewString("x")}

	// Before expiry: all three match (and filter-first == fallback).
	got := assertMVFilterFirstMatchesFallback(t, m, q, 10, f)
	if len(got) != 3 {
		t.Fatalf("before expiry got %d, want 3", len(got))
	}

	// Advance past id 3's per-key deadline: its "category" key is now logically
	// expired. The index still lists id 3 under category=x (over-cover) but the
	// Stage-2 live-meta re-check drops it — filter-first stays == fallback.
	fakeNow += 100
	got = assertMVFilterFirstMatchesFallback(t, m, q, 10, f)
	if !equalU64(sortedMultiIDs(got), []uint64{1, 2}) {
		t.Fatalf("after key expiry got %v, want ids [1 2] (id 3's category key expired)", sortedMultiIDs(got))
	}
}

// TestMVFilterFirstMutationUpdatesIndex proves payload mutations keep the index
// correct: set_payload / overwrite that drop the filtered value EXCLUDE the doc;
// a delete removes it.
func TestMVFilterFirstMutationUpdatesIndex(t *testing.T) {
	const dim = 8
	m, err := NewMultiVectorIndex(MultiVectorConfig{Dim: dim, M: 16, EfConstruction: 200, EfSearch: 128, Seed: 4})
	if err != nil {
		t.Fatalf("NewMultiVectorIndex: %v", err)
	}
	defer m.Close()
	rng := rand.New(rand.NewSource(4))
	for id := uint64(1); id <= 5; id++ {
		if err := m.Add(id, randTokens(rng, 3, dim), Metadata{"category": NewString("x")}); err != nil {
			t.Fatalf("add %d: %v", id, err)
		}
	}
	q := randTokens(rng, 3, dim)
	f := Filter{Op: FilterEq, Field: "category", Value: NewString("x")}

	if got := assertMVFilterFirstMatchesFallback(t, m, q, 10, f); len(got) != 5 {
		t.Fatalf("initial got %d, want 5", len(got))
	}

	// set_payload moves id 1 to category=y → no longer matches category=x.
	if err := m.SetPayload(1, Metadata{"category": NewString("y")}, nil); err != nil {
		t.Fatalf("set_payload 1: %v", err)
	}
	got := assertMVFilterFirstMatchesFallback(t, m, q, 10, f)
	for _, r := range got {
		if r.ID == 1 {
			t.Fatalf("id 1 still matched after set_payload to category=y")
		}
	}
	if len(got) != 4 {
		t.Fatalf("after set_payload got %d, want 4", len(got))
	}

	// overwrite id 2 to a payload WITHOUT category → no longer matches.
	if err := m.OverwritePayload(2, Metadata{"other": NewString("z")}, nil); err != nil {
		t.Fatalf("overwrite 2: %v", err)
	}
	got = assertMVFilterFirstMatchesFallback(t, m, q, 10, f)
	if len(got) != 3 {
		t.Fatalf("after overwrite got %d, want 3", len(got))
	}

	// delete id 3 → removed from the index.
	if !m.Delete(3) {
		t.Fatalf("delete 3 returned false")
	}
	got = assertMVFilterFirstMatchesFallback(t, m, q, 10, f)
	want := []uint64{4, 5}
	if !equalU64(sortedMultiIDs(got), want) {
		t.Fatalf("after delete ids %v, want %v", sortedMultiIDs(got), want)
	}
}

// TestMVFilterFirstRebuildOnRestore proves the index repopulates on snapshot
// save+restore (it is never serialized) so filter-first stays correct.
func TestMVFilterFirstRebuildOnRestore(t *testing.T) {
	const dim = 8
	cfg := MultiVectorConfig{Dim: dim, M: 16, EfConstruction: 200, EfSearch: 128, Seed: 6}
	m, err := NewMultiVectorIndex(cfg)
	if err != nil {
		t.Fatalf("NewMultiVectorIndex: %v", err)
	}
	rng := rand.New(rand.NewSource(6))
	for id := uint64(1); id <= 6; id++ {
		cat := "x"
		if id > 3 {
			cat = "y"
		}
		if err := m.Add(id, randTokens(rng, 3, dim), Metadata{"category": NewString(cat)}); err != nil {
			t.Fatalf("add %d: %v", id, err)
		}
	}
	q := randTokens(rng, 3, dim)

	var buf bytes.Buffer
	if err := m.snapshot(&buf); err != nil {
		t.Fatalf("snapshot: %v", err)
	}
	m.Close()

	restored, err := NewMultiVectorIndex(cfg)
	if err != nil {
		t.Fatalf("new restored: %v", err)
	}
	defer restored.Close()
	if err := restored.restore(bytes.NewReader(buf.Bytes())); err != nil {
		t.Fatalf("restore: %v", err)
	}

	// The restored index must drive filter-first identically to the fallback.
	f := Filter{Op: FilterEq, Field: "category", Value: NewString("x")}
	got := assertMVFilterFirstMatchesFallback(t, restored, q, 10, f)
	if !equalU64(sortedMultiIDs(got), []uint64{1, 2, 3}) {
		t.Fatalf("restored filter-first ids %v, want [1 2 3]", sortedMultiIDs(got))
	}
}

// TestMVFilterFirstNoFilterAndNonSelectiveUnchanged confirms a no-filter search
// and a non-selective filter (matches nearly everything → over-fetch fallback)
// are both correct and unaffected by the filter-first path.
func TestMVFilterFirstNoFilterAndNonSelectiveUnchanged(t *testing.T) {
	const n, dim = 30, 10
	m, err := NewMultiVectorIndex(MultiVectorConfig{Dim: dim, M: 16, EfConstruction: 200, EfSearch: 128, Seed: 2})
	if err != nil {
		t.Fatalf("NewMultiVectorIndex: %v", err)
	}
	defer m.Close()
	rng := rand.New(rand.NewSource(2))
	for id := uint64(1); id <= uint64(n); id++ {
		if err := m.Add(id, randTokens(rng, 3, dim), Metadata{"category": NewString("x")}); err != nil {
			t.Fatalf("add %d: %v", id, err)
		}
	}
	q := randTokens(rng, 3, dim)

	// No-filter search: never enters the filter-first path; returns top-k by MaxSim.
	res, err := m.Search(q, 5, MultiSearchOpts{})
	if err != nil {
		t.Fatalf("no-filter search: %v", err)
	}
	if len(res) != 5 {
		t.Fatalf("no-filter got %d, want 5", len(res))
	}

	// Non-selective filter (category=x matches all 30): too-many candidates ⇒ the
	// cost gate falls back to the over-fetch path. Result still == the fallback.
	f := Filter{Op: FilterEq, Field: "category", Value: NewString("x")}
	got := assertMVFilterFirstMatchesFallback(t, m, q, 5, f)
	if len(got) != 5 {
		t.Fatalf("non-selective got %d, want 5", len(got))
	}
}
