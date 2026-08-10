// SPDX-License-Identifier: Apache-2.0

package vector

import (
	"bytes"
	"math/rand"
	"sort"
	"testing"
	"time"
)

// Filter-first payload-index integration tests for NamedCollection.
//
// The central proof is that the filter-first path (index-narrowed brute-force
// scoring with the live-meta predicate re-check) returns the SAME top-k as the
// predicate-eval graph fallback. Both paths are exercised by toggling
// nc.payloadIdx: with the index present the selective filters take filter-first;
// nilling it forces the existing SearchFilteredWith fallback VERBATIM.

// withFallback runs fn with the payload index disabled so the call takes the
// predicate-eval graph fallback, then restores it. Returns nothing; fn captures.
func (nc *NamedCollection) forcePredicateEval(disable bool) (restore func()) {
	nc.mu.Lock()
	saved := nc.payloadIdx
	if disable {
		nc.payloadIdx = nil
	}
	nc.mu.Unlock()
	return func() {
		nc.mu.Lock()
		nc.payloadIdx = saved
		nc.mu.Unlock()
	}
}

func sortedResultIDs(res []Result) []uint64 {
	ids := make([]uint64, len(res))
	for i, r := range res {
		ids[i] = r.ID
	}
	sort.Slice(ids, func(a, b int) bool { return ids[a] < ids[b] })
	return ids
}

func equalU64(a, b []uint64) bool {
	if len(a) != len(b) {
		return false
	}
	for i := range a {
		if a[i] != b[i] {
			return false
		}
	}
	return true
}

// assertFilterFirstMatchesFallback runs the same SearchNamed twice — once with
// the index (filter-first) and once with it disabled (predicate-eval fallback) —
// and asserts identical result id sets AND identical per-id distances.
func assertFilterFirstMatchesFallback(t *testing.T, nc *NamedCollection, space string, q []float32, k int, f Filter) []Result {
	t.Helper()

	ffRes, err := nc.SearchNamed(space, q, k, f)
	if err != nil {
		t.Fatalf("filter-first search: %v", err)
	}

	restore := nc.forcePredicateEval(true)
	fbRes, err := nc.SearchNamed(space, q, k, f)
	restore()
	if err != nil {
		t.Fatalf("fallback search: %v", err)
	}

	if !equalU64(sortedResultIDs(ffRes), sortedResultIDs(fbRes)) {
		t.Fatalf("filter-first ids %v != fallback ids %v", sortedResultIDs(ffRes), sortedResultIDs(fbRes))
	}
	// Per-id distances must match exactly (both compute the same metric on the same
	// stored vectors). Build id->dist maps and compare.
	dist := func(rs []Result) map[uint64]float32 {
		m := make(map[uint64]float32, len(rs))
		for _, r := range rs {
			m[r.ID] = r.Distance
		}
		return m
	}
	dff, dfb := dist(ffRes), dist(fbRes)
	for id, d := range dff {
		if dfb[id] != d {
			t.Fatalf("id %d distance filter-first %v != fallback %v", id, d, dfb[id])
		}
	}
	return ffRes
}

// TestNamedFilterFirstSameAsFallback covers the accelerated clause families:
// eq, numeric range, datetime range, In, geo — each must yield the SAME top-k as
// the predicate-eval fallback.
func TestNamedFilterFirstSameAsFallback(t *testing.T) {
	nc, err := NewNamedCollection("default/named", map[string]NamedVectorParams{
		"title": {Dim: 4, Metric: L2},
	})
	if err != nil {
		t.Fatalf("new: %v", err)
	}
	defer nc.Close()

	// 40 points spread across category/score/ts/tag/loc so the filters are
	// selective (filter-first preferred) but every clause family has live matches.
	base := time.Date(2024, 1, 1, 0, 0, 0, 0, time.UTC)
	for id := uint64(1); id <= 40; id++ {
		cat := "even"
		if id%2 == 1 {
			cat = "odd"
		}
		// Datetime fields are stored as int64 unix-ms by convention (the dt_* ops
		// lower their RFC3339 bound to unix-ms and compare numerically).
		tsMs := base.Add(time.Duration(id) * time.Hour).UnixMilli()
		meta := Metadata{
			"category": NewString(cat),
			"score":    NewInt(int64(id)),
			"ts":       NewInt(tsMs),
			"tag":      NewString("t" + string(rune('a'+id%5))),
			"loc":      NewGeo(40.0+float64(id)*0.01, -74.0+float64(id)*0.01),
		}
		if err := nc.Insert(id, map[string][]float32{
			"title": {float32(id), float32(id % 3), 0, 1},
		}, meta, 0); err != nil {
			t.Fatalf("insert %d: %v", id, err)
		}
	}
	q := []float32{5, 1, 0, 1}

	cases := []struct {
		name string
		f    Filter
	}{
		{"eq", Filter{Op: FilterEq, Field: "category", Value: NewString("odd")}},
		{"numeric-range", Filter{Op: FilterGte, Field: "score", Value: NewInt(30)}},
		{"numeric-band", Filter{Op: FilterAnd, And: []Filter{
			{Op: FilterGte, Field: "score", Value: NewInt(10)},
			{Op: FilterLte, Field: "score", Value: NewInt(20)},
		}}},
		{"datetime-range", Filter{Op: FilterDtGte, Field: "ts", Value: NewString(base.Add(30 * time.Hour).Format(time.RFC3339))}},
		{"in", Filter{Op: FilterIn, Field: "tag", Value: NewStrings([]string{"ta", "tc"})}},
		{"geo-box", Filter{Op: FilterGeoBox, Field: "loc", Geo: &GeoCondition{
			MinLat: 40.0, MinLon: -74.0, MaxLat: 40.2, MaxLon: -73.8,
		}}},
		{"and-eq-range", Filter{Op: FilterAnd, And: []Filter{
			{Op: FilterEq, Field: "category", Value: NewString("even")},
			{Op: FilterGte, Field: "score", Value: NewInt(20)},
		}}},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			res := assertFilterFirstMatchesFallback(t, nc, "title", q, 8, tc.f)
			if len(res) == 0 {
				t.Fatalf("%s: expected non-empty result (filter should match some points)", tc.name)
			}
		})
	}
}

// TestNamedFilterFirstMatch proves a `match` filter on a named search is now
// filter-first (the id-keyed token index narrows it through candidates) and
// returns the SAME top-k as the predicate-only fallback — single token, match-
// ALL multi-token, ValueStrings multi-value, and And(eq, match).
func TestNamedFilterFirstMatch(t *testing.T) {
	nc, err := NewNamedCollection("default/named", map[string]NamedVectorParams{
		"title": {Dim: 4, Metric: L2},
	})
	if err != nil {
		t.Fatalf("new: %v", err)
	}
	defer nc.Close()

	rng := rand.New(rand.NewSource(11))
	vocab := []string{"alpha", "beta", "gamma", "delta", "epsilon", "zeta"}
	for id := uint64(1); id <= 60; id++ {
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
		if err := nc.Insert(id, map[string][]float32{"title": {float32(id), float32(id % 3), 0, 1}}, meta, 0); err != nil {
			t.Fatalf("insert %d: %v", id, err)
		}
	}
	q := []float32{5, 1, 0, 1}

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
			// filter-first == predicate-only (the index narrows, the re-check gates).
			assertFilterFirstMatchesFallback(t, nc, "title", q, 8, tc.f)
		})
	}
}

// TestNamedFilterFirstNonAccelerableFallsBack confirms a non-index-narrowable
// filter (regex / or / is-null) still returns correct results via the fallback
// (candidates() returns ok=false ⇒ filter-first declines).
func TestNamedFilterFirstNonAccelerableFallsBack(t *testing.T) {
	nc, err := NewNamedCollection("default/named", map[string]NamedVectorParams{
		"title": {Dim: 4, Metric: L2},
	})
	if err != nil {
		t.Fatalf("new: %v", err)
	}
	defer nc.Close()
	for id := uint64(1); id <= 10; id++ {
		cat := "alpha"
		if id%2 == 0 {
			cat = "beta"
		}
		if err := nc.Insert(id, map[string][]float32{"title": {float32(id), 0, 0, 1}},
			Metadata{"category": NewString(cat)}, 0); err != nil {
			t.Fatalf("insert %d: %v", id, err)
		}
	}
	q := []float32{3, 0, 0, 1}

	// Regex (not accelerated): the filter-first path declines (ok=false) and the
	// fallback predicate-eval runs. The result must still be correct.
	regex := Filter{Op: FilterMatch, Field: "category", Value: NewString("^alpha$")}
	got := assertFilterFirstMatchesFallback(t, nc, "title", q, 10, regex)
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
	all := assertFilterFirstMatchesFallback(t, nc, "title", q, 10, or)
	if len(all) != 10 {
		t.Fatalf("or(alpha,beta) returned %d, want all 10", len(all))
	}
}

// TestNamedFilterFirstTTLReCheck proves the live-meta re-check drops an expired
// point and an expired per-key TTL key that the index over-covers (the index has
// no notion of TTL — correctness comes from the re-check).
func TestNamedFilterFirstTTLReCheck(t *testing.T) {
	nc, err := NewNamedCollection("default/named", map[string]NamedVectorParams{
		"title": {Dim: 4, Metric: L2},
	})
	if err != nil {
		t.Fatalf("new: %v", err)
	}
	defer nc.Close()
	var fakeNow int64 = 1_000_000
	nc.now = func() int64 { return fakeNow }

	// id 1: live. id 2: live. Both match category=x with no per-key TTL.
	if err := nc.Insert(1, map[string][]float32{"title": {1, 0, 0, 1}},
		Metadata{"category": NewString("x")}, 0); err != nil {
		t.Fatalf("insert 1: %v", err)
	}
	if err := nc.Insert(2, map[string][]float32{"title": {2, 0, 0, 1}},
		Metadata{"category": NewString("x")}, 0); err != nil {
		t.Fatalf("insert 2: %v", err)
	}
	// id 3: per-key TTL on the FILTERED key "category" that will expire. Once the
	// key expires the live-meta view drops it, so category=x no longer matches —
	// even though the payload index still lists id 3 (it has no notion of TTL).
	// The re-check is what makes filter-first correct here.
	if _, err := nc.InsertCASKeyTTL(3, map[string][]float32{"title": {3, 0, 0, 1}}, nil,
		Metadata{"category": NewString("x")}, 0,
		map[string]int64{"category": 50}, CASCond{}); err != nil {
		t.Fatalf("insert 3: %v", err)
	}

	f := Filter{Op: FilterEq, Field: "category", Value: NewString("x")}

	// Before expiry: all three match (and filter-first == fallback).
	got := assertFilterFirstMatchesFallback(t, nc, "title", []float32{1, 0, 0, 1}, 10, f)
	if len(got) != 3 {
		t.Fatalf("before expiry got %d, want 3", len(got))
	}

	// Advance past id 3's per-key deadline: its "category" key is now logically
	// expired. The index still lists id 3 under category=x (over-cover) but the
	// live-meta re-check drops it — filter-first stays == fallback.
	fakeNow += 100
	got = assertFilterFirstMatchesFallback(t, nc, "title", []float32{1, 0, 0, 1}, 10, f)
	if !equalU64(sortedResultIDs(got), []uint64{1, 2}) {
		t.Fatalf("after key expiry got %v, want ids [1 2] (id 3's category key expired)", sortedResultIDs(got))
	}
}

// TestNamedFilterFirstMutationUpdatesIndex proves payload mutations keep the
// index correct: set_payload / overwrite that drop the filtered value EXCLUDE
// the point; a delete removes it.
func TestNamedFilterFirstMutationUpdatesIndex(t *testing.T) {
	nc, err := NewNamedCollection("default/named", map[string]NamedVectorParams{
		"title": {Dim: 4, Metric: L2},
	})
	if err != nil {
		t.Fatalf("new: %v", err)
	}
	defer nc.Close()
	for id := uint64(1); id <= 5; id++ {
		if err := nc.Insert(id, map[string][]float32{"title": {float32(id), 0, 0, 1}},
			Metadata{"category": NewString("x")}, 0); err != nil {
			t.Fatalf("insert %d: %v", id, err)
		}
	}
	q := []float32{1, 0, 0, 1}
	f := Filter{Op: FilterEq, Field: "category", Value: NewString("x")}

	if got := assertFilterFirstMatchesFallback(t, nc, "title", q, 10, f); len(got) != 5 {
		t.Fatalf("initial got %d, want 5", len(got))
	}

	// set_payload moves id 1 to category=y → no longer matches category=x.
	if err := nc.SetPayload(1, Metadata{"category": NewString("y")}, nil); err != nil {
		t.Fatalf("set_payload 1: %v", err)
	}
	got := assertFilterFirstMatchesFallback(t, nc, "title", q, 10, f)
	for _, r := range got {
		if r.ID == 1 {
			t.Fatalf("id 1 still matched after set_payload to category=y")
		}
	}
	if len(got) != 4 {
		t.Fatalf("after set_payload got %d, want 4", len(got))
	}

	// overwrite id 2 to a payload WITHOUT category → no longer matches.
	if err := nc.OverwritePayload(2, Metadata{"other": NewString("z")}, nil); err != nil {
		t.Fatalf("overwrite 2: %v", err)
	}
	got = assertFilterFirstMatchesFallback(t, nc, "title", q, 10, f)
	if len(got) != 3 {
		t.Fatalf("after overwrite got %d, want 3", len(got))
	}

	// delete id 3 → removed from the index.
	if _, err := nc.Delete(3); err != nil {
		t.Fatalf("delete 3: %v", err)
	}
	got = assertFilterFirstMatchesFallback(t, nc, "title", q, 10, f)
	if len(got) != 2 {
		t.Fatalf("after delete got %d, want 2 (ids 4,5)", len(got))
	}
	want := []uint64{4, 5}
	if !equalU64(sortedResultIDs(got), want) {
		t.Fatalf("after delete ids %v, want %v", sortedResultIDs(got), want)
	}
}

// TestNamedFilterFirstRebuildOnRestore proves the index repopulates on snapshot
// save+restore (it is never serialized) so filter-first stays correct.
func TestNamedFilterFirstRebuildOnRestore(t *testing.T) {
	cfg := map[string]NamedVectorParams{"title": {Dim: 4, Metric: L2}}
	nc, err := NewNamedCollection("default/named", cfg)
	if err != nil {
		t.Fatalf("new: %v", err)
	}
	for id := uint64(1); id <= 6; id++ {
		cat := "x"
		if id > 3 {
			cat = "y"
		}
		if err := nc.Insert(id, map[string][]float32{"title": {float32(id), 0, 0, 1}},
			Metadata{"category": NewString(cat)}, 0); err != nil {
			t.Fatalf("insert %d: %v", id, err)
		}
	}

	var buf bytes.Buffer
	if err := nc.Snapshot(&buf); err != nil {
		t.Fatalf("snapshot: %v", err)
	}
	nc.Close()

	restored, err := NewNamedCollection("default/named", cfg)
	if err != nil {
		t.Fatalf("new restored: %v", err)
	}
	defer restored.Close()
	if err := restored.Restore(bytes.NewReader(buf.Bytes())); err != nil {
		t.Fatalf("restore: %v", err)
	}

	// The restored index must drive filter-first identically to the fallback.
	q := []float32{1, 0, 0, 1}
	f := Filter{Op: FilterEq, Field: "category", Value: NewString("x")}
	got := assertFilterFirstMatchesFallback(t, restored, "title", q, 10, f)
	if !equalU64(sortedResultIDs(got), []uint64{1, 2, 3}) {
		t.Fatalf("restored filter-first ids %v, want [1 2 3]", sortedResultIDs(got))
	}
}

// TestNamedFilterFirstNoFilterUnchanged confirms a no-filter search is unaffected
// (it never enters the filter-first path).
func TestNamedFilterFirstNoFilterUnchanged(t *testing.T) {
	nc, err := NewNamedCollection("default/named", map[string]NamedVectorParams{
		"title": {Dim: 4, Metric: L2},
	})
	if err != nil {
		t.Fatalf("new: %v", err)
	}
	defer nc.Close()
	for id := uint64(1); id <= 5; id++ {
		if err := nc.Insert(id, map[string][]float32{"title": {float32(id), 0, 0, 1}},
			Metadata{"category": NewString("x")}, 0); err != nil {
			t.Fatalf("insert %d: %v", id, err)
		}
	}
	res, err := nc.SearchNamed("title", []float32{1, 0, 0, 1}, 5, Filter{})
	if err != nil {
		t.Fatalf("no-filter search: %v", err)
	}
	if len(res) != 5 {
		t.Fatalf("no-filter got %d, want 5", len(res))
	}
	if res[0].ID != 1 {
		t.Fatalf("no-filter nearest = %d, want 1", res[0].ID)
	}
}
