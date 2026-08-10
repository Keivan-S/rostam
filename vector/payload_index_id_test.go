// SPDX-License-Identifier: Apache-2.0

package vector

import (
	"math/rand"
	"sort"
	"testing"
)

// idSet returns the candidate ids as a sorted set, asserting ok==wantOK.
func idCandidates(t *testing.T, p *payloadIndexID, f Filter, limit int, wantOK bool) []uint64 {
	t.Helper()
	ids, ok := p.candidates(f, limit)
	if ok != wantOK {
		t.Fatalf("candidates ok = %v, want %v (filter op=%v)", ok, wantOK, f.Op)
	}
	return sortedIDs(ids)
}

// bruteMatch returns the sorted ids whose metadata satisfies the compiled
// filter predicate — the exact ground-truth match set.
func bruteMatch(t *testing.T, metas map[uint64]Metadata, f Filter) []uint64 {
	t.Helper()
	pred, err := f.Compile()
	if err != nil {
		t.Fatalf("compile filter: %v", err)
	}
	var out []uint64
	for id, m := range metas {
		if pred == nil || pred(m) {
			out = append(out, id)
		}
	}
	return sortedIDs(out)
}

// isSuperset reports whether super ⊇ sub (both sorted).
func isSuperset(super, sub []uint64) bool {
	have := make(map[uint64]struct{}, len(super))
	for _, x := range super {
		have[x] = struct{}{}
	}
	for _, x := range sub {
		if _, ok := have[x]; !ok {
			return false
		}
	}
	return true
}

func TestPayloadIndexIDReindexAddUpdateRemove(t *testing.T) {
	p := newPayloadIndexID()

	// add
	p.reindex(1, Metadata{"color": NewString("red"), "size": NewInt(10)})
	p.reindex(2, Metadata{"color": NewString("red"), "size": NewInt(20)})
	p.reindex(3, Metadata{"color": NewString("blue"), "size": NewInt(10)})

	red := idCandidates(t, p, Filter{Op: FilterEq, Field: "color", Value: NewString("red")}, 1000, true)
	if !eqUint64(red, []uint64{1, 2}) {
		t.Fatalf("color=red got %v want [1 2]", red)
	}

	// update id 1's color red -> green; its OLD red posting must be removed.
	p.reindex(1, Metadata{"color": NewString("green"), "size": NewInt(10)})
	red = idCandidates(t, p, Filter{Op: FilterEq, Field: "color", Value: NewString("red")}, 1000, true)
	if !eqUint64(red, []uint64{2}) {
		t.Fatalf("after update color=red got %v want [2] (stale OLD posting not dropped)", red)
	}
	green := idCandidates(t, p, Filter{Op: FilterEq, Field: "color", Value: NewString("green")}, 1000, true)
	if !eqUint64(green, []uint64{1}) {
		t.Fatalf("color=green got %v want [1]", green)
	}

	// remove id 2 via nil meta (pure removal)
	p.reindex(2, nil)
	red = idCandidates(t, p, Filter{Op: FilterEq, Field: "color", Value: NewString("red")}, 1000, true)
	if len(red) != 0 {
		t.Fatalf("after remove color=red got %v want []", red)
	}
	if _, ok := p.idKeys[2]; ok {
		t.Fatalf("idKeys still has removed id 2")
	}

	// size=10 should still see ids 1 and 3
	s10 := idCandidates(t, p, Filter{Op: FilterEq, Field: "size", Value: NewInt(10)}, 1000, true)
	if !eqUint64(s10, []uint64{1, 3}) {
		t.Fatalf("size=10 got %v want [1 3]", s10)
	}
}

func TestPayloadIndexIDRebuildFromMap(t *testing.T) {
	p := newPayloadIndexID()
	// pre-existing garbage that rebuild must clear
	p.reindex(99, Metadata{"color": NewString("ghost")})

	metas := map[uint64]Metadata{
		1: {"color": NewString("red"), "size": NewInt(5)},
		2: {"color": NewString("blue"), "size": NewInt(7)},
		3: {"color": NewString("red"), "size": NewInt(9)},
	}
	p.rebuild(metas)

	ghost := idCandidates(t, p, Filter{Op: FilterEq, Field: "color", Value: NewString("ghost")}, 1000, true)
	if len(ghost) != 0 {
		t.Fatalf("rebuild did not clear old state: ghost=%v", ghost)
	}
	red := idCandidates(t, p, Filter{Op: FilterEq, Field: "color", Value: NewString("red")}, 1000, true)
	if !eqUint64(red, []uint64{1, 3}) {
		t.Fatalf("rebuilt color=red got %v want [1 3]", red)
	}
}

// TestPayloadIndexIDContentFieldSkipped pins that $content is not indexed AND —
// the part this test used to get wrong — that a filter on it is reported NOT
// NARROWABLE rather than narrowed to the empty set.
//
// The original assertion was `wantOK=true` with an empty candidate list, i.e. it
// asserted "the index answers this filter, and the answer is: nothing matches".
// That reads as a reasonable encoding of "content is not indexed", and it is the
// bug: an empty candidate set with ok=true is a promise to the caller that it
// may skip the predicate, so the planner brute-forced zero candidates and
// returned nothing for filters every row satisfied. The postings are empty
// because reindex never writes them, not because nothing matches — a distinction
// ok=false is the only way to express.
//
// So the expectation flips deliberately: ok=false, meaning "I cannot help with
// this one, go evaluate the predicate". Do not flip it back.
func TestPayloadIndexIDContentFieldSkipped(t *testing.T) {
	p := newPayloadIndexID()
	p.reindex(1, Metadata{contentField: NewString("a long document body"), "tag": NewString("x")})
	// content must NOT be indexed -> a filter on it declines to narrow entirely
	idCandidates(t, p, Filter{Op: FilterEq, Field: contentField, Value: NewString("a long document body")}, 1000, false)
	// ...and so must every other narrowing op on it.
	idCandidates(t, p, Filter{Op: FilterMatch, Field: contentField, Value: NewString("document")}, 1000, false)
	idCandidates(t, p, Filter{Op: FilterIn, Field: contentField, Value: NewStrings([]string{"a long document body"})}, 1000, false)
	idCandidates(t, p, Filter{Op: FilterGt, Field: contentField, Value: NewString("a")}, 1000, false)

	// An ordinary field is unaffected, including as $content's sibling in an And:
	// declining the content conjunct must not disable narrowing on the rest.
	tag := idCandidates(t, p, Filter{Op: FilterEq, Field: "tag", Value: NewString("x")}, 1000, true)
	if !eqUint64(tag, []uint64{1}) {
		t.Fatalf("tag=x got %v want [1]", tag)
	}
	both := idCandidates(t, p, Filter{Op: FilterAnd, And: []Filter{
		{Op: FilterEq, Field: "tag", Value: NewString("x")},
		{Op: FilterMatch, Field: contentField, Value: NewString("document")},
	}}, 1000, true)
	if !eqUint64(both, []uint64{1}) {
		t.Fatalf("And(tag=x, match($content)) got %v want [1] — the content conjunct "+
			"should be dropped from the plan, not empty the whole intersection", both)
	}
}

func TestPayloadIndexIDScalarKeyEdgeCases(t *testing.T) {
	p := newPayloadIndexID()
	// ValueNone and slice values are declined from the scalar index.
	p.reindex(1, Metadata{
		"none":  {Kind: ValueNone},
		"slice": NewStrings([]string{"a", "b"}),
		"ok":    NewInt(42),
	})
	if len(p.idKeys[1]) != 1 {
		t.Fatalf("expected only 1 indexed key (the int), got %v", p.idKeys[1])
	}
	// eq on the declined fields cannot narrow to a match (the field never entered
	// the scalar index). Eq with a slice value is itself not equality-indexable.
	if _, ok := p.candidates(Filter{Op: FilterEq, Field: "slice", Value: NewStrings([]string{"a", "b"})}, 1000); ok {
		t.Fatalf("eq on slice value should not be index-narrowable (scalarKeyOf declines)")
	}
}

func TestPayloadIndexIDCandidatesEqExact(t *testing.T) {
	p := newPayloadIndexID()
	metas := map[uint64]Metadata{
		1: {"c": NewString("red"), "n": NewInt(1)},
		2: {"c": NewString("red"), "n": NewInt(2)},
		3: {"c": NewString("blue"), "n": NewInt(1)},
		4: {"c": NewString("red"), "n": NewInt(1)},
	}
	p.rebuild(metas)

	// And(eq,eq) is exact -> equality with brute-force.
	f := Filter{Op: FilterAnd, And: []Filter{
		{Op: FilterEq, Field: "c", Value: NewString("red")},
		{Op: FilterEq, Field: "n", Value: NewInt(1)},
	}}
	got := idCandidates(t, p, f, 1000, true)
	want := bruteMatch(t, metas, f)
	if !eqUint64(got, want) {
		t.Fatalf("eq-and: got %v want %v", got, want)
	}
}

func TestPayloadIndexIDCandidatesNumericRange(t *testing.T) {
	p := newPayloadIndexID()
	metas := map[uint64]Metadata{}
	for id := uint64(1); id <= 20; id++ {
		metas[id] = Metadata{"v": NewInt(int64(id))}
	}
	p.rebuild(metas)

	cases := []struct {
		op FilterOp
	}{{FilterGt}, {FilterGte}, {FilterLt}, {FilterLte}}
	for _, c := range cases {
		f := Filter{Op: c.op, Field: "v", Value: NewInt(10)}
		got := idCandidates(t, p, f, 1000, true)
		want := bruteMatch(t, metas, f)
		if !eqUint64(got, want) {
			t.Fatalf("range %v: got %v want %v", c.op, got, want)
		}
	}
}

func TestPayloadIndexIDCandidatesDatetimeRange(t *testing.T) {
	p := newPayloadIndexID()
	// datetime fields are stored as int64 unix-ms.
	metas := map[uint64]Metadata{
		1: {"ts": NewInt(mustMs(t, "2020-01-01T00:00:00Z"))},
		2: {"ts": NewInt(mustMs(t, "2021-01-01T00:00:00Z"))},
		3: {"ts": NewInt(mustMs(t, "2022-01-01T00:00:00Z"))},
		4: {"ts": NewInt(mustMs(t, "2023-01-01T00:00:00Z"))},
	}
	p.rebuild(metas)

	f := Filter{Op: FilterDtGte, Field: "ts", Value: NewString("2022-01-01T00:00:00Z")}
	got := idCandidates(t, p, f, 1000, true)
	want := bruteMatch(t, metas, f)
	if !eqUint64(got, want) {
		t.Fatalf("dt_gte: got %v want %v", got, want)
	}
}

func mustMs(t *testing.T, rfc string) int64 {
	t.Helper()
	ms, ok := datetimeBound(NewString(rfc))
	if !ok {
		t.Fatalf("datetimeBound(%q) failed", rfc)
	}
	return ms
}

func TestPayloadIndexIDCandidatesIn(t *testing.T) {
	p := newPayloadIndexID()
	metas := map[uint64]Metadata{
		1: {"c": NewString("red")},
		2: {"c": NewString("blue")},
		3: {"c": NewString("green")},
		4: {"c": NewString("red")},
	}
	p.rebuild(metas)

	f := Filter{Op: FilterIn, Field: "c", Value: NewStrings([]string{"red", "green"})}
	got := idCandidates(t, p, f, 1000, true)
	want := bruteMatch(t, metas, f)
	if !eqUint64(got, want) {
		t.Fatalf("in: got %v want %v", got, want)
	}
}

func TestPayloadIndexIDCandidatesGeo(t *testing.T) {
	p := newPayloadIndexID()
	// Cluster around San Francisco; one point far away (NYC).
	metas := map[uint64]Metadata{
		1: {"loc": NewGeo(37.7749, -122.4194)},
		2: {"loc": NewGeo(37.7849, -122.4094)},
		3: {"loc": NewGeo(37.3382, -121.8863)}, // San Jose ~70km
		4: {"loc": NewGeo(40.7128, -74.0060)},  // NYC
	}
	p.rebuild(metas)

	f := Filter{Op: FilterGeoRadius, Field: "loc", Geo: &GeoCondition{
		CenterLat: 37.7749, CenterLon: -122.4194, RadiusM: 5000,
	}}
	got := idCandidates(t, p, f, 1000, true)
	want := bruteMatch(t, metas, f)
	if !isSuperset(got, want) {
		t.Fatalf("geo candidates %v not a superset of true match %v", got, want)
	}
	// SF points must be present; NYC must be excluded by the geohash narrowing.
	if !isSuperset(got, []uint64{1, 2}) {
		t.Fatalf("geo superset missing nearby points: %v", got)
	}
	for _, id := range got {
		if id == 4 {
			t.Fatalf("geo superset wrongly includes far point 4 (NYC): %v", got)
		}
	}
}

func TestPayloadIndexIDCandidatesAndGeoEq(t *testing.T) {
	p := newPayloadIndexID()
	metas := map[uint64]Metadata{
		1: {"loc": NewGeo(37.7749, -122.4194), "open": NewBool(true)},
		2: {"loc": NewGeo(37.7849, -122.4094), "open": NewBool(false)},
		3: {"loc": NewGeo(40.7128, -74.0060), "open": NewBool(true)},
	}
	p.rebuild(metas)

	f := Filter{Op: FilterAnd, And: []Filter{
		{Op: FilterGeoRadius, Field: "loc", Geo: &GeoCondition{CenterLat: 37.7749, CenterLon: -122.4194, RadiusM: 5000}},
		{Op: FilterEq, Field: "open", Value: NewBool(true)},
	}}
	got := idCandidates(t, p, f, 1000, true)
	want := bruteMatch(t, metas, f)
	if !isSuperset(got, want) {
		t.Fatalf("and(geo,eq) %v not superset of %v", got, want)
	}
	// id 1 (SF, open) must be in; id 3 (NYC) and id 2 (closed) excluded.
	if !isSuperset(got, []uint64{1}) {
		t.Fatalf("and(geo,eq) missing true match 1: %v", got)
	}
}

func TestPayloadIndexIDNonAccelerable(t *testing.T) {
	p := newPayloadIndexID()
	p.rebuild(map[uint64]Metadata{
		1: {"c": NewString("red"), "name": NewString("alpha")},
		2: {"c": NewString("blue"), "name": NewString("beta")},
	})

	nonAccel := []Filter{
		{Op: FilterRegex, Field: "name", Value: NewString("^al.*")},
		{Op: FilterNe, Field: "c", Value: NewString("red")},
		{Op: FilterIsNull, Field: "c"},
		{Op: FilterIsEmpty, Field: "c"},
		// NOTE: FilterMatch is now index-narrowable via the inverted token index
		// (see payload_match_index_id_test.go), and FilterContains via the inverted
		// element index (see payload_contains_index_test.go), so neither is non-
		// accelerable anymore.
		{Op: FilterOr, Or: []Filter{
			{Op: FilterEq, Field: "c", Value: NewString("red")},
			{Op: FilterEq, Field: "c", Value: NewString("blue")},
		}},
		{Op: FilterNot, Not: &Filter{Op: FilterEq, Field: "c", Value: NewString("red")}},
	}
	for _, f := range nonAccel {
		if _, ok := p.candidates(f, 1000); ok {
			t.Fatalf("filter op %v should NOT be index-narrowable", f.Op)
		}
	}
}

// TestPayloadIndexIDSupersetSafety is the central safety invariant: for every
// accelerable filter over random payloads, the candidate superset contains
// EVERY id whose payload actually matches (never misses a true match), and for
// the exact (eq/In) accelerators it equals the brute-force set.
func TestPayloadIndexIDSupersetSafety(t *testing.T) {
	rng := rand.New(rand.NewSource(20260615))
	colors := []string{"red", "green", "blue", "yellow"}

	const N = 400
	metas := make(map[uint64]Metadata, N)
	for id := uint64(1); id <= N; id++ {
		m := Metadata{
			"color": NewString(colors[rng.Intn(len(colors))]),
			"price": NewInt(int64(rng.Intn(100))),
			"score": NewFloat(rng.Float64() * 50),
		}
		// some points carry geo, some don't
		if rng.Intn(2) == 0 {
			m["loc"] = NewGeo(37.0+rng.Float64(), -122.0+rng.Float64())
		}
		metas[id] = m
	}
	p := newPayloadIndexID()
	p.rebuild(metas)

	filters := []Filter{
		{Op: FilterEq, Field: "color", Value: NewString("red")},
		{Op: FilterGt, Field: "price", Value: NewInt(50)},
		{Op: FilterGte, Field: "price", Value: NewInt(50)},
		{Op: FilterLt, Field: "score", Value: NewFloat(25)},
		{Op: FilterLte, Field: "score", Value: NewFloat(25)},
		{Op: FilterIn, Field: "color", Value: NewStrings([]string{"red", "blue"})},
		{Op: FilterAnd, And: []Filter{
			{Op: FilterEq, Field: "color", Value: NewString("green")},
			{Op: FilterGte, Field: "price", Value: NewInt(20)},
		}},
		{Op: FilterAnd, And: []Filter{
			{Op: FilterGt, Field: "price", Value: NewInt(10)},
			{Op: FilterLt, Field: "price", Value: NewInt(90)},
		}},
		{Op: FilterGeoBox, Field: "loc", Geo: &GeoCondition{
			MinLat: 37.2, MinLon: -121.8, MaxLat: 37.8, MaxLon: -121.2,
		}},
	}

	// exact accelerators (the index set == predicate set, no over-cover)
	exact := map[int]bool{0: true, 5: true}

	for i, f := range filters {
		ids, ok := p.candidates(f, 100000)
		if !ok {
			t.Fatalf("filter %d (op=%v) expected accelerable", i, f.Op)
		}
		got := sortedIDs(ids)
		want := bruteMatch(t, metas, f)
		if !isSuperset(got, want) {
			// find a missing id for a precise failure
			missing := setDiff(want, got)
			t.Fatalf("filter %d (op=%v): candidate superset MISSES true matches %v", i, f.Op, missing)
		}
		if exact[i] && !eqUint64(got, want) {
			t.Fatalf("filter %d (op=%v): expected EXACT got %v want %v", i, f.Op, got, want)
		}
	}
}

// setDiff returns the elements of a not in b (both sorted).
func setDiff(a, b []uint64) []uint64 {
	have := make(map[uint64]struct{}, len(b))
	for _, x := range b {
		have[x] = struct{}{}
	}
	var out []uint64
	for _, x := range a {
		if _, ok := have[x]; !ok {
			out = append(out, x)
		}
	}
	sort.Slice(out, func(i, j int) bool { return out[i] < out[j] })
	return out
}

// TestPayloadIndexIDMutationUpdatesSuperset proves the safety invariant holds
// across mutation: after re-indexing a point with new payload, the superset for
// the new value contains it and the old value's superset no longer does.
func TestPayloadIndexIDMutationUpdatesSuperset(t *testing.T) {
	p := newPayloadIndexID()
	p.rebuild(map[uint64]Metadata{
		1: {"c": NewString("red"), "p": NewInt(5)},
		2: {"c": NewString("red"), "p": NewInt(15)},
	})
	// mutate id 1 -> price 25, color blue
	p.reindex(1, Metadata{"c": NewString("blue"), "p": NewInt(25)})

	// old eq red no longer includes 1
	red := idCandidates(t, p, Filter{Op: FilterEq, Field: "c", Value: NewString("red")}, 1000, true)
	if !eqUint64(red, []uint64{2}) {
		t.Fatalf("post-mutation color=red got %v want [2]", red)
	}
	// new range p>20 now includes 1
	hi := idCandidates(t, p, Filter{Op: FilterGt, Field: "p", Value: NewInt(20)}, 1000, true)
	if !eqUint64(hi, []uint64{1}) {
		t.Fatalf("post-mutation p>20 got %v want [1]", hi)
	}
}
