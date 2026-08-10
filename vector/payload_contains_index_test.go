// SPDX-License-Identifier: Apache-2.0

package vector

import (
	"bytes"
	"math/rand"
	"testing"
)

// Tests for the per-element `contains` inverted index (dense payloadIndex +
// id-keyed payloadIndexID) and the candidates() contains arm. The central proof
// is the EQUIVALENCE ORACLE: a filter-first search/scroll with a `contains`
// filter returns the SAME matched set as the predicate-eval fallback, for
// string, int AND float arrays — i.e. the index posting list is a no-false-
// negative superset of the compileContains predicate matches.

// slotCands runs the dense candidates() asserting ok, returning the sorted slots.
func slotCands(t *testing.T, p *payloadIndex, f Filter, limit int, wantOK bool) []uint32 {
	t.Helper()
	slots, ok := p.candidates(f, limit)
	if ok != wantOK {
		t.Fatalf("candidates ok = %v, want %v (filter op=%v)", ok, wantOK, f.Op)
	}
	out := append([]uint32(nil), slots...)
	for i := 1; i < len(out); i++ {
		for j := i; j > 0 && out[j-1] > out[j]; j-- {
			out[j-1], out[j] = out[j], out[j-1]
		}
	}
	return out
}

func hasSlot(slots []uint32, s uint32) bool {
	for _, x := range slots {
		if x == s {
			return true
		}
	}
	return false
}

// TestContainsIndexMatchesPredicate: a doc with array [a,b,c]; contains a/b/c
// hit, contains d miss — exactly mirroring compileContains. Strings, ints, floats.
func TestContainsIndexMatchesPredicate(t *testing.T) {
	p := newPayloadIndex()
	p.reindex(1, Metadata{
		"tags":  NewStrings([]string{"a", "b", "c"}),
		"perms": NewInts([]int64{10, 20, 30}),
		"score": NewFloats([]float64{1.5, 2.5}),
	})

	str := func(s string) Filter { return Filter{Op: FilterContains, Field: "tags", Value: NewString(s)} }
	for _, s := range []string{"a", "b", "c"} {
		got := slotCands(t, p, str(s), 1000, true)
		if !hasSlot(got, 1) {
			t.Fatalf("contains tags=%q got %v, want includes slot 1", s, got)
		}
	}
	if got := slotCands(t, p, str("d"), 1000, true); hasSlot(got, 1) {
		t.Fatalf("contains tags=d got %v, want NOT slot 1", got)
	}

	for _, n := range []int64{10, 20, 30} {
		got := slotCands(t, p, Filter{Op: FilterContains, Field: "perms", Value: NewInt(n)}, 1000, true)
		if !hasSlot(got, 1) {
			t.Fatalf("contains perms=%d got %v, want includes slot 1", n, got)
		}
	}
	if got := slotCands(t, p, Filter{Op: FilterContains, Field: "perms", Value: NewInt(99)}, 1000, true); hasSlot(got, 1) {
		t.Fatalf("contains perms=99 got %v, want NOT slot 1", got)
	}

	for _, x := range []float64{1.5, 2.5} {
		got := slotCands(t, p, Filter{Op: FilterContains, Field: "score", Value: NewFloat(x)}, 1000, true)
		if !hasSlot(got, 1) {
			t.Fatalf("contains score=%v got %v, want includes slot 1", x, got)
		}
	}
	if got := slotCands(t, p, Filter{Op: FilterContains, Field: "score", Value: NewFloat(9.9)}, 1000, true); hasSlot(got, 1) {
		t.Fatalf("contains score=9.9 got %v, want NOT slot 1", got)
	}
}

// TestContainsKindMismatch: a string want vs an int array (and vice versa) must
// not hit — exactly as compileContains' got.Kind check requires.
func TestContainsKindMismatch(t *testing.T) {
	p := newPayloadIndex()
	p.reindex(1, Metadata{"perms": NewInts([]int64{1, 2, 3})})

	// string want against the int array field -> the scalarKey kind differs, so no
	// posting (the predicate also returns false: got.Kind != ValueStrings).
	got := slotCands(t, p, Filter{Op: FilterContains, Field: "perms", Value: NewString("1")}, 1000, true)
	if len(got) != 0 {
		t.Fatalf("contains perms(string '1') got %v, want empty (kind mismatch)", got)
	}
	// And confirm the predicate agrees.
	pred, err := Filter{Op: FilterContains, Field: "perms", Value: NewString("1")}.Compile()
	if err != nil {
		t.Fatal(err)
	}
	if pred(Metadata{"perms": NewInts([]int64{1, 2, 3})}) {
		t.Fatalf("compileContains string vs int array should be false")
	}
}

// TestContainsDedupWithinDoc: array [a,a,b] posts under a exactly once (the
// reverse list stays distinct so the per-entry drop is exact).
func TestContainsDedupWithinDoc(t *testing.T) {
	p := newPayloadIndex()
	p.reindex(1, Metadata{"tags": NewStrings([]string{"a", "a", "b"})})

	set := p.contains["tags"][scalarKey{kind: ValueString, str: "a"}]
	if len(set) != 1 {
		t.Fatalf("contains[tags][a] posting size = %d, want 1 (slot posted once)", len(set))
	}
	// The reverse list must contain (tags,a) exactly once, else dropContains on a
	// later reindex would leave a stale entry. Count distinct.
	var aCount int
	for _, fk := range p.containsSlotKeys[1] {
		if fk.field == "tags" && fk.key.str == "a" {
			aCount++
		}
	}
	if aCount != 1 {
		t.Fatalf("reverse key (tags,a) appears %d times, want 1 (dedup within value)", aCount)
	}
}

// TestContainsDeleteReindexRemovesStale is the #1 maintenance bug guard: reindex
// a doc to a different array -> the OLD elements' postings no longer include it;
// delete (nil meta) -> gone entirely; no empty maps left.
func TestContainsDeleteReindexRemovesStale(t *testing.T) {
	p := newPayloadIndex()
	p.reindex(1, Metadata{"tags": NewStrings([]string{"a", "b"})})
	p.reindex(2, Metadata{"tags": NewStrings([]string{"a", "c"})})

	a := slotCands(t, p, Filter{Op: FilterContains, Field: "tags", Value: NewString("a")}, 1000, true)
	if !hasSlot(a, 1) || !hasSlot(a, 2) {
		t.Fatalf("contains a = %v, want {1,2}", a)
	}

	// Reindex slot 1 to [x,y]: OLD elements a,b must no longer post slot 1.
	p.reindex(1, Metadata{"tags": NewStrings([]string{"x", "y"})})
	a = slotCands(t, p, Filter{Op: FilterContains, Field: "tags", Value: NewString("a")}, 1000, true)
	if hasSlot(a, 1) {
		t.Fatalf("after reindex contains a = %v, slot 1 stale (OLD posting not dropped)", a)
	}
	if !hasSlot(a, 2) {
		t.Fatalf("after reindex contains a = %v, slot 2 must remain", a)
	}
	b := slotCands(t, p, Filter{Op: FilterContains, Field: "tags", Value: NewString("b")}, 1000, true)
	if hasSlot(b, 1) {
		t.Fatalf("after reindex contains b = %v, slot 1 stale", b)
	}
	x := slotCands(t, p, Filter{Op: FilterContains, Field: "tags", Value: NewString("x")}, 1000, true)
	if !hasSlot(x, 1) {
		t.Fatalf("after reindex contains x = %v, want slot 1", x)
	}

	// Delete slot 1 (nil meta) and slot 2: contains x and a must be empty, maps clean.
	p.reindex(1, nil)
	p.reindex(2, nil)
	if got := slotCands(t, p, Filter{Op: FilterContains, Field: "tags", Value: NewString("x")}, 1000, true); len(got) != 0 {
		t.Fatalf("after delete contains x = %v, want empty", got)
	}
	if _, ok := p.containsSlotKeys[1]; ok {
		t.Fatalf("containsSlotKeys[1] not removed after delete")
	}
	if len(p.contains) != 0 {
		t.Fatalf("contains map not cleaned up after all deletes: %v", p.contains)
	}
}

// TestContainsAndCompose: contains ∧ eq intersects on both conjuncts.
func TestContainsAndCompose(t *testing.T) {
	p := newPayloadIndex()
	p.reindex(1, Metadata{"tags": NewStrings([]string{"a", "b"}), "color": NewString("red")})
	p.reindex(2, Metadata{"tags": NewStrings([]string{"a", "c"}), "color": NewString("blue")})
	p.reindex(3, Metadata{"tags": NewStrings([]string{"a"}), "color": NewString("red")})

	f := Filter{Op: FilterAnd, And: []Filter{
		{Op: FilterContains, Field: "tags", Value: NewString("a")},
		{Op: FilterEq, Field: "color", Value: NewString("red")},
	}}
	got := slotCands(t, p, f, 1000, true)
	if !hasSlot(got, 1) || !hasSlot(got, 3) || hasSlot(got, 2) {
		t.Fatalf("and(contains a, color=red) = %v, want {1,3}", got)
	}
}

// TestContainsSelectivityBail: a contains value present on more docs than the
// limit bails ok=false (so the caller falls back to graph/predicate — still
// correct). A selective value within the limit narrows ok=true.
func TestContainsSelectivityBail(t *testing.T) {
	p := newPayloadIndex()
	for i := uint32(1); i <= 20; i++ {
		p.reindex(i, Metadata{"tags": NewStrings([]string{"common", "rare" + string(rune('a'+i%2))})})
	}
	// "common" is on all 20 -> with limit 5 the posting list exceeds it -> bail.
	if _, ok := p.candidates(Filter{Op: FilterContains, Field: "tags", Value: NewString("common")}, 5); ok {
		t.Fatalf("contains common with limit 5 expected ok=false (selectivity bail)")
	}
	// With a generous limit it narrows.
	if _, ok := p.candidates(Filter{Op: FilterContains, Field: "tags", Value: NewString("common")}, 1000); !ok {
		t.Fatalf("contains common with limit 1000 expected ok=true")
	}
}

// TestContainsEquivalenceOracleDense is the soundness proof on the dense engine:
// a filter-first SearchFiltered with a contains filter == the predicate-eval
// brute-force result, for string, int and float arrays.
func TestContainsEquivalenceOracleDense(t *testing.T) {
	const (
		n   = 400
		dim = 8
		k   = 25
	)
	rng := rand.New(rand.NewSource(424242))
	h, err := newHNSW(Config{Dim: dim, Metric: L2, M: 16, EfConstruction: 200, EfSearch: 128, Seed: 1})
	if err != nil {
		t.Fatal(err)
	}
	strVocab := []string{"red", "green", "blue", "amber"}
	intVocab := []int64{1, 2, 3, 4, 5}
	fltVocab := []float64{0.5, 1.5, 2.5}
	corpus := make(map[uint64][]float32, n)
	metas := make(map[uint64]Metadata, n)
	for i := 1; i <= n; i++ {
		v := make([]float32, dim)
		for j := range v {
			v[j] = float32(rng.NormFloat64())
		}
		id := uint64(i)
		// 1-3 distinct-ish elements per array (with intentional repeats to exercise dedup).
		tags := []string{strVocab[rng.Intn(len(strVocab))], strVocab[rng.Intn(len(strVocab))]}
		perms := []int64{intVocab[rng.Intn(len(intVocab))], intVocab[rng.Intn(len(intVocab))]}
		scores := []float64{fltVocab[rng.Intn(len(fltVocab))]}
		meta := Metadata{
			"tags":  NewStrings(tags),
			"perms": NewInts(perms),
			"score": NewFloats(scores),
		}
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

	type oracle struct {
		name string
		f    Filter
		pred func(Metadata) bool
	}
	cases := []oracle{
		{"contains-string", Filter{Op: FilterContains, Field: "tags", Value: NewString("red")}, func(m Metadata) bool {
			for _, s := range m["tags"].Strs {
				if s == "red" {
					return true
				}
			}
			return false
		}},
		{"contains-int", Filter{Op: FilterContains, Field: "perms", Value: NewInt(3)}, func(m Metadata) bool {
			for _, x := range m["perms"].Ints {
				if x == 3 {
					return true
				}
			}
			return false
		}},
		{"contains-float", Filter{Op: FilterContains, Field: "score", Value: NewFloat(1.5)}, func(m Metadata) bool {
			for _, x := range m["score"].Flts {
				if x == 1.5 {
					return true
				}
			}
			return false
		}},
		{"and-contains-eq-likeRange", Filter{Op: FilterAnd, And: []Filter{
			{Op: FilterContains, Field: "tags", Value: NewString("blue")},
			{Op: FilterContains, Field: "perms", Value: NewInt(2)},
		}}, func(m Metadata) bool {
			hb := false
			for _, s := range m["tags"].Strs {
				if s == "blue" {
					hb = true
				}
			}
			h2 := false
			for _, x := range m["perms"].Ints {
				if x == 2 {
					h2 = true
				}
			}
			return hb && h2
		}},
	}
	for _, c := range cases {
		t.Run(c.name, func(t *testing.T) {
			got, err := h.SearchFiltered(q, k, c.f)
			if err != nil {
				t.Fatal(err)
			}
			want := bruteForceFiltered(corpus, metas, q, k, c.pred)
			if !eqUint64(resultIDs(got), want) {
				t.Fatalf("%s: filter-first ids %v != brute-force %v", c.name, resultIDs(got), want)
			}
		})
	}
}

// TestContainsRebuildOnRestoreDense proves rebuild reconstructs the contains
// index: a restored dense collection answers a contains filter filter-first and
// correctly (matches the predicate).
func TestContainsRebuildOnRestoreDense(t *testing.T) {
	src, err := newHNSW(Config{Dim: 4, Metric: L2, M: 8, EfConstruction: 50, EfSearch: 64, Seed: 1})
	if err != nil {
		t.Fatal(err)
	}
	want := map[uint64]bool{}
	for i := 1; i <= 40; i++ {
		tags := []string{"x"}
		if i%3 == 0 {
			tags = []string{"x", "target"}
			want[uint64(i)] = true
		}
		if _, _, err := src.Insert(uint64(i), []float32{float32(i), 0, 0, 0}, 0, Metadata{"tags": NewStrings(tags)}, nil, nil, CASCond{}); err != nil {
			t.Fatal(err)
		}
	}
	var buf bytes.Buffer
	if err := src.Snapshot(&buf); err != nil {
		t.Fatal(err)
	}
	dst, err := newHNSW(Config{Dim: 4, Metric: L2, M: 8, EfConstruction: 50, EfSearch: 64, Seed: 1})
	if err != nil {
		t.Fatal(err)
	}
	if err := dst.Restore(&buf); err != nil {
		t.Fatal(err)
	}
	// The restored payload index must carry the rebuilt contains postings.
	if len(dst.payloadIdx.contains["tags"][scalarKey{kind: ValueString, str: "target"}]) == 0 {
		t.Fatalf("restored contains[tags][target] is empty -> rebuild did not reconstruct contains")
	}
	got, err := dst.SearchFiltered([]float32{20, 0, 0, 0}, 40, Filter{Op: FilterContains, Field: "tags", Value: NewString("target")})
	if err != nil {
		t.Fatal(err)
	}
	if len(got) != len(want) {
		t.Fatalf("restored contains target returned %d ids, want %d", len(got), len(want))
	}
	for _, r := range got {
		if !want[r.ID] {
			t.Fatalf("restored contains target returned non-matching id %d", r.ID)
		}
	}
}

// --- id-keyed (named/MV) mirror ---

// TestContainsIndexIDMatchesPredicate mirrors the dense index==predicate test for
// the id-keyed payloadIndexID.
func TestContainsIndexIDMatchesPredicate(t *testing.T) {
	p := newPayloadIndexID()
	p.reindex(1, Metadata{
		"tags":  NewStrings([]string{"a", "b", "c"}),
		"perms": NewInts([]int64{10, 20}),
		"score": NewFloats([]float64{1.5}),
	})

	for _, s := range []string{"a", "b", "c"} {
		got := idCandidates(t, p, Filter{Op: FilterContains, Field: "tags", Value: NewString(s)}, 1000, true)
		if !eqUint64(got, []uint64{1}) {
			t.Fatalf("id contains tags=%q got %v, want [1]", s, got)
		}
	}
	if got := idCandidates(t, p, Filter{Op: FilterContains, Field: "tags", Value: NewString("d")}, 1000, true); len(got) != 0 {
		t.Fatalf("id contains tags=d got %v, want []", got)
	}
	if got := idCandidates(t, p, Filter{Op: FilterContains, Field: "perms", Value: NewInt(10)}, 1000, true); !eqUint64(got, []uint64{1}) {
		t.Fatalf("id contains perms=10 got %v, want [1]", got)
	}
	if got := idCandidates(t, p, Filter{Op: FilterContains, Field: "score", Value: NewFloat(1.5)}, 1000, true); !eqUint64(got, []uint64{1}) {
		t.Fatalf("id contains score=1.5 got %v, want [1]", got)
	}
}

// TestContainsIDDeleteReindexRemovesStale mirrors the dense stale-posting guard.
func TestContainsIDDeleteReindexRemovesStale(t *testing.T) {
	p := newPayloadIndexID()
	p.reindex(1, Metadata{"tags": NewStrings([]string{"a", "b"})})
	p.reindex(2, Metadata{"tags": NewStrings([]string{"a"})})

	p.reindex(1, Metadata{"tags": NewStrings([]string{"z"})}) // a,b must drop from id 1
	a := idCandidates(t, p, Filter{Op: FilterContains, Field: "tags", Value: NewString("a")}, 1000, true)
	if !eqUint64(a, []uint64{2}) {
		t.Fatalf("after reindex id contains a = %v, want [2] (stale id 1 not dropped)", a)
	}
	p.reindex(2, nil)
	if got := idCandidates(t, p, Filter{Op: FilterContains, Field: "tags", Value: NewString("a")}, 1000, true); len(got) != 0 {
		t.Fatalf("after delete id contains a = %v, want []", got)
	}
	if _, ok := p.containsIDKeys[2]; ok {
		t.Fatalf("containsIDKeys[2] not removed after delete")
	}
}

// TestNamedFilterFirstContains is the id-keyed equivalence oracle: a named
// filter-first contains search == the predicate-eval fallback, for string, int
// and float arrays, plus And(eq, contains). Proves the id-keyed contains index +
// the rebuild-on-load path (NewNamedCollection rebuilds) are sound.
func TestNamedFilterFirstContains(t *testing.T) {
	nc, err := NewNamedCollection("default/named", map[string]NamedVectorParams{
		"title": {Dim: 4, Metric: L2},
	})
	if err != nil {
		t.Fatalf("new: %v", err)
	}
	defer nc.Close()

	rng := rand.New(rand.NewSource(99))
	strVocab := []string{"red", "green", "blue", "amber"}
	intVocab := []int64{1, 2, 3, 4}
	fltVocab := []float64{0.5, 1.5, 2.5}
	for id := uint64(1); id <= 60; id++ {
		cat := "even"
		if id%2 == 1 {
			cat = "odd"
		}
		meta := Metadata{
			"tags":  NewStrings([]string{strVocab[rng.Intn(len(strVocab))], strVocab[rng.Intn(len(strVocab))]}),
			"perms": NewInts([]int64{intVocab[rng.Intn(len(intVocab))], intVocab[rng.Intn(len(intVocab))]}),
			"score": NewFloats([]float64{fltVocab[rng.Intn(len(fltVocab))]}),
			"cat":   NewString(cat),
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
		{"contains-string", Filter{Op: FilterContains, Field: "tags", Value: NewString("red")}},
		{"contains-int", Filter{Op: FilterContains, Field: "perms", Value: NewInt(2)}},
		{"contains-float", Filter{Op: FilterContains, Field: "score", Value: NewFloat(1.5)}},
		{"and-eq-contains", Filter{Op: FilterAnd, And: []Filter{
			{Op: FilterEq, Field: "cat", Value: NewString("odd")},
			{Op: FilterContains, Field: "tags", Value: NewString("blue")},
		}}},
		{"and-contains-contains", Filter{Op: FilterAnd, And: []Filter{
			{Op: FilterContains, Field: "tags", Value: NewString("green")},
			{Op: FilterContains, Field: "perms", Value: NewInt(3)},
		}}},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			assertFilterFirstMatchesFallback(t, nc, "title", q, 12, tc.f)
		})
	}
}

// TestMVFilterFirstContains is the multi-vector (MaxSim) equivalence oracle: a
// filter-first contains search == the post-filter fallback (ids + MaxSim scores)
// for string, int and float arrays + And(eq, contains). MV uses the same id-keyed
// payloadIndexID, so this exercises the id-keyed contains arm on the MV path.
func TestMVFilterFirstContains(t *testing.T) {
	const n, dim = 60, 12
	m, err := NewMultiVectorIndex(MultiVectorConfig{Dim: dim, M: 16, EfConstruction: 200, EfSearch: 128, Seed: 7})
	if err != nil {
		t.Fatalf("NewMultiVectorIndex: %v", err)
	}
	defer m.Close()

	rng := rand.New(rand.NewSource(7))
	strVocab := []string{"red", "green", "blue", "amber"}
	intVocab := []int64{1, 2, 3, 4}
	fltVocab := []float64{0.5, 1.5, 2.5}
	for id := uint64(1); id <= uint64(n); id++ {
		cat := "even"
		if id%2 == 1 {
			cat = "odd"
		}
		meta := Metadata{
			"tags":  NewStrings([]string{strVocab[rng.Intn(len(strVocab))], strVocab[rng.Intn(len(strVocab))]}),
			"perms": NewInts([]int64{intVocab[rng.Intn(len(intVocab))], intVocab[rng.Intn(len(intVocab))]}),
			"score": NewFloats([]float64{fltVocab[rng.Intn(len(fltVocab))]}),
			"cat":   NewString(cat),
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
		{"contains-string", Filter{Op: FilterContains, Field: "tags", Value: NewString("red")}},
		{"contains-int", Filter{Op: FilterContains, Field: "perms", Value: NewInt(2)}},
		{"contains-float", Filter{Op: FilterContains, Field: "score", Value: NewFloat(1.5)}},
		{"and-eq-contains", Filter{Op: FilterAnd, And: []Filter{
			{Op: FilterEq, Field: "cat", Value: NewString("odd")},
			{Op: FilterContains, Field: "tags", Value: NewString("blue")},
		}}},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			assertMVFilterFirstMatchesFallback(t, m, q, 8, tc.f)
		})
	}
}
