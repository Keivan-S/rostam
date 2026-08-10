// SPDX-License-Identifier: Apache-2.0

package vector

import (
	"math/rand"
	"testing"
)

// matchPredicate brute-force evaluates a FilterMatch over Metadata exactly as
// compileMatch does, for the superset proof in the tests below.
func matchPredicate(field, query string) func(Metadata) bool {
	pred, err := compileMatch(field, NewString(query))
	if err != nil {
		panic(err)
	}
	return pred
}

// candidateSlots calls the dense payload index's candidates() under the owner's
// read lock and returns the resulting slot set (or nil, ok=false).
func candidateSlots(h *hnsw, f Filter, limit int) ([]uint32, bool) {
	h.mu.RLock()
	defer h.mu.RUnlock()
	return h.payloadIdx.candidates(f, limit)
}

// slotIDSet maps a slot slice to the set of user ids it resolves to (under the
// read lock), so a candidates() result can be compared to a brute-force id set.
func slotIDSet(h *hnsw, slots []uint32) map[uint64]struct{} {
	h.mu.RLock()
	defer h.mu.RUnlock()
	out := make(map[uint64]struct{}, len(slots))
	for _, s := range slots {
		out[h.arena.ID(s)] = struct{}{}
	}
	return out
}

// bruteMatchIDs returns the set of corpus ids whose metadata satisfies pred.
func bruteMatchIDs(metas map[uint64]Metadata, pred func(Metadata) bool) map[uint64]struct{} {
	out := make(map[uint64]struct{})
	for id, m := range metas {
		if pred(m) {
			out[id] = struct{}{}
		}
	}
	return out
}

func assertSuperset(t *testing.T, label string, cand, truth map[uint64]struct{}) {
	t.Helper()
	for id := range truth {
		if _, ok := cand[id]; !ok {
			t.Errorf("%s: candidate set MISSING true match id %d (not a superset)", label, id)
		}
	}
}

// randWord returns a deterministic lowercase word from a small vocabulary.
func randWord(rng *rand.Rand, vocab []string) string { return vocab[rng.Intn(len(vocab))] }

// TestMatchIndexSupersetSingleToken: a single-field single-token Match returns a
// candidate set EXACTLY equal to the brute-force compileMatch matches (the token
// index intersection of one token == the per-doc posting list, and per-element
// detail can't differ for a single whole token across all docs containing it).
func TestMatchIndexSupersetSingleToken(t *testing.T) {
	const n = 400
	rng := rand.New(rand.NewSource(7))
	vocab := []string{"alpha", "beta", "gamma", "delta", "epsilon", "zeta", "eta", "theta"}
	h, err := newHNSW(Config{Dim: 4, Metric: L2, M: 8, EfConstruction: 50, EfSearch: 50, Seed: 1})
	if err != nil {
		t.Fatal(err)
	}
	metas := make(map[uint64]Metadata, n)
	for i := 1; i <= n; i++ {
		// 1-3 words joined with punctuation/spaces so tokenize splits them.
		w := randWord(rng, vocab) + " " + randWord(rng, vocab) + "-" + randWord(rng, vocab)
		meta := Metadata{"body": NewString(w)}
		metas[uint64(i)] = meta
		if _, _, err := h.Insert(uint64(i), []float32{float32(i), 0, 0, 0}, 0, meta, nil, nil, CASCond{}); err != nil {
			t.Fatal(err)
		}
	}
	for _, q := range vocab {
		f := Filter{Op: FilterMatch, Field: "body", Value: NewString(q)}
		slots, ok := candidateSlots(h, f, n+10)
		if !ok {
			t.Fatalf("match %q: expected narrowable (ok=true)", q)
		}
		cand := slotIDSet(h, slots)
		truth := bruteMatchIDs(metas, matchPredicate("body", q))
		assertSuperset(t, "match "+q, cand, truth)
		// Single token => index set == exact match set.
		if len(cand) != len(truth) {
			t.Errorf("match %q: |cand|=%d != |truth|=%d (single token should be exact)", q, len(cand), len(truth))
		}
	}
}

// TestMatchIndexMultiTokenIntersect: a multi-token Match (match-ALL) returns a
// superset; a doc qualifies only if it has EVERY token. We assert ⊇ and that the
// candidate set never contains a doc the predicate rejects beyond the superset
// guarantee (i.e. it's a tight superset == the exact set for whole-string docs).
func TestMatchIndexMultiTokenIntersect(t *testing.T) {
	h, err := newHNSW(Config{Dim: 4, Metric: L2, M: 8, EfConstruction: 50, EfSearch: 50, Seed: 1})
	if err != nil {
		t.Fatal(err)
	}
	docs := map[uint64]string{
		1: "quick brown fox",
		2: "quick brown dog",
		3: "lazy brown fox",
		4: "quick red fox",
		5: "brown fox quick", // word order irrelevant
	}
	metas := make(map[uint64]Metadata)
	for id, s := range docs {
		m := Metadata{"body": NewString(s)}
		metas[id] = m
		if _, _, err := h.Insert(id, []float32{float32(id), 0, 0, 0}, 0, m, nil, nil, CASCond{}); err != nil {
			t.Fatal(err)
		}
	}
	// "quick fox" => docs containing BOTH quick AND fox: 1, 4, 5.
	f := Filter{Op: FilterMatch, Field: "body", Value: NewString("quick fox")}
	slots, ok := candidateSlots(h, f, 100)
	if !ok {
		t.Fatal("expected narrowable")
	}
	cand := slotIDSet(h, slots)
	truth := bruteMatchIDs(metas, matchPredicate("body", "quick fox"))
	assertSuperset(t, "quick fox", cand, truth)
	wantIDs := map[uint64]struct{}{1: {}, 4: {}, 5: {}}
	if len(cand) != len(wantIDs) {
		t.Errorf("quick fox cand=%v want exactly %v", cand, wantIDs)
	}
	for id := range wantIDs {
		if _, ok := cand[id]; !ok {
			t.Errorf("quick fox missing id %d", id)
		}
	}
}

// TestMatchIndexSelectivityBail: a token present in > limit docs makes the
// intersected set overflow the limit, so candidates bails (ok=false) for a bare
// Match leaf -> the predicate-only fallback stays correct.
func TestMatchIndexSelectivityBail(t *testing.T) {
	const n = 200
	h, err := newHNSW(Config{Dim: 4, Metric: L2, M: 8, EfConstruction: 50, EfSearch: 50, Seed: 1})
	if err != nil {
		t.Fatal(err)
	}
	for i := 1; i <= n; i++ {
		// EVERY doc contains "common".
		m := Metadata{"body": NewString("common token here")}
		if _, _, err := h.Insert(uint64(i), []float32{float32(i), 0, 0, 0}, 0, m, nil, nil, CASCond{}); err != nil {
			t.Fatal(err)
		}
	}
	// limit far below n => the "common" posting set overflows -> bail.
	f := Filter{Op: FilterMatch, Field: "body", Value: NewString("common")}
	if slots, ok := candidateSlots(h, f, 10); ok {
		t.Errorf("expected selectivity bail (ok=false) for non-selective token, got %d slots ok=true", len(slots))
	}
	// With a generous limit it narrows fine.
	if _, ok := candidateSlots(h, f, n+10); !ok {
		t.Error("expected narrowable with a generous limit")
	}
}

// TestMatchIndexValueStrings: a match on a multi-value (ValueStrings) field posts
// per-document (a token in ANY element posts the doc); the candidate set is a
// superset of the predicate's per-element matches.
func TestMatchIndexValueStrings(t *testing.T) {
	h, err := newHNSW(Config{Dim: 4, Metric: L2, M: 8, EfConstruction: 50, EfSearch: 50, Seed: 1})
	if err != nil {
		t.Fatal(err)
	}
	docs := map[uint64][]string{
		1: {"red apple", "green pear"},
		2: {"blue sky", "red car"},
		3: {"green grass", "yellow sun"},
		4: {"red", "apple"}, // "red apple" only across DIFFERENT elements
	}
	metas := make(map[uint64]Metadata)
	for id, ss := range docs {
		m := Metadata{"tags": NewStrings(ss)}
		metas[id] = m
		if _, _, err := h.Insert(id, []float32{float32(id), 0, 0, 0}, 0, m, nil, nil, CASCond{}); err != nil {
			t.Fatal(err)
		}
	}
	// "red apple": per-doc postings -> doc 4 has red+apple (different elements);
	// predicate (per-element tokensContainAll) requires ONE element with both, so
	// doc 4 is in the index superset but NOT a true predicate match. doc 1 IS.
	f := Filter{Op: FilterMatch, Field: "tags", Value: NewString("red apple")}
	slots, ok := candidateSlots(h, f, 100)
	if !ok {
		t.Fatal("expected narrowable")
	}
	cand := slotIDSet(h, slots)
	truth := bruteMatchIDs(metas, matchPredicate("tags", "red apple"))
	assertSuperset(t, "red apple (strings)", cand, truth)
	if _, ok := truth[1]; !ok {
		t.Error("expected doc 1 to be a true predicate match")
	}
	if _, ok := truth[4]; ok {
		t.Error("doc 4 should NOT be a predicate match (cross-element)")
	}
	if _, ok := cand[4]; !ok {
		t.Error("doc 4 SHOULD be in the index superset (per-doc posting)")
	}
}

// TestMatchIndexMutationDropsOldTokens: changing a doc's text drops its old
// tokens (no ghost postings); deleting removes; rebuild repopulates.
func TestMatchIndexMutationDropsOldTokens(t *testing.T) {
	h, err := newHNSW(Config{Dim: 4, Metric: L2, M: 8, EfConstruction: 50, EfSearch: 50, Seed: 1})
	if err != nil {
		t.Fatal(err)
	}
	if _, _, err := h.Insert(1, []float32{1, 0, 0, 0}, 0, Metadata{"body": NewString("orange fruit")}, nil, nil, CASCond{}); err != nil {
		t.Fatal(err)
	}
	if _, _, err := h.Insert(2, []float32{2, 0, 0, 0}, 0, Metadata{"body": NewString("apple fruit")}, nil, nil, CASCond{}); err != nil {
		t.Fatal(err)
	}
	mustHave := func(q string, wantIDs ...uint64) {
		t.Helper()
		slots, ok := candidateSlots(h, Filter{Op: FilterMatch, Field: "body", Value: NewString(q)}, 100)
		if !ok {
			t.Fatalf("match %q: expected narrowable", q)
		}
		cand := slotIDSet(h, slots)
		if len(cand) != len(wantIDs) {
			t.Errorf("match %q: cand=%v want ids %v", q, cand, wantIDs)
		}
		for _, id := range wantIDs {
			if _, ok := cand[id]; !ok {
				t.Errorf("match %q: missing id %d", q, id)
			}
		}
	}
	mustHave("orange", 1)
	// Overwrite doc 1's text: "orange" must no longer post slot 1.
	if _, _, _, err := h.SetPayload(1, Metadata{"body": NewString("banana fruit")}, nil, CASCond{}); err != nil {
		t.Fatal(err)
	}
	if slots, ok := candidateSlots(h, Filter{Op: FilterMatch, Field: "body", Value: NewString("orange")}, 100); ok {
		if cand := slotIDSet(h, slots); len(cand) != 0 {
			t.Errorf("after mutation, 'orange' should match nothing, got %v", cand)
		}
	}
	mustHave("banana", 1)
	mustHave("fruit", 1, 2)
	// Delete doc 2: "apple" already gone (was overwritten? no, doc2 unchanged) ->
	// deleting doc 2 removes it from "fruit".
	if _, err := h.Delete(2, CASCond{}); err != nil {
		t.Fatal(err)
	}
	h.Reclaim()
	mustHave("fruit", 1)
	// rebuild (via Reclaim already ran) keeps doc 1 intact.
	mustHave("banana", 1)
}

// TestMatchIndexInsideAnd: a Match inside an And with an eq term narrows to the
// intersection (the candidate set ⊇ the And's true matches AND is constrained by
// both terms via the index intersection).
func TestMatchIndexInsideAnd(t *testing.T) {
	const n = 300
	rng := rand.New(rand.NewSource(3))
	vocab := []string{"alpha", "beta", "gamma", "delta"}
	h, err := newHNSW(Config{Dim: 4, Metric: L2, M: 8, EfConstruction: 50, EfSearch: 50, Seed: 1})
	if err != nil {
		t.Fatal(err)
	}
	metas := make(map[uint64]Metadata, n)
	for i := 1; i <= n; i++ {
		m := Metadata{
			"body": NewString(randWord(rng, vocab) + " " + randWord(rng, vocab)),
			"cat":  NewInt(int64(i % 5)),
		}
		metas[uint64(i)] = m
		if _, _, err := h.Insert(uint64(i), []float32{float32(i), 0, 0, 0}, 0, m, nil, nil, CASCond{}); err != nil {
			t.Fatal(err)
		}
	}
	f := Filter{Op: FilterAnd, And: []Filter{
		{Op: FilterEq, Field: "cat", Value: NewInt(2)},
		{Op: FilterMatch, Field: "body", Value: NewString("alpha")},
	}}
	slots, ok := candidateSlots(h, f, n+10)
	if !ok {
		t.Fatal("expected narrowable")
	}
	cand := slotIDSet(h, slots)
	truth := bruteMatchIDs(metas, func(m Metadata) bool {
		return m["cat"].Int == 2 && matchPredicate("body", "alpha")(m)
	})
	assertSuperset(t, "and(eq,match)", cand, truth)
	// Every candidate must satisfy BOTH the cat==2 and the alpha-token constraint
	// (the index intersection already enforces both since both are indexed).
	for id := range cand {
		if metas[id]["cat"].Int != 2 {
			t.Errorf("and: candidate id %d has cat=%d (index should have intersected cat==2)", id, metas[id]["cat"].Int)
		}
	}
}

// TestMatchIndexEmptyQuery: an empty (no-token) query is NOT narrowable for a
// bare leaf (ok=false), and inside an And it is skipped (the other term narrows).
func TestMatchIndexEmptyQuery(t *testing.T) {
	h, err := newHNSW(Config{Dim: 4, Metric: L2, M: 8, EfConstruction: 50, EfSearch: 50, Seed: 1})
	if err != nil {
		t.Fatal(err)
	}
	for i := 1; i <= 20; i++ {
		m := Metadata{"body": NewString("word here"), "cat": NewInt(int64(i % 2))}
		if _, _, err := h.Insert(uint64(i), []float32{float32(i), 0, 0, 0}, 0, m, nil, nil, CASCond{}); err != nil {
			t.Fatal(err)
		}
	}
	// Bare empty-query match leaf (only punctuation -> zero tokens) -> ok=false.
	if slots, ok := candidateSlots(h, Filter{Op: FilterMatch, Field: "body", Value: NewString("   --- ")}, 100); ok {
		t.Errorf("empty-query match leaf should be ok=false, got %d slots", len(slots))
	}
	// Inside an And: the empty match is skipped, the eq term still narrows.
	f := Filter{Op: FilterAnd, And: []Filter{
		{Op: FilterEq, Field: "cat", Value: NewInt(1)},
		{Op: FilterMatch, Field: "body", Value: NewString("   ")},
	}}
	slots, ok := candidateSlots(h, f, 100)
	if !ok {
		t.Fatal("and(eq, empty-match) should narrow via the eq term")
	}
	cand := slotIDSet(h, slots)
	for id := range cand {
		if id%2 != 1 {
			t.Errorf("and(eq, empty-match): candidate id %d violates cat==1", id)
		}
	}
}

// TestMatchIndexFilterFirstEqualsPredicate: an end-to-end dense search with a
// match filter returns the SAME results as the brute-force predicate path. This
// exercises the full filter-first acceleration (candidates -> predicate re-check)
// and proves filter-first == predicate-only.
func TestMatchIndexFilterFirstEqualsPredicate(t *testing.T) {
	const (
		n   = 500
		dim = 8
		k   = 12
	)
	rng := rand.New(rand.NewSource(99))
	vocab := []string{"alpha", "beta", "gamma", "delta", "epsilon", "zeta"}
	h, err := newHNSW(Config{Dim: dim, Metric: L2, M: 16, EfConstruction: 200, EfSearch: 128, Seed: 1})
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
		text := randWord(rng, vocab) + " " + randWord(rng, vocab) + " " + randWord(rng, vocab)
		m := Metadata{"body": NewString(text)}
		corpus[id] = v
		metas[id] = m
		if _, _, err := h.Insert(id, v, 0, m, nil, nil, CASCond{}); err != nil {
			t.Fatal(err)
		}
	}
	q := make([]float32, dim)
	for j := range q {
		q[j] = float32(rng.NormFloat64())
	}

	cases := []string{"alpha", "alpha beta", "gamma delta", "zeta"}
	for _, qstr := range cases {
		f := Filter{Op: FilterMatch, Field: "body", Value: NewString(qstr)}
		got, err := h.SearchFiltered(q, k, f)
		if err != nil {
			t.Fatal(err)
		}
		want := bruteForceFiltered(corpus, metas, q, k, matchPredicate("body", qstr))
		if !eqUint64(resultIDs(got), want) {
			t.Errorf("match %q: filter-first ids %v != brute-force %v", qstr, resultIDs(got), want)
		}
	}

	// And(match, range-ish via second match) end-to-end.
	andF := Filter{Op: FilterAnd, And: []Filter{
		{Op: FilterMatch, Field: "body", Value: NewString("alpha")},
		{Op: FilterMatch, Field: "body", Value: NewString("beta")},
	}}
	got, err := h.SearchFiltered(q, k, andF)
	if err != nil {
		t.Fatal(err)
	}
	want := bruteForceFiltered(corpus, metas, q, k, func(m Metadata) bool {
		return matchPredicate("body", "alpha")(m) && matchPredicate("body", "beta")(m)
	})
	if !eqUint64(resultIDs(got), want) {
		t.Errorf("and(match,match): filter-first %v != brute-force %v", resultIDs(got), want)
	}
}

// TestMatchIndexRandomSupersetFuzz: random string payloads, assert the index
// candidate set is ALWAYS a superset of the brute-force compileMatch matches for
// random single- and multi-token queries (the core safety invariant).
func TestMatchIndexRandomSupersetFuzz(t *testing.T) {
	const n = 600
	rng := rand.New(rand.NewSource(2024))
	vocab := []string{"aa", "bb", "cc", "dd", "ee", "ff", "gg", "hh", "ii", "jj"}
	h, err := newHNSW(Config{Dim: 4, Metric: L2, M: 8, EfConstruction: 50, EfSearch: 50, Seed: 1})
	if err != nil {
		t.Fatal(err)
	}
	metas := make(map[uint64]Metadata, n)
	for i := 1; i <= n; i++ {
		nw := 1 + rng.Intn(4)
		var ws []string
		for j := 0; j < nw; j++ {
			ws = append(ws, randWord(rng, vocab))
		}
		text := ""
		for j, w := range ws {
			if j > 0 {
				text += " "
			}
			text += w
		}
		m := Metadata{"body": NewString(text)}
		metas[uint64(i)] = m
		if _, _, err := h.Insert(uint64(i), []float32{float32(i), 0, 0, 0}, 0, m, nil, nil, CASCond{}); err != nil {
			t.Fatal(err)
		}
	}
	for trial := 0; trial < 200; trial++ {
		nq := 1 + rng.Intn(3)
		var qws []string
		for j := 0; j < nq; j++ {
			qws = append(qws, randWord(rng, vocab))
		}
		qstr := ""
		for j, w := range qws {
			if j > 0 {
				qstr += " "
			}
			qstr += w
		}
		f := Filter{Op: FilterMatch, Field: "body", Value: NewString(qstr)}
		slots, ok := candidateSlots(h, f, n+10)
		truth := bruteMatchIDs(metas, matchPredicate("body", qstr))
		if !ok {
			// Bail is only legitimate when the truth set exceeds the limit; with
			// limit=n+10 it never should, so a bail here is a bug.
			t.Fatalf("trial %d query %q: unexpected ok=false (limit generous)", trial, qstr)
		}
		cand := slotIDSet(h, slots)
		assertSuperset(t, "fuzz "+qstr, cand, truth)
	}
}
