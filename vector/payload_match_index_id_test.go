// SPDX-License-Identifier: Apache-2.0

package vector

import (
	"math/rand"
	"testing"
)

// id-keyed token-index tests for payloadIndexID.candidates on a FilterMatch.
// They mirror the dense payload_match_index_test.go superset/selectivity/
// mutation proofs but operate directly on a payloadIndexID (uint64 ids), which
// is what the named + MV collections key their payloads by. The core invariant
// is the SAME as dense: candidates() for a Match returns a SUPERSET ⊇ the brute-
// force compileMatch matches; a single whole token == the exact match set.

// idMatchSuperset asserts the id candidate set ⊇ the brute-force match truth.
func idMatchSuperset(t *testing.T, p *payloadIndexID, metas map[uint64]Metadata, field, query string, limit int) []uint64 {
	t.Helper()
	f := Filter{Op: FilterMatch, Field: field, Value: NewString(query)}
	cand := idCandidates(t, p, f, limit, true)
	truth := bruteMatch(t, metas, f)
	if !isSuperset(cand, truth) {
		t.Fatalf("match %q: candidate %v NOT a superset of truth %v", query, cand, truth)
	}
	return cand
}

// TestMatchIndexIDSupersetSingleToken: a single whole token == the exact match
// set (the posting list of that one token IS exactly the docs containing it).
func TestMatchIndexIDSupersetSingleToken(t *testing.T) {
	const n = 400
	rng := rand.New(rand.NewSource(7))
	vocab := []string{"alpha", "beta", "gamma", "delta", "epsilon", "zeta", "eta", "theta"}
	p := newPayloadIndexID()
	metas := make(map[uint64]Metadata, n)
	for i := uint64(1); i <= n; i++ {
		w := vocab[rng.Intn(len(vocab))] + " " + vocab[rng.Intn(len(vocab))] + "-" + vocab[rng.Intn(len(vocab))]
		m := Metadata{"body": NewString(w)}
		metas[i] = m
		p.reindex(i, m)
	}
	for _, q := range vocab {
		cand := idMatchSuperset(t, p, metas, "body", q, n+10)
		truth := bruteMatch(t, metas, Filter{Op: FilterMatch, Field: "body", Value: NewString(q)})
		if len(cand) != len(truth) {
			t.Errorf("match %q: |cand|=%d != |truth|=%d (single token should be exact)", q, len(cand), len(truth))
		}
	}
}

// TestMatchIndexIDMultiTokenIntersect: match-ALL intersects (a doc qualifies
// only if it has EVERY query token).
func TestMatchIndexIDMultiTokenIntersect(t *testing.T) {
	p := newPayloadIndexID()
	docs := map[uint64]string{
		1: "quick brown fox",
		2: "quick brown dog",
		3: "lazy brown fox",
		4: "quick red fox",
		5: "brown fox quick",
	}
	metas := make(map[uint64]Metadata)
	for id, s := range docs {
		m := Metadata{"body": NewString(s)}
		metas[id] = m
		p.reindex(id, m)
	}
	// "quick fox" => docs with BOTH quick AND fox: 1, 4, 5.
	cand := idMatchSuperset(t, p, metas, "body", "quick fox", 100)
	want := map[uint64]struct{}{1: {}, 4: {}, 5: {}}
	if len(cand) != len(want) {
		t.Errorf("quick fox cand=%v want exactly %v", cand, want)
	}
	for id := range want {
		found := false
		for _, c := range cand {
			if c == id {
				found = true
			}
		}
		if !found {
			t.Errorf("quick fox missing id %d", id)
		}
	}
}

// TestMatchIndexIDSelectivityBail: a token in > limit docs overflows the cap so a
// bare Match leaf bails (ok=false); a generous limit narrows fine.
func TestMatchIndexIDSelectivityBail(t *testing.T) {
	const n = 200
	p := newPayloadIndexID()
	for i := uint64(1); i <= n; i++ {
		p.reindex(i, Metadata{"body": NewString("common token here")})
	}
	f := Filter{Op: FilterMatch, Field: "body", Value: NewString("common")}
	if ids, ok := p.candidates(f, 10); ok {
		t.Errorf("expected selectivity bail (ok=false), got %d ids ok=true", len(ids))
	}
	if _, ok := p.candidates(f, n+10); !ok {
		t.Error("expected narrowable with a generous limit")
	}
}

// TestMatchIndexIDValueStrings: per-document postings on a multi-value field; a
// token in ANY element posts the id once -> superset of per-element matches.
func TestMatchIndexIDValueStrings(t *testing.T) {
	p := newPayloadIndexID()
	docs := map[uint64][]string{
		1: {"red apple", "green pear"},
		2: {"blue sky", "red car"},
		3: {"green grass", "yellow sun"},
		4: {"red", "apple"}, // red+apple only across DIFFERENT elements
	}
	metas := make(map[uint64]Metadata)
	for id, ss := range docs {
		m := Metadata{"tags": NewStrings(ss)}
		metas[id] = m
		p.reindex(id, m)
	}
	cand := idMatchSuperset(t, p, metas, "tags", "red apple", 100)
	truth := bruteMatch(t, metas, Filter{Op: FilterMatch, Field: "tags", Value: NewString("red apple")})
	// doc 1 is a true predicate match; doc 4 is in the superset but NOT a match.
	if !contains(truth, 1) {
		t.Error("expected doc 1 to be a true predicate match")
	}
	if contains(truth, 4) {
		t.Error("doc 4 should NOT be a predicate match (cross-element)")
	}
	if !contains(cand, 4) {
		t.Error("doc 4 SHOULD be in the index superset (per-doc posting)")
	}
}

// TestMatchIndexIDMutationDropsOldTokens: reindex on mutation drops old tokens
// (no ghosts), delete removes, rebuild repopulates.
func TestMatchIndexIDMutationDropsOldTokens(t *testing.T) {
	p := newPayloadIndexID()
	live := map[uint64]Metadata{
		1: {"body": NewString("orange fruit")},
		2: {"body": NewString("apple fruit")},
	}
	for id, m := range live {
		p.reindex(id, m)
	}
	mustHave := func(q string, wantIDs ...uint64) {
		t.Helper()
		ids := idCandidates(t, p, Filter{Op: FilterMatch, Field: "body", Value: NewString(q)}, 100, true)
		if len(ids) != len(wantIDs) {
			t.Errorf("match %q: cand=%v want %v", q, ids, wantIDs)
		}
		for _, id := range wantIDs {
			if !contains(ids, id) {
				t.Errorf("match %q: missing id %d", q, id)
			}
		}
	}
	mustHave("orange", 1)
	// Overwrite id 1's text: "orange" must no longer post id 1.
	live[1] = Metadata{"body": NewString("banana fruit")}
	p.reindex(1, live[1])
	if ids, ok := p.candidates(Filter{Op: FilterMatch, Field: "body", Value: NewString("orange")}, 100); ok && len(ids) != 0 {
		t.Errorf("after mutation, 'orange' should match nothing, got %v", ids)
	}
	mustHave("banana", 1)
	mustHave("fruit", 1, 2)
	// Delete id 2 via reindex(nil), then rebuild from the live map.
	p.reindex(2, nil)
	delete(live, 2)
	mustHave("fruit", 1)
	p.rebuild(live)
	mustHave("banana", 1)
	mustHave("fruit", 1)
}

// TestMatchIndexIDInsideAnd: a Match inside an And with an eq term narrows to the
// intersection of both indexed constraints.
func TestMatchIndexIDInsideAnd(t *testing.T) {
	const n = 300
	rng := rand.New(rand.NewSource(3))
	vocab := []string{"alpha", "beta", "gamma", "delta"}
	p := newPayloadIndexID()
	metas := make(map[uint64]Metadata, n)
	for i := uint64(1); i <= n; i++ {
		m := Metadata{
			"body": NewString(vocab[rng.Intn(len(vocab))] + " " + vocab[rng.Intn(len(vocab))]),
			"cat":  NewInt(int64(i % 5)),
		}
		metas[i] = m
		p.reindex(i, m)
	}
	f := Filter{Op: FilterAnd, And: []Filter{
		{Op: FilterEq, Field: "cat", Value: NewInt(2)},
		{Op: FilterMatch, Field: "body", Value: NewString("alpha")},
	}}
	cand := idCandidates(t, p, f, n+10, true)
	truth := bruteMatch(t, metas, f)
	if !isSuperset(cand, truth) {
		t.Fatalf("and(eq,match): cand %v not superset of truth %v", cand, truth)
	}
	// The index intersection already enforces cat==2 (it is indexed too).
	for _, id := range cand {
		if metas[id]["cat"].Int != 2 {
			t.Errorf("and: candidate id %d has cat=%d (should have intersected cat==2)", id, metas[id]["cat"].Int)
		}
	}
}

// TestMatchIndexIDEmptyQuery: a bare empty-token match leaf is ok=false; inside
// an And it is skipped and the eq term still narrows.
func TestMatchIndexIDEmptyQuery(t *testing.T) {
	p := newPayloadIndexID()
	for i := uint64(1); i <= 20; i++ {
		p.reindex(i, Metadata{"body": NewString("word here"), "cat": NewInt(int64(i % 2))})
	}
	if ids, ok := p.candidates(Filter{Op: FilterMatch, Field: "body", Value: NewString("   --- ")}, 100); ok {
		t.Errorf("empty-query match leaf should be ok=false, got %d ids", len(ids))
	}
	f := Filter{Op: FilterAnd, And: []Filter{
		{Op: FilterEq, Field: "cat", Value: NewInt(1)},
		{Op: FilterMatch, Field: "body", Value: NewString("   ")},
	}}
	ids := idCandidates(t, p, f, 100, true)
	for _, id := range ids {
		if id%2 != 1 {
			t.Errorf("and(eq, empty-match): candidate id %d violates cat==1", id)
		}
	}
}

// TestMatchIndexIDRandomSupersetFuzz: random payloads, the candidate set is
// ALWAYS a superset of the brute-force compileMatch matches (the core safety
// invariant), for random single- and multi-token queries.
func TestMatchIndexIDRandomSupersetFuzz(t *testing.T) {
	const n = 600
	rng := rand.New(rand.NewSource(2024))
	vocab := []string{"aa", "bb", "cc", "dd", "ee", "ff", "gg", "hh", "ii", "jj"}
	p := newPayloadIndexID()
	metas := make(map[uint64]Metadata, n)
	for i := uint64(1); i <= n; i++ {
		nw := 1 + rng.Intn(4)
		text := ""
		for j := 0; j < nw; j++ {
			if j > 0 {
				text += " "
			}
			text += vocab[rng.Intn(len(vocab))]
		}
		m := Metadata{"body": NewString(text)}
		metas[i] = m
		p.reindex(i, m)
	}
	for trial := 0; trial < 200; trial++ {
		nq := 1 + rng.Intn(3)
		qstr := ""
		for j := 0; j < nq; j++ {
			if j > 0 {
				qstr += " "
			}
			qstr += vocab[rng.Intn(len(vocab))]
		}
		f := Filter{Op: FilterMatch, Field: "body", Value: NewString(qstr)}
		cand := idCandidates(t, p, f, n+10, true)
		truth := bruteMatch(t, metas, f)
		if !isSuperset(cand, truth) {
			t.Fatalf("fuzz %q: cand %v NOT superset of truth %v", qstr, cand, truth)
		}
	}
}

// contains reports whether ids contains x.
func contains(ids []uint64, x uint64) bool {
	for _, id := range ids {
		if id == x {
			return true
		}
	}
	return false
}
