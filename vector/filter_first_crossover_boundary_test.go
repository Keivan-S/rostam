// SPDX-License-Identifier: Apache-2.0

package vector

import (
	"fmt"
	"math/rand"
	"testing"
)

// denseCrossoverFixture is an hnsw whose filter-first crossover has been
// computed from the index's OWN cost model, plus the ids available for tagging
// into precisely-sized groups.
//
// The two-phase construction is the point. The crossover is a function of
// (arena size, k, EfSearch, M, MaxEfSearch, quantizer) — NOT of any group's
// size — so inserting the whole corpus with no metadata first pins arena.Size(),
// which lets filterFirstCrossover be evaluated before a single group exists.
// Groups are then tagged to sizes derived from that answer. Hardcoding sizes
// (as the older sweep does) cannot express "exactly the crossover", because the
// crossover moves whenever the fixture does.
type denseCrossoverFixture struct {
	h        *hnsw
	ids      []uint64
	k        int
	limit    int
	maxCand  int
	nextFree int // cursor into ids for successive tagging
}

// newDenseCrossoverFixture builds the index and resolves its crossover.
func newDenseCrossoverFixture(t *testing.T) *denseCrossoverFixture {
	t.Helper()
	const dim, n, k = 8, 10_000, 10
	// Small M/EfSearch keep the crossover well inside a test-sized corpus, and a
	// huge FilterFirstThreshold keeps `limit` from being the binding constraint —
	// so the cap under test is the COST MODEL's, not the absolute ceiling's.
	cfg := Config{Dim: dim, Metric: L2, M: 4, EfConstruction: 50, EfSearch: 10, Seed: 1,
		FilterFirstThreshold: 100_000}
	h, err := newHNSW(cfg)
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() { h.Close() })

	rng := rand.New(rand.NewSource(11))
	ids := make([]uint64, 0, n)
	for i := 1; i <= n; i++ {
		v := make([]float32, dim)
		for j := range v {
			v[j] = float32(rng.NormFloat64())
		}
		if _, _, err := h.Insert(uint64(i), v, 0, nil, nil, nil, CASCond{}); err != nil {
			t.Fatal(err)
		}
		ids = append(ids, uint64(i))
	}

	f := &denseCrossoverFixture{h: h, ids: ids, k: k}
	f.limit = h.effectiveFilterFirstLimit(h.arena.Size())
	f.maxCand = h.filterFirstCrossover(f.k, f.limit)

	// Loud guards: every one of these is a precondition for the boundary cases
	// below to mean anything. If a cost-model or fixture change moves the
	// crossover, this fails with the reason rather than quietly degrading into a
	// test that exercises only one side.
	if f.maxCand < 2 {
		t.Fatalf("crossover %d is too small to straddle (need >= 2 so maxCand-1 is a real group); "+
			"raise M/EfSearch or the corpus size", f.maxCand)
	}
	if f.maxCand+1 > f.limit {
		t.Fatalf("crossover %d is at the materialization limit %d, so maxCand+1 cannot be "+
			"distinguished from a limit rejection; raise FilterFirstThreshold", f.maxCand, f.limit)
	}
	if need := 3*f.maxCand + 3; need > len(ids) {
		t.Fatalf("crossover %d needs %d taggable points but the corpus has %d; enlarge n",
			f.maxCand, need, len(ids))
	}
	return f
}

// tag applies meta to the next `size` untagged ids and returns nothing; the
// caller filters on the metadata it supplied. Tagging never changes
// arena.Size(), so the crossover computed at construction stays valid.
func (f *denseCrossoverFixture) tag(t *testing.T, size int, meta Metadata) {
	t.Helper()
	if f.nextFree+size > len(f.ids) {
		t.Fatalf("fixture exhausted: need %d more ids, have %d", size, len(f.ids)-f.nextFree)
	}
	for i := 0; i < size; i++ {
		if _, _, _, err := f.h.SetPayload(f.ids[f.nextFree], meta, nil, CASCond{}); err != nil {
			t.Fatal(err)
		}
		f.nextFree++
	}
}

// checkDecision asserts that the capped planner path makes exactly the decision
// the old materialize-then-check made, and returns that decision.
func (f *denseCrossoverFixture) checkDecision(t *testing.T, label string, filter Filter, wantN int) bool {
	t.Helper()
	h := f.h

	// OLD decision: materialize fully via the uncapped candidates(), then apply
	// the two checks searchIntoWith used to make inline.
	oldCands, oldMatOk := h.payloadIdx.candidates(filter, f.limit)
	if !oldMatOk {
		t.Fatalf("%s: fixture filter is not index-narrowable", label)
	}
	if len(oldCands) != wantN {
		t.Fatalf("%s: fixture matched %d slots, want exactly %d "+
			"(group sizing is the whole point of this test)", label, len(oldCands), wantN)
	}
	oldOk := len(oldCands) <= f.limit && h.preferFilterFirst(len(oldCands), f.k)

	// NEW decision: candidatesCapped with the crossover cap, exactly as
	// searchIntoWith now calls it.
	newCands, newOk := h.payloadIdx.candidatesCapped(filter, f.limit, f.maxCand)
	if newOk != oldOk {
		t.Fatalf("%s (ncand=%d, crossover=%d): strategy decision differs: old=%v new=%v",
			label, wantN, f.maxCand, oldOk, newOk)
	}
	if newOk {
		if len(newCands) != len(oldCands) {
			t.Fatalf("%s: candidate count differs: old=%d new=%d", label, len(oldCands), len(newCands))
		}
		oldSet := make(map[uint32]bool, len(oldCands))
		for _, s := range oldCands {
			oldSet[s] = true
		}
		for _, s := range newCands {
			if !oldSet[s] {
				t.Fatalf("%s: candidate slot %d returned by new path but not old", label, s)
			}
		}
	}
	return oldOk
}

// TestDenseFilterFirstCrossoverBoundary pins the ncand == crossover case on the
// DENSE planner, which the existing sweep in filter_first_crossover_test.go
// cannot reach: its group sizes are hardcoded, so none of them lands on the
// computed crossover, and relaxing intersectSlotSets' abort from `> maxCand` to
// `>= maxCand` leaves the whole suite green.
//
// That mutation is not cosmetic. At exactly maxCand matching documents the
// planner still prefers filter-first, so `>=` would abandon a materialization it
// was about to use and silently fall back to filtered graph traversal — a recall
// regression on some of the most selective filters there are, in the exact
// regime filter-first exists to serve.
//
// The single-Eq filters here produce ONE posting set, so they exercise the
// single-set fast path (payload_index.go, intersectSlotSets' len(sets) == 1 arm)
// which decides from the map length alone.
func TestDenseFilterFirstCrossoverBoundary(t *testing.T) {
	f := newDenseCrossoverFixture(t)
	t.Logf("dense crossover = %d (limit %d, k %d, n %d)", f.maxCand, f.limit, f.k, f.h.arena.Size())

	// Three groups whose sizes straddle the crossover exactly.
	sizes := []int{f.maxCand - 1, f.maxCand, f.maxCand + 1}
	for g, size := range sizes {
		f.tag(t, size, Metadata{"grp": NewInt(int64(g))})
	}

	var sawTrue, sawFalse bool
	for g, size := range sizes {
		name := fmt.Sprintf("ncand=%d", size)
		switch size {
		case f.maxCand - 1:
			name += "(crossover-1)"
		case f.maxCand:
			name += "(crossover)"
		case f.maxCand + 1:
			name += "(crossover+1)"
		}
		t.Run(name, func(t *testing.T) {
			filter := Filter{Op: FilterEq, Field: "grp", Value: NewInt(int64(g))}
			ok := f.checkDecision(t, name, filter, size)
			// State the expected side explicitly rather than only asserting old ==
			// new: if BOTH paths regressed identically the equivalence check alone
			// would still pass.
			wantOK := size <= f.maxCand
			if ok != wantOK {
				t.Fatalf("%s: filter-first chosen = %v, want %v (crossover = %d)",
					name, ok, wantOK, f.maxCand)
			}
			if ok {
				sawTrue = true
			} else {
				sawFalse = true
			}
		})
	}
	if !sawTrue || !sawFalse {
		t.Fatalf("boundary cases did not straddle the crossover: sawTrue=%v sawFalse=%v", sawTrue, sawFalse)
	}
}

// TestDenseFilterFirstCrossoverConjunctiveAbort drives the MULTI-SET arm of
// intersectSlotSets, which no dense test reached: a single Eq yields one posting
// set and takes the fast path, leaving the in-loop `len(out) > maxCand` abort
// dead code under test.
//
// And(Eq grp, Eq band) yields two posting sets, so the intersection is built
// element by element and the abort fires mid-scan. Band sizes straddle the
// crossover for the same boundary reason as above.
func TestDenseFilterFirstCrossoverConjunctiveAbort(t *testing.T) {
	f := newDenseCrossoverFixture(t)

	const grp = 7
	sizes := []int{f.maxCand - 1, f.maxCand, f.maxCand + 1}
	for b, size := range sizes {
		f.tag(t, size, Metadata{"grp": NewInt(grp), "band": NewInt(int64(b))})
	}
	// Padding carrying only the group key, so the grp posting set is strictly
	// larger than every band set. intersectSlotSets scans the SMALLEST set, so
	// this guarantees it scans the band and probes the group — a real narrowing
	// rather than a one-set intersection wearing two names.
	f.tag(t, 500, Metadata{"grp": NewInt(grp)})

	var sawTrue, sawFalse bool
	for b, size := range sizes {
		name := fmt.Sprintf("and/ncand=%d", size)
		t.Run(name, func(t *testing.T) {
			filter := Filter{Op: FilterAnd, And: []Filter{
				{Op: FilterEq, Field: "grp", Value: NewInt(grp)},
				{Op: FilterEq, Field: "band", Value: NewInt(int64(b))},
			}}
			ok := f.checkDecision(t, name, filter, size)
			wantOK := size <= f.maxCand
			if ok != wantOK {
				t.Fatalf("%s: filter-first chosen = %v, want %v (crossover = %d)",
					name, ok, wantOK, f.maxCand)
			}
			if ok {
				sawTrue = true
			} else {
				sawFalse = true
			}
		})
	}
	if !sawTrue || !sawFalse {
		t.Fatalf("conjunctive cases did not straddle the crossover: sawTrue=%v sawFalse=%v", sawTrue, sawFalse)
	}
}
