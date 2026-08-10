// SPDX-License-Identifier: Apache-2.0

package vector

import (
	"fmt"
	"math/rand"
	"slices"
	"testing"
)

// TestIVFPreferFilterFirstIsMonotone pins the property filterFirstCrossover's
// binary search depends on and that this task was told NOT to assume carries
// over from hnsw's model: IVF's preferFilterFirst is monotonically non-increasing
// in ncand. It holds because probeCost — the whole right-hand side — is a
// function of (arena size, nprobe, nlist, k) with no ncand term, so the decision
// is the single comparison ncand <= probeCost.
func TestIVFPreferFilterFirstIsMonotone(t *testing.T) {
	ix := newTrainedIVFForCrossover(t)

	for _, k := range []int{1, 10, 100} {
		limit := 5000
		sawTrue, sawFalse := false, false
		prev := true
		for ncand := 0; ncand <= limit; ncand++ {
			got := ix.preferFilterFirst(ncand, k)
			if got && !prev {
				t.Fatalf("k=%d: preferFilterFirst went false->true at ncand=%d (not monotone)", k, ncand)
			}
			prev = got
			if got {
				sawTrue = true
			} else {
				sawFalse = true
			}
		}
		if !sawTrue || !sawFalse {
			t.Fatalf("k=%d: sweep did not straddle the crossover (sawTrue=%v sawFalse=%v)", k, sawTrue, sawFalse)
		}
		// The crossover found by binary search must be the exact boundary.
		cross := ix.filterFirstCrossover(k, limit)
		if !ix.preferFilterFirst(cross, k) {
			t.Fatalf("k=%d: crossover %d is not itself preferred", k, cross)
		}
		if cross < limit && ix.preferFilterFirst(cross+1, k) {
			t.Fatalf("k=%d: crossover %d is not the LARGEST preferred ncand", k, cross)
		}
	}
}

// TestIVFFilterFirstCrossoverMatchesUncappedDecision is the IVF analogue of
// TestFilterFirstCrossoverMatchesUncappedDecision: it proves the early-abort
// planner path (filterFirstCrossover + payloadIndexID.candidatesCapped, used by
// searchNamedFilterFirst and mvFilterFirstCands) makes EXACTLY the same
// filter-first-vs-index decision as the old code, which fully materialized the
// candidate set via candidates() and then separately checked
// len(cands) <= limit && preferFilterFirst(len(cands), k).
//
// It sweeps ncand via Eq filters on groups of a precisely-controlled size,
// across a range straddling the crossover, so both outcomes are exercised.
func TestIVFFilterFirstCrossoverMatchesUncappedDecision(t *testing.T) {
	ix := newTrainedIVFForCrossover(t)

	// The id-keyed payload index is the one the named/MV planner consults; build
	// it over the same group layout so len(candidates) is exactly the group size.
	p := newPayloadIndexID()
	// 499/500/501 straddle the crossover EXACTLY (the fixture pins probeCost to
	// 500). Without the ncand == maxCand case, relaxing the abort test from
	// `> maxCand` to `>= maxCand` still passes every assertion, while silently
	// dropping the filter-first path for a filter matching exactly maxCand docs —
	// i.e. a recall regression on some of the most selective filters there are.
	groupSizes := []int{1, 2, 5, 10, 20, 40, 80, 150, 300, 499, 500, 501, 600, 1200, 2500, 5000}
	id := uint64(1)
	for g, size := range groupSizes {
		for i := 0; i < size; i++ {
			p.reindex(id, Metadata{"grp": NewInt(int64(g))})
			id++
		}
	}

	const k = 10
	limit := ix.effectiveFilterFirstLimit(ix.arena.Size())
	maxCand := ix.filterFirstCrossover(k, limit)

	// The boundary cases above are only meaningful if the crossover actually lands
	// on one of the swept sizes. Assert it rather than trust the arithmetic, so a
	// fixture change that moves probeCost fails loudly instead of quietly
	// surrendering the mutation coverage.
	if !slices.Contains(groupSizes, maxCand) {
		t.Fatalf("crossover %d is not among the swept group sizes %v: the ncand == maxCand "+
			"boundary is untested (adjust groupSizes or the fixture's nlist/nprobe)", maxCand, groupSizes)
	}

	sawTrue, sawFalse := false, false
	for g, size := range groupSizes {
		t.Run(fmt.Sprintf("ncand=%d", size), func(t *testing.T) {
			filter := Filter{Op: FilterEq, Field: "grp", Value: NewInt(int64(g))}

			// OLD decision: materialize fully via the uncapped candidates(), then
			// apply the two checks searchNamedFilterFirst used to make inline.
			oldCands, oldMatOk := p.candidates(filter, limit)
			oldOk := oldMatOk && len(oldCands) <= limit && ix.preferFilterFirst(len(oldCands), k)

			// NEW decision: candidatesCapped with the crossover cap, exactly as the
			// named/MV gates now call it.
			newCands, newOk := p.candidatesCapped(filter, limit, maxCand)

			if newOk != oldOk {
				t.Fatalf("ncand=%d: strategy decision differs: old=%v new=%v", size, oldOk, newOk)
			}
			if oldOk {
				sawTrue = true
			} else {
				sawFalse = true
			}
			if newOk {
				if len(newCands) != len(oldCands) {
					t.Fatalf("ncand=%d: candidate count differs: old=%d new=%d", size, len(oldCands), len(newCands))
				}
				oldSet := make(map[uint64]bool, len(oldCands))
				for _, x := range oldCands {
					oldSet[x] = true
				}
				for _, x := range newCands {
					if !oldSet[x] {
						t.Fatalf("ncand=%d: candidate id %d returned by new path but not old", size, x)
					}
				}
			}
		})
	}
	if !sawTrue || !sawFalse {
		t.Fatalf("sweep did not straddle the crossover: sawTrue=%v sawFalse=%v (adjust groupSizes)", sawTrue, sawFalse)
	}
}

// TestIVFCrossoverConjunctiveAbortMatchesUncapped drives the MULTI-SET arm of
// intersectIDSets, which the single-field sweep above never reaches: with one
// Eq term there is exactly one posting set, so candidatesCapped takes the
// fast path that decides from the map length alone and the in-loop
// `len(out) > maxCand` abort is dead code under that test.
//
// And(Eq grp, Eq band) yields two posting sets, so the intersection is built
// element by element and the abort fires mid-scan. The band sizes straddle the
// crossover exactly (499/500/501) for the same mutation-coverage reason as the
// sweep: at ncand == maxCand the intersection must still be RETURNED, not
// abandoned.
func TestIVFCrossoverConjunctiveAbortMatchesUncapped(t *testing.T) {
	ix := newTrainedIVFForCrossover(t)

	// One wide group, partitioned into bands whose sizes straddle the crossover.
	// The band set is always the smaller of the two, so intersectIDSets scans it
	// and probes the group set — the multi-set path.
	const grp = 7
	bandSizes := []int{499, 500, 501}
	p := newPayloadIndexID()
	id := uint64(1)
	for b, size := range bandSizes {
		for i := 0; i < size; i++ {
			p.reindex(id, Metadata{"grp": NewInt(grp), "band": NewInt(int64(b))})
			id++
		}
	}
	// Padding in the same group but no band, so the group set is strictly larger
	// than every band set and the intersection is a real narrowing.
	for i := 0; i < 2000; i++ {
		p.reindex(id, Metadata{"grp": NewInt(grp)})
		id++
	}

	const k = 10
	limit := ix.effectiveFilterFirstLimit(ix.arena.Size())
	maxCand := ix.filterFirstCrossover(k, limit)
	if !slices.Contains(bandSizes, maxCand) {
		t.Fatalf("crossover %d is not among the band sizes %v: the ncand == maxCand "+
			"boundary of the multi-set path is untested", maxCand, bandSizes)
	}

	sawTrue, sawFalse := false, false
	for b, size := range bandSizes {
		t.Run(fmt.Sprintf("ncand=%d", size), func(t *testing.T) {
			filter := Filter{Op: FilterAnd, And: []Filter{
				{Op: FilterEq, Field: "grp", Value: NewInt(grp)},
				{Op: FilterEq, Field: "band", Value: NewInt(int64(b))},
			}}

			oldCands, oldMatOk := p.candidates(filter, limit)
			oldOk := oldMatOk && len(oldCands) <= limit && ix.preferFilterFirst(len(oldCands), k)
			if oldMatOk && len(oldCands) != size {
				t.Fatalf("fixture: And(grp,band=%d) matched %d ids, want %d "+
					"(the intersection must be exactly the band)", b, len(oldCands), size)
			}

			newCands, newOk := p.candidatesCapped(filter, limit, maxCand)
			if newOk != oldOk {
				t.Fatalf("ncand=%d: strategy decision differs: old=%v new=%v", size, oldOk, newOk)
			}
			if oldOk {
				sawTrue = true
			} else {
				sawFalse = true
			}
			if newOk {
				if len(newCands) != len(oldCands) {
					t.Fatalf("ncand=%d: candidate count differs: old=%d new=%d", size, len(oldCands), len(newCands))
				}
				oldSet := make(map[uint64]bool, len(oldCands))
				for _, x := range oldCands {
					oldSet[x] = true
				}
				for _, x := range newCands {
					if !oldSet[x] {
						t.Fatalf("ncand=%d: candidate id %d returned by new path but not old", size, x)
					}
				}
			}
		})
	}
	if !sawTrue || !sawFalse {
		t.Fatalf("conjunctive sweep did not straddle the crossover: sawTrue=%v sawFalse=%v", sawTrue, sawFalse)
	}
}

// TestUntrainedIVFCrossoverDoesNotCap covers the degenerate arm of IVF's cost
// model: an untrained index (no centroids) brute-forces the whole arena, so
// preferFilterFirst is unconditionally true and the crossover must be the full
// limit — never a tighter cap that would silently reject a materializable set.
func TestUntrainedIVFCrossoverDoesNotCap(t *testing.T) {
	ix, err := newIVF(ivfTestConfig(8))
	if err != nil {
		t.Fatal(err)
	}
	defer ix.Close()
	rng := rand.New(rand.NewSource(5))
	for i := 1; i <= 100; i++ {
		v := make([]float32, 8)
		for j := range v {
			v[j] = float32(rng.NormFloat64())
		}
		if _, _, err := ix.Insert(uint64(i), v, 0, nil, nil, nil, CASCond{}); err != nil {
			t.Fatal(err)
		}
	}
	if len(ix.centroids) != 0 {
		t.Skip("index trained on Insert; the untrained arm is unreachable here")
	}
	if got := ix.filterFirstCrossover(10, 10_000); got != 10_000 {
		t.Fatalf("untrained crossover = %d, want the full limit 10000", got)
	}
}

// newTrainedIVFForCrossover builds a TRAINED IVF whose probe cost sits inside
// the crossover sweep's ncand range: nlist/nprobe are pinned so probeCost is a
// known fraction of the arena rather than whatever the auto-tuner picks.
func newTrainedIVFForCrossover(t *testing.T) *ivf {
	t.Helper()
	const dim, n = 8, 10_000
	cfg := ivfTestConfig(dim)
	cfg.IVFNlist = 100
	cfg.IVFNprobe = 5 // probeCost ~= n * 5/100 = 500 candidates
	cfg.FilterFirstThreshold = 100_000
	ix, err := newIVF(cfg)
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() { ix.Close() })

	rng := rand.New(rand.NewSource(17))
	ids := make([]uint64, n)
	vecs := make([][]float32, n)
	for i := 0; i < n; i++ {
		v := make([]float32, dim)
		for j := range v {
			v[j] = float32(rng.NormFloat64())
		}
		ids[i] = uint64(i + 1)
		vecs[i] = v
	}
	if err := ix.BuildConcurrent(ids, vecs, 0); err != nil {
		t.Fatal(err)
	}
	if len(ix.centroids) == 0 {
		t.Fatal("IVF did not train: the crossover sweep needs a trained cost model")
	}
	return ix
}
