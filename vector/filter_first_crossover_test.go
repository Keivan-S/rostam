// SPDX-License-Identifier: Apache-2.0

package vector

import (
	"fmt"
	"math/rand"
	"testing"
)

// TestFilterFirstCrossoverMatchesUncappedDecision proves the early-abort planner
// path (filterFirstCrossover + candidatesCapped, used by searchIntoWith) makes
// EXACTLY the same filter-first-vs-graph decision as the old code, which fully
// materialized the candidate set via candidates() and then separately checked
// len(cands) <= limit && preferFilterFirst(len(cands), k).
//
// It sweeps ncand (via Eq filters on groups of a precisely-controlled size) across
// a range straddling preferFilterFirst's crossover point, so the sweep exercises
// both the filter-first and graph outcomes, not just one side.
func TestFilterFirstCrossoverMatchesUncappedDecision(t *testing.T) {
	const dim = 8
	// Small M/EfSearch so the cost-model crossover sits inside a test-sized
	// corpus (mirrors TestRelativeGateCostModelStillRejects's setup).
	cfg := Config{Dim: dim, Metric: L2, M: 4, EfConstruction: 50, EfSearch: 10, Seed: 1,
		FilterFirstThreshold: 100_000}
	h, err := newHNSW(cfg)
	if err != nil {
		t.Fatal(err)
	}
	defer h.Close()

	rng := rand.New(rand.NewSource(11))
	// Each group g gets exactly groupSizes[g] points tagged {"grp": g}, so an
	// Eq(grp, g) filter narrows to exactly that many candidates -> precise
	// control of ncand for the sweep below.
	groupSizes := []int{1, 2, 5, 10, 20, 40, 80, 150, 300, 600, 1200, 2500, 5000}
	id := uint64(1)
	for g, size := range groupSizes {
		for i := 0; i < size; i++ {
			v := make([]float32, dim)
			for j := range v {
				v[j] = float32(rng.NormFloat64())
			}
			meta := Metadata{"grp": NewInt(int64(g))}
			if _, _, err := h.Insert(id, v, 0, meta, nil, nil, CASCond{}); err != nil {
				t.Fatal(err)
			}
			id++
		}
	}

	const k = 10
	limit := h.effectiveFilterFirstLimit(h.arena.Size())
	cap := h.filterFirstCrossover(k, limit)

	sawTrue, sawFalse := false, false
	for g, size := range groupSizes {
		t.Run(fmt.Sprintf("ncand=%d", size), func(t *testing.T) {
			filter := Filter{Op: FilterEq, Field: "grp", Value: NewInt(int64(g))}

			// OLD decision: materialize fully via the uncapped candidates(),
			// then apply the two checks searchIntoWith used to make inline.
			oldCands, oldMatOk := h.payloadIdx.candidates(filter, limit)
			oldOk := oldMatOk && len(oldCands) <= limit && h.preferFilterFirst(len(oldCands), k)

			// NEW decision: candidatesCapped with the crossover cap, exactly as
			// searchIntoWith now calls it.
			newCands, newOk := h.payloadIdx.candidatesCapped(filter, limit, cap)

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
				oldSet := make(map[uint32]bool, len(oldCands))
				for _, s := range oldCands {
					oldSet[s] = true
				}
				for _, s := range newCands {
					if !oldSet[s] {
						t.Fatalf("ncand=%d: candidate slot %d returned by new path but not old", size, s)
					}
				}
			}
		})
	}
	if !sawTrue || !sawFalse {
		t.Fatalf("sweep did not straddle the crossover: sawTrue=%v sawFalse=%v (adjust groupSizes)", sawTrue, sawFalse)
	}
}
