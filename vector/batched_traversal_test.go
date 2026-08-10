// SPDX-License-Identifier: Apache-2.0

package vector

import (
	"testing"
)

// These tests pin the claim that batching changes NOTHING observable: the
// batched and per-pair search paths return the same result sets, with the same
// distances, after rejecting the same candidates.
//
// That is a stronger claim than batching usually supports, and the reason it
// holds is worth stating because it is exactly the thing a future refactor might
// break. Hoisting every distance in a neighbor list out of the admission loop
// would normally make the "closer than the worst kept" gate read a stale bound —
// candidates gated against a heap their own block-mates should already have
// tightened. It does not here, because expandBatched's admission pass re-reads
// s.nearest.peek() on every iteration and admits in the original neighbor order,
// so each gate sees precisely the heap it saw before. Combined with kernels that
// are bit-identical to the per-pair ones (distance_ny_test.go asserts that
// directly), the two paths are indistinguishable.
//
// So these tests assert EQUALITY, not "no worse". An earlier revision asserted
// only that batched recall was >= per-pair recall, which is worse than weak: it
// is backwards. Reintroducing staleness admits a SUPERSET of candidates, which
// nudges recall UP, so that form would have actively welcomed the regression.
//
// DO NOT DELETE THE selective-predicate ARM AS REDUNDANT. It is not a second
// copy of the no-predicate arm; it is the only assertion here that detects
// admission staleness at all. That was verified by injecting the regression —
// hoisting s.nearest.peek() to a value read once before the loop — and running
// this file against it:
//
//	top-10 result sets   0/200 differ   (insensitive)
//	full ef-sized sets   0/200 differ   (insensitive)
//	filterRejects        41581 vs 40246 (CAUGHT)
//
// The reason is that a stale bound lets extra candidates past the "closer than
// the worst kept" gate and into admit(), but those candidates are by
// construction not actually closer than the true current worst, so they never
// displace anything in the result heap. The extra work is therefore invisible in
// the OUTPUT and visible only in the reject tally. Without a predicate, admit()
// rejects almost nothing and the tally stays flat — hence the arm.
//
// The result-set assertions are still worth their cost: they cover the other
// failure modes (wrong distances, dropped or duplicated candidates, an admission
// loop out of step with its slot list), which the reject tally would not catch.
//
// All arms run the two paths over the SAME index by clearing layerScorer.batch
// on a copy of the scorer, so nothing but the expansion strategy differs — no
// second build, no separate graph.

// perPair returns sc with the batched kernel removed, forcing searchLayerCore
// down the per-slot path.
func perPair(sc layerScorer) layerScorer {
	sc.batch = batchKernel{}
	return sc
}

// TestBatchedTraversalMatchesPerPair is the primary gate. For every query the
// two paths must return the SAME result set, and consequently the same recall
// against an exact brute-force oracle.
//
// Per-query set equality is the assertion that has teeth. Recall equality alone
// does not: recall is an aggregate over 200 queries, so a change that swapped
// one admitted candidate for an equally-good one on many queries would leave it
// untouched. And the earlier "batched recall >= per-pair recall" form was weaker
// still — reintroducing genuine admission staleness admits a SUPERSET of
// candidates, which nudges recall UP, so that assertion would have waved through
// exactly the regression it was written to catch.
//
// The predicate arm additionally compares the filter-reject tallies, so the
// admit-gate interaction is covered and not just the unfiltered walk.
func TestBatchedTraversalMatchesPerPair(t *testing.T) {
	const (
		n    = 6000
		dim  = 64
		k    = 10
		seed = 42
	)
	ids, vecs := siftLikeCorpus(n, dim, seed)
	_, queries := siftLikeCorpus(200, dim, 7)

	h, err := newHNSW(Config{Dim: dim, Metric: L2, M: 16, EfConstruction: 200, EfSearch: 64, Seed: seed})
	if err != nil {
		t.Fatal(err)
	}
	defer h.Close()
	for i, v := range vecs {
		// Tag half the corpus so the selective-predicate arm below rejects
		// roughly every other candidate the walk touches.
		meta := Metadata{"keep": NewBool(i%2 == 0)}
		if _, _, err := h.Insert(ids[i], v, 0, meta, nil, nil, CASCond{}); err != nil {
			t.Fatalf("insert %d: %v", i, err)
		}
	}

	keepEven := func(m Metadata) bool {
		v, ok := m["keep"]
		return ok && v.Bool
	}

	for _, arm := range []struct {
		name string
		pred Predicate
	}{
		{"no-predicate", nil},
		{"selective-predicate", keepEven},
	} {
		t.Run(arm.name, func(t *testing.T) {
			var batchedHits, pairHits, differing, fullDiffer int
			var rejB, rejP uint64
			for qi, q := range queries {
				truth := bruteTopK(vecs, q, k)

				sc := h.exactScorer(q)
				if !sc.batch.ok() {
					t.Fatal("fixture: the unquantized L2 path must have a batched kernel")
				}
				eps := []uint32{h.entryPoint}
				now := uint64(h.now())

				// searchLayerCore flushes each traversal's filter rejections to
				// the shared atomic on return, so the delta around one call is
				// that call's tally. Stats() surfaces the same counter and is
				// checked in aggregate below.
				before := h.filterRejects.Load()
				fullB := append([]slotDist(nil), h.searchLayer(&layerScratch{}, sc, eps, h.cfg.EfSearch, 0, arm.pred, now)...)
				mid := h.filterRejects.Load()
				fullP := append([]slotDist(nil), h.searchLayer(&layerScratch{}, perPair(sc), eps, h.cfg.EfSearch, 0, arm.pred, now)...)
				after := h.filterRejects.Load()
				batched, pair := topKIDs(h, fullB, k), topKIDs(h, fullP, k)
				if !sameSlotDists(fullB, fullP) {
					fullDiffer++
					if fullDiffer <= 3 {
						t.Errorf("query %d: the full ef-sized candidate sets differ (%d vs %d entries)",
							qi, len(fullB), len(fullP))
					}
				}
				rejB += mid - before
				rejP += after - mid

				if !sameIDs(batched, pair) {
					differing++
					if differing <= 3 {
						t.Errorf("query %d: batched and per-pair returned different sets\n  batched=%v\n  perpair=%v",
							qi, batched, pair)
					}
				}
				if got, want := mid-before, after-mid; got != want {
					t.Errorf("query %d: batched rejected %d candidates, per-pair rejected %d — "+
						"the admit gate must fire on the same slots in the same order", qi, got, want)
				}
				for _, id := range batched {
					if truth[id] {
						batchedHits++
					}
				}
				for _, id := range pair {
					if truth[id] {
						pairHits++
					}
				}
			}

			total := float64(len(queries) * k)
			rb, rp := float64(batchedHits)/total, float64(pairHits)/total
			t.Logf("recall@%d batched=%.4f per-pair=%.4f; differing top-%d sets %d/%d; differing ef-sets %d/%d; filterRejects batched=%d per-pair=%d",
				k, rb, rp, k, differing, len(queries), fullDiffer, len(queries), rejB, rejP)

			if fullDiffer != 0 {
				t.Errorf("%d/%d queries produced a different ef-sized candidate set. This is the "+
					"most sensitive signal available without a predicate: admission staleness "+
					"changes which candidates reach the frontier long before it changes the top-%d.",
					fullDiffer, len(queries), k)
			}
			if differing != 0 {
				t.Errorf("%d/%d queries returned a different result set; batching must be "+
					"observationally identical to the per-pair path", differing, len(queries))
			}
			if rb != rp {
				t.Errorf("recall@%d differs: batched=%.6f per-pair=%.6f", k, rb, rp)
			}
			if rejB != rejP {
				t.Errorf("filter rejections differ: batched=%d per-pair=%d", rejB, rejP)
			}
			if arm.pred != nil {
				// Guard the fixture itself: a predicate that rejected nothing
				// would make the whole arm vacuous.
				if rejB == 0 {
					t.Error("fixture: the selective predicate rejected no candidates, so this arm proved nothing")
				}
				if rb == 0 {
					t.Error("fixture: the filtered search returned no true positives")
				}
			}
			// Stats() must agree with the raw counter the loop sampled.
			if got := h.Stats().FilterRejects; got < rejB+rejP {
				t.Errorf("Stats().FilterRejects = %d, below the %d this arm observed", got, rejB+rejP)
			}
		})
	}
}

// TestBatchedExpandToggle exercises batchedExpand=false, the A/B knob that
// otherwise has no test coverage at all — it is declared, documented and read
// once, which is exactly the shape of a switch that silently stops working.
// Flipping it must reproduce the per-pair path, so a search run with it off has
// to match one run with it on.
func TestBatchedExpandToggle(t *testing.T) {
	const (
		n    = 2000
		dim  = 32
		k    = 10
		seed = 5
	)
	ids, vecs := siftLikeCorpus(n, dim, seed)
	_, queries := siftLikeCorpus(50, dim, 9)

	h, err := newHNSW(Config{Dim: dim, Metric: L2, M: 16, EfConstruction: 100, EfSearch: 64, Seed: seed})
	if err != nil {
		t.Fatal(err)
	}
	defer h.Close()
	for i, v := range vecs {
		if _, _, err := h.Insert(ids[i], v, 0, nil, nil, nil, CASCond{}); err != nil {
			t.Fatal(err)
		}
	}

	// The knob is read when the scorer is BUILT, so it must be flipped before
	// exactScorer runs, not before the traversal.
	defer func(prev bool) { batchedExpand = prev }(batchedExpand)

	for qi, q := range queries {
		batchedExpand = true
		scOn := h.exactScorer(q)
		if !scOn.batch.ok() {
			t.Fatal("fixture: batchedExpand=true must yield a batched kernel")
		}
		on := topKIDs(h, h.searchLayer(&layerScratch{}, scOn, []uint32{h.entryPoint}, h.cfg.EfSearch, 0, nil, uint64(h.now())), k)

		batchedExpand = false
		scOff := h.exactScorer(q)
		if scOff.batch.ok() {
			t.Fatal("batchedExpand=false still produced a batched kernel; the knob does not work")
		}
		off := topKIDs(h, h.searchLayer(&layerScratch{}, scOff, []uint32{h.entryPoint}, h.cfg.EfSearch, 0, nil, uint64(h.now())), k)

		if !sameIDs(on, off) {
			t.Fatalf("query %d: batchedExpand on/off disagree\n  on =%v\n  off=%v", qi, on, off)
		}
	}
}

// TestBatchedTraversalDistancesAreExact separates the two ways the two paths
// could disagree. For every slot BOTH traversals returned, the distance recorded
// must be bit-identical — because the kernels are. If this passes and the result
// sets still differ, the difference is admission ordering (the documented,
// accepted effect); if it fails, the kernel is wrong and no amount of
// tie-breaking explains it.
func TestBatchedTraversalDistancesAreExact(t *testing.T) {
	const (
		n    = 3000
		dim  = 32
		seed = 11
	)
	for _, m := range []Metric{L2, Cosine, DotProduct} {
		ids, vecs := siftLikeCorpus(n, dim, seed)
		_, queries := siftLikeCorpus(60, dim, 5)

		h, err := newHNSW(Config{Dim: dim, Metric: m, M: 16, EfConstruction: 100, EfSearch: 64, Seed: seed})
		if err != nil {
			t.Fatal(err)
		}
		for i, v := range vecs {
			if _, _, err := h.Insert(ids[i], v, 0, nil, nil, nil, CASCond{}); err != nil {
				t.Fatalf("insert %d: %v", i, err)
			}
		}

		var compared int
		for _, q := range queries {
			sc := h.exactScorer(normalizedFor(m, q))
			eps := []uint32{h.entryPoint}
			now := uint64(h.now())

			byBatch := map[uint32]float32{}
			for _, sd := range h.searchLayer(&layerScratch{}, sc, eps, h.cfg.EfSearch, 0, nil, now) {
				byBatch[sd.slot] = sd.dist
			}
			for _, sd := range h.searchLayer(&layerScratch{}, perPair(sc), eps, h.cfg.EfSearch, 0, nil, now) {
				d, ok := byBatch[sd.slot]
				if !ok {
					continue // admission-order difference; covered by the recall test
				}
				compared++
				if d != sd.dist {
					t.Fatalf("metric=%v slot %d: batched dist=%v per-pair dist=%v — the two "+
						"paths must produce identical distances", m, sd.slot, d, sd.dist)
				}
			}
		}
		if compared == 0 {
			t.Fatalf("metric=%v: compared no overlapping slots; the test proved nothing", m)
		}
		t.Logf("metric=%v: %d overlapping slots all bit-identical", m, compared)
		h.Close()
	}
}

// normalizedFor mirrors what the search path does to a query before scoring:
// Cosine collapses to 1-dot only on unit vectors, so the query is normalized.
func normalizedFor(m Metric, q []float32) []float32 {
	c := append([]float32(nil), q...)
	if m == Cosine {
		normalize(c)
	}
	return c
}

// topKIDs maps the first k traversal results to arena ids.
func topKIDs(h *hnsw, res []slotDist, k int) []uint64 {
	if len(res) > k {
		res = res[:k]
	}
	out := make([]uint64, 0, len(res))
	for _, sd := range res {
		out = append(out, h.arena.ID(sd.slot))
	}
	return out
}

func sameIDs(a, b []uint64) bool {
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

// sameSlotDists compares two traversal result slices exactly, distances
// included. It is deliberately stricter than comparing the top-k ids: the
// ef-sized set reacts to a change in which candidates reached the frontier,
// which is the first observable symptom of admission staleness.
func sameSlotDists(a, b []slotDist) bool {
	if len(a) != len(b) {
		return false
	}
	for i := range a {
		if a[i].slot != b[i].slot || a[i].dist != b[i].dist {
			return false
		}
	}
	return true
}
