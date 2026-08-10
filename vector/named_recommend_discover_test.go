// SPDX-License-Identifier: Apache-2.0

package vector

import (
	"errors"
	"math"
	"sort"
	"testing"
)

// namedRDConfig builds a NamedCollection config with THREE dense spaces of
// DIFFERENT metrics — "cos" (cosine), "dot" (dot product), "l2" (L2) — all dim 3.
// The distinct metrics let the per-space-metric tests prove the named recommend/
// discover scorers use cfg[Space].Metric (a wrong-metric derive is the #1 named
// bug class), since the same example ids derive/rank differently per space.
func namedRDConfig() map[string]NamedVectorParams {
	return map[string]NamedVectorParams{
		"cos": {Dim: 3, Metric: Cosine},
		"dot": {Dim: 3, Metric: DotProduct},
		"l2":  {Dim: 3, Metric: L2},
	}
}

// newNamedRDCorpus inserts the SAME vectors into all three spaces so a per-space
// recommend/discover query has real, metric-sensitive rankings. Points are chosen
// so candidate scores are strictly distinct (tiebreak-agnostic oracles).
func newNamedRDCorpus(t *testing.T) *NamedCollection {
	t.Helper()
	nc, err := NewNamedCollection("default/nrd", namedRDConfig())
	if err != nil {
		t.Fatalf("new named: %v", err)
	}
	t.Cleanup(func() { nc.Close() })

	pts := map[uint64][]float32{
		1: {1.0, 0.1, 0.0},
		2: {0.2, 1.0, 0.1},
		3: {0.0, 0.2, 1.0},
		4: {0.9, 0.4, 0.05},
		5: {0.3, 0.9, 0.6},
		6: {0.7, 0.7, 0.2},
		7: {0.1, 0.5, 0.95},
		8: {0.6, 0.2, 0.8},
	}
	for id, v := range pts {
		dense := map[string][]float32{"cos": v, "dot": v, "l2": v}
		if err := nc.Insert(id, dense, Metadata{"k": NewInt(int64(id))}, 0); err != nil {
			t.Fatalf("insert %d: %v", id, err)
		}
	}
	return nc
}

// oracleDiscoverScore re-implements the discover per-candidate score INDEPENDENTLY
// of the production discoverScore (different code, same contract): each pair scores
// +1 when the candidate is closer to the positive than the negative (sim is -dist),
// else simPos-simNeg. Uses oracleSim's distance via the metric.
func oracleDiscoverScore(cv []float32, pairs []DiscoverPair, metric Metric) float32 {
	negDist := func(a, b []float32) float32 {
		var dot, l2 float32
		for i := range a {
			dot += a[i] * b[i]
			d := a[i] - b[i]
			l2 += d * d
		}
		switch metric {
		case Cosine:
			return dot - 1 // -dist where dist = 1-dot (stored cosine vectors are unit-norm)
		case DotProduct:
			return dot // -dist where dist = -dot  →  dot
		default: // L2
			return -l2 // -dist where dist = squared-L2
		}
	}
	var score float32
	for _, p := range pairs {
		simPos := negDist(cv, p.Pos)
		simNeg := negDist(cv, p.Neg)
		if simPos >= simNeg {
			score++
		} else {
			score += simPos - simNeg
		}
	}
	return score
}

// spaceVecs returns the stored vectors for ids in a named space (the SAME
// metric-normalized vectors the engine scores).
func spaceVecs(t *testing.T, nc *NamedCollection, space string, ids []uint64) map[uint64][]float32 {
	t.Helper()
	idx, ok := nc.indexes[space]
	if !ok {
		t.Fatalf("space %q not found", space)
	}
	return idx.vecsForIDs(ids)
}

func allCorpusIDs() []uint64 { return []uint64{1, 2, 3, 4, 5, 6, 7, 8} }

func resultIDsSorted(res []Result) []uint64 {
	ids := make([]uint64, len(res))
	for i, r := range res {
		ids[i] = r.ID
	}
	return ids
}

// TestNamedAverageRecommendInSpace: a named AVERAGE_VECTOR recommend leaf == an
// independent per-space mean-diff dense-in-space search oracle (examples excluded).
// The oracle derives normalize(mean(pos)-mean(neg)) in the SPACE metric WITHOUT
// calling DeriveRecommendVector, then runs SearchNamed (the in-space engine search)
// and prunes the example ids — proving the named pre-pass derives in the right space
// + metric and the rewrite→LeafDense→SearchNamed path returns that exact ranking.
func TestNamedAverageRecommendInSpace(t *testing.T) {
	for _, space := range []string{"cos", "dot", "l2"} {
		t.Run(space, func(t *testing.T) {
			nc := newNamedRDCorpus(t)
			metric := namedRDConfig()[space].Metric
			positive := []uint64{1, 4}
			negative := []uint64{3}
			k := 3

			spec := QuerySpec{
				Mode: ModeFusion,
				K:    k,
				Prefetch: srcs([]QueryLeaf{
					{Kind: LeafRecommend, Space: space, Positive: positive, Negative: negative, K: k},
				}...),
			}
			qr, err := nc.Query(spec)
			if err != nil {
				t.Fatalf("Query: %v", err)
			}

			// Oracle: independent mean-diff derive in this space's metric.
			stored := spaceVecs(t, nc, space, allCorpusIDs())
			derived := make([]float32, 3)
			for _, id := range positive {
				for i := range derived {
					derived[i] += stored[id][i]
				}
			}
			for i := range derived {
				derived[i] /= float32(len(positive))
			}
			negMean := make([]float32, 3)
			for _, id := range negative {
				for i := range negMean {
					negMean[i] += stored[id][i]
				}
			}
			for i := range negMean {
				negMean[i] /= float32(len(negative))
			}
			for i := range derived {
				derived[i] -= negMean[i]
			}
			if metric == Cosine {
				normalize(derived)
			}
			oracleRes, err := nc.SearchNamed(space, derived, k+len(positive)+len(negative), Filter{})
			if err != nil {
				t.Fatalf("oracle SearchNamed: %v", err)
			}
			exclude := map[uint64]struct{}{}
			for _, id := range append(append([]uint64{}, positive...), negative...) {
				exclude[id] = struct{}{}
			}
			var wantIDs []uint64
			for _, r := range oracleRes {
				if _, drop := exclude[r.ID]; drop {
					continue
				}
				wantIDs = append(wantIDs, r.ID)
				if len(wantIDs) == k {
					break
				}
			}
			gotIDs := resultIDsSorted(qr.Fused)
			if !uint64SliceEq(gotIDs, wantIDs) {
				t.Errorf("named AVERAGE recommend in %q\n got=%v\nwant=%v", space, gotIDs, wantIDs)
			}
			for _, id := range gotIDs {
				if _, bad := exclude[id]; bad {
					t.Errorf("example id %d not excluded from result %v", id, gotIDs)
				}
			}
		})
	}
}

// TestNamedBestScoreRecommendInSpace: a named BEST_SCORE recommend leaf == an
// INDEPENDENT per-candidate best-score oracle in the SPACE metric (oracleBestScore,
// NOT the production bestScore). Examples excluded; inputs chosen for distinct scores.
func TestNamedBestScoreRecommendInSpace(t *testing.T) {
	for _, space := range []string{"cos", "dot", "l2"} {
		t.Run(space, func(t *testing.T) {
			nc := newNamedRDCorpus(t)
			metric := namedRDConfig()[space].Metric
			positive := []uint64{1}
			// L2 + negatives is fail-loud, so test L2 with no negatives.
			var negative []uint64
			if metric != L2 {
				negative = []uint64{3}
			}
			k := 3

			spec := QuerySpec{
				Mode: ModeFusion,
				K:    k,
				Prefetch: srcs([]QueryLeaf{
					{Kind: LeafRecommend, Strategy: RecommendBestScore, ScoreDesc: true, Space: space, Positive: positive, Negative: negative, K: k},
				}...),
			}
			qr, err := nc.Query(spec)
			if err != nil {
				t.Fatalf("Query: %v", err)
			}

			// Independent oracle: score every candidate by oracleBestScore over the SPACE's
			// stored vectors, exclude examples, top-k score-desc (ties → lower id).
			stored := spaceVecs(t, nc, space, allCorpusIDs())
			posVecs := [][]float32{stored[positive[0]]}
			var negVecs [][]float32
			for _, id := range negative {
				negVecs = append(negVecs, stored[id])
			}
			exclude := map[uint64]struct{}{}
			for _, id := range append(append([]uint64{}, positive...), negative...) {
				exclude[id] = struct{}{}
			}
			type sc struct {
				id    uint64
				score float32
			}
			var scored []sc
			for _, id := range allCorpusIDs() {
				if _, drop := exclude[id]; drop {
					continue
				}
				scored = append(scored, sc{id, oracleBestScore(stored[id], posVecs, negVecs, metric)})
			}
			sort.SliceStable(scored, func(a, b int) bool {
				if scored[a].score != scored[b].score {
					return scored[a].score > scored[b].score
				}
				return scored[a].id < scored[b].id
			})
			assertDistinctScores(t, len(scored), func(i int) float32 { return scored[i].score }, k)
			var wantIDs []uint64
			for i := 0; i < k && i < len(scored); i++ {
				wantIDs = append(wantIDs, scored[i].id)
			}
			gotIDs := resultIDsSorted(qr.Fused)
			if !uint64SliceEq(gotIDs, wantIDs) {
				t.Errorf("named BEST_SCORE recommend in %q\n got=%v\nwant=%v", space, gotIDs, wantIDs)
			}
			for _, id := range gotIDs {
				if _, bad := exclude[id]; bad {
					t.Errorf("example id %d not excluded from result %v", id, gotIDs)
				}
			}
		})
	}
}

// TestNamedDiscoverInSpace: a named discover leaf == an INDEPENDENT per-candidate
// discover oracle in the SPACE metric (oracleDiscoverScore). Inputs chosen distinct.
func TestNamedDiscoverInSpace(t *testing.T) {
	for _, space := range []string{"cos", "dot", "l2"} {
		t.Run(space, func(t *testing.T) {
			nc := newNamedRDCorpus(t)
			metric := namedRDConfig()[space].Metric
			targetID := uint64(6)
			ctxIDs := []ContextPair{{Positive: 1, Negative: 3}, {Positive: 2, Negative: 8}}
			k := 3

			spec := QuerySpec{
				Mode: ModeFusion,
				K:    k,
				Prefetch: srcs([]QueryLeaf{
					{Kind: LeafDiscover, ScoreDesc: true, Space: space, DiscoverTargetID: []uint64{targetID}, DiscoverContextIDs: ctxIDs, K: k},
				}...),
			}
			qr, err := nc.Query(spec)
			if err != nil {
				t.Fatalf("Query: %v", err)
			}

			stored := spaceVecs(t, nc, space, allCorpusIDs())
			pairs := make([]DiscoverPair, len(ctxIDs))
			for i, cp := range ctxIDs {
				pairs[i] = DiscoverPair{Pos: stored[cp.Positive], Neg: stored[cp.Negative]}
			}
			// The discover score is integer-dominated (+1/pair), so many candidates tie on
			// score; the engine breaks score ties on the SEED-POOL distance (the target is
			// the seed). Replicate that tiebreak in the oracle: score desc, then distance to
			// the (metric-normalized) seed asc, then id — matching sortDiscover.
			seed := make([]float32, 3)
			copy(seed, stored[targetID])
			if metric == Cosine {
				normalize(seed)
			}
			dist := pickDistDim(metric, 3)
			type sc struct {
				id    uint64
				score float32
				d     float32
			}
			var scored []sc
			for _, id := range allCorpusIDs() {
				scored = append(scored, sc{id, oracleDiscoverScore(stored[id], pairs, metric), dist(seed, stored[id])})
			}
			sort.SliceStable(scored, func(a, b int) bool {
				if scored[a].score != scored[b].score {
					return scored[a].score > scored[b].score
				}
				if scored[a].d != scored[b].d {
					return scored[a].d < scored[b].d
				}
				return scored[a].id < scored[b].id
			})
			var wantIDs []uint64
			for i := 0; i < k && i < len(scored); i++ {
				wantIDs = append(wantIDs, scored[i].id)
			}
			gotIDs := resultIDsSorted(qr.Fused)
			if !uint64SliceEq(gotIDs, wantIDs) {
				t.Errorf("named discover in %q\n got=%v\nwant=%v", space, gotIDs, wantIDs)
			}
		})
	}
}

// TestNamedRecommendPerSpaceMetricDiffers proves the SPACE metric drives the
// ranking: the SAME BEST_SCORE recommend over the cosine space vs the dot space vs
// the L2 space yields DIFFERENT top-k orderings (a hardcoded/dense metric would make
// them identical). At least one pair of spaces must differ.
func TestNamedRecommendPerSpaceMetricDiffers(t *testing.T) {
	nc := newNamedRDCorpus(t)
	positive := []uint64{1}
	k := 4
	run := func(space string) []uint64 {
		spec := QuerySpec{
			Mode: ModeFusion,
			K:    k,
			Prefetch: srcs([]QueryLeaf{
				{Kind: LeafRecommend, Strategy: RecommendBestScore, ScoreDesc: true, Space: space, Positive: positive, K: k},
			}...),
		}
		qr, err := nc.Query(spec)
		if err != nil {
			t.Fatalf("Query %q: %v", space, err)
		}
		return resultIDsSorted(qr.Fused)
	}
	cos := run("cos")
	dot := run("dot")
	l2 := run("l2")
	if uint64SliceEq(cos, dot) && uint64SliceEq(cos, l2) {
		t.Errorf("per-space metric ignored: cos=%v dot=%v l2=%v (all equal — metric not applied)", cos, dot, l2)
	}
}

// TestNamedRecommendDiscoverExamplesExcluded explicitly verifies the example ids
// never appear in a named recommend result (both strategies).
func TestNamedRecommendDiscoverExamplesExcluded(t *testing.T) {
	nc := newNamedRDCorpus(t)
	positive := []uint64{1, 2}
	negative := []uint64{3}
	k := 8 // ask for everything so an un-excluded example would surface
	for _, strat := range []RecommendStrategy{RecommendAverageVector, RecommendBestScore} {
		neg := negative
		// BEST_SCORE over cosine allows negatives; keep both consistent on cos space.
		spec := QuerySpec{
			Mode: ModeFusion,
			K:    k,
			Prefetch: srcs([]QueryLeaf{
				{Kind: LeafRecommend, Strategy: strat, ScoreDesc: strat == RecommendBestScore, Space: "cos", Positive: positive, Negative: neg, K: k},
			}...),
		}
		qr, err := nc.Query(spec)
		if err != nil {
			t.Fatalf("Query strat=%d: %v", strat, err)
		}
		for _, r := range qr.Fused {
			if r.ID == 1 || r.ID == 2 || r.ID == 3 {
				t.Errorf("strat=%d: example id %d present in result %v", strat, r.ID, resultIDsSorted(qr.Fused))
			}
		}
	}
}

// TestMVRecommendDiscoverFailLoud: an MV query carrying a recommend or a discover
// leaf is rejected fail-loud (inherent semantic limitation, documented), NOT pooled.
func TestMVRecommendDiscoverFailLoud(t *testing.T) {
	m, _ := mvHybridFixture(t)
	cases := []struct {
		name string
		spec QuerySpec
		want error
	}{
		{
			name: "recommend",
			spec: QuerySpec{Mode: ModeFusion, K: 3, Prefetch: srcs([]QueryLeaf{
				{Kind: LeafRecommend, Strategy: RecommendBestScore, ScoreDesc: true, Positive: []uint64{1}, RecPosVecs: [][]float32{{1, 0, 0}}},
			}...)},
			want: ErrQueryMVRecommendUnsupported,
		},
		{
			name: "discover",
			spec: QuerySpec{Mode: ModeFusion, K: 3, Prefetch: srcs([]QueryLeaf{
				{Kind: LeafDiscover, ScoreDesc: true, DiscoverContext: []DiscoverPair{{Pos: []float32{1, 0, 0}, Neg: []float32{0, 1, 0}}}},
			}...)},
			want: ErrQueryMVDiscoverUnsupported,
		},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			if _, err := m.Query(tc.spec); !errors.Is(err, tc.want) {
				t.Fatalf("MV %s err = %v, want %v", tc.name, err, tc.want)
			}
		})
	}
}

// TestNamedRecommendL2Negatives: BEST_SCORE recommend on an L2 space WITH negatives
// is fail-loud (the -max_neg sign-flip inverts the non-positive L2 ranking), using
// the SPACE metric (not a dense default).
func TestNamedRecommendL2Negatives(t *testing.T) {
	nc := newNamedRDCorpus(t)
	spec := QuerySpec{
		Mode: ModeFusion,
		K:    3,
		Prefetch: srcs([]QueryLeaf{
			{Kind: LeafRecommend, Strategy: RecommendBestScore, ScoreDesc: true, Space: "l2", Positive: []uint64{1}, Negative: []uint64{3}, K: 3},
		}...),
	}
	if _, err := nc.Query(spec); !errors.Is(err, ErrRecommendBestScoreL2Negatives) {
		t.Fatalf("L2 best-score + negatives err = %v, want ErrRecommendBestScoreL2Negatives", err)
	}
}

// TestNamedRerankRootKindBoundary documents the precise boundary of named
// RERANK: discover and BEST_SCORE-recommend root leaves fail loud with
// ErrQueryNamedRerankRootKind (they work only as PREFETCH lanes, not as rerank
// roots — scoreByIDs cannot reduce a multi-exemplar scorer to a candidate
// re-ranking). AVERAGE-recommend is the positive contrast: it rewrites to a
// LeafDense in the pre-pass, so it IS legal as a root.
//
// This test locks in three behaviours simultaneously:
//
//	(a) discover root → ErrQueryNamedRerankRootKind, no panic, no empty
//	(b) BEST_SCORE-recommend root → ErrQueryNamedRerankRootKind, no panic, no empty
//	(c) AVERAGE-recommend root → OK (the rewrite-to-dense path works end-to-end)
func TestNamedRerankRootKindBoundary(t *testing.T) {
	nc := newNamedRDCorpus(t)

	// A minimal prefetch lane (dense, "cos") used as the candidate source in the
	// RERANK specs below. Its identity doesn't affect the boundary under test —
	// the error fires in scoreByIDs after candidate union, before any scoring.
	prefetchLane := QueryLeaf{
		Kind:  LeafDense,
		Space: "cos",
		Dense: []float32{1.0, 0.1, 0.0},
		K:     5,
	}

	// (a) DISCOVER root — must fail loud, not panic, not empty.
	discoverRoot := QueryLeaf{
		Kind:  LeafDiscover,
		Space: "cos",
		K:     3,
		// DiscoverTargetID / DiscoverContextIDs are resolved to vectors by the
		// named pre-pass; the failure fires in scoreByIDs after resolution, so we
		// supply valid ids that exist in the corpus.
		DiscoverTargetID:   []uint64{1},
		DiscoverContextIDs: []ContextPair{{Positive: 2, Negative: 3}},
	}
	discoverSpec := QuerySpec{
		Mode:     ModeRerank,
		K:        3,
		Root:     discoverRoot,
		Prefetch: []QuerySource{{Leaf: &prefetchLane}},
	}
	_, discErr := nc.Query(discoverSpec)
	if discErr == nil {
		t.Fatal("discover RERANK root: got nil error, want ErrQueryNamedRerankRootKind")
	}
	if !errors.Is(discErr, ErrQueryNamedRerankRootKind) {
		t.Errorf("discover RERANK root: err = %v, want ErrQueryNamedRerankRootKind", discErr)
	}

	// (b) BEST_SCORE-recommend root — must fail loud, not panic, not empty.
	bestScoreRoot := QueryLeaf{
		Kind:     LeafRecommend,
		Space:    "cos",
		K:        3,
		Strategy: RecommendBestScore,
		Positive: []uint64{1, 4},
		Negative: []uint64{3},
	}
	bestScoreSpec := QuerySpec{
		Mode:     ModeRerank,
		K:        3,
		Root:     bestScoreRoot,
		Prefetch: []QuerySource{{Leaf: &prefetchLane}},
	}
	_, bsErr := nc.Query(bestScoreSpec)
	if bsErr == nil {
		t.Fatal("BEST_SCORE-recommend RERANK root: got nil error, want ErrQueryNamedRerankRootKind")
	}
	if !errors.Is(bsErr, ErrQueryNamedRerankRootKind) {
		t.Errorf("BEST_SCORE-recommend RERANK root: err = %v, want ErrQueryNamedRerankRootKind", bsErr)
	}

	// (c) AVERAGE-recommend root — the pre-pass rewrites it to a LeafDense, so
	// scoreByIDs receives a dense leaf and succeeds. This is the positive contrast
	// documenting that the boundary is at discover/BEST_SCORE, not all recommend.
	averageRoot := QueryLeaf{
		Kind:     LeafRecommend,
		Space:    "cos",
		K:        3,
		Strategy: RecommendAverageVector,
		Positive: []uint64{1, 4},
	}
	averageSpec := QuerySpec{
		Mode:     ModeRerank,
		K:        3,
		Root:     averageRoot,
		Prefetch: []QuerySource{{Leaf: &prefetchLane}},
	}
	avgResult, avgErr := nc.Query(averageSpec)
	if avgErr != nil {
		t.Fatalf("AVERAGE-recommend RERANK root: unexpected error %v", avgErr)
	}
	if len(avgResult.Fused) == 0 {
		t.Error("AVERAGE-recommend RERANK root: got empty Fused results, want non-empty")
	}
}

// TestDenseRecommendDiscoverStillRejectSpace: the DENSE Query still rejects a
// Space-bearing recommend/discover leaf (byte/behaviour-identical to before — named
// recommend/discover is a NEW leaf-kind+space combo, not a dense behaviour change).
func TestDenseRecommendDiscoverStillRejectSpace(t *testing.T) {
	c := newRecommendQueryCorpus(t, Cosine)
	recSpec := QuerySpec{
		Mode: ModeFusion, K: 3,
		Prefetch: srcs([]QueryLeaf{{Kind: LeafRecommend, Space: "cos", Positive: []uint64{1}}}...),
	}
	if _, err := c.Query(recSpec); !errors.Is(err, ErrQueryRecommendHasSpace) && !errors.Is(err, ErrQueryDenseLeafHasSpace) {
		t.Errorf("dense Space-bearing recommend err = %v, want ErrQueryRecommendHasSpace or ErrQueryDenseLeafHasSpace", err)
	}
	discSpec := QuerySpec{
		Mode: ModeFusion, K: 3,
		Prefetch: srcs([]QueryLeaf{{Kind: LeafDiscover, ScoreDesc: true, Space: "cos", DiscoverContext: []DiscoverPair{{Pos: []float32{1, 0}, Neg: []float32{0, 1}}}}}...),
	}
	if _, err := c.Query(discSpec); !errors.Is(err, ErrQueryDiscoverHasSpace) && !errors.Is(err, ErrQueryDenseLeafHasSpace) {
		t.Errorf("dense Space-bearing discover err = %v, want ErrQueryDiscoverHasSpace or ErrQueryDenseLeafHasSpace", err)
	}
}

// --- small shared helpers (this file only) ---

func uint64SliceEq(a, b []uint64) bool {
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

// assertDistinctScores fails if the top-(k+1) scores are not strictly distinct, so
// the oracle's id ordering is unambiguous (tiebreak-agnostic vs the engine's seed-
// distance tiebreak). n is the scored count; score(i) the i-th score (sorted desc).
func assertDistinctScores(t *testing.T, n int, score func(int) float32, k int) {
	t.Helper()
	lim := k + 1
	if lim > n {
		lim = n
	}
	for i := 1; i < lim; i++ {
		if math.Abs(float64(score(i)-score(i-1))) < 1e-6 {
			t.Fatalf("oracle scores not strictly distinct at rank %d (%.6f ~= %.6f) — pick distinct inputs", i, score(i), score(i-1))
		}
	}
}
