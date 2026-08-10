// SPDX-License-Identifier: Apache-2.0

package vector

import (
	"errors"
	"fmt"
	"math"
	"sort"
	"testing"
)

// recommendBestSpec builds a single-leaf FUSION spec with one BEST_SCORE recommend
// prefetch leaf (the simplest path: the lane IS the answer after the scorer runs).
func recommendBestSpec(positive, negative []uint64, k int) QuerySpec {
	return QuerySpec{
		Mode: ModeFusion,
		Prefetch: srcs([]QueryLeaf{
			{Kind: LeafRecommend, Strategy: RecommendBestScore, ScoreDesc: true, Positive: positive, Negative: negative, K: k},
		}...),
		K: k,
	}
}

// oracleSim is the test's OWN, formula-independent similarity (higher = more
// similar), written here a SECOND time so the oracle never calls the production
// recommendSim: Cosine = the raw dot product (vectors are unit-normalized at insert,
// so dot == cosine); DotProduct = the raw inner product; L2 = -squared-L2. A formula
// bug in the production sim is then caught by divergence, not masked by shared code.
func oracleSim(a, b []float32, metric Metric) float32 {
	var dot, l2 float32
	for i := range a {
		dot += a[i] * b[i]
		d := a[i] - b[i]
		l2 += d * d
	}
	switch metric {
	case Cosine, DotProduct:
		return dot
	default: // L2
		return -l2
	}
}

// oracleBestScore is the BEST_SCORE merge RE-IMPLEMENTED independently of the
// production bestScore (it uses oracleSim, not recommendSim): max_pos = max sim to
// any positive; max_neg = max sim to any negative; merge = max_pos when no negatives
// or max_pos > max_neg, else -max_neg. Writing the merge a second time means a
// formula bug in production diverges from this oracle (the self-referential trap the
// old oracle fell into by calling the production bestScore is gone).
func oracleBestScore(cv []float32, posVecs, negVecs [][]float32, metric Metric) float32 {
	maxPos := float32(-math.MaxFloat32)
	for _, p := range posVecs {
		if s := oracleSim(cv, p, metric); s > maxPos {
			maxPos = s
		}
	}
	if len(negVecs) == 0 {
		return maxPos
	}
	maxNeg := float32(-math.MaxFloat32)
	for _, n := range negVecs {
		if s := oracleSim(cv, n, metric); s > maxNeg {
			maxNeg = s
		}
	}
	if maxPos > maxNeg {
		return maxPos
	}
	return -maxNeg
}

// bruteBestScoreTopK is the BEST_SCORE oracle: it scores EVERY doc in the index by an
// INDEPENDENT merge (oracleBestScore — NOT the production bestScore) over the index's
// OWN STORED vectors (c.idx.vecsForIDs — the SAME metric-normalized vectors the engine
// scores), excludes the example ids, and returns the top-k ids score-desc. Computing
// the score from first principles (a second implementation) is what makes this a real
// oracle: a production formula bug is caught by divergence, not hidden by calling the
// code under test. To stay tiebreak-agnostic (the engine breaks score ties on the seed
// pool distance, which a brute oracle cannot replicate) the caller MUST choose inputs
// whose candidate scores are strictly distinct; this oracle asserts that and breaks any
// residual tie on lower id deterministically.
func bruteBestScoreTopK(t *testing.T, c *Collection, allIDs []uint64, metric Metric, positive, negative []uint64, k int) []uint64 {
	t.Helper()
	stored := c.idx.vecsForIDs(allIDs)
	posVecs := make([][]float32, 0, len(positive))
	for _, id := range positive {
		posVecs = append(posVecs, stored[id])
	}
	negVecs := make([][]float32, 0, len(negative))
	for _, id := range negative {
		negVecs = append(negVecs, stored[id])
	}
	exclude := make(map[uint64]bool)
	for _, id := range positive {
		exclude[id] = true
	}
	for _, id := range negative {
		exclude[id] = true
	}
	type sc struct {
		id    uint64
		score float32
	}
	scored := make([]sc, 0, len(stored))
	for _, id := range allIDs {
		if exclude[id] {
			continue
		}
		scored = append(scored, sc{id: id, score: oracleBestScore(stored[id], posVecs, negVecs, metric)})
	}
	sort.SliceStable(scored, func(a, b int) bool {
		if scored[a].score != scored[b].score {
			return scored[a].score > scored[b].score
		}
		return scored[a].id < scored[b].id
	})
	// Assert strictly-distinct scores across the top-k boundary so the lower-id
	// tiebreak is never exercised (keeps the oracle == the engine's distance-tiebreak).
	for i := 1; i < len(scored) && i <= k; i++ {
		if scored[i-1].score == scored[i].score {
			t.Fatalf("oracle tie at rank %d (score %g) — choose distinct-score inputs", i, scored[i].score)
		}
	}
	out := make([]uint64, 0, k)
	for _, s := range scored {
		out = append(out, s.id)
		if len(out) == k {
			break
		}
	}
	return out
}

// TestQueryRecommendBestEqualsBruteOracle is the BEST_SCORE equivalence proof: a
// BEST_SCORE recommend leaf must rank the corpus IDENTICALLY to a brute
// per-candidate bestScore oracle (over every doc, excluding the examples). Run for
// every metric so the sim-not-distance handling is proven across Cosine / L2 /
// DotProduct. k is chosen small relative to the corpus so the seed pool (max(4k,50)
// >= the whole 6-doc corpus) surfaces every candidate — the engine top-k then
// matches the brute top-k exactly.
func TestQueryRecommendBestEqualsBruteOracle(t *testing.T) {
	// A graded corpus along a single axis so every candidate's similarity to a
	// positive/negative example is STRICTLY distinct (no score ties → the engine's
	// pool-distance tiebreak never diverges from the brute oracle). 10 docs at evenly
	// graded angles between [1,0] and [0,1].
	allIDs := make([]uint64, 0, 10)
	for _, metric := range []Metric{Cosine, L2, DotProduct} {
		t.Run(fmt.Sprintf("metric=%d", metric), func(t *testing.T) {
			c, err := NewCollection("recobest", Config{Dim: 2, M: 16, EfConstruction: 200, EfSearch: 64, Seed: 1, Metric: metric})
			if err != nil {
				t.Fatalf("NewCollection: %v", err)
			}
			allIDs = allIDs[:0]
			for i := 0; i < 10; i++ {
				id := uint64(i + 1)
				// x decreases, y increases monotonically → strictly graded distances.
				x := 1.0 - float32(i)*0.1
				y := 0.1 + float32(i)*0.1
				if err := c.Insert(id, []float32{x, y}, 0, nil, nil); err != nil {
					t.Fatalf("Insert %d: %v", id, err)
				}
				allIDs = append(allIDs, id)
			}
			cases := []struct {
				pos, neg []uint64
				k        int
			}{
				{pos: []uint64{1}, neg: nil, k: 4},
				{pos: []uint64{1}, neg: []uint64{10}, k: 4},
				{pos: []uint64{2, 3}, neg: []uint64{9}, k: 3},
			}
			for _, tc := range cases {
				// L2 + best_score + negatives is fail-loud (the -max_neg sign-flip inverts
				// the ranking for the non-positive L2 similarity): skip the with-negative
				// cases for L2 here — TestQueryRecommendBestL2NegativesRejected covers the
				// reject, and TestQueryRecommendBestL2NoNegatives covers L2-no-neg correctness.
				if metric == L2 && len(tc.neg) > 0 {
					if _, err := c.Query(recommendBestSpec(tc.pos, tc.neg, tc.k)); !errors.Is(err, ErrRecommendBestScoreL2Negatives) {
						t.Errorf("L2 BEST_SCORE(pos=%v neg=%v) err = %v, want ErrRecommendBestScoreL2Negatives", tc.pos, tc.neg, err)
					}
					continue
				}
				qr, err := c.Query(recommendBestSpec(tc.pos, tc.neg, tc.k))
				if err != nil {
					t.Fatalf("BEST_SCORE Query(pos=%v neg=%v): %v", tc.pos, tc.neg, err)
				}
				got := resultIDs(qr.Fused)
				want := bruteBestScoreTopK(t, c, allIDs, metric, tc.pos, tc.neg, tc.k)
				if len(got) != len(want) {
					t.Fatalf("BEST_SCORE(pos=%v neg=%v) len = %d (%v), want %d (%v)", tc.pos, tc.neg, len(got), got, len(want), want)
				}
				for i := range want {
					if got[i] != want[i] {
						t.Errorf("BEST_SCORE(pos=%v neg=%v) = %v, want brute oracle %v", tc.pos, tc.neg, got, want)
						break
					}
				}
			}
		})
	}
}

// TestQueryRecommendBestPositiveRanksHigh: a doc near a positive ranks high. With a
// cluster-A positive (id 1) the top BEST_SCORE results are the OTHER cluster-A docs
// (2, 3), not cluster-B docs.
func TestQueryRecommendBestPositiveRanksHigh(t *testing.T) {
	c := newRecommendQueryCorpus(t, Cosine)
	qr, err := c.Query(recommendBestSpec([]uint64{1}, nil, 2))
	if err != nil {
		t.Fatalf("Query: %v", err)
	}
	got := sortedIDs(resultIDs(qr.Fused))
	if len(got) != 2 || got[0] != 2 || got[1] != 3 {
		t.Errorf("BEST_SCORE(+1) = %v, want cluster-A neighbors [2 3] (example 1 excluded)", got)
	}
}

// TestQueryRecommendBestNegativeRanksLow is the FORMULA-INDEPENDENT semantic
// ground-truth for the score-style metrics (Cosine + DotProduct, the two metrics for
// which best_score+negatives is well-defined): a doc near a positive must OUTRANK a
// doc near a negative. This is a directional/structural claim — it does NOT depend on
// the exact merge formula, so it catches a ranking inversion (the L2 bug class) even
// if the engine and oracle shared the same wrong formula. Two well-separated clusters
// (A near [1,0], B near [0,1]); positive in A, negative in B. The cluster-B doc
// dominated by the negative also gets a NEGATIVE score (max_neg > max_pos ⇒ -max_neg).
func TestQueryRecommendBestNegativeRanksLow(t *testing.T) {
	for _, metric := range []Metric{Cosine, DotProduct} {
		t.Run(fmt.Sprintf("metric=%d", metric), func(t *testing.T) {
			c, err := NewCollection("recobestneg", Config{Dim: 2, M: 16, EfConstruction: 200, EfSearch: 64, Seed: 1, Metric: metric})
			if err != nil {
				t.Fatalf("NewCollection: %v", err)
			}
			// Strongly separated clusters so the directional claim is unambiguous.
			corpus := map[uint64][]float32{
				1: {1, 0}, 2: {0.99, 0.14}, 3: {0.97, 0.24}, // cluster A
				4: {0, 1}, 5: {0.14, 0.99}, 6: {0.24, 0.97}, // cluster B
			}
			for id, v := range corpus {
				if err := c.Insert(id, v, 0, nil, nil); err != nil {
					t.Fatalf("Insert %d: %v", id, err)
				}
			}
			// Positive cluster A (1), negative cluster B (4).
			qr, err := c.Query(recommendBestSpec([]uint64{1}, []uint64{4}, 4))
			if err != nil {
				t.Fatalf("Query: %v", err)
			}
			scoreOf := func(id uint64) (float32, bool) {
				for _, r := range qr.Fused {
					if r.ID == id {
						return r.Score, true
					}
				}
				return 0, false
			}
			// SEMANTIC GROUND-TRUTH: a cluster-A doc (near the positive) must outrank a
			// cluster-B doc (near the negative). Formula-independent ranking check.
			sA, okA := scoreOf(2)
			sB, okB := scoreOf(5)
			if !okA || !okB {
				t.Fatalf("missing docs: A(2)=%v B(5)=%v in %v", okA, okB, resultIDs(qr.Fused))
			}
			if sA <= sB {
				t.Errorf("near-positive doc 2 score %g should outrank near-negative doc 5 score %g (negative steers away)", sA, sB)
			}
			// The cluster-B doc dominated by the negative must have a NEGATIVE score.
			if sB >= 0 {
				t.Errorf("near-negative doc 5 (dominated by the negative) score = %g, want < 0", sB)
			}
		})
	}
}

// TestQueryRecommendBestL2NegativesRejected: L2 + best_score + negatives must FAIL
// LOUD with ErrRecommendBestScoreL2Negatives — the -max_neg sign-flip inverts the
// ranking for the non-positive L2 similarity (-squared-L2 <= 0). Covers BOTH the
// single-node resolve path (c.Query → resolveRecommendLeaves) and the cluster
// coordinator path (RewriteRecommendLeavesWith) so the reject fires identically on
// P==1 and P>1. L2 WITHOUT negatives stays valid (see TestQueryRecommendBestL2NoNegatives).
func TestQueryRecommendBestL2NegativesRejected(t *testing.T) {
	c, err := NewCollection("recobestl2neg", Config{Dim: 2, M: 16, EfConstruction: 200, EfSearch: 64, Seed: 1, Metric: L2})
	if err != nil {
		t.Fatalf("NewCollection: %v", err)
	}
	for i := 0; i < 6; i++ {
		id := uint64(i + 1)
		if err := c.Insert(id, []float32{1.0 - float32(i)*0.1, 0.1 + float32(i)*0.1}, 0, nil, nil); err != nil {
			t.Fatalf("Insert %d: %v", id, err)
		}
	}

	// Single-node path (P==1): resolveRecommendLeaves rejects before scoring.
	if _, err := c.Query(recommendBestSpec([]uint64{1}, []uint64{6}, 3)); !errors.Is(err, ErrRecommendBestScoreL2Negatives) {
		t.Errorf("single-node L2 best_score + negatives err = %v, want ErrRecommendBestScoreL2Negatives", err)
	}

	// Coordinator/fan-out path (P>1): RewriteRecommendLeavesWith rejects with the same
	// sentinel, given the L2 metric (so P>1 fails loud identically to P==1).
	spec := recommendBestSpec([]uint64{1}, []uint64{6}, 3)
	resolved := c.idx.vecsForIDs([]uint64{1, 6})
	_, rerr := RewriteRecommendLeavesWith(&spec, L2, resolved, func(positive, negative []uint64) ([]float32, error) {
		return DeriveRecommendVector(2, L2, resolved, positive, negative)
	})
	if !errors.Is(rerr, ErrRecommendBestScoreL2Negatives) {
		t.Errorf("coordinator L2 best_score + negatives err = %v, want ErrRecommendBestScoreL2Negatives", rerr)
	}

	// Cosine/DotProduct + negatives stay ALLOWED on the same coordinator path (the
	// reject is L2-specific, not a blanket negatives ban).
	for _, m := range []Metric{Cosine, DotProduct} {
		spec2 := recommendBestSpec([]uint64{1}, []uint64{6}, 3)
		if _, err := RewriteRecommendLeavesWith(&spec2, m, resolved, func(positive, negative []uint64) ([]float32, error) {
			return DeriveRecommendVector(2, m, resolved, positive, negative)
		}); err != nil {
			t.Errorf("metric=%d best_score + negatives coordinator rewrite err = %v, want nil", m, err)
		}
	}
}

// TestQueryRecommendBestL2NoNegatives: L2 + best_score with NO negatives is correct
// (score = nearest-positive similarity, monotonic) and stays ALLOWED. Proven against
// the formula-independent oracle: the engine top-k equals the brute oracle (nearest
// positive ranks highest). A graded single-axis corpus keeps scores strictly distinct.
func TestQueryRecommendBestL2NoNegatives(t *testing.T) {
	c, err := NewCollection("recobestl2nopos", Config{Dim: 2, M: 16, EfConstruction: 200, EfSearch: 64, Seed: 1, Metric: L2})
	if err != nil {
		t.Fatalf("NewCollection: %v", err)
	}
	allIDs := make([]uint64, 0, 10)
	for i := 0; i < 10; i++ {
		id := uint64(i + 1)
		if err := c.Insert(id, []float32{1.0 - float32(i)*0.1, 0.1 + float32(i)*0.1}, 0, nil, nil); err != nil {
			t.Fatalf("Insert %d: %v", id, err)
		}
		allIDs = append(allIDs, id)
	}
	pos := []uint64{1}
	k := 4
	qr, err := c.Query(recommendBestSpec(pos, nil, k))
	if err != nil {
		t.Fatalf("L2 best_score no-neg Query: %v", err)
	}
	got := resultIDs(qr.Fused)
	want := bruteBestScoreTopK(t, c, allIDs, L2, pos, nil, k)
	if len(got) != len(want) {
		t.Fatalf("L2 best_score no-neg len = %d (%v), want %d (%v)", len(got), got, len(want), want)
	}
	for i := range want {
		if got[i] != want[i] {
			t.Errorf("L2 best_score no-neg = %v, want oracle %v", got, want)
			break
		}
	}
	// Nearest-positive ground-truth: the doc adjacent to the positive (id 2) ranks first.
	if len(got) == 0 || got[0] != 2 {
		t.Errorf("L2 best_score no-neg top = %v, want nearest-positive doc 2 first", got)
	}
}

// TestQueryRecommendBestExamplesExcluded: every positive AND negative example id is
// excluded from the BEST_SCORE result set.
func TestQueryRecommendBestExamplesExcluded(t *testing.T) {
	c := newRecommendQueryCorpus(t, Cosine)
	qr, err := c.Query(recommendBestSpec([]uint64{1, 2}, []uint64{4}, 6))
	if err != nil {
		t.Fatalf("Query: %v", err)
	}
	for _, r := range qr.Fused {
		if r.ID == 1 || r.ID == 2 || r.ID == 4 {
			t.Errorf("example id %d leaked into BEST_SCORE results %v", r.ID, resultIDs(qr.Fused))
		}
	}
}

// TestQueryRecommendBestComposesInFusion: a BEST_SCORE recommend leaf composes as
// one prefetch lane in a multi-lane FUSION spec (score-descending, like a sparse/MV
// lane) and the query succeeds, with the example excluded from its lane.
func TestQueryRecommendBestComposesInFusion(t *testing.T) {
	c := newRecommendQueryCorpus(t, Cosine)
	spec := QuerySpec{
		Mode: ModeFusion,
		Prefetch: srcs([]QueryLeaf{
			{Kind: LeafRecommend, Strategy: RecommendBestScore, ScoreDesc: true, Positive: []uint64{1}, K: 5},
			{Kind: LeafDense, Dense: []float32{1, 0}, K: 5},
		}...),
		Method: FusionRRF,
		K:      4,
	}
	qr, err := c.Query(spec)
	if err != nil {
		t.Fatalf("Query: %v", err)
	}
	if qr.Mode != ModeFusion || len(qr.Lanes) != 2 {
		t.Fatalf("mode=%d lanes=%d, want fusion + 2 lanes", qr.Mode, len(qr.Lanes))
	}
	for _, r := range qr.Lanes[0] {
		if r.ID == 1 {
			t.Errorf("BEST_SCORE example id 1 leaked into lane 0 %v", resultIDs(qr.Lanes[0]))
		}
	}
}

// TestQueryRecommendBestComposesAsRerankRoot: a BEST_SCORE recommend leaf composes
// as the RERANK root — the union of the prefetch candidates is re-scored by the
// bestScore merge, top result a cluster-A doc.
func TestQueryRecommendBestComposesAsRerankRoot(t *testing.T) {
	c := newRecommendQueryCorpus(t, Cosine)
	spec := QuerySpec{
		Mode: ModeRerank,
		Root: QueryLeaf{Kind: LeafRecommend, Strategy: RecommendBestScore, ScoreDesc: true, Positive: []uint64{1}, K: 3},
		Prefetch: srcs([]QueryLeaf{
			{Kind: LeafDense, Dense: []float32{1, 0}, K: 6},
		}...),
		K: 3,
	}
	qr, err := c.Query(spec)
	if err != nil {
		t.Fatalf("Query: %v", err)
	}
	if qr.Mode != ModeRerank {
		t.Fatalf("mode=%d, want rerank", qr.Mode)
	}
	for _, r := range qr.Fused {
		if r.ID == 1 {
			t.Errorf("BEST_SCORE example id 1 leaked into rerank result %v", resultIDs(qr.Fused))
		}
	}
	if len(qr.Fused) == 0 {
		t.Fatal("rerank returned no results")
	}
	if top := qr.Fused[0].ID; top != 2 && top != 3 {
		t.Errorf("BEST_SCORE rerank top = %d, want a cluster-A neighbor (2 or 3)", top)
	}
}

// TestQueryRecommendBestFailLoud covers the BEST_SCORE fail-loud edges: no
// positives, a Space-bearing leaf, and all-positives-missing.
func TestQueryRecommendBestFailLoud(t *testing.T) {
	c := newRecommendQueryCorpus(t, Cosine)

	if _, err := c.Query(recommendBestSpec(nil, []uint64{4}, 3)); !errors.Is(err, ErrNoRecommendExamples) {
		t.Errorf("no-positives err = %v, want ErrNoRecommendExamples", err)
	}
	if _, err := c.Query(recommendBestSpec([]uint64{9998, 9999}, nil, 3)); !errors.Is(err, ErrIDNotFound) {
		t.Errorf("missing-positives err = %v, want ErrIDNotFound", err)
	}
	spaceSpec := QuerySpec{
		Mode: ModeFusion,
		Prefetch: srcs([]QueryLeaf{
			{Kind: LeafRecommend, Strategy: RecommendBestScore, ScoreDesc: true, Space: "title", Positive: []uint64{1}, K: 3},
		}...),
		K: 3,
	}
	if _, err := c.Query(spaceSpec); err == nil {
		t.Error("Space-bearing BEST_SCORE recommend leaf should fail loud")
	} else if !errors.Is(err, ErrQueryRecommendHasSpace) && !errors.Is(err, ErrQueryDenseLeafHasSpace) {
		t.Errorf("Space-bearing BEST_SCORE err = %v, want ErrQueryRecommendHasSpace or ErrQueryDenseLeafHasSpace", err)
	}
}

// TestQueryRecommendBestPQDrop: BEST_SCORE on a PQ-HNSW + PQDropVecs collection
// scores candidates against the RECONSTRUCTED example vectors (vecsForIDs → vecFor)
// and still ranks the positive's cluster neighbors first, with the example excluded.
// The equivalence is proven against the brute oracle over the RECONSTRUCTED vectors.
func TestQueryRecommendBestPQDrop(t *testing.T) {
	const (
		dim       = 64
		m         = 8
		nClusters = 8
		n         = 600
		seed      = 7
	)
	ids, vecs, _ := buildPQDropCorpus(n, dim, nClusters, seed)
	cfg := Config{Dim: dim, Metric: L2, M: 16, EfConstruction: 200, EfSearch: 64, Seed: seed,
		Quant: QuantPQ, QuantPQM: m, PQDropVecs: true}
	c, err := NewCollection("recobestpq", cfg)
	if err != nil {
		t.Fatalf("NewCollection: %v", err)
	}
	if err := c.BuildConcurrent(ids, vecs, 4); err != nil {
		t.Fatalf("BuildConcurrent: %v", err)
	}
	h, ok := c.idx.(*hnsw)
	if !ok {
		t.Fatalf("collection index is not *hnsw")
	}
	if !h.vecsDropped() {
		t.Fatal("PQDropVecs did not drop the resident floats (auto-train did not trip)")
	}

	pos := []uint64{ids[0]}
	k := 5
	qr, err := c.Query(recommendBestSpec(pos, nil, k))
	if err != nil {
		t.Fatalf("BEST_SCORE Query on PQ-drop: %v", err)
	}
	if len(qr.Fused) == 0 {
		t.Fatal("BEST_SCORE on PQ-drop returned no results")
	}
	for _, r := range qr.Fused {
		if r.ID == ids[0] {
			t.Errorf("example id %d leaked into PQ-drop BEST_SCORE result %v", ids[0], resultIDs(qr.Fused))
		}
	}

	// Equivalence on the PQ-drop path: the engine top-k must equal the brute oracle
	// over the RECONSTRUCTED corpus (vecsForIDs), restricted to the SAME candidate set
	// the engine's seed pool surfaced (so an ANN pool miss is not counted as a
	// disagreement — the proof is the SCORING, not the pool recall).
	reconstructed := c.idx.vecsForIDs(ids)
	posVecs := [][]float32{reconstructed[ids[0]]}
	dist := pickDist(L2)
	type sc struct {
		id    uint64
		score float32
	}
	cand := make([]sc, 0, len(qr.Fused))
	for _, r := range qr.Fused {
		cand = append(cand, sc{id: r.ID, score: bestScore(reconstructed[r.ID], posVecs, nil, L2, dist)})
	}
	// The engine results must already be score-descending (the BEST_SCORE contract).
	for i := 1; i < len(cand); i++ {
		if cand[i-1].score < cand[i].score {
			t.Errorf("PQ-drop BEST_SCORE not score-descending: %g before %g", cand[i-1].score, cand[i].score)
		}
	}
	// And each engine Result.Score must equal the brute bestScore over the same
	// reconstructed vector (the engine scored the same vectors the oracle did).
	for _, r := range qr.Fused {
		want := bestScore(reconstructed[r.ID], posVecs, nil, L2, dist)
		if r.Score != want {
			t.Errorf("PQ-drop BEST_SCORE doc %d engine score %g != brute %g", r.ID, r.Score, want)
		}
	}
}
