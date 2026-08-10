// SPDX-License-Identifier: Apache-2.0

package vector

import (
	"errors"
	"fmt"
	"testing"
)

// newRecommendQueryCorpus builds a Collection with two cosine clusters: ids 1-3
// in cluster A (near [1,0]) and ids 4-6 in cluster B (near [0,1]). The recommend
// Query API derives mean(positive) - mean(negative) over these example vectors.
func newRecommendQueryCorpus(t *testing.T, metric Metric) *Collection {
	t.Helper()
	c, err := NewCollection("reco", Config{Dim: 2, M: 16, EfConstruction: 200, EfSearch: 64, Seed: 1, Metric: metric})
	if err != nil {
		t.Fatalf("NewCollection: %v", err)
	}
	corpus := map[uint64][]float32{
		1: {1, 0.05},
		2: {1, -0.05},
		3: {0.99, 0.1},
		4: {0.05, 1},
		5: {-0.05, 1},
		6: {0.1, 0.99},
	}
	for id, v := range corpus {
		if err := c.Insert(id, v, 0, nil, nil); err != nil {
			t.Fatalf("Insert %d: %v", id, err)
		}
	}
	return c
}

// recommendSpec builds a single-leaf FUSION spec with one recommend prefetch leaf
// (the simplest path: the lane IS the answer after derive).
func recommendSpec(positive, negative []uint64, k int) QuerySpec {
	return QuerySpec{
		Mode: ModeFusion,
		Prefetch: srcs([]QueryLeaf{
			{Kind: LeafRecommend, Positive: positive, Negative: negative, K: k},
		}...),
		K: k,
	}
}

// TestQueryRecommendPositiveNeighbors: recommend by a positive id returns the
// other cluster-A neighbors (the centroid's near-neighbors), with the example id
// excluded.
func TestQueryRecommendPositiveNeighbors(t *testing.T) {
	c := newRecommendQueryCorpus(t, Cosine)
	qr, err := c.Query(recommendSpec([]uint64{1}, nil, 2))
	if err != nil {
		t.Fatalf("Query: %v", err)
	}
	got := sortedIDs(resultIDs(qr.Fused))
	if len(got) != 2 || got[0] != 2 || got[1] != 3 {
		t.Errorf("recommend(+1) = %v, want cluster-A neighbors [2 3] (example 1 excluded)", got)
	}
}

// TestQueryRecommendExamplesExcluded: every positive AND negative example id is
// excluded from the result set.
func TestQueryRecommendExamplesExcluded(t *testing.T) {
	c := newRecommendQueryCorpus(t, Cosine)
	qr, err := c.Query(recommendSpec([]uint64{1, 2}, []uint64{4}, 6))
	if err != nil {
		t.Fatalf("Query: %v", err)
	}
	for _, r := range qr.Fused {
		if r.ID == 1 || r.ID == 2 || r.ID == 4 {
			t.Errorf("example id %d leaked into recommend results %v", r.ID, resultIDs(qr.Fused))
		}
	}
}

// TestQueryRecommendNegativeSteersAway: adding a cluster-B negative pushes a
// cluster-B doc (id 6 — never an example in either query, so the comparison is
// apples-to-apples) STRICTLY FARTHER from the derived query than the same
// recommend without the negative. Compares the doc's distance directly (rank is
// confounded by the differing exclusion-set sizes).
func TestQueryRecommendNegativeSteersAway(t *testing.T) {
	c := newRecommendQueryCorpus(t, Cosine)

	distOf := func(res []Result, id uint64) (float32, bool) {
		for _, r := range res {
			if r.ID == id {
				return r.Distance, true
			}
		}
		return 0, false
	}

	// Without a negative: derived ≈ cluster A; measure cluster-B doc 6's distance.
	without, err := c.Query(recommendSpec([]uint64{1}, nil, 6))
	if err != nil {
		t.Fatalf("Query without negative: %v", err)
	}
	// With a cluster-B negative (4): the derive steers AWAY from cluster B, so doc 6
	// (cluster B) must be farther.
	with, err := c.Query(recommendSpec([]uint64{1}, []uint64{4}, 6))
	if err != nil {
		t.Fatalf("Query with negative: %v", err)
	}
	dWithout, ok1 := distOf(without.Fused, 6)
	dWith, ok2 := distOf(with.Fused, 6)
	if !ok1 || !ok2 {
		t.Fatalf("cluster-B doc 6 missing from results: without=%v with=%v", resultIDs(without.Fused), resultIDs(with.Fused))
	}
	if dWith <= dWithout {
		t.Errorf("negative did not steer away: cluster-B doc 6 distance without=%g with=%g (want with > without)", dWithout, dWith)
	}
}

// TestQueryRecommendEqualsManualMeanDense is the equivalence oracle: a recommend
// leaf must produce the SAME top-k as a plain dense query whose vector is the
// manually-derived normalize(mean(positive) - mean(negative)) — proving the
// coordinator derive + rewrite-to-dense is exact.
func TestQueryRecommendEqualsManualMeanDense(t *testing.T) {
	for _, metric := range []Metric{Cosine, L2, DotProduct} {
		t.Run(fmt.Sprintf("metric=%d", metric), func(t *testing.T) {
			c := newRecommendQueryCorpus(t, metric)
			positive := []uint64{1, 3}
			negative := []uint64{4}
			k := 3

			recoQR, err := c.Query(recommendSpec(positive, negative, k))
			if err != nil {
				t.Fatalf("recommend Query: %v", err)
			}

			// Manual derive: mean(pos) - mean(neg), metric-normalized exactly like the
			// helper, then a plain dense lane EXCLUDING the example ids by hand.
			derived, err := deriveRecommendVector(c.cfg.Dim, c.cfg.Metric, c.idx.vecsForIDs, positive, negative)
			if err != nil {
				t.Fatalf("manual derive: %v", err)
			}
			exclude := map[uint64]bool{1: true, 3: true, 4: true}
			lane, err := c.SearchFiltered(derived, k+len(exclude), Filter{})
			if err != nil {
				t.Fatalf("manual dense search: %v", err)
			}
			want := make([]Result, 0, k)
			for _, r := range lane {
				if exclude[r.ID] {
					continue
				}
				want = append(want, r)
				if len(want) == k {
					break
				}
			}
			if !queryResultsEqual(recoQR.Fused, want) {
				t.Errorf("recommend != manual mean(pos)-mean(neg) dense\n got=%+v\nwant=%+v", recoQR.Fused, want)
			}
		})
	}
}

// TestQueryRecommendComposesInFusion: a recommend leaf composes as one prefetch
// lane in a multi-lane FUSION spec (it is a dense lane after derive) and the query
// succeeds, returning a fused top-k with the example excluded.
func TestQueryRecommendComposesInFusion(t *testing.T) {
	c := newRecommendQueryCorpus(t, Cosine)
	spec := QuerySpec{
		Mode: ModeFusion,
		Prefetch: srcs([]QueryLeaf{
			{Kind: LeafRecommend, Positive: []uint64{1}, K: 5},
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
	for _, r := range qr.Fused {
		if r.ID == 1 {
			t.Errorf("recommend example id 1 leaked into fused result %v", resultIDs(qr.Fused))
		}
	}
	// The recommend lane (lane 0) must also have id 1 pruned.
	for _, r := range qr.Lanes[0] {
		if r.ID == 1 {
			t.Errorf("recommend example id 1 leaked into lane 0 %v", resultIDs(qr.Lanes[0]))
		}
	}
}

// TestQueryRecommendComposesAsRerankRoot: a recommend leaf composes as the RERANK
// root (it is a dense root after derive) — the union of the prefetch candidates is
// re-scored by the derived recommend vector.
func TestQueryRecommendComposesAsRerankRoot(t *testing.T) {
	c := newRecommendQueryCorpus(t, Cosine)
	spec := QuerySpec{
		Mode: ModeRerank,
		Root: QueryLeaf{Kind: LeafRecommend, Positive: []uint64{1}, K: 3},
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
			t.Errorf("recommend example id 1 leaked into rerank result %v", resultIDs(qr.Fused))
		}
	}
	// The rerank root re-scores by the derived recommend vector ≈ cluster A: the
	// top result must be a cluster-A doc (2 or 3), not a cluster-B doc.
	if len(qr.Fused) == 0 {
		t.Fatal("rerank returned no results")
	}
	if top := qr.Fused[0].ID; top != 2 && top != 3 {
		t.Errorf("rerank top = %d, want a cluster-A neighbor (2 or 3)", top)
	}
}

// TestQueryRecommendFailLoud covers the v1 fail-loud edges: no positives, a
// Space-bearing (named) recommend leaf, and all-positives-missing.
func TestQueryRecommendFailLoud(t *testing.T) {
	c := newRecommendQueryCorpus(t, Cosine)

	// No positives → ErrNoRecommendExamples.
	if _, err := c.Query(recommendSpec(nil, []uint64{4}, 3)); !errors.Is(err, ErrNoRecommendExamples) {
		t.Errorf("no-positives err = %v, want ErrNoRecommendExamples", err)
	}

	// All positives missing → ErrIDNotFound (the mean is undefined).
	if _, err := c.Query(recommendSpec([]uint64{9998, 9999}, nil, 3)); !errors.Is(err, ErrIDNotFound) {
		t.Errorf("missing-positives err = %v, want ErrIDNotFound", err)
	}

	// A Space-bearing recommend leaf → ErrQueryRecommendHasSpace (v1 dense-only).
	spaceSpec := QuerySpec{
		Mode: ModeFusion,
		Prefetch: srcs([]QueryLeaf{
			{Kind: LeafRecommend, Space: "title", Positive: []uint64{1}, K: 3},
		}...),
		K: 3,
	}
	// A dense collection rejects ANY Space-bearing leaf before the recommend check;
	// either rejection is a valid fail-loud (the leaf never runs as a recommend).
	if _, err := c.Query(spaceSpec); err == nil {
		t.Error("Space-bearing recommend leaf should fail loud")
	} else if !errors.Is(err, ErrQueryRecommendHasSpace) && !errors.Is(err, ErrQueryDenseLeafHasSpace) {
		t.Errorf("Space-bearing recommend err = %v, want ErrQueryRecommendHasSpace or ErrQueryDenseLeafHasSpace", err)
	}
}

// TestQueryRecommendPQDrop: recommend on a PQ-HNSW + PQDropVecs collection (the
// incremental auto-train trips, then the resident floats are DROPPED) derives the
// query vector from the RECONSTRUCTED vectors (vecsForIDs → vecFor) and still
// returns the positive's cluster neighbors, with the example excluded.
func TestQueryRecommendPQDrop(t *testing.T) {
	const (
		dim       = 64
		m         = 8
		nClusters = 8
		n         = 600
		seed      = 7
	)
	ids, vecs, _ := buildPQDropCorpus(n, dim, nClusters, seed)
	// PQ-HNSW with float-drop: a bulk BuildConcurrent trains the PQ codebooks and
	// DROPS the resident floats in one pass (the same path pq_drop_hnsw_test uses),
	// so reads reconstruct from the codes via vecFor.
	cfg := Config{Dim: dim, Metric: L2, M: 16, EfConstruction: 200, EfSearch: 64, Seed: seed,
		Quant: QuantPQ, QuantPQM: m, PQDropVecs: true}
	c, err := NewCollection("recopq", cfg)
	if err != nil {
		t.Fatalf("NewCollection: %v", err)
	}
	if err := c.BuildConcurrent(ids, vecs, 4); err != nil {
		t.Fatalf("BuildConcurrent: %v", err)
	}
	// The floats must actually be dropped (the reconstruct path is what we exercise).
	h, ok := c.idx.(*hnsw)
	if !ok {
		t.Fatalf("collection index is not *hnsw")
	}
	if !h.vecsDropped() {
		t.Fatal("PQDropVecs did not drop the resident floats (auto-train did not trip)")
	}

	// Pick two positives in the same cluster (ids whose stored vectors are close)
	// so the derived mean is a coherent query. Use the first id and its exact
	// nearest neighbor (by reconstructed vector) as the positive pair.
	pos := []uint64{ids[0]}
	qr, err := c.Query(recommendSpec(pos, nil, 5))
	if err != nil {
		t.Fatalf("recommend Query on PQ-drop: %v", err)
	}
	if len(qr.Fused) == 0 {
		t.Fatal("recommend on PQ-drop returned no results")
	}
	for _, r := range qr.Fused {
		if r.ID == ids[0] {
			t.Errorf("example id %d leaked into PQ-drop recommend result %v", ids[0], resultIDs(qr.Fused))
		}
	}

	// Equivalence on the PQ-drop path: recommend == a plain dense query with the
	// manually derived (reconstructed) vector, same exclusion. This proves the
	// derive consumes the SAME reconstructed vectors as a manual mean.
	derived, err := deriveRecommendVector(c.cfg.Dim, c.cfg.Metric, c.idx.vecsForIDs, pos, nil)
	if err != nil {
		t.Fatalf("manual derive on PQ-drop: %v", err)
	}
	lane, err := c.SearchFiltered(derived, 6, Filter{})
	if err != nil {
		t.Fatalf("manual dense search on PQ-drop: %v", err)
	}
	want := make([]Result, 0, 5)
	for _, r := range lane {
		if r.ID == ids[0] {
			continue
		}
		want = append(want, r)
		if len(want) == 5 {
			break
		}
	}
	if !queryResultsEqual(qr.Fused, want) {
		t.Errorf("PQ-drop recommend != manual derived dense\n got=%+v\nwant=%+v", qr.Fused, want)
	}
}
