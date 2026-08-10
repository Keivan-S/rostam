// SPDX-License-Identifier: Apache-2.0

package vector

import (
	"errors"
	"testing"
)

// newDiscoverQueryCorpus builds a Collection with two cosine clusters: ids 1-3
// in cluster A (near [1,0]) and ids 4-6 in cluster B (near [0,1]). The target
// [0.7,0.7] sits between them, so a context pair (positive in A, negative in B)
// steers discover toward cluster A — the same corpus discover_test.go uses.
func newDiscoverQueryCorpus(t *testing.T, metric Metric) *Collection {
	t.Helper()
	c, err := NewCollection("disc", Config{Dim: 2, M: 16, EfConstruction: 200, EfSearch: 64, Seed: 1, Metric: metric})
	if err != nil {
		t.Fatalf("NewCollection: %v", err)
	}
	corpus := map[uint64][]float32{
		1: {1, 0.02},
		2: {1, -0.02},
		3: {0.98, 0.05},
		4: {0.02, 1},
		5: {-0.02, 1},
		6: {0.05, 0.98},
	}
	for id, v := range corpus {
		if err := c.Insert(id, v, 0, nil, nil); err != nil {
			t.Fatalf("Insert %d: %v", id, err)
		}
	}
	return c
}

// discoverSpec builds a single-leaf FUSION spec with one discover prefetch leaf
// carrying the UNRESOLVED target + context-pair ids (the coordinator resolve
// pre-pass embeds the vectors before the execLeaf runs).
func discoverSpec(targetID []uint64, context []ContextPair, k int) QuerySpec {
	return QuerySpec{
		Mode: ModeFusion,
		Prefetch: srcs([]QueryLeaf{
			{
				Kind:               LeafDiscover,
				DiscoverTargetID:   targetID,
				DiscoverContextIDs: context,
				K:                  k,
				ScoreDesc:          true,
			},
		}...),
		K: k,
	}
}

// TestQueryDiscoverEqualsEngine is the equivalence oracle: discover via the Query
// API (id-form, resolved by the single-node pre-pass) must produce the SAME top-k
// as the engine (*hnsw).Discover on the same target + context. Proving the leaf
// executor reuses discover.go's exact scorer.
func TestQueryDiscoverEqualsEngine(t *testing.T) {
	c := newDiscoverQueryCorpus(t, Cosine)
	target := []float32{0.7, 0.7}
	context := []ContextPair{{Positive: 1, Negative: 4}}
	k := 3

	// Engine oracle: the raw (*hnsw).Discover.
	h, ok := c.idx.(*hnsw)
	if !ok {
		t.Fatalf("collection index is not *hnsw")
	}
	want, err := h.Discover(k, DiscoverOpts{Target: target, Context: context})
	if err != nil {
		t.Fatalf("engine Discover: %v", err)
	}

	// Query API: the discover leaf carries the target VECTOR + context ids (the
	// pre-pass resolves the context ids; the target rides as a raw vector here so the
	// seed is byte-identical to the engine path).
	spec := QuerySpec{
		Mode: ModeFusion,
		Prefetch: srcs([]QueryLeaf{
			{
				Kind:               LeafDiscover,
				DiscoverTarget:     target,
				DiscoverContextIDs: context,
				K:                  k,
				ScoreDesc:          true,
			},
		}...),
		K: k,
	}
	qr, err := c.Query(spec)
	if err != nil {
		t.Fatalf("discover Query: %v", err)
	}
	if !queryResultsEqual(qr.Fused, want) {
		t.Errorf("discover Query != engine Discover\n got=%+v\nwant=%+v", qr.Fused, want)
	}
}

// TestQueryDiscoverEqualsEngineTargetID is the equivalence oracle for the FULL
// id-form: BOTH the target and the context resolve from ids via the pre-pass, and
// must equal the engine Discover whose Target is the same id's stored vector.
func TestQueryDiscoverEqualsEngineTargetID(t *testing.T) {
	c := newDiscoverQueryCorpus(t, Cosine)
	context := []ContextPair{{Positive: 1, Negative: 4}}
	k := 3

	h := c.idx.(*hnsw)
	// The target id 2's stored vector is the engine oracle's Target.
	tv := c.idx.vecsForIDs([]uint64{2})[2]
	want, err := h.Discover(k, DiscoverOpts{Target: tv, Context: context})
	if err != nil {
		t.Fatalf("engine Discover: %v", err)
	}

	qr, err := c.Query(discoverSpec([]uint64{2}, context, k))
	if err != nil {
		t.Fatalf("discover Query (target id): %v", err)
	}
	if !queryResultsEqual(qr.Fused, want) {
		t.Errorf("discover Query (target id) != engine Discover\n got=%+v\nwant=%+v", qr.Fused, want)
	}
}

// TestQueryDiscoverContextSteers: a context pair (positive in cluster A, negative
// in cluster B) steers the top-k toward cluster A even though the target sits
// between the clusters. The result must be all cluster-A ids (1,2,3).
func TestQueryDiscoverContextSteers(t *testing.T) {
	c := newDiscoverQueryCorpus(t, Cosine)
	qr, err := c.Query(QuerySpec{
		Mode: ModeFusion,
		Prefetch: srcs([]QueryLeaf{
			{
				Kind:               LeafDiscover,
				DiscoverTarget:     []float32{0.7, 0.7},
				DiscoverContextIDs: []ContextPair{{Positive: 1, Negative: 4}},
				K:                  3,
				ScoreDesc:          true,
			},
		}...),
		K: 3,
	})
	if err != nil {
		t.Fatalf("discover Query: %v", err)
	}
	got := sortedIDs(resultIDs(qr.Fused))
	if len(got) != 3 || got[0] != 1 || got[1] != 2 || got[2] != 3 {
		t.Errorf("discover steered = %v, want cluster-A ids [1 2 3]", got)
	}

	// Flipping the pair (positive in B, negative in A) steers to cluster B.
	qrB, err := c.Query(QuerySpec{
		Mode: ModeFusion,
		Prefetch: srcs([]QueryLeaf{
			{
				Kind:               LeafDiscover,
				DiscoverTarget:     []float32{0.7, 0.7},
				DiscoverContextIDs: []ContextPair{{Positive: 4, Negative: 1}},
				K:                  3,
				ScoreDesc:          true,
			},
		}...),
		K: 3,
	})
	if err != nil {
		t.Fatalf("discover Query (flipped): %v", err)
	}
	gotB := sortedIDs(resultIDs(qrB.Fused))
	if len(gotB) != 3 || gotB[0] != 4 || gotB[1] != 5 || gotB[2] != 6 {
		t.Errorf("discover steered (flipped) = %v, want cluster-B ids [4 5 6]", gotB)
	}
}

// TestQueryDiscoverScoreDesc: the discover lane is score-descending and the leaf
// is tagged ScoreDesc=true (so the fan-out fold/merge orients it correctly).
func TestQueryDiscoverScoreDesc(t *testing.T) {
	c := newDiscoverQueryCorpus(t, Cosine)
	qr, err := c.Query(discoverSpec([]uint64{1}, []ContextPair{{Positive: 1, Negative: 4}}, 5))
	if err != nil {
		t.Fatalf("discover Query: %v", err)
	}
	if qr.Mode != ModeFusion || len(qr.Lanes) != 1 {
		t.Fatalf("mode=%d lanes=%d, want fusion + 1 lane", qr.Mode, len(qr.Lanes))
	}
	lane := qr.Lanes[0]
	for i := 1; i < len(lane); i++ {
		if lane[i].Score > lane[i-1].Score {
			t.Errorf("discover lane not score-descending at %d: %g > %g", i, lane[i].Score, lane[i-1].Score)
		}
	}
}

// TestQueryDiscoverComposesInFusion: a discover leaf composes as one prefetch lane
// in a multi-lane FUSION spec (alongside a dense lane), folded via the score-desc
// orientation (lane0 = discover, ScoreDesc=true).
func TestQueryDiscoverComposesInFusion(t *testing.T) {
	c := newDiscoverQueryCorpus(t, Cosine)
	spec := QuerySpec{
		Mode: ModeFusion,
		Prefetch: srcs([]QueryLeaf{
			{
				Kind:               LeafDiscover,
				DiscoverTarget:     []float32{0.7, 0.7},
				DiscoverContextIDs: []ContextPair{{Positive: 1, Negative: 4}},
				K:                  5,
				ScoreDesc:          true,
			},
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
	if len(qr.Fused) == 0 {
		t.Fatal("fusion returned no results")
	}
	// Both the discover lane and the dense lane favor cluster A; the fused top must
	// be a cluster-A id.
	if top := qr.Fused[0].ID; top != 1 && top != 2 && top != 3 {
		t.Errorf("fused top = %d, want a cluster-A id (1/2/3)", top)
	}
}

// TestQueryDiscoverComposesAsRerankRoot: a discover leaf composes as the RERANK
// root — the union of the prefetch candidates is re-scored by the discover
// context-pair scorer (score-desc), surfacing cluster A first.
func TestQueryDiscoverComposesAsRerankRoot(t *testing.T) {
	c := newDiscoverQueryCorpus(t, Cosine)
	spec := QuerySpec{
		Mode: ModeRerank,
		Root: QueryLeaf{
			Kind:               LeafDiscover,
			DiscoverTarget:     []float32{0.7, 0.7},
			DiscoverContextIDs: []ContextPair{{Positive: 1, Negative: 4}},
			K:                  3,
			ScoreDesc:          true,
		},
		Prefetch: srcs([]QueryLeaf{
			{Kind: LeafDense, Dense: []float32{0.7, 0.7}, K: 6},
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
	if len(qr.Fused) == 0 {
		t.Fatal("rerank returned no results")
	}
	// The discover rerank scores by the context pair (positive A / negative B): the
	// top result must be a cluster-A doc.
	if top := qr.Fused[0].ID; top != 1 && top != 2 && top != 3 {
		t.Errorf("rerank top = %d, want a cluster-A id (1/2/3)", top)
	}
}

// TestQueryDiscoverFailLoud covers the v1 fail-loud edges: a Space-bearing (named)
// discover leaf, no context pairs, an unresolvable target id, and all-context
// unresolvable.
func TestQueryDiscoverFailLoud(t *testing.T) {
	c := newDiscoverQueryCorpus(t, Cosine)

	// A Space-bearing discover leaf → fail loud (v1 dense-only). A dense collection
	// rejects ANY Space-bearing leaf before the discover check; either rejection is
	// a valid fail-loud (the leaf never runs as a discover).
	spaceSpec := QuerySpec{
		Mode: ModeFusion,
		Prefetch: srcs([]QueryLeaf{
			{Kind: LeafDiscover, Space: "title", DiscoverContextIDs: []ContextPair{{Positive: 1, Negative: 4}}, K: 3, ScoreDesc: true},
		}...),
		K: 3,
	}
	if _, err := c.Query(spaceSpec); err == nil {
		t.Error("Space-bearing discover leaf should fail loud")
	} else if !errors.Is(err, ErrQueryDiscoverHasSpace) && !errors.Is(err, ErrQueryDenseLeafHasSpace) {
		t.Errorf("Space-bearing discover err = %v, want ErrQueryDiscoverHasSpace or ErrQueryDenseLeafHasSpace", err)
	}

	// No context pairs → ErrQueryDiscoverNoContext.
	if _, err := c.Query(discoverSpec([]uint64{1}, nil, 3)); !errors.Is(err, ErrQueryDiscoverNoContext) {
		t.Errorf("no-context err = %v, want ErrQueryDiscoverNoContext", err)
	}

	// Unresolvable target id → ErrIDNotFound (the anchor cannot resolve).
	if _, err := c.Query(discoverSpec([]uint64{9999}, []ContextPair{{Positive: 1, Negative: 4}}, 3)); !errors.Is(err, ErrIDNotFound) {
		t.Errorf("missing-target err = %v, want ErrIDNotFound", err)
	}

	// All context pairs reference missing ids → ErrIDNotFound (no pair resolves).
	if _, err := c.Query(discoverSpec(nil, []ContextPair{{Positive: 9998, Negative: 9999}}, 3)); !errors.Is(err, ErrIDNotFound) {
		t.Errorf("missing-context err = %v, want ErrIDNotFound", err)
	}
}

// TestQueryDiscoverPQDrop: discover on a PQ-HNSW + PQDropVecs collection (the
// resident floats are DROPPED) resolves the context-pair + target vectors and
// scores candidates from the RECONSTRUCTED vectors (vecsForIDs → vecFor), and
// must equal the engine Discover on the same reconstructed inputs.
func TestQueryDiscoverPQDrop(t *testing.T) {
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
	c, err := NewCollection("discpq", cfg)
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

	// Use ids from two different clusters as a context pair, and a third as target.
	context := []ContextPair{{Positive: ids[0], Negative: ids[n-1]}}
	target := []uint64{ids[1]}
	k := 5

	// Engine oracle: target id's reconstructed vector + the same context.
	tv := c.idx.vecsForIDs([]uint64{ids[1]})[ids[1]]
	want, err := h.Discover(k, DiscoverOpts{Target: tv, Context: context})
	if err != nil {
		t.Fatalf("engine Discover on PQ-drop: %v", err)
	}

	qr, err := c.Query(discoverSpec(target, context, k))
	if err != nil {
		t.Fatalf("discover Query on PQ-drop: %v", err)
	}
	if len(qr.Fused) == 0 {
		t.Fatal("discover on PQ-drop returned no results")
	}
	if !queryResultsEqual(qr.Fused, want) {
		t.Errorf("PQ-drop discover Query != engine Discover\n got=%+v\nwant=%+v", qr.Fused, want)
	}
}
