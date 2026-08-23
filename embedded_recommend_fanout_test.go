// SPDX-License-Identifier: Apache-2.0

package rostam

import (
	"context"
	"errors"
	"testing"

	"github.com/rostamlabs/rostam/ops"
	"github.com/rostamlabs/rostam/sdk/pb"
	"github.com/rostamlabs/rostam/vector"
)

// seedRecommendCollection creates a P-partition (or P==1) DENSE collection (dim 4,
// Cosine — so the metric-normalize derive path is exercised) and inserts ids 1..n
// each with a deterministic dense vector spread over the unit-ish space so a derived
// centroid has well-separated near-neighbors. Cosine is used so the coordinator-derive
// L2-normalizes the mean(pos)-mean(neg) vector (the same as the single-node path).
func seedRecommendCollection(t *testing.T, s Store, coll string, P, n int) {
	t.Helper()
	ctx := context.Background()
	if err := s.CreateCollection(ctx, coll, VectorConfig{
		Dim: 4, M: 8, EfConstruction: 50, EfSearch: 64, Seed: 1, Metric: vector.Cosine, Partitions: P,
	}); err != nil {
		t.Fatalf("CreateCollection %q (P=%d): %v", coll, P, err)
	}
	for id := uint64(1); id <= uint64(n); id++ {
		// A direction that rotates with id so each point is a distinct direction; the
		// derived centroid of a few positives picks out their neighbors deterministically.
		f := float32(id)
		v := []float32{f, f * 0.5, float32(id%5) + 1, float32(id%3) + 1}
		if err := s.VectorInsertExt(ctx, coll, id, v, VectorInsertOpts{}); err != nil {
			t.Fatalf("VectorInsertExt %s/%d: %v", coll, id, err)
		}
	}
}

func recommendLeaf(positive, negative []uint64, k int) *pb.QueryLeaf {
	return &pb.QueryLeaf{Leaf: &pb.QueryLeaf_Recommend{Recommend: &pb.RecommendLeaf{
		Positive: positive, Negative: negative, K: int32(k),
	}}}
}

// recommendBestLeaf builds a BEST_SCORE recommend leaf (a custom per-candidate
// scorer, score-descending — like discover, NOT a dense rewrite).
func recommendBestLeaf(positive, negative []uint64, k int) *pb.QueryLeaf {
	return &pb.QueryLeaf{Leaf: &pb.QueryLeaf_Recommend{Recommend: &pb.RecommendLeaf{
		Positive: positive, Negative: negative, K: int32(k),
		Strategy: pb.RecommendStrategy_RECOMMEND_BEST_SCORE,
	}}}
}

// recommendBestSpec builds a FUSION spec with a single BEST_SCORE recommend prefetch
// leaf.
func recommendBestSpec(t *testing.T, positive, negative []uint64, k int) ([]byte, vector.QuerySpec) {
	t.Helper()
	return buildQuerySpec(t, &pb.QuerySpec{
		Mode:         pb.QueryMode_QUERY_MODE_FUSION,
		Prefetch:     []*pb.QueryLeaf{recommendBestLeaf(positive, negative, k)},
		FusionMethod: "rrf",
		K:            int32(k),
	})
}

// TestRecommendBestFanOutMatchesP1 is the BEST_SCORE exactness invariant: a BEST_SCORE
// recommend query over a P=4 collection returns the EXACT same top-k as P=1. BEST_SCORE
// is a per-candidate scorer over partition-disjoint ids (like discover), so each
// partition produces the globally-correct bestScore for its OWN ids and the score-desc
// merge is exact. A SMALL corpus (n=40) keeps every partition's doc count below the
// seed pool (max(4k,50)=50) so the candidate pool surfaces ALL docs on every
// partitioning — isolating the SCORER's partition-invariance from ANN pool recall.
func TestRecommendBestFanOutMatchesP1(t *testing.T) {
	const n, k = 40, 5
	ctx := context.Background()
	positive := []uint64{3, 4}
	negative := []uint64{38}
	specBytes, spec := recommendBestSpec(t, positive, negative, k)

	s1 := newSingleEmbedded(t)
	waitLeaderEmbedded(t, s1)
	seedRecommendCollection(t, s1, "rb1", 1, n)
	got1, _, err := s1.(*embedded).VectorQuery(ctx, "rb1", specBytes, spec, ReadOpts{})
	if err != nil {
		t.Fatalf("P1 BEST_SCORE recommend: %v", err)
	}
	if len(got1) == 0 {
		t.Fatal("P1 BEST_SCORE recommend returned no results")
	}

	const P = 4
	sP := newSingleEmbedded(t)
	waitLeaderEmbedded(t, sP)
	seedRecommendCollection(t, sP, "rb4", P, n)
	gotP, meta, err := sP.(*embedded).VectorQuery(ctx, "rb4", specBytes, spec, ReadOpts{})
	if err != nil {
		t.Fatalf("P4 BEST_SCORE recommend: %v", err)
	}
	if meta.Degraded {
		t.Fatalf("P4 BEST_SCORE recommend unexpectedly degraded: %+v", meta)
	}
	if len(gotP) == 0 {
		t.Fatal("P4 BEST_SCORE recommend returned no results")
	}
	if !eqHybridKeys(queryResultKeys(gotP), queryResultKeys(got1)) {
		t.Fatalf("BEST_SCORE recommend P4 != P1:\n P4=%v\n P1=%v", queryResultKeys(gotP), queryResultKeys(got1))
	}
	// Examples are excluded from the BEST_SCORE result too.
	excluded := map[uint64]bool{3: true, 4: true, 38: true}
	for _, r := range gotP {
		if excluded[r.ID] {
			t.Fatalf("BEST_SCORE example id %d not excluded: %v", r.ID, queryResultKeys(gotP))
		}
	}
}

// TestRecommendBestFanOutCrossPartitionPositives proves the BEST_SCORE cluster
// id-resolution: positive ids chosen to span >=3 partitions must all be resolved
// cluster-wide and embedded before fan-out, so the P=4 result equals the P=1 result.
func TestRecommendBestFanOutCrossPartitionPositives(t *testing.T) {
	const n, k = 40, 5
	const P = 4
	ctx := context.Background()

	var positive []uint64
	seenPart := map[int]bool{}
	for id := uint64(1); id <= uint64(n) && len(seenPart) < 3; id++ {
		p := ops.PartitionOf(id, P)
		if !seenPart[p] {
			seenPart[p] = true
			positive = append(positive, id)
		}
	}
	if len(seenPart) < 3 {
		t.Fatalf("could not find positives spanning >=3 partitions (got %d)", len(seenPart))
	}

	specBytes, spec := recommendBestSpec(t, positive, nil, k)

	s1 := newSingleEmbedded(t)
	waitLeaderEmbedded(t, s1)
	seedRecommendCollection(t, s1, "rbx1", 1, n)
	got1, _, err := s1.(*embedded).VectorQuery(ctx, "rbx1", specBytes, spec, ReadOpts{})
	if err != nil {
		t.Fatalf("P1 cross-partition BEST_SCORE: %v", err)
	}

	sP := newSingleEmbedded(t)
	waitLeaderEmbedded(t, sP)
	seedRecommendCollection(t, sP, "rbx4", P, n)
	gotP, meta, err := sP.(*embedded).VectorQuery(ctx, "rbx4", specBytes, spec, ReadOpts{})
	if err != nil {
		t.Fatalf("P4 cross-partition BEST_SCORE: %v", err)
	}
	if meta.Degraded {
		t.Fatalf("P4 cross-partition BEST_SCORE unexpectedly degraded: %+v", meta)
	}
	if len(gotP) == 0 {
		t.Fatal("P4 cross-partition BEST_SCORE returned no results (cluster id-resolution likely missed cross-partition positives)")
	}
	if !eqHybridKeys(queryResultKeys(gotP), queryResultKeys(got1)) {
		t.Fatalf("cross-partition BEST_SCORE P4 != P1:\n P4=%v\n P1=%v", queryResultKeys(gotP), queryResultKeys(got1))
	}
}

// TestRecommendBestFanOutFailLoudNoPositives: a BEST_SCORE leaf with no positive ids
// cannot score candidates → fail-loud on the cluster path.
func TestRecommendBestFanOutFailLoudNoPositives(t *testing.T) {
	const n, k = 40, 5
	const P = 4
	ctx := context.Background()
	specBytes, spec := recommendBestSpec(t, nil, []uint64{5}, k)

	sP := newSingleEmbedded(t)
	waitLeaderEmbedded(t, sP)
	seedRecommendCollection(t, sP, "rbf4", P, n)
	if _, _, err := sP.(*embedded).VectorQuery(ctx, "rbf4", specBytes, spec, ReadOpts{}); err == nil {
		t.Fatal("expected fail-loud error for a BEST_SCORE leaf with no positives")
	}
}

// TestRecommendBestFanOutL2NegativesRejected: BEST_SCORE + negatives over an L2
// collection must FAIL LOUD on the cluster (P>1) path with
// ErrRecommendBestScoreL2Negatives — the same reject the single-node path enforces
// (the -max_neg sign-flip inverts the L2 ranking). Proves the coordinator
// (resolveRecommendForFanOut → RewriteRecommendLeavesWith) rejects before fan-out, so
// P>1 == P1 for the reject. L2 + BEST_SCORE with NO negatives stays valid on the cluster.
func TestRecommendBestFanOutL2NegativesRejected(t *testing.T) {
	const n, k = 40, 5
	const P = 4
	ctx := context.Background()

	seedL2 := func(s Store, coll string) {
		if err := s.CreateCollection(ctx, coll, VectorConfig{
			Dim: 4, M: 8, EfConstruction: 50, EfSearch: 64, Seed: 1, Metric: vector.L2, Partitions: P,
		}); err != nil {
			t.Fatalf("CreateCollection %q: %v", coll, err)
		}
		for id := uint64(1); id <= uint64(n); id++ {
			f := float32(id)
			v := []float32{f, f * 0.5, float32(id%5) + 1, float32(id%3) + 1}
			if err := s.VectorInsertExt(ctx, coll, id, v, VectorInsertOpts{}); err != nil {
				t.Fatalf("VectorInsertExt %s/%d: %v", coll, id, err)
			}
		}
	}

	sP := newSingleEmbedded(t)
	waitLeaderEmbedded(t, sP)
	seedL2(sP, "rbl2neg")

	// L2 + negatives → fail loud on the cluster path.
	negSpecBytes, negSpec := recommendBestSpec(t, []uint64{3, 4}, []uint64{38}, k)
	if _, _, err := sP.(*embedded).VectorQuery(ctx, "rbl2neg", negSpecBytes, negSpec, ReadOpts{}); err == nil {
		t.Fatal("expected fail-loud error for L2 BEST_SCORE + negatives on the cluster path")
	} else if !errors.Is(err, vector.ErrRecommendBestScoreL2Negatives) {
		t.Fatalf("L2 BEST_SCORE + negatives cluster err = %v, want ErrRecommendBestScoreL2Negatives", err)
	}

	// L2 + NO negatives stays valid on the cluster path.
	okSpecBytes, okSpec := recommendBestSpec(t, []uint64{3, 4}, nil, k)
	if _, _, err := sP.(*embedded).VectorQuery(ctx, "rbl2neg", okSpecBytes, okSpec, ReadOpts{}); err != nil {
		t.Fatalf("L2 BEST_SCORE no-neg on cluster should succeed, got %v", err)
	}
}

// recommendSpec builds a FUSION spec with a single recommend prefetch leaf (the
// canonical recommend query: one recommend node → dense search after derive).
func recommendSpec(t *testing.T, positive, negative []uint64, k int) ([]byte, vector.QuerySpec) {
	t.Helper()
	return buildQuerySpec(t, &pb.QuerySpec{
		Mode:         pb.QueryMode_QUERY_MODE_FUSION,
		Prefetch:     []*pb.QueryLeaf{recommendLeaf(positive, negative, k)},
		FusionMethod: "rrf",
		K:            int32(k),
	})
}

// TestRecommendFanOutMatchesP1 is the recommend exactness invariant: a recommend
// query (positive+negative example ids → coordinator-derive → dense search) over a
// P=4 collection returns the EXACT same top-k (id+score+distance, in order) as the
// same query over P=1. The derived query vector is identical regardless of
// partitioning (the resolved example vectors are the same), so the rewritten dense
// search is partition-invariant. Goes RED if the coordinator-derive diverges from
// the single-node derive or the over-fetch/exclude diverges across P.
func TestRecommendFanOutMatchesP1(t *testing.T) {
	const n, k = 200, 10
	ctx := context.Background()
	positive := []uint64{10, 11, 12}
	negative := []uint64{190, 191}
	specBytes, spec := recommendSpec(t, positive, negative, k)

	s1 := newSingleEmbedded(t)
	waitLeaderEmbedded(t, s1)
	seedRecommendCollection(t, s1, "rc1", 1, n)
	got1, _, err := s1.(*embedded).VectorQuery(ctx, "rc1", specBytes, spec, ReadOpts{})
	if err != nil {
		t.Fatalf("P1 recommend: %v", err)
	}
	if len(got1) == 0 {
		t.Fatal("P1 recommend returned no results")
	}

	const P = 4
	sP := newSingleEmbedded(t)
	waitLeaderEmbedded(t, sP)
	seedRecommendCollection(t, sP, "rc4", P, n)
	gotP, meta, err := sP.(*embedded).VectorQuery(ctx, "rc4", specBytes, spec, ReadOpts{})
	if err != nil {
		t.Fatalf("P4 recommend: %v", err)
	}
	if meta.Degraded {
		t.Fatalf("P4 recommend unexpectedly degraded: %+v", meta)
	}
	if len(gotP) == 0 {
		t.Fatal("P4 recommend returned no results")
	}
	if !eqHybridKeys(queryResultKeys(gotP), queryResultKeys(got1)) {
		t.Fatalf("recommend P4 != P1:\n P4=%v\n P1=%v", queryResultKeys(gotP), queryResultKeys(got1))
	}
}

// TestRecommendFanOutNestedTreeWalk proves the coordinator recommend pre-pass
// TREE-WALK over P>1: a parent spec whose prefetch is [a dense leaf, a NESTED
// single-lane FUSION sub-spec carrying a RECOMMEND leaf] must have the nested recommend
// leaf DERIVED-to-dense on the coordinator (cluster-wide example resolution) BEFORE
// fan-out — exactly as the top-level recommend does. If the coordinator pre-pass
// (SpecHasRecommendLeaves / RecommendExampleIDs / RewriteRecommendLeavesWith) did NOT
// recurse into the nested sub-spec, the un-rewritten LeafRecommend would reach a
// partition handler and be rejected fail-loud (ErrQueryBadLeafKind) — so a successful,
// non-degraded, example-excluded result proves the tree-walk resolved the nested leaf.
// (Exact cross-partition equality is asserted by the provably partition-invariant
// nested-RERANK test; a derived-recommend HNSW lane carries pool/ANN tie ambiguity
// across partitionings, so this coordinator test asserts the resolution contract.)
func TestRecommendFanOutNestedTreeWalk(t *testing.T) {
	const n, k = 200, 10
	const P = 4
	ctx := context.Background()
	positive := []uint64{10, 11, 12}
	denseQ := []float32{1, 0.5, 1, 1}

	sub := &pb.QuerySpec{
		Mode: pb.QueryMode_QUERY_MODE_FUSION,
		PrefetchSources: []*pb.QuerySource{
			{Source: &pb.QuerySource_Leaf{Leaf: recommendLeaf(positive, nil, k)}},
		},
		FusionMethod: "rrf",
		K:            int32(k),
	}
	specBytes, spec := buildQuerySpec(t, &pb.QuerySpec{
		Mode: pb.QueryMode_QUERY_MODE_FUSION,
		PrefetchSources: []*pb.QuerySource{
			{Source: &pb.QuerySource_Leaf{Leaf: denseLeaf(denseQ, k)}},
			{Source: &pb.QuerySource_Spec{Spec: sub}},
		},
		FusionMethod: "rrf",
		K:            int32(k),
	})

	sP := newSingleEmbedded(t)
	waitLeaderEmbedded(t, sP)
	seedRecommendCollection(t, sP, "rn4", P, n)
	gotP, meta, err := sP.(*embedded).VectorQuery(ctx, "rn4", specBytes, spec, ReadOpts{})
	if err != nil {
		// The fail-loud signal that the tree-walk did NOT recurse: an un-rewritten nested
		// recommend leaf reaching a partition handler.
		t.Fatalf("P4 nested recommend (tree-walk pre-pass did not resolve the nested leaf): %v", err)
	}
	if meta.Degraded {
		t.Fatalf("P4 nested recommend unexpectedly degraded: %+v", meta)
	}
	if len(gotP) == 0 {
		t.Fatal("P4 nested recommend returned no results — the nested derive likely failed")
	}
	// The recommend example ids must be excluded from the result (the coordinator's
	// post-merge prune over the tree-walk-collected nested example ids).
	excluded := map[uint64]bool{10: true, 11: true, 12: true}
	for _, r := range gotP {
		if excluded[r.ID] {
			t.Fatalf("nested recommend example id %d not excluded: %v", r.ID, queryResultKeys(gotP))
		}
	}
}

// TestRecommendFanOutCrossPartitionPositives proves the cluster id-resolution: the
// positive example ids are chosen to hash to DIFFERENT partitions (P=4), so the
// coordinator's cluster-wide batch-get must reach multiple partitions to resolve
// them all before deriving. The P=4 result must equal the P=1 result on the SAME
// example set (a single-node derive over all examples). Goes RED if the
// coordinator resolves only the coordinator-local partition's ids.
func TestRecommendFanOutCrossPartitionPositives(t *testing.T) {
	const n, k = 200, 10
	const P = 4
	ctx := context.Background()

	// Pick positive ids that span >=3 distinct partitions for P=4 (the cross-
	// partition resolution is the thing under test).
	var positive []uint64
	seenPart := map[int]bool{}
	for id := uint64(1); id <= uint64(n) && len(seenPart) < 3; id++ {
		p := ops.PartitionOf(id, P)
		if !seenPart[p] {
			seenPart[p] = true
			positive = append(positive, id)
		}
	}
	if len(seenPart) < 3 {
		t.Fatalf("could not find positives spanning >=3 partitions (got %d)", len(seenPart))
	}
	// Sanity: the chosen positives really do live on distinct partitions.
	parts := map[int]bool{}
	for _, id := range positive {
		parts[ops.PartitionOf(id, P)] = true
	}
	if len(parts) < 3 {
		t.Fatalf("positives only span %d partitions, want >=3 (cross-partition resolution not exercised)", len(parts))
	}

	specBytes, spec := recommendSpec(t, positive, nil, k)

	s1 := newSingleEmbedded(t)
	waitLeaderEmbedded(t, s1)
	seedRecommendCollection(t, s1, "rx1", 1, n)
	got1, _, err := s1.(*embedded).VectorQuery(ctx, "rx1", specBytes, spec, ReadOpts{})
	if err != nil {
		t.Fatalf("P1 cross-partition recommend: %v", err)
	}

	sP := newSingleEmbedded(t)
	waitLeaderEmbedded(t, sP)
	seedRecommendCollection(t, sP, "rx4", P, n)
	gotP, meta, err := sP.(*embedded).VectorQuery(ctx, "rx4", specBytes, spec, ReadOpts{})
	if err != nil {
		t.Fatalf("P4 cross-partition recommend: %v", err)
	}
	if meta.Degraded {
		t.Fatalf("P4 cross-partition recommend unexpectedly degraded: %+v", meta)
	}
	if len(gotP) == 0 {
		t.Fatal("P4 cross-partition recommend returned no results (cluster id-resolution likely missed cross-partition positives)")
	}
	if !eqHybridKeys(queryResultKeys(gotP), queryResultKeys(got1)) {
		t.Fatalf("cross-partition recommend P4 != P1:\n P4=%v\n P1=%v", queryResultKeys(gotP), queryResultKeys(got1))
	}
}

// TestQueryFanOutNestedFusionRecommendBothStrategiesMatchesP1 is the dense nested
// recommend correctness test covering BOTH AVERAGE_VECTOR and BEST_SCORE strategies.
// A parent FUSION whose prefetch is [a dense leaf, a NESTED 2-lane FUSION sub-spec
// carrying a RECOMMEND leaf] must be P4==P1 exact with example ids excluded.
//
// This exercises widenNestedSpecsK in the dense path (vector/query.go QueryTreeLanes +
// recommend_fanout.go coordinator): the sub-spec K is widened by nExclude so a recommend
// leaf buried at depth>=1 over-fetches enough candidates to survive the post-fold prune.
// BEST_SCORE strategy (RecPosVecs/RecNegVecs coordinator path) is not covered by the
// existing TestQueryFanOutNestedFusionWithRecommendMatchesP1 (AVERAGE_VECTOR only).
func TestQueryFanOutNestedFusionRecommendBothStrategiesMatchesP1(t *testing.T) {
	const n, k = 200, 10
	ctx := context.Background()
	// Positive ids spanning all 4 partitions so the coordinator resolves cross-partition.
	positive := []uint64{3, 5, 7}
	sIdx := []uint32{3}
	sVal := []float32{1}

	cases := []struct {
		name   string
		leafFn func([]uint64, []uint64, int) *pb.QueryLeaf
	}{
		{"average_vector", recommendLeaf},
		{"best_score", recommendBestLeaf},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			// Nested sub-spec: [recommend + sparse]. dbsf for continuous scores (no ties).
			subSpec := &pb.QuerySpec{
				Mode: pb.QueryMode_QUERY_MODE_FUSION,
				PrefetchSources: []*pb.QuerySource{
					{Source: &pb.QuerySource_Leaf{Leaf: tc.leafFn(positive, nil, k)}},
					{Source: &pb.QuerySource_Leaf{Leaf: sparseLeaf(sIdx, sVal, k)}},
				},
				FusionMethod: "dbsf",
				Alpha:        0.5,
				K:            int32(k),
			}
			pspec := &pb.QuerySpec{
				Mode: pb.QueryMode_QUERY_MODE_FUSION,
				PrefetchSources: []*pb.QuerySource{
					{Source: &pb.QuerySource_Leaf{Leaf: denseLeaf([]float32{0.5, 0.5, 1, 1}, k)}},
					{Source: &pb.QuerySource_Spec{Spec: subSpec}},
				},
				FusionMethod: "dbsf",
				Alpha:        0.5,
				K:            int32(k),
			}
			specBytes, spec := buildQuerySpec(t, pspec)
			if !vector.SpecHasNestedFusion(spec) {
				t.Fatalf("%s: pspec should have a nested multi-lane FUSION node", tc.name)
			}

			s1 := newSingleEmbedded(t)
			waitLeaderEmbedded(t, s1)
			seedQueryCollection(t, s1, "qnrec1_"+tc.name, 1, n)
			got1, _, err := s1.(*embedded).VectorQuery(ctx, "qnrec1_"+tc.name, specBytes, spec, ReadOpts{})
			if err != nil {
				t.Fatalf("%s P1: %v", tc.name, err)
			}
			if len(got1) == 0 {
				t.Fatalf("%s P1 returned no results", tc.name)
			}
			// Example ids must be excluded from P1 result.
			excluded := map[uint64]bool{3: true, 5: true, 7: true}
			for _, r := range got1 {
				if excluded[r.ID] {
					t.Fatalf("%s P1 result contains positive example id %d (should be excluded)", tc.name, r.ID)
				}
			}

			const P = 4
			// Re-decode pspec so the in-place rewrite from P1 doesn't affect P4.
			_, specP := buildQuerySpec(t, pspec)
			sP := newSingleEmbedded(t)
			waitLeaderEmbedded(t, sP)
			seedQueryCollection(t, sP, "qnrec4_"+tc.name, P, n)
			touched := map[int]bool{}
			for id := uint64(1); id <= uint64(n); id++ {
				touched[ops.PartitionOf(id, P)] = true
			}
			if len(touched) != P {
				t.Fatalf("ids only touch %d/%d partitions", len(touched), P)
			}
			gotP, meta, err := sP.(*embedded).VectorQuery(ctx, "qnrec4_"+tc.name, specBytes, specP, ReadOpts{})
			if err != nil {
				t.Fatalf("%s P4: %v", tc.name, err)
			}
			if meta.Degraded {
				t.Fatalf("%s P4 unexpectedly degraded: %+v", tc.name, meta)
			}
			for _, r := range gotP {
				if excluded[r.ID] {
					t.Fatalf("%s P4 result contains positive example id %d (should be excluded)", tc.name, r.ID)
				}
			}
			if !eqHybridKeys(queryResultKeys(gotP), queryResultKeys(got1)) {
				t.Fatalf("%s nested-FUSION-with-recommend P4 != P1:\n P4=%v\n P1=%v", tc.name, queryResultKeys(gotP), queryResultKeys(got1))
			}
		})
	}
}

// TestRecommendFanOutExcludesExamples verifies the example ids are pruned from the
// cluster result (AFTER the global fuse), mirroring the single-node exclusion. The
// positive ids are their own nearest neighbors (the centroid is near them), so
// without exclusion they would dominate the top-k; with exclusion none appear.
func TestRecommendFanOutExcludesExamples(t *testing.T) {
	const n, k = 200, 10
	const P = 4
	ctx := context.Background()
	positive := []uint64{20, 21, 22}
	negative := []uint64{180}
	specBytes, spec := recommendSpec(t, positive, negative, k)

	sP := newSingleEmbedded(t)
	waitLeaderEmbedded(t, sP)
	seedRecommendCollection(t, sP, "re4", P, n)
	gotP, _, err := sP.(*embedded).VectorQuery(ctx, "re4", specBytes, spec, ReadOpts{})
	if err != nil {
		t.Fatalf("P4 recommend: %v", err)
	}
	excluded := map[uint64]bool{}
	for _, id := range append(append([]uint64{}, positive...), negative...) {
		excluded[id] = true
	}
	for _, r := range gotP {
		if excluded[r.ID] {
			t.Fatalf("example id %d appeared in the recommend result (must be excluded): %v", r.ID, queryResultKeys(gotP))
		}
	}
	if len(gotP) == 0 {
		t.Fatal("recommend result empty after exclusion")
	}
}

// TestRecommendFanOutFailLoudNoPositives: a recommend leaf with no positive ids
// cannot derive a query vector → fail-loud (mirrors the single-node
// ErrNoRecommendExamples). The error must surface on the cluster path, not produce
// an empty/garbage result.
func TestRecommendFanOutFailLoudNoPositives(t *testing.T) {
	const n, k = 50, 10
	const P = 4
	ctx := context.Background()
	specBytes, spec := recommendSpec(t, nil, []uint64{5}, k)

	sP := newSingleEmbedded(t)
	waitLeaderEmbedded(t, sP)
	seedRecommendCollection(t, sP, "rf4", P, n)
	if _, _, err := sP.(*embedded).VectorQuery(ctx, "rf4", specBytes, spec, ReadOpts{}); err == nil {
		t.Fatal("expected fail-loud error for a recommend leaf with no positives")
	}
}

// TestRecommendFanOutFailLoudAllPositivesMissing: a recommend whose positive ids
// do NOT exist anywhere in the cluster cannot derive a query vector → fail-loud
// (mirrors the single-node ErrIDNotFound: the cluster batch-get resolves nothing,
// so the mean is undefined). The error must surface, not an empty result.
func TestRecommendFanOutFailLoudAllPositivesMissing(t *testing.T) {
	const n, k = 80, 10
	const P = 4
	ctx := context.Background()
	// ids well above n never exist in the collection.
	specBytes, spec := recommendSpec(t, []uint64{9001, 9002, 9003}, nil, k)

	sP := newSingleEmbedded(t)
	waitLeaderEmbedded(t, sP)
	seedRecommendCollection(t, sP, "rm4", P, n)
	if _, _, err := sP.(*embedded).VectorQuery(ctx, "rm4", specBytes, spec, ReadOpts{}); err == nil {
		t.Fatal("expected fail-loud error when NO positive ids resolve cluster-wide")
	}
}

// TestRecommendFanOutPartialPositivesResolve proves graceful partial resolution:
// when SOME positive ids do not exist (mixed with valid ones), the cluster derive
// uses only the resolved ids — matching a single-node derive over the SAME resolved
// (valid-only) example set. Mirrors vecsForIDs skipping unknown ids; fail-loud only
// fires when NONE resolve.
// TestHasCatalogPartitionsKeySelection verifies the physConfig key-selection logic in
// resolveRecommendForFanOut. CreateCollection writes to the catalog ONLY for P>1
// (line: "if cfg.Partitions <= 1 { return e.Call(...) }") — a P==1 collection takes
// the direct single-create path with NO catalog write. As a result hasCatalogPartitions
// returns false for P==1 regardless of catalog implementation:
//
//   - P==1: no catalog entry → PartitionsGen returns ok=false → hasCatalogPartitions=false
//     → physConfig = collection (the logical collection, which holds the full config).
//   - P>1: catalog entry with Partitions>1 → hasCatalogPartitions=true
//     → physConfig = PartitionKeyGen(collection, gen, 0) = "collection#0".
//
// This is consistent across singleNodeCatalog (ops.Catalog) and metaCatalog (cluster
// multi-node): both return ok=false when no catalog entry exists. The logical
// collection is valid for vector_get_config because CreateCollection always creates it
// with the full dim/metric config (Partitions reset to 0 in the physical cfg, but
// dim/metric preserved), and the P==1 direct path creates a collection under the bare
// logical name directly.
func TestHasCatalogPartitionsKeySelection(t *testing.T) {
	ctx := context.Background()
	s := newSingleEmbedded(t)
	waitLeaderEmbedded(t, s)
	emb := s.(*embedded)

	// P==1 collection: CreateCollection takes the DIRECT path (no catalog write).
	// hasCatalogPartitions must return false so physConfig = bare logical name.
	if err := s.CreateCollection(ctx, "hcp1", VectorConfig{
		Dim: 4, M: 8, EfConstruction: 50, EfSearch: 32, Seed: 1, Partitions: 1,
	}); err != nil {
		t.Fatalf("CreateCollection P=1: %v", err)
	}
	coll1 := ops.CanonicalName("hcp1")
	if hasCatalogPartitions(emb, coll1) {
		t.Errorf("P==1: hasCatalogPartitions = true, want false (no catalog entry for P==1)")
	}

	// P>1 collection: catalog entry written → hasCatalogPartitions=true → physConfig=coll#0.
	if err := s.CreateCollection(ctx, "hcp4", VectorConfig{
		Dim: 4, M: 8, EfConstruction: 50, EfSearch: 32, Seed: 1, Partitions: 4,
	}); err != nil {
		t.Fatalf("CreateCollection P=4: %v", err)
	}
	coll4 := ops.CanonicalName("hcp4")
	if !hasCatalogPartitions(emb, coll4) {
		t.Errorf("P==4: hasCatalogPartitions = false, want true")
	}

	// Unregistered collection: no entry → false.
	if hasCatalogPartitions(emb, "default/nonexistent") {
		t.Errorf("unregistered: hasCatalogPartitions = true, want false")
	}
}

func TestRecommendFanOutPartialPositivesResolve(t *testing.T) {
	const n, k = 200, 10
	const P = 4
	ctx := context.Background()
	valid := []uint64{30, 31, 32}
	// Mix valid + non-existent positives; the derive must use only the valid ones.
	mixed := []uint64{30, 9001, 31, 9002, 32}

	// Oracle: P=1 over the VALID-only positives.
	validBytes, validSpec := recommendSpec(t, valid, nil, k)
	s1 := newSingleEmbedded(t)
	waitLeaderEmbedded(t, s1)
	seedRecommendCollection(t, s1, "rp1", 1, n)
	want, _, err := s1.(*embedded).VectorQuery(ctx, "rp1", validBytes, validSpec, ReadOpts{})
	if err != nil {
		t.Fatalf("P1 valid-only recommend: %v", err)
	}

	// P=4 over the MIXED positives must equal the valid-only oracle.
	mixedBytes, mixedSpec := recommendSpec(t, mixed, nil, k)
	sP := newSingleEmbedded(t)
	waitLeaderEmbedded(t, sP)
	seedRecommendCollection(t, sP, "rp4", P, n)
	got, meta, err := sP.(*embedded).VectorQuery(ctx, "rp4", mixedBytes, mixedSpec, ReadOpts{})
	if err != nil {
		t.Fatalf("P4 mixed recommend: %v", err)
	}
	if meta.Degraded {
		t.Fatalf("P4 mixed recommend unexpectedly degraded: %+v", meta)
	}
	// Note: the mixed query excludes the non-existent ids too (they are in the
	// example set), but since they never appear as results the exclusion is a no-op;
	// the resolved-derive must match the valid-only oracle's top-k.
	if !eqHybridKeys(queryResultKeys(got), queryResultKeys(want)) {
		t.Fatalf("partial-resolve P4(mixed) != P1(valid):\n got=%v\n want=%v", queryResultKeys(got), queryResultKeys(want))
	}
}
