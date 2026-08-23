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

// seedNamedRecDiscCollection creates a P-partition (or P==1) NAMED collection with a
// COSINE dense space ("title", dim 4 — so the AVERAGE-recommend per-space derive
// L2-normalizes mean(pos)-mean(neg) with the SPACE metric, the named-correctness
// path), an L2 dense space ("image", dim 3), and a sparse space ("terms"). Every
// point populates all three spaces + a shared payload {"id": id}. Ids carry
// deterministic spread directions so recommend/discover steer cleanly. A HIGH
// EfSearch + small n make the discover/best-score seed pool recall every doc on BOTH
// P=1 and P=4 (the precondition for the per-partition custom scorer to be
// partition-invariant — same caveat as dense discover).
func seedNamedRecDiscCollection(t *testing.T, coll string, P, n int) Store {
	t.Helper()
	s := newSingleEmbedded(t)
	waitLeaderEmbedded(t, s)
	ctx := context.Background()
	cfg := map[string]NamedVectorParams{
		"title": {Dim: 4, Metric: vector.Cosine, M: 8, EfConstruction: 200, EfSearch: 400, Seed: 1},
		"image": {Dim: 3, Metric: vector.L2, M: 8, EfConstruction: 200, EfSearch: 400, Seed: 1},
		"terms": {Sparse: true},
	}
	if err := s.VectorNamedCreateCollection(ctx, coll, cfg, P); err != nil {
		t.Fatalf("create named rec/disc collection %q (P=%d): %v", coll, P, err)
	}
	emb := s.(*embedded)
	for id := uint64(1); id <= uint64(n); id++ {
		f := float32(id)
		title := []float32{f, f * 0.5, float32(id%5) + 1, float32(id%3) + 1}
		image := []float32{float32(id%9)*0.11 + 0.03, f*0.007 + 0.2, float32(id%4)*0.17 + 0.01}
		sv := &vector.SparseVector{Indices: []uint32{uint32(id%7) + 1}, Values: []float32{f*0.01 + 1}}
		meta := VectorMetadata{"id": vector.NewInt(int64(id))}
		if err := emb.VectorNamedInsertSparse(ctx, coll, id,
			map[string][]float32{"title": title, "image": image},
			map[string]*vector.SparseVector{"terms": sv}, meta, 0); err != nil {
			t.Fatalf("named rec/disc insert %d: %v", id, err)
		}
	}
	return s
}

// namedRecommendLeaf / namedRecommendBestLeaf / namedDiscoverLeaf build Space-bearing
// recommend/discover pb leaves (the named family has its own proto space).
func namedRecommendLeaf(space string, positive, negative []uint64, k int) *pb.QueryLeaf {
	return &pb.QueryLeaf{Leaf: &pb.QueryLeaf_Recommend{Recommend: &pb.RecommendLeaf{
		Positive: positive, Negative: negative, K: int32(k), Space: space,
	}}}
}

func namedRecommendBestLeaf(space string, positive, negative []uint64, k int) *pb.QueryLeaf {
	return &pb.QueryLeaf{Leaf: &pb.QueryLeaf_Recommend{Recommend: &pb.RecommendLeaf{
		Positive: positive, Negative: negative, K: int32(k), Space: space,
		Strategy: pb.RecommendStrategy_RECOMMEND_BEST_SCORE,
	}}}
}

func namedDiscoverLeaf(space string, targetID []uint64, context [][2]uint64, k int) *pb.QueryLeaf {
	ids := make([]*pb.ContextPairIDs, len(context))
	for i, cp := range context {
		ids[i] = &pb.ContextPairIDs{Positive: cp[0], Negative: cp[1]}
	}
	return &pb.QueryLeaf{Leaf: &pb.QueryLeaf_Discover{Discover: &pb.DiscoverLeaf{
		TargetId: targetID, ContextIds: ids, K: int32(k), Space: space,
	}}}
}

func namedRecDiscSpec(t *testing.T, leaf *pb.QueryLeaf, k int) ([]byte, vector.QuerySpec) {
	t.Helper()
	return buildQuerySpec(t, &pb.QuerySpec{
		Mode:         pb.QueryMode_QUERY_MODE_FUSION,
		Prefetch:     []*pb.QueryLeaf{leaf},
		FusionMethod: "rrf",
		K:            int32(k),
	})
}

// TestNamedRecommendAverageFanOutMatchesP1 is the named AVERAGE-recommend exactness
// invariant: a named recommend (AVERAGE_VECTOR) query against the COSINE "title"
// space over P=4 returns the EXACT same top-k as P=1. The coordinator resolves the
// example ids → the SPACE'S vectors cluster-wide, derives normalize(mean(pos)-mean(neg))
// ONCE with the SPACE metric, and rewrites the leaf → a dense leaf (Space PRESERVED)
// before fan-out — partition-invariant. Goes RED if the coordinator picks the wrong
// space's vectors, derives with the wrong metric, or the over-fetch/exclude diverges.
func TestNamedRecommendAverageFanOutMatchesP1(t *testing.T) {
	const n, k = 120, 10
	ctx := context.Background()
	positive := []uint64{10, 11, 12}
	negative := []uint64{110, 111}
	specBytes, spec := namedRecDiscSpec(t, namedRecommendLeaf("title", positive, negative, k), k)

	s1 := seedNamedRecDiscCollection(t, "nra1", 1, n)
	got1, _, err := s1.(*embedded).VectorNamedQuery(ctx, "nra1", specBytes, spec, ReadOpts{})
	if err != nil {
		t.Fatalf("P1 named AVERAGE recommend: %v", err)
	}
	if len(got1) == 0 {
		t.Fatal("P1 named AVERAGE recommend returned no results")
	}

	const P = 4
	sP := seedNamedRecDiscCollection(t, "nra4", P, n)
	gotP, meta, err := sP.(*embedded).VectorNamedQuery(ctx, "nra4", specBytes, spec, ReadOpts{})
	if err != nil {
		t.Fatalf("P4 named AVERAGE recommend: %v", err)
	}
	if meta.Degraded {
		t.Fatalf("P4 named AVERAGE recommend unexpectedly degraded: %+v", meta)
	}
	if !eqHybridKeys(queryResultKeys(gotP), queryResultKeys(got1)) {
		t.Fatalf("named AVERAGE recommend P4 != P1:\n P4=%v\n P1=%v", queryResultKeys(gotP), queryResultKeys(got1))
	}
	// Examples excluded from the merged cluster result.
	excluded := map[uint64]bool{10: true, 11: true, 12: true, 110: true, 111: true}
	for _, r := range gotP {
		if excluded[r.ID] {
			t.Fatalf("named recommend example id %d not excluded: %v", r.ID, queryResultKeys(gotP))
		}
	}
}

// TestNamedRecommendAverageFanOutCrossPartition proves the cluster per-space
// id-resolution for named AVERAGE recommend: the positive ids span >=3 partitions, so
// the coordinator's VectorNamedGetBatch must reach multiple partitions and pick each
// point's "title"-space vector. P=4 must equal P=1 on the same examples.
func TestNamedRecommendAverageFanOutCrossPartition(t *testing.T) {
	const n, k = 120, 10
	const P = 4
	ctx := context.Background()

	var positive []uint64
	seen := map[int]bool{}
	for id := uint64(1); id <= uint64(n) && len(seen) < 3; id++ {
		p := ops.PartitionOf(id, P)
		if !seen[p] {
			seen[p] = true
			positive = append(positive, id)
		}
	}
	if len(seen) < 3 {
		t.Fatalf("could not find positives spanning >=3 partitions (got %d)", len(seen))
	}
	specBytes, spec := namedRecDiscSpec(t, namedRecommendLeaf("title", positive, nil, k), k)

	s1 := seedNamedRecDiscCollection(t, "nrx1", 1, n)
	got1, _, err := s1.(*embedded).VectorNamedQuery(ctx, "nrx1", specBytes, spec, ReadOpts{})
	if err != nil {
		t.Fatalf("P1 cross-partition named recommend: %v", err)
	}
	sP := seedNamedRecDiscCollection(t, "nrx4", P, n)
	gotP, meta, err := sP.(*embedded).VectorNamedQuery(ctx, "nrx4", specBytes, spec, ReadOpts{})
	if err != nil {
		t.Fatalf("P4 cross-partition named recommend: %v", err)
	}
	if meta.Degraded {
		t.Fatalf("P4 cross-partition named recommend unexpectedly degraded: %+v", meta)
	}
	if len(gotP) == 0 {
		t.Fatal("P4 cross-partition named recommend returned no results (per-space id-resolution likely missed cross-partition positives)")
	}
	if !eqHybridKeys(queryResultKeys(gotP), queryResultKeys(got1)) {
		t.Fatalf("cross-partition named recommend P4 != P1:\n P4=%v\n P1=%v", queryResultKeys(gotP), queryResultKeys(got1))
	}
}

// TestNamedRecommendBestFanOutMatchesP1 is the named BEST_SCORE exactness invariant: a
// BEST_SCORE recommend query against the COSINE "title" space over P=4 returns the
// EXACT same top-k as P=1. The coordinator embeds the resolved per-space example
// vectors + clears the ids; the leaf stays a LeafRecommend the named best-score
// execLeaf scores per partition over partition-disjoint ids, score-desc merged. A
// SMALL n (<= the seed pool max(4k,50)) makes every partition surface all its docs so
// the scorer's partition-invariance is isolated from ANN pool recall.
func TestNamedRecommendBestFanOutMatchesP1(t *testing.T) {
	const n, k = 40, 5
	ctx := context.Background()
	positive := []uint64{3, 4}
	negative := []uint64{38}
	specBytes, spec := namedRecDiscSpec(t, namedRecommendBestLeaf("title", positive, negative, k), k)

	s1 := seedNamedRecDiscCollection(t, "nrb1", 1, n)
	got1, _, err := s1.(*embedded).VectorNamedQuery(ctx, "nrb1", specBytes, spec, ReadOpts{})
	if err != nil {
		t.Fatalf("P1 named BEST_SCORE recommend: %v", err)
	}
	if len(got1) == 0 {
		t.Fatal("P1 named BEST_SCORE recommend returned no results")
	}

	const P = 4
	sP := seedNamedRecDiscCollection(t, "nrb4", P, n)
	gotP, meta, err := sP.(*embedded).VectorNamedQuery(ctx, "nrb4", specBytes, spec, ReadOpts{})
	if err != nil {
		t.Fatalf("P4 named BEST_SCORE recommend: %v", err)
	}
	if meta.Degraded {
		t.Fatalf("P4 named BEST_SCORE recommend unexpectedly degraded: %+v", meta)
	}
	if !eqHybridKeys(queryResultKeys(gotP), queryResultKeys(got1)) {
		t.Fatalf("named BEST_SCORE recommend P4 != P1:\n P4=%v\n P1=%v", queryResultKeys(gotP), queryResultKeys(got1))
	}
	excluded := map[uint64]bool{3: true, 4: true, 38: true}
	for _, r := range gotP {
		if excluded[r.ID] {
			t.Fatalf("named BEST_SCORE example id %d not excluded: %v", r.ID, queryResultKeys(gotP))
		}
	}
}

// TestNamedDiscoverFanOutMatchesP1 is the named discover exactness invariant: a named
// discover query (target + context-pair ids → coordinator per-space resolve → embed →
// per-partition scorer → score-desc merge) against the COSINE "title" space over P=4
// returns the EXACT same top-k as P=1. Run under COMPLETE seed recall (small n + high
// EfSearch) so the per-partition discover scorer is partition-invariant — the same
// caveat as the dense discover fan-out.
func TestNamedDiscoverFanOutMatchesP1(t *testing.T) {
	const n, k = 40, 10
	ctx := context.Background()
	target := []uint64{20}
	pairs := [][2]uint64{{5, 38}, {6, 37}}
	specBytes, spec := namedRecDiscSpec(t, namedDiscoverLeaf("title", target, pairs, k), k)

	s1 := seedNamedRecDiscCollection(t, "nd1", 1, n)
	got1, _, err := s1.(*embedded).VectorNamedQuery(ctx, "nd1", specBytes, spec, ReadOpts{})
	if err != nil {
		t.Fatalf("P1 named discover: %v", err)
	}
	if len(got1) == 0 {
		t.Fatal("P1 named discover returned no results")
	}

	const P = 4
	sP := seedNamedRecDiscCollection(t, "nd4", P, n)
	gotP, meta, err := sP.(*embedded).VectorNamedQuery(ctx, "nd4", specBytes, spec, ReadOpts{})
	if err != nil {
		t.Fatalf("P4 named discover: %v", err)
	}
	if meta.Degraded {
		t.Fatalf("P4 named discover unexpectedly degraded: %+v", meta)
	}
	if !eqHybridKeys(queryResultKeys(gotP), queryResultKeys(got1)) {
		t.Fatalf("named discover P4 != P1:\n P4=%v\n P1=%v", queryResultKeys(gotP), queryResultKeys(got1))
	}
}

// TestNamedDiscoverFanOutCrossPartition proves the cluster per-space id-resolution for
// named discover: the target + context-pair ids span >=4 partitions, so the
// coordinator's VectorNamedGetBatch must reach multiple partitions and pick each
// point's "title"-space vector before embedding. P=4 == P=1 on the same inputs.
func TestNamedDiscoverFanOutCrossPartition(t *testing.T) {
	const n, k = 40, 10
	const P = 4
	ctx := context.Background()

	var ids []uint64
	seen := map[int]bool{}
	for id := uint64(1); id <= uint64(n) && len(ids) < 5; id++ {
		p := ops.PartitionOf(id, P)
		if !seen[p] || len(ids) < 5 {
			seen[p] = true
			ids = append(ids, id)
		}
	}
	if len(ids) < 5 {
		t.Fatalf("need 5 ids, got %d", len(ids))
	}
	target := []uint64{ids[0]}
	pairs := [][2]uint64{{ids[1], ids[2]}, {ids[3], ids[4]}}
	specBytes, spec := namedRecDiscSpec(t, namedDiscoverLeaf("title", target, pairs, k), k)

	s1 := seedNamedRecDiscCollection(t, "ndx1", 1, n)
	got1, _, err := s1.(*embedded).VectorNamedQuery(ctx, "ndx1", specBytes, spec, ReadOpts{})
	if err != nil {
		t.Fatalf("P1 cross-partition named discover: %v", err)
	}
	sP := seedNamedRecDiscCollection(t, "ndx4", P, n)
	gotP, meta, err := sP.(*embedded).VectorNamedQuery(ctx, "ndx4", specBytes, spec, ReadOpts{})
	if err != nil {
		t.Fatalf("P4 cross-partition named discover: %v", err)
	}
	if meta.Degraded {
		t.Fatalf("P4 cross-partition named discover unexpectedly degraded: %+v", meta)
	}
	if len(gotP) == 0 {
		t.Fatal("P4 cross-partition named discover returned no results (per-space id-resolution likely missed cross-partition ids)")
	}
	if !eqHybridKeys(queryResultKeys(gotP), queryResultKeys(got1)) {
		t.Fatalf("cross-partition named discover P4 != P1:\n P4=%v\n P1=%v", queryResultKeys(gotP), queryResultKeys(got1))
	}
}

// TestNamedRecommendFanOutSparseSpaceRejected proves the coordinator named pre-pass
// fail-loud per-space: a recommend leaf pointing at the SPARSE "terms" space is
// rejected with ErrSpaceModalityMismatch on the cluster path (recommend/discover are
// dense-vector concepts). The error must surface, not a panic/empty result, and P>1
// rejects the same as P1 (the single-node denseSpace reject).
func TestNamedRecommendFanOutSparseSpaceRejected(t *testing.T) {
	const n, k = 40, 5
	const P = 4
	ctx := context.Background()
	specBytes, spec := namedRecDiscSpec(t, namedRecommendLeaf("terms", []uint64{3, 4}, nil, k), k)

	sP := seedNamedRecDiscCollection(t, "nrs4", P, n)
	if _, _, err := sP.(*embedded).VectorNamedQuery(ctx, "nrs4", specBytes, spec, ReadOpts{}); err == nil {
		t.Fatal("expected fail-loud error for a recommend leaf on a SPARSE space")
	} else if !errors.Is(err, vector.ErrSpaceModalityMismatch) {
		t.Fatalf("sparse-space named recommend err = %v, want ErrSpaceModalityMismatch", err)
	}
}

// TestNamedRecommendFanOutFailLoudAllMissing: a named recommend whose positive ids do
// NOT exist anywhere in the cluster cannot derive a query vector → fail-loud
// (ErrIDNotFound). The error must surface on the cluster path, not an empty result.
func TestNamedRecommendFanOutFailLoudAllMissing(t *testing.T) {
	const n, k = 40, 5
	const P = 4
	ctx := context.Background()
	specBytes, spec := namedRecDiscSpec(t, namedRecommendLeaf("title", []uint64{9001, 9002}, nil, k), k)

	sP := seedNamedRecDiscCollection(t, "nrf4", P, n)
	if _, _, err := sP.(*embedded).VectorNamedQuery(ctx, "nrf4", specBytes, spec, ReadOpts{}); err == nil {
		t.Fatal("expected fail-loud error when NO positive ids resolve cluster-wide")
	}
}

// TestMVQueryRecommendDiscoverRejectedAtWire proves the MV fail-loud surfaces as an
// ERROR (not a panic/empty) when a recommend or discover leaf reaches an MV query via
// the wire path (VectorMVQuery → validateMVLeafPayload). MV stores token SETS with no
// single-vector pooling semantics, so recommend/discover are inherently unsupported.
func TestMVQueryRecommendDiscoverRejectedAtWire(t *testing.T) {
	const n, k = 40, 5
	ctx := context.Background()
	s := seedMVQueryCollection(t, "mvrd", 1, n)
	emb := s.(*embedded)

	recBytes, recSpec := buildQuerySpec(t, &pb.QuerySpec{
		Mode:         pb.QueryMode_QUERY_MODE_FUSION,
		Prefetch:     []*pb.QueryLeaf{recommendLeaf([]uint64{3, 4}, nil, k)},
		FusionMethod: "rrf",
		K:            int32(k),
	})
	if _, _, err := emb.VectorMVQuery(ctx, "mvrd", recBytes, recSpec, ReadOpts{}); err == nil {
		t.Fatal("expected MV recommend to be rejected at the wire")
	} else if !errors.Is(err, vector.ErrQueryMVRecommendUnsupported) {
		t.Fatalf("MV recommend err = %v, want ErrQueryMVRecommendUnsupported", err)
	}

	discBytes, discSpec := buildQuerySpec(t, &pb.QuerySpec{
		Mode:         pb.QueryMode_QUERY_MODE_FUSION,
		Prefetch:     []*pb.QueryLeaf{discoverLeafPB([]uint64{20}, [][2]uint64{{5, 38}}, k)},
		FusionMethod: "rrf",
		K:            int32(k),
	})
	if _, _, err := emb.VectorMVQuery(ctx, "mvrd", discBytes, discSpec, ReadOpts{}); err == nil {
		t.Fatal("expected MV discover to be rejected at the wire")
	} else if !errors.Is(err, vector.ErrQueryMVDiscoverUnsupported) {
		t.Fatalf("MV discover err = %v, want ErrQueryMVDiscoverUnsupported", err)
	}
}
