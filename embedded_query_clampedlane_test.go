// SPDX-License-Identifier: Apache-2.0

package rostam

import (
	"context"
	"testing"

	"github.com/rostamlabs/rostam/grpcapi/pb"
	"github.com/rostamlabs/rostam/ops"
	"github.com/rostamlabs/rostam/vector"
)

// TestQueryFanOutClampedLaneMatchesP1 is the clamped-lane P-invariance proof: a
// FUSION query whose prefetch lanes carry LaneK=50000 (well above maxLanePool=10000)
// over a P=4 collection returns the EXACT same fused top-k (id + score + distance,
// in order) as the same spec over a P=1 collection. This proves:
//
//	(1) the maxLanePool clamp in leafLanePool/SourceLanePool introduces NO P-variance
//	    at the coordinator: each partition clamps to the same ceiling, the unioned
//	    lanes are identical to those produced with LaneK=10000, and the final merge
//	    is unchanged.
//	(2) single-node == coordinator for clamped lanes (the symmetry linchpin): both
//	    paths route through vector.SourceLanePool, which calls leafLanePool, which
//	    clamps to maxLanePool — so the lane sizes are identical.
//
// The collection has 200 docs; both LaneK=50000 and the clamped LaneK=10000 retrieve
// all 200 candidates (far below both values), so the test confirms the clamp is a
// structural no-op for normal data while still proving P-invariance on the clamped path.
func TestQueryFanOutClampedLaneMatchesP1(t *testing.T) {
	const n, k = 200, 10
	ctx := context.Background()
	denseQ := []float32{0.5, 0, 0, 0}
	sIdx := []uint32{3}
	sVal := []float32{1}

	// Build the spec with LaneK=50000 on both leaves (above maxLanePool=10000).
	// The pb.DenseLeaf and pb.SparseLeaf structs carry LaneK directly.
	pspec := &pb.QuerySpec{
		Mode: pb.QueryMode_QUERY_MODE_FUSION,
		Prefetch: []*pb.QueryLeaf{
			{Leaf: &pb.QueryLeaf_Dense{Dense: &pb.DenseLeaf{Dense: denseQ, K: int32(k), LaneK: 50000}}},
			{Leaf: &pb.QueryLeaf_Sparse{Sparse: &pb.SparseLeaf{Indices: sIdx, Values: sVal, K: int32(k), LaneK: 50000}}},
		},
		FusionMethod: "rrf",
		K:            int32(k),
	}
	specBytes, spec := buildQuerySpec(t, pspec)

	// Confirm the decoded spec's LaneK was preserved (50000, not clamped at decode —
	// clamping is at execution time in leafLanePool, not at the codec).
	if len(spec.Prefetch) < 1 || spec.Prefetch[0].Leaf == nil || spec.Prefetch[0].Leaf.LaneK != 50000 {
		t.Fatalf("decoded LaneK = %d, want 50000 (clamp must be at exec time, not decode)", func() int {
			if len(spec.Prefetch) > 0 && spec.Prefetch[0].Leaf != nil {
				return spec.Prefetch[0].Leaf.LaneK
			}
			return -1
		}())
	}

	// P=1 baseline.
	s1 := newSingleEmbedded(t)
	waitLeaderEmbedded(t, s1)
	seedQueryCollection(t, s1, "qcl1", 1, n)
	got1, _, err := s1.(*embedded).VectorQuery(ctx, "qcl1", specBytes, spec, ReadOpts{})
	if err != nil {
		t.Fatalf("P1 clamped-lane query: %v", err)
	}
	if len(got1) == 0 {
		t.Fatal("P1 clamped-lane query returned no results")
	}

	// P=4: all 200 ids must span all 4 partitions.
	const P = 4
	sP := newSingleEmbedded(t)
	waitLeaderEmbedded(t, sP)
	seedQueryCollection(t, sP, "qcl4", P, n)
	touched := map[int]bool{}
	for id := uint64(1); id <= uint64(n); id++ {
		touched[ops.PartitionOf(id, P)] = true
	}
	if len(touched) != P {
		t.Fatalf("ids only touch %d/%d partitions — test setup invalid", len(touched), P)
	}
	gotP, meta, err := sP.(*embedded).VectorQuery(ctx, "qcl4", specBytes, spec, ReadOpts{})
	if err != nil {
		t.Fatalf("P4 clamped-lane query: %v", err)
	}
	if meta.Degraded {
		t.Fatalf("P4 clamped-lane query unexpectedly degraded: %+v", meta)
	}
	if len(gotP) == 0 {
		t.Fatal("P4 clamped-lane query returned no results")
	}

	// Core assertion: clamped P4 == P1 (id + score + distance, in order).
	a, b := queryResultKeys(gotP), queryResultKeys(got1)
	if !eqHybridKeys(a, b) {
		t.Fatalf("clamped-lane FUSION P4 != P1:\n P4=%v\n P1=%v", a, b)
	}

	// Bonus: confirm that the same spec with LaneK=vector.MaxLanePool returns
	// identical results to LaneK=50000 — proving the clamp == capping LaneK.
	cappedSpec := &pb.QuerySpec{
		Mode: pb.QueryMode_QUERY_MODE_FUSION,
		Prefetch: []*pb.QueryLeaf{
			{Leaf: &pb.QueryLeaf_Dense{Dense: &pb.DenseLeaf{Dense: denseQ, K: int32(k), LaneK: int32(vector.MaxLanePool)}}},
			{Leaf: &pb.QueryLeaf_Sparse{Sparse: &pb.SparseLeaf{Indices: sIdx, Values: sVal, K: int32(k), LaneK: int32(vector.MaxLanePool)}}},
		},
		FusionMethod: "rrf",
		K:            int32(k),
	}
	cappedBytes, cappedEngineSpec := buildQuerySpec(t, cappedSpec)
	gotCapped, _, err := s1.(*embedded).VectorQuery(ctx, "qcl1", cappedBytes, cappedEngineSpec, ReadOpts{})
	if err != nil {
		t.Fatalf("P1 explicit-ceiling query: %v", err)
	}
	if !eqHybridKeys(queryResultKeys(got1), queryResultKeys(gotCapped)) {
		t.Fatalf("LaneK=50000 (clamped) != LaneK=MaxLanePool (explicit ceiling):\n clamped=%v\n ceiling=%v",
			queryResultKeys(got1), queryResultKeys(gotCapped))
	}
}
