// SPDX-License-Identifier: Apache-2.0

package rostam

import (
	"context"
	"testing"

	"github.com/rostamlabs/rostam/grpcapi/pb"
	"github.com/rostamlabs/rostam/vector"
)

// TestDirectNestedFusionMatchesOracle proves finding 004: a Direct store's
// VectorQuery on a nested MULTI-lane FUSION spec must fold the expanded tree-lanes
// via treeFusionMergeFanOut, exactly like the single-node engine. The oracle is an
// embedded P=1 VectorQuery (which already guards on SpecHasNestedFusion). Before the
// fix, directStore.VectorQuery unconditionally called the flat fusionMergeFanOut,
// which consumes only len(spec.Prefetch) top-level lanes — mis-associating the nested
// sub-lanes and dropping trailing top-level leaves — so it returned a DIFFERENT,
// silently-wrong top-k with no error. This test goes RED without the guard.
func TestDirectNestedFusionMatchesOracle(t *testing.T) {
	ctx := context.Background()
	const n, k = 200, 10
	parentDense := []float32{0.5, 0, 0, 0}
	subDense := []float32{12.3, 0, 0, 0} // a different anchor → its top ids differ from the parent's
	sIdx := []uint32{3}
	sVal := []float32{1}

	// The nested sub-spec: a 2-lane (dense+sparse) FUSION with a small per-lane pool.
	subSpec := &pb.QuerySpec{
		Mode: pb.QueryMode_QUERY_MODE_FUSION,
		PrefetchSources: []*pb.QuerySource{
			{Source: &pb.QuerySource_Leaf{Leaf: denseLeafLaneK(subDense, k, 6)}},
			{Source: &pb.QuerySource_Leaf{Leaf: sparseLeafLaneK(sIdx, sVal, k, 6)}},
		},
		FusionMethod: "rrf",
		K:            k,
	}
	// The parent FUSION: [a dense leaf source, the nested MULTI-lane FUSION sub-spec].
	pspec := &pb.QuerySpec{
		Mode: pb.QueryMode_QUERY_MODE_FUSION,
		PrefetchSources: []*pb.QuerySource{
			{Source: &pb.QuerySource_Leaf{Leaf: denseLeaf(parentDense, k)}},
			{Source: &pb.QuerySource_Spec{Spec: subSpec}},
		},
		FusionMethod: "rrf",
		K:            k,
	}
	specBytes, spec := buildQuerySpec(t, pspec)
	if !vector.SpecHasNestedFusion(spec) {
		t.Fatal("spec should have a nested multi-lane FUSION node")
	}

	// Oracle: the single-node engine via embedded P=1 (already nested-fusion-aware).
	oracle := newSingleEmbedded(t)
	waitLeaderEmbedded(t, oracle)
	seedQueryCollection(t, oracle, "nf", 1, n)
	want, _, err := oracle.VectorQuery(ctx, "nf", specBytes, spec, ReadOpts{})
	if err != nil {
		t.Fatalf("oracle VectorQuery: %v", err)
	}
	if len(want) == 0 {
		t.Fatal("oracle returned no results")
	}

	// Subject: the Direct store over the identical data + spec.
	d := newDirectForTest(t)
	seedQueryCollection(t, d, "nf", 1, n)
	got, _, err := d.VectorQuery(ctx, "nf", specBytes, spec, ReadOpts{})
	if err != nil {
		t.Fatalf("direct VectorQuery: %v", err)
	}

	if !eqHybridKeys(queryResultKeys(got), queryResultKeys(want)) {
		t.Fatalf("Direct nested-FUSION != single-node oracle:\n direct=%v\n oracle=%v",
			queryResultKeys(got), queryResultKeys(want))
	}
}
