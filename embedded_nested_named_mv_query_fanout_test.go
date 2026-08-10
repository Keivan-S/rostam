// SPDX-License-Identifier: Apache-2.0

package rostam

import (
	"context"
	"testing"

	"github.com/rostamlabs/rostam/grpcapi/pb"
	"github.com/rostamlabs/rostam/ops"
	"github.com/rostamlabs/rostam/vector"
)

// Cross-partition (P>1==P1) exactness for NESTED prefetch in the NAMED + MV families:
// a spec containing a nested MULTI-lane FUSION node ships per-partition UNFUSED
// tree-lanes and the coordinator folds each FUSION node over the cross-partition
// global union ⇒ P4==P1 EXACT. These inherit the dense tree-lanes invariant (the
// orientation-aware fold keys per-lane sort on the source orientation), now verified
// for named multi-space dense/sparse lanes and MV score-desc lanes.

// TestNamedQueryFanOutNestedMultiLaneFusionMatchesP1 is the named-family #1 nested
// invariant: a parent FUSION whose prefetch is [a dense "title" leaf, a NESTED 2-lane
// FUSION sub-spec over (image dense + terms sparse) — a MULTI-SPACE nested fusion]
// returns the EXACT same fused top-k over P=4 as over P=1, for RRF/weighted/dbsf.
func TestNamedQueryFanOutNestedMultiLaneFusionMatchesP1(t *testing.T) {
	const n, k = 200, 10
	ctx := context.Background()
	titleQ := []float32{1.2, 0.6, 0.3, 0.4}
	imageQ := []float32{0.5, 0.5, 0.2}
	sIdx := []uint32{1, 14, 26}
	sVal := []float32{2, 3, 1}

	cases := []struct {
		name   string
		method string
		alpha  float64
	}{
		{"rrf", "rrf", 0},
		{"weighted", "weighted", 0.35},
		{"dbsf", "dbsf", 0.35},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			// Nested MULTI-SPACE 2-lane FUSION: image dense + terms sparse.
			sub := &pb.QuerySpec{
				Mode: pb.QueryMode_QUERY_MODE_FUSION,
				PrefetchSources: []*pb.QuerySource{
					{Source: &pb.QuerySource_Leaf{Leaf: namedDenseLeaf("image", imageQ, k)}},
					{Source: &pb.QuerySource_Leaf{Leaf: namedSparseLeaf("terms", sIdx, sVal, k)}},
				},
				FusionMethod: tc.method,
				Alpha:        tc.alpha,
				K:            int32(k),
			}
			pspec := &pb.QuerySpec{
				Mode: pb.QueryMode_QUERY_MODE_FUSION,
				PrefetchSources: []*pb.QuerySource{
					{Source: &pb.QuerySource_Leaf{Leaf: namedDenseLeaf("title", titleQ, k)}},
					{Source: &pb.QuerySource_Spec{Spec: sub}},
				},
				FusionMethod: tc.method,
				Alpha:        tc.alpha,
				K:            int32(k),
			}
			specBytes, spec := buildQuerySpec(t, pspec)

			s1 := seedNamedQueryCollection(t, "nnf1_"+tc.name, 1, n)
			got1, _, err := s1.(*embedded).VectorNamedQuery(ctx, "nnf1_"+tc.name, specBytes, spec, ReadOpts{})
			if err != nil {
				t.Fatalf("P1 named nested fusion: %v", err)
			}
			if len(got1) == 0 {
				t.Fatal("P1 named nested fusion returned no results")
			}

			const P = 4
			sP := seedNamedQueryCollection(t, "nnf4_"+tc.name, P, n)
			touched := map[int]bool{}
			for id := uint64(1); id <= uint64(n); id++ {
				touched[ops.PartitionOf(id, P)] = true
			}
			if len(touched) != P {
				t.Fatalf("ids only touch %d/%d partitions", len(touched), P)
			}
			gotP, meta, err := sP.(*embedded).VectorNamedQuery(ctx, "nnf4_"+tc.name, specBytes, spec, ReadOpts{})
			if err != nil {
				t.Fatalf("P4 named nested fusion: %v", err)
			}
			if meta.Degraded {
				t.Fatalf("P4 named nested fusion degraded: %+v", meta)
			}
			a, b := queryResultKeys(gotP), queryResultKeys(got1)
			if !eqHybridKeys(a, b) {
				t.Fatalf("named nested multi-lane FUSION P4 != P1 (%s):\n P4=%v\n P1=%v", tc.name, a, b)
			}
		})
	}
}

// TestNamedQueryFanOutNestedFusionDepth3MatchesP1 is the depth-3 named invariant: a
// 3-level nested FUSION (grandchild multi-space FUSION wrapped by a child, wrapped by
// the parent) is P4==P1 exact (the coordinator folds EVERY FUSION node over the global
// union recursively).
func TestNamedQueryFanOutNestedFusionDepth3MatchesP1(t *testing.T) {
	const n, k = 200, 10
	ctx := context.Background()
	titleQ := []float32{1.2, 0.6, 0.3, 0.4}
	imageQ := []float32{0.5, 0.5, 0.2}
	sIdx := []uint32{1, 14, 26}
	sVal := []float32{2, 3, 1}

	grand := &pb.QuerySpec{
		Mode: pb.QueryMode_QUERY_MODE_FUSION,
		PrefetchSources: []*pb.QuerySource{
			{Source: &pb.QuerySource_Leaf{Leaf: namedDenseLeaf("image", imageQ, k)}},
			{Source: &pb.QuerySource_Leaf{Leaf: namedSparseLeaf("terms", sIdx, sVal, k)}},
		},
		FusionMethod: "rrf",
		K:            int32(k),
	}
	child := &pb.QuerySpec{
		Mode: pb.QueryMode_QUERY_MODE_FUSION,
		PrefetchSources: []*pb.QuerySource{
			{Source: &pb.QuerySource_Leaf{Leaf: namedDenseLeaf("title", titleQ, k)}},
			{Source: &pb.QuerySource_Spec{Spec: grand}},
		},
		FusionMethod: "rrf",
		K:            int32(k),
	}
	pspec := &pb.QuerySpec{
		Mode: pb.QueryMode_QUERY_MODE_FUSION,
		PrefetchSources: []*pb.QuerySource{
			{Source: &pb.QuerySource_Leaf{Leaf: namedDenseLeaf("title", titleQ, k)}},
			{Source: &pb.QuerySource_Spec{Spec: child}},
		},
		FusionMethod: "rrf",
		K:            int32(k),
	}
	specBytes, spec := buildQuerySpec(t, pspec)

	s1 := seedNamedQueryCollection(t, "nnd31", 1, n)
	got1, _, err := s1.(*embedded).VectorNamedQuery(ctx, "nnd31", specBytes, spec, ReadOpts{})
	if err != nil {
		t.Fatalf("P1 named depth-3: %v", err)
	}
	if len(got1) == 0 {
		t.Fatal("P1 named depth-3 returned no results")
	}

	const P = 4
	sP := seedNamedQueryCollection(t, "nnd34", P, n)
	gotP, meta, err := sP.(*embedded).VectorNamedQuery(ctx, "nnd34", specBytes, spec, ReadOpts{})
	if err != nil {
		t.Fatalf("P4 named depth-3: %v", err)
	}
	if meta.Degraded {
		t.Fatalf("P4 named depth-3 degraded: %+v", meta)
	}
	if !eqHybridKeys(queryResultKeys(gotP), queryResultKeys(got1)) {
		t.Fatalf("named depth-3 nested FUSION P4 != P1:\n P4=%v\n P1=%v", queryResultKeys(gotP), queryResultKeys(got1))
	}
}

// TestNamedQueryFanOutNestedRerankMatchesP1 is the named nested RERANK invariant: a
// parent FUSION whose prefetch is [a dense leaf, a nested RERANK sub-spec] is P4==P1
// exact (a nested RERANK sub-spec is partition-invariant — partition-disjoint ids,
// per-candidate re-score).
func TestNamedQueryFanOutNestedRerankMatchesP1(t *testing.T) {
	const n, k = 200, 10
	ctx := context.Background()
	titleQ := []float32{1.2, 0.6, 0.3, 0.4}
	imageQ := []float32{0.5, 0.5, 0.2}
	sIdx := []uint32{1, 14, 26}
	sVal := []float32{2, 3, 1}

	// Nested RERANK: prefetch [image dense, terms sparse] (pools >= n so both P recall
	// the same candidate set) → rerank by the dense "title" root.
	sub := &pb.QuerySpec{
		Mode: pb.QueryMode_QUERY_MODE_RERANK,
		Root: namedDenseLeaf("title", titleQ, k),
		PrefetchSources: []*pb.QuerySource{
			{Source: &pb.QuerySource_Leaf{Leaf: namedDenseLeaf("image", imageQ, n)}},
			{Source: &pb.QuerySource_Leaf{Leaf: namedSparseLeaf("terms", sIdx, sVal, n)}},
		},
		K: int32(k),
	}
	pspec := &pb.QuerySpec{
		Mode: pb.QueryMode_QUERY_MODE_FUSION,
		PrefetchSources: []*pb.QuerySource{
			{Source: &pb.QuerySource_Leaf{Leaf: namedDenseLeaf("title", titleQ, k)}},
			{Source: &pb.QuerySource_Spec{Spec: sub}},
		},
		FusionMethod: "rrf",
		K:            int32(k),
	}
	specBytes, spec := buildQuerySpec(t, pspec)

	s1 := seedNamedQueryCollection(t, "nnr1", 1, n)
	got1, _, err := s1.(*embedded).VectorNamedQuery(ctx, "nnr1", specBytes, spec, ReadOpts{})
	if err != nil {
		t.Fatalf("P1 named nested rerank: %v", err)
	}

	const P = 4
	sP := seedNamedQueryCollection(t, "nnr4", P, n)
	gotP, meta, err := sP.(*embedded).VectorNamedQuery(ctx, "nnr4", specBytes, spec, ReadOpts{})
	if err != nil {
		t.Fatalf("P4 named nested rerank: %v", err)
	}
	if meta.Degraded {
		t.Fatalf("P4 named nested rerank degraded: %+v", meta)
	}
	if !eqHybridKeys(queryResultKeys(gotP), queryResultKeys(got1)) {
		t.Fatalf("named nested RERANK P4 != P1:\n P4=%v\n P1=%v", queryResultKeys(gotP), queryResultKeys(got1))
	}
}

// TestMVQueryFanOutNestedMultiLaneFusionMatchesP1 is the MV-family nested invariant: a
// parent FUSION whose prefetch is [a MaxSim leaf, a NESTED 2-lane FUSION sub-spec over
// (MaxSim + doc-sparse)] is P4==P1 exact for RRF/weighted/dbsf (all MV lanes score-desc;
// the orientation-aware coordinator fold handles them at every FUSION node).
func TestMVQueryFanOutNestedMultiLaneFusionMatchesP1(t *testing.T) {
	const n, k = 200, 12
	ctx := context.Background()
	parentQ := [][]float32{{1.2, 0.6, 0.3, 0.4}}
	subQ := [][]float32{{0.2, 0.9, 0.1, 0.5}, {0.5, 0.1, 0.7, 0.2}}
	sIdx := []uint32{1, 14, 26}
	sVal := []float32{2, 3, 1}

	cases := []struct {
		name   string
		method string
		alpha  float64
	}{
		{"rrf", "rrf", 0},
		{"weighted", "weighted", 0.35},
		{"dbsf", "dbsf", 0.35},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			sub := &pb.QuerySpec{
				Mode: pb.QueryMode_QUERY_MODE_FUSION,
				PrefetchSources: []*pb.QuerySource{
					{Source: &pb.QuerySource_Leaf{Leaf: mvMaxSimLeaf(subQ, k)}},
					{Source: &pb.QuerySource_Leaf{Leaf: sparseLeaf(sIdx, sVal, k)}},
				},
				FusionMethod: tc.method,
				Alpha:        tc.alpha,
				K:            int32(k),
			}
			pspec := &pb.QuerySpec{
				Mode: pb.QueryMode_QUERY_MODE_FUSION,
				PrefetchSources: []*pb.QuerySource{
					{Source: &pb.QuerySource_Leaf{Leaf: mvMaxSimLeaf(parentQ, k)}},
					{Source: &pb.QuerySource_Spec{Spec: sub}},
				},
				FusionMethod: tc.method,
				Alpha:        tc.alpha,
				K:            int32(k),
			}
			specBytes, spec := buildQuerySpec(t, pspec)

			s1 := seedMVQueryCollection(t, "mnf1_"+tc.name, 1, n)
			got1, _, err := s1.(*embedded).VectorMVQuery(ctx, "mnf1_"+tc.name, specBytes, spec, ReadOpts{})
			if err != nil {
				t.Fatalf("P1 mv nested fusion: %v", err)
			}
			if len(got1) == 0 {
				t.Fatal("P1 mv nested fusion returned no results")
			}

			const P = 4
			sP := seedMVQueryCollection(t, "mnf4_"+tc.name, P, n)
			touched := map[int]bool{}
			for id := uint64(1); id <= uint64(n); id++ {
				touched[ops.PartitionOf(id, P)] = true
			}
			if len(touched) != P {
				t.Fatalf("ids only touch %d/%d partitions", len(touched), P)
			}
			gotP, meta, err := sP.(*embedded).VectorMVQuery(ctx, "mnf4_"+tc.name, specBytes, spec, ReadOpts{})
			if err != nil {
				t.Fatalf("P4 mv nested fusion: %v", err)
			}
			if meta.Degraded {
				t.Fatalf("P4 mv nested fusion degraded: %+v", meta)
			}
			if !eqHybridKeys(queryResultKeys(gotP), queryResultKeys(got1)) {
				t.Fatalf("mv nested multi-lane FUSION P4 != P1 (%s):\n P4=%v\n P1=%v", tc.name, queryResultKeys(gotP), queryResultKeys(got1))
			}
		})
	}
}

// TestMVQueryFanOutNestedRerankMatchesP1 is the MV nested RERANK invariant: a parent
// FUSION whose prefetch is [a MaxSim leaf, a nested RERANK sub-spec] is P4==P1 exact.
func TestMVQueryFanOutNestedRerankMatchesP1(t *testing.T) {
	const n, k = 200, 10
	ctx := context.Background()
	parentQ := [][]float32{{1.2, 0.6, 0.3, 0.4}}
	subQ := [][]float32{{0.2, 0.9, 0.1, 0.5}}
	rootQ := [][]float32{{0.5, 0.1, 0.7, 0.2}}
	sIdx := []uint32{1, 14, 26}
	sVal := []float32{2, 3, 1}

	sub := &pb.QuerySpec{
		Mode: pb.QueryMode_QUERY_MODE_RERANK,
		Root: mvMaxSimLeaf(rootQ, k),
		PrefetchSources: []*pb.QuerySource{
			{Source: &pb.QuerySource_Leaf{Leaf: mvMaxSimLeaf(subQ, n)}},
			{Source: &pb.QuerySource_Leaf{Leaf: sparseLeaf(sIdx, sVal, n)}},
		},
		K: int32(k),
	}
	pspec := &pb.QuerySpec{
		Mode: pb.QueryMode_QUERY_MODE_FUSION,
		PrefetchSources: []*pb.QuerySource{
			{Source: &pb.QuerySource_Leaf{Leaf: mvMaxSimLeaf(parentQ, k)}},
			{Source: &pb.QuerySource_Spec{Spec: sub}},
		},
		FusionMethod: "rrf",
		K:            int32(k),
	}
	specBytes, spec := buildQuerySpec(t, pspec)

	s1 := seedMVQueryCollection(t, "mnr1", 1, n)
	got1, _, err := s1.(*embedded).VectorMVQuery(ctx, "mnr1", specBytes, spec, ReadOpts{})
	if err != nil {
		t.Fatalf("P1 mv nested rerank: %v", err)
	}

	const P = 4
	sP := seedMVQueryCollection(t, "mnr4", P, n)
	gotP, meta, err := sP.(*embedded).VectorMVQuery(ctx, "mnr4", specBytes, spec, ReadOpts{})
	if err != nil {
		t.Fatalf("P4 mv nested rerank: %v", err)
	}
	if meta.Degraded {
		t.Fatalf("P4 mv nested rerank degraded: %+v", meta)
	}
	if !eqHybridKeys(queryResultKeys(gotP), queryResultKeys(got1)) {
		t.Fatalf("mv nested RERANK P4 != P1:\n P4=%v\n P1=%v", queryResultKeys(gotP), queryResultKeys(got1))
	}
}

// TestMVQueryFanOutNestedFusionDepth3MatchesP1 is the MV depth-3 fan-out invariant:
// a 3-level nested FUSION (grandchild MaxSim+sparse wrapped by child FUSION, wrapped
// by the parent) is P4==P1 exact — the coordinator folds EVERY FUSION node over the
// global union recursively.
func TestMVQueryFanOutNestedFusionDepth3MatchesP1(t *testing.T) {
	const n, k = 200, 10
	ctx := context.Background()
	parentQ := [][]float32{{1.2, 0.6, 0.3, 0.4}}
	subQ := [][]float32{{0.2, 0.9, 0.1, 0.5}, {0.5, 0.1, 0.7, 0.2}}
	grandQ := [][]float32{{0.7, 0.3, 0.5, 0.1}}
	sIdx := []uint32{1, 14, 26}
	sVal := []float32{2, 3, 1}

	grand := &pb.QuerySpec{
		Mode: pb.QueryMode_QUERY_MODE_FUSION,
		PrefetchSources: []*pb.QuerySource{
			{Source: &pb.QuerySource_Leaf{Leaf: mvMaxSimLeaf(grandQ, k)}},
			{Source: &pb.QuerySource_Leaf{Leaf: sparseLeaf(sIdx, sVal, k)}},
		},
		FusionMethod: "rrf",
		K:            int32(k),
	}
	child := &pb.QuerySpec{
		Mode: pb.QueryMode_QUERY_MODE_FUSION,
		PrefetchSources: []*pb.QuerySource{
			{Source: &pb.QuerySource_Leaf{Leaf: mvMaxSimLeaf(subQ, k)}},
			{Source: &pb.QuerySource_Spec{Spec: grand}},
		},
		FusionMethod: "rrf",
		K:            int32(k),
	}
	pspec := &pb.QuerySpec{
		Mode: pb.QueryMode_QUERY_MODE_FUSION,
		PrefetchSources: []*pb.QuerySource{
			{Source: &pb.QuerySource_Leaf{Leaf: mvMaxSimLeaf(parentQ, k)}},
			{Source: &pb.QuerySource_Spec{Spec: child}},
		},
		FusionMethod: "rrf",
		K:            int32(k),
	}
	specBytes, spec := buildQuerySpec(t, pspec)
	if !vector.SpecHasNestedFusion(spec) {
		t.Fatal("pspec should have a nested multi-lane FUSION node")
	}

	s1 := seedMVQueryCollection(t, "mvd31", 1, n)
	got1, _, err := s1.(*embedded).VectorMVQuery(ctx, "mvd31", specBytes, spec, ReadOpts{})
	if err != nil {
		t.Fatalf("P1 MV depth-3: %v", err)
	}
	if len(got1) == 0 {
		t.Fatal("P1 MV depth-3 returned no results")
	}

	const P = 4
	sP := seedMVQueryCollection(t, "mvd34", P, n)
	touched := map[int]bool{}
	for id := uint64(1); id <= uint64(n); id++ {
		touched[ops.PartitionOf(id, P)] = true
	}
	if len(touched) != P {
		t.Fatalf("ids only touch %d/%d partitions", len(touched), P)
	}
	gotP, meta, err := sP.(*embedded).VectorMVQuery(ctx, "mvd34", specBytes, spec, ReadOpts{})
	if err != nil {
		t.Fatalf("P4 MV depth-3: %v", err)
	}
	if meta.Degraded {
		t.Fatalf("P4 MV depth-3 degraded: %+v", meta)
	}
	if !eqHybridKeys(queryResultKeys(gotP), queryResultKeys(got1)) {
		t.Fatalf("MV depth-3 nested FUSION P4 != P1:\n P4=%v\n P1=%v", queryResultKeys(gotP), queryResultKeys(got1))
	}
}

// TestNamedQueryFanOutNestedFusionWithRecommendMatchesP1 is the nested
// recommend-in-named-FUSION coverage test: a parent FUSION whose prefetch is [a dense
// leaf, a NESTED 2-lane FUSION sub-spec carrying a RECOMMEND leaf] is P4==P1 exact,
// example ids excluded, no under-fill. Tests BOTH AVERAGE_VECTOR and BEST_SCORE.
// Mirrors the dense TestQueryFanOutNestedFusionWithRecommendMatchesP1.
//
// UNDER-FILL INVESTIGATION (reviewer request): the over-fetch widen previously applied
// only to the root spec.K; collectTreeLanesAt uses each sub-spec's own K as the lane
// pool for its leaf children. A recommend leaf buried inside the nested sub-spec (now
// rewritten to dense/best-score) would fetch sub.K candidates (no room for exclude).
// The fix (widenNestedSpecsK) widens every nested sub-spec K by nExclude. The test
// asserts the result has EXACTLY k items (no under-fill) and example ids are absent.
func TestNamedQueryFanOutNestedFusionWithRecommendMatchesP1(t *testing.T) {
	const n, k = 40, 5
	ctx := context.Background()
	positive := []uint64{3, 5, 7}
	sIdx := []uint32{3}
	sVal := []float32{1}

	cases := []struct {
		name    string
		leafFn  func(string, []uint64, []uint64, int) *pb.QueryLeaf
		negPart bool // whether negatives are used (AVERAGE uses neg, BEST_SCORE uses neg only with Cosine)
	}{
		{"average_vector", namedRecommendLeaf, false},
		{"best_score", namedRecommendBestLeaf, false},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			// Nested sub-spec: [recommend-in-"title" + sparse-"terms"]. Use dbsf to avoid
			// tie-based rank divergence between P1 and P4 (continuous scores).
			subSpec := &pb.QuerySpec{
				Mode: pb.QueryMode_QUERY_MODE_FUSION,
				PrefetchSources: []*pb.QuerySource{
					{Source: &pb.QuerySource_Leaf{Leaf: tc.leafFn("title", positive, nil, k)}},
					{Source: &pb.QuerySource_Leaf{Leaf: namedSparseLeaf("terms", sIdx, sVal, k)}},
				},
				FusionMethod: "dbsf",
				Alpha:        0.5,
				K:            int32(k),
			}
			pspec := &pb.QuerySpec{
				Mode: pb.QueryMode_QUERY_MODE_FUSION,
				PrefetchSources: []*pb.QuerySource{
					{Source: &pb.QuerySource_Leaf{Leaf: namedDenseLeaf("title", []float32{0.5, 0.5, 1, 1}, k)}},
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

			s1 := seedNamedRecDiscCollection(t, "nnrec1_"+tc.name, 1, n)
			got1, _, err := s1.(*embedded).VectorNamedQuery(ctx, "nnrec1_"+tc.name, specBytes, spec, ReadOpts{})
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
			// No under-fill: must have exactly k results (widenNestedSpecsK fixed this).
			if len(got1) != k {
				t.Fatalf("%s P1: got %d results, want %d (under-fill detected)", tc.name, len(got1), k)
			}

			const P = 4
			// Re-decode from pspec so the in-place recommend→dense rewrite from the P1 call
			// does not affect the P4 call's spec (mirrors the dense nested-recommend test).
			_, specP := buildQuerySpec(t, pspec)
			sP := seedNamedRecDiscCollection(t, "nnrec4_"+tc.name, P, n)
			gotP, meta, err := sP.(*embedded).VectorNamedQuery(ctx, "nnrec4_"+tc.name, specBytes, specP, ReadOpts{})
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
			if len(gotP) != k {
				t.Fatalf("%s P4: got %d results, want %d (under-fill detected)", tc.name, len(gotP), k)
			}
			if !eqHybridKeys(queryResultKeys(gotP), queryResultKeys(got1)) {
				t.Fatalf("%s nested-FUSION-with-recommend P4 != P1:\n P4=%v\n P1=%v", tc.name, queryResultKeys(gotP), queryResultKeys(got1))
			}
		})
	}
}

// TestNamedQueryFanOutNestedFusionWithDiscoverMatchesP1 is the nested
// discover-in-named-FUSION coordinator tree-walk test: a parent FUSION whose prefetch
// is [a dense leaf, a NESTED 2-lane FUSION sub-spec carrying a DISCOVER leaf] must
// succeed without error and return a non-empty result at P=4 — proving the coordinator
// tree-walk pre-pass resolves the nested discover leaf cluster-wide before fan-out.
//
// Strict P4==P1 is NOT asserted: the discover lane's per-candidate score depends on the
// per-partition candidate POOL (each partition surfaces its own subset of docs under the
// discover seed query), so the lane sizes differ between P=1 (all n docs) and P=4 (~n/4
// per partition) — the same pool-size caveat as the flat named discover fan-out test.
// The important invariant here is: (a) no error (the nested discover leaf was resolved
// by the coordinator tree-walk pre-pass, not rejected partition-side as ErrIDNotFound),
// (b) a non-empty result is returned.
func TestNamedQueryFanOutNestedFusionWithDiscoverMatchesP1(t *testing.T) {
	const n, k = 40, 5
	ctx := context.Background()
	target := []uint64{20}
	pairs := [][2]uint64{{5, 38}, {6, 37}}
	sIdx := []uint32{3}
	sVal := []float32{1}

	// Nested sub-spec: [discover-in-"title" + sparse-"terms"]. rrf (not dbsf) so even a
	// pool-size-sensitive discover lane yields a scored rank rather than a [0,1] min/max.
	subSpec := &pb.QuerySpec{
		Mode: pb.QueryMode_QUERY_MODE_FUSION,
		PrefetchSources: []*pb.QuerySource{
			{Source: &pb.QuerySource_Leaf{Leaf: namedDiscoverLeaf("title", target, pairs, k)}},
			{Source: &pb.QuerySource_Leaf{Leaf: namedSparseLeaf("terms", sIdx, sVal, k)}},
		},
		FusionMethod: "rrf",
		K:            int32(k),
	}
	pspec := &pb.QuerySpec{
		Mode: pb.QueryMode_QUERY_MODE_FUSION,
		PrefetchSources: []*pb.QuerySource{
			{Source: &pb.QuerySource_Leaf{Leaf: namedDenseLeaf("title", []float32{0.5, 0.5, 1, 1}, k)}},
			{Source: &pb.QuerySource_Spec{Spec: subSpec}},
		},
		FusionMethod: "rrf",
		K:            int32(k),
	}
	specBytes, spec := buildQuerySpec(t, pspec)
	if !vector.SpecHasNestedFusion(spec) {
		t.Fatal("pspec should have a nested multi-lane FUSION node")
	}

	// P=1 baseline: no error, non-empty result.
	s1 := seedNamedRecDiscCollection(t, "nndisc1", 1, n)
	got1, _, err := s1.(*embedded).VectorNamedQuery(ctx, "nndisc1", specBytes, spec, ReadOpts{})
	if err != nil {
		t.Fatalf("P1 nested discover: %v", err)
	}
	if len(got1) == 0 {
		t.Fatal("P1 nested discover returned no results")
	}

	const P = 4
	_, specP := buildQuerySpec(t, pspec)
	sP := seedNamedRecDiscCollection(t, "nndisc4", P, n)
	gotP, meta, err := sP.(*embedded).VectorNamedQuery(ctx, "nndisc4", specBytes, specP, ReadOpts{})
	if err != nil {
		// Fail-loud: an un-resolved nested discover leaf would reach the partition handler
		// and surface as an error (ErrIDNotFound or ErrQueryDiscoverNoContext). A successful
		// call here proves the coordinator tree-walk pre-pass resolved the nested leaf.
		t.Fatalf("P4 nested discover (coordinator tree-walk pre-pass must resolve nested leaf): %v", err)
	}
	if meta.Degraded {
		t.Fatalf("P4 nested discover unexpectedly degraded: %+v", meta)
	}
	if len(gotP) == 0 {
		t.Fatal("P4 nested discover returned no results")
	}
	// Confirm all returned ids are valid (1..n), not garbage.
	for _, r := range gotP {
		if r.ID < 1 || r.ID > uint64(n) {
			t.Fatalf("P4 nested discover returned out-of-range id %d", r.ID)
		}
	}
}
