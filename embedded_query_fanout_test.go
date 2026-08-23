// SPDX-License-Identifier: Apache-2.0

package rostam

import (
	"context"
	"testing"

	"google.golang.org/protobuf/proto"

	"github.com/rostamlabs/rostam/sdk/pb"
	"github.com/rostamlabs/rostam/ops"
	"github.com/rostamlabs/rostam/vector"
)

// seedQueryCollection creates a P-partition (or P==1) DENSE collection (dim 4,
// L2) with an inline sparse lane, then inserts ids 1..n each with a deterministic
// dense vector AND a deterministic sparse vector so both prefetch lanes can score
// every point. Mirrors the hybrid fan-out oracle seed (a dense collection with
// sparse), but for the unified Query API.
func seedQueryCollection(t *testing.T, s Store, coll string, P, n int) {
	t.Helper()
	ctx := context.Background()
	if err := s.CreateCollection(ctx, coll, VectorConfig{
		Dim: 4, M: 8, EfConstruction: 50, EfSearch: 32, Seed: 1, Metric: vector.L2, Partitions: P,
	}); err != nil {
		t.Fatalf("CreateCollection %q (P=%d): %v", coll, P, err)
	}
	for id := uint64(1); id <= uint64(n); id++ {
		// Dense vector below the smallest query so distances are tie-free in id.
		v := []float32{float32(id), 0, 0, 0}
		sp := VectorSparse{Indices: []uint32{uint32(id % 7)}, Values: []float32{float32(id)*0.01 + 1}}
		if err := s.VectorInsertExt(ctx, coll, id, v, VectorInsertOpts{Sparse: sp}); err != nil {
			t.Fatalf("VectorInsertExt %s/%d: %v", coll, id, err)
		}
	}
}

// buildQuerySpec builds a marshaled pb.QuerySpec + the matching engine
// vector.QuerySpec (via the same ops decode path the handler/coordinator uses),
// so a test passes both to VectorQuery exactly like the wire layers will.
func buildQuerySpec(t *testing.T, p *pb.QuerySpec) ([]byte, vector.QuerySpec) {
	t.Helper()
	specBytes, err := proto.Marshal(p)
	if err != nil {
		t.Fatalf("marshal QuerySpec: %v", err)
	}
	// Round-trip through the public coordinator decoder to obtain the engine spec
	// exactly as the dispatcher will (also validates the spec converts cleanly).
	_, _, spec, _, _, _, err := ops.DecodeQuerySpecArgs(ops.EncodeQueryArgs("x", specBytes, 0, 0, 0))
	if err != nil {
		t.Fatalf("DecodeQuerySpecArgs: %v", err)
	}
	return specBytes, spec
}

func denseLeaf(dense []float32, k int) *pb.QueryLeaf {
	return &pb.QueryLeaf{Leaf: &pb.QueryLeaf_Dense{Dense: &pb.DenseLeaf{Dense: dense, K: int32(k)}}}
}

func sparseLeaf(idx []uint32, vals []float32, k int) *pb.QueryLeaf {
	return &pb.QueryLeaf{Leaf: &pb.QueryLeaf_Sparse{Sparse: &pb.SparseLeaf{Indices: idx, Values: vals, K: int32(k)}}}
}

func queryResultKeys(res []VectorResult) []hybridResultKey {
	out := make([]hybridResultKey, len(res))
	for i, r := range res {
		out[i] = hybridResultKey{r.ID, r.Score, r.Distance}
	}
	return out
}

// TestQueryFanOutFusionMatchesP1 is the #1 exactness invariant: a FUSION query
// (dense + sparse prefetch lanes) over a P=4 collection returns the EXACT same
// fused top-k (id + score + distance, in order) as the same query over a P=1
// collection — for RRF, weighted, AND dbsf. This proves queryFanOut unions the
// lanes + truncates per-lane to the global K + fuses ONCE (no per-partition
// pre-fuse). It goes RED if a partition pre-fuses or the per-lane truncation is
// wrong.
func TestQueryFanOutFusionMatchesP1(t *testing.T) {
	const n, k = 200, 10
	ctx := context.Background()
	denseQ := []float32{0.5, 0, 0, 0}
	sIdx := []uint32{3}
	sVal := []float32{1}

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
			pspec := &pb.QuerySpec{
				Mode: pb.QueryMode_QUERY_MODE_FUSION,
				Prefetch: []*pb.QueryLeaf{
					denseLeaf(denseQ, k),
					sparseLeaf(sIdx, sVal, k),
				},
				FusionMethod: tc.method,
				Alpha:        tc.alpha,
				K:            int32(k),
			}
			specBytes, spec := buildQuerySpec(t, pspec)

			s1 := newSingleEmbedded(t)
			waitLeaderEmbedded(t, s1)
			seedQueryCollection(t, s1, "q1", 1, n)
			got1, _, err := s1.(*embedded).VectorQuery(ctx, "q1", specBytes, spec, ReadOpts{})
			if err != nil {
				t.Fatalf("P1 query: %v", err)
			}

			const P = 4
			sP := newSingleEmbedded(t)
			waitLeaderEmbedded(t, sP)
			seedQueryCollection(t, sP, "q4", P, n)
			touched := map[int]bool{}
			for id := uint64(1); id <= uint64(n); id++ {
				touched[ops.PartitionOf(id, P)] = true
			}
			if len(touched) != P {
				t.Fatalf("ids only touch %d/%d partitions", len(touched), P)
			}
			gotP, meta, err := sP.(*embedded).VectorQuery(ctx, "q4", specBytes, spec, ReadOpts{})
			if err != nil {
				t.Fatalf("P4 query: %v", err)
			}
			if meta.Degraded {
				t.Fatalf("P4 query unexpectedly degraded: %+v", meta)
			}
			if len(gotP) == 0 {
				t.Fatal("P4 query returned no results")
			}
			a, b := queryResultKeys(gotP), queryResultKeys(got1)
			if !eqHybridKeys(a, b) {
				t.Fatalf("FUSION P4 != P1:\n P4=%v\n P1=%v", a, b)
			}
		})
	}
}

// TestQueryFanOutRerankMatchesP1 proves RERANK partition-invariance: a rerank
// query (prefetch dense+sparse → rerank by a dense root) over P=4 returns the
// exact same reranked top-k as P=1. Exact because point ids are partition-
// disjoint (each doc is prefetched + reranked on its sole owning partition), so
// the coordinator's merge-sort of the per-partition reranked top-ks == the
// single-partition rerank.
func TestQueryFanOutRerankMatchesP1(t *testing.T) {
	const n, k = 200, 10
	ctx := context.Background()
	denseRoot := []float32{0.5, 0, 0, 0}
	denseP := []float32{0.5, 0, 0, 0}
	sIdx := []uint32{3}
	sVal := []float32{1}

	pspec := &pb.QuerySpec{
		Mode: pb.QueryMode_QUERY_MODE_RERANK,
		Root: denseLeaf(denseRoot, k),
		Prefetch: []*pb.QueryLeaf{
			denseLeaf(denseP, 50),
			sparseLeaf(sIdx, sVal, 50),
		},
		K: int32(k),
	}
	specBytes, spec := buildQuerySpec(t, pspec)

	s1 := newSingleEmbedded(t)
	waitLeaderEmbedded(t, s1)
	seedQueryCollection(t, s1, "qr1", 1, n)
	got1, _, err := s1.(*embedded).VectorQuery(ctx, "qr1", specBytes, spec, ReadOpts{})
	if err != nil {
		t.Fatalf("P1 rerank: %v", err)
	}

	const P = 4
	sP := newSingleEmbedded(t)
	waitLeaderEmbedded(t, sP)
	seedQueryCollection(t, sP, "qr4", P, n)
	gotP, meta, err := sP.(*embedded).VectorQuery(ctx, "qr4", specBytes, spec, ReadOpts{})
	if err != nil {
		t.Fatalf("P4 rerank: %v", err)
	}
	if meta.Degraded {
		t.Fatalf("P4 rerank unexpectedly degraded: %+v", meta)
	}
	if len(gotP) == 0 {
		t.Fatal("P4 rerank returned no results")
	}
	a, b := queryResultKeys(gotP), queryResultKeys(got1)
	if !eqHybridKeys(a, b) {
		t.Fatalf("RERANK P4 != P1:\n P4=%v\n P1=%v", a, b)
	}
}

// TestQueryFanOutDegradation: dropping one physical partition makes the
// coordinator's fan-out to that partition fail. With the default Partial mode the
// query returns partial results flagged Degraded with the missing partition
// index; with OnPartitionUnavailable=Fail (1) the whole query errors. Mirrors
// TestEmbeddedSearchDegraded for the unified Query API.
func TestQueryFanOutDegradationFailErrors(t *testing.T) {
	const n, k = 80, 10
	ctx := context.Background()
	pspec := &pb.QuerySpec{
		Mode: pb.QueryMode_QUERY_MODE_FUSION,
		Prefetch: []*pb.QueryLeaf{
			denseLeaf([]float32{0.5, 0, 0, 0}, k),
			sparseLeaf([]uint32{3}, []float32{1}, k),
		},
		FusionMethod: "rrf",
		K:            int32(k),
	}
	specBytes, spec := buildQuerySpec(t, pspec)

	const P = 4
	s := newSingleEmbedded(t)
	waitLeaderEmbedded(t, s)
	seedQueryCollection(t, s, "qd", P, n)
	emb := s.(*embedded)

	// Sanity: healthy query succeeds, not degraded.
	if _, meta, err := emb.VectorQuery(ctx, "qd", specBytes, spec, ReadOpts{}); err != nil || meta.Degraded {
		t.Fatalf("healthy query: err=%v degraded=%v", err, meta.Degraded)
	}

	// Make partition 1 unreachable by dropping its physical collection (gen 0).
	if _, err := emb.Call(ctx, "vector_drop_collection",
		ops.EncodeDropCollectionArgs(string(ops.PartitionKeyGen("qd", 0, 1)))); err != nil {
		t.Fatalf("drop partition 1: %v", err)
	}

	// Fail mode (OnPartitionUnavailable=1) → error.
	if _, _, err := emb.VectorQuery(ctx, "qd", specBytes, spec, ReadOpts{OnPartitionUnavailable: 1}); err == nil {
		t.Fatal("expected error with OnPartitionUnavailable=Fail and an unreachable partition")
	}
	// Partial mode (default) → degraded + partial (no error).
	res, meta, err := emb.VectorQuery(ctx, "qd", specBytes, spec, ReadOpts{})
	if err != nil {
		t.Fatalf("partial-mode query errored: %v", err)
	}
	if !meta.Degraded {
		t.Fatal("expected Degraded=true after a partition was made unreachable")
	}
	if len(meta.Missing) == 0 {
		t.Fatal("expected non-empty Missing after a partition was made unreachable")
	}
	if len(res) == 0 {
		t.Fatal("expected partial results from the reachable partitions")
	}
}

// TestQueryFanOutRcRidesEveryArg is the anti-silent-drop guard: the exact
// per-partition encode line queryFanOut's Encode closure uses must carry the
// Linearizable rc so ReadConsistencyOf("vector_query", arg) arms each shard's
// data barrier. If rc were dropped from the per-partition encode, this fails
// (RED-reproduces the silent-degrade hole). The bound also rides for a bounded
// query.
func TestQueryFanOutRcRidesEveryArg(t *testing.T) {
	pspec := &pb.QuerySpec{
		Mode:     pb.QueryMode_QUERY_MODE_FUSION,
		Prefetch: []*pb.QueryLeaf{denseLeaf([]float32{1, 0, 0, 0}, 10), sparseLeaf([]uint32{1}, []float32{1}, 10)},
		K:        10,
	}
	specBytes, _ := buildQuerySpec(t, pspec)

	// The exact per-partition encode line from queryFanOut (rc/opa/bound ride
	// every arg). Linearizable rc must survive the encode → ReadConsistencyOf.
	arg := ops.EncodeQueryArgs("phys#2", specBytes, ops.ConsistencyLinearizable, 0, 0)
	rc, ok := ops.ReadConsistencyOf("vector_query", arg)
	if !ok {
		t.Fatal("ReadConsistencyOf(vector_query) not covered — a Linearizable query would NOT arm the shard barrier")
	}
	if rc != ops.ConsistencyLinearizable {
		t.Fatalf("per-partition arg rc = %d, want Linearizable(%d) — rc dropped (silent-degrade)", rc, ops.ConsistencyLinearizable)
	}

	// Bounded staleness: the bound must round-trip on the per-partition arg too.
	const bound = uint64(1234)
	bArg := ops.EncodeQueryArgs("phys#2", specBytes, ops.ConsistencyBoundedStaleness, 0, bound)
	gotBound, okB := ops.ReadStalenessOf("vector_query", bArg)
	if !okB || gotBound != bound {
		t.Fatalf("ReadStalenessOf(vector_query) = (%d,%v), want (%d,true)", gotBound, okB, bound)
	}
}

// TestFanQueryWireIsFlatDegraded locks the Task-3 coordinator-result CONTRACT:
// the fanoutDispatcher's fanQuery ALWAYS re-encodes a FLAT (RERANK-tagged)
// fused/reranked top-k + degraded/missing trailer — for BOTH a partitioned (P>1)
// AND an unpartitioned (P<=1) FUSION query — so the dedicated VectorQuery RPC,
// the HTTP /query route, and the networked client can all decode one uniform
// shape via DecodeQueryResultDegraded. The unpartitioned case is the regression
// guard: the per-shard handler returns UNFUSED lanes for FUSION, so a pass-
// through would leak the lanes shape; fanQuery must run the local fusion merge.
func TestFanQueryWireIsFlatDegraded(t *testing.T) {
	const n, k = 120, 10
	pspec := &pb.QuerySpec{
		Mode: pb.QueryMode_QUERY_MODE_FUSION,
		Prefetch: []*pb.QueryLeaf{
			denseLeaf([]float32{0.5, 0, 0, 0}, k),
			sparseLeaf([]uint32{3}, []float32{1}, k),
		},
		FusionMethod: "rrf",
		K:            int32(k),
	}
	specBytes, _ := buildQuerySpec(t, pspec)

	for _, P := range []int{1, 4} {
		t.Run("P"+itoaTest(P), func(t *testing.T) {
			s := newSingleEmbedded(t)
			waitLeaderEmbedded(t, s)
			coll := "fq" + itoaTest(P)
			seedQueryCollection(t, s, coll, P, n)
			emb := s.(*embedded)
			fan := newFanoutDispatcher(emb, emb.node)

			body, err := fan.Call("vector_query", ops.EncodeQueryArgs(coll, specBytes, 0, 0, 0))
			if err != nil {
				t.Fatalf("fanQuery P=%d: %v", P, err)
			}
			// Must decode as a FLAT fused result (NOT unfused FUSION lanes).
			res, degraded, missing, derr := ops.DecodeQueryResultDegraded(body)
			if derr != nil {
				t.Fatalf("DecodeQueryResultDegraded P=%d: %v (wire is not flat fused)", P, derr)
			}
			if degraded || len(missing) != 0 {
				t.Fatalf("healthy P=%d query degraded=%v missing=%v, want false/empty", P, degraded, missing)
			}
			if len(res) == 0 || len(res) > k {
				t.Fatalf("P=%d fused top-k len = %d, want 1..%d", P, len(res), k)
			}
			// Cross-check against the embedded VectorQuery's flat top-k (same merge).
			want, _, werr := emb.VectorQuery(context.Background(), coll, specBytes, mustEngineSpec(t, specBytes), ReadOpts{})
			if werr != nil {
				t.Fatalf("embedded VectorQuery P=%d: %v", P, werr)
			}
			if !eqHybridKeys(queryResultKeys(res), queryResultKeys(want)) {
				t.Fatalf("P=%d fanQuery wire top-k != embedded VectorQuery top-k", P)
			}
		})
	}
}

// TestQueryFanOutNestedRerankMatchesP1 is the depth-2 P>1==P1 invariant: a parent
// FUSION spec whose prefetch is [a dense leaf, a NESTED RERANK sub-spec] returns the
// EXACT same fused top-k over P=4 as over P=1. A nested RERANK sub-spec is partition-
// invariant (its candidate union is partition-disjoint and the rerank score is a
// per-candidate function → each partition produces the globally-correct score for its
// own ids), so the recursion holds at depth 2 exactly like a leaf lane. This is the
// meaningful Qdrant multi-stage case (prefetch → rerank sub-stage → fuse at parent)
// and exercises runQuerySpecAt(depth+1) + SourceOrientation/SourceLanePool for a
// nested spec source on every partition.
func TestQueryFanOutNestedRerankMatchesP1(t *testing.T) {
	const n, k = 200, 10
	ctx := context.Background()
	denseQ := []float32{0.5, 0, 0, 0}
	sIdx := []uint32{3}
	sVal := []float32{1}

	// The nested RERANK sub-spec: prefetch [dense, sparse] → rerank by a dense root.
	subSpec := &pb.QuerySpec{
		Mode: pb.QueryMode_QUERY_MODE_RERANK,
		Root: denseLeaf(denseQ, k),
		PrefetchSources: []*pb.QuerySource{
			{Source: &pb.QuerySource_Leaf{Leaf: denseLeaf(denseQ, 50)}},
			{Source: &pb.QuerySource_Leaf{Leaf: sparseLeaf(sIdx, sVal, 50)}},
		},
		K: int32(k),
	}
	// The parent FUSION: [a dense leaf source, the nested RERANK sub-spec source].
	pspec := &pb.QuerySpec{
		Mode: pb.QueryMode_QUERY_MODE_FUSION,
		PrefetchSources: []*pb.QuerySource{
			{Source: &pb.QuerySource_Leaf{Leaf: denseLeaf(denseQ, k)}},
			{Source: &pb.QuerySource_Spec{Spec: subSpec}},
		},
		FusionMethod: "rrf",
		K:            int32(k),
	}
	specBytes, spec := buildQuerySpec(t, pspec)

	s1 := newSingleEmbedded(t)
	waitLeaderEmbedded(t, s1)
	seedQueryCollection(t, s1, "qn1", 1, n)
	got1, _, err := s1.(*embedded).VectorQuery(ctx, "qn1", specBytes, spec, ReadOpts{})
	if err != nil {
		t.Fatalf("P1 nested query: %v", err)
	}
	if len(got1) == 0 {
		t.Fatal("P1 nested query returned no results")
	}

	const P = 4
	sP := newSingleEmbedded(t)
	waitLeaderEmbedded(t, sP)
	seedQueryCollection(t, sP, "qn4", P, n)
	touched := map[int]bool{}
	for id := uint64(1); id <= uint64(n); id++ {
		touched[ops.PartitionOf(id, P)] = true
	}
	if len(touched) != P {
		t.Fatalf("ids only touch %d/%d partitions", len(touched), P)
	}
	gotP, meta, err := sP.(*embedded).VectorQuery(ctx, "qn4", specBytes, spec, ReadOpts{})
	if err != nil {
		t.Fatalf("P4 nested query: %v", err)
	}
	if meta.Degraded {
		t.Fatalf("P4 nested query unexpectedly degraded: %+v", meta)
	}
	a, b := queryResultKeys(gotP), queryResultKeys(got1)
	if !eqHybridKeys(a, b) {
		t.Fatalf("nested-RERANK depth-2 P4 != P1:\n P4=%v\n P1=%v", a, b)
	}
}

// TestQueryFanOutNestedSingleLaneFusionMatchesP1 is the second partition-invariant
// nested-FUSION case: a parent FUSION spec whose prefetch is [a sparse leaf, a NESTED
// SINGLE-LANE FUSION sub-spec (one dense lane)]. A single-lane FUSION sub-spec returns
// its lone lane verbatim (no fold, no rank-derived score synthesis), so it carries the
// inner leaf's PARTITION-INVARIANT key (dense Distance) straight through — exact at
// P>1. It locks the SourceOrientation/SourceLanePool 1-lane branches in the fan-out.
// (A nested MULTI-lane FUSION sub-spec is now ALSO P>1==P1 exact — it ships its UNFUSED
// lanes and the coordinator folds the nested FUSION node over the global union — asserted
// by TestQueryFanOutNestedMultiLaneFusionMatchesP1 and its depth-3 / mixed-tree siblings.)
func TestQueryFanOutNestedSingleLaneFusionMatchesP1(t *testing.T) {
	const n, k = 200, 10
	ctx := context.Background()
	denseQ := []float32{0.5, 0, 0, 0}
	sIdx := []uint32{3}
	sVal := []float32{1}

	subSpec := &pb.QuerySpec{
		Mode: pb.QueryMode_QUERY_MODE_FUSION,
		PrefetchSources: []*pb.QuerySource{
			{Source: &pb.QuerySource_Leaf{Leaf: denseLeaf(denseQ, k)}},
		},
		FusionMethod: "rrf",
		K:            int32(k),
	}
	pspec := &pb.QuerySpec{
		Mode: pb.QueryMode_QUERY_MODE_FUSION,
		PrefetchSources: []*pb.QuerySource{
			{Source: &pb.QuerySource_Leaf{Leaf: sparseLeaf(sIdx, sVal, k)}},
			{Source: &pb.QuerySource_Spec{Spec: subSpec}},
		},
		FusionMethod: "rrf",
		K:            int32(k),
	}
	specBytes, spec := buildQuerySpec(t, pspec)

	s1 := newSingleEmbedded(t)
	waitLeaderEmbedded(t, s1)
	seedQueryCollection(t, s1, "qs1", 1, n)
	got1, _, err := s1.(*embedded).VectorQuery(ctx, "qs1", specBytes, spec, ReadOpts{})
	if err != nil {
		t.Fatalf("P1 nested single-lane query: %v", err)
	}
	if len(got1) == 0 {
		t.Fatal("P1 nested single-lane query returned no results")
	}

	const P = 4
	sP := newSingleEmbedded(t)
	waitLeaderEmbedded(t, sP)
	seedQueryCollection(t, sP, "qs4", P, n)
	gotP, meta, err := sP.(*embedded).VectorQuery(ctx, "qs4", specBytes, spec, ReadOpts{})
	if err != nil {
		t.Fatalf("P4 nested single-lane query: %v", err)
	}
	if meta.Degraded {
		t.Fatalf("P4 nested single-lane query unexpectedly degraded: %+v", meta)
	}
	a, b := queryResultKeys(gotP), queryResultKeys(got1)
	if !eqHybridKeys(a, b) {
		t.Fatalf("nested single-lane FUSION depth-2 P4 != P1:\n P4=%v\n P1=%v", a, b)
	}
}

// itoaTest is a tiny local int→string for subtest names (avoids importing strconv
// just for the test name).
func itoaTest(n int) string {
	if n == 0 {
		return "0"
	}
	var b []byte
	for n > 0 {
		b = append([]byte{byte('0' + n%10)}, b...)
		n /= 10
	}
	return string(b)
}

// mustEngineSpec decodes the engine spec from marshaled spec bytes via the public
// coordinator decoder (the same path the dispatcher uses).
func mustEngineSpec(t *testing.T, specBytes []byte) vector.QuerySpec {
	t.Helper()
	_, _, spec, _, _, _, err := ops.DecodeQuerySpecArgs(ops.EncodeQueryArgs("x", specBytes, 0, 0, 0))
	if err != nil {
		t.Fatalf("DecodeQuerySpecArgs: %v", err)
	}
	return spec
}

// denseLeafLaneK / sparseLeafLaneK build a dense/sparse leaf with an explicit per-lane
// pool (LaneK) so a nested sub-spec's candidate set is BOUNDED — small enough that the
// globally-best candidates span multiple partitions and a PARTITION-LOCAL nested fuse
// (the v1 lossy step) diverges from the global fuse.
func denseLeafLaneK(dense []float32, k, laneK int) *pb.QueryLeaf {
	return &pb.QueryLeaf{Leaf: &pb.QueryLeaf_Dense{Dense: &pb.DenseLeaf{Dense: dense, K: int32(k), LaneK: int32(laneK)}}}
}

func sparseLeafLaneK(idx []uint32, vals []float32, k, laneK int) *pb.QueryLeaf {
	return &pb.QueryLeaf{Leaf: &pb.QueryLeaf_Sparse{Sparse: &pb.SparseLeaf{Indices: idx, Values: vals, K: int32(k), LaneK: int32(laneK)}}}
}

// TestQueryFanOutNestedMultiLaneFusionMatchesP1 is THE HEADLINE: a parent FUSION spec
// whose prefetch is [a dense leaf, a NESTED 2-lane (dense+sparse) FUSION sub-spec]
// returns the EXACT same fused top-k (id+score+distance, in order) over P=4 as over
// P=1, with the sub-spec's globally-best candidates SPANNING partitions. On master the
// nested multi-lane FUSION is fused PARTITION-LOCALLY (each partition RRF-normalizes over
// ITS OWN candidates → a partition-dependent fused score) so P4 DIVERGED from P1; after
// the fix each partition ships the sub-spec's UNFUSED lanes and the coordinator folds the
// nested FUSION node over the GLOBAL union ⇒ P4==P1 exact. The sub-spec's small per-lane
// pool (LaneK=6) forces its top candidates to live on different partitions under P=4
// (ids are partition-striped by PartitionOf), so the partition-local fuse on master is
// provably wrong — this test is RED before the fix and GREEN after.
func TestQueryFanOutNestedMultiLaneFusionMatchesP1(t *testing.T) {
	const n, k = 200, 10
	ctx := context.Background()
	parentDense := []float32{0.5, 0, 0, 0}
	subDense := []float32{12.3, 0, 0, 0} // a different anchor → its top ids differ from the parent's
	sIdx := []uint32{3}
	sVal := []float32{1}

	// The nested sub-spec: a 2-lane (dense+sparse) FUSION with a SMALL per-lane pool so
	// its globally-best candidates span partitions under P=4.
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

	s1 := newSingleEmbedded(t)
	waitLeaderEmbedded(t, s1)
	seedQueryCollection(t, s1, "qm1", 1, n)
	got1, _, err := s1.(*embedded).VectorQuery(ctx, "qm1", specBytes, spec, ReadOpts{})
	if err != nil {
		t.Fatalf("P1 nested multi-lane query: %v", err)
	}
	if len(got1) == 0 {
		t.Fatal("P1 nested multi-lane query returned no results")
	}

	const P = 4
	sP := newSingleEmbedded(t)
	waitLeaderEmbedded(t, sP)
	seedQueryCollection(t, sP, "qm4", P, n)
	touched := map[int]bool{}
	for id := uint64(1); id <= uint64(n); id++ {
		touched[ops.PartitionOf(id, P)] = true
	}
	if len(touched) != P {
		t.Fatalf("ids only touch %d/%d partitions", len(touched), P)
	}
	gotP, meta, err := sP.(*embedded).VectorQuery(ctx, "qm4", specBytes, spec, ReadOpts{})
	if err != nil {
		t.Fatalf("P4 nested multi-lane query: %v", err)
	}
	if meta.Degraded {
		t.Fatalf("P4 nested multi-lane query unexpectedly degraded: %+v", meta)
	}
	a, b := queryResultKeys(gotP), queryResultKeys(got1)
	if !eqHybridKeys(a, b) {
		t.Fatalf("nested MULTI-lane FUSION depth-2 P4 != P1:\n P4=%v\n P1=%v", a, b)
	}
}

// TestQueryFanOutNestedMultiLaneFusionDepth3MatchesP1 extends the headline to DEPTH-3:
// a parent FUSION whose prefetch is [a dense leaf, a nested FUSION whose prefetch is [a
// dense leaf, a nested 2-lane FUSION]] — a multi-lane FUSION node at TWO levels. The
// coordinator must fold BOTH nested FUSION nodes over the global union (the recursion is
// bottom-up). P4==P1 exact with candidates spanning partitions.
func TestQueryFanOutNestedMultiLaneFusionDepth3MatchesP1(t *testing.T) {
	const n, k = 200, 10
	ctx := context.Background()
	parentDense := []float32{0.5, 0, 0, 0}
	midDense := []float32{30.0, 0, 0, 0}
	leafDense := []float32{12.3, 0, 0, 0}
	sIdx := []uint32{3}
	sVal := []float32{1}

	innerSub := &pb.QuerySpec{
		Mode: pb.QueryMode_QUERY_MODE_FUSION,
		PrefetchSources: []*pb.QuerySource{
			{Source: &pb.QuerySource_Leaf{Leaf: denseLeafLaneK(leafDense, k, 6)}},
			{Source: &pb.QuerySource_Leaf{Leaf: sparseLeafLaneK(sIdx, sVal, k, 6)}},
		},
		FusionMethod: "rrf",
		K:            k,
	}
	midSub := &pb.QuerySpec{
		Mode: pb.QueryMode_QUERY_MODE_FUSION,
		PrefetchSources: []*pb.QuerySource{
			{Source: &pb.QuerySource_Leaf{Leaf: denseLeafLaneK(midDense, k, 6)}},
			{Source: &pb.QuerySource_Spec{Spec: innerSub}},
		},
		FusionMethod: "rrf",
		K:            k,
	}
	pspec := &pb.QuerySpec{
		Mode: pb.QueryMode_QUERY_MODE_FUSION,
		PrefetchSources: []*pb.QuerySource{
			{Source: &pb.QuerySource_Leaf{Leaf: denseLeaf(parentDense, k)}},
			{Source: &pb.QuerySource_Spec{Spec: midSub}},
		},
		FusionMethod: "rrf",
		K:            k,
	}
	specBytes, spec := buildQuerySpec(t, pspec)
	if !vector.SpecHasNestedFusion(spec) {
		t.Fatal("depth-3 spec should have a nested multi-lane FUSION node")
	}

	s1 := newSingleEmbedded(t)
	waitLeaderEmbedded(t, s1)
	seedQueryCollection(t, s1, "qd1", 1, n)
	got1, _, err := s1.(*embedded).VectorQuery(ctx, "qd1", specBytes, spec, ReadOpts{})
	if err != nil {
		t.Fatalf("P1 depth-3 query: %v", err)
	}
	if len(got1) == 0 {
		t.Fatal("P1 depth-3 query returned no results")
	}

	const P = 4
	sP := newSingleEmbedded(t)
	waitLeaderEmbedded(t, sP)
	seedQueryCollection(t, sP, "qd4", P, n)
	gotP, meta, err := sP.(*embedded).VectorQuery(ctx, "qd4", specBytes, spec, ReadOpts{})
	if err != nil {
		t.Fatalf("P4 depth-3 query: %v", err)
	}
	if meta.Degraded {
		t.Fatalf("P4 depth-3 query unexpectedly degraded: %+v", meta)
	}
	a, b := queryResultKeys(gotP), queryResultKeys(got1)
	if !eqHybridKeys(a, b) {
		t.Fatalf("nested MULTI-lane FUSION depth-3 P4 != P1:\n P4=%v\n P1=%v", a, b)
	}
}

// TestQueryFanOutNestedFusionOfFusionRerankMatchesP1 is the MIXED tree: a parent FUSION
// whose prefetch is [a nested MULTI-lane FUSION, a nested RERANK]. The coordinator must
// EXPAND the FUSION child (fold over the global union) while shipping the RERANK child as
// ONE partition-exact fused lane. P4==P1 exact.
func TestQueryFanOutNestedFusionOfFusionRerankMatchesP1(t *testing.T) {
	const n, k = 200, 10
	ctx := context.Background()
	denseQ := []float32{0.5, 0, 0, 0}
	subDense := []float32{12.3, 0, 0, 0}
	sIdx := []uint32{3}
	sVal := []float32{1}

	fusionSub := &pb.QuerySpec{
		Mode: pb.QueryMode_QUERY_MODE_FUSION,
		PrefetchSources: []*pb.QuerySource{
			{Source: &pb.QuerySource_Leaf{Leaf: denseLeafLaneK(subDense, k, 6)}},
			{Source: &pb.QuerySource_Leaf{Leaf: sparseLeafLaneK(sIdx, sVal, k, 6)}},
		},
		FusionMethod: "rrf",
		K:            k,
	}
	rerankSub := &pb.QuerySpec{
		Mode: pb.QueryMode_QUERY_MODE_RERANK,
		Root: denseLeaf(denseQ, k),
		PrefetchSources: []*pb.QuerySource{
			{Source: &pb.QuerySource_Leaf{Leaf: denseLeaf(denseQ, 50)}},
			{Source: &pb.QuerySource_Leaf{Leaf: sparseLeaf(sIdx, sVal, 50)}},
		},
		K: k,
	}
	pspec := &pb.QuerySpec{
		Mode: pb.QueryMode_QUERY_MODE_FUSION,
		PrefetchSources: []*pb.QuerySource{
			{Source: &pb.QuerySource_Spec{Spec: fusionSub}},
			{Source: &pb.QuerySource_Spec{Spec: rerankSub}},
		},
		FusionMethod: "rrf",
		K:            k,
	}
	specBytes, spec := buildQuerySpec(t, pspec)
	if !vector.SpecHasNestedFusion(spec) {
		t.Fatal("mixed spec should have a nested multi-lane FUSION node")
	}

	s1 := newSingleEmbedded(t)
	waitLeaderEmbedded(t, s1)
	seedQueryCollection(t, s1, "qx1", 1, n)
	got1, _, err := s1.(*embedded).VectorQuery(ctx, "qx1", specBytes, spec, ReadOpts{})
	if err != nil {
		t.Fatalf("P1 mixed query: %v", err)
	}
	if len(got1) == 0 {
		t.Fatal("P1 mixed query returned no results")
	}

	const P = 4
	sP := newSingleEmbedded(t)
	waitLeaderEmbedded(t, sP)
	seedQueryCollection(t, sP, "qx4", P, n)
	gotP, meta, err := sP.(*embedded).VectorQuery(ctx, "qx4", specBytes, spec, ReadOpts{})
	if err != nil {
		t.Fatalf("P4 mixed query: %v", err)
	}
	if meta.Degraded {
		t.Fatalf("P4 mixed query unexpectedly degraded: %+v", meta)
	}
	a, b := queryResultKeys(gotP), queryResultKeys(got1)
	if !eqHybridKeys(a, b) {
		t.Fatalf("mixed FUSION-of-(FUSION,RERANK) P4 != P1:\n P4=%v\n P1=%v", a, b)
	}
}

// TestTreeFusionCursorExact is a structural unit test for the emit/consume predicate
// agreement: it constructs synthetic per-partition QueryResult lane lists whose lane
// count EXACTLY matches the number the consume-side (mergeTreeFusionNode) expects for
// the given spec shape, calls treeFusionMergeFanOutCursor, and asserts the cursor
// advanced to EXACTLY len(lanes). Any divergence between the emit predicate
// (collectTreeLanesAt) and the consume predicate (mergeTreeFusionNode) changes the
// expected lane count — the cursor assertion goes RED regardless of whether the P4==P1
// result comparison would catch it (e.g. the >=1 over-expand on a single-lane child is
// invisible to P4==P1 but changes the expected cursor by +1 for the over-expanded lane).
//
// Spec shape: parent [dense_leaf, multi_sub(2 lanes→expanded), single_sub(1 lane→NOT expanded)]
// Emit produces 4 lanes per partition: 1 (dense) + 2 (multi expanded) + 1 (single fused) = 4.
// Consume must advance the cursor to exactly 4.
//
// The >=3 under-expand mutation: multi_sub (2 lanes) is NOT expanded on the emit side →
// emits 1 fused lane for multi_sub → 3 total lanes. The consume side (>=2) tries to
// expand multi_sub → reads 2 lanes → cursor desync → this test goes RED because
// expectedLanes stays 4 but emit only produced 3, causing parts[p].Lanes to be short.
//
// The >=1 over-expand mutation: single_sub (1 lane) IS expanded on the emit side →
// emits 1 raw lane for single_sub → still 4 total lanes (1+2+1). Consume side (>=2)
// does NOT expand it → reads 1 lane from cursor → cursor still advances to 4.
// P4==P1 is preserved, but if the single_sub leaf content differed per-partition,
// the cursor position would still be 4 — this assertion catches over-expand only when
// the expansion changes the lane count. Use an explicit expectedCursor to make the
// boundary explicit.
func TestTreeFusionCursorExact(t *testing.T) {
	// Build a spec matching: parent [dense_leaf, multiSub(2 lanes), singleSub(1 lane)].
	// This is the SAME shape as TestQueryFanOutNestedSingleUnderMultiBoundaryMatchesP1.
	sIdx := []uint32{3}
	sVal := []float32{1}
	multiSub := &pb.QuerySpec{
		Mode: pb.QueryMode_QUERY_MODE_FUSION,
		PrefetchSources: []*pb.QuerySource{
			{Source: &pb.QuerySource_Leaf{Leaf: denseLeafLaneK([]float32{1, 0, 0, 0}, 5, 6)}},
			{Source: &pb.QuerySource_Leaf{Leaf: sparseLeafLaneK(sIdx, sVal, 5, 6)}},
		},
		FusionMethod: "rrf",
		K:            5,
	}
	singleSub := &pb.QuerySpec{
		Mode: pb.QueryMode_QUERY_MODE_FUSION,
		PrefetchSources: []*pb.QuerySource{
			{Source: &pb.QuerySource_Leaf{Leaf: denseLeaf([]float32{2, 0, 0, 0}, 5)}},
		},
		FusionMethod: "rrf",
		K:            5,
	}
	pspec := &pb.QuerySpec{
		Mode: pb.QueryMode_QUERY_MODE_FUSION,
		PrefetchSources: []*pb.QuerySource{
			{Source: &pb.QuerySource_Leaf{Leaf: denseLeaf([]float32{0.5, 0, 0, 0}, 5)}},
			{Source: &pb.QuerySource_Spec{Spec: multiSub}},
			{Source: &pb.QuerySource_Spec{Spec: singleSub}},
		},
		FusionMethod: "rrf",
		K:            5,
	}
	_, spec := buildQuerySpec(t, pspec)

	// Emit produces 4 lanes per partition for this spec shape:
	//   lane 0: dense_leaf (leaf → 1 lane)
	//   lane 1: multiSub.Prefetch[0] dense (expanded: 2 lanes)
	//   lane 2: multiSub.Prefetch[1] sparse
	//   lane 3: singleSub fused result (NOT expanded: 1 fused lane)
	const expectedCursor = 4

	// Synthetic per-partition lane lists: 2 partitions × 4 lanes each, distinct ids.
	r := func(id uint64, score float32) vector.Result { return vector.Result{ID: id, Score: score} }
	parts := []vector.QueryResult{
		{Lanes: [][]vector.Result{
			{r(1, 0.9), r(5, 0.7)},  // lane 0: dense leaf P0
			{r(1, 0.8), r(5, 0.6)},  // lane 1: multiSub dense P0
			{r(1, 0.5)},             // lane 2: multiSub sparse P0
			{r(1, 0.95), r(5, 0.3)}, // lane 3: singleSub fused P0
		}},
		{Lanes: [][]vector.Result{
			{r(2, 0.85), r(6, 0.65)}, // lane 0: dense leaf P1
			{r(2, 0.75), r(6, 0.55)}, // lane 1: multiSub dense P1
			{r(2, 0.45)},             // lane 2: multiSub sparse P1
			{r(2, 0.90), r(6, 0.25)}, // lane 3: singleSub fused P1
		}},
	}

	_, cursor, err := treeFusionMergeFanOutCursor(parts, spec, 5)
	if err != nil {
		t.Fatalf("treeFusionMergeFanOutCursor: %v", err)
	}
	if cursor != expectedCursor {
		t.Fatalf("cursor desync: consumed %d lanes, expected %d\n"+
			"  A cursor < expectedCursor means the consume side under-consumed (>=3 under-expand mutation);\n"+
			"  a cursor > expectedCursor means the consume side over-consumed (a lane was read twice).",
			cursor, expectedCursor)
	}
}

// TestQueryFanOutNestedSingleUnderMultiBoundaryMatchesP1 pins the emit/consume
// expand-predicate BOUNDARY at len(Prefetch)==2: a parent multi-lane FUSION whose
// children are [dense leaf, nested MULTI-lane FUSION (>=2 lanes, expanded), nested
// SINGLE-lane FUSION (1 lane, NOT expanded)]. The >=3 UNDER-expand mutation of the
// emit predicate (collectTreeLanesAt no longer expanding the 2-lane multi child)
// produces a FUSED single lane where the consume side (mergeTreeFusionNode, still
// >=2) expects 2 raw lanes — cursor desync → P4 diverges from P1, test goes RED.
//
// NOTE: the >=1 OVER-expand mutation on the single-lane child is a NO-OP for P4==P1
// equality: emit expands the 1-lane child into 1 raw lane; consume (>=2 unchanged)
// does NOT expand it but still advances the cursor by 1 for the fused lane, so the
// lane counts and cursor positions are identical. The structural cursor-exact guard
// (TestTreeFusionCursorExact) catches ANY emit/consume predicate divergence directly
// by asserting the cursor advances to exactly the expected lane count regardless of
// the P4==P1 result comparison.
func TestQueryFanOutNestedSingleUnderMultiBoundaryMatchesP1(t *testing.T) {
	const n, k = 200, 10
	ctx := context.Background()
	parentDense := []float32{0.5, 0, 0, 0}
	subDense := []float32{12.3, 0, 0, 0}
	singleDense := []float32{30.0, 0, 0, 0}
	sIdx := []uint32{3}
	sVal := []float32{1}

	// A nested MULTI-lane (2-lane) FUSION child: expanded by both emit and consume.
	multiSub := &pb.QuerySpec{
		Mode: pb.QueryMode_QUERY_MODE_FUSION,
		PrefetchSources: []*pb.QuerySource{
			{Source: &pb.QuerySource_Leaf{Leaf: denseLeafLaneK(subDense, k, 6)}},
			{Source: &pb.QuerySource_Leaf{Leaf: sparseLeafLaneK(sIdx, sVal, k, 6)}},
		},
		FusionMethod: "rrf",
		K:            k,
	}
	// A nested SINGLE-lane FUSION child: NOT expanded (len(Prefetch)==1 < 2).
	// Emit and consume must AGREE on not expanding it; a >=1 mutation on emit
	// would expand it, producing an extra lane the consume side does not consume.
	singleSub := &pb.QuerySpec{
		Mode: pb.QueryMode_QUERY_MODE_FUSION,
		PrefetchSources: []*pb.QuerySource{
			{Source: &pb.QuerySource_Leaf{Leaf: denseLeaf(singleDense, k)}},
		},
		FusionMethod: "rrf",
		K:            k,
	}
	// Parent: [dense leaf, nested multi-lane FUSION, nested single-lane FUSION].
	pspec := &pb.QuerySpec{
		Mode: pb.QueryMode_QUERY_MODE_FUSION,
		PrefetchSources: []*pb.QuerySource{
			{Source: &pb.QuerySource_Leaf{Leaf: denseLeaf(parentDense, k)}},
			{Source: &pb.QuerySource_Spec{Spec: multiSub}},
			{Source: &pb.QuerySource_Spec{Spec: singleSub}},
		},
		FusionMethod: "rrf",
		K:            k,
	}
	specBytes, spec := buildQuerySpec(t, pspec)
	if !vector.SpecHasNestedFusion(spec) {
		t.Fatal("spec should have a nested multi-lane FUSION node")
	}

	s1 := newSingleEmbedded(t)
	waitLeaderEmbedded(t, s1)
	seedQueryCollection(t, s1, "qb1", 1, n)
	got1, _, err := s1.(*embedded).VectorQuery(ctx, "qb1", specBytes, spec, ReadOpts{})
	if err != nil {
		t.Fatalf("P1 boundary query: %v", err)
	}
	if len(got1) == 0 {
		t.Fatal("P1 boundary query returned no results")
	}

	const P = 4
	sP := newSingleEmbedded(t)
	waitLeaderEmbedded(t, sP)
	seedQueryCollection(t, sP, "qb4", P, n)
	touched := map[int]bool{}
	for id := uint64(1); id <= uint64(n); id++ {
		touched[ops.PartitionOf(id, P)] = true
	}
	if len(touched) != P {
		t.Fatalf("ids only touch %d/%d partitions", len(touched), P)
	}
	gotP, meta, err := sP.(*embedded).VectorQuery(ctx, "qb4", specBytes, spec, ReadOpts{})
	if err != nil {
		t.Fatalf("P4 boundary query: %v", err)
	}
	if meta.Degraded {
		t.Fatalf("P4 boundary query unexpectedly degraded: %+v", meta)
	}
	a, b := queryResultKeys(gotP), queryResultKeys(got1)
	if !eqHybridKeys(a, b) {
		t.Fatalf("single-under-multi boundary P4 != P1:\n P4=%v\n P1=%v", a, b)
	}
}

// TestQueryFanOutNestedFusionWithRecommendMatchesP1 is the recommend+discover-in-tree
// coverage test: a parent FUSION spec with a nested MULTI-lane FUSION sub-spec that
// contains a RECOMMEND leaf (AVERAGE_VECTOR, which is rewritten to a dense leaf by the
// coordinator pre-pass before fan-out). Verifies the coordinator-side over-fetch
// (wantK+len(exclude)) and post-fold prune in QueryTreeLanes prevent under-fill AND
// that P4==P1 holds with candidates spanning partitions. Uses the standard L2 dense
// collection (same as other nested-fusion tests) so a stable seeded corpus is used.
func TestQueryFanOutNestedFusionWithRecommendMatchesP1(t *testing.T) {
	const n, k = 200, 5
	ctx := context.Background()
	const coll1, collP = "qrec1", "qrec4"
	const P = 4

	s1 := newSingleEmbedded(t)
	waitLeaderEmbedded(t, s1)
	seedQueryCollection(t, s1, coll1, 1, n)

	sP := newSingleEmbedded(t)
	waitLeaderEmbedded(t, sP)
	seedQueryCollection(t, sP, collP, P, n)

	// Positive examples that span partitions (PartitionOf(3,4)=3, PartitionOf(7,4)=3,
	// PartitionOf(5,4)=1 — exclude set = {3,5,7}).
	positive := []uint64{3, 5, 7}
	sIdx := []uint32{3}
	sVal := []float32{1}

	// The nested sub-spec: a 2-lane FUSION whose lane 0 is a RECOMMEND leaf and lane 1
	// is a sparse leaf. The recommend leaf is rewritten to dense by the pre-pass.
	// Use dbsf (not rrf) so the per-candidate score is a continuous [0,1] value from
	// each lane's min/max — no two candidates share the same dbsf score, avoiding
	// tie-breaking divergence between P1 and P4 (which would produce the same set but
	// in an arbitrary stable-sort order that differs by partition append order).
	subSpec := &pb.QuerySpec{
		Mode: pb.QueryMode_QUERY_MODE_FUSION,
		PrefetchSources: []*pb.QuerySource{
			{Source: &pb.QuerySource_Leaf{Leaf: recommendLeaf(positive, nil, k+len(positive))}},
			{Source: &pb.QuerySource_Leaf{Leaf: sparseLeafLaneK(sIdx, sVal, k, 6)}},
		},
		FusionMethod: "dbsf",
		Alpha:        0.5,
		K:            k,
	}
	// Parent: [dense leaf, nested FUSION(recommend+sparse)]. dbsf here too for the
	// same tie-avoidance reason.
	pspec := &pb.QuerySpec{
		Mode: pb.QueryMode_QUERY_MODE_FUSION,
		PrefetchSources: []*pb.QuerySource{
			{Source: &pb.QuerySource_Leaf{Leaf: denseLeaf([]float32{0.5, 0.5, 0, 0}, k)}},
			{Source: &pb.QuerySource_Spec{Spec: subSpec}},
		},
		FusionMethod: "dbsf",
		Alpha:        0.5,
		K:            k,
	}
	specBytes, spec := buildQuerySpec(t, pspec)
	if !vector.SpecHasNestedFusion(spec) {
		t.Fatal("recommend spec should have a nested multi-lane FUSION node")
	}

	got1, _, err := s1.(*embedded).VectorQuery(ctx, coll1, specBytes, spec, ReadOpts{})
	if err != nil {
		t.Fatalf("P1 recommend query: %v", err)
	}
	if len(got1) == 0 {
		t.Fatal("P1 recommend query returned no results")
	}
	// Confirm none of the positive examples appear in the result (recommend exclusion).
	for _, r := range got1 {
		for _, pos := range positive {
			if r.ID == pos {
				t.Fatalf("P1 result contains positive example id %d (should be excluded)", pos)
			}
		}
	}

	// Re-decode specBytes into a fresh engine spec for the P4 call: the P1
	// VectorQuery mutated the shared sub-spec pointer (resolveRecommendForFanOut
	// rewrites recommend→dense in-place on a struct reachable via spec.Prefetch[i].Spec)
	// so the test's spec copy would already show a dense leaf, causing SpecHasRecommendLeaves
	// to return false and skip the coordinator rewrite for the P4 path.  Decoding fresh
	// from the immutable specBytes gives a new pointer tree with the original recommend leaf.
	_, specP := buildQuerySpec(t, pspec)
	gotP, meta, err := sP.(*embedded).VectorQuery(ctx, collP, specBytes, specP, ReadOpts{})
	if err != nil {
		t.Fatalf("P4 recommend query: %v", err)
	}
	if meta.Degraded {
		t.Fatalf("P4 recommend query unexpectedly degraded: %+v", meta)
	}
	a, b := queryResultKeys(gotP), queryResultKeys(got1)
	if !eqHybridKeys(a, b) {
		t.Fatalf("nested-FUSION-with-recommend P4 != P1:\n P4=%v\n P1=%v", a, b)
	}
}

// TestQueryFanOutNestedFusionDegradedPartition verifies the degraded-partition path
// through treeFusionMergeFanOut: dropping one physical partition makes the fan-out
// return Degraded=true with a sane (non-empty, no panic) partial result. The
// *cursor < len(parts[p].Lanes) guard in mergeTreeFusionNode skips the missing
// partition's lanes — there is no desync or panic, just fewer candidates.
func TestQueryFanOutNestedFusionDegradedPartition(t *testing.T) {
	const n, k = 80, 10
	ctx := context.Background()
	subDense := []float32{12.3, 0, 0, 0}
	sIdx := []uint32{3}
	sVal := []float32{1}

	multiSub := &pb.QuerySpec{
		Mode: pb.QueryMode_QUERY_MODE_FUSION,
		PrefetchSources: []*pb.QuerySource{
			{Source: &pb.QuerySource_Leaf{Leaf: denseLeafLaneK(subDense, k, 6)}},
			{Source: &pb.QuerySource_Leaf{Leaf: sparseLeafLaneK(sIdx, sVal, k, 6)}},
		},
		FusionMethod: "rrf",
		K:            k,
	}
	pspec := &pb.QuerySpec{
		Mode: pb.QueryMode_QUERY_MODE_FUSION,
		PrefetchSources: []*pb.QuerySource{
			{Source: &pb.QuerySource_Leaf{Leaf: denseLeaf([]float32{0.5, 0, 0, 0}, k)}},
			{Source: &pb.QuerySource_Spec{Spec: multiSub}},
		},
		FusionMethod: "rrf",
		K:            k,
	}
	specBytes, spec := buildQuerySpec(t, pspec)
	if !vector.SpecHasNestedFusion(spec) {
		t.Fatal("spec should have a nested multi-lane FUSION node")
	}

	const P = 4
	s := newSingleEmbedded(t)
	waitLeaderEmbedded(t, s)
	seedQueryCollection(t, s, "qdeg", P, n)
	emb := s.(*embedded)

	// Sanity: healthy query succeeds, not degraded.
	healthy, meta, err := emb.VectorQuery(ctx, "qdeg", specBytes, spec, ReadOpts{})
	if err != nil || meta.Degraded {
		t.Fatalf("healthy nested-fusion query: err=%v degraded=%v", err, meta.Degraded)
	}
	if len(healthy) == 0 {
		t.Fatal("healthy nested-fusion query returned no results")
	}

	// Drop partition 1 (gen 0) to simulate a missing partition.
	if _, err := emb.Call(ctx, "vector_drop_collection",
		ops.EncodeDropCollectionArgs(string(ops.PartitionKeyGen("qdeg", 0, 1)))); err != nil {
		t.Fatalf("drop partition 1: %v", err)
	}

	// Fail mode → error (treeFusionMergeFanOut path should propagate the fan-out error).
	if _, _, err := emb.VectorQuery(ctx, "qdeg", specBytes, spec, ReadOpts{OnPartitionUnavailable: 1}); err == nil {
		t.Fatal("expected error with OnPartitionUnavailable=Fail and a missing partition")
	}

	// Partial mode (default) → Degraded=true, no panic, some results from remaining partitions.
	res, meta, err := emb.VectorQuery(ctx, "qdeg", specBytes, spec, ReadOpts{})
	if err != nil {
		t.Fatalf("partial-mode nested-fusion query errored: %v", err)
	}
	if !meta.Degraded {
		t.Fatal("expected Degraded=true after a partition was made unreachable")
	}
	if len(meta.Missing) == 0 {
		t.Fatal("expected non-empty Missing after a partition was made unreachable")
	}
	if len(res) == 0 {
		t.Fatal("expected partial results from the reachable partitions")
	}
}
