// SPDX-License-Identifier: Apache-2.0

package rostam

import (
	"context"
	"sort"
	"testing"

	"github.com/rostamlabs/rostam/sdk/pb"
	"github.com/rostamlabs/rostam/ops"
	"github.com/rostamlabs/rostam/vector"
)

// seedMVQueryCollection creates a P-partition (or P==1) MV collection (dim 4) and
// adds ids 1..n each with a deterministic token matrix (the MaxSim lane) AND a
// deterministic doc-level sparse vector (the doc-sparse lane), plus a shared payload
// {"id": id}. The Query-API MV analogue of seedMVHybridCollection /
// seedNamedQueryCollection: every doc populates BOTH MV modalities so each prefetch
// lane can score every doc.
func seedMVQueryCollection(t *testing.T, coll string, P, n int) Store {
	t.Helper()
	s := newSingleEmbedded(t)
	waitLeaderEmbedded(t, s)
	ctx := context.Background()
	if err := s.VectorMVCreateCollection(ctx, coll, MultiVectorConfig{Dim: 4, Partitions: P}); err != nil {
		t.Fatalf("create mv query collection %q (P=%d): %v", coll, P, err)
	}
	for id := uint64(1); id <= uint64(n); id++ {
		tokens := [][]float32{
			{float32(id)*0.01 + 0.1, float32(id%5)*0.2 + 0.05, float32(id%7)*0.13 + 0.02, float32(id%3)*0.31 + 0.07},
			{float32(id%4)*0.17 + 0.03, float32(id)*0.005 + 0.2, float32(id%9)*0.11 + 0.01, float32(id%6)*0.23 + 0.04},
		}
		idx := []uint32{uint32(id % 7), uint32(id%11) + 12, uint32(id%13) + 24}
		sort.Slice(idx, func(i, j int) bool { return idx[i] < idx[j] })
		uniq := idx[:1]
		for _, v := range idx[1:] {
			if v != uniq[len(uniq)-1] {
				uniq = append(uniq, v)
			}
		}
		sv := &vector.SparseVector{Indices: uniq, Values: make([]float32, len(uniq))}
		for i := range sv.Values {
			sv.Values[i] = float32(id)*0.013 + float32(uniq[i])*0.001 + float32(i)*0.1 + 1
		}
		meta := VectorMetadata{"id": vector.NewInt(int64(id))}
		if err := s.VectorMVAdd(ctx, coll, id, tokens, meta, WriteOpts{Sparse: sv}); err != nil {
			t.Fatalf("mv query add %d: %v", id, err)
		}
	}
	return s
}

// mvMaxSimLeaf builds an MV MaxSim pb.QueryLeaf (the MV oneof arm). tokens is the
// query token matrix; the MV engine scores docs by MaxSim (score-descending).
func mvMaxSimLeaf(tokens [][]float32, k int) *pb.QueryLeaf {
	tv := make([]*pb.TokenVector, len(tokens))
	for i, row := range tokens {
		tv[i] = &pb.TokenVector{Values: row}
	}
	return &pb.QueryLeaf{Leaf: &pb.QueryLeaf_MvMaxsim{MvMaxsim: &pb.MVMaxSimLeaf{
		Query: tv, K: int32(k),
	}}}
}

// TestMVQueryFanOutFusionMatchesP1 is the #1 MV-family exactness invariant: a FUSION
// query over an MV collection's MaxSim lane + doc-sparse lane over a P=4 collection
// returns the EXACT same fused top-k (id + score + distance, in order) as the same
// query over a P=1 collection — for RRF, weighted, AND dbsf. This proves mvQueryFanOut
// unions each lane + truncates per-lane (SCORE-descending, both MV lanes) + folds ONCE
// via FuseScoreLanes (lane0 ScoreDesc=true), reusing the orientation-aware
// fusionMergeFanOut verbatim. It goes RED if a partition pre-fuses, the per-lane
// truncation uses the wrong orientation, or the fold starts with dense Fuse.
func TestMVQueryFanOutFusionMatchesP1(t *testing.T) {
	const n, k = 200, 12
	ctx := context.Background()
	maxsimQ := [][]float32{{1.2, 0.6, 0.3, 0.4}, {0.2, 0.9, 0.1, 0.5}}
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
			pspec := &pb.QuerySpec{
				Mode: pb.QueryMode_QUERY_MODE_FUSION,
				Prefetch: []*pb.QueryLeaf{
					mvMaxSimLeaf(maxsimQ, k),
					sparseLeaf(sIdx, sVal, k),
				},
				FusionMethod: tc.method,
				Alpha:        tc.alpha,
				K:            int32(k),
			}
			specBytes, spec := buildQuerySpec(t, pspec)

			s1 := seedMVQueryCollection(t, "mq1_"+tc.name, 1, n)
			got1, _, err := s1.(*embedded).VectorMVQuery(ctx, "mq1_"+tc.name, specBytes, spec, ReadOpts{})
			if err != nil {
				t.Fatalf("P1 mv query: %v", err)
			}

			const P = 4
			sP := seedMVQueryCollection(t, "mq4_"+tc.name, P, n)
			touched := map[int]bool{}
			for id := uint64(1); id <= uint64(n); id++ {
				touched[ops.PartitionOf(id, P)] = true
			}
			if len(touched) != P {
				t.Fatalf("ids only touch %d/%d partitions", len(touched), P)
			}
			gotP, meta, err := sP.(*embedded).VectorMVQuery(ctx, "mq4_"+tc.name, specBytes, spec, ReadOpts{})
			if err != nil {
				t.Fatalf("P4 mv query: %v", err)
			}
			if meta.Degraded {
				t.Fatalf("P4 mv query unexpectedly degraded: %+v", meta)
			}
			if len(gotP) == 0 {
				t.Fatal("P4 mv query returned no results")
			}
			a, b := queryResultKeys(gotP), queryResultKeys(got1)
			if !eqHybridKeys(a, b) {
				t.Fatalf("FUSION P4 != P1:\n P4=%v\n P1=%v", a, b)
			}
		})
	}
}

// TestMVQueryFanOutRerankMatchesP1 proves RERANK partition-invariance for the MV
// family: a rerank query (prefetch a MaxSim lane + the doc-sparse lane → rerank by a
// MaxSim root) over P=4 returns the exact same reranked top-k as P=1. Exact because MV
// point ids are partition-disjoint (each doc is prefetched + reranked on its sole
// owning partition), so the coordinator's merge-sort of the per-partition reranked
// top-ks (by the root's score-descending orientation) == the single-partition rerank.
func TestMVQueryFanOutRerankMatchesP1(t *testing.T) {
	const n, k = 200, 10
	ctx := context.Background()
	rootQ := [][]float32{{1.2, 0.6, 0.3, 0.4}}
	maxsimQ := [][]float32{{0.2, 0.9, 0.1, 0.5}}
	sIdx := []uint32{1, 14, 26}
	sVal := []float32{2, 3, 1}

	// Prefetch pools >= n so BOTH P1 and P4 surface the full candidate set per lane;
	// the root then re-scores the identical global union. This isolates the
	// partition-invariance of rerankMergeFanOut from any prefetch-recall difference a
	// small pool would introduce (see the named-family rerank oracle).
	pspec := &pb.QuerySpec{
		Mode: pb.QueryMode_QUERY_MODE_RERANK,
		Root: mvMaxSimLeaf(rootQ, k),
		Prefetch: []*pb.QueryLeaf{
			mvMaxSimLeaf(maxsimQ, n),
			sparseLeaf(sIdx, sVal, n),
		},
		K: int32(k),
	}
	specBytes, spec := buildQuerySpec(t, pspec)

	s1 := seedMVQueryCollection(t, "mqr1", 1, n)
	got1, _, err := s1.(*embedded).VectorMVQuery(ctx, "mqr1", specBytes, spec, ReadOpts{})
	if err != nil {
		t.Fatalf("P1 mv rerank: %v", err)
	}

	const P = 4
	sP := seedMVQueryCollection(t, "mqr4", P, n)
	gotP, meta, err := sP.(*embedded).VectorMVQuery(ctx, "mqr4", specBytes, spec, ReadOpts{})
	if err != nil {
		t.Fatalf("P4 mv rerank: %v", err)
	}
	if meta.Degraded {
		t.Fatalf("P4 mv rerank unexpectedly degraded: %+v", meta)
	}
	if len(gotP) == 0 {
		t.Fatal("P4 mv rerank returned no results")
	}
	a, b := queryResultKeys(gotP), queryResultKeys(got1)
	if !eqHybridKeys(a, b) {
		t.Fatalf("RERANK P4 != P1:\n P4=%v\n P1=%v", a, b)
	}
}

// TestMVQueryFusionEqualsMVHybrid is the UNIFICATION cross-check: a 2-lane FUSION MV
// query (the MaxSim lane + the doc-sparse lane) via vector_mv_query returns the EXACT
// same fused top-k as the dedicated VectorMVHybridSearch over the same modalities.
// This proves the MV Query API path is a strict generalization of the existing MV
// hybrid (the MaxSim+sparse case collapses to it), and validates the orientation-aware
// fold (both lanes score-desc → FuseScoreLanes). Run for both P=1 and P=4 so the
// fan-out path is also cross-checked.
func TestMVQueryFusionEqualsMVHybrid(t *testing.T) {
	const n, k = 200, 12
	ctx := context.Background()
	maxsimQ := [][]float32{{1.2, 0.6, 0.3, 0.4}, {0.2, 0.9, 0.1, 0.5}}
	sIdx := []uint32{1, 14, 26}
	sVal := []float32{2, 3, 1}
	sparseQ := vector.SparseVector{Indices: sIdx, Values: sVal}

	for _, P := range []int{1, 4} {
		t.Run("P"+itoaTest(P), func(t *testing.T) {
			coll := "mqu" + itoaTest(P)
			s := seedMVQueryCollection(t, coll, P, n)

			// Hybrid oracle: MaxSim fused with doc-sparse via RRF.
			want, err := s.VectorMVHybridSearch(ctx, coll, maxsimQ, sparseQ, k,
				MVHybridOpts{Method: FusionRRF})
			if err != nil {
				t.Fatalf("VectorMVHybridSearch: %v", err)
			}

			// The same two modalities as a 2-lane FUSION MV query. The hybrid engine uses
			// denseK/sparseK = max(k,50); set the prefetch lane K to match so the
			// candidate lists fused are identical.
			pspec := &pb.QuerySpec{
				Mode: pb.QueryMode_QUERY_MODE_FUSION,
				Prefetch: []*pb.QueryLeaf{
					mvMaxSimLeaf(maxsimQ, 50),
					sparseLeaf(sIdx, sVal, 50),
				},
				FusionMethod: "rrf",
				K:            int32(k),
			}
			specBytes, spec := buildQuerySpec(t, pspec)
			got, _, err := s.(*embedded).VectorMVQuery(ctx, coll, specBytes, spec, ReadOpts{})
			if err != nil {
				t.Fatalf("VectorMVQuery: %v", err)
			}
			if !eqHybridKeys(queryResultKeys(got), hybridKeys(want)) {
				t.Fatalf("2-lane MV FUSION != MVHybrid (P=%d):\n query=%v\n hybrid=%v",
					P, queryResultKeys(got), hybridKeys(want))
			}
		})
	}
}

// TestMVQueryFanOutDegradationFailErrors mirrors the named-family degradation guard
// for the MV family: dropping one physical partition makes the fan-out fail. Partial
// mode (default) returns partial results flagged Degraded + the missing partition
// index; Fail mode errors.
func TestMVQueryFanOutDegradationFailErrors(t *testing.T) {
	const n, k = 80, 10
	ctx := context.Background()
	pspec := &pb.QuerySpec{
		Mode: pb.QueryMode_QUERY_MODE_FUSION,
		Prefetch: []*pb.QueryLeaf{
			mvMaxSimLeaf([][]float32{{1.2, 0.6, 0.3, 0.4}}, k),
			sparseLeaf([]uint32{1, 14, 26}, []float32{2, 3, 1}, k),
		},
		FusionMethod: "rrf",
		K:            int32(k),
	}
	specBytes, spec := buildQuerySpec(t, pspec)

	const P = 4
	s := seedMVQueryCollection(t, "mqd", P, n)
	emb := s.(*embedded)

	// Sanity: healthy query succeeds, not degraded.
	if _, meta, err := emb.VectorMVQuery(ctx, "mqd", specBytes, spec, ReadOpts{}); err != nil || meta.Degraded {
		t.Fatalf("healthy mv query: err=%v degraded=%v", err, meta.Degraded)
	}

	// Make partition 1 unreachable by dropping its physical collection (gen 0).
	if _, err := emb.Call(ctx, "vector_mv_drop_collection",
		ops.EncodeMVDeleteArgs(string(ops.PartitionKeyGen("mqd", 0, 1)), 0)); err != nil {
		t.Fatalf("drop partition 1: %v", err)
	}

	// Fail mode → error.
	if _, _, err := emb.VectorMVQuery(ctx, "mqd", specBytes, spec, ReadOpts{OnPartitionUnavailable: 1}); err == nil {
		t.Fatal("expected error with OnPartitionUnavailable=Fail and an unreachable partition")
	}
	// Partial mode (default) → degraded + partial (no error).
	res, meta, err := emb.VectorMVQuery(ctx, "mqd", specBytes, spec, ReadOpts{})
	if err != nil {
		t.Fatalf("partial-mode mv query errored: %v", err)
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

// TestMVQueryFanOutRcRidesEveryArg is the anti-silent-drop guard for the MV family:
// the exact per-partition encode line mvQueryFanOut's Encode closure uses must carry
// the Linearizable rc so ReadConsistencyOf("vector_mv_query", arg) arms each shard's
// data barrier. The bound also rides for a bounded query.
func TestMVQueryFanOutRcRidesEveryArg(t *testing.T) {
	pspec := &pb.QuerySpec{
		Mode: pb.QueryMode_QUERY_MODE_FUSION,
		Prefetch: []*pb.QueryLeaf{
			mvMaxSimLeaf([][]float32{{1, 0, 0, 0}}, 10),
			sparseLeaf([]uint32{1}, []float32{1}, 10),
		},
		K: 10,
	}
	specBytes, _ := buildQuerySpec(t, pspec)

	arg := ops.EncodeQueryArgs("phys#2", specBytes, ops.ConsistencyLinearizable, 0, 0)
	rc, ok := ops.ReadConsistencyOf("vector_mv_query", arg)
	if !ok {
		t.Fatal("ReadConsistencyOf(vector_mv_query) not covered — a Linearizable mv query would NOT arm the shard barrier")
	}
	if rc != ops.ConsistencyLinearizable {
		t.Fatalf("per-partition arg rc = %d, want Linearizable(%d) — rc dropped (silent-degrade)", rc, ops.ConsistencyLinearizable)
	}

	const bound = uint64(1234)
	bArg := ops.EncodeQueryArgs("phys#2", specBytes, ops.ConsistencyBoundedStaleness, 0, bound)
	gotBound, okB := ops.ReadStalenessOf("vector_mv_query", bArg)
	if !okB || gotBound != bound {
		t.Fatalf("ReadStalenessOf(vector_mv_query) = (%d,%v), want (%d,true)", gotBound, okB, bound)
	}
}

// TestDirectStoreMVQueryFusionNonEmpty is the directStore bug-class guard: a
// single-node Direct store MV FUSION query must route the mode-tagged result through
// fusionMergeFanOut, NOT read qr.Fused directly (which is nil for FUSION). A non-empty
// result proves the merge ran.
func TestDirectStoreMVQueryFusionNonEmpty(t *testing.T) {
	const n, k = 120, 10
	ctx := context.Background()
	maxsimQ := [][]float32{{1.2, 0.6, 0.3, 0.4}, {0.2, 0.9, 0.1, 0.5}}
	pspec := &pb.QuerySpec{
		Mode: pb.QueryMode_QUERY_MODE_FUSION,
		Prefetch: []*pb.QueryLeaf{
			mvMaxSimLeaf(maxsimQ, k),
			sparseLeaf([]uint32{1, 14, 26}, []float32{2, 3, 1}, k),
		},
		FusionMethod: "rrf",
		K:            int32(k),
	}
	specBytes, spec := buildQuerySpec(t, pspec)

	d := newSingleDirect(t)
	if err := d.VectorMVCreateCollection(ctx, "dmq", MultiVectorConfig{Dim: 4}); err != nil {
		t.Fatalf("direct create mv collection: %v", err)
	}
	for id := uint64(1); id <= uint64(n); id++ {
		tokens := [][]float32{
			{float32(id)*0.01 + 0.1, float32(id%5)*0.2 + 0.05, float32(id%7)*0.13 + 0.02, float32(id%3)*0.31 + 0.07},
			{float32(id%4)*0.17 + 0.03, float32(id)*0.005 + 0.2, float32(id%9)*0.11 + 0.01, float32(id%6)*0.23 + 0.04},
		}
		idx := []uint32{uint32(id % 7), uint32(id%11) + 12, uint32(id%13) + 24}
		sort.Slice(idx, func(i, j int) bool { return idx[i] < idx[j] })
		uniq := idx[:1]
		for _, v := range idx[1:] {
			if v != uniq[len(uniq)-1] {
				uniq = append(uniq, v)
			}
		}
		sv := &vector.SparseVector{Indices: uniq, Values: make([]float32, len(uniq))}
		for i := range sv.Values {
			sv.Values[i] = float32(id)*0.013 + float32(uniq[i])*0.001 + float32(i)*0.1 + 1
		}
		if err := d.VectorMVAdd(ctx, "dmq", id, tokens, VectorMetadata{"id": vector.NewInt(int64(id))}, WriteOpts{Sparse: sv}); err != nil {
			t.Fatalf("direct mv add %d: %v", id, err)
		}
	}

	got, _, err := d.VectorMVQuery(ctx, "dmq", specBytes, spec, ReadOpts{})
	if err != nil {
		t.Fatalf("directStore mv query: %v", err)
	}
	if len(got) == 0 {
		t.Fatal("directStore MV FUSION returned EMPTY — qr.Fused read directly instead of routed through the merge")
	}
}
