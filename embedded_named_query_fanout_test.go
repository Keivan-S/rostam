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

// seedNamedQueryCollection creates a P-partition (or P==1) NAMED collection with
// THREE spaces — two dense ("title" dim 4, "image" dim 3, both L2) and one sparse
// ("terms") — then inserts ids 1..n each populating all three spaces (so every
// prefetch lane can score every point), plus a shared payload {"id": id}. This is
// the named-family multi-space analogue of seedQueryCollection: a named collection
// has MANY spaces, so a FUSION query can fuse across N>2 lanes (the distinctive
// value v1 dense could not deliver).
func seedNamedQueryCollection(t *testing.T, coll string, P, n int) Store {
	t.Helper()
	s := newSingleEmbedded(t)
	waitLeaderEmbedded(t, s)
	ctx := context.Background()
	cfg := map[string]NamedVectorParams{
		"title": {Dim: 4, Metric: vector.L2},
		"image": {Dim: 3, Metric: vector.L2},
		"terms": {Sparse: true},
	}
	if err := s.VectorNamedCreateCollection(ctx, coll, cfg, P); err != nil {
		t.Fatalf("create named query collection %q (P=%d): %v", coll, P, err)
	}
	emb := s.(*embedded)
	for id := uint64(1); id <= uint64(n); id++ {
		title := []float32{
			float32(id)*0.01 + 0.1,
			float32(id%5)*0.2 + 0.05,
			float32(id%7)*0.13 + 0.02,
			float32(id%3)*0.31 + 0.07,
		}
		image := []float32{
			float32(id%9)*0.11 + 0.03,
			float32(id)*0.007 + 0.2,
			float32(id%4)*0.17 + 0.01,
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
		if err := emb.VectorNamedInsertSparse(ctx, coll, id,
			map[string][]float32{"title": title, "image": image},
			map[string]*vector.SparseVector{"terms": sv}, meta, 0); err != nil {
			t.Fatalf("named query insert %d: %v", id, err)
		}
	}
	return s
}

// namedDenseLeaf / namedSparseLeaf build a Space-bearing pb.QueryLeaf (the named
// family), so a named QuerySpec round-trips through the named oneof arms.
func namedDenseLeaf(space string, dense []float32, k int) *pb.QueryLeaf {
	return &pb.QueryLeaf{Leaf: &pb.QueryLeaf_NamedDense{NamedDense: &pb.NamedDenseLeaf{
		Space: space, Dense: dense, K: int32(k),
	}}}
}

func namedSparseLeaf(space string, idx []uint32, vals []float32, k int) *pb.QueryLeaf {
	return &pb.QueryLeaf{Leaf: &pb.QueryLeaf_NamedSparse{NamedSparse: &pb.NamedSparseLeaf{
		Space: space, Indices: idx, Values: vals, K: int32(k),
	}}}
}

// TestNamedQueryFanOutFusionMatchesP1 is the #1 named-family exactness invariant: a
// FUSION query over THREE named spaces (two dense + one sparse prefetch lanes) over
// a P=4 collection returns the EXACT same fused top-k (id + score + distance, in
// order) as the same query over a P=1 collection — for RRF, weighted, AND dbsf.
// This proves namedQueryFanOut unions each lane + truncates per-lane to the global
// K + folds the 3 lanes ONCE (no per-partition pre-fuse), reusing the v1
// fusionMergeFanOut verbatim. It goes RED if a partition pre-fuses or the per-lane
// truncation is wrong.
func TestNamedQueryFanOutFusionMatchesP1(t *testing.T) {
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
			pspec := &pb.QuerySpec{
				Mode: pb.QueryMode_QUERY_MODE_FUSION,
				Prefetch: []*pb.QueryLeaf{
					namedDenseLeaf("title", titleQ, k),
					namedDenseLeaf("image", imageQ, k),
					namedSparseLeaf("terms", sIdx, sVal, k),
				},
				FusionMethod: tc.method,
				Alpha:        tc.alpha,
				K:            int32(k),
			}
			specBytes, spec := buildQuerySpec(t, pspec)

			s1 := seedNamedQueryCollection(t, "nq1_"+tc.name, 1, n)
			got1, _, err := s1.(*embedded).VectorNamedQuery(ctx, "nq1_"+tc.name, specBytes, spec, ReadOpts{})
			if err != nil {
				t.Fatalf("P1 named query: %v", err)
			}

			const P = 4
			sP := seedNamedQueryCollection(t, "nq4_"+tc.name, P, n)
			touched := map[int]bool{}
			for id := uint64(1); id <= uint64(n); id++ {
				touched[ops.PartitionOf(id, P)] = true
			}
			if len(touched) != P {
				t.Fatalf("ids only touch %d/%d partitions", len(touched), P)
			}
			gotP, meta, err := sP.(*embedded).VectorNamedQuery(ctx, "nq4_"+tc.name, specBytes, spec, ReadOpts{})
			if err != nil {
				t.Fatalf("P4 named query: %v", err)
			}
			if meta.Degraded {
				t.Fatalf("P4 named query unexpectedly degraded: %+v", meta)
			}
			if len(gotP) == 0 {
				t.Fatal("P4 named query returned no results")
			}
			a, b := queryResultKeys(gotP), queryResultKeys(got1)
			if !eqHybridKeys(a, b) {
				t.Fatalf("FUSION P4 != P1:\n P4=%v\n P1=%v", a, b)
			}
		})
	}
}

// TestNamedQueryFanOutRerankMatchesP1 proves RERANK partition-invariance for the
// named family: a rerank query (prefetch the two dense + the sparse space → rerank
// by a dense root space) over P=4 returns the exact same reranked top-k as P=1.
// Exact because named point ids are partition-disjoint (each doc is prefetched +
// reranked on its sole owning partition regardless of space), so the coordinator's
// merge-sort of the per-partition reranked top-ks == the single-partition rerank.
func TestNamedQueryFanOutRerankMatchesP1(t *testing.T) {
	const n, k = 200, 10
	ctx := context.Background()
	titleQ := []float32{1.2, 0.6, 0.3, 0.4}
	imageQ := []float32{0.5, 0.5, 0.2}
	sIdx := []uint32{1, 14, 26}
	sVal := []float32{2, 3, 1}

	// Prefetch pools are set to >= n so BOTH P1 and P4 surface the full candidate
	// set per lane; the root (title) then re-scores the identical global union. This
	// isolates the partition-invariance of rerankMergeFanOut (merge per-partition
	// reranked top-Ks by the root's orientation, exact via id-disjointness) from the
	// prefetch-recall difference a small pool would introduce (a partition with ~n/P
	// docs surfaces nearly all its docs at any pool, so a small global pool would let
	// P4 rerank a strict superset of P1's candidates — not a fan-out bug).
	pspec := &pb.QuerySpec{
		Mode: pb.QueryMode_QUERY_MODE_RERANK,
		Root: namedDenseLeaf("title", titleQ, k),
		Prefetch: []*pb.QueryLeaf{
			namedDenseLeaf("image", imageQ, n),
			namedSparseLeaf("terms", sIdx, sVal, n),
		},
		K: int32(k),
	}
	specBytes, spec := buildQuerySpec(t, pspec)

	s1 := seedNamedQueryCollection(t, "nqr1", 1, n)
	got1, _, err := s1.(*embedded).VectorNamedQuery(ctx, "nqr1", specBytes, spec, ReadOpts{})
	if err != nil {
		t.Fatalf("P1 named rerank: %v", err)
	}

	const P = 4
	sP := seedNamedQueryCollection(t, "nqr4", P, n)
	gotP, meta, err := sP.(*embedded).VectorNamedQuery(ctx, "nqr4", specBytes, spec, ReadOpts{})
	if err != nil {
		t.Fatalf("P4 named rerank: %v", err)
	}
	if meta.Degraded {
		t.Fatalf("P4 named rerank unexpectedly degraded: %+v", meta)
	}
	if len(gotP) == 0 {
		t.Fatal("P4 named rerank returned no results")
	}
	a, b := queryResultKeys(gotP), queryResultKeys(got1)
	if !eqHybridKeys(a, b) {
		t.Fatalf("RERANK P4 != P1:\n P4=%v\n P1=%v", a, b)
	}
}

// TestNamedQueryFusionEqualsNamedHybrid is the UNIFICATION cross-check: a 2-space
// FUSION named query (one dense space + one sparse space) via vector_named_query
// returns the EXACT same fused top-k as the dedicated VectorNamedHybridSearch over
// the same two spaces. This proves the Query API named path is a strict
// generalization of the existing named hybrid (the dense+sparse case collapses to
// it). Run for both P=1 and P=4 so the fan-out path is also cross-checked.
func TestNamedQueryFusionEqualsNamedHybrid(t *testing.T) {
	const n, k = 200, 12
	ctx := context.Background()
	titleQ := []float32{1.2, 0.6, 0.3, 0.4}
	sIdx := []uint32{1, 14, 26}
	sVal := []float32{2, 3, 1}
	sparseQ := vector.SparseVector{Indices: sIdx, Values: sVal}

	for _, P := range []int{1, 4} {
		t.Run("P"+itoaTest(P), func(t *testing.T) {
			coll := "nqu" + itoaTest(P)
			s := seedNamedQueryCollection(t, coll, P, n)

			// Hybrid oracle: dense "title" fused with sparse "terms" via RRF.
			want, err := s.VectorNamedHybridSearch(ctx, coll, "title", titleQ, "terms", sparseQ, k,
				NamedHybridOpts{Method: FusionRRF})
			if err != nil {
				t.Fatalf("VectorNamedHybridSearch: %v", err)
			}

			// The same two spaces as a 2-lane FUSION named query. The hybrid engine
			// uses denseK/sparseK = max(k,50); set the prefetch lane K to match so the
			// candidate lists fused are identical.
			pspec := &pb.QuerySpec{
				Mode: pb.QueryMode_QUERY_MODE_FUSION,
				Prefetch: []*pb.QueryLeaf{
					namedDenseLeaf("title", titleQ, 50),
					namedSparseLeaf("terms", sIdx, sVal, 50),
				},
				FusionMethod: "rrf",
				K:            int32(k),
			}
			specBytes, spec := buildQuerySpec(t, pspec)
			got, _, err := s.(*embedded).VectorNamedQuery(ctx, coll, specBytes, spec, ReadOpts{})
			if err != nil {
				t.Fatalf("VectorNamedQuery: %v", err)
			}
			if !eqHybridKeys(queryResultKeys(got), hybridKeys(want)) {
				t.Fatalf("2-space named FUSION != NamedHybrid (P=%d):\n query=%v\n hybrid=%v",
					P, queryResultKeys(got), hybridKeys(want))
			}
		})
	}
}

// TestNamedQueryFanOutDegradationFailErrors mirrors TestQueryFanOutDegradation for
// the named family: dropping one physical partition makes the fan-out fail. Partial
// mode (default) returns partial results flagged Degraded + the missing partition
// index; Fail mode errors.
func TestNamedQueryFanOutDegradationFailErrors(t *testing.T) {
	const n, k = 80, 10
	ctx := context.Background()
	pspec := &pb.QuerySpec{
		Mode: pb.QueryMode_QUERY_MODE_FUSION,
		Prefetch: []*pb.QueryLeaf{
			namedDenseLeaf("title", []float32{1.2, 0.6, 0.3, 0.4}, k),
			namedDenseLeaf("image", []float32{0.5, 0.5, 0.2}, k),
			namedSparseLeaf("terms", []uint32{1, 14, 26}, []float32{2, 3, 1}, k),
		},
		FusionMethod: "rrf",
		K:            int32(k),
	}
	specBytes, spec := buildQuerySpec(t, pspec)

	const P = 4
	s := seedNamedQueryCollection(t, "nqd", P, n)
	emb := s.(*embedded)

	// Sanity: healthy query succeeds, not degraded.
	if _, meta, err := emb.VectorNamedQuery(ctx, "nqd", specBytes, spec, ReadOpts{}); err != nil || meta.Degraded {
		t.Fatalf("healthy named query: err=%v degraded=%v", err, meta.Degraded)
	}

	// Make partition 1 unreachable by dropping its physical collection (gen 0).
	if _, err := emb.Call(ctx, "vector_named_drop_collection",
		ops.EncodeNamedNameArgs(string(ops.PartitionKeyGen("nqd", 0, 1)))); err != nil {
		t.Fatalf("drop partition 1: %v", err)
	}

	// Fail mode → error.
	if _, _, err := emb.VectorNamedQuery(ctx, "nqd", specBytes, spec, ReadOpts{OnPartitionUnavailable: 1}); err == nil {
		t.Fatal("expected error with OnPartitionUnavailable=Fail and an unreachable partition")
	}
	// Partial mode (default) → degraded + partial (no error).
	res, meta, err := emb.VectorNamedQuery(ctx, "nqd", specBytes, spec, ReadOpts{})
	if err != nil {
		t.Fatalf("partial-mode named query errored: %v", err)
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

// TestNamedQueryFanOutRcRidesEveryArg is the anti-silent-drop guard for the named
// family: the exact per-partition encode line namedQueryFanOut's Encode closure
// uses must carry the Linearizable rc so ReadConsistencyOf("vector_named_query",
// arg) arms each shard's data barrier. Identical to the v1 guard but for the named
// op (which shares the EXACT same arg wire). The bound also rides for a bounded
// query.
func TestNamedQueryFanOutRcRidesEveryArg(t *testing.T) {
	pspec := &pb.QuerySpec{
		Mode: pb.QueryMode_QUERY_MODE_FUSION,
		Prefetch: []*pb.QueryLeaf{
			namedDenseLeaf("title", []float32{1, 0, 0, 0}, 10),
			namedSparseLeaf("terms", []uint32{1}, []float32{1}, 10),
		},
		K: 10,
	}
	specBytes, _ := buildQuerySpec(t, pspec)

	arg := ops.EncodeQueryArgs("phys#2", specBytes, ops.ConsistencyLinearizable, 0, 0)
	rc, ok := ops.ReadConsistencyOf("vector_named_query", arg)
	if !ok {
		t.Fatal("ReadConsistencyOf(vector_named_query) not covered — a Linearizable named query would NOT arm the shard barrier")
	}
	if rc != ops.ConsistencyLinearizable {
		t.Fatalf("per-partition arg rc = %d, want Linearizable(%d) — rc dropped (silent-degrade)", rc, ops.ConsistencyLinearizable)
	}

	const bound = uint64(1234)
	bArg := ops.EncodeQueryArgs("phys#2", specBytes, ops.ConsistencyBoundedStaleness, 0, bound)
	gotBound, okB := ops.ReadStalenessOf("vector_named_query", bArg)
	if !okB || gotBound != bound {
		t.Fatalf("ReadStalenessOf(vector_named_query) = (%d,%v), want (%d,true)", gotBound, okB, bound)
	}
}

// TestDirectStoreNamedQueryFusionNonEmpty is the directStore bug-class guard (the
// v1 directStore.VectorQuery class): a single-node Direct store named FUSION query
// must route the mode-tagged result through fusionMergeFanOut, NOT read qr.Fused
// directly (which is nil for FUSION). A non-empty result proves the merge ran. Also
// cross-checked against the embedded single-shard path for exactness.
func TestDirectStoreNamedQueryFusionNonEmpty(t *testing.T) {
	const n, k = 120, 10
	ctx := context.Background()
	titleQ := []float32{1.2, 0.6, 0.3, 0.4}
	imageQ := []float32{0.5, 0.5, 0.2}
	pspec := &pb.QuerySpec{
		Mode: pb.QueryMode_QUERY_MODE_FUSION,
		Prefetch: []*pb.QueryLeaf{
			namedDenseLeaf("title", titleQ, k),
			namedDenseLeaf("image", imageQ, k),
			namedSparseLeaf("terms", []uint32{1, 14, 26}, []float32{2, 3, 1}, k),
		},
		FusionMethod: "rrf",
		K:            int32(k),
	}
	specBytes, spec := buildQuerySpec(t, pspec)

	// Seed a single-node Direct store with the same 3-space named collection, then
	// run the named FUSION query through directStore.VectorNamedQuery so the
	// directStore merge path is exercised end-to-end on a non-clustered store.
	d := newSingleDirect(t)
	cfg := map[string]NamedVectorParams{
		"title": {Dim: 4, Metric: vector.L2},
		"image": {Dim: 3, Metric: vector.L2},
		"terms": {Sparse: true},
	}
	if err := d.VectorNamedCreateCollection(ctx, "dnq", cfg, 0); err != nil {
		t.Fatalf("direct create named collection: %v", err)
	}
	for id := uint64(1); id <= uint64(n); id++ {
		title := []float32{
			float32(id)*0.01 + 0.1, float32(id%5)*0.2 + 0.05,
			float32(id%7)*0.13 + 0.02, float32(id%3)*0.31 + 0.07,
		}
		image := []float32{
			float32(id%9)*0.11 + 0.03, float32(id)*0.007 + 0.2, float32(id%4)*0.17 + 0.01,
		}
		idx := []uint32{uint32(id % 7), uint32(id%11) + 12, uint32(id%13) + 24}
		sort.Slice(idx, func(i, j int) bool { return idx[i] < idx[j] })
		uniq := idx[:1]
		for _, v := range idx[1:] {
			if v != uniq[len(uniq)-1] {
				uniq = append(uniq, v)
			}
		}
		sv := vector.SparseVector{Indices: uniq, Values: make([]float32, len(uniq))}
		for i := range sv.Values {
			sv.Values[i] = float32(id)*0.013 + float32(uniq[i])*0.001 + float32(i)*0.1 + 1
		}
		ds := d.(*directStore)
		if _, err := ds.Call(ctx, "vector_named_insert",
			ops.EncodeNamedInsertArgsSparseCASKeyTTL("dnq", id,
				map[string][]float32{"title": title, "image": image},
				map[string]*vector.SparseVector{"terms": &sv},
				VectorMetadata{"id": vector.NewInt(int64(id))}, 0, 0, false, nil)); err != nil {
			t.Fatalf("direct named insert %d: %v", id, err)
		}
	}

	got, _, err := d.VectorNamedQuery(ctx, "dnq", specBytes, spec, ReadOpts{})
	if err != nil {
		t.Fatalf("directStore named query: %v", err)
	}
	if len(got) == 0 {
		t.Fatal("directStore named FUSION returned EMPTY — qr.Fused read directly instead of routed through the merge")
	}
}
