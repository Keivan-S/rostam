// SPDX-License-Identifier: Apache-2.0

package rostam

import (
	"context"
	"testing"

	"github.com/rostamlabs/rostam/ops"
	"github.com/rostamlabs/rostam/sdk/pb"
	"github.com/rostamlabs/rostam/vector"
)

// seedDiscoverCollection creates a P-partition (or P==1) DENSE collection (dim 4,
// Cosine) with a HIGH EfSearch and a SMALL n so the discover SEED pool (an HNSW
// search over the target/mean seed) recalls EVERY doc on BOTH P=1 and P=4 — the
// precondition for discover's per-partition scorer to be partition-invariant. (With
// a small n <= EfSearch the seed search returns the full corpus, so each partition
// scores the same global candidate set and the score-desc merge reproduces P=1.) Ids
// 1..n carry deterministic spread directions so the discover context steers cleanly.
func seedDiscoverCollection(t *testing.T, s Store, coll string, P, n int) {
	t.Helper()
	ctx := context.Background()
	if err := s.CreateCollection(ctx, coll, VectorConfig{
		Dim: 4, M: 8, EfConstruction: 200, EfSearch: 400, Seed: 1, Metric: vector.Cosine, Partitions: P,
	}); err != nil {
		t.Fatalf("CreateCollection %q (P=%d): %v", coll, P, err)
	}
	for id := uint64(1); id <= uint64(n); id++ {
		f := float32(id)
		v := []float32{f, f * 0.5, float32(id%5) + 1, float32(id%3) + 1}
		if err := s.VectorInsertExt(ctx, coll, id, v, VectorInsertOpts{}); err != nil {
			t.Fatalf("VectorInsertExt %s/%d: %v", coll, id, err)
		}
	}
}

// discoverLeaf builds a proto discover QueryLeaf carrying the UNRESOLVED target +
// context-pair ids (the coordinator pre-pass resolves them cluster-wide → vectors
// and embeds them before fan-out). The discover lane is score-descending.
func discoverLeafPB(targetID []uint64, context [][2]uint64, k int) *pb.QueryLeaf {
	ids := make([]*pb.ContextPairIDs, len(context))
	for i, cp := range context {
		ids[i] = &pb.ContextPairIDs{Positive: cp[0], Negative: cp[1]}
	}
	return &pb.QueryLeaf{Leaf: &pb.QueryLeaf_Discover{Discover: &pb.DiscoverLeaf{
		TargetId: targetID, ContextIds: ids, K: int32(k),
	}}}
}

// discoverFanSpec builds a FUSION spec with a single discover prefetch leaf (the
// canonical discover query: one discover node → score-desc lane).
func discoverFanSpec(t *testing.T, targetID []uint64, context [][2]uint64, k int) ([]byte, vector.QuerySpec) {
	t.Helper()
	return buildQuerySpec(t, &pb.QuerySpec{
		Mode:         pb.QueryMode_QUERY_MODE_FUSION,
		Prefetch:     []*pb.QueryLeaf{discoverLeafPB(targetID, context, k)},
		FusionMethod: "rrf",
		K:            int32(k),
	})
}

// TestDiscoverFanOutMatchesP1 is the discover exactness invariant: a discover query
// (target + context-pair ids → coordinator-resolve → embed → per-partition scorer →
// score-desc merge) over a P=4 collection returns the EXACT same top-k (id + score +
// distance, in order) as the same query over P=1. The per-partition discover scorer
// + the score-desc fan-out merge are partition-invariant (every doc is scored on its
// sole owning partition; the context/target vectors are resolved once on the
// coordinator and embedded identically). Goes RED if the coordinator resolve diverges
// across P, the leaf is re-resolved per-partition, or the score-desc merge mis-orients.
func TestDiscoverFanOutMatchesP1(t *testing.T) {
	const n, k = 40, 10
	ctx := context.Background()
	target := []uint64{20}
	pairs := [][2]uint64{{5, 38}, {6, 37}}
	specBytes, spec := discoverFanSpec(t, target, pairs, k)

	s1 := newSingleEmbedded(t)
	waitLeaderEmbedded(t, s1)
	seedDiscoverCollection(t, s1, "dc1", 1, n)
	got1, _, err := s1.(*embedded).VectorQuery(ctx, "dc1", specBytes, spec, ReadOpts{})
	if err != nil {
		t.Fatalf("P1 discover: %v", err)
	}
	if len(got1) == 0 {
		t.Fatal("P1 discover returned no results")
	}

	const P = 4
	sP := newSingleEmbedded(t)
	waitLeaderEmbedded(t, sP)
	seedDiscoverCollection(t, sP, "dc4", P, n)
	gotP, meta, err := sP.(*embedded).VectorQuery(ctx, "dc4", specBytes, spec, ReadOpts{})
	if err != nil {
		t.Fatalf("P4 discover: %v", err)
	}
	if meta.Degraded {
		t.Fatalf("P4 discover unexpectedly degraded: %+v", meta)
	}
	if len(gotP) == 0 {
		t.Fatal("P4 discover returned no results")
	}
	if !eqHybridKeys(queryResultKeys(gotP), queryResultKeys(got1)) {
		t.Fatalf("discover P4 != P1:\n P4=%v\n P1=%v", queryResultKeys(gotP), queryResultKeys(got1))
	}
}

// TestDiscoverFanOutCrossPartitionIDs proves the cluster id-resolution: the target +
// every context-pair id are chosen to hash to DIFFERENT partitions (P=4), so the
// coordinator's cluster-wide batch-get must reach multiple partitions to resolve them
// all before embedding. The P=4 result must equal the P=1 result on the SAME inputs
// (a single-node resolve over all ids). Goes RED if the coordinator resolves only the
// coordinator-local partition's ids (cross-partition target/context lost).
func TestDiscoverFanOutCrossPartitionIDs(t *testing.T) {
	const n, k = 40, 10
	const P = 4
	ctx := context.Background()

	// Pick a target + 2 context pairs (5 ids total) that span >=4 distinct partitions.
	var ids []uint64
	seen := map[int]bool{}
	for id := uint64(1); id <= uint64(n) && len(ids) < 5; id++ {
		p := ops.PartitionOf(id, P)
		if !seen[p] || len(ids) >= P { // first take one per partition, then any to fill 5
			seen[p] = true
			ids = append(ids, id)
		}
	}
	if len(ids) < 5 {
		t.Fatalf("could not pick 5 ids (got %d)", len(ids))
	}
	parts := map[int]bool{}
	for _, id := range ids {
		parts[ops.PartitionOf(id, P)] = true
	}
	if len(parts) < 4 {
		t.Fatalf("ids only span %d partitions, want >=4 (cross-partition resolution not exercised)", len(parts))
	}
	target := []uint64{ids[0]}
	pairs := [][2]uint64{{ids[1], ids[2]}, {ids[3], ids[4]}}
	specBytes, spec := discoverFanSpec(t, target, pairs, k)

	s1 := newSingleEmbedded(t)
	waitLeaderEmbedded(t, s1)
	seedDiscoverCollection(t, s1, "dx1", 1, n)
	got1, _, err := s1.(*embedded).VectorQuery(ctx, "dx1", specBytes, spec, ReadOpts{})
	if err != nil {
		t.Fatalf("P1 cross-partition discover: %v", err)
	}

	sP := newSingleEmbedded(t)
	waitLeaderEmbedded(t, sP)
	seedDiscoverCollection(t, sP, "dx4", P, n)
	gotP, meta, err := sP.(*embedded).VectorQuery(ctx, "dx4", specBytes, spec, ReadOpts{})
	if err != nil {
		t.Fatalf("P4 cross-partition discover: %v", err)
	}
	if meta.Degraded {
		t.Fatalf("P4 cross-partition discover unexpectedly degraded: %+v", meta)
	}
	if len(gotP) == 0 {
		t.Fatal("P4 cross-partition discover returned no results (cluster id-resolution likely missed cross-partition ids)")
	}
	if !eqHybridKeys(queryResultKeys(gotP), queryResultKeys(got1)) {
		t.Fatalf("cross-partition discover P4 != P1:\n P4=%v\n P1=%v", queryResultKeys(gotP), queryResultKeys(got1))
	}
}

// TestDiscoverFanOutRerankRootMatchesP1: a discover RERANK ROOT (the union of a dense
// prefetch's candidates re-scored by the discover context-pair scorer, score-desc)
// over P=4 must equal P=1. Because ids are partition-disjoint each doc is reranked on
// its owning partition; the score-desc rerankMergeFanOut merge reproduces the global
// rerank. Exercises the discover leaf as the RERANK root (root.ScoreDesc=true).
func TestDiscoverFanOutRerankRootMatchesP1(t *testing.T) {
	const n, k = 40, 10
	ctx := context.Background()
	rootDiscover := discoverLeafPB([]uint64{20}, [][2]uint64{{5, 38}}, k)
	build := func() ([]byte, vector.QuerySpec) {
		return buildQuerySpec(t, &pb.QuerySpec{
			Mode:     pb.QueryMode_QUERY_MODE_RERANK,
			Root:     rootDiscover,
			Prefetch: []*pb.QueryLeaf{denseLeaf([]float32{20, 10, 1, 1}, 50)},
			K:        int32(k),
		})
	}
	specBytes, spec := build()

	s1 := newSingleEmbedded(t)
	waitLeaderEmbedded(t, s1)
	seedDiscoverCollection(t, s1, "dr1", 1, n)
	got1, _, err := s1.(*embedded).VectorQuery(ctx, "dr1", specBytes, spec, ReadOpts{})
	if err != nil {
		t.Fatalf("P1 discover rerank: %v", err)
	}
	if len(got1) == 0 {
		t.Fatal("P1 discover rerank returned no results")
	}

	const P = 4
	sP := newSingleEmbedded(t)
	waitLeaderEmbedded(t, sP)
	seedDiscoverCollection(t, sP, "dr4", P, n)
	gotP, meta, err := sP.(*embedded).VectorQuery(ctx, "dr4", specBytes, spec, ReadOpts{})
	if err != nil {
		t.Fatalf("P4 discover rerank: %v", err)
	}
	if meta.Degraded {
		t.Fatalf("P4 discover rerank unexpectedly degraded: %+v", meta)
	}
	if !eqHybridKeys(queryResultKeys(gotP), queryResultKeys(got1)) {
		t.Fatalf("discover rerank P4 != P1:\n P4=%v\n P1=%v", queryResultKeys(gotP), queryResultKeys(got1))
	}
}

// TestDiscoverFanOutFailLoudMissingTarget: a discover whose TARGET id does NOT exist
// anywhere in the cluster cannot resolve the anchor → fail-loud (mirrors the single-
// node ErrIDNotFound). The error must surface on the cluster path, not an empty result.
func TestDiscoverFanOutFailLoudMissingTarget(t *testing.T) {
	const n, k = 40, 10
	const P = 4
	ctx := context.Background()
	specBytes, spec := discoverFanSpec(t, []uint64{9001}, [][2]uint64{{10, 11}}, k)

	sP := newSingleEmbedded(t)
	waitLeaderEmbedded(t, sP)
	seedDiscoverCollection(t, sP, "df4", P, n)
	if _, _, err := sP.(*embedded).VectorQuery(ctx, "df4", specBytes, spec, ReadOpts{}); err == nil {
		t.Fatal("expected fail-loud error when the discover target id does not resolve cluster-wide")
	}
}

// TestDiscoverFanOutFailLoudAllContextMissing: a discover whose context-pair ids do
// NOT exist cluster-wide cannot score candidates → fail-loud (no pair resolves →
// ErrIDNotFound). The error must surface, not an empty/garbage result.
func TestDiscoverFanOutFailLoudAllContextMissing(t *testing.T) {
	const n, k = 40, 10
	const P = 4
	ctx := context.Background()
	specBytes, spec := discoverFanSpec(t, nil, [][2]uint64{{9001, 9002}}, k)

	sP := newSingleEmbedded(t)
	waitLeaderEmbedded(t, sP)
	seedDiscoverCollection(t, sP, "dam4", P, n)
	if _, _, err := sP.(*embedded).VectorQuery(ctx, "dam4", specBytes, spec, ReadOpts{}); err == nil {
		t.Fatal("expected fail-loud error when NO discover context pair resolves cluster-wide")
	}
}
