// SPDX-License-Identifier: Apache-2.0

package inttest

import (
	"context"
	"math/rand"
	"testing"

	"github.com/rostamlabs/rostam"
	"github.com/rostamlabs/rostam/ops"
	"github.com/rostamlabs/rostam/vector"
)

// benchSrcs wraps query leaves as prefetch sources (the inttest-local counterpart
// of the srcs helper in the ops/vector test packages). The Go 1.22+ per-iteration
// loop variable makes &leaves[i] safe.
func benchSrcs(leaves ...vector.QueryLeaf) []vector.QuerySource {
	out := make([]vector.QuerySource, len(leaves))
	for i := range leaves {
		out[i] = vector.QuerySource{Leaf: &leaves[i]}
	}
	return out
}

// Coordinator-bound benchmarks that REQUIRE a real multi-node cluster (RF>1):
// the write-consistency barrier and bounded-staleness reads only differ once a
// shard has replicas on distinct nodes, so they cannot be measured on the
// single-node embedded / bufconn harnesses (they were the BLOCKED items in the
// Wave-10 / B2 findings). These drive the real RF=3 in-process cluster
// (newInmemEmbeddedClusterServers), whose ops forward over TCP through the fan-out
// coordinator. Cluster construction (Raft election settle) is excluded from the
// timed region via b.ResetTimer / per-sub-benchmark setup.
//
// NOTE: these spin up 3-node Raft groups and are election-latency-heavy; run them
// deliberately (`-run '^$' -bench BenchmarkClusterWriteConsistency -benchtime Nx`),
// not as part of a broad sweep, and on a host that is not CPU-oversubscribed (see
// the test-suite ops notes on inttest flakiness).

func benchVec(dim int, seed int64) []float32 {
	r := rand.New(rand.NewSource(seed)) //nolint:gosec // deterministic benchmark data
	v := make([]float32, dim)
	for i := range v {
		v[i] = r.Float32()
	}
	return v
}

// BenchmarkClusterWriteConsistency measures the per-write cost of the tunable
// write-consistency barrier on a single-partition RF=3 collection. factor=1 is
// leader-only (no replica barrier); factor=2 waits for one follower ack; factor=3
// waits for both; nowait skips the barrier entirely (Wait=false). The spread
// between them IS the barrier latency — the knob that was unmeasurable until a true
// multi-replica cluster existed.
func BenchmarkClusterWriteConsistency(b *testing.B) {
	const dim = 64
	stores, _ := newInmemEmbeddedClusterServers(b, 3, 1, 3) // n=3, 1 shard, RF=3
	store := stores[0]
	ctx := context.Background()
	const coll = "wc"
	if err := store.CreateCollection(ctx, coll, rostam.VectorConfig{
		Dim: dim, Metric: vector.Cosine, M: 16, EfConstruction: 200, EfSearch: 64, Partitions: 1,
	}); err != nil {
		b.Fatal(err)
	}
	v := benchVec(dim, 7)

	cases := []struct {
		name string
		opts rostam.VectorInsertOpts
	}{
		{"factor=1", rostam.VectorInsertOpts{WriteOpts: rostam.WriteOpts{WriteConsistencyFactor: 1}}},
		{"factor=2", rostam.VectorInsertOpts{WriteOpts: rostam.WriteOpts{WriteConsistencyFactor: 2}}},
		{"factor=3", rostam.VectorInsertOpts{WriteOpts: rostam.WriteOpts{WriteConsistencyFactor: 3}}},
		{"nowait", rostam.VectorInsertOpts{WriteOpts: rostam.WriteOpts{Wait: bptr(false)}}},
	}
	id := uint64(1)
	for _, tc := range cases {
		b.Run(tc.name, func(b *testing.B) {
			for b.Loop() {
				if err := store.VectorInsertExt(ctx, coll, id, v, tc.opts); err != nil {
					b.Fatal(err)
				}
				id++
			}
		})
	}
}

// BenchmarkClusterBoundedStalenessRead measures read latency by consistency level
// on a single-partition RF=3 collection: leader-only (1), linearizable readIndex
// barrier (4), and bounded-staleness (3) with a slack vs zero MaxStaleness. The
// readIndex barrier and the replica-lag check are coordinator/raft round-trips that
// only exist with replicas — so this is the multi-node counterpart to the embedded
// search benchmarks.
func BenchmarkClusterBoundedStalenessRead(b *testing.B) {
	const dim, n, k = 64, 2_000, 10
	stores, _ := newInmemEmbeddedClusterServers(b, 3, 1, 3)
	store := stores[0]
	ctx := context.Background()
	const coll = "bs"
	if err := store.CreateCollection(ctx, coll, rostam.VectorConfig{
		Dim: dim, Metric: vector.Cosine, M: 16, EfConstruction: 200, EfSearch: 64, Partitions: 1,
	}); err != nil {
		b.Fatal(err)
	}
	for i := 0; i < n; i++ {
		if err := store.VectorInsert(ctx, coll, uint64(i+1), benchVec(dim, int64(i+1))); err != nil {
			b.Fatal(err)
		}
	}
	q := benchVec(dim, 99)

	cases := []struct {
		name string
		opts rostam.VectorSearchOpts
	}{
		{"leaderOnly", rostam.VectorSearchOpts{ReadConsistency: 1}},
		{"linearizable", rostam.VectorSearchOpts{ReadConsistency: 4}},
		{"boundedStaleness/slack", rostam.VectorSearchOpts{ReadConsistency: 3, MaxStaleness: 1_000_000}},
		{"boundedStaleness/zero", rostam.VectorSearchOpts{ReadConsistency: 3, MaxStaleness: 0}},
	}
	for _, tc := range cases {
		b.Run(tc.name, func(b *testing.B) {
			for b.Loop() {
				if _, _, err := store.VectorSearchExt(ctx, coll, q, k, tc.opts); err != nil {
					b.Fatal(err)
				}
			}
		})
	}
}

// BenchmarkClusterQueryFusionFanout measures the unified Query API in FUSION mode
// across partitions — the path that was BLOCKED on the single-node bufconn harness
// (Wave 16): a multi-lane fusion op returns the UNFUSED per-lane results and the
// fan-out COORDINATOR fuses them into a flat top-k. The real cluster store
// (embedded.VectorQuery → queryFanOut → merge) IS that coordinator, so a 2-lane
// dense RRF fusion runs end-to-end here. P=1 (no fan-out) vs P=4 (fan out to 4
// partitions across 3 nodes, then RRF-merge per lane) isolates the cross-partition
// fan-out + fusion-merge cost.
func BenchmarkClusterQueryFusionFanout(b *testing.B) {
	const dim, nDocs, k = 64, 4_000, 10
	stores, _ := newInmemEmbeddedClusterServers(b, 3, 4, 1) // n=3, 4 shards, RF=1
	store := stores[0]
	ctx := context.Background()

	q1 := benchVec(dim, 99)
	q2 := benchVec(dim, 123) // a second, distinct lane so RRF actually merges two rankings
	fusionSpec := func(k int) ([]byte, vector.QuerySpec) {
		spec := vector.QuerySpec{
			Mode: vector.ModeFusion, Method: vector.FusionRRF, K: k,
			Prefetch: benchSrcs(
				vector.QueryLeaf{Kind: vector.LeafDense, Dense: q1, K: 50},
				vector.QueryLeaf{Kind: vector.LeafDense, Dense: q2, K: 50},
			),
		}
		blob, err := ops.MarshalEngineQuerySpec(spec)
		if err != nil {
			b.Fatal(err)
		}
		return blob, spec
	}

	for _, P := range []int{1, 4} {
		coll := "fus" + map[int]string{1: "1", 4: "4"}[P]
		if err := store.CreateCollection(ctx, coll, rostam.VectorConfig{
			Dim: dim, Metric: vector.Cosine, M: 16, EfConstruction: 200, EfSearch: 64, Partitions: P,
		}); err != nil {
			b.Fatal(err)
		}
		for i := 0; i < nDocs; i++ {
			if err := store.VectorInsert(ctx, coll, uint64(i+1), benchVec(dim, int64(i+1))); err != nil {
				b.Fatal(err)
			}
		}
		specBytes, spec := fusionSpec(k)
		b.Run("P="+map[int]string{1: "1", 4: "4"}[P], func(b *testing.B) {
			for b.Loop() {
				if _, _, err := store.VectorQuery(ctx, coll, specBytes, spec, rostam.ReadOpts{}); err != nil {
					b.Fatal(err)
				}
			}
		})
	}
}

// BenchmarkClusterScrollFanout measures one cross-partition scroll PAGE through the
// coordinator — the last unmeasured coordinator read path. A scroll page fans out
// to every partition for the next `limit` ids after the cursor, then merges them
// id-ascending and returns the global top-`limit` (plus the resume cursor). P=1 (no
// fan-out) vs P=4 (fan out to 4 partitions across 3 nodes + k-way id merge)
// isolates the cross-partition page cost. The first page (empty cursor) is timed
// repeatedly — the steady per-page coordinator cost.
func BenchmarkClusterScrollFanout(b *testing.B) {
	const dim, nDocs, limit = 64, 4_000, 100
	stores, _ := newInmemEmbeddedClusterServers(b, 3, 4, 1)
	store := stores[0]
	ctx := context.Background()

	for _, P := range []int{1, 4} {
		coll := "scr" + map[int]string{1: "1", 4: "4"}[P]
		if err := store.CreateCollection(ctx, coll, rostam.VectorConfig{
			Dim: dim, Metric: vector.Cosine, M: 16, EfConstruction: 200, EfSearch: 64, Partitions: P,
		}); err != nil {
			b.Fatal(err)
		}
		for i := 0; i < nDocs; i++ {
			if err := store.VectorInsert(ctx, coll, uint64(i+1), benchVec(dim, int64(i+1))); err != nil {
				b.Fatal(err)
			}
		}
		b.Run("P="+map[int]string{1: "1", 4: "4"}[P], func(b *testing.B) {
			for b.Loop() {
				if _, _, _, err := store.VectorScroll(ctx, coll, rostam.VectorFilter{}, limit, rostam.VectorScrollOpts{}); err != nil {
					b.Fatal(err)
				}
			}
		})
	}
}
