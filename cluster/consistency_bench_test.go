// SPDX-License-Identifier: Apache-2.0

package cluster

import (
	"context"
	"math/rand"
	"testing"

	"github.com/rostamlabs/rostam/ops"
	"github.com/rostamlabs/rostam/vector"
)

// Wave-9 benchmarks for the distributed READ-CONSISTENCY overhead on vector
// search: AnyReplica (default, barrier-free) vs LeaderOnly (leader-pinned) vs
// Linearizable (leader + readIndex barrier: VerifyLeader + commit-index catch-up).
// The matrix flagged these as unmeasured. A single-node multi-shard cluster
// suffices: the readIndex barrier (verifyLeaderAndCatchUp) runs on the serving
// leader regardless of replica count, so this captures the barrier CODE-PATH
// overhead relative to the default path (a real multi-node cluster adds the
// VerifyLeader heartbeat-quorum round on top — noted, not measured here to avoid
// multi-node Raft flakiness in a benchmark).

func benchRandVecs(n, dim int, seed int64) [][]float32 {
	rng := rand.New(rand.NewSource(seed))
	out := make([][]float32, n)
	for i := range out {
		v := make([]float32, dim)
		for j := range v {
			v[j] = float32(rng.NormFloat64())
		}
		out[i] = v
	}
	return out
}

// BenchmarkVectorSearchConsistency measures vector_search latency over the cluster
// client at each read-consistency level against one single-partition collection
// (10k × 128). The Linearizable vs AnyReplica delta is the readIndex-barrier cost.
func BenchmarkVectorSearchConsistency(b *testing.B) {
	const n, dim, k = 10_000, 128, 10
	c, stop := benchClusterStack(b, 4) // single node, 4 shards
	defer stop()
	ctx := context.Background()

	// Single-partition physical collection so the search routes to exactly one
	// shard (no fan-out) — isolates the per-shard read-consistency cost.
	const coll = "bench/docs"
	physCol := string(ops.PartitionKeyGen(coll, 0, 0))
	cfg := vector.Config{Dim: dim, Metric: vector.L2, M: 16, EfConstruction: 200, EfSearch: 64, Seed: 1}
	if _, err := c.Call(ctx, "vector_create_collection", ops.EncodeCreateCollectionArgs(physCol, cfg)); err != nil {
		b.Fatalf("create collection: %v", err)
	}
	corpus := benchRandVecs(n, dim, 42)
	for i := 0; i < n; i++ {
		if _, err := c.Call(ctx, "vector_upsert",
			ops.EncodeVectorUpsertArgs(physCol, uint64(i+1), corpus[i], "", 0, nil, vector.SparseVector{})); err != nil {
			b.Fatalf("upsert %d: %v", i, err)
		}
	}
	queries := benchRandVecs(1_000, dim, 99)

	levels := []struct {
		name string
		rc   uint8
	}{
		{"any_replica", 0},
		{"leader_only", 1},
		{"linearizable", 2},
	}
	for _, lvl := range levels {
		b.Run(lvl.name, func(b *testing.B) {
			b.ResetTimer()
			for i := 0; i < b.N; i++ {
				args := ops.EncodeVectorSearchArgsOpts(physCol, k, queries[i%len(queries)], vector.Filter{}, lvl.rc, 0, 0)
				if _, err := c.Call(ctx, "vector_search", args); err != nil {
					b.Fatal(err)
				}
			}
		})
	}
}
