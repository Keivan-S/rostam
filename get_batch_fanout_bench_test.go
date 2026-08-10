// SPDX-License-Identifier: Apache-2.0

package rostam

import (
	"context"
	"strconv"
	"testing"
	"time"

	"github.com/rostamlabs/rostam/ops"
	"github.com/rostamlabs/rostam/vector"
)

// Wave-9 (distributed) benchmark for the GetBatch scatter/gather fanout: a batched
// multi-id get must route each id to its owning partition, call each partition
// once with its id subset, and reassemble. The matrix flagged it as unmeasured.
// This uses an in-process embedded store with P partitions — real fanout
// (FanOut/CallPhysical over local partitions), no networked multi-node Raft.

// newBenchEmbedded builds a single-node embedded store with NumShards shards and
// waits for leadership. Mirrors the test helpers but takes testing.TB.
func newBenchEmbedded(b *testing.B) Store {
	b.Helper()
	reg := ops.NewRegistry()
	if err := ops.RegisterBuiltins(reg); err != nil {
		b.Fatalf("RegisterBuiltins: %v", err)
	}
	s, err := NewEmbedded(EmbeddedConfig{
		NodeID:    "bench-node",
		DataDir:   b.TempDir(),
		NumShards: 1, // matches the proven seedBatchCollection helper; partitions live on this shard
		Bootstrap: true,
		Ops:       reg,
	})
	if err != nil {
		b.Fatalf("NewEmbedded: %v", err)
	}
	deadline := time.Now().Add(10 * time.Second)
	for time.Now().Before(deadline) {
		if s.LeaderAddr(nil) != "" {
			break
		}
		time.Sleep(20 * time.Millisecond)
	}
	return s
}

// BenchmarkGetBatchFanout measures a 100-id batched get scattered across P
// partitions (each id routed to its owning partition, one call per partition,
// merged). Swept over partition count P — the fanout breadth.
func BenchmarkGetBatchFanout(b *testing.B) {
	const n, batch = 5_000, 100
	for _, P := range []int{1, 4, 16} {
		b.Run("P="+strconv.Itoa(P), func(b *testing.B) {
			s := newBenchEmbedded(b)
			defer s.Close()
			ctx := context.Background()
			coll := "docs"
			cfg := VectorConfig{Dim: 64, M: 16, EfConstruction: 100, EfSearch: 64, Seed: 1, Metric: vector.L2, Partitions: P}
			// Retry briefly to absorb any residual leader-election race after bootstrap.
			var cerr error
			for attempt := 0; attempt < 50; attempt++ {
				if cerr = s.CreateCollection(ctx, coll, cfg); cerr == nil {
					break
				}
				time.Sleep(50 * time.Millisecond)
			}
			if cerr != nil {
				b.Fatalf("CreateCollection: %v", cerr)
			}
			for id := uint64(1); id <= uint64(n); id++ {
				v := make([]float32, 64)
				v[0] = float32(id)
				if err := s.VectorInsertExt(ctx, coll, id, v, VectorInsertOpts{}); err != nil {
					b.Fatalf("insert %d: %v", id, err)
				}
			}
			// A batch of ids spread across the id space (hence across partitions).
			ids := make([]uint64, batch)
			for i := range ids {
				ids[i] = uint64((i*(n/batch))%n + 1)
			}
			b.ResetTimer()
			for i := 0; i < b.N; i++ {
				if _, _, err := s.VectorGetBatch(ctx, coll, ids, true, true); err != nil {
					b.Fatal(err)
				}
			}
		})
	}
}
