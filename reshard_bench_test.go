// SPDX-License-Identifier: Apache-2.0

package rostam

import (
	"context"
	"strconv"
	"testing"
	"time"

	"github.com/rostamlabs/rostam/vector"
)

// Wave-10 benchmark for RESHARD throughput: VectorResplit re-partitions a
// collection by rebuilding its data into a new partition count (offline bulk
// re-partition). The matrix flagged reshard throughput as unmeasured. It runs on
// an in-process embedded store (the resplit coordinator is an embedded-layer op),
// so no networked multi-node cluster is needed.
//
// Each iteration resplits a FRESH collection (unique name) so the timed op always
// does a full P_old -> P_new rebuild; the create+insert setup is excluded via
// StopTimer. Resplit is heavy (rebuilds every partition), so run with a small
// fixed count, e.g. -benchtime 10x.

func BenchmarkResplit(b *testing.B) {
	const n, dim, pOld, pNew = 5_000, 64, 2, 4
	s := newBenchEmbedded(b) // NumShards=1; partitions live on the one shard
	defer s.Close()
	ctx := context.Background()
	vecs := make([][]float32, n)
	for i := range vecs {
		v := make([]float32, dim)
		v[0] = float32(i + 1)
		vecs[i] = v
	}

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		b.StopTimer()
		coll := "resh" + strconv.Itoa(i)
		cfg := VectorConfig{Dim: dim, M: 16, EfConstruction: 100, EfSearch: 64, Seed: 1, Metric: vector.L2, Partitions: pOld}
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
			if err := s.VectorInsertExt(ctx, coll, id, vecs[id-1], VectorInsertOpts{}); err != nil {
				b.Fatalf("insert %d: %v", id, err)
			}
		}
		b.StartTimer()
		if err := s.VectorResplit(ctx, coll, pNew); err != nil {
			b.Fatalf("VectorResplit %d->%d: %v", pOld, pNew, err)
		}
		b.StopTimer()
	}
	b.ReportMetric(float64(n), "vecs/resplit")
}
