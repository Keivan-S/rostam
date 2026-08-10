// SPDX-License-Identifier: Apache-2.0

package rostam

import (
	"context"
	"math/rand"
	"strconv"
	"testing"
	"time"
)

// BenchmarkMVResplit measures multi-vector resplit (offline bulk re-partition)
// throughput. Mirrors BenchmarkResplit for the MV path: each iteration resplits a
// fresh MV collection P2->P4; the create+add setup is excluded via StopTimer, so
// run with a small fixed count (e.g. -benchtime 10x). The copy is batched via the
// vector_mv_add_batch op (one Raft commit per target partition per scan) instead
// of one commit per document.
func BenchmarkMVResplit(b *testing.B) {
	const n, dim, ntok, pOld, pNew = 2_000, 32, 8, 2, 4
	s := newBenchEmbedded(b) // NumShards=1; partitions live on the one shard
	defer s.Close()
	ctx := context.Background()

	// Pre-build n token matrices (reused across iterations — same data each resplit).
	rng := rand.New(rand.NewSource(42))
	docs := make([][][]float32, n)
	for i := range docs {
		toks := make([][]float32, ntok)
		for t := range toks {
			v := make([]float32, dim)
			for j := range v {
				v[j] = float32(rng.NormFloat64())
			}
			toks[t] = v
		}
		docs[i] = toks
	}

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		b.StopTimer()
		coll := "mvresh" + strconv.Itoa(i)
		var cerr error
		for attempt := 0; attempt < 50; attempt++ {
			if cerr = s.VectorMVCreateCollection(ctx, coll, MultiVectorConfig{Dim: dim, Partitions: pOld}); cerr == nil {
				break
			}
			time.Sleep(50 * time.Millisecond)
		}
		if cerr != nil {
			b.Fatalf("VectorMVCreateCollection: %v", cerr)
		}
		for id := uint64(1); id <= uint64(n); id++ {
			if err := s.VectorMVAdd(ctx, coll, id, docs[id-1], nil); err != nil {
				b.Fatalf("VectorMVAdd %d: %v", id, err)
			}
		}
		b.StartTimer()
		if err := s.VectorMVResplit(ctx, coll, pNew); err != nil {
			b.Fatalf("VectorMVResplit %d->%d: %v", pOld, pNew, err)
		}
		b.StopTimer()
	}
	b.ReportMetric(float64(n), "docs/resplit")
}
