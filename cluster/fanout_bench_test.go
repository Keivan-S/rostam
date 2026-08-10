// SPDX-License-Identifier: Apache-2.0

package cluster

import (
	"strconv"
	"testing"
)

// Wave-7 benchmarks for the cross-shard SEARCH fanout coordinator. The existing
// cluster benchmarks only cover KV Put; the multi-partition search scatter/gather
// (FanOut) and its top-K merge had no coverage (see BENCHMARKS.md). FanOut is
// driven here with a fake partition caller so the benchmark isolates the
// COORDINATOR overhead (per-partition goroutine scatter + decode + merge) as a
// function of partition count P — the real per-partition shard latency is the
// single-partition VectorSearchTCP benchmark's job.

const fanBenchK = 10

// BenchmarkFanOutSearch measures the search fanout coordinator: scatter to P
// partitions (concurrent), decode each partition's K results, and merge to top-K.
// The fake caller returns instantly, so this is pure coordination cost vs P.
func BenchmarkFanOutSearch(b *testing.B) {
	caller := func(physCol, op string, args []byte, leaderOnly bool) ([]byte, error) {
		// Echo the physical name's last byte as a per-partition seed.
		return []byte{physCol[len(physCol)-1]}, nil
	}
	decode := func(bts []byte) ([]Scored, error) {
		seed := uint64(bts[0])
		out := make([]Scored, fanBenchK)
		for i := range out {
			out[i] = Scored{ID: seed*1000 + uint64(i), Dist: float32(seed) + float32(i)*0.1}
		}
		return out, nil
	}
	merge := func(parts [][]Scored, k int) []Scored {
		return MergeTopK(parts, k, func(a, b Scored) bool { return a.Dist < b.Dist })
	}
	for _, p := range []int{2, 4, 16, 64} {
		b.Run("P="+strconv.Itoa(p), func(b *testing.B) {
			b.ResetTimer()
			for i := 0; i < b.N; i++ {
				_, _, err := FanOut(FanArgs{
					Collection: "default/docs", P: p, K: fanBenchK,
					Op: "vector_search", Consistency: AnyReplica, OnUnavailable: Partial,
					Encode: func(physCol string) []byte { return nil },
				}, caller, decode, merge)
				if err != nil {
					b.Fatal(err)
				}
			}
		})
	}
}

// BenchmarkMergeTopK isolates the merge step (concat P*K candidates, then top-K),
// the CPU-bound part of fanout that scales with P. Pre-builds the per-partition
// result sets so only the merge is timed.
func BenchmarkMergeTopK(b *testing.B) {
	less := func(a, b Scored) bool { return a.Dist < b.Dist }
	for _, p := range []int{2, 4, 16, 64} {
		parts := make([][]Scored, p)
		for pi := range parts {
			parts[pi] = make([]Scored, fanBenchK)
			for i := range parts[pi] {
				// Interleave distances across partitions so the merge can't shortcut.
				parts[pi][i] = Scored{ID: uint64(pi*1000 + i), Dist: float32(i*p + pi)}
			}
		}
		b.Run("P="+strconv.Itoa(p), func(b *testing.B) {
			b.ResetTimer()
			for i := 0; i < b.N; i++ {
				_ = MergeTopK(parts, fanBenchK, less)
			}
		})
	}
}
