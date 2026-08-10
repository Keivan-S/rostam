// SPDX-License-Identifier: Apache-2.0

package rostam

import (
	"context"
	"math/rand"
	"testing"

	"github.com/rostamlabs/rostam/ops"
	"github.com/rostamlabs/rostam/vector"
)

// benchNetVectors builds n random dim-dimensional vectors (standard normal).
func benchNetVectors(n, dim int, seed int64) [][]float32 {
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

// setupVectorServer stands up a Direct-backed TCP server with a populated
// "bench" collection and returns a connected client plus a cleanup func. The
// measured path is the full wire round-trip: client encode → TCP → server
// decode → HNSW search → encode → TCP → client decode.
func setupVectorServer(tb testing.TB, n, dim int) (Store, func()) {
	tb.Helper()
	reg := ops.NewRegistry()
	if err := ops.RegisterBuiltins(reg); err != nil {
		tb.Fatal(err)
	}
	srv, err := NewDirectServer("127.0.0.1:0", DirectConfig{
		DataDir: tb.TempDir(),
		Ops:     reg,
		Cache:   CacheConfig{NumShardsPerNode: 4},
	})
	if err != nil {
		tb.Fatal(err)
	}
	cli, err := NewClient(ClientConfig{Servers: []string{srv.Addr()}})
	if err != nil {
		_ = srv.Close()
		tb.Fatal(err)
	}
	ctx := context.Background()
	cfg := VectorConfig{Dim: dim, Metric: vector.L2, M: 16, EfConstruction: 200, EfSearch: 64, Seed: 1}
	if err := cli.CreateCollection(ctx, "bench", cfg); err != nil {
		tb.Fatal(err)
	}
	for i, v := range benchNetVectors(n, dim, 42) {
		if err := cli.VectorInsert(ctx, "bench", uint64(i+1), v); err != nil {
			tb.Fatal(err)
		}
	}
	return cli, func() { _ = cli.Close(); _ = srv.Close() }
}

// BenchmarkVectorSearchTCP measures serial client→server kNN latency over TCP.
func BenchmarkVectorSearchTCP(b *testing.B) {
	const dim = 128
	cli, cleanup := setupVectorServer(b, 10_000, dim)
	defer cleanup()
	queries := benchNetVectors(1_000, dim, 99)
	ctx := context.Background()
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		if _, err := cli.VectorSearch(ctx, "bench", queries[i%len(queries)], 10); err != nil {
			b.Fatal(err)
		}
	}
}

// BenchmarkVectorSearchIntoTCP measures the zero-copy client path: a reused dst
// slice + CallFunc, so the client decodes the response with no defensive copy
// and no result-slice allocation. Compare allocs/op against BenchmarkVectorSearchTCP.
func BenchmarkVectorSearchIntoTCP(b *testing.B) {
	const dim = 128
	cli, cleanup := setupVectorServer(b, 10_000, dim)
	defer cleanup()
	queries := benchNetVectors(1_000, dim, 99)
	ctx := context.Background()
	dst := make([]VectorResult, 0, 10)
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		var err error
		dst, err = cli.VectorSearchInto(ctx, "bench", queries[i%len(queries)], 10, dst[:0])
		if err != nil {
			b.Fatal(err)
		}
	}
}

// BenchmarkVectorSearchTCP_Parallel measures saturated throughput/latency with
// many concurrent connections hitting the search path.
func BenchmarkVectorSearchTCP_Parallel(b *testing.B) {
	const dim = 128
	cli, cleanup := setupVectorServer(b, 10_000, dim)
	defer cleanup()
	queries := benchNetVectors(1_000, dim, 99)
	b.ReportAllocs()
	b.ResetTimer()
	b.RunParallel(func(pb *testing.PB) {
		ctx := context.Background()
		i := 0
		for pb.Next() {
			if _, err := cli.VectorSearch(ctx, "bench", queries[i%len(queries)], 10); err != nil {
				b.Fatal(err)
			}
			i++
		}
	})
}
