// SPDX-License-Identifier: Apache-2.0

package rostam

import (
	"context"
	"runtime"
	"sync/atomic"
	"testing"

	"github.com/rostamlabs/rostam/ops"
)

// Parallel benchmarks comparing the three Store backends head-to-head.
// Run with: `go test -run=^$ -bench=. -benchtime=3s -count=3 -benchmem`.

const benchValSize = 256

func benchValue() []byte { return make([]byte, benchValSize) }

func BenchmarkDirectGet_Parallel(b *testing.B) {
	reg := ops.NewRegistry()
	_ = ops.RegisterBuiltins(reg)
	s, err := NewDirect(DirectConfig{
		Ops:   reg,
		Cache: CacheConfig{NumShardsPerNode: 16},
	})
	if err != nil {
		b.Fatal(err)
	}
	defer func() { _ = s.Close() }()

	val := benchValue()
	ctx := context.Background()
	for i := range 10_000 {
		k := []byte{byte(i & 0xff), byte((i >> 8) & 0xff)} //nolint:gosec // benchmark int range
		_ = s.Put(ctx, k, val, 0)
	}

	b.ResetTimer()
	b.RunParallel(func(pb *testing.PB) {
		c := context.Background()
		i := 0
		for pb.Next() {
			k := []byte{byte(i & 0xff), byte((i >> 8) & 0xff)} //nolint:gosec // benchmark int range
			_, _ = s.Get(c, k)
			i++
			if i == 10_000 {
				i = 0
			}
		}
	})
}

func BenchmarkDirectPut_Parallel(b *testing.B) {
	reg := ops.NewRegistry()
	_ = ops.RegisterBuiltins(reg)
	s, err := NewDirect(DirectConfig{
		Ops:   reg,
		Cache: CacheConfig{NumShardsPerNode: 16},
	})
	if err != nil {
		b.Fatal(err)
	}
	defer func() { _ = s.Close() }()

	val := benchValue()
	b.ResetTimer()
	var counter int64
	b.RunParallel(func(pb *testing.PB) {
		ctx := context.Background()
		for pb.Next() {
			i := atomic.AddInt64(&counter, 1)
			k := []byte{byte(i & 0xff), byte((i >> 8) & 0xff), byte((i >> 16) & 0xff)} //nolint:gosec // bytes are masked
			_ = s.Put(ctx, k, val, 0)
		}
	})
}

// Embedded parallel benchmarks (for direct comparison against Direct).

func BenchmarkEmbeddedGet_Parallel(b *testing.B) {
	reg := ops.NewRegistry()
	_ = ops.RegisterBuiltins(reg)
	s, err := NewEmbedded(EmbeddedConfig{
		NodeID:    "bench-embedded",
		DataDir:   b.TempDir(),
		NumShards: 1,
		Bootstrap: true,
		Ops:       reg,
		// go test merges the binary's stdout and stderr, so raft at its default
		// verbosity shreds the benchmark result columns. See EmbeddedConfig.
		RaftLogLevel: "ERROR",
	})
	if err != nil {
		b.Fatal(err)
	}
	defer func() { _ = s.Close() }()
	// Wait for leader; election is fast in single-node mode.
	for !s.IsLeader([]byte("probe")) {
		runtime.Gosched()
	}

	val := benchValue()
	ctx := context.Background()
	for i := range 10_000 {
		k := []byte{byte(i & 0xff), byte((i >> 8) & 0xff)} //nolint:gosec // benchmark int range
		_ = s.Put(ctx, k, val, 0)
	}

	b.ResetTimer()
	b.RunParallel(func(pb *testing.PB) {
		c := context.Background()
		i := 0
		for pb.Next() {
			k := []byte{byte(i & 0xff), byte((i >> 8) & 0xff)} //nolint:gosec // benchmark int range
			_, _ = s.Get(c, k)
			i++
			if i == 10_000 {
				i = 0
			}
		}
	})
}

func BenchmarkEmbeddedPut_Parallel(b *testing.B) {
	reg := ops.NewRegistry()
	_ = ops.RegisterBuiltins(reg)
	s, err := NewEmbedded(EmbeddedConfig{
		NodeID:    "bench-embedded-put",
		DataDir:   b.TempDir(),
		NumShards: 1,
		Bootstrap: true,
		Ops:       reg,
		// go test merges the binary's stdout and stderr, so raft at its default
		// verbosity shreds the benchmark result columns. See EmbeddedConfig.
		RaftLogLevel: "ERROR",
		NoSync:       true,
	})
	if err != nil {
		b.Fatal(err)
	}
	defer func() { _ = s.Close() }()
	for !s.IsLeader([]byte("probe")) {
		runtime.Gosched()
	}

	val := benchValue()
	b.ResetTimer()
	var counter int64
	b.RunParallel(func(pb *testing.PB) {
		ctx := context.Background()
		for pb.Next() {
			i := atomic.AddInt64(&counter, 1)
			k := []byte{byte(i & 0xff), byte((i >> 8) & 0xff), byte((i >> 16) & 0xff)} //nolint:gosec // bytes are masked
			_ = s.Put(ctx, k, val, 0)
		}
	})
}
