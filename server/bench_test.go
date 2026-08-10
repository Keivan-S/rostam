// SPDX-License-Identifier: Apache-2.0

package server

import (
	"context"
	"fmt"
	"sync/atomic"
	"testing"
	"time"

	"github.com/rostamlabs/rostam/cache"
	"github.com/rostamlabs/rostam/client"
	"github.com/rostamlabs/rostam/ops"
	"github.com/rostamlabs/rostam/shard"
)

func benchStack(b *testing.B, noSync bool) (*client.Client, func()) {
	b.Helper()
	reg := ops.NewRegistry()
	_ = ops.RegisterBuiltins(reg)
	cc := cache.DefaultConfig()
	cc.NumShards = 1
	store, err := shard.New(shard.Config{
		NodeID: "node-bench", DataDir: b.TempDir(),
		Cache: cc, Ops: reg,
		Bootstrap:       true,
		RaftHeartbeatMs: 50, RaftElectionMs: 100, NoSync: noSync,
	})
	if err != nil {
		b.Fatal(err)
	}
	deadline := time.Now().Add(3 * time.Second)
	for time.Now().Before(deadline) {
		if store.IsLeader() {
			break
		}
		time.Sleep(10 * time.Millisecond)
	}
	srv, err := New(Config{Addr: "127.0.0.1:0", Dispatcher: store})
	if err != nil {
		b.Fatal(err)
	}
	go func() {
		_ = srv.Serve()
	}()

	c, err := client.New(client.Config{Servers: []string{srv.Addr().String()}})
	if err != nil {
		b.Fatal(err)
	}
	return c, func() {
		_ = c.Close()
		_ = srv.Close()
		_ = store.Close()
	}
}

func BenchmarkClientCallGet(b *testing.B) {
	c, stop := benchStack(b, true)
	defer stop()
	val := make([]byte, 256)
	for i := range 10_000 {
		k := []byte{byte(i), byte(i >> 8)}
		_, _ = c.Call(context.Background(), "put", ops.EncodePutArgs(k, val, 0))
	}
	b.ResetTimer()
	i := 0
	for b.Loop() {
		k := []byte{byte(i), byte(i >> 8)}
		_, _ = c.Call(context.Background(), "get", ops.EncodeKeyArgs(k))
		i++
		if i == 10_000 {
			i = 0
		}
	}
}

func BenchmarkClientCallPut_NoSync(b *testing.B) {
	c, stop := benchStack(b, true)
	defer stop()
	val := make([]byte, 256)
	b.ResetTimer()
	i := 0
	for b.Loop() {
		k := []byte{byte(i), byte(i >> 8), byte(i >> 16)}
		_, _ = c.Call(context.Background(), "put", ops.EncodePutArgs(k, val, 0))
		i++
	}
}

func BenchmarkClientCallPut_Sync(b *testing.B) {
	c, stop := benchStack(b, false)
	defer stop()
	val := make([]byte, 256)
	b.ResetTimer()
	i := 0
	for b.Loop() {
		k := []byte{byte(i), byte(i >> 8), byte(i >> 16)}
		_, _ = c.Call(context.Background(), "put", ops.EncodePutArgs(k, val, 0))
		i++
	}
}

func BenchmarkClientCallPing(b *testing.B) {
	c, stop := benchStack(b, true)
	defer stop()
	b.ResetTimer()
	for b.Loop() {
		_, _ = c.Call(context.Background(), "__ping__", nil)
	}
}

// Parallel variants — match the workload shape used by external clients
// like asbench (20 worker threads each driving the connection pool).

func benchStackParallel(b *testing.B, noSync bool, maxConns int) (*client.Client, func()) {
	b.Helper()
	reg := ops.NewRegistry()
	_ = ops.RegisterBuiltins(reg)
	cc := cache.DefaultConfig()
	cc.NumShards = 1
	store, err := shard.New(shard.Config{
		NodeID: "node-bench-parallel", DataDir: b.TempDir(),
		Cache: cc, Ops: reg,
		Bootstrap:       true,
		RaftHeartbeatMs: 50, RaftElectionMs: 100, NoSync: noSync,
	})
	if err != nil {
		b.Fatal(err)
	}
	deadline := time.Now().Add(3 * time.Second)
	for time.Now().Before(deadline) {
		if store.IsLeader() {
			break
		}
		time.Sleep(10 * time.Millisecond)
	}
	srv, err := New(Config{Addr: "127.0.0.1:0", Dispatcher: store})
	if err != nil {
		b.Fatal(err)
	}
	go func() { _ = srv.Serve() }()

	c, err := client.New(client.Config{
		Servers:           []string{srv.Addr().String()},
		MaxConnsPerServer: int32(maxConns), //nolint:gosec // benchmark-supplied positive int
	})
	if err != nil {
		b.Fatal(err)
	}
	return c, func() {
		_ = c.Close()
		_ = srv.Close()
		_ = store.Close()
	}
}

func BenchmarkClientCallGet_Parallel(b *testing.B) {
	c, stop := benchStackParallel(b, true, 64)
	defer stop()
	val := make([]byte, 256)
	for i := range 10_000 {
		k := []byte{byte(i), byte(i >> 8)}
		_, _ = c.Call(context.Background(), "put", ops.EncodePutArgs(k, val, 0))
	}
	b.ResetTimer()
	b.RunParallel(func(pb *testing.PB) {
		ctx := context.Background()
		i := 0
		for pb.Next() {
			k := []byte{byte(i), byte(i >> 8)}
			_, _ = c.Call(ctx, "get", ops.EncodeKeyArgs(k))
			i++
			if i == 10_000 {
				i = 0
			}
		}
	})
}

func BenchmarkClientCallPut_NoSync_Parallel(b *testing.B) {
	c, stop := benchStackParallel(b, true, 64)
	defer stop()
	val := make([]byte, 256)
	b.ResetTimer()
	var counter int64
	b.RunParallel(func(pb *testing.PB) {
		ctx := context.Background()
		for pb.Next() {
			i := atomic.AddInt64(&counter, 1)
			k := []byte{byte(i & 0xff), byte((i >> 8) & 0xff), byte((i >> 16) & 0xff)} //nolint:gosec // bytes are masked
			_, _ = c.Call(ctx, "put", ops.EncodePutArgs(k, val, 0))
		}
	})
}

// directDispatcher is a server.Dispatcher backed by a bare cache.Cache
// — no Raft. Used by the Direct-over-TCP benchmarks below to measure
// what the network surface costs when the storage layer is the Phase
// 8.5 NewDirect fast path instead of the Raft-replicated shard.Store.
type directDispatcher struct {
	cache    *cache.Cache
	registry *ops.Registry
	tx       *ops.TxContext // mirrors the prod directStore — reused across calls
}

func (d *directDispatcher) Call(name string, args []byte) ([]byte, error) {
	handler, _, _, ok := d.registry.Lookup(name)
	if !ok {
		return nil, fmt.Errorf("server: op %q not registered", name)
	}
	return handler(d.tx, args)
}

func (d *directDispatcher) LeaderAddr() string { return "" }

func benchStackDirect(b *testing.B, maxConns int) (*client.Client, func()) {
	b.Helper()
	reg := ops.NewRegistry()
	_ = ops.RegisterBuiltins(reg)
	cc := cache.DefaultConfig()
	cc.NumShards = 16
	c, err := cache.New(cc)
	if err != nil {
		b.Fatal(err)
	}
	disp := &directDispatcher{cache: c, registry: reg, tx: ops.NewTxContext(c)}
	srv, err := New(Config{Addr: "127.0.0.1:0", Dispatcher: disp})
	if err != nil {
		_ = c.Close()
		b.Fatal(err)
	}
	go func() { _ = srv.Serve() }()

	cli, err := client.New(client.Config{
		Servers:           []string{srv.Addr().String()},
		MaxConnsPerServer: int32(maxConns), //nolint:gosec // benchmark-supplied positive int
	})
	if err != nil {
		_ = srv.Close()
		_ = c.Close()
		b.Fatal(err)
	}
	return cli, func() {
		_ = cli.Close()
		_ = srv.Close()
		_ = c.Close()
	}
}

func BenchmarkClientCallGet_Direct_Parallel(b *testing.B) {
	c, stop := benchStackDirect(b, 64)
	defer stop()
	val := make([]byte, 256)
	for i := range 10_000 {
		k := []byte{byte(i), byte(i >> 8)}
		_, _ = c.Call(context.Background(), "put", ops.EncodePutArgs(k, val, 0))
	}
	b.ResetTimer()
	b.RunParallel(func(pb *testing.PB) {
		ctx := context.Background()
		i := 0
		for pb.Next() {
			k := []byte{byte(i), byte(i >> 8)}
			_, _ = c.Call(ctx, "get", ops.EncodeKeyArgs(k))
			i++
			if i == 10_000 {
				i = 0
			}
		}
	})
}

// BenchmarkClientCallGetFunc_Direct_Parallel mirrors the above but uses
// CallFunc with a no-op consumer to measure the zero-copy path's alloc
// budget. The delta vs the Call variant is exactly the per-Call
// defensive response copy that Call performs.
func BenchmarkClientCallGetFunc_Direct_Parallel(b *testing.B) {
	c, stop := benchStackDirect(b, 64)
	defer stop()
	val := make([]byte, 256)
	for i := range 10_000 {
		k := []byte{byte(i), byte(i >> 8)}
		_, _ = c.Call(context.Background(), "put", ops.EncodePutArgs(k, val, 0))
	}
	b.ResetTimer()
	b.RunParallel(func(pb *testing.PB) {
		ctx := context.Background()
		i := 0
		for pb.Next() {
			k := []byte{byte(i), byte(i >> 8)}
			_ = c.CallFunc(ctx, "get", ops.EncodeKeyArgs(k), nil)
			i++
			if i == 10_000 {
				i = 0
			}
		}
	})
}

func BenchmarkClientCallPut_Direct_Parallel(b *testing.B) {
	c, stop := benchStackDirect(b, 64)
	defer stop()
	val := make([]byte, 256)
	b.ResetTimer()
	var counter int64
	b.RunParallel(func(pb *testing.PB) {
		ctx := context.Background()
		for pb.Next() {
			i := atomic.AddInt64(&counter, 1)
			k := []byte{byte(i & 0xff), byte((i >> 8) & 0xff), byte((i >> 16) & 0xff)} //nolint:gosec // bytes are masked
			_, _ = c.Call(ctx, "put", ops.EncodePutArgs(k, val, 0))
		}
	})
}
