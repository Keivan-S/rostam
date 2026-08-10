// SPDX-License-Identifier: Apache-2.0

package cluster

import (
	"context"
	"fmt"
	"testing"

	"github.com/rostamlabs/rostam/cache"
	"github.com/rostamlabs/rostam/client"
	"github.com/rostamlabs/rostam/ops"
	"github.com/rostamlabs/rostam/server"
	"github.com/rostamlabs/rostam/shard"
)

func benchClusterStack(b *testing.B, numShards int) (*client.Client, func()) {
	b.Helper()
	reg := ops.NewRegistry()
	_ = ops.RegisterBuiltins(reg)
	cc := cache.DefaultConfig()
	cc.NumShards = 1
	node, err := New(Config{
		NodeID: "node-bench", DataDir: b.TempDir(),
		NumShards: numShards, Bootstrap: true,
		ShardCfg: shard.Config{
			NodeID: "ignored", DataDir: "ignored",
			Cache: cc, Ops: reg,
			Bootstrap:       true,
			RaftHeartbeatMs: 50, RaftElectionMs: 100, NoSync: true,
		},
		Ops: reg,
	})
	if err != nil {
		b.Fatal(err)
	}
	// Wait for all shards to elect a leader before accepting writes.
	// Without this, early iterations return ErrNotLeader and inflate ns/op.
	waitAllLeaders(b, node)

	srv, err := server.New(server.Config{Addr: "127.0.0.1:0", Dispatcher: node})
	if err != nil {
		b.Fatal(err)
	}
	go srv.Serve() //nolint:errcheck // returns nil on clean Close

	c, err := client.New(client.Config{
		Servers:           []string{srv.Addr().String()},
		MaxConnsPerServer: 16,
	})
	if err != nil {
		b.Fatal(err)
	}
	return c, func() {
		_ = c.Close()
		_ = srv.Close()
		_ = node.Close()
	}
}

func benchClusterPut(b *testing.B, numShards int) {
	b.Helper()
	c, stop := benchClusterStack(b, numShards)
	defer stop()
	val := make([]byte, 256)
	b.ResetTimer()
	b.RunParallel(func(pb *testing.PB) {
		var i int
		for pb.Next() {
			k := fmt.Appendf(nil, "k%07d", i)
			_, _ = c.Call(context.Background(), "put", ops.EncodePutArgs(k, val, 0))
			i++
		}
	})
}

func BenchmarkClusterPut_1Shard(b *testing.B)   { benchClusterPut(b, 1) }
func BenchmarkClusterPut_16Shard(b *testing.B)  { benchClusterPut(b, 16) }
func BenchmarkClusterPut_256Shard(b *testing.B) { benchClusterPut(b, 256) }
