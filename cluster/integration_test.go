// SPDX-License-Identifier: Apache-2.0

package cluster

import (
	"bytes"
	"context"
	"encoding/binary"
	"fmt"
	"sync"
	"testing"

	"github.com/rostamlabs/rostam/cache"
	"github.com/rostamlabs/rostam/client"
	"github.com/rostamlabs/rostam/ops"
	"github.com/rostamlabs/rostam/server"
	"github.com/rostamlabs/rostam/shard"
)

func startClusterServer(t *testing.T, numShards int) (*client.Client, *Node, func()) {
	t.Helper()
	reg := ops.NewRegistry()
	if err := ops.RegisterBuiltins(reg); err != nil {
		t.Fatal(err)
	}
	cc := cache.DefaultConfig()
	cc.NumShards = 1
	cfg := Config{
		NodeID: "node1", DataDir: t.TempDir(),
		NumShards: numShards, Bootstrap: true,
		ShardCfg: shard.Config{
			NodeID: "ignored", DataDir: "ignored",
			Cache: cc, Ops: reg,
			Bootstrap:       true,
			RaftHeartbeatMs: 50, RaftElectionMs: 100, NoSync: true,
		},
		Ops: reg,
	}
	node, err := New(cfg)
	if err != nil {
		t.Fatal(err)
	}
	// Wait for all shards to elect a leader before the server starts accepting
	// writes; without this, put calls return ErrNotLeader.
	waitAllLeaders(t, node)

	srv, err := server.New(server.Config{Addr: "127.0.0.1:0", Dispatcher: node})
	if err != nil {
		t.Fatal(err)
	}
	go srv.Serve() //nolint:errcheck // returns nil on clean Close()

	c, err := client.New(client.Config{Servers: []string{srv.Addr().String()}})
	if err != nil {
		t.Fatal(err)
	}
	return c, node, func() {
		_ = c.Close()
		_ = srv.Close()
		_ = node.Close()
	}
}

func TestClusterServerEndToEnd(t *testing.T) {
	c, node, stop := startClusterServer(t, 16)
	defer stop()

	ctx := context.Background()
	for i := range 100 {
		k := fmt.Appendf(nil, "k%03d", i)
		if _, err := c.Call(ctx, "put", ops.EncodePutArgs(k, []byte{1}, 0)); err != nil {
			t.Fatalf("put %d: %v", i, err)
		}
	}
	for i := range 100 {
		k := fmt.Appendf(nil, "k%03d", i)
		got, err := c.Call(ctx, "get", ops.EncodeKeyArgs(k))
		if err != nil {
			t.Fatalf("get %d: %v", i, err)
		}
		if !bytes.Equal(got, []byte{1}) {
			t.Fatalf("get %d = %v, want [1]", i, got)
		}
	}

	// Verify routing actually distributes across shards through the wire.
	// 100 keys / 16 shards ≈ 6 per shard mean; require ≥10/16 shards saw writes.
	st := node.Stats()
	occupied := 0
	for _, s := range st.PerShard {
		if s.Cache.Puts > 0 {
			occupied++
		}
	}
	if occupied < 10 {
		t.Errorf("routing through wire not distributing: only %d/16 shards saw writes", occupied)
	}
}

func TestClusterServerConcurrentClients(t *testing.T) {
	c, _, stop := startClusterServer(t, 16)
	defer stop()

	const goroutines = 16
	const iters = 50

	// Write phase: each goroutine writes its own deterministic 4-byte value.
	var wg sync.WaitGroup
	for g := range goroutines {
		wg.Add(1)
		go func(id int) {
			defer wg.Done()
			ctx := context.Background()
			v := make([]byte, 4)
			binary.BigEndian.PutUint32(v, uint32(id)) //nolint:gosec // id ≤ goroutines-1 = 15, no overflow
			for i := range iters {
				k := fmt.Appendf(nil, "g%02d-k%05d", id, i)
				if _, err := c.Call(ctx, "put", ops.EncodePutArgs(k, v, 0)); err != nil {
					t.Errorf("put g=%d i=%d: %v", id, i, err)
					return
				}
			}
		}(g)
	}
	wg.Wait()

	// Read phase: verify each goroutine's keys return the correct value.
	// A response-mixing bug (wrong goroutine gets a reply) would surface here.
	var wg2 sync.WaitGroup
	for g := range goroutines {
		wg2.Add(1)
		go func(id int) {
			defer wg2.Done()
			ctx := context.Background()
			wantV := make([]byte, 4)
			binary.BigEndian.PutUint32(wantV, uint32(id)) //nolint:gosec // id ≤ goroutines-1 = 15, no overflow
			for i := range iters {
				k := fmt.Appendf(nil, "g%02d-k%05d", id, i)
				got, err := c.Call(ctx, "get", ops.EncodeKeyArgs(k))
				if err != nil {
					t.Errorf("get g=%d i=%d: %v", id, i, err)
					return
				}
				if !bytes.Equal(got, wantV) {
					t.Errorf("g=%d i=%d: got %v, want %v (possible response mix-up)", id, i, got, wantV)
					return
				}
			}
		}(g)
	}
	wg2.Wait()
}

func TestClusterServerPingThroughClient(t *testing.T) {
	c, _, stop := startClusterServer(t, 8)
	defer stop()
	if _, err := c.Call(context.Background(), "__ping__", nil); err != nil {
		t.Fatalf("__ping__: %v", err)
	}
}
