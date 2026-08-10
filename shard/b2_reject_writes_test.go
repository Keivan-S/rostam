// SPDX-License-Identifier: Apache-2.0

package shard

import (
	"net"
	"testing"
	"time"

	hraft "github.com/hashicorp/raft"

	"github.com/rostamlabs/rostam/cache"
	"github.com/rostamlabs/rostam/ops"
)

// stubStreamLayer is a non-nil hraft.StreamLayer for the gate table test — it is
// never dialed/accepted, only nil-checked by replicatedCacheShard.
type stubStreamLayer struct{}

func (stubStreamLayer) Dial(hraft.ServerAddress, time.Duration) (net.Conn, error) {
	return nil, nil
}
func (stubStreamLayer) Accept() (net.Conn, error) { return nil, nil }
func (stubStreamLayer) Close() error              { return nil }
func (stubStreamLayer) Addr() net.Addr            { return nil }

// TestReplicatedCacheShardGate covers the B2 decision (#4 Phase B): a shard is
// forced to reject-writes iff it participates in cluster replication — PB mode or
// a cluster-wired Raft transport (StreamLayer/Transport). A bare store is not.
func TestReplicatedCacheShardGate(t *testing.T) {
	_, inmem := hraft.NewInmemTransport("")
	cases := []struct {
		name string
		cfg  Config
		want bool
	}{
		{"bare (no transport, no mode)", Config{}, false},
		{"bare raft mode, no transport", Config{ReplicationMode: ReplicationModeRaft}, false},
		{"pb mode", Config{ReplicationMode: ReplicationModePB}, true},
		{"raft stream layer (mux cluster)", Config{RaftStreamLayer: stubStreamLayer{}}, true},
		{"raft transport (fabric cluster)", Config{RaftTransport: inmem}, true},
	}
	for _, c := range cases {
		if got := replicatedCacheShard(c.cfg); got != c.want {
			t.Errorf("%s: replicatedCacheShard = %v, want %v", c.name, got, c.want)
		}
	}
}

// TestNewForcesRejectWritesForReplicatedShard proves New() actually applies the
// B2 policy end-to-end: a cluster shard (external Raft transport) with the DEFAULT
// RingbufEvict cache is constructed with the cache forced to reject-writes, while
// a bare in-memory store keeps its configured evict policy.
func TestNewForcesRejectWritesForReplicatedShard(t *testing.T) {
	reg := ops.NewRegistry()
	if err := ops.RegisterBuiltins(reg); err != nil {
		t.Fatal(err)
	}

	// Cluster shard: external inmem Raft transport ⇒ replicatedCacheShard ⇒ forced.
	repCfg := DefaultConfig(t.TempDir(), "rep", reg)
	repCfg.Bootstrap = true
	repCfg.RaftHeartbeatMs = 50
	repCfg.RaftElectionMs = 100
	repCfg.NoSync = true
	_, inmem := hraft.NewInmemTransport("")
	repCfg.RaftTransport = inmem
	if repCfg.Cache.AtCapPolicy != cache.PolicyRingbufEvict {
		t.Fatalf("precondition: DefaultConfig cache policy = %d, want RingbufEvict(0)", repCfg.Cache.AtCapPolicy)
	}
	rep, err := New(repCfg)
	if err != nil {
		t.Fatalf("shard.New(replicated): %v", err)
	}
	t.Cleanup(func() { _ = rep.Close() })
	if got := rep.cache.AtCapPolicy(); got != cache.PolicyRejectWrites {
		t.Fatalf("replicated shard cache policy = %d, want PolicyRejectWrites(%d) — B2 not applied", got, cache.PolicyRejectWrites)
	}

	// Bare store (no external transport): keeps the configured evict policy.
	bareCfg := DefaultConfig(t.TempDir(), "bare", reg)
	bareCfg.Bootstrap = true
	bareCfg.RaftHeartbeatMs = 50
	bareCfg.RaftElectionMs = 100
	bareCfg.NoSync = true
	bare, err := New(bareCfg)
	if err != nil {
		t.Fatalf("shard.New(bare): %v", err)
	}
	t.Cleanup(func() { _ = bare.Close() })
	if got := bare.cache.AtCapPolicy(); got != cache.PolicyRingbufEvict {
		t.Fatalf("bare store cache policy = %d, want RingbufEvict(%d) unchanged", got, cache.PolicyRingbufEvict)
	}
}
