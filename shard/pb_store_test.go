// SPDX-License-Identifier: Apache-2.0

package shard

import (
	"testing"

	"github.com/rostamlabs/rostam/cache"
	"github.com/rostamlabs/rostam/ops"
	"github.com/rostamlabs/rostam/shard/pbisr"
)

func TestSingleNodePBStorePutGet(t *testing.T) {
	reg := ops.NewRegistry()
	ops.RegisterBuiltins(reg)
	cc := cache.DefaultConfig()
	cc.NumShards = 1
	cfg := Config{
		NodeID:          "n1",
		DataDir:         t.TempDir(),
		Cache:           cc,
		Ops:             reg,
		ReplicationMode: ReplicationModePB,
		PBControl:       fakeControl{node: "n1"},
	}
	s, err := New(cfg)
	if err != nil {
		t.Fatalf("New(pb): %v", err)
	}
	t.Cleanup(func() { _ = s.Close() })

	if !s.IsLeader() {
		t.Fatal("single-node PB primary must be leader")
	}
	if err := s.Put([]byte("k"), []byte("v"), 0); err != nil {
		t.Fatalf("Put: %v", err)
	}
	got, err := s.Get([]byte("k"))
	if err != nil || string(got) != "v" {
		t.Fatalf("Get: got=%q err=%v", got, err)
	}
}

// shardScopedControl is a static control plane whose Primary/Epoch answers
// depend on the queried shard index, unlike fakeControl (which answers the
// same regardless of shard). It exists to prove shard.New's PB branch
// threads cfg.ShardIndex through to Control lookups rather than hardcoding 0:
// a Store configured for a non-zero ShardIndex must consult THAT shard's
// primary, not shard 0's.
type shardScopedControl struct {
	primary map[int]string
}

func (c shardScopedControl) Epoch(int) uint64         { return 1 }
func (c shardScopedControl) Primary(shard int) string { return c.primary[shard] }
func (c shardScopedControl) ISR(shard int) []string   { return []string{c.primary[shard]} }
func (c shardScopedControl) MinISR(int) int           { return 1 }

// TestPBStoreRegistersEngine confirms that when Config.PBRegister is set,
// shard.New's PB branch calls it exactly once with (ShardIndex, eng), and
// that the passed receiver is a live, usable pbisr.Receiver.
func TestPBStoreRegistersEngine(t *testing.T) {
	reg := ops.NewRegistry()
	ops.RegisterBuiltins(reg)
	cc := cache.DefaultConfig()
	cc.NumShards = 1

	var calls int
	var gotShard int
	var gotReceiver pbisr.Receiver
	cfg := Config{
		NodeID:          "n1",
		DataDir:         t.TempDir(),
		Cache:           cc,
		Ops:             reg,
		ReplicationMode: ReplicationModePB,
		PBControl:       fakeControl{node: "n1"},
		PBRegister: func(shard int, r pbisr.Receiver) {
			calls++
			gotShard = shard
			gotReceiver = r
		},
	}
	s, err := New(cfg)
	if err != nil {
		t.Fatalf("New(pb): %v", err)
	}
	t.Cleanup(func() { _ = s.Close() })

	if calls != 1 {
		t.Fatalf("PBRegister calls: want 1, got %d", calls)
	}
	if gotShard != cfg.ShardIndex {
		t.Fatalf("PBRegister shard: want %d, got %d", cfg.ShardIndex, gotShard)
	}
	if gotReceiver == nil {
		t.Fatal("PBRegister receiver: want non-nil")
	}
	// The receiver must be usable: Receive returns an AckMsg without panicking
	// (the exact value is not asserted here — see TestPBReplicatorSatisfiesSeamAndApplies
	// for behavior coverage of Engine.Receive).
	_ = gotReceiver.Receive(pbisr.ReplicateMsg{})
}

// TestPBStoreShardIndexNonZero confirms shard.New's PB branch consults
// Config.ShardIndex (not a hardcoded 0) for Control lookups: a Store
// configured with ShardIndex=7, where shardScopedControl names "n1" primary
// of shard 7 (and someone else primary of shard 0), must become leader and
// accept writes.
func TestPBStoreShardIndexNonZero(t *testing.T) {
	reg := ops.NewRegistry()
	ops.RegisterBuiltins(reg)
	cc := cache.DefaultConfig()
	cc.NumShards = 1
	ctrl := shardScopedControl{primary: map[int]string{
		0: "someone-else",
		7: "n1",
	}}
	cfg := Config{
		NodeID:          "n1",
		DataDir:         t.TempDir(),
		Cache:           cc,
		Ops:             reg,
		ReplicationMode: ReplicationModePB,
		PBControl:       ctrl,
		ShardIndex:      7,
	}
	s, err := New(cfg)
	if err != nil {
		t.Fatalf("New(pb): %v", err)
	}
	t.Cleanup(func() { _ = s.Close() })

	if !s.IsLeader() {
		t.Fatal("shard-7 primary must be leader (proves ShardIndex, not 0, was consulted)")
	}
	if err := s.Put([]byte("k"), []byte("v"), 0); err != nil {
		t.Fatalf("Put: %v", err)
	}
	got, err := s.Get([]byte("k"))
	if err != nil || string(got) != "v" {
		t.Fatalf("Get: got=%q err=%v", got, err)
	}
}
