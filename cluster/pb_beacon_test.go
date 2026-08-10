// SPDX-License-Identifier: Apache-2.0

package cluster

import (
	"reflect"
	"testing"
	"time"

	"github.com/hashicorp/raft"
)

// applyShardLeaseRenew is a test helper that encodes and applies an
// OpShardLeaseRenew beacon entry (node + a batch of (shard,epoch) pairs).
func applyShardLeaseRenew(t *testing.T, f *MetaFSM, node string, renews []ShardEpochPair) {
	t.Helper()
	data, err := encodeLogEntry(LogEntry{Op: OpShardLeaseRenew, Node: node, LeaseRenew: renews})
	if err != nil {
		t.Fatal(err)
	}
	if resp := f.Apply(&raft.Log{Data: data}); resp != nil {
		t.Fatalf("apply ShardLeaseRenew(node=%s): %v", node, resp)
	}
}

// TestLogEntryShardLeaseRenewRoundtrip proves the beacon payload (node + batched
// (shard,epoch) pairs) gob round-trips, mirroring the other op round-trip tests.
func TestLogEntryShardLeaseRenewRoundtrip(t *testing.T) {
	in := LogEntry{
		Op:         OpShardLeaseRenew,
		Node:       "n2",
		LeaseRenew: []ShardEpochPair{{ShardID: 0, Epoch: 3}, {ShardID: 7, Epoch: 1}},
	}
	b, err := encodeLogEntry(in)
	if err != nil {
		t.Fatal(err)
	}
	got, err := decodeLogEntry(b)
	if err != nil {
		t.Fatal(err)
	}
	if got.Op != OpShardLeaseRenew || got.Node != "n2" {
		t.Fatalf("round-trip = %+v, want ShardLeaseRenew node=n2", got)
	}
	if !reflect.DeepEqual(got.LeaseRenew, in.LeaseRenew) {
		t.Fatalf("LeaseRenew round-trip = %v, want %v", got.LeaseRenew, in.LeaseRenew)
	}
}

// TestFSMApplyShardLeaseRenewGuard is the Stage-1 correctness test: applying an
// OpShardLeaseRenew beacon fires the leader-local observer ONLY for a pair whose
// node is exactly the shard's current committed primary at exactly its current
// epoch, and IGNORES every stale pair (wrong primary, wrong epoch, or a shard the
// node does not primary). It also proves the beacon mutates NO replicated state.
func TestFSMApplyShardLeaseRenewGuard(t *testing.T) {
	f := NewMetaFSM()
	// Establish the committed control plane: shard 0 -> (n1, epoch 1),
	// shard 1 -> (n2, epoch 2).
	applyShardEpoch(t, f, 0, 1, "n1")
	applyShardEpoch(t, f, 1, 2, "n2")

	// Record every observer firing.
	type fire struct {
		shard int
		epoch uint64
	}
	var fires []fire
	f.SetLeaseRenewObserver(func(shard int, epoch uint64) {
		fires = append(fires, fire{shard, epoch})
	})

	// Snapshot the replicated state so we can prove the beacon leaves it untouched.
	before := f.State()

	// A valid batch from n1: only shard 0 matches (n1 primaries shard 0 at epoch 1).
	// The (shard 1, epoch 2) pair claims a shard n1 does NOT primary → ignored.
	applyShardLeaseRenew(t, f, "n1", []ShardEpochPair{{0, 1}, {1, 2}})
	if !reflect.DeepEqual(fires, []fire{{0, 1}}) {
		t.Fatalf("after valid n1 beacon, fires = %+v, want [{0 1}]", fires)
	}

	// Stale-epoch pair (shard 0 at epoch 0, and at a future epoch 2): both ignored.
	fires = nil
	applyShardLeaseRenew(t, f, "n1", []ShardEpochPair{{0, 0}, {0, 2}})
	if len(fires) != 0 {
		t.Fatalf("stale-epoch beacon fired observer: %+v, want none", fires)
	}

	// Wrong-primary pair (n3 claims shard 0, which n1 primaries): ignored.
	fires = nil
	applyShardLeaseRenew(t, f, "n3", []ShardEpochPair{{0, 1}})
	if len(fires) != 0 {
		t.Fatalf("wrong-primary beacon fired observer: %+v, want none", fires)
	}

	// A valid batch from n2 for shard 1: fires exactly once.
	fires = nil
	applyShardLeaseRenew(t, f, "n2", []ShardEpochPair{{1, 2}})
	if !reflect.DeepEqual(fires, []fire{{1, 2}}) {
		t.Fatalf("after valid n2 beacon, fires = %+v, want [{1 2}]", fires)
	}

	// The beacon is INERT in the FSM: epoch/primary/ISR maps are byte-identical.
	after := f.State()
	if !reflect.DeepEqual(before.ShardEpoch, after.ShardEpoch) ||
		!reflect.DeepEqual(before.ShardPrimary, after.ShardPrimary) ||
		!reflect.DeepEqual(before.ShardISR, after.ShardISR) {
		t.Fatalf("beacon mutated replicated state:\n before %+v/%+v/%+v\n after  %+v/%+v/%+v",
			before.ShardEpoch, before.ShardPrimary, before.ShardISR,
			after.ShardEpoch, after.ShardPrimary, after.ShardISR)
	}

	// With NO observer set, the beacon is a pure no-op (no panic, no state change).
	f.SetLeaseRenewObserver(nil)
	applyShardLeaseRenew(t, f, "n1", []ShardEpochPair{{0, 1}})
}

// TestPBBeaconRenewalsObservedOnLeader is the Stage-2 integration test: with
// PBAutoFailover ON, every node runs a beacon (a follower-hosted primary forwards
// its OpShardLeaseRenew to the meta leader), and after a couple of intervals the
// meta leader's pbFailoverTracker has observed a renewal — at the shard's current
// committed epoch — for every seeded shard. This exercises the batched op, the
// forward-to-leader path, the guarded FSM apply, and the observer end to end.
func TestPBBeaconRenewalsObservedOnLeader(t *testing.T) {
	const numShards = 4
	tc := newPBTestCluster(t, 3, numShards, 2, func(c *Config) {
		c.PBAutoFailover = true
		c.PBRenewIntervalMs = 200 // beacon briskly so the test is quick
	})

	// Find the meta leader node (the one whose tracker the failover ticker consumes).
	leaderIdx := func() int {
		for i, nd := range tc.nodes {
			if nd.meta.Raft.State() == raft.Leader {
				return i
			}
		}
		return -1
	}

	deadline := time.Now().Add(15 * time.Second)
	for {
		li := leaderIdx()
		if li >= 0 {
			leader := tc.nodes[li]
			// Every seeded shard must have (a) a committed primary and (b) a beacon
			// observed on the leader's tracker at the shard's CURRENT epoch.
			allObserved := true
			for sh := 0; sh < numShards; sh++ {
				epoch := leader.meta.FSM.ShardEpoch(sh)
				primary := leader.meta.FSM.ShardPrimary(sh)
				if primary == "" {
					allObserved = false
					break
				}
				_, obsEp, ok := leader.pbTracker.observed(sh)
				if !ok || obsEp != epoch {
					allObserved = false
					break
				}
			}
			if allObserved {
				return // success
			}
		}
		if !time.Now().Before(deadline) {
			li := leaderIdx()
			if li < 0 {
				t.Fatal("no meta leader within 15s")
			}
			leader := tc.nodes[li]
			for sh := 0; sh < numShards; sh++ {
				ns, ep, ok := leader.pbTracker.observed(sh)
				t.Logf("shard %d: primary=%q epoch=%d observed=(ns=%d ep=%d ok=%v)",
					sh, leader.meta.FSM.ShardPrimary(sh), leader.meta.FSM.ShardEpoch(sh), ns, ep, ok)
			}
			t.Fatal("leader tracker did not observe all shard renewals within 15s")
		}
		time.Sleep(50 * time.Millisecond)
	}
}
