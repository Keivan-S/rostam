// SPDX-License-Identifier: Apache-2.0

package cluster

import (
	"errors"
	"testing"
	"time"

	"github.com/rostamlabs/rostam/ops"
)

// writeToShardLeader applies a put for key (which routes to shard sh) via the
// shard's current leader, returning the leader's node index. It uses the
// leader-following dispatcher so the write commits at majority regardless of
// which node we start from. It does NOT wait for followers to apply — callers
// that need that use waitOwnersApplied or let the barrier itself do the waiting.
// The returned leaderIdx is the node hosting the shard leader.
func writeToShardLeader(t *testing.T, tc *testCluster, sh int, key, val []byte) int {
	t.Helper()
	leaderIdx, ok := tc.findShardLeader(sh)
	if !ok {
		t.Fatalf("no leader for shard %d", sh)
	}
	wd := tc.nodes[leaderIdx].LeaderFollowingDispatcher()
	if _, err := wd.Call("put", ops.EncodePutArgs(key, val, 0)); err != nil {
		t.Fatalf("write to shard %d leader: %v", sh, err)
	}
	return leaderIdx
}

// waitOwnersApplied blocks until every live owner of sh has applied at least
// target, or the deadline elapses (failing the test). It is the happy-path
// ground truth the barrier is supposed to wait for.
func waitOwnersApplied(t *testing.T, tc *testCluster, sh int, target uint64, d time.Duration) {
	t.Helper()
	leader := tc.nodes[0]
	owners := leader.ownersFor(sh)
	deadline := time.Now().Add(d)
	for time.Now().Before(deadline) {
		all := true
		for _, owner := range owners {
			if leader.appliedReplicas(sh, []string{owner}, target) == 0 {
				all = false
				break
			}
		}
		if all {
			return
		}
		time.Sleep(20 * time.Millisecond)
	}
	t.Fatalf("owners of shard %d did not all reach applied index %d within %s", sh, target, d)
}

// TestBarrierForShardSatisfied (case a): after a normal write to shard S on a
// 3-node RF=3 cluster, BarrierForShard(S, 3, true, 2s) returns nil and all three
// owners' applied index has caught up to the leader's.
func TestBarrierForShardSatisfied(t *testing.T) {
	tc := newTestCluster(t, 3, 4, 3) // RF=3: every node owns every shard
	key := []byte("wc-sat")
	sh := shardOf(key, 4)

	leaderIdx := writeToShardLeader(t, tc, sh, key, []byte("v"))
	leader := tc.nodes[leaderIdx]
	target := leader.localStatus(sh).AppliedIndex
	if target == 0 {
		t.Fatalf("leader applied index unexpectedly 0 after write")
	}

	if err := tc.nodes[0].BarrierForShard(sh, 3, true, 2*time.Second); err != nil {
		t.Fatalf("BarrierForShard(S,3,true): %v", err)
	}

	// Independent ground truth: every owner has applied >= target.
	owners := tc.nodes[0].ownersFor(sh)
	if len(owners) != 3 {
		t.Fatalf("RF: want 3 owners, got %d (%v)", len(owners), owners)
	}
	for _, owner := range owners {
		if tc.nodes[0].appliedReplicas(sh, []string{owner}, target) != 1 {
			t.Errorf("owner %s did not reach applied index %d", owner, target)
		}
	}
}

// TestBarrierForShardNoOpFastPath (case b + d): factors <= majority and
// wait=false are immediate no-ops (eff <= maj=2 for RF=3). Each must return nil
// in well under the timeout — proving no poll loop ran.
func TestBarrierForShardNoOpFastPath(t *testing.T) {
	tc := newTestCluster(t, 3, 4, 3)
	key := []byte("wc-noop")
	sh := shardOf(key, 4)
	writeToShardLeader(t, tc, sh, key, []byte("v"))

	cases := []struct {
		name string
		wcf  uint8
		wait bool
	}{
		{"wcf0", 0, true},
		{"wcf1", 1, true},
		{"wcf2-eq-majority", 2, true},
		{"wait-false", 3, false},
	}
	for _, c := range cases {
		t.Run(c.name, func(t *testing.T) {
			start := time.Now()
			// A 10s timeout would be paid only if the barrier actually polled;
			// a no-op returns immediately.
			if err := tc.nodes[0].BarrierForShard(sh, c.wcf, c.wait, 10*time.Second); err != nil {
				t.Fatalf("BarrierForShard(%d,%v): %v", c.wcf, c.wait, err)
			}
			if el := time.Since(start); el > 50*time.Millisecond {
				t.Fatalf("expected no-op fast path, took %s (poll loop ran)", el)
			}
		})
	}
}

// TestBarrierForShardTimeout (case c): with one owner of shard S down so its
// applied index can never reach the target, BarrierForShard(S,3,true,300ms)
// returns *ErrWriteConsistency with Applied==2; the write remains durable and
// readable on the two live owners.
//
// Follower-lag induction: closing one owner node removes it from the live set
// while leaving the placement (owner list) at 3 — its remote __rb_status__ poll
// then fails and degrades to a zero-value status (AppliedIndex 0 < target), so it
// never counts. This is a realistic partition/down-replica scenario, preferred
// over a synthetic future target.
func TestBarrierForShardTimeout(t *testing.T) {
	tc := newTestCluster(t, 3, 4, 3)
	key := []byte("wc-timeout")
	val := []byte("durable")
	sh := shardOf(key, 4)

	leaderIdx := writeToShardLeader(t, tc, sh, key, val)
	leader := tc.nodes[leaderIdx]
	target := leader.localStatus(sh).AppliedIndex
	// Make sure all three owners have it before we kill one, so durability is
	// proven on BOTH survivors regardless of which non-leader we down.
	waitOwnersApplied(t, tc, sh, target, 5*time.Second)

	// Pick a non-leader owner to take down. Closing its SERVER (TCP listener)
	// makes the survivor's __rb_status__ poll to it fail at the transport — which
	// degrades to a zero-value adminStatus (AppliedIndex 0 < target), so it never
	// counts. (Closing only the Node would leave the server answering from the
	// node's last-applied state, which would still satisfy the target.) We also
	// close the Node to free its resources; both are nilled to avoid double-close.
	downIdx := (leaderIdx + 1) % 3
	if err := tc.servers[downIdx].Close(); err != nil {
		t.Fatalf("closing owner %d server: %v", downIdx, err)
	}
	tc.servers[downIdx] = nil
	if err := tc.nodes[downIdx].Close(); err != nil {
		t.Fatalf("closing owner %d node: %v", downIdx, err)
	}
	tc.nodes[downIdx] = nil // prevent double-close in tc.Close()

	// From a surviving node, the barrier can never reach 3 (the downed owner's
	// status poll fails forever) → timeout with Applied==2.
	var survivor *Node
	for i, nd := range tc.nodes {
		if nd != nil && i != downIdx {
			survivor = nd
			break
		}
	}
	start := time.Now()
	err := survivor.BarrierForShard(sh, 3, true, 300*time.Millisecond)
	el := time.Since(start)
	var wcErr *ErrWriteConsistency
	if !errors.As(err, &wcErr) {
		t.Fatalf("BarrierForShard(S,3) with one owner down: want *ErrWriteConsistency, got %v", err)
	}
	if wcErr.Applied != 2 {
		t.Errorf("ErrWriteConsistency.Applied = %d, want 2 (leader + 1 live follower)", wcErr.Applied)
	}
	if wcErr.Requested != 3 {
		t.Errorf("ErrWriteConsistency.Requested = %d, want 3", wcErr.Requested)
	}
	if el < 250*time.Millisecond {
		t.Errorf("barrier returned in %s, expected to poll until ~300ms deadline", el)
	}

	// Durability: the write is still present on both live owners' local stores.
	live := 0
	for i, nd := range tc.nodes {
		if nd == nil || i == downIdx {
			continue
		}
		s := nd.getShard(sh)
		if s == nil {
			continue
		}
		if v, gerr := s.Get(key); gerr == nil && string(v) == string(val) {
			live++
		}
	}
	if live != 2 {
		t.Errorf("write durable on %d live owners, want 2", live)
	}
}

// TestBarrierSingleNodeNoOp: RF=1 (single-node cluster) ⇒ majority == RF == 1,
// so WCF can never exceed majority and the barrier never engages, for any factor.
func TestBarrierSingleNodeNoOp(t *testing.T) {
	tc := newTestCluster(t, 1, 4) // 1 node, RF defaults so each shard has 1 owner
	key := []byte("wc-single")
	sh := shardOf(key, 4)
	writeToShardLeader(t, tc, sh, key, []byte("v"))

	if got := len(tc.nodes[0].ownersFor(sh)); got != 1 {
		t.Fatalf("single-node RF: want 1 owner, got %d", got)
	}
	for _, wcf := range []uint8{0, 1, 2, 3, 255} {
		start := time.Now()
		if err := tc.nodes[0].BarrierForShard(sh, wcf, true, 10*time.Second); err != nil {
			t.Fatalf("single-node BarrierForShard(%d): %v", wcf, err)
		}
		if el := time.Since(start); el > 50*time.Millisecond {
			t.Fatalf("single-node barrier wcf=%d took %s, want immediate no-op", wcf, el)
		}
	}
}

// TestShardIndexForName confirms ShardIndexForName reuses the routing path's
// canonicalization + hash: the index for a bare name equals shardOf over the
// canonical ("default/<name>") form, which is exactly what the vector key
// extractor produces for a write to that collection.
func TestShardIndexForName(t *testing.T) {
	tc := newTestCluster(t, 1, 8)
	n := tc.nodes[0]
	cases := []string{"docs", "default/docs", "tenant/coll", "x"}
	for _, name := range cases {
		want := shardOf([]byte(ops.CanonicalName(name)), 8)
		if got := n.ShardIndexForName(name); got != want {
			t.Errorf("ShardIndexForName(%q) = %d, want %d (shardOf canonical)", name, got, want)
		}
	}
	// A bare name and its "default/"-qualified form must map to the same shard
	// (they are the same collection), matching vectorRouteKey.
	if n.ShardIndexForName("docs") != n.ShardIndexForName("default/docs") {
		t.Error("bare and qualified name disagree on shard index")
	}
}
