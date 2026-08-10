// SPDX-License-Identifier: Apache-2.0

package cluster

import (
	"encoding/json"
	"strings"
	"testing"
	"time"

	"github.com/rostamlabs/rostam/ops"
)

// replMetricsBody mirrors the __repl_metrics__ JSON the handler emits (the
// handler's structs are unexported, so the test decodes into a local twin).
type replMetricsBody struct {
	Node   string `json:"node"`
	Shards []struct {
		Shard           int               `json:"shard"`
		Mode            string            `json:"mode"`
		IsPrimary       bool              `json:"is_primary"`
		Primary         string            `json:"primary"`
		Epoch           uint64            `json:"epoch"`
		ISRSize         int               `json:"isr_size"`
		MinISR          int               `json:"min_isr"`
		UnderReplicated bool              `json:"under_replicated"`
		PlacementSize   int               `json:"placement_size"`
		BelowPlacement  bool              `json:"below_placement"`
		GrowAborts      map[string]uint64 `json:"grow_aborts"`
		LastSeq         uint64            `json:"last_seq"`
		Committed       uint64            `json:"committed"`
		Backups         []struct {
			Node  string `json:"node"`
			Acked uint64 `json:"acked_seq"`
			Lag   uint64 `json:"lag"`
		} `json:"backups"`
	} `json:"shards"`
}

// findShardPrimaryNode finds the node index that is the current primary of shard
// sh and to which a Put succeeds, mirroring the polling in
// TestStaticPBClusterReplicatesAndReads (the self-lease is granted only once the
// leaseKeeper observes the FSM primary).
func findShardPrimaryNode(t *testing.T, tc *pbTestCluster, sh int, putArgs []byte) int {
	t.Helper()
	deadline := time.Now().Add(15 * time.Second)
	for time.Now().Before(deadline) {
		primary := tc.nodes[0].meta.FSM.ShardPrimary(sh)
		if primary == "" {
			time.Sleep(50 * time.Millisecond)
			continue
		}
		idx := -1
		for i, p := range tc.peers {
			if p.NodeID == primary {
				idx = i
				break
			}
		}
		if idx < 0 {
			t.Fatalf("primary %q not in peer list", primary)
		}
		if _, err := tc.nodes[idx].Call("put", putArgs); err == nil {
			return idx
		}
		time.Sleep(50 * time.Millisecond)
	}
	t.Fatal("no shard primary accepted a write within 15s")
	return -1
}

// TestReplMetricsReportsISRAndLag brings up a 3-node PB cluster (RF=full,
// MinISR=2), writes to a shard's primary, and asserts __repl_metrics__ on the
// primary node reports that shard as PB-mode, fully in-sync (ISR=3 >= min-ISR=2,
// not under-replicated), and with its two backups tracked (acked >= 1).
func TestReplMetricsReportsISRAndLag(t *testing.T) {
	const numShards = 4
	tc := newPBTestCluster(t, 3, numShards, 2)

	key := []byte("repl-metrics-key")
	sh := shardOf(key, numShards)
	putArgs := ops.EncodePutArgs(key, []byte("v"), 0)
	primaryIdx := findShardPrimaryNode(t, tc, sh, putArgs)

	body, err := tc.nodes[primaryIdx].Call(ops.ReplMetricsOp, nil)
	if err != nil {
		t.Fatalf("__repl_metrics__: %v", err)
	}
	var rm replMetricsBody
	if err := json.Unmarshal(body, &rm); err != nil {
		t.Fatalf("unmarshal %s: %v", body, err)
	}
	if rm.Node != tc.peers[primaryIdx].NodeID {
		t.Fatalf("node = %q, want %q", rm.Node, tc.peers[primaryIdx].NodeID)
	}
	// Every hosted shard is reported; find the one under test.
	var found bool
	for _, s := range rm.Shards {
		if s.Shard != sh {
			continue
		}
		found = true
		if s.Mode != ReplicationModePB {
			t.Fatalf("mode = %q, want pb", s.Mode)
		}
		if !s.IsPrimary {
			t.Fatalf("is_primary = false on the primary node")
		}
		if s.ISRSize != 3 || s.MinISR != 2 {
			t.Fatalf("isr_size/min_isr = %d/%d, want 3/2", s.ISRSize, s.MinISR)
		}
		if s.UnderReplicated {
			t.Fatalf("under_replicated = true with full ISR")
		}
		// The two backups must be tracked, each having acked the committed write.
		if len(s.Backups) != 2 {
			t.Fatalf("backups = %d, want 2", len(s.Backups))
		}
		for _, b := range s.Backups {
			if b.Acked < 1 {
				t.Fatalf("backup %s acked = %d, want >= 1", b.Node, b.Acked)
			}
		}
	}
	if !found {
		t.Fatalf("shard %d not in repl metrics: %s", sh, body)
	}
}

// TestReadinessFailsOnUnderReplicatedShard verifies the #3 linkage: after a
// shard's ISR is shrunk below min-ISR, the __ready__ probe on a node hosting it
// reports NOT ready (an under-replicated shard cannot durably commit).
func TestReadinessFailsOnUnderReplicatedShard(t *testing.T) {
	const numShards = 4
	tc := newPBTestCluster(t, 3, numShards, 2)

	key := []byte("ready-isr-key")
	sh := shardOf(key, numShards)
	putArgs := ops.EncodePutArgs(key, []byte("v"), 0)
	primaryIdx := findShardPrimaryNode(t, tc, sh, putArgs)
	primary := tc.nodes[primaryIdx]

	// Once the bootstrap ISR seeding converges (every hosted shard reaches its
	// full seeded ISR), the node is ready. Poll rather than assert immediately —
	// seeding is asynchronous, so a shard OTHER than the target can still be
	// mid-seed right after startup and would fail readiness spuriously.
	readyDeadline := time.Now().Add(10 * time.Second)
	for {
		if _, err := primary.Call(ops.ReadyOp, nil); err == nil {
			break
		} else if !time.Now().Before(readyDeadline) {
			t.Fatalf("node not ready with full seeded ISR within 10s: %v", err)
		}
		time.Sleep(50 * time.Millisecond)
	}

	// Reset the shard's ISR to just the primary (below min-ISR=2) via an EPOCH BUMP:
	// OpSetShardEpoch resets the ISR to {primary}, which is exactly how an
	// under-replicated ISR actually arises (a failover). A direct OpSetShardISR below
	// the durability floor is now REJECTED by the FSM's structural min-ISR floor
	// (ISR hardening — see TestFSMMinISRFloorStructural), so it can no longer be
	// used to synthesize this state. The apply is monotonic (epoch+1), on whichever
	// node is the meta-Raft leader.
	epoch := primary.meta.FSM.ShardEpoch(sh)
	primaryID := primary.meta.FSM.ShardPrimary(sh)
	var applied bool
	for _, nd := range tc.nodes {
		if err := nd.meta.ApplySetShardEpoch(sh, epoch+1, primaryID, 5*time.Second); err == nil {
			applied = true
			break
		}
	}
	if !applied {
		t.Fatal("could not apply epoch bump on any node (no meta leader?)")
	}

	// Wait for the primary node's own FSM replica to observe the shrink, then the
	// readiness probe must fail closed with the under-replicated reason.
	deadline := time.Now().Add(10 * time.Second)
	for len(primary.meta.FSM.ShardISR(sh)) >= 2 {

		if !time.Now().Before(deadline) {
			t.Fatal("ISR shrink did not propagate to the primary node within 10s")
		}
		time.Sleep(50 * time.Millisecond)
	}
	_, err := primary.Call(ops.ReadyOp, nil)
	if err == nil {
		t.Fatal("ready = nil with an under-replicated shard, want not-ready")
	}
	if !strings.Contains(err.Error(), "under-replicated") {
		t.Fatalf("not-ready reason = %q, want it to mention under-replicated", err.Error())
	}
}

// TestReplMetricsBelowPlacementInvisibleCase reproduces the audit's
// motivating case: with min-ISR=1, an ISR reset to {primary} on a
// full-replication (placement=3) shard is NOT under-replicated (1 >= 1
// floor), so before this stage it was reported as fully healthy on every
// surface — {"isr_size":1,"min_isr":1,"under_replicated":false} and
// __ready__ "ready" — while the shard has silently lost two of its three
// replicas. This asserts the new below_placement field (ISRSize <
// PlacementSize) makes exactly that condition visible, while leaving
// under_replicated's existing floor-only meaning untouched.
func TestReplMetricsBelowPlacementInvisibleCase(t *testing.T) {
	const numShards = 4
	const wantPlacement = 3 // RF=0 (full replication) over a 3-node cluster
	tc := newPBTestCluster(t, 3, numShards, 1)

	key := []byte("below-placement-key")
	sh := shardOf(key, numShards)
	putArgs := ops.EncodePutArgs(key, []byte("v"), 0)
	primaryIdx := findShardPrimaryNode(t, tc, sh, putArgs)
	primary := tc.nodes[primaryIdx]

	// This test asserts placement_size == 3 below, so the placement table must
	// actually be visible first — it is not guaranteed to be. See
	// requireVisiblePlacement.
	requireVisiblePlacement(t, tc.nodes, primary, sh, numShards)

	// Sanity: before the reset, fully in sync — neither signal fires.
	body, err := primary.Call(ops.ReplMetricsOp, nil)
	if err != nil {
		t.Fatalf("__repl_metrics__ (pre-reset): %v", err)
	}
	var pre replMetricsBody
	if err := json.Unmarshal(body, &pre); err != nil {
		t.Fatalf("unmarshal %s: %v", body, err)
	}
	for _, s := range pre.Shards {
		if s.Shard != sh {
			continue
		}
		if s.PlacementSize != wantPlacement {
			t.Fatalf("pre-reset placement_size = %d, want %d", s.PlacementSize, wantPlacement)
		}
		if s.BelowPlacement {
			t.Fatalf("pre-reset below_placement = true with a full ISR")
		}
	}

	// Reset the ISR to {primary} via the election-reset op (same technique as
	// TestReadinessFailsOnUnderReplicatedShard) — exactly how this state
	// actually arises (a failover), and the only way to reach an ISR below
	// placement now that a direct below-floor OpSetShardISR is FSM-rejected.
	epoch := primary.meta.FSM.ShardEpoch(sh)
	primaryID := primary.meta.FSM.ShardPrimary(sh)
	var applied bool
	for _, nd := range tc.nodes {
		if err := nd.meta.ApplySetShardEpoch(sh, epoch+1, primaryID, 5*time.Second); err == nil {
			applied = true
			break
		}
	}
	if !applied {
		t.Fatal("could not apply epoch bump on any node (no meta leader?)")
	}

	deadline := time.Now().Add(10 * time.Second)
	for len(primary.meta.FSM.ShardISR(sh)) >= wantPlacement {

		if !time.Now().Before(deadline) {
			t.Fatal("ISR reset did not propagate to the primary node within 10s")
		}
		time.Sleep(50 * time.Millisecond)
	}

	// __ready__ must stay READY: min-ISR=1 is still met (#5 constraint — this
	// stage does not change readiness semantics, floor-only).
	if _, err := primary.Call(ops.ReadyOp, nil); err != nil {
		t.Fatalf("ready = %v with ISR(1) >= min-ISR(1); readiness semantics must stay floor-only", err)
	}

	body, err = primary.Call(ops.ReplMetricsOp, nil)
	if err != nil {
		t.Fatalf("__repl_metrics__ (post-reset): %v", err)
	}
	var post replMetricsBody
	if err := json.Unmarshal(body, &post); err != nil {
		t.Fatalf("unmarshal %s: %v", body, err)
	}
	var found bool
	for _, s := range post.Shards {
		if s.Shard != sh {
			continue
		}
		found = true
		if s.ISRSize != 1 || s.MinISR != 1 {
			t.Fatalf("isr_size/min_isr = %d/%d, want 1/1", s.ISRSize, s.MinISR)
		}
		// THE INVISIBLE-CASE ASSERTION: under_replicated's meaning is UNCHANGED
		// (floor-only) — false here — while below_placement is the new, separate
		// signal that makes the lost redundancy visible.
		if s.UnderReplicated {
			t.Fatalf("under_replicated = true at ISR(1) >= min-ISR(1); its meaning must stay floor-only")
		}
		if s.PlacementSize != wantPlacement {
			t.Fatalf("placement_size = %d, want %d", s.PlacementSize, wantPlacement)
		}
		if !s.BelowPlacement {
			t.Fatalf("below_placement = false with ISR(1) < placement(%d) — the invisible case is still invisible", wantPlacement)
		}
	}
	if !found {
		t.Fatalf("shard %d not in repl metrics: %s", sh, body)
	}
}
