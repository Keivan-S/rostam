// SPDX-License-Identifier: Apache-2.0

package cluster

import (
	"errors"
	"testing"
	"time"

	"github.com/rostamlabs/rostam/ops"
	"github.com/rostamlabs/rostam/shard"
)

// TestPBBackupReturnsFollowableLeaderHint is the regression for a write sent to
// a node that hosts the target shard as a BACKUP under PB replication.
//
// Such a write cannot be served locally, so it must come back as a
// NotLeaderError carrying the PRIMARY's client-facing server address — the hint
// an HTTP or gRPC caller follows. It used to come back with the hint DROPPED
// (bare "shard: not leader"), because callHostedShard resolved the identifier
// only through raftToServerAddr, which matches RAFT TRANSPORT addresses, while
// pbReplicator.LeaderAddr returns ctrl.Primary(shard) — a NODE ID. No match
// meant no hint.
//
// The native client masked this (it routes from its own __topology__ cache and
// falls back to round-robin), which is why the KV benchmarks never saw it; an
// HTTP caller simply failed. At RF=2 across 3 nodes that is 2-3 of every 8
// shards for any single endpoint.
//
// The assertion is deliberately on the ADDRESS, not merely on non-emptiness: a
// hint that is present but unfollowable is the bug wearing a disguise.
func TestPBBackupReturnsFollowableLeaderHint(t *testing.T) {
	const numShards = 4
	tc := newPBTestCluster(t, 3, numShards, 2)

	key := []byte("pb-hint-key")
	sh := shardOf(key, numShards)
	putArgs := ops.EncodePutArgs(key, []byte("v"), 0)

	// Same readiness idiom as TestStaticPBClusterReplicatesAndReads: poll until a
	// write to the designated primary actually succeeds, so the lease is live
	// before we make any claim about what a NON-primary returns.
	primaryIdx := -1
	var lastErr error
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
		_, err := tc.nodes[idx].Call("put", putArgs)
		if err == nil {
			primaryIdx = idx
			break
		}
		lastErr = err
		time.Sleep(50 * time.Millisecond)
	}
	if primaryIdx < 0 {
		t.Fatalf("PB write to primary never succeeded within 15s (lastErr=%v)", lastErr)
	}

	// A backup is a node that HOSTS the shard but is not its primary. This is the
	// case the bug hit; a node that does not host the shard at all takes
	// Node.forward instead and was never affected.
	backupIdx := -1
	for i, nd := range tc.nodes {
		if i == primaryIdx {
			continue
		}
		if nd.getShard(sh) != nil {
			backupIdx = i
			break
		}
	}
	if backupIdx < 0 {
		t.Fatalf("no node hosts shard %d as a backup; harness cannot exercise the case", sh)
	}

	_, err := tc.nodes[backupIdx].Call("put", putArgs)
	if err == nil {
		t.Fatalf("write on backup %s unexpectedly succeeded; it is not the primary",
			tc.peers[backupIdx].NodeID)
	}
	var nle *shard.NotLeaderError
	if !errors.As(err, &nle) {
		t.Fatalf("backup returned %v (%T); want *shard.NotLeaderError", err, err)
	}
	if nle.LeaderAddr == "" {
		t.Fatalf("backup %s dropped the leader hint: an HTTP/gRPC caller has no address to follow",
			tc.peers[backupIdx].NodeID)
	}
	if want := tc.peers[primaryIdx].ServerAddr; nle.LeaderAddr != want {
		t.Fatalf("leader hint = %q; want the primary's client-facing server addr %q",
			nle.LeaderAddr, want)
	}
}

// TestPBLeaderFollowingDispatcherServesOnBackup pins the mechanism that the
// partitioned-collection coordinator depends on: LeaderFollowingDispatcher must
// turn a backup's NotLeader into a SUCCESSFUL write, server-side.
//
// The dispatcher is gated on `nle.LeaderAddr != ""`. Under PB that address was
// always empty, so the wrapper — which HTTP and gRPC both rely on — silently did
// nothing in PB mode and every caller behind it failed. A test asserting only
// that the hint is non-empty (as the test above does) would NOT catch a
// regression that broke the wrapper itself, which is why this asserts the
// end-to-end outcome instead.
func TestPBLeaderFollowingDispatcherServesOnBackup(t *testing.T) {
	const numShards = 4
	tc := newPBTestCluster(t, 3, numShards, 2)

	key := []byte("pb-lf-key")
	sh := shardOf(key, numShards)
	putArgs := ops.EncodePutArgs(key, []byte("v"), 0)

	primaryIdx := -1
	deadline := time.Now().Add(15 * time.Second)
	for time.Now().Before(deadline) {
		primary := tc.nodes[0].meta.FSM.ShardPrimary(sh)
		if primary == "" {
			time.Sleep(50 * time.Millisecond)
			continue
		}
		for i, p := range tc.peers {
			if p.NodeID == primary {
				primaryIdx = i
			}
		}
		if primaryIdx >= 0 {
			if _, err := tc.nodes[primaryIdx].Call("put", putArgs); err == nil {
				break
			}
		}
		time.Sleep(50 * time.Millisecond)
	}
	if primaryIdx < 0 {
		t.Fatal("no primary for shard within 15s")
	}

	backupIdx := -1
	for i, nd := range tc.nodes {
		if i != primaryIdx && nd.getShard(sh) != nil {
			backupIdx = i
			break
		}
	}
	if backupIdx < 0 {
		t.Fatalf("no node hosts shard %d as a backup", sh)
	}

	// The bare node must still report NotLeader — the TCP client follows that hint
	// itself, and changing it would be a separate behavior change.
	if _, err := tc.nodes[backupIdx].Call("put", putArgs); err == nil {
		t.Fatal("bare Node.Call on a backup succeeded; expected a NotLeader hint")
	}
	// The wrapper must complete the same write.
	if _, err := tc.nodes[backupIdx].LeaderFollowingDispatcher().Call("put", putArgs); err != nil {
		t.Fatalf("LeaderFollowingDispatcher.Call on backup %s = %v; want success via server-side redirect",
			tc.peers[backupIdx].NodeID, err)
	}
}

// TestResolveLeaderHintAcceptsBothIdentifierForms pins the contract that made the
// bug possible: the identifier a replicator puts in NotLeaderError.LeaderAddr has
// a DIFFERENT FORM per engine — a raft transport address under Raft, a node id
// under PB — and only the cluster layer can resolve both.
//
// It also pins the precedence. Raft is tried first so that path stays
// byte-identical; a regression that flipped the order would be invisible to the
// test above (which only exercises PB) but would change Raft's behavior.
func TestResolveLeaderHintAcceptsBothIdentifierForms(t *testing.T) {
	n := &Node{cfg: Config{Peers: []Peer{
		{NodeID: "n1", RaftAddr: "127.0.0.1:7401", ServerAddr: "127.0.0.1:7001"},
		{NodeID: "n2", RaftAddr: "127.0.0.1:7402", ServerAddr: "127.0.0.1:7002"},
	}}}

	for _, tt := range []struct {
		name, hint, want string
	}{
		{"raft transport addr (raft path)", "127.0.0.1:7402", "127.0.0.1:7002"},
		{"node id (pb path)", "n2", "127.0.0.1:7002"},
		{"unknown identifier resolves to nothing", "n9", ""},
		{"empty identifier resolves to nothing", "", ""},
	} {
		t.Run(tt.name, func(t *testing.T) {
			if got := n.resolveLeaderHint(tt.hint); got != tt.want {
				t.Fatalf("resolveLeaderHint(%q) = %q; want %q", tt.hint, got, tt.want)
			}
		})
	}
}
