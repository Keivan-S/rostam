// SPDX-License-Identifier: Apache-2.0

package cluster

import (
	"context"
	"fmt"
	"testing"
	"time"

	"github.com/rostamlabs/rostam/ops"
)

// TestSlice1ShardMembershipPrimitive exercises the online-rebalancing slice-1
// primitives on a live cluster: a node that does NOT host a shard joins it
// (AddShardOwner + the leader's ShardAddVoter), catches up the shard's data via
// Raft, and serves it locally; then it is removed again.
func TestSlice1ShardMembershipPrimitive(t *testing.T) {
	// RF=1, 2 nodes, 2 shards: shard 0 is owned only by n1, shard 1 only by n2.
	tc := newTestCluster(t, 2, 2, 1)
	ctx := context.Background()

	n1, n2 := tc.nodes[0], tc.nodes[1] // NodeIDs "n1","n2"; n1 owns shard 0
	if n1.getShard(0) == nil {
		t.Fatal("precondition: n1 should own shard 0")
	}
	if n2.getShard(0) != nil {
		t.Fatal("precondition: n2 should NOT own shard 0")
	}

	// Seed a key that routes to shard 0 (written via the client → shard 0 leader).
	var key []byte
	for i := 0; ; i++ {
		k := fmt.Appendf(nil, "key-%d", i)
		if shardOf(k, 2) == 0 {
			key = k
			break
		}
	}
	val := []byte("payload")
	if _, err := tc.client.Call(ctx, "put", ops.EncodePutArgs(key, val, 0)); err != nil {
		t.Fatalf("seed put: %v", err)
	}

	// --- Join: n2 becomes an owner of shard 0. ---
	if err := n2.AddShardOwner(0); err != nil { // create the store (join mode)
		t.Fatalf("AddShardOwner: %v", err)
	}
	if n2.getShard(0) == nil {
		t.Fatal("n2 should host shard 0 after AddShardOwner")
	}
	// n1 (shard 0's leader) adds n2 as a voter → Raft streams it the snapshot+log.
	if err := n1.ShardAddVoter(0, "n2", tc.peers[1].RaftAddr); err != nil {
		t.Fatalf("ShardAddVoter: %v", err)
	}

	// n2's shard-0 replica must catch up and serve the seeded key locally.
	deadline := time.Now().Add(10 * time.Second)
	var got []byte
	for time.Now().Before(deadline) {
		if v, err := n2.getShard(0).Get(key); err == nil {
			got = v
			break
		}
		time.Sleep(50 * time.Millisecond)
	}
	if string(got) != string(val) {
		t.Fatalf("n2 shard-0 catch-up: got %q, want %q", got, val)
	}

	// --- Leave: remove n2 from shard 0 again. ---
	if err := n1.ShardRemoveVoter(0, "n2"); err != nil {
		t.Fatalf("ShardRemoveVoter: %v", err)
	}
	if err := n2.RemoveShardOwner(0); err != nil {
		t.Fatalf("RemoveShardOwner: %v", err)
	}
	if n2.getShard(0) != nil {
		t.Fatal("n2 should not host shard 0 after RemoveShardOwner")
	}

	// The shard is still served by n1, reachable from either node (n2 forwards).
	if v, err := tc.client.Call(ctx, "get", ops.EncodeKeyArgs(key)); err != nil || string(v) != string(val) {
		t.Fatalf("post-remove get = %q,%v, want %q", v, err, val)
	}
}
