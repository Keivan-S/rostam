// SPDX-License-Identifier: Apache-2.0

package cluster

import (
	"errors"
	"testing"
	"time"

	"github.com/rostamlabs/rostam/ops"
	"github.com/rostamlabs/rostam/shard"
)

// TestLeaderFollowingDispatcher verifies the HTTP/gRPC leader-redirect: a write
// that lands on a follower of a hosted shard returns a NotLeaderError through
// plain Node.Call (the binary-hint path the TCP client follows), but the
// LeaderFollowingDispatcher forwards it to the leader and succeeds — so HTTP/gRPC
// clients that can't act on the hint still get their write committed.
func TestLeaderFollowingDispatcher(t *testing.T) {
	tc := newTestCluster(t, 3, 4) // full replication: every node hosts every shard
	key := []byte("lf-key")
	sh := shardOf(key, 4)

	leaderIdx, ok := tc.findShardLeader(sh)
	if !ok {
		t.Fatalf("no leader for shard %d", sh)
	}
	followerIdx := (leaderIdx + 1) % 3
	follower := tc.nodes[followerIdx]
	if follower.getShard(sh) == nil {
		t.Fatal("follower should host the shard (full replication)")
	}
	if follower.shardIsLeader(sh) {
		t.Fatal("picked node is the leader, not a follower")
	}

	val := []byte("payload")
	putArgs := ops.EncodePutArgs(key, val, 0)

	// Plain Call on a follower surfaces NotLeaderError — the gap HTTP/gRPC hit.
	if _, err := follower.Call("put", putArgs); err == nil {
		t.Fatal("plain follower.Call(put) should return NotLeaderError")
	} else {
		var nle *shard.NotLeaderError
		if !errors.As(err, &nle) {
			t.Fatalf("plain follower.Call(put): want NotLeaderError, got %v", err)
		}
	}

	// The leader-following dispatcher forwards the write to the leader → success.
	wd := follower.LeaderFollowingDispatcher()
	if _, err := wd.Call("put", putArgs); err != nil {
		t.Fatalf("leader-following put via follower: %v", err)
	}

	// The committed write replicates back to the follower's local store.
	replicated := false
	deadline := time.Now().Add(5 * time.Second)
	for time.Now().Before(deadline) {
		if v, err := follower.getShard(sh).Get(key); err == nil && string(v) == string(val) {
			replicated = true
			break
		}
		time.Sleep(50 * time.Millisecond)
	}
	if !replicated {
		t.Fatal("forwarded write did not replicate back to the follower")
	}

	// A read through the dispatcher also returns the value.
	got, err := wd.Call("get", ops.EncodeKeyArgs(key))
	if err != nil || string(got) != string(val) {
		t.Fatalf("get via dispatcher = %q, %v; want %q", got, err, val)
	}
}
