// SPDX-License-Identifier: Apache-2.0

package cluster

import (
	"bytes"
	"context"
	"sync"
	"testing"
	"time"

	"github.com/rostamlabs/rostam/ops"
	"github.com/rostamlabs/rostam/shard"
)

// readRecorder captures every OpReadOnly serve reported via
// shard.SetReadServedHook, recording whether the serving replica was the
// shard leader. Mutex-guarded so it is safe under concurrent serves.
type readRecorder struct {
	mu     sync.Mutex
	serves []bool // each entry: isLeader of the replica that served
}

func (r *readRecorder) hook(isLeader bool) {
	r.mu.Lock()
	r.serves = append(r.serves, isLeader)
	r.mu.Unlock()
}

func (r *readRecorder) reset() {
	r.mu.Lock()
	r.serves = nil
	r.mu.Unlock()
}

func (r *readRecorder) snapshot() []bool {
	r.mu.Lock()
	defer r.mu.Unlock()
	return append([]bool(nil), r.serves...)
}

func (r *readRecorder) sawFollowerServe() bool {
	for _, isLeader := range r.snapshot() {
		if !isLeader {
			return true
		}
	}
	return false
}

func (r *readRecorder) sawLeaderServe() bool {
	for _, isLeader := range r.snapshot() {
		if isLeader {
			return true
		}
	}
	return false
}

// TestCallPhysicalLeaderOnlyRouting verifies that CallPhysical honors the
// leaderOnly flag in an RF=2 cluster: when driven on a node that hosts the
// target shard as a FOLLOWER, leaderOnly=false serves locally (follower
// serve observed) while leaderOnly=true routes to the shard leader (only a
// leader serve observed, never the local follower) and returns the correct
// value.
func TestCallPhysicalLeaderOnlyRouting(t *testing.T) {
	const numShards = 4
	tc := newTestCluster(t, 3, numShards, 2) // RF=2
	defer tc.Close()
	ctx := context.Background()

	// Seed a key; reuse the built-in OpReadOnly "get" op (routable via the
	// standard key extractor, so CallPhysical routes it by the embedded key).
	key := []byte("leaderonly-key")
	val := []byte("v")
	if _, err := tc.client.Call(ctx, "put", ops.EncodePutArgs(key, val, 0)); err != nil {
		t.Fatalf("seed put: %v", err)
	}
	// Read-only ops bypass Raft; wait for every replica to apply the write.
	for _, n := range tc.nodes {
		if n != nil {
			waitAllApplied(t, n)
		}
	}

	// The shard the key routes to.
	shardID := shardOf(key, numShards)

	// Find a node that hosts shardID as a FOLLOWER (hosts it, not leader).
	var follower *Node
	deadline := time.Now().Add(10 * time.Second)
	for time.Now().Before(deadline) && follower == nil {
		if _, ok := tc.findShardLeader(shardID); ok {
			for _, n := range tc.nodes {
				if n == nil {
					continue
				}
				if s := n.getShard(shardID); s != nil && !s.IsLeader() {
					follower = n
					break
				}
			}
		}
		if follower == nil {
			time.Sleep(50 * time.Millisecond)
		}
	}
	if follower == nil {
		t.Fatalf("no follower hosting shard %d found (RF=2 should place a replica on a non-leader)", shardID)
	}

	rec := &readRecorder{}
	shard.SetReadServedHook(rec.hook)
	defer shard.SetReadServedHook(nil)

	physCol := "default/docs" // physical name is irrelevant: get routes by key

	// leaderOnly=false: the local follower serves the read.
	rec.reset()
	got, err := follower.CallPhysical(physCol, "get", ops.EncodeKeyArgs(key), false)
	if err != nil {
		t.Fatalf("CallPhysical leaderOnly=false: %v", err)
	}
	if !bytes.Equal(got, val) {
		t.Fatalf("CallPhysical leaderOnly=false value = %q, want %q", got, val)
	}
	if !rec.sawFollowerServe() {
		t.Fatalf("leaderOnly=false: expected the local follower to serve (an isLeader=false serve), serves=%v", rec.snapshot())
	}

	// leaderOnly=true: the read must be served by the LEADER only.
	rec.reset()
	got, err = follower.CallPhysical(physCol, "get", ops.EncodeKeyArgs(key), true)
	if err != nil {
		t.Fatalf("CallPhysical leaderOnly=true: %v", err)
	}
	if !bytes.Equal(got, val) {
		t.Fatalf("CallPhysical leaderOnly=true value = %q, want %q", got, val)
	}
	if rec.sawFollowerServe() {
		t.Fatalf("leaderOnly=true: a follower served the read (leaderOnly ignored), serves=%v", rec.snapshot())
	}
	if !rec.sawLeaderServe() {
		t.Fatalf("leaderOnly=true: no leader serve observed, serves=%v", rec.snapshot())
	}
}
