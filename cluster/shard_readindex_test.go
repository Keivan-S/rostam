// SPDX-License-Identifier: Apache-2.0

package cluster

import (
	"sync"
	"sync/atomic"
	"testing"
	"time"
)

// shardLeaderAndFollower returns a node that leads shard idx and one that hosts it as
// a follower, waiting briefly for a shard leader to be elected.
func shardLeaderAndFollower(t *testing.T, c *testCluster, idx int) (leader, follower *Node) {
	t.Helper()
	deadline := time.Now().Add(5 * time.Second)
	for time.Now().Before(deadline) {
		leader, follower = nil, nil
		for _, n := range c.nodes {
			if n == nil {
				continue
			}
			s := n.getShard(idx)
			if s == nil {
				continue
			}
			if s.IsLeader() {
				leader = n
			} else {
				follower = n
			}
		}
		if leader != nil && follower != nil {
			return leader, follower
		}
		time.Sleep(20 * time.Millisecond)
	}
	t.Fatalf("shard %d: leader=%v follower=%v not both found", idx, leader != nil, follower != nil)
	return nil, nil
}

// TestShardReadIndexHandlerReturnsFrontier exercises handleShardReadIndex directly:
// on the shard leader it returns OK=true with a non-zero committed frontier; on a
// follower it returns OK=false (not-leader signal) so the caller fails closed.
func TestShardReadIndexHandlerReturnsFrontier(t *testing.T) {
	c := newTestCluster(t, 3, 1)
	defer c.Close()

	leader, follower := shardLeaderAndFollower(t, c, 0)

	args, err := gobEncode(shardReadIndexReq{Version: shardReadIndexVersion, ShardIdx: 0})
	if err != nil {
		t.Fatal(err)
	}

	raw, err := leader.handleShardReadIndex(args)
	if err != nil {
		t.Fatalf("handleShardReadIndex on leader: %v", err)
	}
	var reply shardReadIndexReply
	if err := gobDecode(raw, &reply); err != nil {
		t.Fatal(err)
	}
	if !reply.OK {
		t.Fatal("__shard_readindex__ on the leader returned OK=false")
	}
	if reply.Frontier == 0 {
		t.Fatal("__shard_readindex__ on the leader returned a zero frontier")
	}
	// The reported frontier must be the leader's committed index.
	if want := leader.getShard(0).CommitIndex(); reply.Frontier > want {
		t.Fatalf("frontier %d exceeds leader CommitIndex %d", reply.Frontier, want)
	}

	fraw, err := follower.handleShardReadIndex(args)
	if err != nil {
		t.Fatalf("handleShardReadIndex on follower returned a hard error: %v", err)
	}
	var freply shardReadIndexReply
	if err := gobDecode(fraw, &freply); err != nil {
		t.Fatal(err)
	}
	if freply.OK {
		t.Fatal("__shard_readindex__ on a FOLLOWER returned OK=true — must signal not-leader (OK=false)")
	}
}

// TestShardLeaderFrontierLocalLeader: shardLeaderFrontier answers locally on the
// leader (returns its CommitIndex) and returns errNotShardLeader on a follower.
func TestShardLeaderFrontierLocalLeader(t *testing.T) {
	c := newTestCluster(t, 3, 1)
	defer c.Close()

	leader, follower := shardLeaderAndFollower(t, c, 0)

	f, err := leader.shardLeaderFrontier(0, time.Now().Add(2*time.Second))
	if err != nil {
		t.Fatalf("shardLeaderFrontier on leader: %v", err)
	}
	if f == 0 {
		t.Fatal("shardLeaderFrontier on leader returned 0")
	}

	if _, err := follower.shardLeaderFrontier(0, time.Now().Add(2*time.Second)); err == nil {
		t.Fatal("shardLeaderFrontier on a follower returned nil error, want errNotShardLeader")
	}
}

// TestShardFrontierCoalescerSharesInFlight: N concurrent bounded reads for the SAME
// shard on a follower coalesce into FEWER than N leader pings.
func TestShardFrontierCoalescerSharesInFlight(t *testing.T) {
	var c shardFrontierCoalescer
	const n = 8

	var calls atomic.Int64
	release := make(chan struct{})
	registered := make(chan struct{}, n)

	var wg sync.WaitGroup
	for i := 0; i < n; i++ {
		wg.Add(1)
		go func() {
			defer wg.Done()
			registered <- struct{}{}
			_, _ = c.do(0, time.Now().Add(2*time.Second), func(time.Time) (uint64, error) {
				calls.Add(1)
				<-release
				return 7, nil
			})
		}()
	}
	for i := 0; i < n; i++ {
		<-registered
	}
	time.Sleep(50 * time.Millisecond)
	close(release)
	wg.Wait()

	if got := calls.Load(); got >= n {
		t.Fatalf("coalescer ran %d pings for %d concurrent followers, want < %d", got, n, n)
	}
	if calls.Load() == 0 {
		t.Fatal("coalescer never ran the ping")
	}
}

// TestShardFrontierCoalescerPerShardIndependent: forwards for DIFFERENT shards do not
// coalesce with each other (each shard has its own flight).
func TestShardFrontierCoalescerPerShardIndependent(t *testing.T) {
	var c shardFrontierCoalescer
	var shard0, shard1 atomic.Int64

	var wg sync.WaitGroup
	wg.Add(2)
	go func() {
		defer wg.Done()
		_, _ = c.do(0, time.Now().Add(2*time.Second), func(time.Time) (uint64, error) {
			shard0.Add(1)
			return 1, nil
		})
	}()
	go func() {
		defer wg.Done()
		_, _ = c.do(1, time.Now().Add(2*time.Second), func(time.Time) (uint64, error) {
			shard1.Add(1)
			return 2, nil
		})
	}()
	wg.Wait()

	if shard0.Load() != 1 || shard1.Load() != 1 {
		t.Fatalf("per-shard flights: shard0=%d shard1=%d, want 1 and 1", shard0.Load(), shard1.Load())
	}
}

// TestFetchShardLeaderFrontierForwardCoalesces: N concurrent fetchShardLeaderFrontier
// forwards on a follower (through the coalescer) issue ONE remote leader ping,
// observed via the forward hook.
func TestFetchShardLeaderFrontierForwardCoalesces(t *testing.T) {
	c := newTestCluster(t, 3, 1)
	defer c.Close()

	_, follower := shardLeaderAndFollower(t, c, 0)

	var pings atomic.Int64
	SetShardReadIndexForwardHook(func() { pings.Add(1) })
	defer SetShardReadIndexForwardHook(nil)

	const n = 10
	gate := make(chan struct{})
	var wg sync.WaitGroup
	var anyErr atomic.Pointer[error]
	for i := 0; i < n; i++ {
		wg.Add(1)
		go func() {
			defer wg.Done()
			<-gate
			_, err := follower.shardFrontier.do(0, time.Now().Add(3*time.Second), func(d time.Time) (uint64, error) {
				return follower.fetchShardLeaderFrontier(0, d)
			})
			if err != nil {
				anyErr.Store(&err)
			}
		}()
	}
	close(gate)
	wg.Wait()

	if p := anyErr.Load(); p != nil {
		t.Fatalf("a coalesced bounded fetch errored: %v", *p)
	}
	if got := pings.Load(); got == 0 {
		t.Fatal("no remote __shard_readindex__ ping fired — the follower must forward to the leader")
	} else if got >= n {
		t.Fatalf("%d concurrent bounded fetches fired %d leader pings, want < %d (must coalesce)", n, got, n)
	}
}
