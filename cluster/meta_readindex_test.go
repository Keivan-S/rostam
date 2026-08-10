// SPDX-License-Identifier: Apache-2.0

package cluster

import (
	"errors"
	"sync"
	"sync/atomic"
	"testing"
	"time"

	hraft "github.com/hashicorp/raft"
)

// metaFollowerNode returns a node whose meta-Raft is currently a follower (NOT the
// leader). It waits briefly for a stable leader so exactly one follower is returned.
func metaFollowerNode(t *testing.T, c *testCluster) *Node {
	t.Helper()
	leader := metaLeaderNode(t, c)
	for _, n := range c.nodes {
		if n != nil && n != leader && n.meta != nil && n.meta.Raft.State() != hraft.Leader {
			return n
		}
	}
	t.Fatal("no meta follower found")
	return nil
}

// TestMetaLeaderFrontierFastPathNoBarrier: when the leader's FSM command frontier is
// already >= the verified commit index (the common steady-state case), metaLeaderFrontier
// returns WITHOUT committing a Barrier entry. We assert the meta-Raft LastIndex and
// CommitIndex are UNCHANGED across the call (a Barrier would commit a no-op and bump both).
func TestMetaLeaderFrontierFastPathNoBarrier(t *testing.T) {
	c := newTestCluster(t, 3, 8)
	defer c.Close()

	leader := metaLeaderNode(t, c)
	// Commit a catalog entry, then let the leader's own FSM fully drain so the fast
	// path is taken (AppliedIndex >= ci).
	if err := leader.SetCollectionPartitions("default/docs", 6, 1, 5*time.Second); err != nil {
		t.Fatalf("SetCollectionPartitions: %v", err)
	}
	// Wait until the leader FSM is caught up to its commit index so we are firmly on
	// the fast path (no idle no-op tail driving us into the Barrier).
	deadline := time.Now().Add(5 * time.Second)
	for time.Now().Before(deadline) {
		if leader.meta.FSM.AppliedIndex() >= leader.meta.Raft.CommitIndex() {
			break
		}
		time.Sleep(5 * time.Millisecond)
	}

	beforeLast := leader.meta.Raft.LastIndex()
	beforeCommit := leader.meta.Raft.CommitIndex()

	frontier, err := leader.metaLeaderFrontier(time.Now().Add(2 * time.Second))
	if err != nil {
		t.Fatalf("metaLeaderFrontier: %v", err)
	}
	if frontier == 0 {
		t.Fatal("metaLeaderFrontier returned 0 frontier on a leader with committed catalog state")
	}
	if frontier < beforeCommit {
		t.Fatalf("frontier %d < commit %d — frontier should reflect all commands <= commit", frontier, beforeCommit)
	}

	afterLast := leader.meta.Raft.LastIndex()
	afterCommit := leader.meta.Raft.CommitIndex()
	if afterLast != beforeLast {
		t.Fatalf("fast path committed an extra log entry: LastIndex %d -> %d (a Barrier no-op leaked)", beforeLast, afterLast)
	}
	if afterCommit != beforeCommit {
		t.Fatalf("fast path advanced CommitIndex %d -> %d (a Barrier no-op leaked)", beforeCommit, afterCommit)
	}
}

// TestMetaLeaderFrontierBehindCITriggersBarrier: when the leader's own FSM is held
// BEHIND its commit index (apply-gated), metaLeaderFrontier must take the Barrier slow
// path, BLOCK until the FSM drains, then return a frontier that reflects the gated
// command. We gate the leader's apply of a catalog commit, call metaLeaderFrontier
// concurrently, assert it does not return early, release the gate, and assert it returns
// a frontier covering the gated command's index.
func TestMetaLeaderFrontierBehindCITriggersBarrier(t *testing.T) {
	// Register the gate-clear BEFORE newTestCluster so it runs AFTER the cluster's own
	// t.Cleanup (Close) — t.Cleanup is LIFO. Clearing the global apply gate only once the
	// cluster is fully shut down avoids a -race report between this write and a trailing
	// meta-Apply reading the gate during teardown.
	t.Cleanup(func() { SetMetaApplyCatalogGate(nil) })

	c := newTestCluster(t, 3, 8)
	defer c.Close()

	leader := metaLeaderNode(t, c)
	leaderID := leader.cfg.NodeID

	release := make(chan struct{})
	var releaseOnce sync.Once
	lagEntered := make(chan struct{})
	var lagEnteredOnce sync.Once
	// Gate ONLY the leader's apply of this catalog commit. The gate fires BEFORE the
	// FSM write lock and BEFORE the deferred advanceApplied, so blocking it holds the
	// leader's FSM command frontier behind the committed index (commit advances on
	// quorum ack from the followers; the leader's local apply is what we stall).
	SetMetaApplyCatalogGate(func(nodeID, collection string, partitions, generation uint32) {
		if nodeID != leaderID {
			return
		}
		lagEnteredOnce.Do(func() { close(lagEntered) })
		<-release
	})
	// Always release the gate, even on an early t.Fatal, so the gated apply drains and
	// teardown (Close) cannot hang.
	defer releaseOnce.Do(func() { close(release) })

	// Snapshot the leader's command frontier BEFORE the gated write, so we can prove the
	// returned frontier advanced PAST it (i.e. the Barrier genuinely drained the gated
	// catalog command into the FSM).
	frontierBeforeWrite := leader.meta.FSM.AppliedIndex()

	// Commit the catalog entry from a FOLLOWER so the leader applies it via replication
	// while the gate holds its local apply. Forwarded write reaches the leader's log;
	// commit advances on quorum; the leader's own apply blocks in the gate.
	follower := metaFollowerNode(t, c)
	writeDone := make(chan error, 1)
	go func() { writeDone <- follower.SetCollectionPartitions("default/gated", 4, 1, 8*time.Second) }()

	select {
	case <-lagEntered:
	case <-time.After(10 * time.Second):
		releaseOnce.Do(func() { close(release) })
		t.Fatal("leader apply gate never entered")
	}

	// While the gate holds the leader FSM behind ci, metaLeaderFrontier must NOT return
	// before the gate releases (it sits in the Barrier draining the FSM).
	frontierCh := make(chan uint64, 1)
	errCh := make(chan error, 1)
	go func() {
		f, err := leader.metaLeaderFrontier(time.Now().Add(8 * time.Second))
		if err != nil {
			errCh <- err
			return
		}
		frontierCh <- f
	}()

	select {
	case f := <-frontierCh:
		releaseOnce.Do(func() { close(release) })
		t.Fatalf("metaLeaderFrontier returned %d while leader FSM was gated behind ci — should block in Barrier", f)
	case err := <-errCh:
		releaseOnce.Do(func() { close(release) })
		t.Fatalf("metaLeaderFrontier errored while gated: %v", err)
	case <-time.After(300 * time.Millisecond):
		// Good: still blocked in the Barrier.
	}

	releaseOnce.Do(func() { close(release) })

	select {
	case f := <-frontierCh:
		// The returned frontier is the leader's last-applied COMMAND index. It MUST have
		// advanced past the pre-write frontier (proving the Barrier drained the gated
		// catalog command into the FSM). NOTE it is intentionally a COMMAND index, NOT the
		// raw CommitIndex — the Barrier's own no-op entry bumps CommitIndex PAST the last
		// command, so frontier < CommitIndex is correct (the no-op-entry landmine).
		if f <= frontierBeforeWrite {
			t.Fatalf("frontier %d did not advance past pre-write frontier %d — Barrier returned before draining the gated command", f, frontierBeforeWrite)
		}
	case err := <-errCh:
		t.Fatalf("metaLeaderFrontier errored after release: %v", err)
	case <-time.After(8 * time.Second):
		t.Fatal("metaLeaderFrontier did not return after gate release")
	}

	if err := <-writeDone; err != nil {
		t.Fatalf("gated write: %v", err)
	}
}

// TestMetaReadBarrierFollowerForwardsAndCatchesUp (THE follower proof): a follower's
// meta apply of a catalog commit is gated so its local FSM lags. The follower's
// metaReadBarrier must FORWARD to the leader (1 RTT), learn the leader frontier, and
// BLOCK until its local FSM reaches it — then return nil. We assert it blocks while the
// gate holds, then unblocks promptly after release.
func TestMetaReadBarrierFollowerForwardsAndCatchesUp(t *testing.T) {
	// See the BehindCI test: clear the gate AFTER cluster teardown (LIFO Cleanup) to keep
	// -race clean against a trailing meta-Apply.
	t.Cleanup(func() { SetMetaApplyCatalogGate(nil) })

	c := newTestCluster(t, 3, 8)
	defer c.Close()

	leader := metaLeaderNode(t, c)
	follower := metaFollowerNode(t, c)
	followerID := follower.cfg.NodeID

	var forwards int32
	SetMetaReadIndexForwardHook(func() { atomic.AddInt32(&forwards, 1) })
	defer SetMetaReadIndexForwardHook(nil)

	release := make(chan struct{})
	var releaseOnce sync.Once
	lagEntered := make(chan struct{})
	var lagEnteredOnce sync.Once
	// Gate ONLY the lagging follower's apply of the catalog commit so its local FSM
	// command frontier stays behind the leader frontier.
	SetMetaApplyCatalogGate(func(nodeID, collection string, partitions, generation uint32) {
		if nodeID != followerID {
			return
		}
		lagEnteredOnce.Do(func() { close(lagEntered) })
		<-release
	})
	// Always release the gate, even on an early t.Fatal, so teardown cannot hang.
	defer releaseOnce.Do(func() { close(release) })

	// Commit from the leader so the lagging follower applies via replication (gated).
	writeDone := make(chan error, 1)
	go func() { writeDone <- leader.SetCollectionPartitions("default/lag", 4, 1, 8*time.Second) }()

	select {
	case <-lagEntered:
	case <-time.After(10 * time.Second):
		releaseOnce.Do(func() { close(release) })
		t.Fatal("follower apply gate never entered")
	}

	// The follower's barrier must forward + block while its FSM lags.
	barrierErr := make(chan error, 1)
	go func() { barrierErr <- follower.metaReadBarrier(time.Now().Add(8 * time.Second)) }()

	select {
	case err := <-barrierErr:
		releaseOnce.Do(func() { close(release) })
		t.Fatalf("metaReadBarrier returned (%v) while follower FSM was gated — should block until caught up", err)
	case <-time.After(300 * time.Millisecond):
		// Good: still blocked waiting for the local FSM to reach the leader frontier.
	}

	if atomic.LoadInt32(&forwards) == 0 {
		releaseOnce.Do(func() { close(release) })
		t.Fatal("follower metaReadBarrier issued ZERO __meta_readindex__ forwards — it must forward to the leader")
	}

	releaseStart := time.Now()
	releaseOnce.Do(func() { close(release) })

	select {
	case err := <-barrierErr:
		if err != nil {
			t.Fatalf("metaReadBarrier after release: %v", err)
		}
		if d := time.Since(releaseStart); d > 3*time.Second {
			t.Fatalf("metaReadBarrier took %s after gate release — should unblock promptly once caught up", d)
		}
	case <-time.After(8 * time.Second):
		t.Fatal("metaReadBarrier did not return after gate release")
	}

	if err := <-writeDone; err != nil {
		t.Fatalf("gated write: %v", err)
	}
}

// TestMetaReadBarrierLeaderUnreachableTimesOut: a follower whose meta leader is
// unreachable must time out with *ErrMetaLinearizableTimeout within the deadline
// (bounded; never hangs), never serving stale.
func TestMetaReadBarrierLeaderUnreachableTimesOut(t *testing.T) {
	c := newTestCluster(t, 3, 8)
	defer c.Close()

	leader := metaLeaderNode(t, c)
	follower := metaFollowerNode(t, c)

	// Bring the meta leader DOWN (stop its node + server). The follower can no longer
	// resolve/reach a meta leader; a new election may or may not complete, but with the
	// SHORT deadline below the barrier must fail loud rather than hang.
	var leaderIdx = -1
	for i, n := range c.nodes {
		if n == leader {
			leaderIdx = i
			break
		}
	}
	if leaderIdx < 0 {
		t.Fatal("could not locate leader index")
	}
	if c.servers[leaderIdx] != nil {
		_ = c.servers[leaderIdx].Close()
		c.servers[leaderIdx] = nil
	}
	if c.nodes[leaderIdx] != nil {
		_ = c.nodes[leaderIdx].Close()
		c.nodes[leaderIdx] = nil
	}

	// Use a short deadline. Even if a new leader is mid-election, the barrier on this
	// follower must return a typed timeout within roughly the deadline, never hang.
	start := time.Now()
	err := follower.metaReadBarrier(time.Now().Add(600 * time.Millisecond))
	elapsed := time.Since(start)

	// A new leader could conceivably be elected within 600ms and the follower (already
	// caught up) could pass. The contract under test is "bounded, never hangs": accept
	// either a clean nil (election completed + caught up) or the typed timeout, but FAIL
	// if it hangs well past the deadline.
	if elapsed > 3*time.Second {
		t.Fatalf("metaReadBarrier took %s with the leader down — must be bounded by the ~600ms deadline", elapsed)
	}
	if err != nil {
		var to *ErrMetaLinearizableTimeout
		if !errors.As(err, &to) {
			t.Fatalf("error = %v, want *ErrMetaLinearizableTimeout (or nil if a new leader was elected in time)", err)
		}
	}
}

// TestMetaReadBarrierSingleNodeNoOp: a single-node cluster (len(Peers)<=1) short-circuits
// metaReadBarrier to a trivial no-op — ZERO forwards, returns instantly.
func TestMetaReadBarrierSingleNodeNoOp(t *testing.T) {
	c := newTestCluster(t, 1, 8)
	defer c.Close()

	var forwards int32
	SetMetaReadIndexForwardHook(func() { atomic.AddInt32(&forwards, 1) })
	defer SetMetaReadIndexForwardHook(nil)

	n := c.nodes[0]
	start := time.Now()
	if err := n.metaReadBarrier(time.Now().Add(2 * time.Second)); err != nil {
		t.Fatalf("single-node metaReadBarrier should be a trivial no-op, got %v", err)
	}
	if elapsed := time.Since(start); elapsed > 100*time.Millisecond {
		t.Fatalf("single-node metaReadBarrier took %s — should return instantly", elapsed)
	}
	if f := atomic.LoadInt32(&forwards); f != 0 {
		t.Fatalf("single-node metaReadBarrier issued %d __meta_readindex__ forwards, want 0", f)
	}
}

// TestMetaReadIndexHandlerReturnsFrontier exercises the __meta_readindex__ handler
// directly on the leader: it returns OK=true with a non-zero frontier reflecting the
// committed catalog state.
func TestMetaReadIndexHandlerReturnsFrontier(t *testing.T) {
	c := newTestCluster(t, 3, 8)
	defer c.Close()

	leader := metaLeaderNode(t, c)
	if err := leader.SetCollectionPartitions("default/docs", 6, 1, 5*time.Second); err != nil {
		t.Fatalf("SetCollectionPartitions: %v", err)
	}

	args, err := gobEncode(metaReadIndexReq{Version: metaReadIndexVersion})
	if err != nil {
		t.Fatal(err)
	}
	raw, err := leader.handleMetaReadIndex(args)
	if err != nil {
		t.Fatalf("handleMetaReadIndex on leader: %v", err)
	}
	var reply metaReadIndexReply
	if err := gobDecode(raw, &reply); err != nil {
		t.Fatal(err)
	}
	if !reply.OK {
		t.Fatal("__meta_readindex__ on the leader returned OK=false")
	}
	if reply.Frontier == 0 {
		t.Fatal("__meta_readindex__ on the leader returned a zero frontier")
	}

	// On a follower the handler must return OK=false (not-leader signal) so the caller
	// re-resolves the leader — never a stale/zero frontier as if authoritative.
	follower := metaFollowerNode(t, c)
	fraw, err := follower.handleMetaReadIndex(args)
	if err != nil {
		t.Fatalf("handleMetaReadIndex on follower returned a hard error: %v", err)
	}
	var freply metaReadIndexReply
	if err := gobDecode(fraw, &freply); err != nil {
		t.Fatal(err)
	}
	if freply.OK {
		t.Fatal("__meta_readindex__ on a FOLLOWER returned OK=true — must signal not-leader (OK=false)")
	}
}
