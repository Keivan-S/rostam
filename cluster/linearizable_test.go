// SPDX-License-Identifier: Apache-2.0

package cluster

import (
	"context"
	"sync/atomic"
	"testing"
	"time"

	"github.com/rostamlabs/rostam/client"
	"github.com/rostamlabs/rostam/ops"
	"github.com/rostamlabs/rostam/shard"
	"github.com/rostamlabs/rostam/vector"
)

// TestFanOutRoutesByConsistency asserts the routing decision FanOut threads to
// the partitionCaller for every consistency level: AnyReplica reads any replica
// (route-to-leader=false); LeaderOnly AND Linearizable both pin to the leader
// (route-to-leader=true). It mirrors TestFanOutForwardsLeaderOnly and adds the
// Linearizable case (the routing addition) plus the unchanged-default
// regression cases. The Linearizable FRESHNESS barrier is NOT signalled by this
// bool — it travels in the op args (read_consistency byte) and is enforced by
// the serving shard — so the routing-level contract is exactly "pin to leader".
func TestFanOutRoutesByConsistency(t *testing.T) {
	for _, tc := range []struct {
		name string
		cons Consistency
		want bool // route-to-leader (the leaderOnly arg the caller receives)
	}{
		{"any-replica", AnyReplica, false},
		{"leader-only", LeaderOnly, true},
		{"linearizable", Linearizable, true},
	} {
		t.Run(tc.name, func(t *testing.T) {
			caller := func(physCol, op string, args []byte, leaderOnly bool) ([]byte, error) {
				if leaderOnly != tc.want {
					t.Errorf("consistency=%d: leaderOnly=%v, want %v", tc.cons, leaderOnly, tc.want)
				}
				return []byte{0}, nil
			}
			decode := func(b []byte) ([]Scored, error) { return []Scored{{ID: uint64(b[0])}}, nil }
			merge := func(parts [][]Scored, k int) []Scored {
				return MergeTopK(parts, k, func(a, b Scored) bool { return a.Dist < b.Dist })
			}
			_, _, err := FanOut(FanArgs{
				Collection: "default/docs", P: 2, K: 10,
				Consistency: tc.cons,
				Encode:      func(physCol string) []byte { return []byte(physCol) },
			}, caller, decode, merge)
			if err != nil {
				t.Fatal(err)
			}
		})
	}
}

// linearizableSearchArgs builds vector_search args tagged ConsistencyLinearizable
// for the given physical collection. The read_consistency byte rides in the opts
// trailer, so shard.Store.Call peeks it via ops.ReadConsistencyOf and runs the
// readIndex barrier before serving — regardless of whether the read arrived via a
// local-leader serve or a forwardToLeader hop.
func linearizableSearchArgs(physCol string, k int, query []float32) []byte {
	return ops.EncodeVectorSearchArgsOpts(physCol, k, query, vector.Filter{}, ops.ConsistencyLinearizable, 0, 0)
}

// TestCallPhysicalLinearizableRoutesToLeaderAndBarriers proves the routing
// contract end-to-end on a real RF=2 cluster: a Linearizable read submitted to a
// node that hosts the target shard as a FOLLOWER is forwarded to the shard leader,
// the leader runs the readIndex barrier (VerifyLeader + commit-index catch-up)
// BEFORE serving, only the leader serves it (never the local follower), and it
// returns the latest committed data. This exercises the forwardToLeader path with
// the linearizable byte intact in the forwarded args.
func TestCallPhysicalLinearizableRoutesToLeaderAndBarriers(t *testing.T) {
	const numShards = 4
	tc := newTestCluster(t, 3, numShards, 2) // RF=2
	defer tc.Close()
	ctx := context.Background()

	// Single-partition collection so the physical name is deterministic and the
	// read routes to exactly one shard. The collection name doubles as the routing
	// key (vectorKeyColAt2), so creates/upserts/searches all hit the same shard.
	const coll = "lin/docs"
	physCol := string(ops.PartitionKeyGen(coll, 0, 0))
	cfg := vector.Config{Dim: 4, Metric: vector.L2, M: 8, EfConstruction: 50, EfSearch: 64, Seed: 1}
	if _, err := tc.client.Call(ctx, "vector_create_collection", ops.EncodeCreateCollectionArgs(physCol, cfg)); err != nil {
		t.Fatalf("create collection: %v", err)
	}
	// Upsert one vector (a committed write through Raft).
	const wantID = uint64(7)
	vec := []float32{1, 0, 0, 0}
	if _, err := tc.client.Call(ctx, "vector_upsert",
		ops.EncodeVectorUpsertArgs(physCol, wantID, vec, "c", 0, nil, vector.SparseVector{})); err != nil {
		t.Fatalf("upsert: %v", err)
	}
	// Read-only ops bypass Raft; wait for every replica to apply.
	for _, n := range tc.nodes {
		if n != nil {
			waitAllApplied(t, n)
		}
	}

	shardID := shardOf([]byte(physCol), numShards)

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

	// Observe (a) which replica serves and (b) that the readIndex barrier is entered.
	rec := &readRecorder{}
	shard.SetReadServedHook(rec.hook)
	defer shard.SetReadServedHook(nil)
	var barriers atomic.Int32
	shard.SetBarrierEnteredHook(func() { barriers.Add(1) })
	defer shard.SetBarrierEnteredHook(nil)

	args := linearizableSearchArgs(physCol, 1, vec)
	raw, err := follower.CallPhysical(physCol, "vector_search", args, true)
	if err != nil {
		t.Fatalf("CallPhysical linearizable: %v", err)
	}
	if barriers.Load() == 0 {
		t.Fatal("readIndex barrier was NOT entered on the serving leader — a linearizable read reached a serve without VerifyLeader (correctness hole)")
	}
	if rec.sawFollowerServe() {
		t.Fatalf("linearizable read served by a follower (leader routing failed), serves=%v", rec.snapshot())
	}
	if !rec.sawLeaderServe() {
		t.Fatalf("linearizable read produced no leader serve, serves=%v", rec.snapshot())
	}

	// The leader served the LATEST committed data.
	results, err := ops.DecodeVectorSearchResults(raw)
	if err != nil {
		t.Fatalf("decode results: %v", err)
	}
	if len(results) != 1 || results[0].ID != wantID {
		t.Fatalf("linearizable search results = %+v, want exactly [id=%d]", results, wantID)
	}
}

// TestCallPhysicalLinearizableSingleNodeFree confirms that on a single-node
// cluster a Linearizable read works and the barrier resolves immediately: with a
// quorum of one, VerifyLeader returns without a round-trip and the FSM is trivially
// caught up, so the read is "free" (no hang) and serves correct data.
func TestCallPhysicalLinearizableSingleNodeFree(t *testing.T) {
	tc := newTestCluster(t, 1, 1) // single node, single shard
	defer tc.Close()
	ctx := context.Background()

	const coll = "lin1/docs"
	physCol := string(ops.PartitionKeyGen(coll, 0, 0))
	cfg := vector.Config{Dim: 4, Metric: vector.L2, M: 8, EfConstruction: 50, EfSearch: 64, Seed: 1}
	if _, err := tc.client.Call(ctx, "vector_create_collection", ops.EncodeCreateCollectionArgs(physCol, cfg)); err != nil {
		t.Fatalf("create collection: %v", err)
	}
	const wantID = uint64(3)
	vec := []float32{0, 1, 0, 0}
	if _, err := tc.client.Call(ctx, "vector_upsert",
		ops.EncodeVectorUpsertArgs(physCol, wantID, vec, "c", 0, nil, vector.SparseVector{})); err != nil {
		t.Fatalf("upsert: %v", err)
	}
	for _, n := range tc.nodes {
		if n != nil {
			waitAllApplied(t, n)
		}
	}

	var barriers atomic.Int32
	shard.SetBarrierEnteredHook(func() { barriers.Add(1) })
	defer shard.SetBarrierEnteredHook(nil)

	// Bound the call so a hung VerifyLeader (a regression) fails the test instead
	// of blocking forever; on RF=1 it must resolve in well under this budget.
	type res struct {
		raw []byte
		err error
	}
	done := make(chan res, 1)
	go func() {
		raw, err := tc.nodes[0].CallPhysical(physCol, "vector_search", linearizableSearchArgs(physCol, 1, vec), true)
		done <- res{raw, err}
	}()
	var got res
	select {
	case got = <-done:
	case <-time.After(3 * time.Second):
		t.Fatal("single-node linearizable read did not resolve within 3s (VerifyLeader should be immediate at quorum=1)")
	}
	if got.err != nil {
		t.Fatalf("single-node linearizable read: %v", got.err)
	}
	if barriers.Load() == 0 {
		t.Fatal("barrier not entered: the linearizable byte did not reach the shard")
	}
	results, err := ops.DecodeVectorSearchResults(got.raw)
	if err != nil {
		t.Fatalf("decode results: %v", err)
	}
	if len(results) != 1 || results[0].ID != wantID {
		t.Fatalf("single-node linearizable results = %+v, want exactly [id=%d]", results, wantID)
	}
}

// TestCallPhysicalAnyReplicaSkipsBarrier is a no-regression guard: the default
// (AnyReplica) path must NEVER enter the readIndex barrier — it costs only the
// cheap consistency peek. We drive the SAME vector_search op but with no opts
// trailer (rc=0) and leaderOnly=false, then assert the barrier hook never fired.
func TestCallPhysicalAnyReplicaSkipsBarrier(t *testing.T) {
	tc := newTestCluster(t, 1, 1)
	defer tc.Close()
	ctx := context.Background()

	const coll = "any/docs"
	physCol := string(ops.PartitionKeyGen(coll, 0, 0))
	cfg := vector.Config{Dim: 4, Metric: vector.L2, M: 8, EfConstruction: 50, EfSearch: 64, Seed: 1}
	if _, err := tc.client.Call(ctx, "vector_create_collection", ops.EncodeCreateCollectionArgs(physCol, cfg)); err != nil {
		t.Fatalf("create collection: %v", err)
	}
	vec := []float32{0, 0, 1, 0}
	if _, err := tc.client.Call(ctx, "vector_upsert",
		ops.EncodeVectorUpsertArgs(physCol, 1, vec, "c", 0, nil, vector.SparseVector{})); err != nil {
		t.Fatalf("upsert: %v", err)
	}
	for _, n := range tc.nodes {
		if n != nil {
			waitAllApplied(t, n)
		}
	}

	var barriers atomic.Int32
	shard.SetBarrierEnteredHook(func() { barriers.Add(1) })
	defer shard.SetBarrierEnteredHook(nil)

	// Legacy (no-opts) search args ⇒ rc=0 (AnyReplica); leaderOnly=false.
	anyArgs := ops.EncodeVectorSearchArgs(physCol, 1, vec)
	if _, err := tc.nodes[0].CallPhysical(physCol, "vector_search", anyArgs, false); err != nil {
		t.Fatalf("AnyReplica search: %v", err)
	}
	if barriers.Load() != 0 {
		t.Fatalf("AnyReplica read entered the readIndex barrier %d time(s) — default path must be barrier-free", barriers.Load())
	}
}

// TestClusterLinearizableFailover is the cluster-level linearizability-across-
// failover proof. On a real 3-node RF=3 TCP cluster (every node replicates the
// target shard) it:
//
//  1. commits value X (vector id=wantX) through Raft and confirms a Linearizable
//     read returns X (the readIndex barrier serves the latest committed state);
//  2. KILLS the node hosting the shard leader (close node + server, drop the
//     pooled client), then waits — gated on actual leadership convergence, not a
//     fixed sleep — for a NEW leader to be elected in the surviving majority;
//  3. commits a NEW value Y (vector id=wantY) via the new leader;
//  4. issues a Linearizable read from a SURVIVING node through the real cluster
//     path (CallPhysical → leader routing → forwardToLeader → shard readIndex
//     barrier) and asserts it returns Y — the LATEST committed value, never the
//     dead leader's stale X-or-older state.
//
// The barrier-entered hook confirms the read genuinely ran VerifyLeader on the
// serving leader (not a stale local-state bypass), and the read-served hook
// confirms a leader (never a follower) served it. RF=3 guarantees both survivors
// hold a replica, so one can win the post-kill election and serve the read.
func TestClusterLinearizableFailover(t *testing.T) {
	const numShards = 4
	tc := newTestCluster(t, 3, numShards, 3) // RF=3: every node replicates every shard
	ctx := context.Background()

	// Single-partition collection ⇒ deterministic physical name + single target
	// shard, so the kill/elect/read all concern one Raft group.
	const coll = "failover/docs"
	physCol := string(ops.PartitionKeyGen(coll, 0, 0))
	cfg := vector.Config{Dim: 4, Metric: vector.L2, M: 8, EfConstruction: 50, EfSearch: 64, Seed: 1}
	if _, err := tc.client.Call(ctx, "vector_create_collection", ops.EncodeCreateCollectionArgs(physCol, cfg)); err != nil {
		t.Fatalf("create collection: %v", err)
	}

	// ── Commit X (the pre-failover committed state) ──────────────────────────
	const wantX = uint64(7)
	vecX := []float32{1, 0, 0, 0}
	if _, err := tc.client.Call(ctx, "vector_upsert",
		ops.EncodeVectorUpsertArgs(physCol, wantX, vecX, "x", 0, nil, vector.SparseVector{})); err != nil {
		t.Fatalf("upsert X: %v", err)
	}
	for _, n := range tc.nodes {
		if n != nil {
			waitAllApplied(t, n)
		}
	}

	shardID := shardOf([]byte(physCol), numShards)

	// readLinearizable issues a Linearizable read through the cluster path on the
	// given surviving node and returns the decoded results. It asserts the
	// readIndex barrier was entered (VerifyLeader genuinely ran) and that a leader
	// — never a follower — served the read.
	readLinearizable := func(t *testing.T, n *Node, query []float32) []vector.Result {
		t.Helper()
		rec := &readRecorder{}
		shard.SetReadServedHook(rec.hook)
		defer shard.SetReadServedHook(nil)
		var barriers atomic.Int32
		shard.SetBarrierEnteredHook(func() { barriers.Add(1) })
		defer shard.SetBarrierEnteredHook(nil)

		raw, err := n.CallPhysical(physCol, "vector_search", linearizableSearchArgs(physCol, 1, query), true)
		if err != nil {
			t.Fatalf("CallPhysical linearizable: %v", err)
		}
		if barriers.Load() == 0 {
			t.Fatal("readIndex barrier was NOT entered — a linearizable read served without VerifyLeader (correctness hole)")
		}
		if rec.sawFollowerServe() {
			t.Fatalf("linearizable read served by a follower (leader routing failed), serves=%v", rec.snapshot())
		}
		if !rec.sawLeaderServe() {
			t.Fatalf("linearizable read produced no leader serve, serves=%v", rec.snapshot())
		}
		results, err := ops.DecodeVectorSearchResults(raw)
		if err != nil {
			t.Fatalf("decode results: %v", err)
		}
		return results
	}

	// ── Baseline: a Linearizable read returns X ──────────────────────────────
	// Drive it from a node that hosts the shard as a follower so the read
	// genuinely traverses leader routing (not a same-node shortcut).
	var preReader *Node
	for _, n := range tc.nodes {
		if n == nil {
			continue
		}
		if s := n.getShard(shardID); s != nil && !s.IsLeader() {
			preReader = n
			break
		}
	}
	if preReader == nil {
		t.Fatalf("RF=3: expected a follower hosting shard %d for the baseline read", shardID)
	}
	if r := readLinearizable(t, preReader, vecX); len(r) != 1 || r[0].ID != wantX {
		t.Fatalf("baseline linearizable read = %+v, want exactly [id=%d]", r, wantX)
	}

	// ── Kill the shard leader ────────────────────────────────────────────────
	leaderIdx, ok := tc.findShardLeader(shardID)
	if !ok {
		t.Fatalf("could not find shard %d leader to kill", shardID)
	}
	t.Logf("killing shard %d leader: node %d", shardID, leaderIdx)

	// Drop the pooled client first (its conns to the soon-dead server would
	// otherwise stall server.Close), then close the node + server and nil the
	// slots so tc.Close / helpers skip them.
	_ = tc.client.Close()
	tc.client = nil
	_ = tc.nodes[leaderIdx].Close()
	_ = tc.servers[leaderIdx].Close()
	tc.nodes[leaderIdx] = nil
	tc.servers[leaderIdx] = nil

	// ── Wait for a NEW leader in the surviving majority (gate on convergence) ─
	// findShardLeader scans only non-nil (surviving) nodes, so this resolves
	// only once a survivor reports Leader for the shard — true re-election, not
	// a fixed sleep. 30s matches the leader-kill deadline elsewhere (race-mode
	// re-election under CPU contention).
	var newLeader int
	electDeadline := time.Now().Add(30 * time.Second)
	for {
		if idx, found := tc.findShardLeader(shardID); found {
			newLeader = idx
			break
		}
		if time.Now().After(electDeadline) {
			t.Fatalf("no new shard %d leader elected within deadline after kill", shardID)
		}
		time.Sleep(50 * time.Millisecond)
	}
	t.Logf("new shard %d leader: node %d", shardID, newLeader)

	// Fresh client wired only to the surviving servers.
	survAddrs := make([]string, 0, len(tc.peers)-1)
	for i, p := range tc.peers {
		if i == leaderIdx {
			continue
		}
		survAddrs = append(survAddrs, p.ServerAddr)
	}
	c2, err := client.New(client.Config{
		Servers:           survAddrs,
		MaxConnsPerServer: 8,
		MaxNotLeaderHops:  5,
	})
	if err != nil {
		t.Fatalf("post-kill client: %v", err)
	}
	tc.client = c2 // adopt so t.Cleanup → tc.Close closes it

	// ── Commit Y via the new leader ──────────────────────────────────────────
	// Retry across the (possibly still-settling) re-election window; a write
	// that commits proves the new leader is serving.
	const wantY = uint64(11)
	vecY := []float32{0, 0, 0, 1}
	writeDeadline := time.Now().Add(30 * time.Second)
	var lastErr error
	wrote := false
	for time.Now().Before(writeDeadline) {
		_, lastErr = c2.Call(ctx, "vector_upsert",
			ops.EncodeVectorUpsertArgs(physCol, wantY, vecY, "y", 0, nil, vector.SparseVector{}))
		if lastErr == nil {
			wrote = true
			break
		}
		time.Sleep(100 * time.Millisecond)
	}
	if !wrote {
		t.Fatalf("write Y never committed on the new leader: %v", lastErr)
	}
	for _, n := range tc.nodes {
		if n != nil {
			waitAllApplied(t, n)
		}
	}

	// ── The core proof: a Linearizable read sees Y (the LATEST), never stale ──
	// Drive it from a SURVIVING follower so it routes to the new leader, which
	// re-runs the readIndex barrier. Query toward Y; the nearest neighbour must
	// be Y (and X — the dead leader's pre-failover state — must NOT win).
	var postReader *Node
	for i, n := range tc.nodes {
		if n == nil || i == newLeader {
			continue
		}
		if s := n.getShard(shardID); s != nil && !s.IsLeader() {
			postReader = n
			break
		}
	}
	if postReader == nil {
		// Fall back to the new leader itself if no surviving follower hosts the
		// shard (shouldn't happen at RF=3 with 2 survivors, but stay robust).
		postReader = tc.nodes[newLeader]
	}
	results := readLinearizable(t, postReader, vecY)
	if len(results) != 1 || results[0].ID != wantY {
		t.Fatalf("post-failover linearizable read = %+v, want exactly [id=%d] (Y, the latest committed value) — a stale/dead-leader serve would return id=%d or nothing", results, wantY, wantX)
	}
}
