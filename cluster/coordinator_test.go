// SPDX-License-Identifier: Apache-2.0

package cluster

import (
	"context"
	"fmt"
	"sync"
	"sync/atomic"
	"testing"
	"time"

	"github.com/rostamlabs/rostam/ops"
)

// TestSlice3Coordinator drives the rebalance coordinator through two membership
// changes on a live 3-node cluster: first growing replication (RF 1->2, every
// shard gains an owner and the load redistributes ~evenly), then decommissioning
// a node (target the {n1,n2} subset, so n3's shards re-home to n1/n2). A
// background reader runs throughout and asserts no acknowledged key ever reads
// back a wrong value; afterwards every seeded key is still readable.
func TestSlice3Coordinator(t *testing.T) {
	const numShards = 6
	tc := newTestCluster(t, 3, numShards, 1) // RF=1: each shard on exactly one node
	nodes := map[string]*Node{"n1": tc.nodes[0], "n2": tc.nodes[1], "n3": tc.nodes[2]}
	mc := MigrationClusterFromPeers(nodes, tc.peers)
	coord := Coordinator{MC: mc, MaxParallel: 3, StepTimeout: 20 * time.Second}
	ctx := context.Background()

	// --- Seed keys spread across all shards. ---
	seeded := map[string]string{}
	put := func(k, v []byte) bool {
		for attempt := 0; attempt < 50; attempt++ {
			if _, err := tc.client.Call(ctx, "put", ops.EncodePutArgs(k, v, 0)); err == nil {
				return true
			}
			time.Sleep(20 * time.Millisecond)
		}
		return false
	}
	for i := 0; i < 120; i++ {
		k := fmt.Appendf(nil, "key-%d", i)
		v := fmt.Appendf(nil, "val-%d", i)
		if !put(k, v) {
			t.Fatalf("seed put %q failed", k)
		}
		seeded[string(k)] = string(v)
	}

	// --- Background reader: any successful read must match the seeded value. ---
	var (
		stop    = make(chan struct{})
		wg      sync.WaitGroup
		badVals int64
	)
	wg.Add(1)
	go func() {
		defer wg.Done()
		rctx := context.Background()
		for {
			select {
			case <-stop:
				return
			default:
			}
			for k, want := range seeded {
				v, err := tc.client.Call(rctx, "get", ops.EncodeKeyArgs([]byte(k)))
				if err == nil && len(v) > 0 && string(v) != want {
					atomic.AddInt64(&badVals, 1)
					t.Errorf("stale/wrong read: key %q = %q, want %q", k, v, want)
				}
			}
			time.Sleep(5 * time.Millisecond)
		}
	}()

	// --- Grow: RF 1 -> 2. Every shard gains exactly one owner. ---
	plan, err := coord.Rebalance(ctx, tc.peers, numShards, 2)
	if err != nil {
		close(stop)
		wg.Wait()
		t.Fatalf("Rebalance RF=2: %v (plan moves=%d)", err, len(plan.Moves))
	}
	if len(plan.Moves) != numShards {
		t.Fatalf("RF1->2 plan: %d moves, want %d", len(plan.Moves), numShards)
	}
	assertPlacement(t, mc, computePlacement(tc.peers, numShards, 2))
	assertMetaConverged(t, tc.nodes[0], computePlacement(tc.peers, numShards, 2))
	assertEvenHosting(t, nodes, numShards, 2)

	// Idempotent: re-running the same target is a no-op.
	plan2, err := coord.Rebalance(ctx, tc.peers, numShards, 2)
	if err != nil || len(plan2.Moves) != 0 {
		t.Fatalf("idempotent re-run: moves=%d err=%v, want 0/nil", len(plan2.Moves), err)
	}

	// --- Decommission n3: target the {n1,n2} subset. n3's shards re-home. ---
	twoPeers := []Peer{tc.peers[0], tc.peers[1]} // n1, n2
	plan3, err := coord.Rebalance(ctx, twoPeers, numShards, 2)
	if err != nil {
		close(stop)
		wg.Wait()
		t.Fatalf("Rebalance decommission n3: %v (plan moves=%d)", err, len(plan3.Moves))
	}
	wantAfter := computePlacement(twoPeers, numShards, 2) // full replication on {n1,n2}
	assertPlacement(t, mc, wantAfter)
	assertMetaConverged(t, tc.nodes[0], wantAfter)
	// n3 hosts nothing; n1 and n2 host every shard.
	for s := 0; s < numShards; s++ {
		if tc.nodes[2].getShard(s) != nil {
			t.Fatalf("n3 should host no shards after decommission; still hosts shard %d", s)
		}
		if tc.nodes[0].getShard(s) == nil || tc.nodes[1].getShard(s) == nil {
			t.Fatalf("n1 and n2 should both host shard %d after decommission", s)
		}
	}

	// --- Stop the reader; assert no wrong values were ever observed. ---
	close(stop)
	wg.Wait()
	if n := atomic.LoadInt64(&badVals); n != 0 {
		t.Fatalf("%d stale/wrong reads observed during rebalance", n)
	}

	// --- Every seeded key is still readable with its value. ---
	for k, want := range seeded {
		var got []byte
		for attempt := 0; attempt < 50; attempt++ {
			v, err := tc.client.Call(ctx, "get", ops.EncodeKeyArgs([]byte(k)))
			if err == nil {
				got = v
				break
			}
			time.Sleep(20 * time.Millisecond)
		}
		if string(got) != want {
			t.Fatalf("lost key after rebalance: %q = %q, want %q", k, got, want)
		}
	}
	t.Logf("coordinator verified: %d keys survived RF1->2 then n3 decommission", len(seeded))
}

// assertPlacement checks every node's local routing view matches want (per-shard
// set equality), polling briefly since commits propagate asynchronously.
func assertPlacement(t *testing.T, mc MigrationCluster, want [][]string) {
	t.Helper()
	deadline := time.Now().Add(5 * time.Second)
	for {
		ok := true
		for _, nd := range mc.Nodes {
			got := nd.placementCopy()
			for s := range want {
				if s >= len(got) || !sameSet(got[s], want[s]) {
					ok = false
					break
				}
			}
			if !ok {
				break
			}
		}
		if ok {
			return
		}
		if !time.Now().Before(deadline) {
			for id, nd := range mc.Nodes {
				t.Logf("node %s placement=%v", id, nd.placementCopy())
			}
			t.Fatalf("placement did not converge to %v", want)
		}
		time.Sleep(50 * time.Millisecond)
	}
}

// assertMetaConverged checks the meta-Raft FSM placement on nd matches want.
func assertMetaConverged(t *testing.T, nd *Node, want [][]string) {
	t.Helper()
	deadline := time.Now().Add(5 * time.Second)
	for {
		st := nd.meta.FSM.State()
		ok := len(st.Placement) >= len(want)
		for s := 0; ok && s < len(want); s++ {
			if !sameSet(st.Placement[s], want[s]) {
				ok = false
			}
		}
		if ok {
			return
		}
		if !time.Now().Before(deadline) {
			t.Fatalf("meta placement did not converge: got %v want %v", st.Placement, want)
		}
		time.Sleep(50 * time.Millisecond)
	}
}

// assertEvenHosting checks each node hosts exactly numShards*rf/len(nodes)
// shard replicas (the sliding-window placement is exactly even when it divides).
func assertEvenHosting(t *testing.T, nodes map[string]*Node, numShards, rf int) {
	t.Helper()
	want := numShards * rf / len(nodes)
	for id, nd := range nodes {
		count := 0
		for s := 0; s < numShards; s++ {
			if nd.getShard(s) != nil {
				count++
			}
		}
		if count != want {
			t.Fatalf("node %s hosts %d shards, want %d (even spread)", id, count, want)
		}
	}
}
