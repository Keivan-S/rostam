// SPDX-License-Identifier: Apache-2.0

package cluster

import (
	"context"
	"errors"
	"os"
	"path/filepath"
	"testing"
	"time"

	"github.com/rostamlabs/rostam/ops"
	"github.com/rostamlabs/rostam/shard"
)

// TestThreeNodeWASMRegistration registers a WASM op via the cluster's shared
// client, waits for Raft to replicate the registration to all nodes, then
// verifies the op is callable on every individual node by connecting a
// dedicated single-server client to each.
func TestThreeNodeWASMRegistration(t *testing.T) {
	const wasmPath = "../wasm/testdata/incr.wasm"
	if _, err := os.Stat(wasmPath); err != nil {
		t.Skipf("incr.wasm not found (%v); skipping", err)
	}

	wasmBytes, err := os.ReadFile(wasmPath)
	if err != nil {
		t.Fatalf("read incr.wasm: %v", err)
	}

	tc := newTestCluster(t, 3, 4)
	ctx := context.Background()

	// Settle the cluster BEFORE snapshotting the log heads. newTestCluster's
	// waitReady only requires SOME node to report a leader per shard, so a
	// follower can still sit at last_log_index=0 here. Snapshotting that as
	// "before" made the growth assertion below VACUOUS: the leader's own election
	// no-op entry, replicated to that follower, is by itself enough to satisfy
	// got > before, with or without the registration ever arriving. The
	// single-node test in wasm_broadcast_test.go settles for the same reason and
	// says so.
	waitClusterApplied(t, tc.nodes)

	// Snapshot every node's per-group log head. The registration is the only op
	// this test issues, so any group whose log does not grow past this point never
	// received it. This is what exercises the CROSS-NODE leg of the broadcast: a
	// node leads only some of the 4 groups, so the rest are reached by forwarding
	// __register_wasm_shard__ to the group's leader.
	before := make([][]uint64, len(tc.nodes))
	for i, n := range tc.nodes {
		if n != nil {
			before[i] = lastLogIndexes(t, n)
		}
	}

	// Register the WASM op through the cluster's multi-server client.
	// RegisterWASM routes to the Raft leader and blocks until the log entry is
	// committed on a majority. It replicates through the META group, whose leader
	// election waitReady does NOT gate on (waitReady covers the data shards), so
	// on a CPU-starved CI runner the registration can race ahead of meta election
	// and exhaust its NotLeader hops. Retry through the bringup window: a transient
	// pre-commit NotLeader is expected, not a failure (retry is safe — a failed
	// registration commits nothing). Finite so a real failure still fails loud.
	regDeadline := time.Now().Add(cpuScaled(30 * time.Second))
	for {
		pushReport, err := tc.client.RegisterWASM(ctx, ops.WASMRegistration{
			Name:       "wasm_incr",
			Kind:       ops.OpReadWrite,
			Blob:       ops.WASMBlobFingerprint(wasmBytes),
			ExportName: "apply",
			// "raw" makes the op ROUTABLE (RegisterRoutableCrossShard), so its
			// invocations land in shardOf(args)'s Raft log rather than shard 0's.
			// That is the ONLY configuration in which the defect this test guards
			// is reachable: with an empty handle the op is shardless, every
			// invocation goes to group 0 alongside the registration, and the
			// cross-group ordering that motivates the broadcast never arises. The
			// test used to register with an empty handle and therefore proved
			// nothing about it.
		}, wasmBytes)
		if err == nil {
			// Every node in this cluster is up, so every member must have taken
			// the module's bytes and rendered a compile verdict. A non-empty
			// report here means a member was skipped on a healthy cluster, which
			// is the state that turns into an op that group cannot run.
			if pushReport != "" {
				t.Fatalf("every member is up, so the blob push must have reached all of them; report: %s", pushReport)
			}
			break
		}
		if time.Now().After(regDeadline) {
			t.Fatalf("RegisterWASM did not succeed within budget: %v", err)
		}
		time.Sleep(200 * time.Millisecond)
	}

	// Wait for all nodes' FSMs to apply the committed log entry before
	// querying. Without this, a per-node call can race the replication
	// even though the registration RPC returned. waitClusterApplied (not the
	// per-node waitAllApplied) is what actually proves that: it holds every
	// hosting node to the LEADER's last_log_index, whereas a per-node
	// applied>=own-last check passes trivially on a follower that was never
	// sent the entry.
	waitClusterApplied(t, tc.nodes)

	// AND THEN WAIT FOR THE THING THIS TEST ACTUALLY NEEDS, because every
	// index-based barrier is a PROXY for it and no proxy is exact.
	//
	// applied_index cannot be one: hashicorp/raft advances it when an entry is
	// ENQUEUED onto fsmMutateCh, not when the FSM goroutine has run it (see
	// waitClusterApplied). Requiring fsm_pending==0 narrows that from 128 entries
	// to one in-flight batch — measured, it took this test from ~20-25% failures
	// under load to ~7% — but a batch already dequeued is still a batch not yet
	// applied, so the window is narrowed, not closed.
	//
	// The route gate is not a proxy. checkWASMRouteGate reads exactly this map,
	// and the hook that fills it is the one whose completion the barriers above
	// are trying to infer. So ask it directly.
	waitWASMRouteGateOpen(t, tc.nodes, "wasm_incr")

	// EVERY shard group on EVERY node must now carry the registration. Landing it
	// in shard 0's group alone is the divergence bug: a routable WASM op is
	// invoked from shardOf(key)'s log, and a replica caught up on that group but
	// lagging on group 0 has no way to serve it. Reaching every group is what
	// makes the op USABLE everywhere: under the route gate (checkWASMRouteGate) a
	// group whose log never received the registration is a group no node will
	// propose an invocation into, so every key that routes there fails with a
	// client-visible error until a retry lands the entry.
	for i, n := range tc.nodes {
		if n == nil {
			continue
		}
		for j, got := range lastLogIndexes(t, n) {
			if n.shards[j] == nil {
				continue // not hosted here
			}
			if got <= before[i][j] {
				t.Errorf("node %d shard %d: log did not grow (last_log_index %d -> %d); the registration never reached this group",
					i, j, before[i][j], got)
			}
		}
	}

	// Verify the op is callable through the cluster-wide client for a key routing
	// to EACH of the 4 groups, from EVERY node, and that the nodes agree on the
	// result. A routable op's invocation is logged in shardOf(key)'s group, so
	// this is what actually walks the cross-group path: calling it once with a
	// nil key (as this test used to) exercises one group and says nothing about
	// the other three.
	const numShards = 4
	for i := 0; i < numShards; i++ {
		key := keyForShard(t, i, numShards)

		// Through the cluster client first: this genuinely EXECUTES the op in
		// group i, which is the cross-group path a routable op takes.
		want, err := tc.client.Call(ctx, "wasm_incr", key)
		if err != nil {
			t.Fatalf("cluster client: Call wasm_incr with a key routing to shard %d: %v", i, err)
		}

		// Then per node. A follower answers NotLeader for a write op — Node.Call
		// does not hop for ordinary ops — and that is a PASS here: reaching the
		// routing stage at all proves the op was found in that node's registry.
		// The failure this is looking for is ErrUnknownOp, which is returned
		// BEFORE any routing and means the registration never landed on the node.
		for ni, n := range tc.nodes {
			if n == nil {
				continue
			}
			got, err := n.Call("wasm_incr", key)
			if err != nil {
				if errors.Is(err, ErrUnknownOp) {
					t.Errorf("node %d: wasm_incr is not registered (key routes to shard %d): the registration never reached this node", ni, i)
					continue
				}
				var nle *shard.NotLeaderError
				if errors.As(err, &nle) {
					continue // registered here, just not this group's leader
				}
				t.Errorf("node %d: Call wasm_incr with a key routing to shard %d: %v", ni, i, err)
				continue
			}
			if string(got) != string(want) {
				t.Errorf("node %d shard %d: wasm_incr returned %q, but the cluster client got %q — replicas are executing different code",
					ni, i, got, want)
			}
		}
	}

	// Verify each node persisted the module — that's the canonical proof the FSM
	// apply landed on every replica. The bytes live at their content address, and
	// the metadata sidecar is what names them as this op; both must be present or
	// the reload finds nothing. The reload-on-restart test in wasm_node_test.go
	// separately proves callability after restart.
	for i, n := range tc.nodes {
		for _, path := range []string{
			blobPathFor(t, n.cfg.DataDir, wasmBytes),
			filepath.Join(n.cfg.DataDir, "wasm", "wasm_incr.json"),
		} {
			if _, err := os.Stat(path); err != nil {
				t.Errorf("node %d: expected %s on disk: %v", i, path, err)
			}
		}
	}
}
