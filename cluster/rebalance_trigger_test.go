// SPDX-License-Identifier: Apache-2.0

package cluster

import (
	"context"
	"fmt"
	"testing"
	"time"

	"github.com/rostamlabs/rostam/ops"
)

// TestSlice4RemoteRebalance drives a full rebalance entirely over the network:
// it triggers the __rebalance__ op through the cluster client (TriggerRebalance),
// so the coordinator runs on the meta leader and drives its peers via the
// network admin ops (remoteNode) — not in-process *Node handles. It grows RF
// 1->2 (every shard gains an owner) then decommissions n3 (its shards re-home),
// asserting redistribution and that every seeded key stays readable.
func TestSlice4RemoteRebalance(t *testing.T) {
	const numShards = 6
	tc := newTestCluster(t, 3, numShards, 1) // RF=1 over real TCP servers
	nodes := map[string]*Node{"n1": tc.nodes[0], "n2": tc.nodes[1], "n3": tc.nodes[2]}
	ctx := context.Background()

	// --- Seed keys across all shards. ---
	seeded := map[string]string{}
	for i := 0; i < 60; i++ {
		k := fmt.Appendf(nil, "rk-%d", i)
		v := fmt.Appendf(nil, "rv-%d", i)
		ok := false
		for attempt := 0; attempt < 50; attempt++ {
			if _, err := tc.client.Call(ctx, "put", ops.EncodePutArgs(k, v, 0)); err == nil {
				ok = true
				break
			}
			time.Sleep(20 * time.Millisecond)
		}
		if !ok {
			t.Fatalf("seed put %q failed", k)
		}
		seeded[string(k)] = string(v)
	}
	readAll := func(stage string) {
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
				t.Fatalf("%s: lost key %q = %q, want %q", stage, k, got, want)
			}
		}
	}

	// --- Grow RF 1 -> 2, triggered over the network. ---
	gctx, cancel := context.WithTimeout(ctx, 60*time.Second)
	defer cancel()
	res, err := TriggerRebalance(gctx, tc.client, tc.peers, 2)
	if err != nil {
		t.Fatalf("TriggerRebalance RF=2: %v", err)
	}
	if res.Moves != numShards || res.Done != numShards || res.Failed != 0 {
		t.Fatalf("RF1->2 result = %+v, want Moves=%d Done=%d Failed=0", res, numShards, numShards)
	}
	assertEvenHosting(t, nodes, numShards, 2)
	assertPlacement(t, MigrationClusterFromPeers(nodes, tc.peers), computePlacement(tc.peers, numShards, 2))
	readAll("after RF1->2")

	// --- Decommission n3, triggered over the network: target the {n1,n2} subset. ---
	twoPeers := []Peer{tc.peers[0], tc.peers[1]}
	dctx, cancel2 := context.WithTimeout(ctx, 60*time.Second)
	defer cancel2()
	res2, err := TriggerRebalance(dctx, tc.client, twoPeers, 2)
	if err != nil {
		t.Fatalf("TriggerRebalance decommission n3: %v", err)
	}
	if res2.Failed != 0 {
		t.Fatalf("decommission result = %+v, want Failed=0", res2)
	}
	wantAfter := computePlacement(twoPeers, numShards, 2)
	assertPlacement(t, MigrationClusterFromPeers(nodes, tc.peers), wantAfter)
	for s := 0; s < numShards; s++ {
		if tc.nodes[2].getShard(s) != nil {
			t.Fatalf("n3 should host no shards after decommission; still hosts shard %d", s)
		}
		if tc.nodes[0].getShard(s) == nil || tc.nodes[1].getShard(s) == nil {
			t.Fatalf("n1 and n2 should both host shard %d after decommission", s)
		}
	}
	readAll("after n3 decommission")
	t.Logf("remote rebalance verified: %d keys survived RF1->2 + n3 decommission, all driven over TCP", len(seeded))
}
