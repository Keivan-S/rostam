// SPDX-License-Identifier: Apache-2.0

package client

import (
	"testing"
	"time"

	"github.com/cespare/xxhash/v2"

	"github.com/rostamlabs/rostam/ops"
)

// TestPickInitialTargetClientSharding verifies the client-side sharding fast
// path: with a cached topology, the client routes a key directly to its shard's
// leader when known, and to an owner (via Placement) when the leader is unknown
// — never falling back to the round-robin server list.
func TestPickInitialTargetClientSharding(t *testing.T) {
	reg := ops.NewRegistry()
	if err := ops.RegisterBuiltins(reg); err != nil {
		t.Fatal(err)
	}
	c, err := New(Config{
		Servers:                 []string{"fallback:9999"}, // must NOT be chosen
		Ops:                     reg,
		TopologyRefreshInterval: time.Second,
	})
	if err != nil {
		t.Fatal(err)
	}
	defer c.Close()

	const numShards = 4
	members := []ops.TopologyMember{
		{NodeID: "n1", ServerAddr: "addr-n1"},
		{NodeID: "n2", ServerAddr: "addr-n2"},
	}
	placement := [][]string{{"n1"}, {"n2"}, {"n1"}, {"n2"}}

	key := []byte("some-key")
	args := ops.EncodePutArgs(key, []byte("v"), 0)
	shard := int(xxhash.Sum64(key) % numShards)
	wantOwnerAddr := members[map[int]int{0: 0, 1: 1, 2: 0, 3: 1}[shard]].ServerAddr

	// Leader unknown for this shard → route to the owner from Placement.
	c.topology.set(ops.Topology{
		NumShards: numShards,
		Members:   members,
		Leaders:   make([]string, numShards), // all ""
		Placement: placement,
	})
	if got := c.pickInitialTarget("put", args); got != wantOwnerAddr {
		t.Errorf("leader unknown: target = %q, want owner %q (shard %d)", got, wantOwnerAddr, shard)
	}

	// Leader known → prefer the leader over the owner.
	leaders := make([]string, numShards)
	leaders[shard] = "addr-leader"
	c.topology.set(ops.Topology{
		NumShards: numShards,
		Members:   members,
		Leaders:   leaders,
		Placement: placement,
	})
	if got := c.pickInitialTarget("put", args); got != "addr-leader" {
		t.Errorf("leader known: target = %q, want %q", got, "addr-leader")
	}
}
