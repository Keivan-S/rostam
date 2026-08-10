// SPDX-License-Identifier: Apache-2.0

package ops

import "testing"

func TestTopologyOwnerAddr(t *testing.T) {
	top := Topology{
		NumShards: 3,
		Members: []TopologyMember{
			{NodeID: "n1", ServerAddr: "a:1"},
			{NodeID: "n2", ServerAddr: "a:2"},
			{NodeID: "n3", ServerAddr: "a:3"},
		},
		Placement: [][]string{
			{"n2", "n3"}, // shard 0 owned by n2 (first) then n3
			{"n1"},       // shard 1 owned by n1
			{"n3", "n1"}, // shard 2 owned by n3 (first)
		},
	}
	cases := map[int]string{0: "a:2", 1: "a:1", 2: "a:3"}
	for shard, want := range cases {
		if got := top.OwnerAddr(shard); got != want {
			t.Errorf("OwnerAddr(%d) = %q, want %q", shard, got, want)
		}
	}
	// Out-of-range and empty placement return "".
	if got := top.OwnerAddr(99); got != "" {
		t.Errorf("OwnerAddr(99) = %q, want \"\"", got)
	}
	if got := (Topology{}).OwnerAddr(0); got != "" {
		t.Errorf("empty topology OwnerAddr = %q, want \"\"", got)
	}
}
