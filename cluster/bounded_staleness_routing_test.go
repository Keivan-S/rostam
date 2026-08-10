// SPDX-License-Identifier: Apache-2.0

package cluster

import (
	"testing"

	"github.com/rostamlabs/rostam/ops"
)

// TestRouteToLeaderBoundedStaleness is the #1 routing-regression guard: it asserts
// routeToLeader() is FALSE for AnyReplica AND BoundedStaleness (both serve from any
// replica) and TRUE for LeaderOnly + Linearizable (leader-pinned). If someone
// reverts routeToLeader() to `c >= LeaderOnly`, BoundedStaleness (== 3) would pin to
// the leader and this test goes RED — defeating the whole leader-offload feature.
func TestRouteToLeaderBoundedStaleness(t *testing.T) {
	cases := []struct {
		name string
		c    Consistency
		want bool
	}{
		{"AnyReplica", AnyReplica, false},
		{"LeaderOnly", LeaderOnly, true},
		{"Linearizable", Linearizable, true},
		{"BoundedStaleness", BoundedStaleness, false},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			if got := tc.c.routeToLeader(); got != tc.want {
				t.Fatalf("%s.routeToLeader() = %v, want %v", tc.name, got, tc.want)
			}
		})
	}
}

// TestConsistencyEnumMatchesOps proves the cluster.Consistency enum stays
// wire-aligned with the ops.Consistency* constants (the rc byte threads from the API
// edge through fan-out to the shard, so a drift would mis-route reads).
func TestConsistencyEnumMatchesOps(t *testing.T) {
	if uint8(AnyReplica) != ops.ConsistencyAnyReplica ||
		uint8(LeaderOnly) != ops.ConsistencyLeaderOnly ||
		uint8(Linearizable) != ops.ConsistencyLinearizable ||
		uint8(BoundedStaleness) != ops.ConsistencyBoundedStaleness {
		t.Fatalf("cluster.Consistency enum drifted from ops: any=%d leader=%d lin=%d bounded=%d (ops: %d %d %d %d)",
			AnyReplica, LeaderOnly, Linearizable, BoundedStaleness,
			ops.ConsistencyAnyReplica, ops.ConsistencyLeaderOnly, ops.ConsistencyLinearizable, ops.ConsistencyBoundedStaleness)
	}
}
