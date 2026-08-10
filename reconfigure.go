// SPDX-License-Identifier: Apache-2.0

package rostam

import (
	"context"

	"github.com/rostamlabs/rostam/client"
	"github.com/rostamlabs/rostam/cluster"
)

// RebalanceResult summarizes a triggered online rebalance: how many shard moves
// the plan contained and how they resolved. Alias of cluster.RebalanceResult.
type RebalanceResult = cluster.RebalanceResult

// Reconfigure connects to a running cluster (any of serverAddrs) and triggers an
// online rebalance to the target member set + replication factor, returning when
// it completes. This is the client side of the operator surface — the same
// action `rostam-server -reconfigure` performs.
//
// target is the desired membership: to grow, include the new node(s); to
// decommission, omit the departing node(s) (they keep running and forwarding
// until their shards have re-homed). rf is the target replication factor (0 or
// >= len(target) means full replication). The call blocks until the rebalance
// finishes; use a context deadline sized to the data volume being moved.
func Reconfigure(ctx context.Context, serverAddrs []string, target []Peer, rf int) (RebalanceResult, error) {
	cl, err := client.New(client.Config{Servers: serverAddrs, MaxNotLeaderHops: 5})
	if err != nil {
		return RebalanceResult{}, err
	}
	defer func() { _ = cl.Close() }()

	ct := make([]cluster.Peer, len(target))
	for i, p := range target {
		ct[i] = cluster.Peer{NodeID: p.NodeID, RaftAddr: p.RaftAddr, ServerAddr: p.ServerAddr}
	}
	return cluster.TriggerRebalance(ctx, cl, ct, rf)
}
