// SPDX-License-Identifier: Apache-2.0

package cluster

import (
	"context"
	"errors"
	"fmt"
	"time"

	hraft "github.com/hashicorp/raft"

	"github.com/rostamlabs/rostam/client"
)

// rebalanceParallel and rebalanceStepTimeout are the coordinator settings used
// by the network-triggered __rebalance__ op. Conservative defaults: a handful
// of shards move at once, each bounded so a stuck migration can't hang the op.
const (
	rebalanceParallel    = 4
	rebalanceStepTimeout = 20 * time.Second
)

// handleRebalance is the operator-facing trigger (the __rebalance__ op). It runs
// the coordinator on the meta-Raft leader, redistributing shards to the target
// member set / replication factor. If the receiving node is not the meta leader,
// it forwards the op there (the leader never re-forwards, so there is no loop).
//
// Args (adminReq): Members = the target member set (empty → current cfg.Peers,
// a no-op); RF = the target replication factor.
func (n *Node) handleRebalance(args []byte) ([]byte, error) {
	if n.meta == nil {
		return nil, errors.New("cluster: __rebalance__ requires multi-node mode")
	}
	req, err := decodeAdminReq(args)
	if err != nil {
		return nil, err
	}

	// Must run on the meta leader (it owns placement). Forward if we are not it.
	if n.meta.Raft.State() != hraft.Leader {
		addr := n.metaLeaderServerAddr()
		if addr == "" || addr == n.serverAddrFor(n.cfg.NodeID) {
			return nil, errors.New("cluster: __rebalance__: no meta-Raft leader yet")
		}
		cl, err := n.peerClient(addr)
		if err != nil {
			return nil, err
		}
		return cl.Call(context.Background(), opRBRebalanceName, args)
	}

	target := req.Members
	if len(target) == 0 {
		target = n.cfg.Peers // default: current membership (no-op rebalance)
	}

	coord := &Coordinator{
		MC:          n.buildMigrationCluster(),
		MaxParallel: rebalanceParallel,
		StepTimeout: rebalanceStepTimeout,
	}
	plan, rerr := coord.Rebalance(context.Background(), target, n.cfg.NumShards, req.RF)
	stats := coord.Stats()
	res := RebalanceResult{Moves: len(plan.Moves), Done: stats.Done, Failed: stats.Failed}
	out, encErr := gobEncode(res)
	if encErr != nil {
		return nil, encErr
	}
	return out, rerr
}

// buildMigrationCluster assembles a MigrationCluster spanning every known member
// (cfg.Peers — the union of current and target owners, so a node being
// decommissioned is still reachable to be torn down). The local node is an
// in-process shardAdmin; every peer is a network-backed remoteNode reusing the
// node's pooled forwarding clients.
func (n *Node) buildMigrationCluster() MigrationCluster {
	nodes := make(map[string]shardAdmin, len(n.cfg.Peers))
	raftAddr := make(map[string]string, len(n.cfg.Peers))
	for _, p := range n.cfg.Peers {
		raftAddr[p.NodeID] = p.RaftAddr
		if p.NodeID == n.cfg.NodeID {
			nodes[p.NodeID] = n
			continue
		}
		cl, err := n.peerClient(p.ServerAddr)
		if err != nil {
			continue // unreachable peer; migrations touching it will surface the error
		}
		nodes[p.NodeID] = &remoteNode{nodeID: p.NodeID, cl: cl}
	}
	return MigrationCluster{
		Nodes:     nodes,
		RaftAddr:  raftAddr,
		NumShards: n.cfg.NumShards,
		Local:     n.cfg.NodeID,
	}
}

// metaLeaderServerAddr returns the client-facing server address of the current
// meta-Raft leader, or "" if unknown. The meta leader's Raft transport addr is
// the node's shared mux addr, so it maps through cfg.Peers.
func (n *Node) metaLeaderServerAddr() string {
	addr, _ := n.meta.Raft.LeaderWithID()
	if addr == "" {
		return ""
	}
	for _, p := range n.cfg.Peers {
		if p.RaftAddr == string(addr) {
			return p.ServerAddr
		}
	}
	return ""
}

// TriggerRebalance sends a __rebalance__ op through cl to redistribute shards to
// the target member set + replication factor. A convenience wrapper over the
// wire op for operator tooling (rostam-server reconfigure). It returns the
// server's RebalanceResult.
func TriggerRebalance(ctx context.Context, cl *client.Client, target []Peer, rf int) (RebalanceResult, error) {
	args, err := gobEncode(adminReq{Members: target, RF: rf})
	if err != nil {
		return RebalanceResult{}, err
	}
	raw, err := cl.Call(ctx, opRBRebalanceName, args)
	if err != nil {
		return RebalanceResult{}, err
	}
	var res RebalanceResult
	if err := gobDecode(raw, &res); err != nil {
		return RebalanceResult{}, fmt.Errorf("cluster: decode rebalance result: %w", err)
	}
	return res, nil
}
