// SPDX-License-Identifier: Apache-2.0

package cluster

import (
	"context"
	"errors"
	"time"

	hraft "github.com/hashicorp/raft"

	"github.com/rostamlabs/rostam/client"
)

// shardAdmin is the per-node control surface a single-shard migration drives.
// *Node satisfies it locally (in-process); remoteNode satisfies it over the
// network via the admin ops, so the coordinator on the meta leader can drive
// peers in a real multi-process cluster. The migration logic (MigrateShard /
// Coordinator) is written against this interface and is oblivious to whether a
// given node is local or remote.
type shardAdmin interface {
	addOwner(shardID int) error
	removeOwner(shardID int) error
	addVoter(shardID int, peerNodeID, peerRaftAddr string) error
	removeVoter(shardID int, peerNodeID string) error
	transferLeadership(shardID int, toNodeID, toRaftAddr string) error
	// setPlacement updates this node's hot-path routing view for shardID.
	setPlacement(shardID int, owners []string) error
	// commitMetaPlacement commits shardID's owners to meta-Raft. Returns
	// hraft.ErrNotLeader if this node is not the meta leader (the caller rotates
	// to another node), or errNoMeta in a single-node deployment.
	commitMetaPlacement(shardID, numShards int, owners []string, timeout time.Duration) error
	// status reports a shard's local Raft progress (leadership + indices).
	status(shardID int) adminStatus
	// placementCopy returns this node's full local routing view.
	placementCopy() [][]string
}

// errNoMeta marks a deployment with no meta-Raft (single node): there is no
// placement log to commit to; routing is updated locally only.
var errNoMeta = errors.New("cluster: no meta-Raft in this deployment")

// --- *Node: the local (in-process) shardAdmin. ---

func (n *Node) addOwner(shardID int) error    { return n.AddShardOwner(shardID) }
func (n *Node) removeOwner(shardID int) error { return n.RemoveShardOwner(shardID) }

func (n *Node) addVoter(shardID int, peerNodeID, peerRaftAddr string) error {
	return n.ShardAddVoter(shardID, peerNodeID, peerRaftAddr)
}

func (n *Node) removeVoter(shardID int, peerNodeID string) error {
	return n.ShardRemoveVoter(shardID, peerNodeID)
}

func (n *Node) transferLeadership(shardID int, toNodeID, toRaftAddr string) error {
	return n.ShardTransferLeadership(shardID, toNodeID, toRaftAddr)
}

func (n *Node) setPlacement(shardID int, owners []string) error {
	n.SetShardPlacement(shardID, owners)
	return nil
}

func (n *Node) commitMetaPlacement(shardID, numShards int, owners []string, timeout time.Duration) error {
	if n.meta == nil {
		return errNoMeta
	}
	return n.meta.ApplySetPlacement(shardID, numShards, owners, timeout)
}

func (n *Node) status(shardID int) adminStatus { return n.localStatus(shardID) }

// *Node already provides placementCopy() (used by Topology); it satisfies the
// shardAdmin method directly.

// --- remoteNode: a peer driven over the network via the admin ops. ---

// remoteNode is a shardAdmin backed by a client to one peer's server address.
// Every method is one admin-op round-trip to that peer, which dispatches it
// node-locally (Node.Call's admin bypass). Query methods degrade to a zero
// value on transport error — callers (findLeader/waitCaughtUp) poll, so a
// transient failure simply retries.
type remoteNode struct {
	nodeID string
	cl     *client.Client
}

func (r *remoteNode) do(op string, req adminReq) ([]byte, error) {
	args, err := gobEncode(req)
	if err != nil {
		return nil, err
	}
	return r.cl.Call(context.Background(), op, args)
}

func (r *remoteNode) addOwner(shardID int) error {
	_, err := r.do(opRBAddOwner, adminReq{ShardID: shardID})
	return err
}

func (r *remoteNode) removeOwner(shardID int) error {
	_, err := r.do(opRBRemoveOwner, adminReq{ShardID: shardID})
	return err
}

func (r *remoteNode) addVoter(shardID int, peerNodeID, peerRaftAddr string) error {
	_, err := r.do(opRBAddVoter, adminReq{ShardID: shardID, PeerID: peerNodeID, RaftAddr: peerRaftAddr})
	return err
}

func (r *remoteNode) removeVoter(shardID int, peerNodeID string) error {
	_, err := r.do(opRBRemoveVoter, adminReq{ShardID: shardID, PeerID: peerNodeID})
	return err
}

func (r *remoteNode) transferLeadership(shardID int, toNodeID, toRaftAddr string) error {
	_, err := r.do(opRBTransfer, adminReq{ShardID: shardID, PeerID: toNodeID, RaftAddr: toRaftAddr})
	return err
}

func (r *remoteNode) setPlacement(shardID int, owners []string) error {
	_, err := r.do(opRBSetPlacement, adminReq{ShardID: shardID, Owners: owners})
	return err
}

// commitMetaPlacement is never the remote node's job: the coordinator runs on
// the meta leader and commits placement through its own local *Node. Reporting
// ErrNotLeader keeps commitPlacement rotating to the local leader.
func (r *remoteNode) commitMetaPlacement(_, _ int, _ []string, _ time.Duration) error {
	return hraft.ErrNotLeader
}

func (r *remoteNode) status(shardID int) adminStatus {
	b, err := r.do(opRBStatus, adminReq{ShardID: shardID})
	if err != nil {
		return adminStatus{}
	}
	var st adminStatus
	if err := gobDecode(b, &st); err != nil {
		return adminStatus{}
	}
	return st
}

func (r *remoteNode) placementCopy() [][]string {
	b, err := r.do(opRBPlacement, adminReq{})
	if err != nil {
		return nil
	}
	var p adminPlacement
	if err := gobDecode(b, &p); err != nil {
		return nil
	}
	return p.Placement
}
