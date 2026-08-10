// SPDX-License-Identifier: Apache-2.0

package cluster

import (
	"context"
	"errors"
	"fmt"
	"sync/atomic"
	"time"
)

// Bounded-staleness follower reads (ConsistencyBoundedStaleness) confirm freshness
// via a coalesced RTT to the SHARD leader for its true committed frontier
// (__shard_readindex__), mirroring the meta readindex (cluster/meta_readindex.go).
// The leader answers a tiny, coalesced frontier ping (VerifyLeader + report
// CommitIndex, NO Barrier wait — this is a cheap freshness probe, not a catch-up)
// while the follower absorbs the expensive vector-search data path: leader CPU
// offload with a freshness SLO. The RTT fails CLOSED on a partition (the shard
// Store's serveBoundedStaleness turns an error into a NotLeaderError ⇒ route to the
// leader), so a partitioned follower never serves data beyond its bound.

// errNotShardLeader is the internal not-leader sentinel: this node is not the leader
// of the requested shard (not hosted, not leader, or VerifyLeader reported
// not-leader). The follower caller treats it as "re-route to the leader / fail
// closed". It is not exported: it never reaches a client.
var errNotShardLeader = errors.New("cluster: shard readindex: not the shard-Raft leader")

// shardReadIndexForwardHook holds an optional test-only observer fired ONCE each time
// a follower actually FORWARDS a __shard_readindex__ op to the shard leader (issues
// the leader RTT). Stored atomically; nil in prod ⇒ zero overhead. Used to PROVE the
// coalescer fires ONE ping per N concurrent bounded reads on a follower.
var shardReadIndexForwardHook atomic.Pointer[func()]

// SetShardReadIndexForwardHook installs (or clears with nil) the __shard_readindex__
// forward observer. Test-only; nil in production.
func SetShardReadIndexForwardHook(fn func()) {
	if fn == nil {
		shardReadIndexForwardHook.Store(nil)
		return
	}
	shardReadIndexForwardHook.Store(&fn)
}

// shardLeaderFrontier runs ON the shard leader (locally for a leader, AND inside
// handleShardReadIndex). It confirms (VerifyLeader, a quorum heartbeat ⇒ no stale
// partitioned leader) it is the leader of shard idx and returns its committed
// frontier (CommitIndex). Unlike the meta readindex it does NOT Barrier-drain — a
// bounded-staleness read only needs the leader's committed frontier to measure the
// follower's lag against; the follower decides locally whether its applied index is
// within bound. errNotShardLeader when this node does not lead shard idx.
func (n *Node) shardLeaderFrontier(idx int, _ time.Time) (uint64, error) {
	s := n.getShard(idx)
	if s == nil || !s.IsLeader() {
		return 0, errNotShardLeader
	}
	if err := s.VerifyLeader(); err != nil {
		// A follower / lost-quorum leader: not authoritative for the frontier.
		return 0, errNotShardLeader
	}
	return s.CommitIndex(), nil
}

// fetchShardLeaderFrontier resolves the current leader of shard idx and obtains its
// committed frontier: if the leader is this node it answers locally; otherwise it
// forwards __shard_readindex__ to the leader's server address. It returns an error
// (the shard Store fails closed ⇒ NotLeaderError ⇒ route to leader) when the leader
// is unknown, unreachable, or replies not-leader. Mirrors fetchMetaLeaderFrontier.
func (n *Node) fetchShardLeaderFrontier(idx int, deadline time.Time) (uint64, error) {
	addr := n.leaderServerAddr(idx)
	if addr == "" {
		return 0, errNotShardLeader
	}
	if addr == n.serverAddrFor(n.cfg.NodeID) {
		// The leader resolves to us — answer locally (no self-RTT).
		return n.shardLeaderFrontier(idx, deadline)
	}
	cl, err := n.peerClient(addr)
	if err != nil {
		return 0, err
	}
	if h := shardReadIndexForwardHook.Load(); h != nil {
		(*h)()
	}
	budget := time.Until(deadline)
	if budget < 0 {
		budget = 0
	}
	args, err := gobEncode(shardReadIndexReq{
		Version:     shardReadIndexVersion,
		ShardIdx:    uint32(idx),
		BudgetNanos: int64(budget),
	})
	if err != nil {
		return 0, err
	}
	ctx, cancel := context.WithDeadline(context.Background(), deadline)
	defer cancel()
	raw, err := cl.Call(ctx, opShardReadIndexName, args)
	if err != nil {
		return 0, err
	}
	var reply shardReadIndexReply
	if err := gobDecode(raw, &reply); err != nil {
		return 0, fmt.Errorf("cluster: __shard_readindex__ decode reply: %w", err)
	}
	if !reply.OK {
		return 0, errNotShardLeader
	}
	return reply.Frontier, nil
}
