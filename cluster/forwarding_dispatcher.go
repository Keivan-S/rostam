// SPDX-License-Identifier: Apache-2.0

package cluster

import (
	"context"
	"errors"

	"github.com/rostamlabs/rostam/shard"
)

// LeaderFollowingDispatcher wraps a Node so that a NotLeader result for a shard
// the node *hosts* is transparently forwarded to that shard's leader, instead of
// surfacing the hint to the caller.
//
// Plain Node.Call returns a NotLeaderError (with the leader's server address) for
// a write that lands on a follower of a hosted shard. The binary TCP client
// follows that hint itself (one cheap redirect), so the TCP transport uses the
// node directly. HTTP and gRPC clients can't act on the binary hint — they'd see
// a 503 / Unavailable — so those transports use this wrapper, which performs the
// redirect server-side over the existing peer-forwarding path. The leader then
// executes the op and its result flows back. (The partition case — an op for a
// shard this node does not host — is already forwarded inside Node.Call, so it
// never reaches here as a NotLeaderError.)
type LeaderFollowingDispatcher struct{ n *Node }

// LeaderFollowingDispatcher returns a dispatcher (Call + LeaderAddr) that follows
// hosted-shard NotLeader results to the leader. Intended for the HTTP/gRPC
// transports; the TCP transport should use the Node directly.
func (n *Node) LeaderFollowingDispatcher() *LeaderFollowingDispatcher {
	return &LeaderFollowingDispatcher{n: n}
}

// Call runs the op on the node and, if it returns a hosted-shard NotLeader hint,
// forwards it once to the leader's server address. The per-peer client follows
// any further NotLeader hops itself, so a momentarily stale hint still lands on
// the current leader.
func (d *LeaderFollowingDispatcher) Call(name string, args []byte) ([]byte, error) {
	res, err := d.n.Call(name, args)
	if err == nil {
		return res, nil
	}
	var nle *shard.NotLeaderError
	if !errors.As(err, &nle) || nle.LeaderAddr == "" {
		return res, err // not a followable NotLeader hint
	}
	cl, cerr := d.n.peerClient(nle.LeaderAddr)
	if cerr != nil {
		return nil, err // can't reach the leader; surface the original hint
	}
	return cl.Call(context.Background(), name, args)
}

// LeaderAddr mirrors the node's leader address for shard 0 (the Dispatcher
// contract's secondary hint source).
func (d *LeaderFollowingDispatcher) LeaderAddr() string { return d.n.LeaderAddr() }
