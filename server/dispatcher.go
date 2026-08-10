// SPDX-License-Identifier: Apache-2.0

package server

// Dispatcher is the minimal interface a server uses to dispatch frames
// into a backing store. Both *shard.Store and *cluster.Node satisfy it.
//
// Call runs an op by name and returns the encoded result bytes (or an
// error that the server maps to a wire status code).
//
// LeaderAddr returns an address of a current Raft leader, or an empty
// string when no leader is known. For *shard.Store this is the leader
// of its single Raft group. For *cluster.Node (multi-node)
// this is shard 0's leader — used as a fallback when the per-shard
// hint from *shard.NotLeaderError is absent.
//
// Per-shard NotLeader hints are the primary mechanism: server's
// mapResult prefers *shard.NotLeaderError.LeaderAddr from the
// dispatched Call's error chain. LeaderAddr() is only consulted
// when the wrapped hint is empty.
type Dispatcher interface {
	Call(name string, args []byte) ([]byte, error)
	LeaderAddr() string
}
