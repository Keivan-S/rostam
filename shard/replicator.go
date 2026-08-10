// SPDX-License-Identifier: Apache-2.0

package shard

import (
	"time"

	"github.com/rostamlabs/rostam/raft"
)

// replicator is the data-plane replication engine a Store writes and reads
// through. It is the seam that lets a shard run on either the default per-shard
// Raft group (raftReplicator, a *raft.Node — byte-identical to before this
// abstraction existed) or, when Config.ReplicationMode selects it, the
// primary-backup/ISR engine (pbReplicator; see shard/pbisr/DESIGN.md).
//
// The method set is exactly what Store already called on *raft.Node, so the
// Raft path is a zero-behavior-change extraction: *raft.Node satisfies this
// interface directly (see the compile-time assertion below).
//
// Method semantics under pb-isr (a v1 pb engine
// returns an explicit unimplemented error for the ones not yet built):
//   - ApplyIndexed: assign (epoch,seq), apply, replicate to ISR, ack when the
//     min-ISR floor is met; the returned index is the seq.
//   - IsLeader/LeaderAddr: "am I the current primary?" / who is.
//   - VerifyLeader/Barrier/CommitIndex/AppliedIndex/LastIndex: the linearizable
//     read barrier — confirm still-primary for this epoch + catch up.
//   - AddVoter/RemoveServer/LeadershipTransferToServer: ISR / primary changes
//     driven by the control plane.
type replicator interface {
	ApplyIndexed(data []byte, timeout time.Duration) (resp any, index uint64, err error)
	IsLeader() bool
	LeaderAddr() string
	CommitIndex() uint64
	AppliedIndex() uint64
	LastIndex() uint64
	VerifyLeader() error
	Barrier(timeout time.Duration) error
	AddVoter(id, addr string, prevIndex uint64, timeout time.Duration) error
	RemoveServer(id string, prevIndex uint64, timeout time.Duration) error
	LeadershipTransferToServer(id, addr string) error
	Stats() map[string]string
	Shutdown() error
}

// The default Raft path IS the replicator — no wrapper, no indirection cost.
var _ replicator = (*raft.Node)(nil)
