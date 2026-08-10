// SPDX-License-Identifier: Apache-2.0

package shard

import "errors"

// NotLeaderError indicates the local shard.Store is not the Raft leader
// for its group. LeaderAddr names the current leader (per the Raft
// library's view); empty when no leader is known.
//
// All NotLeaderError values satisfy errors.Is(err, ErrNotLeader). Use
// errors.As(err, &nle) to extract the leader hint.
type NotLeaderError struct {
	LeaderAddr string
}

func (e *NotLeaderError) Error() string { return "shard: not leader" }

// Is reports whether target is a *NotLeaderError, enabling errors.Is matching
// against ErrNotLeader for any NotLeaderError value regardless of LeaderAddr.
func (e *NotLeaderError) Is(target error) bool {
	_, ok := target.(*NotLeaderError)
	return ok
}

// ErrNotLeader is the sentinel for "not leader" failures, primarily
// for callers using errors.Is. Use errors.As to extract LeaderAddr.
var ErrNotLeader error = &NotLeaderError{}

// _ asserts NotLeaderError satisfies the error interface at compile time.
var _ error = (*NotLeaderError)(nil)

// ErrOpNotRegistered is returned when Call references a name that is not
// in the ops registry.
var ErrOpNotRegistered = errors.New("shard: op not registered")

// errApplyAbandoned is the response every apply gets once this FSM has walked
// away from a classRetry block during Store.Close (see fsm.abandoned). It is not
// a durability signal to a client — the writes it answers were committed and
// replay on the next start — it is the FSM declining to move its state, and
// therefore raft's applied index, past a hole. It never escapes a live process:
// the only thing that sets the flag is shutdown.
var errApplyAbandoned = errors.New("shard: apply abandoned: the store is closing and an entry was left un-applied mid-block")
