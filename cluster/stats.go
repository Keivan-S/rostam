// SPDX-License-Identifier: Apache-2.0

package cluster

import (
	"time"

	"github.com/rostamlabs/rostam/shard"
)

// Stats is a cluster-level snapshot aggregating cache and Raft stats
// across every shard.
type Stats struct {
	// NumShards is the configured shard count.
	NumShards int

	// PerShard holds the underlying shard.Stats indexed by shard ID.
	PerShard []shard.Stats

	// WASMGate reports the route gate's state and its refusal counter.
	WASMGate WASMGateStats

	// WASMBlobPush reports the pre-registration module-blob push.
	WASMBlobPush WASMBlobPushStats

	// WASMBlock reports shard groups currently PARKED waiting for module bytes
	// (see WASMBlockStats). It is a sibling of WASMGate rather than a field of it
	// because the two answer different questions about different mechanisms: the
	// gate refuses to PROPOSE into a group whose log lacks a registration, while a
	// block is a group that cannot APPLY an entry its log already carries.
	WASMBlock WASMBlockStats

	// WASMBlobRetire reports the blob retirement sweeper (see
	// WASMBlobRetireStats). Retention == 0 means retirement is OFF, which is the
	// default and the only configuration in which nothing can ever be removed.
	WASMBlobRetire WASMBlobRetireStats
}

// WASMBlobRetireStats makes WASM blob retirement observable — including, and
// especially, the fact that it is switched off.
//
// Retirement is the one WASM mechanism that DELETES something, and the failure
// it can cause (a replica that needed a retired version blocks until someone
// supplies the bytes) surfaces somewhere else entirely, in WASMBlockStats. So
// the two numbers an operator correlating those needs are how many files this
// node has removed and whether the sweeper is even running — neither of which is
// inferable from anything else.
type WASMBlobRetireStats struct {
	// Retention echoes Config.WASMBlobRetention. ZERO MEANS OFF: no sweeper
	// goroutine exists, and nothing has been or can be removed.
	Retention time.Duration

	// Sweeps counts retirement passes since process start. It is what separates
	// "off" from "on and finding nothing to do" — with Retention non-zero and
	// Sweeps flat, the sweeper is not running.
	Sweeps uint64

	// Retired counts blob FILES removed since process start. Correlate a rise
	// here with a later WASMBlock entry naming a fingerprint: that pairing is
	// what a retention window set too short looks like.
	Retired uint64

	// Pending is how many unreferenced blobs are currently waiting out their
	// window — the sweeper's backlog, and an upper bound on what the next few
	// sweeps can remove.
	Pending int64
}

// WASMBlobPushStats makes the pre-registration blob push observable.
//
// The push tolerates an unreachable member by design — refusing would let any
// node being restarted stop the cluster from registering a module — so a member
// that is persistently unreachable produces no error anywhere: every registration
// succeeds. The reply payload names it, but only to the caller of that one call;
// Skips is what makes the STANDING state visible from the server side, to an
// operator who was not holding the return value of any particular registration.
type WASMBlobPushStats struct {
	// Acks counts per-member push legs that were delivered and acked (the peer
	// verified the hash and compiled the module) since process start.
	Acks uint64

	// Skips counts per-member push legs skipped because the member could not be
	// reached. A steadily climbing Skips means some member is missing every
	// module's bytes and is relying entirely on fetching them on demand.
	Skips uint64
}

// WASMGateStats makes the WASM route gate observable.
//
// The gate deliberately trades a shard-wide halt for a client-visible retryable
// error (ErrWASMOpNotInThisGroup), which means a gate that never opens is a
// silent, permanent refusal of exactly the keys that route to the unproven
// group — indistinguishable from a client bug unless something reports it. That
// makes the error's visibility part of the design, not a nice-to-have.
type WASMGateStats struct {
	// Refusals counts Calls the gate has declined to propose since process
	// start, across all ops and groups.
	Refusals uint64

	// ProvenGroups maps each gated (replicated) op name to the sorted shard
	// groups whose Raft log this node knows carries its registration. An op
	// present here with a group MISSING is a wedged (op, group) pair — that is
	// the diagnostic. Freshly allocated per call.
	ProvenGroups map[string][]int
}
