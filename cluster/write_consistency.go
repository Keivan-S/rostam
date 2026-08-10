// SPDX-License-Identifier: Apache-2.0

package cluster

import (
	"fmt"
	"time"

	"github.com/rostamlabs/rostam/ops"
)

// Post-commit write-consistency barrier.
//
// Each data shard is its own hashicorp/raft group; raft.Apply already resolves
// only after a MAJORITY of voters have committed the entry — there is no API to
// ask for fewer acks. A write_consistency_factor (WCF) is therefore a post-commit
// barrier that waits for MORE than majority (up to all RF replicas) to have
// APPLIED the entry, by polling each owner's per-shard applied index. Values
// <= majority are the free default (no barrier, no round-trips); only values
// > majority engage the poll loop.
//
// This mirrors the migration precedent migrate.go waitCaughtUp, which polls a
// specific peer's adminStatus.AppliedIndex against a target log index via the
// __rb_status__ admin op. The barrier reuses that same machinery on the live
// write path.

// writeConsistencyTimeout is the default deadline for BarrierForShard when a
// caller does not supply its own. Transports may override it.
const writeConsistencyTimeout = 5 * time.Second

// WriteConsistencyTimeout is the exported default deadline the root-package
// callers (the embedded write methods and the __wc__ envelope handler)
// pass into BarrierForShard. It equals the internal writeConsistencyTimeout so
// there is a single source of truth; transports may pass their own.
const WriteConsistencyTimeout = writeConsistencyTimeout

// writeConsistencyTick is the poll interval, matching waitCaughtUp's 20ms.
const writeConsistencyTick = 20 * time.Millisecond

// ErrWriteConsistency reports that a write committed at Raft quorum (and is
// therefore durable) but the requested write-consistency factor was not met
// within the timeout: fewer than Requested replicas had applied the entry. The
// message carries a stable, greppable "cluster: write " prefix so the transport
// layer can prefix-match it, like the alias sentinels.
type ErrWriteConsistency struct {
	Requested      int    // the effective (clamped) factor that was sought
	Applied        int    // how many replicas had applied by the deadline
	CommittedIndex uint64 // the catch-up target (the shard leader's applied index)
}

func (e *ErrWriteConsistency) Error() string {
	return fmt.Sprintf(
		"cluster: write committed at quorum but consistency factor %d not met (%d replicas applied at index %d)",
		e.Requested, e.Applied, e.CommittedIndex,
	)
}

// waitWriteConsistency polls the owners of shardIdx until at least wcf of them
// have applied the entry at the catch-up target, or the timeout elapses. It
// mirrors migrate.go waitCaughtUp but for the live write path: the LOCAL node's
// per-shard applied index is read directly; each remote owner is polled via the
// __rb_status__ admin op through a cached peer-forwarding client. Transient
// per-peer errors degrade to a zero-value status (treated as not-yet-applied)
// exactly like remoteNode.status, so the loop simply retries.
//
// The catch-up target is the shard LEADER's applied index, captured ONCE the
// first time a leader is observed (the leader applied this write before Apply
// returned, so its applied index is >= this write's committed index — a safe,
// always-correct target). Crucially, the barrier FAILS CLOSED: if no leader can
// be located within the window it never falls back to a follower's (possibly
// lagging) index — a lagging follower could be below this write's index and
// falsely satisfy the factor. With no established target the count stays 0 and
// the call times out with a typed *ErrWriteConsistency (the write is still
// durable at majority).
//
// Returns the count of replicas that had applied at the moment it returns. On
// success applied >= wcf and err is nil; on deadline it returns the last count
// with a typed *ErrWriteConsistency. A single deadline bounds the whole wait
// (leader discovery + follower catch-up share one budget).
func (n *Node) waitWriteConsistency(shardIdx int, owners []string, wcf int, timeout time.Duration) (applied int, err error) {
	deadline := time.Now().Add(timeout)
	var target uint64
	var haveTarget bool
	for {
		if !haveTarget {
			if idx, ok := n.leaderAppliedIndex(shardIdx, owners); ok {
				target, haveTarget = idx, true
			}
		}
		if haveTarget {
			applied = n.appliedReplicas(shardIdx, owners, target)
			if applied >= wcf {
				return applied, nil
			}
		}
		if !time.Now().Before(deadline) {
			return applied, &ErrWriteConsistency{Requested: wcf, Applied: applied, CommittedIndex: target}
		}
		time.Sleep(writeConsistencyTick)
	}
}

// appliedReplicas counts how many of owners have applied shardIdx's log up to at
// least target. The local node (if it owns the shard) is read directly with no
// round-trip; each remote owner is queried via __rb_status__ (a cheap local raft
// index read on that peer). A peer that errors or is not yet caught up simply
// does not count this round — the caller polls again.
func (n *Node) appliedReplicas(shardIdx int, owners []string, target uint64) int {
	applied := 0
	for _, owner := range owners {
		var st adminStatus
		if owner == n.cfg.NodeID {
			st = n.localStatus(shardIdx)
		} else {
			st = n.remoteOwnerStatus(owner, shardIdx)
		}
		if st.AppliedIndex >= target {
			applied++
		}
	}
	return applied
}

// remoteOwnerStatus fetches an owner's per-shard adminStatus over the network,
// reusing the cached peer-forwarding client and the remoteNode helper (which
// degrades to a zero-value adminStatus on any transport error). Returns the
// zero value if the owner has no resolvable server address or no client.
func (n *Node) remoteOwnerStatus(ownerID string, shardIdx int) adminStatus {
	addr := n.serverAddrFor(ownerID)
	if addr == "" {
		return adminStatus{}
	}
	cl, err := n.peerClient(addr)
	if err != nil {
		return adminStatus{}
	}
	r := &remoteNode{nodeID: ownerID, cl: cl}
	return r.status(shardIdx)
}

// ShardIndexForName maps a physical collection name to its shard index using the
// exact same routing the write path uses: shardOf over the CANONICAL name bytes
// (bare "docs" -> "default/docs"), so the barrier targets the SAME shard the
// write landed on. ops.CanonicalName is the same canonicalization vectorRouteKey
// applies inside the key extractors (vector_routing.go vectorKeyColAt1/At2), and
// shardOf is the cluster router (router.go:11) — both reused verbatim.
func (n *Node) ShardIndexForName(name string) int {
	return shardOf([]byte(ops.CanonicalName(name)), n.cfg.NumShards)
}

// BarrierForShard is the public post-commit write-consistency barrier for a
// single shard's Raft group. It is invoked by callers that have wcf/wait as
// explicit values (the embedded path and the __wc__ envelope handler),
// AFTER the inner write has committed at majority.
//
// Raft-floor honesty: a write already reaches a MAJORITY of voters before Apply
// returns, so any effective factor <= majority is satisfied for free — the
// barrier short-circuits with NO round-trips. Only eff > majority polls.
//
//   - RF      = len(ownersFor(idx))      (replicas of the target shard)
//   - maj     = RF/2 + 1                 (the Raft commit floor)
//   - eff     = clamp(int(wcf), 1, RF)   (a request > RF clamps to RF; <1 -> 1)
//
// If !wait, or eff <= maj, or RF <= 1, the barrier is a no-op and returns nil
// immediately (no peer round-trips). Otherwise it captures the catch-up target —
// the target shard LEADER's applied index — and polls until eff owners have
// applied at least that index, or the timeout elapses (returning a typed
// *ErrWriteConsistency while the write remains durable at majority).
//
// Target choice: we use the shard leader's current RaftAppliedIndex (>= this
// write's committed index, since the leader applied it before Apply returned)
// rather than threading the exact per-write index up through Call's ([]byte,
// error) return. This is slightly over-strict — it may also wait for a few
// concurrent entries the leader applied — but is always correct: it guarantees
// THIS write is applied on >= eff replicas. If the leader is unreachable for the
// whole window the barrier FAILS CLOSED (times out unmet) rather than trusting a
// possibly-lagging follower's index — see waitWriteConsistency.
func (n *Node) BarrierForShard(shardIdx int, wcf uint8, wait bool, timeout time.Duration) error {
	owners := n.ownersFor(shardIdx)
	RF := len(owners)
	maj := RF/2 + 1
	eff := int(wcf)
	if eff < 1 {
		eff = 1
	}
	if eff > RF {
		eff = RF
	}
	// Fast path: nothing to wait for. wait=false is a latency knob (return at
	// majority); eff <= maj is already satisfied by the Raft commit floor; RF<=1
	// (single replica / single-node embedded) can never exceed majority.
	if !wait || eff <= maj || RF <= 1 {
		return nil
	}
	_, err := n.waitWriteConsistency(shardIdx, owners, eff, timeout)
	return err
}

// leaderAppliedIndex does ONE pass over owners and returns the applied index of
// the one reporting IsLeader (read locally with no round-trip if that is this
// node, else via __rb_status__). Returns ok=false if no owner reports leadership
// this pass — the caller retries within its deadline and, finding no leader,
// fails closed rather than substituting a follower's index. By Raft's leader
// completeness a newly elected leader necessarily holds every committed entry,
// so once ANY leader is observed its applied index is a safe catch-up target.
func (n *Node) leaderAppliedIndex(shardIdx int, owners []string) (uint64, bool) {
	for _, owner := range owners {
		var st adminStatus
		if owner == n.cfg.NodeID {
			st = n.localStatus(shardIdx)
		} else {
			st = n.remoteOwnerStatus(owner, shardIdx)
		}
		if st.IsLeader {
			return st.AppliedIndex, true
		}
	}
	return 0, false
}
