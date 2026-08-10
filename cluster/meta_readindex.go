// SPDX-License-Identifier: Apache-2.0

package cluster

import (
	"context"
	"errors"
	"fmt"
	"sync/atomic"
	"time"

	hraft "github.com/hashicorp/raft"
)

// metaReadIndexForwardHook holds an optional test-only observer fired ONCE each
// time a follower actually FORWARDS a __meta_readindex__ op to the meta leader
// (i.e. issues the leader RTT). Stored atomically; nil in prod ⇒ zero overhead.
// Used to PROVE the single-node/no-peers short-circuit issues ZERO forwards and to
// observe the follower-forward path in tests.
var metaReadIndexForwardHook atomic.Pointer[func()]

// SetMetaReadIndexForwardHook installs (or clears with nil) the __meta_readindex__
// forward observer. Test-only; nil in production.
func SetMetaReadIndexForwardHook(fn func()) {
	if fn == nil {
		metaReadIndexForwardHook.Store(nil)
		return
	}
	metaReadIndexForwardHook.Store(&fn)
}

// Read-side meta readIndex barrier (Linearizable catalog reads — A1: true
// readIndex with local-serve on caught-up followers). A Linearizable read, BEFORE
// it resolves (P, gen)/aliases from its LOCAL meta-FSM, confirms its local meta-FSM
// has applied every catalog command committed as of a leader-verified read point.
//
// THE LANDMINE (designed around): hashicorp/raft calls fsm.Apply ONLY for
// LogCommand entries — NEVER for the election no-op or config entries. So
// Raft.CommitIndex() can point PAST the FSM's last-applied COMMAND index on an
// otherwise caught-up node. A follower waiting for FSM.AppliedIndex() >=
// rawCommitIndex would therefore wait FOREVER whenever the tail of the log is a
// no-op/config. Fix: the leader does NOT return its raw commit index; it returns
// its FSM COMMAND FRONTIER — the index of its last-applied command after ensuring
// it has drained everything <= its verified commit (via Barrier). That frontier is
// a real command index every follower applies in the same order and WILL reach.

// errMetaNotLeader is the internal not-leader sentinel returned by
// metaLeaderFrontier when this node is not the meta-Raft leader (VerifyLeader or a
// Barrier reported ErrNotLeader / ErrLeadershipLost). The caller re-resolves the
// leader and re-forwards. It is not exported: it never reaches a client — the
// follower path turns a persistent inability to reach a leader into the typed
// *ErrMetaLinearizableTimeout below.
var errMetaNotLeader = errors.New("cluster: meta readindex: not the meta-Raft leader")

// isMetaNotLeader reports whether err signals that this node lost (or never had)
// meta leadership for the readindex. It maps BOTH hraft.ErrNotLeader (a follower
// calling VerifyLeader, or a Barrier rejected because we are not leader) AND
// hraft.ErrLeadershipLost (leadership lost while the Barrier was in flight) to the
// not-leader signal — the strict-linearizable work learned ErrLeadershipLost must
// be handled too, else a leadership change mid-barrier surfaces as a hard error
// instead of a re-forward.
func isMetaNotLeader(err error) bool {
	return errors.Is(err, hraft.ErrNotLeader) || errors.Is(err, hraft.ErrLeadershipLost)
}

// ErrMetaLinearizableTimeout reports that a Linearizable read's meta readIndex
// barrier reached its deadline before this node's local meta-FSM caught up to the
// leader-verified command frontier (or before a reachable meta leader could be
// resolved at all). It is fail-loud: a Linearizable read NEVER serves a stale
// catalog on timeout — the read returns this error. The "cluster: meta
// linearizable" prefix is stable/greppable like the other cluster sentinels.
type ErrMetaLinearizableTimeout struct {
	WantFrontier uint64        // the leader frontier we were waiting for (0 if never resolved)
	Timeout      time.Duration // the deadline budget the barrier was given
}

func (e *ErrMetaLinearizableTimeout) Error() string {
	return fmt.Sprintf(
		"cluster: meta linearizable read timed out after %s waiting for local meta-FSM to reach leader frontier %d",
		e.Timeout, e.WantFrontier,
	)
}

// metaReadIndexFollowerTick is the poll interval at which a forwarding follower
// re-checks its local meta-FSM applied index against the leader frontier. A
// caught-up follower passes on the FIRST check (so the common case adds only the 1
// leader RTT, no poll delay); the tick only matters while genuinely catching up.
const metaReadIndexFollowerTick = 3 * time.Millisecond

// metaLeaderFrontier runs ON the meta-Raft leader (locally for a leader-coordinator
// AND inside handleMetaReadIndex). It returns the leader's FSM command frontier —
// the index of its last-applied catalog command, after confirming (VerifyLeader, a
// quorum heartbeat ⇒ no stale partitioned leader) it is the leader and draining its
// FSM to >= the verified commit index. The returned frontier is a real command
// index every follower deterministically reaches (NOT the raw CommitIndex; see the
// landmine note above).
//
// It calls hashicorp methods directly on n.meta.Raft — that field is the RAW
// *hraft.Raft (hashicorp API), NOT the raft/node.go wrapper. VerifyLeader() is
// LEADER-ONLY: a follower gets hraft.ErrNotLeader immediately.
func (n *Node) metaLeaderFrontier(deadline time.Time) (uint64, error) {
	if n.meta == nil {
		return 0, errNoMeta
	}
	// VerifyLeader: a quorum heartbeat confirming we are still the leader at a fresh
	// read point. A follower gets ErrNotLeader here immediately ⇒ surface the
	// not-leader sentinel so the caller re-resolves the leader and re-forwards.
	if err := n.meta.Raft.VerifyLeader().Error(); err != nil {
		if isMetaNotLeader(err) {
			return 0, errMetaNotLeader
		}
		return 0, err
	}
	// Capture the commit index AFTER the verify, so it reflects the verified leader
	// term's committed tail.
	ci := n.meta.Raft.CommitIndex()

	// Fast path (common, no log write, no Barrier): the FSM has DIRECTLY applied a
	// command at or beyond ci, so by in-order apply every command <= ci is in local
	// state. Return the drained command frontier.
	if fa := n.meta.FSM.AppliedIndex(); fa >= ci {
		return fa, nil
	}

	// Slow path: the FSM command frontier is behind ci (an idle no-op/config tail, or
	// a still-draining FSM). Barrier commits a no-op and resolves only after the FSM
	// has applied everything before it — every command <= ci, in order. Per
	// hashicorp/raft api.go the timeout passed to Barrier bounds only the ENQUEUE of
	// the no-op (an enqueue stall returns ErrEnqueueTimeout); the subsequent .Error()
	// then waits for the no-op to DRAIN the FSM and is NOT bounded by that timeout
	// (harmless here: the leader's own FSM apply is never gated). If leadership is lost
	// mid-barrier the future resolves with not-leader/leadership-lost (mapped) rather
	// than hanging.
	timeout := time.Until(deadline)
	if timeout <= 0 {
		return 0, &ErrMetaLinearizableTimeout{Timeout: 0}
	}
	if err := n.meta.Raft.Barrier(timeout).Error(); err != nil {
		if isMetaNotLeader(err) {
			return 0, errMetaNotLeader
		}
		return 0, err
	}
	// After the Barrier resolves, the FSM has applied every command <= ci. Return its
	// last-applied COMMAND index — a real command index reflecting all those commands.
	return n.meta.FSM.AppliedIndex(), nil
}

// metaReadBarrier is the per-Linearizable-read coordinator barrier: it confirms
// this node's LOCAL meta-FSM has applied every catalog command committed as of a
// leader-verified read point, so a subsequent LOCAL catalog read (alias/partition
// resolution) is linearized. It NEVER serves on timeout — it returns a typed error.
//
// Cost contract:
//   - single-node / no-meta-peers: ZERO work, ZERO forward (the local catalog is
//     always fresh — the local node IS all nodes).
//   - leader-coordinator: VerifyLeader + self catch-up (its own FSM is already AT the
//     returned frontier ⇒ serve, no wait).
//   - caught-up follower: 1 leader RTT (__meta_readindex__), then the FIRST local
//     check passes ⇒ no poll delay.
//
// It is wired into the Linearizable read path in a later task; here it is only the
// helper + op + tests. It is a true no-op off the Linearizable path (callers that
// are not Linearizable never invoke it).
// MetaReadBarrier is the exported entry point the embedded read path
// (rostam.resolveCollectionForRead) calls ONCE per Linearizable read, on the
// coordinator, BEFORE it resolves the catalog (alias + partition gen) from the
// LOCAL meta-FSM. It delegates to the unexported metaReadBarrier (the cross-package
// rostam layer cannot reach the unexported method). The cost contract below is the
// metaReadBarrier contract verbatim: single-node/no-meta-peers is a true local
// no-op; off the Linearizable read path it is never invoked at all.
func (n *Node) MetaReadBarrier(deadline time.Time) error {
	return n.metaReadBarrier(deadline)
}

// MetaIsLeader reports whether this node is currently the meta-Raft leader. A
// single-node / no-meta-peers node (n.meta == nil) reports false (there is no
// meta-Raft; the local catalog is authoritative and the barrier is a no-op). It is
// a cheap introspection accessor used by the read-path wiring tests to pick a
// follower coordinator (whose Linearizable read forwards) vs the leader (fast-path).
func (n *Node) MetaIsLeader() bool {
	if n.meta == nil {
		return false
	}
	return n.meta.Raft.State() == hraft.Leader
}

// metaContactStaleness bounds how long a meta FOLLOWER may go without contact
// from the current leader and still be considered connected to the quorum (for
// the PB lease gate). It must exceed the meta heartbeat interval by a
// comfortable margin (so a healthy follower is never falsely judged partitioned)
// and stay well under pbLeaseTTL (so a genuinely partitioned follower stops
// renewing, then lapses, with time to spare).
const metaContactStaleness = 2 * time.Second

// errMetaNoRecentContact is returned by confirmMetaView on a follower that has
// not heard from the leader within metaContactStaleness (a suspected partition).
var errMetaNoRecentContact = errors.New("cluster: no recent meta-leader contact")

// confirmMetaView reports (nil) that this node is currently connected to the
// meta-Raft quorum — the gate the PB leaseKeeper checks before renewing a
// primary lease, closing the OH1 double-primary window. It is a
// PARTITION detector, deliberately NOT the read-index-forward barrier
// (MetaReadBarrier): that barrier's follower arm forwards to the leader and, in
// a PB cluster where primaries are spread across all nodes, starves every
// follower-hosted primary (measured: follower forwards time out while the
// leader's own VerifyLeader succeeds). Instead:
//
//   - Meta LEADER: VerifyLeader() — a quorum heartbeat proving we still lead at
//     a fresh point (a partitioned ex-leader fails when it cannot reach quorum).
//   - Meta FOLLOWER: LastContact() within metaContactStaleness — a follower
//     still receiving the leader's AppendEntries/heartbeats is, by definition,
//     connected to the quorum and applying its log (so its local ShardPrimary
//     view is bounded-fresh). A partitioned follower's LastContact ages out.
//
// Single-node / no meta peers: always connected, nil (there is no quorum to
// lose). The deadline bounds the leader's VerifyLeader; the follower check is a
// timestamp read and returns immediately.
//
// OH1/failover note: this confirms CONNECTION, not that our FSM has applied the
// very latest committed epoch. With no failover (today) epochs never change, so
// a connected node's ShardPrimary view is correct. Once failover introduces
// epoch bumps, the promotion honor-rule (grant the new primary its lease only
// after the old lease has provably lapsed) must budget for up to
// metaContactStaleness + the follower's max apply-lag of stale renewal — see
// shard/pbisr/DESIGN.md.
func (n *Node) confirmMetaView(deadline time.Time) error {
	if n.meta == nil || len(n.cfg.Peers) <= 1 {
		return nil
	}
	if n.meta.Raft.State() == hraft.Leader {
		return n.meta.Raft.VerifyLeader().Error()
	}
	last := n.meta.Raft.LastContact()
	// Use the per-node configured staleness bound (PBMetaContactStalenessMs), falling
	// back to the metaContactStaleness default when unset (0). Config-driven so the
	// no-acked-loss failover gate test can shrink it alongside the other real clocks.
	stale := n.pbMetaContactStaleness
	if stale <= 0 {
		stale = metaContactStaleness
	}
	if last.IsZero() || time.Since(last) > stale {
		return errMetaNoRecentContact
	}
	return nil
}

func (n *Node) metaReadBarrier(deadline time.Time) error {
	// Single-node / no-meta-peers: the local catalog is always fresh. Zero cost.
	if n.meta == nil || len(n.cfg.Peers) <= 1 {
		return nil
	}

	// Leader-coordinator: confirm leadership + drain locally. The leader's own FSM is
	// already AT the returned frontier, so there is nothing to wait for — serve local.
	if n.meta.Raft.State() == hraft.Leader {
		_, err := n.metaLeaderFrontier(deadline)
		// errMetaNotLeader here means leadership was lost between the State() check and
		// the verify; fall through to the follower path to re-resolve + forward.
		if err != nil && errors.Is(err, errMetaNotLeader) {
			return n.metaReadBarrierFollower(deadline)
		}
		return err
	}

	// Follower: forward to the meta leader for the frontier, then wait until our local
	// FSM reaches it (or fail loud on the deadline).
	return n.metaReadBarrierFollower(deadline)
}

// metaReadBarrierFollower is the follower arm of metaReadBarrier: resolve the meta
// leader, forward __meta_readindex__ to learn its FSM command frontier, then poll
// the LOCAL meta-FSM applied index until it reaches that frontier (local-serve once
// caught up) or the deadline elapses. If the leader is unknown/unreachable it
// RETRIES resolving+forwarding within the deadline (an election may be in flight),
// then fails loud with *ErrMetaLinearizableTimeout. NEVER serves on timeout.
func (n *Node) metaReadBarrierFollower(deadline time.Time) error {
	timeout := time.Until(deadline)
	var frontier uint64
	haveFrontier := false

	ticker := time.NewTicker(metaReadIndexFollowerTick)
	defer ticker.Stop()

	for {
		if !haveFrontier {
			// Resolve the leader and forward for the frontier. On any failure (no
			// leader yet, unreachable peer, transient not-leader at the target) we do
			// NOT give up — an election may be mid-flight — we retry on the next tick
			// within the deadline.
			f, err := n.metaFrontier.do(deadline, n.fetchMetaLeaderFrontier)
			if err == nil {
				frontier = f
				haveFrontier = true
				// Fast common case: a caught-up follower already satisfies the frontier,
				// so check immediately without waiting a tick.
				if n.meta.FSM.AppliedIndex() >= frontier {
					return nil
				}
			}
		} else if n.meta.FSM.AppliedIndex() >= frontier {
			// Caught up: local catalog is now linearized to the leader frontier.
			return nil
		}

		<-ticker.C
		if time.Now().After(deadline) {
			return &ErrMetaLinearizableTimeout{WantFrontier: frontier, Timeout: timeout}
		}
	}
}

// fetchMetaLeaderFrontier resolves the current meta-Raft leader's server address
// and forwards the __meta_readindex__ op to it, returning the leader's FSM command
// frontier. It returns an error (which the follower loop retries within the
// deadline) when the leader is unknown, points at ourselves, is unreachable, or the
// op fails (e.g. the target just lost leadership and replied not-leader).
func (n *Node) fetchMetaLeaderFrontier(deadline time.Time) (uint64, error) {
	addr := n.metaLeaderServerAddr()
	if addr == "" || addr == n.serverAddrFor(n.cfg.NodeID) {
		// No leader resolved yet, or it resolves to us while State() != Leader (a
		// transient view during an election). Retry.
		return 0, errMetaNotLeader
	}
	cl, err := n.peerClient(addr)
	if err != nil {
		return 0, err
	}
	if h := metaReadIndexForwardHook.Load(); h != nil {
		(*h)()
	}
	// Thread our remaining budget so the leader's Barrier is bounded by
	// min(its own default, our deadline) — it must not block past the moment we give
	// up (N1). A non-positive budget (we are already at/past the deadline) is encoded
	// as 0, which the leader treats as "use your default"; but the timed-out branch in
	// metaReadBarrierFollower will have returned before we reach here in that case.
	budget := time.Until(deadline)
	if budget < 0 {
		budget = 0
	}
	args, err := gobEncode(metaReadIndexReq{Version: metaReadIndexVersion, BudgetNanos: int64(budget)})
	if err != nil {
		return 0, err
	}
	ctx, cancel := context.WithDeadline(context.Background(), deadline)
	defer cancel()
	raw, err := cl.Call(ctx, opMetaReadIndexName, args)
	if err != nil {
		return 0, err
	}
	var reply metaReadIndexReply
	if err := gobDecode(raw, &reply); err != nil {
		return 0, fmt.Errorf("cluster: __meta_readindex__ decode reply: %w", err)
	}
	if !reply.OK {
		// The target was not the leader (or could not drain): re-resolve + retry.
		return 0, errMetaNotLeader
	}
	return reply.Frontier, nil
}
