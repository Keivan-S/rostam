// SPDX-License-Identifier: Apache-2.0

package cluster

import (
	"fmt"
	"sync"
	"time"

	"github.com/rostamlabs/rostam/shard/pbisr"
)

// pbResolvingTransport adapts a pbisr.Transport (which dials network
// addresses) to the ISR's view of the world, which names peers by node-ID.
// Replicate rewrites the node-ID peer to its PB dial address via addrOf
// before delegating to base; a node-ID absent from addrOf is a hard error —
// this transport NEVER dials a node-ID as if it were a hostname.
type pbResolvingTransport struct {
	base   pbisr.Transport
	addrOf map[string]string // node-ID -> PB dial address
}

var _ pbisr.Transport = (*pbResolvingTransport)(nil)

func (t *pbResolvingTransport) Replicate(peer string, msg pbisr.ReplicateMsg, done func(pbisr.AckMsg, error)) error {
	addr, ok := t.addrOf[peer]
	if !ok {
		// Unresolved node-ID: a submission error (done is NOT invoked). We never
		// dial a node-ID as a hostname.
		return fmt.Errorf("pbisr wiring: no PB address for node %q", peer)
	}
	return t.base.Replicate(addr, msg, done)
}

// TryReplicate forwards the optional inline-submit capability (lever 1) with
// the node-ID resolved to its PB dial address. A base without the capability
// (or an unresolvable node-ID) just declines — false is always a safe answer
// (the engine falls back to its ordered sender path).
func (t *pbResolvingTransport) TryReplicate(peer string, msg pbisr.ReplicateMsg, done func(pbisr.AckMsg, error)) bool {
	addr, ok := t.addrOf[peer]
	if !ok {
		return false
	}
	it, ok := t.base.(pbisr.InlineTransport)
	if !ok {
		return false
	}
	return it.TryReplicate(addr, msg, done)
}

// pbResolvingGroupTransport is a pbResolvingTransport whose base also has the
// optional pbisr.GroupTransport capability. It exists as a separate
// type so the capability is only advertised to the engine's type-assert when
// the base truly supports it — see newPBResolvingTransport.
type pbResolvingGroupTransport struct {
	pbResolvingTransport
	group pbisr.GroupTransport // == base, pre-asserted
}

var _ pbisr.GroupTransport = (*pbResolvingGroupTransport)(nil)

func (t *pbResolvingGroupTransport) ReplicateGroup(peer string, msgs []pbisr.ReplicateMsg, done func(pbisr.AckMsg, error)) error {
	addr, ok := t.addrOf[peer]
	if !ok {
		return fmt.Errorf("pbisr wiring: no PB address for node %q", peer)
	}
	return t.group.ReplicateGroup(addr, msgs, done)
}

var _ pbisr.CatchupTransport = (*pbResolvingGroupTransport)(nil)

// CatchupRequest forwards the grow handshake with the node-ID resolved to
// its PB dial address. Advertised only on the group-capable resolving transport
// (catch-up ships the delta over group frames, so it is meaningless without the
// group capability). A base lacking the catch-up capability, or an unresolvable
// node-ID, is a hard error the grow treats as an aborted attempt (retried).
func (t *pbResolvingGroupTransport) CatchupRequest(peer string, epoch uint64) (pbisr.CatchupInfoMsg, error) {
	addr, ok := t.addrOf[peer]
	if !ok {
		return pbisr.CatchupInfoMsg{}, fmt.Errorf("pbisr wiring: no PB address for node %q", peer)
	}
	ct, ok := t.base.(pbisr.CatchupTransport)
	if !ok {
		return pbisr.CatchupInfoMsg{}, fmt.Errorf("pbisr wiring: base transport has no catch-up capability")
	}
	return ct.CatchupRequest(addr, epoch)
}

var _ pbisr.SnapshotTransport = (*pbResolvingGroupTransport)(nil)

// SendSnapshotChunk forwards one snapshot-transfer chunk with the
// node-ID resolved to its PB dial address. Advertised on the group-capable
// resolving transport alongside CatchupRequest, because a snapshot is only ever
// the PREFIX-ESTABLISHING step in front of the group-frame delta — shipping one
// without the ability to then backfill and flip would be pointless.
func (t *pbResolvingGroupTransport) SendSnapshotChunk(peer string, c pbisr.SnapshotChunk) (pbisr.AckMsg, error) {
	addr, ok := t.addrOf[peer]
	if !ok {
		return pbisr.AckMsg{}, fmt.Errorf("pbisr wiring: no PB address for node %q", peer)
	}
	st, ok := t.base.(pbisr.SnapshotTransport)
	if !ok {
		return pbisr.AckMsg{}, fmt.Errorf("pbisr wiring: base transport has no snapshot capability")
	}
	return st.SendSnapshotChunk(addr, c)
}

// newPBResolvingTransport wraps base in node-ID address resolution, preserving
// base's optional GroupTransport capability when present.
func newPBResolvingTransport(base pbisr.Transport, addrOf map[string]string) pbisr.Transport {
	rt := pbResolvingTransport{base: base, addrOf: addrOf}
	if g, ok := base.(pbisr.GroupTransport); ok {
		return &pbResolvingGroupTransport{pbResolvingTransport: rt, group: g}
	}
	return &rt
}

// pbAddrMap builds a node-ID -> PBAddr lookup from the cluster's peer list,
// for use by pbResolvingTransport. Peers with an empty PBAddr (e.g. raft-mode
// peers) are skipped.
func pbAddrMap(peers []Peer) map[string]string {
	m := make(map[string]string, len(peers))
	for _, p := range peers {
		if p.PBAddr == "" {
			continue
		}
		m[p.NodeID] = p.PBAddr
	}
	return m
}

// PB primary-lease timing (static-cluster wiring). pbLeaseRefresh is how often
// the lease-keeper re-grants each owned primary's lease; pbLeaseTTL is the
// granted validity window. TTL is many refresh intervals so a delayed tick
// (CPU oversubscription) never lapses a still-primary lease, while a keeper that
// truly stops (node shutdown) still lets the lease expire within pbLeaseTTL.
const (
	pbLeaseRefresh = 200 * time.Millisecond
	pbLeaseTTL     = 5 * time.Second
	// pbLeaseBarrierTimeout bounds each tick's quorum-confirmed meta read (Plan
	// 4a). Well under pbLeaseTTL so an occasional slow round never itself lapses
	// a still-valid lease; comfortably above a healthy meta heartbeat so a
	// transiently busy leader is not treated as unreachable.
	pbLeaseBarrierTimeout = 1 * time.Second
	// pbRenewInterval is the default primary-liveness beacon commit interval (Plan
	// 4 automatic failover): every interval a primary commits an OpShardLeaseRenew
	// so the meta leader keeps seeing it alive. Several intervals fit inside
	// pbFailoverTimeout so a couple of dropped beacons never trip a spurious
	// failover. Overridable via Config.PBRenewIntervalMs.
	pbRenewInterval = 1 * time.Second
	// pbFailoverTickInterval is how often the failover ticker evaluates liveness.
	// Sub-second so a real primary loss is detected promptly once the timeout
	// elapses, while cheap (a lock-free FSM read + a map walk when leader).
	pbFailoverTickInterval = 500 * time.Millisecond
	// pbFailoverTimeout is the default silent-primary promotion threshold.
	// It MUST exceed pbLeaseTTL + metaContactStaleness (the HONOR RULE) so a
	// meta-partitioned old primary has provably self-fenced before the leader names
	// a replacement: 10s > 5s + 2s = 7s, a 3s margin. Overridable via
	// Config.PBFailoverTimeoutMs (the honor rule is re-asserted at construction).
	pbFailoverTimeout = 10 * time.Second
	// pbFailoverApplyTimeout bounds each promotion's OpSetShardEpoch commit from the
	// ticker (mirrors the other meta-apply deadlines).
	pbFailoverApplyTimeout = 5 * time.Second
	// pbBeaconSubmitTimeout bounds each beacon's commit/forward. Well under
	// pbLeaseTTL so a slow round never itself lapses a lease.
	pbBeaconSubmitTimeout = 1 * time.Second
	// pbShrinkTickInterval is how often the shrink driver evaluates each
	// owned primary's wedge signal. Sub-second so a wedged shard un-stalls promptly
	// once the threshold trips, while cheap (an FSM read + a per-shard map walk).
	pbShrinkTickInterval = 500 * time.Millisecond
	// pbShrinkSubmitTimeout bounds each shrink's OpSetShardISR commit/forward.
	pbShrinkSubmitTimeout = 2 * time.Second
	// pbShrinkFailureThreshold is the default per-backup consecutive-replication-
	// failure count at which the shrink driver requests a member's removal. Set
	// WELL above one RTT of transient blips so a momentary hiccup never shrinks a
	// healthy ISR; a genuinely dead/dropping backup crosses it within a few writes.
	// Overridable via Config.PBShrinkThreshold.
	pbShrinkFailureThreshold = 50
	// pbGrowTickInterval is how often the grow driver evaluates each owned
	// primary for an under-replicated ISR (a placement owner missing from the ISR).
	// A little slower than shrink — re-adding is a recovery action, not an
	// availability-critical one. Overridable via Config.PBGrowTickMs.
	pbGrowTickInterval = 1 * time.Second
	// pbGrowSubmitTimeout bounds each grow's OpSetShardISR(re-add) commit/forward.
	pbGrowSubmitTimeout = 2 * time.Second
	// pbGrowCatchupTimeout bounds ONE grow candidate's DELTA catch-up (handshake +
	// backfill + flip). A slow/unreachable candidate simply fails and is retried
	// next tick; kept generous since a deep lag legitimately takes many rounds.
	pbGrowCatchupTimeout = 10 * time.Second
	// pbGrowSnapshotTimeout bounds a catch-up on an engine that can fall back to
	// SNAPSHOT TRANSFER. It needs its own, far larger knob because the
	// two are not the same kind of operation at all: a delta is a handful of round
	// trips over a bounded ring, while a transfer serializes, ships and re-installs
	// the ENTIRE FSM — work proportional to shard size and disk, not to lag.
	// pbGrowCatchupTimeout would abort every real snapshot before its first chunk
	// landed, which would make the whole repair path unreachable in production while
	// passing every small-fixture test.
	//
	// COST, NAMED: the grow driver's tick is synchronous, so a transfer in progress
	// blocks that node's grow evaluation for every OTHER shard until it finishes.
	// That serialization is deliberate — two concurrent full-state transfers out of
	// one primary would contend for the same quiesce and the same disk, converging
	// slower than running them one at a time — but it is a real bound, and it is why
	// the value is minutes rather than unbounded.
	pbGrowSnapshotTimeout = 120 * time.Second
)

// pbSeedRetry is how often the background seeder re-attempts the control-plane
// seed until it lands, and pbSeedDeadline bounds the whole effort (a safety
// backstop — the seeder exits as soon as the seed is observed committed).
const (
	pbSeedRetry    = 200 * time.Millisecond
	pbSeedDeadline = 60 * time.Second
)

// seedPBControlPlane retries bootstrapPBShardControl until the control plane is
// seeded (every seed shard has a primary), this node is asked to stop, or the
// deadline elapses. bootstrapPBShardControl commits only when this node is the
// meta leader and is idempotent (epoch is monotonic), so running it on every
// bootstrapping node is safe: whichever node becomes leader lands the seed once,
// and every seeder then observes it and exits.
func (n *Node) seedPBControlPlane(p shardControlProposer, seeds []pbShardSeed, stop <-chan struct{}) {
	if len(seeds) == 0 {
		return
	}
	ticker := time.NewTicker(pbSeedRetry)
	defer ticker.Stop()
	deadline := time.After(pbSeedDeadline)
	for {
		if n.pbControlSeeded(seeds) {
			return
		}
		_ = bootstrapPBShardControl(p, seeds, 1, 5*time.Second) //nolint:errcheck,gosec // no-op on followers; retried until seeded
		if n.pbControlSeeded(seeds) {
			return
		}
		select {
		case <-stop:
			return
		case <-deadline:
			return
		case <-ticker.C:
		}
	}
}

// pbControlSeeded reports whether every seed shard now has a committed primary in
// the local (replicated) MetaFSM — the signal the seed has landed cluster-wide.
func (n *Node) pbControlSeeded(seeds []pbShardSeed) bool {
	for _, s := range seeds {
		if n.meta.FSM.ShardPrimary(s.ShardID) == "" {
			return false
		}
	}
	return true
}

// leaseKeeper renews pbisr.Engine primary leases, gated on a QUORUM-CONFIRMED
// meta view — the OH1 double-primary fix.
//
// Each refresh tick first calls barrier — a QUORUM-CONNECTION check
// (Node.confirmMetaView: VerifyLeader on the meta leader, recent LastContact on
// a follower). Only if it succeeds — proving this node is still connected to the
// meta quorum, so its local MetaFSM view is bounded-fresh — does the keeper read
// ShardPrimary/ShardEpoch and (re)grant the local-clock lease. A node
// partitioned from the meta quorum fails the check, renews nothing, and its
// engines self-fence when their local leases lapse (`pbisr.Propose` →
// ErrLeaseExpired). This closes the window the old static local-read keeper left
// open: a partitioned node can no longer keep renewing a lease the cluster's
// quorum no longer backs. (It is deliberately NOT the read-index barrier, whose
// follower-forward starves non-leader primaries — see confirmMetaView.)
//
// The lease itself remains a LOCAL construct (expiry on this process's
// monotonic clock, shared with its engines via WithClock) — the barrier does
// not change that; it changes only WHETHER a renewal is permitted. barrier may
// be nil (single-node / no meta peers), in which case there is no quorum to
// confirm and renewal is unconditional, exactly as before.
type leaseKeeper struct {
	fsm            *MetaFSM
	nodeID         string
	engines        map[int]*pbisr.Engine
	leaseTTL       time.Duration
	refresh        time.Duration
	now            func() int64          // monotonic-ns clock, shared with the engines (WithClock)
	barrier        func(time.Time) error // per-tick quorum-CONNECTION check; nil = single-node (skip)
	readBarrier    func(time.Time) error // per-EPOCH read-index barrier (Node.metaReadBarrier); nil = skip
	barrierTimeout time.Duration         // per-tick bound on barrier; << leaseTTL so a slow round never spuriously lapses

	// barriered records, per shard, the epoch whose FIRST lease grant was preceded
	// by a successful readBarrier. Keeper-goroutine-owned (only tick touches it, and
	// run is the sole caller), so no lock — same ownership rule as the grow driver's
	// lastSig/stableFor.
	barriered map[int]uint64

	mu      sync.Mutex
	started bool
	done    chan struct{}
	stopped chan struct{}
}

// newLeaseKeeper constructs a leaseKeeper. now must return monotonic
// nanoseconds from the SAME clock source given to every engine in engines
// (via pbisr.WithClock) so the lease math (expiry comparisons) lines up.
// barrier is the quorum-confirmed meta read gating each renewal; pass
// nil to disable the gate (single-node / no meta peers). barrierTimeout bounds
// each barrier call and MUST be well under leaseTTL so an occasional slow
// quorum round does not itself cause a lease to lapse. readBarrier is the
// read-index barrier run ONCE PER (shard, epoch) before that epoch's first lease
// grant (see tick); pass nil to disable it.
func newLeaseKeeper(fsm *MetaFSM, nodeID string, engines map[int]*pbisr.Engine, leaseTTL, refresh time.Duration, now func() int64, barrier, readBarrier func(time.Time) error, barrierTimeout time.Duration) *leaseKeeper {
	return &leaseKeeper{
		fsm:            fsm,
		nodeID:         nodeID,
		engines:        engines,
		leaseTTL:       leaseTTL,
		refresh:        refresh,
		now:            now,
		barrier:        barrier,
		readBarrier:    readBarrier,
		barrierTimeout: barrierTimeout,
		barriered:      make(map[int]uint64),
	}
}

// start spawns the keeper's refresh goroutine. It is a no-op if already
// started.
func (k *leaseKeeper) start() {
	k.mu.Lock()
	defer k.mu.Unlock()
	if k.started {
		return
	}
	k.started = true
	k.done = make(chan struct{})
	k.stopped = make(chan struct{})
	go k.run()
}

// stop signals the refresh goroutine to exit and blocks until it has. It is
// safe to call stop without a prior start (no-op), and safe to call more
// than once.
func (k *leaseKeeper) stop() {
	k.mu.Lock()
	if !k.started {
		k.mu.Unlock()
		return
	}
	done := k.done
	stopped := k.stopped
	k.started = false
	k.mu.Unlock()

	select {
	case <-done:
		// already closed by a previous stop call
	default:
		close(done)
	}
	<-stopped
}

func (k *leaseKeeper) run() {
	defer close(k.stopped)
	ticker := time.NewTicker(k.refresh)
	defer ticker.Stop()
	for {
		select {
		case <-k.done:
			return
		case <-ticker.C:
			k.tick()
		}
	}
}

// tick renews leases for every shard this node still primaries — but ONLY after a
// quorum-confirmed meta read. If barrier fails (this node cannot reach
// the meta quorum, or is no longer the leader and cannot forward), NO lease is
// renewed this tick; the engines' leases lapse and they self-fence.
//
// TWO DIFFERENT BARRIERS, for two different jobs:
//
//   - barrier (Node.confirmMetaView) runs EVERY tick. It is a CONNECTION check —
//     VerifyLeader on the meta leader, recent LastContact on a follower. It proves
//     we are still attached to the quorum. It establishes NO frontier and therefore
//     says NOTHING about how fresh our local FSM is.
//
//   - readBarrier (Node.metaReadBarrier) runs ONCE PER (shard, epoch), before that
//     epoch's FIRST lease grant. It resolves the leader's commit frontier and blocks
//     until our LOCAL meta-FSM has applied up to it.
//
// Why the per-epoch read barrier is here (the correctness argument). A lease is a
// licence to ACK, and proposeSequenced builds each write's peer set from an
// UNBARRIERED read of this same local MetaFSM. If that FSM is behind, the primary
// reads an ISR NARROWER than the committed one, passes the MinISR floor, and
// commits on itself alone — writes it acks are then lost when it dies and a
// survivor that never saw them is (correctly) promoted. The read barrier makes the
// local FSM provably current at the moment the epoch's first grant is issued; and
// because the FSM's applied index only ADVANCES, every ctrl.ISR read the engine
// makes afterwards is at least as fresh. That is why the barriered ISR does not
// need to be copied into the engine: the engine reads the very FSM the barrier
// advanced. Cost is one barrier per EPOCH TRANSITION, not per write and not per
// tick — an already-barriered epoch renews with no meta round-trip at all.
//
// The previous comment here claimed the post-barrier view "is at least as fresh as
// the barrier's confirmed frontier" because "the MetaFSM is monotonic". That does
// not follow. Monotone means the view never goes BACKWARDS; it does not make it
// CURRENT. And confirmMetaView never produced a frontier to be as fresh as in the
// first place — it is a liveness check, not a read-index barrier.
//
// If the read barrier fails for a NEW epoch, that epoch is simply not granted this
// tick (retried next tick). Availability cost is bounded by the refresh interval;
// correctness is not traded for it. Epochs already barriered keep renewing, so a
// transient meta hiccup cannot fence a healthy steady-state primary.
func (k *leaseKeeper) tick() {
	if k.barrier != nil {
		if err := k.barrier(time.Now().Add(k.barrierTimeout)); err != nil {
			return // unconfirmed leadership view — renew nothing, let leases lapse
		}
	}
	nowNs := k.now()
	// At most ONE read barrier per tick even when several shards transition epoch
	// together (bootstrap seeds every shard at once): the frontier it establishes is
	// FSM-wide, not per shard.
	readBarriered := false
	for shard, eng := range k.engines {
		if k.fsm.ShardPrimary(shard) != k.nodeID {
			continue
		}
		epoch := k.fsm.ShardEpoch(shard)
		if k.barriered[shard] != epoch {
			if k.readBarrier != nil && !readBarriered {
				if err := k.readBarrier(time.Now().Add(k.barrierTimeout)); err != nil {
					continue // freshness unconfirmed — grant nothing new for this epoch
				}
				readBarriered = true
			}
			// Re-read AFTER the barrier: the pre-barrier read that got us here may
			// have been the stale one.
			if k.fsm.ShardPrimary(shard) != k.nodeID {
				continue
			}
			epoch = k.fsm.ShardEpoch(shard)
			k.barriered[shard] = epoch
		}
		expiry := nowNs + int64(k.leaseTTL)
		if epoch > eng.Epoch() {
			// This node is the shard's primary for an epoch its engine has NOT yet
			// adopted as primary — a FRESH PROMOTION: either bootstrap
			// (epoch 0 → 1) or a failover where this backup was just named primary.
			// Promote (not AdoptEpoch): continue seq assignment from the applied
			// high-water (lastSeq = lastApplied). Plain AdoptEpoch would leave
			// lastSeq at 0 and the first Propose would re-use already-applied seqs,
			// gap-rejecting every future write. Promote is a safe superset for a
			// fresh empty primary (lastApplied 0 ⇒ same result as AdoptEpoch).
			eng.Promote(epoch, expiry)
		} else {
			// Steady-state renewal for an epoch we already hold as primary.
			eng.GrantLease(epoch, expiry)
		}
	}
}
