// SPDX-License-Identifier: Apache-2.0

package cluster

import (
	"context"
	"errors"
	"fmt"
	"log/slog"
	"sort"
	"sync"
	"time"

	"github.com/rostamlabs/rostam/shard/pbisr"
)

// ISR GROW driver (primary-driven, meta-leader-committed). The INVERSE
// of the shrink driver (pb_shrink.go): it re-opens an under-replicated shard by
// catching a lagging placement owner up to the write frontier and re-adding it to
// the ISR — restoring a minISR>=2 shard's writability after a 4b failover reset
// the ISR to {newPrimary}, and un-doing a prior shrink once a member recovers.
//
// Like shrink it is PRIMARY-driven and runs ONLY under PBAutoFailover: the primary
// holds the freshest catch-up capability (its engine's ring + the group-frame
// delta), catches the survivor up via Engine.StartLearnerCatchup (the learner
// flip), then asks the meta leader to commit the widened ISR (OpSetShardISR,
// epoch-guarded + floor-checked at the FSM). Once it observes that committed widen
// in its own MetaFSM it reconciles Engine.GrowISR so proposeSequenced's intersect
// stops narrowing the re-added member back out after a prior same-epoch shrink.
//
// THE CRUX (see Engine.StartLearnerCatchup): the flip sets learners[M]=true
// STRICTLY BEFORE this driver submits OpSetShardISR(E, S∪M), so M is in the ship-
// set for every seq from the flip onward and the re-add can never open a seq gap.

// pbAppliedISR is one shard's last-reconciled ISR, tagged with the epoch it was
// decided for. Shared between the shrink and grow drivers via pbISRReconcile.
type pbAppliedISR struct {
	epoch uint64
	isr   []string
}

// pbISRReconcile is the per-node ISR-reconcile baseline SHARED by the shrink and
// grow drivers: shardID → the ISR last reconciled into that shard's engine at its
// epoch. Sharing it is load-bearing — a grow WIDENING and a shrink NARROWING at
// the same epoch would otherwise misread each other's committed change (a stale
// baseline could hide a re-narrow after a widen, or a re-widen after a narrow).
// Each driver updates the baseline ONLY in its own direction (shrink on a
// narrowing/first-sight, grow on a widening/first-sight), so neither steals the
// other's reconcile. Guarded by mu.
type pbISRReconcile struct {
	mu      sync.Mutex
	applied map[int]pbAppliedISR
}

func newPBISRReconcile() *pbISRReconcile {
	return &pbISRReconcile{applied: make(map[int]pbAppliedISR)}
}

func (r *pbISRReconcile) get(shardID int) (pbAppliedISR, bool) {
	r.mu.Lock()
	defer r.mu.Unlock()
	v, ok := r.applied[shardID]
	return v, ok
}

func (r *pbISRReconcile) set(shardID int, epoch uint64, isr []string) {
	r.mu.Lock()
	defer r.mu.Unlock()
	r.applied[shardID] = pbAppliedISR{epoch: epoch, isr: append([]string(nil), isr...)}
}

func (r *pbISRReconcile) forget(shardID int) {
	r.mu.Lock()
	defer r.mu.Unlock()
	delete(r.applied, shardID)
}

// pbCandidateHighWater verifies a failover promotion candidate's applied
// high-water — the DURABLE backstop that makes ISR membership NECESSARY BUT NOT
// SUFFICIENT for promotion (a grow can transiently leave a GAPPED member
// in the ISR; promoting it would lose acked writes). It returns (LastApplied,
// true) for a reachable candidate, or (0, false) when the candidate cannot be
// reached/verified — in which case the failover gate refuses to promote it.
//
// For the local node it reads the engine directly (no self-dial); for a remote
// candidate it round-trips the PB catch-up handshake (CatchupInfo returns the
// backup's applied high-water in AckMsg.Seq), resolving the node-ID to its PB
// dial address. Called only on the failover path (rare), never on the hot path.
// STAGE 4.3 — the OK FLAG IS NOW LOAD-BEARING, AND WAS PREVIOUSLY IGNORED.
// Before this stage nothing ever answered CatchupInfo{OK:false}, so both branches
// below reported (highWater, true) unconditionally. Snapshot transfer introduces a
// node that genuinely cannot state its log identity — a POISON-FENCED node, whose
// install began and did not provably complete, so its FSM may be half-wiped and
// its watermark means nothing. Promoting one destroys committed data outright.
//
// So both branches now honour it: the remote path checks info.OK, and the local
// path (which never round-trips at all) checks the engine's poison latch directly.
// "Unverifiable ⇒ not promotable" was already this function's contract; these two
// checks are what make a node actually able to declare itself unverifiable.
func (n *Node) pbCandidateHighWater(shardID int, candidate string) (uint64, bool) {
	if candidate == n.cfg.NodeID {
		if eng := n.pbEngines[shardID]; eng != nil {
			if eng.Poisoned() {
				return 0, false // half-wiped by an aborted snapshot install ⇒ NOT promotable
			}
			return eng.LastApplied(), true
		}
		return 0, false
	}
	if n.pbTransport == nil {
		return 0, false
	}
	tr := newPBResolvingTransport(n.pbTransport.For(shardID), n.pbAddrOf)
	ct, ok := tr.(pbisr.CatchupTransport)
	if !ok {
		return 0, false
	}
	epoch := uint64(0)
	if n.meta != nil {
		epoch = n.meta.FSM.ShardEpoch(shardID)
	}
	info, err := ct.CatchupRequest(candidate, epoch)
	if err != nil {
		return 0, false // unreachable / unverifiable ⇒ NOT promotable
	}
	if !info.OK {
		return 0, false // the candidate declares itself unverifiable (poison-fenced)
	}
	// AppliedSeq — NOT FrontierSeq. The failover gate ranks candidates by how much
	// of the COMMITTED tail they received AS A REPLICA; the frontier additionally
	// counts writes a candidate proposed itself while primary at an older epoch,
	// which are exactly the possibly-uncommitted ones this gate must not reward.
	// (Log matching split the two apart; before it, both were AckMsg.Seq.)
	return info.AppliedSeq, true
}

// decidePBGrow returns the placement owners that are NOT in the current ISR — the
// re-add candidates, in placement order (deterministic). The PRIMARY is always in
// the ISR (it holds the epoch), so it is never a candidate. Pure and
// order-preserving; the caller catches candidates up one at a time.
func decidePBGrow(placement, isr []string) []string {
	if len(placement) == 0 {
		return nil
	}
	inISR := make(map[string]struct{}, len(isr))
	for _, m := range isr {
		inISR[m] = struct{}{}
	}
	var out []string
	for _, owner := range placement {
		if _, ok := inISR[owner]; !ok {
			out = append(out, owner)
		}
	}
	return out
}

// growAbortReason classifies a grow catch-up error into a short, stable
// reason string for logging and the per-shard abort counters
// (cluster/repl_metrics.go's grow_aborts). The pbisr sentinels are distinct
// causes with very different operator implications, so each gets its OWN
// bucket: folding any of them into a neighbour or into "other" would hand an
// operator a count they cannot act on.
//
// SEVERITY, which is the whole point of the classification:
//
//   - RETRYABLE — the next driver tick genuinely may succeed. "snapshot_rejected"
//     (the target nacked an install, e.g. a stale epoch), "snapshot_exhausted"
//     (the bounded snapshot→catch-up loop did not converge because the primary
//     out-wrote the ring), "epoch_changed", "peer_ahead". A steadily CLIMBING
//     snapshot_exhausted is the one to watch: it means write rate is outrunning
//     state transfer, and no number of retries fixes that by itself.
//
//   - REPAIRABLE BY SNAPSHOT — "ring_evicted" (the target lagged past the
//     retained backlog) and "diverged" (the target is not a PREFIX of
//     this primary's history — provably reachable on a NON-quiesced failover, so
//     at minISR>=2 it is the ordinary steady state of a failover under live
//     traffic, not a rare edge case). Both USED TO BE PERMANENT and are NOT any
//     more: snapshot transfer exists, and pbisr routes both into it
//     automatically. Seeing either here means the repair path did not run or did
//     not help — check for a "no_snapshot_store"/"no_snapshot_transport" count on
//     the same shard first.
//
//   - NEEDS RE-SNAPSHOTTING — "unverifiable": the target is
//     POISON-FENCED. A snapshot install began on it and did not provably
//     complete, so its FSM may be half-wiped; it now refuses to serve and cannot
//     even state its log identity. It is neither behind nor forked, it is UNKNOWN.
//     Distinct from "diverged" because the node is REFUSING TRAFFIC, which an
//     operator must be able to see directly.
//
//   - CONFIGURATION — "no_snapshot_store" / "no_snapshot_transport": no repair
//     capability is wired, so ring_evicted/diverged/unverifiable on this shard
//     will never self-heal. Also "no_catchup_transport" / "no_group_transport",
//     which disable the grow outright. Retries are futile until an operator acts.
//
// Anything else (a generic transport/context error) falls back to "other".
func growAbortReason(err error) string {
	switch {
	case errors.Is(err, pbisr.ErrGrowRingEvicted):
		return "ring_evicted"
	case errors.Is(err, pbisr.ErrCatchupDiverged):
		return "diverged"
	case errors.Is(err, pbisr.ErrCatchupUnverifiable):
		return "unverifiable"
	case errors.Is(err, pbisr.ErrGrowSnapshotRejected):
		return "snapshot_rejected"
	case errors.Is(err, pbisr.ErrGrowSnapshotExhausted):
		return "snapshot_exhausted"
	case errors.Is(err, pbisr.ErrGrowNoSnapshotStore):
		return "no_snapshot_store"
	case errors.Is(err, pbisr.ErrGrowNoSnapshotTransport):
		return "no_snapshot_transport"
	case errors.Is(err, pbisr.ErrGrowEpochChanged):
		return "epoch_changed"
	case errors.Is(err, pbisr.ErrGrowPeerAhead):
		return "peer_ahead"
	case errors.Is(err, pbisr.ErrGrowNoCatchupTransport):
		return "no_catchup_transport"
	case errors.Is(err, pbisr.ErrGrowNoGroupTransport):
		return "no_group_transport"
	default:
		return "other"
	}
}

// pbGrowDriver is the per-node ISR-grow driver goroutine, mirroring
// pbShrinkDriver's lifecycle. Started ONLY when PBAutoFailover is on — when off,
// no driver runs, no OpSetShardISR grow is ever logged, and the static cluster's
// replicated state stays byte-identical.
type pbGrowDriver struct {
	node           *Node
	interval       time.Duration
	submitTimeout  time.Duration
	catchupTimeout time.Duration
	// snapshotTimeout bounds a catch-up on a SNAPSHOT-CAPABLE engine;
	// see pbGrowSnapshotTimeout for why a state transfer cannot share the delta's
	// deadline.
	snapshotTimeout time.Duration

	// baseline is SHARED with the shrink driver (see pbISRReconcile).
	baseline *pbISRReconcile

	// abortStats is the grow-abort observability: rate-limited logging
	// (log the transition, not every tick) plus cumulative per-reason counters
	// surfaced on /v1/replication (AbortCounts). See pbAbortTracker.
	abortStats *pbAbortTracker

	// lastSig/stableFor track, per shard, how many CONSECUTIVE ticks the committed
	// (epoch, ISR) has been unchanged in this driver's local FSM view. They gate the
	// ISR-MUTATING actions (grow re-add, abandon compensation) on the control state
	// having SETTLED, so the driver never overwrites a shard's ISR from a transiently
	// changing view — most importantly the bootstrap window, where the ISR
	// transitions {primary}→Placement and a premature grow would re-commit a NARROWER
	// ISR (dropping a seeded member → acked-loss). Grow-driver-goroutine-owned (only
	// tick touches them), so no lock. See tick.
	lastSig   map[int]string
	stableFor map[int]int

	mu      sync.Mutex
	started bool
	done    chan struct{}
	stopped chan struct{}
}

// growStableTicks is how many consecutive ticks a shard's committed (epoch, ISR)
// must be UNCHANGED before the grow driver will mutate that ISR. It filters out
// bootstrap/propagation transients (which settle in well under one tick) with NO
// meta traffic; a genuine under-replication is stable and grows after this many
// ticks. A quorum-fresh read barrier still gates the mutation itself, so this is a
// cheap pre-filter, not the correctness guarantee.
const growStableTicks = 2

func newPBGrowDriver(n *Node, interval, submitTimeout, catchupTimeout time.Duration, baseline *pbISRReconcile) *pbGrowDriver {
	return &pbGrowDriver{
		node:            n,
		interval:        interval,
		submitTimeout:   submitTimeout,
		catchupTimeout:  catchupTimeout,
		snapshotTimeout: pbGrowSnapshotTimeout,
		baseline:        baseline,
		lastSig:         make(map[int]string),
		stableFor:       make(map[int]int),
		abortStats:      newPBAbortTracker(),
	}
}

// AbortCounts returns a snapshot of shardID's cumulative grow-abort counts by
// reason (nil if none recorded), for the /v1/replication observability
// surface (cluster/repl_metrics.go). Nil-receiver-safe (PBAutoFailover off ⇒
// n.pbGrow is nil) and safe from any goroutine.
func (d *pbGrowDriver) AbortCounts(shardID int) map[string]uint64 {
	if d == nil {
		return nil
	}
	return d.abortStats.snapshot(shardID)
}

// isrSig is a deterministic signature of a shard's committed (epoch, ISR), used to
// detect whether the control state changed since the last tick.
func isrSig(epoch uint64, isr []string) string {
	s := append([]string(nil), isr...)
	sort.Strings(s)
	return fmt.Sprintf("%d:%v", epoch, s)
}

func (d *pbGrowDriver) start() {
	d.mu.Lock()
	defer d.mu.Unlock()
	if d.started {
		return
	}
	d.started = true
	d.done = make(chan struct{})
	d.stopped = make(chan struct{})
	go d.run()
}

func (d *pbGrowDriver) stop() {
	d.mu.Lock()
	if !d.started {
		d.mu.Unlock()
		return
	}
	done := d.done
	stopped := d.stopped
	d.started = false
	d.mu.Unlock()

	select {
	case <-done:
	default:
		close(done)
	}
	<-stopped
}

func (d *pbGrowDriver) run() {
	defer close(d.stopped)
	ticker := time.NewTicker(d.interval)
	defer ticker.Stop()
	for {
		select {
		case <-d.done:
			return
		case <-ticker.C:
			d.tick()
		}
	}
}

// tick runs one grow evaluation over every shard this node currently primaries.
// For each it does TWO independent things (mirroring the shrink driver):
//
//  1. RECONCILE (observe → apply): if the committed ISR in the local MetaFSM is a
//     strict WIDENING of the shared baseline (at the same epoch), call
//     Engine.GrowISR so proposeSequenced's intersect stops narrowing the re-added
//     member back out after a prior shrink at this epoch.
//
//  2. DETECT → CATCH-UP → RE-ADD: for each placement owner missing from the ISR
//     (decidePBGrow), catch it up via Engine.StartLearnerCatchup (the learner
//     flip), then commit the widened ISR (OpSetShardISR, floor-checked at the FSM).
//     One successful re-add per shard per tick; the reconcile in step (1) applies
//     the committed widen to the engine on a later tick.
//
// Both steps are best-effort and idempotent: GrowISR re-checks epoch+lease and is
// future-only; StartLearnerCatchup aborts cleanly on any epoch change; the
// OpSetShardISR is epoch-guarded and floor-checked at the FSM.
func (d *pbGrowDriver) tick() {
	n := d.node
	nodeID := n.cfg.NodeID
	if n.meta == nil {
		return
	}
	st := n.meta.FSM.State()
	fresh := false // whether st was refreshed via the quorum read barrier this tick

	for shardID, eng := range n.pbEngines {
		if eng == nil {
			continue
		}
		if st.ShardPrimary[shardID] != nodeID {
			// Not this node's primary: it has no business holding grow learners for
			// this shard, so tear any down (reclaims their sender goroutines), and
			// forget its stability tracking. The shrink driver owns forgetting the
			// shared baseline (both run the same condition).
			eng.TeardownLearners()
			delete(d.lastSig, shardID)
			delete(d.stableFor, shardID)
			d.abortStats.clear(shardID)
			continue
		}
		// Reclaim any learner sender an epoch advance orphaned (clearLearnersLocked
		// parked them because it could not take writeMu). Cheap; usually a no-op.
		eng.ReclaimOrphanLearners()

		epoch := st.ShardEpoch[shardID]
		isr := st.ShardISR[shardID]
		// Only drive grow for the epoch the engine actually holds as primary.
		if eng.Epoch() != epoch {
			continue
		}

		// (1) RECONCILE committed widenings into the live engine. READ-ONLY on the
		// committed state, so safe on the local (possibly not-yet-barriered) view: it
		// acts only on a MONOTONE observed widening, which a stale read shows as an
		// older state — never a spurious narrowing. Kept responsive (not stability-
		// gated) so a real widen reaches the engine promptly.
		base, seen := d.baseline.get(shardID)
		switch {
		case !seen || base.epoch != epoch:
			// First sight at this epoch: establish the baseline WITHOUT re-evaluating
			// (a fresh epoch starts with a clean pipeline). The shrink driver may
			// establish it first — set() is idempotent.
			d.baseline.set(shardID, epoch, isr)
		case isStrictSubset(base.isr, isr):
			// The committed ISR WIDENED at this epoch: reconcile the same-epoch override
			// so a re-added member is not intersected back out (future-only).
			eng.GrowISR(epoch, isr)
			d.baseline.set(shardID, epoch, isr)
		}

		// STABILITY GATE for the ISR-MUTATING actions below (abandon compensation +
		// grow re-add). Only act once the committed (epoch, ISR) has been UNCHANGED for
		// growStableTicks ticks, so a transiently-changing view never drives an ISR
		// OVERWRITE — a grow computed from a mid-transition view could re-commit
		// {primary, target} and DROP other members that belong in the ISR (acked-loss).
		// While unsettled the driver stays DORMANT (no meta traffic), so it also cannot
		// delay a bootstrapping primary's own FSM catch-up.
		//
		// The bootstrap seed is no longer such a transition: it commits (epoch, primary,
		// full ISR) as ONE entry (ApplySetShardSeed), so there is no window in which the
		// ISR is the singleton {primary}. The gate is still load-bearing for FAILOVER,
		// where OpSetShardEpoch resets the ISR to {newPrimary} deliberately and the
		// driver must not act until it has settled.
		sig := isrSig(epoch, isr)
		if d.lastSig[shardID] != sig {
			d.lastSig[shardID] = sig
			d.stableFor[shardID] = 0
			continue
		}
		d.stableFor[shardID]++
		if d.stableFor[shardID] < growStableTicks {
			continue
		}

		// Is there ISR-mutating work — an abandoned gapped voter to re-narrow, or an
		// under-replicated placement owner to re-add?
		var placement []string
		if shardID >= 0 && shardID < len(st.Placement) {
			placement = st.Placement[shardID]
		}
		abandoned := false
		for _, m := range isr {
			if m != nodeID && eng.LearnerAbandoned(m) {
				abandoned = true
				break
			}
		}
		if !abandoned && len(decidePBGrow(placement, isr)) == 0 {
			d.abortStats.clear(shardID) // healthy: no wedge to compensate, nothing to grow
			continue                    // nothing to mutate
		}

		// About to OVERWRITE the committed ISR. The stability gate above filters
		// transients, but it is measured on LOCAL reads; a stably-STALE local FSM (rare
		// — only under sustained meta lag) could still slip through, so CONFIRM a
		// quorum-FRESH view before committing. metaReadBarrier blocks until the local
		// FSM has applied up to the committed frontier. Done at most ONCE per tick, and
		// bounded SHORT (pbLeaseBarrierTimeout) so a meta election/partition is not
		// hammered. If freshness cannot be confirmed, skip (retry next tick).
		if !fresh {
			if err := n.metaReadBarrier(time.Now().Add(pbLeaseBarrierTimeout)); err != nil {
				return
			}
			st = n.meta.FSM.State()
			fresh = true
		}
		if st.ShardPrimary[shardID] != nodeID {
			continue
		}
		epoch = st.ShardEpoch[shardID]
		isr = st.ShardISR[shardID]
		if eng.Epoch() != epoch || isrSig(epoch, isr) != sig {
			// The fresh view differs from the settled local view we gated on — the
			// local read WAS stale. Reset stability and wait for the fresh view to
			// settle before any mutation.
			d.lastSig[shardID] = isrSig(epoch, isr)
			d.stableFor[shardID] = 0
			continue
		}
		placement = nil
		if shardID >= 0 && shardID < len(st.Placement) {
			placement = st.Placement[shardID]
		}

		// (2) ABANDON COMPENSATION (CRITICAL — a gapped voter must never persist). If
		// the grow ABANDONED a member (its learner channel filled — submitLearnerLocked)
		// AFTER that member was committed into the ISR, it is a GAPPED required voter:
		// under CommitFullISR every write requiring it gap-rejects forever (wedge);
		// under CommitPrimary the primary commits without it, so a later failover to it
		// would lose acked writes. Re-narrow it back out (idempotent; the shrink driver
		// reconciles the narrowing → ShrinkISR un-wedges the pipeline). The FSM floor
		// rejects a below-floor re-narrow; that at-floor corner is the DURABLE backstop's
		// job — the failover high-water gate never promotes a gapped member. The abandon
		// signal persists (idempotent re-narrow each tick) until the member leaves the
		// committed ISR; only a fresh grow flip clears it.
		compensated := false
		compFailed := false
		for _, m := range isr {
			if m == nodeID || !eng.LearnerAbandoned(m) {
				continue
			}
			narrowed := make([]string, 0, len(isr)-1)
			for _, x := range isr {
				if x != m {
					narrowed = append(narrowed, x)
				}
			}
			if err := n.submitSetShardISR(shardID, epoch, narrowed, d.submitTimeout); err != nil {
				compFailed = true
				if d.abortStats.record(shardID, m, "abandon_compensation_submit_failed") {
					slog.Warn("pb: grow abandon-compensation submit failed", "component", "cluster", "shard", shardID, "target", m, "epoch", epoch, "err", err)
				}
			}
			compensated = true // idempotent; retried next tick regardless of this submit's outcome
		}
		if compensated {
			if !compFailed {
				d.abortStats.clear(shardID)
			}
			continue // re-evaluate next tick on the reconciled ISR
		}

		// (3) DETECT under-replication and CATCH-UP → RE-ADD (one per tick, FRESH isr,
		// so `isr ∪ target` is a true WIDENING and never drops a real member).
		grew := false
		for _, target := range decidePBGrow(placement, isr) {
			// A snapshot-capable engine may need to ship the WHOLE FSM to
			// repair a ring-cold or diverged target — work proportional to shard size,
			// not to lag. The delta deadline would abort that before its first chunk
			// landed, so pick the bound from the capability rather than from optimism.
			timeout := d.catchupTimeout
			if eng.SnapshotCapable() {
				timeout = d.snapshotTimeout
			}
			ctx, cancel := context.WithTimeout(context.Background(), timeout)
			err := eng.StartLearnerCatchup(ctx, target)
			cancel()
			if err != nil {
				reason := growAbortReason(err)
				if d.abortStats.record(shardID, target, reason) {
					slog.Warn("pb: grow catch-up aborted", "component", "cluster", "shard", shardID, "target", target, "epoch", epoch, "reason", reason, "err", err)
				}
				continue // unreachable / racing / un-repairable: try the next, retry next tick
			}
			// PRE-SUBMIT GUARD: the target must still be a live, un-abandoned learner. A
			// write burst during catch-up could have filled its channel and abandoned it
			// between the flip and here; committing now would install a GAPPED voter.
			// (A post-submit abandon is caught by step (2)'s compensation next tick.)
			if !eng.IsLearner(target) || eng.LearnerAbandoned(target) {
				eng.AbortLearnerCatchup(target)
				if d.abortStats.record(shardID, target, "learner_abandoned_pre_submit") {
					slog.Warn("pb: grow re-add aborted, learner abandoned before submit", "component", "cluster", "shard", shardID, "target", target, "epoch", epoch)
				}
				continue
			}
			// The learner flipped (learners[target]=true) BEFORE this submit — the
			// re-add ordering crux. Commit the widened ISR (epoch-guarded, floor-checked).
			newISR := append(append([]string(nil), isr...), target)
			if err := n.submitSetShardISR(shardID, epoch, newISR, d.submitTimeout); err != nil {
				eng.AbortLearnerCatchup(target)
				if d.abortStats.record(shardID, target, "submit_failed") {
					slog.Warn("pb: grow re-add submit failed", "component", "cluster", "shard", shardID, "target", target, "epoch", epoch, "err", err)
				}
				continue
			}
			grew = true
			break // one grow committed this tick for this shard
		}
		if grew {
			d.abortStats.clear(shardID)
		}
	}
}
