// SPDX-License-Identifier: Apache-2.0

package pbisr

import (
	"context"
	"errors"
)

// ============================================================================
// ISR GROW (learner-before-voter with an atomic writeMu flip).
//
// GROW is the INVERSE of SHRINK (pb_pipeline.go): it re-opens a shard to writes
// by catching a lagging survivor M to the write frontier and re-adding it to the
// ISR, restoring a minISR>=2 shard's writability after a 4b failover reset the
// ISR to {newPrimary}, and un-doing a prior shrink once a member recovers.
//
// THE HAZARD grow must avoid (silent divergence): writes W(h+1..) proposed while
// M ∉ ISR are NOT shipped to M; a naive re-add makes a later write require M, and
// M — never having received W(h+1..) — gap-REJECTS it (receiveLocked's
// PrevSeq != lastApplied check). Senders are fire-once with no resend, so that is
// a permanent shard wedge. The learner-before-voter protocol below closes it.
//
// THE LOAD-BEARING ASYMMETRY (mirror of shrink's): grow NEVER widens an already-
// registered in-flight record's required set. A learner is shipped every write
// but is added to NO in-flight record's pending set; it only becomes required for
// FUTURE writes, once its re-add commits. Widening a live record's required set
// would demand an ack for a write the member may never have received — UNSOUND.
// ============================================================================

// Grow sentinel errors (all abort the grow cleanly; the shard is unaffected
// because a learner is never a required/committing member).
var (
	// ErrGrowNoCatchupTransport / ErrGrowNoGroupTransport — the injected transport
	// lacks the catch-up handshake or the group-frame delta capability. A grow
	// cannot run without both; the caller falls back to leaving the shard as-is.
	ErrGrowNoCatchupTransport = errors.New("pbisr: transport does not support catch-up handshake")
	ErrGrowNoGroupTransport   = errors.New("pbisr: transport does not support group-frame delta")

	// ErrGrowPeerAhead — the catch-up target reports an epoch strictly HIGHER than
	// the epoch we are growing at, so it has already adopted a newer leadership
	// generation. Our grow is stale; abort (its OpSetShardISR would no-op anyway).
	ErrGrowPeerAhead = errors.New("pbisr: catch-up target is ahead of the growing epoch")

	// ErrGrowEpochChanged — the engine's epoch/lease moved off the growing epoch E
	// mid-catch-up (a failover, a lost lease). The grow is void; abort and tear
	// down. Any late OpSetShardISR(E,...) is epoch-guarded to a no-op.
	ErrGrowEpochChanged = errors.New("pbisr: epoch/lease changed during catch-up")

	// ErrGrowRingEvicted — the delta the target needs has fallen out of the primary's
	// catch-up ring (the target lagged past the retained backlog). A ring-cold delta
	// needs a full snapshot transfer, which SHIPS: like ErrCatchupDiverged
	// this is REPAIRABLE, and reaches the caller only when no snapshot store is wired
	// or the bounded snapshot→catch-up loop failed to converge.
	ErrGrowRingEvicted = errors.New("pbisr: catch-up delta evicted from ring (snapshot required)")

	// ErrCatchupDiverged — LOG MATCHING failure. The target does not
	// hold a PREFIX of our history: either its applied frontier is past our own
	// write frontier (it holds writes we never made), or the write it holds at a
	// shared seq was assigned under a DIFFERENT epoch than ours (the seq counter is
	// reset by Promote, so the same seq under two epochs is two different writes).
	//
	// A grow by APPEND cannot repair this: appending our history onto a fork
	// silently interleaves two contradictory logs (the exact defect log matching
	// closes). Repair needs a full state transfer — which now SHIPS, so
	// this sentinel is REPAIRABLE, not terminal: StartLearnerCatchup routes it to
	// snapshotCatchup and retries the delta path from the newly established prefix.
	// It only reaches the caller (and the abort counters) when no snapshot store is
	// wired or the bounded snapshot→catch-up loop failed to converge.
	//
	// It is raised BOTH up front (checkCatchupDivergenceLocked, from the handshake's
	// reported frontier) and reactively, when a backfill round advances the target's
	// high-water by NOTHING: with the chain check in receiveLocked, a rejected first
	// frame IS a log-matching failure — the only other cause, an out-of-order or
	// duplicate delivery, is impossible on the single synchronous backfill stream.
	//
	// Exported for observability so cluster/'s grow-abort classifier can
	// distinguish it from the other grow sentinels without string-matching; it
	// SUPERSEDES what used to be the separate, unexported errGrowNoProgress — Stage
	// 4.1 folded "no progress" into the more precise log-matching diagnosis.
	ErrCatchupDiverged = errors.New("pbisr: catch-up target has diverged (snapshot required)")

	// ErrCatchupUnverifiable — the target answered the handshake NOT-OK,
	// i.e. it cannot state a log identity at all. Today that is exactly a
	// POISON-FENCED node: one whose snapshot install began and did not provably
	// complete, whose FSM may be half-wiped and whose watermarks are therefore
	// deliberately zeroed. It is neither "behind" nor "forked" — it is UNKNOWN, and
	// the only repair is a full state transfer.
	//
	// Exported by the same convention so cluster/'s grow-abort classifier can give a
	// half-installed node its OWN bucket: unlike the other sentinels it names a node
	// that is currently REFUSING TO SERVE, which an operator must be able to tell
	// apart from an ordinary lagging or forked one.
	ErrCatchupUnverifiable = errors.New("pbisr: catch-up target cannot state its log identity (snapshot required)")
)

// growBackfillBatch caps how many retained writes one far-backfill round ships in
// a single group frame. Bounded so a round's ringDeltaLocked copy and the group
// frame stay modest; the loop simply runs more rounds for a deeper lag.
const growBackfillBatch = 256

// CatchupTransport is an OPTIONAL Transport capability for ISR grow: the
// learner-catch-up HANDSHAKE. CatchupRequest asks peer for its log identity
// (Engine.CatchupInfo on the remote side) so the growing primary can decide
// whether the target is a PREFIX of its history and, if so, compute the ring delta
// to backfill. epoch is the primary's growing epoch E (informational — the primary
// itself fences on the returned CatchupInfoMsg.Epoch).
type CatchupTransport interface {
	CatchupRequest(peer string, epoch uint64) (CatchupInfoMsg, error)
}

// CatchupInfo answers a learner-catch-up handshake with THIS engine's log
// identity. Serialized by e.mu, like the other read paths.
//
// It reports BOTH watermarks because they answer different questions and log matching
// stopped conflating them:
//
//   - AppliedSeq (lastApplied) is the backup-role high-water the FAILOVER gate
//     reads to rank promotion candidates. Unchanged in meaning.
//   - FrontierSeq/FrontierEpoch are the LOG identity the GROW path resumes from.
//     A node that proposed as primary holds those writes even though lastApplied
//     never counted them; resuming from lastApplied would ship it a delta starting
//     below its own frontier, which the receiver now (correctly) rejects.
//
// A POISONED node — one whose snapshot install began and did not
// provably complete — answers NOT-OK with ZEROED watermarks. Both halves matter.
// The not-OK is what makes cluster's pbCandidateHighWater treat it as
// UNVERIFIABLE so the failover gate never promotes a half-wiped node; the zeroing
// is so that no caller which forgets to check OK can read a watermark this node
// cannot back with FSM state. A poisoned node is not "behind", it is UNKNOWN, and
// the only honest answer to "what do you hold" is "I cannot say".
func (e *Engine) CatchupInfo() CatchupInfoMsg {
	e.mu.Lock()
	defer e.mu.Unlock()
	if e.poisoned {
		return CatchupInfoMsg{Epoch: e.epoch, OK: false}
	}
	fs, fe := e.appliedFrontierLocked()
	return CatchupInfoMsg{
		Epoch:         e.epoch,
		AppliedSeq:    e.lastApplied,
		FrontierSeq:   fs,
		FrontierEpoch: fe,
		OK:            true,
	}
}

// checkCatchupDivergenceLocked decides, from the target's reported log identity
// (k, kEpoch), whether the target holds a PREFIX of this primary's history — the
// precondition for a catch-up by log append. It returns ErrCatchupDiverged when it
// provably does not. Caller holds e.mu.
//
// Three cases, in the order they are cheap to decide:
//
//	k > frontier   — the target holds a write at a seq we never assigned. It
//	                 forked ahead of us (the ex-primary case: it proposed alone
//	                 while we were promoted from a lower high-water). Appending
//	                 our log onto it would interleave two histories.
//	k == frontier  — same position; agree only if the write there is the SAME
//	                 write, i.e. same epoch.
//	k < frontier   — the target is behind. Our own entry at k+1 records the epoch
//	                 of ITS predecessor (ringEntry.prevEpoch) — i.e. our epoch at
//	                 seq k — so comparing that against kEpoch verifies the join
//	                 point exactly. If k+1 is no longer retained we cannot decide
//	                 locally; that is not an error here, because the very next step
//	                 (ringDeltaLocked) fails it as ring-cold anyway.
//
// This is a fail-fast optimization, NOT the safety boundary: receiveLocked's chain
// check is authoritative and catches any divergence this cannot see locally
// (surfacing as ErrCatchupDiverged from backfillLearner's no-progress branch). It
// exists because the alternative is shipping a delta that will certainly be
// rejected — and because for k > frontier the backfill loop's `tail - k` would
// underflow.
func (e *Engine) checkCatchupDivergenceLocked(k, kEpoch uint64) error {
	frontier, frontierEpoch := e.appliedFrontierLocked()
	switch {
	case k > frontier:
		return ErrCatchupDiverged
	case k == frontier:
		if kEpoch != frontierEpoch {
			return ErrCatchupDiverged
		}
	default:
		if ent, has := e.backlog.at(k + 1); has && ent.prevEpoch != kEpoch {
			return ErrCatchupDiverged
		}
	}
	return nil
}

// clearLearnersLocked drops every in-progress grow learner from the ship-set. It
// is called on EVERY epoch advance (alongside clearEffISRLocked) so a learner
// caught up under one epoch can never linger in a later epoch's ship-set. It only
// deletes the MAP entries (stopping proposeSequenced from shipping to them) — it
// does NOT tear down their sender goroutines, because the epoch-advance callers
// (AdoptEpoch / GrantLease / Promote / receiveLocked) hold ONLY e.mu, and a
// sender teardown (dropPeerLocked) additionally requires writeMu, which they must
// not acquire while holding e.mu (the writeMu-before-e.mu ordering). An orphaned
// learner sender stops receiving new frames (it is out of the ship-set), drains
// what it holds, and is reclaimed by the grow driver's abort (AbortLearnerCatchup,
// which holds both locks) or by Shutdown. Caller holds e.mu.
func (e *Engine) clearLearnersLocked() {
	if len(e.learners) == 0 {
		return
	}
	if e.learnerTeardown == nil {
		e.learnerTeardown = make(map[string]bool)
	}
	for peer := range e.learners {
		// Park the sender for reclaim under writeMu+e.mu by the grow driver's
		// ReclaimOrphanLearners (we hold only e.mu here and must not take writeMu).
		e.learnerTeardown[peer] = true
		delete(e.learners, peer)
	}
}

// ringDeltaLocked returns the retained writes for seqs [from, to] inclusive as
// ordered, (PrevSeq, PrevEpoch)-chained ReplicateMsgs read from the catch-up ring,
// COPYING each payload (the returned msgs are shipped OFF e.mu, so they must not
// alias a ring buffer a later window wrap could recycle). ok is false when `from`
// precedes the oldest retained seq — the delta has fallen out of the ring and a
// snapshot transfer would be required. An empty range (to < from)
// returns (nil, true). Caller holds e.mu.
//
// THE PrevEpoch SUBTLETY, solved at the SOURCE rather than here.
// A replayed frame must carry the epoch of ITS predecessor, which is the epoch of
// the ring entry one seq below it. Deriving that at replay time from "the previous
// entry in this delta" fails at the delta's FIRST element: its predecessor is
// `from-1`, which is by construction NOT in the delta and is frequently not in the
// ring at all (the common case is from-1 == oldest-1, i.e. exactly the entry the
// ring last evicted). Reconstructing it from what happens to still be retained
// would make a WIRE-VISIBLE identity depend on local eviction state — the same
// class of mistake as reading lastApplied for the frontier. So each entry carries
// its own chain link, stamped at append time from the primary's live frontier
// (see ringEntry.prevEpoch and proposeSequenced), and replay just reads it back.
// A replayed frame is then byte-identical to the one originally shipped, whatever
// the ring still holds.
//
// UNIFORM-EPOCH RUNS. The delta is additionally TRUNCATED at the first epoch
// change: the group wire format (encodeReplicateGroup) declares ONE epoch for the
// whole frame and rebuilds the per-record chain from it, so a delta spanning an
// epoch boundary would be re-stamped to the first record's epoch on the wire — it
// would arrive claiming a history that never existed. The ring CAN span epochs (a
// primary that survives an epoch bump keeps its backlog), so this is reachable.
// Truncating costs nothing: backfillLearner simply runs another round from the
// boundary.
func (e *Engine) ringDeltaLocked(from, to uint64) ([]ReplicateMsg, bool) {
	if to < from {
		return nil, true // nothing to ship
	}
	oldest, newest, ok := e.backlog.span()
	if !ok || from < oldest {
		return nil, false // ring-cold / evicted: needs a snapshot
	}
	if to > newest {
		to = newest
	}
	msgs := make([]ReplicateMsg, 0, to-from+1)
	var runEpoch uint64
	for s := from; s <= to; s++ {
		ent, has := e.backlog.at(s)
		if !has {
			return nil, false // a hole in the retained range — treat as ring-cold
		}
		if len(msgs) == 0 {
			runEpoch = ent.epoch
		} else if ent.epoch != runEpoch {
			break // epoch boundary: end this uniform-epoch run (see the doc above)
		}
		msgs = append(msgs, ReplicateMsg{
			Epoch:     ent.epoch,
			Seq:       ent.seq,
			PrevSeq:   ent.seq - 1,
			PrevEpoch: ent.prevEpoch,                    // the chain link as originally shipped
			Data:      append([]byte(nil), ent.data...), // COPY: shipped off-lock
		})
	}
	return msgs, true
}

// backfillLearner ships the ring delta to peer in group-frame rounds until the
// target's applied high-water k is within `gap` of the current write frontier
// (LastSeq). It runs OFF writeMu and is the target's SOLE replication source
// while it runs (the target is not yet a peer or a learner). It re-fences the
// growing epoch E on every round and returns the final high-water k it drove the
// target to, or an error on any abort. Caller must NOT hold writeMu or e.mu.
//
// gap == 0 catches the target to the exact frozen high-water (a
// quiesced frontier); gap == pipelineWindow leaves the final <=W tail for the
// atomic flip to hand off.
func (e *Engine) backfillLearner(ctx context.Context, peer string, gt GroupTransport, E, k uint64, gap uint64) (uint64, error) {
	for {
		if err := ctx.Err(); err != nil {
			return k, err
		}
		e.mu.Lock()
		// Re-fence: only keep streaming while WE still hold epoch E's lease. A
		// failover/lease loss voids the grow (a late OpSetShardISR(E) no-ops).
		if e.epoch != E || e.leaseEpoch != E {
			e.mu.Unlock()
			return k, ErrGrowEpochChanged
		}
		tail := e.lastSeq
		if tail-k <= gap {
			e.mu.Unlock()
			return k, nil // caught up to within the requested gap
		}
		upTo := k + growBackfillBatch
		if upTo > tail {
			upTo = tail
		}
		msgs, ringOK := e.ringDeltaLocked(k+1, upTo)
		e.mu.Unlock()
		if !ringOK {
			return k, ErrGrowRingEvicted
		}

		ack, err := e.shipGroupSync(ctx, gt, peer, msgs)
		if err != nil {
			return k, err
		}
		if ack.Epoch > E {
			return k, ErrGrowPeerAhead
		}
		// Cumulative-ack semantics (ReceiveGroup): ack.Seq is the target's new
		// applied high-water whether the whole group applied (OK) or only a prefix
		// (a nack short of upTo). Advance k to it; no advance means the target
		// rejected the FIRST frame, i.e. its applied frontier is not the
		// (PrevSeq, PrevEpoch) that frame names — a log-matching failure, not a
		// transient. There is no resend and no snapshot path, so abort.
		// BOUND the credit by what this group actually shipped. ack.Seq arrives
		// from a peer over the wire, and everything below drives off k: the next
		// delta, the flip's tail-k arithmetic, and the loop's termination. An
		// unvalidated value could advance k past the frontier — a peer claiming
		// to have applied writes that were never sent — and the subsequent
		// tail-k subtraction is unsigned. Nothing downstream re-checks it.
		//
		// The upper bound is the last seq in the group just shipped: a
		// cumulative ack cannot legitimately credit beyond it. Over-credit is
		// treated as a log-matching failure rather than clamped, because a peer
		// reporting a frontier it cannot hold is not a peer whose deltas should
		// be trusted.
		if ack.Seq <= k || ack.Seq > msgs[len(msgs)-1].Seq {
			return k, ErrCatchupDiverged
		}
		k = ack.Seq
	}
}

// shipGroupSync ships one group frame and BLOCKS (off-lock) for its cumulative
// ack, adapting the async GroupTransport.ReplicateGroup contract to the
// synchronous backfill loop. A submit error IS the completion (done will not
// fire); otherwise done fires exactly once into the buffered channel. ctx
// cancellation abandons the wait (the late callback lands harmlessly in the
// buffered channel).
func (e *Engine) shipGroupSync(ctx context.Context, gt GroupTransport, peer string, msgs []ReplicateMsg) (AckMsg, error) {
	type result struct {
		ack AckMsg
		err error
	}
	ch := make(chan result, 1)
	if err := gt.ReplicateGroup(peer, msgs, func(ack AckMsg, cbErr error) {
		ch <- result{ack, cbErr}
	}); err != nil {
		return AckMsg{}, err
	}
	select {
	case r := <-ch:
		return r.ack, r.err
	case <-ctx.Done():
		return AckMsg{}, ctx.Err()
	}
}

// StartLearnerCatchup catches a lagging survivor `peer` up to this primary's
// write frontier and installs it as a LEARNER via the atomic flip, so the
// driver's subsequent OpSetShardISR(E, S∪M) re-add cannot open a seq gap on M.
// On success M is in the engine's learner ship-set (receiving every future seq)
// with the final ring delta pre-loaded on its fresh sender; the caller (the grow
// driver) then commits the ISR widening and calls GrowISR.
//
// It runs entirely off writeMu (a catch-up is a NON-propose path, so it is not
// blocked by a below-minISR admission gate — see the recovery proof).
//
// Ordering that makes the no-gap / no-committed-required-absent proofs hold:
//   - No SEQ GAP reaches M. Pre-flip, M's ONLY source is the single synchronous
//     backfill stream (dense, k exact). The backfill is quiesced before the flip,
//     so no two sources are ever concurrent. At the flip (seq assignment frozen
//     under writeMu) the ring hands M a CONTIGUOUS [k+1..tail], pre-loaded onto
//     M's fresh sender's channel; the engine's normal ship path then appends
//     tail+1.. behind it. One sender goroutine drains that channel FIFO →
//     strictly dense on M.
//   - No COMMITTED write is required-but-absent on M. learners[M]=true is set at
//     the flip STRICTLY BEFORE the driver submits OpSetShardISR(E, S∪M), and a
//     write becomes M-required only once proposeSequenced observes ctrl.ISR ⊇ M
//     (strictly after that submit commits). So for every M-required seq r, M was
//     in the ship-set since the flip ≤ r and received k+1..r-1 gap-free.
//     In-flight-at-flip records keep their pre-grow required set (P6); M's acks
//     for them no-op (removePending is idempotent on an absent member).
//
// STAGE 4.3 — SNAPSHOT FALLBACK. The delta path below is unchanged; what is new
// is that its two TERMINAL failures are no longer terminal. ErrGrowRingEvicted
// (the target needs a delta that fell out of the ring) and ErrCatchupDiverged
// (the target holds a fork, which append cannot repair) both mean "no delta
// exists that would work" — and the answer to that is a full state transfer,
// after which the SAME delta path runs again from the newly established prefix.
//
// The loop is what handles the ring-origin coupling: a snapshot establishes
// frontier F on the target, but if the primary writes more than ringCapacity
// entries before the flip, F+1 has fallen out of the ring and the retry surfaces
// ErrGrowRingEvicted AGAIN — so we re-snapshot from the newer frontier and try
// once more. Bounded by maxGrowSnapshotRounds, then a clean abort with nothing
// half-installed.
//
// Without a snapshot capability wired, the ORIGINAL error is returned unchanged,
// so an engine built without WithSnapshotStore behaves byte-identically to
// the durable frontier.
func (e *Engine) StartLearnerCatchup(ctx context.Context, peer string) error {
	err := e.catchupOnce(ctx, peer)
	if !needsSnapshot(err) || e.snap == nil {
		return err
	}
	E := e.Epoch()
	for round := 0; round < maxGrowSnapshotRounds; round++ {
		if _, _, serr := e.snapshotCatchup(ctx, peer, E); serr != nil {
			return serr
		}
		err = e.catchupOnce(ctx, peer)
		if !needsSnapshot(err) {
			return err // success, or an abort a snapshot cannot fix (epoch change, ctx, peer ahead)
		}
	}
	return ErrGrowSnapshotExhausted
}

// needsSnapshot reports whether err is one of the two states that ONLY a full
// state transfer can exit: the needed delta is no longer retained, or the target
// is not a prefix of our history at all.
func needsSnapshot(err error) bool {
	return errors.Is(err, ErrGrowRingEvicted) ||
		errors.Is(err, ErrCatchupDiverged) ||
		errors.Is(err, ErrCatchupUnverifiable)
}

// catchupOnce is the StartLearnerCatchup body verbatim: handshake, log
// -matching gate, far-backfill, atomic flip. It is factored out so the snapshot
// fallback can re-run it after establishing a prefix, WITHOUT this stage
// modifying the flip or its proof in any way.
func (e *Engine) catchupOnce(ctx context.Context, peer string) error {
	ct, ok := e.tr.(CatchupTransport)
	if !ok {
		return ErrGrowNoCatchupTransport
	}
	gt, ok := e.tr.(GroupTransport)
	if !ok {
		return ErrGrowNoGroupTransport
	}
	E := e.Epoch()

	// (1) Handshake: learn the target's LOG IDENTITY. Abort if it has already
	// adopted a higher epoch than the one we are growing at.
	info, err := ct.CatchupRequest(peer, E)
	if err != nil {
		return err
	}
	if info.Epoch > E {
		return ErrGrowPeerAhead
	}
	// An un-OK handshake means the target CANNOT state its log identity
	// — today that is exactly a poison-fenced node, mid-repair from an earlier
	// aborted install. Its reported watermarks are zeroed on purpose, and resuming
	// a delta from them would be resuming from a number nothing backs. Route it to
	// the snapshot path, which is the only thing that can give it an identity
	// again. (Checking OK at all is new: before this stage nothing ever answered
	// not-OK, so the field was dead.)
	if !info.OK {
		return ErrCatchupUnverifiable
	}
	// Resume from the target's applied FRONTIER, not its backup lastApplied: a node
	// that proposed as primary HOLDS those writes even though lastApplied never
	// counted them, so backfilling from lastApplied would ship it a delta starting
	// below its own log end — which receiveLocked now rejects (and used to accept,
	// silently interleaving two histories).
	k := info.FrontierSeq

	// (1b) LOG MATCHING gate, before anything is shipped: is the target a PREFIX of
	// our history at all? A fork cannot be repaired by append (hence snapshot),
	// and a target ahead of our frontier would underflow the backfill's `tail - k`.
	e.mu.Lock()
	derr := e.checkCatchupDivergenceLocked(k, info.FrontierEpoch)
	e.mu.Unlock()
	if derr != nil {
		return derr
	}

	// (2) Far-backfill (off writeMu) until the target is within ONE pipeline window
	// of the frontier, so the final gap the flip hands off fits in the fresh
	// sender's buffered channel (<= pipelineWindow) and never blocks under the lock.
	k, err = e.backfillLearner(ctx, peer, gt, E, k, pipelineWindow)
	if err != nil {
		return err
	}

	// (3) ATOMIC FLIP: under writeMu+e.mu, re-verify E, pre-load the final ring
	// delta onto M's fresh sender, and add M to the learner ship-set.
	return e.flipLearner(ctx, peer, gt, E, k)
}

// flipLearner performs the atomic learner flip (StartLearnerCatchup step 3). It
// holds writeMu+e.mu so seq assignment is frozen for the critical section, does
// NO network I/O under the lock (only channel sends), and installs the learner
// AFTER pre-loading the delta so M's fresh sender is a single gap-free FIFO
// source. If the frontier raced past one window between the backfill exit and the
// lock, it resumes the backfill (off-lock) and retries. Aborts (returning an
// error, having installed nothing) if the growing epoch/lease is gone.
func (e *Engine) flipLearner(ctx context.Context, peer string, gt GroupTransport, E, k uint64) error {
	for {
		if err := ctx.Err(); err != nil {
			return err
		}
		e.writeMu.Lock()
		e.mu.Lock()
		if e.closed {
			e.mu.Unlock()
			e.writeMu.Unlock()
			return ErrGrowEpochChanged
		}
		// Re-verify the growing epoch AND a still-live lease UNDER writeMu (seq
		// assignment is now frozen). A failover/lease loss voids the grow; a late
		// OpSetShardISR(E) is epoch-guarded to a no-op, so nothing diverges.
		if e.epoch != E || e.leaseEpoch != E || e.now() >= e.leaseExpiry {
			e.mu.Unlock()
			e.writeMu.Unlock()
			return ErrGrowEpochChanged
		}
		tail := e.lastSeq
		remaining := tail - k // tail >= k (k only ever advanced from applied writes)
		if remaining > pipelineWindow {
			// The frontier advanced past our window between the backfill exit and
			// this lock. Release and catch the extra up, then retry the flip. Bounded
			// by ctx (checked at the loop top); the driver retries on a ctx timeout.
			e.mu.Unlock()
			e.writeMu.Unlock()
			var err error
			k, err = e.backfillLearner(ctx, peer, gt, E, k, pipelineWindow)
			if err != nil {
				return err
			}
			continue
		}
		// Build the final contiguous delta [k+1..tail] (<= pipelineWindow entries).
		delta, ringOK := e.ringDeltaLocked(k+1, tail)
		if !ringOK {
			e.mu.Unlock()
			e.writeMu.Unlock()
			return ErrGrowRingEvicted
		}
		// ringDeltaLocked truncates at an epoch boundary (uniform-epoch groups), so
		// the delta may be SHORT of tail — and a short pre-load would hand M a gap
		// between the delta's end and the tail+1.. frames the normal ship path
		// appends behind it. Never flip on a short delta: drop the locks, backfill
		// the remainder off-lock (which crosses the boundary in its own round), and
		// retry the flip. Bounded by ctx like the window-overrun branch above.
		//
		// gap == 0 (not pipelineWindow) is REQUIRED here: the truncation is usually
		// INSIDE the final window, and a windowed backfill would return immediately
		// without shipping anything, spinning this loop forever. gap == 0 forces at
		// least one round, which crosses the boundary (each round re-reads the ring
		// from k+1 and so starts a fresh uniform-epoch run) and advances k.
		if uint64(len(delta)) != tail-k {
			e.mu.Unlock()
			e.writeMu.Unlock()
			var err error
			k, err = e.backfillLearner(ctx, peer, gt, E, k, 0)
			if err != nil {
				return err
			}
			continue
		}
		e.createLearnerSenderLocked(peer, delta)
		if e.learners == nil {
			e.learners = make(map[string]bool)
		}
		// A FRESH grow supersedes any stale abandon/teardown signal for this peer, so
		// the driver reads this attempt's outcome, not a prior one. Cleared UNDER
		// writeMu+e.mu (held here), serialized against ReclaimOrphanLearners so the
		// fresh sender is never reclaimed out from under us.
		delete(e.abandonedLearners, peer)
		delete(e.learnerTeardown, peer)
		e.learners[peer] = true // ship-set membership BEFORE the driver's OpSetShardISR
		e.mu.Unlock()
		e.writeMu.Unlock()
		return nil
	}
}

// createLearnerSenderLocked builds M's FRESH per-peer sender and pre-loads the
// final ring delta [k+1..tail] onto its channel (channel sends ONLY — no network
// I/O under the lock). remaining <= pipelineWindow == the channel buffer, so the
// pre-load never blocks. Each push bumps pending so the inline fast path stays
// disabled until the sender drains the pre-loaded frames — the normal ship path
// then appends tail+1.. behind them into ONE gap-free FIFO. Caller holds
// writeMu AND e.mu.
func (e *Engine) createLearnerSenderLocked(peer string, delta []ReplicateMsg) {
	// Defensive: a stale sender for this peer would be a second, racing source.
	// Drop it (discard-without-submit) before building the fresh one. In the real
	// grow paths (post-failover, or shrink-then-grow) M has no sender here.
	if _, exists := e.peerQ[peer]; exists {
		e.dropPeerLocked(peer)
	}
	ps := &peerSender{ch: make(chan ReplicateMsg, pipelineWindow)}
	if e.peerQ == nil {
		e.peerQ = make(map[string]*peerSender)
	}
	e.peerQ[peer] = ps
	e.senderWG.Add(1)
	go e.runSender(peer, ps)
	for i := range delta {
		ps.pending.Add(1)
		ps.ch <- delta[i] // <= pipelineWindow sends into a pipelineWindow-buffered channel
	}
}

// learnerShipSetLocked returns the learners that are NOT already required peers
// for this write — the set proposeSequenced ships a NON-BLOCKING learner copy to.
// A member in BOTH sets (the brief learner→voter overlap once ctrl.ISR ⊇ M) is
// shipped ONCE, via the required `peers` path; excluding it here is what keeps
// the transition duplicate-free. Caller holds e.mu.
func (e *Engine) learnerShipSetLocked(peers []string) []string {
	if len(e.learners) == 0 {
		return nil
	}
	var out []string
	for l := range e.learners {
		inPeers := false
		for _, p := range peers {
			if p == l {
				inPeers = true
				break
			}
		}
		if !inPeers {
			out = append(out, l)
		}
	}
	return out
}

// submitLearnerLocked ships msg to a LEARNER via a NON-BLOCKING channel send. A
// learner does NOT gate commit, so it must NEVER block the write path under
// writeMu. A learner ALWAYS routes through its sender channel (never the inline
// fast path): its channel was pre-loaded at the flip with [k+1..tail] and post-
// flip ordering IS the single sender draining that FIFO, so an inline link-append
// would jump the queued delta and open a gap. If the channel is FULL the learner
// cannot keep up and the grow is hopeless — ABANDON it: discard its parked frames
// (dropPeerLocked) and drop it from the learner set, so the write path proceeds
// (committed still advances via the required peers). Caller holds writeMu.
func (e *Engine) submitLearnerLocked(peer string, msg ReplicateMsg) {
	if e.closed {
		return
	}
	ps := e.peerQ[peer]
	if ps == nil {
		return // no sender (torn down / never flipped): nothing to ship
	}
	ps.pending.Add(1)
	select {
	case ps.ch <- msg:
	default:
		// Full channel: undo the pending bump and abandon this grow. dropPeerLocked,
		// the learner delete, AND the abandon record all need e.mu (we already hold
		// writeMu). RECORDING the abandon (abandonedLearners[peer] = msg.Seq) is the
		// coordination signal the grow driver reads: if a widen for this peer already
		// committed (or commits), the driver sees the peer is a GAPPED voter and
		// compensates it back out of the ISR. Without this the abandon would be silent
		// and the widen would leave a gapped required member → wedge/acked-loss.
		ps.pending.Add(-1)
		e.mu.Lock()
		e.dropPeerLocked(peer)
		delete(e.learners, peer)
		e.markLearnerAbandonedLocked(peer, msg.Seq)
		e.mu.Unlock()
	}
}

// GrowISR reconciles a MetaRaft-committed ISR WIDENING to newISR for epoch — the
// FUTURE-ONLY, INVERSE-direction twin of ShrinkISR. It installs the same-epoch
// effective-ISR override so proposeSequenced's intersect stops narrowing a
// re-added member back out after a PRIOR shrink at this epoch. Unlike ShrinkISR it
// does NOT narrow any already-registered in-flight record and does NOT drop any
// sender: in-flight safety is automatic (P6 — a record's required set is
// snapshotted at propose time, and widening the committed ISR never touches it).
// Widening a live record's required set would demand an ack for a write a member
// may never have received — the load-bearing shrink/grow asymmetry.
//
// It deliberately does NOT remove newISR members from the learner set: the
// learnerShipSetLocked `\ peers` guard already makes M shipped-exactly-once across
// the learner→voter transition, and removing M here could open a gap for a
// concurrently in-flight Propose that snapshotted a stale (pre-widen) ctrl.ISR
// (that Propose would then ship M via NEITHER peers NOR learners). M lingers as an
// INERT learner (always excluded via `\ peers` once it is a voter) until the next
// epoch advance clears it.
//
// Epoch/lease-fenced exactly like ShrinkISR: a stale-epoch grow, or one after this
// node was fenced, touches nothing (a late OpSetShardISR(E) already no-oped at the
// FSM). It needs ONLY e.mu (no sender/in-flight mutation): proposeSequenced reads
// the override under e.mu, so the install serializes against it there.
func (e *Engine) GrowISR(epoch uint64, newISR []string) {
	e.mu.Lock()
	defer e.mu.Unlock()
	if epoch != e.epoch || e.leaseEpoch != epoch {
		return
	}
	// Unifying rule with shrink: effISR ALWAYS mirrors the driver's last-reconciled
	// committed ISR at the current epoch (shrink → narrower + narrow-in-flight +
	// drop; grow → wider, future-only).
	e.effISR = append([]string(nil), newISR...)
	e.effISREpoch = epoch
}

// AbortLearnerCatchup tears down an in-progress or completed grow for peer: under
// writeMu+e.mu it drops the learner's sender (discarding parked frames) and
// removes it from the learner set. It deliberately does NOT clear the
// abandonedLearners signal — that must survive until the grow driver has
// compensated the peer OUT of the committed ISR (only a fresh flipLearner clears
// it). Idempotent and safe when no grow exists. The grow driver calls it when it
// abandons a grow so an orphaned learner sender does not linger until Shutdown.
func (e *Engine) AbortLearnerCatchup(peer string) {
	e.writeMu.Lock()
	e.mu.Lock()
	delete(e.learners, peer)
	delete(e.learnerTeardown, peer)
	e.dropPeerLocked(peer)
	e.mu.Unlock()
	e.writeMu.Unlock()
}

// TeardownLearners abandons EVERY in-progress grow learner for this engine and
// reclaims their sender goroutines (both live learners and any parked by an epoch
// advance). The grow driver calls it for a shard this node no longer primaries, so
// no learner lingers when this node has no business growing the shard. It does NOT
// clear abandonedLearners (those are epoch-scoped compensation signals cleared by a
// fresh grow). Idempotent.
func (e *Engine) TeardownLearners() {
	e.writeMu.Lock()
	e.mu.Lock()
	for peer := range e.learners {
		e.dropPeerLocked(peer)
		delete(e.learners, peer)
	}
	for peer := range e.learnerTeardown {
		e.dropPeerLocked(peer)
		delete(e.learnerTeardown, peer)
	}
	e.mu.Unlock()
	e.writeMu.Unlock()
}

// ReclaimOrphanLearners drops the sender goroutines of learners that an epoch
// advance removed from the ship-set (clearLearnersLocked parked them because it
// holds only e.mu and cannot drop a sender). The grow driver calls it each tick;
// it is cheap (usually a no-op — the parked set is empty in steady state).
func (e *Engine) ReclaimOrphanLearners() {
	e.writeMu.Lock()
	e.mu.Lock()
	for peer := range e.learnerTeardown {
		e.dropPeerLocked(peer)
		delete(e.learnerTeardown, peer)
	}
	e.mu.Unlock()
	e.writeMu.Unlock()
}

// markLearnerAbandonedLocked records that a grow for peer was abandoned at seq
// (its learner channel filled). Caller holds e.mu.
func (e *Engine) markLearnerAbandonedLocked(peer string, seq uint64) {
	if e.abandonedLearners == nil {
		e.abandonedLearners = make(map[string]uint64)
	}
	e.abandonedLearners[peer] = seq
}

// LearnerAbandonedAt reports whether a grow for peer was abandoned (channel-full),
// and the seq at which it happened — the grow driver's signal to compensate a
// gapped-voter peer out of the ISR. It takes e.mu.
func (e *Engine) LearnerAbandonedAt(peer string) (uint64, bool) {
	e.mu.Lock()
	defer e.mu.Unlock()
	seq, ok := e.abandonedLearners[peer]
	return seq, ok
}

// LearnerAbandoned reports whether a grow for peer was abandoned (channel-full).
func (e *Engine) LearnerAbandoned(peer string) bool {
	_, ok := e.LearnerAbandonedAt(peer)
	return ok
}

// IsLearner reports whether peer is currently in this engine's grow learner
// ship-set (test/introspection helper). It takes e.mu.
func (e *Engine) IsLearner(peer string) bool {
	e.mu.Lock()
	defer e.mu.Unlock()
	return e.learners[peer]
}
