// SPDX-License-Identifier: Apache-2.0

package pbisr

import (
	"context"
	"errors"
	"fmt"
)

// ============================================================================
// SNAPSHOT TRANSFER (the only exit from ring-cold and diverged).
//
// PB catch-up (pb_grow.go) ships a delta out of the primary's BOUNDED in-memory
// ring. Two states have no delta to ship, and before this stage both were
// silent and PERMANENT:
//
//  1. RING-COLD (ErrGrowRingEvicted). The delta the target needs precedes the
//     primary's oldest retained seq. This is not an edge case after a failover:
//     Promote leaves the backlog UNTOUCHED and the sole ring-append site is
//     proposeSequenced, so a freshly promoted primary's ring is EMPTY and its
//     origin is promotionHighWater+1. A survivor lagging by ONE write needs a
//     delta starting below that origin. The grow then succeeds only on an exact
//     high-water tie — i.e. only when there was nothing to transfer.
//
//  2. DIVERGED (ErrCatchupDiverged). A forked node is correctly
//     REFUSED rather than silently interleaved into the ISR, but the refusal is
//     terminal: catch-up is log APPEND, and no amount of appending repairs a
//     fork.
//
// With -min-isr >= 2 these are worse than lost redundancy. OpSetShardEpoch
// resets the ISR to {newPrimary}; Propose then returns ErrBelowMinISR; the
// primary cannot write; its ring never fills; the grow can never be served. The
// shard is permanently WRITE-DEAD. Snapshot transfer is the only exit.
//
// THE HAND-OFF OBLIGATION (and it is the whole contract). The learner flip's
// soundness proof requires a gap-free [k+1..tail]. This path's ONLY obligation
// is:
//
//	after install, the target's FSM is byte-identical to the primary's FSM at
//	frontier F, AND the target's engine reports frontier F with F's epoch —
//	ATOMICALLY with the install, in memory AND durably.
//
// Given that, k := F and backfillLearner + flipLearner apply VERBATIM. This
// stage modifies NEITHER. The snapshot is a prefix-establishment step in FRONT
// of the existing flip, never a replacement for it.
//
// WHY THE POST-SNAPSHOT FLIP IS TRIVIAL IN THE CASE THAT MATTERS. On a freshly
// promoted primary lastSeq == lastApplied, so the applied frontier F IS lastSeq
// == tail. After the install backfillLearner sees tail-k == 0 and returns
// without touching the ring at all, and flipLearner's ringDeltaLocked(F+1, F)
// takes the empty-range early return. The write-dead shard therefore converges
// in exactly ONE snapshot round, over a ring that is still empty — which is the
// point, since a below-floor primary's ring cannot fill.
//
// RING-ORIGIN COUPLING (the healthy-shard case). F+1 must still be >= the
// primary's ring origin when the flip runs. If the primary wrote more than
// ringCapacity entries during the transfer, F+1 has fallen out and the delta is
// unservable again — which surfaces as ErrGrowRingEvicted from the very same
// backfill. StartLearnerCatchup therefore LOOPS: snapshot -> catch-up -> if
// ring-cold again, re-snapshot from the newer frontier -> ... bounded by
// maxGrowSnapshotRounds, then a clean abort. Note the minISR interaction that
// makes this converge trivially in the important case: a write-blocked primary's
// ring does not advance at all, so round 1 always wins.
//
// NEITHER RunExclusive NOR flipLearner REQUIRES THE ISR FLOOR. Both require the
// LEASE, which a below-floor primary still holds (the floor is checked in
// proposeSequenced step (b) only). That is what lets a write-dead shard run a
// snapshot transfer and a flip at all.
//
// THE DISCARD IS SAFE — WITH ONE DOCUMENTED EXCEPTION. restoreSnapshot WIPES the
// target's pre-restore key set; the divergent tail is DISCARDED, not merged. The
// target is a LEARNER: it is in no in-flight record's required set, and under
// CommitFullISR the primary holds a superset of every acked write (P6), so
// nothing client-acked is lost. Under CommitPrimary that is NOT true: a demoted
// ex-primary may hold writes it acked alone, and discarding them IS loss by that
// contract. CommitPrimary already trades exactly that durability for latency;
// this stage does not pretend otherwise, it names it.
// ============================================================================

// Snapshot sentinel errors.
//
// All are EXPORTED under the convention (see pb_grow.go's sentinel
// block) so cluster/'s growAbortReason can classify a snapshot-path abort into
// its own bucket instead of collapsing it into "other". They name failure modes
// that did not exist when that classifier was written, and an abort counter
// that cannot name a failure is one an operator cannot act on.
var (
	// ErrGrowNoSnapshotStore / ErrGrowNoSnapshotTransport — this engine (or its
	// transport) has no snapshot capability wired, so a ring-cold or diverged
	// target CANNOT be repaired. The grow surfaces the original ring-cold/diverged
	// error unchanged, preserving pre-Stage-4.3 behavior exactly. CONFIGURATION,
	// not a transient: retrying will not fix it until an operator wires a store.
	ErrGrowNoSnapshotStore     = errors.New("pbisr: engine has no snapshot store (cannot repair a ring-cold/diverged target)")
	ErrGrowNoSnapshotTransport = errors.New("pbisr: transport does not support snapshot transfer")

	// ErrGrowSnapshotRejected — the target refused the snapshot install (a stale
	// epoch at the target, a malformed/failed install, or a nack). Abort the grow;
	// the driver retries next tick. RETRYABLE.
	ErrGrowSnapshotRejected = errors.New("pbisr: snapshot install rejected by target")

	// ErrGrowSnapshotExhausted — the snapshot -> catch-up loop ran its bounded
	// rounds without converging (the primary kept out-writing the ring faster than
	// a snapshot could be transferred). A clean abort: nothing was installed into
	// the ISR and the target is left consistent (it holds SOME complete snapshot,
	// just not a currently-servable one). RETRYABLE, but a persistent count means
	// the write rate is outrunning state transfer and needs operator attention.
	ErrGrowSnapshotExhausted = errors.New("pbisr: snapshot catch-up did not converge in the allowed rounds")

	// ErrSnapshotPending — this engine is POISON-FENCED: a snapshot install
	// began and did not provably complete, so its FSM may be half-wiped and its
	// watermark means nothing. Every write path, receive path and promotion path
	// refuses until a fresh snapshot install succeeds. See the poison fence note
	// on installSnapshot.
	ErrSnapshotPending = errors.New("pbisr: snapshot install pending — state unusable until re-snapshotted")
)

// maxGrowSnapshotRounds bounds the snapshot -> catch-up convergence loop inside
// ONE StartLearnerCatchup attempt. Each round costs a full state transfer, so the
// bound is small: if the primary is out-writing a whole ring capacity per
// snapshot for three consecutive rounds, this grow is hopeless right now and the
// driver's next tick is a better place to retry than an unbounded loop holding a
// catch-up context.
const maxGrowSnapshotRounds = 3

// pbSnapshotChunkBytes is the payload each snapshot frame carries. Comfortably
// under pbMaxPayload (64 MiB) so a chunk frame is never itself oversize, and
// large enough that a multi-GiB shard is a few hundred frames rather than tens of
// thousands.
const pbSnapshotChunkBytes = 4 << 20

// pbSnapshotMaxBytes bounds the TOTAL a peer may declare for one snapshot, so a
// corrupt or hostile Total cannot drive an unbounded receive-side allocation.
// Deliberately enforced per appended chunk as well — the declared Total is never
// used to SIZE a reservation, only to validate the stream (the repo's
// bound-every-wire-declared-count rule).
const pbSnapshotMaxBytes = 4 << 30

// SnapshotChunk is one frame of a snapshot transfer.
//
// The two epochs are DIFFERENT things and conflating them is the classic bug:
//
//   - Epoch is the SHIPPING primary's current leadership generation. It is the
//     fence: a target that has adopted a higher epoch rejects the chunk, exactly
//     as receiveLocked fences a stale replicate frame.
//   - FrontierSeq/FrontierEpoch are the log IDENTITY of the state in the blob —
//     the (seq, epoch) of the newest write the serialized FSM materializes. On a
//     freshly promoted primary FrontierEpoch is the INHERITED epoch of the write
//     it was promoted at, strictly below Epoch, and installing it as anything
//     else would make the target advertise a predecessor no peer holds (the same
//     mistake Promote avoids by inheriting rather than stamping).
type SnapshotChunk struct {
	Epoch         uint64
	FrontierSeq   uint64
	FrontierEpoch uint64
	Offset        uint64 // byte offset of Data within the whole blob
	Total         uint64 // total blob length
	Final         bool   // last chunk: install now
	Data          []byte
}

// SnapshotStore is an OPTIONAL engine capability: the FSM-level
// serialize/install pair, plus the DURABLE POISON FENCE that makes a partial
// install unusable rather than merely stale. pbisr deliberately does not import
// shard, so the implementation is injected (WithSnapshotStore); shard's
// pbSnapshotStore is the production one.
//
// THE FENCE CONTRACT, stated as an ordering:
//
//	BeginInstall(F, Fe)   durably records "a restore to (F, Fe) is in progress"
//	InstallFSM(blob)      wipes and installs — NOT atomic, by construction
//	CommitInstall(F, Fe)  durably records frontier (F, Fe), THEN clears the marker
//
// A crash or abort anywhere between Begin and a SUCCESSFUL Commit leaves the
// marker set, and InstallPending() then reports true forever until a fresh
// install succeeds. The frontier must be persisted BEFORE the marker is cleared,
// never after: the reverse order admits a crash window in which an un-poisoned
// node carries its PRE-install watermark over POST-install state, and if the node
// was diverged AHEAD that stale watermark OVER-reports — which is precisely the
// direction the durable frontier exists to make unreachable.
type SnapshotStore interface {
	// SnapshotFSM serializes the whole local FSM into one self-describing blob,
	// stamping appliedIndex into it. The engine calls it with writeMu+e.mu held,
	// so no Applier.Apply can race it (see RunExclusive's argument).
	SnapshotFSM(appliedIndex uint64) ([]byte, error)

	// BeginInstall durably raises the poison fence for a pending restore to
	// (seq, epoch). Called OFF the engine locks.
	BeginInstall(seq, epoch uint64) error

	// InstallFSM discards local state and installs blob. Called with writeMu+e.mu
	// held. A non-nil error leaves the FSM in an INDETERMINATE state — which is
	// exactly what the fence exists to make safe.
	InstallFSM(blob []byte) error

	// CommitInstall persists the frontier (seq, epoch) and THEN clears the poison
	// fence, in that order, both durably. Called with writeMu+e.mu held.
	CommitInstall(seq, epoch uint64) error

	// InstallPending reports whether a durable poison fence is raised — i.e. this
	// node booted (or aborted) mid-install and must refuse to serve.
	InstallPending() bool

	// AbortInstall lowers the fence WITHOUT installing anything. The engine calls
	// it on the ONE abort path where the FSM is provably untouched: a newer epoch
	// observed between BeginInstall and the install critical section. Every other
	// abort deliberately leaves the fence raised.
	AbortInstall() error
}

// SnapshotTransport is an OPTIONAL Transport capability: ship ONE
// snapshot chunk to peer and block for its ack.
//
// It is deliberately a SEPARATE, SYNCHRONOUS channel rather than the peer sender
// path. submitLearnerLocked is non-blocking by design and ABANDONS the grow on a
// full channel — a snapshot would trip that on its first frame. shipGroupSync is
// the existing precedent for a blocking, off-both-locks control transfer driven
// by the grow goroutine; this follows it.
type SnapshotTransport interface {
	SendSnapshotChunk(peer string, c SnapshotChunk) (AckMsg, error)
}

// snapshotStage is the RECEIVE-side accumulation buffer for an in-progress
// transfer. It is PURE MEMORY and touches no FSM state, which is what makes an
// aborted transfer leave the target pristine: a dropped connection, an epoch
// change, or a bad offset simply discards the staging buffer, and the poison
// fence is never raised because the install never began. The fence therefore
// only ever covers the LOCAL wipe+install window (microseconds-to-seconds of
// disk work with no network in it), not the whole multi-second transfer.
// Guarded by e.mu.
type snapshotStage struct {
	epoch         uint64
	frontierSeq   uint64
	frontierEpoch uint64
	total         uint64
	buf           []byte
}

// WithSnapshotStore installs the FSM serialize/install/fence capability (Stage
// 4.3). Without it an engine behaves exactly as it did before this stage: a
// ring-cold or diverged target aborts the grow and is never repaired.
//
// It also seeds the engine's POISONED latch from the durable fence, so a node
// that crashed mid-install comes back refusing to serve rather than silently
// presenting a half-wiped FSM behind a meaningless watermark.
func WithSnapshotStore(ss SnapshotStore) Option {
	return func(e *Engine) {
		if ss == nil {
			return
		}
		e.snap = ss
		if ss.InstallPending() {
			e.poisoned = true
		}
	}
}

// SnapshotStats is a point-in-time view of this engine's snapshot-transfer
// activity, including the WRITE-STALL cost of the quiesced serialization. The
// stall is the honest price of flow-control option (a) (see BENCHMARK.md and the
// review report): SnapshotFSM runs under writeMu+e.mu, which excludes both
// Applier.Apply sites, so a shard's write path is frozen for its duration.
// Operators size shards against StallMaxNs.
type SnapshotStats struct {
	Taken       uint64 // snapshots serialized as a source
	Installed   uint64 // snapshots installed as a target
	StallLastNs int64  // duration of the most recent quiesced serialization
	StallMaxNs  int64  // longest quiesced serialization observed
	Poisoned    bool   // a snapshot install is pending: this node refuses to serve
}

// SnapshotStats snapshots the counters. Lock-free except for the poisoned read.
func (e *Engine) SnapshotStats() SnapshotStats {
	e.mu.Lock()
	poisoned := e.poisoned
	e.mu.Unlock()
	return SnapshotStats{
		Taken:       e.snapTaken.Load(),
		Installed:   e.snapInstalled.Load(),
		StallLastNs: e.snapStallLastNs.Load(),
		StallMaxNs:  e.snapStallMaxNs.Load(),
		Poisoned:    poisoned,
	}
}

// SnapshotCapable reports whether this engine can repair a ring-cold or diverged
// target by full state transfer. The grow driver reads it to decide how long a
// catch-up attempt may run: a delta is a handful of round trips, a state transfer
// is bounded by shard size and disk.
func (e *Engine) SnapshotCapable() bool {
	if e.snap == nil {
		return false
	}
	_, ok := e.tr.(SnapshotTransport)
	return ok
}

// Poisoned reports whether a snapshot install is pending on this engine — i.e.
// its FSM may be half-wiped and its watermark is meaningless. A poisoned engine
// refuses to propose, refuses to receive, and reports CatchupInfo{OK:false} so
// the failover gate treats it as UNVERIFIABLE and never promotes it. The only
// exit is a successful snapshot install.
func (e *Engine) Poisoned() bool {
	e.mu.Lock()
	defer e.mu.Unlock()
	return e.poisoned
}

// noteStall records one quiesced-serialization duration into the metrics.
func (e *Engine) noteStall(ns int64) {
	e.snapStallLastNs.Store(ns)
	for {
		cur := e.snapStallMaxNs.Load()
		if ns <= cur || e.snapStallMaxNs.CompareAndSwap(cur, ns) {
			return
		}
	}
}

// takeSnapshot serializes the local FSM under the SAME quiesce BackupSnapshot
// uses (writeMu+e.mu, excluding both Applier.Apply sites) and returns the blob
// together with the (seq, epoch) log identity it materializes.
//
// FLOW CONTROL — option (a), deliberately and with its cost named. The whole
// serialization runs inside the critical section, so a shard's writes are FROZEN
// for its duration. The alternatives were considered and rejected on evidence:
//
//   - (b) freeze-under-lock / serialize-off-lock is not reachable today. The
//     cache half has a plausible freeze point (mmap page-append with a
//     rebuildIndexFromPages recovery path), but the VECTOR half does not:
//     CollectionStore.SnapshotAll walks HNSW graphs that are mutated in place
//     under the collection lock, with no versioned or copy-on-write view to pin.
//     Freezing only the cache would produce a blob whose two halves are at
//     different logical points — precisely the torn snapshot RunExclusive exists
//     to prevent.
//
//   - (c) shipping from the primary's on-disk backup artifact has zero write-path
//     impact but makes convergence STRICTLY WORSE where it matters: an artifact's
//     frontier F is older, so F+1 is further outside the ring and the re-snapshot
//     loop is more likely to spin, not less.
//
// So the stall is accepted, MEASURED, and bounded by an operational shard-size
// ceiling rather than hidden. BenchmarkPBSnapshotStall (shard package,
// i9-13900H, 100-byte values, KV only):
//
//	keys      blob      stall
//	1e3      ~0.12 MB   0.68 ms
//	1e4      ~1.2 MB    2.70 ms
//	1e5      ~12 MB     31.2 ms
//	5e5      ~60 MB    134.9 ms
//
// i.e. ~450 MB/s, LINEAR in serialized bytes — so the operational rule is
//
//	stall ≈ shard_bytes / 450 MB/s
//
// and the documented CEILING is: a shard whose serialized size exceeds ~450 MB
// will stall its write path for over a second per transfer. Shards above that
// should be split. Vector collections serialize through the same critical section
// and add to the blob, so size the ceiling on TOTAL serialized bytes, not on the
// KV set alone. SnapshotStats.StallMaxNs reports the live figure per shard, which
// is what an operator should alert on rather than inferring from key counts.
//
// The stall is also mitigated by exactly the case that motivates this stage: a
// below-minISR primary is ALREADY refusing writes, so in the write-dead shard —
// the one that cannot recover any other way — there is no write traffic to stall.
//
// Caller must NOT hold writeMu or e.mu.
func (e *Engine) takeSnapshot(E uint64) (blob []byte, fseq, fepoch uint64, err error) {
	ss := e.snap
	if ss == nil {
		return nil, 0, 0, ErrGrowNoSnapshotStore
	}
	e.writeMu.Lock()
	e.mu.Lock()
	if e.closed {
		e.mu.Unlock()
		e.writeMu.Unlock()
		return nil, 0, 0, ErrGrowEpochChanged
	}
	// Fence the growing epoch under the SAME locks the serialization runs under, so
	// the blob provably belongs to epoch E's history and not a successor's.
	if e.epoch != E || e.leaseEpoch != E || e.now() >= e.leaseExpiry {
		e.mu.Unlock()
		e.writeMu.Unlock()
		return nil, 0, 0, ErrGrowEpochChanged
	}
	fseq, fepoch = e.appliedFrontierLocked()
	start := e.now()
	blob, err = ss.SnapshotFSM(fseq)
	stall := e.now() - start
	e.mu.Unlock()
	e.writeMu.Unlock()

	e.noteStall(stall)
	if err != nil {
		return nil, 0, 0, fmt.Errorf("pbisr: snapshot serialize: %w", err)
	}
	e.snapTaken.Add(1)
	return blob, fseq, fepoch, nil
}

// shipSnapshot streams blob to peer in chunks over the SYNCHRONOUS snapshot
// transport, off both engine locks, re-fencing epoch E on every ack. It is the
// grow-driver goroutine's own blocking work — never the write path's.
func (e *Engine) shipSnapshot(ctx context.Context, peer string, st SnapshotTransport, E, fseq, fepoch uint64, blob []byte) error {
	total := uint64(len(blob))
	for off := uint64(0); ; {
		if err := ctx.Err(); err != nil {
			return err
		}
		end := off + pbSnapshotChunkBytes
		if end > total {
			end = total
		}
		c := SnapshotChunk{
			Epoch:         E,
			FrontierSeq:   fseq,
			FrontierEpoch: fepoch,
			Offset:        off,
			Total:         total,
			Final:         end == total,
			Data:          blob[off:end],
		}
		ack, err := st.SendSnapshotChunk(peer, c)
		if err != nil {
			return err
		}
		// A target that has adopted a HIGHER epoch has moved past our leadership
		// generation; our transfer is stale and must not be completed.
		if ack.Epoch > E {
			return ErrGrowPeerAhead
		}
		if !ack.OK {
			return ErrGrowSnapshotRejected
		}
		if c.Final {
			return nil
		}
		off = end
		// Re-fence LOCALLY between chunks too: a lease loss mid-transfer voids the
		// grow, and continuing to stream would waste a target's disk on a snapshot
		// from a superseded primary.
		e.mu.Lock()
		stale := e.epoch != E || e.leaseEpoch != E
		e.mu.Unlock()
		if stale {
			return ErrGrowEpochChanged
		}
	}
}

// snapshotCatchup runs ONE full snapshot transfer to peer: serialize under the
// quiesce, stream it, and (on the target) install it atomically with both
// watermarks. It returns the frontier the target now holds.
func (e *Engine) snapshotCatchup(ctx context.Context, peer string, E uint64) (fseq, fepoch uint64, err error) {
	st, ok := e.tr.(SnapshotTransport)
	if !ok {
		return 0, 0, ErrGrowNoSnapshotTransport
	}
	blob, fseq, fepoch, err := e.takeSnapshot(E)
	if err != nil {
		return 0, 0, err
	}
	if err := e.shipSnapshot(ctx, peer, st, E, fseq, fepoch, blob); err != nil {
		return 0, 0, err
	}
	return fseq, fepoch, nil
}

// ============================================================================
// RECEIVE SIDE
// ============================================================================

// ReceiveSnapshotChunk is the TARGET side of a snapshot transfer: it accumulates
// chunks into a staging buffer and, on the final chunk, installs.
//
// Everything before the final chunk is pure memory: no FSM state is touched and
// no poison fence is raised, so an abort at ANY point mid-transfer (transport
// failure, epoch change, a bad offset, a hostile stream) leaves the target
// EXACTLY as it was. That is a stronger abort property than a streaming install
// could offer, and it is why the fence's window is only the local install.
//
// It is serialized per shard by e.mu, like Receive.
func (e *Engine) ReceiveSnapshotChunk(c SnapshotChunk) AckMsg {
	nack := AckMsg{Epoch: c.Epoch, Seq: c.FrontierSeq, OK: false}

	e.mu.Lock()
	// (a) Epoch fence (H1/H5), identical in shape to receiveLocked: reject a stale
	// primary; adopt a higher epoch (which also voids any shrink override and grow
	// learners this node held as an older-epoch primary).
	if c.Epoch < e.epoch {
		e.snapStage = nil
		e.mu.Unlock()
		return AckMsg{Epoch: e.epoch, Seq: c.FrontierSeq, OK: false}
	}
	if c.Epoch > e.epoch {
		e.epoch = c.Epoch
		e.clearEffISRLocked()
		e.clearLearnersLocked()
		e.flushEpochLocked(e.epoch)
		e.snapStage = nil // a staged transfer from an older epoch is void
	}
	if c.Total > pbSnapshotMaxBytes || c.Offset > c.Total || uint64(len(c.Data)) > c.Total-c.Offset {
		e.snapStage = nil
		e.mu.Unlock()
		return nack
	}

	// (b) Stream identity. A chunk that does not continue the staged transfer
	// either STARTS a new one (offset 0) or is garbage. Never merge two streams.
	st := e.snapStage
	if st == nil || st.epoch != c.Epoch || st.frontierSeq != c.FrontierSeq ||
		st.frontierEpoch != c.FrontierEpoch || st.total != c.Total {
		if c.Offset != 0 {
			e.snapStage = nil
			e.mu.Unlock()
			return nack
		}
		st = &snapshotStage{
			epoch:         c.Epoch,
			frontierSeq:   c.FrontierSeq,
			frontierEpoch: c.FrontierEpoch,
			total:         c.Total,
		}
		e.snapStage = st
	}
	// (c) Strict contiguity. The declared Total is used ONLY to validate, never to
	// size a reservation — a hostile Total can therefore cost at most the bytes
	// actually delivered.
	if c.Offset != uint64(len(st.buf)) {
		e.snapStage = nil
		e.mu.Unlock()
		return nack
	}
	st.buf = append(st.buf, c.Data...)
	if !c.Final {
		e.mu.Unlock()
		return AckMsg{Epoch: c.Epoch, Seq: c.FrontierSeq, OK: true}
	}
	if uint64(len(st.buf)) != c.Total {
		e.snapStage = nil
		e.mu.Unlock()
		return nack
	}
	blob := st.buf
	e.snapStage = nil
	e.mu.Unlock()

	if err := e.installSnapshot(c.Epoch, c.FrontierSeq, c.FrontierEpoch, blob); err != nil {
		return nack
	}
	return AckMsg{Epoch: c.Epoch, Seq: c.FrontierSeq, OK: true}
}

// installSnapshot performs the ATOMIC-ENOUGH install: it makes the FSM, the
// in-memory watermark and the DURABLE watermark agree, and makes every state in
// between UNUSABLE rather than merely stale.
//
// THE ATOMICITY ARGUMENT, in the three parts the obligation names.
//
//	FSM + IN-MEMORY WATERMARK are genuinely atomic: both happen inside ONE
//	writeMu+e.mu critical section, which excludes both Applier.Apply sites, so no
//	observer can see one without the other. Every reader of the frontier
//	(CatchupInfo, receiveLocked's log matching, appliedFrontierLocked) takes e.mu.
//
//	DURABLE WATERMARK cannot be atomic with the FSM — restoreSnapshot is
//	delete-then-put across a wipe and a replay, and no single durable store spans
//	it. So it is made FAIL-SAFE instead of atomic, by the poison fence: the
//	marker goes down durably BEFORE the wipe and comes up only AFTER both the
//	install and the durable frontier write have succeeded. Every intermediate
//	state — mid-wipe, mid-replay, installed-but-unstamped — is therefore a state
//	in which the marker is set, and a node that boots or aborts with the marker
//	set refuses to serve, reports CatchupInfo{OK:false} (so the failover gate
//	never promotes it), and can only be re-snapshotted.
//
// WHY lastSeq IS FORCED DOWN. appliedFrontierLocked is max(lastSeq, lastApplied).
// A DIVERGED target is typically an ex-primary with lastSeq far ABOVE the seq we
// are installing; leaving it would make the node report a frontier its wiped FSM
// does not hold — an OVER-report, the one direction that lets log matching
// certify a divergent append. Both watermarks are therefore SET (not maxed) to
// the installed identity. This is the one place a seq watermark legitimately
// REGRESSES, and it is legitimate precisely because the state under it was
// discarded rather than kept.
func (e *Engine) installSnapshot(shipEpoch, fseq, fepoch uint64, blob []byte) error {
	ss := e.snap
	if ss == nil {
		return ErrGrowNoSnapshotStore
	}

	// (1) Raise the durable poison fence BEFORE anything is touched, and OFF the
	// engine locks (it is an fsync). A fence raised for an install that then never
	// starts is conservative-safe: the node is poisoned, refuses to serve, and is
	// re-snapshotable — which is the designed recovery path anyway. It is cleared
	// again below on the pre-install abort branch, where the FSM is provably
	// untouched.
	if err := ss.BeginInstall(fseq, fepoch); err != nil {
		return fmt.Errorf("pbisr: snapshot fence: %w", err)
	}

	e.writeMu.Lock()
	e.mu.Lock()
	if e.closed || shipEpoch < e.epoch {
		// A newer epoch arrived between the fence and the lock. NOTHING has been
		// installed yet, so the fence can be safely lowered again.
		e.mu.Unlock()
		e.writeMu.Unlock()
		_ = ss.AbortInstall() //nolint:errcheck,gosec // a fence we fail to lower just re-poisons: safe
		return ErrGrowEpochChanged
	}
	// From here on the FSM is being mutated. Latch POISONED in memory immediately:
	// it is what protects a heap-mode / no-DataDir shard, where the durable fence
	// is necessarily a no-op but a half-wiped in-memory FSM is just as unusable.
	e.poisoned = true

	if err := ss.InstallFSM(blob); err != nil {
		e.mu.Unlock()
		e.writeMu.Unlock()
		return fmt.Errorf("pbisr: snapshot install: %w", err)
	}

	// The FSM now IS the source's FSM at (fseq, fepoch). Make every watermark say
	// exactly that, in the same critical section.
	e.lastApplied, e.lastAppliedEpoch = fseq, fepoch
	e.lastSeq, e.lastSeqEpoch = fseq, fepoch
	e.committed = fseq
	// Any in-flight record is void: its write is either inside the installed state
	// or was discarded by the wipe. Either way it can never full-ISR-commit under
	// this identity, so resolve it as unknown-outcome rather than stall the sweep
	// behind it (the same honesty rule as the epoch flush).
	for i := 0; i < e.inflightN; i++ {
		e.resolveLocked(e.inflightRing[(e.inflightHead+i)%len(e.inflightRing)], ErrReplicationTimeout)
	}
	for e.inflightN > 0 {
		e.popInflightHeadLocked()
	}
	// The catch-up ring described the DISCARDED history; retaining it would let a
	// later grow replay entries this node no longer holds.
	e.backlog = newRing(len(e.backlog.buf))
	e.clearEffISRLocked()
	e.clearLearnersLocked()
	if e.windowCond != nil {
		e.windowCond.Broadcast() // committed moved: admission may proceed
	}

	// (2) Durable frontier FIRST, marker cleared SECOND — never the reverse. A
	// crash between them re-poisons a node whose state is actually fine (costing
	// one more transfer); a crash in the reverse order would un-poison a node
	// carrying a PRE-install watermark over POST-install state, which for a
	// diverged target is an OVER-report.
	if err := ss.CommitInstall(fseq, fepoch); err != nil {
		e.mu.Unlock()
		e.writeMu.Unlock()
		return fmt.Errorf("pbisr: snapshot commit: %w", err)
	}
	e.poisoned = false
	e.mu.Unlock()
	e.writeMu.Unlock()

	e.snapInstalled.Add(1)
	return nil
}
