// SPDX-License-Identifier: Apache-2.0

// Package pbisr is the transport-agnostic primary-backup / ISR replication
// engine for the Rostam data plane (Option B — see
// shard/pbisr/DESIGN.md). It is the correctness-critical protocol core:
// primary write path (lease fence, assign seq, apply locally, ship to ISR,
// commit only when the FULL current ISR acks the exact (epoch,seq)) and backup
// receive path (epoch fence, gap check, apply, ack).
//
// The package is deliberately self-contained: it does NOT import shard or
// cluster (to avoid import cycles). Everything it needs from the control plane
// and local state arrives through the injected Control, Transport, and Applier
// interfaces. The real Control delegates to the MetaRaft-authoritative
// cluster.MetaFSM; tests inject fakes.
package pbisr

import (
	"context"
	"errors"
	"sync"
	"sync/atomic"
	"time"
)

// defaultRingCapacity bounds the in-memory catch-up backlog per engine.
const defaultRingCapacity = 4096

// monoStart anchors the process-wide monotonic clock. monotonicNow reads the
// monotonic component of the wall clock (time.Since uses the monotonic reading)
// and is the ONLY place the engine touches the real clock. The correctness core
// never calls time.Now directly; it calls e.now() so tests inject a controllable
// clock and the lease decision path stays deterministic (DESIGN: OH1 lease).
var monoStart = time.Now()

// monotonicNow returns monotonic nanoseconds since process start. It is the
// default clock for a real Engine; tests inject their own via WithClock.
func monotonicNow() int64 { return int64(time.Since(monoStart)) }

// Sentinel errors returned by Propose.
var (
	// ErrNotPrimary — this node is not the control plane's named primary for the
	// shard. The write is rejected before any state changes. (Loss of the epoch
	// lease is reported as ErrLeaseExpired instead — see below.)
	ErrNotPrimary = errors.New("pbisr: not primary for shard (fenced)")

	// ErrLeaseExpired — this node holds no currently-valid MetaRaft lease for its
	// adopted epoch: the lease is absent, is for a different epoch, or has expired
	// by the engine's own monotonic clock. A partitioned primary that cannot renew
	// its lease self-fences here WITHOUT observing any new control-plane state —
	// this is the OH1 write-path defense (see DESIGN "Open hazards / OH1").
	ErrLeaseExpired = errors.New("pbisr: primary lease absent or expired (fenced)")

	// ErrBelowMinISR — the in-sync set is smaller than the durability floor.
	// The primary chooses unavailability and NEVER acks below the floor (H3).
	ErrBelowMinISR = errors.New("pbisr: ISR below min-ISR floor")

	// ErrReplicationTimeout — the write was applied locally but the full current
	// ISR did not ack this exact (epoch,seq) before the context deadline (or a
	// reachable member rejected/failed the write). The caller treats this as a
	// failed, non-durable write.
	ErrReplicationTimeout = errors.New("pbisr: replication did not reach full ISR before deadline")

	// ErrPBUnimplemented — a replicator method whose full behavior arrives in a
	// later phase (control-plane-driven membership/leadership changes). The seam
	// contract in shard/replicator.go permits an explicit unimplemented error.
	ErrPBUnimplemented = errors.New("pbisr: operation not implemented in this phase")

	// ErrPipelineStalled — the pipeline window is full of in-flight writes that
	// cannot commit because a backup stopped acking below full ISR (hole class
	// (ii) in PIPELINE-REDESIGN §3). Admission of new writes is refused until an
	// ISR shrink or catch-up unwedges the shard. Mapped at the seam like a
	// timeout (unknown outcome). Not yet returned on this path.
	ErrPipelineStalled = errors.New("pbisr: pipeline stalled below full ISR")
)

// Control reads the MetaRaft-authoritative control plane for one shard. The
// real implementation delegates to cluster.MetaFSM (ShardEpoch / ShardPrimary
// / ShardISR + the min-ISR floor); tests inject a fake.
type Control interface {
	Epoch(shard int) uint64
	Primary(shard int) string
	ISR(shard int) []string
	MinISR(shard int) int
}

// Transport ships one write to one backup. The backup side is an Engine.Receive
// registered by peer ID inside the transport (see inmem_transport.go for the
// in-memory wiring used by tests).
//
// It is an ASYNCHRONOUS, ordered-submit + completion-callback contract (the old
// blocking Replicate could not express pipelined ordered submission — see
// PIPELINE-REDESIGN Q6):
//
//   - Calls from a SINGLE goroutine to a SINGLE peer are submitted in call order.
//     The engine's per-peer sender is that single goroutine, and per-shard seq
//     order on the wire (P1) depends on this.
//   - done is invoked EXACTLY ONCE — with the backup's ack, or a transport error
//     — from the transport's delivery goroutine, and must NOT block (the engine's
//     callback only takes e.mu briefly).
//   - Replicate itself MAY block for backpressure (a full send queue) but must
//     NOT block on the remote round trip; the engine owns the deadline (Propose
//     ctx), so the transport carries none.
//   - On a submission failure Replicate returns the error and does NOT invoke
//     done (the returned error IS the completion). Otherwise it returns nil and
//     done fires later, exactly once.
type Transport interface {
	Replicate(peer string, msg ReplicateMsg, done func(AckMsg, error)) error
}

// GroupTransport is an OPTIONAL Transport capability: ship k>=2
// ordered, uniform-epoch, seq-dense writes to one backup as ONE wire frame,
// answered by ONE cumulative ack. Submission rules are identical to Replicate
// (single-caller per-peer ordering; done invoked exactly once, non-blocking;
// a submit error IS the completion). msgs is only valid for the duration of
// the call — implementations must not retain it after returning (the sender
// reuses the backing array).
//
// The cumulative AckMsg contract (see Engine.ReceiveGroup): OK==true with
// Seq==lastSeq means every record applied; OK==false with Seq==s means the
// prefix through s applied and everything after s did not.
type GroupTransport interface {
	ReplicateGroup(peer string, msgs []ReplicateMsg, done func(AckMsg, error)) error
}

// InlineTransport is an OPTIONAL Transport capability (BENCHMARK.md "epoll
// inline-dispatch ceiling", lever 1): attempt a NON-BLOCKING, NO-DIAL submit
// of one write from the caller's own goroutine, skipping the per-peer sender
// hop (one scheduler wakeup per write) when the peer link is connected and
// uncontended. Returns true iff the write was submitted — done then fires
// exactly once, same contract as Replicate. Returns false (nothing submitted,
// done will NOT fire) when the link is down, its send queue is full, or the
// implementation cannot guarantee a non-blocking submit; the caller falls
// back to the ordered sender-goroutine path.
//
// ORDERING: callers must only invoke this when no submission for the same
// peer is queued or in flight on the sender path (the engine enforces this
// with a per-peer pending count under writeMu), so per-shard seq order on the
// wire (P1) is preserved.
type InlineTransport interface {
	TryReplicate(peer string, msg ReplicateMsg, done func(AckMsg, error)) bool
}

// pbGroupBatchMax caps how many queued writes the per-peer sender drains into
// one group frame. Grouping NEVER waits (no linger): it only engages when the
// peer queue already holds >1 message — exactly the regime where per-message
// framing/ack/lock costs dominate.
const pbGroupBatchMax = 64

// pbGroupBytesMax caps a group's cumulative record-data bytes so a group of
// large values can never approach the receiver's pbMaxPayload frame bound —
// exceeding it would make the backup drop the conn (errPBOversize) and fail
// the whole group, a liveness cliff for large-value workloads. A write bigger
// than this on its own still ships fine: it just rides alone (the single
// message path has no group overhead to amortize anyway).
const pbGroupBytesMax = 1 << 20

// Applier applies a committed write to local state and returns the op result.
// On the primary it is the authoritative in-memory FSM apply; on a backup it
// materializes the replicated write.
type Applier interface {
	Apply(data []byte) ([]byte, error)
}

// ringEntry is one retained write in the catch-up backlog.
//
// prevEpoch is the epoch of THIS entry's predecessor (the write at seq-1),
// captured at append time from the primary's own applied frontier. It is stored
// per entry — rather than derived at replay time from the previous ring entry —
// because a replayed delta's FIRST element's predecessor may have already been
// evicted, and because a wire-visible chain link must never depend on how much
// backlog a node happens to still retain. See ringDeltaLocked.
type ringEntry struct {
	epoch     uint64
	seq       uint64
	prevEpoch uint64
	data      []byte
}

// ring is a fixed-capacity in-memory backlog of the most recent writes. On
// overflow it drops the oldest entry (catch-up falls back to a snapshot later).
// Not safe for concurrent use; callers hold the engine mutex.
type ring struct {
	buf   []ringEntry
	head  int // index of the oldest entry
	count int
}

func newRing(capacity int) *ring {
	if capacity < 1 {
		capacity = 1
	}
	return &ring{buf: make([]ringEntry, capacity)}
}

// append retains e, returning the DATA of the entry it evicted (nil when the
// ring was not yet full). The evicted buffer's ownership passes back to the
// caller — the hook behind WithDataRelease's buffer recycling.
func (r *ring) append(e ringEntry) (evicted []byte) {
	if r.count < len(r.buf) {
		r.buf[(r.head+r.count)%len(r.buf)] = e
		r.count++
		return nil
	}
	// Full: overwrite the oldest and advance head (drop-oldest).
	evicted = r.buf[r.head].data
	r.buf[r.head] = e
	r.head = (r.head + 1) % len(r.buf)
	return evicted
}

func (r *ring) len() int { return r.count }

// reset drops every retained entry, handing each one's data back through
// release (the WithDataRelease recycling hook) so a reset does not leak the
// buffers append would otherwise have returned on eviction. release may be nil.
//
// Used by Promote: the entries belong to an epoch this engine no longer serves,
// and retaining them across a promotion is what creates the hole documented on
// at(). Callers hold the engine mutex.
func (r *ring) reset(release func([]byte)) {
	for i := 0; i < r.count; i++ {
		idx := (r.head + i) % len(r.buf)
		if release != nil && r.buf[idx].data != nil {
			release(r.buf[idx].data)
		}
		r.buf[idx] = ringEntry{}
	}
	r.head = 0
	r.count = 0
}

// span reports the oldest and newest seq currently retained in the ring, and
// whether it holds anything. [oldest, newest] BOUNDS the replayable seqs but,
// since Promote can leave a hole (see at), does not by itself prove every seq
// inside it is present. Callers hold the engine mutex.
func (r *ring) span() (oldest, newest uint64, ok bool) {
	if r.count == 0 {
		return 0, 0, false
	}
	oldest = r.buf[r.head].seq
	newest = r.buf[(r.head+r.count-1)%len(r.buf)].seq
	return oldest, newest, true
}

// at returns the retained entry for seq, or ok=false if that seq is not in the
// ring (evicted, never appended, or skipped by a hole). Callers hold the engine
// mutex.
//
// The seq re-check is load-bearing, not defensive. Indexing off `oldest`
// assumes the retained seqs are DENSE, and that assumption is false after a
// re-promotion: Promote resets lastSeq to lastApplied but the ring keeps the
// node's older primary-era entries, while writes it took as a BACKUP never
// append here at all (proposeSequenced is the only append site). Proposals then
// resume above the received range, so the ring holds [1..5, 9..11] with 6-8
// absent while span() still reports [1..11] as if dense.
//
// Without this check the offset arithmetic walks off the end of the live
// prefix and returns a ZERO-VALUED entry with ok=true. That is worse than a
// miss: a zero entry has prevEpoch == 0, so checkCatchupDivergenceLocked reads
// it as a chain link that fails to match and permanently rejects a replica
// holding a perfectly clean prefix — the exact inverse of the bug log matching
// exists to catch.
//
// Returning false instead routes those seqs down the ring-cold path, which is
// what a hole actually means: this engine cannot serve that delta and the
// caller must fall back to a snapshot.
func (r *ring) at(seq uint64) (ringEntry, bool) {
	oldest, newest, ok := r.span()
	if !ok || seq < oldest || seq > newest {
		return ringEntry{}, false
	}
	ent := r.buf[(r.head+int(seq-oldest))%len(r.buf)]
	if ent.seq != seq {
		return ringEntry{}, false
	}
	return ent, true
}

// Option customizes an Engine at construction. It exists mainly so tests can
// inject a controllable clock (WithClock); production callers use the defaults.
type Option func(*Engine)

// WithClock overrides the engine's monotonic clock source. The supplied function
// must return monotonic nanoseconds. The core lease decision path calls it in
// place of the real clock so tests control time deterministically. A nil now is
// ignored (the default monotonic clock is kept).
// WithDataRelease installs a release hook for write payloads the engine has
// finished retaining: today that is exactly the catch-up ring's drop-oldest
// eviction. The caller (the shard seam) uses it to recycle its encode
// buffers. SAFETY CONTRACT: an evicted buffer may still be aliased by frames
// parked in a DEAD peer's sender channel. A dead peer stalls the window (no
// commit, no ring wrap, no eviction) until it is removed. ISR shrink is the
// path that removes it, and it closes this aliasing window at the source:
// dropPeerLocked latches the removed peer's sender into DISCARD mode, so every
// parked frame is drained WITHOUT being submitted to the transport — the ring
// may then safely wrap and recycle those buffers. (The complementary GROW
// re-add path is a separate, deferred increment; it re-adds a member via a
// FRESH sender, so it likewise cannot resurrect a stale parked frame.)
func WithDataRelease(release func([]byte)) Option {
	return func(e *Engine) { e.releaseData = release }
}

func WithClock(now func() int64) Option {
	return func(e *Engine) {
		if now != nil {
			e.now = now
		}
	}
}

// WithFrontierSink installs a callback invoked whenever this engine's applied
// frontier advances (the durable frontier). It receives the SAME (seq,
// epoch) pair AppliedFrontier would return at that instant, and it is called
// while holding e.mu, immediately after the write is materialized — so a value it
// receives is always covered by the local FSM. It must not block or re-enter the
// engine. A nil sink is ignored.
func WithFrontierSink(sink func(seq, epoch uint64)) Option {
	return func(e *Engine) {
		if sink != nil {
			e.frontierSink = sink
		}
	}
}

// WithRestoredFrontier seeds a freshly constructed engine with a frontier read
// back from durable storage. It is how a RESTARTED PB node stops
// presenting a genesis (0,0) identity over a warm-restarted FSM.
//
// It seeds lastApplied/lastAppliedEpoch, NOT lastSeq. That is sufficient and it is
// the honest place to put it: a new Engine has lastSeq == 0, so
// appliedFrontierLocked's max() returns exactly (seq, epoch) — and lastApplied is
// the "materialized, not necessarily proposed by me" watermark, which is precisely
// what a restored value is. Restoring into lastSeq instead would additionally
// claim the node ASSIGNED those seqs as primary, which it may never have done.
//
// It deliberately does NOT restore e.epoch (the leasing watermark) or e.committed.
// The epoch watermark is control-plane state that MetaRaft holds durably and
// re-supplies via GrantLease/AdoptEpoch; seeding it here from a write's epoch
// would let a restored node out-rank its own control plane and refuse to take a
// legitimate lease. committed is left at 0 for the same reason it is safe to:
// under-reporting durability costs catch-up work, never correctness.
//
// A (0, 0) restore is a no-op, which is what makes this safe to call
// unconditionally at construction.
func WithRestoredFrontier(seq, epoch uint64) Option {
	return func(e *Engine) {
		if seq == 0 {
			return
		}
		e.lastApplied = seq
		e.lastAppliedEpoch = epoch
	}
}

// CommitLevel selects a shard's durability contract — how many replicas a write
// must reach before Propose acks the client. It is the direct analogue of
// Aerospike's commitLevel and Kafka's acks.
type CommitLevel uint8

const (
	// CommitFullISR (default, zero value) commits only when EVERY member of the
	// write's propose-time ISR has acked the exact (epoch,seq) in memory (P6).
	// No acked write is lost while any one ISR member survives. This is the
	// contract every other property in DESIGN.md is written against.
	CommitFullISR CommitLevel = iota

	// CommitPrimary commits as soon as the write is applied locally on the
	// primary (behind the same OH1 lease fence), shipping to backups
	// asynchronously in the background. It trades durability for latency: a
	// write the client saw as committed CAN be lost if the primary dies before
	// any backup received it. Opt-in only — the operator is choosing Aerospike's
	// commit-master / async-replication posture. Throughput is unaffected on a
	// pipelined, server-underutilized path (the full-ISR ack waits already
	// overlap); the win is per-write latency (one fewer round trip).
	CommitPrimary
)

// WithCommitLevel sets the engine's durability contract (default CommitFullISR
// — byte-identical to the pre-knob behavior).
func WithCommitLevel(level CommitLevel) Option {
	return func(e *Engine) { e.commitLevel = level }
}

// Engine is the per-shard primary-backup / ISR replication core for one node.
type Engine struct {
	nodeID string
	shard  int
	ctrl   Control
	tr     Transport
	ap     Applier

	// now returns monotonic nanoseconds. The lease fence reads it INSTEAD of
	// time.Now so the decision is deterministic and test-controllable.
	now func() int64

	// writeMu serializes the entire primary write path so that seq assignment,
	// local apply, and per-backup dispatch all happen in a total per-shard
	// order (DESIGN: the primary serializes writes). Backups therefore observe
	// each shard's writes gap-free and in seq order.
	writeMu sync.Mutex

	// mu guards the fields below (epoch, lease, lastSeq, lastApplied, backlog).
	// It is also the per-shard serialization point for the backup Receive path.
	mu          sync.Mutex
	epoch       uint64 // adopted/leased epoch watermark (fencing)
	leaseEpoch  uint64 // epoch this node holds a MetaRaft primary lease for
	leaseExpiry int64  // monotonic-ns deadline after which the lease is void
	lastSeq     uint64 // highest seq assigned as primary (assigned, NOT necessarily durable)
	committed   uint64 // highest seq that reached the full ISR (durable)

	// lastSeqEpoch / lastAppliedEpoch are the EPOCH halves of the two seq
	// watermarks — the log-matching identity. A seq alone is a position,
	// not a history: Promote resets the counter to the promoted node's applied
	// high-water, so seq N under epoch 2 can be a completely different write from
	// seq N under epoch 1. Pairing each watermark with the epoch of the write that
	// set it is what lets receiveLocked tell "you are my prefix" from "we forked".
	//
	//   - lastSeqEpoch is the epoch under which lastSeq was ASSIGNED as primary
	//     (set in registerInflightLocked). Promote INHERITS lastAppliedEpoch rather
	//     than stamping the new epoch: promotion does not rewrite the write already
	//     sitting at that seq, it only continues past it.
	//   - lastAppliedEpoch is the epoch of the write at lastApplied (set in
	//     receiveLocked, the only place lastApplied moves).
	//
	// Both are guarded by e.mu like their seq counterparts.
	lastApplied      uint64 // highest seq applied as backup
	lastSeqEpoch     uint64 // epoch of the write at lastSeq
	lastAppliedEpoch uint64 // epoch of the write at lastApplied
	// peerAcked is the per-backup replication high-water: the highest (epoch,seq)
	// each backup has H6-exactly acked to THIS primary. It is observability-only
	// (ReplicationStatus reads it for the /v1/replication lag metric) and NEVER
	// gates commit — the full-ISR sweep is the sole commit authority. Updated only
	// in the credit branch of completeSend/completeGroupSend, which already hold
	// e.mu, so it adds one map store per credited ack on the already-locked
	// completion path (never on the network-bound send path). Lazily allocated on
	// the first credit; a pure backup / never-shipped shard leaves it nil.
	peerAcked   map[string]uint64
	backlog     *ring
	releaseData func([]byte) // optional WithDataRelease hook (ring-evicted buffers)
	commitLevel CommitLevel  // durability contract; CommitFullISR (0) = default

	// frontierSink (optional; WithFrontierSink) is notified every time the applied
	// frontier advances — i.e. every time a write finishes materializing into the
	// local FSM, in EITHER role. The durable frontier uses it to persist into the
	// cache header so a restarted node can tell the truth about what it holds.
	//
	// CALLED UNDER e.mu, on the write path. The implementation MUST be cheap and
	// non-blocking (record-and-return); the durable stamp it eventually triggers is
	// the caller's problem to amortise and must happen off this goroutine. See
	// shard.pbFrontierStamper.
	frontierSink func(seq, epoch uint64)

	// effISR is the durable per-epoch EFFECTIVE-ISR override installed by
	// ShrinkISR. It is live iff effISREpoch == e.epoch; on any
	// epoch advance it is cleared (clearEffISRLocked) so it can NEVER outlive the
	// epoch it was decided for. While live it narrows the required-peer set of
	// BOTH already-registered in-flight records (ShrinkISR's narrowing pass) AND
	// records registered afterward (the proposeSequenced intersect) — the two
	// halves that together close the stale-read race (a Propose that snapshotted a
	// pre-shrink ISR must still drop the removed member). It is MONOTONE-NARROWING
	// only: shrink removes members, never adds. Grow (widening) is a separate,
	// deferred increment and must NOT reuse this field to widen in-flight records
	// (that asymmetry is load-bearing for the no-acked-loss proof). Guarded by e.mu.
	effISR      []string
	effISREpoch uint64

	// learners is the ISR GROW "learner ship-set": catching-up members that a
	// grow is streaming toward but that are NOT yet committed ISR members and do
	// NOT gate commit. proposeSequenced ships each write to peers ∪ learners, yet
	// registers the in-flight record's required set as `peers` ONLY — a learner is
	// NEVER added to an in-flight record's pending set (the no-widen-in-flight
	// rule; widening a live record's required set is UNSOUND, mirroring shrink's
	// load-bearing narrow-only asymmetry). A member is promoted from learner to
	// required only for FUTURE writes, once the driver's OpSetShardISR(E, S∪M)
	// commits and proposeSequenced observes it in ctrl.ISR. Guarded by e.mu; like
	// effISR it is epoch-scoped and cleared on every epoch advance
	// (clearLearnersLocked) so a grow decided for one epoch can never leak into
	// another. The map value is always true (membership set).
	learners map[string]bool

	// abandonedLearners records, per peer, the seq at which submitLearnerLocked
	// ABANDONED a grow (its non-blocking learner channel filled — a hopeless grow).
	// It is the DURABLE signal the grow driver reads (LearnerAbandonedAt) instead of
	// inferring from IsLearner (which cannot distinguish "abandoned" from
	// "transitioned to voter" from "never started"): a peer the grow committed into
	// the ISR while it is marked here is a GAPPED voter the driver must compensate
	// out (see cluster/pb_grow.go's abandon coordination). Set on abandon; CLEARED
	// only by a fresh flipLearner (a new grow start), so the signal survives across
	// driver ticks until a fresh grow supersedes it. Guarded by e.mu.
	abandonedLearners map[string]uint64

	// learnerTeardown parks learners that clearLearnersLocked removed from the
	// ship-set on an epoch advance: that path holds ONLY e.mu, so it cannot drop the
	// learner's sender goroutine (dropPeerLocked needs writeMu too). The grow driver
	// calls ReclaimOrphanLearners each tick to drop these senders under both locks,
	// so an epoch-cleared learner's goroutine is reclaimed promptly rather than
	// leaking until Shutdown. Guarded by e.mu (writes) / writeMu+e.mu (the reclaim).
	learnerTeardown map[string]bool

	// peerFailures counts CONSECUTIVE replication failures per peer (ISR
	// shrink wedge signal): incremented on every non-OK/errored completion in
	// completeSend/completeGroupSend, reset to 0 on the peer's next OK ack. The
	// primary-side shrink driver reads it via StalledPeers(threshold) to decide a
	// member is dead enough to request removal. Observability/decision-only — it
	// never gates a commit. Lazily allocated. Guarded by e.mu.
	peerFailures map[string]int

	// In-flight pipeline machinery. Guarded by e.mu; wired into Propose
	// later. Records live in a seq-dense ring FIFO: the physical index of the
	// record for seq s is (inflightHead + (s - baseSeq)) % cap, giving O(1) ack
	// routing (seqs are dense, so s - baseSeq is the record's FIFO offset). The
	// commit watermark advances only from the head (PIPELINE-REDESIGN §"In-flight
	// record and queue"). windowCond wakes admission waiters when committed
	// advances (a freed window slot) or a record resolves. See pb_pipeline.go.
	inflightRing []*inflight
	inflightHead int        // physical index of the oldest unpopped record
	inflightN    int        // number of unpopped records
	baseSeq      uint64     // seq of the record at inflightHead (valid iff inflightN > 0)
	windowCond   *sync.Cond // L == &e.mu; signalled on window-slot changes

	// SNAPSHOT TRANSFER state.
	//
	// snap is the optional FSM serialize/install/fence capability
	// (WithSnapshotStore); nil restores exact pre-4.3 behavior (a ring-cold or
	// diverged target aborts the grow and is never repaired).
	//
	// snapStage is the RECEIVE-side chunk accumulation buffer — pure memory, so an
	// aborted transfer leaves the FSM pristine. Guarded by e.mu.
	//
	// poisoned latches "a snapshot install began and did not provably complete".
	// It is seeded at construction from the DURABLE fence (so a crash mid-install
	// comes back refusing to serve) and set again in memory across every install,
	// which is what protects a heap-mode shard whose durable fence is a no-op.
	// While set, this engine refuses to propose, refuses to receive, and reports
	// CatchupInfo{OK:false} so the failover gate never promotes it. Guarded by e.mu.
	snap      SnapshotStore
	snapStage *snapshotStage
	poisoned  bool

	// Snapshot metrics (lock-free). StallLast/Max measure the QUIESCED
	// serialization — the write-path freeze that is the named cost of this stage's
	// flow-control choice. See takeSnapshot.
	snapTaken       atomic.Uint64
	snapInstalled   atomic.Uint64
	snapStallLastNs atomic.Int64
	snapStallMaxNs  atomic.Int64

	// Per-peer ordered senders. peerQ maps a peer to its sender state:
	// a buffered, seq-ordered frame channel drained by ONE goroutine per peer
	// that submits to the Transport in submission (== seq) order — the ordering
	// backbone behind per-shard seq order on the wire (P1) — plus the pending
	// count that gates the inline fast path (see submitPeerLocked). Senders are
	// created lazily during the sequencing critical section and torn down by
	// Shutdown. Guarded by writeMu (NOT e.mu): the enqueue happens under
	// writeMu, so a full channel would block only that path — never a sender's
	// ack callback (which takes e.mu, not writeMu) — so no lock cycle exists.
	// closed latches shutdown so no sender is created after teardown.
	peerQ    map[string]*peerSender
	senderWG sync.WaitGroup
	closed   bool
}

// peerSender is one peer's ordered submission state. pending counts messages
// handed to ch whose transport submission has not yet RETURNED — incremented
// under writeMu before the channel send, decremented by the sender goroutine
// only after the Transport call returns (the link-append ordering point).
// The inline fast path (submitPeerLocked) may bypass the channel ONLY when
// pending == 0: the channel is then provably empty AND the sender goroutine is
// provably between submissions, so an inline link-append cannot overtake or be
// overtaken by a sender-path submission for this peer (P1-safe; both writers
// of new submissions are serialized by writeMu).
type peerSender struct {
	ch      chan ReplicateMsg
	pending atomic.Int64
	// discard latches this sender into DROP mode (ISR shrink). Set by
	// dropPeerLocked when the peer is removed from the ISR: the sender then
	// drains every remaining parked frame WITHOUT submitting it to the Transport
	// and exits when the channel closes. This is the fix for the WithDataRelease
	// aliasing hazard — once a shrink un-stalls the window the catch-up ring can
	// wrap and recycle a payload buffer that is still aliased by a frame parked
	// in a removed dead peer's channel; discarding those frames unsubmitted means
	// no recycled buffer is ever handed to the transport. Read/written with
	// atomics because the sender goroutine reads it without holding writeMu.
	discard atomic.Bool
}

// New constructs an Engine with the default catch-up ring capacity.
func New(nodeID string, shard int, ctrl Control, tr Transport, ap Applier, opts ...Option) *Engine {
	return NewWithRingCapacity(nodeID, shard, ctrl, tr, ap, defaultRingCapacity, opts...)
}

// NewWithRingCapacity constructs an Engine with an explicit catch-up ring
// capacity (the backlog is bounded; on overflow the oldest write is dropped).
func NewWithRingCapacity(nodeID string, shard int, ctrl Control, tr Transport, ap Applier, ringCapacity int, opts ...Option) *Engine {
	// Hardening: the catch-up ring MUST be able to hold at least a full
	// pipeline window, so every in-flight (uncommitted) write is always replayable
	// from the backlog — the grow flip enqueues up to pipelineWindow ring entries
	// onto a re-added member's fresh sender, and shrink/OH2 both rely on the same
	// invariant. A capacity below the window would silently evict a still-in-flight
	// write, so fail LOUD at construction rather than corrupt catch-up later. The
	// default path already holds this via the compile-time assert in pb_pipeline.go.
	if ringCapacity < pipelineWindow {
		panic("pbisr: ringCapacity < pipelineWindow")
	}
	e := &Engine{
		nodeID:  nodeID,
		shard:   shard,
		ctrl:    ctrl,
		tr:      tr,
		ap:      ap,
		now:     monotonicNow,
		backlog: newRing(ringCapacity),
	}
	e.windowCond = sync.NewCond(&e.mu)
	for _, o := range opts {
		o(e)
	}
	return e
}

// AdoptEpoch advances this engine's cached epoch watermark to epoch if it is
// higher (monotonic). It models MetaRaft leasing a leadership generation to a
// new primary, and is a no-op for a stale or equal epoch. Backups also adopt
// higher epochs implicitly through Receive.
func (e *Engine) AdoptEpoch(epoch uint64) {
	e.mu.Lock()
	if epoch > e.epoch {
		e.epoch = epoch
		// Epoch advanced: a shrink override decided for the OLD epoch is void —
		// clear it so it can never narrow a new epoch's writes.
		e.clearEffISRLocked()
		// ... and every in-progress GROW learner is epoch-scoped too: a learner
		// caught up under the OLD epoch must not stay in the ship-set for the new
		// one (ISR grow). This stops shipping to it; its orphaned sender is
		// reclaimed by the grow driver's abort or by Shutdown.
		e.clearLearnersLocked()
		// Epoch advanced: in-flight records of a superseded epoch can never
		// commit under the lease fence (Q5), so fail them promptly instead of
		// stalling the sweep behind them (PIPELINE-REDESIGN §3, epoch flush).
		e.flushEpochLocked(e.epoch)
	}
	e.mu.Unlock()
}

// GrantLease models MetaRaft granting (or renewing) a primary lease to this node
// for epoch, valid until expiryMonoNs on the engine's monotonic clock. It adopts
// epoch as the engine's watermark if higher (monotonic). A grant for a stale
// epoch (below the currently held lease epoch) is a no-op; a grant for the same
// epoch is a renewal that extends the expiry. This is the ONLY thing that lets a
// primary pass the Propose lease fence: a partitioned primary that stops
// receiving renewals self-fences the moment the injected clock passes expiry.
func (e *Engine) GrantLease(epoch uint64, expiryMonoNs int64) {
	e.mu.Lock()
	defer e.mu.Unlock()
	if epoch < e.leaseEpoch {
		return // stale grant: never regress the lease epoch.
	}
	e.leaseEpoch = epoch
	e.leaseExpiry = expiryMonoNs
	if epoch > e.epoch {
		e.epoch = epoch
		// New epoch: void any prior-epoch shrink override. A same-epoch
		// renewal (the else branch) keeps effISR — the override is still current.
		e.clearEffISRLocked()
		e.clearLearnersLocked() // and abandon any prior-epoch grow learners
		// Same epoch-flush as AdoptEpoch: a same-node re-election (new epoch, same
		// primary) starts a clean pipeline; stale-epoch in-flight records fail (Q5).
		e.flushEpochLocked(e.epoch)
	}
}

// Promote turns a BACKUP into this shard's writable PRIMARY for newEpoch (Plan
// 4b failover). It continues seq assignment from the durable high-water this
// node applied as a backup (lastSeq = lastApplied), marks that high-water
// committed (the node is now the source of truth), and grants itself the lease
// for newEpoch until expiryMonoNs. Called by the control-plane failover driver
// AFTER MetaRaft has committed newEpoch naming this node primary, and (OH1)
// after the previous epoch's lease has provably lapsed. A stale/equal newEpoch
// (<= the adopted watermark) is a no-op — never regress a live primary's state.
//
// Lossless-failover argument: a client-acked write reached the FULL ISR before
// the ack (P6), so every ISR member — including this promotion target, which
// MetaRaft picks from the ISR — applied it. It is therefore <= lastApplied and
// preserved by continuing past it. Writes this backup applied that were never
// client-acked (the old primary died mid-flight) simply survive under the new
// epoch, which is safe: the client never observed them as absent.
func (e *Engine) Promote(newEpoch uint64, expiryMonoNs int64) {
	e.mu.Lock()
	defer e.mu.Unlock()
	if newEpoch <= e.epoch {
		return // stale/equal promotion — a live primary's lastSeq must not regress
	}
	// Snapshot-transfer poison fence, defence in depth. The AUTHORITATIVE guard is the
	// failover gate (cluster's pbCandidateHighWater refuses an un-OK candidate), but
	// the lease keeper calls Promote from a purely LOCAL decision ("the FSM says I
	// am primary for an epoch my engine has not adopted"), so a poisoned node must
	// also refuse here. Promoting it would set lastSeq from a lastApplied its
	// half-wiped FSM does not back — a fabricated frontier every peer would then
	// log-match against.
	if e.poisoned {
		return
	}
	e.lastSeq = e.lastApplied
	// INHERIT the applied frontier's epoch, do NOT stamp newEpoch. The write
	// sitting at lastApplied was assigned under an OLDER epoch and promotion does
	// not rewrite it — it only continues past it. Stamping newEpoch here would make
	// this node advertise a predecessor identity (lastSeq, newEpoch) that no peer
	// holds, and every one of its first frames would log-match-reject on backups
	// that hold the very same write under its real epoch.
	e.lastSeqEpoch = e.lastAppliedEpoch
	if e.lastApplied > e.committed {
		e.committed = e.lastApplied
	}
	e.epoch = newEpoch
	e.leaseEpoch = newEpoch
	e.leaseExpiry = expiryMonoNs
	// DROP the retained catch-up ring. Everything in it was proposed under an
	// epoch this node no longer serves, and proposals now resume from
	// lastApplied — which, for a node promoted out of the backup role, sits
	// ABOVE the ring's newest entry, because writes taken via Receive never
	// append here. Keeping the old entries would leave the ring holding a
	// disjoint low range with a hole between it and the new proposals, and
	// span() reports [oldest..newest] across that hole as though it were dense.
	//
	// at() now re-checks the seq so a hole can no longer be served as a
	// fabricated entry, but that only makes the hole SAFE, not absent: every
	// lookup into it still costs a grow its delta and forces a snapshot.
	// Clearing the ring makes the state honest — a freshly promoted primary
	// genuinely has nothing to replay, and a catch-up asking for anything is
	// correctly told ring-cold rather than being handed a stale epoch's writes
	// that no longer sit on the chain it is extending.
	e.backlog.reset(e.releaseData)
	// New epoch: a shrink override from any prior epoch is void.
	e.clearEffISRLocked()
	e.clearLearnersLocked() // a freshly promoted primary starts with no grow learners
	// Defensive: fail any stale-epoch in-flight record. A freshly promoted backup
	// has none (it was applying via Receive, not Proposing), so this is a no-op.
	e.flushEpochLocked(e.epoch)
}

// Epoch returns the engine's current cached epoch watermark.
func (e *Engine) Epoch() uint64 {
	e.mu.Lock()
	defer e.mu.Unlock()
	return e.epoch
}

// RunExclusive runs fn while holding BOTH the primary write lock (writeMu) and
// the backup/state lock (e.mu), so NO Applier.Apply can mutate FSM state while
// fn runs. This is the quiesce primitive shard.Store.BackupSnapshot uses to take
// a torn-free, point-in-time serialization of the cache + vector state under PB
// replication.
//
// Consistency argument (no torn snapshot under concurrent PB applies):
//   - The Applier is invoked from EXACTLY two sites, and each holds exactly one
//     of these two locks across the apply:
//   - PRIMARY: proposeSequenced runs e.ap.Apply(data) (step (d)) while
//     holding writeMu (it releases e.mu before the apply and re-takes it only
//     AFTER, to assign the seq). writeMu serializes the whole primary write
//     path, so no primary apply is in progress once RunExclusive holds writeMu
//     — and a primary apply that had started blocks RunExclusive at
//     writeMu.Lock until it finishes.
//   - BACKUP: receiveLocked (Receive / ReceiveGroup) runs e.ap.Apply(msg.Data)
//     (step (c)) while holding e.mu. So no backup apply is in progress once
//     RunExclusive holds e.mu.
//     Holding BOTH locks therefore excludes BOTH apply sites: while fn runs, the
//     cache and vector state are frozen at a single logical point (every applied
//     write is fully materialized; no partially-applied write is visible), and the
//     applied frontier (max(lastSeq, lastApplied)) names that point.
//   - Lock order writeMu→e.mu is IDENTICAL to proposeSequenced's, so RunExclusive
//     can never deadlock with the write path. The ack-completion callbacks
//     (completeSend/completeGroupSend) take only e.mu (not writeMu) and simply
//     block briefly until fn returns; the per-peer sender goroutines drain their
//     channels without either lock, so replication cannot wedge on the quiesce.
//
// fn must be short (a single in-memory serialization) — it stalls the write path
// for its duration.
func (e *Engine) RunExclusive(fn func(appliedFrontier uint64)) {
	e.writeMu.Lock()
	e.mu.Lock()
	defer e.mu.Unlock()
	defer e.writeMu.Unlock()
	// The applied frontier reflected in local FSM state: on a primary that is
	// lastSeq (each seq is applied locally before assignment, step (d)); on a pure
	// backup it is lastApplied. max() covers a node in either role (and a promoted
	// backup where lastSeq was just set to lastApplied).
	frontier, _ := e.appliedFrontierLocked()
	fn(frontier)
}

// appliedFrontierLocked returns the (seq, epoch) IDENTITY of the newest write
// this node has materialized into its FSM, in EITHER role. It is the single
// notion of "what this node's log ends with", and the log-matching
// check is defined against it.
//
// WHY max(lastSeq, lastApplied) AND NOT lastApplied — THE GENESIS CRUX.
// lastApplied moves ONLY in receiveLocked; a node's own writes as PRIMARY advance
// lastSeq and hit the Applier but leave lastApplied at 0. So lastApplied is a
// ROLE-SPECIFIC counter ("how much did I receive as a replica"), never a
// statement about what the node holds. Reading it as one is precisely the defect
// this stage closes: an ex-primary that proposed five writes reports 0, and a
// PrevSeq==0 frame — the "extends from nothing" genesis claim — then matches it.
//
// The genesis convention is therefore NOT a special case in the check, and MUST
// NOT be written as one. There is exactly one right rule:
//
//	a frame extends from nothing IFF the receiver's applied FRONTIER is (0, 0).
//
// A node with lastApplied == 0 but a non-empty state (the ex-primary, or a
// promoted-then-demoted survivor) has a frontier of (lastSeq, lastSeqEpoch) ≠
// (0,0) and so is NOT at genesis — a genesis frame log-match-rejects on it, which
// is the whole point. Any implementation that instead keys genesis off
// `lastApplied == 0` re-opens the divergent append verbatim.
//
// The two watermarks can only be EQUAL at genesis (both 0) or right after Promote
// (which inherits the epoch, so both halves agree), so the tie-break below is
// never observable; it prefers lastSeq to match RunExclusive's long-standing
// frontier definition. Caller holds e.mu.
func (e *Engine) appliedFrontierLocked() (seq, epoch uint64) {
	if e.lastApplied > e.lastSeq {
		return e.lastApplied, e.lastAppliedEpoch
	}
	return e.lastSeq, e.lastSeqEpoch
}

// noteFrontierLocked hands the CURRENT applied frontier to the durable-frontier
// sink, if one is installed. Caller holds e.mu and must call it only after the
// write that moved the frontier has fully materialized (the Applier returned) —
// that is what makes the value it reports provably <= what the FSM holds, which is
// the never-over-report rule the persisted watermark depends on.
//
// It re-reads appliedFrontierLocked rather than taking the caller's (seq, epoch)
// so there is exactly ONE definition of the value that gets persisted, and it is
// the same one CatchupInfo and receiveLocked's log-matching check read. A caller
// passing its own pair could drift from that definition; this cannot.
func (e *Engine) noteFrontierLocked() {
	if e.frontierSink == nil {
		return
	}
	seq, epoch := e.appliedFrontierLocked()
	e.frontierSink(seq, epoch)
}

// AppliedFrontier returns this engine's log identity — the (seq, epoch) of the
// newest write it holds in either role. It takes e.mu. See
// appliedFrontierLocked for why this, and not LastApplied, is the node's history.
func (e *Engine) AppliedFrontier() (seq, epoch uint64) {
	e.mu.Lock()
	defer e.mu.Unlock()
	return e.appliedFrontierLocked()
}

// LeaseValid reports whether this node currently holds a valid MetaRaft primary
// lease for its adopted epoch, judged by the engine's own monotonic clock. It is
// the read-side form of the Propose lease fence (OH1): a primary that cannot
// renew self-fences here without observing any new control-plane state.
// A POISONED engine reports no valid lease. This is what makes the
// seam's VerifyLeader (and through it every linearizable read and the write
// redirect) refuse to serve from a node whose FSM may be half-wiped — the
// "must refuse to serve" half of the poison fence, expressed in the one predicate
// the whole seam already consults.
func (e *Engine) LeaseValid() bool {
	e.mu.Lock()
	defer e.mu.Unlock()
	if e.poisoned {
		return false
	}
	return e.leaseEpoch == e.epoch && e.now() < e.leaseExpiry
}

// LastSeq returns the highest seq assigned by this node as primary (assigned,
// NOT necessarily durable — use Committed for the durable frontier).
func (e *Engine) LastSeq() uint64 {
	e.mu.Lock()
	defer e.mu.Unlock()
	return e.lastSeq
}

// LastApplied returns the highest seq this engine has applied as a backup.
func (e *Engine) LastApplied() uint64 {
	e.mu.Lock()
	defer e.mu.Unlock()
	return e.lastApplied
}

// Propose is the PRIMARY write path, PIPELINED (PIPELINE-REDESIGN). It splits
// into a serialized sequencing stage under writeMu (fence → local apply → assign
// seq → register in-flight → enqueue to per-peer FIFOs; microseconds, NO network
// I/O) and an asynchronous commit stage (per-peer senders + ack-driven sweep)
// that overlaps up to W writes per shard. It still BLOCKS until THIS write
// full-ISR-commits or fails, so the client contract is unchanged: it fences a
// primary that lost its lease (OH1), refuses to ack below the durability floor
// (H3), counts only exact (epoch,seq) acks (H6), and commits only when EVERY
// propose-time ISR member acked. Concurrency comes from concurrent callers.
//
// Returns (result, seq, err): result is this write's own local-apply output;
// err == nil means durably committed; a non-nil err (timeout / lease flush /
// stall) means the write is an applied-but-non-durable tail — unknown, possibly
// transitively durable later via a following commit (P9/P10).
func (e *Engine) Propose(ctx context.Context, data []byte) (result []byte, seq uint64, err error) {
	for {
		// 1. Admission gate (ctx-bounded), OUTSIDE writeMu: block while the window
		// (lastSeq - committed) is full so the OH2 uncommitted tail stays bounded
		// by W; a wedged shard refuses admission here (PIPELINE-REDESIGN §4). On a
		// non-nil return no slot came free before ctx ended — distinguish the two
		// honestly (the seam maps these): a caller cancellation surfaces
		// the real ctx.Err(), while a deadline that expired with the window still
		// full is the designed admission-stall signal, ErrPipelineStalled.
		if werr := e.windowWait(ctx); werr != nil {
			if errors.Is(werr, context.Canceled) {
				return nil, 0, werr
			}
			return nil, 0, ErrPipelineStalled
		}

		rec, result, seq, retry, err := e.proposeSequenced(data)
		if retry {
			continue
		}
		if err != nil {
			return nil, 0, err
		}

		// 3. Wait OUTSIDE the lock for this record to resolve — full-ISR commit,
		// ctx timeout, or an epoch/lease-change flush.
		select {
		case rerr := <-rec.doneCh:
			return result, seq, rerr
		case <-ctx.Done():
			// Time out this record. resolveLocked is exactly-once, so a commit that
			// won the race keeps its nil outcome; read the resolved err back rather
			// than assuming the timeout stuck (avoids reporting a committed write as
			// failed).
			e.mu.Lock()
			e.resolveLocked(rec, ErrReplicationTimeout)
			rerr := rec.err
			e.mu.Unlock()
			return result, seq, rerr
		}
	}
}

// pbTimerPool recycles ProposeDeadline's commit-wait timers so the hot seam
// path allocates no timer per write (2026-07-22 alloc profile: per-write
// context+timer machinery was ~19% of write-path bytes).
var pbTimerPool sync.Pool

func getPBTimer(d time.Duration) *time.Timer {
	if t, _ := pbTimerPool.Get().(*time.Timer); t != nil {
		t.Reset(d)
		return t
	}
	return time.NewTimer(d)
}

func putPBTimer(t *time.Timer) {
	if !t.Stop() {
		select {
		case <-t.C:
		default:
		}
	}
	pbTimerPool.Put(t)
}

// ProposeDeadline is Propose for callers with a plain timeout and no
// cancellation source: semantics identical to Propose under a
// context.WithTimeout, minus the per-write context, cancel channel, and timer
// allocations. timeout bounds admission and the commit wait TOGETHER, like a
// context deadline would.
func (e *Engine) ProposeDeadline(data []byte, timeout time.Duration) ([]byte, uint64, error) {
	start := time.Now()
	for {
		if werr := e.windowWaitTimeout(timeout - time.Since(start)); werr != nil {
			return nil, 0, werr
		}
		rec, result, seq, retry, err := e.proposeSequenced(data)
		if retry {
			continue
		}
		if err != nil {
			return nil, 0, err
		}
		t := getPBTimer(timeout - time.Since(start))
		select {
		case rerr := <-rec.doneCh:
			putPBTimer(t)
			return result, seq, rerr
		case <-t.C:
			pbTimerPool.Put(t) // fired and drained: ready for Reset on reuse
			e.mu.Lock()
			e.resolveLocked(rec, ErrReplicationTimeout)
			rerr := rec.err
			e.mu.Unlock()
			return result, seq, rerr
		}
	}
}

// windowWaitTimeout is windowWait for deadline callers: block until the window
// has room or timeout elapses (ErrPipelineStalled — the admission-stall
// signal). The watcher goroutine + timer engage only on the rare blocking
// path; the common non-full case is a lock, a check, and out.
func (e *Engine) windowWaitTimeout(timeout time.Duration) error {
	e.mu.Lock()
	defer e.mu.Unlock()
	if !e.windowFullLocked() {
		return nil
	}
	if timeout <= 0 {
		return ErrPipelineStalled
	}
	timedOut := false
	stop := make(chan struct{})
	defer close(stop)
	go func() {
		t := time.NewTimer(timeout)
		defer t.Stop()
		select {
		case <-t.C:
			e.mu.Lock()
			timedOut = true
			e.windowCond.Broadcast()
			e.mu.Unlock()
		case <-stop:
		}
	}()
	for e.windowFullLocked() {
		if timedOut {
			return ErrPipelineStalled
		}
		e.windowCond.Wait()
	}
	return nil
}

// proposeSequenced is the sequencing stage shared by Propose and
// ProposeDeadline: fences, local apply, seq assignment, backlog append,
// in-flight registration, and per-peer submission — steps 2..(f) of the
// pipelined write path. retry means the window filled between admission and
// the locked re-check; the caller must re-admit. On success the caller owns
// the wait on rec.doneCh.
func (e *Engine) proposeSequenced(data []byte) (rec *inflight, result []byte, seq uint64, retry bool, err error) {
	// Snapshot control-plane state once, BEFORE the serialized section: these
	// are advisory recent-view reads (per-shard fencing authority is the LEASE,
	// checked under e.mu below — we deliberately do NOT read ctrl.Epoch, since
	// a partitioned node's local MetaFSM would agree with its own stale cached
	// epoch, OH1), and the control plane gives no ordering guarantee tying them
	// to writeMu anyway. Off the critical path they stop serializing every
	// shard's writers on the global MetaFSM lock and its per-call ISR copy.
	primary := e.ctrl.Primary(e.shard)
	isr := e.ctrl.ISR(e.shard)
	minISR := e.ctrl.MinISR(e.shard)

	// 2. Sequencing critical section. writeMu serializes the whole stage so
	// seq assignment and per-peer enqueue happen in one total per-shard order
	// (P1). No network I/O and no blocking channel op happens under it.
	e.writeMu.Lock()

	e.mu.Lock()
	// (a0) POISON FENCE: a node with a pending snapshot install must
	// never ack a write. Its FSM may be half-wiped, so a local apply on top of it
	// would build committed state over an indeterminate base — and its watermark
	// cannot even name where that base ends. Refused BEFORE the primary/lease
	// checks so it holds even for a node the control plane still calls primary.
	if e.poisoned {
		e.mu.Unlock()
		e.writeMu.Unlock()
		return nil, nil, 0, false, ErrSnapshotPending
	}
	// (a) Named-primary check.
	if primary != e.nodeID {
		e.mu.Unlock()
		e.writeMu.Unlock()
		return nil, nil, 0, false, ErrNotPrimary
	}
	// (a') Lease fence (OH1): self-fence on the engine's own monotonic clock.
	if e.leaseEpoch != e.epoch || e.now() >= e.leaseExpiry {
		e.mu.Unlock()
		e.writeMu.Unlock()
		return nil, nil, 0, false, ErrLeaseExpired
	}
	// (b) Min-ISR floor (H3): never ack below the durability floor.
	if len(isr) < minISR {
		e.mu.Unlock()
		e.writeMu.Unlock()
		return nil, nil, 0, false, ErrBelowMinISR
	}
	// (c) Re-check the window UNDER the sequencing lock. windowWait admission is
	// check-then-act, so N proposers can pass it concurrently; without this
	// re-check they would overshoot W and soften the OH2 bound. If full, back
	// out and re-admit (the outer loop blocks in windowWait again).
	if e.windowFullLocked() {
		e.mu.Unlock()
		e.writeMu.Unlock()
		return nil, nil, 0, true, nil
	}
	epoch := e.epoch
	e.mu.Unlock()

	// (d) Apply locally FIRST, before assigning a seq (P2). A failed apply must
	// burn NO seq: a phantom seq would gap-reject every future write at backups
	// forever (a shard wedge). writeMu is held, so no other Propose interleaves
	// between apply and the seq assignment below.
	result, err = e.ap.Apply(data)
	if err != nil {
		e.writeMu.Unlock()
		return nil, nil, 0, false, err
	}

	// (e) Assign the dense seq, retain in the catch-up ring, and register the
	// in-flight record — all under e.mu, in seq order (P1). needed == 0
	// (single node, no backups) commits immediately through the sweep, behind
	// the same commit-time lease fence as any other record.
	peers := distinctPeers(e.nodeID, isr)
	e.mu.Lock()
	// STALE-READ RACE FIX — half 2 of 2 (ISR shrink; MUST stay paired with
	// ShrinkISR's narrowing pass, or the wedge reopens). `isr` was snapshotted
	// OUTSIDE any lock (see the read above), so a Propose that captured a
	// PRE-shrink ISR could otherwise register the removed member M into this
	// record's pending set and re-wedge the pipeline the shrink just un-wedged.
	// ShrinkISR installs effISR/effISREpoch under e.mu; HERE, also under e.mu,
	// we intersect the stale snapshot with the live override. Because BOTH the
	// install and this intersect run under e.mu there is NO window: this Propose
	// either observes the override (and drops M now) or does not — in which case
	// ShrinkISR's later narrowing pass, ordered after this record's registration
	// on e.mu, removes M from it. Intersect is NARROWING-ONLY (it can drop peers,
	// never add), so it never widens a record — the load-bearing shrink/grow
	// asymmetry. `epoch` was captured under e.mu above and is this record's epoch;
	// the override applies only when it was decided for exactly that epoch.
	if e.effISREpoch == epoch {
		peers = intersectPeers(peers, e.effISR)
	}
	// Capture the predecessor's IDENTITY, not just its position. The
	// write at seq-1 is exactly this node's applied frontier right now (still under
	// e.mu, before this seq is registered), so its epoch is the frontier's epoch.
	// On a freshly promoted primary that is the INHERITED epoch of the write it was
	// promoted at — not the new epoch — which is what lets its first frame match on
	// backups that already hold that write. Stamped onto the ring entry so a later
	// replay reproduces the exact same chain link (see ringEntry.prevEpoch).
	prevSeq, prevEpoch := e.appliedFrontierLocked()
	seq = prevSeq + 1
	if evicted := e.backlog.append(ringEntry{epoch: epoch, seq: seq, prevEpoch: prevEpoch, data: data}); evicted != nil && e.releaseData != nil {
		e.releaseData(evicted)
	}
	rec = e.registerInflightLocked(epoch, seq, peers)
	// The frontier advanced to this write, which step (d) already
	// materialized locally. Persisting an UNCOMMITTED primary write is correct and
	// intentional — the persisted value describes what this node's FSM HOLDS, which
	// is exactly what appliedFrontierLocked means and exactly what a peer's log
	// matching will compare against. Persisting only committed writes would
	// under-report a primary's own applied tail, which is the (0,0)-over-real-data
	// class of lie this stage exists to remove.
	e.noteFrontierLocked()
	if len(peers) == 0 {
		e.sweepLocked()
	}
	// ISR GROW — snapshot the learner ship-set UNDER e.mu (learners is e.mu-
	// guarded), as `learners \ peers`. A learner is shipped every write so a later
	// re-add finds it gap-free, but it is registered into NO in-flight record's
	// pending set (registerInflightLocked got `peers` only) — the no-widen-in-flight
	// rule. The `\ peers` guard is what makes the learner→voter transition
	// duplicate-free AND gap-free across BOTH orderings of "OpSetShardISR observed"
	// vs "GrowISR ran": once M enters ctrl.ISR it is in `peers` and drops out of the
	// learner ship-set (shipped once, as a voter); until then it is a learner
	// (shipped once, as a learner). It is therefore in EXACTLY one of the two sets
	// for every seq — no gap, no double-send — so GrowISR need not (and must not,
	// lest it open a gap for an in-flight stale-ctrl.ISR Propose) remove M here.
	learnerShip := e.learnerShipSetLocked(peers)
	e.mu.Unlock()

	// (f) Submit the frame to every peer — still under writeMu, so frames
	// reach each peer in strict seq order (P1). Fast path: an uncontended,
	// connected link takes the write inline from THIS goroutine (no sender
	// wakeup — lever 1, see submitPeerLocked); otherwise the frame enters the
	// peer's ordered channel and the per-peer sender submits asynchronously.
	// The ack wait is OUTSIDE the lock below. Under STATIC ISR membership
	// (the only mode today — membership changes are ErrPBUnimplemented) the
	// channel is buffered to W and the window bounds undrained frames to
	// <= W per peer, so the send never blocks under the lock.
	// ISR shrink alone cannot overflow this channel: a shrink only NARROWS
	// the peer set (the removed peer's sender is dropped, not fed), and the
	// proposeSequenced intersect guarantees no frame is ever enqueued for a
	// removed peer. The depth-past-W hazard needs a shrink-then-RE-ADD (grow),
	// which is a separate deferred increment; grow must re-add via a FRESH
	// sender and still switch to an engine-owned unbounded deque
	// (PIPELINE-REDESIGN §Q1) to bound a caught-up member's backlog.
	// grow re-add path re-adds a member via a FRESH sender pre-loaded
	// with the final ring delta, and ships to it via a NON-BLOCKING learner send
	// (submitLearnerLocked) that abandons a hopeless grow instead of blocking here.
	if len(peers) > 0 || len(learnerShip) > 0 {
		msg := ReplicateMsg{Epoch: epoch, Seq: seq, PrevSeq: prevSeq, PrevEpoch: prevEpoch, Data: data}
		for _, peer := range peers {
			e.submitPeerLocked(peer, msg) // required: blocking, window-bounded
		}
		// Learners never gate commit, so they must never block the write path: a
		// full learner channel means the grow is hopeless and is abandoned. Learners
		// are shipped AFTER the required peers so the (rare) abandon path runs last.
		for _, learner := range learnerShip {
			e.submitLearnerLocked(learner, msg)
		}
		// CommitPrimary: the backups now have the write queued (best-effort,
		// async); commit on local apply without waiting for their acks. Done
		// under writeMu, in seq order, so committed advances densely and the
		// head is this record (or a resolved-failed predecessor the sweep pops
		// transitively). The lease fence still governs inside sweepLocked. Gated on
		// a real required set (a learner-only ship never commits — commit is peers-only).
		if len(peers) > 0 && e.commitLevel == CommitPrimary {
			e.mu.Lock()
			e.sweepLocked()
			e.mu.Unlock()
		}
	}
	e.writeMu.Unlock()
	return rec, result, seq, false, nil
}

// distinctPeers returns every DISTINCT ISR member other than self. Dedup: a
// duplicated backup id must not contribute two acks (that would let one real
// backup stand in for another, defeating full-ISR durability). Identical
// construction to the pre-pipeline Propose.
func distinctPeers(self string, isr []string) []string {
	seen := map[string]struct{}{self: {}}
	peers := make([]string, 0, len(isr))
	for _, m := range isr {
		if _, dup := seen[m]; dup {
			continue
		}
		seen[m] = struct{}{}
		peers = append(peers, m)
	}
	return peers
}

// submitPeerLocked ships msg to peer: the inline fast path submits from THIS
// goroutine when the transport supports it, the link can take the write
// without blocking or dialing, AND no sender-path submission is pending for
// this peer (pending == 0 — the P1 ordering gate, see peerSender); otherwise
// it appends to the peer's ordered send channel, lazily starting the peer's
// single sender goroutine on first use. Caller holds writeMu. The channel is
// buffered to pipelineWindow; under STATIC ISR membership the admission
// window bounds undrained frames to <= W per peer, so the send never blocks
// under the lock. That bound is NOT structural under dynamic membership —
// see the TODO at the Propose submit site. A no-op after Shutdown (the record
// then resolves via its ctx timeout).
func (e *Engine) submitPeerLocked(peer string, msg ReplicateMsg) {
	if e.closed {
		return
	}
	ps := e.peerQ[peer]
	if ps == nil {
		if e.peerQ == nil {
			e.peerQ = make(map[string]*peerSender)
		}
		ps = &peerSender{ch: make(chan ReplicateMsg, pipelineWindow)}
		e.peerQ[peer] = ps
		e.senderWG.Add(1)
		go e.runSender(peer, ps)
	}
	if ps.pending.Load() == 0 {
		if it, ok := e.tr.(InlineTransport); ok {
			m := msg
			if it.TryReplicate(peer, m, func(ack AckMsg, cbErr error) {
				e.completeSend(peer, m, ack, cbErr)
			}) {
				return
			}
		}
	}
	ps.pending.Add(1)
	ps.ch <- msg
}

// runSender is the ONE goroutine per peer that drains that peer's ordered frame
// channel and submits to the Transport in channel (== seq) order — the
// structural guarantee behind per-shard seq order on the wire (P1). Every
// submission completes EXACTLY ONCE: via the completion callback (ack or
// transport error), or — when the submit call returns an error (the
// returned error IS the completion, the callback will NOT fire) — directly
// from the return path in submitOne/submitGroup.
//
// When the Transport also implements GroupTransport, the sender GROUPS what is
// already queued: after receiving one message it greedily drains —
// never waits — up to pbGroupBatchMax-1 more, breaking the group where the
// epoch changes or seq continuity breaks, so every group is uniform-epoch and
// seq-dense by construction. A group of 1 submits through the plain per-message
// Replicate path, byte-identical to pre-Plan-G behavior.
func (e *Engine) runSender(peer string, ps *peerSender) {
	defer e.senderWG.Done()
	ch := ps.ch
	gt, hasGroup := e.tr.(GroupTransport)
	if !hasGroup {
		for msg := range ch {
			// ISR shrink: once dropPeerLocked has removed this peer from the
			// ISR it latches discard — drain the remaining parked frames but do
			// NOT submit them (the payload buffers may already be recycled by a
			// post-shrink ring wrap; submitting would alias them). See peerSender.
			if ps.discard.Load() {
				ps.pending.Add(-1)
				continue
			}
			e.submitOne(peer, msg)
			ps.pending.Add(-1) // after the submit returned: the link append happened
		}
		return
	}
	var batch []ReplicateMsg
	var next ReplicateMsg
	pendingNext := false
	for {
		var first ReplicateMsg
		if pendingNext {
			first, pendingNext = next, false
		} else {
			m, ok := <-ch
			if !ok {
				return
			}
			first = m
		}
		batch = append(batch[:0], first)
		bytes := len(first.Data)
	drain:
		for len(batch) < pbGroupBatchMax {
			select {
			case m, ok := <-ch:
				if !ok {
					break drain // submit what we have; the next blocking recv exits
				}
				if m.Epoch != first.Epoch || m.Seq != batch[len(batch)-1].Seq+1 ||
					bytes+len(m.Data) > pbGroupBytesMax {
					next, pendingNext = m, true // discontinuity/full: m starts the next group
					break drain
				}
				batch = append(batch, m)
				bytes += len(m.Data)
			default:
				break drain
			}
		}
		// ISR shrink: a removed peer's sender discards its parked frames
		// unsubmitted (see the non-group path above and peerSender.discard).
		if ps.discard.Load() {
			ps.pending.Add(-int64(len(batch)))
			continue
		}
		if len(batch) == 1 {
			e.submitOne(peer, batch[0])
		} else {
			e.submitGroup(gt, peer, batch)
		}
		// Decrement ONLY after the transport call returned (the link-append
		// ordering point) so the inline fast path can never interleave with a
		// sender-path submission (P1). A held-back `next` stays counted: it has
		// left the channel but not yet been submitted.
		ps.pending.Add(-int64(len(batch)))
	}
}

// submitOne ships a single write via the per-message Replicate path.
func (e *Engine) submitOne(peer string, msg ReplicateMsg) {
	if err := e.tr.Replicate(peer, msg, func(ack AckMsg, cbErr error) {
		e.completeSend(peer, msg, ack, cbErr)
	}); err != nil {
		e.completeSend(peer, msg, AckMsg{}, err)
	}
}

// submitGroup ships a uniform-epoch, seq-dense group of writes as one frame.
// The completion closure captures only scalars (epoch, seq bounds) — never the
// msgs slice, whose backing array the sender reuses for the next group.
func (e *Engine) submitGroup(gt GroupTransport, peer string, msgs []ReplicateMsg) {
	epoch := msgs[0].Epoch
	firstSeq := msgs[0].Seq
	lastSeq := msgs[len(msgs)-1].Seq
	if err := gt.ReplicateGroup(peer, msgs, func(ack AckMsg, cbErr error) {
		e.completeGroupSend(peer, epoch, firstSeq, lastSeq, ack, cbErr)
	}); err != nil {
		e.completeGroupSend(peer, epoch, firstSeq, lastSeq, AckMsg{}, err)
	}
}

// completeSend routes one submission's completion to its in-flight record under
// e.mu. An H6-exact OK ack (byte-identical match to engine.go's original counting
// rule) advances the record via ackInflightLocked, which may commit through the
// sweep. ANYTHING else — a transport error, or a non-OK / wrong-epoch / wrong-seq
// response — means this REQUIRED peer can never satisfy full-ISR for this exact
// (epoch,seq), so the record is doomed: resolve it FAILED promptly rather than
// waiting out its Propose ctx (fail-fast, matching the pre-pipeline Propose and
// the design's "identical failure semantics" promise). Failing on the FIRST bad
// response is correct — full-ISR needs ALL peers, so one errored/rejecting
// required peer dooms the write; it stays applied-locally-but-non-durable
// (P9/P10 unchanged). No resend (deferred). The exactly-once latch
// makes this a no-op if the ctx timeout, a full-ack, or an epoch flush already
// resolved the record. Caller must NOT hold e.mu.
func (e *Engine) completeSend(peer string, msg ReplicateMsg, ack AckMsg, err error) {
	ok := err == nil && ack.OK && ack.Epoch == msg.Epoch && ack.Seq == msg.Seq
	e.mu.Lock()
	if ok {
		e.notePeerSuccessLocked(peer) // reset the shrink wedge counter
		e.creditPeerAckedLocked(peer, ack.Seq)
		e.ackInflightLocked(peer, ack)
	} else {
		// This required peer failed this exact frame — bump its consecutive-failure
		// count so the shrink driver can detect a dead member. Counted on
		// EVERY failed completion, independent of whether the record is still live.
		e.notePeerFailureLocked(peer)
		if rec := e.inflightAtLocked(msg.Seq); rec != nil && rec.epoch == msg.Epoch {
			// Guard rec.epoch == msg.Epoch: only fail the record this frame belongs to
			// (defends against a popped/replaced FIFO slot).
			e.resolveLocked(rec, ErrReplicationTimeout)
			e.sweepLocked()
		}
	}
	e.mu.Unlock()
}

// completeGroupSend routes a group submission's ONE completion to every record
// in [firstSeq, lastSeq] under a single e.mu acquisition. Credit follows the
// cumulative-ack contract (Engine.ReceiveGroup):
//
//   - full OK (OK, epoch match, Seq == lastSeq): every record's pending set
//     drops peer. This is H6-sound: the group was uniform-epoch and seq-dense,
//     and the backup applies in order behind its gap check, so one cumulative
//     ack attests each exact (epoch, seq <= lastSeq) individually (P5).
//   - prefix nack (!OK, epoch match, firstSeq-1 <= Seq < lastSeq): records
//     through ack.Seq get peer credit; every later record is resolved FAILED —
//     the identical outcome the per-message path produces when the backup acks
//     a prefix then nacks the first gap and everything behind it (fail-fast,
//     matching completeSend).
//   - anything else (transport error, wrong epoch, out-of-range Seq): credit
//     NOTHING, fail every record. Under-crediting is always safe — it can only
//     fail writes, never falsely commit (P6).
//
// Records are matched via inflightAtLocked and epoch-guarded exactly like
// completeSend, so a popped/replaced FIFO slot is never misattributed. One
// sweep at the end advances the commit frontier. Caller must NOT hold e.mu.
func (e *Engine) completeGroupSend(peer string, epoch, firstSeq, lastSeq uint64, ack AckMsg, err error) {
	creditThrough := firstSeq - 1 // exclusive floor: no credit
	switch {
	case err == nil && ack.OK && ack.Epoch == epoch && ack.Seq == lastSeq:
		creditThrough = lastSeq
	case err == nil && !ack.OK && ack.Epoch == epoch && ack.Seq >= firstSeq-1 && ack.Seq < lastSeq:
		creditThrough = ack.Seq
	}
	e.mu.Lock()
	// Shrink wedge counter: a full-group OK clears it; any short credit
	// (a prefix nack, transport error, or wrong-epoch ack) is a failure signal.
	if creditThrough == lastSeq {
		e.notePeerSuccessLocked(peer)
	} else {
		e.notePeerFailureLocked(peer)
	}
	if creditThrough >= firstSeq {
		// The cumulative ack attests this backup applied every seq through
		// creditThrough (P5), so its high-water is that seq (observability only).
		e.creditPeerAckedLocked(peer, creditThrough)
	}
	for s := firstSeq; s <= lastSeq; s++ {
		rec := e.inflightAtLocked(s)
		if rec == nil || rec.epoch != epoch {
			continue
		}
		if s <= creditThrough {
			rec.removePending(peer)
		} else {
			e.resolveLocked(rec, ErrReplicationTimeout)
		}
	}
	e.sweepLocked()
	e.mu.Unlock()
}

// Shutdown stops every per-peer sender goroutine: it latches closed, closes the
// senders' channels (each drains any already-enqueued frames, then exits), and
// waits for them all to return. Safe to call more than once. It does NOT close
// the Transport (its owner does that) and does NOT resolve in-flight records —
// after Shutdown, an in-flight Propose resolves through its own ctx timeout.
func (e *Engine) Shutdown() {
	e.writeMu.Lock()
	if e.closed {
		e.writeMu.Unlock()
		return
	}
	e.closed = true
	for peer, ps := range e.peerQ {
		close(ps.ch)
		delete(e.peerQ, peer)
	}
	e.writeMu.Unlock()
	e.senderWG.Wait()
}

// markCommittedLocked advances the durable high-watermark under e.mu (the sweep
// path already holds the lock, so it must not re-lock). committed is a max, so a
// transitive commit can jump it past resolved-failed holes (P7/P10). Advancing
// it frees window slots, so it wakes admission waiters.
func (e *Engine) markCommittedLocked(seq uint64) {
	if seq > e.committed {
		e.committed = seq
		if e.windowCond != nil {
			e.windowCond.Broadcast()
		}
	}
}

// Committed returns the highest seq durably acked by the min-ISR floor.
func (e *Engine) Committed() uint64 {
	e.mu.Lock()
	defer e.mu.Unlock()
	return e.committed
}

// creditPeerAckedLocked advances a backup's replication high-water to seq (a
// monotonic max — an out-of-order or duplicate ack never rewinds it). It is
// observability bookkeeping only: it feeds ReplicationStatus and has NO bearing
// on the commit decision (the full-ISR sweep owns that). Caller holds e.mu.
func (e *Engine) creditPeerAckedLocked(peer string, seq uint64) {
	if e.peerAcked == nil {
		e.peerAcked = make(map[string]uint64, 2) // typical ISR: 1-2 backups
	}
	if seq > e.peerAcked[peer] {
		e.peerAcked[peer] = seq
	}
}

// PeerLag is one backup's replication progress relative to this primary, as of
// the ReplicationStatus snapshot. Acked is the highest seq the backup has
// H6-exactly acked; Lag is LastSeq-Acked (assigned writes the backup is behind).
type PeerLag struct {
	Peer  string
	Acked uint64
	Lag   uint64
}

// ReplicationStatus is a point-in-time snapshot of this engine's PRIMARY-side
// replication progress: the assigned frontier (LastSeq), the durable full-ISR
// frontier (Committed), and each backup's acked high-water (Peers). A pure
// backup (this node never proposed) reports LastSeq==0 and no Peers. It is a
// read-only introspection view for the replication-metrics op — it takes e.mu
// exactly like the other engine reads and copies out, so the caller never
// aliases engine-internal state.
type ReplicationStatus struct {
	Epoch     uint64
	LastSeq   uint64
	Committed uint64
	Peers     []PeerLag
}

// ReplicationStatus snapshots the primary-side replication progress under e.mu.
func (e *Engine) ReplicationStatus() ReplicationStatus {
	e.mu.Lock()
	defer e.mu.Unlock()
	st := ReplicationStatus{
		Epoch:     e.epoch,
		LastSeq:   e.lastSeq,
		Committed: e.committed,
	}
	if len(e.peerAcked) > 0 {
		st.Peers = make([]PeerLag, 0, len(e.peerAcked))
		for peer, acked := range e.peerAcked {
			// A backup can momentarily read acked > lastSeq only across an epoch
			// change race; clamp the lag at 0 so the metric never underflows.
			var lag uint64
			if e.lastSeq > acked {
				lag = e.lastSeq - acked
			}
			st.Peers = append(st.Peers, PeerLag{Peer: peer, Acked: acked, Lag: lag})
		}
	}
	return st
}

// Receive is the BACKUP receive path. It fences stale primaries by epoch
// (H1/H5), rejects out-of-order writes via the PrevSeq gap check, and only
// then applies locally and acks the exact (epoch,seq). It is serialized per
// shard by e.mu.
func (e *Engine) Receive(msg ReplicateMsg) AckMsg {
	e.mu.Lock()
	defer e.mu.Unlock()
	return e.receiveLocked(msg)
}

// ReceiveGroup is the BACKUP receive path for one group frame: it runs
// the exact per-message Receive logic over msgs IN ORDER under ONE e.mu
// acquisition, stopping at the first failure, and answers with ONE cumulative
// ack: OK with Seq == last seq when everything applied, else !OK with Seq = the
// last seq of THIS group that did apply (or firstSeq-1 when none did). The
// caller (the primary's completeGroupSend) credits exactly that prefix.
func (e *Engine) ReceiveGroup(msgs []ReplicateMsg) AckMsg {
	if len(msgs) == 0 {
		return AckMsg{OK: false}
	}
	e.mu.Lock()
	defer e.mu.Unlock()
	// The cumulative-ack baseline is "the position we hold if the FIRST frame is
	// rejected", i.e. one before it. On a seq-0 group that subtraction UNDERFLOWS
	// to MaxUint64, and the nack then reports an applied high-water of
	// 18446744073709551615 to the sender — a peer-controlled counter that feeds
	// straight into backfillLearner's delta arithmetic (tail - k, k + batch).
	//
	// Seq 0 is never assignable: proposeSequenced hands out frontier+1 from a
	// frontier that starts at 0, so the lowest legitimate seq is 1. A seq-0 group
	// can therefore only come from a malformed or hostile peer on the wire-format
	// branch — exactly the input that must not be trusted to be well-formed. The
	// honest ack for a group we applied nothing from is 0.
	//
	// backfillLearner separately rejects an ack past the shipped range (which
	// catches this MaxUint64 downstream), but defence there does not justify
	// emitting a nonsense high-water here: every OTHER consumer of this ack would
	// have to repeat that guard.
	if msgs[0].Seq == 0 {
		return AckMsg{Epoch: msgs[0].Epoch, Seq: 0, OK: false}
	}
	lastOK := msgs[0].Seq - 1
	for i := range msgs {
		if ack := e.receiveLocked(msgs[i]); !ack.OK {
			return AckMsg{Epoch: msgs[0].Epoch, Seq: lastOK, OK: false}
		}
		lastOK = msgs[i].Seq
	}
	return AckMsg{Epoch: msgs[0].Epoch, Seq: lastOK, OK: true}
}

// receiveLocked is the Receive body; caller holds e.mu.
func (e *Engine) receiveLocked(msg ReplicateMsg) AckMsg {
	// (a0) POISON FENCE. A pending snapshot install means this node's
	// FSM may be half-wiped and its applied frontier means nothing — so the log
	// matching in step (b), which is a comparison AGAINST that frontier, cannot be
	// trusted to distinguish a prefix from a fork. Refuse every write until a fresh
	// snapshot install restores a frontier the FSM actually backs. Nacking (rather
	// than silently accepting) is what turns this into a clean grow abort upstream.
	if e.poisoned {
		return AckMsg{Epoch: e.epoch, Seq: msg.Seq, OK: false}
	}
	// (a) Epoch fence (H1/H5): reject a stale primary; adopt a higher epoch.
	if msg.Epoch < e.epoch {
		return AckMsg{Epoch: e.epoch, Seq: msg.Seq, OK: false}
	}
	if msg.Epoch > e.epoch {
		e.epoch = msg.Epoch
		// New epoch observed as a backup: any shrink override this node installed
		// while it was primary at an older epoch is void.
		e.clearEffISRLocked()
		e.clearLearnersLocked() // and any grow learners it was streaming to as that old primary
		// A node that was primary at an older epoch and is now hearing a newer
		// epoch as a backup must fail its stale-epoch in-flight primary records
		// (Q5 epoch flush) — they can never commit under the fence.
		e.flushEpochLocked(e.epoch)
	}

	// (b) LOG MATCHING. The write must extend the EXACT history this
	// node already holds: the frame names its predecessor as (PrevSeq, PrevEpoch),
	// and we accept only if that pair IS our applied frontier. This is the direct
	// analogue of Raft's (prevLogIndex, prevLogTerm) check, and it upgrades the old
	// bare `PrevSeq != lastApplied` gap check on two independent axes:
	//
	//   - THE EPOCH HALF. Promote continues seq assignment from the promoted node's
	//     applied high-water, so a seq is REUSED across epochs with different
	//     content. Position alone proves nothing; only (seq, epoch) identifies a
	//     write. Without the epoch, a successor's seq 3 silently overwrites a
	//     predecessor's different seq 3.
	//   - THE FRONTIER HALF. The old check read e.lastApplied, which moves ONLY
	//     here — a node's own writes as PRIMARY leave it at 0. An ex-primary
	//     therefore looked like a blank node and accepted a genesis frame
	//     (PrevSeq==0) on top of its own divergent history. appliedFrontierLocked is
	//     the node's real log end in EITHER role, and genesis is simply the case
	//     where that frontier is (0, 0) — see its doc for why this must not be
	//     written as a `lastApplied == 0` special case.
	//
	// A mismatch is NOT a recoverable gap: with no resend and no snapshot transfer
	// (deferred), the sender cannot bridge it. Nacking is what turns a
	// silent divergent append into an abort the grow driver reads as
	// ErrCatchupDiverged.
	// The frame must also be its own predecessor's SUCCESSOR (Seq == PrevSeq+1).
	// Every sender builds it that way; enforcing it here is what makes the chain a
	// chain rather than two independent numbers, and it stops a malformed/hostile
	// frame from REGRESSING the applied frontier (accepted at PrevSeq==frontier but
	// carrying a lower Seq), which would re-open the very append this check exists
	// to close.
	if msg.Seq != msg.PrevSeq+1 {
		return AckMsg{Epoch: msg.Epoch, Seq: msg.Seq, OK: false}
	}
	frontierSeq, frontierEpoch := e.appliedFrontierLocked()
	if msg.PrevSeq != frontierSeq || msg.PrevEpoch != frontierEpoch {
		return AckMsg{Epoch: msg.Epoch, Seq: msg.Seq, OK: false}
	}

	// (c) Apply locally, advance the applied frontier (BOTH halves — a seq without
	// its epoch is not an identity), and ack the exact (epoch,seq). Within a group
	// frame this is what re-chains the check for the next record: after applying
	// msg, lastApplied == msg.Seq >= lastSeq, so the frontier is exactly
	// (msg.Seq, msg.Epoch) — which is precisely the predecessor the next record
	// declares (see decodeReplicateGroup's implied chain).
	if _, err := e.ap.Apply(msg.Data); err != nil {
		return AckMsg{Epoch: msg.Epoch, Seq: msg.Seq, OK: false}
	}
	e.lastApplied = msg.Seq
	e.lastAppliedEpoch = msg.Epoch
	// (d) the frontier just advanced over a MATERIALIZED write (the
	// Apply above returned), so it is now eligible to be persisted. The sink only
	// records it; the durable stamp is amortised off this path.
	e.noteFrontierLocked()
	return AckMsg{Epoch: msg.Epoch, Seq: msg.Seq, OK: true}
}

// backlogLen reports the number of retained writes (test/introspection helper).
func (e *Engine) backlogLen() int {
	e.mu.Lock()
	defer e.mu.Unlock()
	return e.backlog.len()
}
