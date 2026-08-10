# PB-ISR Pipelined Primary Write Path — Design

**Status:** proposed (redesign of `shard/pbisr/engine.go` Propose path)
**Prereq reading:** `shard/pbisr/DESIGN.md` (H1–H6, OH1–OH3), `shard/pbisr/engine.go`

## Summary

Today `Engine.Propose` holds `writeMu` across the full write path — fence, local apply, seq assignment, **and the ship-to-all-ISR network round trip** (engine.go:271-272, wait loop engine.go:360-372). Per shard, the write path is strict ping-pong: the next seq cannot be assigned until the previous write has committed across the network. Throughput is bounded by commit RTT, not by pipe or CPU; the writev batcher never sees more than ~1 frame per flush (measured 1.09).

The redesign splits Propose into a **serialized sequencing stage** (fence → local apply → seq assign → backlog append → enqueue to per-peer FIFOs → register an in-flight record) that holds the lock for microseconds, and an **asynchronous commit stage** (per-peer ordered senders, ack routing, an in-order commit sweep over a FIFO of in-flight records) that overlaps up to `W` writes per shard. The client-facing contract of Propose is unchanged: it still blocks until *its* write full-ISR-commits or fails. Concurrency comes from concurrent callers, which the benchmark workload already provides.

Every H/OH property is preserved; one is **strengthened**: the design adds a commit-time lease fence, closing a late-commit-after-lease-expiry window that exists in the current code (width: one RTT today; width: `W` writes after pipelining — which is why it must be added now).

One interface must change: the blocking `Transport.Replicate` (engine.go:81-83) cannot support ordered pipelined submission and is replaced by an ordered submit + completion-callback contract. The **wire format does not change at all**. The backup `Receive` path does not change at all.

## Root cause (why the current path is ping-pong)

- engine.go:271-272 — `e.writeMu.Lock(); defer e.writeMu.Unlock()` spans the entire function.
- engine.go:351-372 — the concurrent per-peer ship happens *inside* that critical section; `Propose` returns (and releases `writeMu`) only after every ISR ack or deadline.
- Consequence: per shard, at most one `ReplicateMsg` is ever outstanding per peer. The transport underneath is already fully capable of pipelining — `pbPeerLink` correlates by `reqID` and supports arbitrarily many outstanding requests (pb_link.go:54, pb_link.go:59, pb_link.go:124-148) — the engine just never gives it more than one.
- Secondary cost: 2 goroutines are spawned **per write** (engine.go:352-358), plus a `time.Timer` per roundTrip (pb_link.go:136). Per the fast-raft profiling, Go scheduler churn is the dominant write-path cost in this codebase; the redesign eliminates per-write goroutines entirely.

## The contract: properties that must survive

Enumerated from engine.go + DESIGN.md; each gets a correctness argument below.

| # | Property | Where enforced today |
|---|----------|---------------------|
| P1 | Dense, gap-free, total per-shard seq order; backups gap-check `PrevSeq == lastApplied` | engine.go:314-317, engine.go:421 |
| P2 | Local-apply-before-seq-assignment (failed apply burns no seq — no phantom-seq wedge) | engine.go:305-312 |
| P3 | OH1 lease fence: no writing without a valid lease on the engine's own monotonic clock | engine.go:289-296 |
| P4 | H3 min-ISR floor: reject below the durability floor | engine.go:297-301 |
| P5 | H6: only exact `(epoch,seq)` OK acks count | engine.go:355 |
| P6 | Full-ISR commit: every distinct member of the propose-time ISR must ack | engine.go:326-345, 373 |
| P7 | `committed` advances monotonically, in seq order, and never exposes a non-durable write | engine.go:389-395 |
| P8 | H1/H5 backup epoch fence + adopt-higher | engine.go:412-418 |
| P9 | Propose returns `(result, seq, err)`; on failure the write is a known-applied, non-durable uncommitted tail (OH2), result and seq still returned | engine.go:376-381 |
| P10 | `ErrReplicationTimeout` means *unknown*, not *guaranteed absent*: a timed-out write can later become transitively durable when a later write commits (markCommitted is a max) | engine.go:389-395 — existing, subtle, preserved |

## Architecture

```
Propose(ctx, data)                                   (many concurrent callers)
  │
  ├─ 1. admission: acquire window slot  (lastSeq - committed < W), ctx-bounded
  ├─ 2. writeMu ─────────────────────────────── SEQUENCING (µs, no I/O) ──┐
  │      fence: named-primary, lease, minISR          (unchanged checks)  │
  │      apply locally (Applier.Apply)                (P2: before seq)    │
  │      seq = ++lastSeq; backlog.append              (P1)                │
  │      rec = inflight{epoch, seq, required, doneCh}; queue.push(rec)    │
  │      for each peer: peerQ[peer].push(frame)       (in seq order)      │
  ├─ 3. writeMu released ──────────────────────────────────────────────-──┘
  └─ 4. select { <-rec.doneCh ; <-ctx.Done() }        (wait OUTSIDE lock)

per-peer sender goroutine (one per peer per engine, permanent):
  drain peerQ in order → transport ordered submit (may block = backpressure)

transport completion (runs in the link's reader goroutine):
  onAck(peer, ack, err) → mu: match exact (epoch,seq) to rec → sweep()

sweep() (under mu, driven by acks / timeouts / epoch changes):
  while head resolved:
    committed-success → commit-time lease fence → markCommitted → doneCh<-nil
    failed            → pop without advancing committed → doneCh<-err
  freed window slots wake blocked admissions
```

### In-flight record and queue

```go
type inflight struct {
    epoch    uint64
    seq      uint64
    pending  map[string]struct{} // propose-time ISR peers not yet exact-acked
    resolved bool                // exactly-once resolution latch
    err      error               // nil = committed
    doneCh   chan error          // buffered(1); Propose waits here
}
```

Records live in a seq-ordered FIFO (ring buffer; seqs are dense so index = `seq - baseSeq`, O(1) ack routing). Guarded by `e.mu` alongside the existing fields. Ack matching is **literal H6**: an ack removes `pending[peer]` iff `err == nil && ack.OK && ack.Epoch == rec.epoch && ack.Seq == rec.seq` — byte-identical logic to engine.go:355. No cumulative-ack inference in v1 (it would be sound — a backup's OK ack of seq N proves it applied 1..N by the gap-check invariant — but it is an optimization; exact matching preserves H6 verbatim).

## The seven questions

### 1. What leaves the critical section, what stays

**Stays under `writeMu`, in this order** (all CPU-only — no I/O, no channel ops that can block on a remote peer): fence checks → local apply → seq assign → backlog append → in-flight record registration → per-peer FIFO enqueue. The per-peer enqueue must be inside the lock: it is what guarantees frames enter every peer's queue in seq order (P1). The enqueue targets an engine-owned deque (≤ W entries, bounded by the window), so it never blocks on a slow peer while holding `writeMu`.

**Moves out:** everything network — submission to the link, the ack wait, timeout handling, and commit resolution. The local apply itself stays serialized (the FSM sees the identical serial apply order as today); only the *wait* leaves the lock.

### 2. Commit tracking with out-of-order acks

Acks from different peers arrive in any interleaving; acks from **one** peer for one shard are generated in seq order at the backup (Receive is serialized per shard, engine.go:408-431). With callback delivery from the link's single reader goroutine, per-peer observation is in-order in practice, but the design does not rely on it.

The committed watermark advances **only from the head of the FIFO**: `sweep` pops the head when its `pending` set is empty (full-ISR), so `committed` reaches N only after every seq in `(committed, N]` has either fully-acked or been explicitly resolved-failed (see Q3 for the hole rule). This is Raft's matchIndex/commitIndex computation specialized to full-ISR: commit index = min over required peers, enforced structurally by the FIFO rather than by a min-scan. Per-record state (who still owes an ack) lives in the record; nothing per-write lives in the transport.

### 3. Failure / timeout / reject of an in-flight write — holes

**(i) Ack lost or late (backup DID apply seq s).** Record s times out via its Propose ctx → resolved-failed → client gets `ErrReplicationTimeout` (= unknown, P10). The backup's `lastApplied` includes s, so later seqs are *not* gap-rejected; when a later record s′ fully-acks, `sweep` pops the failed s without advancing `committed`, then commits s′ — `committed` jumps from s−1 to s′. Exposing s transitively is **sound and is exactly today's behavior**: s′'s full-ISR acks prove (via the backup gap-check invariant) that every ISR member applied s. The current code produces the identical outcome through `markCommitted`'s max semantics (engine.go:389-395). Nothing wedges.

**(ii) Backup never applied s (nack, link death mid-stream, remote apply error).** That backup gap-rejects every subsequent seq (engine.go:421) — acceptable and unchanged. Under full-ISR commit nothing can commit past s−1, so the pipeline stalls, the window fills with resolved-failed records, and new Proposes block at admission until ctx deadline (new sentinel `ErrPipelineStalled`, mapped at the seam like a timeout). Recovery is exactly the recovery the protocol already prescribes: MetaRaft ISR-shrink removes the broken member (subsequent writes' propose-time required set excludes it; commit resumes and `committed` sweeps past the holes as in (i)), or P4 catch-up replays the ring to the backup. Already-in-flight records keep their propose-time required set and fail — matching today's per-write semantics. Re-evaluating in-flight required sets on an ISR shrink is deliberately **out of scope**: only safe if the shrink is MetaRaft-committed before the commit decision, and it adds a race for zero v1 benefit.

Compare today: the same broken backup makes *every* subsequent Propose individually apply-then-time-out, growing the uncommitted tail without bound. The pipelined design fails at most W writes and then stops admitting — strictly better (Q4).

**Resolution is exactly-once** via the `resolved` latch: whichever of {full-ack sweep, ctx timeout, transport-failure callback, epoch-change flush} fires first wins; later events on a resolved record are no-ops. This latch is the riskiest code in the design (see §Risk).

### 4. Backpressure

Window: **`lastSeq − committed ≤ W`** (suggest W = 256 initially; must be ≤ ring capacity 4096 so in-flight entries are always catch-up-replayable — trivially satisfied). Admission blocks (ctx-bounded) before `writeMu` when the window is full; the sweep wakes waiters as `committed` advances.

Choosing `lastSeq − committed` rather than "unresolved records ≤ W" is deliberate: it **hard-bounds the OH2 uncommitted tail at W** — an improvement over today, where every timed-out Propose grows the tail forever. A wedged shard (dead backup, no ISR shrink yet) fails at most W writes and then refuses admission — H3's "choose unavailability" philosophy applied to the pipeline. The demoted-primary snapshot-reload rule (OH2/P4) stands regardless; the bound caps divergence.

Secondary backpressure: the per-peer sender blocks on the link's bounded `sendCh` (pb_link.go:17, 256 deep) — outside all engine locks, stalling only that peer's drain, which full-ISR commit already gates on.

### 5. The lease fence under pipelining — argued through

The question: the lease is checked at seq-assignment (engine.go:293); is that still sufficient once the network wait happens after `writeMu` is released?

**No — and it was not fully sufficient before, either.** Key observation: in the *current* code the network wait already happens after the lease check, and `markCommitted` (engine.go:373-375, 389-395) never re-checks the lease. `writeMu` never bounded the check-to-commit window — the ctx deadline did (one RTT-ish). Concretely: primary P's lease expires at t1 while an ack is in flight from B1 (co-partitioned with P); MetaRaft honors the lease, waits out expiry + skew, grants epoch E+1 to a member that never saw seq s; at t3 > t1 the delayed ack arrives and P commits s and acks the client. The write is client-acked at E after E+1 exists, on a commit set (the old full ISR) that the new epoch's ISR need not contain — the OH1 loss, through the late-commit window. Today that window is one write wide and one RTT long; pipelining widens it to **W writes**, so it must be closed now:

**Commit-time lease fence.** `sweep` resolves a head record as *committed* only if `leaseEpoch == rec.epoch && epoch == rec.epoch && e.now() < leaseExpiry`. Otherwise the record (and, transitively, everything behind it in the same epoch) resolves as failed → clients get the unknown-outcome timeout, which is the honest answer. With this fence the per-write safety argument becomes fully compositional and independent of where the wait happens:

1. Seq assigned only under a valid lease (unchanged, engine.go:293).
2. Client-visible commit issued only under a valid lease for that record's epoch (new).
3. MetaRaft (pending half, unchanged obligation) grants E+1 only after E's lease provably lapsed (+ skew bound).
∴ No commit at E can be concurrent with or later than the existence of E+1 — the commit set (full propose-time ISR, P6) is evaluated entirely within E's tenancy, and full-ISR commit guarantees it intersects any member promotable from the MetaRaft-committed ISR. The number of writes in flight never appears in the argument.

Liveness cost: a transiently late lease renewal fails otherwise-healthy in-flight writes instead of committing them. That is the correct trade (unavailability over loss, same class as H3), and the renewal cadence already must beat expiry for Propose to work at all.

The same fence closes the analogous mid-flight `AdoptEpoch` race (epoch bumped while records are in flight — also unchecked at commit today). On any epoch/lease change, `sweep` flushes in-flight records of older epochs as failed. Same-node re-election (new epoch, same primary) starts a clean pipeline; seq continuity is preserved because `lastSeq` persists.

### 6. Protocol / Transport / Receive / ring changes

- **Wire format: zero changes.** Frames already carry `(epoch, seq, prevSeq)` and acks `(epoch, seq, ok)` (pb_frame.go:127-178); `reqID` correlation untouched.
- **`Transport` interface: must change.** The blocking `Replicate` (engine.go:81-83) cannot express "submit in order, complete later" — with the wait outside `writeMu`, two Propose goroutines calling blocking Replicate could submit out of order to the same peer and trip the gap check. Replace with:

  ```go
  // Calls from one goroutine to one peer are delivered in call order.
  // done is invoked exactly once — ack, nack, or transport error —
  // from the transport's delivery goroutine. May block for backpressure.
  Replicate(peer string, msg ReplicateMsg, done func(AckMsg, error)) error
  ```

  `pbPeerLink` needs a small change: `pending` holds the callback instead of a waiter channel; `deliver` (pb_link.go:160-170) invokes it in the reader goroutine; `fail` (pb_link.go:178-198) invokes all pending callbacks with the error — prompt failure of all in-flight records on link death for free. Per-request timers (pb_link.go:136) disappear; deadlines are owned by the engine (Propose ctx). `inmem_transport.go` gets the same treatment.
- **Backup `Receive` path: zero changes.** engine.go:408-431 already serializes per shard, gap-checks, and acks — it handles pipelined arrivals natively. The strongest simplicity argument for this design.
- **Server response path: zero changes** — serveConn already pipelines and batches responses (net_transport.go:89-142).
- **Catch-up ring: zero structural changes.** Constraint recorded: `W ≤ ringCapacity`. Ring append stays inside the sequencing critical section (as at engine.go:318).
- **Deleted:** the 2-goroutines-per-write ship pattern (engine.go:351-358) and its per-write timer. Steady-state new goroutines: one sender per peer per engine (≈ 2 peers × 8 shards = 16, permanent) — a large scheduler-churn reduction, which fast-raft profiling says is where the write path's CPU actually goes.

### 7. Does the writev batcher become useful again?

Partially — and only off-loopback. After pipelining, a peer's `sendCh` genuinely accumulates frames (up to W per shard × shards sharing the link), so the greedy linger=0 drain (pb_batcher.go:167-220) will finally see multi-frame batches without any forced linger; expect frames/writev well above 1.09 under load. On **loopback**, the honest position stands: the path was latency-bound and the copy dominates; pipelining removes the latency bound, making the path CPU-bound, where fewer syscalls/wakeups help modestly at best — do not expect the batcher to be a headline win in the loopback benchmark. On a **real NIC** (the dedicated cloud box), batching finally has both raw material (queued frames during real RTTs) and a payoff (per-packet/syscall costs loopback hides). Keep linger=0; re-measure frames/writev on real hardware before spending more on batching. (And per the hardware findings: benchmark on the dedicated box, not the laptop.)

## Correctness arguments (per preserved property)

- **P1 (dense seq order, gap-free at backups):** seq assignment and per-peer FIFO enqueue are atomic under `writeMu`; one sender per peer drains FIFO in order; TCP + single link per peer preserves order; backup applies serially under gap check. Reordering two seqs to the same peer would require two senders or enqueue outside the lock — both structurally excluded.
- **P2 (no phantom seq):** apply-then-assign, both inside the same critical section, failure path exits before assignment — byte-for-byte today's logic (engine.go:305-319).
- **P3 (lease at assignment):** unchanged check, same lock. **Strengthened** by the commit-time fence (Q5).
- **P4 (min-ISR floor):** unchanged check at admission, under the lock, against the same control snapshot.
- **P5 (H6 exact acks):** literal reimplementation of engine.go:355 against per-record state; no cumulative inference in v1.
- **P6 (full-ISR):** required set = deduped propose-time ISR minus self (same construction as engine.go:326-334); commit iff `pending` empty. The `needed == 0` single-node path resolves at sweep like any other record.
- **P7 (in-order committed, never non-durable):** head-only sweep ⇒ `committed` is monotone and reaches N only when 1..N are each fully-acked or resolved-failed; a failed seq is exposed only transitively via a later full-ISR commit, which proves its presence on every ISR member (gap-check invariant) — the exact semantics of today's `markCommitted` max (engine.go:389-395), now with the OH2 tail additionally bounded by W.
- **P8:** backup path untouched.
- **P9/P10:** Propose still returns `(result, seq, err)` with result from its own local apply; failure still means "unknown, uncommitted tail"; transitive later durability preserved as P7.

## Staging

**Engine pipeline + async transport core (the change).** In-flight FIFO, window admission, per-peer FIFOs + senders, callback Transport (net + inmem), sweep with commit-time lease fence, epoch-change flush. All 16 existing engine tests must pass unmodified in meaning. New tests: concurrent-callers pipelining depth > 1 (assert overlap via a blocking fake transport); out-of-order cross-peer acks; hole class (i) (lost ack → later transitive commit, assert `committed` jump); hole class (ii) (dead peer → W failures then `ErrPipelineStalled`, then ISR-shrink recovery); commit-fence-on-expiry (lease lapses with acks in flight → all fail, `committed` frozen); mid-flight `GrantLease`/`AdoptEpoch`; window-slot exactly-once release; `-race` on all of the above. A deterministic model test with a scripted fake transport driving adversarial interleavings is strongly recommended.

**Measurement.** Re-run the 3-node benchmark on the dedicated box; record frames/writev to see whether the batcher engages; compare against `-nosync -volatile-log` Raft. No code.

**Optional, separately gated:** (a) group-commit frames — batch k consecutive seqs into one frame (new frame kind, version bump; backup applies the batch under one `mu` hold); (b) in-engine nack/timeout resend from the ring (Raft-style nextIndex backoff) instead of fail-fast — a semantics change (fewer client-visible failures) requiring its own review; (c) cumulative-ack cursors if per-record maps show up in profiles.

## Riskiest part

The **record-resolution state machine**: four concurrent resolvers (full-ack sweep, Propose ctx timeout, transport-failure callback, epoch/lease-change flush) must each resolve a record at most once, free its window accounting exactly once, and never leave the sweep stalled behind a resolved record or a `doneCh` unsignaled. The happy path is easy; the failure interleavings are combinatorial. Mitigation: the single `resolved` latch under `e.mu` as the only resolution point, buffered `doneCh`, the deterministic model test, and `-race` throughout. Second riskiest: the commit-time lease fence is new hot-path code implementing the OH1 argument — it needs its own adversarial test (delayed-ack-after-expiry) proving no commit escapes.

## Trade-offs

| Decision | Pros | Cons |
|----------|------|------|
| Pipeline with per-record FIFO vs keep ping-pong | Throughput bounded by pipe/CPU not RTT; batching can engage; tail bounded | New concurrency surface in a consensus-adjacent core; more state |
| Callback Transport vs keep blocking `Replicate` | Ordered submission guaranteed; kills 2 goroutines + 1 timer per write | Interface break (net + inmem transports touched); callbacks run on reader goroutine (must stay cheap) |
| Window = `lastSeq − committed` vs unresolved-count | Hard-bounds OH2 tail; wedged shard stops burning seqs | Wedged shard rejects writes until ISR shrink (deliberate, H3-consistent) |
| Commit-time lease fence added | Closes late-commit OH1 window (pre-existing, widened by pipelining); makes the safety argument per-write and W-independent | Late lease renewal fails healthy in-flight writes (unavailability over loss); hot-path clock read |
| Per-write required-set snapshot vs live-ISR re-eval | Exactly today's semantics; no shrink/commit race | A mid-flight ISR shrink doesn't rescue already-in-flight writes (same as today) |
| Fail-fast on nack (no resend) in v1 | Identical failure semantics to current code | Transient single-frame loss fails writes Raft would quietly retry |
