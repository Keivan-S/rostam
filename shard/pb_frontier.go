// SPDX-License-Identifier: Apache-2.0

package shard

import (
	"sync"
	"time"

	"github.com/rostamlabs/rostam/cache"
)

// Defaults for the PB durable-frontier stamp (knobs:
// Config.PBFrontierStampEvery / PBFrontierStampInterval).
const (
	// defaultPBFrontierStampEvery DISABLES the count trigger. The default policy is
	// purely time-based, and that is a measured decision, not a guess.
	//
	// A full-region msync costs O(REGION SIZE), not O(bytes changed): it walks the
	// whole VMA (256 MiB per shard by default) looking for dirty pages. So stamping
	// more often does not cost proportionally less per stamp — it just multiplies a
	// fixed, large cost. Measured on a single-shard PB store (300k Puts, ns/op,
	// median of 3):
	//
	//	no stamping at all ........................ 1599
	//	time-based, 1s ............................ 1538
	//	time-based, 100ms ......................... 1554   (free — inside the noise)
	//	+ count trigger every 1024 applies ........ 2760   (+73%)
	//	+ count trigger every 64 applies .......... 3175   (+99%)
	//
	// The count trigger's bound is also weaker than it looks under exactly the load
	// where it costs the most: at ~500k writes/s, 1024 writes elapse in ~2 ms, far
	// below any sane interval, so the flusher's wakeups coalesce and the effective
	// bound collapses back to "whatever one msync takes" — while the flusher runs
	// msyncs back to back, competing with the writers for the mmap lock and memory
	// bandwidth. The ticker already bounds the under-report by "writes in one
	// interval" at every rate, so the count trigger can only ever make the watermark
	// tighter at a cost measured above, never make it correct.
	//
	// It is kept as an opt-in knob because a deployment with a small region (cheap
	// msync) may genuinely prefer a write-count bound — e.g. to keep the restart
	// delta inside the primary's catch-up ring (pbisr defaultRingCapacity = 4096),
	// which is what a lagging restart is served from.
	defaultPBFrontierStampEvery = 0
	// defaultPBFrontierStampInterval bounds the under-report in TIME: a crash loses
	// at most this much frontier advance. Measured free at full write rate (above).
	//
	// A 1s default was tried and REVERTED — recorded here so it is not re-tried on
	// the same reasoning. The hypothesis was that the ns/op table above, being a
	// MEAN on a SINGLE-shard store, cannot see this flusher's real cost (one
	// full-region msync PER SHARD per tick) and that the tick rate therefore drove
	// PB write TAIL latency on a real cluster. A first paired A/B seemed to confirm
	// it: stamping-off beat 100ms 3/3 on ops/s, p99 and p999, with p999 landing on
	// the pre-Stage-4 baseline.
	//
	// It did not survive. Raising the default to 1s (10x fewer msyncs) produced NO
	// improvement — 1s won 1 of 3 pairs and was marginally worse at the median. And
	// the control arm itself was unstable BETWEEN sessions: the identical 100ms
	// binary posted a p999 median of 2790µs in one run and 1689µs in the next, a
	// ~1.7x swing that overlaps the whole effect being chased.
	//
	// RESOLVED on a quiet dedicated host — one binary, the arms differing ONLY by
	// -pb-frontier-stamp-interval, 6 pairs, randomized within-pair order, warm-up
	// discarded (12-vCPU EPYC, 3-node PB RF=2, 8 shards, 64 conns, KV puts):
	//
	//	              ops/s        p99      p999
	//	100ms ..... 101,612     1931µs    3123µs
	//	1s ........ 105,735     1801µs    8306µs
	//
	// Every metric was 6/6 — and they point in OPPOSITE directions. Fewer ticks
	// means less steady-state interference (throughput +4.1%, p99 −6.7%), but each
	// msync then covers 10x more dirty pages, so each individual stall is far
	// longer and p999 blows out 2.7x. The interval trades stall FREQUENCY against
	// stall SIZE; there is no free setting.
	//
	// That also settles the earlier open question: the stamper IS the p999 driver,
	// since a runtime flag alone moves the tail 2.7x with nothing else changed.
	//
	// 100ms stays the default because a 2.7x tail regression is the worse trade for
	// a database, and because the earlier ns/op table shows the mean barely moves.
	// A throughput-bound deployment that does not care about p999 can raise it via
	// Config.PBFrontierStampInterval / -pb-frontier-stamp-interval (this knob used
	// to be dead config, which is why the question went unanswered so long).
	//
	// MEASUREMENT NOTE, since three laptop A/Bs got this wrong before the server
	// settled it: a same-session paired A/B controls for drift INSIDE a run and
	// cannot rescue a 3-sample median when between-run variance exceeds the effect.
	// It also matters WHICH metric you look at — a single-metric framing here would
	// have reported 1s as a clean win.
	defaultPBFrontierStampInterval = 100 * time.Millisecond
)

// pbFrontierStamper persists the PB engine's applied frontier into the cache
// header, AMORTISED.
//
// WHY AMORTISED, AND WHY NOT CHEAPER. cache.SetPBFrontier is crash-ordered: it
// msyncs the page data before writing and msyncing the header, so the persisted
// watermark can never name a write whose data is not on disk. That ordering is the
// entire safety property, and it costs a full-region msync per call. The PB engine
// applies ONE write at a time (unlike the Raft path, which amortises the same call
// over a whole Apply batch), and PB exists for write throughput — so a durable
// stamp per write would cost more than the engine is worth.
//
// The resolution is to amortise the ORDERED write rather than to weaken it: stamp
// periodically (every T, and optionally also every N applies), using the same
// crash-ordered path. The persisted watermark then trails the true frontier by at
// most one interval, and trailing is the SAFE direction — a restarted node
// under-reports, is offered a delta from further back, and pbisr's log matching
// accepts it as a true prefix. The unsafe direction (dropping the msync ordering
// so the header can reach disk ahead of the pages it certifies) buys the same
// amortisation at the cost of the property.
//
// SELF-LIMITING. The flusher is a single goroutine woken through a
// capacity-1 channel, so wakeups that arrive while a stamp is in flight COALESCE
// into one. The stamp rate is therefore bounded by how fast msync completes, never
// by how fast writes arrive: a write burst cannot queue up an msync backlog.
//
// KNOWN RESIDUAL (snapshot transfer's, not this path's). An under-report is safe when the
// re-shipped delta is the SAME delta — the writes are re-applied deterministically
// and the node converges. It is not a repair for a FORK: a node whose watermark
// trails into a range where the cluster's history diverged will present a frontier
// the primary considers a valid prefix, and the append will land on top of state
// the primary does not have. Detecting that needs the node's true frontier, which a
// crash by definition did not persist; repairing it needs a full state transfer.
// This design strictly SHRINKS that window (from "the whole log", which is what a
// (0,0) frontier over real data meant, to "one stamp interval"), and Store.Close's
// synchronous final flush removes it entirely for a clean shutdown.
type pbFrontierStamper struct {
	cache    *cache.Cache
	everyN   uint64
	interval time.Duration

	// flushMu serializes the STAMPING ITSELF (the crash-ordered SetPBFrontier
	// call), which mu deliberately does not: mu is dropped before the msync so a
	// write recording a newer frontier never waits on the disk.
	//
	// Snapshot transfer made that separation load-bearing. A snapshot install writes the
	// frontier through installFrontier, and without flushMu a periodic flush that
	// had already snapshotted the PRE-install pair could land its stale (and, for a
	// diverged target, HIGHER) value after the install's — an over-report, the one
	// direction the durable watermark must never take. flushMu + reading pend
	// INSIDE it makes the two orderable: whichever acquires last writes last, and
	// pend by then always names the installed frontier.
	flushMu sync.Mutex

	mu           sync.Mutex
	pendSeq      uint64 // newest frontier recorded (may not be persisted yet)
	pendEpoch    uint64
	stampedSeq   uint64 // newest frontier handed to cache.SetPBFrontier
	stampedEpoch uint64
	sinceStamp   uint64 // applies recorded since the last stamp (only if everyN > 0)

	wake     chan struct{}
	stop     chan struct{}
	stopOnce sync.Once
	wg       sync.WaitGroup
}

// newPBFrontierStamper starts the background flusher. everyN <= 0 disables the
// count trigger (the default — see defaultPBFrontierStampEvery); interval <= 0
// selects the default interval. c may be a heap-mode cache, in which case
// SetPBFrontier is a no-op and this whole path costs one mutex per write.
func newPBFrontierStamper(c *cache.Cache, everyN int, interval time.Duration) *pbFrontierStamper {
	if everyN < 0 {
		everyN = defaultPBFrontierStampEvery
	}
	if interval <= 0 {
		interval = defaultPBFrontierStampInterval
	}
	s := &pbFrontierStamper{
		cache:    c,
		everyN:   uint64(everyN), //nolint:gosec // guarded non-negative above
		interval: interval,
		wake:     make(chan struct{}, 1),
		stop:     make(chan struct{}),
	}
	s.wg.Add(1)
	go s.run()
	return s
}

// record is the pbisr.WithFrontierSink callback. It runs UNDER THE ENGINE LOCK on
// the write path, so it does exactly two things: publish the pair, and possibly
// poke the flusher. No I/O, no blocking, no allocation.
//
// The pair it publishes is always covered by the local FSM — the engine calls it
// only after the Applier returned — which is the precondition cache.SetPBFrontier
// documents and the reason the persisted value can never over-report.
func (s *pbFrontierStamper) record(seq, epoch uint64) {
	s.mu.Lock()
	s.pendSeq = seq
	s.pendEpoch = epoch
	due := false
	if s.everyN > 0 {
		s.sinceStamp++
		due = s.sinceStamp >= s.everyN
	}
	s.mu.Unlock()
	if due {
		select {
		case s.wake <- struct{}{}:
		default: // a flush is already pending or in flight; it will pick this up
		}
	}
}

func (s *pbFrontierStamper) run() {
	defer s.wg.Done()
	t := time.NewTicker(s.interval)
	defer t.Stop()
	for {
		select {
		case <-s.stop:
			return
		case <-s.wake:
			s.flush()
		case <-t.C:
			s.flush()
		}
	}
}

// flush stamps the pending frontier if it has moved since the last stamp. The
// value is snapshotted under s.mu and the msync happens OUTSIDE it, so a write
// recording a newer frontier never waits on the disk.
//
// Snapshotting before the msync is also what makes the ordering argument hold end
// to end: every write up to the snapshotted pair had already returned from apply
// when the pair was recorded, hence its pages were already dirty in the mapping
// when SetPBFrontier's msync began, hence they are on disk before the header names
// them. Writes that land DURING the msync only advance the frontier past what is
// being stamped — the wrong direction to cause an over-report.
func (s *pbFrontierStamper) flush() {
	s.flushMu.Lock()
	defer s.flushMu.Unlock()
	s.mu.Lock()
	seq, epoch := s.pendSeq, s.pendEpoch
	if seq == s.stampedSeq && epoch == s.stampedEpoch {
		s.mu.Unlock()
		return // nothing new; skip the msync entirely on an idle shard
	}
	s.stampedSeq, s.stampedEpoch = seq, epoch
	s.sinceStamp = 0
	s.mu.Unlock()
	s.cache.SetPBFrontier(seq, epoch)
}

// installFrontier REBASES the stamper onto a snapshot-installed frontier and
// persists it, crash-ordered, under the flush lock.
//
// It exists because the amortised design has one state a plain SetPBFrontier
// cannot survive: after a snapshot install the stamper still holds the
// PRE-install pending pair, and its very next tick would flush that back over the
// installed value. On the target this path exists to repair — a DIVERGED node —
// the stale pair is HIGHER than the installed frontier, so the resurrection would
// be an OVER-report, which is the direction that lets log matching certify a
// divergent append.
//
// Rebasing pend AND stamped together is what makes the later flushes no-ops
// (flush skips when they are equal), and holding flushMu across the write is what
// orders it against a flush that is already in flight with the old value. The
// engine calls it with writeMu+e.mu held, inside the poison fence, so the msync
// cost is part of the documented install stall and never on the write path.
func (s *pbFrontierStamper) installFrontier(seq, epoch uint64) {
	s.flushMu.Lock()
	defer s.flushMu.Unlock()
	s.mu.Lock()
	s.pendSeq, s.pendEpoch = seq, epoch
	s.stampedSeq, s.stampedEpoch = seq, epoch
	s.sinceStamp = 0
	s.mu.Unlock()
	s.cache.SetPBFrontier(seq, epoch)
}

// Close stops the flusher and stamps the final frontier synchronously, so a CLEAN
// shutdown persists an EXACT watermark rather than one up to an interval stale.
// (An unclean stop still under-reports by at most an interval, which is the whole
// point of the design.) Idempotent.
//
// The final flush runs only after the goroutine has exited, so there is never more
// than one stamper in cache.SetPBFrontier at a time. The caller must have stopped
// the engine first — Store.Close shuts replication down before reaching here — so
// no further frontier advance can be missed.
func (s *pbFrontierStamper) Close() {
	s.stopOnce.Do(func() {
		close(s.stop)
		s.wg.Wait()
		s.flush()
	})
}
