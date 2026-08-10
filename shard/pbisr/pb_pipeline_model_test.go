// SPDX-License-Identifier: Apache-2.0

package pbisr

import (
	"context"
	"errors"
	"sync"
	"testing"
	"time"
)

// This file is the deterministic ADVERSARIAL model test for the record-resolution
// state machine — the part PIPELINE-REDESIGN §"Riskiest part" calls the highest
// risk: four concurrent resolvers (full-ack sweep, Propose ctx timeout,
// transport-error/nack callback, epoch/lease-change flush) must each resolve a
// record at most once, advance committed only from a fully-acked head behind the
// lease fence, and never leave a doneCh unsignaled or the sweep stalled behind a
// resolved record.
//
// Unlike pb_pipeline_test.go (which drives the internals directly under e.mu),
// these tests drive the REAL Propose path — sequencing, the per-peer senders, and
// completeSend — but replace the transport with a SCRIPTED one that captures each
// submission's completion callback and fires it only when the test releases it,
// in a test-chosen order. That makes adversarial interleavings deterministic
// (no sleeps, no timing) while still exercising the concurrent completion
// callbacks and the concurrent Propose waiters.
//
// Invariants asserted after every scenario:
//   - committed is MONOTONE and never exceeds the fully-acked PREFIX (head-only);
//   - every Propose's doneCh is signaled EXACTLY ONCE (no lost signal — proved by
//     every goroutine returning; no double — proved by unique dense seqs + the
//     resolved latch, run under -race);
//   - no non-durable seq is ever exposed via committed (the commit-time lease fence);
//   - senders exit on Shutdown (no goroutine leak).

// errModelLink models a link-death transport error (a completion firing with a
// non-nil error, as NetTransport.fail delivers on a dead peer link).
var errModelLink = errors.New("pbisr: model link closed")

// --- scripted transport -----------------------------------------------------

// submitKey identifies one captured submission. Seqs are dense and unique per
// (peer), so (peer, seq) is a unique key for the in-flight frame to that peer.
type submitKey struct {
	peer string
	seq  uint64
}

// scriptedSubmit is one captured, not-yet-completed submission: the engine's
// completion callback plus the frame it was submitted for (so a release can
// synthesize the exactly-matching (epoch,seq) ack).
type scriptedSubmit struct {
	done func(AckMsg, error)
	msg  ReplicateMsg
}

func (s scriptedSubmit) okAck() AckMsg  { return AckMsg{Epoch: s.msg.Epoch, Seq: s.msg.Seq, OK: true} }
func (s scriptedSubmit) nakAck() AckMsg { return AckMsg{Epoch: s.msg.Epoch, Seq: s.msg.Seq, OK: false} }

// scriptedTransport implements the async Transport contract but NEVER completes a
// submission on its own: Replicate captures (done, msg) keyed by (peer, seq) and
// returns nil immediately (non-blocking, as the contract requires). The test then
// releases stored completions in whatever order it scripts. Because a release
// runs completeSend synchronously on the test goroutine, the sweep it drives has
// fully executed by the time the release call returns — so committed can be read
// deterministically after each step.
type scriptedTransport struct {
	mu      sync.Mutex
	pending map[submitKey]scriptedSubmit
}

func newScriptedTransport() *scriptedTransport {
	return &scriptedTransport{pending: make(map[submitKey]scriptedSubmit)}
}

func (s *scriptedTransport) Replicate(peer string, msg ReplicateMsg, done func(AckMsg, error)) error {
	s.mu.Lock()
	s.pending[submitKey{peer, msg.Seq}] = scriptedSubmit{done: done, msg: msg}
	s.mu.Unlock()
	return nil
}

// count reports how many submissions are currently captured (not yet released).
func (s *scriptedTransport) count() int {
	s.mu.Lock()
	defer s.mu.Unlock()
	return len(s.pending)
}

// take removes and returns the captured completion for (peer, seq). The caller
// fires it OUTSIDE s.mu (completeSend takes e.mu; holding s.mu across the engine
// callback would invert lock order).
func (s *scriptedTransport) take(peer string, seq uint64) (scriptedSubmit, bool) {
	s.mu.Lock()
	defer s.mu.Unlock()
	k := submitKey{peer, seq}
	sub, ok := s.pending[k]
	if ok {
		delete(s.pending, k)
	}
	return sub, ok
}

// releaseAck fires (peer, seq)'s completion with a positive, H6-exact ack.
func (s *scriptedTransport) releaseAck(t *testing.T, peer string, seq uint64) {
	t.Helper()
	sub, ok := s.take(peer, seq)
	if !ok {
		t.Fatalf("releaseAck: no captured submission for peer=%s seq=%d", peer, seq)
	}
	sub.done(sub.okAck(), nil)
}

// releaseNack fires with a definitive negative ack (a required peer rejecting the
// exact (epoch,seq) — e.g. a gap/epoch rejection at the backup).
func (s *scriptedTransport) releaseNack(t *testing.T, peer string, seq uint64) {
	t.Helper()
	sub, ok := s.take(peer, seq)
	if !ok {
		t.Fatalf("releaseNack: no captured submission for peer=%s seq=%d", peer, seq)
	}
	sub.done(sub.nakAck(), nil)
}

// releaseErr fires with a transport error (one frame's link death).
func (s *scriptedTransport) releaseErr(t *testing.T, peer string, seq uint64, err error) {
	t.Helper()
	sub, ok := s.take(peer, seq)
	if !ok {
		t.Fatalf("releaseErr: no captured submission for peer=%s seq=%d", peer, seq)
	}
	sub.done(AckMsg{}, err)
}

// closeAll fires EVERY remaining captured completion with err, modeling a link
// close that fails all in-flight submissions at once (NetTransport.fail on a dead
// peer link). Map iteration order is intentionally arbitrary: the asserted
// invariants must hold under any completion order.
func (s *scriptedTransport) closeAll(err error) {
	s.mu.Lock()
	subs := make([]scriptedSubmit, 0, len(s.pending))
	for k, sub := range s.pending {
		subs = append(subs, sub)
		delete(s.pending, k)
	}
	s.mu.Unlock()
	for _, sub := range subs {
		sub.done(AckMsg{}, err)
	}
}

// --- model harness ----------------------------------------------------------

// modelHarness drives concurrent Proposes over a scriptedTransport and collects
// each Propose's outcome keyed by its assigned seq. It reuses fakeControl /
// fakeClock / fakeApplier from engine_test.go (same package).
type modelHarness struct {
	t   *testing.T
	eng *Engine
	tr  *scriptedTransport
	clk *fakeClock

	mu       sync.Mutex
	results  map[uint64]error // assigned seq -> resolved outcome (recorded once)
	returned int
	wg       sync.WaitGroup
}

// newModelHarness builds a single primary "n1" with the given ISR, a valid lease
// (epoch 1, far-future expiry), and the scripted transport. minISR is 2; every
// distinct backup in isr is a required full-ISR peer.
func newModelHarness(t *testing.T, isr []string) *modelHarness {
	t.Helper()
	ctrl := &fakeControl{epoch: 1, primary: "n1", isr: isr, minISR: 2}
	tr := newScriptedTransport()
	clk := &fakeClock{t: t0}
	ap := &fakeApplier{}
	eng := New("n1", testShard, ctrl, tr, ap, WithClock(clk.now))
	eng.GrantLease(1, t0+leaseDur)
	return &modelHarness{t: t, eng: eng, tr: tr, clk: clk, results: make(map[uint64]error)}
}

// launch starts one Propose on its own goroutine, recording its (seq, err) when
// it returns. Each goroutine gets a distinct dense seq, so results is keyed 1:1.
func (h *modelHarness) launch(ctx context.Context) {
	h.wg.Add(1)
	go func() {
		defer h.wg.Done()
		_, seq, err := h.eng.Propose(ctx, []byte("op"))
		h.mu.Lock()
		if _, dup := h.results[seq]; dup && seq != 0 {
			h.t.Errorf("seq %d returned from Propose more than once (double signal)", seq)
		}
		h.results[seq] = err
		h.returned++
		h.mu.Unlock()
	}()
}

// waitSubmissions blocks until the scripted transport has captured want
// submissions (numRecords * numPeers) — i.e. every launched Propose has been
// sequenced and enqueued to every peer's sender. Only after this is it safe to
// script releases (all records are registered in the FIFO).
func (h *modelHarness) waitSubmissions(want int) {
	h.t.Helper()
	deadline := time.Now().Add(5 * time.Second)
	for h.tr.count() < want {
		if time.Now().After(deadline) {
			h.t.Fatalf("captured %d/%d submissions before deadline", h.tr.count(), want)
		}
		time.Sleep(time.Millisecond)
	}
}

// wait blocks until every launched Propose has returned, with a watchdog. A hang
// here is the direct signal of a LOST doneCh signal (a resolver that failed to
// wake its waiter).
func (h *modelHarness) wait() {
	h.t.Helper()
	done := make(chan struct{})
	go func() { h.wg.Wait(); close(done) }()
	select {
	case <-done:
	case <-time.After(5 * time.Second):
		h.t.Fatal("not all Propose goroutines returned — a doneCh signal was lost")
	}
}

// committed reads the durable frontier (deterministic after a synchronous release).
func (h *modelHarness) committed() uint64 { return h.eng.Committed() }

// result returns the recorded outcome for an assigned seq (call after wait).
func (h *modelHarness) result(seq uint64) error {
	h.mu.Lock()
	defer h.mu.Unlock()
	return h.results[seq]
}

// shutdownWithin asserts Shutdown (which joins every per-peer sender) returns
// within d — a hang means a sender goroutine leaked or is stuck.
func (h *modelHarness) shutdownWithin(d time.Duration) {
	h.t.Helper()
	done := make(chan struct{})
	go func() { h.eng.Shutdown(); close(done) }()
	select {
	case <-done:
	case <-time.After(d):
		h.t.Fatalf("Shutdown did not return within %v — a sender goroutine leaked", d)
	}
}

// --- Scenario 1: ack racing the ctx timeout for the SAME record --------------

// TestModelAckThenTimeoutIsNoOp: the ack arrives (and commits) STRICTLY before
// the ctx timeout fires. The later timeout must be a no-op — Propose reads the
// resolved outcome back and returns the committed (nil) result, not a spurious
// timeout error.
func TestModelAckThenTimeoutIsNoOp(t *testing.T) {
	h := newModelHarness(t, []string{"n1", "b1"})
	defer h.eng.Shutdown()
	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()

	h.launch(ctx)
	h.waitSubmissions(1)

	h.tr.releaseAck(t, "b1", 1) // ack wins: full-ISR commit
	if got := h.committed(); got != 1 {
		t.Fatalf("committed = %d, want 1 (ack committed the record)", got)
	}
	cancel() // timeout fires just after — must not override the committed outcome
	h.wait()
	if err := h.result(1); err != nil {
		t.Fatalf("seq 1 outcome = %v, want nil (committed; a late timeout must not win)", err)
	}
}

// TestModelTimeoutThenAckIsNoOp: the ctx timeout fires STRICTLY before the ack.
// Propose resolves failed and committed stays 0; the late ack then pops the
// resolved head cleanly WITHOUT advancing committed and without a double-signal.
func TestModelTimeoutThenAckIsNoOp(t *testing.T) {
	h := newModelHarness(t, []string{"n1", "b1"})
	defer h.eng.Shutdown()
	ctx, cancel := context.WithCancel(context.Background())

	h.launch(ctx)
	h.waitSubmissions(1)

	cancel() // timeout wins
	h.wait()
	if err := h.result(1); !errors.Is(err, ErrReplicationTimeout) {
		t.Fatalf("seq 1 outcome = %v, want ErrReplicationTimeout", err)
	}
	if got := h.committed(); got != 0 {
		t.Fatalf("committed = %d, want 0 (a timed-out record must not commit)", got)
	}

	// The ack now arrives late (backup DID apply): a clean no-op.
	h.tr.releaseAck(t, "b1", 1)
	if got := h.committed(); got != 0 {
		t.Fatalf("committed = %d after late ack, want 0 (resolved head popped, not committed)", got)
	}
}

// TestModelConcurrentAckTimeoutRace fires the ack completion and the ctx timeout
// CONCURRENTLY for the same single-peer record, many times. The two paths race
// resolveLocked under e.mu; exactly one wins. The asserted invariant is that the
// outcome and committed are always CONSISTENT: nil outcome IFF committed advanced
// to the record's seq. Run under -race, this shakes the exactly-once latch.
func TestModelConcurrentAckTimeoutRace(t *testing.T) {
	const iters = 100
	for iter := 0; iter < iters; iter++ {
		h := newModelHarness(t, []string{"n1", "b1"})
		ctx, cancel := context.WithCancel(context.Background())

		h.launch(ctx)
		h.waitSubmissions(1)

		// Capture the completion on the test goroutine (no *testing.T use inside the
		// racing goroutines — that would violate the from-test-goroutine rule).
		sub, ok := h.tr.take("b1", 1)
		if !ok {
			t.Fatalf("iter %d: no captured submission for b1/seq1", iter)
		}

		var start sync.WaitGroup
		start.Add(1)
		var rg sync.WaitGroup
		rg.Add(2)
		go func() { defer rg.Done(); start.Wait(); sub.done(sub.okAck(), nil) }()
		go func() { defer rg.Done(); start.Wait(); cancel() }()
		start.Done()
		rg.Wait()
		h.wait()

		err := h.result(1)
		committed := h.committed()
		switch {
		case err == nil && committed != 1:
			t.Fatalf("iter %d: nil outcome but committed = %d (want 1)", iter, committed)
		case err != nil && committed != 0:
			t.Fatalf("iter %d: failed outcome (%v) but committed = %d (want 0)", iter, err, committed)
		}
		h.eng.Shutdown()
	}
}

// --- Scenario 2: out-of-order cross-peer acks (head-only in-order commit) -----

// TestModelOutOfOrderCrossPeerAcks scripts acks arriving out of seq order AND out
// of peer order: a higher seq (3) fully acks before the head (1), and the two
// peers ack different seqs in interleaved order. committed must stay 0 until the
// HEAD itself is fully acked, then staircase 1 -> 3 as the now-acked tail commits
// transitively. This is the direct proof of head-only, in-order, never-exceeds-
// the-fully-acked-prefix commit (P7).
func TestModelOutOfOrderCrossPeerAcks(t *testing.T) {
	h := newModelHarness(t, []string{"n1", "b1", "b2"})
	defer h.eng.Shutdown()
	ctx, cancel := context.WithTimeout(context.Background(), 30*time.Second)
	defer cancel()

	const n = 3
	for i := 0; i < n; i++ {
		h.launch(ctx)
	}
	h.waitSubmissions(n * 2) // 3 seqs x 2 peers

	step := func(peer string, seq, wantCommitted uint64) {
		t.Helper()
		h.tr.releaseAck(t, peer, seq)
		if got := h.committed(); got != wantCommitted {
			t.Fatalf("after ack peer=%s seq=%d: committed = %d, want %d", peer, seq, got, wantCommitted)
		}
	}

	step("b2", 3, 0) // seq3 half-acked, and not the head
	step("b1", 3, 0) // seq3 FULLY acked — but head is seq1, so head-only holds committed at 0
	step("b2", 2, 0) // seq2 half-acked
	step("b2", 1, 0) // seq1 half-acked
	step("b1", 1, 1) // seq1 FULLY acked -> commit 1; new head seq2 still owes b1 -> stop
	step("b1", 2, 3) // seq2 FULLY acked -> commit 2; head seq3 already full -> commit 3

	h.wait()
	for s := uint64(1); s <= n; s++ {
		if err := h.result(s); err != nil {
			t.Fatalf("seq %d outcome = %v, want nil (committed)", s, err)
		}
	}
	if got := h.committed(); got != n {
		t.Fatalf("final committed = %d, want %d", got, n)
	}
}

// --- Scenario 3: epoch bump / flush mid-flight -------------------------------

// TestModelEpochFlushMidFlight bumps the epoch while records are pending (a
// partially-acked one included). flushEpochLocked must fail EVERY superseded-epoch
// record; committed stays 0 (nothing of the old epoch may commit under the fence);
// each Propose resolves ErrLeaseExpired exactly once. Late completions arriving
// after the flush (the records are already popped) must be clean no-ops.
func TestModelEpochFlushMidFlight(t *testing.T) {
	h := newModelHarness(t, []string{"n1", "b1", "b2"})
	ctx, cancel := context.WithTimeout(context.Background(), 30*time.Second)
	defer cancel()

	const n = 3
	for i := 0; i < n; i++ {
		h.launch(ctx)
	}
	h.waitSubmissions(n * 2)

	h.tr.releaseAck(t, "b1", 1) // seq1 half-acked but still in flight
	h.eng.AdoptEpoch(2)         // epoch bump -> flush every epoch-1 in-flight record

	if got := h.committed(); got != 0 {
		t.Fatalf("committed = %d after epoch flush, want 0", got)
	}
	h.wait()
	for s := uint64(1); s <= n; s++ {
		if err := h.result(s); !errors.Is(err, ErrLeaseExpired) {
			t.Fatalf("seq %d outcome = %v, want ErrLeaseExpired (epoch-flushed)", s, err)
		}
	}

	// Every remaining captured completion arrives late (records already popped):
	// must be a clean no-op, committed unchanged.
	h.tr.closeAll(errModelLink)
	if got := h.committed(); got != 0 {
		t.Fatalf("committed = %d after late completions, want 0", got)
	}
	h.shutdownWithin(3 * time.Second)
}

// --- Scenario 4: transport error / link close during a sweep -----------------

// TestModelTransportErrorTransitiveCommit: a transport error resolves the HEAD
// (seq1) as failed; a later seq (2) then fully acks. The sweep pops the failed
// head WITHOUT committing it and commits seq2 — committed jumps 0 -> 2, exposing
// the failed seq1 transitively (P7/P10), exactly as markCommitted's max does.
func TestModelTransportErrorTransitiveCommit(t *testing.T) {
	h := newModelHarness(t, []string{"n1", "b1"})
	defer h.eng.Shutdown()
	ctx, cancel := context.WithTimeout(context.Background(), 30*time.Second)
	defer cancel()

	const n = 2
	for i := 0; i < n; i++ {
		h.launch(ctx)
	}
	h.waitSubmissions(n) // single required peer -> n submissions

	h.tr.releaseErr(t, "b1", 1, errModelLink) // head fails on transport error
	if got := h.committed(); got != 0 {
		t.Fatalf("committed = %d after head transport error, want 0", got)
	}
	h.tr.releaseAck(t, "b1", 2) // seq2 fully acks -> transitive commit past seq1
	if got := h.committed(); got != 2 {
		t.Fatalf("committed = %d, want 2 (transitive commit past failed seq1)", got)
	}

	h.wait()
	if err := h.result(1); !errors.Is(err, ErrReplicationTimeout) {
		t.Fatalf("seq 1 outcome = %v, want ErrReplicationTimeout (fail-fast on transport error)", err)
	}
	if err := h.result(2); err != nil {
		t.Fatalf("seq 2 outcome = %v, want nil (committed)", err)
	}
}

// TestModelLinkCloseFailsAllInFlight: after one write commits, the peer link dies
// and fails EVERY remaining in-flight submission at once. committed must freeze at
// the durable prefix (1) — no non-durable seq is ever exposed — and every stalled
// Propose resolves failed exactly once. The senders then exit on Shutdown.
func TestModelLinkCloseFailsAllInFlight(t *testing.T) {
	h := newModelHarness(t, []string{"n1", "b1", "b2"})
	ctx, cancel := context.WithTimeout(context.Background(), 30*time.Second)
	defer cancel()

	const n = 4
	for i := 0; i < n; i++ {
		h.launch(ctx)
	}
	h.waitSubmissions(n * 2)

	h.tr.releaseAck(t, "b1", 1)
	h.tr.releaseAck(t, "b2", 1) // seq1 full-ISR commits
	if got := h.committed(); got != 1 {
		t.Fatalf("committed = %d, want 1 (seq1 committed before link death)", got)
	}

	h.tr.closeAll(errModelLink) // link dies: fail all remaining in-flight
	if got := h.committed(); got != 1 {
		t.Fatalf("committed = %d after link close, want 1 (frozen at durable prefix)", got)
	}
	h.wait()
	if err := h.result(1); err != nil {
		t.Fatalf("seq 1 outcome = %v, want nil (committed before link death)", err)
	}
	for s := uint64(2); s <= n; s++ {
		if err := h.result(s); !errors.Is(err, ErrReplicationTimeout) {
			t.Fatalf("seq %d outcome = %v, want ErrReplicationTimeout (link death)", s, err)
		}
	}
	h.shutdownWithin(3 * time.Second)
}

// --- Scenario 5: commit-time lease fence — delayed ack after expiry ----------

// TestModelDelayedAckAfterLeaseExpiry is the second-riskiest mitigation (the
// commit-time lease fence, PIPELINE-REDESIGN Q5) driven through the real
// completion path: the lease lapses while acks are in flight, then FULL acks
// arrive after expiry. The commit-time fence must refuse to commit — no late/
// non-durable seq is ever exposed via committed — and every record resolves
// ErrLeaseExpired.
func TestModelDelayedAckAfterLeaseExpiry(t *testing.T) {
	h := newModelHarness(t, []string{"n1", "b1", "b2"})
	defer h.eng.Shutdown()
	ctx, cancel := context.WithTimeout(context.Background(), 30*time.Second)
	defer cancel()

	const n = 2
	for i := 0; i < n; i++ {
		h.launch(ctx)
	}
	h.waitSubmissions(n * 2)

	// Partitioned primary: no lease renewal, clock advances past expiry.
	h.clk.set(t0 + leaseDur + 1)

	// Full acks for both records arrive AFTER expiry.
	h.tr.releaseAck(t, "b1", 1)
	h.tr.releaseAck(t, "b2", 1)
	h.tr.releaseAck(t, "b1", 2)
	h.tr.releaseAck(t, "b2", 2)

	if got := h.committed(); got != 0 {
		t.Fatalf("committed = %d, want 0 — the commit-time fence must reject acks past lease expiry", got)
	}
	h.wait()
	for s := uint64(1); s <= n; s++ {
		if err := h.result(s); !errors.Is(err, ErrLeaseExpired) {
			t.Fatalf("seq %d outcome = %v, want ErrLeaseExpired (commit-time fence)", s, err)
		}
	}
}

// TestModelNackFailsRequiredPeer: a definitive negative ack (OK:false) from a
// required peer for the exact (epoch,seq) dooms the write — full-ISR needs ALL
// peers, so one rejecting peer fails the record fast. Here the head is nacked;
// a later fully-acked seq then commits transitively over it.
func TestModelNackFailsRequiredPeer(t *testing.T) {
	h := newModelHarness(t, []string{"n1", "b1", "b2"})
	defer h.eng.Shutdown()
	ctx, cancel := context.WithTimeout(context.Background(), 30*time.Second)
	defer cancel()

	const n = 2
	for i := 0; i < n; i++ {
		h.launch(ctx)
	}
	h.waitSubmissions(n * 2)

	// seq1 gets one OK ack and one NACK from its two required peers -> doomed.
	h.tr.releaseAck(t, "b1", 1)
	h.tr.releaseNack(t, "b2", 1)
	if got := h.committed(); got != 0 {
		t.Fatalf("committed = %d after nack on head, want 0", got)
	}
	// seq2 fully acks -> commits transitively past the failed seq1.
	h.tr.releaseAck(t, "b1", 2)
	h.tr.releaseAck(t, "b2", 2)
	if got := h.committed(); got != 2 {
		t.Fatalf("committed = %d, want 2 (transitive over nacked seq1)", got)
	}
	h.wait()
	if err := h.result(1); !errors.Is(err, ErrReplicationTimeout) {
		t.Fatalf("seq 1 outcome = %v, want ErrReplicationTimeout (nacked required peer)", err)
	}
	if err := h.result(2); err != nil {
		t.Fatalf("seq 2 outcome = %v, want nil (committed)", err)
	}
}
