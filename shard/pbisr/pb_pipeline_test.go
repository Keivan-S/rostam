// SPDX-License-Identifier: Apache-2.0

package pbisr

import (
	"context"
	"errors"
	"sync"
	"sync/atomic"
	"testing"
	"time"
)

// --- test harness -----------------------------------------------------------
//
// The pipeline machinery is driven directly under e.mu with synthetic in-flight
// records and an injected clock — no transport, no Propose. testClocks maps an
// engine to its controllable monotonic clock so advanceClock can find it from
// the *Engine alone (the brief's helper signatures take only the engine).

var testClocks sync.Map // *Engine -> *atomic.Int64

// newInflightTestEngine builds an engine with a VALID lease (epoch=5, leaseEpoch=5,
// leaseExpiry=100), committed=0, and an injected clock starting at now()=0.
func newInflightTestEngine(t *testing.T) *Engine {
	t.Helper()
	clk := new(atomic.Int64) // now() == 0
	e := New("p", 0, nil, nil, nil, WithClock(func() int64 { return clk.Load() }))
	e.mu.Lock()
	e.epoch = 5
	e.leaseEpoch = 5
	e.leaseExpiry = 100
	e.committed = 0
	e.mu.Unlock()
	testClocks.Store(e, clk)
	return e
}

// advanceClock moves the injected monotonic clock forward by delta ns.
func advanceClock(e *Engine, delta int64) {
	v, ok := testClocks.Load(e)
	if !ok {
		panic("advanceClock: engine has no test clock")
	}
	v.(*atomic.Int64).Add(delta)
}

// pushInflight registers a synthetic in-flight record (mirrors Propose's
// sequencing stage: seq assigned, record queued).
func pushInflight(e *Engine, epoch, seq uint64, peers []string) *inflight {
	e.mu.Lock()
	defer e.mu.Unlock()
	return e.registerInflightLocked(epoch, seq, peers)
}

// ackAll delivers an exact (epoch,seq) OK ack from every peer still pending on
// rec, driving the sweep after each.
func ackAll(e *Engine, rec *inflight, epoch, seq uint64) {
	e.mu.Lock()
	defer e.mu.Unlock()
	peers := append([]string(nil), rec.pendingPeers()...)
	for _, p := range peers {
		e.ackInflightLocked(p, AckMsg{Epoch: epoch, Seq: seq, OK: true})
	}
}

// resolveFail resolves rec as failed (models a ctx timeout / transport failure)
// and sweeps.
func resolveFail(e *Engine, rec *inflight, err error) {
	e.mu.Lock()
	defer e.mu.Unlock()
	e.resolveLocked(rec, err)
	e.sweepLocked()
}

// waitResolve blocks for rec's resolution and returns its outcome.
func waitResolve(rec *inflight) error { return <-rec.doneCh }

// mustResolveNil asserts rec committed successfully.
func mustResolveNil(t *testing.T, rec *inflight) {
	t.Helper()
	if err := <-rec.doneCh; err != nil {
		t.Fatalf("record seq=%d resolved err=%v, want nil (committed)", rec.seq, err)
	}
}

// --- brief tests ------------------------------------------------------------

func TestSweepCommitsInOrder(t *testing.T) {
	e := newInflightTestEngine(t) // lease valid, epoch=5, committed=0
	r1 := pushInflight(e, 5, 1, []string{"b1", "b2"})
	r2 := pushInflight(e, 5, 2, []string{"b1", "b2"})
	// r2 fully acked first — must NOT commit ahead of r1 (head-only).
	ackAll(e, r2, 5, 2)
	if e.Committed() != 0 {
		t.Fatalf("committed=%d, want 0 (r1 still pending)", e.Committed())
	}
	ackAll(e, r1, 5, 1)
	if e.Committed() != 2 {
		t.Fatalf("committed=%d, want 2 (both now durable)", e.Committed())
	}
	mustResolveNil(t, r1)
	mustResolveNil(t, r2)
}

func TestSweepHoleTransitiveCommit(t *testing.T) {
	// r1 resolved-failed (timeout), r2 full-ISR acked -> committed jumps to 2.
	e := newInflightTestEngine(t)
	r1 := pushInflight(e, 5, 1, []string{"b1"})
	r2 := pushInflight(e, 5, 2, []string{"b1"})
	resolveFail(e, r1, ErrReplicationTimeout)
	if e.Committed() != 0 {
		t.Fatalf("committed=%d want 0", e.Committed())
	}
	ackAll(e, r2, 5, 2)
	if e.Committed() != 2 {
		t.Fatalf("committed=%d want 2 (transitive)", e.Committed())
	}
	if err := waitResolve(r1); !errors.Is(err, ErrReplicationTimeout) {
		t.Fatalf("r1 err=%v, want ErrReplicationTimeout", err)
	}
	mustResolveNil(t, r2)
}

func TestCommitLeaseFenceBlocksLateCommit(t *testing.T) {
	// Lease expires while r1's acks are in flight; sweep must NOT commit it.
	e := newInflightTestEngine(t) // now()=0, leaseExpiry=100, leaseEpoch=epoch=5
	r1 := pushInflight(e, 5, 1, []string{"b1"})
	advanceClock(e, 200) // now()=200 >= leaseExpiry
	ackAll(e, r1, 5, 1)
	if e.Committed() != 0 {
		t.Fatalf("committed=%d, want 0 — expired lease must not commit", e.Committed())
	}
	if err := waitResolve(r1); !errors.Is(err, ErrLeaseExpired) {
		t.Fatalf("r1 err=%v, want ErrLeaseExpired", err)
	}
}

func TestResolveExactlyOnce(t *testing.T) {
	e := newInflightTestEngine(t)
	r := pushInflight(e, 5, 1, []string{"b1"})
	// Race timeout vs ack: both call resolve; doneCh (buffered 1) gets one value,
	// window slot freed once.
	e.mu.Lock()
	e.resolveLocked(r, ErrReplicationTimeout)
	e.mu.Unlock()
	e.mu.Lock()
	e.resolveLocked(r, nil) // no-op
	e.mu.Unlock()
	if got := len(r.doneCh); got != 1 {
		t.Fatalf("doneCh has %d, want 1", got)
	}
	if err := <-r.doneCh; !errors.Is(err, ErrReplicationTimeout) {
		t.Fatalf("first resolution won? err=%v, want ErrReplicationTimeout", err)
	}
}

func TestFlushEpochFailsOlder(t *testing.T) {
	e := newInflightTestEngine(t)
	r := pushInflight(e, 5, 1, []string{"b1"})
	e.mu.Lock()
	e.flushEpochLocked(6)
	e.mu.Unlock()
	if err := waitResolve(r); err == nil {
		t.Fatal("want failure on epoch flush")
	}
}

// --- additional edge-case coverage (P5/P6/P7, fence, window) ----------------

// TestAckH6ExactMatch: a non-OK ack, a wrong-epoch ack, and a wrong-seq ack must
// NOT count toward the record's pending set (P5 / H6 exact matching).
func TestAckH6ExactMatch(t *testing.T) {
	e := newInflightTestEngine(t)
	r := pushInflight(e, 5, 1, []string{"b1"})

	e.mu.Lock()
	e.ackInflightLocked("b1", AckMsg{Epoch: 5, Seq: 1, OK: false}) // not OK
	e.ackInflightLocked("b1", AckMsg{Epoch: 4, Seq: 1, OK: true})  // wrong epoch
	e.ackInflightLocked("b1", AckMsg{Epoch: 5, Seq: 2, OK: true})  // wrong seq (routes elsewhere/none)
	e.mu.Unlock()

	if e.Committed() != 0 {
		t.Fatalf("committed=%d, want 0 — no exact ack yet", e.Committed())
	}
	if r.resolved {
		t.Fatal("record resolved on non-matching acks")
	}
	ackAll(e, r, 5, 1) // exact match now
	if e.Committed() != 1 {
		t.Fatalf("committed=%d, want 1 after exact ack", e.Committed())
	}
	mustResolveNil(t, r)
}

// TestFullISRRequiresEveryPeer: a record commits only when EVERY distinct
// propose-time peer has exact-acked (P6).
func TestFullISRRequiresEveryPeer(t *testing.T) {
	e := newInflightTestEngine(t)
	r := pushInflight(e, 5, 1, []string{"b1", "b2"})

	e.mu.Lock()
	e.ackInflightLocked("b1", AckMsg{Epoch: 5, Seq: 1, OK: true}) // partial ISR
	e.mu.Unlock()
	if e.Committed() != 0 {
		t.Fatalf("committed=%d, want 0 — only 1 of 2 acked", e.Committed())
	}
	e.mu.Lock()
	e.ackInflightLocked("b2", AckMsg{Epoch: 5, Seq: 1, OK: true}) // full ISR
	e.mu.Unlock()
	if e.Committed() != 1 {
		t.Fatalf("committed=%d, want 1 — full ISR acked", e.Committed())
	}
	mustResolveNil(t, r)
}

// TestFlushEpochKeepsSameEpoch: flushEpochLocked fails records strictly OLDER
// than minEpoch and leaves current-epoch records untouched.
func TestFlushEpochKeepsSameEpoch(t *testing.T) {
	e := newInflightTestEngine(t)
	old := pushInflight(e, 5, 1, []string{"b1"})
	// Move to epoch 6 for the newer record (dense seq continues).
	cur := pushInflight(e, 6, 2, []string{"b1"})

	e.mu.Lock()
	e.flushEpochLocked(6) // fail everything < epoch 6
	e.mu.Unlock()

	if err := waitResolve(old); err == nil {
		t.Fatal("older-epoch record should be failed by flush")
	}
	if cur.resolved {
		t.Fatal("current-epoch record must survive the flush")
	}
	// cur is now head; a full ack under its own epoch still commits it. Note the
	// fence checks e.epoch/e.leaseEpoch, so bring the engine to epoch 6.
	e.mu.Lock()
	e.epoch = 6
	e.leaseEpoch = 6
	e.mu.Unlock()
	ackAll(e, cur, 6, 2)
	if e.Committed() != 2 {
		t.Fatalf("committed=%d, want 2 (cur committed transitively over failed old)", e.Committed())
	}
}

// TestWindowFullAndWait: window = lastSeq - committed. A full window blocks
// windowWait until committed advances (a freed slot), then it returns.
func TestWindowFullAndWait(t *testing.T) {
	e := newInflightTestEngine(t)
	e.mu.Lock()
	e.lastSeq = pipelineWindow // lastSeq - committed == W
	e.committed = 0
	e.mu.Unlock()

	if !e.windowFull() {
		t.Fatal("windowFull should be true at lastSeq-committed == W")
	}
	done := make(chan error, 1)
	go func() { done <- e.windowWait(context.Background()) }()

	// Advancing committed frees a slot and wakes the waiter.
	e.mu.Lock()
	e.markCommittedLocked(1)
	e.mu.Unlock()

	select {
	case err := <-done:
		if err != nil {
			t.Fatalf("windowWait returned %v, want nil after slot freed", err)
		}
	case <-time.After(3 * time.Second):
		t.Fatal("windowWait did not wake after committed advanced")
	}
	if e.windowFull() {
		t.Fatal("window should have room after committed advanced")
	}
}

// TestWindowWaitCtxCancel: a full window that never drains fails admission when
// the caller's context is cancelled (ctx-bounded gate).
func TestWindowWaitCtxCancel(t *testing.T) {
	e := newInflightTestEngine(t)
	e.mu.Lock()
	e.lastSeq = pipelineWindow
	e.committed = 0
	e.mu.Unlock()

	ctx, cancel := context.WithCancel(context.Background())
	done := make(chan error, 1)
	go func() { done <- e.windowWait(ctx) }()
	cancel()

	select {
	case err := <-done:
		if err == nil {
			t.Fatal("windowWait should return ctx error on a wedged full window")
		}
	case <-time.After(3 * time.Second):
		t.Fatal("windowWait did not return after ctx cancel")
	}
}

// TestWindowWaitImmediateWhenRoom: no blocking when the window has room.
func TestWindowWaitImmediateWhenRoom(t *testing.T) {
	e := newInflightTestEngine(t)
	if err := e.windowWait(context.Background()); err != nil {
		t.Fatalf("windowWait with empty window returned %v, want nil", err)
	}
}

// TestProposeStallVsCancelAtFullWindow pins the Propose-level admission mapping
// that the seam relies on: with a FULL window (lastSeq-committed == W)
// that never drains, a Propose whose ctx DEADLINE expires while full returns the
// designed admission-stall signal ErrPipelineStalled, whereas a Propose whose ctx
// is CANCELLED returns the ctx error unchanged — never ErrPipelineStalled. The
// window-full short-circuit in Propose precedes any control-plane read, so the
// ctrl==nil engine from newInflightTestEngine never gets dereferenced here.
func TestProposeStallVsCancelAtFullWindow(t *testing.T) {
	e := newInflightTestEngine(t)
	e.mu.Lock()
	e.lastSeq = pipelineWindow // window == W: full, and it never drains (committed=0)
	e.committed = 0
	e.mu.Unlock()

	// (a) Deadline expires with the window still full → ErrPipelineStalled.
	ctx, cancel := context.WithTimeout(context.Background(), 20*time.Millisecond)
	defer cancel()
	if _, _, err := e.Propose(ctx, []byte("w")); !errors.Is(err, ErrPipelineStalled) {
		t.Fatalf("deadline-while-full: err = %v, want ErrPipelineStalled", err)
	}

	// (b) Caller cancellation → the ctx error, NOT ErrPipelineStalled.
	ctx2, cancel2 := context.WithCancel(context.Background())
	cancel2()
	_, _, err := e.Propose(ctx2, []byte("w"))
	if errors.Is(err, ErrPipelineStalled) {
		t.Fatalf("cancellation must not surface as ErrPipelineStalled; got %v", err)
	}
	if !errors.Is(err, context.Canceled) {
		t.Fatalf("cancellation: err = %v, want context.Canceled", err)
	}
}
