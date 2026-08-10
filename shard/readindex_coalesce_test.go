// SPDX-License-Identifier: Apache-2.0

package shard

import (
	"sync"
	"sync/atomic"
	"testing"
	"time"
)

// TestReadindexCoalescerSharesInFlight: readers that all arrive while one flight is
// open share ONE fn invocation (the coalescing win). We gate fn so it cannot finish
// until all readers have registered; that forces them into the same open batch.
func TestReadindexCoalescerSharesInFlight(t *testing.T) {
	var c readindexCoalescer
	const n = 8

	var calls atomic.Int64
	release := make(chan struct{})
	registered := make(chan struct{}, n)

	var wg sync.WaitGroup
	for i := 0; i < n; i++ {
		wg.Add(1)
		go func() {
			defer wg.Done()
			// Signal arrival, then run the coalesced barrier.
			registered <- struct{}{}
			_ = c.do(time.Now().Add(2*time.Second), func(time.Time) error {
				calls.Add(1)
				<-release // hold the flight open so latecomers (the rest) coalesce
				return nil
			})
		}()
	}

	// Wait until all n readers have at least entered do() (registered), then let the
	// single in-flight fn finish. With the batch held open by the gate, the leader's
	// fn runs once and the rest join it.
	for i := 0; i < n; i++ {
		<-registered
	}
	// Give the goroutines a beat to all reach do()/register into the batch before the
	// leader closes it. The leader only closes after it is scheduled; the gate keeps
	// fn blocked so followers that registered before the close all coalesce.
	time.Sleep(50 * time.Millisecond)
	close(release)
	wg.Wait()

	if got := calls.Load(); got >= n {
		t.Fatalf("coalescer ran fn %d times for %d concurrent readers, want < %d (coalesced)", got, n, n)
	}
	if calls.Load() == 0 {
		t.Fatal("coalescer never ran fn")
	}
}

// TestReadindexCoalescerLatecomerGetsFreshFlight is the NO-STALENESS proof at the
// coalescer level: a reader that ARRIVES AFTER an in-flight flight has captured
// (its leader is already inside fn) must NOT receive that flight's result; it must
// get a FRESH flight whose capture is after its arrival. We tag each flight with a
// monotonically increasing capture sequence and assert the latecomer's observed
// sequence is strictly greater than the in-flight one.
func TestReadindexCoalescerLatecomerGetsFreshFlight(t *testing.T) {
	var c readindexCoalescer

	// captureSeq tags each fn capture so we can prove R2's capture is STRICTLY after
	// R1's (a fresh post-arrival flight), never R1's already-captured one.
	var captureSeq atomic.Int64
	flight1Capturing := make(chan struct{})
	flight1Release := make(chan struct{})
	var captureOnce sync.Once

	// Each fn records its capture sequence into the batch's result via a closure over
	// a shared slot keyed by call order. R1's fn (the first) blocks until released,
	// AFTER capturing — so R2 demonstrably arrives after R1's capture.
	seqOf := func(name string, slot *int64) func(time.Time) error {
		return func(time.Time) error {
			s := captureSeq.Add(1)
			*slot = s
			if name == "R1" {
				captureOnce.Do(func() { close(flight1Capturing) })
				<-flight1Release // hold R1's flight open past R2's arrival
			}
			return nil
		}
	}

	var r1Seq, r2Seq int64
	r1Done := make(chan struct{})
	go func() {
		_ = c.do(time.Now().Add(5*time.Second), seqOf("R1", &r1Seq))
		close(r1Done)
	}()

	<-flight1Capturing // R1 has captured (#1) and is holding its flight open.

	// R2 ARRIVES NOW — strictly after R1's capture, while R1's flight is running. It
	// must NOT be served R1's #1 frontier; it waits for a FRESH flight (#2) whose
	// capture is after its arrival. (Under the batch-while-busy model R2 is run by
	// R1's leader in the drain loop AFTER R1 releases — a separate capture.)
	r2Done := make(chan struct{})
	go func() {
		_ = c.do(time.Now().Add(5*time.Second), seqOf("R2", &r2Seq))
		close(r2Done)
	}()

	// Give R2 time to register into the pending batch, then release R1.
	time.Sleep(50 * time.Millisecond)
	close(flight1Release)

	<-r1Done
	<-r2Done

	if r1Seq != 1 {
		t.Fatalf("R1 capture seq = %d, want 1", r1Seq)
	}
	if r2Seq <= r1Seq {
		t.Fatalf("latecomer R2 capture seq = %d, must be > R1 seq %d (R2 must NOT be served the pre-arrival flight)", r2Seq, r1Seq)
	}
}

// TestReadindexCoalescerSingleReaderRunsBody: a lone reader forms a batch of one and
// runs fn directly (behavior identical to the un-coalesced path).
func TestReadindexCoalescerSingleReaderRunsBody(t *testing.T) {
	var c readindexCoalescer
	var ran atomic.Bool
	if err := c.do(time.Now().Add(time.Second), func(time.Time) error {
		ran.Store(true)
		return nil
	}); err != nil {
		t.Fatalf("do: %v", err)
	}
	if !ran.Load() {
		t.Fatal("lone reader did not run the barrier body")
	}
}
