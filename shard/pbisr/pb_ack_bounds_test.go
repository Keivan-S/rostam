// SPDX-License-Identifier: Apache-2.0

package pbisr

import (
	"math"
	"sync"
	"testing"
	"time"
)

// --- M5: the ack is a PEER-CONTROLLED counter that drives delta computation ----

// TestReceiveGroupSeq0DoesNotUnderflowAck pins the underflow at its source.
//
// ReceiveGroup's cumulative-ack baseline is `msgs[0].Seq - 1` — the position the
// target holds if the FIRST frame is rejected. On a seq-0 group that expression
// wraps to MaxUint64, and the nack then reports an applied high-water of
// 18446744073709551615 to the sender, which feeds it straight into
// backfillLearner's delta arithmetic (`tail - k`, `k + growBackfillBatch`).
//
// Seq 0 is never assignable (proposeSequenced assigns frontier+1 from a frontier
// that starts at 0), so such a group can only come from a malformed or hostile
// peer on the wire-format branch — precisely the input that must not be trusted
// to be well-formed. The honest ack for a group we applied nothing from is 0.
func TestReceiveGroupSeq0DoesNotUnderflowAck(t *testing.T) {
	e := New("n2", testShard, &fakeControl{epoch: 1, primary: "n1", isr: []string{"n1", "n2"}, minISR: 2}, newInMemTransport(), &fakeApplier{})
	defer e.Shutdown()

	ack := e.ReceiveGroup([]ReplicateMsg{
		{Epoch: 1, Seq: 0, PrevSeq: math.MaxUint64, PrevEpoch: 0, Data: []byte("x")},
		{Epoch: 1, Seq: 1, PrevSeq: 0, PrevEpoch: 1, Data: []byte("y")},
	})
	if ack.OK {
		t.Fatal("a group whose first frame claims seq 0 must be rejected")
	}
	if ack.Seq != 0 {
		t.Fatalf("ack.Seq = %d, want 0 — the baseline underflowed (MaxUint64 = %d)", ack.Seq, uint64(math.MaxUint64))
	}
	// And nothing was applied: the frontier is untouched.
	if fs, fe := e.AppliedFrontier(); fs != 0 || fe != 0 {
		t.Fatalf("frontier = (%d,%d) after a rejected seq-0 group, want (0,0)", fs, fe)
	}

	// The ordinary rejection path is unchanged: a well-formed group that fails log
	// matching still reports the position the target actually holds (here, 0 —
	// nothing applied — via `msgs[0].Seq - 1` with no wrap).
	ack = e.ReceiveGroup([]ReplicateMsg{
		{Epoch: 1, Seq: 5, PrevSeq: 4, PrevEpoch: 1, Data: []byte("z")},
	})
	if ack.OK || ack.Seq != 4 {
		t.Fatalf("ack = %+v, want OK=false Seq=4 (the baseline below a rejected first frame)", ack)
	}
}

// overAckingGroupTransport delivers group frames normally but REWRITES the ack's
// Seq to a value of the test's choosing — modelling a peer whose reported
// high-water is not something this exchange could have produced (a buggy or
// hostile implementation, or an ack corrupted on the wire-format branch).
type overAckingGroupTransport struct {
	*inMemTransport
	mu     sync.Mutex
	seq    uint64 // the Seq every group ack is rewritten to
	rounds int    // how many group frames were shipped
}

func (w *overAckingGroupTransport) ReplicateGroup(peer string, msgs []ReplicateMsg, done func(AckMsg, error)) error {
	return w.inMemTransport.ReplicateGroup(peer, msgs, func(ack AckMsg, err error) {
		w.mu.Lock()
		w.rounds++
		forged := w.seq
		w.mu.Unlock()
		if err == nil {
			ack.Seq = forged
			ack.OK = true
		}
		done(ack, err)
	})
}

func (w *overAckingGroupTransport) shipped() int {
	w.mu.Lock()
	defer w.mu.Unlock()
	return w.rounds
}

// TestBackfillRejectsAckPastShippedRange is the M5 regression: backfillLearner
// accepted ANY ack.Seq > k, with no bound on what was actually shipped.
//
// Two ways that hurts, both driven entirely by the peer's number:
//   - A target claiming a position ABOVE the delta it was sent makes the loop
//     resume from a k this primary never established, so the next round's delta
//     starts above seqs the target does not hold — a gap manufactured by the ack.
//   - The underflowed MaxUint64 ack (see above) lands here as a colossal
//     "advance", after which `tail - k` and `k + growBackfillBatch` are both
//     wrapped arithmetic.
//
// The bound is (k, shipped]. Anything outside is not repairable by appending, so
// it takes the same exit as any divergence — and that is routed to a full
// snapshot, the only thing that can re-establish a known position.
func TestBackfillRejectsAckPastShippedRange(t *testing.T) {
	for _, tc := range []struct {
		name string
		ack  uint64
	}{
		{"just past the shipped range", uint64(growBackfillBatch) + 2},
		{"absurdly far ahead", 1 << 40},
		{"the MaxUint64 underflow value", math.MaxUint64},
	} {
		t.Run(tc.name, func(t *testing.T) {
			ctrl := &fakeControl{epoch: 1, primary: "n1", isr: []string{"n1"}, minISR: 1}
			clk := &fakeClock{t: t0}
			inner := newInMemTransport()
			tr := &overAckingGroupTransport{inMemTransport: inner, seq: tc.ack}
			primary := New("n1", testShard, ctrl, tr, &fakeApplier{}, WithClock(clk.now))
			inner.register("n1", primary)
			primary.GrantLease(1, t0+leaseDur)
			defer primary.Shutdown()

			lag := New("n3", testShard, ctrl, inner, &fakeApplier{}, WithClock(clk.now))
			inner.register("n3", lag)
			defer lag.Shutdown()

			// More than one backfill round's worth, so a forged ack that IS accepted
			// visibly skips ahead instead of just ending the loop.
			for i := 0; i < 3*growBackfillBatch; i++ {
				if _, _, err := primary.ProposeDeadline([]byte{byte(i)}, 2*time.Second); err != nil {
					t.Fatalf("propose %d: %v", i, err)
				}
			}

			err := primary.StartLearnerCatchup(ctxWithTimeout(t, 3*time.Second), "n3")
			if err != ErrCatchupDiverged {
				t.Fatalf("over-reported ack (%d): err = %v, want ErrCatchupDiverged", tc.ack, err)
			}
			if primary.IsLearner("n3") {
				t.Fatal("a grow aborted on an unbounded ack must not leave a learner installed")
			}
			// It must abort on the FIRST forged ack, not wander through more rounds
			// on a position the primary never established.
			if got := tr.shipped(); got != 1 {
				t.Fatalf("shipped %d group frames before aborting, want 1 (the bound must reject the first forged ack)", got)
			}
		})
	}
}

// TestBackfillAcceptsHonestAckWithinShippedRange is the control: the bound must
// not touch the ordinary path. A target that applies everything it is sent acks
// exactly the shipped tail (ack.Seq == shipped, the upper end of the accepted
// range), and a multi-round backfill must still converge and flip.
func TestBackfillAcceptsHonestAckWithinShippedRange(t *testing.T) {
	ctrl := &fakeControl{epoch: 1, primary: "n1", isr: []string{"n1"}, minISR: 1}
	clk := &fakeClock{t: t0}
	tr := newInMemTransport()
	primary := New("n1", testShard, ctrl, tr, &fakeApplier{}, WithClock(clk.now))
	tr.register("n1", primary)
	primary.GrantLease(1, t0+leaseDur)
	defer primary.Shutdown()

	ap3 := &fakeApplier{}
	lag := New("n3", testShard, ctrl, tr, ap3, WithClock(clk.now))
	tr.register("n3", lag)
	defer lag.Shutdown()

	const n = 3 * growBackfillBatch // several rounds, each acking its exact shipped tail
	for i := 0; i < n; i++ {
		if _, _, err := primary.ProposeDeadline([]byte{byte(i)}, 2*time.Second); err != nil {
			t.Fatalf("propose %d: %v", i, err)
		}
	}
	if err := primary.StartLearnerCatchup(ctxWithTimeout(t, 5*time.Second), "n3"); err != nil {
		t.Fatalf("an honest multi-round backfill must succeed: %v", err)
	}
	if !primary.IsLearner("n3") {
		t.Fatal("n3 must be flipped into the learner ship-set")
	}
	eventually(t, func() bool { return lag.LastApplied() == uint64(n) }, "n3 catches up to the frontier")
	if got := ap3.count(); got != n {
		t.Fatalf("n3 applied %d ops, want %d (gap-free)", got, n)
	}
}
