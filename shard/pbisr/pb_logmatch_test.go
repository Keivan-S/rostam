// SPDX-License-Identifier: Apache-2.0

package pbisr

import (
	"fmt"
	"testing"
	"time"
)

// payloads renders every applied op as a string, in apply order — the direct
// read-out of a node's materialized history (used to show divergence, not just
// counters).
func (a *fakeApplier) payloads() []string {
	a.mu.Lock()
	defer a.mu.Unlock()
	out := make([]string, 0, len(a.applied))
	for _, d := range a.applied {
		out = append(out, string(d))
	}
	return out
}

// divergedPair builds the two-node stage for the log-matching regression: n1 is
// primary at epoch 1 over a single-node ISR {n1} (so it commits ALONE — its
// writes reach no one), and n2 is a would-be successor over the SAME transport.
// Each node gets its own control plane so both can believe they are primary at
// their own epoch, which is exactly the split-brain-then-recover shape the
// catch-up protocol must survive.
type divergedPair struct {
	tr     *inMemTransport
	clk    *fakeClock
	ctrl1  *fakeControl
	ctrl2  *fakeControl
	n1, n2 *Engine
	ap1    *fakeApplier
	ap2    *fakeApplier
}

func newDivergedPair() *divergedPair {
	clk := &fakeClock{t: t0}
	tr := newInMemTransport()
	ctrl1 := &fakeControl{epoch: 1, primary: "n1", isr: []string{"n1"}, minISR: 1}
	ctrl2 := &fakeControl{epoch: 2, primary: "n2", isr: []string{"n2"}, minISR: 1}
	ap1, ap2 := &fakeApplier{}, &fakeApplier{}
	n1 := New("n1", testShard, ctrl1, tr, ap1, WithClock(clk.now))
	n2 := New("n2", testShard, ctrl2, tr, ap2, WithClock(clk.now))
	tr.register("n1", n1)
	tr.register("n2", n2)
	n1.GrantLease(1, t0+leaseDur)
	return &divergedPair{tr: tr, clk: clk, ctrl1: ctrl1, ctrl2: ctrl2, n1: n1, n2: n2, ap1: ap1, ap2: ap2}
}

func (p *divergedPair) close() {
	p.n1.Shutdown()
	p.n2.Shutdown()
}

// TestCatchupRejectsDivergentExPrimary is the regression test.
//
// THE DEFECT it pins: a seq number is not a history identity. n1 proposes 5
// writes ALONE as primary at epoch 1 — those advance lastSeq but NOT lastApplied
// (lastApplied only moves in the backup receive path), so n1's catch-up
// handshake used to report a high-water of ZERO. n2 then promotes at epoch 2
// from lastApplied 0, resets its seq counter to 0, and proposes three DIFFERENT
// writes at seqs 1/2/3. Shipping that delta to n1 used to pass the gap check
// (PrevSeq 0 == n1.lastApplied 0), so n1 silently appended n2's history ON TOP
// of its own five divergent ops, catch-up returned nil, and the control plane
// flipped n1 into the learner set believing the two nodes were in sync.
//
// The log-matching property makes that a clean, named abort instead.
func TestCatchupRejectsDivergentExPrimary(t *testing.T) {
	p := newDivergedPair()
	defer p.close()

	// n1 writes ALONE at epoch 1. Single-node ISR ⇒ every write commits locally
	// and reaches nobody.
	for i := 0; i < 5; i++ {
		if _, _, err := p.n1.ProposeDeadline([]byte(fmt.Sprintf("a%03d", i)), 2*time.Second); err != nil {
			t.Fatalf("n1 propose %d: %v", i, err)
		}
	}
	t.Logf("n1 (old primary, wrote alone): LastSeq=%d LastApplied=%d appliedCount=%d",
		p.n1.LastSeq(), p.n1.LastApplied(), p.ap1.count())

	// n2 promotes at epoch 2 from its own (empty) applied high-water and proposes
	// a DIFFERENT history at the very same seqs 1/2/3.
	p.n2.Promote(2, t0+leaseDur)
	t.Logf("n2 promoted at epoch 2 with LastApplied=%d -> LastSeq=%d", p.n2.LastApplied(), p.n2.LastSeq())
	for i := 0; i < 3; i++ {
		if _, _, err := p.n2.ProposeDeadline([]byte(fmt.Sprintf("c%03d", i)), 2*time.Second); err != nil {
			t.Fatalf("n2 propose %d: %v", i, err)
		}
	}

	err := p.n2.StartLearnerCatchup(ctxWithTimeout(t, 2*time.Second), "n1")
	t.Logf("n2.StartLearnerCatchup(n1) err = %v ; n2.IsLearner(n1) = %v", err, p.n2.IsLearner("n1"))
	// Settle: a successful flip pre-loads the final delta onto a FRESH sender that
	// drains asynchronously, so the damage (if any) lands after the call returns.
	// Fixed wait rather than a poll — the assertion below is that NOTHING lands.
	time.Sleep(200 * time.Millisecond)
	t.Logf("FINAL: n1.appliedCount=%d n1.LastApplied=%d n1.LastSeq=%d | n2.appliedCount=%d n2.LastSeq=%d",
		p.ap1.count(), p.n1.LastApplied(), p.n1.LastSeq(), p.ap2.count(), p.n2.LastSeq())
	t.Logf("n1 applied payloads: %s", p.ap1.payloads())

	if err == nil {
		t.Fatal("catch-up of a DIVERGENT ex-primary must fail: n2's history contradicts n1's at seqs 1..3")
	}
	if err != ErrCatchupDiverged {
		t.Fatalf("err = %v, want ErrCatchupDiverged", err)
	}
	if p.n2.IsLearner("n1") {
		t.Fatal("a diverged target must NOT be flipped into the learner ship-set")
	}
	if got := p.ap1.count(); got != 5 {
		t.Fatalf("n1 applied %d ops, want 5 — n2's contradicting writes must NOT have been appended", got)
	}
	if got := p.n1.LastApplied(); got != 0 {
		t.Fatalf("n1 LastApplied = %d, want 0 (nothing was accepted from n2)", got)
	}
}

// TestCatchupCleanExPrimaryStillFlips is the CONTROL for the divergence case
// above, and the case that pins why the handshake must report the applied
// FRONTIER rather than lastApplied.
//
// n1 is primary at epoch 1 over the writable ISR {n1,n2} and commits 5 writes, so
// n2 holds every one of them. n2 is then promoted at epoch 2 (continuing from
// seq 5) and proposes 3 more alone. n1 now rejoins with the exact shape that
// trips the naive reading of state: lastApplied == 0 (it only ever PROPOSED,
// never received) yet a non-empty, five-write history — which is a TRUE PREFIX of
// n2's, not a fork. The grow must therefore succeed and append exactly the 3
// missing writes: no rejection, and no re-application of the 5 it already has.
func TestCatchupCleanExPrimaryStillFlips(t *testing.T) {
	clk := &fakeClock{t: t0}
	tr := newInMemTransport()
	ctrl1 := &fakeControl{epoch: 1, primary: "n1", isr: []string{"n1", "n2"}, minISR: 2}
	ctrl2 := &fakeControl{epoch: 2, primary: "n2", isr: []string{"n2"}, minISR: 1}
	ap1, ap2 := &fakeApplier{}, &fakeApplier{}
	n1 := New("n1", testShard, ctrl1, tr, ap1, WithClock(clk.now))
	n2 := New("n2", testShard, ctrl2, tr, ap2, WithClock(clk.now))
	tr.register("n1", n1)
	tr.register("n2", n2)
	defer n1.Shutdown()
	defer n2.Shutdown()
	n1.GrantLease(1, t0+leaseDur)

	// n1 commits 5 writes on the FULL ISR, so n2 holds all of them.
	for i := 0; i < 5; i++ {
		if _, _, err := n1.ProposeDeadline([]byte(fmt.Sprintf("a%03d", i)), 2*time.Second); err != nil {
			t.Fatalf("n1 propose %d: %v", i, err)
		}
	}
	eventually(t, func() bool { return n2.LastApplied() == 5 }, "n2 receives all 5 writes")

	// n2 is promoted at epoch 2 and extends the SAME history with 3 more writes.
	n2.Promote(2, t0+leaseDur)
	for i := 0; i < 3; i++ {
		if _, _, err := n2.ProposeDeadline([]byte(fmt.Sprintf("b%03d", i)), 2*time.Second); err != nil {
			t.Fatalf("n2 propose %d: %v", i, err)
		}
	}

	// n1's own view: five writes held, but ZERO of them counted by lastApplied.
	if got := n1.LastApplied(); got != 0 {
		t.Fatalf("n1 LastApplied = %d, want 0 (it only ever proposed)", got)
	}
	fs, fe := n1.AppliedFrontier()
	if fs != 5 || fe != 1 {
		t.Fatalf("n1 AppliedFrontier = (%d,%d), want (5,1)", fs, fe)
	}

	if err := n2.StartLearnerCatchup(ctxWithTimeout(t, 5*time.Second), "n1"); err != nil {
		t.Fatalf("catch-up of a CLEAN ex-primary must succeed: %v", err)
	}
	if !n2.IsLearner("n1") {
		t.Fatal("a caught-up target must be flipped into the learner ship-set")
	}
	eventually(t, func() bool { return n1.LastApplied() == 8 }, "n1 catches up to the frontier")

	want := []string{"a000", "a001", "a002", "a003", "a004", "b000", "b001", "b002"}
	got := ap1.payloads()
	if len(got) != len(want) {
		t.Fatalf("n1 applied %v, want %v (the 5 it already held must NOT be re-applied)", got, want)
	}
	for i := range want {
		if got[i] != want[i] {
			t.Fatalf("n1 applied %v, want %v", got, want)
		}
	}
}

// TestCatchupLaggingBackupStillFlips is the plain control: a pure backup that
// simply fell behind (never proposed anything, so lastApplied IS its frontier)
// must still catch up and flip. This is the ordinary grow the log-matching check
// must leave completely untouched.
func TestCatchupLaggingBackupStillFlips(t *testing.T) {
	c := newCluster([]string{"n1", "n2", "n3"}, "n1", 1, []string{"n1", "n2"}, 2)
	primary := c.engines["n1"]
	defer primary.Shutdown()

	const n = 20
	for i := 0; i < n; i++ {
		if _, _, err := primary.ProposeDeadline([]byte(fmt.Sprintf("w%03d", i)), 2*time.Second); err != nil {
			t.Fatalf("propose %d: %v", i, err)
		}
	}
	// n3 is out of the ISR and holds nothing: a genuine (empty) prefix.
	if fs, fe := c.engines["n3"].AppliedFrontier(); fs != 0 || fe != 0 {
		t.Fatalf("n3 frontier = (%d,%d), want (0,0) — a true genesis node", fs, fe)
	}

	if err := primary.StartLearnerCatchup(ctxWithTimeout(t, 5*time.Second), "n3"); err != nil {
		t.Fatalf("catch-up of a genuinely lagging backup: %v", err)
	}
	if !primary.IsLearner("n3") {
		t.Fatal("n3 must be a learner after the flip")
	}
	eventually(t, func() bool { return c.engines["n3"].LastApplied() == n }, "n3 catches to the frontier")
	if got := c.appliers["n3"].count(); got != n {
		t.Fatalf("n3 applied %d ops, want %d (gap-free)", got, n)
	}
}

// TestRingDeltaStampsPrevEpochAndSplitsRuns is the direct unit assertion on the
// replay path's two log-matching obligations, without the e2e machinery in between:
//
//  1. every replayed frame carries the chain link the ORIGINAL frame carried,
//     read back from the entry itself — including the delta's FIRST element,
//     whose predecessor is not in the delta and may not be in the ring at all;
//  2. the delta stops at an epoch boundary, because the group wire format
//     declares one epoch per frame and would otherwise re-stamp the tail of the
//     run into a history that never existed.
func TestRingDeltaStampsPrevEpochAndSplitsRuns(t *testing.T) {
	c := newCluster([]string{"n1"}, "n1", 1, []string{"n1"}, 1)
	e := c.engines["n1"]
	defer e.Shutdown()

	for i := 0; i < 3; i++ {
		if _, _, err := e.ProposeDeadline([]byte("e1"), 2*time.Second); err != nil {
			t.Fatalf("epoch-1 propose %d: %v", i, err)
		}
	}
	e.GrantLease(2, t0+leaseDur)
	for i := 0; i < 3; i++ {
		if _, _, err := e.ProposeDeadline([]byte("e2"), 2*time.Second); err != nil {
			t.Fatalf("epoch-2 propose %d: %v", i, err)
		}
	}

	e.mu.Lock()
	first, ok1 := e.ringDeltaLocked(1, 6)
	second, ok2 := e.ringDeltaLocked(4, 6)
	e.mu.Unlock()
	if !ok1 || !ok2 {
		t.Fatalf("ring delta not replayable: ok1=%v ok2=%v", ok1, ok2)
	}

	// [1..6] spans the boundary and must stop at seq 3.
	if len(first) != 3 {
		t.Fatalf("delta [1..6] returned %d records, want 3 (truncated at the epoch boundary)", len(first))
	}
	wantFirst := []ReplicateMsg{
		{Epoch: 1, Seq: 1, PrevSeq: 0, PrevEpoch: 0}, // genesis link
		{Epoch: 1, Seq: 2, PrevSeq: 1, PrevEpoch: 1},
		{Epoch: 1, Seq: 3, PrevSeq: 2, PrevEpoch: 1},
	}
	for i := range wantFirst {
		if first[i].Epoch != wantFirst[i].Epoch || first[i].Seq != wantFirst[i].Seq ||
			first[i].PrevSeq != wantFirst[i].PrevSeq || first[i].PrevEpoch != wantFirst[i].PrevEpoch {
			t.Fatalf("record %d: got (e%d s%d pv%d pe%d), want (e%d s%d pv%d pe%d)", i,
				first[i].Epoch, first[i].Seq, first[i].PrevSeq, first[i].PrevEpoch,
				wantFirst[i].Epoch, wantFirst[i].Seq, wantFirst[i].PrevSeq, wantFirst[i].PrevEpoch)
		}
	}
	// The resumed run's FIRST element crosses the boundary: its predecessor is an
	// epoch-1 write that is NOT in this delta. That link can only come from the
	// entry's own stamp, which is the whole reason it is stored per entry.
	if len(second) != 3 {
		t.Fatalf("delta [4..6] returned %d records, want 3", len(second))
	}
	if second[0].Epoch != 2 || second[0].PrevSeq != 3 || second[0].PrevEpoch != 1 {
		t.Fatalf("boundary-crossing record: got (e%d pv%d pe%d), want (e2 pv3 pe1)",
			second[0].Epoch, second[0].PrevSeq, second[0].PrevEpoch)
	}
	if second[1].PrevEpoch != 2 || second[2].PrevEpoch != 2 {
		t.Fatalf("in-run records must chain within epoch 2: pe=%d,%d", second[1].PrevEpoch, second[2].PrevEpoch)
	}
}

// TestCatchupAcrossRingEpochBoundary exercises the ONE structural consequence of
// putting an epoch on every chain link: a replayed delta may no longer span an
// epoch boundary, because the group wire format declares a single epoch for the
// whole frame and rebuilds every record's predecessor from it.
//
// n1 writes 3 at epoch 1, is re-elected at epoch 2, and writes 3 more, so its
// catch-up ring holds a two-epoch run. Catching n3 (which holds nothing) must
// therefore split the backfill at the boundary — and the flip must refuse to
// pre-load a delta truncated short of the tail, or n3 would be handed a gap
// between the pre-load and the frames the normal ship path appends behind it.
// The invariant: n3 ends holding all six writes, in order, gap-free.
func TestCatchupAcrossRingEpochBoundary(t *testing.T) {
	c := newCluster([]string{"n1", "n3"}, "n1", 1, []string{"n1"}, 1)
	primary := c.engines["n1"]
	defer primary.Shutdown()

	for i := 0; i < 3; i++ {
		if _, _, err := primary.ProposeDeadline([]byte(fmt.Sprintf("e1-%d", i)), 2*time.Second); err != nil {
			t.Fatalf("epoch-1 propose %d: %v", i, err)
		}
	}
	// Same node re-elected at epoch 2: the ring now spans two epochs.
	primary.GrantLease(2, t0+leaseDur)
	for i := 0; i < 3; i++ {
		if _, _, err := primary.ProposeDeadline([]byte(fmt.Sprintf("e2-%d", i)), 2*time.Second); err != nil {
			t.Fatalf("epoch-2 propose %d: %v", i, err)
		}
	}

	if err := primary.StartLearnerCatchup(ctxWithTimeout(t, 5*time.Second), "n3"); err != nil {
		t.Fatalf("catch-up across an epoch boundary: %v", err)
	}
	eventually(t, func() bool { return c.engines["n3"].LastApplied() == 6 }, "n3 catches across the boundary")

	want := []string{"e1-0", "e1-1", "e1-2", "e2-0", "e2-1", "e2-2"}
	got := c.appliers["n3"].payloads()
	if len(got) != len(want) {
		t.Fatalf("n3 applied %v, want %v", got, want)
	}
	for i := range want {
		if got[i] != want[i] {
			t.Fatalf("n3 applied %v, want %v (order/content must survive the split)", got, want)
		}
	}
	// And the frontier it reports back names the epoch-2 write, not the epoch-1 one.
	if fs, fe := c.engines["n3"].AppliedFrontier(); fs != 6 || fe != 2 {
		t.Fatalf("n3 frontier = (%d,%d), want (6,2)", fs, fe)
	}
}

// TestReceiveRejectsSeqReuseAcrossEpochs isolates the EPOCH half of the check
// from the frontier half: the receiver is a plain backup at a legitimate
// lastApplied, and the frame extends the right POSITION but names a predecessor
// from a different generation. That is a fork at the join point, and it must be
// refused even though the seq arithmetic lines up perfectly.
func TestReceiveRejectsSeqReuseAcrossEpochs(t *testing.T) {
	c := newCluster([]string{"n1", "n2"}, "n1", 1, []string{"n1", "n2"}, 2)
	backup := c.engines["n2"]

	if ack := backup.Receive(ReplicateMsg{Epoch: 1, Seq: 1, PrevSeq: 0, PrevEpoch: 0, Data: []byte("a")}); !ack.OK {
		t.Fatal("genesis frame on an empty node must be accepted")
	}
	before := c.appliers["n2"].count()

	// Right position (PrevSeq == 1 == lastApplied), WRONG history: the sender
	// believes seq 1 was written at epoch 3; this node holds an epoch-1 write there.
	ack := backup.Receive(ReplicateMsg{Epoch: 3, Seq: 2, PrevSeq: 1, PrevEpoch: 3, Data: []byte("b")})
	if ack.OK {
		t.Fatal("a frame whose predecessor epoch disagrees must be rejected (log matching)")
	}
	if got := c.appliers["n2"].count(); got != before {
		t.Fatalf("applied count changed on a rejected frame: %d -> %d", before, got)
	}
	if got := backup.LastApplied(); got != 1 {
		t.Fatalf("LastApplied = %d, want 1 (unchanged)", got)
	}
}

// TestReceiveRejectsNonSuccessorFrame pins the chain's successor half: a frame
// accepted at PrevSeq == frontier but carrying a Seq that is not PrevSeq+1 would
// REGRESS the applied frontier and re-open the divergent append.
func TestReceiveRejectsNonSuccessorFrame(t *testing.T) {
	c := newCluster([]string{"n1", "n2"}, "n1", 1, []string{"n1", "n2"}, 2)
	backup := c.engines["n2"]

	for s := uint64(1); s <= 3; s++ {
		var pe uint64
		if s > 1 {
			pe = 1
		}
		if ack := backup.Receive(ReplicateMsg{Epoch: 1, Seq: s, PrevSeq: s - 1, PrevEpoch: pe, Data: []byte("d")}); !ack.OK {
			t.Fatalf("seq %d must be accepted", s)
		}
	}
	if ack := backup.Receive(ReplicateMsg{Epoch: 1, Seq: 2, PrevSeq: 3, PrevEpoch: 1, Data: []byte("regress")}); ack.OK {
		t.Fatal("a frame whose Seq is not PrevSeq+1 must be rejected")
	}
	if got := backup.LastApplied(); got != 3 {
		t.Fatalf("LastApplied = %d, want 3 (the frontier must not regress)", got)
	}
}
