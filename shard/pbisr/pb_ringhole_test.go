// SPDX-License-Identifier: Apache-2.0

package pbisr

import (
	"fmt"
	"testing"
	"time"
)

// setView mutates a fakeControl's whole view atomically (test-only helper): the
// re-failover scenarios below move one node through THREE control-plane views
// (backup → promoted primary → ...), which the per-field setters cannot express.
func (c *fakeControl) setView(epoch uint64, primary string, isr []string, minISR int) {
	c.mu.Lock()
	c.epoch, c.primary, c.isr, c.minISR = epoch, primary, isr, minISR
	c.mu.Unlock()
}

// TestRingAtRejectsHoleSeq is the DIRECT probe for the C1 defect: ring.at()
// derived its slot arithmetically from `oldest`, which is only valid while the
// retained seqs are DENSE. A ring holding a HOLE (see
// TestPromoteResetsBacklogSoNoHoleFormsInRing for how one used to form) then
// answered `ok=true` for a seq it does not hold, returning either a ZERO-VALUED
// entry (epoch 0, prevEpoch 0) or — worse — a DIFFERENT write's entry.
//
// A zero entry read back through checkCatchupDivergenceLocked is garbage that
// compares unequal to any real epoch, so a replica holding a perfectly clean
// prefix was declared ErrCatchupDiverged. That is the bug; this pins the fix at
// its source, independent of how a hole comes to exist: at() must verify that
// the slot it computed actually HOLDS the requested seq.
func TestRingAtRejectsHoleSeq(t *testing.T) {
	r := newRing(16)
	// A non-dense retention: 1..5 from an old primary term, then 9 from a new one
	// (the shape Promote used to leave behind). span() reports [1..9] either way.
	for _, s := range []uint64{1, 2, 3, 4, 5} {
		r.append(ringEntry{epoch: 1, seq: s, prevEpoch: 1, data: []byte(fmt.Sprintf("a%d", s))})
	}
	r.append(ringEntry{epoch: 3, seq: 9, prevEpoch: 2, data: []byte("b9")})

	oldest, newest, ok := r.span()
	if !ok || oldest != 1 || newest != 9 {
		t.Fatalf("span = (%d,%d,%v), want (1,9,true) — the pre-condition for the defect", oldest, newest, ok)
	}
	if r.len() != 6 {
		t.Fatalf("ring len = %d, want 6 (the span is 9 wide but only 6 seqs are retained)", r.len())
	}

	// Every seq inside the span but NOT retained must be reported absent.
	for _, hole := range []uint64{6, 7, 8} {
		if ent, has := r.at(hole); has {
			t.Fatalf("at(%d) = (%+v, true) for a seq the ring does not hold — a hole must report ok=false", hole, ent)
		}
	}
	// Every seq BELOW the hole still resolves, correctly, at its own identity: the
	// fix must not over-reject the dense prefix.
	for _, s := range []uint64{1, 2, 3, 4, 5} {
		ent, has := r.at(s)
		if !has || ent.seq != s || ent.epoch != 1 {
			t.Fatalf("at(%d) = (%+v, %v), want the retained epoch-1 entry", s, ent, has)
		}
	}
	// Seq 9 is PHYSICALLY retained but sits at the slot the hole displaced it from,
	// so a derived-index lookup cannot address it: at() reports it absent. That is
	// the CONSERVATIVE direction and the only safe one — "I hold something here"
	// must never be answered from arithmetic alone. Callers read absent as
	// ring-cold and fall back to a snapshot, which is correct for a ring whose
	// density has been broken. (Promote no longer breaks it; see
	// TestPromoteResetsBacklogSoNoHoleFormsInRing.)
	if ent, has := r.at(9); has {
		t.Fatalf("at(9) = (%+v, true) across a hole; a displaced entry must report absent, not be served from a derived slot", ent)
	}
	if _, has := r.at(10); has {
		t.Fatal("at(10) past the newest retained seq must be absent")
	}
}

// TestRingDeltaRejectsHoledRange reaches ringDeltaLocked's HOLE branch (`if
// !has { return nil, false }`). Before the at() fix that branch was DEAD CODE:
// at() answered ok=true for every seq inside the span, hole or not, so the loop
// silently shipped zero-valued entries as if they were writes. With at() honest
// the branch is live, and it is the thing that turns "I cannot serve this range
// from the ring" into a clean ring-cold abort (→ snapshot) instead of a
// fabricated delta.
func TestRingDeltaRejectsHoledRange(t *testing.T) {
	e := New("n1", testShard, &fakeControl{epoch: 1, primary: "n1", isr: []string{"n1"}, minISR: 1}, newInMemTransport(), &fakeApplier{})
	defer e.Shutdown()

	// Hand-build a HOLED backlog: seqs 1..3 retained, 4..5 missing, 6 retained.
	e.mu.Lock()
	for _, s := range []uint64{1, 2, 3} {
		e.backlog.append(ringEntry{epoch: 1, seq: s, prevEpoch: 1, data: []byte("x")})
	}
	e.backlog.append(ringEntry{epoch: 2, seq: 6, prevEpoch: 2, data: []byte("y")})

	// A range that CROSSES the hole cannot be served: ring-cold, not a short read
	// and certainly not a fabricated one.
	if msgs, ok := e.ringDeltaLocked(2, 6); ok {
		e.mu.Unlock()
		t.Fatalf("ringDeltaLocked(2,6) over a hole returned ok=true with %d msgs; want (nil,false)", len(msgs))
	}
	// A range that STARTS in the hole is equally unservable.
	if _, ok := e.ringDeltaLocked(4, 6); ok {
		e.mu.Unlock()
		t.Fatal("ringDeltaLocked(4,6) starting inside a hole returned ok=true; want (nil,false)")
	}
	// A range entirely BELOW the hole is still served normally (the branch must
	// not over-reject).
	msgs, ok := e.ringDeltaLocked(2, 3)
	e.mu.Unlock()
	if !ok || len(msgs) != 2 || msgs[0].Seq != 2 || msgs[1].Seq != 3 {
		t.Fatalf("ringDeltaLocked(2,3) = (%+v,%v), want the dense 2..3 delta", msgs, ok)
	}
}

// TestPromoteResetsBacklogSoNoHoleFormsInRing is the reviewer's END-TO-END probe.
//
// It builds the exact state that made a CLEAN replica report as diverged, on the
// blessed path:
//
//	epoch 1 — n1 is primary and commits 5 writes to the full ISR {n1,n2,n3}, so
//	          n1's backlog retains 1..5 (a primary appends its OWN writes only).
//	epoch 2 — n2 is promoted and extends the same history with 6..8. n1 and n3
//	          RECEIVE those as backups — and a backup receive never appends to the
//	          ring — so n1's backlog still holds exactly 1..5 while its applied
//	          frontier is now 8.
//	epoch 3 — n1 is promoted back. Promote resumes seq assignment from the applied
//	          frontier, so its next write is seq 9, and appending it to a backlog
//	          that still ends at 5 left a HOLE at 6..8 under a span of [1..9].
//
// n3 then holds a PERFECT PREFIX of n1's history (1..8) and needs exactly one
// write. Before the fix, checkCatchupDivergenceLocked read at(9), got a
// zero-valued entry with ok=true, compared its epoch 0 against n3's real epoch 2
// and returned ErrCatchupDiverged — condemning a clean replica to a full snapshot
// transfer (and, pre-4.3, to permanent exclusion).
//
// The assertion is therefore twofold: no hole may form at all (Promote resets the
// backlog), and the clean-prefix replica must FLIP, applying only the one write
// it is actually missing.
func TestPromoteResetsBacklogSoNoHoleFormsInRing(t *testing.T) {
	clk := &fakeClock{t: t0}
	tr := newInMemTransport()
	ctrl1 := &fakeControl{epoch: 1, primary: "n1", isr: []string{"n1", "n2", "n3"}, minISR: 3}
	ctrl2 := &fakeControl{epoch: 2, primary: "n2", isr: []string{"n2", "n3"}, minISR: 2}
	ctrl3 := &fakeControl{epoch: 1, primary: "n1", isr: []string{"n1", "n2", "n3"}, minISR: 3}
	ap1, ap2, ap3 := &fakeApplier{}, &fakeApplier{}, &fakeApplier{}
	n1 := New("n1", testShard, ctrl1, tr, ap1, WithClock(clk.now))
	n2 := New("n2", testShard, ctrl2, tr, ap2, WithClock(clk.now))
	n3 := New("n3", testShard, ctrl3, tr, ap3, WithClock(clk.now))
	tr.register("n1", n1)
	tr.register("n2", n2)
	tr.register("n3", n3)
	defer n1.Shutdown()
	defer n2.Shutdown()
	defer n3.Shutdown()

	// EPOCH 1 — n1 commits 5 writes on the FULL ISR; n2 and n3 hold all of them.
	n1.GrantLease(1, t0+leaseDur)
	for i := 0; i < 5; i++ {
		if _, _, err := n1.ProposeDeadline([]byte(fmt.Sprintf("a%03d", i)), 2*time.Second); err != nil {
			t.Fatalf("n1 propose %d: %v", i, err)
		}
	}
	eventually(t, func() bool { return n2.LastApplied() == 5 && n3.LastApplied() == 5 }, "n2/n3 receive all 5 epoch-1 writes")
	if got := n1.backlogLen(); got != 5 {
		t.Fatalf("n1 backlog holds %d entries after 5 own writes, want 5", got)
	}

	// EPOCH 2 — n2 takes over and extends the SAME history with 3 more writes.
	n2.Promote(2, t0+leaseDur)
	for i := 0; i < 3; i++ {
		if _, _, err := n2.ProposeDeadline([]byte(fmt.Sprintf("b%03d", i)), 2*time.Second); err != nil {
			t.Fatalf("n2 propose %d: %v", i, err)
		}
	}
	eventually(t, func() bool { return n3.LastApplied() == 8 }, "n3 receives the 3 epoch-2 writes")
	// n1 was OUT of epoch 2's ISR; catch it up so it, too, holds 1..8. This is the
	// step that makes n1's applied frontier (8) run AHEAD of its own backlog (1..5)
	// — backup receives never append to the ring.
	ctrl1.setView(2, "n2", []string{"n2", "n3"}, 2)
	if err := n2.StartLearnerCatchup(ctxWithTimeout(t, 5*time.Second), "n1"); err != nil {
		t.Fatalf("n1 catch-up under epoch 2: %v", err)
	}
	eventually(t, func() bool { return n1.LastApplied() == 8 }, "n1 catches up to seq 8 as a backup")

	// EPOCH 3 — n1 is promoted BACK. Its backlog must not survive the promotion:
	// the entries in it belong to a superseded term, and the next seq it assigns is
	// frontier+1 == 9, which would leave 6..8 as a hole under a [1..9] span.
	n1.Promote(3, t0+leaseDur)
	if got := n1.backlogLen(); got != 0 {
		t.Fatalf("n1 backlog holds %d entries after Promote, want 0 (a superseded term's writes must not survive)", got)
	}
	ctrl1.setView(3, "n1", []string{"n1"}, 1) // write alone: n3 must stay at seq 8
	if _, _, err := n1.ProposeDeadline([]byte("c000"), 2*time.Second); err != nil {
		t.Fatalf("n1 propose under epoch 3: %v", err)
	}

	// The backlog is DENSE at the new term's writes, and nothing below them is
	// claimed: span == [9..9], and every would-be hole reports absent.
	n1.mu.Lock()
	oldest, newest, ok := n1.backlog.span()
	holes := make([]uint64, 0, 3)
	for s := uint64(6); s <= 8; s++ {
		if _, has := n1.backlog.at(s); has {
			holes = append(holes, s)
		}
	}
	n1.mu.Unlock()
	if !ok || oldest != 9 || newest != 9 {
		t.Fatalf("n1 backlog span = (%d,%d,%v), want (9,9,true) — dense at the new term", oldest, newest, ok)
	}
	if len(holes) != 0 {
		t.Fatalf("n1 backlog claims to hold superseded-term seqs %v", holes)
	}

	// THE PAYOFF: n3 holds a perfect prefix (1..8 at epoch 2) and needs exactly the
	// one write it is missing. It must FLIP — not be declared diverged, and not be
	// routed to a full snapshot transfer for a one-write lag.
	if fs, fe := n3.AppliedFrontier(); fs != 8 || fe != 2 {
		t.Fatalf("n3 frontier = (%d,%d), want (8,2) — a clean prefix of n1's history", fs, fe)
	}
	if err := n1.StartLearnerCatchup(ctxWithTimeout(t, 5*time.Second), "n3"); err != nil {
		t.Fatalf("catch-up of a CLEAN-PREFIX replica must succeed, got: %v", err)
	}
	if !n1.IsLearner("n3") {
		t.Fatal("n3 must be flipped into the learner ship-set")
	}
	eventually(t, func() bool { return n3.LastApplied() == 9 }, "n3 receives the single missing write")

	want := []string{"a000", "a001", "a002", "a003", "a004", "b000", "b001", "b002", "c000"}
	got := ap3.payloads()
	if len(got) != len(want) {
		t.Fatalf("n3 applied %v, want %v (no re-application of the 8 it already held)", got, want)
	}
	for i := range want {
		if got[i] != want[i] {
			t.Fatalf("n3 applied %v, want %v", got, want)
		}
	}
}
