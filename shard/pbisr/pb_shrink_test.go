// SPDX-License-Identifier: Apache-2.0

package pbisr

import (
	"sync"
	"testing"
	"time"
)

// ============================================================================
// ISR SHRINK — engine tests.
//
// These exercise the live in-flight re-evaluation (ShrinkISR), the two halves of
// the stale-read-race fix, dropPeerLocked's discard-without-submit, and the
// idempotency of a late ack from a dropped member. The adversarial invariant
// under test throughout: a shrink NARROWS (never widens) and never loses an
// acked write nor issues a false commit.
// ============================================================================

// gatedTransport blocks each Replicate call on a shared gate so a test can park
// frames in a peer sender's channel deterministically (the sender goroutine is
// stuck in the first Replicate; later submits queue behind it). It records the
// seqs actually handed to Replicate so a test can assert a DROPPED peer's parked
// frames were discarded, never submitted. It is NOT an InlineTransport, so the
// engine always routes through the per-peer sender channel.
type gatedTransport struct {
	mu        sync.Mutex
	submitted []uint64
	checksum  byte          // xor of every submitted payload (forces a Data read under -race)
	gate      chan struct{} // closed to release every blocked Replicate
}

func newGatedTransport() *gatedTransport {
	return &gatedTransport{gate: make(chan struct{})}
}

func (g *gatedTransport) Replicate(peer string, msg ReplicateMsg, done func(AckMsg, error)) error {
	// Read the payload AT SUBMIT TIME so the -race detector can catch a frame that
	// is (wrongly) submitted while a concurrent buffer-release overwrites its Data.
	var sum byte
	for _, b := range msg.Data {
		sum ^= b
	}
	g.mu.Lock()
	g.submitted = append(g.submitted, msg.Seq)
	g.checksum ^= sum
	g.mu.Unlock()
	<-g.gate // block until the test releases
	done(AckMsg{Epoch: msg.Epoch, Seq: msg.Seq, OK: true}, nil)
	return nil
}

func (g *gatedTransport) submittedSeqs() []uint64 {
	g.mu.Lock()
	defer g.mu.Unlock()
	return append([]uint64(nil), g.submitted...)
}

func (g *gatedTransport) release() { close(g.gate) }

// --- (a) A shrink un-wedges a stalled head, which commits on the survivor -----

// TestShrinkUnwedgesStalledHead: with a silent backup, an in-flight write's head
// stalls owing the dead member. A MetaRaft-committed shrink that removes the dead
// member live-re-evaluates the in-flight record; it now owes only the survivor
// (which already acked), so it commits and committed sweeps to it — WITHOUT the
// write ever having been shipped to, or acked by, the removed member.
func TestShrinkUnwedgesStalledHead(t *testing.T) {
	c := newCluster([]string{"n1", "n2", "n3"}, "n1", 1, []string{"n1", "n2", "n3"}, 2)
	primary := c.engines["n1"]
	defer primary.Shutdown()

	// n3 is silent: full-ISR commit REQUIRES it, so the write stalls in flight.
	c.tr.setFault("n3", peerFault{partition: true})

	done := make(chan struct {
		seq uint64
		err error
	}, 1)
	go func() {
		_, seq, err := primary.Propose(ctxWithTimeout(t, 5*time.Second), []byte("w"))
		done <- struct {
			seq uint64
			err error
		}{seq, err}
	}()

	// The reachable backup applies the in-flight write; the head now owes only n3.
	eventually(t, func() bool { return c.engines["n2"].LastApplied() == 1 },
		"n2 applies the in-flight write")
	if got := primary.Committed(); got != 0 {
		t.Fatalf("committed = %d, want 0 (silent n3 blocks the full-ISR commit)", got)
	}

	// MetaRaft-committed shrink drops the dead n3. The in-flight head re-evaluates
	// and commits against {n1,n2} without n3 ever acking.
	primary.ShrinkISR(1, []string{"n1", "n2"})

	select {
	case r := <-done:
		if r.err != nil {
			t.Fatalf("post-shrink propose: %v (re-wedged?)", r.err)
		}
		if r.seq != 1 {
			t.Fatalf("seq = %d, want 1", r.seq)
		}
	case <-time.After(3 * time.Second):
		t.Fatal("propose did not commit after shrink — the stalled head never re-evaluated")
	}
	if got := primary.Committed(); got != 1 {
		t.Fatalf("committed = %d, want 1 (shrink un-wedged the head)", got)
	}
	// The removed member never received the write — commit rests on {n1,n2} only.
	if got := c.appliers["n3"].count(); got != 0 {
		t.Fatalf("removed n3 applied %d, want 0", got)
	}
}

// --- (b) A stale-epoch or unleased shrink is a no-op ---------------------------

func TestShrinkStaleEpochAndUnleasedAreNoOp(t *testing.T) {
	// Future-epoch shrink: epoch != e.epoch → no-op, no override installed.
	c := newCluster([]string{"n1", "n2", "n3"}, "n1", 1, []string{"n1", "n2", "n3"}, 2)
	primary := c.engines["n1"]
	defer primary.Shutdown()

	primary.ShrinkISR(2, []string{"n1", "n2"}) // epoch 2, engine is at 1
	primary.mu.Lock()
	stale := primary.effISREpoch
	primary.mu.Unlock()
	if stale != 0 {
		t.Fatalf("future-epoch shrink installed an override (effISREpoch=%d), want none", stale)
	}

	primary.ShrinkISR(0, []string{"n1"}) // epoch 0, below current
	primary.mu.Lock()
	stale = primary.effISREpoch
	primary.mu.Unlock()
	if stale != 0 {
		t.Fatalf("stale-epoch shrink installed an override (effISREpoch=%d), want none", stale)
	}

	// A node that adopted an epoch but holds NO lease for it must not shrink.
	unleased := New("x1", testShard, c.ctrl, c.tr, &fakeApplier{}, WithClock(c.clk.now))
	defer unleased.Shutdown()
	unleased.AdoptEpoch(1) // epoch=1, but leaseEpoch stays 0
	unleased.ShrinkISR(1, []string{"x1"})
	unleased.mu.Lock()
	got := unleased.effISREpoch
	unleased.mu.Unlock()
	if got != 0 {
		t.Fatalf("unleased shrink installed an override (effISREpoch=%d), want none", got)
	}
}

// --- (c) THE STALE-READ RACE — the override drops a member a stale snapshot kept

// TestShrinkStaleReadRaceOverrideDropsMember is the correctness crux. It models a
// Propose that snapshotted the PRE-shrink ISR (ctrl.ISR is deliberately left
// unchanged, still naming the dead member) AFTER the shrink override was
// installed. Without half 2 of the fix (the proposeSequenced intersect) that
// Propose would register the dead member as required and re-wedge the pipeline.
// With it, the override narrows the stale snapshot under e.mu, so the write needs
// only the survivor and commits.
func TestShrinkStaleReadRaceOverrideDropsMember(t *testing.T) {
	c := newCluster([]string{"n1", "n2", "n3"}, "n1", 1, []string{"n1", "n2", "n3"}, 2)
	primary := c.engines["n1"]
	defer primary.Shutdown()
	c.tr.setFault("n3", peerFault{partition: true}) // n3 dead & silent

	// Install the shrink override for epoch 1, but LEAVE ctrl.ISR = {n1,n2,n3}: a
	// subsequent Propose therefore snapshots the stale pre-shrink set.
	primary.ShrinkISR(1, []string{"n1", "n2"})

	_, seq, err := primary.Propose(ctxWithTimeout(t, 2*time.Second), []byte("w"))
	if err != nil {
		t.Fatalf("propose after shrink: %v — the stale snapshot re-registered the dead n3 (intersect missing)", err)
	}
	if seq != 1 {
		t.Fatalf("seq = %d, want 1", seq)
	}
	if got := primary.Committed(); got != 1 {
		t.Fatalf("committed = %d, want 1 (override must narrow the stale snapshot)", got)
	}
	// Proof the intersect (not the ctrl view) did the narrowing: n3 got nothing.
	if got := c.appliers["n3"].count(); got != 0 {
		t.Fatalf("removed n3 applied %d, want 0 — the override did not drop it", got)
	}
}

// --- (d) Under -race: shrink un-stalls, the ring wraps, no aliased-buffer reuse -

// TestShrinkDiscardNoAliasedBufferReuseRace drives the WithDataRelease aliasing
// scenario: parked frames for a dead peer alias payload buffers the catch-up ring
// later evicts and RELEASES (the release hook overwrites the buffer). Once the
// shrink drops that peer, its sender must discard the parked frames WITHOUT
// submitting them — so the transport never reads a buffer the release hook is
// concurrently overwriting. Run under -race; the detector is the assertion.
func TestShrinkDiscardNoAliasedBufferReuseRace(t *testing.T) {
	ctrl := &fakeControl{epoch: 1, primary: "n1", isr: []string{"n1", "n2"}, minISR: 1}
	clk := &fakeClock{t: t0}
	g := newGatedTransport()
	e := New("n1", testShard, ctrl, g, &fakeApplier{}, WithClock(clk.now))
	e.GrantLease(1, t0+leaseDur)
	defer e.Shutdown()

	// Park frames 1..6 for n2, each aliasing its own payload buffer. The sender
	// goroutine blocks in the first gated Replicate (having read frame 1's Data);
	// frames 2..6 queue behind it, unsubmitted.
	const nFrames = 6
	bufs := make([][]byte, nFrames+1)
	e.writeMu.Lock()
	for s := uint64(1); s <= nFrames; s++ {
		buf := make([]byte, 8)
		for i := range buf {
			buf[i] = byte(s)
		}
		bufs[s] = buf
		e.submitPeerLocked("n2", ReplicateMsg{Epoch: 1, Seq: s, PrevSeq: s - 1, Data: buf})
	}
	e.writeMu.Unlock()

	eventually(t, func() bool { return len(g.submittedSeqs()) == 1 },
		"frame 1 submitted (Data read), 2..6 parked")

	// Shrink n2 out: dropPeerLocked latches discard + closes the channel, so the
	// sender will drain frames 2..6 WITHOUT submitting them.
	ctrl.setISR([]string{"n1"})
	e.ShrinkISR(1, []string{"n1"})

	// Release the blocked Replicate AND concurrently recycle the parked frames'
	// buffers (models the catch-up ring's WithDataRelease eviction now that the
	// window can advance). If discard were broken, the sender would submit a parked
	// frame — reading its Data — concurrently with this overwrite, and -race would
	// fire. With discard correct, no parked frame is ever read: no race.
	go func() {
		for s := 2; s <= nFrames; s++ {
			for i := range bufs[s] {
				bufs[s][i] = 0xEE
			}
		}
	}()
	g.release()

	eventually(t, func() bool {
		e.writeMu.Lock()
		_, has := e.peerQ["n2"]
		e.writeMu.Unlock()
		return !has
	}, "n2 sender torn down")
	// Only frame 1 (in flight at drop time) was ever submitted; 2..6 were discarded.
	if got := g.submittedSeqs(); len(got) != 1 || got[0] != 1 {
		t.Fatalf("submitted = %v, want [1] (parked frames must be discarded, not submitted)", got)
	}
}

// --- (e) dropPeerLocked discards parked frames without submitting --------------

func TestDropPeerDiscardsParkedFramesWithoutSubmitting(t *testing.T) {
	ctrl := &fakeControl{epoch: 1, primary: "n1", isr: []string{"n1", "n2"}, minISR: 1}
	clk := &fakeClock{t: t0}
	g := newGatedTransport()
	e := New("n1", testShard, ctrl, g, &fakeApplier{}, WithClock(clk.now))
	e.GrantLease(1, t0+leaseDur)
	defer e.Shutdown()

	// Park frames 1..4 for n2 (sender blocks in Replicate on frame 1).
	e.writeMu.Lock()
	for s := uint64(1); s <= 4; s++ {
		e.submitPeerLocked("n2", ReplicateMsg{Epoch: 1, Seq: s, PrevSeq: s - 1, Data: []byte("x")})
	}
	e.writeMu.Unlock()

	eventually(t, func() bool { return len(g.submittedSeqs()) == 1 },
		"frame 1 submitted, 2..4 parked")

	// Drop n2 directly (the ShrinkISR path calls this under writeMu+e.mu).
	e.writeMu.Lock()
	e.mu.Lock()
	e.dropPeerLocked("n2")
	e.mu.Unlock()
	e.writeMu.Unlock()

	g.release() // let the blocked Replicate return; the sender then discards 2..4

	eventually(t, func() bool {
		e.writeMu.Lock()
		_, has := e.peerQ["n2"]
		e.writeMu.Unlock()
		return !has
	}, "n2 sender removed from peerQ")
	// Give the sender a moment to (not) submit the parked frames after release.
	time.Sleep(50 * time.Millisecond)
	if got := g.submittedSeqs(); len(got) != 1 || got[0] != 1 {
		t.Fatalf("submitted = %v, want [1] — parked frames 2..4 must be discarded", got)
	}
}

// --- (f) A late ack from the dropped member is harmless ------------------------

// TestLateAckFromDroppedPeerNoDoubleCommit: the removed member acks AFTER the
// shrink already committed and popped the record. That late ack must be a no-op —
// removePending is idempotent and resolveLocked is exactly-once — so no double
// commit and no panic.
func TestLateAckFromDroppedPeerNoDoubleCommit(t *testing.T) {
	c := newCluster([]string{"n1", "n2", "n3"}, "n1", 1, []string{"n1", "n2", "n3"}, 2)
	primary := c.engines["n1"]
	defer primary.Shutdown()

	// n3 acks LATE (after we will have shrunk it out); n2 acks promptly.
	c.tr.setFault("n3", peerFault{delay: 400 * time.Millisecond})

	done := make(chan error, 1)
	go func() {
		_, _, err := primary.Propose(ctxWithTimeout(t, 5*time.Second), []byte("w"))
		done <- err
	}()

	eventually(t, func() bool { return c.engines["n2"].LastApplied() == 1 },
		"n2 applies the in-flight write")

	// Shrink n3 out BEFORE its delayed ack lands: the head commits on {n1,n2} and
	// the record is popped.
	primary.ShrinkISR(1, []string{"n1", "n2"})
	if err := <-done; err != nil {
		t.Fatalf("propose: %v", err)
	}
	if got := primary.Committed(); got != 1 {
		t.Fatalf("committed = %d, want 1", got)
	}

	// n3's delayed ack now arrives for the already-committed, already-popped record.
	// It must not double-commit or panic; committed stays 1.
	time.Sleep(600 * time.Millisecond)
	if got := primary.Committed(); got != 1 {
		t.Fatalf("committed = %d after the late ack, want 1 (no double-commit)", got)
	}
}
