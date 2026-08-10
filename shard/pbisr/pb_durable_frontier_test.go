// SPDX-License-Identifier: Apache-2.0

package pbisr

import (
	"testing"
	"time"
)

// ============================================================================
// RESTORED FRONTIER (the engine half).
//
// A PB node's DATA survives a restart (the cache warm-restarts from its mmap
// pages) but a fresh Engine has every watermark at zero. The durable frontier persists the
// applied frontier and seeds it back through WithRestoredFrontier. These tests
// pin the engine-visible consequences:
//
//	(1) a restored node's log identity is the restored pair, in every read path;
//	(2) it therefore REJECTS a genesis frame — the 4.1 divergent-append check now
//	    also covers the restarted-node case it could not see before;
//	(3) an UNDER-reported restore (the amortised watermark trailing reality, which
//	    is the only direction the persisted value can be wrong) is caught up by the
//	    normal grow path;
//	(4) a restore that disagrees with the primary's history is rejected CLEANLY,
//	    never silently mis-applied.
// ============================================================================

// restartEngine models a process restart of nodeID: a brand-new Engine — every
// in-memory watermark back to zero — seeded ONLY with what durable storage could
// hand back, then re-registered on the shared transport. The fresh applier stands
// in for the warm-restarted FSM.
func restartEngine(c *cluster, nodeID string, seq, epoch uint64) *fakeApplier {
	ap := &fakeApplier{}
	eng := New(nodeID, testShard, c.ctrl, c.tr, ap,
		WithClock(c.clk.now), WithRestoredFrontier(seq, epoch))
	c.engines[nodeID] = eng
	c.appliers[nodeID] = ap
	c.tr.register(nodeID, eng)
	return ap
}

// TestRestoredFrontierIsTheNodesLogIdentity: a restored node reports the restored
// pair from every path that answers "what does your log end with" — and reports it
// WITHOUT claiming it assigned those seqs as primary (lastSeq stays 0, which is
// the truth: the node did not propose them in this process).
func TestRestoredFrontierIsTheNodesLogIdentity(t *testing.T) {
	c := newCluster([]string{"n1", "n2"}, "n1", 1, []string{"n1", "n2"}, 2)
	defer c.engines["n1"].Shutdown()
	restartEngine(c, "n2", 30, 2)
	n2 := c.engines["n2"]
	defer n2.Shutdown()

	if seq, epoch := n2.AppliedFrontier(); seq != 30 || epoch != 2 {
		t.Fatalf("AppliedFrontier = (%d,%d), want (30,2)", seq, epoch)
	}
	info := n2.CatchupInfo()
	if info.FrontierSeq != 30 || info.FrontierEpoch != 2 {
		t.Fatalf("CatchupInfo frontier = (%d,%d), want (30,2) — a restarted node must not answer a grow handshake with a lie",
			info.FrontierSeq, info.FrontierEpoch)
	}
	if info.AppliedSeq != 30 {
		t.Fatalf("CatchupInfo AppliedSeq = %d, want 30", info.AppliedSeq)
	}
	if got := n2.LastSeq(); got != 0 {
		t.Fatalf("LastSeq = %d, want 0 — restoring must not claim the node PROPOSED those writes", got)
	}
}

// TestRestoredFrontierRejectsGenesisFrame is the defect this stage closes, stated
// as a single assertion. Before 4.2 a restarted node presented (0,0) over real
// data, so a genesis frame — "I extend from nothing" — matched it and appended a
// foreign history onto a full FSM. With the frontier restored, that frame
// log-match-rejects and only the true successor is accepted.
func TestRestoredFrontierRejectsGenesisFrame(t *testing.T) {
	c := newCluster([]string{"n1", "n2"}, "n1", 1, []string{"n1", "n2"}, 2)
	defer c.engines["n1"].Shutdown()
	ap := restartEngine(c, "n2", 30, 2)
	n2 := c.engines["n2"]
	defer n2.Shutdown()

	genesis := ReplicateMsg{Epoch: 2, Seq: 1, PrevSeq: 0, PrevEpoch: 0, Data: []byte("forged")}
	if ack := n2.Receive(genesis); ack.OK {
		t.Fatal("a restarted node with real data ACCEPTED a genesis frame — the divergent append is back")
	}
	if ap.count() != 0 {
		t.Fatal("the rejected frame was applied anyway")
	}

	// The true successor of the restored frontier is accepted.
	ok := ReplicateMsg{Epoch: 2, Seq: 31, PrevSeq: 30, PrevEpoch: 2, Data: []byte("real")}
	if ack := n2.Receive(ok); !ack.OK {
		t.Fatal("the genuine successor of the restored frontier was rejected")
	}
	if seq, epoch := n2.AppliedFrontier(); seq != 31 || epoch != 2 {
		t.Fatalf("frontier after accepting the successor = (%d,%d), want (31,2)", seq, epoch)
	}
}

// TestRestoredFrontierRejectsMidHistoryReplay: a frame naming a predecessor the
// restored node is PAST (or one it never reached) is rejected. Only the exact
// successor matches — the frontier is an identity, not a floor.
func TestRestoredFrontierRejectsMidHistoryReplay(t *testing.T) {
	c := newCluster([]string{"n1", "n2"}, "n1", 1, []string{"n1", "n2"}, 2)
	defer c.engines["n1"].Shutdown()
	restartEngine(c, "n2", 30, 2)
	n2 := c.engines["n2"]
	defer n2.Shutdown()

	cases := []struct {
		name string
		msg  ReplicateMsg
	}{
		{"predecessor already passed", ReplicateMsg{Epoch: 2, Seq: 20, PrevSeq: 19, PrevEpoch: 2}},
		{"predecessor not yet reached", ReplicateMsg{Epoch: 2, Seq: 41, PrevSeq: 40, PrevEpoch: 2}},
		{"right position, wrong epoch", ReplicateMsg{Epoch: 3, Seq: 31, PrevSeq: 30, PrevEpoch: 1}},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			if ack := n2.Receive(tc.msg); ack.OK {
				t.Fatalf("accepted %+v against a restored frontier of (30,2)", tc.msg)
			}
		})
	}
}

// TestRestoredUnderReportIsCaughtUpByGrow is requirement (4) of the stage: the
// persisted watermark is amortised, so a restarted node's frontier can TRAIL what
// its FSM actually holds. That node must still be recoverable — it is offered a
// delta from the earlier point, and the existing grow path must carry it to the
// primary's high-water without a gap.
//
// n2 applies 50 writes as a backup, then "restarts" having only persisted 30. The
// primary's ring still holds 31..50, so the handshake resumes there.
func TestRestoredUnderReportIsCaughtUpByGrow(t *testing.T) {
	c := newCluster([]string{"n1", "n2"}, "n1", 1, []string{"n1", "n2"}, 2)
	primary := c.engines["n1"]
	defer primary.Shutdown()

	const n = 50
	for i := 0; i < n; i++ {
		if _, _, err := primary.Propose(ctxWithTimeout(t, 5*time.Second), []byte{byte(i)}); err != nil {
			t.Fatalf("propose %d: %v", i, err)
		}
	}
	if got := c.engines["n2"].LastApplied(); got != n {
		t.Fatalf("setup: n2 LastApplied = %d, want %d", got, n)
	}
	c.engines["n2"].Shutdown()

	// The restart: the amortised stamp had only reached 30 when the process died,
	// so that is all durable storage can hand back. Under-reporting by 20 writes.
	const persisted = 30
	restartEngine(c, "n2", persisted, 1)
	defer c.engines["n2"].Shutdown()
	// The primary must keep writing while n2 is out, so drop it from the ISR.
	c.ctrl.setISR([]string{"n1"})
	c.ctrl.minISR = 1

	if err := primary.StartLearnerCatchup(ctxWithTimeout(t, 5*time.Second), "n2"); err != nil {
		t.Fatalf("StartLearnerCatchup on an under-reporting restart: %v", err)
	}
	eventually(t, func() bool {
		seq, _ := c.engines["n2"].AppliedFrontier()
		return seq == n
	}, "the under-reporting restart is caught up to the primary's high-water")

	// It received exactly the delta it was missing — 31..50 — not the whole log.
	if got := c.appliers["n2"].count(); got != n-persisted {
		t.Fatalf("n2 re-applied %d writes, want exactly the %d-write delta from its persisted watermark",
			got, n-persisted)
	}
	// ...and it now extends cleanly: the next real write lands.
	c.ctrl.setISR([]string{"n1", "n2"})
	primary.GrowISR(1, []string{"n1", "n2"})
	if _, _, err := primary.Propose(ctxWithTimeout(t, 5*time.Second), []byte{0xFF}); err != nil {
		t.Fatalf("post-catch-up propose: %v — a gap or mismatch would surface here", err)
	}
}

// TestRestoredDivergentFrontierRejectsCleanly is requirement (4)'s other half. An
// under-report must be caught up; a restore that names a write the primary's
// history does NOT contain must be REFUSED, loudly, not papered over by appending
// on top of it. (Repairing that case needs a full state transfer.)
func TestRestoredDivergentFrontierRejectsCleanly(t *testing.T) {
	t.Run("restored ahead of the primary", func(t *testing.T) {
		c := newCluster([]string{"n1", "n2"}, "n1", 1, []string{"n1"}, 1)
		primary := c.engines["n1"]
		defer primary.Shutdown()
		for i := 0; i < 10; i++ {
			if _, _, err := primary.Propose(ctxWithTimeout(t, 5*time.Second), []byte{byte(i)}); err != nil {
				t.Fatalf("propose %d: %v", i, err)
			}
		}
		// n2 restarts holding a write the primary never assigned (it forked ahead).
		restartEngine(c, "n2", 99, 1)
		defer c.engines["n2"].Shutdown()

		err := primary.StartLearnerCatchup(ctxWithTimeout(t, 5*time.Second), "n2")
		if err == nil {
			t.Fatal("catch-up ACCEPTED a target whose restored frontier is ahead of the primary's own")
		}
	})

	t.Run("same position, different history", func(t *testing.T) {
		c := newCluster([]string{"n1", "n2"}, "n1", 1, []string{"n1"}, 1)
		primary := c.engines["n1"]
		defer primary.Shutdown()
		for i := 0; i < 10; i++ {
			if _, _, err := primary.Propose(ctxWithTimeout(t, 5*time.Second), []byte{byte(i)}); err != nil {
				t.Fatalf("propose %d: %v", i, err)
			}
		}
		// Seq 5 exists on both sides — but under a different epoch, so it is a
		// DIFFERENT write. Position alone must not be mistaken for agreement.
		restartEngine(c, "n2", 5, 7)
		defer c.engines["n2"].Shutdown()

		if err := primary.StartLearnerCatchup(ctxWithTimeout(t, 5*time.Second), "n2"); err == nil {
			t.Fatal("catch-up accepted a target holding a different write at the same seq")
		}
		if got := c.appliers["n2"].count(); got != 0 {
			t.Fatalf("the diverged target was shipped %d writes; it must be shipped none", got)
		}
	})
}

// TestRestoredGenesisIsANoOp: (0,0) is what a fresh data dir, a heap-mode cache,
// and a Raft-mode shard all hand back, so the seam is called unconditionally. It
// must leave a genuinely blank node genuinely blank — still accepting a genesis
// frame.
func TestRestoredGenesisIsANoOp(t *testing.T) {
	c := newCluster([]string{"n1", "n2"}, "n1", 1, []string{"n1", "n2"}, 2)
	defer c.engines["n1"].Shutdown()
	restartEngine(c, "n2", 0, 0)
	n2 := c.engines["n2"]
	defer n2.Shutdown()

	if seq, epoch := n2.AppliedFrontier(); seq != 0 || epoch != 0 {
		t.Fatalf("AppliedFrontier = (%d,%d), want genesis (0,0)", seq, epoch)
	}
	genesis := ReplicateMsg{Epoch: 1, Seq: 1, PrevSeq: 0, PrevEpoch: 0, Data: []byte("first")}
	if ack := n2.Receive(genesis); !ack.OK {
		t.Fatal("a genuinely blank node rejected a genesis frame")
	}
}

// TestFrontierSinkSeesEveryMaterializedAdvance pins the persistence seam itself:
// the sink must fire on BOTH apply paths, must report exactly what
// AppliedFrontier would report at that instant, and must never report a value the
// Applier has not returned from — that last property is what makes the persisted
// watermark unable to over-report.
func TestFrontierSinkSeesEveryMaterializedAdvance(t *testing.T) {
	t.Run("primary path", func(t *testing.T) {
		ctrl := &fakeControl{epoch: 1, primary: "n1", isr: []string{"n1"}, minISR: 1}
		clk := &fakeClock{t: t0}
		ap := &fakeApplier{}
		var seen [][2]uint64
		eng := New("n1", testShard, ctrl, newInMemTransport(), ap, WithClock(clk.now),
			WithFrontierSink(func(seq, epoch uint64) {
				// Called under e.mu, right after the apply: the write must already be
				// materialized. applied[seq-1] is this very write.
				if int(seq) > ap.count() {
					t.Errorf("sink reported seq %d but only %d writes have been applied — an OVER-REPORT",
						seq, ap.count())
				}
				seen = append(seen, [2]uint64{seq, epoch})
			}))
		eng.GrantLease(1, t0+leaseDur)
		defer eng.Shutdown()

		for i := 0; i < 5; i++ {
			if _, _, err := eng.Propose(ctxWithTimeout(t, 5*time.Second), []byte{byte(i)}); err != nil {
				t.Fatalf("propose %d: %v", i, err)
			}
		}
		want := [][2]uint64{{1, 1}, {2, 1}, {3, 1}, {4, 1}, {5, 1}}
		if len(seen) != len(want) {
			t.Fatalf("sink fired %d times, want %d", len(seen), len(want))
		}
		for i := range want {
			if seen[i] != want[i] {
				t.Fatalf("sink[%d] = %v, want %v", i, seen[i], want[i])
			}
		}
	})

	t.Run("backup path", func(t *testing.T) {
		ap := &fakeApplier{}
		var last [2]uint64
		eng := New("n2", testShard, &fakeControl{epoch: 1, primary: "n1", isr: []string{"n1", "n2"}, minISR: 2},
			newInMemTransport(), ap,
			WithFrontierSink(func(seq, epoch uint64) { last = [2]uint64{seq, epoch} }))
		defer eng.Shutdown()

		for i := uint64(1); i <= 4; i++ {
			ack := eng.Receive(ReplicateMsg{Epoch: 2, Seq: i, PrevSeq: i - 1, PrevEpoch: map[bool]uint64{true: 0, false: 2}[i == 1], Data: []byte{byte(i)}})
			if !ack.OK {
				t.Fatalf("receive %d: not OK", i)
			}
		}
		if last != [2]uint64{4, 2} {
			t.Fatalf("sink last = %v, want (4,2)", last)
		}
		// A REJECTED frame must not move the persisted watermark: it was never applied.
		if ack := eng.Receive(ReplicateMsg{Epoch: 2, Seq: 9, PrevSeq: 8, PrevEpoch: 2}); ack.OK {
			t.Fatal("setup: the gap frame should have been rejected")
		}
		if last != [2]uint64{4, 2} {
			t.Fatalf("a rejected frame advanced the sink to %v — the watermark would name a write that never applied", last)
		}
	})
}
