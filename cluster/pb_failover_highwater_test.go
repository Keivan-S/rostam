// SPDX-License-Identifier: Apache-2.0

package cluster

import (
	"context"
	"testing"
	"time"

	"github.com/rostamlabs/rostam/shard/pbisr"
)

// ============================================================================
// M3 — the AppliedSeq / FrontierSeq SPLIT is a safety property of the failover
// gate, and until this file nothing tested it.
//
// Log matching split one number into two: AppliedSeq (what a node received AS A
// REPLICA — the committed tail) and FrontierSeq (what it HOLDS in either role,
// which additionally counts writes it proposed itself while primary at an older
// epoch). pbCandidateHighWater must rank promotion candidates by the FORMER.
//
// The distinction is the whole no-acked-loss argument. A write a node PROPOSED
// but never full-ISR-committed is uncommitted by definition: no client was told
// it exists, and no other member is required to hold it. Ranking by frontier
// rewards a candidate for exactly those writes — so a stale ex-primary that
// proposed a long uncommitted tail alone outranks a backup that actually
// received the committed one, and promoting it DISCARDS committed writes the
// backup was holding.
//
// The reviewer's mutation — return FrontierSeq instead of AppliedSeq, in either
// branch of pbCandidateHighWater — passed the entire cluster suite. These tests
// are what fails it. They are written against the DECISION (which node gets
// promoted), not against the return expression, so they stay meaningful if the
// function is refactored: the property is "the gate must not promote a node
// whose lead is made of uncommitted self-proposals".
// ============================================================================

// pbHighWaterCandidate builds a real engine in one of the two shapes the gate
// must tell apart, and reports the (AppliedSeq, FrontierSeq) it should present.
//
//	received  — writes this node took AS A BACKUP (they land in lastApplied).
//	proposed  — writes it then made ITSELF after a promotion (they advance the
//	            frontier only; lastApplied never counts a node's own proposals).
//
// A pure backup (proposed == 0) has AppliedSeq == FrontierSeq, so it is the
// control shape: nothing about it distinguishes the two fields.
func pbHighWaterCandidate(t *testing.T, nodeID string, received, proposed int) *pbisr.Engine {
	t.Helper()
	clock := newFakeClock(0)
	fsm := NewMetaFSM()
	// Epoch 1, some OTHER node primary: this engine is a backup while it receives.
	applyMeta(t, fsm, LogEntry{Op: OpSetShardEpoch, ShardID: 0, Epoch: 1, Primary: "origin", ISR: []string{"origin", nodeID}})
	eng := pbisr.New(nodeID, 0, newMetaControl(fsm, 1), nil,
		applierFunc(func(d []byte) ([]byte, error) { return d, nil }), pbisr.WithClock(clock.now))
	t.Cleanup(eng.Shutdown)

	// (1) Receive the committed tail as a replica. Each frame names its
	// predecessor's (seq, epoch) — the chain link; seq 1 extends genesis.
	for i := uint64(1); i <= uint64(received); i++ {
		var prevEpoch uint64
		if i > 1 {
			prevEpoch = 1
		}
		if ack := eng.Receive(pbisr.ReplicateMsg{Epoch: 1, Seq: i, PrevSeq: i - 1, PrevEpoch: prevEpoch, Data: []byte("w")}); !ack.OK {
			t.Fatalf("%s: backup Receive seq %d not OK", nodeID, i)
		}
	}
	if got := eng.LastApplied(); got != uint64(received) {
		t.Fatalf("%s: LastApplied = %d, want %d", nodeID, got, received)
	}
	if proposed == 0 {
		return eng
	}

	// (2) Promote it and let it propose ALONE. These writes never reach a second
	// member, so they are uncommitted — and they are precisely what inflates the
	// frontier above the applied high-water.
	applyMeta(t, fsm, LogEntry{Op: OpSetShardEpoch, ShardID: 0, Epoch: 2, Primary: nodeID, ISR: []string{nodeID}})
	eng.Promote(2, clock.now()+int64(time.Hour))
	for i := 0; i < proposed; i++ {
		ctx, cancel := context.WithTimeout(context.Background(), 2*time.Second)
		_, _, err := eng.Propose(ctx, []byte("self"))
		cancel()
		if err != nil {
			t.Fatalf("%s: self-propose %d: %v", nodeID, i, err)
		}
	}

	// The shape this whole test rests on: two DIFFERENT numbers, with the frontier
	// the larger one.
	fs, _ := eng.AppliedFrontier()
	if eng.LastApplied() != uint64(received) || fs != uint64(received+proposed) {
		t.Fatalf("%s: (applied,frontier) = (%d,%d), want (%d,%d) — the ex-primary shape did not materialize",
			nodeID, eng.LastApplied(), fs, received, received+proposed)
	}
	return eng
}

// pbHighWaterNode wires a Node whose pbCandidateHighWater can answer for the
// LOCAL candidate (from n.pbEngines) and for REMOTE ones (over a real
// NetTransport, exactly as the failover ticker does in production).
func pbHighWaterNode(t *testing.T, localID string, local *pbisr.Engine, remotes map[string]*pbisr.Engine) *Node {
	t.Helper()
	localTr, err := pbisr.NewNetTransport("127.0.0.1:0", nil, nil, nil)
	if err != nil {
		t.Fatalf("local pb transport: %v", err)
	}
	t.Cleanup(func() { _ = localTr.Close() }) //nolint:errcheck,gosec // test cleanup

	addrOf := make(map[string]string, len(remotes))
	for id, eng := range remotes {
		tr, err := pbisr.NewNetTransport("127.0.0.1:0", nil, nil, nil)
		if err != nil {
			t.Fatalf("pb transport for %s: %v", id, err)
		}
		t.Cleanup(func() { _ = tr.Close() }) //nolint:errcheck,gosec // test cleanup
		tr.Register(0, eng)                  // serves the catch-up handshake for this candidate
		addrOf[id] = tr.Addr()
	}

	n := &Node{cfg: Config{NodeID: localID}, pbTransport: localTr, pbAddrOf: addrOf}
	if local != nil {
		n.pbEngines = map[int]*pbisr.Engine{0: local}
	}
	return n
}

// TestPBCandidateHighWaterReportsAppliedNotFrontier pins the two branches of
// pbCandidateHighWater against engines whose two watermarks DIFFER, so the
// returned number identifies which field was read.
func TestPBCandidateHighWaterReportsAppliedNotFrontier(t *testing.T) {
	// The ex-primary shape: 5 committed writes received as a replica, then 7 more
	// proposed alone after a promotion. applied=5, frontier=12.
	exPrimary := pbHighWaterCandidate(t, "n-ex", 5, 7)
	// A plain backup that simply received more of the committed tail: 10/10.
	backup := pbHighWaterCandidate(t, "n-backup", 10, 0)

	t.Run("remote branch", func(t *testing.T) {
		n := pbHighWaterNode(t, "n-self", nil, map[string]*pbisr.Engine{"n-ex": exPrimary, "n-backup": backup})
		hw, ok := n.pbCandidateHighWater(0, "n-ex")
		if !ok {
			t.Fatal("a reachable candidate must verify")
		}
		if hw != 5 {
			t.Fatalf("remote high-water = %d, want 5 (AppliedSeq). 12 is FrontierSeq — the gate must not "+
				"count writes the candidate proposed itself and never committed", hw)
		}
		if hw, ok := n.pbCandidateHighWater(0, "n-backup"); !ok || hw != 10 {
			t.Fatalf("remote high-water for the plain backup = (%d,%v), want (10,true)", hw, ok)
		}
	})

	t.Run("local branch", func(t *testing.T) {
		// Same engine, now read through the self-candidate path (no round trip).
		n := pbHighWaterNode(t, "n-ex", exPrimary, nil)
		hw, ok := n.pbCandidateHighWater(0, "n-ex")
		if !ok {
			t.Fatal("the local candidate must verify")
		}
		if hw != 5 {
			t.Fatalf("local high-water = %d, want 5 (LastApplied). 12 is the applied FRONTIER — the local "+
				"branch must read the same field the remote one does, or the gate ranks self and peers on different scales", hw)
		}
	})
}

// TestFailoverGateNeverPromotesOnUncommittedFrontier is the M3 payoff: the same
// split, asserted where it MATTERS — the promotion decision.
//
// The dead primary's ISR holds two survivors:
//
//	n-ex     — an ex-primary from an older epoch. It received 5 committed writes
//	           as a replica, then proposed 7 more ALONE that nobody else holds.
//	           applied=5, frontier=12.
//	n-backup — a plain backup that received 10 of the committed tail.
//	           applied=10, frontier=10.
//
// Ranked by APPLIED high-water, n-backup wins (10 > 5) and the 5 committed
// writes n-ex is missing survive. Ranked by FRONTIER, n-ex wins (12 > 10) — and
// promoting it silently discards committed writes 6..10, which no member is then
// required to hold. That is acked-loss produced by the gate itself.
//
// The candidates are deliberately named so the FRONTIER ranking and the
// LOWEST-NODEID tie-break disagree with the applied ranking in opposite
// directions: "n-backup" sorts AFTER "n-ex", so a gate that lost the ranking
// entirely (all candidates equal) would also pick n-ex. Only reading AppliedSeq
// produces n-backup.
func TestFailoverGateNeverPromotesOnUncommittedFrontier(t *testing.T) {
	exPrimary := pbHighWaterCandidate(t, "n-ex", 5, 7)
	backup := pbHighWaterCandidate(t, "n-backup", 10, 0)
	n := pbHighWaterNode(t, "n-self", nil, map[string]*pbisr.Engine{"n-ex": exPrimary, "n-backup": backup})

	const timeout = int64(2_000_000_000)
	const now = int64(100_000_000_000)
	shards := []pbShardLiveness{{
		shardID:     0,
		epoch:       3,
		primary:     "n-dead",
		isr:         []string{"n-dead", "n-ex", "n-backup"},
		lastRenewNs: now - 20_000_000_000, // silent past the timeout ⇒ presumed dead
	}}

	promos := decidePBPromotions(shards, now, timeout, n.pbCandidateHighWater)
	if len(promos) != 1 {
		t.Fatalf("decidePBPromotions = %+v, want exactly one promotion", promos)
	}
	if promos[0].newPrimary != "n-backup" {
		t.Fatalf("promoted %q, want \"n-backup\".\n"+
			"n-ex leads only on FRONTIER (12 vs 10), and every one of those 7 writes is an uncommitted "+
			"self-proposal no other member holds. On APPLIED position n-backup leads 10 to 5, and promoting "+
			"n-ex instead discards committed writes 6..10 outright.", promos[0].newPrimary)
	}
	if promos[0].newEpoch != 4 {
		t.Fatalf("promoted at epoch %d, want 4 (the dead primary's epoch + 1)", promos[0].newEpoch)
	}
}
