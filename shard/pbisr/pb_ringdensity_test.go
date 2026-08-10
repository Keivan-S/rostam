// SPDX-License-Identifier: Apache-2.0

package pbisr

import (
	"testing"
	"time"
)

// RING DENSITY AFTER RE-PROMOTION.
//
// The catch-up ring is indexed by offset from its oldest retained seq, which is
// correct only while the retained seqs are DENSE. Promotion used to break that:
// it reset lastSeq to lastApplied but left the ring holding this node's older
// primary-era entries, while writes taken as a BACKUP never append to the ring
// at all (proposeSequenced is the only append site). Proposals then resumed
// ABOVE the received range, leaving a hole — ring [1..5, 9..11] with 6-8 absent
// — while span() still advertised [1..11] as though it were contiguous.
//
// The consequence was not a miss but a FABRICATION: at() indexed past the live
// prefix and returned a zero-valued entry with ok=true. A zero entry carries
// prevEpoch == 0, which checkCatchupDivergenceLocked reads as a chain link that
// fails to match — so a replica holding a perfectly clean prefix was rejected
// as diverged. Snapshot transfer means that no longer strands the replica, but it does
// spend a whole snapshot transfer repairing a node that needed a few writes,
// and reports it to operators under a repair reason.
//
// Two independent guards, asserted separately:
//   - at() re-checks the entry's seq, so a hole can never be served (safety net)
//   - Promote drops the ring, so the hole never forms (the fix)

// ringAppendSeqs appends entries at the given seqs, each stamped with a
// non-zero prevEpoch so a fabricated zero entry is distinguishable from a real
// one on inspection.
func ringAppendSeqs(r *ring, epoch uint64, seqs ...uint64) {
	for _, s := range seqs {
		r.append(ringEntry{epoch: epoch, seq: s, prevEpoch: epoch - 1, data: []byte{byte(s)}})
	}
}

// TestRingAtRefusesAHole builds the post-re-promotion shape directly and
// asserts every seq in the hole reports missing rather than fabricated.
func TestRingAtRefusesAHole(t *testing.T) {
	r := newRing(16)
	ringAppendSeqs(r, 2, 1, 2, 3, 4, 5) // primary-era entries
	ringAppendSeqs(r, 3, 9, 10, 11)     // post-promotion proposals; 6-8 never appended

	oldest, newest, ok := r.span()
	if !ok || oldest != 1 || newest != 11 {
		t.Fatalf("span = (%d, %d, %v), want (1, 11, true) — this test needs that shape", oldest, newest, ok)
	}

	// Seqs BEFORE the hole are still locatable: their offset from oldest equals
	// their slot, because nothing has been skipped yet.
	for _, seq := range []uint64{1, 2, 3, 4, 5} {
		ent, has := r.at(seq)
		if !has {
			t.Errorf("at(%d): missing, want present — seqs before the hole must still resolve", seq)
			continue
		}
		if ent.seq != seq {
			t.Errorf("at(%d): returned the entry for seq %d — offset indexing is wrong", seq, ent.seq)
		}
	}

	// Everything from the hole ONWARD must report missing — including 9-11,
	// which are physically retained.
	//
	// This is not a shortcoming of the seq check, it is the arithmetic: at()
	// locates a slot as (head + seq - oldest), while append stores entries
	// contiguously in arrival order. Once a seq is skipped, those two indices
	// diverge permanently, so no entry at or past the hole can be found by
	// offset. Before the fix that mismatch was invisible — the lookup landed on
	// whatever occupied the computed slot (often a zero entry, whose
	// prevEpoch == 0 reads as a failed chain link) and returned it with
	// ok=true, so a clean replica was reported diverged.
	//
	// Reporting missing is the correct degradation: a holed ring genuinely
	// cannot serve those deltas, so the caller falls back to a snapshot instead
	// of acting on a fabricated one. The ring going cold from the hole onward
	// is precisely why Promote clears it rather than relying on this guard.
	for _, seq := range []uint64{6, 7, 8, 9, 10, 11} {
		if ent, has := r.at(seq); has {
			t.Errorf("at(%d): returned {seq:%d epoch:%d prevEpoch:%d} with ok=true; every seq at "+
				"or past a hole must report missing rather than a mislocated entry",
				seq, ent.seq, ent.epoch, ent.prevEpoch)
		}
	}
}

// TestPromoteDropsTheRing asserts the hole cannot form in the first place: a
// promoted primary retains nothing from the epoch it no longer serves, so a
// catch-up asking for those seqs is correctly told ring-cold.
func TestPromoteDropsTheRing(t *testing.T) {
	c := newCluster([]string{"n1", "n2"}, "n1", 1, []string{"n1", "n2"}, 2)
	primary := c.engines["n1"]
	backup := c.engines["n2"]

	// Commit writes so the PRIMARY's ring holds real primary-era entries.
	for i := 0; i < 3; i++ {
		if _, _, err := primary.Propose(ctxWithTimeout(t, time.Second), []byte("w")); err != nil {
			t.Fatalf("propose %d: %v", i, err)
		}
	}
	if got := primary.backlog.len(); got != 3 {
		t.Fatalf("primary ring len = %d, want 3 — setup did not retain the proposals", got)
	}
	eventually(t, func() bool { return backup.LastApplied() == 3 }, "backup applies all 3")

	// Re-promote the node that already holds primary-era entries. This is the
	// grow -> re-failover shape: its ring is non-empty at promotion time.
	primary.Promote(2, t0+leaseDur)

	if got := primary.backlog.len(); got != 0 {
		t.Errorf("after Promote: ring len = %d, want 0 — a promoted primary must not retain "+
			"a superseded epoch's entries", got)
	}
	if _, _, ok := primary.backlog.span(); ok {
		t.Error("after Promote: span() still reports entries; the ring should be empty")
	}
	if _, has := primary.backlog.at(2); has {
		t.Error("after Promote: at(2) still resolves a pre-promotion entry")
	}
}

// TestPromoteReleasesRingBuffers pins that clearing the ring hands each entry's
// data back through the WithDataRelease hook, so a promotion does not leak the
// buffers eviction would otherwise have recycled.
func TestPromoteReleasesRingBuffers(t *testing.T) {
	var released int
	e := New("n1", testShard, nil, nil, &fakeApplier{},
		WithDataRelease(func([]byte) { released++ }))

	ringAppendSeqs(e.backlog, 2, 1, 2, 3)
	e.Promote(7, t0+leaseDur)

	if released != 3 {
		t.Errorf("released %d buffers, want 3 — Promote must return retained data to the "+
			"release hook rather than dropping it", released)
	}
	if got := e.backlog.len(); got != 0 {
		t.Errorf("ring len = %d after Promote, want 0", got)
	}
}

// TestCatchupInfoSeparatesAppliedFromFrontier pins the split at its
// source: CatchupInfo reports the applied-as-replica watermark and the frontier
// as DISTINCT fields, and they genuinely differ on an ex-primary.
//
// The failover gate (cluster's pbCandidateHighWater) ranks candidates on
// AppliedSeq. The frontier additionally counts writes a node proposed itself at
// an older epoch — possibly acked by nobody — so ranking on it can hand the
// epoch to a node whose lead is uncommitted. Review found the gate correct but
// untested, and the reason nothing caught a mutation was that every engine in
// the suite had the two numbers equal. This is the state where they don't.
func TestCatchupInfoSeparatesAppliedFromFrontier(t *testing.T) {
	c := newCluster([]string{"n1", "n2"}, "n1", 1, []string{"n1", "n2"}, 2)
	primary := c.engines["n1"]
	defer primary.Shutdown()

	const n = 4
	for i := 0; i < n; i++ {
		if _, _, err := primary.Propose(ctxWithTimeout(t, time.Second), []byte("w")); err != nil {
			t.Fatalf("propose %d: %v", i, err)
		}
	}

	// The primary proposed every write and received none as a replica, so its
	// frontier has advanced while its applied-as-replica watermark has not.
	info := primary.CatchupInfo()
	if !info.OK {
		t.Fatal("CatchupInfo not OK on a healthy primary")
	}
	if info.FrontierSeq != n {
		t.Errorf("FrontierSeq = %d, want %d (the primary assigned all of them)", info.FrontierSeq, n)
	}
	if info.AppliedSeq == info.FrontierSeq {
		t.Fatalf("AppliedSeq == FrontierSeq == %d on a primary that received nothing as a "+
			"replica; the two watermarks must be reported separately, or a gate ranking on "+
			"one silently gets the other", info.AppliedSeq)
	}
	if info.AppliedSeq != primary.LastApplied() {
		t.Errorf("AppliedSeq = %d, want LastApplied = %d", info.AppliedSeq, primary.LastApplied())
	}

	// The backup is the mirror image: it applied everything as a replica and
	// proposed nothing, so both watermarks agree.
	eventually(t, func() bool { return c.engines["n2"].LastApplied() == n }, "backup applies all")
	binfo := c.engines["n2"].CatchupInfo()
	if binfo.AppliedSeq != n {
		t.Errorf("backup AppliedSeq = %d, want %d", binfo.AppliedSeq, n)
	}
}
