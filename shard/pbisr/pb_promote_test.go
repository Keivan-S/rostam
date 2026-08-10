// SPDX-License-Identifier: Apache-2.0

package pbisr

import (
	"testing"
	"time"
)

// TestPromoteBackupContinuesFromHighWater is the engine primitive: a
// backup that applied writes up to a high-water becomes a writable primary that
// continues seq assignment from there, with every acked write preserved.
func TestPromoteBackupContinuesFromHighWater(t *testing.T) {
	// n1 primary, n2 backup; ISR {n1,n2}, minISR 2 → full-ISR commit.
	c := newCluster([]string{"n1", "n2"}, "n1", 1, []string{"n1", "n2"}, 2)
	primary := c.engines["n1"]
	backup := c.engines["n2"]

	// Commit 3 writes to the full ISR — n2 (the promotion target) applies all 3.
	for i := 0; i < 3; i++ {
		if _, _, err := primary.Propose(ctxWithTimeout(t, time.Second), []byte("w")); err != nil {
			t.Fatalf("propose %d: %v", i, err)
		}
	}
	eventually(t, func() bool { return backup.LastApplied() == 3 }, "backup applies all 3")
	appliedBefore := c.appliers["n2"].count()

	// Old primary dies; MetaRaft promotes n2 to epoch 2 with a fresh lease.
	backup.Promote(2, t0+leaseDur)

	if backup.Epoch() != 2 {
		t.Fatalf("epoch = %d, want 2", backup.Epoch())
	}
	if backup.LastSeq() != 3 {
		t.Fatalf("LastSeq = %d, want 3 (continue from applied high-water)", backup.LastSeq())
	}
	if backup.Committed() != 3 {
		t.Fatalf("Committed = %d, want 3 (applied high-water is now canonical)", backup.Committed())
	}
	if !backup.LeaseValid() {
		t.Fatal("promoted node holds no valid lease")
	}
	if got := c.appliers["n2"].count(); got != appliedBefore {
		t.Fatalf("promotion changed applied state (%d → %d) — acked writes must be untouched", appliedBefore, got)
	}

	// The promoted node is now a writable primary: update the control plane to
	// name it, and a new write gets seq 4 and commits (single-member ISR {n2}).
	c.ctrl.mu.Lock()
	c.ctrl.primary = "n2"
	c.ctrl.epoch = 2
	c.ctrl.isr = []string{"n2"}
	c.ctrl.minISR = 1
	c.ctrl.mu.Unlock()

	_, seq, err := backup.Propose(ctxWithTimeout(t, time.Second), []byte("post-promote"))
	if err != nil {
		t.Fatalf("post-promote propose: %v", err)
	}
	if seq != 4 {
		t.Fatalf("post-promote seq = %d, want 4", seq)
	}
	if backup.Committed() != 4 {
		t.Fatalf("Committed = %d, want 4", backup.Committed())
	}
}

// TestPromoteRejectsStaleEpoch: a promotion for an epoch not strictly higher
// than the engine's watermark is a no-op (never regress a live primary).
func TestPromoteRejectsStaleEpoch(t *testing.T) {
	c := newCluster([]string{"n1", "n2"}, "n1", 5, []string{"n1", "n2"}, 2)
	primary := c.engines["n1"]
	// Assign some seqs as the epoch-5 primary.
	for i := 0; i < 2; i++ {
		_, _, _ = primary.Propose(ctxWithTimeout(t, time.Second), []byte("w"))
	}
	lastSeqBefore := primary.LastSeq()

	primary.Promote(5, t0+leaseDur) // equal epoch — must be a no-op
	if primary.LastSeq() != lastSeqBefore {
		t.Fatalf("equal-epoch Promote regressed LastSeq %d → %d", lastSeqBefore, primary.LastSeq())
	}
	primary.Promote(3, t0+leaseDur) // stale epoch — must be a no-op
	if primary.LastSeq() != lastSeqBefore {
		t.Fatalf("stale-epoch Promote regressed LastSeq %d → %d", lastSeqBefore, primary.LastSeq())
	}
}
