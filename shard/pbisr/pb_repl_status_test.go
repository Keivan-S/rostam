// SPDX-License-Identifier: Apache-2.0

package pbisr

import (
	"testing"
	"time"
)

// TestReplicationStatusPrimaryReportsPerBackupLag verifies the primary-side
// replication snapshot the /v1/replication metric reads: after committing
// writes, the primary reports its assigned + durable frontiers and each backup's
// acked high-water with zero lag once the backup has caught up.
func TestReplicationStatusPrimaryReportsPerBackupLag(t *testing.T) {
	c := newCluster([]string{"n1", "n2"}, "n1", 1, []string{"n1", "n2"}, 2)
	primary := c.engines["n1"]

	for i := 0; i < 3; i++ {
		if _, _, err := primary.Propose(ctxWithTimeout(t, time.Second), []byte("w")); err != nil {
			t.Fatalf("propose %d: %v", i, err)
		}
	}

	st := primary.ReplicationStatus()
	if st.LastSeq != 3 {
		t.Fatalf("LastSeq = %d, want 3", st.LastSeq)
	}
	if st.Committed != 3 {
		t.Fatalf("Committed = %d, want 3", st.Committed)
	}
	if len(st.Peers) != 1 {
		t.Fatalf("Peers = %d, want 1 (the single backup n2)", len(st.Peers))
	}
	p := st.Peers[0]
	if p.Peer != "n2" {
		t.Fatalf("peer = %q, want n2", p.Peer)
	}
	// A full-ISR commit means n2 H6-acked seq 3; it is fully caught up (lag 0).
	if p.Acked != 3 {
		t.Fatalf("n2 acked = %d, want 3", p.Acked)
	}
	if p.Lag != 0 {
		t.Fatalf("n2 lag = %d, want 0 (caught up)", p.Lag)
	}
}

// TestReplicationStatusBackupReportsEmpty verifies a node that has proposed
// nothing (a pure backup) reports an empty primary-side view: it holds no
// per-backup high-water because it never shipped or heard acks.
func TestReplicationStatusBackupReportsEmpty(t *testing.T) {
	c := newCluster([]string{"n1", "n2"}, "n1", 1, []string{"n1", "n2"}, 2)
	// Drive real writes through the primary so n2 applies as a backup.
	if _, _, err := c.engines["n1"].Propose(ctxWithTimeout(t, time.Second), []byte("w")); err != nil {
		t.Fatalf("propose: %v", err)
	}
	eventually(t, func() bool { return c.engines["n2"].LastApplied() == 1 }, "n2 applies seq 1")

	st := c.engines["n2"].ReplicationStatus()
	if st.LastSeq != 0 {
		t.Fatalf("backup LastSeq = %d, want 0 (never proposed)", st.LastSeq)
	}
	if len(st.Peers) != 0 {
		t.Fatalf("backup Peers = %d, want 0 (holds no per-backup high-water)", len(st.Peers))
	}
}

// TestReplicationStatusLagAndMonotonicCredit white-box tests the lag arithmetic
// the full-ISR-commit tests can never reach (a committed write always leaves the
// backup caught up, lag 0): a backup behind the assigned frontier reports a real
// positive lag, a duplicate/out-of-order lower ack never rewinds the high-water,
// and an ack ahead of LastSeq (only reachable across an epoch-change race) clamps
// the lag at 0 rather than underflowing the unsigned subtraction.
func TestReplicationStatusLagAndMonotonicCredit(t *testing.T) {
	e := &Engine{} // bare engine: only mu/lastSeq/peerAcked drive this arithmetic
	e.mu.Lock()
	e.lastSeq = 10
	e.creditPeerAckedLocked("n2", 4)  // behind the frontier
	e.creditPeerAckedLocked("n3", 10) // caught up
	e.mu.Unlock()

	peers := func() map[string]PeerLag {
		m := map[string]PeerLag{}
		for _, p := range e.ReplicationStatus().Peers {
			m[p.Peer] = p
		}
		return m
	}

	m := peers()
	if m["n2"].Acked != 4 || m["n2"].Lag != 6 {
		t.Fatalf("n2 = {acked %d, lag %d}, want {4, 6}", m["n2"].Acked, m["n2"].Lag)
	}
	if m["n3"].Acked != 10 || m["n3"].Lag != 0 {
		t.Fatalf("n3 = {acked %d, lag %d}, want {10, 0} (caught up)", m["n3"].Acked, m["n3"].Lag)
	}

	// A lower (duplicate/out-of-order) ack must NOT rewind the high-water.
	e.mu.Lock()
	e.creditPeerAckedLocked("n2", 3)
	e.mu.Unlock()
	if m = peers(); m["n2"].Acked != 4 || m["n2"].Lag != 6 {
		t.Fatalf("after stale ack n2 = {acked %d, lag %d}, want {4, 6} (no rewind)", m["n2"].Acked, m["n2"].Lag)
	}

	// An ack ahead of LastSeq (epoch-change race) clamps lag at 0, no underflow.
	e.mu.Lock()
	e.creditPeerAckedLocked("n2", 20)
	e.mu.Unlock()
	if m = peers(); m["n2"].Acked != 20 || m["n2"].Lag != 0 {
		t.Fatalf("ahead-of-frontier n2 = {acked %d, lag %d}, want {20, 0} (clamped)", m["n2"].Acked, m["n2"].Lag)
	}
}
