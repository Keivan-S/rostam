// SPDX-License-Identifier: Apache-2.0

package cluster

import (
	"errors"
	"fmt"
	"testing"

	"github.com/rostamlabs/rostam/shard/pbisr"
)

// ============================================================================
// ISR grow/shrink observability: pure-unit coverage of the two
// building blocks the drivers' tick() wires into their abort paths
// (cluster/pb_grow.go, cluster/pb_shrink.go): the reason classifier
// (growAbortReason) and the rate-limiting/counter tracker (pbAbortTracker).
// The end-to-end payoff — a real stuck grow actually logging and counting,
// rate-limited — is TestGrowAbortLoggedAndRateLimited in pb_grow_test.go.
// ============================================================================

// TestGrowAbortReason pins the sentinel→reason mapping used for both the log
// line and the /v1/replication grow_aborts counter key. Exercised via
// errors.Is (not ==) so a future wrap of these sentinels keeps classifying
// correctly.
//
// The snapshot-transfer entries are here because the classifier was written before
// snapshot transfer existed and therefore could not name any of its failure
// modes: they would every one have landed in "other", which is exactly as
// useless to an operator as the silent discard 4.0 set out to fix.
func TestGrowAbortReason(t *testing.T) {
	cases := []struct {
		err  error
		want string
	}{
		{pbisr.ErrGrowRingEvicted, "ring_evicted"},
		{pbisr.ErrCatchupDiverged, "diverged"},
		{pbisr.ErrGrowEpochChanged, "epoch_changed"},
		{pbisr.ErrGrowPeerAhead, "peer_ahead"},
		{pbisr.ErrGrowNoCatchupTransport, "no_catchup_transport"},
		{pbisr.ErrGrowNoGroupTransport, "no_group_transport"},
		// Snapshot transfer's own failure modes.
		{pbisr.ErrCatchupUnverifiable, "unverifiable"},
		{pbisr.ErrGrowSnapshotRejected, "snapshot_rejected"},
		{pbisr.ErrGrowSnapshotExhausted, "snapshot_exhausted"},
		{pbisr.ErrGrowNoSnapshotStore, "no_snapshot_store"},
		{pbisr.ErrGrowNoSnapshotTransport, "no_snapshot_transport"},
		{fmt.Errorf("wrapped: %w", pbisr.ErrGrowRingEvicted), "ring_evicted"},
		{fmt.Errorf("wrapped: %w", pbisr.ErrCatchupUnverifiable), "unverifiable"},
		{errors.New("connection refused"), "other"},
	}
	for _, tc := range cases {
		if got := growAbortReason(tc.err); got != tc.want {
			t.Errorf("growAbortReason(%v) = %q, want %q", tc.err, got, tc.want)
		}
	}
}

// TestGrowAbortReasonsAreDistinct pins the property the table above can only
// pin one row at a time: every grow sentinel gets its OWN bucket. A counter
// that merges two causes with different operator responses — "retry is
// pointless until you wire a snapshot store" vs "this is repairing itself" —
// is worse than no counter, because it reads as authoritative.
func TestGrowAbortReasonsAreDistinct(t *testing.T) {
	sentinels := []error{
		pbisr.ErrGrowRingEvicted,
		pbisr.ErrCatchupDiverged,
		pbisr.ErrCatchupUnverifiable,
		pbisr.ErrGrowSnapshotRejected,
		pbisr.ErrGrowSnapshotExhausted,
		pbisr.ErrGrowNoSnapshotStore,
		pbisr.ErrGrowNoSnapshotTransport,
		pbisr.ErrGrowEpochChanged,
		pbisr.ErrGrowPeerAhead,
		pbisr.ErrGrowNoCatchupTransport,
		pbisr.ErrGrowNoGroupTransport,
	}
	seen := make(map[string]error, len(sentinels))
	for _, err := range sentinels {
		reason := growAbortReason(err)
		if reason == "other" {
			t.Errorf("sentinel %v fell through to the catch-all bucket", err)
			continue
		}
		if prev, dup := seen[reason]; dup {
			t.Errorf("bucket %q is shared by %v and %v", reason, prev, err)
			continue
		}
		seen[reason] = err
	}
}

// TestPBAbortTrackerRateLimiting pins pbAbortTracker's contract directly: it
// is the sole gate between every tick's abort and an actual log line
// (cluster/pb_grow.go's `if d.abortStats.record(...) { slog.Warn(...) }`), so
// this test is the precise spec for "log the transition, not every tick".
func TestPBAbortTrackerRateLimiting(t *testing.T) {
	tr := newPBAbortTracker()
	const shard = 0

	// N consecutive ticks with the SAME reason: only the FIRST should signal a
	// log (the transition into failure); the rest are rate-limited.
	if !tr.record(shard, "n2", "ring_evicted") {
		t.Fatal("first abort at a new reason must signal a log")
	}
	for i := 0; i < 9; i++ {
		if tr.record(shard, "n2", "ring_evicted") {
			t.Fatalf("tick %d: repeat of the SAME reason re-signaled a log (rate limiting broken)", i)
		}
	}
	if counts := tr.snapshot(shard); counts["ring_evicted"] != 10 {
		t.Fatalf("cumulative count = %d, want 10 (every tick counts, even when not logged)", counts["ring_evicted"])
	}

	// A DIFFERENT reason is a transition too, even with no intervening success.
	if !tr.record(shard, "n2", "submit_failed") {
		t.Fatal("a changed reason must signal a log")
	}
	if tr.record(shard, "n2", "submit_failed") {
		t.Fatal("repeat of the new reason must NOT re-signal")
	}

	// clear() (a tick with no abort — success or nothing-to-do) resets the
	// transition state, so a LATER recurrence of the same reason logs again.
	tr.clear(shard)
	if !tr.record(shard, "n2", "submit_failed") {
		t.Fatal("recurrence after a clear() must signal a log again")
	}
	// Counts are cumulative and unaffected by clear().
	counts := tr.snapshot(shard)
	if counts["submit_failed"] != 3 {
		t.Fatalf("submit_failed count = %d, want 3 (2 before clear + 1 after)", counts["submit_failed"])
	}
	if counts["ring_evicted"] != 10 {
		t.Fatalf("clear() for one reason must not disturb another reason's count: ring_evicted = %d, want 10", counts["ring_evicted"])
	}

	// Shards are independent.
	const other = 1
	if snap := tr.snapshot(other); snap != nil {
		t.Fatalf("untouched shard has a non-nil snapshot: %v", snap)
	}
	tr.record(other, "n2", "ring_evicted")
	if tr.snapshot(shard)["ring_evicted"] != 10 {
		t.Fatal("recording on a different shard must not affect shard 0's counts")
	}

	// A nil tracker (PBAutoFailover off ⇒ no driver ⇒ no tracker) must be safe
	// to call through the same methods the drivers use — AbortCounts on a nil
	// *pbGrowDriver relies on this.
	var nilTr *pbAbortTracker
	if nilTr.record(0, "n2", "x") {
		t.Fatal("nil tracker record() must return false, not panic/signal")
	}
	nilTr.clear(0) // must not panic
	if snap := nilTr.snapshot(0); snap != nil {
		t.Fatalf("nil tracker snapshot() = %v, want nil", snap)
	}
}

// TestPBAbortTrackerTwoTargetsDoNotFlap pins the M4 regression: the tracker
// suppresses per (shard, target), not per shard.
//
// One grow tick calls record() once per candidate outside the ISR, and a shard
// commonly has several. When the suppression state was keyed on the shard
// alone, two targets failing for DIFFERENT reasons overwrote each other on
// every call, so each one always saw a "changed" reason and every abort logged
// — 100 lines for 100 aborts, roughly 2 WARN/s/shard at the default tick, and
// worse as the reason vocabulary grows.
//
// The suppression is supposed to mean "this target keeps failing this way, say
// it once", so alternating targets must not defeat it.
func TestPBAbortTrackerTwoTargetsDoNotFlap(t *testing.T) {
	tr := newPBAbortTracker()
	const shard = 3
	const ticks = 50

	logged := 0
	for i := 0; i < ticks; i++ {
		// Exactly the driver's shape: one record per target, per tick, each
		// failing persistently for its own reason.
		if tr.record(shard, "n2", "diverged") {
			logged++
		}
		if tr.record(shard, "n3", "ring_evicted") {
			logged++
		}
	}

	// One line per target for the first occurrence, then silence.
	if logged != 2 {
		t.Errorf("logged %d lines over %d ticks x 2 targets, want 2 (one per target); "+
			"keying suppression on the shard alone makes the two targets overwrite each "+
			"other and every abort logs", logged, ticks)
	}

	// Counts are shard-scoped and must record every abort regardless.
	counts := tr.snapshot(shard)
	if counts["diverged"] != ticks {
		t.Errorf(`counts["diverged"] = %d, want %d — suppression must never drop a count`, counts["diverged"], ticks)
	}
	if counts["ring_evicted"] != ticks {
		t.Errorf(`counts["ring_evicted"] = %d, want %d`, counts["ring_evicted"], ticks)
	}
}

// TestPBAbortTrackerClearResetsEveryTarget pins that a shard-level clear (the
// driver observing a healthy tick) re-arms logging for ALL of that shard's
// targets, not just whichever one happened to be recorded last.
func TestPBAbortTrackerClearResetsEveryTarget(t *testing.T) {
	tr := newPBAbortTracker()
	const shard = 1

	tr.record(shard, "n2", "diverged")
	tr.record(shard, "n3", "diverged")
	// Both are now suppressed.
	if tr.record(shard, "n2", "diverged") || tr.record(shard, "n3", "diverged") {
		t.Fatal("setup: expected both targets suppressed after their first record")
	}

	tr.clear(shard) // a healthy tick for this shard

	if !tr.record(shard, "n2", "diverged") {
		t.Error("after clear: n2 did not re-log; clear must re-arm every target of the shard")
	}
	if !tr.record(shard, "n3", "diverged") {
		t.Error("after clear: n3 did not re-log; clear must re-arm every target of the shard")
	}
}
