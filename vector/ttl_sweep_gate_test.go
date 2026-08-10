// SPDX-License-Identifier: Apache-2.0

package vector

import (
	"testing"
	"time"
)

// TestSweepOnceFastPathNoLock proves the Fix-4 gate: a collection carrying no
// point deadline and no per-key deadline never takes h.mu at all — sweepOnce
// returns immediately via the arena.DeadlineSlots()==0 check BEFORE the
// h.mu.Lock() call. We hold h.mu ourselves (as a stand-in write-locker) and run
// sweepOnce concurrently: the slow path would block forever trying to acquire
// h.mu, so a prompt return is direct proof the fast path never touched the lock.
func TestSweepOnceFastPathNoLock(t *testing.T) {
	h, err := newHNSW(Config{Dim: 4, Metric: L2, M: 16, EfConstruction: 200, EfSearch: 64, Seed: 1})
	if err != nil {
		t.Fatalf("newHNSW: %v", err)
	}
	if got := h.arena.DeadlineSlots(); got != 0 {
		t.Fatalf("DeadlineSlots on a fresh index = %d, want 0", got)
	}
	// A few TTL-less inserts must never open the gate.
	for i, v := range [][]float32{{1, 0, 0, 0}, {0, 1, 0, 0}, {0, 0, 1, 0}} {
		if _, _, err := h.Insert(uint64(i+1), v, 0, nil, nil, nil, CASCond{}); err != nil {
			t.Fatalf("Insert %d: %v", i+1, err)
		}
	}
	if got := h.arena.DeadlineSlots(); got != 0 {
		t.Fatalf("DeadlineSlots after TTL-less inserts = %d, want 0", got)
	}

	h.mu.Lock()
	defer h.mu.Unlock()

	done := make(chan int, 1)
	go func() { done <- h.sweepOnce() }()
	select {
	case n := <-done:
		if n != 0 {
			t.Fatalf("sweepOnce fast path returned %d, want 0", n)
		}
	case <-time.After(2 * time.Second):
		t.Fatal("sweepOnce blocked while h.mu was held elsewhere — it took the slow path instead of the DeadlineSlots==0 fast path")
	}
}

// TestSweepOnceGateTracksPointTTL covers (b) and (c) of the Fix-4 requirement: a
// single point-TTL insert opens the gate (DeadlineSlots goes 0 -> 1), the sweeper
// keeps running (and eventually expires it) while the gate is open, and once the
// point is swept the gate closes again (DeadlineSlots back to 0).
func TestSweepOnceGateTracksPointTTL(t *testing.T) {
	h, err := newHNSW(Config{Dim: 4, Metric: L2, M: 16, EfConstruction: 200, EfSearch: 64, Seed: 1})
	if err != nil {
		t.Fatalf("newHNSW: %v", err)
	}
	var fakeNow int64 = 1_000_000
	h.now = func() int64 { return fakeNow }

	if got := h.arena.DeadlineSlots(); got != 0 {
		t.Fatalf("DeadlineSlots before insert = %d, want 0", got)
	}

	if _, _, err := h.Insert(1, []float32{1, 0, 0, 0}, 50*time.Millisecond, nil, nil, nil, CASCond{}); err != nil {
		t.Fatalf("Insert: %v", err)
	}
	if got := h.arena.DeadlineSlots(); got != 1 {
		t.Fatalf("DeadlineSlots after a TTL insert = %d, want 1 (gate must open)", got)
	}

	// (b) Before expiry the gate is open, so sweepOnce runs the full scan and
	// correctly finds nothing to sweep yet.
	if n := h.sweepOnce(); n != 0 {
		t.Fatalf("sweepOnce pre-expiry = %d, want 0", n)
	}
	if got := h.arena.DeadlineSlots(); got != 1 {
		t.Fatalf("DeadlineSlots pre-expiry = %d, want 1 (still pending)", got)
	}

	fakeNow += 100

	// (b) After expiry sweepOnce tombstones the point.
	if n := h.sweepOnce(); n != 1 {
		t.Fatalf("sweepOnce post-expiry = %d, want 1", n)
	}
	// (c) The gate closes once the only TTL point is swept.
	if got := h.arena.DeadlineSlots(); got != 0 {
		t.Fatalf("DeadlineSlots after expiry = %d, want 0 (gate must close)", got)
	}

	// A subsequent sweep with the gate closed takes the fast path and stays a no-op.
	if n := h.sweepOnce(); n != 0 {
		t.Fatalf("sweepOnce with the gate closed = %d, want 0", n)
	}
}

// TestSweepOnceGateTracksKeyTTL proves the gate also opens/closes for a
// PER-KEY deadline (no point TTL at all): SetPayload's key_ttl opens it,
// Delete (not sweep — key-only entries have no point deadline to age past) or
// the key-TTL sweep pass closes it once the key is swept.
func TestSweepOnceGateTracksKeyTTL(t *testing.T) {
	h, err := newHNSW(keyTTLCfg())
	if err != nil {
		t.Fatalf("newHNSW: %v", err)
	}
	var fakeNow int64 = 1_000_000
	h.now = func() int64 { return fakeNow }

	if _, _, err := h.Insert(1, []float32{1, 0, 0, 0}, 0, nil, nil, nil, CASCond{}); err != nil {
		t.Fatalf("Insert: %v", err)
	}
	if got := h.arena.DeadlineSlots(); got != 0 {
		t.Fatalf("DeadlineSlots after a TTL-less insert = %d, want 0", got)
	}

	if _, _, _, err := h.SetPayload(1, Metadata{"temp": NewString("hot")}, map[string]int64{"temp": 1000}, CASCond{}); err != nil {
		t.Fatalf("SetPayload: %v", err)
	}
	if got := h.arena.DeadlineSlots(); got != 1 {
		t.Fatalf("DeadlineSlots after a per-key TTL = %d, want 1 (gate must open)", got)
	}

	fakeNow += 1500
	h.sweepOnce()

	if got := h.arena.DeadlineSlots(); got != 0 {
		t.Fatalf("DeadlineSlots after the key-TTL sweep = %d, want 0 (gate must close)", got)
	}
}

// TestSweepOnceGateClosesOnDelete proves Delete (not just sweepOnce) clears a
// live point's deadline out of the gate: deleting a point with a pending TTL
// must close the gate immediately rather than leaving a phantom count that only
// a future sweep would (wrongly) never resolve.
func TestSweepOnceGateClosesOnDelete(t *testing.T) {
	h, err := newHNSW(Config{Dim: 4, Metric: L2, M: 16, EfConstruction: 200, EfSearch: 64, Seed: 1})
	if err != nil {
		t.Fatalf("newHNSW: %v", err)
	}
	if _, _, err := h.Insert(1, []float32{1, 0, 0, 0}, time.Hour, nil, nil, nil, CASCond{}); err != nil {
		t.Fatalf("Insert: %v", err)
	}
	if got := h.arena.DeadlineSlots(); got != 1 {
		t.Fatalf("DeadlineSlots after a TTL insert = %d, want 1", got)
	}
	if ok, err := h.Delete(1, CASCond{}); err != nil || !ok {
		t.Fatalf("Delete: ok=%v err=%v", ok, err)
	}
	if got := h.arena.DeadlineSlots(); got != 0 {
		t.Fatalf("DeadlineSlots after deleting the only TTL point = %d, want 0", got)
	}
}

// assertDeadlineCountsBalanced compares the INCREMENTALLY maintained counters
// against an authoritative recount of the same arrays. This is the both-
// directions proof the gate needs: an overcount only pins the sweep gate open
// (a perf loss), but an UNDERCOUNT closes the gate on an index that still has
// pending deadlines — TTLs would silently never expire, a correctness bug.
// RecomputeDeadlineCounts is the authoritative recount, so we snapshot the
// incremental values, recount, and compare.
func assertDeadlineCountsBalanced(t *testing.T, a *arena, stage string) {
	t.Helper()
	gotPoints := a.deadlinePoints.Load()
	gotKeys := a.deadlineKeys.Load()
	a.RecomputeDeadlineCounts()
	wantPoints := a.deadlinePoints.Load()
	wantKeys := a.deadlineKeys.Load()
	if gotPoints != wantPoints || gotKeys != wantKeys {
		dir := "OVERCOUNT (gate pinned open — perf loss)"
		if gotPoints < wantPoints || gotKeys < wantKeys {
			dir = "UNDERCOUNT (gate closes early — TTLs never expire, CORRECTNESS BUG)"
		}
		t.Fatalf("%s: incremental counters (points=%d keys=%d) disagree with the recount (points=%d keys=%d) — %s",
			stage, gotPoints, gotKeys, wantPoints, wantKeys, dir)
	}
}

// TestDeadlineCountsBalancedAcrossSlotLifecycle walks slots through every
// teardown path that returns one to the free list — tombstone, Reclaim, and the
// insert-over-a-dead-slot resurrect — checking after each step that the
// incremental counters still equal an authoritative recount.
//
// What this pins: arena.Insert's free-list reuse zeroes expires/keyExpires
// DIRECTLY, so a slot freed while still carrying a deadline would leak a
// permanent +1 unless the decrement happens on the way out. arena.Delete now
// owns that clear-and-decrement for every caller, and because SetExpires/
// SetKeyExpires are idempotent in the clearing direction, the paths that
// already clear at tombstone time cannot double-decrement it back down.
func TestDeadlineCountsBalancedAcrossSlotLifecycle(t *testing.T) {
	h, err := newHNSW(Config{Dim: 4, Metric: L2, M: 16, EfConstruction: 200, EfSearch: 64, Seed: 1})
	if err != nil {
		t.Fatalf("newHNSW: %v", err)
	}
	var fakeNow int64 = 1_000_000
	h.now = func() int64 { return fakeNow }

	vec := func(i int) []float32 { return []float32{float32(i), 0, 0, 0} }
	meta := Metadata{"k": NewInt(7)}
	keyTTL := map[string]int64{"k": 3_600_000}

	// 6 points covering all four shapes: both TTL kinds (1,2), point-only (3,4),
	// key-only (5), TTL-free (6).
	for i := 1; i <= 2; i++ {
		if _, _, err := h.Insert(uint64(i), vec(i), time.Hour, meta, nil, keyTTL, CASCond{}); err != nil {
			t.Fatalf("Insert %d: %v", i, err)
		}
	}
	for i := 3; i <= 4; i++ {
		if _, _, err := h.Insert(uint64(i), vec(i), time.Hour, nil, nil, nil, CASCond{}); err != nil {
			t.Fatalf("Insert %d: %v", i, err)
		}
	}
	if _, _, err := h.Insert(5, vec(5), 0, meta, nil, keyTTL, CASCond{}); err != nil {
		t.Fatalf("Insert 5: %v", err)
	}
	if _, _, err := h.Insert(6, vec(6), 0, nil, nil, nil, CASCond{}); err != nil {
		t.Fatalf("Insert 6: %v", err)
	}
	if got := h.arena.deadlinePoints.Load(); got != 4 { // 1,2,3,4
		t.Fatalf("deadlinePoints after seeding = %d, want 4", got)
	}
	if got := h.arena.deadlineKeys.Load(); got != 3 { // 1,2,5
		t.Fatalf("deadlineKeys after seeding = %d, want 3", got)
	}
	assertDeadlineCountsBalanced(t, h.arena, "after seeding")

	// Tombstone one of each shape. The gate must drop them NOW, not at Reclaim.
	for _, id := range []uint64{1, 3} {
		if ok, derr := h.Delete(id, CASCond{}); derr != nil || !ok {
			t.Fatalf("Delete(%d): ok=%v err=%v", id, ok, derr)
		}
	}
	if got := h.arena.DeadlineSlots(); got != 4 { // points 2,4 + keys 2,5
		t.Fatalf("DeadlineSlots after tombstoning 1 and 3 = %d, want 4", got)
	}
	assertDeadlineCountsBalanced(t, h.arena, "after tombstone")

	// Reclaim frees those slots through arena.Delete. Its clear is a no-op here
	// (the tombstone path already cleared them) — a double-decrement would show
	// up immediately as an undercount.
	if n := h.Reclaim(); n != 2 {
		t.Fatalf("Reclaim reclaimed %d slots, want 2", n)
	}
	if got := h.arena.DeadlineSlots(); got != 4 {
		t.Fatalf("DeadlineSlots after Reclaim = %d, want 4 (Reclaim must not double-decrement)", got)
	}
	assertDeadlineCountsBalanced(t, h.arena, "after Reclaim")

	// Reuse a freed slot for a TTL-free insert: no stale deadline may ride along
	// and the counters must not move.
	if _, _, err := h.Insert(10, vec(10), 0, nil, nil, nil, CASCond{}); err != nil {
		t.Fatalf("Insert 10 (slot reuse): %v", err)
	}
	if got := h.arena.DeadlineSlots(); got != 4 {
		t.Fatalf("DeadlineSlots after a TTL-free insert into a reclaimed slot = %d, want 4", got)
	}
	assertDeadlineCountsBalanced(t, h.arena, "after free-list reuse")

	// The resurrect path in insertLocked: tombstone 2 (which carries BOTH kinds),
	// then re-insert the same id with no TTL — insertLocked frees the dead slot
	// mid-insert via arena.Delete.
	if ok, derr := h.Delete(2, CASCond{}); derr != nil || !ok {
		t.Fatalf("Delete(2): ok=%v err=%v", ok, derr)
	}
	if _, _, err := h.Insert(2, vec(2), 0, nil, nil, nil, CASCond{}); err != nil {
		t.Fatalf("Insert 2 (resurrect): %v", err)
	}
	if got := h.arena.DeadlineSlots(); got != 2 { // point 4 + key 5
		t.Fatalf("DeadlineSlots after the resurrect = %d, want 2", got)
	}
	assertDeadlineCountsBalanced(t, h.arena, "after resurrect")

	// THE case the centralization exists for: a slot that reaches arena.Delete
	// with its deadlines still LIVE. insertLocked's resurrect branch fires on
	// `tombstoned OR expired`, and an expired-but-not-yet-swept slot was never
	// cleared by anything — its expires is still nonzero when the slot goes on
	// the free list, and arena.Insert's reuse then zeroes the array DIRECTLY.
	// Without the clear inside arena.Delete both counters leak +1 here, forever.
	if _, _, err := h.Insert(20, vec(20), time.Minute, meta, nil, keyTTL, CASCond{}); err != nil {
		t.Fatalf("Insert 20: %v", err)
	}
	if got := h.arena.DeadlineSlots(); got != 4 { // point 4,20 + key 5,20
		t.Fatalf("DeadlineSlots after the short-TTL insert = %d, want 4", got)
	}
	fakeNow += 2 * 60 * 1000 // past the deadline, but the sweeper never runs
	if _, _, err := h.Insert(20, vec(20), 0, nil, nil, nil, CASCond{}); err != nil {
		t.Fatalf("Insert 20 (resurrect over an expired slot): %v", err)
	}
	if got := h.arena.DeadlineSlots(); got != 2 { // back to point 4 + key 5
		t.Fatalf("DeadlineSlots after resurrecting an EXPIRED slot = %d, want 2 — the freed slot leaked its deadline counts", got)
	}
	assertDeadlineCountsBalanced(t, h.arena, "after expired-slot resurrect")

	// Drain: the gate must land on EXACTLY 0 or the sweeper scans forever.
	for _, id := range []uint64{2, 4, 5, 6, 10, 20} {
		if ok, derr := h.Delete(id, CASCond{}); derr != nil || !ok {
			t.Fatalf("drain Delete(%d): ok=%v err=%v", id, ok, derr)
		}
	}
	h.Reclaim()
	if got := h.arena.DeadlineSlots(); got != 0 {
		t.Fatalf("DeadlineSlots after draining every point = %d, want exactly 0", got)
	}
	assertDeadlineCountsBalanced(t, h.arena, "after drain")
}
