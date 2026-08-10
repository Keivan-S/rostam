// SPDX-License-Identifier: Apache-2.0

package vector

import (
	"testing"
	"time"
)

// slotContains reports whether the candidate slot list returned by the
// payloadIdx for filter f (filter-first path) includes slot.
func slotContains(slots []uint32, want uint32) bool {
	for _, s := range slots {
		if s == want {
			return true
		}
	}
	return false
}

// TestSweepKeyTTLPayloadIdxHygiene is THE proof: after a payload key's deadline
// passes, sweepOnce PHYSICALLY removes the key from arena.Metadata(slot) AND
// drops its posting from the dense payloadIdx — so candidates() (the
// filter-first index path) no longer returns the slot for that field/value.
// Before the sweep the lazy read path already returns correct results, but the
// index still carries the stale posting; the sweep is what reclaims it.
func TestSweepKeyTTLPayloadIdxHygiene(t *testing.T) {
	cfg := keyTTLCfg()
	h, err := newHNSW(cfg)
	if err != nil {
		t.Fatalf("newHNSW: %v", err)
	}
	var fakeNow int64 = 1_000_000
	h.now = func() int64 { return fakeNow }

	if _, _, err := h.Insert(1, []float32{1, 0, 0, 0}, 0, nil, nil, nil, CASCond{}); err != nil {
		t.Fatalf("Insert 1: %v", err)
	}
	// Seed other points with temp==hot so the posting list exists and
	// candidates(filter) returns ok=true (filter-first path).
	seedID := uint64(100)
	for j := 0; j < 10; j++ {
		if _, _, err := h.Insert(seedID, []float32{0, 0, float32(seedID), 0}, 0, Metadata{"temp": NewString("hot")}, nil, nil, CASCond{}); err != nil {
			t.Fatalf("Insert seeder %d: %v", seedID, err)
		}
		seedID++
	}
	// id=1 gets temp==hot with a 1000ms TTL.
	if _, _, _, err := h.SetPayload(1, Metadata{"temp": NewString("hot")}, map[string]int64{"temp": 1000}, CASCond{}); err != nil {
		t.Fatalf("SetPayload: %v", err)
	}
	slot, _ := h.arena.Slot(1)
	f := Filter{Op: FilterEq, Field: "temp", Value: NewString("hot")}

	// Before expiry: the index posting for slot exists (stale-posting baseline).
	cands, ok := h.payloadIdx.candidates(f, cfg.FilterFirstThreshold)
	if !ok {
		t.Fatal("candidates ok=false — test would not exercise the index path")
	}
	if !slotContains(cands, slot) {
		t.Fatalf("pre-sweep candidates = %v, want to contain slot %d", cands, slot)
	}

	// Advance past the deadline and sweep.
	fakeNow += 1500
	h.sweepOnce()

	// The key is physically gone from the arena metadata.
	if _, present := h.arena.Metadata(slot)["temp"]; present {
		t.Fatalf("post-sweep arena meta still has expired key 'temp': %v", h.arena.Metadata(slot))
	}
	// And the posting is physically removed from the index: candidates no longer
	// lists the slot for temp==hot. THIS is the hygiene proof.
	cands, ok = h.payloadIdx.candidates(f, cfg.FilterFirstThreshold)
	if !ok {
		t.Fatal("candidates ok=false after sweep — seeders should keep the posting list alive")
	}
	if slotContains(cands, slot) {
		t.Fatalf("post-sweep candidates = %v, want NOT to contain slot %d (stale posting not reclaimed)", cands, slot)
	}
}

// TestSweepKeyTTLIdempotentWithLazy: a Get after the sweep returns the SAME
// payload that the lazy liveMeta returns before the sweep (same predicate ⇒ the
// swept slot reads identically to a lazily-dropped one).
func TestSweepKeyTTLIdempotentWithLazy(t *testing.T) {
	h, err := newHNSW(keyTTLCfg())
	if err != nil {
		t.Fatalf("newHNSW: %v", err)
	}
	var fakeNow int64 = 1_000_000
	h.now = func() int64 { return fakeNow }

	if _, _, err := h.Insert(1, []float32{1, 0, 0, 0}, 0, Metadata{"perm": NewInt(1)}, nil, nil, CASCond{}); err != nil {
		t.Fatalf("Insert: %v", err)
	}
	if _, _, _, err := h.SetPayload(1, Metadata{"temp": NewInt(9), "perm2": NewString("x")},
		map[string]int64{"temp": 1000}, CASCond{}); err != nil {
		t.Fatalf("SetPayload: %v", err)
	}

	fakeNow += 1500
	// Lazy result BEFORE sweep (liveMeta hides the expired key).
	_, lazyMeta, _, _, _, ok := h.Get(1)
	if !ok {
		t.Fatal("Get pre-sweep ok=false")
	}
	if _, present := lazyMeta["temp"]; present {
		t.Fatalf("lazy meta still has expired key: %v", lazyMeta)
	}

	h.sweepOnce()

	// Result AFTER sweep must equal the lazy result.
	_, sweptMeta, _, _, _, ok := h.Get(1)
	if !ok {
		t.Fatal("Get post-sweep ok=false (the point must survive a key-only sweep)")
	}
	if len(sweptMeta) != len(lazyMeta) {
		t.Fatalf("post-sweep meta %v != lazy meta %v (size)", sweptMeta, lazyMeta)
	}
	for k, v := range lazyMeta {
		got, present := sweptMeta[k]
		if !present || !got.Equal(v) {
			t.Fatalf("post-sweep meta %v != lazy meta %v (key %q)", sweptMeta, lazyMeta, k)
		}
	}
}

// TestSweepKeyTTLNoIdSetVersionBump: a key-only sweep does NOT bump idSetVersion
// (a key drop does not change the live id set). A POINT-expiry sweep in the same
// process still bumps — verifying the point path is unaffected.
func TestSweepKeyTTLNoIdSetVersionBump(t *testing.T) {
	h, err := newHNSW(keyTTLCfg())
	if err != nil {
		t.Fatalf("newHNSW: %v", err)
	}
	var fakeNow int64 = 1_000_000
	h.now = func() int64 { return fakeNow }

	// Point with no point-TTL but a per-key TTL.
	if _, _, err := h.Insert(1, []float32{1, 0, 0, 0}, 0, Metadata{"perm": NewInt(1)}, nil, nil, CASCond{}); err != nil {
		t.Fatalf("Insert 1: %v", err)
	}
	if _, _, _, err := h.SetPayload(1, Metadata{"temp": NewInt(9)}, map[string]int64{"temp": 1000}, CASCond{}); err != nil {
		t.Fatalf("SetPayload: %v", err)
	}

	fakeNow += 1500
	verBefore := h.idSetVersion
	if n := h.sweepOnce(); n != 0 {
		t.Fatalf("key-only sweep tombstoned %d points, want 0", n)
	}
	if h.idSetVersion != verBefore {
		t.Fatalf("key-only sweep bumped idSetVersion %d -> %d, want unchanged", verBefore, h.idSetVersion)
	}
	// Confirm the key was actually swept (the predicate fired) so the no-bump
	// assertion is meaningful.
	if h.keysSwept.Load() == 0 {
		t.Fatal("expected at least one key swept")
	}

	// Now a POINT-expiry sweep must still bump idSetVersion (point path intact).
	if _, _, err := h.Insert(2, []float32{0, 1, 0, 0}, 50*time.Millisecond, nil, nil, nil, CASCond{}); err != nil {
		t.Fatalf("Insert 2 (point ttl): %v", err)
	}
	verBefore = h.idSetVersion
	fakeNow += 100
	if n := h.sweepOnce(); n != 1 {
		t.Fatalf("point sweep tombstoned %d, want 1", n)
	}
	if h.idSetVersion != verBefore+1 {
		t.Fatalf("point sweep bumped idSetVersion %d -> %d, want +1", verBefore, h.idSetVersion)
	}
}

// TestSweepKeyTTLNonExpiredSurvive: a non-expired key and the point survive the
// sweep; a sweep of a slot with no keyExpires is a no-op.
func TestSweepKeyTTLNonExpiredSurvive(t *testing.T) {
	h, err := newHNSW(keyTTLCfg())
	if err != nil {
		t.Fatalf("newHNSW: %v", err)
	}
	var fakeNow int64 = 1_000_000
	h.now = func() int64 { return fakeNow }

	// id=1: a not-yet-expired TTL key + a permanent key.
	if _, _, err := h.Insert(1, []float32{1, 0, 0, 0}, 0, Metadata{"perm": NewInt(1)}, nil, nil, CASCond{}); err != nil {
		t.Fatalf("Insert 1: %v", err)
	}
	if _, _, _, err := h.SetPayload(1, Metadata{"temp": NewInt(9)}, map[string]int64{"temp": 5000}, CASCond{}); err != nil {
		t.Fatalf("SetPayload: %v", err)
	}
	// id=2: no per-key TTL at all (fast-path no-op).
	if _, _, err := h.Insert(2, []float32{0, 1, 0, 0}, 0, Metadata{"a": NewInt(2)}, nil, nil, CASCond{}); err != nil {
		t.Fatalf("Insert 2: %v", err)
	}

	fakeNow += 1000 // past nothing — temp deadline is now+5000
	if n := h.sweepOnce(); n != 0 {
		t.Fatalf("sweep tombstoned %d, want 0", n)
	}
	if h.keysSwept.Load() != 0 {
		t.Fatalf("keysSwept = %d, want 0 (nothing expired yet)", h.keysSwept.Load())
	}
	if _, meta, _, _, _, ok := h.Get(1); !ok || meta["temp"].Int != 9 || meta["perm"].Int != 1 {
		t.Fatalf("id=1 lost a live key after no-op sweep: meta=%v ok=%v", meta, ok)
	}
	if _, meta, _, _, _, ok := h.Get(2); !ok || meta["a"].Int != 2 {
		t.Fatalf("id=2 (no per-key TTL) changed after sweep: meta=%v ok=%v", meta, ok)
	}
}

// TestSweepKeyTTLPointExpiredNotDoubleProcessed: a point-expired slot is
// tombstoned by the point pass and its keys are NOT separately swept (the point
// gate fires first and continues).
func TestSweepKeyTTLPointExpiredNotDoubleProcessed(t *testing.T) {
	h, err := newHNSW(keyTTLCfg())
	if err != nil {
		t.Fatalf("newHNSW: %v", err)
	}
	var fakeNow int64 = 1_000_000
	h.now = func() int64 { return fakeNow }

	// A point with BOTH a point TTL and a per-key TTL that both expire.
	if _, _, err := h.Insert(1, []float32{1, 0, 0, 0}, 50*time.Millisecond, Metadata{"perm": NewInt(1)}, nil, nil, CASCond{}); err != nil {
		t.Fatalf("Insert: %v", err)
	}
	if _, _, _, err := h.SetPayload(1, Metadata{"temp": NewInt(9)}, map[string]int64{"temp": 10}, CASCond{}); err != nil {
		t.Fatalf("SetPayload: %v", err)
	}
	slot, _ := h.arena.Slot(1)

	fakeNow += 100 // both deadlines passed
	if n := h.sweepOnce(); n != 1 {
		t.Fatalf("sweep tombstoned %d, want 1 (the point)", n)
	}
	if !h.tombstoned[slot] {
		t.Fatal("point-expired slot not tombstoned")
	}
	// The per-key pass must NOT have run on the tombstoned slot.
	if h.keysSwept.Load() != 0 {
		t.Fatalf("keysSwept = %d, want 0 (point gate fires first, skips key pass)", h.keysSwept.Load())
	}
}
