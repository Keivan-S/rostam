// SPDX-License-Identifier: Apache-2.0

package vector

import (
	"sync"
	"testing"
	"time"
)

// newIfAbsentHNSW builds a tiny L2 index for the if-absent / exists tests.
func newIfAbsentHNSW(t *testing.T) *hnsw {
	t.Helper()
	h, err := newHNSW(Config{Dim: 4, Metric: L2, M: 16, EfConstruction: 200, EfSearch: 64, Seed: 1})
	if err != nil {
		t.Fatalf("newHNSW: %v", err)
	}
	return h
}

// vecAt returns the stored vector for id (live slot), or nil if absent.
func vecAt(h *hnsw, id uint64) []float32 {
	h.mu.RLock()
	defer h.mu.RUnlock()
	slot, ok := h.arena.Slot(id)
	if !ok || h.tombstoned[slot] || h.isExpired(slot) {
		return nil
	}
	return append([]float32(nil), h.arena.Vec(slot)...)
}

func TestInsertIfAbsentInsertsWhenAbsent(t *testing.T) {
	h := newIfAbsentHNSW(t)
	inserted, err := h.InsertIfAbsent(1, []float32{1, 0, 0, 0}, 0, nil, nil)
	if err != nil {
		t.Fatalf("InsertIfAbsent: %v", err)
	}
	if !inserted {
		t.Fatalf("InsertIfAbsent on absent id returned inserted=false, want true")
	}
	if !h.Exists(1) {
		t.Fatalf("Exists(1) = false after insert, want true")
	}
}

func TestInsertIfAbsentNoOpWhenLive(t *testing.T) {
	h := newIfAbsentHNSW(t)
	if _, _, err := h.Insert(1, []float32{1, 0, 0, 0}, 0, nil, nil, nil, CASCond{}); err != nil {
		t.Fatalf("Insert: %v", err)
	}
	// A second if-absent with a DIFFERENT value must be a no-op and must not
	// clobber the live value.
	inserted, err := h.InsertIfAbsent(1, []float32{0, 1, 0, 0}, 0, nil, nil)
	if err != nil {
		t.Fatalf("InsertIfAbsent: %v", err)
	}
	if inserted {
		t.Fatalf("InsertIfAbsent on live id returned inserted=true, want false")
	}
	got := vecAt(h, 1)
	if len(got) != 4 || got[0] != 1 || got[1] != 0 {
		t.Fatalf("live value clobbered by no-op if-absent: got %v, want [1 0 0 0]", got)
	}
}

func TestInsertIfAbsentTreatsTombstonedAsAbsent(t *testing.T) {
	h := newIfAbsentHNSW(t)
	if _, _, err := h.Insert(1, []float32{1, 0, 0, 0}, 0, nil, nil, nil, CASCond{}); err != nil {
		t.Fatalf("Insert: %v", err)
	}
	if ok, _ := h.Delete(1, CASCond{}); !ok {
		t.Fatalf("Delete(1) = false, want true")
	}
	if h.Exists(1) {
		t.Fatalf("Exists(1) = true after delete, want false")
	}
	// A tombstoned slot is NOT live, so if-absent must insert (resurrect with the
	// new value).
	inserted, err := h.InsertIfAbsent(1, []float32{0, 1, 0, 0}, 0, nil, nil)
	if err != nil {
		t.Fatalf("InsertIfAbsent: %v", err)
	}
	if !inserted {
		t.Fatalf("InsertIfAbsent on tombstoned id returned inserted=false, want true")
	}
	got := vecAt(h, 1)
	if len(got) != 4 || got[1] != 1 {
		t.Fatalf("if-absent did not insert over tombstone: got %v, want [0 1 0 0]", got)
	}
}

func TestInsertIfAbsentTreatsExpiredAsAbsent(t *testing.T) {
	h := newIfAbsentHNSW(t)
	var fakeNow int64 = 1_000_000
	h.now = func() int64 { return fakeNow }
	if _, _, err := h.Insert(1, []float32{1, 0, 0, 0}, 50*time.Millisecond, nil, nil, nil, CASCond{}); err != nil {
		t.Fatalf("Insert: %v", err)
	}
	fakeNow += 100 // age past TTL
	if h.Exists(1) {
		t.Fatalf("Exists(1) = true after expiry, want false")
	}
	inserted, err := h.InsertIfAbsent(1, []float32{0, 0, 1, 0}, 0, nil, nil)
	if err != nil {
		t.Fatalf("InsertIfAbsent: %v", err)
	}
	if !inserted {
		t.Fatalf("InsertIfAbsent on expired id returned inserted=false, want true")
	}
	if !h.Exists(1) {
		t.Fatalf("Exists(1) = false after re-insert, want true")
	}
}

func TestExistsLiveDeletedExpiredAbsent(t *testing.T) {
	h := newIfAbsentHNSW(t)
	var fakeNow int64 = 1_000_000
	h.now = func() int64 { return fakeNow }

	// absent
	if h.Exists(99) {
		t.Fatalf("Exists(absent) = true, want false")
	}
	// live
	if _, _, err := h.Insert(1, []float32{1, 0, 0, 0}, 0, nil, nil, nil, CASCond{}); err != nil {
		t.Fatalf("Insert: %v", err)
	}
	if !h.Exists(1) {
		t.Fatalf("Exists(live) = false, want true")
	}
	// deleted
	if _, _, err := h.Insert(2, []float32{0, 1, 0, 0}, 0, nil, nil, nil, CASCond{}); err != nil {
		t.Fatalf("Insert: %v", err)
	}
	h.Delete(2, CASCond{})
	if h.Exists(2) {
		t.Fatalf("Exists(deleted) = true, want false")
	}
	// expired
	if _, _, err := h.Insert(3, []float32{0, 0, 1, 0}, 50*time.Millisecond, nil, nil, nil, CASCond{}); err != nil {
		t.Fatalf("Insert: %v", err)
	}
	fakeNow += 100
	if h.Exists(3) {
		t.Fatalf("Exists(expired) = true, want false")
	}
	// the un-expired live one still exists (sanity that expiry probe is per-slot)
	if !h.Exists(1) {
		t.Fatalf("Exists(1) = false after aging clock, want true (no TTL)")
	}
}

// upsertReplace models the dual-write upsert leg as it lands as ONE op on a
// partition's Raft log: a single critical section that establishes v2 regardless
// of the prior live value (delete-then-insert under one lock — the engine's
// Insert already hard-removes a tombstoned slot, and we tombstone first so the
// id is never a live duplicate). This is the engine-level analogue of the
// vector_upsert op.
func upsertReplace(h *hnsw, id uint64, v []float32) {
	h.Delete(id, CASCond{}) // tombstone any live value so Insert resurrects rather than colliding
	_, _, _ = h.Insert(id, append([]float32(nil), v...), 0, nil, nil, nil, CASCond{})
}

// TestInsertIfAbsentAtomicVsUpsert is the core correctness gate for Race A.
// Because each op (if-absent and upsert) lands as a single serialized op on the
// partition's Raft log, the two never interleave; whatever their order, the
// upsert (v2 — the live write) must always win, never be clobbered to v1 by the
// copy's stale if-absent. We mirror that Raft serialization with an op-mutex and
// randomize the order across many iterations. The -race build additionally
// exercises InsertIfAbsent's single-critical-section locking under contention
// (see TestInsertIfAbsentRaceNoCorruption).
func TestInsertIfAbsentAtomicVsUpsert(t *testing.T) {
	const iters = 500
	v1 := []float32{1, 0, 0, 0}
	v2 := []float32{0, 1, 0, 0}
	for it := 0; it < iters; it++ {
		h := newIfAbsentHNSW(t)
		const id = 7
		var opMu sync.Mutex // mirrors single-Raft-log op serialization on one partition
		var wg sync.WaitGroup
		wg.Add(2)
		go func() {
			defer wg.Done()
			opMu.Lock()
			upsertReplace(h, id, v2)
			opMu.Unlock()
		}()
		go func() {
			defer wg.Done()
			opMu.Lock()
			_, _ = h.InsertIfAbsent(id, append([]float32(nil), v1...), 0, nil, nil)
			opMu.Unlock()
		}()
		wg.Wait()

		// Whatever the serialized order, the upsert (v2) must win:
		//   if-absent before upsert -> v1 then upsert replaces -> v2
		//   upsert before if-absent -> v2 then if-absent sees live -> no-op -> v2
		got := vecAt(h, id)
		if len(got) != 4 || got[0] != 0 || got[1] != 1 {
			t.Fatalf("iter %d: final value = %v, want v2 [0 1 0 0] (Race A: stale if-absent clobbered the upsert)", it, got)
		}
	}
}

// TestInsertIfAbsentRaceNoCorruption hammers InsertIfAbsent against a concurrent
// plain Insert/Delete on the same id with NO external serialization, purely to
// shake out a check-then-act lock gap inside InsertIfAbsent under -race. It does
// not assert a particular winner (the two ops are not Raft-serialized here); it
// asserts the index never corrupts and the id is in a consistent live/absent
// state afterwards.
func TestInsertIfAbsentRaceNoCorruption(t *testing.T) {
	const iters = 300
	for it := 0; it < iters; it++ {
		h := newIfAbsentHNSW(t)
		const id = 11
		var wg sync.WaitGroup
		wg.Add(3)
		go func() { defer wg.Done(); _, _ = h.InsertIfAbsent(id, []float32{1, 0, 0, 0}, 0, nil, nil) }()
		go func() { defer wg.Done(); _, _ = h.InsertIfAbsent(id, []float32{0, 1, 0, 0}, 0, nil, nil) }()
		go func() { defer wg.Done(); h.Delete(id, CASCond{}) }()
		wg.Wait()
		// Consistency: Exists must agree with a fresh point read.
		if h.Exists(id) && vecAt(h, id) == nil {
			t.Fatalf("iter %d: Exists=true but point read is nil (corrupted liveness)", it)
		}
	}
}
