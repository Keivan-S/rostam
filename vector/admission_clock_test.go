// SPDX-License-Identifier: Apache-2.0

package vector

import (
	"sync/atomic"
	"testing"
	"time"
)

// countingClock returns a now func (unix millis) that counts how many times it is
// read, plus a pointer to the counter so a test can reset it around one search.
func countingClock(base int64) (func() int64, *atomic.Int64) {
	var calls atomic.Int64
	return func() int64 {
		calls.Add(1)
		return base
	}, &calls
}

// TestHNSWAdmissionReadsClockOncePerSearch pins the loop-invariant wall-clock read
// to ONE per filtered search. The admission gate (TTL expiry + per-key-TTL live
// metadata) used to re-read the clock per admitted candidate — once via isExpired
// and again via liveMeta when a predicate is set — i.e. ~N to ~2N reads over N
// probed candidates. The fix snapshots now once at the search entry and threads it
// into admits, so a whole query reads the clock exactly once regardless of how many
// candidates it admits. This test would report ~N reads on the old code.
func TestHNSWAdmissionReadsClockOncePerSearch(t *testing.T) {
	h, err := newHNSW(Config{Dim: 2, Metric: L2, M: 16, EfConstruction: 200, EfSearch: 128, Seed: 1})
	if err != nil {
		t.Fatalf("newHNSW: %v", err)
	}
	defer h.Close()

	// Insert many points, all carrying the filtered field so the predicate admits a
	// large candidate set (the pre-fix per-candidate clock reads scale with this).
	const n = 60
	for id := uint64(1); id <= n; id++ {
		meta := Metadata{"g": NewInt(int64(id % 3))}
		if _, _, err := h.Insert(id, []float32{float32(id), float32(id % 5)}, 0, meta, nil, nil, CASCond{}); err != nil {
			t.Fatalf("insert %d: %v", id, err)
		}
	}

	// Inject a counting clock AFTER the inserts, then reset the counter so we measure
	// exactly one search. A filter with no payload index forces the graph-traversal
	// path, so admits fires per traversed candidate.
	clock, calls := countingClock(1_000_000)
	h.now = clock
	calls.Store(0)

	filter := Filter{Op: FilterEq, Field: "g", Value: NewInt(0)}
	if _, err := h.SearchFiltered([]float32{5, 2}, 5, filter); err != nil {
		t.Fatalf("SearchFiltered: %v", err)
	}
	if got := calls.Load(); got != 1 {
		t.Fatalf("HNSW filtered search read the clock %d times, want exactly 1 (loop-invariant snapshot)", got)
	}
}

// TestIVFAdmissionReadsClockOncePerSearch is the IVF analogue: the gather* probe
// loops snapshot now once instead of re-reading it per probed candidate.
func TestIVFAdmissionReadsClockOncePerSearch(t *testing.T) {
	cfg := DefaultConfig()
	cfg.Dim = 2
	cfg.Metric = L2
	cfg.Seed = 1
	ix, err := newIVF(cfg)
	if err != nil {
		t.Fatalf("newIVF: %v", err)
	}
	defer ix.Close()

	const n = 60
	for id := uint64(1); id <= n; id++ {
		meta := Metadata{"g": NewInt(int64(id % 3))}
		if _, _, err := ix.Insert(id, []float32{float32(id), float32(id % 5)}, 0, meta, nil, nil, CASCond{}); err != nil {
			t.Fatalf("insert %d: %v", id, err)
		}
	}

	clock, calls := countingClock(1_000_000)
	ix.now = clock
	calls.Store(0)

	filter := Filter{Op: FilterEq, Field: "g", Value: NewInt(0)}
	if _, err := ix.SearchFiltered([]float32{5, 2}, 5, filter); err != nil {
		t.Fatalf("SearchFiltered: %v", err)
	}
	if got := calls.Load(); got != 1 {
		t.Fatalf("IVF filtered search read the clock %d times, want exactly 1 (loop-invariant snapshot)", got)
	}
}

// TestAdmissionSnapshotPreservesTTL confirms the once-per-query snapshot did not
// change expiry semantics: a point past its TTL is still filtered, one within TTL
// still admitted, with the injected clock fixed for the whole search.
func TestAdmissionSnapshotPreservesTTL(t *testing.T) {
	h, err := newHNSW(Config{Dim: 2, Metric: L2, M: 16, EfConstruction: 200, EfSearch: 64, Seed: 1})
	if err != nil {
		t.Fatalf("newHNSW: %v", err)
	}
	defer h.Close()

	var fakeNow int64 = 1_000_000
	h.now = func() int64 { return fakeNow }

	// id 1: 50ms TTL (will expire). id 2: no TTL (always live).
	if _, _, err := h.Insert(1, []float32{1, 0}, 50*time.Millisecond, nil, nil, nil, CASCond{}); err != nil {
		t.Fatalf("insert 1: %v", err)
	}
	if _, _, err := h.Insert(2, []float32{0, 1}, 0, nil, nil, nil, CASCond{}); err != nil {
		t.Fatalf("insert 2: %v", err)
	}

	// Before expiry: both visible.
	got, err := h.Search([]float32{1, 0}, 10)
	if err != nil {
		t.Fatalf("search before expiry: %v", err)
	}
	if len(got) != 2 {
		t.Fatalf("before expiry got %d results, want 2", len(got))
	}

	// Advance the fixed clock past id 1's deadline: it must be filtered, id 2 kept.
	fakeNow += 100
	got, err = h.Search([]float32{1, 0}, 10)
	if err != nil {
		t.Fatalf("search after expiry: %v", err)
	}
	if len(got) != 1 || got[0].ID != 2 {
		t.Fatalf("after expiry got %+v, want only id 2 (id 1 TTL-expired)", got)
	}
}
