// SPDX-License-Identifier: Apache-2.0

package vector

import (
	"testing"
)

// TestScanVectorsCarriesKeyExpires: scanVectors clones the arena's ABSOLUTE
// per-key deadline map into ScanRecord.KeyExpires (an owned copy, not an alias),
// for both insert-time and set_payload-set per-key TTLs; points without per-key
// TTL get nil.
func TestScanVectorsCarriesKeyExpires(t *testing.T) {
	h, err := newHNSW(keyTTLCfg())
	if err != nil {
		t.Fatalf("newHNSW: %v", err)
	}
	var fakeNow int64 = 5_000_000
	h.now = func() int64 { return fakeNow }

	// id 1: insert-time per-key TTL.
	if _, _, err := h.Insert(1, []float32{1, 0, 0, 0}, 0,
		Metadata{"temp": NewString("x")}, nil, map[string]int64{"temp": 1000}, CASCond{}); err != nil {
		t.Fatalf("Insert 1: %v", err)
	}
	// id 2: set_payload per-key TTL.
	if _, _, err := h.Insert(2, []float32{0, 1, 0, 0}, 0, Metadata{"p": NewString("y")}, nil, nil, CASCond{}); err != nil {
		t.Fatalf("Insert 2: %v", err)
	}
	if _, _, _, err := h.SetPayload(2, Metadata{"sp": NewString("z")}, map[string]int64{"sp": 2000}, CASCond{}); err != nil {
		t.Fatalf("SetPayload 2: %v", err)
	}
	// id 3: no per-key TTL.
	if _, _, err := h.Insert(3, []float32{0, 0, 1, 0}, 0, Metadata{"k": NewInt(1)}, nil, nil, CASCond{}); err != nil {
		t.Fatalf("Insert 3: %v", err)
	}

	byID := map[uint64]ScanRecord{}
	for _, r := range h.scanVectors() {
		byID[r.ID] = r
	}
	if got := byID[1].KeyExpires["temp"]; got != uint64(fakeNow)+1000 {
		t.Fatalf("id1 KeyExpires[temp] = %d, want %d (absolute insert-time)", got, uint64(fakeNow)+1000)
	}
	if got := byID[2].KeyExpires["sp"]; got != uint64(fakeNow)+2000 {
		t.Fatalf("id2 KeyExpires[sp] = %d, want %d (absolute set_payload)", got, uint64(fakeNow)+2000)
	}
	if byID[3].KeyExpires != nil {
		t.Fatalf("id3 KeyExpires = %v, want nil", byID[3].KeyExpires)
	}
	// The clone must NOT alias arena storage: mutating the scan copy is harmless.
	byID[1].KeyExpires["temp"] = 0
	slot, _ := h.arena.Slot(1)
	if got := h.arena.KeyExpires(slot)["temp"]; got != uint64(fakeNow)+1000 {
		t.Fatalf("scan KeyExpires aliased arena storage (got %d after mutating the copy)", got)
	}
}

// TestRestoreInsertKeyExpiresTimeStable simulates the OFFLINE reshard reinsert:
// a point's ABSOLUTE per-key deadline is restored VERBATIM at a LATER now, and the
// key expires at its ORIGINAL absolute deadline — NOT recomputed now+ttl. The
// revert-fails-it twin (RestoreInsert with nil keyExpires) leaves the key
// PERMANENT, proving the carriage is load-bearing.
func TestRestoreInsertKeyExpiresTimeStable(t *testing.T) {
	// Source: insert at T0 with a 1000ms per-key TTL → absolute deadline T0+1000.
	src, err := newHNSW(keyTTLCfg())
	if err != nil {
		t.Fatalf("newHNSW src: %v", err)
	}
	var t0 int64 = 10_000_000
	src.now = func() int64 { return t0 }
	if _, _, err := src.Insert(1, []float32{1, 0, 0, 0}, 0,
		Metadata{"temp": NewString("bye"), "perm": NewString("keep")}, nil,
		map[string]int64{"temp": 1000}, CASCond{}); err != nil {
		t.Fatalf("src Insert: %v", err)
	}
	rec := src.scanVectors()[0]
	wantDeadline := uint64(t0) + 1000
	if rec.KeyExpires["temp"] != wantDeadline {
		t.Fatalf("scanned KeyExpires[temp] = %d, want %d", rec.KeyExpires["temp"], wantDeadline)
	}

	// Dest: the reshard copy runs MUCH later (T0+5000, already past the deadline+...
	// well past). RestoreInsert sets the deadline VERBATIM, so 'temp' must be
	// considered expired immediately (its ORIGINAL deadline is in the past), NOT
	// recomputed to dstNow+1000.
	dst, err := newHNSW(keyTTLCfg())
	if err != nil {
		t.Fatalf("newHNSW dst: %v", err)
	}
	dstNow := t0 + 5000
	dst.now = func() int64 { return dstNow }
	if err := dst.RestoreInsert(rec.ID, rec.Vec, rec.TTL, rec.Metadata, rec.Sparse, rec.KeyExpires, rec.Version); err != nil {
		t.Fatalf("dst RestoreInsert: %v", err)
	}
	slot, _ := dst.arena.Slot(1)
	if got := dst.arena.KeyExpires(slot)["temp"]; got != wantDeadline {
		t.Fatalf("VERBATIM violated: dst arena KeyExpires[temp] = %d, want %d (absolute, NOT recomputed dstNow+1000=%d)",
			got, wantDeadline, dstNow+1000)
	}
	// Read: 'temp' already expired at its original deadline; 'perm' survives.
	_, meta, _, _, _, ok := dst.Get(1)
	if !ok {
		t.Fatal("Get(1) ok=false after reshard (the POINT must still live)")
	}
	if hasKey(meta, "temp") {
		t.Fatalf("time-stable violated: 'temp' still present at dstNow=%d (original deadline %d): %v", dstNow, wantDeadline, meta)
	}
	if meta["perm"].Str != "keep" {
		t.Fatalf("permanent key lost after reshard: %v", meta)
	}

	// revert-fails-it twin: copying with nil keyExpires (the pre-fix behavior) makes
	// 'temp' PERMANENT — it would NOT expire at its original deadline.
	dst2, err := newHNSW(keyTTLCfg())
	if err != nil {
		t.Fatalf("newHNSW dst2: %v", err)
	}
	dst2.now = func() int64 { return dstNow }
	if err := dst2.RestoreInsert(rec.ID, rec.Vec, rec.TTL, rec.Metadata, rec.Sparse, nil, rec.Version); err != nil {
		t.Fatalf("dst2 RestoreInsert: %v", err)
	}
	if _, meta2, _, _, _, ok2 := dst2.Get(1); !ok2 || !hasKey(meta2, "temp") {
		t.Fatalf("revert-fails-it sanity broken: with nil keyExpires 'temp' should be permanent (ok=%v meta=%v)", ok2, meta2)
	}
}

// TestInsertIfAbsentVersionKeyExpiresTimeStable mirrors the ONLINE reshard copy
// (vector_insert_if_absent → InsertIfAbsentVersion): on a REAL insert the ABSOLUTE
// per-key deadline is set VERBATIM (NOT recomputed) so the copied key expires at
// its original absolute time. version==0 && nil keyExpires reproduces today's
// behavior.
func TestInsertIfAbsentVersionKeyExpiresTimeStable(t *testing.T) {
	src, err := newHNSW(keyTTLCfg())
	if err != nil {
		t.Fatalf("newHNSW src: %v", err)
	}
	var t0 int64 = 20_000_000
	src.now = func() int64 { return t0 }
	if _, _, err := src.Insert(1, []float32{1, 0, 0, 0}, 0,
		Metadata{"temp": NewString("bye")}, nil, map[string]int64{"temp": 1000}, CASCond{}); err != nil {
		t.Fatalf("src Insert: %v", err)
	}
	rec := src.scanVectors()[0]
	wantDeadline := uint64(t0) + 1000

	dst, err := newHNSW(keyTTLCfg())
	if err != nil {
		t.Fatalf("newHNSW dst: %v", err)
	}
	dstNow := t0 + 5000
	dst.now = func() int64 { return dstNow }
	inserted, err := dst.InsertIfAbsentVersion(rec.ID, rec.Vec, rec.TTL, rec.Metadata, rec.Sparse, rec.KeyExpires, rec.Version)
	if err != nil || !inserted {
		t.Fatalf("InsertIfAbsentVersion: inserted=%v err=%v", inserted, err)
	}
	slot, _ := dst.arena.Slot(1)
	if got := dst.arena.KeyExpires(slot)["temp"]; got != wantDeadline {
		t.Fatalf("VERBATIM violated (online): dst KeyExpires[temp] = %d, want %d (NOT dstNow+1000=%d)",
			got, wantDeadline, dstNow+1000)
	}
	if _, meta, _, _, _, ok := dst.Get(1); !ok || hasKey(meta, "temp") {
		t.Fatalf("time-stable violated (online): 'temp' should be expired at dstNow=%d (deadline %d): ok=%v meta=%v",
			dstNow, wantDeadline, ok, meta)
	}
}

// TestInsertIfAbsentVersionNilKeyExpiresByteIdentical: a plain if-absent (version
// 0, nil keyExpires) leaves the slot with no per-key deadlines — the no-reshard
// path is unchanged.
func TestInsertIfAbsentVersionNilKeyExpiresByteIdentical(t *testing.T) {
	h, err := newHNSW(keyTTLCfg())
	if err != nil {
		t.Fatalf("newHNSW: %v", err)
	}
	inserted, err := h.InsertIfAbsentVersion(1, []float32{1, 0, 0, 0}, 0, Metadata{"k": NewInt(1)}, nil, nil, 0)
	if err != nil || !inserted {
		t.Fatalf("InsertIfAbsentVersion: inserted=%v err=%v", inserted, err)
	}
	slot, _ := h.arena.Slot(1)
	if ke := h.arena.KeyExpires(slot); ke != nil {
		t.Fatalf("plain if-absent set per-key deadlines %v, want nil", ke)
	}
}
