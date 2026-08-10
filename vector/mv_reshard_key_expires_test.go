// SPDX-License-Identifier: Apache-2.0

package vector

import (
	"testing"
)

// mvKeyExpiresCfg is the shared config for the MV reshard-keyExpires tests.
func mvKeyExpiresCfg() MultiVectorConfig { return MultiVectorConfig{Dim: 4, Seed: 1} }

// hasMVKey reports whether payload key k is present (a per-key TTL drops the key
// lazily on Get once its absolute deadline passes).
func hasMVKey(pay Metadata, k string) bool {
	if pay == nil {
		return false
	}
	_, ok := pay[k]
	return ok
}

// TestMVScanDocumentsCarriesKeyExpires: ScanDocuments clones each doc's ABSOLUTE
// per-key deadline map into MultiScanRecord.KeyExpires (an owned copy, not an
// alias), for both add-time and set_payload-set per-key TTLs; docs without per-key
// TTL get nil.
func TestMVScanDocumentsCarriesKeyExpires(t *testing.T) {
	m, err := NewMultiVectorIndex(mvKeyExpiresCfg())
	if err != nil {
		t.Fatalf("new: %v", err)
	}
	var fakeNow int64 = 5_000_000
	m.now = func() int64 { return fakeNow }

	// doc 1: add-time per-key TTL.
	if _, err := m.AddCASKeyTTL(1, [][]float32{{1, 0, 0, 0}}, Metadata{"temp": NewString("x")}, map[string]int64{"temp": 1000}, CASCond{}); err != nil {
		t.Fatalf("Add 1: %v", err)
	}
	// doc 2: set_payload per-key TTL.
	if err := m.Add(2, [][]float32{{0, 1, 0, 0}}, Metadata{"p": NewString("y")}); err != nil {
		t.Fatalf("Add 2: %v", err)
	}
	if err := m.SetPayload(2, Metadata{"sp": NewString("z")}, map[string]int64{"sp": 2000}); err != nil {
		t.Fatalf("SetPayload 2: %v", err)
	}
	// doc 3: no per-key TTL.
	if err := m.Add(3, [][]float32{{0, 0, 1, 0}}, Metadata{"k": NewInt(1)}); err != nil {
		t.Fatalf("Add 3: %v", err)
	}

	byID := map[uint64]MultiScanRecord{}
	for _, r := range m.ScanDocuments() {
		byID[r.ID] = r
	}
	if got := byID[1].KeyExpires["temp"]; got != uint64(fakeNow)+1000 {
		t.Fatalf("doc1 KeyExpires[temp] = %d, want %d (absolute add-time)", got, uint64(fakeNow)+1000)
	}
	if got := byID[2].KeyExpires["sp"]; got != uint64(fakeNow)+2000 {
		t.Fatalf("doc2 KeyExpires[sp] = %d, want %d (absolute set_payload)", got, uint64(fakeNow)+2000)
	}
	if byID[3].KeyExpires != nil {
		t.Fatalf("doc3 KeyExpires = %v, want nil", byID[3].KeyExpires)
	}
	// The clone must NOT alias m.keyTTL storage: mutating the scan copy is harmless.
	byID[1].KeyExpires["temp"] = 0
	m.mu.RLock()
	stored := m.keyTTL[1]["temp"]
	m.mu.RUnlock()
	if stored != int64(fakeNow)+1000 {
		t.Fatalf("scan KeyExpires aliased m.keyTTL storage (got %d after mutating the copy)", stored)
	}
}

// TestMVRestoreAddKeyExpiresTimeStable simulates the OFFLINE MV reshard reinsert:
// a doc's ABSOLUTE per-key deadline is restored VERBATIM at a LATER now, and the
// key expires at its ORIGINAL absolute deadline — NOT recomputed now+ttl. The
// revert-fails-it twin (MultiRestoreAdd with nil keyExpires) leaves the key
// PERMANENT, proving the carriage is load-bearing.
func TestMVRestoreAddKeyExpiresTimeStable(t *testing.T) {
	src, err := NewMultiVectorIndex(mvKeyExpiresCfg())
	if err != nil {
		t.Fatalf("new src: %v", err)
	}
	var t0 int64 = 10_000_000
	src.now = func() int64 { return t0 }
	if _, err := src.AddCASKeyTTL(1, [][]float32{{1, 0, 0, 0}},
		Metadata{"temp": NewString("bye"), "perm": NewString("keep")},
		map[string]int64{"temp": 1000}, CASCond{}); err != nil {
		t.Fatalf("src Add: %v", err)
	}
	rec := src.ScanDocuments()[0]
	wantDeadline := uint64(t0) + 1000
	if rec.KeyExpires["temp"] != wantDeadline {
		t.Fatalf("scanned KeyExpires[temp] = %d, want %d", rec.KeyExpires["temp"], wantDeadline)
	}

	// Dest: the reshard copy runs MUCH later (already past the deadline).
	dst, err := NewMultiVectorIndex(mvKeyExpiresCfg())
	if err != nil {
		t.Fatalf("new dst: %v", err)
	}
	dstNow := t0 + 5000
	dst.now = func() int64 { return dstNow }
	if err := dst.MultiRestoreAdd(rec.ID, rec.Tokens, rec.Metadata, rec.KeyExpires, rec.Version); err != nil {
		t.Fatalf("dst MultiRestoreAdd: %v", err)
	}
	dst.mu.RLock()
	stored := dst.keyTTL[1]["temp"]
	dst.mu.RUnlock()
	if uint64(stored) != wantDeadline {
		t.Fatalf("VERBATIM violated: dst keyTTL[temp] = %d, want %d (absolute, NOT recomputed dstNow+1000=%d)",
			stored, wantDeadline, dstNow+1000)
	}
	// Read: 'temp' already expired at its original deadline; 'perm' survives.
	_, pay, _, ok := dst.Get(1)
	if !ok {
		t.Fatal("Get(1) ok=false after reshard (the DOC must still live)")
	}
	if hasMVKey(pay, "temp") {
		t.Fatalf("time-stable violated: 'temp' still present at dstNow=%d (original deadline %d): %v", dstNow, wantDeadline, pay)
	}
	if pay["perm"].Str != "keep" {
		t.Fatalf("permanent key lost after reshard: %v", pay)
	}

	// revert-fails-it twin: copying with nil keyExpires (the pre-fix behavior) makes
	// 'temp' PERMANENT — it would NOT expire at its original deadline.
	dst2, err := NewMultiVectorIndex(mvKeyExpiresCfg())
	if err != nil {
		t.Fatalf("new dst2: %v", err)
	}
	dst2.now = func() int64 { return dstNow }
	if err := dst2.MultiRestoreAdd(rec.ID, rec.Tokens, rec.Metadata, nil, rec.Version); err != nil {
		t.Fatalf("dst2 MultiRestoreAdd: %v", err)
	}
	if _, pay2, _, ok2 := dst2.Get(1); !ok2 || !hasMVKey(pay2, "temp") {
		t.Fatalf("revert-fails-it sanity broken: with nil keyExpires 'temp' should be permanent (ok=%v pay=%v)", ok2, pay2)
	}
}

// TestMVAddIfAbsentVersionKeyExpiresTimeStable mirrors the ONLINE MV reshard copy
// (vector_mv_add_if_absent → MultiAddIfAbsentVersion): on a REAL add the ABSOLUTE
// per-key deadline is set VERBATIM (NOT recomputed) so the copied key expires at
// its original absolute time.
func TestMVAddIfAbsentVersionKeyExpiresTimeStable(t *testing.T) {
	src, err := NewMultiVectorIndex(mvKeyExpiresCfg())
	if err != nil {
		t.Fatalf("new src: %v", err)
	}
	var t0 int64 = 20_000_000
	src.now = func() int64 { return t0 }
	if _, err := src.AddCASKeyTTL(1, [][]float32{{1, 0, 0, 0}},
		Metadata{"temp": NewString("bye")}, map[string]int64{"temp": 1000}, CASCond{}); err != nil {
		t.Fatalf("src Add: %v", err)
	}
	rec := src.ScanDocuments()[0]
	wantDeadline := uint64(t0) + 1000

	dst, err := NewMultiVectorIndex(mvKeyExpiresCfg())
	if err != nil {
		t.Fatalf("new dst: %v", err)
	}
	dstNow := t0 + 5000
	dst.now = func() int64 { return dstNow }
	inserted, err := dst.MultiAddIfAbsentVersion(rec.ID, rec.Tokens, rec.Metadata, rec.KeyExpires, rec.Version)
	if err != nil || !inserted {
		t.Fatalf("MultiAddIfAbsentVersion: inserted=%v err=%v", inserted, err)
	}
	dst.mu.RLock()
	stored := dst.keyTTL[1]["temp"]
	dst.mu.RUnlock()
	if uint64(stored) != wantDeadline {
		t.Fatalf("VERBATIM violated (online): dst keyTTL[temp] = %d, want %d (NOT dstNow+1000=%d)",
			stored, wantDeadline, dstNow+1000)
	}
	if _, pay, _, ok := dst.Get(1); !ok || hasMVKey(pay, "temp") {
		t.Fatalf("time-stable violated (online): 'temp' should be expired at dstNow=%d (deadline %d): ok=%v pay=%v",
			dstNow, wantDeadline, ok, pay)
	}
}

// TestMVAddIfAbsentVersionNilKeyExpiresByteIdentical: a plain if-absent (version 0,
// nil keyExpires) leaves the doc with no per-key deadlines — the no-reshard path is
// unchanged (a fresh add → version 1).
func TestMVAddIfAbsentVersionNilKeyExpiresByteIdentical(t *testing.T) {
	m, err := NewMultiVectorIndex(mvKeyExpiresCfg())
	if err != nil {
		t.Fatalf("new: %v", err)
	}
	inserted, err := m.MultiAddIfAbsentVersion(1, [][]float32{{1, 0, 0, 0}}, Metadata{"k": NewInt(1)}, nil, 0)
	if err != nil || !inserted {
		t.Fatalf("MultiAddIfAbsentVersion: inserted=%v err=%v", inserted, err)
	}
	m.mu.RLock()
	ke := m.keyTTL[1]
	m.mu.RUnlock()
	if ke != nil {
		t.Fatalf("plain if-absent set per-key deadlines %v, want nil", ke)
	}
	if _, _, v, ok := m.Get(1); !ok || v != 1 {
		t.Fatalf("plain if-absent version = %d (ok=%v), want 1", v, ok)
	}
}
