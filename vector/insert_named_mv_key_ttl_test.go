// SPDX-License-Identifier: Apache-2.0

package vector

import (
	"bytes"
	"testing"
	"time"
)

// Insert/add-time per-key payload TTL for the named + MV families: the engine
// turns a RELATIVE-ms key TTL supplied AT INSERT/ADD into an ABSOLUTE deadline
// now+ttl (mirroring set_payload), drops the key lazily on read after the deadline
// while the point/document lives on, persists the absolute deadline VERBATIM
// (snapshot + WAL replay, time-stable), and stays byte-identical / zero-overhead on
// the no-key_ttl path. Mirrors the dense insert_key_ttl_test. The injectable clock
// ages deterministically; no sleeps.

// --- named ---

// TestNamedInsertKeyTTLDropsExpiredKey: a per-key TTL supplied at named Insert is
// present before its deadline and DROPPED after, while the point + permanent key
// survive. Exercises NamedCollection.InsertCASKeyTTL → insertLocked → nc.keyTTL.
func TestNamedInsertKeyTTLDropsExpiredKey(t *testing.T) {
	nc, err := NewNamedCollection("default/named", namedTestConfig())
	if err != nil {
		t.Fatalf("new: %v", err)
	}
	defer nc.Close()
	var fakeNow int64 = 2_000_000
	nc.now = func() int64 { return fakeNow }

	if _, err := nc.InsertCASKeyTTL(1, map[string][]float32{"title": {1, 0, 0, 0}}, nil,
		Metadata{"perm": NewString("keep"), "temp": NewString("bye")}, 0,
		map[string]int64{"temp": 1000}, CASCond{}); err != nil {
		t.Fatalf("InsertCASKeyTTL: %v", err)
	}
	wantDeadline := fakeNow + 1000
	if got := nc.keyTTL[1]["temp"]; got != wantDeadline {
		t.Fatalf("nc.keyTTL[temp] = %d, want %d (absolute now+ttl)", got, wantDeadline)
	}

	// Before the deadline: both keys present.
	_, pay, _, _, ok := nc.Get(1)
	if !ok {
		t.Fatal("Get(1) ok=false before key expiry")
	}
	if pay["temp"].Str != "bye" || pay["perm"].Str != "keep" {
		t.Fatalf("pre-expiry payload = %v, want both keys", pay)
	}

	// Past the deadline: temp dropped (lazy-drop), point + perm survive.
	fakeNow += 1500
	_, pay, _, _, ok = nc.Get(1)
	if !ok {
		t.Fatal("Get(1) ok=false after key expiry (the POINT must still live)")
	}
	if _, has := pay["temp"]; has {
		t.Fatalf("post-expiry payload still has expired key 'temp': %v", pay)
	}
	if pay["perm"].Str != "keep" {
		t.Fatalf("post-expiry lost permanent key: %v", pay)
	}
}

// TestNamedInsertKeyTTLOnlyPresentKeys: a key TTL whose key is NOT in the payload
// sets no deadline; a ttl<=0 is ignored (the key is permanent). Mirrors
// computeInsertKeyExpires / set_payload semantics.
func TestNamedInsertKeyTTLOnlyPresentKeys(t *testing.T) {
	nc, err := NewNamedCollection("default/named", namedTestConfig())
	if err != nil {
		t.Fatalf("new: %v", err)
	}
	defer nc.Close()
	var fakeNow int64 = 100
	nc.now = func() int64 { return fakeNow }

	if _, err := nc.InsertCASKeyTTL(1, map[string][]float32{"title": {1, 0, 0, 0}}, nil,
		Metadata{"a": NewInt(1), "b": NewInt(2)}, 0,
		map[string]int64{
			"a":       1000, // present → deadline
			"b":       0,    // ttl<=0 → permanent, no deadline
			"missing": 1000, // absent from payload → skipped
		}, CASCond{}); err != nil {
		t.Fatalf("InsertCASKeyTTL: %v", err)
	}
	if len(nc.keyTTL[1]) != 1 {
		t.Fatalf("nc.keyTTL[1] = %v, want exactly {a}", nc.keyTTL[1])
	}
	if got := nc.keyTTL[1]["a"]; got != fakeNow+1000 {
		t.Fatalf("nc.keyTTL[1][a] = %d, want %d", got, fakeNow+1000)
	}
}

// TestNamedInsertNoKeyTTLLeavesMapClear: a plain Insert (nil key TTL) leaves no
// keyTTL entry — the zero-overhead path (byte-identical to the pre-feature engine).
func TestNamedInsertNoKeyTTLLeavesMapClear(t *testing.T) {
	nc, err := NewNamedCollection("default/named", namedTestConfig())
	if err != nil {
		t.Fatalf("new: %v", err)
	}
	defer nc.Close()
	if err := nc.Insert(1, map[string][]float32{"title": {1, 0, 0, 0}}, Metadata{"a": NewInt(1)}, 0); err != nil {
		t.Fatalf("Insert: %v", err)
	}
	if _, present := nc.keyTTL[1]; present {
		t.Fatalf("nc.keyTTL[1] = %v, want absent for a no-key_ttl insert", nc.keyTTL[1])
	}
}

// TestNamedUpsertKeyTTLSetsFreshDeadline: re-inserting (upsert) a point REPLACES
// the payload AND its per-key deadlines; the new key TTL applies to the fresh
// payload, and a prior deadline on a now-absent key is gone.
func TestNamedUpsertKeyTTLSetsFreshDeadline(t *testing.T) {
	nc, err := NewNamedCollection("default/named", namedTestConfig())
	if err != nil {
		t.Fatalf("new: %v", err)
	}
	defer nc.Close()
	var fakeNow int64 = 5_000_000
	nc.now = func() int64 { return fakeNow }

	// First insert: key "old" with a deadline.
	if _, err := nc.InsertCASKeyTTL(1, map[string][]float32{"title": {1, 0, 0, 0}}, nil,
		Metadata{"old": NewInt(1)}, 0, map[string]int64{"old": 500}, CASCond{}); err != nil {
		t.Fatalf("insert: %v", err)
	}
	// Upsert: fresh payload {new}, fresh deadline on "new"; "old" is gone entirely.
	if _, err := nc.InsertCASKeyTTL(1, map[string][]float32{"title": {0, 1, 0, 0}}, nil,
		Metadata{"new": NewInt(2)}, 0, map[string]int64{"new": 700}, CASCond{}); err != nil {
		t.Fatalf("upsert: %v", err)
	}
	if _, has := nc.keyTTL[1]["old"]; has {
		t.Error("upsert did not drop the replaced-away key's deadline")
	}
	if got := nc.keyTTL[1]["new"]; got != fakeNow+700 {
		t.Fatalf("upsert keyTTL[new] = %d, want %d (fresh now+ttl)", got, fakeNow+700)
	}
}

// TestNamedInsertKeyTTLSnapshotAbsolute: the insert-time absolute key deadline
// survives a snapshot round-trip VERBATIM (time-stable) — snapshot serializes
// nc.keyTTL wholesale, so persistence is automatic for the insert path.
func TestNamedInsertKeyTTLSnapshotAbsolute(t *testing.T) {
	nc, err := NewNamedCollection("default/named", namedTestConfig())
	if err != nil {
		t.Fatalf("new: %v", err)
	}
	defer nc.Close()
	var fakeNow int64 = 3_000_000
	nc.now = func() int64 { return fakeNow }
	if _, err := nc.InsertCASKeyTTL(1, map[string][]float32{"title": {1, 0, 0, 0}}, nil,
		Metadata{"perm": NewInt(1), "temp": NewInt(9)}, 0,
		map[string]int64{"temp": 1000}, CASCond{}); err != nil {
		t.Fatalf("insert: %v", err)
	}
	wantDeadline := fakeNow + 1000

	var buf bytes.Buffer
	if err := nc.Snapshot(&buf); err != nil {
		t.Fatalf("snapshot: %v", err)
	}
	restored, err := NewNamedCollection("default/named", namedTestConfig())
	if err != nil {
		t.Fatalf("new restored: %v", err)
	}
	defer restored.Close()
	if err := restored.Restore(bytes.NewReader(buf.Bytes())); err != nil {
		t.Fatalf("restore: %v", err)
	}
	if got := restored.keyTTL[1]["temp"]; got != wantDeadline {
		t.Fatalf("restored keyTTL[temp] = %d, want %d (snapshot not time-stable)", got, wantDeadline)
	}
	restored.now = func() int64 { return wantDeadline + 1 }
	_, pay, _, _, ok := restored.Get(1)
	if !ok {
		t.Fatal("restored point absent")
	}
	if _, has := pay["temp"]; has {
		t.Errorf("post-restore post-deadline key still present: %v", pay)
	}
}

// TestNamedInsertKeyTTLWALReplayAbsolute: a per-key payload TTL set AT INSERT in a
// WAL-mode named collection survives a CRASH (reopen = checkpoint + WAL tail, NO
// flush) as an ABSOLUTE deadline — NOT recomputed on replay. This proves the
// named WAL thread-not-nil + replay-restore closed the gap (without it the insert
// dropped the per-key TTL).
func TestNamedInsertKeyTTLWALReplayAbsolute(t *testing.T) {
	dir := t.TempDir()
	const base = int64(1_000_000_000_000)

	cs, err := OpenCollectionStore(dir)
	if err != nil {
		t.Fatal(err)
	}
	if err := cs.CreateCollection("named", namedWALConfig()); err != nil {
		t.Fatal(err)
	}
	nc, ok := cs.GetNamed("named")
	if !ok {
		t.Fatal("named collection missing after create")
	}
	if nc.wal == nil {
		t.Fatal("WAL-mode named collection has nil wal")
	}
	freezeNamedClock(nc, base)

	// Insert with a permanent key + a 1000ms key TTL → absolute deadline base+1000.
	if _, err := nc.InsertCASKeyTTL(1, map[string][]float32{"title": {1, 0, 0, 0}}, nil,
		Metadata{"perm": NewInt(1), "temp": NewInt(9)}, time.Hour,
		map[string]int64{"temp": 1000}, CASCond{}); err != nil {
		t.Fatalf("InsertCASKeyTTL: %v", err)
	}
	wantDeadline := base + 1000
	// Crash: ONLY the WAL holds the insert (no flush).
	_ = cs.Close()

	cs2, err := OpenCollectionStore(dir)
	if err != nil {
		t.Fatalf("reopen: %v", err)
	}
	defer func() { _ = cs2.Close() }()
	nc2, ok := cs2.GetNamed("named")
	if !ok {
		t.Fatal("named collection missing after reopen")
	}
	freezeNamedClock(nc2, base)

	if got := nc2.keyTTL[1]["temp"]; got != wantDeadline {
		t.Fatalf("replayed keyTTL[temp] = %d, want %d (absolute deadline not preserved / WAL gap open)", got, wantDeadline)
	}
	if _, pay, _, _, ok := nc2.Get(1); !ok || pay["temp"].Int != 9 {
		t.Fatalf("post-replay pre-deadline payload=%v ok=%v, want temp:9", pay, ok)
	}
	// Advance past the ORIGINAL absolute deadline: the key drops (proves NOT
	// recomputed as now+ttl at replay time).
	freezeNamedClock(nc2, base+2000)
	_, pay, _, _, ok := nc2.Get(1)
	if !ok {
		t.Fatal("point disappeared after aging (it still has hours of point-TTL left)")
	}
	if _, has := pay["temp"]; has {
		t.Errorf("post-replay post-deadline key still present (replay must NOT recompute now+ttl): %v", pay)
	}
	if pay["perm"].Int != 1 {
		t.Errorf("post-replay lost permanent key: %v", pay)
	}
}

// --- MV ---

// TestMVAddKeyTTLDropsExpiredKey: a per-key TTL supplied at MV Add is present
// before its deadline and DROPPED after, while the doc + permanent key survive.
// Exercises MultiVectorIndex.AddCASKeyTTL → addLocked → m.keyTTL.
func TestMVAddKeyTTLDropsExpiredKey(t *testing.T) {
	m, _ := NewMultiVectorIndex(MultiVectorConfig{Dim: 4, Seed: 1})
	defer m.Close()
	var fakeNow int64 = 2_000_000
	m.now = func() int64 { return fakeNow }

	if _, err := m.AddCASKeyTTL(1, [][]float32{{1, 0, 0, 0}},
		Metadata{"perm": NewString("keep"), "temp": NewString("bye")},
		map[string]int64{"temp": 1000}, CASCond{}); err != nil {
		t.Fatalf("AddCASKeyTTL: %v", err)
	}
	wantDeadline := fakeNow + 1000
	if got := m.keyTTL[1]["temp"]; got != wantDeadline {
		t.Fatalf("m.keyTTL[temp] = %d, want %d (absolute now+ttl)", got, wantDeadline)
	}

	_, pay, _, ok := m.Get(1)
	if !ok {
		t.Fatal("Get(1) ok=false before key expiry")
	}
	if pay["temp"].Str != "bye" || pay["perm"].Str != "keep" {
		t.Fatalf("pre-expiry payload = %v, want both keys", pay)
	}

	fakeNow += 1500
	_, pay, _, ok = m.Get(1)
	if !ok {
		t.Fatal("Get(1) ok=false after key expiry (the DOC must still live)")
	}
	if _, has := pay["temp"]; has {
		t.Fatalf("post-expiry payload still has expired key 'temp': %v", pay)
	}
	if pay["perm"].Str != "keep" {
		t.Fatalf("post-expiry lost permanent key: %v", pay)
	}
}

// TestMVAddNoKeyTTLLeavesMapClear: a plain Add (nil key TTL) leaves no keyTTL
// entry — the zero-overhead path.
func TestMVAddNoKeyTTLLeavesMapClear(t *testing.T) {
	m, _ := NewMultiVectorIndex(MultiVectorConfig{Dim: 4, Seed: 1})
	defer m.Close()
	if err := m.Add(1, [][]float32{{1, 0, 0, 0}}, Metadata{"a": NewInt(1)}); err != nil {
		t.Fatalf("Add: %v", err)
	}
	if _, present := m.keyTTL[1]; present {
		t.Fatalf("m.keyTTL[1] = %v, want absent for a no-key_ttl add", m.keyTTL[1])
	}
}

// TestMVAddKeyTTLReplaceSetsFreshDeadline: re-adding (replace) a doc REPLACES the
// payload AND its per-key deadlines; the new key TTL applies to the fresh payload.
func TestMVAddKeyTTLReplaceSetsFreshDeadline(t *testing.T) {
	m, _ := NewMultiVectorIndex(MultiVectorConfig{Dim: 4, Seed: 1})
	defer m.Close()
	var fakeNow int64 = 5_000_000
	m.now = func() int64 { return fakeNow }

	if _, err := m.AddCASKeyTTL(1, [][]float32{{1, 0, 0, 0}},
		Metadata{"old": NewInt(1)}, map[string]int64{"old": 500}, CASCond{}); err != nil {
		t.Fatalf("add: %v", err)
	}
	if _, err := m.AddCASKeyTTL(1, [][]float32{{0, 1, 0, 0}},
		Metadata{"new": NewInt(2)}, map[string]int64{"new": 700}, CASCond{}); err != nil {
		t.Fatalf("replace: %v", err)
	}
	if _, has := m.keyTTL[1]["old"]; has {
		t.Error("replace did not drop the replaced-away key's deadline")
	}
	if got := m.keyTTL[1]["new"]; got != fakeNow+700 {
		t.Fatalf("replace keyTTL[new] = %d, want %d (fresh now+ttl)", got, fakeNow+700)
	}
}

// TestMVAddKeyTTLWALReplayAbsolute: a per-key payload TTL set AT ADD in a WAL-mode
// MV collection survives a CRASH (reopen = checkpoint + WAL tail, NO flush) as an
// ABSOLUTE deadline — NOT recomputed on replay. Proves the MV WAL thread-not-nil +
// replay-restore closed the gap.
func TestMVAddKeyTTLWALReplayAbsolute(t *testing.T) {
	dir := t.TempDir()
	const base = int64(1_000_000_000_000)

	cs, err := OpenCollectionStore(dir)
	if err != nil {
		t.Fatal(err)
	}
	if err := cs.CreateMultiVector("mv", mvWALConfig()); err != nil {
		t.Fatal(err)
	}
	idx, ok := cs.GetMultiVector("mv")
	if !ok {
		t.Fatal("MV collection missing after create")
	}
	if idx.wal == nil {
		t.Fatal("WAL-mode MV collection has nil wal")
	}
	freezeMVClock(idx, base)

	if _, err := idx.AddCASKeyTTL(1, [][]float32{{1, 0, 0, 0}, {0, 1, 0, 0}},
		Metadata{"perm": NewInt(1), "temp": NewInt(9)},
		map[string]int64{"temp": 1000}, CASCond{}); err != nil {
		t.Fatalf("AddCASKeyTTL: %v", err)
	}
	wantDeadline := base + 1000
	_ = cs.Close()

	cs2, err := OpenCollectionStore(dir)
	if err != nil {
		t.Fatalf("reopen: %v", err)
	}
	defer func() { _ = cs2.Close() }()
	idx2, ok := cs2.GetMultiVector("mv")
	if !ok {
		t.Fatal("MV collection missing after reopen")
	}
	freezeMVClock(idx2, base)

	if got := idx2.keyTTL[1]["temp"]; got != wantDeadline {
		t.Fatalf("replayed keyTTL[temp] = %d, want %d (absolute deadline not preserved / WAL gap open)", got, wantDeadline)
	}
	if _, pay, _, ok := idx2.Get(1); !ok || pay["temp"].Int != 9 {
		t.Fatalf("post-replay pre-deadline payload=%v ok=%v, want temp:9", pay, ok)
	}
	freezeMVClock(idx2, base+2000)
	_, pay, _, ok := idx2.Get(1)
	if !ok {
		t.Fatal("doc disappeared after aging (MV has no doc TTL)")
	}
	if _, has := pay["temp"]; has {
		t.Errorf("post-replay post-deadline key still present (replay must NOT recompute now+ttl): %v", pay)
	}
	if pay["perm"].Int != 1 {
		t.Errorf("post-replay lost permanent key: %v", pay)
	}
}

// TestMVAddKeyTTLSnapshotAbsolute: the add-time absolute key deadline survives a
// snapshot round-trip VERBATIM (time-stable).
func TestMVAddKeyTTLSnapshotAbsolute(t *testing.T) {
	m, _ := NewMultiVectorIndex(MultiVectorConfig{Dim: 4, Seed: 1})
	defer m.Close()
	var fakeNow int64 = 3_000_000
	m.now = func() int64 { return fakeNow }
	if _, err := m.AddCASKeyTTL(1, [][]float32{{1, 0, 0, 0}},
		Metadata{"perm": NewInt(1), "temp": NewInt(9)},
		map[string]int64{"temp": 1000}, CASCond{}); err != nil {
		t.Fatalf("add: %v", err)
	}
	wantDeadline := fakeNow + 1000

	var buf bytes.Buffer
	if err := m.snapshot(&buf); err != nil {
		t.Fatalf("snapshot: %v", err)
	}
	m2, _ := NewMultiVectorIndex(MultiVectorConfig{Dim: 4, Seed: 1})
	defer m2.Close()
	if err := m2.restore(bytes.NewReader(buf.Bytes())); err != nil {
		t.Fatalf("restore: %v", err)
	}
	if got := m2.keyTTL[1]["temp"]; got != wantDeadline {
		t.Fatalf("restored keyTTL[temp] = %d, want %d (snapshot not time-stable)", got, wantDeadline)
	}
	m2.now = func() int64 { return wantDeadline + 1 }
	_, pay, _, ok := m2.Get(1)
	if !ok {
		t.Fatal("restored doc absent")
	}
	if _, has := pay["temp"]; has {
		t.Errorf("post-restore post-deadline key still present: %v", pay)
	}
}
