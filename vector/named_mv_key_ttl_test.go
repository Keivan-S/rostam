// SPDX-License-Identifier: Apache-2.0

package vector

import (
	"bytes"
	"testing"
)

// Per-key payload TTL for the named + MV families (lazy-drop + snapshot). Mirrors
// the dense key_ttl_test, but named/MV store payload in plain maps with NO payload
// index and NO WAL — so the expiry is a parallel deadline map dropped lazily on
// every read (Get / scroll / the filter predicate view) and persisted in the
// SNAPSHOT only. The injectable clock (nc.now / m.now) ages deterministically; no
// sleeps.

// --- named ---

// TestNamedKeyTTLGetDropsExpiredKey: a key with a per-key TTL is returned on Get
// before expiry and DROPPED after, while the point + a permanent key survive.
func TestNamedKeyTTLGetDropsExpiredKey(t *testing.T) {
	nc, err := NewNamedCollection("default/named", namedTestConfig())
	if err != nil {
		t.Fatalf("new: %v", err)
	}
	defer nc.Close()
	var fakeNow int64 = 1_000_000
	nc.now = func() int64 { return fakeNow }

	if err := nc.Insert(1, map[string][]float32{"title": {1, 0, 0, 0}}, Metadata{"perm": NewString("p")}, 0); err != nil {
		t.Fatalf("insert: %v", err)
	}
	// Set an expiring key (50ms) + a permanent key via set_payload.
	if err := nc.SetPayload(1, Metadata{"temp": NewString("hot"), "keep": NewString("k")}, map[string]int64{"temp": 50}); err != nil {
		t.Fatalf("set_payload: %v", err)
	}

	// Before expiry: all three keys present.
	_, pay, _, _, ok := nc.Get(1)
	if !ok {
		t.Fatal("point absent before expiry")
	}
	for _, k := range []string{"perm", "temp", "keep"} {
		if _, has := pay[k]; !has {
			t.Errorf("pre-expiry payload missing key %q: %v", k, pay)
		}
	}

	// Advance PAST the temp deadline.
	fakeNow += 100
	_, pay, _, _, ok = nc.Get(1)
	if !ok {
		t.Fatal("point dropped after key expiry (point must survive)")
	}
	if _, has := pay["temp"]; has {
		t.Errorf("expired key temp still returned: %v", pay)
	}
	for _, k := range []string{"perm", "keep"} {
		if _, has := pay[k]; !has {
			t.Errorf("non-expired key %q dropped: %v", k, pay)
		}
	}
}

// TestNamedKeyTTLFilterNoMatch: a filter on an expired key returns no match (the
// predicate must see the TTL-filtered metadata view), while a filter on a
// permanent key still matches.
func TestNamedKeyTTLFilterNoMatch(t *testing.T) {
	nc, err := NewNamedCollection("default/named", namedTestConfig())
	if err != nil {
		t.Fatalf("new: %v", err)
	}
	defer nc.Close()
	var fakeNow int64 = 1_000_000
	nc.now = func() int64 { return fakeNow }

	if err := nc.Insert(1, map[string][]float32{"title": {1, 0, 0, 0}}, nil, 0); err != nil {
		t.Fatalf("insert: %v", err)
	}
	if err := nc.SetPayload(1, Metadata{"temp": NewString("hot"), "perm": NewString("p")}, map[string]int64{"temp": 50}); err != nil {
		t.Fatalf("set_payload: %v", err)
	}

	tempFilter := Filter{Op: FilterEq, Field: "temp", Value: NewString("hot")}
	permFilter := Filter{Op: FilterEq, Field: "perm", Value: NewString("p")}

	// Before expiry: both filters match via scroll AND search.
	if docs, _ := nc.ScrollDocs(tempFilter, 0); len(docs) != 1 {
		t.Fatalf("pre-expiry scroll on temp = %d, want 1", len(docs))
	}
	if res, _ := nc.SearchNamed("title", []float32{1, 0, 0, 0}, 5, tempFilter); len(res) != 1 {
		t.Fatalf("pre-expiry search on temp = %d, want 1", len(res))
	}

	// Advance past temp's deadline.
	fakeNow += 100
	if docs, _ := nc.ScrollDocs(tempFilter, 0); len(docs) != 0 {
		t.Errorf("post-expiry scroll on expired key matched %d, want 0", len(docs))
	}
	if res, _ := nc.SearchNamed("title", []float32{1, 0, 0, 0}, 5, tempFilter); len(res) != 0 {
		t.Errorf("post-expiry search on expired key matched %d, want 0", len(res))
	}
	// The permanent key still matches.
	if docs, _ := nc.ScrollDocs(permFilter, 0); len(docs) != 1 {
		t.Errorf("post-expiry scroll on perm key = %d, want 1", len(docs))
	}
}

// TestNamedKeyTTLOverwriteDeleteClear: overwrite REPLACES the deadline set;
// delete drops the deleted key's deadline; clear drops all deadlines.
func TestNamedKeyTTLOverwriteDeleteClear(t *testing.T) {
	nc, err := NewNamedCollection("default/named", namedTestConfig())
	if err != nil {
		t.Fatalf("new: %v", err)
	}
	defer nc.Close()
	var fakeNow int64 = 1_000_000
	nc.now = func() int64 { return fakeNow }

	if err := nc.Insert(1, map[string][]float32{"title": {1, 0, 0, 0}}, nil, 0); err != nil {
		t.Fatalf("insert: %v", err)
	}

	// Overwrite sets a fresh deadline set; a prior deadline on a now-absent key is gone.
	if err := nc.SetPayload(1, Metadata{"a": NewInt(1)}, map[string]int64{"a": 50}); err != nil {
		t.Fatalf("set a: %v", err)
	}
	if err := nc.OverwritePayload(1, Metadata{"b": NewInt(2)}, map[string]int64{"b": 50}); err != nil {
		t.Fatalf("overwrite: %v", err)
	}
	if _, ke := nc.keyTTL[1]["a"]; ke {
		t.Error("overwrite did not drop the replaced-away key's deadline")
	}
	if _, ke := nc.keyTTL[1]["b"]; !ke {
		t.Error("overwrite did not set the new key's deadline")
	}

	// Delete the keyed deadline.
	if err := nc.DeletePayloadKeys(1, []string{"b"}); err != nil {
		t.Fatalf("delete: %v", err)
	}
	if _, present := nc.keyTTL[1]; present {
		t.Error("delete left a stale keyTTL entry for the only deadlined key")
	}

	// Re-set then clear.
	if err := nc.SetPayload(1, Metadata{"c": NewInt(3)}, map[string]int64{"c": 50}); err != nil {
		t.Fatalf("set c: %v", err)
	}
	if err := nc.ClearPayload(1); err != nil {
		t.Fatalf("clear: %v", err)
	}
	if _, present := nc.keyTTL[1]; present {
		t.Error("clear left a stale keyTTL entry")
	}
}

// TestNamedKeyTTLSnapshotRestore: a pending per-key deadline survives a
// snapshot->restore round-trip with its ABSOLUTE value, so post-restore expiry is
// time-stable (the same clock value drops it).
func TestNamedKeyTTLSnapshotRestore(t *testing.T) {
	nc, err := NewNamedCollection("default/named", namedTestConfig())
	if err != nil {
		t.Fatalf("new: %v", err)
	}
	defer nc.Close()
	var fakeNow int64 = 1_000_000
	nc.now = func() int64 { return fakeNow }

	if err := nc.Insert(1, map[string][]float32{"title": {1, 0, 0, 0}}, nil, 0); err != nil {
		t.Fatalf("insert: %v", err)
	}
	if err := nc.SetPayload(1, Metadata{"temp": NewString("hot"), "perm": NewString("p")}, map[string]int64{"temp": 50}); err != nil {
		t.Fatalf("set_payload: %v", err)
	}
	wantDeadline := nc.keyTTL[1]["temp"]
	if wantDeadline != fakeNow+50 {
		t.Fatalf("absolute deadline = %d, want %d", wantDeadline, fakeNow+50)
	}

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
		t.Fatalf("restored absolute deadline = %d, want %d (must be verbatim)", got, wantDeadline)
	}

	// Time-stable: the SAME post-deadline clock drops the key after restore.
	restored.now = func() int64 { return wantDeadline + 1 }
	_, pay, _, _, ok := restored.Get(1)
	if !ok {
		t.Fatal("restored point absent")
	}
	if _, has := pay["temp"]; has {
		t.Errorf("restored expired key still present: %v", pay)
	}
	if _, has := pay["perm"]; !has {
		t.Errorf("restored permanent key dropped: %v", pay)
	}
}

// TestNamedSnapshotBackwardCompatV1 ensures a v1 named snapshot (no keyTTL block)
// restores with an empty keyTTL map (no per-key TTL) — backward compatible.
func TestNamedSnapshotBackwardCompatV1(t *testing.T) {
	nc, err := NewNamedCollection("default/named", namedTestConfig())
	if err != nil {
		t.Fatalf("new: %v", err)
	}
	defer nc.Close()
	if err := nc.Insert(1, map[string][]float32{"title": {1, 0, 0, 0}}, Metadata{"k": NewString("v")}, 0); err != nil {
		t.Fatalf("insert: %v", err)
	}

	// Write a v1-shaped snapshot: capture the v2 blob, then rewrite the version byte
	// to 1 and TRUNCATE the trailing keyTTL block (which a v1 writer never emits).
	// Simplest faithful v1 image: the collection has no per-key TTL, so the v2 tail
	// is just a zero count (u32) — drop those 4 bytes and flip the version byte.
	var buf bytes.Buffer
	if err := nc.Snapshot(&buf); err != nil {
		t.Fatalf("snapshot: %v", err)
	}
	blob := buf.Bytes()
	// A dense-only collection writes the dense-only version (v3) with NO trailing
	// sparse block — byte-identical to the pre-sparse format (the v4 bump only
	// applies to collections that actually have a sparse space).
	if blob[len(namedSnapshotMagic)] != namedSnapshotVersionDenseOnly {
		t.Fatalf("expected version byte %d", namedSnapshotVersionDenseOnly)
	}
	v1 := make([]byte, len(blob)-4) // drop the 4-byte keyTTL zero-count tail
	copy(v1, blob[:len(blob)-4])
	v1[len(namedSnapshotMagic)] = 1 // version 1

	restored, err := NewNamedCollection("default/named", namedTestConfig())
	if err != nil {
		t.Fatalf("new restored: %v", err)
	}
	defer restored.Close()
	if err := restored.Restore(bytes.NewReader(v1)); err != nil {
		t.Fatalf("restore v1: %v", err)
	}
	if len(restored.keyTTL) != 0 {
		t.Errorf("v1 restore produced non-empty keyTTL: %v", restored.keyTTL)
	}
	if _, _, _, _, ok := restored.Get(1); !ok {
		t.Error("v1 restore lost the point")
	}
}

// --- MV ---

// TestMVKeyTTLGetDropsExpiredKey: MV Get drops an expired key after the deadline
// while the doc + permanent keys survive.
func TestMVKeyTTLGetDropsExpiredKey(t *testing.T) {
	m, _ := NewMultiVectorIndex(MultiVectorConfig{Dim: 4, Seed: 1})
	defer m.Close()
	var fakeNow int64 = 1_000_000
	m.now = func() int64 { return fakeNow }

	if err := m.Add(1, [][]float32{{1, 0, 0, 0}, {0, 1, 0, 0}}, Metadata{"perm": NewString("p")}); err != nil {
		t.Fatalf("add: %v", err)
	}
	if err := m.SetPayload(1, Metadata{"temp": NewString("hot"), "keep": NewString("k")}, map[string]int64{"temp": 50}); err != nil {
		t.Fatalf("set_payload: %v", err)
	}

	_, pay, _, ok := m.Get(1)
	if !ok {
		t.Fatal("doc absent before expiry")
	}
	for _, k := range []string{"perm", "temp", "keep"} {
		if _, has := pay[k]; !has {
			t.Errorf("pre-expiry payload missing %q: %v", k, pay)
		}
	}

	fakeNow += 100
	_, pay, _, ok = m.Get(1)
	if !ok {
		t.Fatal("doc dropped after key expiry (doc must survive)")
	}
	if _, has := pay["temp"]; has {
		t.Errorf("expired key temp still returned: %v", pay)
	}
	for _, k := range []string{"perm", "keep"} {
		if _, has := pay[k]; !has {
			t.Errorf("non-expired key %q dropped: %v", k, pay)
		}
	}
}

// TestMVKeyTTLFilterNoMatch: an MV search/scroll filter on an expired key returns
// no match (the Stage-2 predicate must see the TTL-filtered view).
func TestMVKeyTTLFilterNoMatch(t *testing.T) {
	m, _ := NewMultiVectorIndex(MultiVectorConfig{Dim: 4, Seed: 1})
	defer m.Close()
	var fakeNow int64 = 1_000_000
	m.now = func() int64 { return fakeNow }

	if err := m.Add(1, [][]float32{{1, 0, 0, 0}, {0, 1, 0, 0}}, nil); err != nil {
		t.Fatalf("add: %v", err)
	}
	if err := m.SetPayload(1, Metadata{"temp": NewString("hot"), "perm": NewString("p")}, map[string]int64{"temp": 50}); err != nil {
		t.Fatalf("set_payload: %v", err)
	}

	tempFilter := Filter{Op: FilterEq, Field: "temp", Value: NewString("hot")}
	permFilter := Filter{Op: FilterEq, Field: "perm", Value: NewString("p")}
	query := [][]float32{{1, 0, 0, 0}}

	if res, _ := m.Search(query, 5, MultiSearchOpts{Filter: tempFilter}); len(res) != 1 {
		t.Fatalf("pre-expiry search on temp = %d, want 1", len(res))
	}
	if docs, _, _, _ := m.ScrollDocsPage(tempFilter, 0, false, 0); len(docs) != 1 {
		t.Fatalf("pre-expiry scroll on temp = %d, want 1", len(docs))
	}

	fakeNow += 100
	if res, _ := m.Search(query, 5, MultiSearchOpts{Filter: tempFilter}); len(res) != 0 {
		t.Errorf("post-expiry search on expired key matched %d, want 0", len(res))
	}
	if docs, _, _, _ := m.ScrollDocsPage(tempFilter, 0, false, 0); len(docs) != 0 {
		t.Errorf("post-expiry scroll on expired key matched %d, want 0", len(docs))
	}
	if res, _ := m.Search(query, 5, MultiSearchOpts{Filter: permFilter}); len(res) != 1 {
		t.Errorf("post-expiry search on perm key = %d, want 1", len(res))
	}
}

// TestMVKeyTTLOverwriteDeleteClear mirrors the named overwrite/delete/clear
// deadline handling for the MV family.
func TestMVKeyTTLOverwriteDeleteClear(t *testing.T) {
	m, _ := NewMultiVectorIndex(MultiVectorConfig{Dim: 4, Seed: 1})
	defer m.Close()
	m.now = func() int64 { return 1_000_000 }

	if err := m.Add(1, [][]float32{{1, 0, 0, 0}}, nil); err != nil {
		t.Fatalf("add: %v", err)
	}
	if err := m.SetPayload(1, Metadata{"a": NewInt(1)}, map[string]int64{"a": 50}); err != nil {
		t.Fatalf("set a: %v", err)
	}
	if err := m.OverwritePayload(1, Metadata{"b": NewInt(2)}, map[string]int64{"b": 50}); err != nil {
		t.Fatalf("overwrite: %v", err)
	}
	if _, ke := m.keyTTL[1]["a"]; ke {
		t.Error("overwrite did not drop the replaced-away key's deadline")
	}
	if _, ke := m.keyTTL[1]["b"]; !ke {
		t.Error("overwrite did not set the new key's deadline")
	}
	if err := m.DeletePayloadKeys(1, []string{"b"}); err != nil {
		t.Fatalf("delete: %v", err)
	}
	if _, present := m.keyTTL[1]; present {
		t.Error("delete left a stale keyTTL entry")
	}
	if err := m.SetPayload(1, Metadata{"c": NewInt(3)}, map[string]int64{"c": 50}); err != nil {
		t.Fatalf("set c: %v", err)
	}
	if err := m.ClearPayload(1); err != nil {
		t.Fatalf("clear: %v", err)
	}
	if _, present := m.keyTTL[1]; present {
		t.Error("clear left a stale keyTTL entry")
	}
}

// TestMVKeyTTLSnapshotRestore: a pending MV per-key deadline survives the
// snapshot->restore round-trip verbatim (absolute, time-stable).
func TestMVKeyTTLSnapshotRestore(t *testing.T) {
	m, _ := NewMultiVectorIndex(MultiVectorConfig{Dim: 4, Seed: 1})
	defer m.Close()
	var fakeNow int64 = 1_000_000
	m.now = func() int64 { return fakeNow }

	if err := m.Add(1, [][]float32{{1, 0, 0, 0}}, nil); err != nil {
		t.Fatalf("add: %v", err)
	}
	if err := m.SetPayload(1, Metadata{"temp": NewString("hot"), "perm": NewString("p")}, map[string]int64{"temp": 50}); err != nil {
		t.Fatalf("set_payload: %v", err)
	}
	wantDeadline := m.keyTTL[1]["temp"]
	if wantDeadline != fakeNow+50 {
		t.Fatalf("absolute deadline = %d, want %d", wantDeadline, fakeNow+50)
	}

	var buf bytes.Buffer
	if err := m.snapshot(&buf); err != nil {
		t.Fatalf("snapshot: %v", err)
	}
	restored, _ := NewMultiVectorIndex(MultiVectorConfig{Dim: 4, Seed: 1})
	defer restored.Close()
	if err := restored.restore(bytes.NewReader(buf.Bytes())); err != nil {
		t.Fatalf("restore: %v", err)
	}
	if got := restored.keyTTL[1]["temp"]; got != wantDeadline {
		t.Fatalf("restored deadline = %d, want %d (verbatim)", got, wantDeadline)
	}

	restored.now = func() int64 { return wantDeadline + 1 }
	_, pay, _, ok := restored.Get(1)
	if !ok {
		t.Fatal("restored doc absent")
	}
	if _, has := pay["temp"]; has {
		t.Errorf("restored expired key still present: %v", pay)
	}
	if _, has := pay["perm"]; !has {
		t.Errorf("restored permanent key dropped: %v", pay)
	}
}

// TestMVSnapshotBackwardCompatOldBlob ensures an MV maps blob WITHOUT the trailing
// per-key TTL marker (an old-format blob) decodes with an empty keyTTL map. The old
// format ended right after the doc loop; decodeMaps probes for the marker and a
// clean EOF means "old blob, no per-key TTL".
func TestMVSnapshotBackwardCompatOldBlob(t *testing.T) {
	m, _ := NewMultiVectorIndex(MultiVectorConfig{Dim: 4, Seed: 1})
	defer m.Close()
	if err := m.Add(1, [][]float32{{1, 0, 0, 0}}, Metadata{"k": NewString("v")}); err != nil {
		t.Fatalf("add: %v", err)
	}

	// A no-per-key-TTL collection emits a keyTTL marker(0) byte, then the per-doc
	// CAS version block. To forge a genuine OLD-format blob (one that ended right
	// after the doc loop, before BOTH the keyTTL marker AND the version block) we
	// re-snapshot and drop the whole trailing region: the keyTTL marker (1 byte) +
	// the version block ([marker:1][count:u32][per-doc docID:u64+version:u64]).
	var buf bytes.Buffer
	if err := m.snapshot(&buf); err != nil {
		t.Fatalf("snapshot: %v", err)
	}
	blob := buf.Bytes()
	live := len(m.docTokens)
	tail := 1 /*keyTTL marker(0)*/ + 1 /*version marker(1)*/ + 4 /*count*/ + live*16 /*docID+version*/
	old := blob[:len(blob)-tail]                                                     // forge a pre-keyTTL, pre-version blob

	restored, _ := NewMultiVectorIndex(MultiVectorConfig{Dim: 4, Seed: 1})
	defer restored.Close()
	if err := restored.restore(bytes.NewReader(old)); err != nil {
		t.Fatalf("restore old blob: %v", err)
	}
	if len(restored.keyTTL) != 0 {
		t.Errorf("old-blob restore produced non-empty keyTTL: %v", restored.keyTTL)
	}
	if _, _, _, ok := restored.Get(1); !ok {
		t.Error("old-blob restore lost the doc")
	}
}
