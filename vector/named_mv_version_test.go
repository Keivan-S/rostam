// SPDX-License-Identifier: Apache-2.0

package vector

import (
	"bytes"
	"errors"
	"testing"
)

// Per-point version + optimistic CAS for the NAMED + MV families.
// Mirrors the dense version/CAS tests: lifecycle (insert=1, bumps, delete drops,
// reinsert=1), CAS match/mismatch, insert-if-absent (expected=0), Get returns the
// version, and persistence (snapshot/restore + the family WAL replay) preserving
// the version VERBATIM. Plus the MV maps-blob backward-compat probe.

func newTestNamed(t *testing.T) *NamedCollection {
	t.Helper()
	nc, err := NewNamedCollection("v", map[string]NamedVectorParams{
		"a": {Dim: 3, Metric: Cosine},
	})
	if err != nil {
		t.Fatalf("NewNamedCollection: %v", err)
	}
	return nc
}

func namedVecs() map[string][]float32 { return map[string][]float32{"a": {1, 0, 0}} }

func TestNamedVersionLifecycle(t *testing.T) {
	nc := newTestNamed(t)

	// Fresh insert → version 1.
	v, err := nc.InsertCAS(1, namedVecs(), Metadata{"k": NewString("x")}, 0, CASCond{})
	if err != nil || v != 1 {
		t.Fatalf("insert: v=%d err=%v (want 1, nil)", v, err)
	}
	if _, _, _, gv, ok := nc.Get(1); !ok || gv != 1 {
		t.Fatalf("Get after insert: ok=%v version=%d (want true, 1)", ok, gv)
	}

	// In-place upsert → +1.
	v, err = nc.InsertCAS(1, namedVecs(), nil, 0, CASCond{})
	if err != nil || v != 2 {
		t.Fatalf("upsert: v=%d err=%v (want 2, nil)", v, err)
	}

	// A payload op bumps.
	pv, err := nc.SetPayloadCAS(1, Metadata{"k2": NewInt(5)}, nil, CASCond{})
	if err != nil || pv != 3 {
		t.Fatalf("set_payload: v=%d err=%v (want 3, nil)", pv, err)
	}
	if _, _, _, gv, _ := nc.Get(1); gv != 3 {
		t.Fatalf("Get version after set_payload=%d (want 3)", gv)
	}

	// Delete drops the version (absent reads 0); reinsert restarts at 1.
	if removed, _, err := nc.DeleteCAS(1, CASCond{}); err != nil || !removed {
		t.Fatalf("delete: removed=%v err=%v", removed, err)
	}
	v, err = nc.InsertCAS(1, namedVecs(), nil, 0, CASCond{})
	if err != nil || v != 1 {
		t.Fatalf("reinsert: v=%d err=%v (want 1, nil)", v, err)
	}
}

func TestNamedCASMatchAndMismatch(t *testing.T) {
	nc := newTestNamed(t)
	if _, err := nc.InsertCAS(1, namedVecs(), nil, 0, CASCond{}); err != nil {
		t.Fatalf("seed: %v", err)
	} // version 1

	// CAS match: expected=1 applies + bumps → 2.
	v, err := nc.SetPayloadCAS(1, Metadata{"k": NewInt(1)}, nil, CASCond{Expected: 1, Has: true})
	if err != nil || v != 2 {
		t.Fatalf("CAS match: v=%d err=%v (want 2, nil)", v, err)
	}

	// CAS mismatch: expected=1 (current=2) → ErrVersionConflict, NO mutation.
	_, err = nc.SetPayloadCAS(1, Metadata{"k": NewInt(99)}, nil, CASCond{Expected: 1, Has: true})
	if !errors.Is(err, ErrVersionConflict) {
		t.Fatalf("CAS mismatch: err=%v (want ErrVersionConflict)", err)
	}
	// No mutation: version unchanged (still 2) and payload not 99.
	_, payload, _, gv, _ := nc.Get(1)
	if gv != 2 {
		t.Fatalf("version after conflict=%d (want 2 — no bump)", gv)
	}
	if payload["k"].Int != 1 {
		t.Fatalf("payload mutated on conflict: %+v", payload)
	}

	// Insert-if-absent CAS (expected=0+Has) succeeds for an absent id.
	v, err = nc.InsertCAS(7, namedVecs(), nil, 0, CASCond{Expected: 0, Has: true})
	if err != nil || v != 1 {
		t.Fatalf("insert-if-absent absent: v=%d err=%v (want 1, nil)", v, err)
	}
	// Insert-if-absent CAS on a PRESENT id conflicts (current=1 != 0).
	_, err = nc.InsertCAS(7, namedVecs(), nil, 0, CASCond{Expected: 0, Has: true})
	if !errors.Is(err, ErrVersionConflict) {
		t.Fatalf("insert-if-absent present: err=%v (want ErrVersionConflict)", err)
	}

	// CAS delete mismatch → conflict, no removal.
	if removed, _, derr := nc.DeleteCAS(1, CASCond{Expected: 99, Has: true}); !errors.Is(derr, ErrVersionConflict) || removed {
		t.Fatalf("CAS delete mismatch: removed=%v err=%v", removed, derr)
	}
	if _, _, _, _, ok := nc.Get(1); !ok {
		t.Fatalf("point removed despite CAS-delete conflict")
	}
}

func TestNamedSnapshotPreservesVersion(t *testing.T) {
	nc := newTestNamed(t)
	_, _ = nc.InsertCAS(1, namedVecs(), Metadata{"k": NewInt(1)}, 0, CASCond{})
	_, _ = nc.SetPayloadCAS(1, Metadata{"k2": NewInt(2)}, nil, CASCond{}) // version 2
	_, _ = nc.InsertCAS(2, namedVecs(), nil, 0, CASCond{})                // version 1

	var buf bytes.Buffer
	if err := nc.Snapshot(&buf); err != nil {
		t.Fatalf("Snapshot: %v", err)
	}
	restored := newTestNamed(t)
	if err := restored.Restore(bytes.NewReader(buf.Bytes())); err != nil {
		t.Fatalf("Restore: %v", err)
	}
	if _, _, _, v1, ok := restored.Get(1); !ok || v1 != 2 {
		t.Fatalf("restored id1 version=%d ok=%v (want 2, true — verbatim)", v1, ok)
	}
	if _, _, _, v2, ok := restored.Get(2); !ok || v2 != 1 {
		t.Fatalf("restored id2 version=%d ok=%v (want 1, true)", v2, ok)
	}
}

func TestNamedV2SnapshotDefaultsVersion(t *testing.T) {
	// A v2 (pre-version-block) snapshot must restore live points with version 1.
	nc := newTestNamed(t)
	_, _ = nc.InsertCAS(1, namedVecs(), nil, 0, CASCond{})

	var buf bytes.Buffer
	if err := nc.Snapshot(&buf); err != nil {
		t.Fatalf("Snapshot: %v", err)
	}
	// Force the snapshot version byte back to 2 (the magic is 4 bytes, then version).
	blob := buf.Bytes()
	blob[len(namedSnapshotMagic)] = 2 // pretend it is a v2 blob (the version block is ignored on read)

	restored := newTestNamed(t)
	if err := restored.Restore(bytes.NewReader(blob)); err != nil {
		t.Fatalf("Restore v2: %v", err)
	}
	if _, _, _, v, ok := restored.Get(1); !ok || v != 1 {
		t.Fatalf("v2 restore version=%d ok=%v (want 1, true — sane default)", v, ok)
	}
}

func TestNamedWALReplayPreservesVersion(t *testing.T) {
	dir := t.TempDir()
	w, err := openWAL(dir+"/n.wal", false)
	if err != nil {
		t.Fatalf("openWAL: %v", err)
	}
	nc := newTestNamed(t)
	nc.wal = w
	_, _ = nc.InsertCAS(1, namedVecs(), Metadata{"k": NewInt(1)}, 0, CASCond{}) // v1
	_, _ = nc.SetPayloadCAS(1, Metadata{"k": NewInt(9)}, nil, CASCond{})        // v2
	_, _ = nc.InsertCAS(1, namedVecs(), nil, 0, CASCond{})                      // v3
	_ = w.close()

	// Replay onto a fresh collection (no snapshot) — version must come back VERBATIM.
	nc2 := newTestNamed(t)
	if err := replayNamedWAL(dir+"/n.wal", nc2); err != nil {
		t.Fatalf("replayNamedWAL: %v", err)
	}
	if _, _, _, v, ok := nc2.Get(1); !ok || v != 3 {
		t.Fatalf("replayed version=%d ok=%v (want 3, true — verbatim)", v, ok)
	}
}

// ---- MV ----

func newTestMV(t *testing.T) *MultiVectorIndex {
	t.Helper()
	m, err := NewMultiVectorIndex(MultiVectorConfig{Dim: 3})
	if err != nil {
		t.Fatalf("NewMultiVectorIndex: %v", err)
	}
	return m
}

func mvTokensV() [][]float32 { return [][]float32{{1, 0, 0}, {0, 1, 0}} }

func TestMVVersionLifecycle(t *testing.T) {
	m := newTestMV(t)

	v, err := m.AddCAS(1, mvTokensV(), Metadata{"k": NewInt(1)}, CASCond{})
	if err != nil || v != 1 {
		t.Fatalf("add: v=%d err=%v (want 1, nil)", v, err)
	}
	if _, _, gv, ok := m.Get(1); !ok || gv != 1 {
		t.Fatalf("Get after add: ok=%v version=%d (want true, 1)", ok, gv)
	}

	// In-place replace-add → +1 (NOT reset to 1).
	v, err = m.AddCAS(1, mvTokensV(), nil, CASCond{})
	if err != nil || v != 2 {
		t.Fatalf("replace-add: v=%d err=%v (want 2, nil)", v, err)
	}
	// Payload op bumps.
	pv, err := m.SetPayloadCAS(1, Metadata{"k2": NewInt(7)}, nil, CASCond{})
	if err != nil || pv != 3 {
		t.Fatalf("set_payload: v=%d err=%v (want 3, nil)", pv, err)
	}

	// Delete drops; re-add restarts at 1.
	if removed, _, err := m.DeleteCAS(1, CASCond{}); err != nil || !removed {
		t.Fatalf("delete: removed=%v err=%v", removed, err)
	}
	v, err = m.AddCAS(1, mvTokensV(), nil, CASCond{})
	if err != nil || v != 1 {
		t.Fatalf("re-add: v=%d err=%v (want 1, nil)", v, err)
	}

	// AddIfAbsent on an absent doc → version 1.
	ins, err := m.AddIfAbsent(2, mvTokensV(), nil)
	if err != nil || !ins {
		t.Fatalf("add-if-absent: inserted=%v err=%v", ins, err)
	}
	if _, _, gv, _ := m.Get(2); gv != 1 {
		t.Fatalf("add-if-absent version=%d (want 1)", gv)
	}
}

func TestMVCASMatchAndMismatch(t *testing.T) {
	m := newTestMV(t)
	if _, err := m.AddCAS(1, mvTokensV(), nil, CASCond{}); err != nil {
		t.Fatalf("seed: %v", err)
	} // v1

	v, err := m.SetPayloadCAS(1, Metadata{"k": NewInt(1)}, nil, CASCond{Expected: 1, Has: true})
	if err != nil || v != 2 {
		t.Fatalf("CAS match: v=%d err=%v (want 2, nil)", v, err)
	}
	_, err = m.SetPayloadCAS(1, Metadata{"k": NewInt(9)}, nil, CASCond{Expected: 1, Has: true})
	if !errors.Is(err, ErrVersionConflict) {
		t.Fatalf("CAS mismatch: err=%v (want ErrVersionConflict)", err)
	}
	_, payload, gv, _ := m.Get(1)
	if gv != 2 || payload["k"].Int != 1 {
		t.Fatalf("conflict mutated state: version=%d payload=%+v", gv, payload)
	}

	// add-if-absent CAS (expected=0) on a present doc conflicts.
	_, err = m.AddCAS(1, mvTokensV(), nil, CASCond{Expected: 0, Has: true})
	if !errors.Is(err, ErrVersionConflict) {
		t.Fatalf("add-if-absent present: err=%v (want ErrVersionConflict)", err)
	}
}

func TestMVSnapshotPreservesVersion(t *testing.T) {
	m := newTestMV(t)
	_, _ = m.AddCAS(1, mvTokensV(), Metadata{"k": NewInt(1)}, CASCond{})
	_, _ = m.SetPayloadCAS(1, Metadata{"k2": NewInt(2)}, nil, CASCond{}) // v2
	_, _ = m.AddCAS(2, mvTokensV(), nil, CASCond{})                      // v1

	var buf bytes.Buffer
	if err := m.snapshot(&buf); err != nil {
		t.Fatalf("snapshot: %v", err)
	}
	restored := newTestMV(t)
	if err := restored.restore(bytes.NewReader(buf.Bytes())); err != nil {
		t.Fatalf("restore: %v", err)
	}
	if _, _, v1, ok := restored.Get(1); !ok || v1 != 2 {
		t.Fatalf("restored doc1 version=%d ok=%v (want 2, true)", v1, ok)
	}
	if _, _, v2, ok := restored.Get(2); !ok || v2 != 1 {
		t.Fatalf("restored doc2 version=%d ok=%v (want 1, true)", v2, ok)
	}
}

// TestMVDecodeMapsOldBlobBackwardCompat verifies a maps blob with NO version block
// (the format BEFORE this feature) decodes cleanly and defaults live docs to
// version 1 — the careful EOF-probe resequence must not choke on the absent block.
func TestMVDecodeMapsOldBlobBackwardCompat(t *testing.T) {
	// Build a current-format maps blob, then TRUNCATE the trailing version block to
	// simulate an old blob. encodeMaps writes: magic + body + keyTTL-marker(0) +
	// version-marker(0). An "old" blob (no version block) ends right after the
	// keyTTL marker. We reproduce that by re-encoding with a writer that drops the
	// final version-marker byte.
	m := newTestMV(t)
	_, _ = m.AddCAS(1, mvTokensV(), Metadata{"k": NewInt(1)}, CASCond{})
	_, _ = m.AddCAS(2, mvTokensV(), nil, CASCond{})

	var full bytes.Buffer
	if err := m.encodeMaps(&full); err != nil {
		t.Fatalf("encodeMaps: %v", err)
	}
	blob := full.Bytes()
	// The version block here is a single marker byte (0 = no versions) at the tail
	// (no docs carry a non-zero... they DO: version 1). So drop the whole version
	// block. Simplest robust approach: re-decode a blob that simply lacks it by
	// re-encoding WITHOUT versions — emulate by zeroing then truncating. Instead,
	// build an old blob directly: take the bytes up to BEFORE the version block by
	// decoding the current blob and re-serializing the legacy way is complex, so we
	// assert the strong invariant via the truncation that drops the trailing
	// version block (marker 1 + count + entries). We locate it by decoding twice.

	// Decode the full blob into a fresh index (sanity: versions preserved).
	chk := newTestMV(t)
	if err := chk.decodeMaps(bytes.NewReader(blob[len(mvMapsMagic):])); err != nil {
		t.Fatalf("decodeMaps full: %v", err)
	}
	if _, _, v, _ := chk.Get(1); v != 1 {
		t.Fatalf("full-blob version=%d (want 1)", v)
	}

	// Now emulate an OLD blob: a maps body that ends right after the keyTTL marker
	// (no version block). Re-encode WITHOUT the version tail.
	old := encodeMapsLegacyNoVersion(t, m)
	restored := newTestMV(t)
	if err := restored.decodeMaps(bytes.NewReader(old)); err != nil {
		t.Fatalf("decodeMaps old blob: %v", err)
	}
	if _, _, v1, ok := restored.Get(1); !ok || v1 != 1 {
		t.Fatalf("old-blob doc1 version=%d ok=%v (want 1, true — defaulted)", v1, ok)
	}
	if _, _, v2, ok := restored.Get(2); !ok || v2 != 1 {
		t.Fatalf("old-blob doc2 version=%d ok=%v (want 1, true — defaulted)", v2, ok)
	}
}

// encodeMapsLegacyNoVersion writes the maps body in the PRE-version-block format:
// the body (nextToken + docs) followed by the keyTTL present marker (0), and NO
// version block — exactly what an old binary produced. Returns the body WITHOUT
// the magic (decodeMaps reads after the magic).
func encodeMapsLegacyNoVersion(t *testing.T, m *MultiVectorIndex) []byte {
	t.Helper()
	var buf bytes.Buffer
	// Reuse encodeMaps then strip the trailing version block. encodeMaps with no
	// per-key TTL writes: magic, nextToken, numDocs, [docs...], keyTTL-marker(0),
	// version-marker(...). The version block is the final bytes; for our fixture no
	// doc has a non-zero keyTTL, so keyTTL-marker is the single 0 byte, and the
	// version block (marker 1 + count + per-doc) follows. We rebuild WITHOUT it by
	// decoding the full blob to find the keyTTL marker offset, but simplest: write
	// our own legacy body.
	if err := m.encodeMaps(&buf); err != nil {
		t.Fatalf("encodeMaps: %v", err)
	}
	body := buf.Bytes()[len(mvMapsMagic):]
	// Walk the body to the end of the keyTTL block, dropping everything after it
	// (the version block). For a fixture with NO per-key TTL the keyTTL block is a
	// single 0 marker byte; the version block follows. Find it by re-decoding into a
	// throwaway and recording how many bytes the keyTTL block consumed is intricate,
	// so instead: since our fixture has no keyTTL, the layout is deterministic — the
	// keyTTL marker is one 0 byte placed right before the version block. We search
	// for the LAST occurrence of the version block (marker 1) from the end and cut.
	// Robust alternative: the version block is [1][u32 count][count*(u64+u64)].
	// count == number of live docs (2). Trailing length = 1 + 4 + count*16.
	live := len(m.docTokens)
	trailing := 1 + 4 + live*16
	if trailing >= len(body) {
		t.Fatalf("fixture too small to trim version block")
	}
	return append([]byte(nil), body[:len(body)-trailing]...)
}
