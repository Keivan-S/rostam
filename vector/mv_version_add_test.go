// SPDX-License-Identifier: Apache-2.0

package vector

import (
	"testing"
)

// TestMVAddIfAbsentVersion_RealAddVerbatim confirms MultiAddIfAbsentVersion sets a
// REAL add's per-document version VERBATIM (not 1), and that re-adding an existing
// doc is a no-op (version unchanged, inserted=false). version==0 falls back to the
// plain if-absent (fresh add → version 1).
func TestMVAddIfAbsentVersion_RealAddVerbatim(t *testing.T) {
	m, err := NewMultiVectorIndex(MultiVectorConfig{Dim: 4, Seed: 1})
	if err != nil {
		t.Fatal(err)
	}
	// Real add with a verbatim version > 1.
	inserted, err := m.MultiAddIfAbsentVersion(1, [][]float32{{1, 0, 0, 0}}, Metadata{"k": NewString("a")}, nil, 7)
	if err != nil || !inserted {
		t.Fatalf("first add: inserted=%v err=%v, want inserted=true nil", inserted, err)
	}
	if _, _, v, ok := m.Get(1); !ok || v != 7 {
		t.Fatalf("doc 1 version = %d (ok=%v), want 7 VERBATIM (not 1)", v, ok)
	}

	// Doc already exists ⇒ no-op, version unchanged.
	inserted, err = m.MultiAddIfAbsentVersion(1, [][]float32{{0, 1, 0, 0}}, nil, nil, 99)
	if err != nil {
		t.Fatalf("second add err: %v", err)
	}
	if inserted {
		t.Fatal("second add reported inserted=true on a live doc (must be a no-op)")
	}
	if _, _, v, _ := m.Get(1); v != 7 {
		t.Fatalf("version after no-op = %d, want 7 unchanged", v)
	}

	// version==0 ⇒ plain if-absent (fresh add → version 1).
	inserted, err = m.MultiAddIfAbsentVersion(2, [][]float32{{0, 0, 1, 0}}, nil, nil, 0)
	if err != nil || !inserted {
		t.Fatalf("v0 add: inserted=%v err=%v", inserted, err)
	}
	if _, _, v, _ := m.Get(2); v != 1 {
		t.Fatalf("v0 fresh add version = %d, want 1", v)
	}
}

// TestMVRestoreAdd_VerbatimReplace confirms the verbatim (non-if-absent) versioned
// add sets the version verbatim, even when the doc already exists (a replace), and
// falls back to the normal bump for version==0.
func TestMVRestoreAdd_VerbatimReplace(t *testing.T) {
	m, err := NewMultiVectorIndex(MultiVectorConfig{Dim: 4, Seed: 1})
	if err != nil {
		t.Fatal(err)
	}
	if err := m.MultiRestoreAdd(5, [][]float32{{1, 0, 0, 0}}, nil, nil, 42); err != nil {
		t.Fatalf("restore add: %v", err)
	}
	if _, _, v, ok := m.Get(5); !ok || v != 42 {
		t.Fatalf("doc 5 version = %d (ok=%v), want 42 VERBATIM", v, ok)
	}
	// Replace the SAME doc with a different verbatim version (not bumped to 43).
	if err := m.MultiRestoreAdd(5, [][]float32{{0, 1, 0, 0}}, nil, nil, 100); err != nil {
		t.Fatalf("restore replace: %v", err)
	}
	if _, _, v, _ := m.Get(5); v != 100 {
		t.Fatalf("replace version = %d, want 100 VERBATIM (not bumped)", v)
	}
	// version==0 on a fresh doc ⇒ normal bump (→ 1).
	if err := m.MultiRestoreAdd(6, [][]float32{{0, 0, 1, 0}}, nil, nil, 0); err != nil {
		t.Fatalf("restore v0: %v", err)
	}
	if _, _, v, _ := m.Get(6); v != 1 {
		t.Fatalf("v0 restore version = %d, want 1 (bumped)", v)
	}
}

// TestMVAddIfAbsentVersion_WALVerbatim confirms a versioned if-absent add WAL-logs
// the VERBATIM version and replay restores it WITHOUT a re-bump (no double-bump).
func TestMVAddIfAbsentVersion_WALVerbatim(t *testing.T) {
	dir := t.TempDir()
	cs, err := OpenCollectionStore(dir)
	if err != nil {
		t.Fatal(err)
	}
	if err := cs.CreateMultiVector("mv", mvWALConfig()); err != nil {
		t.Fatal(err)
	}
	idx, ok := cs.GetMultiVector("mv")
	if !ok || idx.wal == nil {
		t.Fatalf("MV WAL collection not wired (ok=%v wal=%v)", ok, idx.wal)
	}
	inserted, err := idx.MultiAddIfAbsentVersion(1, [][]float32{{1, 0, 0, 0}}, nil, nil, 9)
	if err != nil || !inserted {
		t.Fatalf("versioned if-absent add: inserted=%v err=%v", inserted, err)
	}
	// Crash (no Flush) — only the WAL is on disk.
	_ = cs.Close()

	cs2, err := OpenCollectionStore(dir)
	if err != nil {
		t.Fatal(err)
	}
	defer func() { _ = cs2.Close() }()
	idx2, ok := cs2.GetMultiVector("mv")
	if !ok {
		t.Fatal("MV collection missing after reopen")
	}
	if _, _, v, live := idx2.Get(1); !live || v != 9 {
		t.Fatalf("replayed doc 1 version = %d (live=%v), want 9 VERBATIM (no double-bump)", v, live)
	}
}

// TestMVRestoreAdd_WALVerbatim confirms the verbatim replace-add WAL-logs the
// verbatim version and replay restores it without a re-bump.
func TestMVRestoreAdd_WALVerbatim(t *testing.T) {
	dir := t.TempDir()
	cs, err := OpenCollectionStore(dir)
	if err != nil {
		t.Fatal(err)
	}
	if err := cs.CreateMultiVector("mv", mvWALConfig()); err != nil {
		t.Fatal(err)
	}
	idx, ok := cs.GetMultiVector("mv")
	if !ok || idx.wal == nil {
		t.Fatalf("MV WAL collection not wired (ok=%v wal=%v)", ok, idx.wal)
	}
	if err := idx.MultiRestoreAdd(3, [][]float32{{0, 0, 1, 0}}, Metadata{"p": NewInt(1)}, nil, 55); err != nil {
		t.Fatalf("restore add: %v", err)
	}
	_ = cs.Close()

	cs2, err := OpenCollectionStore(dir)
	if err != nil {
		t.Fatal(err)
	}
	defer func() { _ = cs2.Close() }()
	idx2, ok := cs2.GetMultiVector("mv")
	if !ok {
		t.Fatal("MV collection missing after reopen")
	}
	if _, _, v, live := idx2.Get(3); !live || v != 55 {
		t.Fatalf("replayed doc 3 version = %d (live=%v), want 55 VERBATIM", v, live)
	}
}

// TestMVScanDocuments_CarriesVersion confirms ScanDocuments populates the per-doc
// version on each MultiScanRecord (0 for a fresh add via the if-absent path, >1
// after replace-bumps).
func TestMVScanDocuments_CarriesVersion(t *testing.T) {
	m, err := NewMultiVectorIndex(MultiVectorConfig{Dim: 4, Seed: 1})
	if err != nil {
		t.Fatal(err)
	}
	// Doc 1: add then replace twice ⇒ version 3.
	_ = m.Add(1, [][]float32{{1, 0, 0, 0}}, nil)
	_ = m.Add(1, [][]float32{{1, 0, 0, 0}}, nil)
	_ = m.Add(1, [][]float32{{1, 0, 0, 0}}, nil)
	// Doc 2: single add ⇒ version 1.
	_ = m.Add(2, [][]float32{{0, 1, 0, 0}}, nil)

	got := map[uint64]uint64{}
	for _, r := range m.ScanDocuments() {
		got[r.ID] = r.Version
	}
	if got[1] != 3 {
		t.Fatalf("doc 1 scan version = %d, want 3", got[1])
	}
	if got[2] != 1 {
		t.Fatalf("doc 2 scan version = %d, want 1", got[2])
	}
}
