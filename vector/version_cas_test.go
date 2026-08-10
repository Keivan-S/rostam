// SPDX-License-Identifier: Apache-2.0

package vector

import (
	"bytes"
	"errors"
	"path/filepath"
	"runtime"
	"testing"
	"time"
)

// versionTestConfig is the small L2 config the version/CAS tests build on (L2 so
// stored vectors are not cosine-normalized — version assertions are independent of
// the vector values).
func versionTestConfig() Config {
	return Config{Dim: 2, M: 4, EfConstruction: 10, EfSearch: 10, Seed: 1, Metric: L2}
}

func mustInsertVer(t *testing.T, h *hnsw, id uint64, vec []float32, cas CASCond) uint64 {
	t.Helper()
	v, _, err := h.Insert(id, vec, 0, nil, nil, nil, cas)
	if err != nil {
		t.Fatalf("Insert(%d): %v", id, err)
	}
	return v
}

func getVersion(t *testing.T, h *hnsw, id uint64) (uint64, bool) {
	t.Helper()
	_, _, _, _, ver, ok := h.Get(id)
	return ver, ok
}

// TestVersion_FreshInsertIsOne — a new insert of an absent id is version 1; an
// absent point reads version 0.
func TestVersion_FreshInsertIsOne(t *testing.T) {
	h, _ := newHNSW(versionTestConfig())

	if ver, ok := getVersion(t, h, 1); ok || ver != 0 {
		t.Fatalf("absent point: want (0,false), got (%d,%v)", ver, ok)
	}
	if got := mustInsertVer(t, h, 1, []float32{1, 2}, CASCond{}); got != 1 {
		t.Fatalf("fresh insert: want version 1, got %d", got)
	}
	if ver, ok := getVersion(t, h, 1); !ok || ver != 1 {
		t.Fatalf("Get after insert: want (1,true), got (%d,%v)", ver, ok)
	}
}

// TestVersion_PayloadOpsBump — each in-place payload mutation bumps the version
// by exactly 1, and Get reflects it.
func TestVersion_PayloadOpsBump(t *testing.T) {
	h, _ := newHNSW(versionTestConfig())
	mustInsertVer(t, h, 1, []float32{1, 2}, CASCond{}) // version 1

	_, _, v, err := h.SetPayload(1, Metadata{"a": NewInt(1)}, nil, CASCond{})
	if err != nil || v != 2 {
		t.Fatalf("SetPayload: want version 2, got %d err=%v", v, err)
	}
	_, _, v, err = h.OverwritePayload(1, Metadata{"b": NewInt(2)}, nil, CASCond{})
	if err != nil || v != 3 {
		t.Fatalf("OverwritePayload: want version 3, got %d err=%v", v, err)
	}
	_, _, v, err = h.DeletePayloadKeys(1, []string{"b"}, CASCond{})
	if err != nil || v != 4 {
		t.Fatalf("DeletePayloadKeys: want version 4, got %d err=%v", v, err)
	}
	_, _, v, err = h.ClearPayload(1, CASCond{})
	if err != nil || v != 5 {
		t.Fatalf("ClearPayload: want version 5, got %d err=%v", v, err)
	}
	if ver, ok := getVersion(t, h, 1); !ok || ver != 5 {
		t.Fatalf("Get final: want (5,true), got (%d,%v)", ver, ok)
	}
}

// TestCAS_MatchAppliesAndBumps — a CAS with the correct expected version applies
// the mutation and bumps the version.
func TestCAS_MatchAppliesAndBumps(t *testing.T) {
	h, _ := newHNSW(versionTestConfig())
	mustInsertVer(t, h, 1, []float32{1, 2}, CASCond{}) // version 1

	m, _, v, err := h.SetPayload(1, Metadata{"a": NewInt(7)}, nil, CASCond{Expected: 1, Has: true})
	if err != nil {
		t.Fatalf("CAS match: unexpected err %v", err)
	}
	if v != 2 {
		t.Fatalf("CAS match: want version 2, got %d", v)
	}
	if m["a"].Int != 7 {
		t.Fatalf("CAS match: payload not applied: %v", m)
	}
}

// TestCAS_MismatchConflictsNoMutation — a CAS with a wrong expected version
// returns ErrVersionConflict, applies NO mutation, and does NOT bump.
func TestCAS_MismatchConflictsNoMutation(t *testing.T) {
	h, _ := newHNSW(versionTestConfig())
	mustInsertVer(t, h, 1, []float32{1, 2}, CASCond{})        // version 1
	h.SetPayload(1, Metadata{"a": NewInt(1)}, nil, CASCond{}) //nolint:errcheck // version 2

	_, _, _, err := h.SetPayload(1, Metadata{"a": NewInt(99)}, nil, CASCond{Expected: 1, Has: true})
	if !errors.Is(err, ErrVersionConflict) {
		t.Fatalf("CAS mismatch: want ErrVersionConflict, got %v", err)
	}
	// No mutation, no bump: version still 2, payload still {a:1}.
	if ver, _ := getVersion(t, h, 1); ver != 2 {
		t.Fatalf("CAS mismatch must not bump: want version 2, got %d", ver)
	}
	_, meta, _, _, _, ok := h.Get(1)
	if !ok || meta["a"].Int != 1 {
		t.Fatalf("CAS mismatch must not mutate payload: got %v", meta)
	}
}

// TestCAS_InsertIfAbsent — Expected==0 + Has is "insert-if-absent": it succeeds
// iff the point is absent and conflicts when the point is present.
func TestCAS_InsertIfAbsent(t *testing.T) {
	h, _ := newHNSW(versionTestConfig())

	// Absent → expected 0 succeeds, version 1.
	v, _, err := h.Insert(1, []float32{1, 2}, 0, nil, nil, nil, CASCond{Expected: 0, Has: true})
	if err != nil || v != 1 {
		t.Fatalf("insert-if-absent on absent: want version 1, got %d err=%v", v, err)
	}
	// Present (live) → a payload CAS expecting 0 conflicts.
	_, _, _, err = h.SetPayload(1, Metadata{"a": NewInt(1)}, nil, CASCond{Expected: 0, Has: true})
	if !errors.Is(err, ErrVersionConflict) {
		t.Fatalf("insert-if-absent on present: want ErrVersionConflict, got %v", err)
	}
}

// TestCAS_DeleteConditional — Delete honors the CAS precondition: a wrong
// expected version conflicts (no delete), the right one removes the point, and a
// reinsert of the deleted id starts at version 1.
func TestCAS_DeleteConditional(t *testing.T) {
	h, _ := newHNSW(versionTestConfig())
	mustInsertVer(t, h, 1, []float32{1, 2}, CASCond{})        // version 1
	h.SetPayload(1, Metadata{"a": NewInt(1)}, nil, CASCond{}) //nolint:errcheck // version 2

	if _, err := h.Delete(1, CASCond{Expected: 1, Has: true}); !errors.Is(err, ErrVersionConflict) {
		t.Fatalf("Delete CAS mismatch: want ErrVersionConflict, got %v", err)
	}
	if _, ok := getVersion(t, h, 1); !ok {
		t.Fatalf("Delete CAS mismatch must not delete the point")
	}
	removed, err := h.Delete(1, CASCond{Expected: 2, Has: true})
	if err != nil || !removed {
		t.Fatalf("Delete CAS match: want removed, got removed=%v err=%v", removed, err)
	}
	// Reinsert of the previously-deleted id → fresh point, version 1.
	if got := mustInsertVer(t, h, 1, []float32{3, 4}, CASCond{}); got != 1 {
		t.Fatalf("reinsert after delete: want version 1, got %d", got)
	}
}

// TestVersion_DeleteReinsertResetsToOne — a non-CAS delete then reinsert of the
// same id resets the version to 1 (the point is new).
func TestVersion_DeleteReinsertResetsToOne(t *testing.T) {
	h, _ := newHNSW(versionTestConfig())
	mustInsertVer(t, h, 1, []float32{1, 2}, CASCond{})        // version 1
	h.SetPayload(1, Metadata{"a": NewInt(1)}, nil, CASCond{}) //nolint:errcheck // version 2
	h.SetPayload(1, Metadata{"b": NewInt(2)}, nil, CASCond{}) //nolint:errcheck // version 3
	if ver, _ := getVersion(t, h, 1); ver != 3 {
		t.Fatalf("pre-delete: want version 3, got %d", ver)
	}
	if removed, _ := h.Delete(1, CASCond{}); !removed {
		t.Fatalf("Delete should remove the live point")
	}
	if got := mustInsertVer(t, h, 1, []float32{5, 6}, CASCond{}); got != 1 {
		t.Fatalf("reinsert: want version 1, got %d", got)
	}
}

// TestVersion_NonCASWriteStillBumps — a no-CAS payload write still bumps the
// version (behavior-preserving for existing callers + the new counter).
func TestVersion_NonCASWriteStillBumps(t *testing.T) {
	h, _ := newHNSW(versionTestConfig())
	mustInsertVer(t, h, 1, []float32{1, 2}, CASCond{}) // version 1
	_, _, v, err := h.SetPayload(1, Metadata{"a": NewInt(1)}, nil, CASCond{})
	if err != nil || v != 2 {
		t.Fatalf("non-CAS write: want version 2, got %d err=%v", v, err)
	}
}

// TestVersion_ScanCarriesVersion — scanVectors exports each live point's version.
func TestVersion_ScanCarriesVersion(t *testing.T) {
	h, _ := newHNSW(versionTestConfig())
	mustInsertVer(t, h, 1, []float32{1, 2}, CASCond{})        // version 1
	mustInsertVer(t, h, 2, []float32{3, 4}, CASCond{})        // version 1
	h.SetPayload(2, Metadata{"a": NewInt(1)}, nil, CASCond{}) //nolint:errcheck // version 2

	want := map[uint64]uint64{1: 1, 2: 2}
	for _, rec := range h.scanVectors() {
		if want[rec.ID] != rec.Version {
			t.Fatalf("scan id %d: want version %d, got %d", rec.ID, want[rec.ID], rec.Version)
		}
		delete(want, rec.ID)
	}
	if len(want) != 0 {
		t.Fatalf("scan missing records: %v", want)
	}
}

// TestVersion_RestoreInsertPreservesVersion — RestoreInsert sets the exact
// version verbatim (the reshard/WAL-replay primitive), NOT 1.
func TestVersion_RestoreInsertPreservesVersion(t *testing.T) {
	h, _ := newHNSW(versionTestConfig())
	if err := h.RestoreInsert(7, []float32{1, 2}, 0, nil, nil, nil, 42); err != nil {
		t.Fatalf("RestoreInsert: %v", err)
	}
	if ver, ok := getVersion(t, h, 7); !ok || ver != 42 {
		t.Fatalf("RestoreInsert version: want (42,true), got (%d,%v)", ver, ok)
	}
	// A subsequent in-place mutation bumps from the restored value.
	_, _, v, err := h.SetPayload(7, Metadata{"a": NewInt(1)}, nil, CASCond{})
	if err != nil || v != 43 {
		t.Fatalf("bump after restore: want 43, got %d err=%v", v, err)
	}
}

// TestVersion_SnapshotRoundTrip — a snapshot preserves each point's version
// VERBATIM (restore does not re-bump).
func TestVersion_SnapshotRoundTrip(t *testing.T) {
	h, _ := newHNSW(versionTestConfig())
	mustInsertVer(t, h, 1, []float32{1, 2}, CASCond{})        // version 1
	mustInsertVer(t, h, 2, []float32{3, 4}, CASCond{})        // version 1
	h.SetPayload(2, Metadata{"a": NewInt(1)}, nil, CASCond{}) //nolint:errcheck // version 2
	h.SetPayload(2, Metadata{"b": NewInt(2)}, nil, CASCond{}) //nolint:errcheck // version 3

	var buf bytes.Buffer
	if err := h.Snapshot(&buf); err != nil {
		t.Fatalf("Snapshot: %v", err)
	}
	h2, _ := newHNSW(versionTestConfig())
	if err := h2.Restore(&buf); err != nil {
		t.Fatalf("Restore: %v", err)
	}
	if ver, ok := getVersion(t, h2, 1); !ok || ver != 1 {
		t.Fatalf("restored id 1: want (1,true), got (%d,%v)", ver, ok)
	}
	if ver, ok := getVersion(t, h2, 2); !ok || ver != 3 {
		t.Fatalf("restored id 2: want (3,true), got (%d,%v)", ver, ok)
	}
}

// TestVersion_WALRoundTrip — WAL records carry the version and replay restores it
// VERBATIM (insert + a set-payload, restored without re-bumping). Uses a
// snapshot checkpoint of an EMPTY index + a WAL tail (the dense WAL replay path).
func TestVersion_WALRoundTrip(t *testing.T) {
	dir := t.TempDir()
	walPath := filepath.Join(dir, "wal.log")

	w, err := openWAL(walPath, false)
	if err != nil {
		t.Fatalf("openWAL: %v", err)
	}
	// Log an insert at version 1, then a payload mutation resulting in version 2.
	// Both go through the staged pair (the only way either record is produced now:
	// the insert-family restore paths stage + commit-wait exactly like this, with
	// opMu released in between).
	iseq, err := w.appendInsertStaged(1, []float32{1, 2}, 0, nil, nil, nil, 1)
	if err != nil {
		t.Fatalf("appendInsertStaged: %v", err)
	}
	if err := w.commitWaitStaged(iseq); err != nil {
		t.Fatalf("commitWaitStaged insert: %v", err)
	}
	// Staged write + explicit commit-wait: the only way a setPayload record is
	// produced (payloadOpCAS does exactly this, with opMu released in between).
	seq, err := w.appendSetPayloadStaged(1, Metadata{"a": NewInt(1)}, nil, 2)
	if err != nil {
		t.Fatalf("appendSetPayloadStaged: %v", err)
	}
	if err := w.commitWaitStaged(seq); err != nil {
		t.Fatalf("commitWaitStaged: %v", err)
	}
	if err := w.close(); err != nil {
		t.Fatalf("close wal: %v", err)
	}

	h, _ := newHNSW(versionTestConfig())
	err = replayWAL(walPath,
		func(id uint64, vec []float32, ttl time.Duration, meta Metadata, sparse *SparseVector, keyExpires map[string]uint64, version uint64) {
			_ = h.RestoreInsert(id, vec, ttl, meta, sparse, keyExpires, version)
		},
		func(id uint64) { _, _ = h.Delete(id, CASCond{}) },
		func(id uint64, meta Metadata, keyExpires map[string]uint64, version uint64) {
			_ = h.RestorePayload(id, meta, keyExpires, version)
		},
	)
	if err != nil {
		t.Fatalf("replayWAL: %v", err)
	}
	if ver, ok := getVersion(t, h, 1); !ok || ver != 2 {
		t.Fatalf("WAL replay version: want (2,true), got (%d,%v)", ver, ok)
	}
}

// TestVersion_SidecarRoundTrip — the instant-restart sidecar (SavePersist /
// openPersist) preserves each point's version VERBATIM. A bulk build sets every
// point to version 1; a payload mutation bumps one to 2; after save+reopen both
// are restored unchanged.
func TestVersion_SidecarRoundTrip(t *testing.T) {
	dir := t.TempDir()
	const dim = 8
	cfg := Config{
		Dim: dim, Metric: Cosine, M: 8, EfConstruction: 50, EfSearch: 50, Seed: 1,
		Quant: QuantSQ8, QuantStorage: QuantMmap, MmapPath: filepath.Join(dir, "v.dat"),
		RescoreFactor: 3, GraphMmapPath: filepath.Join(dir, "g.dat"),
	}
	h, err := newHNSW(cfg)
	if err != nil {
		t.Fatalf("newHNSW: %v", err)
	}
	ids := []uint64{1, 2, 3}
	vecs := [][]float32{
		{1, 0, 0, 0, 0, 0, 0, 0},
		{0, 1, 0, 0, 0, 0, 0, 0},
		{0, 0, 1, 0, 0, 0, 0, 0},
	}
	for _, v := range vecs {
		normalize(v)
	}
	if err := h.BuildConcurrent(ids, vecs, runtime.GOMAXPROCS(0)); err != nil {
		t.Fatalf("BuildConcurrent: %v", err) // each built point → version 1
	}
	// Bump id 2 to version 2 via a payload op.
	if _, _, v, err := h.SetPayload(2, Metadata{"a": NewInt(1)}, nil, CASCond{}); err != nil || v != 2 {
		t.Fatalf("SetPayload: want version 2, got %d err=%v", v, err)
	}
	metaPath := filepath.Join(dir, "m.bin")
	if err := h.SavePersist(metaPath); err != nil {
		t.Fatalf("SavePersist: %v", err)
	}
	if err := h.Close(); err != nil {
		t.Fatalf("Close: %v", err)
	}
	h2, err := openPersist(cfg, metaPath)
	if err != nil {
		t.Fatalf("openPersist: %v", err)
	}
	defer h2.Close() //nolint:errcheck
	for id, want := range map[uint64]uint64{1: 1, 2: 2, 3: 1} {
		if ver, ok := getVersion(t, h2, id); !ok || ver != want {
			t.Fatalf("sidecar id %d: want (%d,true), got (%d,%v)", id, want, ver, ok)
		}
	}
}
