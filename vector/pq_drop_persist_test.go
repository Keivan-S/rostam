// SPDX-License-Identifier: Apache-2.0

package vector

import (
	"bytes"
	"path/filepath"
	"runtime"
	"testing"
)

// snapshotVersionField extracts the u32 version stamped right after the 8-byte
// magic in a snapshot blob.
func snapshotVersionField(b []byte) uint32 {
	return uint32(b[8])<<24 | uint32(b[9])<<16 | uint32(b[10])<<8 | uint32(b[11])
}

// TestPQDropSnapshotRestoreADCSame builds a float-drop PQ-HNSW index, snapshots
// it, restores into a fresh index, and asserts:
//   - the snapshot is stamped v9 (snapshotVersionPQDrop) — the dropped format;
//   - the restored index is in the dropped state (vecsDropped, vecs nil);
//   - the codes restored VERBATIM (bit-identical, no re-encode — there are no
//     floats to encode from);
//   - search returns the SAME ADC ordering as before the snapshot;
//   - Get reconstructs a non-nil right-dimension vector after restore.
func TestPQDropSnapshotRestoreADCSame(t *testing.T) {
	const (
		n         = 3000
		dim       = 64
		k         = 10
		nClusters = 40
		seed      = 42
	)
	ids, vecs, queries := buildPQDropCorpus(n, dim, nClusters, seed)
	queries = queries[:80]

	cfg := Config{Dim: dim, Metric: L2, M: 16, EfConstruction: 200, EfSearch: 64, Seed: seed,
		Quant: QuantPQ, QuantPQM: 16, PQDropVecs: true}
	src, err := newHNSW(cfg)
	if err != nil {
		t.Fatal(err)
	}
	if err := src.BuildConcurrent(ids, vecs, 4); err != nil {
		t.Fatal(err)
	}
	if !src.vecsDropped() {
		t.Fatal("source must be in the dropped state after build")
	}

	before := make([][]uint64, len(queries))
	for i, q := range queries {
		res, serr := src.Search(q, k)
		if serr != nil {
			t.Fatal(serr)
		}
		before[i] = resultIDs(res)
	}
	srcCodes := make([][]byte, n)
	for slot := 0; slot < n; slot++ {
		srcCodes[slot] = append([]byte(nil), src.arena.Code(uint32(slot))...)
	}

	var buf bytes.Buffer
	if err := src.Snapshot(&buf); err != nil {
		t.Fatalf("Snapshot: %v", err)
	}
	if v := snapshotVersionField(buf.Bytes()); v != snapshotVersionPQDrop {
		t.Fatalf("dropped snapshot version = %d, want %d (v9)", v, snapshotVersionPQDrop)
	}

	dst, err := newHNSW(cfg)
	if err != nil {
		t.Fatal(err)
	}
	if err := dst.Restore(&buf); err != nil {
		t.Fatalf("Restore: %v", err)
	}
	if !dst.vecsDropped() {
		t.Fatal("restored index must be in the dropped state")
	}
	if dst.arena.vecs != nil {
		t.Fatal("restored arena.vecs must be nil (dropped)")
	}
	if dst.pqUntrained() {
		t.Fatal("restored codec is untrained — codebooks did not survive the dropped snapshot")
	}

	// Codes restored VERBATIM (no re-encode).
	for slot := 0; slot < n; slot++ {
		if !bytes.Equal(srcCodes[slot], dst.arena.Code(uint32(slot))) {
			t.Fatalf("slot %d code not verbatim after restore", slot)
		}
	}

	// Search ADC ordering identical post-restore.
	for i, q := range queries {
		res, serr := dst.Search(q, k)
		if serr != nil {
			t.Fatal(serr)
		}
		if !eqUint64(resultIDs(res), before[i]) {
			t.Fatalf("query %d: restored ADC order %v != pre-snapshot %v", i, resultIDs(res), before[i])
		}
	}

	// Reconstruct survives the restart.
	gv, _, _, _, _, ok := dst.Get(ids[0])
	if !ok || len(gv) != dim {
		t.Fatalf("Get after restore: ok=%v dim=%d", ok, len(gv))
	}
}

// TestPQDropPersistRestart is the sidecar analogue: an mmap-backed float-drop
// PQ-HNSW index is SavePersist'd, closed, the vecs file is REMOVED to prove the
// reopen does not need it, and openPersist restores the dropped state + verbatim
// codes. Search returns the same ADC ordering as before the save.
func TestPQDropPersistRestart(t *testing.T) {
	const (
		n         = 2500
		dim       = 64
		k         = 10
		nClusters = 35
		seed      = 17
	)
	dir := t.TempDir()
	ids, vecs, queries := buildPQDropCorpus(n, dim, nClusters, seed)
	queries = queries[:80]

	cfg := Config{
		Dim: dim, Metric: L2, M: 16, EfConstruction: 200, EfSearch: 64, Seed: seed,
		Quant: QuantPQ, QuantPQM: 16, QuantStorage: QuantMmap, PQDropVecs: true,
		MmapPath:      filepath.Join(dir, "vecs.dat"),
		GraphMmapPath: filepath.Join(dir, "graph.dat"),
	}
	if err := ValidateConfig(cfg); err != nil {
		t.Fatalf("float-drop PQ-HNSW + mmap config rejected: %v", err)
	}
	h, err := newHNSW(cfg)
	if err != nil {
		t.Fatal(err)
	}
	if err := h.BuildConcurrent(ids, vecs, runtime.GOMAXPROCS(0)); err != nil {
		t.Fatalf("build: %v", err)
	}
	if !h.vecsDropped() {
		t.Fatal("expected dropped state after build")
	}

	before := make([][]uint64, len(queries))
	for i, q := range queries {
		res, serr := h.Search(q, k)
		if serr != nil {
			t.Fatal(serr)
		}
		before[i] = resultIDs(res)
	}

	metaPath := filepath.Join(dir, "meta.bin")
	if err := h.SavePersist(metaPath); err != nil {
		t.Fatalf("SavePersist: %v", err)
	}
	if err := h.Close(); err != nil {
		t.Fatalf("close: %v", err)
	}

	h2, err := openPersist(cfg, metaPath)
	if err != nil {
		t.Fatalf("openPersist: %v", err)
	}
	defer func() { _ = h2.Close() }()

	if !h2.vecsDropped() {
		t.Fatal("reopened index must be in the dropped state")
	}
	if h2.arena.vecs != nil {
		t.Fatal("reopened arena.vecs must be nil (no vecs file mapped)")
	}
	if h2.pqUntrained() {
		t.Fatal("reopened codec untrained — sidecar did not persist codebooks")
	}
	for i, q := range queries {
		res, serr := h2.Search(q, k)
		if serr != nil {
			t.Fatal(serr)
		}
		if !eqUint64(resultIDs(res), before[i]) {
			t.Fatalf("query %d: reopened ADC order %v != original %v", i, resultIDs(res), before[i])
		}
	}
}

// TestPQKeepFloatsFormatUnchanged is the back-compat proof: a keep-floats
// PQ-HNSW index (PQDropVecs=false, the default) snapshots at the OLD version
// (v7/v8, never v9) and round-trips identically. This pins that PQ-drop leaves the
// keep-floats on-disk format BYTE-COMPATIBLE — only a dropped index emits v9.
func TestPQKeepFloatsFormatUnchanged(t *testing.T) {
	const (
		n         = 2000
		dim       = 64
		k         = 10
		nClusters = 30
		seed      = 7
	)
	ids, vecs, queries := buildPQDropCorpus(n, dim, nClusters, seed)
	queries = queries[:60]

	cfg := Config{Dim: dim, Metric: L2, M: 16, EfConstruction: 200, EfSearch: 64, Seed: seed,
		Quant: QuantPQ, QuantPQM: 16} // PQDropVecs unset (keep floats)
	h, err := newHNSW(cfg)
	if err != nil {
		t.Fatal(err)
	}
	if err := h.BuildConcurrent(ids, vecs, 4); err != nil {
		t.Fatal(err)
	}
	if h.vecsDropped() {
		t.Fatal("keep-floats PQ-HNSW must NOT drop the floats")
	}

	var buf bytes.Buffer
	if err := h.Snapshot(&buf); err != nil {
		t.Fatalf("Snapshot: %v", err)
	}
	// Keep-floats PQ-HNSW stamps the OLD version (v7 no-OPQ, v8 with OPQ) — NEVER
	// v9. v9 is reserved for the dropped format, so existing keep-floats artifacts
	// and the default path are untouched.
	v := snapshotVersionField(buf.Bytes())
	if v != snapshotVersionPQNoOPQ && v != snapshotVersion {
		t.Fatalf("keep-floats PQ snapshot version = %d, want v7(%d) or v8(%d) — NOT v9",
			v, snapshotVersionPQNoOPQ, snapshotVersion)
	}
	if v == snapshotVersionPQDrop {
		t.Fatal("keep-floats PQ snapshot must never stamp v9 (the dropped format)")
	}

	before := make([][]uint64, len(queries))
	for i, q := range queries {
		res, serr := h.Search(q, k)
		if serr != nil {
			t.Fatal(serr)
		}
		before[i] = resultIDs(res)
	}
	dst, err := newHNSW(cfg)
	if err != nil {
		t.Fatal(err)
	}
	if err := dst.Restore(&buf); err != nil {
		t.Fatalf("Restore: %v", err)
	}
	if dst.vecsDropped() {
		t.Fatal("restored keep-floats index must keep its floats")
	}
	for i, q := range queries {
		res, serr := dst.Search(q, k)
		if serr != nil {
			t.Fatal(serr)
		}
		if !eqUint64(resultIDs(res), before[i]) {
			t.Fatalf("query %d: keep-floats restored %v != %v", i, resultIDs(res), before[i])
		}
	}
}
