// SPDX-License-Identifier: Apache-2.0

package vector

import (
	"bytes"
	"path/filepath"
	"runtime"
	"testing"
)

// TestPQHNSWSnapshotADCSurvives builds a PQ-HNSW index, snapshots it, restores
// into a fresh QuantPQ index, and asserts the restored index (a) has TRAINED
// codebooks (!pqUntrained — ADC navigation, not the exact-float degrade), (b)
// has codes BIT-IDENTICAL to pre-snapshot (the codebooks restored verbatim, so
// the re-encode-from-vecs reproduces the same codes), and (c) returns the SAME
// search results as before the snapshot. Previously the restored index would
// re-encode with an UNTRAINED codec (zero codes) and degrade to exact float.
func TestPQHNSWSnapshotADCSurvives(t *testing.T) {
	const (
		n    = 3000
		dim  = 64
		k    = 10
		seed = 42
	)
	ids, vecs := siftLikeCorpus(n, dim, seed)
	_, queries := siftLikeCorpus(100, dim, 7)

	cfg := Config{Dim: dim, Metric: L2, M: 16, EfConstruction: 200, EfSearch: 64, Seed: seed, Quant: QuantPQ, QuantPQM: 16}
	src, err := newHNSW(cfg)
	if err != nil {
		t.Fatal(err)
	}
	if err := src.BuildConcurrent(ids, vecs, 4); err != nil {
		t.Fatal(err)
	}
	if src.pqUntrained() {
		t.Fatal("source PQ index should be trained after BuildConcurrent")
	}

	// Capture pre-snapshot search results AND the raw codes for each live slot.
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

	dst, err := newHNSW(cfg)
	if err != nil {
		t.Fatal(err)
	}
	if err := dst.Restore(&buf); err != nil {
		t.Fatalf("Restore: %v", err)
	}

	// (a) The restored codec is TRAINED — ADC navigation, not the exact-float degrade.
	if dst.pqUntrained() {
		t.Fatal("restored PQ index is UNTRAINED (degraded to exact-float) — codebooks did not survive")
	}

	// (d) The codebooks restored VERBATIM: compare bit-for-bit with the source.
	sq := src.quant.(*pqQuantizer)
	dq := dst.quant.(*pqQuantizer)
	srcCB, dstCB := sq.codebooks(), dq.codebooks()
	if len(srcCB) != len(dstCB) {
		t.Fatalf("codebook M mismatch: src=%d dst=%d", len(srcCB), len(dstCB))
	}
	for s := range srcCB {
		if len(srcCB[s]) != len(dstCB[s]) {
			t.Fatalf("subspace %d sub-centroid count mismatch: src=%d dst=%d", s, len(srcCB[s]), len(dstCB[s]))
		}
		for c := range srcCB[s] {
			for i := range srcCB[s][c] {
				if srcCB[s][c][i] != dstCB[s][c][i] {
					t.Fatalf("codebook[%d][%d][%d] not verbatim: src=%v dst=%v", s, c, i, srcCB[s][c][i], dstCB[s][c][i])
				}
			}
		}
	}

	// (b) Codes re-encoded from the restored verbatim codebooks are bit-identical.
	for slot := 0; slot < n; slot++ {
		if !bytes.Equal(srcCodes[slot], dst.arena.Code(uint32(slot))) {
			t.Fatalf("slot %d code not bit-identical after restore: src=%v dst=%v", slot, srcCodes[slot], dst.arena.Code(uint32(slot)))
		}
	}

	// (c) Search results are IDENTICAL post-restore (ADC navigate reproduced).
	for i, q := range queries {
		res, serr := dst.Search(q, k)
		if serr != nil {
			t.Fatal(serr)
		}
		if !eqUint64(resultIDs(res), before[i]) {
			t.Fatalf("query %d: restored results %v != pre-snapshot %v", i, resultIDs(res), before[i])
		}
	}
}

// TestPQHNSWSnapshotUntrainedNoBlock snapshots a QuantPQ index that was NEVER
// built (untrained: inserts only) — the presence byte is 0, restore re-encodes
// from vecs with an untrained codec, and the index stays untrained (exact-float),
// matching the pre-train behaviour. This exercises the v7-header / 0-presence-byte
// path without codebooks.
func TestPQHNSWSnapshotUntrainedNoBlock(t *testing.T) {
	const dim = 32
	cfg := Config{Dim: dim, Metric: L2, M: 16, EfConstruction: 200, EfSearch: 64, Seed: 1, Quant: QuantPQ, QuantPQM: 8}
	src, err := newHNSW(cfg)
	if err != nil {
		t.Fatal(err)
	}
	v := make([]float32, dim)
	for j := range v {
		v[j] = float32(j) * 0.01
	}
	if _, _, err := src.Insert(1, v, 0, nil, nil, nil, CASCond{}); err != nil {
		t.Fatal(err)
	}
	if !src.pqUntrained() {
		t.Fatal("insert-only PQ index should be untrained")
	}
	var buf bytes.Buffer
	if err := src.Snapshot(&buf); err != nil {
		t.Fatalf("Snapshot: %v", err)
	}
	dst, err := newHNSW(cfg)
	if err != nil {
		t.Fatal(err)
	}
	if err := dst.Restore(&buf); err != nil {
		t.Fatalf("Restore: %v", err)
	}
	if !dst.pqUntrained() {
		t.Fatal("restored insert-only PQ index should still be untrained")
	}
}

// TestNonPQSnapshotByteIdentical asserts a non-PQ (sq8/bq1/none) snapshot stays
// BYTE-COMPATIBLE: the version field stays v6 (no PQ bump) and no PQ
// block is appended, so the layout is identical to pre-Task-2. (Whole-snapshot
// byte determinism is not asserted because map blocks — idMap/tombstones/metadata
// — iterate in Go's randomized order; that is pre-existing.) The version-field
// check + an unaffected round-trip pin the invariant that must be preserved.
func TestNonPQSnapshotByteIdentical(t *testing.T) {
	const (
		n   = 800
		dim = 32
		k   = 10
	)
	ids, vecs := siftLikeCorpus(n, dim, 5)
	for _, v := range vecs {
		normalize(v)
	}
	_, queries := siftLikeCorpus(40, dim, 11)
	for _, q := range queries {
		normalize(q)
	}

	for _, qm := range []QuantMode{QuantNone, QuantSQ8, QuantBQ1} {
		cfg := Config{Dim: dim, Metric: Cosine, M: 16, EfConstruction: 200, EfSearch: 64, Seed: 3, Quant: qm, RescoreFactor: 3}
		h, err := newHNSW(cfg)
		if err != nil {
			t.Fatalf("quant %d: newHNSW: %v", qm, err)
		}
		if err := h.BuildConcurrent(ids, vecs, 4); err != nil {
			t.Fatalf("quant %d: build: %v", qm, err)
		}
		var b1 bytes.Buffer
		if err := h.Snapshot(&b1); err != nil {
			t.Fatalf("quant %d: snapshot: %v", qm, err)
		}
		// Version field is the u32 right after the 8-byte magic; must stay v6 (no
		// PQ bump for non-PQ indices ⇒ byte-identical layout to pre-Task-2).
		ver := uint32(b1.Bytes()[8])<<24 | uint32(b1.Bytes()[9])<<16 | uint32(b1.Bytes()[10])<<8 | uint32(b1.Bytes()[11])
		if ver != snapshotVersionNoPQ {
			t.Fatalf("quant %d: non-PQ snapshot version = %d, want %d (byte-identity broken)", qm, ver, snapshotVersionNoPQ)
		}

		// Round-trip is unaffected.
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
		if err := dst.Restore(&b1); err != nil {
			t.Fatalf("quant %d: restore: %v", qm, err)
		}
		for i, q := range queries {
			res, serr := dst.Search(q, k)
			if serr != nil {
				t.Fatal(serr)
			}
			if !eqUint64(resultIDs(res), before[i]) {
				t.Fatalf("quant %d query %d: restored %v != %v", qm, i, resultIDs(res), before[i])
			}
		}
	}
}

// TestPQHNSWPersistADCSurvives is the sidecar analogue: a PQ-HNSW index backed by
// mmap (QuantMmap, the only persistable storage) + graph mmap is SavePersist'd,
// closed, and reopened via openPersist (map files, no rebuild). The reopened
// index must have TRAINED codebooks and return identical search results — proving
// the sidecar persists + restores the codebooks before restoreDense re-encodes.
func TestPQHNSWPersistADCSurvives(t *testing.T) {
	const (
		n    = 2000
		dim  = 64
		k    = 10
		seed = 17
	)
	dir := t.TempDir()
	ids, vecs := siftLikeCorpus(n, dim, seed)
	for _, v := range vecs {
		normalize(v)
	}
	_, queries := siftLikeCorpus(80, dim, 23)
	for _, q := range queries {
		normalize(q)
	}

	cfg := Config{
		Dim: dim, Metric: Cosine, M: 16, EfConstruction: 200, EfSearch: 64, Seed: seed,
		Quant: QuantPQ, QuantPQM: 16, QuantStorage: QuantMmap, RescoreFactor: 3,
		MmapPath:      filepath.Join(dir, "vecs.dat"),
		GraphMmapPath: filepath.Join(dir, "graph.dat"),
	}
	if err := ValidateConfig(cfg); err != nil {
		t.Fatalf("PQ-HNSW + Persistent(mmap) config rejected: %v", err)
	}
	h, err := newHNSW(cfg)
	if err != nil {
		t.Fatal(err)
	}
	if err := h.BuildConcurrent(ids, vecs, runtime.GOMAXPROCS(0)); err != nil {
		t.Fatalf("build: %v", err)
	}
	if h.pqUntrained() {
		t.Fatal("PQ index untrained after build")
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

	if h2.pqUntrained() {
		t.Fatal("reopened PQ index is UNTRAINED — sidecar did not persist/restore codebooks")
	}
	for i, q := range queries {
		res, serr := h2.Search(q, k)
		if serr != nil {
			t.Fatal(serr)
		}
		if !eqUint64(resultIDs(res), before[i]) {
			t.Fatalf("query %d: reopened %v != original %v", i, resultIDs(res), before[i])
		}
	}
}
