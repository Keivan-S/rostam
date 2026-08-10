// SPDX-License-Identifier: Apache-2.0

package vector

import (
	"bytes"
	"testing"
)

// TestStoreSnapshotRaftSQ4RoundTrip proves the Raft store-snapshot path
// (SnapshotAll -> RestoreAll) carries a QuantSQ SQBits=4 collection's quantizer
// geometry. Before snapColCfg carried SQBits, the restored collection rebuilt as
// 8-bit (SQBits 0 -> 8) and mis-decoded its 4-bit-packed codes: codes were NOT
// bit-identical and search collapsed. This test FAILS without the SQBits field on
// snapColCfg (and its two copy sites) — the restored collection's hnsw quantizer
// would have bits=8 while the codes were packed at bits=4, so arena.Code lengths
// differ and the per-slot bit-identity assertion fails.
func TestStoreSnapshotRaftSQ4RoundTrip(t *testing.T) {
	const (
		n    = 2_000
		dim  = 64
		k    = 10
		seed = 31
	)
	ids, vecs := siftLikeCorpus(n, dim, seed)
	_, queries := siftLikeCorpus(40, dim, 99)

	cfg := Config{Dim: dim, Metric: L2, M: 16, EfConstruction: 200, EfSearch: 64, Seed: seed,
		Quant: QuantSQ, SQBits: 4}

	src, err := OpenCollectionStore(t.TempDir())
	if err != nil {
		t.Fatal(err)
	}
	defer src.Close()
	if err := src.CreateCollection("sq", cfg); err != nil {
		t.Fatal(err)
	}
	srcCol, ok := src.Get("sq")
	if !ok {
		t.Fatal("source collection missing")
	}
	if err := srcCol.StageBulk(ids, vecs); err != nil {
		t.Fatal(err)
	}
	if err := srcCol.BuildStaged(4); err != nil {
		t.Fatal(err)
	}
	srcH := srcCol.idx.(*hnsw)
	if srcH.sqUntrained() {
		t.Fatal("source SQ index should be trained after BuildStaged")
	}
	if srcH.quant.(*trainedSQ).bits != 4 {
		t.Fatalf("source SQ bits = %d, want 4", srcH.quant.(*trainedSQ).bits)
	}

	before := make([][]uint64, len(queries))
	for i, q := range queries {
		res, serr := srcCol.Search(q, k)
		if serr != nil {
			t.Fatal(serr)
		}
		before[i] = resultIDs(res)
	}
	srcCodes := make(map[uint64][]byte, n)
	for _, id := range ids {
		slot, sok := srcH.arena.Slot(id)
		if !sok {
			t.Fatalf("id %d missing from source arena", id)
		}
		srcCodes[id] = append([]byte(nil), srcH.arena.Code(slot)...)
	}

	var blob bytes.Buffer
	if err := src.SnapshotAll(&blob); err != nil {
		t.Fatalf("SnapshotAll: %v", err)
	}

	dst, err := OpenCollectionStore(t.TempDir())
	if err != nil {
		t.Fatal(err)
	}
	defer dst.Close()
	if err := dst.RestoreAll(bytes.NewReader(blob.Bytes())); err != nil {
		t.Fatalf("RestoreAll: %v", err)
	}

	dstCol, ok := dst.Get("sq")
	if !ok {
		t.Fatal("collection missing after RestoreAll")
	}
	if dstCol.Config().SQBits != 4 {
		t.Fatalf("restored SQBits = %d, want 4 (snapColCfg dropped the field)", dstCol.Config().SQBits)
	}
	dstH := dstCol.idx.(*hnsw)
	if dstH.sqUntrained() {
		t.Fatal("restored SQ index is UNTRAINED — ranges did not survive RestoreAll")
	}
	if dstH.quant.(*trainedSQ).bits != 4 {
		t.Fatalf("restored SQ bits = %d, want 4 (rebuilt as 8-bit -> code mis-decode)", dstH.quant.(*trainedSQ).bits)
	}

	// Per-slot codes BIT-IDENTICAL after restore (the headline corruption check:
	// an 8-bit rebuild packs codes at a different length and would mismatch here).
	for _, id := range ids {
		slot, sok := dstH.arena.Slot(id)
		if !sok {
			t.Fatalf("id %d missing from restored arena", id)
		}
		if !bytes.Equal(srcCodes[id], dstH.arena.Code(slot)) {
			t.Fatalf("id %d code not bit-identical after RestoreAll", id)
		}
	}
	// Search results identical post-restore.
	for i, q := range queries {
		res, serr := dstCol.Search(q, k)
		if serr != nil {
			t.Fatal(serr)
		}
		if !eqUint64(resultIDs(res), before[i]) {
			t.Fatalf("query %d: restored %v != original %v", i, resultIDs(res), before[i])
		}
	}
}

// TestStoreSnapshotRaftPRQRoundTrip proves the Raft store-snapshot path carries a
// QuantPRQ collection's non-default QuantPQM + PRQLayers (+ OPQ) through
// SnapshotAll -> RestoreAll. Before snapColCfg carried these, the restored
// collection's arena was built with default m/layers (wrong arena.codeLen) before
// loadCodebooks could correct it: codes were corrupt / out-of-bounds and search
// collapsed. dim=64 gives defaultPQM=8, so QuantPQM=16 and PRQLayers=3 are both
// non-default; this test FAILS without the PRQLayers/QuantPQM/OPQ fields.
func TestStoreSnapshotRaftPRQRoundTrip(t *testing.T) {
	const (
		n    = 2_000
		dim  = 64
		k    = 10
		seed = 41
		m    = 16 // non-default: defaultPQM(64) == 8
	)
	ids, vecs := siftLikeCorpus(n, dim, seed)
	_, queries := siftLikeCorpus(40, dim, 99)

	cfg := Config{Dim: dim, Metric: L2, M: 16, EfConstruction: 200, EfSearch: 64, Seed: seed,
		Quant: QuantPRQ, QuantPQM: m, PRQLayers: 3, OPQ: true, RescoreFactor: 4}

	src, err := OpenCollectionStore(t.TempDir())
	if err != nil {
		t.Fatal(err)
	}
	defer src.Close()
	if err := src.CreateCollection("prq", cfg); err != nil {
		t.Fatal(err)
	}
	srcCol, ok := src.Get("prq")
	if !ok {
		t.Fatal("source collection missing")
	}
	if err := srcCol.StageBulk(ids, vecs); err != nil {
		t.Fatal(err)
	}
	if err := srcCol.BuildStaged(4); err != nil {
		t.Fatal(err)
	}
	srcH := srcCol.idx.(*hnsw)
	if srcH.prqUntrained() {
		t.Fatal("source PRQ index should be trained after BuildStaged")
	}

	before := make([][]uint64, len(queries))
	for i, q := range queries {
		res, serr := srcCol.Search(q, k)
		if serr != nil {
			t.Fatal(serr)
		}
		before[i] = resultIDs(res)
	}
	srcCodes := make(map[uint64][]byte, n)
	for _, id := range ids {
		slot, sok := srcH.arena.Slot(id)
		if !sok {
			t.Fatalf("id %d missing from source arena", id)
		}
		srcCodes[id] = append([]byte(nil), srcH.arena.Code(slot)...)
	}

	var blob bytes.Buffer
	if err := src.SnapshotAll(&blob); err != nil {
		t.Fatalf("SnapshotAll: %v", err)
	}

	dst, err := OpenCollectionStore(t.TempDir())
	if err != nil {
		t.Fatal(err)
	}
	defer dst.Close()
	if err := dst.RestoreAll(bytes.NewReader(blob.Bytes())); err != nil {
		t.Fatalf("RestoreAll: %v", err)
	}

	dstCol, ok := dst.Get("prq")
	if !ok {
		t.Fatal("collection missing after RestoreAll")
	}
	if dstCol.Config().QuantPQM != m || dstCol.Config().PRQLayers != 3 || !dstCol.Config().OPQ {
		t.Fatalf("restored PRQ cfg lost geometry: QuantPQM=%d PRQLayers=%d OPQ=%v, want %d/3/true",
			dstCol.Config().QuantPQM, dstCol.Config().PRQLayers, dstCol.Config().OPQ, m)
	}
	dstH := dstCol.idx.(*hnsw)
	if dstH.prqUntrained() {
		t.Fatal("restored PRQ index is UNTRAINED — codebooks did not survive RestoreAll")
	}

	// Per-slot codes BIT-IDENTICAL after restore (a default-m/layers rebuild has a
	// different code length and would mismatch / index out of bounds here).
	for _, id := range ids {
		slot, sok := dstH.arena.Slot(id)
		if !sok {
			t.Fatalf("id %d missing from restored arena", id)
		}
		if !bytes.Equal(srcCodes[id], dstH.arena.Code(slot)) {
			t.Fatalf("id %d code not bit-identical after RestoreAll", id)
		}
	}
	for i, q := range queries {
		res, serr := dstCol.Search(q, k)
		if serr != nil {
			t.Fatal(serr)
		}
		if !eqUint64(resultIDs(res), before[i]) {
			t.Fatalf("query %d: restored %v != original %v", i, resultIDs(res), before[i])
		}
	}
}

// TestValidateRejectsIVFQuantSQ asserts the MN1 fix: IndexIVF + QuantSQ is
// rejected by Validate (IVF has no SQ auto-train path), mirroring the QuantPQ /
// QuantPRQ non-HNSW rejection. The HNSW + QuantSQ control must still pass.
func TestValidateRejectsIVFQuantSQ(t *testing.T) {
	ivfSQ := Config{Dim: 64, Metric: L2, M: 16, EfConstruction: 200, EfSearch: 64,
		IndexType: IndexIVF, IVFNlist: 16, Quant: QuantSQ, SQBits: 4}
	if err := ivfSQ.Validate(); err == nil {
		t.Fatal("IVF + QuantSQ should be rejected by Validate (no IVF SQ auto-train)")
	}
	hnswSQ := Config{Dim: 64, Metric: L2, M: 16, EfConstruction: 200, EfSearch: 64,
		IndexType: IndexHNSW, Quant: QuantSQ, SQBits: 4}
	if err := hnswSQ.Validate(); err != nil {
		t.Fatalf("HNSW + QuantSQ should pass Validate: %v", err)
	}
}
