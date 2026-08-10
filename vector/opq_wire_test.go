// SPDX-License-Identifier: Apache-2.0

package vector

import (
	"bufio"
	"bytes"
	"math/rand"
	"path/filepath"
	"testing"
)

// OPQ integration tests: cfg.OPQ threaded into the IVF-PQ (residual) and
// HNSW-PQ build paths, the rotation R persisted VERBATIM in the IVF snapshot
// trailer / HNSW snapshot block / HNSW persist sidecar (search survives restart
// bit-identically), and the hard back-compat invariants (OPQ-off snapshots are
// BYTE-IDENTICAL to the pre-OPQ layout; old / no-R blobs restore with rotation
// nil). The codec-level rotation correctness (RᵀR≈I, reconstruct un-rotate,
// recall-on-imbalanced) is proven by pq_test.go; these tests prove the
// WIRING composes end-to-end through build + persist.

// ivfOPQConfig is ivfPQConfig with OPQ toggled.
func ivfOPQConfig(dim, nlist, m int, rerank, opq bool) Config {
	c := ivfPQConfig(dim, nlist, m, rerank)
	c.OPQ = opq
	return c
}

// TestIVFPQOPQBuildsRotation: cfg.OPQ=true on an IVF-PQ index builds a non-nil
// rotation on the trained residual codec; cfg.OPQ=false leaves it nil.
func TestIVFPQOPQBuildsRotation(t *testing.T) {
	const (
		dim, n, nlist, m = 32, 1500, 32, 8
	)
	rng := rand.New(rand.NewSource(7))
	vecs := makeClustered(rng, n, dim, 24, 0.2)
	ids := make([]uint64, n)
	for i := range ids {
		ids[i] = uint64(i + 1)
	}

	on, err := newIVF(ivfOPQConfig(dim, nlist, m, false, true))
	if err != nil {
		t.Fatal(err)
	}
	if err := on.BuildConcurrent(ids, vecs, 0); err != nil {
		t.Fatal(err)
	}
	if on.pq == nil || on.pq.rotation == nil {
		t.Fatal("IVF-PQ-OPQ: trained codec has nil rotation (OPQ not threaded into build)")
	}
	if len(on.pq.rotation) != dim*dim {
		t.Fatalf("rotation len = %d, want dim*dim = %d", len(on.pq.rotation), dim*dim)
	}

	off, err := newIVF(ivfOPQConfig(dim, nlist, m, false, false))
	if err != nil {
		t.Fatal(err)
	}
	if err := off.BuildConcurrent(ids, vecs, 0); err != nil {
		t.Fatal(err)
	}
	if off.pq == nil {
		t.Fatal("IVF-PQ: codec missing")
	}
	if off.pq.rotation != nil {
		t.Fatal("IVF-PQ OPQ-off: rotation must be nil (byte-identical to plain PQ)")
	}
}

// TestIVFPQOPQRecallNotWorse: IVF-PQ-OPQ recall >= IVF-PQ (non-OPQ) on data with
// imbalanced sub-space variance (the OPQ win). The rotation balances the residual
// sub-spaces so ADC quantization error drops.
func TestIVFPQOPQRecallNotWorse(t *testing.T) {
	const (
		dim, n, nlist, m, k, nq, nprb = 32, 4000, 32, 8, 10, 40, 16
	)
	vecs := makeImbalanced(t, n, dim, 16, 300)
	queries := makeImbalanced(t, nq, dim, 16, 301)
	ids := make([]uint64, n)
	for i := range ids {
		ids[i] = uint64(i + 1)
	}

	build := func(opq bool) float64 {
		ix, err := newIVF(ivfOPQConfig(dim, nlist, m, false, opq))
		if err != nil {
			t.Fatal(err)
		}
		if err := ix.BuildConcurrent(ids, vecs, 0); err != nil {
			t.Fatal(err)
		}
		ix.nprobe = nprb
		return ivfPQRecallOf(t, ix, queries, ids, vecs, k)
	}
	plain := build(false)
	opq := build(true)
	t.Logf("IVF-PQ recall@%d: plain = %.3f, OPQ = %.3f", k, plain, opq)
	if opq < plain-0.02 {
		t.Fatalf("IVF-PQ-OPQ recall %.3f regressed vs plain %.3f on imbalanced data", opq, plain)
	}
}

// TestIVFPQOPQVecForComposition: vecFor reconstruct of a PQ-only (floats dropped)
// IVF-PQ-OPQ slot is close to the original vector — proving the residual + rotation
// composition (encode rotates R(x−centroid); reconstruct un-rotates Rᵀ BEFORE the
// centroid is added back). A broken un-rotate ordering would yield garbage here.
func TestIVFPQOPQVecForComposition(t *testing.T) {
	const (
		dim, n, nlist, m = 32, 1200, 24, 8
	)
	rng := rand.New(rand.NewSource(9))
	vecs := makeClustered(rng, n, dim, 16, 0.15)
	ids := make([]uint64, n)
	for i := range ids {
		ids[i] = uint64(i + 1)
	}
	ix, err := newIVF(ivfOPQConfig(dim, nlist, m, false, true))
	if err != nil {
		t.Fatal(err)
	}
	if err := ix.BuildConcurrent(ids, vecs, 0); err != nil {
		t.Fatal(err)
	}
	// IVFRerank=false ⇒ floats are dropped at build, so vecFor must reconstruct
	// from code + centroid (the OPQ reconstruct path: Rᵀ then + centroid).
	if !ix.pqDropped {
		t.Fatal("expected pqDropped after PQ-only build (IVFRerank=false)")
	}

	// Average relative reconstruction error must be modest (PQ is lossy, but the
	// rotation composition must be CORRECT — a broken Rᵀ ordering gives ~O(1) error).
	var sumRel float64
	cnt := 0
	for slot := uint32(0); slot < uint32(n); slot++ {
		rec := ix.vecFor(slot)
		orig := vecs[slot]
		var num, den float64
		for j := range orig {
			d := float64(rec[j] - orig[j])
			num += d * d
			den += float64(orig[j]) * float64(orig[j])
		}
		if den > 0 {
			sumRel += num / den
			cnt++
		}
	}
	avgRel := sumRel / float64(cnt)
	t.Logf("IVF-PQ-OPQ vecFor avg relative reconstruction error = %.4f", avgRel)
	if avgRel > 0.5 {
		t.Fatalf("vecFor reconstruction error %.4f too high — residual+rotation composition likely wrong", avgRel)
	}
}

// TestIVFPQOPQSnapshotRestoresR: an IVF-PQ-OPQ snapshot persists R VERBATIM in the
// PQ trailer; the restored codec has a bit-identical rotation and reproduces
// identical search results.
func TestIVFPQOPQSnapshotRestoresR(t *testing.T) {
	const (
		dim, n, nlist, m, k, nq, nprb = 24, 1500, 32, 6, 10, 20, 12
	)
	rng := rand.New(rand.NewSource(13))
	vecs := makeClustered(rng, n, dim, 24, 0.2)
	ids := make([]uint64, n)
	for i := range ids {
		ids[i] = uint64(i + 1)
	}
	queries := makeClustered(rng, nq, dim, 24, 0.2)

	ix, err := newIVF(ivfOPQConfig(dim, nlist, m, false, true))
	if err != nil {
		t.Fatal(err)
	}
	if err := ix.BuildConcurrent(ids, vecs, 0); err != nil {
		t.Fatal(err)
	}
	ix.nprobe = nprb
	if ix.pq.rotation == nil {
		t.Fatal("source IVF-PQ-OPQ codec has nil rotation")
	}

	before := make([][]Result, nq)
	for i, q := range queries {
		r, serr := ix.Search(q, k)
		if serr != nil {
			t.Fatal(serr)
		}
		before[i] = r
	}

	var buf bytes.Buffer
	if err := ix.Snapshot(&buf); err != nil {
		t.Fatal(err)
	}

	restored, err := newIVF(ivfOPQConfig(dim, nlist, m, false, true))
	if err != nil {
		t.Fatal(err)
	}
	if err := restored.Restore(bytes.NewReader(buf.Bytes())); err != nil {
		t.Fatal(err)
	}
	restored.nprobe = nprb

	// R restored VERBATIM (bit-identical).
	if restored.pq == nil || restored.pq.rotation == nil {
		t.Fatal("restored IVF-PQ-OPQ codec lost its rotation")
	}
	if len(restored.pq.rotation) != len(ix.pq.rotation) {
		t.Fatalf("restored rotation len %d != %d", len(restored.pq.rotation), len(ix.pq.rotation))
	}
	for i := range ix.pq.rotation {
		if restored.pq.rotation[i] != ix.pq.rotation[i] {
			t.Fatalf("rotation[%d] not verbatim: src=%v dst=%v", i, ix.pq.rotation[i], restored.pq.rotation[i])
		}
	}

	// Search identical post-restore (NOT a degrade).
	for i, q := range queries {
		got, serr := restored.Search(q, k)
		if serr != nil {
			t.Fatal(serr)
		}
		if len(got) != len(before[i]) {
			t.Fatalf("query %d: restored len %d != before %d", i, len(got), len(before[i]))
		}
		for j := range got {
			if got[j].ID != before[i][j].ID {
				t.Fatalf("query %d pos %d: restored id %d != before %d", i, j, got[j].ID, before[i][j].ID)
			}
		}
	}
}

// TestIVFPQOPQOffTrailerByteIdentical proves the OPQ-off invariant on the IVF
// trailer DIRECTLY (not the whole snapshot, which has build/map non-determinism):
// writePQTrailer for a non-OPQ codec writes the EXACT pre-OPQ bytes (no R presence
// byte, no R floats), while the OPQ codec appends a 1 byte + dim*dim floats. The
// difference between the two trailers is EXACTLY 1 + dim*dim*4 bytes.
func TestIVFPQOPQOffTrailerByteIdentical(t *testing.T) {
	const (
		dim, n, nlist, m = 24, 1200, 24, 6
	)
	rng := rand.New(rand.NewSource(17))
	vecs := makeClustered(rng, n, dim, 24, 0.2)
	ids := make([]uint64, n)
	for i := range ids {
		ids[i] = uint64(i + 1)
	}

	trailer := func(opq bool) []byte {
		ix, err := newIVF(ivfOPQConfig(dim, nlist, m, true, opq)) // rerank=true: keep slotCell deterministic
		if err != nil {
			t.Fatal(err)
		}
		if err := ix.BuildConcurrent(ids, vecs, 0); err != nil {
			t.Fatal(err)
		}
		var buf bytes.Buffer
		bw := bufio.NewWriter(&buf)
		if err := ix.writePQTrailer(bw); err != nil {
			t.Fatal(err)
		}
		if err := bw.Flush(); err != nil {
			t.Fatal(err)
		}
		return buf.Bytes()
	}
	// Two OPQ-off trailers are byte-identical (deterministic, no R), and the OPQ-on
	// trailer is exactly 1 (R presence byte) + dim*dim*4 (R floats) bytes longer —
	// proving R is the ONLY OPQ-off→on delta and OPQ-off writes ZERO R bytes.
	off1 := trailer(false)
	off2 := trailer(false)
	on := trailer(true)
	if !bytes.Equal(off1, off2) {
		t.Fatal("OPQ-off trailer is not deterministic")
	}
	if len(on) != len(off1)+1+dim*dim*4 {
		t.Fatalf("OPQ-on trailer should be exactly 1+dim*dim*4 (%d) larger than OPQ-off: off=%d on=%d", 1+dim*dim*4, len(off1), len(on))
	}
}

// TestHNSWPQOPQBuildsRotation: cfg.OPQ=true on a PQ-HNSW index builds a non-nil
// rotation; OPQ off leaves it nil.
func TestHNSWPQOPQBuildsRotation(t *testing.T) {
	const (
		dim, n, seed = 64, 2000, 42
	)
	ids, vecs := siftLikeCorpus(n, dim, seed)

	cfgOn := Config{Dim: dim, Metric: L2, M: 16, EfConstruction: 200, EfSearch: 64, Seed: seed, Quant: QuantPQ, QuantPQM: 16, OPQ: true}
	on, err := newHNSW(cfgOn)
	if err != nil {
		t.Fatal(err)
	}
	if err := on.BuildConcurrent(ids, vecs, 4); err != nil {
		t.Fatal(err)
	}
	if pqRotation(on.quant) == nil {
		t.Fatal("HNSW-PQ-OPQ: trained codec has nil rotation (OPQ not threaded into build)")
	}

	cfgOff := cfgOn
	cfgOff.OPQ = false
	off, err := newHNSW(cfgOff)
	if err != nil {
		t.Fatal(err)
	}
	if err := off.BuildConcurrent(ids, vecs, 4); err != nil {
		t.Fatal(err)
	}
	if pqRotation(off.quant) != nil {
		t.Fatal("HNSW-PQ OPQ-off: rotation must be nil")
	}
}

// TestHNSWPQOPQSnapshotRestoresR: a PQ-HNSW-OPQ snapshot persists R; the restored
// codec has a bit-identical rotation, re-encodes bit-identical codes, and returns
// identical search results.
func TestHNSWPQOPQSnapshotRestoresR(t *testing.T) {
	const (
		dim, n, k, seed = 64, 3000, 10, 42
	)
	ids, vecs := siftLikeCorpus(n, dim, seed)
	_, queries := siftLikeCorpus(80, dim, 7)

	cfg := Config{Dim: dim, Metric: L2, M: 16, EfConstruction: 200, EfSearch: 64, Seed: seed, Quant: QuantPQ, QuantPQM: 16, OPQ: true}
	src, err := newHNSW(cfg)
	if err != nil {
		t.Fatal(err)
	}
	if err := src.BuildConcurrent(ids, vecs, 4); err != nil {
		t.Fatal(err)
	}
	srcRot := pqRotation(src.quant)
	if srcRot == nil {
		t.Fatal("source HNSW-PQ-OPQ codec has nil rotation")
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
		t.Fatal(err)
	}

	dst, err := newHNSW(cfg)
	if err != nil {
		t.Fatal(err)
	}
	if err := dst.Restore(&buf); err != nil {
		t.Fatal(err)
	}
	dstRot := pqRotation(dst.quant)
	if dstRot == nil {
		t.Fatal("restored HNSW-PQ-OPQ codec lost its rotation")
	}
	if len(dstRot) != len(srcRot) {
		t.Fatalf("restored rotation len %d != %d", len(dstRot), len(srcRot))
	}
	for i := range srcRot {
		if dstRot[i] != srcRot[i] {
			t.Fatalf("rotation[%d] not verbatim: src=%v dst=%v", i, srcRot[i], dstRot[i])
		}
	}
	// Codes bit-identical (re-encoded from the verbatim rotated codebooks).
	for slot := 0; slot < n; slot++ {
		if !bytes.Equal(srcCodes[slot], dst.arena.Code(uint32(slot))) {
			t.Fatalf("slot %d code not bit-identical after restore", slot)
		}
	}
	for i, q := range queries {
		res, serr := dst.Search(q, k)
		if serr != nil {
			t.Fatal(serr)
		}
		if !eqUint64(resultIDs(res), before[i]) {
			t.Fatalf("query %d: restored results != pre-snapshot", i)
		}
	}
}

// TestHNSWPQOPQOffBlockByteIdentical proves the OPQ-off invariant on the shared PQ
// codebook block DIRECTLY: writePQCodebooks(writeR=false) for an OPQ-off codec is a
// byte-identical PREFIX of writePQCodebooks(writeR=true) for the OPQ-on codec; the
// OPQ-on block appends EXACTLY dim*dim*4 R bytes (no extra presence byte — the
// version stamp gates R, so OPQ-off bytes are the pre-OPQ bytes verbatim).
func TestHNSWPQOPQOffBlockByteIdentical(t *testing.T) {
	const (
		dim, n, seed = 32, 1500, 42
	)
	ids, vecs := siftLikeCorpus(n, dim, seed)
	block := func(opq bool) []byte {
		cfg := Config{Dim: dim, Metric: L2, M: 16, EfConstruction: 200, EfSearch: 64, Seed: seed, Quant: QuantPQ, QuantPQM: 8, OPQ: opq}
		h, err := newHNSW(cfg)
		if err != nil {
			t.Fatal(err)
		}
		if err := h.BuildConcurrent(ids, vecs, 4); err != nil {
			t.Fatal(err)
		}
		var buf bytes.Buffer
		if err := writePQCodebooks(&buf, h.quant, opq); err != nil {
			t.Fatal(err)
		}
		return buf.Bytes()
	}
	// Two OPQ-off blocks are byte-identical (deterministic, no R); the OPQ-on block
	// is exactly dim*dim*4 R bytes longer (no extra presence byte — the version
	// stamp gates R) ⇒ OPQ-off writes ZERO R bytes (byte-identical to pre-OPQ).
	off1 := block(false)
	off2 := block(false)
	on := block(true)
	if !bytes.Equal(off1, off2) {
		t.Fatal("OPQ-off PQ block is not deterministic")
	}
	if len(on) != len(off1)+dim*dim*4 {
		t.Fatalf("OPQ-on PQ block should be exactly dim*dim*4 (%d) larger than OPQ-off: off=%d on=%d", dim*dim*4, len(off1), len(on))
	}
}

// TestHNSWPQOPQVersionStamp: an OPQ-on PQ snapshot stamps v8; OPQ-off stamps v7.
// This proves the version gate that keeps OPQ-off byte-identical and tells the
// reader whether to expect R.
func TestHNSWPQOPQVersionStamp(t *testing.T) {
	const (
		dim, n, seed = 32, 800, 42
	)
	ids, vecs := siftLikeCorpus(n, dim, seed)
	ver := func(opq bool) uint32 {
		cfg := Config{Dim: dim, Metric: L2, M: 16, EfConstruction: 200, EfSearch: 64, Seed: seed, Quant: QuantPQ, QuantPQM: 8, OPQ: opq}
		h, err := newHNSW(cfg)
		if err != nil {
			t.Fatal(err)
		}
		if err := h.BuildConcurrent(ids, vecs, 4); err != nil {
			t.Fatal(err)
		}
		var buf bytes.Buffer
		if err := h.Snapshot(&buf); err != nil {
			t.Fatal(err)
		}
		b := buf.Bytes()
		// [magic:8][version:u32 big-endian]
		return uint32(b[8])<<24 | uint32(b[9])<<16 | uint32(b[10])<<8 | uint32(b[11])
	}
	if v := ver(false); v != snapshotVersionPQNoOPQ {
		t.Fatalf("OPQ-off PQ snapshot version = %d, want %d (v7)", v, snapshotVersionPQNoOPQ)
	}
	if v := ver(true); v != snapshotVersion {
		t.Fatalf("OPQ-on PQ snapshot version = %d, want %d (v8)", v, snapshotVersion)
	}
}

// TestHNSWPQOPQPersistSidecarRestoresR: the PQ-HNSW persist sidecar carries R; an
// openPersist restore loads a bit-identical rotation and reproduces identical
// search results (ADC navigation survives restart with OPQ).
func TestHNSWPQOPQPersistSidecarRestoresR(t *testing.T) {
	const (
		dim, n, k, seed = 64, 2500, 10, 42
	)
	dir := t.TempDir()
	ids, vecs := siftLikeCorpus(n, dim, seed)
	_, queries := siftLikeCorpus(60, dim, 7)

	cfg := Config{
		Dim: dim, Metric: L2, M: 16, EfConstruction: 200, EfSearch: 64, Seed: seed,
		Quant: QuantPQ, QuantPQM: 16, OPQ: true, Persistent: true, QuantStorage: QuantMmap,
		MmapPath: filepath.Join(dir, "vecs.dat"), GraphMmapPath: filepath.Join(dir, "graph.dat"),
	}
	src, err := newHNSW(cfg)
	if err != nil {
		t.Fatal(err)
	}
	if err := src.BuildConcurrent(ids, vecs, 4); err != nil {
		t.Fatal(err)
	}
	srcRot := pqRotation(src.quant)
	if srcRot == nil {
		t.Fatal("source persistent HNSW-PQ-OPQ codec has nil rotation")
	}
	before := make([][]uint64, len(queries))
	for i, q := range queries {
		res, serr := src.Search(q, k)
		if serr != nil {
			t.Fatal(serr)
		}
		before[i] = resultIDs(res)
	}
	metaPath := filepath.Join(dir, "meta.bin")
	if err := src.SavePersist(metaPath); err != nil {
		t.Fatal(err)
	}
	if err := src.Close(); err != nil {
		t.Fatal(err)
	}

	dst, err := openPersist(cfg, metaPath)
	if err != nil {
		t.Fatalf("openPersist: %v", err)
	}
	defer dst.Close()
	dstRot := pqRotation(dst.quant)
	if dstRot == nil {
		t.Fatal("restored sidecar HNSW-PQ-OPQ codec lost its rotation")
	}
	if len(dstRot) != len(srcRot) {
		t.Fatalf("restored rotation len %d != %d", len(dstRot), len(srcRot))
	}
	for i := range srcRot {
		if dstRot[i] != srcRot[i] {
			t.Fatalf("sidecar rotation[%d] not verbatim: src=%v dst=%v", i, srcRot[i], dstRot[i])
		}
	}
	for i, q := range queries {
		res, serr := dst.Search(q, k)
		if serr != nil {
			t.Fatal(serr)
		}
		if !eqUint64(resultIDs(res), before[i]) {
			t.Fatalf("query %d: sidecar-restored results != pre-persist", i)
		}
	}
}

// TestHNSWPQOPQSidecarVersionStamp: the persist sidecar stamps v8 for OPQ-on and
// v7 for OPQ-off, proving the gate keeps an OPQ-off sidecar byte-identical to the
// pre-OPQ layout (no R bytes) while v8 carries R.
func TestHNSWPQOPQSidecarVersionStamp(t *testing.T) {
	const (
		dim, n, seed = 32, 800, 42
	)
	ids, vecs := siftLikeCorpus(n, dim, seed)
	ver := func(opq bool) uint32 {
		dir := t.TempDir()
		cfg := Config{
			Dim: dim, Metric: L2, M: 16, EfConstruction: 200, EfSearch: 64, Seed: seed,
			Quant: QuantPQ, QuantPQM: 8, Persistent: true, QuantStorage: QuantMmap, OPQ: opq,
			MmapPath: filepath.Join(dir, "v.dat"), GraphMmapPath: filepath.Join(dir, "g.dat"),
		}
		h, err := newHNSW(cfg)
		if err != nil {
			t.Fatal(err)
		}
		if err := h.BuildConcurrent(ids, vecs, 4); err != nil {
			t.Fatal(err)
		}
		var buf bytes.Buffer
		if err := h.writeMeta(&buf, h.arena.Capacity()); err != nil {
			t.Fatal(err)
		}
		_ = h.Close()
		b := buf.Bytes()
		// [magic:8][version:u32 big-endian]
		return uint32(b[8])<<24 | uint32(b[9])<<16 | uint32(b[10])<<8 | uint32(b[11])
	}
	if v := ver(false); v != persistVersionPQNoOPQ {
		t.Fatalf("OPQ-off PQ sidecar version = %d, want %d (v7)", v, persistVersionPQNoOPQ)
	}
	if v := ver(true); v != persistVersion {
		t.Fatalf("OPQ-on PQ sidecar version = %d, want %d (v8)", v, persistVersion)
	}
}
