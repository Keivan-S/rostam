// SPDX-License-Identifier: Apache-2.0

package vector

import (
	"bytes"
	"math"
	"math/rand"
	"testing"
)

// trainPQ4 trains a 4-bit (LUT16) PQ codec on vecs for the codec-level tests.
func trainPQ4(t *testing.T, vecs [][]float32, m, dim int, seed int64, metric Metric) *pq4 {
	t.Helper()
	p, err := trainPQ(vecs, m, dim, seed, metric, 1, false, 0, 1, 4)
	if err != nil {
		t.Fatalf("trainPQ (4-bit): %v", err)
	}
	if p.codebookSize() != pqCodebookSize4 {
		t.Fatalf("4-bit codec codebookSize = %d, want %d", p.codebookSize(), pqCodebookSize4)
	}
	for s := range p.codebooks {
		if len(p.codebooks[s]) > pqCodebookSize4 {
			t.Fatalf("subspace %d has %d centroids, want <= %d", s, len(p.codebooks[s]), pqCodebookSize4)
		}
	}
	return &pq4{codec: p}
}

// referenceADC4 is an INDEPENDENT float ADC reference: it reconstructs the 4-bit
// code into its sub-centroids and computes the metric distance directly against the
// query (NOT via the codec's LUT). The scalar LUT16 ADC must match this within the
// uint8 LUT-quantization tolerance.
func referenceADC4(p *pq4, q, recon []float32) float32 {
	c := p.codec
	switch c.metric {
	case L2:
		return l2SquaredScalar(q, recon)
	case Cosine:
		return 1 - dotScalar(q, recon)
	default: // DotProduct
		return -dotScalar(q, recon)
	}
}

// TestPQ4CodeLenAndPacking checks the nibble-packing geometry: CodeLen == ceil(m/2),
// and a round-trip of subCodeAt recovers each subspace's index.
func TestPQ4CodeLenAndPacking(t *testing.T) {
	for _, m := range []int{1, 2, 3, 4, 7, 8, 16} {
		if got, want := pq4CodeLen(m), (m+1)/2; got != want {
			t.Fatalf("pq4CodeLen(%d)=%d want %d", m, got, want)
		}
	}
	// Pack a known nibble pattern and read it back.
	m := 7
	code := make([]byte, pq4CodeLen(m))
	want := []int{0, 15, 3, 12, 1, 8, 5}
	for s, idx := range want {
		nib := byte(idx & 0x0f)
		if s&1 == 0 {
			code[s>>1] |= nib
		} else {
			code[s>>1] |= nib << 4
		}
	}
	for s, exp := range want {
		if got := subCodeAt(code, s); got != exp {
			t.Fatalf("subCodeAt(s=%d)=%d want %d", s, got, exp)
		}
	}
}

// TestPQ4ScalarADCMatchesReference is the CORRECTNESS anchor: the scalar 4-bit ADC
// (quantized uint8 LUT + dequant) matches an independent reconstruct-and-dot
// reference within the documented LUT-quantization tolerance (<= m*scale/2).
func TestPQ4ScalarADCMatchesReference(t *testing.T) {
	const (
		n    = 1500
		dim  = 64
		m    = 16
		nq   = 40
		seed = 7
	)
	for _, metric := range []Metric{L2, Cosine, DotProduct} {
		vecs := makeCorpus(n, dim, seed)
		if metric == Cosine {
			for _, v := range vecs {
				normalize(v)
			}
		}
		p := trainPQ4(t, vecs, m, dim, seed, metric)
		rng := rand.New(rand.NewSource(seed + 1))
		var maxErr, maxTol float32
		for qi := 0; qi < nq; qi++ {
			q := make([]float32, dim)
			for j := range q {
				q[j] = float32(rng.NormFloat64())
			}
			if metric == Cosine {
				normalize(q)
			}
			lut := p.buildLUT16(q)
			// tolerance = m * scale / 2 (per-entry round-to-nearest over m subspaces).
			tol := float32(m) * lut.scale / 2
			for i := 0; i < 50; i++ {
				v := vecs[rng.Intn(n)]
				code := p.Encode(v)
				recon := p.reconstruct(code)
				got := lut.adcScalar(code)
				want := referenceADC4(p, q, recon)
				e := float32(math.Abs(float64(got - want)))
				if e > maxErr {
					maxErr = e
				}
				if tol > maxTol {
					maxTol = tol
				}
				if e > tol+1e-4 {
					t.Fatalf("metric=%v: scalar ADC %v vs reference %v (err %v > tol %v)", metric, got, want, e, tol)
				}
			}
		}
		t.Logf("metric=%v 4-bit scalar ADC: maxErr=%.5f maxTol(=m*scale/2)=%.5f", metric, maxErr, maxTol)
	}
}

// TestFastScanMatchesScalarADC asserts the ACTIVE fast-scan kernel (AVX2 on amd64,
// NEON on arm64, scalar elsewhere) reproduces the scalar LUT16 ADC EXACTLY (same
// uint8 table, same integer sums) across a range of m and partial-block sizes — the
// memory-safety / scalar-tail check (nVec not a multiple of the 32-lane block).
func TestFastScanMatchesScalarADC(t *testing.T) {
	const (
		n    = 600
		dim  = 64
		seed = 11
	)
	for _, m := range []int{1, 2, 8, 16, 32} {
		if dim%m != 0 {
			continue
		}
		vecs := makeCorpus(n, dim, seed)
		p := trainPQ4(t, vecs, m, dim, seed, L2)
		rng := rand.New(rand.NewSource(seed + 2))
		q := make([]float32, dim)
		for j := range q {
			q[j] = float32(rng.NormFloat64())
		}
		lut := p.buildLUT16(q)

		// Encode a batch of packed codes contiguously.
		codes := make([]byte, n*pq4CodeLen(m))
		for i := 0; i < n; i++ {
			c := p.Encode(vecs[i])
			copy(codes[i*pq4CodeLen(m):], c)
		}
		// Score every partial-block size 1..32 (and a few full blocks) to exercise the
		// scalar tail and the kernel's lane handling.
		for _, nVec := range []int{1, 2, 15, 16, 17, 31, 32} {
			if nVec > n {
				continue
			}
			dists := make([]float32, nVec)
			lut.fastScanBlockInto(dists, codes[:nVec*pq4CodeLen(m)], nVec, nil, nil)
			for i := 0; i < nVec; i++ {
				want := lut.adcScalar(codes[i*pq4CodeLen(m) : i*pq4CodeLen(m)+pq4CodeLen(m)])
				if dists[i] != want {
					t.Fatalf("m=%d nVec=%d lane %d: fastScan=%v scalarADC=%v (must be exact)", m, nVec, i, dists[i], want)
				}
			}
		}
	}
}

// TestFastScanKernelEquivalence directly compares the SCALAR kernel against the
// ACTIVE kernel (the AVX2/NEON swap when present) on random transposed blocks. On a
// scalar-only arch this is a no-op identity; on amd64/arm64 it is the AVX2==scalar /
// NEON==scalar equivalence guarantee the SIMD discipline requires.
func TestFastScanKernelEquivalence(t *testing.T) {
	rng := rand.New(rand.NewSource(99))
	for _, m := range []int{1, 3, 8, 16, 32, 64} {
		tbl := make([]uint8, m*pqCodebookSize4)
		for i := range tbl {
			tbl[i] = uint8(rng.Intn(256))
		}
		block := make([]byte, m*fastScanBlock)
		for i := range block {
			block[i] = byte(rng.Intn(16)) // nibble in [0,15]
		}
		for _, nVec := range []int{1, 7, 16, 31, 32} {
			wantOut := make([]uint16, fastScanBlock)
			gotOut := make([]uint16, fastScanBlock)
			fastScanScalar(tbl, m, block, wantOut, nVec)
			fastScanKernel(tbl, m, block, gotOut, nVec)
			for i := 0; i < nVec; i++ {
				if gotOut[i] != wantOut[i] {
					t.Fatalf("m=%d nVec=%d lane %d: kernel=%d scalar=%d", m, nVec, i, gotOut[i], wantOut[i])
				}
			}
		}
	}
}

// TestPQ4HNSWRecallWithRescore builds an 8-bit PQ-HNSW and a 4-bit PQ-HNSW over the
// same corpus and asserts the 4-bit index (lossier ADC) stays above an absolute
// recall floor thanks to the exact float32 rescore. Reports both numbers.
func TestPQ4HNSWRecallWithRescore(t *testing.T) {
	if testing.Short() {
		t.Skip("skipping recall test in -short mode")
	}
	const (
		n    = 4000
		dim  = 64
		k    = 10
		m    = 16
		seed = 42
	)
	ids, vecs := siftLikeCorpus(n, dim, seed)
	_, queries := siftLikeCorpus(200, dim, 7)

	// 4-bit PQ is a COARSE candidate-gen ADC (16 sub-centroids over dim/m=4 dims on
	// adversarial random-Gaussian data). The exact float32 rescore recovers ranking,
	// but a coarser ADC needs MORE over-collection to land the true top-k in the
	// rescored shortlist — exactly like the BQ1 recall test. RescoreFactor=16 is the
	// regime 4-bit targets; the headline is that rescore lifts 4-bit recall toward the
	// 8-bit baseline at the same m for HALF the code bytes.
	base := Config{Dim: dim, Metric: L2, M: 16, EfConstruction: 200, EfSearch: 128, Seed: seed, RescoreFactor: 16}

	build := func(nbits int) *hnsw {
		cfg := base
		cfg.Quant = QuantPQ
		cfg.QuantPQM = m
		cfg.PQNBits = nbits
		h, err := newHNSW(cfg)
		if err != nil {
			t.Fatal(err)
		}
		if err := h.BuildConcurrent(ids, vecs, 4); err != nil {
			t.Fatal(err)
		}
		return h
	}

	idx8 := build(8)
	idx4 := build(4)

	// The 4-bit index's code length must be half (ceil) the 8-bit one.
	if got, want := idx4.arena.codeLen, pq4CodeLen(m); got != want {
		t.Fatalf("4-bit arena codeLen = %d, want %d", got, want)
	}
	if got, want := idx8.arena.codeLen, m; got != want {
		t.Fatalf("8-bit arena codeLen = %d, want %d", got, want)
	}

	r8 := recallOf(t, idx8, vecs, queries, k)
	r4 := recallOf(t, idx4, vecs, queries, k)
	t.Logf("recall@%d  pq8=%.3f  pq4=%.3f (4-bit: half the code bytes; rescore@%dx recovers ranking)",
		k, r8, r4, base.RescoreFactor)
	if r4 < 0.80 {
		t.Fatalf("4-bit PQ-HNSW recall@%d=%.3f below absolute floor 0.80", k, r4)
	}
}

// TestPQ4SnapshotRestoreIdenticalCodes round-trips a 4-bit PQ-HNSW through the
// collection Snapshot/Restore and asserts the packed codes are BIT-IDENTICAL and the
// restored codec is 4-bit (not an 8-bit rebuild) and trained.
func TestPQ4SnapshotRestoreIdenticalCodes(t *testing.T) {
	const (
		n    = 2000
		dim  = 64
		k    = 10
		m    = 16
		seed = 42
	)
	ids, vecs := siftLikeCorpus(n, dim, seed)
	_, queries := siftLikeCorpus(80, dim, 7)

	cfg := Config{Dim: dim, Metric: L2, M: 16, EfConstruction: 200, EfSearch: 64, Seed: seed,
		Quant: QuantPQ, QuantPQM: m, PQNBits: 4, RescoreFactor: 4}
	src, err := newHNSW(cfg)
	if err != nil {
		t.Fatal(err)
	}
	if err := src.BuildConcurrent(ids, vecs, 4); err != nil {
		t.Fatal(err)
	}
	if src.pqUntrained() {
		t.Fatal("source 4-bit PQ index should be trained after BuildConcurrent")
	}
	if got := src.arena.codeLen; got != pq4CodeLen(m) {
		t.Fatalf("source 4-bit codeLen = %d, want %d", got, pq4CodeLen(m))
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
	dst, err := newHNSW(cfg)
	if err != nil {
		t.Fatal(err)
	}
	if err := dst.Restore(&buf); err != nil {
		t.Fatalf("Restore: %v", err)
	}
	if dst.pqUntrained() {
		t.Fatal("restored 4-bit PQ index is UNTRAINED — codebooks did not survive")
	}
	if got := dst.arena.codeLen; got != pq4CodeLen(m) {
		t.Fatalf("restored codeLen = %d, want %d (rebuilt as 8-bit?)", got, pq4CodeLen(m))
	}
	for slot := 0; slot < n; slot++ {
		if !bytes.Equal(srcCodes[slot], dst.arena.Code(uint32(slot))) {
			t.Fatalf("slot %d packed code not bit-identical after snapshot/restore", slot)
		}
	}
	for i, q := range queries {
		res, serr := dst.Search(q, k)
		if serr != nil {
			t.Fatal(serr)
		}
		if !eqUint64(resultIDs(res), before[i]) {
			t.Fatalf("query %d: restored %v != original %v", i, resultIDs(res), before[i])
		}
	}
}

// TestStoreSnapshotRaftPQ4RoundTrip is the GEOMETRY-ON-RESTORE guard for PQNBits: it
// drives a 4-bit PQ collection through the Raft store-snapshot path (SnapshotAll ->
// RestoreAll) and asserts the restored collection rebuilds AS 4-bit (codeLen ceil(m/2))
// with BIT-IDENTICAL packed codes. Without PQNBits on snapColCfg (+ both copy sites)
// the restore rebuilds as 8-bit (PQNBits 0 -> 8), mis-sizing the arena codes side-array
// and corrupting the nibble codes.
func TestStoreSnapshotRaftPQ4RoundTrip(t *testing.T) {
	const (
		n    = 2000
		dim  = 64
		k    = 10
		m    = 16 // non-default: defaultPQM(64) == 8
		seed = 53
	)
	ids, vecs := siftLikeCorpus(n, dim, seed)
	_, queries := siftLikeCorpus(40, dim, 99)

	cfg := Config{Dim: dim, Metric: L2, M: 16, EfConstruction: 200, EfSearch: 64, Seed: seed,
		Quant: QuantPQ, QuantPQM: m, PQNBits: 4, RescoreFactor: 4}

	src, err := OpenCollectionStore(t.TempDir())
	if err != nil {
		t.Fatal(err)
	}
	defer src.Close()
	if err := src.CreateCollection("pq4", cfg); err != nil {
		t.Fatal(err)
	}
	srcCol, ok := src.Get("pq4")
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
	if srcH.pqUntrained() {
		t.Fatal("source 4-bit PQ index should be trained after BuildStaged")
	}
	if got := srcH.arena.codeLen; got != pq4CodeLen(m) {
		t.Fatalf("source codeLen = %d, want %d", got, pq4CodeLen(m))
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

	dstCol, ok := dst.Get("pq4")
	if !ok {
		t.Fatal("collection missing after RestoreAll")
	}
	if dstCol.Config().PQNBits != 4 {
		t.Fatalf("restored PQNBits = %d, want 4 (snapColCfg dropped the field)", dstCol.Config().PQNBits)
	}
	dstH := dstCol.idx.(*hnsw)
	if dstH.pqUntrained() {
		t.Fatal("restored 4-bit PQ index is UNTRAINED — codebooks did not survive RestoreAll")
	}
	if got := dstH.arena.codeLen; got != pq4CodeLen(m) {
		t.Fatalf("restored codeLen = %d, want %d (rebuilt as 8-bit -> code mis-decode)", got, pq4CodeLen(m))
	}
	for _, id := range ids {
		slot, sok := dstH.arena.Slot(id)
		if !sok {
			t.Fatalf("id %d missing from restored arena", id)
		}
		if !bytes.Equal(srcCodes[id], dstH.arena.Code(slot)) {
			t.Fatalf("id %d packed code not bit-identical after RestoreAll", id)
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

// TestPQ4ValidateGate asserts PQNBits is gated to {0,4,8} and only for QuantPQ.
func TestPQ4ValidateGate(t *testing.T) {
	ok := Config{Dim: 64, Metric: L2, M: 16, EfConstruction: 200, EfSearch: 64, Quant: QuantPQ, QuantPQM: 16, PQNBits: 4}
	if err := ok.Validate(); err != nil {
		t.Fatalf("PQNBits=4 should pass Validate: %v", err)
	}
	bad := ok
	bad.PQNBits = 5
	if err := bad.Validate(); err == nil {
		t.Fatal("PQNBits=5 should be rejected by Validate")
	}
	eight := ok
	eight.PQNBits = 8
	if err := eight.Validate(); err != nil {
		t.Fatalf("PQNBits=8 should pass Validate: %v", err)
	}
}

// Benchmark4BitFastScanVs8BitScalar reports the candidate-gen ADC throughput of the
// 4-bit fast-scan (block-transposed VPSHUFB) against the 8-bit scalar ADC over the
// same number of database vectors.
func Benchmark4BitFastScanVs8BitScalar(b *testing.B) {
	const (
		n    = 4096
		dim  = 128
		m    = 32
		seed = 5
	)
	vecs := makeCorpus(n, dim, seed)
	q := makeCorpus(1, dim, 6)[0]

	// 8-bit codec + scalar ADC.
	p8, err := trainPQ(vecs, m, dim, seed, L2, 1, false, 0, 1, 8)
	if err != nil {
		b.Fatal(err)
	}
	lut8 := p8.queryLUT(q)
	codes8 := make([][]byte, n)
	for i := range codes8 {
		codes8[i] = p8.encode(vecs[i])
	}

	// 4-bit codec + fast-scan blocks.
	p4raw, err := trainPQ(vecs, m, dim, seed, L2, 1, false, 0, 1, 4)
	if err != nil {
		b.Fatal(err)
	}
	p4 := &pq4{codec: p4raw}
	lut4 := p4.buildLUT16(q)
	cl4 := pq4CodeLen(m)
	codes4 := make([]byte, n*cl4)
	for i := 0; i < n; i++ {
		copy(codes4[i*cl4:], p4.Encode(vecs[i]))
	}
	// PRE-TRANSPOSE the codes into fast-scan blocks ONCE (the realistic deployment:
	// the transposed layout is built at index time, then every query reuses it). The
	// per-query cost the bench measures is the VPSHUFB kernel + dequant, NOT the
	// one-time transpose.
	nBlocks := (n + fastScanBlock - 1) / fastScanBlock
	blocks := make([][]byte, nBlocks)
	for blk := 0; blk < nBlocks; blk++ {
		off := blk * fastScanBlock
		nVec := fastScanBlock
		if off+nVec > n {
			nVec = n - off
		}
		blk2 := make([]byte, m*fastScanBlock)
		transposeCodes(blk2, codes4[off*cl4:(off+nVec)*cl4], m, nVec)
		blocks[blk] = blk2
	}
	acc := make([]uint16, fastScanBlock)

	b.Run("8bit-scalar-adc", func(b *testing.B) {
		var sink float32
		for it := 0; it < b.N; it++ {
			for i := 0; i < n; i++ {
				sink += p8.adc(lut8, codes8[i])
			}
		}
		_ = sink
	})
	b.Run("4bit-fastscan", func(b *testing.B) {
		var sink float32
		for it := 0; it < b.N; it++ {
			for blk := 0; blk < nBlocks; blk++ {
				off := blk * fastScanBlock
				nVec := fastScanBlock
				if off+nVec > n {
					nVec = n - off
				}
				fastScanKernel(lut4.tbl, m, blocks[blk], acc, nVec)
				for i := 0; i < nVec; i++ {
					sink += lut4.dequant(uint32(acc[i]))
				}
			}
		}
		_ = sink
	})
}
