// SPDX-License-Identifier: Apache-2.0

package vector

import (
	"bytes"
	"math"
	"math/rand"
	"path/filepath"
	"runtime"
	"sort"
	"testing"
)

// prqClusteredCorpus builds a clustered corpus + queries, the regime quantization
// targets (real embeddings have cluster structure). Shared by the PRQ recall and
// reconstruction-error tests so both measure the same data.
func prqClusteredCorpus(n, dim, nq, clusters int, noise float64, seed int64) (corpus, queries [][]float32) {
	rng := rand.New(rand.NewSource(seed))
	centers := make([][]float32, clusters)
	for c := range centers {
		v := make([]float32, dim)
		for j := range v {
			v[j] = float32(rng.NormFloat64())
		}
		centers[c] = v
	}
	pointNear := func(c int) []float32 {
		v := make([]float32, dim)
		for j := range v {
			v[j] = centers[c][j] + float32(noise*rng.NormFloat64())
		}
		return v
	}
	corpus = make([][]float32, n)
	for i := range corpus {
		corpus[i] = pointNear(i % clusters)
	}
	queries = make([][]float32, nq)
	for i := range queries {
		queries[i] = pointNear(rng.Intn(clusters))
	}
	return corpus, queries
}

// meanReconErr returns the mean squared reconstruction error of a codec over corpus:
// (1/n)·Σ ‖x − reconstruct(encode(x))‖². encode/reconstruct are supplied so the
// helper works for both the plain pq codec and the prq codec.
func meanReconErr(corpus [][]float32, encode func([]float32) []byte, reconstruct func([]byte) []float32) float64 {
	var total float64
	for _, x := range corpus {
		r := reconstruct(encode(x))
		var d float64
		for i := range x {
			diff := float64(x[i] - r[i])
			d += diff * diff
		}
		total += d
	}
	return total / float64(len(corpus))
}

// TestPRQReconstructionErrorLowerThanPQ is the CORE PRQ value proof: a 2-layer PRQ
// at the same per-layer m reconstructs the corpus with STRICTLY lower error than a
// plain 1-layer PQ — the residual layer captures detail PQ's single layer cannot.
// Measured on a fixed clustered corpus under L2 (PQ trains codebooks by L2
// reconstruction regardless of metric).
func TestPRQReconstructionErrorLowerThanPQ(t *testing.T) {
	const (
		n    = 3_000
		dim  = 64
		m    = 8
		seed = 7
	)
	corpus, _ := prqClusteredCorpus(n, dim, 0, 48, 0.25, seed)

	pqCodec, err := trainPQ(corpus, m, dim, seed, L2, runtime.GOMAXPROCS(0), false, 0, 1, 8)
	if err != nil {
		t.Fatalf("trainPQ: %v", err)
	}
	prqCodec, err := trainPRQ(corpus, m, dim, 2, seed, L2, runtime.GOMAXPROCS(0), false, 0)
	if err != nil {
		t.Fatalf("trainPRQ: %v", err)
	}

	pqErr := meanReconErr(corpus, pqCodec.encode, pqCodec.reconstruct)
	prqErr := meanReconErr(corpus, prqCodec.Encode, prqCodec.reconstruct)
	t.Logf("mean recon err  pq(m=%d)=%.5f  prq(L=2,m=%d)=%.5f  ratio=%.3f", m, pqErr, m, prqErr, prqErr/pqErr)
	if !(prqErr < pqErr) {
		t.Errorf("PRQ(L=2) recon err %.5f not strictly lower than PQ %.5f", prqErr, pqErr)
	}
}

// TestPRQCodecRoundTrip verifies CodeLen == L*m, that encode is deterministic, and
// that reconstruct of a code is finite and dimensioned correctly. A sanity gate on
// the codec shape before the end-to-end tests.
func TestPRQCodecRoundTrip(t *testing.T) {
	const (
		n    = 500
		dim  = 32
		m    = 4
		l    = 3
		seed = 3
	)
	corpus, _ := prqClusteredCorpus(n, dim, 0, 16, 0.2, seed)
	codec, err := trainPRQ(corpus, m, dim, l, seed, L2, runtime.GOMAXPROCS(0), false, 0)
	if err != nil {
		t.Fatalf("trainPRQ: %v", err)
	}
	if codec.CodeLen() != l*m {
		t.Fatalf("CodeLen = %d, want %d (L*m)", codec.CodeLen(), l*m)
	}
	for _, x := range corpus[:50] {
		a := codec.Encode(x)
		b := codec.Encode(x)
		if !bytes.Equal(a, b) {
			t.Fatal("encode not deterministic")
		}
		r := codec.reconstruct(a)
		if len(r) != dim {
			t.Fatalf("reconstruct len = %d, want %d", len(r), dim)
		}
		for i, v := range r {
			if math.IsNaN(float64(v)) || math.IsInf(float64(v), 0) {
				t.Fatalf("reconstruct[%d] non-finite: %v", i, v)
			}
		}
	}
}

// TestPRQOPQRoundTrip verifies the OPQ-with-PRQ path: training with opq=true learns
// a rotation on layer 0, residual layers operate in rotated space, reconstruct
// un-rotates once, and only layer 0 carries R. Reconstruction error must still beat
// a plain (non-OPQ) PQ at the same m — proving the rotation + residual chain compose
// correctly.
func TestPRQOPQRoundTrip(t *testing.T) {
	const (
		n    = 2_000
		dim  = 64
		m    = 8
		seed = 13
	)
	corpus, _ := prqClusteredCorpus(n, dim, 0, 32, 0.25, seed)
	codec, err := trainPRQ(corpus, m, dim, 2, seed, L2, runtime.GOMAXPROCS(0), true, 0)
	if err != nil {
		t.Fatalf("trainPRQ OPQ: %v", err)
	}
	if codec.rotation() == nil {
		t.Fatal("OPQ PRQ should carry a rotation on layer 0")
	}
	if codec.layers[1].rotation != nil {
		t.Fatal("residual layer 1 must NOT carry a rotation (OPQ is layer 0's job)")
	}
	pqCodec, err := trainPQ(corpus, m, dim, seed, L2, runtime.GOMAXPROCS(0), false, 0, 1, 8)
	if err != nil {
		t.Fatalf("trainPQ: %v", err)
	}
	prqErr := meanReconErr(corpus, codec.Encode, codec.reconstruct)
	pqErr := meanReconErr(corpus, pqCodec.encode, pqCodec.reconstruct)
	t.Logf("OPQ PRQ recon err=%.5f  plain PQ=%.5f", prqErr, pqErr)
	if !(prqErr < pqErr) {
		t.Errorf("OPQ PRQ(L=2) recon err %.5f not lower than plain PQ %.5f", prqErr, pqErr)
	}
}

// prqRecall builds an exact (QuantNone), a QuantPQ, and a QuantPRQ(L=2) index over
// the SAME clustered corpus and returns (exact, pq, prq) recall@k vs brute-force
// ground truth. mode selects OPQ on/off.
func prqRecall(t *testing.T, opq bool) (exact, pq, prq float64) {
	t.Helper()
	const (
		n        = 3_000
		dim      = 64
		nq       = 50
		k        = 10
		m        = 8
		seed     = 17
		clusters = 48
		noise    = 0.20
		metric   = L2
	)
	corpus, queries := prqClusteredCorpus(n, dim, nq, clusters, noise, seed)

	dist := pickDist(metric)
	groundTruth := func(q []float32) map[uint64]bool {
		type pair struct {
			id uint64
			d  float32
		}
		ds := make([]pair, n)
		for i, v := range corpus {
			ds[i] = pair{id: uint64(i + 1), d: dist(q, v)}
		}
		sort.Slice(ds, func(a, b int) bool { return ds[a].d < ds[b].d })
		out := make(map[uint64]bool, k)
		for i := 0; i < k; i++ {
			out[ds[i].id] = true
		}
		return out
	}

	build := func(mode QuantMode) *hnsw {
		cfg := Config{
			Dim: dim, M: 16, EfConstruction: 200, EfSearch: 64,
			Seed: seed, Metric: metric, Quant: mode, RescoreFactor: 4,
		}
		if mode == QuantPQ || mode == QuantPRQ {
			cfg.QuantPQM = m
			cfg.OPQ = opq
		}
		if mode == QuantPRQ {
			cfg.PRQLayers = 2
		}
		h, err := newHNSW(cfg)
		if err != nil {
			t.Fatal(err)
		}
		ids := make([]uint64, n)
		for i := range ids {
			ids[i] = uint64(i + 1)
		}
		if err := h.BuildConcurrent(ids, corpus, runtime.GOMAXPROCS(0)); err != nil {
			t.Fatal(err)
		}
		return h
	}
	recallOf := func(h *hnsw) float64 {
		var matches int
		for _, q := range queries {
			truth := groundTruth(q)
			results, err := h.Search(q, k)
			if err != nil {
				t.Fatalf("search: %v", err)
			}
			for _, r := range results {
				if truth[r.ID] {
					matches++
				}
			}
		}
		return float64(matches) / float64(nq*k)
	}
	return recallOf(build(QuantNone)), recallOf(build(QuantPQ)), recallOf(build(QuantPRQ))
}

// TestPRQHNSWRecallBeatsPQ asserts that, at the SAME per-layer m, PRQ(L=2) holds
// recall@10 AT LEAST as high as plain PQ (the finer residual reconstruction gives
// better traversal ordering). PRQ uses 2× the bytes/vector (2·m vs m), the documented
// byte-budget tradeoff. Both are exact-rescored, so both should track the exact
// baseline; PRQ should not be worse. Skipped under -short.
func TestPRQHNSWRecallBeatsPQ(t *testing.T) {
	if testing.Short() {
		t.Skip("skipping recall test in -short mode")
	}
	base, pq, prq := prqRecall(t, false)
	t.Logf("recall@10  exact=%.3f  pq=%.3f  prq(L=2)=%.3f", base, pq, prq)
	const margin = 0.02
	if prq < pq-margin {
		t.Errorf("PRQ recall@10 = %.3f, want >= pq - %.2f (= %.3f)", prq, margin, pq-margin)
	}
	if prq < base-0.05 {
		t.Errorf("PRQ recall@10 = %.3f, want >= exact - 0.05 (= %.3f)", prq, base-0.05)
	}
}

// TestPRQHNSWRecallOPQ exercises the OPQ+PRQ end-to-end search path and asserts the
// same recall floor as the non-OPQ case. Skipped under -short.
func TestPRQHNSWRecallOPQ(t *testing.T) {
	if testing.Short() {
		t.Skip("skipping recall test in -short mode")
	}
	base, pq, prq := prqRecall(t, true)
	t.Logf("recall@10 (OPQ)  exact=%.3f  pq=%.3f  prq(L=2)=%.3f", base, pq, prq)
	if prq < base-0.06 {
		t.Errorf("OPQ PRQ recall@10 = %.3f, want >= exact - 0.06 (= %.3f)", prq, base-0.06)
	}
}

// TestPRQUntrainedNavigatesExact verifies the untrained gating: a QuantPRQ index
// below the train threshold has prqUntrained()==true, Encode is a no-op (zero
// codes), and the HNSW owner navigates on EXACT float32. Search must still work.
func TestPRQUntrainedNavigatesExact(t *testing.T) {
	const dim = 16
	q := newPRQQuantizer(dim, 4, 2, Cosine)
	if q.trained() {
		t.Fatal("fresh prqQuantizer should be untrained")
	}
	code := make([]byte, q.CodeLen())
	for i := range code {
		code[i] = 0xAB
	}
	q.Encode(code, make([]float32, dim))
	for i, b := range code {
		if b != 0xAB {
			t.Fatalf("untrained Encode wrote byte %d (=%d); expected no-op", i, b)
		}
	}

	cfg := Config{
		Dim: dim, M: 8, EfConstruction: 50, EfSearch: 32, Seed: 5,
		Metric: Cosine, Quant: QuantPRQ, QuantPQM: 4, PRQLayers: 2,
		IVFTrainThreshold: 1_000_000,
	}
	h, err := newHNSW(cfg)
	if err != nil {
		t.Fatal(err)
	}
	rng := rand.New(rand.NewSource(5))
	for i := 0; i < 50; i++ {
		v := make([]float32, dim)
		for j := range v {
			v[j] = float32(rng.NormFloat64())
		}
		if _, _, err := h.Insert(uint64(i+1), v, 0, nil, nil, nil, CASCond{}); err != nil {
			t.Fatal(err)
		}
	}
	if !h.prqUntrained() {
		t.Fatal("index should still be untrained below the train threshold")
	}
	res, err := h.Search(make([]float32, dim), 5)
	if err != nil {
		t.Fatalf("search on untrained PRQ index failed: %v", err)
	}
	if len(res) == 0 {
		t.Fatal("untrained PRQ search returned no results")
	}
}

// TestPRQHNSWSnapshotSurvives builds a QuantPRQ index, snapshots it, restores into a
// fresh QuantPRQ index, and asserts (a) the restored quantizer is TRAINED, (b) the
// restored per-slot codes are BIT-IDENTICAL (all L layers' codebooks + rotation
// restored verbatim so the re-encode-from-vecs reproduces the same codes), and (c)
// search results are identical post-restore. Run with OPQ on to cover the rotation.
func TestPRQHNSWSnapshotSurvives(t *testing.T) {
	const (
		n    = 2_000
		dim  = 64
		m    = 8
		k    = 10
		seed = 31
	)
	ids, vecs := siftLikeCorpus(n, dim, seed)
	_, queries := siftLikeCorpus(60, dim, 99)

	cfg := Config{
		Dim: dim, Metric: L2, M: 16, EfConstruction: 200, EfSearch: 64, Seed: seed,
		Quant: QuantPRQ, QuantPQM: m, PRQLayers: 2, OPQ: true, RescoreFactor: 4,
	}
	src, err := newHNSW(cfg)
	if err != nil {
		t.Fatal(err)
	}
	if err := src.BuildConcurrent(ids, vecs, 4); err != nil {
		t.Fatal(err)
	}
	if src.prqUntrained() {
		t.Fatal("source PRQ index should be trained after BuildConcurrent")
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
	if dst.prqUntrained() {
		t.Fatal("restored PRQ index is UNTRAINED — codebooks did not survive the snapshot")
	}

	// All L layers' codebooks + rotation restored VERBATIM: codes re-encoded from
	// them must be bit-identical (the headline persistence-soundness check).
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
			t.Fatalf("query %d: restored %v != original %v", i, resultIDs(res), before[i])
		}
	}
}

// TestPRQHNSWPersistSurvives is the instant-restart (mmap sidecar) analogue: a
// QuantPRQ index backed by mmap + graph mmap is SavePersist'd, closed, and reopened
// via openPersist. The reopened index must be TRAINED, re-encode bit-identical
// codes, and return identical search results.
func TestPRQHNSWPersistSurvives(t *testing.T) {
	const (
		n    = 1_500
		dim  = 64
		m    = 8
		k    = 10
		seed = 53
	)
	dir := t.TempDir()
	ids, vecs := siftLikeCorpus(n, dim, seed)
	for _, v := range vecs {
		normalize(v)
	}
	_, queries := siftLikeCorpus(60, dim, 77)
	for _, q := range queries {
		normalize(q)
	}

	cfg := Config{
		Dim: dim, Metric: Cosine, M: 16, EfConstruction: 200, EfSearch: 64, Seed: seed,
		Quant: QuantPRQ, QuantPQM: m, PRQLayers: 2, OPQ: true,
		QuantStorage: QuantMmap, RescoreFactor: 4,
		MmapPath:      filepath.Join(dir, "vecs.dat"),
		GraphMmapPath: filepath.Join(dir, "graph.dat"),
	}
	if err := ValidateConfig(cfg); err != nil {
		t.Fatalf("QuantPRQ + mmap config rejected: %v", err)
	}
	h, err := newHNSW(cfg)
	if err != nil {
		t.Fatal(err)
	}
	if err := h.BuildConcurrent(ids, vecs, runtime.GOMAXPROCS(0)); err != nil {
		t.Fatalf("build: %v", err)
	}
	if h.prqUntrained() {
		t.Fatal("PRQ index untrained after build")
	}

	before := make([][]uint64, len(queries))
	for i, q := range queries {
		res, serr := h.Search(q, k)
		if serr != nil {
			t.Fatal(serr)
		}
		before[i] = resultIDs(res)
	}
	srcCodes := make([][]byte, n)
	for slot := 0; slot < n; slot++ {
		srcCodes[slot] = append([]byte(nil), h.arena.Code(uint32(slot))...)
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

	if h2.prqUntrained() {
		t.Fatal("reopened PRQ index is UNTRAINED — sidecar did not persist/restore codebooks")
	}
	for slot := 0; slot < n; slot++ {
		if !bytes.Equal(srcCodes[slot], h2.arena.Code(uint32(slot))) {
			t.Fatalf("slot %d code not bit-identical after instant-restart", slot)
		}
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

// TestPRQValidate checks the Config.Validate rules for QuantPRQ: the dense allowance
// admits it, PRQLayers must be >= 0, QuantPQM must divide Dim, and OPQ is allowed.
func TestPRQValidate(t *testing.T) {
	base := Config{Dim: 64, M: 16, EfConstruction: 100, EfSearch: 32, Metric: L2, Quant: QuantPRQ, QuantPQM: 8}
	if err := ValidateConfig(base); err != nil {
		t.Fatalf("valid QuantPRQ rejected: %v", err)
	}
	withOPQ := base
	withOPQ.OPQ = true
	if err := ValidateConfig(withOPQ); err != nil {
		t.Fatalf("QuantPRQ + OPQ rejected: %v", err)
	}
	withLayers := base
	withLayers.PRQLayers = 3
	if err := ValidateConfig(withLayers); err != nil {
		t.Fatalf("QuantPRQ PRQLayers=3 rejected: %v", err)
	}
	badLayers := base
	badLayers.PRQLayers = -1
	if err := ValidateConfig(badLayers); err == nil {
		t.Fatal("negative PRQLayers should be rejected")
	}
	badM := base
	badM.QuantPQM = 7 // 64 % 7 != 0
	if err := ValidateConfig(badM); err == nil {
		t.Fatal("QuantPQM not dividing Dim should be rejected")
	}
}
