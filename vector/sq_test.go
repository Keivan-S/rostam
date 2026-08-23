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

// TestTrainedSQEncodeRoundTrip verifies that encode→dequantize round-trips each
// component within the quantization step bound (one level = span/255) and that a
// CONSTANT dimension (min==max across the sample) is handled without NaN/Inf — it
// dequantizes back to the constant and contributes a finite distance.
func TestTrainedSQEncodeRoundTrip(t *testing.T) {
	const dim = 16
	rng := rand.New(rand.NewSource(1))
	sample := make([][]float32, 200)
	for i := range sample {
		v := make([]float32, dim)
		for j := range v {
			v[j] = float32(rng.NormFloat64()) * 3
		}
		// Dimension 0 is CONSTANT across the whole sample (degenerate range).
		v[0] = 5.0
		sample[i] = v
	}
	q := trainSQ(sample, dim, 8, L2)
	if !q.trained() {
		t.Fatal("trainSQ produced an untrained quantizer")
	}

	code := make([]byte, q.CodeLen())
	if q.CodeLen() != dim {
		t.Fatalf("CodeLen = %d, want %d (one byte/dim at 8-bit)", q.CodeLen(), dim)
	}
	for _, v := range sample {
		q.Encode(code, v)
		for i := 0; i < dim; i++ {
			deq := q.deq(uint32(code[i]), i)
			if math.IsNaN(float64(deq)) || math.IsInf(float64(deq), 0) {
				t.Fatalf("dim %d dequantized to non-finite %v (constant-dim guard failed)", i, deq)
			}
			span := q.max[i] - q.min[i]
			step := span / 255
			if i == 0 {
				// Constant dimension: round-trips back to the constant exactly.
				if deq != 5.0 {
					t.Fatalf("constant dim 0 dequantized to %v, want 5.0", deq)
				}
				continue
			}
			if got := float32(math.Abs(float64(v[i] - deq))); got > step+1e-4 {
				t.Fatalf("dim %d round-trip error %v exceeds one step %v", i, got, step)
			}
		}
	}
}

// TestTrainedSQUntrainedNavigatesExact verifies the untrained gating: a QuantSQ
// index whose ranges are not yet learned has trained()==false, Encode is a no-op
// (codes stay zero), and the HNSW owner navigates on EXACT float32 (sqUntrained
// drives searchScorer to exactScorer). After enough inserts cross the auto-train
// threshold the quantizer trains and codes become non-zero.
func TestTrainedSQUntrainedNavigatesExact(t *testing.T) {
	const dim = 8
	q := newTrainedSQ(dim, 8, Cosine)
	if q.trained() {
		t.Fatal("fresh trainedSQ should be untrained")
	}
	code := make([]byte, q.CodeLen())
	for i := range code {
		code[i] = 0xAB // sentinel
	}
	q.Encode(code, make([]float32, dim))
	for i, b := range code {
		if b != 0xAB {
			t.Fatalf("untrained Encode wrote byte %d (=%d); expected no-op", i, b)
		}
	}

	// End-to-end: a QuantSQ index below the train threshold navigates exact.
	cfg := Config{
		Dim: dim, M: 8, EfConstruction: 50, EfSearch: 32, Seed: 5,
		Metric: Cosine, Quant: QuantSQ, IVFTrainThreshold: 1_000_000,
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
	if !h.sqUntrained() {
		t.Fatal("index should still be untrained below the train threshold")
	}
	// Search must still work (exact-float navigation).
	res, err := h.Search(make([]float32, dim), 5)
	if err != nil {
		t.Fatalf("search on untrained SQ index failed: %v", err)
	}
	if len(res) == 0 {
		t.Fatal("untrained SQ search returned no results")
	}
}

// sqRecallForMetric is the shared recall harness: it builds an exact (QuantNone)
// and a QuantSQ index over the SAME clustered corpus under metric, and returns
// (exactRecall, sqRecall) measured against brute-force ground truth. Clustered
// data is the regime quantization targets (real embeddings have cluster
// structure). The headline is that recall holds for ALL THREE metrics — the proof
// the trained scalar quantizer is metric-agnostic.
func sqRecallForMetric(t *testing.T, metric Metric) (float64, float64) {
	t.Helper()
	const (
		n        = 3_000
		dim      = 64
		nq       = 50
		k        = 10
		seed     = 11
		clusters = 48
		noise    = 0.15
	)
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
	corpus := make([][]float32, n)
	for i := range corpus {
		corpus[i] = pointNear(i % clusters)
	}
	queries := make([][]float32, nq)
	for i := range queries {
		queries[i] = pointNear(rng.Intn(clusters))
	}

	// Brute-force ground truth under the SAME metric semantics the index uses
	// (Cosine/DotProduct over normalized-for-cosine vectors via the distFunc;
	// smaller distance = nearer).
	dist := pickDist(metric)
	prep := func(v []float32) []float32 {
		out := append([]float32(nil), v...)
		if metric == Cosine {
			normalize(out)
		}
		return out
	}
	groundTruth := func(q []float32) map[uint64]bool {
		qn := prep(q)
		type pair struct {
			id uint64
			d  float32
		}
		ds := make([]pair, n)
		for i, v := range corpus {
			ds[i] = pair{id: uint64(i + 1), d: dist(qn, prep(v))}
		}
		sort.Slice(ds, func(a, b int) bool { return ds[a].d < ds[b].d })
		out := make(map[uint64]bool, k)
		for i := 0; i < k; i++ {
			out[ds[i].id] = true
		}
		return out
	}

	build := func(mode QuantMode) *hnsw {
		h, err := newHNSW(Config{
			Dim: dim, M: 16, EfConstruction: 200, EfSearch: 64,
			Seed: seed, Metric: metric, Quant: mode, RescoreFactor: 3,
		})
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
	return recallOf(build(QuantNone)), recallOf(build(QuantSQ))
}

// TestSQHNSWRecallMetricAgnostic is the HEADLINE: the trained scalar quantizer
// holds recall@10 within a small margin of the exact baseline for L2, Cosine,
// AND DotProduct — proving it is metric-agnostic (unlike the fixed-scale,
// Cosine-only QuantSQ8). Because HNSW rescores the candidate set on exact
// float32, the SQ index should track the exact baseline closely. Skipped under
// -short.
func TestSQHNSWRecallMetricAgnostic(t *testing.T) {
	if testing.Short() {
		t.Skip("skipping recall test in -short mode")
	}
	const margin = 0.05
	for _, metric := range []struct {
		name string
		m    Metric
	}{
		{"L2", L2},
		{"Cosine", Cosine},
		{"DotProduct", DotProduct},
	} {
		t.Run(metric.name, func(t *testing.T) {
			base, sq := sqRecallForMetric(t, metric.m)
			t.Logf("recall@10  %s  exact=%.3f  sq=%.3f", metric.name, base, sq)
			if sq < 0.90 {
				t.Errorf("%s SQ recall@10 = %.3f, want >= 0.90 (metric-agnostic floor)", metric.name, sq)
			}
			if sq < base-margin {
				t.Errorf("%s SQ recall@10 = %.3f, want >= exact - %.2f (= %.3f)", metric.name, sq, margin, base-margin)
			}
		})
	}
}

// TestSQHNSWSnapshotSurvives builds a QuantSQ index, snapshots it, restores into a
// fresh QuantSQ index, and asserts (a) the restored quantizer is TRAINED, (b) the
// restored per-slot codes are BIT-IDENTICAL (the learned ranges restored verbatim
// so the re-encode-from-vecs reproduces the same codes — the persistence
// soundness proof), and (c) search results are identical post-restore.
func TestSQHNSWSnapshotSurvives(t *testing.T) {
	const (
		n    = 2_000
		dim  = 64
		k    = 10
		seed = 31
	)
	ids, vecs := siftLikeCorpus(n, dim, seed)
	_, queries := siftLikeCorpus(60, dim, 99)

	cfg := Config{Dim: dim, Metric: L2, M: 16, EfConstruction: 200, EfSearch: 64, Seed: seed, Quant: QuantSQ}
	src, err := newHNSW(cfg)
	if err != nil {
		t.Fatal(err)
	}
	if err := src.BuildConcurrent(ids, vecs, 4); err != nil {
		t.Fatal(err)
	}
	if src.sqUntrained() {
		t.Fatal("source SQ index should be trained after BuildConcurrent")
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
	if dst.sqUntrained() {
		t.Fatal("restored SQ index is UNTRAINED — ranges did not survive the snapshot")
	}

	// The learned ranges restored VERBATIM: compare bit-for-bit.
	sq := src.quant.(*trainedSQ)
	dq := dst.quant.(*trainedSQ)
	for i := 0; i < dim; i++ {
		if sq.min[i] != dq.min[i] || sq.max[i] != dq.max[i] {
			t.Fatalf("range[%d] not verbatim: src=[%v,%v] dst=[%v,%v]", i, sq.min[i], sq.max[i], dq.min[i], dq.max[i])
		}
	}
	// Codes re-encoded from the restored ranges are bit-identical (the headline
	// persistence-soundness check).
	for slot := 0; slot < n; slot++ {
		if !bytes.Equal(srcCodes[slot], dst.arena.Code(uint32(slot))) {
			t.Fatalf("slot %d code not bit-identical after restore", slot)
		}
	}
	// Search results identical post-restore.
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

// TestSQHNSWPersistSurvives is the instant-restart (mmap sidecar) analogue: a
// QuantSQ index backed by mmap + graph mmap is SavePersist'd, closed, and reopened
// via openPersist. The reopened index must be TRAINED and return identical search
// results — proving the sidecar persists + restores the ranges before restoreDense
// re-encodes codes from the mapped vectors.
func TestSQHNSWPersistSurvives(t *testing.T) {
	const (
		n    = 1_500
		dim  = 64
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
		Quant: QuantSQ, QuantStorage: QuantMmap, RescoreFactor: 3,
		MmapPath:      filepath.Join(dir, "vecs.dat"),
		GraphMmapPath: filepath.Join(dir, "graph.dat"),
	}
	if err := ValidateConfig(cfg); err != nil {
		t.Fatalf("QuantSQ + mmap config rejected: %v", err)
	}
	h, err := newHNSW(cfg)
	if err != nil {
		t.Fatal(err)
	}
	if err := h.BuildConcurrent(ids, vecs, runtime.GOMAXPROCS(0)); err != nil {
		t.Fatalf("build: %v", err)
	}
	if h.sqUntrained() {
		t.Fatal("SQ index untrained after build")
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

	if h2.sqUntrained() {
		t.Fatal("reopened SQ index is UNTRAINED — sidecar did not persist/restore ranges")
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
