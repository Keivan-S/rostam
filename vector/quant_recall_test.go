// SPDX-License-Identifier: Apache-2.0

package vector

import (
	"math/rand"
	"sort"
	"testing"
)

// TestSQ8HNSWRecall builds two Cosine indexes over identical data — one exact
// float32, one SQ8-quantized — and verifies the quantized index stays within
// 5% recall@10 of the exact baseline. Because the rescore stage re-ranks the
// over-collected candidate set on exact float32, the two should agree closely;
// any gap comes only from the approximate graph traversal reaching slightly
// different candidates. Skipped under -short.
func TestSQ8HNSWRecall(t *testing.T) {
	if testing.Short() {
		t.Skip("skipping recall test in -short mode")
	}
	const (
		n      = 2_000
		dim    = 64
		nq     = 50
		k      = 10
		seed   = 42
		margin = 0.05
	)
	rng := rand.New(rand.NewSource(seed))

	corpus := make([][]float32, n)
	for i := range corpus {
		v := make([]float32, dim)
		for j := range v {
			v[j] = float32(rng.NormFloat64())
		}
		corpus[i] = v
	}
	queries := make([][]float32, nq)
	for i := range queries {
		q := make([]float32, dim)
		for j := range q {
			q[j] = float32(rng.NormFloat64())
		}
		queries[i] = q
	}

	build := func(mode QuantMode) *hnsw {
		h, err := newHNSW(Config{
			Dim: dim, M: 16, EfConstruction: 200, EfSearch: 64,
			Seed: seed, Metric: Cosine, Quant: mode, RescoreFactor: 3,
		})
		if err != nil {
			t.Fatal(err)
		}
		for i, v := range corpus {
			if _, _, err := h.Insert(uint64(i+1), v, 0, nil, nil, nil, CASCond{}); err != nil {
				t.Fatalf("insert %d: %v", i, err)
			}
		}
		return h
	}

	// Brute-force Cosine ground truth over normalized vectors (max dot product).
	groundTruth := func(q []float32) map[uint64]bool {
		qn := append([]float32(nil), q...)
		normalize(qn)
		type pair struct {
			id  uint64
			dot float32
		}
		dists := make([]pair, n)
		for i, v := range corpus {
			vn := append([]float32(nil), v...)
			normalize(vn)
			dists[i] = pair{id: uint64(i + 1), dot: dotScalar(qn, vn)}
		}
		sort.Slice(dists, func(a, b int) bool { return dists[a].dot > dists[b].dot })
		out := make(map[uint64]bool, k)
		for i := 0; i < k; i++ {
			out[dists[i].id] = true
		}
		return out
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

	base := recallOf(build(QuantNone))
	sq := recallOf(build(QuantSQ8))
	t.Logf("recall@%d  exact=%.3f  sq8=%.3f", k, base, sq)
	if sq < base-margin {
		t.Errorf("SQ8 recall@%d = %.3f, want >= exact - %.2f (= %.3f)", k, sq, margin, base-margin)
	}
}

// TestBQ1HNSWRecall builds a binary-quantized (BQ1) Cosine index and verifies
// that, despite navigating the graph on 1-bit-per-dimension Hamming codes, the
// exact float32 rescore stage recovers recall close to the exact baseline.
//
// The corpus is clustered (random cluster centers + small noise), which is the
// regime binary quantization targets: real embeddings have directional cluster
// structure, so sign-bit patterns are informative. On pure random Gaussian —
// the adversarial worst case — sign-bit Hamming is too coarse and BQ1 recall
// collapses; that is a known property of binary quantization, not a defect.
// Binary traversal is still coarser than SQ8, so it over-collects more
// candidates: at RescoreFactor=32 recall@10 reaches ~0.96 of the exact
// baseline on this corpus (the recall/over-collection curve is roughly
// 8->0.82, 16->0.89, 32->0.96, 64->1.00). Skipped under -short.
func TestBQ1HNSWRecall(t *testing.T) {
	if testing.Short() {
		t.Skip("skipping recall test in -short mode")
	}
	const (
		n        = 4_000
		dim      = 128
		nq       = 50
		k        = 10
		seed     = 7
		clusters = 64
		noise    = 0.15
		margin   = 0.10
	)
	rng := rand.New(rand.NewSource(seed))

	// Random unit-ish cluster centers.
	centers := make([][]float32, clusters)
	for c := range centers {
		v := make([]float32, dim)
		for j := range v {
			v[j] = float32(rng.NormFloat64())
		}
		normalize(v)
		centers[c] = v
	}
	// Corpus point = a cluster center + small Gaussian noise (left unnormalized;
	// the index normalizes for Cosine).
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

	groundTruth := func(q []float32) map[uint64]bool {
		qn := append([]float32(nil), q...)
		normalize(qn)
		type pair struct {
			id  uint64
			dot float32
		}
		dists := make([]pair, n)
		for i, v := range corpus {
			vn := append([]float32(nil), v...)
			normalize(vn)
			dists[i] = pair{id: uint64(i + 1), dot: dotScalar(qn, vn)}
		}
		sort.Slice(dists, func(a, b int) bool { return dists[a].dot > dists[b].dot })
		out := make(map[uint64]bool, k)
		for i := 0; i < k; i++ {
			out[dists[i].id] = true
		}
		return out
	}

	build := func(mode QuantMode, rescore int) *hnsw {
		h, err := newHNSW(Config{
			Dim: dim, M: 16, EfConstruction: 200, EfSearch: 64,
			Seed: seed, Metric: Cosine, Quant: mode, RescoreFactor: rescore,
		})
		if err != nil {
			t.Fatal(err)
		}
		for i, v := range corpus {
			if _, _, err := h.Insert(uint64(i+1), v, 0, nil, nil, nil, CASCond{}); err != nil {
				t.Fatalf("insert %d: %v", i, err)
			}
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

	base := recallOf(build(QuantNone, 0))
	bq := recallOf(build(QuantBQ1, 32))
	t.Logf("recall@%d  exact=%.3f  bq1=%.3f", k, base, bq)
	if bq < base-margin {
		t.Errorf("BQ1 recall@%d = %.3f, want >= exact - %.2f (= %.3f)", k, bq, margin, base-margin)
	}
}
