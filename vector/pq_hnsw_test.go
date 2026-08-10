// SPDX-License-Identifier: Apache-2.0

package vector

import (
	"math/rand"
	"testing"
)

// TestPQHNSWBuildRecall builds the same corpus as an exact HNSW and as a PQ-HNSW
// (QuantPQ, trained at BuildConcurrent), and asserts the PQ index's recall@10 is
// close to the exact index's. PQ-HNSW navigates the graph on ADC codes but
// exact-rescores the over-collected shortlist on the kept float vectors, so recall
// stays near exact (the rescore re-ranks on the true metric).
func TestPQHNSWBuildRecall(t *testing.T) {
	const (
		n    = 4000
		dim  = 64
		k    = 10
		seed = 42
	)
	ids, vecs := siftLikeCorpus(n, dim, seed)
	_, queries := siftLikeCorpus(200, dim, 7)

	base := Config{Dim: dim, Metric: L2, M: 16, EfConstruction: 200, EfSearch: 64, Seed: seed}

	exact, err := newHNSW(base)
	if err != nil {
		t.Fatal(err)
	}
	if err := exact.BuildConcurrent(ids, vecs, 4); err != nil {
		t.Fatal(err)
	}
	exactRecall := recallOf(t, exact, vecs, queries, k)

	pqCfg := base
	pqCfg.Quant = QuantPQ
	pqCfg.QuantPQM = 16 // 64 % 16 == 0
	pq, err := newHNSW(pqCfg)
	if err != nil {
		t.Fatal(err)
	}
	if err := pq.BuildConcurrent(ids, vecs, 4); err != nil {
		t.Fatal(err)
	}
	// The codec must be trained and codes populated after BuildConcurrent.
	pqq, ok := pq.quant.(*pqQuantizer)
	if !ok || !pqq.trained() {
		t.Fatalf("PQ codec not trained after BuildConcurrent (ok=%v)", ok)
	}
	// At least one slot must carry a non-zero code (training produced real codes).
	nonZero := false
	for slot := 0; slot < n && !nonZero; slot++ {
		for _, b := range pq.arena.Code(uint32(slot)) {
			if b != 0 {
				nonZero = true
				break
			}
		}
	}
	if !nonZero {
		t.Fatal("all PQ codes are zero after BuildConcurrent (encode did not run)")
	}

	pqRecall := recallOf(t, pq, vecs, queries, k)
	t.Logf("recall@%d exact=%.3f pq=%.3f", k, exactRecall, pqRecall)
	// Exact-rescore on kept floats keeps PQ recall close to exact. Allow a modest
	// gap from the ADC navigation (graph reaches a slightly different shortlist).
	if pqRecall < exactRecall-0.10 {
		t.Fatalf("PQ-HNSW recall@%d=%.3f too far below exact=%.3f", k, pqRecall, exactRecall)
	}
	if pqRecall < 0.80 {
		t.Fatalf("PQ-HNSW recall@%d=%.3f below absolute floor 0.80", k, pqRecall)
	}
}

// TestPQHNSWInsertBeforeBuild verifies the insert-before-train policy: an
// incremental Insert into a fresh QuantPQ index (no BuildConcurrent yet, so the
// codebooks are untrained) keeps the floats and answers search on EXACT float
// distance — it must not panic and must return correct nearest neighbors.
func TestPQHNSWInsertBeforeBuild(t *testing.T) {
	const (
		dim  = 32
		seed = 11
	)
	cfg := Config{Dim: dim, Metric: L2, M: 16, EfConstruction: 200, EfSearch: 64, Seed: seed, Quant: QuantPQ, QuantPQM: 8}
	h, err := newHNSW(cfg)
	if err != nil {
		t.Fatal(err)
	}
	if !h.pqUntrained() {
		t.Fatal("fresh QuantPQ index should report pqUntrained() == true")
	}
	rng := rand.New(rand.NewSource(seed))
	const n = 300
	vecs := make([][]float32, n)
	for i := 0; i < n; i++ {
		v := make([]float32, dim)
		for j := range v {
			v[j] = float32(rng.NormFloat64())
		}
		vecs[i] = v
		if _, _, err := h.Insert(uint64(i+1), v, 0, nil, nil, nil, CASCond{}); err != nil {
			t.Fatalf("insert %d before build: %v", i, err)
		}
	}
	// Search the corpus itself: each vector's own nearest neighbor must be itself,
	// since the untrained path scores exact float distance (no approximation).
	for i := 0; i < 50; i++ {
		res, err := h.Search(vecs[i], 1)
		if err != nil {
			t.Fatal(err)
		}
		if len(res) == 0 || res[0].ID != uint64(i+1) {
			t.Fatalf("insert-before-build exact search: query %d got %v, want self id %d", i, res, i+1)
		}
	}
}

// TestPQHNSWInsertAfterBuild verifies an incremental Insert AFTER BuildConcurrent
// (codebooks trained) encodes the new point's code and the index keeps returning
// correct results (ADC navigate + exact rescore).
func TestPQHNSWInsertAfterBuild(t *testing.T) {
	const (
		n    = 1000
		dim  = 32
		seed = 5
	)
	ids, vecs := siftLikeCorpus(n, dim, seed)
	cfg := Config{Dim: dim, Metric: L2, M: 16, EfConstruction: 200, EfSearch: 64, Seed: seed, Quant: QuantPQ, QuantPQM: 8}
	h, err := newHNSW(cfg)
	if err != nil {
		t.Fatal(err)
	}
	if err := h.BuildConcurrent(ids, vecs, 4); err != nil {
		t.Fatal(err)
	}
	if h.pqUntrained() {
		t.Fatal("index should be trained after BuildConcurrent")
	}
	// Insert a fresh point; must encode a (real) code and be findable as its own NN.
	newVec := make([]float32, dim)
	rng := rand.New(rand.NewSource(99))
	for j := range newVec {
		newVec[j] = float32(rng.NormFloat64())
	}
	const newID = 999999
	if _, _, err := h.Insert(newID, newVec, 0, nil, nil, nil, CASCond{}); err != nil {
		t.Fatal(err)
	}
	res, err := h.Search(newVec, 1)
	if err != nil {
		t.Fatal(err)
	}
	if len(res) == 0 || res[0].ID != newID {
		t.Fatalf("post-build insert: search got %v, want self id %d", res, newID)
	}
}

// TestPQHNSWConfigValidate checks the config gate: QuantPQ is allowed for dense
// (HNSW), dim%M==0 is enforced, and QuantPQ on a non-HNSW index type is rejected.
func TestPQHNSWConfigValidate(t *testing.T) {
	// Allowed: QuantPQ on HNSW with a divisor M.
	ok := Config{Dim: 64, Metric: Cosine, M: 16, EfConstruction: 200, EfSearch: 64, Quant: QuantPQ, QuantPQM: 16}
	if err := ok.Validate(); err != nil {
		t.Fatalf("valid QuantPQ HNSW config rejected: %v", err)
	}
	// Allowed: QuantPQM == 0 resolves to defaultPQM(dim).
	zeroM := ok
	zeroM.QuantPQM = 0
	if err := zeroM.Validate(); err != nil {
		t.Fatalf("QuantPQM=0 (defaultPQM) rejected: %v", err)
	}
	// Rejected: dim not divisible by M.
	badM := ok
	badM.Dim = 65 // 65 % 16 != 0
	if err := badM.Validate(); err != ErrInvalidQuantPQM {
		t.Fatalf("dim%%M!=0 should give ErrInvalidQuantPQM, got %v", err)
	}
	// Rejected: QuantPQ on IVF (IVF drives PQ via IVFPQ, never Quant==QuantPQ).
	onIVF := ok
	onIVF.IndexType = IndexIVF
	if err := onIVF.Validate(); err != ErrInvalidQuant {
		t.Fatalf("QuantPQ on IVF should give ErrInvalidQuant, got %v", err)
	}
}
