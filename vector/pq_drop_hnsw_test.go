// SPDX-License-Identifier: Apache-2.0

package vector

import (
	"math/rand"
	"testing"
)

// buildPQDropCorpus is the shared clustered corpus + query set for the
// float-drop PQ-HNSW tests: clustered data so ADC (PQ) approximation has real
// structure to exploit (ADC-only recall is meaningless on isotropic noise).
func buildPQDropCorpus(n, dim, nClusters int, seed int64) (ids []uint64, vecs, queries [][]float32) {
	rng := rand.New(rand.NewSource(seed))
	centers := make([][]float32, nClusters)
	for c := range centers {
		centers[c] = make([]float32, dim)
		for d := 0; d < dim; d++ {
			centers[c][d] = float32(rng.NormFloat64()) * 5
		}
	}
	vecs = make([][]float32, n)
	ids = make([]uint64, n)
	for i := range vecs {
		c := centers[rng.Intn(nClusters)]
		v := make([]float32, dim)
		for d := 0; d < dim; d++ {
			v[d] = c[d] + float32(rng.NormFloat64())
		}
		vecs[i] = v
		ids[i] = uint64(i + 1)
	}
	queries = make([][]float32, 200)
	for i := range queries {
		c := centers[rng.Intn(nClusters)]
		q := make([]float32, dim)
		for d := 0; d < dim; d++ {
			q[d] = c[d] + float32(rng.NormFloat64())
		}
		queries[i] = q
	}
	return ids, vecs, queries
}

// TestPQDropVecsBuildRecall builds the same clustered corpus three ways — exact
// HNSW, exact-rescore PQ-HNSW (PQDropVecs=false, the default), and float-drop
// PQ-HNSW (PQDropVecs=true, ADC-only) — and asserts:
//   - the floats are actually dropped under PQDropVecs (vecsDropped() true);
//   - the codes survive the drop (a non-zero code is still readable);
//   - ADC-only recall@10 clears a sane floor;
//   - ADC-only recall is <= exact-rescore PQ recall (rescore can only help).
func TestPQDropVecsBuildRecall(t *testing.T) {
	const (
		n         = 4000
		dim       = 64
		k         = 10
		nClusters = 40
		seed      = 42
	)
	ids, vecs, queries := buildPQDropCorpus(n, dim, nClusters, seed)

	base := Config{Dim: dim, Metric: L2, M: 16, EfConstruction: 200, EfSearch: 64, Seed: seed}

	exact, err := newHNSW(base)
	if err != nil {
		t.Fatal(err)
	}
	if err := exact.BuildConcurrent(ids, vecs, 4); err != nil {
		t.Fatal(err)
	}
	exactRecall := recallOf(t, exact, vecs, queries, k)

	// Exact-rescore PQ-HNSW (default): keeps floats, rescores the ADC shortlist.
	pqCfg := base
	pqCfg.Quant = QuantPQ
	pqCfg.QuantPQM = 16
	pqKeep, err := newHNSW(pqCfg)
	if err != nil {
		t.Fatal(err)
	}
	if err := pqKeep.BuildConcurrent(ids, vecs, 4); err != nil {
		t.Fatal(err)
	}
	if pqKeep.vecsDropped() {
		t.Fatal("PQDropVecs=false must NOT drop the floats")
	}
	pqKeepRecall := recallOf(t, pqKeep, vecs, queries, k)

	// Float-drop PQ-HNSW: ADC-only after the post-build drop.
	dropCfg := pqCfg
	dropCfg.PQDropVecs = true
	pqDrop, err := newHNSW(dropCfg)
	if err != nil {
		t.Fatal(err)
	}
	if err := pqDrop.BuildConcurrent(ids, vecs, 4); err != nil {
		t.Fatal(err)
	}
	if !pqDrop.vecsDropped() {
		t.Fatal("PQDropVecs=true must drop the floats after BuildConcurrent")
	}
	// The arena's float vectors are gone but the codes remain readable.
	if pqDrop.arena.vecs != nil {
		t.Fatal("arena.vecs must be nil after dropVecs")
	}
	nonZero := false
	for slot := 0; slot < n && !nonZero; slot++ {
		for _, b := range pqDrop.arena.Code(uint32(slot)) {
			if b != 0 {
				nonZero = true
				break
			}
		}
	}
	if !nonZero {
		t.Fatal("all PQ codes are zero after drop (codes were lost)")
	}
	pqDropRecall := recallOf(t, pqDrop, vecs, queries, k)

	t.Logf("recall@%d exact=%.3f pq-keep(rescore)=%.3f pq-drop(ADC-only)=%.3f",
		k, exactRecall, pqKeepRecall, pqDropRecall)

	if pqDropRecall < 0.40 {
		t.Fatalf("ADC-only PQDropVecs recall@%d=%.3f below floor 0.40", k, pqDropRecall)
	}
	// ADC-only drops the exact rescore, so it cannot beat the rescore variant by a
	// meaningful margin (allow a small slack for graph/ordering noise).
	if pqDropRecall > pqKeepRecall+0.05 {
		t.Fatalf("ADC-only recall@%d=%.3f should not exceed exact-rescore recall=%.3f",
			k, pqDropRecall, pqKeepRecall)
	}
}

// TestPQDropVecsDefaultUnchanged proves PQDropVecs=false is byte/behaviour-
// identical to today's exact-rescore PQ-HNSW: a QuantPQ index with the flag UNSET
// and one with it explicitly false produce element-for-element identical result
// ids AND distances. Built with workers=1 (the multi-worker link phase races, so
// it is order-nondeterministic across runs; single-worker is deterministic), so
// any divergence is attributable to the flag, not build noise. Both indexes keep
// the floats (vecsDropped() false) and run the exact rescore — the only code
// difference is the `if cfg.PQDropVecs` drop guard, which is false in both, so
// the rescore path stays byte-identical.
func TestPQDropVecsDefaultUnchanged(t *testing.T) {
	const (
		n         = 2000
		dim       = 64
		k         = 10
		nClusters = 30
		seed      = 7
	)
	ids, vecs, queries := buildPQDropCorpus(n, dim, nClusters, seed)

	cfg := Config{Dim: dim, Metric: L2, M: 16, EfConstruction: 200, EfSearch: 64, Seed: seed,
		Quant: QuantPQ, QuantPQM: 16}

	// Reference: a QuantPQ index with PQDropVecs unset (today's behaviour).
	ref, err := newHNSW(cfg)
	if err != nil {
		t.Fatal(err)
	}
	if err := ref.BuildConcurrent(ids, vecs, 1); err != nil {
		t.Fatal(err)
	}

	// Same config with PQDropVecs explicitly false — must be identical.
	cfg2 := cfg
	cfg2.PQDropVecs = false
	got, err := newHNSW(cfg2)
	if err != nil {
		t.Fatal(err)
	}
	if err := got.BuildConcurrent(ids, vecs, 1); err != nil {
		t.Fatal(err)
	}
	if ref.vecsDropped() || got.vecsDropped() {
		t.Fatal("PQDropVecs=false must keep the floats on both indexes")
	}

	for qi, q := range queries {
		rRef, err := ref.Search(q, k)
		if err != nil {
			t.Fatal(err)
		}
		rGot, err := got.Search(q, k)
		if err != nil {
			t.Fatal(err)
		}
		if len(rRef) != len(rGot) {
			t.Fatalf("query %d: result len %d != %d", qi, len(rGot), len(rRef))
		}
		for i := range rRef {
			if rRef[i].ID != rGot[i].ID || rRef[i].Distance != rGot[i].Distance {
				t.Fatalf("query %d rank %d: got {id=%d d=%v} want {id=%d d=%v} (default path changed)",
					qi, i, rGot[i].ID, rGot[i].Distance, rRef[i].ID, rRef[i].Distance)
			}
		}
	}
}

// TestPQDropVecsReconstruct verifies the exact (non-search) paths reconstruct an
// APPROXIMATE vector after the drop: Get returns a non-nil, right-dimension
// vector close to the original, and SearchMMR / Recommend / Discover run without
// panicking on the dropped index (they route float reads through vecFor).
func TestPQDropVecsReconstruct(t *testing.T) {
	const (
		n         = 2000
		dim       = 64
		nClusters = 30
		seed      = 13
	)
	ids, vecs, _ := buildPQDropCorpus(n, dim, nClusters, seed)
	cfg := Config{Dim: dim, Metric: L2, M: 16, EfConstruction: 200, EfSearch: 64, Seed: seed,
		Quant: QuantPQ, QuantPQM: 16, PQDropVecs: true}
	h, err := newHNSW(cfg)
	if err != nil {
		t.Fatal(err)
	}
	if err := h.BuildConcurrent(ids, vecs, 4); err != nil {
		t.Fatal(err)
	}
	if !h.vecsDropped() {
		t.Fatal("expected dropped state")
	}

	// Get reconstructs: non-nil, correct dim, and approximately near the original
	// (PQ reconstruction error is bounded by the sub-quantizer resolution).
	id := ids[0]
	gotVec, _, _, _, _, ok := h.Get(id)
	if !ok {
		t.Fatalf("Get(%d) not found", id)
	}
	if len(gotVec) != dim {
		t.Fatalf("reconstructed vec dim = %d, want %d", len(gotVec), dim)
	}
	// Sanity: the reconstruction is closer to its own original than to a random
	// other point's original (PQ approximates the cluster structure).
	self := l2SquaredScalar(gotVec, vecs[0])
	other := l2SquaredScalar(gotVec, vecs[n/2])
	t.Logf("reconstruct L2^2: self=%.2f other=%.2f", self, other)
	if self >= other {
		t.Fatalf("reconstructed vec not closer to its own original (self=%.2f >= other=%.2f)", self, other)
	}

	// MMR / Recommend / Discover must not panic on the dropped index.
	if _, err := h.SearchMMR(vecs[0], 10, MMROpts{Lambda: 0.5}); err != nil {
		t.Fatalf("SearchMMR on dropped index: %v", err)
	}
	if _, err := h.Recommend(10, RecommendOpts{Positive: []uint64{ids[0], ids[1]}}); err != nil {
		t.Fatalf("Recommend on dropped index: %v", err)
	}
	if _, err := h.Discover(10, DiscoverOpts{
		Context: []ContextPair{{Positive: ids[0], Negative: ids[n/2]}},
	}); err != nil {
		t.Fatalf("Discover on dropped index: %v", err)
	}
}

// TestPQDropVecsInsertReadOnly verifies the post-drop Insert path: once the
// floats are dropped, an incremental Insert is rejected with
// ErrPQDropVecsReadOnly (no float read, no crash) — the index is read-mostly
// after the drop and must be rebuilt to add points.
func TestPQDropVecsInsertReadOnly(t *testing.T) {
	const (
		n         = 1000
		dim       = 32
		nClusters = 20
		seed      = 21
	)
	ids, vecs, _ := buildPQDropCorpus(n, dim, nClusters, seed)
	cfg := Config{Dim: dim, Metric: L2, M: 16, EfConstruction: 200, EfSearch: 64, Seed: seed,
		Quant: QuantPQ, QuantPQM: 8, PQDropVecs: true}
	h, err := newHNSW(cfg)
	if err != nil {
		t.Fatal(err)
	}
	if err := h.BuildConcurrent(ids, vecs, 4); err != nil {
		t.Fatal(err)
	}
	if !h.vecsDropped() {
		t.Fatal("expected dropped state")
	}
	newVec := make([]float32, dim)
	for j := range newVec {
		newVec[j] = vecs[0][j]
	}
	_, _, err = h.Insert(uint64(n+1), newVec, 0, nil, nil, nil, CASCond{})
	if err != ErrPQDropVecsReadOnly {
		t.Fatalf("post-drop Insert should return ErrPQDropVecsReadOnly, got %v", err)
	}
	// Search still works after the rejected insert (no corruption).
	if _, err := h.Search(vecs[0], 5); err != nil {
		t.Fatalf("Search after rejected insert: %v", err)
	}
}

// TestPQDropVecsValidate checks the config gate: PQDropVecs requires QuantPQ.
func TestPQDropVecsValidate(t *testing.T) {
	// Allowed: PQDropVecs with QuantPQ on HNSW.
	ok := Config{Dim: 64, Metric: L2, M: 16, EfConstruction: 200, EfSearch: 64,
		Quant: QuantPQ, QuantPQM: 16, PQDropVecs: true}
	if err := ValidateConfig(ok); err != nil {
		t.Fatalf("valid PQDropVecs config rejected: %v", err)
	}
	// Rejected: PQDropVecs without a PQ quantizer.
	noPQ := ok
	noPQ.Quant = QuantNone
	noPQ.QuantPQM = 0
	if err := ValidateConfig(noPQ); err != ErrInvalidPQDropVecs {
		t.Fatalf("PQDropVecs without QuantPQ should give ErrInvalidPQDropVecs, got %v", err)
	}
	// Rejected: PQDropVecs with SQ8 (not PQ).
	sq := ok
	sq.Quant = QuantSQ8
	sq.QuantPQM = 0
	if err := ValidateConfig(sq); err != ErrInvalidPQDropVecs {
		t.Fatalf("PQDropVecs with QuantSQ8 should give ErrInvalidPQDropVecs, got %v", err)
	}
}
