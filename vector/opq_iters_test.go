// SPDX-License-Identifier: Apache-2.0

package vector

import (
	"bytes"
	"math"
	"testing"
)

// TestOPQItersDeterministicReplicaOracle trains PQ TWICE with OPQIters=5 on the
// IDENTICAL slot-ordered sample and asserts BIT-IDENTICAL rotation + codebooks
// (every float32 compared via math.Float32bits). This proves IN-PROCESS
// reproducibility: same input, same binary, same call → same bits.
//
// Cross-replica determinism (two Raft replicas on different hosts) rests on the
// STRUCTURAL absence of any non-deterministic source in the SVD+Procrustes path:
//   - No map iteration (Go maps have randomized range order).
//   - No random beyond the existing seeded iter-0 rotation.
//   - Fixed Jacobi pair order (i<j ascending) and fixed sweep COUNT (jacobiSweeps=30),
//     never a float-threshold convergence test that could stop at a different sweep
//     due to float rounding.
//   - Slot-ordered M accumulation in procrustesRotation (single accumulator/cell).
//
// In-process perturbations (e.g. a tiny epsilon added then subtracted to M) wash
// out below float32 resolution after SVD + Gram-Schmidt + float32 narrowing, so
// this test cannot detect summation-order non-determinism at the bit level. Cross-
// replica safety therefore relies on the structural absence above, not on this test.
func TestOPQItersDeterministicReplicaOracle(t *testing.T) {
	const dim, m, iters = 32, 8, 5
	vecs := makeImbalanced(t, 3000, dim, 16, 314)

	p1, err := trainPQ(vecs, m, dim, 9, L2, 1, true, iters, 1, 8)
	if err != nil {
		t.Fatal(err)
	}
	p2, err := trainPQ(vecs, m, dim, 9, L2, 1, true, iters, 1, 8)
	if err != nil {
		t.Fatal(err)
	}

	// Rotation bit-identical.
	if len(p1.rotation) != len(p2.rotation) {
		t.Fatalf("rotation length mismatch: %d vs %d", len(p1.rotation), len(p2.rotation))
	}
	for i := range p1.rotation {
		if math.Float32bits(p1.rotation[i]) != math.Float32bits(p2.rotation[i]) {
			t.Fatalf("rotation[%d] NOT bit-identical: %v vs %v (replica determinism broken)", i, p1.rotation[i], p2.rotation[i])
		}
	}
	// Codebooks bit-identical.
	assertCodebooksBitIdentical(t, p1, p2)
}

func assertCodebooksBitIdentical(t *testing.T, a, b *pq) {
	t.Helper()
	if len(a.codebooks) != len(b.codebooks) {
		t.Fatalf("codebook M mismatch: %d vs %d", len(a.codebooks), len(b.codebooks))
	}
	for s := range a.codebooks {
		if len(a.codebooks[s]) != len(b.codebooks[s]) {
			t.Fatalf("subspace %d sub-centroid count mismatch: %d vs %d", s, len(a.codebooks[s]), len(b.codebooks[s]))
		}
		for c := range a.codebooks[s] {
			for i := range a.codebooks[s][c] {
				if math.Float32bits(a.codebooks[s][c][i]) != math.Float32bits(b.codebooks[s][c][i]) {
					t.Fatalf("codebook[%d][%d][%d] NOT bit-identical: %v vs %v", s, c, i, a.codebooks[s][c][i], b.codebooks[s][c][i])
				}
			}
		}
	}
}

// TestOPQItersLEOneByteIdentical is the #1 no-break: OPQIters=0 and OPQIters=1 must
// produce a rotation + codebooks IDENTICAL to OPQ-without-the-field (the v1 path:
// trainPQ with opq=true and the OLD single-rotation behavior, which the existing
// suite pins). The short-circuit in trainPQ guarantees this by CONSTRUCTION (no SVD
// is called for OPQIters<=1).
func TestOPQItersLEOneByteIdentical(t *testing.T) {
	const dim, m = 32, 8
	vecs := makeImbalanced(t, 2500, dim, 12, 271)

	// The v1 reference: opq=true with the pre-OPQIters single-rotation path. Since
	// OPQIters<=1 short-circuits to exactly that path, both 0 and 1 must match it.
	ref, err := trainPQ(vecs, m, dim, 13, L2, 1, true, 0, 1, 8)
	if err != nil {
		t.Fatal(err)
	}
	for _, iters := range []int{0, 1} {
		got, err := trainPQ(vecs, m, dim, 13, L2, 1, true, iters, 1, 8)
		if err != nil {
			t.Fatal(err)
		}
		if len(got.rotation) != len(ref.rotation) {
			t.Fatalf("OPQIters=%d rotation length mismatch", iters)
		}
		for i := range ref.rotation {
			if math.Float32bits(got.rotation[i]) != math.Float32bits(ref.rotation[i]) {
				t.Fatalf("OPQIters=%d rotation[%d] differs from v1 path: %v vs %v (NOT byte-identical)", iters, i, got.rotation[i], ref.rotation[i])
			}
		}
		assertCodebooksBitIdentical(t, got, ref)
	}
}

// reconErr returns the mean ‖x − reconstruct(encode(x))‖ over the sample.
func reconErr(p *pq, vecs [][]float32) float64 {
	var sum float64
	for _, v := range vecs {
		r := p.reconstruct(p.Encode(v))
		sum += math.Sqrt(float64(l2Squared(v, r)))
	}
	return sum / float64(len(vecs))
}

// TestOPQItersReducesReconstructionError is the QUALITY + TRANSPOSE-DIRECTION guard
// for full-OPQ: it proves that the Procrustes rotation R = V·Uᵀ (NOT the wrong
// U·Vᵀ) is used, AND that iterative refinement materially improves reconstruction
// error. Two independent seeds are tested so that a transpose-direction regression
// fails on at least one (seed 555 is insensitive at this magnitude threshold; seed
// 1234 is the known discriminator — U·Vᵀ goes WORSE there).
//
// The MAGNITUDE assertion (e5 < e1*0.99) is what catches a subtle bug: with the
// WRONG transpose (U·Vᵀ) the improvement is below 1% or negative. With the CORRECT
// R = V·Uᵀ both seeds show ≥1% improvement across 5 iters.
func TestOPQItersReducesReconstructionError(t *testing.T) {
	const dim, m = 32, 8

	cases := []struct {
		name     string
		dataSeed int64
		pqSeed   int64
		n        int
	}{
		// seed 555: the original discriminator (correct direction improves >1%).
		{name: "seed555", dataSeed: 555, pqSeed: 17, n: 4000},
		// seed 1234: the known discriminator found during review — U·Vᵀ makes this
		// WORSE (error rises), V·Uᵀ reduces it. Both must pass for the direction to
		// be correct.
		{name: "seed1234", dataSeed: 1234, pqSeed: 19, n: 4000},
	}

	for _, tc := range cases {

		t.Run(tc.name, func(t *testing.T) {
			vecs := makeImbalanced(t, tc.n, dim, 16, tc.dataSeed)

			p1, err := trainPQ(vecs, m, dim, tc.pqSeed, L2, 1, true, 1, 1, 8)
			if err != nil {
				t.Fatal(err)
			}
			p5, err := trainPQ(vecs, m, dim, tc.pqSeed, L2, 1, true, 5, 1, 8)
			if err != nil {
				t.Fatal(err)
			}
			e1 := reconErr(p1, vecs)
			e5 := reconErr(p5, vecs)
			t.Logf("reconstruction error: OPQIters=1 -> %.5f, OPQIters=5 -> %.5f (improvement %.2f%%)",
				e1, e5, 100*(e1-e5)/e1)
			// MAGNITUDE guard: improvement must be ≥1%. A wrong transpose (U·Vᵀ)
			// either raises error or improves by less than 0.5% on these seeds.
			if !(e5 < e1*0.99) {
				t.Fatalf("full-OPQ did NOT reduce reconstruction error by ≥1%%: "+
					"iter=5 (%.5f) is not < iter=1 (%.5f) * 0.99 (%.5f) — "+
					"possible transpose direction bug (R=V·Uᵀ expected, not U·Vᵀ)",
					e5, e1, e1*0.99)
			}
		})
	}
}

// TestOPQItersValidation asserts fail-loud validation: OPQIters > maxOPQIters and
// OPQIters < 0 are rejected with ErrInvalidOPQIters; in-range values pass.
func TestOPQItersValidation(t *testing.T) {
	base := func() Config {
		c := DefaultConfig()
		c.Dim = 32
		c.Metric = L2
		c.Quant = QuantPQ
		c.OPQ = true
		return c
	}
	// Out of range high.
	c := base()
	c.OPQIters = maxOPQIters + 1
	if err := ValidateConfig(c); err != ErrInvalidOPQIters {
		t.Fatalf("OPQIters=%d: err = %v, want ErrInvalidOPQIters", c.OPQIters, err)
	}
	// Negative.
	c = base()
	c.OPQIters = -1
	if err := ValidateConfig(c); err != ErrInvalidOPQIters {
		t.Fatalf("OPQIters=-1: err = %v, want ErrInvalidOPQIters", err)
	}
	// In-range (incl. the boundary and the default 0).
	for _, v := range []int{0, 1, 5, maxOPQIters} {
		c = base()
		c.OPQIters = v
		if err := ValidateConfig(c); err != nil {
			t.Fatalf("OPQIters=%d should be valid, got %v", v, err)
		}
	}
}

// TestOPQItersHNSWHonored is the HNSW-PQ smoke + persist round-trip: a BuildConcurrent
// HNSW-PQ index with OPQ + OPQIters>1 trains a refined rotation, snapshots, and
// restores it BIT-IDENTICALLY (the refined R rides the existing snapshot/sidecar R
// slot — no new persist surface). Codes are bit-identical after restore.
func TestOPQItersHNSWHonored(t *testing.T) {
	const (
		n    = 3000
		dim  = 64
		seed = 42
	)
	ids, vecs := siftLikeCorpus(n, dim, seed)

	cfg := Config{Dim: dim, Metric: L2, M: 16, EfConstruction: 200, EfSearch: 64, Seed: seed, Quant: QuantPQ, QuantPQM: 16, OPQ: true, OPQIters: 4}
	src, err := newHNSW(cfg)
	if err != nil {
		t.Fatal(err)
	}
	if err := src.BuildConcurrent(ids, vecs, 4); err != nil {
		t.Fatal(err)
	}
	sq := src.quant.(*pqQuantizer)
	if sq.codec == nil || sq.codec.rotation == nil {
		t.Fatal("HNSW-PQ OPQ trained a nil rotation")
	}
	srcRot := append([]float32(nil), sq.codec.rotation...)
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
	dq := dst.quant.(*pqQuantizer)
	if dq.codec == nil || dq.codec.rotation == nil {
		t.Fatal("restored HNSW-PQ OPQ rotation is nil (refined R did not persist)")
	}
	// The refined rotation persists + restores BIT-IDENTICALLY.
	if len(dq.codec.rotation) != len(srcRot) {
		t.Fatalf("restored rotation length mismatch: %d vs %d", len(dq.codec.rotation), len(srcRot))
	}
	for i := range srcRot {
		if math.Float32bits(dq.codec.rotation[i]) != math.Float32bits(srcRot[i]) {
			t.Fatalf("restored rotation[%d] NOT bit-identical: %v vs %v", i, dq.codec.rotation[i], srcRot[i])
		}
	}
	// Codes bit-identical after restore (re-encoded from the verbatim R + codebooks).
	for slot := 0; slot < n; slot++ {
		if !bytes.Equal(srcCodes[slot], dst.arena.Code(uint32(slot))) {
			t.Fatalf("slot %d code not bit-identical after restore", slot)
		}
	}
}

// TestOPQItersIVFHonored is the IVF-PQ smoke: an IVF-PQ index with OPQ + OPQIters>1
// builds, trains a refined residual rotation, and searches without error. Confirms
// the IVF caller threads cfg.OPQIters into trainPQ (residual codebooks).
func TestOPQItersIVFHonored(t *testing.T) {
	const (
		n    = 4000
		dim  = 32
		seed = 7
	)
	ids, vecs := siftLikeCorpus(n, dim, seed)
	cfg := Config{
		Dim: dim, Metric: L2, M: 16, EfConstruction: 200, EfSearch: 64, Seed: seed,
		IndexType: IndexIVF, IVFNlist: 32, IVFNprobe: 8, IVFPQ: true, IVFPQM: 8, OPQ: true, OPQIters: 4,
	}
	ix, err := newIVF(cfg)
	if err != nil {
		t.Fatal(err)
	}
	if err := ix.BuildConcurrent(ids, vecs, 4); err != nil {
		t.Fatal(err)
	}
	if ix.pq == nil || ix.pq.rotation == nil {
		t.Fatal("IVF-PQ OPQ trained a nil residual rotation")
	}
	_, qs := siftLikeCorpus(20, dim, 11)
	for _, q := range qs {
		if _, err := ix.Search(q, 10); err != nil {
			t.Fatalf("IVF-PQ OPQIters search: %v", err)
		}
	}
}

// TestOPQItersMVHonored is the named/MV smoke: an MV collection whose inner index is
// HNSW-PQ with OPQ + OPQIters>1 inherits the knob via innerConfig() and trains a
// refined rotation, then searches without error.
func TestOPQItersMVHonored(t *testing.T) {
	const (
		dim  = 32
		seed = 21
	)
	cfg := MultiVectorConfig{
		Dim: dim, M: 16, EfConstruction: 200, EfSearch: 64, Seed: seed,
		Quant: QuantPQ, IVFTrainThreshold: 256, OPQ: true, OPQIters: 4,
	}
	if got := mvInnerConfig(cfg).OPQIters; got != 4 {
		t.Fatalf("MV innerConfig did not inherit OPQIters: got %d want 4", got)
	}
	mv, err := NewMultiVectorIndex(cfg)
	if err != nil {
		t.Fatal(err)
	}
	// Add enough docs to cross the train threshold so the inner PQ auto-trains.
	_, vecs := siftLikeCorpus(600, dim, seed)
	for i := 0; i < 300; i++ {
		toks := [][]float32{vecs[2*i], vecs[2*i+1]}
		if err := mv.Add(uint64(i+1), toks, nil); err != nil {
			t.Fatal(err)
		}
	}
	if _, err := mv.Search([][]float32{vecs[0], vecs[1]}, 10, MultiSearchOpts{CandidatesPerToken: 100}); err != nil {
		t.Fatalf("MV OPQIters search: %v", err)
	}
}
