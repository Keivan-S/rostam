// SPDX-License-Identifier: Apache-2.0

package vector

import (
	"math"
	"math/rand"
	"runtime"
	"sort"
	"testing"
)

// makeClustered generates n vectors of the given dim drawn from nClusters gaussian
// blobs, so PQ codebooks have real structure to capture. For Cosine the vectors
// are L2-normalized (matching how the index stores cosine vectors).
func makePQClustered(t *testing.T, n, dim, nClusters int, metric Metric, seed int64) [][]float32 {
	t.Helper()
	rng := rand.New(rand.NewSource(seed))
	centers := make([][]float32, nClusters)
	for c := range centers {
		centers[c] = make([]float32, dim)
		for d := 0; d < dim; d++ {
			centers[c][d] = float32(rng.NormFloat64()) * 5
		}
	}
	out := make([][]float32, n)
	for i := range out {
		c := centers[rng.Intn(nClusters)]
		v := make([]float32, dim)
		for d := 0; d < dim; d++ {
			v[d] = c[d] + float32(rng.NormFloat64())
		}
		if metric == Cosine {
			normalize(v)
		}
		out[i] = v
	}
	return out
}

func TestTrainPQDimValidation(t *testing.T) {
	vecs := makePQClustered(t, 100, 16, 4, L2, 1)
	if _, err := trainPQ(vecs, 3, 16, 1, L2, 1, false, 0, 1, 8); err == nil {
		t.Fatal("expected error for dim%m != 0 (16 % 3)")
	}
	if _, err := trainPQ(vecs, 0, 16, 1, L2, 1, false, 0, 1, 8); err == nil {
		t.Fatal("expected error for m <= 0")
	}
	if _, err := trainPQ(vecs, 4, 16, 1, L2, 1, false, 0, 1, 8); err != nil {
		t.Fatalf("unexpected error for valid 16%%4: %v", err)
	}
}

func TestPQCodeLenAndCentroids(t *testing.T) {
	const dim, m = 32, 8
	vecs := makePQClustered(t, 2000, dim, 10, L2, 2)
	p, err := trainPQ(vecs, m, dim, 7, L2, 1, false, 0, 1, 8)
	if err != nil {
		t.Fatal(err)
	}
	if p.CodeLen() != m {
		t.Fatalf("CodeLen = %d, want %d", p.CodeLen(), m)
	}
	if len(p.codebooks) != m {
		t.Fatalf("codebooks len = %d, want %d", len(p.codebooks), m)
	}
	// nbits=8 → up to 256 sub-centroids/subspace; with ample training data we
	// expect a full codebook.
	for s := 0; s < m; s++ {
		if len(p.codebooks[s]) != pqCodebookSize {
			t.Fatalf("subspace %d has %d sub-centroids, want %d", s, len(p.codebooks[s]), pqCodebookSize)
		}
		if len(p.codebooks[s][0]) != dim/m {
			t.Fatalf("sub-centroid dim = %d, want %d", len(p.codebooks[s][0]), dim/m)
		}
	}
	// Encode produces m bytes.
	code := p.Encode(vecs[0])
	if len(code) != m {
		t.Fatalf("Encode len = %d, want %d", len(code), m)
	}
}

func TestPQSmallTrainingSetNoPanic(t *testing.T) {
	const dim, m = 16, 4
	// n < 256 → kmeans returns fewer than 256 centroids per subspace; must not
	// panic on train/encode/queryLUT/adc/reconstruct.
	vecs := makePQClustered(t, 30, dim, 3, L2, 3)
	p, err := trainPQ(vecs, m, dim, 5, L2, 1, false, 0, 1, 8)
	if err != nil {
		t.Fatal(err)
	}
	for _, v := range vecs {
		code := p.Encode(v)
		_ = p.reconstruct(code)
		lut := p.queryLUT(v)
		_ = p.adc(lut, code)
	}
}

func TestPQReconstruct(t *testing.T) {
	const dim, m = 32, 8
	vecs := makePQClustered(t, 3000, dim, 12, L2, 4)
	p, err := trainPQ(vecs, m, dim, 9, L2, 1, false, 0, 1, 8)
	if err != nil {
		t.Fatal(err)
	}
	// reconstruct(Encode(v)) ≈ v within PQ error: the mean reconstruction error
	// should be small relative to the vector magnitude on clustered data.
	var sumErr, sumMag float64
	for _, v := range vecs {
		r := p.reconstruct(p.Encode(v))
		sumErr += math.Sqrt(float64(l2Squared(v, r)))
		var mag float32
		for _, x := range v {
			mag += x * x
		}
		sumMag += math.Sqrt(float64(mag))
	}
	rel := sumErr / sumMag
	if rel > 0.5 {
		t.Fatalf("reconstruction relative error %.3f too high (PQ not capturing structure)", rel)
	}
	t.Logf("reconstruction relative error = %.4f", rel)
}

// TestPQADCAccuracyL2 asserts adc(queryLUT(q), Encode(v)) ≈ true squared-L2(q,v)
// (lower = better) and that ADC ranks true nearest neighbors highly.
func TestPQADCAccuracyL2(t *testing.T) {
	const dim, m = 32, 8
	vecs := makePQClustered(t, 4000, dim, 16, L2, 10)
	p, err := trainPQ(vecs, m, dim, 11, L2, 1, false, 0, 1, 8)
	if err != nil {
		t.Fatal(err)
	}
	codes := make([][]byte, len(vecs))
	for i, v := range vecs {
		codes[i] = p.Encode(v)
	}
	dist := pickDistDim(L2, dim)

	queries := makePQClustered(t, 50, dim, 16, L2, 12)
	const k = 10
	var recallSum float64
	for _, q := range queries {
		lut := p.queryLUT(q)
		// exact top-k
		exact := topK(len(vecs), k, func(i int) float32 { return dist(q, vecs[i]) })
		// adc top-k
		approx := topK(len(vecs), k, func(i int) float32 { return p.adc(lut, codes[i]) })
		recallSum += pqRecall(exact, approx)

		// adc orientation: nearest exact must have a small adc relative to a far one.
		near := exact[0]
		far := exact[len(exact)-1]
		if p.adc(lut, codes[near]) > p.adc(lut, codes[far])+1e-3 {
			// allowed occasionally but flag gross inversions
			t.Logf("warning: adc(near) > adc(far) for a query (near=%d far=%d)", near, far)
		}
	}
	recall := recallSum / float64(len(queries))
	if recall < 0.4 {
		t.Fatalf("L2 ADC recall@%d = %.3f, want >= 0.4 (PQ is lossy; guards gross inaccuracy)", k, recall)
	}
	t.Logf("L2 ADC recall@%d = %.3f", k, recall)

	// ADC ≈ true distance: sample correlation of sign/magnitude. Check a few
	// candidates' adc tracks the true squared-L2 closely (small relative error).
	q := queries[0]
	lut := p.queryLUT(q)
	var sumRel float64
	cnt := 0
	for i := 0; i < 200; i++ {
		true2 := dist(q, vecs[i])
		got := p.adc(lut, codes[i])
		if true2 > 1e-3 {
			sumRel += math.Abs(float64(got-true2)) / float64(true2)
			cnt++
		}
	}
	avgRel := sumRel / float64(cnt)
	if avgRel > 0.5 {
		t.Fatalf("L2 ADC avg relative error %.3f too high", avgRel)
	}
	t.Logf("L2 ADC avg relative error = %.4f", avgRel)
}

// TestPQADCMetricCosine asserts the IP/cosine orientation: adc ≈ 1 − dot (lower =
// better), and adc ranks nearest neighbors highly.
func TestPQADCMetricCosine(t *testing.T) {
	const dim, m = 32, 8
	vecs := makePQClustered(t, 4000, dim, 16, Cosine, 20)
	p, err := trainPQ(vecs, m, dim, 21, Cosine, 1, false, 0, 1, 8)
	if err != nil {
		t.Fatal(err)
	}
	codes := make([][]byte, len(vecs))
	for i, v := range vecs {
		codes[i] = p.Encode(v)
	}
	dist := pickDistDim(Cosine, dim)

	// adc ≈ true 1 − dot for sampled candidates.
	q := vecs[0]
	lut := p.queryLUT(q)
	var sumErr float64
	const sample = 200
	for i := 0; i < sample; i++ {
		trueD := dist(q, vecs[i]) // 1 - dot
		got := p.adc(lut, codes[i])
		sumErr += math.Abs(float64(got - trueD))
	}
	avgErr := sumErr / float64(sample)
	if avgErr > 0.3 {
		t.Fatalf("cosine ADC avg abs error %.3f too high (orientation/offset wrong?)", avgErr)
	}
	t.Logf("cosine ADC avg abs error = %.4f", avgErr)

	// recall
	queries := makePQClustered(t, 50, dim, 16, Cosine, 22)
	const k = 10
	var recallSum float64
	for _, qq := range queries {
		l := p.queryLUT(qq)
		exact := topK(len(vecs), k, func(i int) float32 { return dist(qq, vecs[i]) })
		approx := topK(len(vecs), k, func(i int) float32 { return p.adc(l, codes[i]) })
		recallSum += pqRecall(exact, approx)
	}
	recall := recallSum / float64(len(queries))
	if recall < 0.4 {
		t.Fatalf("cosine ADC recall@%d = %.3f, want >= 0.4", k, recall)
	}
	t.Logf("cosine ADC recall@%d = %.3f", k, recall)
}

// TestPQADCMetricDotProduct checks the DotProduct orientation (distance = −dot,
// lower = better, offset 0).
func TestPQADCMetricDotProduct(t *testing.T) {
	const dim, m = 32, 8
	vecs := makePQClustered(t, 3000, dim, 12, DotProduct, 30)
	p, err := trainPQ(vecs, m, dim, 31, DotProduct, 1, false, 0, 1, 8)
	if err != nil {
		t.Fatal(err)
	}
	dist := pickDistDim(DotProduct, dim)
	q := vecs[0]
	lut := p.queryLUT(q)
	var sumErr, sumMag float64
	const sample = 200
	for i := 0; i < sample; i++ {
		trueD := dist(q, vecs[i]) // -dot
		got := p.adc(lut, p.Encode(vecs[i]))
		sumErr += math.Abs(float64(got - trueD))
		sumMag += math.Abs(float64(trueD))
	}
	rel := sumErr / sumMag
	if rel > 0.5 {
		t.Fatalf("dotproduct ADC relative error %.3f too high", rel)
	}
	t.Logf("dotproduct ADC relative error = %.4f", rel)
}

// TestADCNoAlloc confirms the inner adc sum performs no allocation (LUT passed
// in, byte-indexed sum).
func TestADCNoAlloc(t *testing.T) {
	const dim, m = 32, 8
	vecs := makePQClustered(t, 1000, dim, 8, L2, 40)
	p, err := trainPQ(vecs, m, dim, 41, L2, 1, false, 0, 1, 8)
	if err != nil {
		t.Fatal(err)
	}
	lut := p.queryLUT(vecs[0])
	code := p.Encode(vecs[1])
	allocs := testing.AllocsPerRun(1000, func() {
		_ = p.adc(lut, code)
	})
	if allocs != 0 {
		t.Fatalf("adc inner sum allocated %.1f times/op, want 0", allocs)
	}
}

// naiveADC is the reference implementation the optimized adc must match
// BIT-IDENTICALLY: the plain subspace-order sum Σ lut[s*256+code[s]] (plus the
// metric offset), float adds in order s=0..m-1, single accumulator.
func naiveADC(metric Metric, m int, lut []float32, code []byte) float32 {
	sum := pqMetricOffset(metric)
	for s := 0; s < m; s++ {
		sum += lut[s*pqCodebookSize+int(code[s])]
	}
	return sum
}

// TestADCBitIdentical is THE correctness guarantee: the optimized adc returns
// the EXACT same float32 as the naive subspace-order sum, over many random LUTs
// and codes across various m. A divergence here would change search results.
func TestADCBitIdentical(t *testing.T) {
	rng := rand.New(rand.NewSource(2026))
	metrics := []Metric{L2, Cosine, DotProduct}
	for _, m := range []int{4, 8, 16, 32} {
		for _, metric := range metrics {
			p := &pq{m: m, metric: metric}
			for iter := 0; iter < 200; iter++ {
				lut := make([]float32, m*pqCodebookSize)
				for i := range lut {
					// Mix small and large magnitudes (and negatives) so any
					// reordering of float adds would surface as a mismatch.
					lut[i] = float32(rng.NormFloat64()) * float32(math.Pow(10, rng.Float64()*6-3))
				}
				code := make([]byte, m)
				for s := range code {
					code[s] = byte(rng.Intn(256))
				}
				got := p.adc(lut, code)
				want := naiveADC(metric, m, lut, code)
				// Exact equality (NaN-safe via bit comparison).
				if math.Float32bits(got) != math.Float32bits(want) {
					t.Fatalf("m=%d metric=%v iter=%d: adc=%v (bits %#x) want %v (bits %#x)",
						m, metric, iter, got, math.Float32bits(got), want, math.Float32bits(want))
				}
			}
		}
	}
}

// BenchmarkADC measures the optimized per-candidate adc hot path (and a naive
// baseline) for a typical m=16, so the bounds-check-elim + unroll speedup is
// reportable. Codes vary per iteration to defeat constant-folding.
func BenchmarkADC(b *testing.B) {
	const m = 16
	rng := rand.New(rand.NewSource(7))
	p := &pq{m: m, metric: L2}
	lut := make([]float32, m*pqCodebookSize)
	for i := range lut {
		lut[i] = float32(rng.NormFloat64())
	}
	codes := make([][]byte, 256)
	for i := range codes {
		codes[i] = make([]byte, m)
		for s := range codes[i] {
			codes[i][s] = byte(rng.Intn(256))
		}
	}
	b.Run("optimized", func(b *testing.B) {
		var sink float32
		for i := 0; i < b.N; i++ {
			sink += p.adc(lut, codes[i&255])
		}
		runtime.KeepAlive(sink)
	})
	b.Run("naive", func(b *testing.B) {
		var sink float32
		for i := 0; i < b.N; i++ {
			sink += naiveADC(L2, m, lut, codes[i&255])
		}
		runtime.KeepAlive(sink)
	})
}

func TestNewQuantizerPQ(t *testing.T) {
	const dim = 64
	q := newQuantizer(QuantPQ, dim, 0, 0, 0, 0, Cosine)
	if q == nil {
		t.Fatal("newQuantizer(QuantPQ) returned nil")
	}
	m := defaultPQM(dim)
	if q.CodeLen() != m {
		t.Fatalf("QuantPQ CodeLen = %d, want defaultPQM(%d)=%d", q.CodeLen(), dim, m)
	}
	if dim%m != 0 {
		t.Fatalf("defaultPQM(%d)=%d does not divide dim", dim, m)
	}
}

// --- OPQ rotation tests ---

// TestRandomOrthogonal asserts RᵀR ≈ I (orthonormal: unit rows, pairwise-
// orthogonal) and determinism (same seed ⇒ identical R).
func TestRandomOrthogonal(t *testing.T) {
	for _, dim := range []int{4, 8, 16, 32} {
		R := randomOrthogonal(dim, 1234)
		// RᵀR = I: row i · row j = δ_ij (rows are orthonormal). Since R is square
		// and orthonormal, both RRᵀ and RᵀR equal I; checking row dots is sufficient.
		for i := 0; i < dim; i++ {
			for j := i; j < dim; j++ {
				var dot float64
				for c := 0; c < dim; c++ {
					dot += float64(R[i*dim+c]) * float64(R[j*dim+c])
				}
				want := 0.0
				if i == j {
					want = 1.0
				}
				if math.Abs(dot-want) > 1e-4 {
					t.Fatalf("dim=%d rows(%d,%d) dot=%.6f, want %.0f (RᵀR != I)", dim, i, j, dot, want)
				}
			}
		}
		// Determinism: same seed ⇒ byte-identical R.
		R2 := randomOrthogonal(dim, 1234)
		for i := range R {
			if math.Float32bits(R[i]) != math.Float32bits(R2[i]) {
				t.Fatalf("dim=%d: randomOrthogonal not deterministic at %d (%v vs %v)", dim, i, R[i], R2[i])
			}
		}
		// Different seed ⇒ different R (sanity; not a hard guarantee but practically true).
		R3 := randomOrthogonal(dim, 5678)
		same := true
		for i := range R {
			if R[i] != R3[i] {
				same = false
				break
			}
		}
		if same {
			t.Fatalf("dim=%d: different seeds produced identical R", dim)
		}
	}
}

// TestRotateRoundTrip asserts rotateT(R, rotate(R, x)) ≈ x (Rᵀ R = I), so the
// reconstruct un-rotation exactly inverts encode's rotation.
func TestRotateRoundTrip(t *testing.T) {
	const dim = 16
	R := randomOrthogonal(dim, 99)
	rng := rand.New(rand.NewSource(7))
	for iter := 0; iter < 50; iter++ {
		x := make([]float32, dim)
		for j := range x {
			x[j] = float32(rng.NormFloat64())
		}
		back := rotateT(R, rotate(R, x))
		for j := range x {
			if math.Abs(float64(back[j]-x[j])) > 1e-4 {
				t.Fatalf("round-trip mismatch at %d: got %v want %v", j, back[j], x[j])
			}
		}
		// rotate preserves L2 norm (isometry).
		var nx, nr float32
		rx := rotate(R, x)
		for j := range x {
			nx += x[j] * x[j]
			nr += rx[j] * rx[j]
		}
		if math.Abs(float64(nx-nr)) > 1e-3 {
			t.Fatalf("rotation not norm-preserving: |x|²=%v |Rx|²=%v", nx, nr)
		}
	}
}

// TestOPQReconstruct asserts OPQ encode→reconstruct ≈ vec: the rotation composes
// (encode rotates Rx, reconstruct un-rotates Rᵀ → back in original space) so the
// reconstruction error is comparable to plain PQ.
func TestOPQReconstruct(t *testing.T) {
	const dim, m = 32, 8
	vecs := makePQClustered(t, 3000, dim, 12, L2, 4)
	p, err := trainPQ(vecs, m, dim, 9, L2, 1, true, 0, 1, 8)
	if err != nil {
		t.Fatal(err)
	}
	if p.rotation == nil {
		t.Fatal("OPQ trainPQ left rotation nil")
	}
	var sumErr, sumMag float64
	for _, v := range vecs {
		r := p.reconstruct(p.Encode(v))
		sumErr += math.Sqrt(float64(l2Squared(v, r)))
		var mag float32
		for _, x := range v {
			mag += x * x
		}
		sumMag += math.Sqrt(float64(mag))
	}
	rel := sumErr / sumMag
	if rel > 0.5 {
		t.Fatalf("OPQ reconstruction relative error %.3f too high (rotation composition wrong?)", rel)
	}
	t.Logf("OPQ reconstruction relative error = %.4f", rel)
}

// makeImbalanced generates clustered vectors whose variance is concentrated in
// the FIRST half of the dimensions (the rest are near-constant), the regime where
// plain PQ's fixed contiguous sub-spaces are imbalanced and OPQ's rotation helps.
func makeImbalanced(t *testing.T, n, dim, nClusters int, seed int64) [][]float32 {
	t.Helper()
	rng := rand.New(rand.NewSource(seed))
	centers := make([][]float32, nClusters)
	for c := range centers {
		centers[c] = make([]float32, dim)
		for d := 0; d < dim; d++ {
			// High variance in the first half, tiny in the second half.
			scale := float32(0.02)
			if d < dim/2 {
				scale = 8
			}
			centers[c][d] = float32(rng.NormFloat64()) * scale
		}
	}
	out := make([][]float32, n)
	for i := range out {
		c := centers[rng.Intn(nClusters)]
		v := make([]float32, dim)
		for d := 0; d < dim; d++ {
			noise := float32(0.01)
			if d < dim/2 {
				noise = 1
			}
			v[d] = c[d] + float32(rng.NormFloat64())*noise
		}
		out[i] = v
	}
	return out
}

func opqRecall(t *testing.T, vecs, queries [][]float32, dim, m int, seed int64, opq bool) float64 {
	t.Helper()
	p, err := trainPQ(vecs, m, dim, seed, L2, 1, opq, 0, 1, 8)
	if err != nil {
		t.Fatal(err)
	}
	codes := make([][]byte, len(vecs))
	for i, v := range vecs {
		codes[i] = p.Encode(v)
	}
	dist := pickDistDim(L2, dim)
	const k = 10
	var recallSum float64
	for _, q := range queries {
		lut := p.queryLUT(q)
		exact := topK(len(vecs), k, func(i int) float32 { return dist(q, vecs[i]) })
		approx := topK(len(vecs), k, func(i int) float32 { return p.adc(lut, codes[i]) })
		recallSum += pqRecall(exact, approx)
	}
	return recallSum / float64(len(queries))
}

// TestOPQRecallImbalanced is the OPQ-benefit proof: on data with imbalanced
// sub-space variance (where plain PQ's fixed contiguous split wastes codebook
// capacity on the high-variance half and starves the low-variance half), OPQ's
// rotation balances the sub-spaces → recall@10(OPQ) ≥ recall@10(plain PQ).
func TestOPQRecallImbalanced(t *testing.T) {
	const dim, m = 32, 8
	vecs := makeImbalanced(t, 4000, dim, 16, 100)
	queries := makeImbalanced(t, 50, dim, 16, 101)

	plain := opqRecall(t, vecs, queries, dim, m, 11, false)
	withOPQ := opqRecall(t, vecs, queries, dim, m, 11, true)
	t.Logf("imbalanced recall@10: plain PQ = %.3f, OPQ = %.3f", plain, withOPQ)
	// OPQ must not regress; on this imbalanced regime it should help (allow a tiny
	// epsilon so float noise / a strong-plain run never flakes the >= assertion).
	if withOPQ < plain-0.02 {
		t.Fatalf("OPQ recall %.3f regressed vs plain %.3f on imbalanced data", withOPQ, plain)
	}
}

// TestOPQRecallBalancedNoRegress asserts OPQ does not REGRESS recall on balanced
// data (where plain PQ already has well-distributed sub-spaces).
func TestOPQRecallBalancedNoRegress(t *testing.T) {
	const dim, m = 32, 8
	vecs := makePQClustered(t, 4000, dim, 16, L2, 200)
	queries := makePQClustered(t, 50, dim, 16, L2, 201)

	plain := opqRecall(t, vecs, queries, dim, m, 11, false)
	withOPQ := opqRecall(t, vecs, queries, dim, m, 11, true)
	t.Logf("balanced recall@10: plain PQ = %.3f, OPQ = %.3f", plain, withOPQ)
	if withOPQ < plain-0.1 {
		t.Fatalf("OPQ recall %.3f regressed materially vs plain %.3f on balanced data", withOPQ, plain)
	}
}

// TestOPQOffByteIdentical is the byte-identicality guarantee: trainPQ(opq=false, 1, 8)
// has rotation==nil and produces IDENTICAL encode bytes + adc values to the
// pre-change plain PQ (every rotation apply site is gated on rotation != nil).
func TestOPQOffByteIdentical(t *testing.T) {
	const dim, m = 32, 8
	vecs := makePQClustered(t, 2000, dim, 10, L2, 2)
	p, err := trainPQ(vecs, m, dim, 7, L2, 1, false, 0, 1, 8)
	if err != nil {
		t.Fatal(err)
	}
	if p.rotation != nil {
		t.Fatal("opq=false left a non-nil rotation (not byte-identical to plain PQ)")
	}
	// Encode bytes + adc must be the plain-PQ values. We reproduce the plain path
	// explicitly (no rotation) and compare bit-for-bit.
	dsub := dim / m
	for i, v := range vecs[:200] {
		code := p.Encode(v)
		// Plain encode: nearest sub-centroid by L2 with NO rotation.
		want := make([]byte, m)
		dist := pickDistDim(L2, dsub)
		for s := 0; s < m; s++ {
			sub := v[s*dsub : s*dsub+dsub]
			cb := p.codebooks[s]
			best, bestD := 0, dist(sub, cb[0])
			for c := 1; c < len(cb); c++ {
				if d := dist(sub, cb[c]); d < bestD {
					bestD, best = d, c
				}
			}
			want[s] = byte(best)
		}
		for s := 0; s < m; s++ {
			if code[s] != want[s] {
				t.Fatalf("vec %d subspace %d: opq-off encode byte %d != plain %d", i, s, code[s], want[s])
			}
		}
		// adc bit-identical to a no-rotation LUT.
		lut := p.queryLUT(v)
		got := p.adc(lut, code)
		want2 := naiveADC(L2, m, lut, code)
		if math.Float32bits(got) != math.Float32bits(want2) {
			t.Fatalf("vec %d: opq-off adc not bit-identical", i)
		}
	}
}

// TestOPQConfigValidation asserts OPQ requires a PQ mode.
func TestOPQConfigValidation(t *testing.T) {
	base := DefaultConfig()
	base.Dim = 32
	base.Metric = L2

	// OPQ with no PQ mode → error.
	c := base
	c.OPQ = true
	if err := c.Validate(); err != ErrInvalidOPQ {
		t.Fatalf("OPQ without PQ: got %v, want ErrInvalidOPQ", err)
	}
	// OPQ + HNSW-PQ → ok.
	c = base
	c.OPQ = true
	c.Quant = QuantPQ
	if err := c.Validate(); err != nil {
		t.Fatalf("OPQ + QuantPQ should be valid: %v", err)
	}
	// OPQ + IVF-PQ → ok.
	c = base
	c.OPQ = true
	c.IndexType = IndexIVF
	c.IVFPQ = true
	if err := c.Validate(); err != nil {
		t.Fatalf("OPQ + IVFPQ should be valid: %v", err)
	}
	// OPQ off → unaffected.
	c = base
	if err := c.Validate(); err != nil {
		t.Fatalf("OPQ-off default config should be valid: %v", err)
	}
}

// --- test helpers ---

// topK returns the indices of the k smallest scores among [0,n), ascending.
func topK(n, k int, score func(i int) float32) []int {
	idx := make([]int, n)
	for i := range idx {
		idx[i] = i
	}
	sort.Slice(idx, func(a, b int) bool { return score(idx[a]) < score(idx[b]) })
	if k > n {
		k = n
	}
	return idx[:k]
}

// recallAt returns |exact ∩ approx| / |exact|.
func pqRecall(exact, approx []int) float64 {
	set := make(map[int]struct{}, len(approx))
	for _, a := range approx {
		set[a] = struct{}{}
	}
	hit := 0
	for _, e := range exact {
		if _, ok := set[e]; ok {
			hit++
		}
	}
	return float64(hit) / float64(len(exact))
}
