// SPDX-License-Identifier: Apache-2.0

package vector

import "math"

// distFunc is the signature for a distance / similarity function. Smaller
// return values mean "more similar". Functions assume len(a) == len(b);
// callers MUST validate that before invoking.
type distFunc func(a, b []float32) float32

// pickDist returns the distance function for a given metric. Panics on
// unknown metrics — the surrounding API is expected to validate the Config
// before this is ever reached.
func pickDist(m Metric) distFunc {
	switch m {
	case Cosine:
		// Cosine distance over pre-normalized vectors collapses to 1 - dot.
		return func(a, b []float32) float32 { return 1.0 - dotProduct(a, b) }
	case L2:
		return l2Squared
	case DotProduct:
		return func(a, b []float32) float32 { return -dotProduct(a, b) }
	default:
		panic("vector: unknown Metric")
	}
}

// avx512DistDim is a platform hook: on AVX-512-capable amd64 CPUs the package
// init sets it to a function returning a metric distFunc backed by the wider
// ZMM kernels for dim >= threshold (nil below it, to defer to the default
// kernel). It stays nil elsewhere, keeping this file platform-neutral.
var avx512DistDim func(m Metric, dim int) distFunc

// pickDistDim is pickDist with a dimension-aware fast path: at high dimension on
// AVX-512 hardware it returns the 16-lane ZMM kernels, which only pay off at
// large dim (on AMD Zen they tie AVX2 below ~1024 and win above — see
// distance_avx512_amd64.s). Below the threshold, or without AVX-512, it returns
// the standard kernel from pickDist. Called once per scorer construction (not on
// the per-distance hot path), so the dispatch cost is negligible.
func pickDistDim(m Metric, dim int) distFunc {
	if avx512DistDim != nil {
		if f := avx512DistDim(m, dim); f != nil {
			return f
		}
	}
	return pickDist(m)
}

// dotProduct and l2Squared are the active distance kernels. They default to
// the portable scalar implementations; on amd64 with AVX2 an init() in
// distance_amd64.go swaps them for SIMD kernels. Cosine and DotProduct route
// through dotProduct; L2 routes through l2Squared.
var (
	dotProduct distFunc = dotScalar
	l2Squared  distFunc = l2SquaredScalar
)

// dotScalar computes the dot product of two equal-length float32 slices.
func dotScalar(a, b []float32) float32 {
	var sum float32
	for i := range a {
		sum += a[i] * b[i]
	}
	return sum
}

// l2SquaredScalar computes the sum of squared differences (squared L2).
// Square root is deferred to the API boundary; HNSW comparisons only need
// the monotonic squared form.
func l2SquaredScalar(a, b []float32) float32 {
	var sum float32
	for i := range a {
		d := a[i] - b[i]
		sum += d * d
	}
	return sum
}

// normalize divides v by its L2 norm in place. No-op if the norm is zero.
// Used by Cosine on insert so the hot path skips per-candidate normalization.
func normalize(v []float32) {
	var n float32
	for _, x := range v {
		n += x * x
	}
	if n == 0 {
		return
	}
	inv := float32(1.0 / math.Sqrt(float64(n)))
	for i := range v {
		v[i] *= inv
	}
}
