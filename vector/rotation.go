// SPDX-License-Identifier: Apache-2.0

package vector

import (
	"math"
	"math/rand"
)

// OPQ rotation primitives.
//
// OPQ (Optimized Product Quantization) prepends an orthogonal rotation R (d×d)
// to the PQ pipeline: every vector x is rotated to Rx BEFORE it is split into
// PQ sub-vectors, and reconstructed vectors are un-rotated by Rᵀ. Because R is
// orthonormal (RᵀR = I), the rotation is an isometry — it preserves L2 / dot /
// cosine distances exactly — so it never changes the underlying geometry; it
// only re-distributes variance across the dimensions so the M contiguous PQ
// sub-spaces carry balanced variance, which lowers PQ's quantization error and
// raises recall at the same M/nbits.
//
// v1 uses a RANDOM orthogonal R: a seeded Gaussian d×d matrix run through
// Gram-Schmidt orthonormalization. It is deterministic in the seed (same seed ⇒
// identical R) and captures much of OPQ's benefit on imbalanced data without an
// eigensolver/SVD (PCA-rotation and full-OPQ alternating optimization are
// follow-ups). R is stored flat, row-major: R[i*dim+j] is row i, column j.

// randomOrthogonal returns a dim×dim orthonormal matrix (flat, row-major) built
// by Gram-Schmidt orthonormalization of a seeded Gaussian random matrix.
// Deterministic: the same (dim, seed) always yields the identical R. The result
// satisfies RᵀR ≈ I (each row is unit-length and pairwise-orthogonal to the
// others) to within float32 round-off.
func randomOrthogonal(dim int, seed int64) []float32 {
	rng := rand.New(rand.NewSource(seed))

	// rows[i] is the i-th row vector (length dim) we orthonormalize in place.
	rows := make([][]float64, dim)
	for i := 0; i < dim; i++ {
		row := make([]float64, dim)
		for j := 0; j < dim; j++ {
			row[j] = rng.NormFloat64()
		}
		rows[i] = row
	}

	// Modified Gram-Schmidt: orthogonalize each row against all previously
	// finalized rows, then normalize. Modified (subtract-as-you-go) is more
	// numerically stable than classical GS.
	for i := 0; i < dim; i++ {
		ri := rows[i]
		for k := 0; k < i; k++ {
			rk := rows[k]
			var dot float64
			for j := 0; j < dim; j++ {
				dot += ri[j] * rk[j]
			}
			for j := 0; j < dim; j++ {
				ri[j] -= dot * rk[j]
			}
		}
		var norm float64
		for j := 0; j < dim; j++ {
			norm += ri[j] * ri[j]
		}
		// norm is ~never zero for a Gaussian matrix; guard defensively so a
		// degenerate row falls back to a unit basis vector rather than NaN.
		if norm <= 1e-12 {
			for j := 0; j < dim; j++ {
				ri[j] = 0
			}
			ri[i] = 1
			norm = 1
		}
		inv := 1 / math.Sqrt(norm)
		for j := 0; j < dim; j++ {
			ri[j] *= inv
		}
	}

	R := make([]float32, dim*dim)
	for i := 0; i < dim; i++ {
		base := i * dim
		ri := rows[i]
		for j := 0; j < dim; j++ {
			R[base+j] = float32(ri[j])
		}
	}
	return R
}

// rotate returns Rx: the matrix-vector product of R (dim×dim, flat row-major)
// with x (length dim). out[i] = Σ_j R[i*dim+j] * x[j]. Allocates the result.
func rotate(R, x []float32) []float32 {
	dim := len(x)
	out := make([]float32, dim)
	rotateInto(out, R, x)
	return out
}

// rotateInto writes Rx into dst (len(dst) == len(x) == dim). No allocation; used
// on the per-vector encode/query hot path with a reused scratch buffer.
func rotateInto(dst, R, x []float32) {
	dim := len(x)
	for i := 0; i < dim; i++ {
		base := i * dim
		var sum float32
		for j := 0; j < dim; j++ {
			sum += R[base+j] * x[j]
		}
		dst[i] = sum
	}
}

// rotateT returns Rᵀy: the transpose-matrix-vector product. Since R is
// orthonormal (RᵀR = I), Rᵀ is the inverse rotation, so rotateT(R, rotate(R, x))
// ≈ x. out[j] = Σ_i R[i*dim+j] * y[i]. Allocates the result.
func rotateT(R, y []float32) []float32 {
	dim := len(y)
	out := make([]float32, dim)
	for i := 0; i < dim; i++ {
		base := i * dim
		yi := y[i]
		for j := 0; j < dim; j++ {
			out[j] += R[base+j] * yi
		}
	}
	return out
}
