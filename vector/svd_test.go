// SPDX-License-Identifier: Apache-2.0

package vector

import (
	"math"
	"math/rand"
	"testing"
)

// TestJacobiSVDReconstructs asserts jacobiSVD(M) gives U,S,V with M ≈ U·diag(S)·Vᵀ
// within tolerance and U,V orthonormal — on a random well-conditioned matrix.
func TestJacobiSVDReconstructs(t *testing.T) {
	const n = 12
	rng := rand.New(rand.NewSource(42))
	M := make([][]float64, n)
	for i := 0; i < n; i++ {
		M[i] = make([]float64, n)
		for j := 0; j < n; j++ {
			M[i][j] = rng.NormFloat64()
		}
	}
	U, S, V := jacobiSVD(M)

	// Build U·diag(S)·Vᵀ and compare to M.
	for i := 0; i < n; i++ {
		for j := 0; j < n; j++ {
			var sum float64
			for k := 0; k < n; k++ {
				sum += U[i][k] * S[k] * V[j][k]
			}
			if math.Abs(sum-M[i][j]) > 1e-6 {
				t.Fatalf("SVD reconstruct[%d][%d]=%v want %v", i, j, sum, M[i][j])
			}
		}
	}

	// U orthonormal: UᵀU ≈ I.
	assertOrthonormalCols(t, U, n, "U")
	assertOrthonormalCols(t, V, n, "V")
}

// assertOrthonormalCols checks that the columns of the n×n matrix are mutually
// orthonormal (MᵀM ≈ I) within tolerance.
func assertOrthonormalCols(t *testing.T, Mtx [][]float64, n int, name string) {
	t.Helper()
	for a := 0; a < n; a++ {
		for b := 0; b < n; b++ {
			var dot float64
			for k := 0; k < n; k++ {
				dot += Mtx[k][a] * Mtx[k][b]
			}
			want := 0.0
			if a == b {
				want = 1.0
			}
			if math.Abs(dot-want) > 1e-6 {
				t.Fatalf("%s columns not orthonormal: <%d,%d>=%v want %v", name, a, b, dot, want)
			}
		}
	}
}

// TestJacobiSVDDeterministic is part of the determinism linchpin: the SAME M run
// through jacobiSVD TWICE yields BIT-IDENTICAL U, S, V (fixed sweep order + fixed
// sweep count, no random, no map iteration).
func TestJacobiSVDDeterministic(t *testing.T) {
	const n = 16
	rng := rand.New(rand.NewSource(7))
	M := make([][]float64, n)
	for i := 0; i < n; i++ {
		M[i] = make([]float64, n)
		for j := 0; j < n; j++ {
			M[i][j] = rng.NormFloat64() * 3
		}
	}
	// Deep copies so the two runs share no backing storage.
	M1 := copyMat(M)
	M2 := copyMat(M)
	U1, S1, V1 := jacobiSVD(M1)
	U2, S2, V2 := jacobiSVD(M2)
	assertMatBitEqual(t, U1, U2, "U")
	assertMatBitEqual(t, V1, V2, "V")
	for i := range S1 {
		if math.Float64bits(S1[i]) != math.Float64bits(S2[i]) {
			t.Fatalf("S[%d] not bit-identical: %v vs %v", i, S1[i], S2[i])
		}
	}
}

func copyMat(M [][]float64) [][]float64 {
	out := make([][]float64, len(M))
	for i := range M {
		out[i] = make([]float64, len(M[i]))
		copy(out[i], M[i])
	}
	return out
}

func assertMatBitEqual(t *testing.T, A, B [][]float64, name string) {
	t.Helper()
	for i := range A {
		for j := range A[i] {
			if math.Float64bits(A[i][j]) != math.Float64bits(B[i][j]) {
				t.Fatalf("%s[%d][%d] not bit-identical: %v vs %v", name, i, j, A[i][j], B[i][j])
			}
		}
	}
}

// TestProcrustesRotationOrthogonal asserts R = procrustesRotation(X, Xhat) is
// orthogonal (R·Rᵀ ≈ I) on a realistic original/reconstruction pair.
func TestProcrustesRotationOrthogonal(t *testing.T) {
	const dim = 16
	rng := rand.New(rand.NewSource(99))
	n := 300
	X := make([][]float32, n)
	Xhat := make([][]float32, n)
	for i := 0; i < n; i++ {
		x := make([]float32, dim)
		xh := make([]float32, dim)
		for d := 0; d < dim; d++ {
			x[d] = float32(rng.NormFloat64())
			// A perturbed copy stands in for a reconstruction.
			xh[d] = x[d] + float32(rng.NormFloat64())*0.1
		}
		X[i], Xhat[i] = x, xh
	}
	R := procrustesRotation(X, Xhat, dim)

	// R·Rᵀ ≈ I (R flat row-major).
	for i := 0; i < dim; i++ {
		for j := 0; j < dim; j++ {
			var dot float32
			for k := 0; k < dim; k++ {
				dot += R[i*dim+k] * R[j*dim+k]
			}
			want := float32(0)
			if i == j {
				want = 1
			}
			if math.Abs(float64(dot-want)) > 1e-4 {
				t.Fatalf("R·Rᵀ[%d][%d]=%v want %v (R not orthogonal)", i, j, dot, want)
			}
		}
	}
}

// TestProcrustesRotationDegenerate asserts the degenerate cases produce NO NaN/Inf
// and a well-defined (orthogonal) fallback rotation: an all-zero M (zero sample),
// and a rank-deficient M (all reconstructions identical → rank-1 M).
func TestProcrustesRotationDegenerate(t *testing.T) {
	const dim = 8

	// Empty sample → M all-zero → identity-padded U/V → orthogonal R, no NaN.
	R := procrustesRotation(nil, nil, dim)
	assertNoNaNOrthogonal(t, R, dim, "empty-sample")

	// All-zero vectors → M all-zero.
	z := make([][]float32, 10)
	zh := make([][]float32, 10)
	for i := range z {
		z[i] = make([]float32, dim)
		zh[i] = make([]float32, dim)
	}
	R = procrustesRotation(z, zh, dim)
	assertNoNaNOrthogonal(t, R, dim, "all-zero")

	// Rank-deficient: every x maps to the SAME xhat (rank-1 outer-product sum).
	rng := rand.New(rand.NewSource(5))
	xhConst := make([]float32, dim)
	for d := range xhConst {
		xhConst[d] = float32(rng.NormFloat64())
	}
	X := make([][]float32, 50)
	Xhat := make([][]float32, 50)
	for i := range X {
		x := make([]float32, dim)
		for d := range x {
			x[d] = float32(rng.NormFloat64())
		}
		X[i] = x
		Xhat[i] = xhConst
	}
	R = procrustesRotation(X, Xhat, dim)
	assertNoNaNOrthogonal(t, R, dim, "rank-deficient")
}

func assertNoNaNOrthogonal(t *testing.T, R []float32, dim int, label string) {
	t.Helper()
	for i, v := range R {
		if math.IsNaN(float64(v)) || math.IsInf(float64(v), 0) {
			t.Fatalf("%s: R[%d]=%v is NaN/Inf", label, i, v)
		}
	}
	// Orthogonality (R·Rᵀ ≈ I) — the graceful fallback still yields an orthogonal R.
	for i := 0; i < dim; i++ {
		for j := 0; j < dim; j++ {
			var dot float32
			for k := 0; k < dim; k++ {
				dot += R[i*dim+k] * R[j*dim+k]
			}
			want := float32(0)
			if i == j {
				want = 1
			}
			if math.Abs(float64(dot-want)) > 1e-3 {
				t.Fatalf("%s: R·Rᵀ[%d][%d]=%v want %v", label, i, j, dot, want)
			}
		}
	}
}
