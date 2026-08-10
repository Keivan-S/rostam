// SPDX-License-Identifier: Apache-2.0

package vector

import "math"

// Deterministic SVD + Procrustes rotation for full-OPQ (Ge et al. 2013, OPQ_NP).
//
// The current OPQ uses a SEEDED RANDOM orthogonal rotation R applied before a
// single PQ train (see rotation.go). Full-OPQ instead REFINES R from the trained
// codebooks: at each iteration it reconstructs every training vector x̂ from its
// PQ code and solves the orthogonal Procrustes problem
//
//	R* = argmin_{RᵀR=I} Σ_i ‖x_i − Rᵀ·x̂_i‖²   →   R = V·Uᵀ where  M = Σ_i x_i·x̂_iᵀ = U·Σ·Vᵀ
//
// (R rotates ORIGINAL→reconstruction space; encode applies Rx, reconstruct
// un-applies Rᵀ — see pq.go). Solving Procrustes needs an SVD; the codebase has
// none (the v1 random rotation deliberately avoided an eigensolver). This file
// adds a FROM-SCRATCH one-sided Jacobi SVD in pure Go.
//
// ┌─────────────────────────────────────────────────────────────────────────┐
// │ DETERMINISM IS LOAD-BEARING. Replicas apply the SAME Raft ops and MUST   │
// │ converge to a BIT-IDENTICAL rotation R + codebooks. Every step here is   │
// │ a PURE function of the input matrix:                                     │
// │   • NO random init (the only randomness in full-OPQ is the existing      │
// │     seeded iter-0 rotation, which lives in trainPQ, not here).           │
// │   • NO map iteration (maps have a randomized range order in Go).         │
// │   • FIXED Jacobi sweep order: every sweep visits the index pairs (i,j)   │
// │     with i<j in ascending order — never a data-dependent order.          │
// │   • FIXED sweep COUNT (jacobiSweeps): the loop ALWAYS runs exactly the   │
// │     same number of sweeps regardless of float values, so two hosts with  │
// │     different float rounding can NEVER stop at a different sweep (which   │
// │     a "converge when off-diagonal < ε" test could, diverging R).         │
// │   • The M = Σ x·x̂ᵀ accumulation (procrustesRotation) iterates the sample │
// │     in FIXED slot order with a single accumulator per cell.              │
// │ All arithmetic is float64 for numeric stability; only the final R is     │
// │ narrowed to float32 (the stored rotation type).                          │
// └─────────────────────────────────────────────────────────────────────────┘

// jacobiSweeps is the FIXED number of one-sided Jacobi sweeps. It is a CONSTANT
// (not a convergence test) so the iteration count is identical on every replica
// regardless of float rounding — the determinism linchpin. One-sided Jacobi
// converges quadratically; for the d×d matrices here (d up to ~1536) the
// off-diagonal mass is at float64 round-off well before 30 sweeps, so 30 is a
// safe over-provision. Cost: O(d³·sweeps) per OPQ iteration (d²/2 column-pairs
// × O(d) rotation-apply per pair). At d=768 that is ~8.5B flops per SVD call;
// keep OPQIters small (≤5) for high-dimensional collections (d≥512).
const jacobiSweeps = 30

// jacobiSVD computes a deterministic singular value decomposition of the SQUARE
// matrix M (n×n, row-major [][]float64): it returns U, S, V such that
// M ≈ U·diag(S)·Vᵀ, with U and V orthonormal (UᵀU = VᵀV = I) and S the singular
// values (non-negative, NOT sorted — see note). M is not modified.
//
// Algorithm: one-sided Jacobi on a working copy A := M. Each sweep rotates pairs
// of COLUMNS (i,j), i<j ascending, by a Givens rotation chosen to orthogonalize
// columns i and j; the SAME rotation is accumulated into V. After jacobiSweeps
// fixed sweeps the columns of A are mutually orthogonal; their norms are the
// singular values S, and the normalized columns are U. V holds the accumulated
// right rotations. Everything is float64.
//
// Determinism: fixed sweep count, fixed (i,j) ascending pair order, no random,
// no maps. Same M ⇒ bit-identical U, S, V on every host.
//
// Note: S is returned in COLUMN order (not sorted descending). procrustesRotation
// (the only caller) computes V·Uᵀ, which is invariant to a consistent column
// permutation/sign of U and V, so sorting is unnecessary; skipping it keeps the
// routine branch-light and order-stable.
func jacobiSVD(M [][]float64) (U [][]float64, S []float64, V [][]float64) {
	n := len(M)
	// A is the working copy whose COLUMNS we orthogonalize; V accumulates the
	// right rotations (init identity).
	A := make([][]float64, n)
	V = make([][]float64, n)
	for i := 0; i < n; i++ {
		A[i] = make([]float64, n)
		copy(A[i], M[i])
		V[i] = make([]float64, n)
		V[i][i] = 1
	}
	if n == 0 {
		return [][]float64{}, []float64{}, [][]float64{}
	}

	// One-sided Jacobi: a FIXED number of sweeps; each sweep visits column pairs
	// (i,j), i<j, in ascending order. The pair order and sweep count are constant
	// (no convergence test) so the result is bit-identical across replicas.
	for sweep := 0; sweep < jacobiSweeps; sweep++ {
		for i := 0; i < n; i++ {
			for j := i + 1; j < n; j++ {
				// alpha = ‖col i‖², beta = ‖col j‖², gamma = col i · col j.
				var alpha, beta, gamma float64
				for k := 0; k < n; k++ {
					aki := A[k][i]
					akj := A[k][j]
					alpha += aki * aki
					beta += akj * akj
					gamma += aki * akj
				}
				// Columns already orthogonal (or a zero column) → no rotation.
				// The guard is a fixed predicate on a deterministic value, so all
				// replicas take the identical branch; it does NOT change the sweep
				// COUNT (the outer loop runs jacobiSweeps regardless).
				if gamma == 0 || alpha == 0 || beta == 0 {
					continue
				}
				// Jacobi angle that zeroes the (i,j) column inner product.
				zeta := (beta - alpha) / (2 * gamma)
				var t float64
				if zeta >= 0 {
					t = 1 / (zeta + math.Sqrt(1+zeta*zeta))
				} else {
					t = -1 / (-zeta + math.Sqrt(1+zeta*zeta))
				}
				c := 1 / math.Sqrt(1+t*t)
				s := c * t
				// Apply the Givens rotation to columns i and j of A and V.
				for k := 0; k < n; k++ {
					aki := A[k][i]
					akj := A[k][j]
					A[k][i] = c*aki - s*akj
					A[k][j] = s*aki + c*akj
					vki := V[k][i]
					vkj := V[k][j]
					V[k][i] = c*vki - s*vkj
					V[k][j] = s*vki + c*vkj
				}
			}
		}
	}

	// Column norms of the orthogonalized A are the singular values; the
	// normalized columns are U. A degenerate (near-zero) column → that singular
	// value is 0 and the U column falls back to the corresponding identity basis
	// vector (so U stays a valid orthonormal-ish frame and V·Uᵀ has no NaN).
	S = make([]float64, n)
	U = make([][]float64, n)
	for i := 0; i < n; i++ {
		U[i] = make([]float64, n)
	}
	for j := 0; j < n; j++ {
		var norm float64
		for k := 0; k < n; k++ {
			norm += A[k][j] * A[k][j]
		}
		norm = math.Sqrt(norm)
		S[j] = norm
		if norm <= 1e-12 {
			// Zero/rank-deficient column: pick the identity basis vector e_j so U
			// stays well-defined and NaN-free (graceful degenerate handling).
			U[j][j] = 1
			continue
		}
		inv := 1 / norm
		for k := 0; k < n; k++ {
			U[k][j] = A[k][j] * inv
		}
	}
	return U, S, V
}

// procrustesRotation solves the orthogonal Procrustes problem and returns the
// optimal rotation R (n×n, flat row-major, float32) that maps the ORIGINAL
// vectors X to their reconstructions Xhat in the sense used by the OPQ codec:
// R = V·Uᵀ where M = Σ_i x_i · x̂_iᵀ = U·Σ·Vᵀ. Encode then applies Rx, and
// reconstruct un-applies Rᵀ (see pq.go).
//
// X and Xhat are the SAME length (one x̂ per x), each a dim-length row. The
// accumulation of M iterates the sample in FIXED slot order (i = 0..len-1) with
// a single accumulator per cell, so M — and therefore R — is bit-identical on
// every replica (load-bearing for determinism).
//
// Degenerate input (empty sample, all-zero M, or a rank-deficient M) is handled
// gracefully: jacobiSVD returns identity-padded U/V for zero columns, so
// R = V·Uᵀ stays orthogonal and NaN/Inf-free. An empty sample returns the
// identity rotation (a no-op rotate), so the caller keeps its current codebooks.
func procrustesRotation(X, Xhat [][]float32, dim int) []float32 {
	// M = Σ_i x_i · x̂_iᵀ  (dim×dim). FIXED slot order; single accumulator/cell.
	M := make([][]float64, dim)
	for r := 0; r < dim; r++ {
		M[r] = make([]float64, dim)
	}
	n := len(X)
	if n > len(Xhat) {
		n = len(Xhat)
	}
	for i := 0; i < n; i++ {
		x := X[i]
		xh := Xhat[i]
		for r := 0; r < dim; r++ {
			xr := float64(x[r])
			if xr == 0 {
				continue
			}
			mr := M[r]
			for c := 0; c < dim; c++ {
				mr[c] += xr * float64(xh[c])
			}
		}
	}

	U, _, V := jacobiSVD(M)

	// R = V·Uᵀ  →  R[i][j] = Σ_k V[i][k] · U[j][k]. Computed in float64.
	Rf := make([][]float64, dim)
	for i := 0; i < dim; i++ {
		Vi := V[i]
		row := make([]float64, dim)
		for j := 0; j < dim; j++ {
			Uj := U[j]
			var sum float64
			for k := 0; k < dim; k++ {
				sum += Vi[k] * Uj[k]
			}
			row[j] = sum
		}
		Rf[i] = row
	}

	// Re-orthonormalize R (modified Gram-Schmidt on its rows). For a FULL-RANK M,
	// V·Uᵀ is already orthogonal, so GS is a (deterministic) no-op. For a DEGENERATE
	// / rank-deficient M (zero singular values → ill-defined U columns), the raw
	// V·Uᵀ may not be exactly orthogonal; GS deterministically completes it to the
	// nearest orthonormal matrix (a zero row falls back to the identity basis
	// vector), so R·Rᵀ ≈ I and there is no NaN/Inf. Deterministic: fixed row order,
	// no random, no maps — load-bearing for replica determinism.
	orthonormalizeRowsF64(Rf, dim)

	// Narrow to float32 (the stored rotation type) only at the very end.
	R := make([]float32, dim*dim)
	for i := 0; i < dim; i++ {
		base := i * dim
		for j := 0; j < dim; j++ {
			R[base+j] = float32(Rf[i][j])
		}
	}
	return R
}

// orthonormalizeRowsF64 applies modified Gram-Schmidt to the rows of the n×n
// matrix in place: each row is orthogonalized against all previously finalized
// rows then normalized. Idempotent on an already-orthonormal matrix (GS of an
// orthonormal set returns it unchanged to round-off). A degenerate (near-zero)
// row falls back to the identity basis vector e_i so the result stays a valid
// orthonormal frame (no NaN). Deterministic: fixed row order, no random.
func orthonormalizeRowsF64(R [][]float64, n int) {
	for i := 0; i < n; i++ {
		ri := R[i]
		for k := 0; k < i; k++ {
			rk := R[k]
			var dot float64
			for j := 0; j < n; j++ {
				dot += ri[j] * rk[j]
			}
			for j := 0; j < n; j++ {
				ri[j] -= dot * rk[j]
			}
		}
		var norm float64
		for j := 0; j < n; j++ {
			norm += ri[j] * ri[j]
		}
		if norm <= 1e-12 {
			for j := 0; j < n; j++ {
				ri[j] = 0
			}
			ri[i] = 1
			continue
		}
		inv := 1 / math.Sqrt(norm)
		for j := 0; j < n; j++ {
			ri[j] *= inv
		}
	}
}
