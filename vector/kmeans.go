// SPDX-License-Identifier: Apache-2.0

package vector

import (
	"math"
	"math/rand"
	"runtime"
	"sync"
	"sync/atomic"
)

// kmeansMaxIter bounds Lloyd's iterations. Convergence (no assignment changes /
// negligible centroid drift) almost always trips well before this; the cap just
// guarantees termination on pathological inputs.
const kmeansMaxIter = 25

// kmeansShiftEps2 is the early-exit threshold on the summed squared centroid
// drift between iterations. Below it the centroids are considered converged.
const kmeansShiftEps2 = 1e-6

// kmeansSampleMax is the absolute cap on the training-set size. When the input
// exceeds the per-k cap (256*k) and this max, kmeans trains on a deterministic
// sample of min(len, max(256*k, kmeansSampleMax)) vectors. 50k keeps a single
// training pass cheap and bounded in the memory-capped container while still
// giving k-means plenty of signal for the centroid placement.
const kmeansSampleMax = 50000

// kmeans runs Lloyd's algorithm with k-means++ initialization and returns k
// centroids for the given vectors. It is a pure function with no index coupling:
// it reuses the package distance kernels via the supplied metric (the same
// metric the collection uses — L2/Cosine/DotProduct), so the coarse quantizer
// partitions space with the same notion of "near" the search uses.
//
// Determinism: every random choice (k-means++ seeding, sampling, empty-cluster
// re-seeding) is driven by a single seeded math/rand.Rand — matching how hnsw
// seeds its rng (rand.New(rand.NewSource(seed))). The same (vectors, k, seed,
// metric) always yields byte-identical centroids.
//
// Metric handling: assignment uses pickDist(metric), so the result is correct
// for whichever metric the collection stores. For Cosine the caller is expected
// to pass vectors that are already L2-normalized (as hnsw stores them on insert,
// via normalize); on normalized vectors cosine reduces to 1 - dot and the
// recomputed means cluster by angle. kmeans does NOT re-normalize the means here
// — keeping it metric-agnostic — but re-normalizes centroids for Cosine so that
// 1 - dot stays a valid distance against subsequently-assigned normalized
// vectors.
//
// Edge cases:
//   - k <= 0:                guard, returns nil.
//   - len(vectors) == 0:     returns nil (no-op).
//   - k >= len(vectors):     returns the (deduplicated) input vectors as
//     centroids — each point is its own centroid.
//
// Sampling: if len(vectors) exceeds min(len, max(256*k, kmeansSampleMax)), kmeans
// trains on a deterministic sample of that size (drawn with the seeded rng) and
// returns centroids in the input space; assignment of the full set happens later
// at index build time. The cap is documented above.
//
// Parallelism (workers): workers<=1 runs the verbatim serial path. workers>1
// parallelizes ONLY the assignment step (each point → nearest centroid) across
// min(workers, GOMAXPROCS) goroutines via a deterministic range-split — each
// goroutine writes its points' chosen-centroid index into the shared, pre-sized
// assign[] at DISJOINT indices (no lock, no shared accumulator). After a barrier,
// the centroid-update reduce runs SERIALLY iterating points in INDEX ORDER, so the
// per-cluster float sums accumulate in the exact same order as the serial path —
// the result is BIT-IDENTICAL to workers=1 for the same (vectors, k, seed, metric).
// k-means++ init stays serial (sequential-dependent). Parallelism is a pure
// performance lever: it changes nothing about the output bytes.
func kmeans(vectors [][]float32, k int, seed int64, metric Metric, workers int) [][]float32 {
	if k <= 0 || len(vectors) == 0 {
		return nil
	}
	cosine := metric == Cosine

	// k >= n: each point is its own centroid (dedup exact duplicates so we
	// don't return coincident centroids).
	if k >= len(vectors) {
		return dedupVectors(vectors)
	}

	rng := rand.New(rand.NewSource(seed))

	// Sample down a huge input for the training pass. The cap is per-k
	// (256*k) bounded by the absolute kmeansSampleMax.
	train := vectors
	sampleCap := 256 * k
	if sampleCap < kmeansSampleMax {
		sampleCap = kmeansSampleMax
	}
	if len(vectors) > sampleCap {
		train = sampleVectors(vectors, sampleCap, rng)
		// After sampling it's possible (cap >= len after dedup considerations
		// is impossible here since cap < len) that k >= len(train); guard.
		if k >= len(train) {
			return dedupVectors(train)
		}
	}

	dim := len(train[0])
	dist := pickDist(metric)

	// Worker budget for the assignment step. <=1 keeps the verbatim serial path;
	// otherwise cap to GOMAXPROCS and to len(train) so empty chunks never spawn.
	nWorkers := workers
	if nWorkers > 1 {
		if gomax := runtime.GOMAXPROCS(0); nWorkers > gomax {
			nWorkers = gomax
		}
		if nWorkers > len(train) {
			nWorkers = len(train)
		}
	}

	// --- k-means++ initialization (seeded) ---
	// Sequential-dependent (each pick refreshes nearest-dist over all points);
	// kept serial so seeding stays byte-deterministic.
	centroids := kmeansPlusPlusInit(train, k, dist, rng)

	assign := make([]int, len(train))
	for i := range assign {
		assign[i] = -1
	}

	// --- Lloyd iterations ---
	for iter := 0; iter < kmeansMaxIter; iter++ {
		changed := false

		// Assignment step: nearest centroid per vector.
		if nWorkers > 1 {
			changed = assignParallel(train, centroids, k, dist, assign, nWorkers)
		} else {
			for vi, v := range train {
				best := nearestCentroidIdx(v, centroids, k, dist)
				if assign[vi] != best {
					assign[vi] = best
					changed = true
				}
			}
		}

		// Update step: centroid = mean of assigned vectors.
		next := make([][]float32, k)
		counts := make([]int, k)
		for ci := range next {
			next[ci] = make([]float32, dim)
		}
		for vi, v := range train {
			ci := assign[vi]
			counts[ci]++
			acc := next[ci]
			for d := 0; d < dim; d++ {
				acc[d] += v[d]
			}
		}
		for ci := 0; ci < k; ci++ {
			if counts[ci] == 0 {
				// Empty cluster: re-seed to the point farthest from its
				// current centroid (deterministic given the seeded order of
				// the prior steps). This splits the largest/loosest cluster
				// rather than leaving a dead centroid.
				next[ci] = farthestPoint(train, centroids, dist)
				continue
			}
			inv := float32(1.0 / float64(counts[ci]))
			acc := next[ci]
			for d := 0; d < dim; d++ {
				acc[d] *= inv
			}
			if cosine {
				normalize(acc)
			}
		}

		// Convergence: stop if no assignment changed, or the centroids barely
		// moved.
		shift := centroidShift2(centroids, next)
		centroids = next
		if !changed || shift < kmeansShiftEps2 {
			break
		}
	}

	return centroids
}

// kmeansAnisotropic runs Lloyd's algorithm with a ScaNN-style SCORE-AWARE
// (anisotropic) loss instead of plain L2, returning k centroids for the given
// (sub)vectors. It is the opt-in PQ-codebook trainer behind Config.AnisotropicEta
// (η > 1); the existing kmeans(...) is used verbatim for the isotropic η ∈ {0,1}.
//
// THE LOSS. For a (sub)vector x assigned to centroid c, the residual r = x − c is
// decomposed relative to x's OWN direction û = x/‖x‖ into a PARALLEL part
// r_∥ = (r·û)û and an ORTHOGONAL part r_⊥ = r − r_∥. The per-point loss is
//
//	L(x, c) = η·‖r_∥‖² + ‖r_⊥‖²    (η ≥ 1).
//
// Parallel error (error ALONG the datapoint direction) dominates the inner-product
// /MIPS score estimate q·x ≈ q·c, so weighting it by η > 1 trades a little extra
// orthogonal error for less score error — exactly ScaNN's anisotropic quantization.
//   - Assignment: argmin_c L(x, c) (replaces nearest-centroid).
//   - Update: the per-cluster WEIGHTED-least-squares minimizer of Σ_i L(x_i, c),
//     i.e. the solution of A c = b with A = Σ_i (η·P_i + (I−P_i)),
//     b = Σ_i (η·P_i + (I−P_i)) x_i, P_i = x_i x_iᵀ/‖x_i‖² (projection onto x_i's
//     direction). A is a small dsub×dsub system (dsub 4..16), solved by Gaussian
//     elimination with partial pivoting (solveLinearSystem; pure Go, no deps).
//
// η = 1 REDUCES TO PLAIN L2 K-MEANS. When η = 1, η·P + (I−P) = I for every point,
// so the loss is ‖r‖² (plain squared-L2, hence the same argmin as nearest-centroid)
// and the update solves (nI) c = Σ x_i ⇒ c = mean — IDENTICAL to kmeans. To make
// that BYTE-identical rather than merely numerically equal, this function
// SHORT-CIRCUITS η ≤ 1 to kmeans(..., L2, ...) directly. Callers (trainCodebooks)
// only invoke this for η > 1, so the short-circuit is a defensive belt-and-suspenders.
//
// v1 SIMPLIFICATION (per-subspace anisotropy). ScaNN defines parallel/orthogonal
// w.r.t. the FULL vector. For per-subspace PQ this uses the SUB-vector's OWN
// direction for P_i, so each subspace stays independent and the trainer remains a
// drop-in per-subspace k-means. The full-vector-coupled weighting (which couples
// the subspaces through the global datapoint direction) is a documented follow-up;
// the per-subspace direction still captures the dominant score-aware effect.
//
// DETERMINISM. Same seeded k-means++ init (kmeansPlusPlusInit), same fixed
// iteration order, same convergence test (centroidShift2) as kmeans. The
// assignment and the per-cluster linear-system accumulation both iterate points in
// INDEX order, so the result is a pure function of (vectors, k, seed, eta) — the
// same seed yields the same codebooks. The anisotropic path is SERIAL (the workers
// lever only parallelized the isotropic assignment step); the modest extra cost is
// a one-time training cost and dsub is small.
//
// Cosine: kmeans re-normalizes means for the Cosine metric; this trainer is only
// reached for PQ codebooks, which ALWAYS train on L2-internal sub-vectors (see
// pq.go header) — there is no cosine re-normalization here, matching the L2 path
// trainCodebooks invokes.
func kmeansAnisotropic(vectors [][]float32, k int, seed int64, eta float32, workers int) [][]float32 {
	// Defensive: η ≤ 1 is provably plain L2 k-means; run the EXISTING path so the
	// result is byte-identical to the isotropic trainer.
	if eta <= 1 {
		return kmeans(vectors, k, seed, L2, workers)
	}
	if k <= 0 || len(vectors) == 0 {
		return nil
	}
	if k >= len(vectors) {
		return dedupVectors(vectors)
	}

	rng := rand.New(rand.NewSource(seed))

	// Sample identically to kmeans so the training set (hence centroids) matches the
	// isotropic path's sampling decisions for the same (vectors, k, seed).
	train := vectors
	sampleCap := 256 * k
	if sampleCap < kmeansSampleMax {
		sampleCap = kmeansSampleMax
	}
	if len(vectors) > sampleCap {
		train = sampleVectors(vectors, sampleCap, rng)
		if k >= len(train) {
			return dedupVectors(train)
		}
	}

	dim := len(train[0])
	// k-means++ seeding under plain L2 distance (the same init the isotropic path
	// uses), so the two trainers START from the identical centroids; only the Lloyd
	// refinement diverges by the anisotropic loss.
	l2 := pickDist(L2)
	centroids := kmeansPlusPlusInit(train, k, l2, rng)

	// Precompute each training point's squared norm once (used by both the
	// assignment loss and the per-cluster system). normSq[i] ≈ 0 ⇒ x_i has no
	// direction; treat its projection as zero (isotropic for that point).
	normSq := make([]float32, len(train))
	for i, x := range train {
		normSq[i] = dotProduct(x, x)
	}

	assign := make([]int, len(train))
	for i := range assign {
		assign[i] = -1
	}

	for iter := 0; iter < kmeansMaxIter; iter++ {
		changed := false

		// Assignment: argmin over centroids of the anisotropic loss.
		for vi, x := range train {
			best := 0
			bestL := anisotropicLoss(x, centroids[0], normSq[vi], eta)
			for ci := 1; ci < k; ci++ {
				if l := anisotropicLoss(x, centroids[ci], normSq[vi], eta); l < bestL {
					bestL = l
					best = ci
				}
			}
			if assign[vi] != best {
				assign[vi] = best
				changed = true
			}
		}

		// Update: per-cluster weighted-least-squares centroid (solve A c = b).
		next := make([][]float32, k)
		// Per-cluster accumulators: A (dim×dim flat row-major) and b (dim).
		accA := make([][]float32, k)
		accB := make([][]float32, k)
		counts := make([]int, k)
		for ci := 0; ci < k; ci++ {
			accA[ci] = make([]float32, dim*dim)
			accB[ci] = make([]float32, dim)
		}
		// Accumulate in INDEX order so the float sums match across runs.
		for vi, x := range train {
			ci := assign[vi]
			counts[ci]++
			accumulateAnisotropic(accA[ci], accB[ci], x, normSq[vi], eta, dim)
		}
		for ci := 0; ci < k; ci++ {
			if counts[ci] == 0 {
				// Empty cluster: re-seed to the point farthest under L2 (same rule and
				// distance as the isotropic path's empty-cluster handling).
				next[ci] = farthestPoint(train, centroids, l2)
				continue
			}
			c, ok := solveLinearSystem(accA[ci], accB[ci], dim)
			if !ok {
				// Singular system (e.g. all assigned points share one direction with
				// η-degenerate weighting): fall back to the plain mean of the cluster,
				// which is always well-defined. b/η-independent fallback keeps training
				// robust without a deps-heavy pseudo-inverse.
				c = clusterMean(train, assign, ci, dim, counts[ci])
			}
			next[ci] = c
		}

		shift := centroidShift2(centroids, next)
		centroids = next
		if !changed || shift < kmeansShiftEps2 {
			break
		}
	}

	return centroids
}

// anisotropicLoss returns η·‖r_∥‖² + ‖r_⊥‖² for residual r = x − c with the
// parallel/orthogonal split taken w.r.t. x's own direction (normSq = ‖x‖²). When
// ‖x‖² ≈ 0 the direction is undefined, so the projection is skipped and the loss
// degrades to plain ‖r‖² (isotropic for that point). Implemented from the identity
// ‖r‖² + (η−1)·(r·û)² where (r·û)² = (r·x)²/‖x‖², avoiding an explicit projection
// vector and any allocation on this hot assignment-inner-loop path.
func anisotropicLoss(x, c []float32, normSq, eta float32) float32 {
	var rDotR float32 // ‖r‖²
	var rDotX float32 // r·x
	for i := range x {
		ri := x[i] - c[i]
		rDotR += ri * ri
		rDotX += ri * x[i]
	}
	if normSq <= 0 {
		return rDotR
	}
	par := (rDotX * rDotX) / normSq // ‖r_∥‖² = (r·û)²
	return rDotR + (eta-1)*par
}

// accumulateAnisotropic adds one point's contribution to a cluster's weighted
// least-squares system A c = b: A += η·P + (I−P), b += (η·P + (I−P)) x, where
// P = x xᵀ/‖x‖². Using M ≜ η·P + (I−P) = I + (η−1)·P, M x = x + (η−1)(x·x/‖x‖²)x =
// x + (η−1)x = η x (since x·x = ‖x‖²), so the per-point b contribution is simply
// η·x. A's contribution is I + (η−1)·x xᵀ/‖x‖². When ‖x‖² ≈ 0 the P-term is skipped
// (isotropic for that point): A += I, b += x. accA is flat row-major dim×dim.
func accumulateAnisotropic(accA, accB, x []float32, normSq, eta float32, dim int) {
	if normSq <= 0 {
		// Isotropic contribution: A += I, b += x.
		for i := 0; i < dim; i++ {
			accA[i*dim+i]++
			accB[i] += x[i]
		}
		return
	}
	w := (eta - 1) / normSq // scale on x xᵀ
	for i := 0; i < dim; i++ {
		xi := x[i]
		// b += η·x (derived above; exact for x·x = ‖x‖²).
		accB[i] += eta * xi
		row := i * dim
		// A += I (diagonal) + (η−1)/‖x‖² · x xᵀ.
		accA[row+i]++
		wxi := w * xi
		for j := 0; j < dim; j++ {
			accA[row+j] += wxi * x[j]
		}
	}
}

// clusterMean returns the plain mean of the points assigned to cluster ci (the
// η = 1 update), used as the robust fallback when the weighted system is singular.
func clusterMean(train [][]float32, assign []int, ci, dim, count int) []float32 {
	out := make([]float32, dim)
	for vi, x := range train {
		if assign[vi] != ci {
			continue
		}
		for d := 0; d < dim; d++ {
			out[d] += x[d]
		}
	}
	inv := float32(1.0 / float64(count))
	for d := 0; d < dim; d++ {
		out[d] *= inv
	}
	return out
}

// solveLinearSystem solves the dim×dim system A x = b for x via Gaussian
// elimination with partial pivoting (pure Go, no deps). A is flat row-major
// (len dim*dim); b has length dim. It works on COPIES so the caller's accumulators
// are untouched. Returns (solution, true) on success, or (nil, false) when the
// matrix is singular (a pivot column is ~0), letting the caller fall back. dim is
// small (the PQ sub-vector dimension, 4..16), so the O(dim³) cost is negligible.
func solveLinearSystem(a, b []float32, dim int) ([]float32, bool) {
	// Work in float64 for the elimination to keep the small-pivot arithmetic stable,
	// then cast the solution back to float32 (the codebooks are float32).
	m := make([]float64, dim*dim)
	for i := range m {
		m[i] = float64(a[i])
	}
	rhs := make([]float64, dim)
	for i := range rhs {
		rhs[i] = float64(b[i])
	}
	for col := 0; col < dim; col++ {
		// Partial pivot: pick the row (>= col) with the largest |m[row][col]|.
		pivot := col
		maxAbs := math.Abs(m[col*dim+col])
		for row := col + 1; row < dim; row++ {
			if v := math.Abs(m[row*dim+col]); v > maxAbs {
				maxAbs = v
				pivot = row
			}
		}
		if maxAbs < 1e-12 {
			return nil, false // singular
		}
		if pivot != col {
			for j := 0; j < dim; j++ {
				m[col*dim+j], m[pivot*dim+j] = m[pivot*dim+j], m[col*dim+j]
			}
			rhs[col], rhs[pivot] = rhs[pivot], rhs[col]
		}
		// Eliminate column col below the pivot.
		pv := m[col*dim+col]
		for row := col + 1; row < dim; row++ {
			f := m[row*dim+col] / pv
			if f == 0 {
				continue
			}
			for j := col; j < dim; j++ {
				m[row*dim+j] -= f * m[col*dim+j]
			}
			rhs[row] -= f * rhs[col]
		}
	}
	// Back-substitution.
	x := make([]float32, dim)
	for row := dim - 1; row >= 0; row-- {
		sum := rhs[row]
		for j := row + 1; j < dim; j++ {
			sum -= m[row*dim+j] * float64(x[j])
		}
		x[row] = float32(sum / m[row*dim+row])
	}
	return x, true
}

// nearestCentroidIdx returns the index of the centroid closest to v under dist,
// breaking ties toward the lower index (the first-seen strictly-smaller wins).
// Shared by the serial and parallel assignment paths so both pick identically.
func nearestCentroidIdx(v []float32, centroids [][]float32, k int, dist distFunc) int {
	best := 0
	bestD := dist(v, centroids[0])
	for ci := 1; ci < k; ci++ {
		if d := dist(v, centroids[ci]); d < bestD {
			bestD = d
			best = ci
		}
	}
	return best
}

// kmeansAssignChunk is the granularity (in points) at which assignParallel hands
// work to goroutines. Small enough that fast cores can grab many chunks while slow
// cores grab few (load balance on heterogeneous / contended CPUs), large enough
// that the per-chunk atomic add is negligible against the chunk's k*dim distance
// work.
const kmeansAssignChunk = 256

// assignParallel runs the assignment step across nWorkers goroutines using DYNAMIC
// chunk dispatch: workers repeatedly claim the next [start,start+chunk) range from
// a shared atomic cursor until train is exhausted. This load-balances across
// asymmetric cores (e.g. Intel P/E hybrids) where a static equal range-split makes
// fast cores idle at the barrier waiting on the slowest core's oversized chunk.
//
// Determinism is unchanged: chunks are disjoint and cover [0,n) exactly, so each
// point is assigned exactly once and assign[vi] is written by exactly one worker —
// the resulting assign[] is identical regardless of which worker (in which order)
// claimed which chunk. Returns whether any assignment changed; the caller still
// runs the centroid-update reduce serially in index order to stay bit-identical.
func assignParallel(train, centroids [][]float32, k int, dist distFunc, assign []int, nWorkers int) bool {
	n := len(train)
	changedFlags := make([]bool, nWorkers) // one per worker: written once, no atomics needed
	var cursor int64                       // shared dispatch cursor (next unclaimed index)
	var wg sync.WaitGroup
	for w := 0; w < nWorkers; w++ {
		wg.Add(1)
		go func(w int) {
			defer wg.Done()
			local := false
			for {
				start := int(atomic.AddInt64(&cursor, kmeansAssignChunk)) - kmeansAssignChunk
				if start >= n {
					break
				}
				end := start + kmeansAssignChunk
				if end > n {
					end = n
				}
				for vi := start; vi < end; vi++ {
					best := nearestCentroidIdx(train[vi], centroids, k, dist)
					if assign[vi] != best {
						assign[vi] = best
						local = true
					}
				}
			}
			changedFlags[w] = local
		}(w)
	}
	wg.Wait()
	for _, c := range changedFlags {
		if c {
			return true
		}
	}
	return false
}

// kmeansPlusPlusInit picks k initial centroids with the D^2-weighted k-means++
// scheme, driven by the supplied seeded rng for determinism. dist is the metric
// distance kernel; D^2 weighting uses dist directly (for L2 this is the squared
// distance the algorithm wants; for cosine/dot the metric distance is a valid
// monotone proxy that biases toward dissimilar seeds).
func kmeansPlusPlusInit(train [][]float32, k int, dist distFunc, rng *rand.Rand) [][]float32 {
	centroids := make([][]float32, 0, k)

	// First centroid: uniform random pick.
	first := rng.Intn(len(train))
	centroids = append(centroids, cloneVec(train[first]))

	// nearest[i] = distance from train[i] to its closest chosen centroid.
	nearest := make([]float32, len(train))
	for i, v := range train {
		nearest[i] = dist(v, centroids[0])
	}

	for len(centroids) < k {
		// Weighted pick proportional to nearest[i] (clamped to >= 0).
		var total float64
		for _, d := range nearest {
			if d > 0 {
				total += float64(d)
			}
		}
		var chosen int
		if total <= 0 {
			// All remaining points coincide with chosen centroids; fall back
			// to a uniform pick to fill the remaining slots deterministically.
			chosen = rng.Intn(len(train))
		} else {
			target := rng.Float64() * total
			var acc float64
			chosen = len(train) - 1
			for i, d := range nearest {
				if d > 0 {
					acc += float64(d)
				}
				if acc >= target {
					chosen = i
					break
				}
			}
		}
		centroids = append(centroids, cloneVec(train[chosen]))

		// Refresh nearest with the newly added centroid.
		c := centroids[len(centroids)-1]
		for i, v := range train {
			if d := dist(v, c); d < nearest[i] {
				nearest[i] = d
			}
		}
	}
	return centroids
}

// farthestPoint returns a clone of the training vector with the largest distance
// to its nearest centroid — used to re-seed an empty cluster.
func farthestPoint(train [][]float32, centroids [][]float32, dist distFunc) []float32 {
	bestIdx := 0
	var bestD float32 = -1
	for i, v := range train {
		// distance to nearest centroid
		nd := dist(v, centroids[0])
		for ci := 1; ci < len(centroids); ci++ {
			if d := dist(v, centroids[ci]); d < nd {
				nd = d
			}
		}
		if nd > bestD {
			bestD = nd
			bestIdx = i
		}
	}
	return cloneVec(train[bestIdx])
}

// centroidShift2 returns the summed squared L2 drift between two centroid sets.
// Used purely for convergence early-exit (metric-independent: any movement is
// movement).
func centroidShift2(a, b [][]float32) float32 {
	var sum float32
	for ci := range a {
		sum += l2SquaredScalar(a[ci], b[ci])
	}
	return sum
}

// dedupVectors returns clones of the input with exact-duplicate vectors removed,
// preserving first-seen order. Used for the k >= n centroid path.
func dedupVectors(vectors [][]float32) [][]float32 {
	out := make([][]float32, 0, len(vectors))
	for _, v := range vectors {
		dup := false
		for _, o := range out {
			if vecEqual(v, o) {
				dup = true
				break
			}
		}
		if !dup {
			out = append(out, cloneVec(v))
		}
	}
	return out
}

// sampleVectors draws n vectors from the input without replacement using a
// seeded partial Fisher-Yates over an index permutation, returning clones in
// index order so the result is deterministic for a given rng state.
func sampleVectors(vectors [][]float32, n int, rng *rand.Rand) [][]float32 {
	idx := make([]int, len(vectors))
	for i := range idx {
		idx[i] = i
	}
	for i := 0; i < n; i++ {
		j := i + rng.Intn(len(idx)-i)
		idx[i], idx[j] = idx[j], idx[i]
	}
	picked := idx[:n]
	// Sort picked indices so the output order is stable (selection order above
	// is already deterministic, but stable ordering keeps centroid init
	// independent of the swap order).
	insertionSortInts(picked)
	out := make([][]float32, n)
	for i, p := range picked {
		out[i] = cloneVec(vectors[p])
	}
	return out
}

// insertionSortInts sorts a small int slice in place (avoids pulling sort for a
// tiny, hot-path-irrelevant helper).
func insertionSortInts(a []int) {
	for i := 1; i < len(a); i++ {
		v := a[i]
		j := i - 1
		for j >= 0 && a[j] > v {
			a[j+1] = a[j]
			j--
		}
		a[j+1] = v
	}
}

func cloneVec(v []float32) []float32 {
	out := make([]float32, len(v))
	copy(out, v)
	return out
}

func vecEqual(a, b []float32) bool {
	if len(a) != len(b) {
		return false
	}
	for i := range a {
		if a[i] != b[i] {
			return false
		}
	}
	return true
}
