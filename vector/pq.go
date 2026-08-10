// SPDX-License-Identifier: Apache-2.0

package vector

import (
	"errors"
	"sync"
)

// pqRotScratch pools dim-length scratch buffers for the OPQ encode rotation (Rx)
// so the per-insert encodeInto path is allocation-free. Build re-encodes in
// parallel, so the pool must be concurrency-safe (sync.Pool is). Pointers are
// stored so Put doesn't box a slice header into the interface (which would itself
// allocate, defeating the purpose).
var pqRotScratch sync.Pool

// getRotScratch returns a *[]float32 of length n (grown if the pooled buffer is
// too small). Pair every call with putRotScratch.
func getRotScratch(n int) *[]float32 {
	if p, _ := pqRotScratch.Get().(*[]float32); p != nil {
		if cap(*p) < n {
			*p = make([]float32, n)
		} else {
			*p = (*p)[:n]
		}
		return p
	}
	s := make([]float32, n)
	return &s
}

// putRotScratch returns a scratch buffer to the pool.
func putRotScratch(p *[]float32) { pqRotScratch.Put(p) }

// Product Quantization (PQ) codec with Asymmetric Distance Computation (ADC).
//
// A PQ codec splits each dim-dimensional vector into m contiguous sub-vectors of
// dsub = dim/m components and quantizes each sub-vector independently against its
// own 256-entry codebook (nbits=8 → one byte per sub-code, m bytes per vector).
// The codebooks are trained with the existing per-subspace k-means.
//
// CODEBOOK TRAINING + ENCODING ALWAYS USE L2 (squared Euclidean), independent of
// the collection metric. PQ's job is to minimize the sub-vector RECONSTRUCTION
// error ‖x^s − c^s‖²; the nearest sub-centroid by L2 is the minimal-reconstruction
// choice. Only the ADC LUT aggregation depends on the metric (squared-L2 sum for
// L2; dot sum for IP/Cosine). Training/encoding with the collection metric instead
// is both wrong (cosine on tiny non-normalized residual sub-vectors does not
// minimize reconstruction error) and pathologically slow (near-constant cosine
// distances starve k-means++). This is the canonical IVFPQ design.
//
// RESIDUAL-AWARE: this codec is metric-correct and centroid-agnostic; it operates
// on whatever vectors it is given. The IVF integration trains and encodes
// RESIDUALS (vec − coarse-centroid) and builds the query LUT from the residual
// query (q − coarse-centroid) per probed cell. trainPQ/Encode/queryLUT therefore
// take the residual vectors directly — the codebooks become residual codebooks
// and the LUT a residual-query LUT. The codec itself does not know or need the
// centroid; the IVF computes the subtraction per cell.
//
// ADC: the query is kept in full float32. For a query q, queryLUT precomputes a
// flat [m*256]float32 table holding, for each subspace s and sub-centroid c, the
// metric contribution of q's s-th sub-vector against codebook[s][c]. The distance
// to a stored code is then Σ_{s} lut[s*256 + code[s]] — m byte-indexed lookups,
// no allocation in the inner sum.
//
// METRIC ORIENTATION (matches metricDist / pickDistDim — LOWER = better for all
// three metrics):
//   - L2:        per-subspace LUT holds squared-L2(q_sub, centroid_sub); the
//                sum is the full squared-L2. lower = nearer. adc offset = 0.
//   - Cosine:    pre-normalized vectors → distance is 1 − dot. dot is additive
//                across subspaces, so the LUT holds the NEGATED per-subspace dot
//                (−Σ q_sub·centroid_sub) and adc adds the constant 1 once, giving
//                1 − dot. lower = nearer.
//   - DotProduct: distance is −dot. The LUT holds the negated per-subspace dot
//                and adc adds 0, giving −dot. lower = nearer.
//
// This keeps adc(queryLUT(q), Encode(v)) ≈ pickDistDim(metric,dim)(q, v) with the
// same sign/direction the IVF gather sorts by (the shortlist sorts ascending).

// pqCodebookSize is the number of sub-centroids per subspace for nbits=8.
const pqCodebookSize = 256

// maxOPQIters caps Config.OPQIters (full-OPQ iterative Procrustes refinement).
// Each iteration costs O(d²·jacobiSweeps) for the SVD plus a full PQ k-means
// retrain, a one-time training cost; the cap bounds it on large dim (768/1536).
// Validate rejects OPQIters > maxOPQIters fail-loud (ErrInvalidOPQIters).
const maxOPQIters = 20

var (
	// errInvalidPQM is returned by trainPQ when m <= 0.
	errInvalidPQM = errors.New("vector: invalid PQ m (must be > 0)")
	// errPQDimNotDivisible is returned by trainPQ when dim is not a positive
	// multiple of m (PQ requires equal contiguous sub-vectors).
	errPQDimNotDivisible = errors.New("vector: PQ requires dim % m == 0")
)

// pq is a residual-aware product quantizer. codebooks[s] is the 256×dsub codebook
// for subspace s; codebooks[s][c] is the c-th sub-centroid (a dsub-length vector).
type pq struct {
	m         int           // number of sub-quantizers (subspaces)
	dsub      int           // sub-vector dimension = dim/m
	dim       int           // full vector dimension = m*dsub
	metric    Metric        // the collection metric (drives LUT contribution + offset)
	codebooks [][][]float32 // [m][≤K][dsub], K = codebookSize()

	// nbits is the per-subspace code width: 0/8 ⇒ 256 sub-centroids (the default
	// 8-bit codec, byte-identical), 4 ⇒ 16 (the LUT16 fast-scan codec, pq4.go).
	// It controls ONLY the k passed to trainCodebooks' k-means; the 8-bit encode/
	// queryLUT/adc hot paths still use the pqCodebookSize literal so they are
	// provably unchanged. The 4-bit codec drives nibble-packing + the quantized
	// fast-scan LUT through the pq4 wrapper.
	nbits int

	// rotation is the OPQ orthogonal rotation R (dim×dim, flat row-major), or
	// nil when OPQ is off. When non-nil, encode/train/query rotate the input
	// (Rx) BEFORE the per-subspace split so the codebooks live in rotated space,
	// and reconstruct un-rotates (Rᵀ) AFTER concatenating the sub-centroids so
	// the reconstructed vector returns to the ORIGINAL space. nil ⇒ every apply
	// site is a no-op, byte/behaviour-identical to plain PQ.
	rotation []float32
}

// pqMetricOffset is the constant added to the adc sum so the result matches the
// metric's full distance: Cosine's distance is 1 − dot, whose per-subspace dots
// the LUT negates and sums to −dot, so the offset restores the +1. L2 and
// DotProduct are fully additive (offset 0).
func pqMetricOffset(metric Metric) float32 {
	if metric == Cosine {
		return 1
	}
	return 0
}

// codebookSize returns the number of sub-centroids per subspace: 16 for the 4-bit
// LUT16 codec (nbits==4), 256 otherwise (the 8-bit default). This is the k passed
// to the per-subspace k-means in trainCodebooks.
func (p *pq) codebookSize() int {
	if p.nbits == 4 {
		return pqCodebookSize4
	}
	return pqCodebookSize
}

// trainPQ trains an m-subspace PQ codec on vecs (the IVF passes residuals; trainPQ
// is agnostic to whether they are residuals). Each vector is split into m
// contiguous sub-vectors of dsub = dim/m components; per-subspace k-means with
// k=256 yields the residual codebooks. seed is offset per subspace so the m
// k-means runs are independent yet deterministic. Requires dim%m == 0 and m > 0.
//
// Small training sets (n < 256): kmeans already returns the (deduplicated) input
// vectors as centroids when k >= n, so a subspace may end up with fewer than 256
// sub-centroids. Encode/queryLUT use len(codebook[s]) rather than a hard 256, so
// the codec stays correct and panic-free for small n.
//
// workers parallelizes each subspace's k-means assignment step (deterministic;
// workers<=1 ⇒ serial). The m subspaces are trained sequentially so the per-
// subspace seeding and codebooks stay byte-identical regardless of workers.
// opq, when true, builds a seeded random orthogonal rotation R = randomOrthogonal
// (dim, seed) and rotates every training vector to Rx before the per-subspace
// k-means, so the codebooks are learned in the balanced (rotated) space. R is
// stored on the returned pq and re-applied (and un-applied) by encode/query/
// reconstruct. opq=false ⇒ rotation stays nil and trainPQ is byte-identical to
// plain PQ. The rotation seed reuses the codec seed (so the same seed ⇒ same R).
//
// opqIters drives FULL-OPQ iterative Procrustes refinement (Ge et al. 2013). It
// is only consulted when opq is true:
//   - opqIters <= 1 (incl. the default 0): the EXISTING v1 path runs VERBATIM —
//     a single seeded random R + ONE PQ train, NO SVD. This is the BYTE-IDENTICAL
//     guarantee for OPQIters<=1, made structural (the short-circuit below) rather
//     than coincidental.
//   - opqIters > 1: iter-0 is the SAME seeded random R + train (matching the v1
//     starting point), then iters 1..opqIters-1 reconstruct each training vector
//     from its current PQ code, solve the orthogonal Procrustes rotation
//     R = V·Uᵀ from M = Σ x·x̂ᵀ (deterministic Jacobi SVD — see svd.go), set it
//     as the new R, re-rotate the ORIGINAL training set, and re-train PQ. The
//     final R + codebooks are returned. Every step is a pure function of the
//     (slot-ordered) input + fixed seed + deterministic SVD, so replicas converge
//     to bit-identical R + codebooks.
//
// eta is the anisotropic score-aware quantization weight (Config.AnisotropicEta):
// 0 or 1 ⇒ the EXISTING isotropic L2 k-means runs verbatim (byte-identical); > 1
// ⇒ trainCodebooks uses kmeansAnisotropic, weighting parallel (score-direction)
// reconstruction error by eta. The codec (encode/ADC/snapshot) is UNCHANGED — only
// the learned codebooks differ — so eta rides Config only, no wire/snapshot format
// change. With OPQ the codebooks are learned in the rotated space, so eta weights
// the rotated-space sub-vectors' own directions (the per-subspace v1 simplification
// applies to the rotated sub-vectors; see kmeansAnisotropic).
// nbits is the per-subspace code width: 0/8 ⇒ the 256-centroid 8-bit codec
// (byte-identical to before — every existing caller passes 8 or 0), 4 ⇒ the
// 16-centroid LUT16 codec (k-means runs with k=16; the returned codec's nbits is
// stamped so trainCodebooks and the pq4 wrapper agree on the width). Only the
// trained codebook SIZE differs for 4-bit; the training objective (L2 per-subspace
// reconstruction, composing with the anisotropic η) is identical.
func trainPQ(vecs [][]float32, m, dim int, seed int64, metric Metric, workers int, opq bool, opqIters int, eta float32, nbits int) (*pq, error) {
	if m <= 0 {
		return nil, errInvalidPQM
	}
	if dim <= 0 || dim%m != 0 {
		return nil, errPQDimNotDivisible
	}
	dsub := dim / m

	p := &pq{
		m:         m,
		dsub:      dsub,
		dim:       dim,
		metric:    metric,
		nbits:     nbits,
		codebooks: make([][][]float32, m),
	}

	// orig holds the ORIGINAL (un-rotated) training vectors so the >1 refinement
	// loop can re-rotate from scratch with each new R. For the OPQIters<=1 path it
	// is unused (the existing code path is taken verbatim below).
	orig := vecs

	if opq {
		// Build R deterministically from the codec seed and rotate every training
		// vector to Rx so the codebooks are residual-rotated-space codebooks. This
		// is iter-0 of full-OPQ AND the entire v1 path (when opqIters<=1).
		p.rotation = randomOrthogonal(dim, seed)
		rotated := make([][]float32, len(vecs))
		for i, v := range vecs {
			rotated[i] = rotate(p.rotation, v)
		}
		vecs = rotated
	}

	// Train PQ once on the current (possibly rotated) vecs. Factored so the
	// refinement loop can re-invoke it after each new rotation.
	p.trainCodebooks(vecs, seed, workers, eta)

	// FULL-OPQ refinement: only when opq is on AND opqIters>1. The OPQIters<=1 path
	// returns here, having taken the EXACT v1 code path above (seeded random R, one
	// train, no SVD) → byte-identical by construction.
	//
	// MONOTONICITY NOTE: the net error trend across iters is downward, but individual
	// iterations can transiently raise error by ~0.1–0.2% because k-means is re-seeded
	// at each iter (not warm-started). This is NOT a bug: the Procrustes step finds the
	// GLOBALLY optimal R for fixed codebook assignments; k-means then may shift centroids
	// in a way that momentarily increases reconstruction error before converging. More
	// iters ≈ better quality overall, but NOT strictly monotone per-iter. The output is
	// always bit-identical (deterministic Jacobi SVD + fixed slot order) regardless.
	if opq && opqIters > 1 {
		// Scratch buffers reused across iterations: xhat[i] is the reconstruction of
		// orig[i] (the un-rotated original space), rotated[i] is R·orig[i].
		yhat := make([][]float32, len(orig))
		rotated := make([][]float32, len(orig))
		for iter := 1; iter < opqIters; iter++ {
			// For each training vector, in FIXED slot order, recover its quantized
			// reconstruction in ROTATED space ŷ (the concatenated sub-centroids, BEFORE
			// the Rᵀ un-rotation). The OPQ objective minimizes ‖Rx − ŷ‖ over R with the
			// codebook assignments fixed; the Procrustes solution is R = V·Uᵀ from
			// M = Σ x·ŷᵀ (Ge et al. 2013). Pairing orig (x) with the rotated-space ŷ —
			// NOT the un-rotated reconstruction — is what makes the refinement actually
			// reduce the reconstruction error.
			for i, v := range orig {
				code := p.encode(v) // rotates by current R, then nearest sub-centroids
				yhat[i] = p.reconstructRotated(code)
			}
			// Solve the optimal rotation R = V·Uᵀ from M = Σ x·ŷᵀ (slot-ordered
			// accumulation inside procrustesRotation). Deterministic Jacobi SVD.
			p.rotation = procrustesRotation(orig, yhat, dim)
			// Re-rotate the ORIGINAL training set with the new R and re-train.
			for i, v := range orig {
				rotated[i] = rotate(p.rotation, v)
			}
			p.trainCodebooks(rotated, seed, workers, eta)
		}
	}
	return p, nil
}

// trainCodebooks (re)trains the m per-subspace k-means codebooks on the given
// (already-rotated, if OPQ) vectors and stores them on p. The m subspaces are
// trained sequentially with per-subspace seed seed+s so the codebooks stay
// byte-identical regardless of workers (see trainPQ header). Called once per PQ
// train — by trainPQ's initial train and by each full-OPQ refinement iteration.
// eta is the anisotropic score-aware weight (Config.AnisotropicEta): eta > 1 trains
// each subspace with kmeansAnisotropic (parallel/score-direction error weighted by
// eta); eta ∈ {0,1} runs the EXISTING kmeans(..., L2, ...) path VERBATIM, so the
// codebooks are byte-identical to the isotropic trainer (the back-compat anchor).
func (p *pq) trainCodebooks(vecs [][]float32, seed int64, workers int, eta float32) {
	// Reusable buffer of sub-vectors for the current subspace.
	subVecs := make([][]float32, len(vecs))
	for s := 0; s < p.m; s++ {
		lo := s * p.dsub
		hi := lo + p.dsub
		for i, v := range vecs {
			// Slice the contiguous sub-vector; kmeans clones internally
			// (cloneVec / dedupVectors), so sharing backing storage is safe.
			subVecs[i] = v[lo:hi]
		}
		// Train with L2: codebooks minimize sub-vector reconstruction error
		// regardless of the collection metric (see file header). eta > 1 switches
		// to the anisotropic (score-aware) trainer; eta ∈ {0,1} is the verbatim
		// isotropic path (kmeansAnisotropic short-circuits eta ≤ 1 to kmeans, but
		// branch here too so the isotropic path is provably untouched).
		k := p.codebookSize()
		var cb [][]float32
		if eta > 1 {
			cb = kmeansAnisotropic(subVecs, k, seed+int64(s), eta, workers)
		} else {
			cb = kmeans(subVecs, k, seed+int64(s), L2, workers)
		}
		// kmeans returns nil only for empty input; with vecs non-empty it always
		// returns at least one centroid. Guard defensively for the empty case.
		if cb == nil {
			cb = [][]float32{make([]float32, p.dsub)}
		}
		p.codebooks[s] = cb
	}
}

// encode returns the m-byte PQ code for vec (allocating). It is the internal
// twin of the exported Encode, used by the full-OPQ refinement loop. Both rotate
// by the current R (if any) before the subspace split — see encodeInto.
func (p *pq) encode(vec []float32) []byte {
	code := make([]byte, p.m)
	p.encodeInto(code, vec)
	return code
}

// CodeLen returns the encoded length in bytes: one sub-code byte per subspace (m).
func (p *pq) CodeLen() int { return p.m }

// Encode writes the m-byte PQ code for vec into a freshly allocated slice and
// returns it. For each subspace it finds the nearest sub-centroid (via the metric
// distance over the dsub-length sub-vector) and stores its index as a byte. The
// IVF passes the residual (vec − coarse-centroid); Encode is agnostic.
func (p *pq) Encode(vec []float32) []byte {
	code := make([]byte, p.m)
	p.encodeInto(code, vec)
	return code
}

// encodeInto is Encode without allocation; len(dst) must be CodeLen() (==m). Used
// by the quantizer-interface adapter so the arena can encode in place.
func (p *pq) encodeInto(dst []byte, vec []float32) {
	// OPQ: rotate the input to Rx BEFORE the subspace split so the sub-vectors
	// match the rotated-space codebooks. The rotation scratch comes from a pool
	// so this hot insert path stays allocation-free. nil rotation ⇒ no-op (plain
	// PQ), no scratch taken.
	var scratch *[]float32
	if p.rotation != nil {
		scratch = getRotScratch(len(vec))
		rotateInto(*scratch, p.rotation, vec)
		vec = *scratch
	}
	// Encode by L2 (minimal reconstruction error), independent of the metric.
	dist := pickDistDim(L2, p.dsub)
	for s := 0; s < p.m; s++ {
		sub := vec[s*p.dsub : s*p.dsub+p.dsub]
		cb := p.codebooks[s]
		best := 0
		bestD := dist(sub, cb[0])
		for c := 1; c < len(cb); c++ {
			if d := dist(sub, cb[c]); d < bestD {
				bestD = d
				best = c
			}
		}
		//nolint:gosec // best ∈ [0,255]: len(cb) <= 256 by construction.
		dst[s] = byte(best)
	}
	if scratch != nil {
		putRotScratch(scratch)
	}
}

// reconstruct concatenates the m sub-centroids selected by code into a single
// dim-length vector. For residual codes this reconstructs the residual; the caller
// adds back the coarse centroid for the opt-in exact-rerank-without-floats path.
func (p *pq) reconstruct(code []byte) []float32 {
	out := make([]float32, p.dim)
	for s := 0; s < p.m; s++ {
		sub := p.codebooks[s][code[s]]
		copy(out[s*p.dsub:], sub)
	}
	// OPQ: the concatenated sub-centroids live in ROTATED space (codebooks were
	// trained on Rx); un-rotate by Rᵀ so the returned vector is back in the
	// ORIGINAL space. This is the inverse of encode's Rx (RᵀR = I). The IVF
	// reconstruct path (vecFor) then adds the coarse centroid in original space.
	// nil rotation ⇒ no-op (plain PQ).
	if p.rotation != nil {
		out = rotateT(p.rotation, out)
	}
	return out
}

// reconstructRotated concatenates the m sub-centroids selected by code WITHOUT the
// Rᵀ un-rotation, returning the reconstruction in ROTATED space (ŷ). It is the
// codebook decode that the full-OPQ Procrustes step compares against Rx (the OPQ
// objective is ‖Rx − ŷ‖); reconstruct() is this plus the Rᵀ un-rotation back to
// original space (used everywhere else). Used only by the trainPQ refinement loop.
func (p *pq) reconstructRotated(code []byte) []float32 {
	out := make([]float32, p.dim)
	for s := 0; s < p.m; s++ {
		sub := p.codebooks[s][code[s]]
		copy(out[s*p.dsub:], sub)
	}
	return out
}

// queryLUT precomputes the flat [m*256]float32 ADC lookup table for query q (the
// IVF passes the residual query q − coarse-centroid). lut[s*256 + c] holds the
// metric contribution of q's s-th sub-vector against codebook[s][c]:
//   - L2:                squared-L2(q_sub, centroid_sub)   (additive → full L2)
//   - Cosine/DotProduct: −dot(q_sub, centroid_sub)         (adc adds the offset)
//
// Entries for c >= len(codebook[s]) (small-n training) are left at the zero value;
// such indices are never produced by Encode, so they are never read.
//
// Allocates once per (query, cell). The per-candidate adc that consumes it does no
// allocation. Slots are sized to the full m*256 so adc can index code bytes (which
// are < 256) without bounds-special-casing.
func (p *pq) queryLUT(q []float32) []float32 {
	lut := make([]float32, p.lutLen())
	p.queryLUTInto(lut, q)
	return lut
}

// lutLen is the flat ADC LUT length (m sub-quantizers × 256 codes).
func (p *pq) lutLen() int { return p.m * pqCodebookSize }

// queryLUTInto fills dst[:lutLen()] with the ADC LUT for q, allocation-free, so a
// caller (the IVF gather) can reuse ONE buffer across all probed cells instead of
// allocating m*256 floats per cell. Only entries [base, base+len(cb[s])) are
// written per subspace; entries for c >= len(cb[s]) are left as-is and are never
// read by adc (Encode never produces such indices), so a reused buffer needs no
// clearing between cells — the written region is fully overwritten each call and
// the codebooks (hence the written index ranges) are identical across cells.
func (p *pq) queryLUTInto(dst []float32, q []float32) {
	// OPQ: rotate the query to Rq BEFORE the subspace split so its sub-vectors
	// align with the rotated-space codebooks (the rotation is an isometry, so
	// adc(Rq, code) still matches the metric distance). The rotation scratch comes
	// from the same pool as encodeInto so the per-cell IVF gather (one queryLUTInto
	// per probed cell) stays allocation-free. nil ⇒ no-op (plain PQ), no scratch.
	var scratch *[]float32
	if p.rotation != nil {
		scratch = getRotScratch(len(q))
		rotateInto(*scratch, p.rotation, q)
		q = *scratch
	}
	negDot := p.metric != L2
	for s := 0; s < p.m; s++ {
		sub := q[s*p.dsub : s*p.dsub+p.dsub]
		cb := p.codebooks[s]
		base := s * pqCodebookSize
		for c := 0; c < len(cb); c++ {
			if negDot {
				dst[base+c] = -dotProduct(sub, cb[c])
			} else {
				dst[base+c] = l2Squared(sub, cb[c])
			}
		}
	}
	if scratch != nil {
		putRotScratch(scratch)
	}
}

// adc computes the asymmetric distance between the prepared query LUT and a stored
// m-byte code: Σ_{s=0..m-1} lut[s*256 + code[s]], plus the metric offset (1 for
// Cosine so the result is 1 − dot; 0 otherwise). Lower = nearer, matching
// metricDist. The inner sum is m byte-indexed lookups with NO allocation.
func (p *pq) adc(lut []float32, code []byte) float32 {
	sum := pqMetricOffset(p.metric)
	m := p.m
	// Re-slice both operands to their exact, m-derived lengths ONCE, up front:
	//   - code[:m]      → the range loop visits exactly m subspaces (and a short
	//                     code panics here, deterministically, not OOB-reads).
	//   - lut[:m*256]   → makes lut's length a function of m, so the inner index
	//                     s*256 + c (with s < m and c a uint8 < 256, hence
	//                     s*256+c < m*256) is provably in range and the compiler
	//                     elides the per-iteration bounds check on the LUT read.
	// The two slice checks are hoisted out of the loop (one each), versus the
	// naive form's per-iteration IsInBounds on lut[s*256+int(code[s])].
	//
	// The float additions stay in strict subspace order 0,1,2,...,m-1 with a
	// single accumulator, so this is bit-identical to the naive
	// Σ lut[s*256+code[s]]. (A single-accumulator unroll-by-4 was benchmarked
	// and is NOT faster — the strict-order adds form one dependency chain the
	// unroll can't break without reordering, which would break bit-identicality;
	// correctness wins, so we keep the clean bounds-check-eliminated loop.)
	code = code[:m]
	lut = lut[: m*pqCodebookSize : m*pqCodebookSize]
	for s, c := range code {
		sum += lut[s*pqCodebookSize+int(c)]
	}
	return sum
}
