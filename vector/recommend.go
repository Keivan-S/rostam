// SPDX-License-Identifier: Apache-2.0

package vector

import (
	"math"
	"time"
)

// RecommendOpts configures a recommendation query: find vectors similar to the
// positive examples and away from the negative ones. At least one positive
// example id is required.
type RecommendOpts struct {
	Positive []uint64 // ids of liked examples (at least one must exist in the index)
	Negative []uint64 // ids of disliked examples (optional)
	Filter   Filter
}

// Recommend returns up to k vectors similar to the positive examples and away
// from the negatives, using a synthesized query vector of mean(positive) -
// mean(negative). The supplied example ids are excluded from the results.
// Requires at least one positive example present in the index.
func (h *hnsw) Recommend(k int, opts RecommendOpts) ([]Result, error) {
	start := time.Now()
	defer func() { h.searchLat.observe(time.Since(start)) }()

	if k <= 0 {
		return nil, nil
	}
	if len(opts.Positive) == 0 {
		return nil, ErrNoRecommendExamples
	}

	// Build the target = normalize(mean(positive) - mean(negative)) via the shared
	// derive helper (vecsForIDs reconstructs PQ-dropped floats; the cosine
	// normalize is idempotent with SearchInto's own normalize below).
	target, err := h.deriveRecommendVector(opts.Positive, opts.Negative)
	if err != nil {
		return nil, err
	}

	// Exclude the example ids from the result set.
	exclude := make(map[uint64]bool, len(opts.Positive)+len(opts.Negative))
	for _, id := range opts.Positive {
		exclude[id] = true
	}
	for _, id := range opts.Negative {
		exclude[id] = true
	}

	// Over-fetch so k results remain after dropping the examples.
	res, err := h.SearchInto(nil, target, k+len(exclude), opts.Filter)
	if err != nil {
		return nil, err
	}
	out := res[:0]
	for _, r := range res {
		if exclude[r.ID] {
			continue
		}
		out = append(out, r)
		if len(out) == k {
			break
		}
	}
	return out, nil
}

// RecommendStrategy and its RecommendAverageVector / RecommendBestScore constants
// now live in the engine-free vtypes leaf package and are re-exported from
// vtypes_aliases.go.

// RecommendVecsOpts configures a BEST_SCORE recommend whose positive/negative
// example vectors are already RESOLVED to floats (the Query API leaf form). It is
// the recommend analogue of DiscoverVecsOpts: the coordinator resolves the example
// ids cluster-wide and embeds the vectors here, so each partition scores against
// them without re-resolving. Positive must carry at least one vector (the seed
// pool + the max-positive similarity need it); Negative is optional (steers away).
type RecommendVecsOpts struct {
	Positive [][]float32 // resolved positive example vectors (at least one)
	Negative [][]float32 // resolved negative example vectors (optional)
	FetchK   int         // candidate pool size; 0 (or < k) = max(4*k, 50)
	Filter   Filter
}

// recommendSim is the metric's true SIMILARITY (higher = more similar) used by the
// BEST_SCORE merge — distinct from discoverScore's -dist convention because here the
// ABSOLUTE value AND SIGN of the score are returned (a -1 cosine offset would corrupt
// the merge), not just compared. Matching Qdrant's similarity:
//
//   - Cosine: dot = 1 - dist (range [-1,1]; +1 identical, -1 opposite).
//   - DotProduct: dot = -dist (the raw inner product).
//   - L2: -dist = -squared-L2 (a monotonic similarity: nearer ⇒ larger, ≤ 0; L2 has
//     no bounded similarity, so the negated distance is the consistent choice — order-
//     preserving and the SAME convention discover/MMR use for L2).
//
// The caller supplies the metric's distFunc so the same kernels (and SIMD dispatch)
// the engine uses everywhere compute the underlying distance.
func recommendSim(cv, x []float32, metric Metric, dist distFunc) float32 {
	d := dist(cv, x)
	switch metric {
	case Cosine:
		return 1 - d // dot over normalized vectors
	default: // DotProduct (-dist = dot) and L2 (-dist = -squared-L2, monotonic)
		return -d
	}
}

// bestScore is the Qdrant BEST_SCORE per-candidate merge: max_pos is the largest
// SIMILARITY of the candidate to any positive example; max_neg is the largest
// similarity to any negative (recommendSim — the metric's true similarity, higher =
// better). The merge favors candidates dominated by a positive (returns max_pos) and
// penalizes candidates dominated by a negative (returns -max_neg, i.e. a candidate
// closer to a negative than to any positive scores -max_neg, below any positive-
// dominated candidate). With no negatives the candidate keeps its max_pos (max_neg =
// -inf). posVecs MUST be non-empty (the caller validates).
func bestScore(cv []float32, posVecs, negVecs [][]float32, metric Metric, dist distFunc) float32 {
	maxPos := float32(-math.MaxFloat32)
	for _, p := range posVecs {
		if s := recommendSim(cv, p, metric, dist); s > maxPos {
			maxPos = s
		}
	}
	if len(negVecs) == 0 {
		// No negatives: the candidate keeps its best positive similarity (max_neg = -inf
		// ⇒ max_pos > max_neg ⇒ return max_pos).
		return maxPos
	}
	maxNeg := float32(-math.MaxFloat32)
	for _, n := range negVecs {
		if s := recommendSim(cv, n, metric, dist); s > maxNeg {
			maxNeg = s
		}
	}
	if maxPos > maxNeg {
		return maxPos
	}
	return -maxNeg
}

// recommendBestSeed builds the BEST_SCORE candidate-pool SEED query into dst
// (length dim): the mean (centroid) of the positive example vectors, then a Cosine
// L2-normalize (idempotent with the search's own normalize). It is the shared seed
// logic for every BEST_SCORE path (hnsw / ivf) so the pool query is identical, and
// mirrors discoverSeed's positive-centroid seed (best-score has no separate target
// anchor — the positives' centroid is the natural pool center). dst must have
// length dim; posVecs must be non-empty.
func recommendBestSeed(dst []float32, posVecs [][]float32, metric Metric) {
	for i := range dst {
		dst[i] = 0
	}
	for _, p := range posVecs {
		for i := range dst {
			dst[i] += p[i]
		}
	}
	if len(posVecs) > 0 {
		inv := 1.0 / float32(len(posVecs))
		for i := range dst {
			dst[i] *= inv
		}
	}
	if metric == Cosine {
		normalize(dst)
	}
}

// sortRecommendBest sorts cands by the BEST_SCORE contract: score DESCENDING, pool
// distance ASCENDING as the tiebreak (mirrors sortDiscover). Shared by every
// BEST_SCORE path.
func sortRecommendBest(cands []Result) { sortDiscover(cands) }

// deriveRecommendVector resolves the example point-ids to their stored vectors
// and derives the recommend query vector mean(positive) - mean(negative), then
// metric-normalizes it (Cosine → L2-normalize the derived vector so it scores
// identically to a stored cosine vector; DotProduct / L2 → as-is). It is the
// extracted, REUSABLE derive+normalize half of Recommend (the SEARCH half becomes
// the rewritten dense leaf in the Query API). It fails loud (ErrNoRecommendExamples)
// when no positive ids are given and (ErrIDNotFound) when NONE of the positive ids
// resolve to a stored vector (the mean would be undefined). Missing negatives are
// simply skipped (they only steer the result). Vectors are resolved via vecsForIDs
// so the PQ-drop reconstruct path (vecFor) is handled transparently.
func (h *hnsw) deriveRecommendVector(positive, negative []uint64) ([]float32, error) {
	return deriveRecommendVector(h.cfg.Dim, h.cfg.Metric, h.vecsForIDs, positive, negative)
}

// deriveRecommendVector is the metric/dim-parameterized derive used by every
// index family (HNSW / IVF) and by the Query API coordinator pre-pass: it takes
// the dimension, the metric, and a batch id→vector resolver (vecsForIDs, which
// already reconstructs PQ-dropped floats) so the same derive math runs everywhere.
func deriveRecommendVector(dim int, metric Metric, vecsForIDs func([]uint64) map[uint64][]float32, positive, negative []uint64) ([]float32, error) {
	if len(positive) == 0 {
		return nil, ErrNoRecommendExamples
	}
	resolved := vecsForIDs(append(append([]uint64(nil), positive...), negative...))
	return DeriveRecommendVector(dim, metric, resolved, positive, negative)
}

// DeriveRecommendVector is the exported coordinator-side derive: it runs the SAME
// mean(positive) - mean(negative) + metric-normalize math as the single-node
// deriveRecommendVector, but takes an ALREADY-RESOLVED id→vector map instead of a
// vecsForIDs resolver. The cluster coordinator-derive pre-pass builds `resolved`
// via a cluster-wide batch-get (the example ids may span partitions) and calls
// this ONCE to derive the query vector, so the derive is partition-invariant: the
// rewritten dense leaf is then fanned out and produces the SAME top-k as the
// single-node path on the same example set. Fails loud the same way: no positives
// given → ErrNoRecommendExamples; NONE of the positives resolve → ErrIDNotFound.
func DeriveRecommendVector(dim int, metric Metric, resolved map[uint64][]float32, positive, negative []uint64) ([]float32, error) {
	if len(positive) == 0 {
		return nil, ErrNoRecommendExamples
	}
	target := make([]float32, dim)
	nPos := meanInto(target, positive, resolved)
	if nPos == 0 {
		return nil, ErrIDNotFound
	}
	if len(negative) > 0 {
		neg := make([]float32, dim)
		if meanInto(neg, negative, resolved) > 0 {
			for i := range target {
				target[i] -= neg[i]
			}
		}
	}
	// Metric-aware normalize, mirroring Recommend's SearchInto contract: Cosine
	// scores against pre-normalized stored vectors, so the derived query is
	// L2-normalized; DotProduct / L2 keep the raw mean-difference magnitude.
	if metric == Cosine {
		normalize(target)
	}
	return target, nil
}

// meanInto sets dst to the mean of the resolved vectors for ids (skipping ids
// absent from resolved), returning the count averaged. The id→vector map is the
// vecsForIDs result (already a per-id COPY), so this is a pure, lock-free fold —
// the lock-held arena analogue is meanOf.
func meanInto(dst []float32, ids []uint64, resolved map[uint64][]float32) int {
	for i := range dst {
		dst[i] = 0
	}
	n := 0
	for _, id := range ids {
		v, ok := resolved[id]
		if !ok || len(v) != len(dst) {
			continue
		}
		for i := range dst {
			dst[i] += v[i]
		}
		n++
	}
	if n > 0 {
		inv := 1.0 / float32(n)
		for i := range dst {
			dst[i] *= inv
		}
	}
	return n
}
