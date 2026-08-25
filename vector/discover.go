// SPDX-License-Identifier: Apache-2.0

package vector

import (
	"sort"
	"time"
)

// ContextPair (an id-based discovery constraint) and DiscoverPair (its resolved
// vector analogue) now live in the engine-free vtypes leaf package and are
// re-exported from vtypes_aliases.go.

// DiscoverOpts configures a discovery query. At least one context pair is
// required. Target is an optional anchor that seeds the candidate pool; when
// nil, the mean of the context positives is used.
type DiscoverOpts struct {
	Target  []float32     // optional anchor vector (nil = mean of context positives)
	Context []ContextPair // at least one pair required
	FetchK  int           // candidate pool size; 0 (or < k) = max(4*k, 50)
	Filter  Filter
}

// DiscoverVecsOpts configures a discovery query whose target + context pairs are
// already RESOLVED to vectors (the Query API leaf form). It mirrors DiscoverOpts
// but carries the resolved example vectors instead of ids — the coordinator
// resolves the ids cluster-wide and embeds the vectors, so each partition scores
// against them without re-resolving. Target is the optional anchor vector (nil =
// mean of the pair positives). Context must carry at least one pair.
type DiscoverVecsOpts struct {
	Target  []float32      // optional anchor vector (nil = mean of context positives)
	Context []DiscoverPair // at least one resolved pair required
	FetchK  int            // candidate pool size; 0 (or < k) = max(4*k, 50)
	Filter  Filter
}

// discoverFetchK is the shared candidate-pool sizing: at least k, defaulting to
// max(4*k, 50) when the caller's FetchK is below k. Used by every Discover path
// (hnsw/ivf, id-form and resolved-vecs form) so the seed pool is identical.
func discoverFetchK(fetchK, k int) int {
	if fetchK < k {
		if fetchK = 4 * k; fetchK < 50 {
			fetchK = 50
		}
	}
	return fetchK
}

// discoverSeed builds the candidate-pool SEED query into dst (length dim): the
// target when non-nil, else the mean of the pair positives, then a Cosine
// L2-normalize (idempotent with the search's own normalize). It is the shared
// seed logic for every Discover path so the pool query is identical whether the
// example vectors arrived as ids (resolved internally) or as embedded vectors
// (the Query API leaf). dst must have length dim.
func discoverSeed(dst []float32, target []float32, posVecs [][]float32, metric Metric) {
	if target != nil {
		copy(dst, target)
	} else {
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
	}
	if metric == Cosine {
		normalize(dst)
	}
}

// discoverScore is the per-candidate context-pair score: each pair contributes
// +1 when the candidate cv is closer to the pair's positive than its negative,
// and a negative penalty (simPos - simNeg, both = -dist) otherwise. It is the
// shared, exact scoring math for every Discover path (the equivalence oracle).
func discoverScore(cv []float32, pairs []DiscoverPair, dist distFunc) float32 {
	var score float32
	for _, p := range pairs {
		simPos := -dist(cv, p.Pos)
		simNeg := -dist(cv, p.Neg)
		if simPos >= simNeg {
			score++
		} else {
			score += simPos - simNeg // negative penalty
		}
	}
	return score
}

// sortDiscover sorts cands by the discover contract: context score DESCENDING,
// pool distance ASCENDING as the tiebreak. Shared by every Discover path.
func sortDiscover(cands []Result) {
	sort.SliceStable(cands, func(a, b int) bool {
		if cands[a].Score != cands[b].Score {
			return cands[a].Score > cands[b].Score
		}
		return cands[a].Distance < cands[b].Distance
	})
}

// Discover returns up to k results steered by context pairs: it seeds a
// candidate pool (from Target, or the mean of the context positives), then
// ranks candidates by a context score — each pair contributes +1 when the
// candidate is closer to the pair's positive than its negative, and a negative
// penalty otherwise — with the pool distance as a tiebreak. Result.Score
// carries the context score. Requires at least one context pair whose ids
// exist in the index.
func (h *hnsw) Discover(k int, opts DiscoverOpts) ([]Result, error) {
	start := time.Now()
	defer func() { h.searchLat.observe(time.Since(start)) }()

	if k <= 0 {
		return nil, nil
	}
	if len(opts.Context) == 0 {
		return nil, ErrNoContextPairs
	}
	if opts.Target != nil && len(opts.Target) != h.cfg.Dim {
		return nil, ErrDimMismatch
	}
	pred, err := CompileFilter(opts.Filter)
	if err != nil {
		return nil, err
	}
	fetchK := discoverFetchK(opts.FetchK, k)

	s := getLayerScratch()
	defer layerScratchPool.Put(s)

	h.mu.RLock()
	defer h.mu.RUnlock()
	h.searchOps.Add(1)

	// Resolve context pair vectors (skip pairs referencing missing ids).
	pairs := make([]DiscoverPair, 0, len(opts.Context))
	for _, c := range opts.Context {
		ps, okp := h.arena.Slot(c.Positive)
		ns, okn := h.arena.Slot(c.Negative)
		if !okp || !okn {
			continue
		}
		// exact, or reconstructed when PQDropVecs dropped the floats
		pairs = append(pairs, DiscoverPair{Pos: h.vecFor(ps), Neg: h.vecFor(ns)})
	}
	if len(pairs) == 0 {
		return nil, ErrIDNotFound
	}

	return h.discoverScoredLocked(s, k, fetchK, opts.Target, pairs, pred)
}

// DiscoverVecs is the RESOLVED-vectors discovery path (the Query API leaf form):
// the target + context-pair example VECTORS are supplied directly (the
// coordinator resolved the ids and embedded them), so no per-call id resolution
// happens. It runs the IDENTICAL seed + per-candidate context-pair scorer as
// Discover (the equivalence oracle), score-descending. Requires at least one
// resolved context pair. Target, when non-nil, must match the index Dim.
func (h *hnsw) DiscoverVecs(k int, opts DiscoverVecsOpts) ([]Result, error) {
	start := time.Now()
	defer func() { h.searchLat.observe(time.Since(start)) }()

	if k <= 0 {
		return nil, nil
	}
	if len(opts.Context) == 0 {
		return nil, ErrNoContextPairs
	}
	if opts.Target != nil && len(opts.Target) != h.cfg.Dim {
		return nil, ErrDimMismatch
	}
	pred, err := CompileFilter(opts.Filter)
	if err != nil {
		return nil, err
	}
	fetchK := discoverFetchK(opts.FetchK, k)

	s := getLayerScratch()
	defer layerScratchPool.Put(s)

	h.mu.RLock()
	defer h.mu.RUnlock()
	h.searchOps.Add(1)

	return h.discoverScoredLocked(s, k, fetchK, opts.Target, opts.Context, pred)
}

// RecommendVecs is the BEST_SCORE recommend path (the Query API leaf form): the
// positive/negative example VECTORS are supplied directly (the coordinator resolved
// the ids and embedded them), so no per-call id resolution happens. It seeds a
// candidate pool from the positives' centroid (recommendBestSeed), scores each
// candidate by the BEST_SCORE merge (bestScore — max-sim-to-positive vs
// max-sim-to-negative), and sorts score-descending (pool distance tiebreak). It is
// the recommend analogue of DiscoverVecs and the equivalence oracle for BEST_SCORE.
// Requires at least one positive vector. The example ids are NOT excluded here (the
// Query API coordinator pre-pass prunes them from the final result, mirroring the
// AVERAGE_VECTOR path).
func (h *hnsw) RecommendVecs(k int, opts RecommendVecsOpts) ([]Result, error) {
	start := time.Now()
	defer func() { h.searchLat.observe(time.Since(start)) }()

	if k <= 0 {
		return nil, nil
	}
	if len(opts.Positive) == 0 {
		return nil, ErrNoRecommendExamples
	}
	pred, err := CompileFilter(opts.Filter)
	if err != nil {
		return nil, err
	}
	fetchK := discoverFetchK(opts.FetchK, k)

	s := getLayerScratch()
	defer layerScratchPool.Put(s)

	h.mu.RLock()
	defer h.mu.RUnlock()
	h.searchOps.Add(1)

	return h.recommendBestLocked(s, k, fetchK, opts.Positive, opts.Negative, pred)
}

// recommendBestLocked is the shared HNSW BEST_SCORE core: seed the candidate pool
// from the positives' centroid, search it distance-ascending, score each candidate
// by the bestScore merge, and sort score-desc (pool distance tiebreak). Caller
// holds h.mu (read) and has resolved the example VECTORS. Mirrors
// discoverScoredLocked so the leaf == the engine on the same input.
func (h *hnsw) recommendBestLocked(s *layerScratch, k, fetchK int, posVecs, negVecs [][]float32, pred Predicate) ([]Result, error) {
	dim := h.cfg.Dim
	if cap(s.qbuf) < dim {
		s.qbuf = make([]float32, dim)
	} else {
		s.qbuf = s.qbuf[:dim]
	}
	recommendBestSeed(s.qbuf, posVecs, h.cfg.Metric)

	cands := h.searchDenseLocked(s, s.qbuf, fetchK, pred, nil)
	if len(cands) == 0 {
		return nil, nil
	}

	dist := h.metricDist()
	for i := range cands {
		slot, ok := h.arena.Slot(cands[i].ID)
		if !ok {
			continue
		}
		cv := h.vecFor(slot) // exact, or reconstructed when PQDropVecs dropped the floats
		cands[i].Score = bestScore(cv, posVecs, negVecs, h.cfg.Metric, dist)
	}
	sortRecommendBest(cands)
	if len(cands) > k {
		cands = cands[:k]
	}
	return cands, nil
}

// discoverScoredLocked is the shared HNSW discover core: seed the candidate pool
// (target or mean of pair positives), search it distance-ascending, score each
// candidate by the context pairs, and sort score-desc (pool distance as the
// tiebreak). Caller holds h.mu (read) and has resolved the pair VECTORS. This is
// the single body both Discover (id-form) and DiscoverVecs (resolved-vecs form)
// run, so the Query API leaf == the engine Discover on the same input.
func (h *hnsw) discoverScoredLocked(s *layerScratch, k, fetchK int, target []float32, pairs []DiscoverPair, pred Predicate) ([]Result, error) {
	dim := h.cfg.Dim
	if cap(s.qbuf) < dim {
		s.qbuf = make([]float32, dim)
	} else {
		s.qbuf = s.qbuf[:dim]
	}
	posVecs := make([][]float32, len(pairs))
	for i, p := range pairs {
		posVecs[i] = p.Pos
	}
	discoverSeed(s.qbuf, target, posVecs, h.cfg.Metric)

	cands := h.searchDenseLocked(s, s.qbuf, fetchK, pred, nil)
	if len(cands) == 0 {
		return nil, nil
	}

	dist := h.metricDist()
	for i := range cands {
		slot, ok := h.arena.Slot(cands[i].ID)
		if !ok {
			continue
		}
		cv := h.vecFor(slot) // exact, or reconstructed when PQDropVecs dropped the floats
		cands[i].Score = discoverScore(cv, pairs, dist)
	}
	sortDiscover(cands)
	if len(cands) > k {
		cands = cands[:k]
	}
	return cands, nil
}
