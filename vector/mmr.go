// SPDX-License-Identifier: Apache-2.0

package vector

import "time"

// MMROpts configures Maximal Marginal Relevance re-ranking. The zero value is
// valid: Lambda defaults to 0.5 and FetchK to max(4*k, 50).
type MMROpts struct {
	// Lambda is the relevance/diversity tradeoff in (0,1]: 1 = pure relevance
	// (equivalent to plain Search), smaller values favor diversity. Values <= 0
	// are treated as the 0.5 default; values > 1 are clamped to 1.
	Lambda float64
	// FetchK is the candidate pool size to re-rank. 0 (or < k) = max(4*k, 50),
	// so there is a pool to diversify over.
	FetchK int
	// Filter is an optional metadata predicate applied during candidate search.
	Filter Filter
}

// SearchMMR returns up to k results re-ranked by Maximal Marginal Relevance: it
// over-collects FetchK candidates by relevance, then greedily selects results
// that balance similarity to the query against similarity to already-selected
// results (the Lambda knob). This reduces near-duplicate results — useful for
// diversifying retrieved context in RAG. Result.Distance carries the original
// metric distance to the query.
func (h *hnsw) SearchMMR(query []float32, k int, opts MMROpts) ([]Result, error) {
	start := time.Now()
	defer func() { h.searchLat.observe(time.Since(start)) }()

	if len(query) != h.cfg.Dim {
		return nil, ErrDimMismatch
	}
	if k <= 0 {
		return nil, nil
	}
	lambda := opts.Lambda
	switch {
	case lambda <= 0:
		lambda = 0.5
	case lambda > 1:
		lambda = 1
	}
	fetchK := opts.FetchK
	if fetchK < k {
		if fetchK = 4 * k; fetchK < 50 {
			fetchK = 50
		}
	}
	pred, err := CompileFilter(opts.Filter)
	if err != nil {
		return nil, err
	}

	s := getLayerScratch()
	defer layerScratchPool.Put(s)

	q := query
	if h.cfg.Metric == Cosine {
		s.qbuf = append(s.qbuf[:0], query...)
		normalize(s.qbuf)
		q = s.qbuf
	}

	h.mu.RLock()
	defer h.mu.RUnlock()
	h.searchOps.Add(1)

	cands := h.searchDenseLocked(s, q, fetchK, pred, nil)
	if len(cands) <= k {
		return cands, nil
	}
	return h.mmrSelect(cands, k, float32(lambda)), nil
}

// mmrSelect greedily picks k results from cands (sorted ascending by distance)
// to maximize lambda*relevance - (1-lambda)*redundancy, where similarity is the
// negated metric distance. Must be called with h.mu held for reading (it reads
// candidate vectors from the arena).
func (h *hnsw) mmrSelect(cands []Result, k int, lambda float32) []Result {
	dist := h.metricDist()
	n := len(cands)

	// Candidate vectors alias arena storage (or are reconstructed from codes when
	// PQDropVecs dropped the floats — vecFor); both are valid only under the held
	// RLock.
	vecs := make([][]float32, n)
	for i := range cands {
		if slot, ok := h.arena.Slot(cands[i].ID); ok {
			vecs[i] = h.vecFor(slot)
		}
	}

	selected := make([]Result, 0, k)
	used := make([]bool, n)

	// First pick is the most relevant candidate (cands is sorted by distance).
	selected = append(selected, cands[0])
	used[0] = true

	// maxSim[i] is the highest similarity of candidate i to any already-
	// selected result. It only grows as the selected set grows (the max of a
	// growing set), so each round folds in just the newest pick instead of
	// rescanning every selected vector — O(k*n) total instead of O(k^2*n) for
	// the same, numerically identical, values.
	maxSim := make([]float32, n)
	for i := 0; i < n; i++ {
		if vecs[i] == nil {
			continue
		}
		maxSim[i] = -dist(vecs[i], vecs[0])
	}

	for len(selected) < k {
		best := -1
		var bestScore float32
		for i := 0; i < n; i++ {
			if used[i] || vecs[i] == nil {
				continue
			}
			relevance := -cands[i].Distance
			score := lambda*relevance - (1-lambda)*maxSim[i]
			if best == -1 || score > bestScore {
				best, bestScore = i, score
			}
		}
		if best == -1 {
			break
		}
		selected = append(selected, cands[best])
		used[best] = true
		for i := 0; i < n; i++ {
			if used[i] || vecs[i] == nil {
				continue
			}
			if sim := -dist(vecs[i], vecs[best]); sim > maxSim[i] {
				maxSim[i] = sim
			}
		}
	}
	return selected
}
