// SPDX-License-Identifier: Apache-2.0

package vector

import (
	"time"

	"github.com/rostamlabs/rostam/vector/analysis"
)

// analyzeCounts runs the resolved analyzer's tf-multiset pipeline over text. It
// is the single write/maintenance entry point for $content → BM25 postings, so
// add/remove/rebuild all tokenize identically (deterministic inverse). Returns
// nil when no analyzer is configured or the text yields no terms.
func (h *hnsw) analyzeCounts(text string) map[uint32]uint32 {
	ca, ok := h.az.(analysis.CountingAnalyzer)
	if !ok {
		return nil
	}
	return ca.AnalyzeCounts(text)
}

// maintainBM25OnPayloadChange re-maintains the BM25 postings + corpus stats for a
// LIVE slot whose metadata is being replaced from oldMeta to newMeta. The reserved
// $content is indexed out of the slot's metadata map (contentOf), so a payload op
// that changes or removes $content must remove the old postings and add the new
// ones or the index drifts (stale postings + skewed n/df/avgdl/docLen) until a
// Reclaim(). It mirrors the upsert-replace path's remove-old/add-new (hnsw.go).
//
// Only the $content delta matters: when old == new (the common SetPayload patch
// that does not touch $content) this is a no-op — no churn, no stat drift. The
// remove is gated on a non-empty old (an empty doc was never add()ed) and the add
// on a non-empty new (an empty doc must not enter the corpus). Caller holds h.mu
// write lock; oldMeta is the metadata BEFORE the in-place SetMetadata.
func (h *hnsw) maintainBM25OnPayloadChange(slot uint32, oldMeta, newMeta Metadata) {
	if h.bm25 == nil {
		return
	}
	oldContent := contentOf(oldMeta)
	newContent := contentOf(newMeta)
	if oldContent == newContent {
		return
	}
	if oldContent != "" {
		h.bm25.remove(slot, h.analyzeCounts(oldContent))
	}
	if newContent != "" {
		h.bm25.add(slot, h.analyzeCounts(newContent))
	}
}

// FullTextEnabled reports whether this index carries a BM25 full-text lane.
func (h *hnsw) FullTextEnabled() bool { return h.bm25 != nil }

// SearchText analyzes query and returns the top-k documents by BM25 relevance,
// descending Score (higher = more relevant). admit gates which slots may appear
// (tombstone/TTL/filter); nil admits every live slot. It returns nil (no error)
// when full text is disabled — callers that need a hard error use the Collection
// wrapper, which checks FullTextEnabled first. Results carry the BM25 Score; the
// dense Distance is 0 (there is no dense query).
func (h *hnsw) SearchText(query string, k int, admit func(slot uint32) bool) []Result {
	if h.bm25 == nil || k <= 0 {
		return nil
	}
	// Analyze under no lock (pure function of the query string), then scan under
	// the read lock — mirroring how the dense query buffer is prepared pre-lock.
	terms := h.az.Analyze(query)
	if len(terms) == 0 {
		return nil
	}
	s := getLayerScratch()
	defer layerScratchPool.Put(s)

	h.mu.RLock()
	defer h.mu.RUnlock()
	h.searchOps.Add(1)
	return h.textLaneLocked(s, terms, k, admit)
}

// textLaneLocked runs the BM25 scan for the already-analyzed query terms and maps
// the winning slots to Results (id + Score). Caller holds the read lock. The
// shared body of SearchText and the hybrid text lane.
func (h *hnsw) textLaneLocked(s *layerScratch, terms []uint32, k int, admit func(slot uint32) bool) []Result {
	var out []Result
	for _, ts := range h.bm25.bm25SearchTopK(s, h.arena.Capacity(), terms, k, admit) {
		// No allocation re-check: the posting lists only contain slots that were
		// given content at insert time, and admit already applied the liveness
		// gate. Emitting the id verbatim is what makes user id 0 returnable.
		out = append(out, Result{ID: h.slotID(ts.slot), Score: ts.score})
	}
	return out
}

// BM25GlobalStats is the coordinator-supplied GLOBAL corpus statistics injected
// into the two-phase (dfs_query_then_fetch) text scorer: N is the summed live
// document count across all partitions, Avgdl is the global average document length
// (summed tokenTotal / N), and DF is the summed per-query-term document frequency.
// Built by the ops layer from each shard's CorpusStats; consumed by the
// *Global scoring entry points so every shard scores with the SAME IDF.
type BM25GlobalStats struct {
	N     int
	Avgdl float32
	DF    map[uint32]int
}

// CorpusStats analyzes query (pre-lock, a pure function of the string) and returns
// this index's CORPUS-WIDE BM25 statistics for the query's terms under the read
// lock: the live document count n, the total live token count tokenTotal (for
// avgdl), and df[t] for each query term (df 0 included explicitly for an absent
// term). It is the phase-0 reader of the global-DF fan-out. Returns zero/nil
// cleanly when full text is disabled (h.bm25 == nil) or the query has no terms.
func (h *hnsw) CorpusStats(query string) (n int, tokenTotal uint64, df map[uint32]int) {
	if h.bm25 == nil {
		return 0, 0, nil
	}
	terms := h.az.Analyze(query)
	if len(terms) == 0 {
		return 0, 0, nil
	}
	h.mu.RLock()
	defer h.mu.RUnlock()
	return h.bm25.corpusStats(terms)
}

// textLaneGlobalLocked is textLaneLocked scored against the injected GLOBAL stats
// (bm25SearchTopKGlobal) instead of this shard's local IDF. Caller holds the read
// lock; the shared body of SearchTextGlobal and the global hybrid text lane.
func (h *hnsw) textLaneGlobalLocked(s *layerScratch, terms []uint32, k int, admit func(slot uint32) bool, g BM25GlobalStats) []Result {
	var out []Result
	for _, ts := range h.bm25.bm25SearchTopKGlobal(s, h.arena.Capacity(), terms, k, admit, g.N, g.Avgdl, g.DF) {
		// No allocation re-check: the posting lists only contain slots that were
		// given content at insert time, and admit already applied the liveness
		// gate. Emitting the id verbatim is what makes user id 0 returnable.
		out = append(out, Result{ID: h.slotID(ts.slot), Score: ts.score})
	}
	return out
}

// SearchTextGlobal is SearchText scored with coordinator-supplied GLOBAL corpus
// stats g (the two-phase text scorer): it scans THIS shard's local postings/tf/
// docLen but injects g.N/g.Avgdl/g.DF for the IDF, so every shard's scores are
// globally comparable. Identical analyze-once + empty-terms degradation + locking
// as SearchText; returns nil when full text is disabled or the query has no terms.
func (h *hnsw) SearchTextGlobal(query string, k int, admit func(slot uint32) bool, g BM25GlobalStats) []Result {
	if h.bm25 == nil || k <= 0 {
		return nil
	}
	terms := h.az.Analyze(query)
	if len(terms) == 0 {
		return nil
	}
	s := getLayerScratch()
	defer layerScratchPool.Put(s)

	h.mu.RLock()
	defer h.mu.RUnlock()
	h.searchOps.Add(1)
	return h.textLaneGlobalLocked(s, terms, k, admit, g)
}

// HybridTextLanes builds the dense and BM25-text candidate lanes WITHOUT fusing
// them, so a cross-partition coordinator can union the lanes and re-fuse globally
// (exact text fan-out). denseRes is ascending by Distance; textRes is descending
// by Score (a BM25 relevance score, NOT a distance). DenseK/SparseK defaults
// match HybridSearch (max(k,50)); opts.Filter gates BOTH lanes. Returns
// ErrFullTextDisabled when this collection has no BM25 lane.
func (h *hnsw) HybridTextLanes(dense []float32, query string, k int, opts HybridOpts) (denseRes, textRes []Result, err error) {
	start := time.Now()
	defer func() { h.searchLat.observe(time.Since(start)) }()
	denseRes, textRes, _, err = h.buildTextLanes(dense, query, k, opts)
	return denseRes, textRes, err
}

// buildTextLanes is the shared dense + BM25-text lane builder. It mirrors
// buildLanes: validate, size the candidate pools, prepare the (cosine-normalized)
// dense query pre-lock, then run both lanes under the read lock gated by the same
// admit rule (tombstone + TTL + filter).
func (h *hnsw) buildTextLanes(dense []float32, query string, k int, opts HybridOpts) (denseRes, textRes []Result, terms []uint32, err error) {
	if h.bm25 == nil {
		return nil, nil, nil, ErrFullTextDisabled
	}
	if k <= 0 {
		return nil, nil, nil, nil
	}
	if len(dense) > 0 && len(dense) != h.cfg.Dim {
		return nil, nil, nil, ErrDimMismatch
	}
	pred, err := CompileFilter(opts.Filter)
	if err != nil {
		return nil, nil, nil, err
	}

	denseK := opts.DenseK
	if denseK <= 0 {
		denseK = k
		if denseK < 50 {
			denseK = 50
		}
	}
	textK := opts.SparseK
	if textK <= 0 {
		textK = k
		if textK < 50 {
			textK = 50
		}
	}

	terms = h.az.Analyze(query)

	s := getLayerScratch()
	defer layerScratchPool.Put(s)

	var q []float32
	if len(dense) > 0 {
		q = dense
		if h.cfg.Metric == Cosine {
			s.qbuf = append(s.qbuf[:0], dense...)
			normalize(s.qbuf)
			q = s.qbuf
		}
	}

	h.mu.RLock()
	defer h.mu.RUnlock()
	h.searchOps.Add(1)

	if q != nil {
		denseRes = h.searchDenseLocked(s, q, denseK, pred, nil)
	}
	if len(terms) > 0 {
		now := uint64(h.now()) // one clock read for the whole text-lane admission scan
		admit := func(slot uint32) bool { return h.admits(slot, pred, now) }
		textRes = h.textLaneLocked(s, terms, textK, admit)
	}
	return denseRes, textRes, terms, nil
}

// HybridTextLanesGlobal is HybridTextLanes with the text lane scored against the
// coordinator-supplied GLOBAL stats g (the two-phase fan-out's phase 1). The dense
// lane is UNCHANGED (global df is a BM25-text-only concern); only the text lane
// routes through the global scorer. Returns ErrFullTextDisabled when this
// collection has no BM25 lane.
func (h *hnsw) HybridTextLanesGlobal(dense []float32, query string, k int, opts HybridOpts, g BM25GlobalStats) (denseRes, textRes []Result, terms []uint32, err error) {
	start := time.Now()
	defer func() { h.searchLat.observe(time.Since(start)) }()
	return h.buildTextLanesGlobal(dense, query, k, opts, g)
}

// buildTextLanesGlobal mirrors buildTextLanes but scores the text lane with the
// injected GLOBAL stats (textLaneGlobalLocked) instead of this shard's local IDF.
// The dense lane is byte-identical to buildTextLanes; the analyze-once + single-
// lane degradation + admit/locking logic is the same.
func (h *hnsw) buildTextLanesGlobal(dense []float32, query string, k int, opts HybridOpts, g BM25GlobalStats) (denseRes, textRes []Result, terms []uint32, err error) {
	if h.bm25 == nil {
		return nil, nil, nil, ErrFullTextDisabled
	}
	if k <= 0 {
		return nil, nil, nil, nil
	}
	if len(dense) > 0 && len(dense) != h.cfg.Dim {
		return nil, nil, nil, ErrDimMismatch
	}
	pred, err := CompileFilter(opts.Filter)
	if err != nil {
		return nil, nil, nil, err
	}

	denseK := opts.DenseK
	if denseK <= 0 {
		denseK = k
		if denseK < 50 {
			denseK = 50
		}
	}
	textK := opts.SparseK
	if textK <= 0 {
		textK = k
		if textK < 50 {
			textK = 50
		}
	}

	terms = h.az.Analyze(query)

	s := getLayerScratch()
	defer layerScratchPool.Put(s)

	var q []float32
	if len(dense) > 0 {
		q = dense
		if h.cfg.Metric == Cosine {
			s.qbuf = append(s.qbuf[:0], dense...)
			normalize(s.qbuf)
			q = s.qbuf
		}
	}

	h.mu.RLock()
	defer h.mu.RUnlock()
	h.searchOps.Add(1)

	if q != nil {
		denseRes = h.searchDenseLocked(s, q, denseK, pred, nil)
	}
	if len(terms) > 0 {
		now := uint64(h.now()) // one clock read for the whole text-lane admission scan
		admit := func(slot uint32) bool { return h.admits(slot, pred, now) }
		textRes = h.textLaneGlobalLocked(s, terms, textK, admit, g)
	}
	return denseRes, textRes, terms, nil
}

// HybridText fuses the dense lane with the BM25 text lane into the top-k. The
// text lane is a DESCENDING relevance SCORE (higher = better), NOT a distance, so
// the weighted/DBSF fusion must treat it as a score (FuseScoreLanes) — never
// inverting it the way the distance-oriented Fuse does for the dense lane. RRF is
// rank-only, so it fuses correctly either way. Single-lane queries (empty text or
// no dense vector) degrade to the present lane, mirroring HybridSearch.
func (h *hnsw) HybridText(dense []float32, query string, k int, opts HybridOpts) ([]Result, error) {
	start := time.Now()
	defer func() { h.searchLat.observe(time.Since(start)) }()

	denseRes, textRes, terms, err := h.buildTextLanes(dense, query, k, opts)
	if err != nil {
		return nil, err
	}

	hasDense := len(dense) > 0
	hasText := len(terms) > 0
	switch {
	case !hasText:
		if len(denseRes) > k {
			denseRes = denseRes[:k]
		}
		return denseRes, nil
	case !hasDense:
		if len(textRes) > k {
			textRes = textRes[:k]
		}
		return textRes, nil
	}

	// Both lanes present. The dense lane is a Distance (asc); the text lane is a
	// Score (desc). FuseScoreLanes' weighted/DBSF paths normalize a SCORE lane
	// without inverting it — but they treat the FIRST lane as a score too. So for
	// the score-shaped text lane we keep it as the SPARSE (second) argument of the
	// standard Fuse, whose sparse path is already score-oriented (normalizeSparse /
	// dbsfNormalizeSparse, higher = better) and whose dense path inverts the
	// distance. That is exactly the orientation we want: dense=distance (lane 1),
	// text=score (lane 2). RRF reads both in rank order regardless.
	alpha := opts.Alpha
	return Fuse(denseRes, textRes, opts.Method, alpha, opts.RRFK, k), nil
}
