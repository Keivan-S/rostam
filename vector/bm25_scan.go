// SPDX-License-Identifier: Apache-2.0

package vector

import "math"

// bm25SearchTopK scores every live document that shares a query term against the
// BM25 ranking function and returns the top-k slots (descending score). It forks
// sparseIndex.searchTopK: instead of a sparse dot product it accumulates, per
// query term t and per posting (slot, tf), the BM25 contribution
//
//	idf(t) · tf·(k1+1) / (tf + k1·(1 - b + b·docLen[slot]/avgdl))
//
// where idf(t) = ln(1 + (n - df(t) + 0.5)/(df(t) + 0.5)) is the
// Robertson/Sparck-Jones IDF with the +1 floor (always non-negative). IDF is
// computed ONCE per query term. queryTerms is the SORTED-UNIQUE term-id set of
// the query (Analyzer.Analyze), so each term's IDF is applied exactly once
// regardless of how often it appeared in the query — standard BM25.
//
// admit gates which slots may contribute (tombstoned/expired/filtered slots are
// skipped, identical to the sparse lane). n is the arena slot capacity (sizes
// the pooled accumulator). The caller holds the read lock.
func (bi *bm25Index) bm25SearchTopK(s *layerScratch, n int, queryTerms []uint32, k int, admit func(slot uint32) bool) []slotScore {
	// LOCAL path: the IDF corpus size, df, and avgdl all come from THIS index's
	// own live stats (bi.n / bi.df / bi.avgdl()) — the single-node scorer. Behavior
	// is byte-identical to the pre-refactor scan; only the IDF inputs are now
	// injected through the shared body.
	return bi.bm25Score(s, n, queryTerms, k, admit, bi.n, func(term uint32) int { return bi.df[term] }, bi.avgdl())
}

// bm25SearchTopKGlobal is the GLOBAL-stats entry point for the two-phase
// (dfs_query_then_fetch) fan-out: it scores THIS shard's LOCAL postings (local tf
// / local docLen) but injects the coordinator's GLOBAL corpus statistics for the
// IDF — gN (the summed live document count across all shards), gDF (the summed
// per-query-term document frequency), and gAvgdl (the global average document
// length, gTokenTotal/gN). Every shard scoring with the SAME global IDF makes the
// per-shard scores globally comparable, reproducing the single-node score exactly.
//
// A query term absent from gDF resolves to df 0 ⇒ idf = ln(1+(gN+0.5)/0.5) — the
// standard rare-term IDF; it is harmless because such a term has no local postings
// to contribute, but it is computed identically to the local path's lookup miss.
// Guards gN==0 / gAvgdl==0 like the local path (returns nil — no scorable corpus).
func (bi *bm25Index) bm25SearchTopKGlobal(s *layerScratch, n int, queryTerms []uint32, k int, admit func(slot uint32) bool, gN int, gAvgdl float32, gDF map[uint32]int) []slotScore {
	return bi.bm25Score(s, n, queryTerms, k, admit, gN, func(term uint32) int { return gDF[term] }, gAvgdl)
}

// bm25Score is the shared scan body for the local and global entry points. It
// always iterates THIS index's LOCAL postings (bi.postings[term]) and reads LOCAL
// per-document tf (p.value) and docLen (bi.docLen[p.slot]); ONLY the IDF inputs —
// the corpus size idfN, the per-term df via dfFor, and avgdl — are injected, so
// the local entry passes bi.n / bi.df / bi.avgdl() and the global entry passes the
// coordinator's summed stats. n is the arena slot capacity (sizes the pooled
// accumulator). The caller holds the read lock.
func (bi *bm25Index) bm25Score(s *layerScratch, n int, queryTerms []uint32, k int, admit func(slot uint32) bool, idfN int, dfFor func(term uint32) int, avgdl float32) []slotScore {
	if k <= 0 || len(queryTerms) == 0 || idfN == 0 {
		return nil
	}
	if avgdl == 0 {
		return nil
	}
	acc := &s.sparseAcc
	acc.prepare(n)
	k1 := bi.k1
	b := bi.b
	for _, term := range queryTerms {
		lst := bi.postings[term]
		if len(lst) == 0 {
			continue
		}
		df := float64(dfFor(term))
		// idf = ln(1 + (N - df + 0.5)/(df + 0.5)); the +1 keeps it >= 0 even when a
		// term appears in more than half the corpus (BM25+ / Lucene convention). The
		// ratio is formed in float64 (n/df promoted before the divide) so it is not
		// computed in float32 then widened into the log.
		idf := float32(math.Log(1 + (float64(idfN)-df+0.5)/(df+0.5)))
		for _, p := range lst {
			if admit != nil && !admit(p.slot) {
				continue
			}
			tf := p.value
			// denom >= 1 always (tf >= 1, k1 > 0, the length term >= 0), so no
			// divide-by-zero guard is needed here.
			denom := tf + k1*(1-b+b*float32(bi.docLen[p.slot])/avgdl)
			acc.add(p.slot, idf*(tf*(k1+1))/denom)
		}
	}
	return acc.topK(k)
}
