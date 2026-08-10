// SPDX-License-Identifier: Apache-2.0

package vector

import "github.com/rostamlabs/rostam/vector/analysis"

// Default BM25 saturation/length-normalization knobs (Robertson/Sparck-Jones).
const (
	defaultBM25K1 float32 = 1.2
	defaultBM25B  float32 = 0.75
)

// bm25Index is a full-text inverted index that stores RAW term frequencies and
// scores BM25 at query time from LIVE corpus statistics. It is structurally a
// sibling of sparseIndex (postings[term] -> []posting{slot, tf}) but maintains
// the extra stats BM25 needs: the live document count, the total token count
// (for avgdl), the live per-term document frequency, and each slot's length.
//
// It is allocated ONLY when a collection enables full text; a collection without
// it keeps every bm25 field nil and is byte/behavior-identical to before.
//
// Not safe for concurrent use; the embedding hnsw guards it with the SAME
// RWMutex it uses for sparseIdx (write lock for add/remove/rebuild, read lock
// for the scan).
type bm25Index struct {
	// postings[term] holds every (slot, tf) that references term. Append-only on
	// add; deletions are exact (remove) or lazy (the scan skips inadmissible
	// slots) and reclaimed by rebuild.
	postings map[uint32][]posting

	// n is the number of LIVE documents carrying text (the BM25 corpus size N).
	n int
	// tokenTotal is the sum of every live document's length (Σ tf over all live
	// postings). avgdl = tokenTotal / n.
	tokenTotal uint64
	// df is the LIVE document frequency per term, maintained explicitly so a
	// reused/tombstoned slot's stale postings never skew IDF (len(postings[term])
	// over-counts until reclaim).
	df map[uint32]int
	// docLen is each live slot's token length (Σ tf for that slot), keyed by slot
	// so the scan's length-normalization is O(1).
	docLen map[uint32]uint32

	// k1 is the term-frequency saturation parameter; b is the length-norm weight.
	k1 float32
	b  float32
}

// newBM25Index constructs an empty index with the given knobs (0 → defaults).
func newBM25Index(k1, b float32) *bm25Index {
	if k1 <= 0 {
		k1 = defaultBM25K1
	}
	// b == 0 resolves to the default (the FullTextConfig contract: 0 → 0.75). A
	// caller wanting no length normalization sets b to a tiny epsilon. Negative is
	// always the default too.
	if b <= 0 {
		b = defaultBM25B
	}
	return &bm25Index{
		postings: make(map[uint32][]posting),
		df:       make(map[uint32]int),
		docLen:   make(map[uint32]uint32),
		k1:       k1,
		b:        b,
	}
}

// add records slot's analyzed term counts (term id -> tf). It appends one
// posting per term, bumps the corpus stats (n, tokenTotal, df), and records the
// slot's length. An empty/nil counts map (a slot with no text) is a no-op, so
// such a slot never enters the corpus and never skews avgdl/IDF. The caller must
// hold the write lock.
func (bi *bm25Index) add(slot uint32, counts map[uint32]uint32) {
	if len(counts) == 0 {
		return
	}
	var length uint64
	for term, tf := range counts {
		bi.postings[term] = append(bi.postings[term], posting{slot: slot, value: float32(tf)})
		bi.df[term]++
		length += uint64(tf)
	}
	bi.n++
	bi.tokenTotal += length
	bi.docLen[slot] = uint32(length)
}

// remove is the exact inverse of add: it drops slot's postings for each term in
// counts and reverses every stat. Used when a slot is hard-removed for reuse
// (the Upsert replace path). An empty/nil counts map is a no-op. The caller must
// hold the write lock and pass the SAME counts that were add()ed for the slot.
func (bi *bm25Index) remove(slot uint32, counts map[uint32]uint32) {
	if len(counts) == 0 {
		return
	}
	var length uint64
	for term, tf := range counts {
		lst := bi.postings[term]
		w := 0
		for _, p := range lst {
			if p.slot != slot {
				lst[w] = p
				w++
			}
		}
		if w == 0 {
			delete(bi.postings, term)
		} else {
			bi.postings[term] = lst[:w]
		}
		if bi.df[term] <= 1 {
			delete(bi.df, term)
		} else {
			bi.df[term]--
		}
		length += uint64(tf)
	}
	bi.n--
	if bi.n < 0 {
		bi.n = 0
	}
	if bi.tokenTotal >= length {
		bi.tokenTotal -= length
	} else {
		bi.tokenTotal = 0
	}
	delete(bi.docLen, slot)
}

// rebuild clears the index and re-derives every statistic from each live (non-
// tombstoned) slot's $content metadata, re-analyzed with az. Called on Reclaim
// (stale postings would otherwise reference reused slots) and on every load path
// (snapshot restore / instant-restart / WAL replay) where the index is never
// serialized — it is rebuilt from the persisted $content. The caller must hold
// the write lock. az must be non-nil whenever the index exists.
func (bi *bm25Index) rebuild(a *arena, tombstoned map[uint32]bool, az analysis.Analyzer) {
	bi.postings = make(map[uint32][]posting)
	bi.df = make(map[uint32]int)
	bi.docLen = make(map[uint32]uint32)
	bi.n = 0
	bi.tokenTotal = 0
	ca, _ := az.(analysis.CountingAnalyzer)
	for _, slot := range a.idMap {
		if tombstoned[slot] {
			continue
		}
		content := contentOf(a.Metadata(slot))
		if content == "" {
			continue
		}
		if ca != nil {
			bi.add(slot, ca.AnalyzeCounts(content))
		}
	}
}

// corpusStats returns this index's CORPUS-WIDE BM25 statistics for the given query
// terms — the live document count n (the IDF corpus size N), the total live token
// count tokenTotal (avgdl = tokenTotal/n), and the per-term live document frequency
// df[t] = bi.df[t] for each t in terms. It is the phase-0 reader of the two-phase
// (dfs_query_then_fetch) fan-out: each shard returns these, the coordinator sums
// them into the global stats, and phase 1 re-scores with the injected globals.
//
// NO filter is applied — df/n are whole-corpus, exactly as the single-node IDF uses
// them (the filter only gates scoring via admit, never the IDF). Every query term
// gets an entry (a term absent from the corpus is reported as df 0, NOT omitted) so
// the returned map is self-describing and the coordinator's per-term sum is uniform
// across shards. The caller holds the read lock.
func (bi *bm25Index) corpusStats(terms []uint32) (n int, tokenTotal uint64, df map[uint32]int) {
	df = make(map[uint32]int, len(terms))
	for _, t := range terms {
		df[t] = bi.df[t] // absent ⇒ 0; included explicitly
	}
	return bi.n, bi.tokenTotal, df
}

// avgdl returns the average document length (tokenTotal / n), guarding the
// empty-corpus case (n == 0 → 0). It is the denominator term BM25 normalizes
// each document's length against.
func (bi *bm25Index) avgdl() float32 {
	if bi.n == 0 {
		return 0
	}
	return float32(bi.tokenTotal) / float32(bi.n)
}

// contentOf extracts the reserved $content string from a slot's metadata, or ""
// when absent/non-string. Mirrors the read in docForLocked.
func contentOf(meta Metadata) string {
	if meta == nil {
		return ""
	}
	if cv, ok := meta[contentField]; ok && cv.Kind == ValueString {
		return cv.Str
	}
	return ""
}
