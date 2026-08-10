// SPDX-License-Identifier: Apache-2.0

package rostam

import (
	"sort"

	"github.com/rostamlabs/rostam/cluster"
	"github.com/rostamlabs/rostam/ops"
	"github.com/rostamlabs/rostam/vector"
)

// textLanes carries one partition's unfused dense + BM25-text candidate lanes
// back to the coordinator so they can be unioned across partitions and fused once.
type textLanes struct {
	dense []vector.Result
	text  []vector.Result
}

// textDocsFanOut scatters vector_search_text to all partitions and merges the
// enriched documents by DESCENDING BM25 score (higher = more relevant) into the
// global top-k. IDs are disjoint across partitions, so the global top-k is a
// subset of the union of per-partition top-k. ⚠ SHARDED-IDF CAVEAT: each
// document carries its OWN partition-LOCAL BM25 score (per-shard n/df/avgdl), so
// the cross-partition ordering is APPROXIMATE (query_then_fetch), NOT a global
// corpus ranking — see the design doc. Honors ReadConsistency / OnPartitionUnavailable.
func (e *embedded) textDocsFanOut(collection string, P int, gen uint32, query string, k int, filter VectorFilter, rc, opa uint8, bound uint64, globalIDF bool) ([]VectorDocument, cluster.FanResult, error) {
	// Phase 0 of the global-DF (dfs_query_then_fetch) fan-out: gather + sum each
	// partition's corpus stats into GLOBAL N/avgdl/df, then inject them into the
	// phase-1 scoring trailer so every shard scores with the SAME IDF (exact global
	// top-k). Only when explicitly requested AND partitioned; the default path runs
	// the single-phase per-shard-local fan-out UNCHANGED.
	var gPtr *vector.BM25GlobalStats
	var phase0 cluster.FanResult
	if globalIDF {
		g, fr0, err := e.bm25StatsFanOut(collection, P, gen, query, rc, opa, bound)
		if err != nil {
			return nil, fr0, err
		}
		phase0 = fr0
		// Σn==0 ⇒ no live BM25 docs across the fleet (or no full-text lane): nothing
		// to score globally; fall through to the single-phase path (gPtr stays nil) so
		// we never divide by zero building avgdl. The result is empty either way.
		if g.N > 0 {
			gPtr = &g
		}
	}

	a := cluster.FanArgs{
		Collection:    collection,
		P:             P,
		Generation:    gen,
		K:             k,
		Op:            "vector_search_text",
		Consistency:   cluster.Consistency(rc),
		OnUnavailable: cluster.OnUnavailable(opa),
		Encode: func(physCol string) []byte {
			return ops.EncodeSearchTextArgsGlobal(physCol, query, k, filter, rc, opa, bound, globalIDF, gPtr)
		},
	}
	decode := func(raw []byte) ([]vector.Document, error) {
		return ops.DecodeVectorDocs(raw)
	}
	merge := func(parts [][]vector.Document, k int) []vector.Document {
		var all []vector.Document
		for _, p := range parts {
			all = append(all, p...)
		}
		// BM25 is score-DESC (higher=better), unlike the dense docsFanOut which
		// sorts by ascending Distance. Secondary ascending ID for cross-partition
		// determinism (equal scores admit any valid top-k order).
		sort.SliceStable(all, func(i, j int) bool {
			if all[i].Score != all[j].Score {
				return all[i].Score > all[j].Score
			}
			return all[i].ID < all[j].ID
		})
		if k >= 0 && len(all) > k {
			all = all[:k]
		}
		return all
	}
	docs, fr, err := cluster.FanOut(a, e.node.CallPhysical, decode, merge)
	if err != nil {
		return docs, fr, err
	}
	return docs, mergeFanResults(phase0, fr), nil
}

// mergeFanResults unions two phases' degraded/missing partition sets (phase 0
// stats gather + phase 1 scoring) into one FanResult so the coordinator's returned
// FanMeta reflects EITHER phase skipping a partition. Under OnUnavailable==Fail the
// fan-out already errored, so this only runs in Partial mode; the missing set is
// the sorted, de-duplicated union of both phases.
func mergeFanResults(a, b cluster.FanResult) cluster.FanResult {
	out := cluster.FanResult{Degraded: a.Degraded || b.Degraded}
	seen := make(map[int]struct{}, len(a.Missing)+len(b.Missing))
	for _, p := range a.Missing {
		if _, ok := seen[p]; !ok {
			seen[p] = struct{}{}
			out.Missing = append(out.Missing, p)
		}
	}
	for _, p := range b.Missing {
		if _, ok := seen[p]; !ok {
			seen[p] = struct{}{}
			out.Missing = append(out.Missing, p)
		}
	}
	sort.Ints(out.Missing)
	return out
}

// bm25StatsFanOut is phase 0 of the global-DF (dfs) text fan-out: it scatters
// vector_bm25_stats to every partition (same consistency/unavailable policy as the
// scoring phase) and SUMS each shard's (n, tokenTotal, df[termID]) into one global
// BM25GlobalStats. Term ids are stateless FNV-1a hashes identical on every node, so
// df is a plain per-term-id sum across the disjoint shards. avgdl is built from the
// summed tokenTotal/N by the caller (guarding N==0). A partition without a BM25
// lane returns zero/empty and contributes nothing.
func (e *embedded) bm25StatsFanOut(collection string, P int, gen uint32, query string, rc, opa uint8, bound uint64) (vector.BM25GlobalStats, cluster.FanResult, error) {
	a := cluster.FanArgs{
		Collection:    collection,
		P:             P,
		Generation:    gen,
		K:             0,
		Op:            "vector_bm25_stats",
		Consistency:   cluster.Consistency(rc),
		OnUnavailable: cluster.OnUnavailable(opa),
		Encode: func(physCol string) []byte {
			return ops.EncodeBM25StatsArgs(physCol, query, rc, opa, bound)
		},
	}
	decode := func(raw []byte) ([]partitionStats, error) {
		n, tokenTotal, df, err := ops.DecodeBM25StatsResult(raw)
		if err != nil {
			return nil, err
		}
		return []partitionStats{{n: n, tokenTotal: tokenTotal, df: df}}, nil
	}
	// FanOut's merge concatenates; the cross-partition SUM happens after (mirroring
	// hybridTextFanOut, which unions lanes then truncates/fuses post-FanOut).
	merge := func(parts [][]partitionStats, _ int) []partitionStats {
		var all []partitionStats
		for _, p := range parts {
			all = append(all, p...)
		}
		return all
	}
	parts, fr, err := cluster.FanOut(a, e.node.CallPhysical, decode, merge)
	if err != nil {
		return vector.BM25GlobalStats{}, fr, err
	}
	var sumN int
	var sumTok uint64
	df := map[uint32]int{}
	for _, ps := range parts {
		sumN += ps.n
		sumTok += ps.tokenTotal
		for term, d := range ps.df {
			df[term] += d
		}
	}
	var avgdl float32
	if sumN > 0 {
		avgdl = float32(sumTok) / float32(sumN)
	}
	return vector.BM25GlobalStats{N: sumN, Avgdl: avgdl, DF: df}, fr, nil
}

// partitionStats is one partition's phase-0 corpus stats decoded from a
// vector_bm25_stats reply, summed by bm25StatsFanOut into the global stats.
type partitionStats struct {
	n          int
	tokenTotal uint64
	df         map[uint32]int
}

// hybridTextFanOut runs vector_hybrid_text_lanes on every physical partition,
// unions the per-partition dense + text lanes, truncates the unioned lanes to the
// GLOBAL denseK/sparseK, then fuses ONCE — mirroring hybridFanOut exactly.
//
// ⚠ SHARDED-IDF CAVEAT (v1, decided + documented in the design doc): the BM25
// text lane's IDF is computed PER-SHARD-LOCAL — each partition scores the query
// against its OWN corpus stats (n/df/avgdl), NOT a global corpus. So partitioned
// BM25 scores are APPROXIMATE vs a single-node corpus — EXACTLY Elasticsearch's
// default query_then_fetch behavior. Consequences for correctness/tests:
//   - The FUSION step itself is exact (truncate-before-fuse, P>1 == P1 for the
//     fusion math given the same per-lane candidates), so a SINGLE-shard placement
//     (one shard / RF≥N, where local corpus == global corpus) is bit-identical to
//     the single-node oracle.
//   - A MULTI-shard placement gives a SANE ranking (the right docs come back,
//     ordering is reasonable) but NOT bit-equal scores — the IDF differs per shard.
//
// A global-DF two-phase (dfs_query_then_fetch: gather df/n across shards, then
// score) is a documented follow-up — NOT built here.
func (e *embedded) hybridTextFanOut(collection string, P int, gen uint32, dense []float32, query string, k int, opts VectorHybridOpts) ([]VectorResult, cluster.FanResult, error) {
	hopts := toVectorHybridOpts(opts)
	denseK := hopts.DenseK
	if denseK <= 0 {
		denseK = k
		if denseK < 50 {
			denseK = 50
		}
	}
	textK := hopts.SparseK
	if textK <= 0 {
		textK = k
		if textK < 50 {
			textK = 50
		}
	}

	// Phase 0 of the global-DF (dfs) hybrid-text fan-out: gather + sum corpus stats,
	// inject into the phase-1 text-lane scoring trailer. The DENSE lane is unaffected
	// (global df is a BM25-text-only concern). Default path (GlobalIDF=false) runs the
	// single-phase per-shard-local fan-out UNCHANGED.
	var gPtr *vector.BM25GlobalStats
	var phase0 cluster.FanResult
	if opts.GlobalIDF {
		g, fr0, err := e.bm25StatsFanOut(collection, P, gen, query, opts.ReadConsistency, opts.OnPartitionUnavailable, opts.MaxStaleness)
		if err != nil {
			return nil, fr0, err
		}
		phase0 = fr0
		if g.N > 0 {
			gPtr = &g
		}
	}

	a := cluster.FanArgs{
		Collection:    collection,
		P:             P,
		Generation:    gen,
		K:             k,
		Op:            "vector_hybrid_text_lanes",
		Consistency:   cluster.Consistency(opts.ReadConsistency),
		OnUnavailable: cluster.OnUnavailable(opts.OnPartitionUnavailable),
		Encode: func(physCol string) []byte {
			// rc/opa trailer carries Linearizable to each partition leader so the
			// readIndex barrier runs there. The vector_hybrid_text_lanes handler
			// decodes with DecodeHybridTextArgsGlobal (which ignores the rc trailer for
			// routing but reads any phase-1 global-stats block), so this is wire-
			// compatible while ops.ReadConsistencyOf reads the rc byte.
			return ops.EncodeHybridTextArgsGlobal(physCol, dense, query, k, hopts, opts.ReadConsistency, opts.OnPartitionUnavailable, opts.MaxStaleness, opts.GlobalIDF, gPtr)
		},
	}
	decode := func(raw []byte) ([]textLanes, error) {
		d, t, err := ops.DecodeHybridLanesResult(raw)
		if err != nil {
			return nil, err
		}
		return []textLanes{{dense: d, text: t}}, nil
	}
	merge := func(parts [][]textLanes, _ int) []textLanes {
		var all []textLanes
		for _, p := range parts {
			all = append(all, p...)
		}
		return all
	}
	parts, fr, err := cluster.FanOut(a, e.node.CallPhysical, decode, merge)
	if err != nil {
		return nil, fr, err
	}
	// Combine phase-0 (stats) degraded/missing with phase-1 (scoring) so the returned
	// FanResult reflects EITHER phase skipping a partition.
	fr = mergeFanResults(phase0, fr)

	var allDense, allText []vector.Result
	for _, p := range parts {
		allDense = append(allDense, p.dense...)
		allText = append(allText, p.text...)
	}

	// EXACTNESS INVARIANT (for the fusion step): truncate to the global
	// denseK/textK BEFORE fusing. Dense lane: ascending Distance, secondary
	// ascending ID for fan-out-internal determinism. Text lane: descending Score
	// (BM25 is score-desc, higher=better), secondary ascending ID — the only
	// cross-partition-stable tiebreak (equal scores admit any valid top-k order).
	sort.SliceStable(allDense, func(i, j int) bool {
		if allDense[i].Distance != allDense[j].Distance {
			return allDense[i].Distance < allDense[j].Distance
		}
		return allDense[i].ID < allDense[j].ID
	})
	if len(allDense) > denseK {
		allDense = allDense[:denseK]
	}
	sort.SliceStable(allText, func(i, j int) bool {
		if allText[i].Score != allText[j].Score {
			return allText[i].Score > allText[j].Score
		}
		return allText[i].ID < allText[j].ID
	})
	if len(allText) > textK {
		allText = allText[:textK]
	}

	// Mirror HybridText's single-lane degradation, then fuse both lanes. The
	// single-node hasText gate is len(analyzed terms) > 0, NOT len(query) > 0: a
	// non-empty query that analyzes to zero terms (e.g. all-stopword "the") has no
	// text lane and single-node degrades to pure dense — gating the fan-out on raw
	// len(query) instead sent it through Fuse(dense, empty) (alpha-normalized dense
	// scores), breaking the P1==P-many invariant. Gate on the empty text lane
	// instead: each partition's buildTextLanes emits text results ONLY when
	// len(terms) > 0, so an empty unioned allText is the analyzed-term-empty signal
	// surviving the fan-out. (The one residual vs single-node: a query with valid
	// terms that match ZERO live docs ALSO yields empty allText and degrades to pure
	// dense here, whereas single-node runs Fuse(dense, empty) — same returned ID
	// order, only the per-result fusion Score differs, which is already documented
	// as per-shard approximate for partitioned text.) Empty dense degrades to pure
	// text symmetrically, matching single-node !hasDense.
	hasText := len(allText) > 0
	hasDense := len(dense) > 0
	var fused []vector.Result
	switch {
	case !hasText:
		fused = allDense
		if len(fused) > k {
			fused = fused[:k]
		}
	case !hasDense:
		fused = allText
		if len(fused) > k {
			fused = fused[:k]
		}
	default:
		fused = vector.Fuse(allDense, allText, hopts.Method, hopts.Alpha, hopts.RRFK, k)
	}

	out := make([]VectorResult, len(fused))
	for i, r := range fused {
		out[i] = VectorResult(r)
	}
	return out, fr, nil
}
