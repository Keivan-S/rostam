// SPDX-License-Identifier: Apache-2.0

package vector

import (
	"errors"
	"sort"
)

// --- Unified Query API engine (prefetch + fusion/rerank, dense family) ---
//
// This is the proto-FREE Go-struct surface for the unified vector_query op. The
// ops layer converts a marshaled pb.QuerySpec into these structs (the engine
// never imports the proto package, matching how the rest of vector takes
// vector.Filter / vector.HybridOpts rather than proto types). A query carries a
// ROOT leaf plus N single-level PREFETCH leaves, combined at the root by either
// FUSION (RRF/Weighted/DBSF over the prefetch lanes) or RERANK (the root leaf
// re-scores the union of the prefetch candidates). v1 = the DENSE family,
// single-level (no recursion); leaf kinds are dense + sparse.

// The unified query data types (LeafKind, QueryMode, QueryLeaf, QuerySource,
// QuerySpec, QueryResult), the LeafSource constructor + QuerySource.IsLeaf method,
// the MaxPrefetchSources bound, and the ErrQuerySpecTooDeep /
// ErrTooManyPrefetchSources sentinels now live in the engine-free vtypes leaf
// package and are re-exported from vtypes_aliases.go.

// Query API validation errors (fail-loud at the engine edge).
var (
	// ErrQueryNoPrefetch is returned when a spec carries no prefetch leaves (a
	// query MUST prefetch at least one lane).
	ErrQueryNoPrefetch = errors.New("vector: query spec has no prefetch leaves")
	// ErrQueryBadMode is returned for an unknown QueryMode.
	ErrQueryBadMode = errors.New("vector: query spec has an unknown mode")
	// ErrQueryBadLeafKind is returned for an unknown LeafKind in any leaf.
	ErrQueryBadLeafKind = errors.New("vector: query leaf has an unknown kind")
	// ErrQueryRerankNoRoot is returned when a RERANK spec's root leaf carries
	// neither a dense nor a sparse query payload (RERANK needs a root to re-score
	// the candidate union).
	ErrQueryRerankNoRoot = errors.New("vector: rerank query spec has an empty root leaf")
	// ErrQueryDenseLeafHasSpace is returned when a leaf in a dense
	// (*Collection).Query carries a non-empty Space (a dense collection has no
	// named spaces — fail loud rather than silently ignoring the Space).
	ErrQueryDenseLeafHasSpace = errors.New("vector: dense query leaf carries a named space")
	// ErrQueryNamedLeafNoSpace is returned when a leaf in a named
	// (*NamedCollection).Query carries an empty Space (every named-query leaf MUST
	// target a configured named space — fail loud).
	ErrQueryNamedLeafNoSpace = errors.New("vector: named query leaf has no space")
	// ErrQueryRecommendHasSpace is returned when a recommend leaf carries a
	// non-empty Space: v1 recommend is DENSE-only (named/MV recommend is a
	// follow-up), so a Space-bearing recommend leaf is rejected fail-loud rather
	// than silently treated as a dense recommend.
	ErrQueryRecommendHasSpace = errors.New("vector: recommend query leaf carries a named space (v1 is dense-only)")
	// ErrQueryDiscoverHasSpace is returned when a discover leaf carries a non-empty
	// Space: v1 discover is DENSE-only (named/MV discover is a follow-up), so a
	// Space-bearing discover leaf is rejected fail-loud rather than silently treated
	// as a dense discover.
	ErrQueryDiscoverHasSpace = errors.New("vector: discover query leaf carries a named space (v1 is dense-only)")
	// ErrQueryDiscoverNoContext is returned when a discover leaf carries neither
	// resolved context pairs nor context-pair ids (discover requires at least one
	// context pair to score candidates).
	ErrQueryDiscoverNoContext = errors.New("vector: discover query leaf has no context pairs")
	// ErrQueryNestedNotSupported is returned when a query FAMILY that does not yet
	// support nested prefetch recursion (the NAMED and MV families in v1) encounters a
	// nested QuerySpec prefetch source. The DENSE (*Collection).Query supports nested
	// recursion; named/MV stay 1-level (flat leaf sources) until a follow-up
	// threads their per-leaf executors through the recursion. A flat leaf-source spec
	// never hits this — it is byte/behaviour-identical to today.
	ErrQueryNestedNotSupported = errors.New("vector: nested query prefetch is not supported for this query family")
	// ErrQueryGroupNotDense is returned when a GROUPED query (spec.GroupBy != "") is
	// run against a non-dense family (named / MV): grouping is DENSE-only in v1 (it
	// post-processes the dense ordered pool, reading each id's Metadata locally via the
	// dense payload accessor). A named/MV grouped query is rejected fail-loud rather
	// than silently returning an ungrouped result; named/MV grouping is a follow-up.
	ErrQueryGroupNotDense = errors.New("vector: grouped query (group_by) is dense-only (named/MV grouping is a follow-up)")
)

// MaxQueryDepthExec is the DEFENSIVE execution-side nesting bound, mirroring the
// ops-codec decode bound (ops.maxQueryDepth): even a malformed IN-MEMORY spec (built
// directly, bypassing the decode) cannot drive runQuerySpec into unbounded recursion.
// The root spec runs at depth 0; each nested sub-spec recurses at depth+1; a depth
// over this bound is rejected fail-loud (ErrQuerySpecTooDeep). Kept in lockstep with
// the decode bound so a spec that decodes cleanly also executes. Exported so the
// coordinator's mergeTreeFusionNode can mirror the same guard for in-memory specs.
const MaxQueryDepthExec = 4

// MaxLanePool clamps the per-lane candidate-pool size (leafLanePool / SourceLanePool):
// a lane requesting LaneK above this is capped to this ceiling — exactly equivalent to
// the client having passed LaneK=MaxLanePool (the lane still returns its score-ordered
// top-N, just bounded), so it is deterministic and P-invariant (every partition clamps
// to the same ceiling). 10K candidates per lane is far above any realistic top-k.
// Exported so tests (incl. the coordinator fan-out P-invariance test) can reference the
// same sentinel without hard-coding the value. Structural safety bound; promote to
// config in a follow-up.
const MaxLanePool = 10000

// validateLeafKind rejects an unknown leaf kind (fail-loud, never silently
// treats a corrupt kind as dense).
func validateLeafKind(l QueryLeaf) error {
	switch l.Kind {
	case LeafDense, LeafSparse, LeafMVMaxSim, LeafDiscover:
		// LeafDiscover is a real execLeaf (a custom per-candidate scorer): unlike
		// LeafRecommend it is executed directly by runQuerySpec (execLeaf →
		// DiscoverVecs), so it is a valid leaf kind at the runQuerySpec edge. The
		// coordinator resolve pre-pass has already filled its DiscoverContext vectors.
		return nil
	case LeafRecommend:
		// AVERAGE_VECTOR recommend: valid at the spec edge but MUST be rewritten to a
		// dense leaf by the coordinator pre-pass before runQuerySpec executes it; if one
		// reaches runQuerySpec un-rewritten it is a coordinator bug, not a client error,
		// so it is rejected here as a bad kind (the shared core has no derive context to
		// execute an AVERAGE recommend leaf directly).
		//
		// BEST_SCORE recommend: a REAL execLeaf (a custom per-candidate scorer, like
		// LeafDiscover) — the resolve pre-pass embeds RecPosVecs/RecNegVecs and the leaf
		// stays a LeafRecommend the execLeaf runs (RecommendVecs), so it IS a valid leaf
		// kind at the runQuerySpec edge.
		if l.Strategy == RecommendBestScore {
			return nil
		}
		return ErrQueryBadLeafKind
	default:
		return ErrQueryBadLeafKind
	}
}

// Query executes a unified query spec on this single collection (one shard /
// one partition): it runs each prefetch leaf, then either FUSES the resulting
// lanes (ModeFusion) or RERANKS the union of the prefetch candidates by the
// root leaf (ModeRerank). It returns a mode-tagged QueryResult so the op handler
// (and, later, the cross-partition coordinator) can encode/decode by mode.
//
// FUSION: each prefetch leaf is executed as a lane (dense → distance-ascending,
// sparse → score-descending). The lanes are returned unfused (for the fan-out
// coordinator) AND locally fused via Fuse into Fused (the P=1 direct top-k). v1
// fuses the dense+sparse 2-lane case via vector.Fuse; for >2 prefetch lanes the
// lanes are folded pairwise (a left-to-right Fuse over the lane list) so the
// shape is natural for the v2 N-lane coordinator while remaining well-defined for
// v1. (>2 lanes is documented as v2 for the cross-partition orchestrator; the
// single-node fold here is exact for any N.)
//
// RERANK: every prefetch leaf is executed, their candidate ids are UNIONED, and
// the ROOT leaf re-scores that candidate set restricted to those ids (dense via
// filterFirstByID — exact brute-force over the set; sparse via an exhaustive
// sparse-lane scan filtered to the set), returning the reranked top-k by the
// root's orientation (dense distance-ascending, sparse score-descending).
func (c *Collection) Query(spec QuerySpec) (QueryResult, error) {
	// GROUPED query (Qdrant-parity group_by): when GroupBy is set the query is a
	// final-stage POST-PROCESS over the flat dense ordered pool — run the spec with a
	// WIDE k (the groups-op FetchK pool), then collapse the ordered Fused pool by the
	// GroupBy field via GroupDocuments (REUSED verbatim). DENSE-only (this *Collection
	// is the dense family; named/MV have their own Query methods that fail-loud). An
	// EMPTY GroupBy falls straight through to queryFlat — byte/behaviour-identical to a
	// non-grouped query (the #1 invariant).
	if spec.GroupBy != "" {
		return c.queryGrouped(spec)
	}
	return c.queryFlat(spec)
}

// queryFlat is the UNCHANGED flat dense Query pipeline (the pre-grouping body). A
// non-grouped query routes here directly from Query, so its behaviour — validation
// order, discover/recommend pre-passes, runQuerySpec, exclusion — is byte/behaviour-
// identical to before grouping existed.
func (c *Collection) queryFlat(spec QuerySpec) (QueryResult, error) {
	// A dense collection has NO named spaces: any Space-bearing leaf is a request
	// error (fail loud rather than silently ignoring the Space and querying the
	// single dense index). Checked before the shared core so the dense path keeps its
	// exact validation surface.
	// A dense query supports NESTED prefetch (a sub-spec source recurses); validate the
	// no-Space rule across the WHOLE tree (a Space-bearing dense leaf at any depth is a
	// request error). A flat leaf-source spec walks only the top level — its exact
	// pre-recursion validation surface.
	if err := denseTreeNoSpace(&spec); err != nil {
		return QueryResult{}, err
	}
	// DISCOVER pre-pass: resolve each discover leaf's target + context-pair IDS to
	// their stored vectors via the LOCAL index and EMBED them into the leaf's vector
	// fields, BEFORE the shared runQuerySpec runs. Unlike recommend, a discover leaf
	// is NOT rewritten to a dense leaf — it stays a LeafDiscover the execLeaf runs
	// (DiscoverVecs over the embedded vectors). v1 is dense-only: a Space-bearing
	// discover leaf is rejected fail-loud.
	if err := c.resolveDiscoverLeaves(&spec); err != nil {
		return QueryResult{}, err
	}
	// RECOMMEND pre-pass: derive the query vector for each recommend leaf (root +
	// prefetch) from the LOCAL index and rewrite the leaf to a dense leaf, BEFORE
	// the shared runQuerySpec runs. After this the spec is a plain dense spec, so
	// the entire existing dense pipeline (lanes / fusion / rerank) runs unchanged.
	// The example ids are collected for post-filter exclusion from the final result.
	exclude, err := c.resolveRecommendLeaves(&spec)
	if err != nil {
		return QueryResult{}, err
	}
	if len(exclude) == 0 {
		// No recommend leaves: the dense path is byte-identical to before.
		return runQuerySpec(spec, c.execLeaf, c.rerankByRoot)
	}
	// Over-fetch by the number of example ids so the final top-k still holds k
	// results after the examples are pruned (mirrors Recommend's k+len(exclude)
	// over-fetch). runQuerySpec truncates the fused/reranked result to spec.K
	// internally, so the over-fetch must be applied to spec.K for the run; the
	// final top-k is re-truncated to the requested k after the prune.
	wantK := spec.K
	if wantK <= 0 {
		wantK = 10 // runQuerySpec's default
	}
	spec.K = wantK + len(exclude)
	qr, err := runQuerySpec(spec, c.execLeaf, c.rerankByRoot)
	if err != nil {
		return QueryResult{}, err
	}
	excludeExamplesFromResult(&qr, exclude, wantK)
	return qr, nil
}

// QueryTreeLanes is the UNFUSED tree-lanes Query variant for a spec that contains a
// nested MULTI-lane FUSION node (SpecHasNestedFusion == true). Instead of fusing each
// FUSION node into a single lane (Query → runQuerySpecAt), it runs the SAME dense
// pipeline (the discover/recommend resolve pre-passes, the per-leaf executor, the
// nested RERANK/single-lane-FUSION fold) but returns the node-expanded UNFUSED lanes in
// the deterministic pre-order traversal collectTreeLanesAt walks — so the coordinator
// (and the single-shard path) folds EVERY FUSION node over the cross-partition GLOBAL
// union ⇒ P>1==P1 EXACT at every level. It is DENSE-only and NON-grouped (the same
// surface as queryFlat); the recommend example ids are pruned from each lane (mirroring
// queryFlat's excludeExamplesFromResult lane prune) so a single-shard recommend tree
// matches the multi-partition path. Query (the flat/grouped single-node fold) is left
// UNCHANGED — only the nested-multi-lane-FUSION wire path routes here.
func (c *Collection) QueryTreeLanes(spec QuerySpec) ([][]Result, error) {
	if err := denseTreeNoSpace(&spec); err != nil {
		return nil, err
	}
	if err := c.resolveDiscoverLeaves(&spec); err != nil {
		return nil, err
	}
	exclude, err := c.resolveRecommendLeaves(&spec)
	if err != nil {
		return nil, err
	}
	if len(exclude) > 0 {
		// Over-fetch so wantK results survive after the coordinator-side post-fold
		// recommend exclusion (the example ids are NOT pruned from the lanes here —
		// they must remain so DBSF/Weighted normalization runs over the FULL candidate
		// set, byte-identical to the P>1 path where the coordinator folds the unioned
		// lanes and THEN prunes the merged result). The widened spec.K propagates into
		// collectTreeLanesAt's execLeaf calls so each lane fetches wantK+len(exclude)
		// candidates; the coordinator (P>1) or the single-shard fold (P=1) prunes post-
		// fold via ExcludeExamplesFromResults.
		wantK := spec.K
		if wantK <= 0 {
			wantK = 10
		}
		nExclude := len(exclude)
		spec.K = wantK + nExclude
		// Widen EVERY nested sub-spec's K by nExclude too: collectTreeLanesAt uses each
		// sub-spec's own K as the lane pool for its leaf children, so a recommend leaf
		// buried at any depth (now rewritten to dense/best-score) would otherwise fetch
		// sub.K candidates — no room for the post-fold prune. Mirrors named QueryTreeLanes.
		widenNestedSpecsK(&spec, nExclude)
	}
	return collectTreeLanesAt(spec, c.execLeaf, c.rerankByRoot, 0)
}

// queryGrouped runs a GROUPED dense query (spec.GroupBy != ""): a final-stage
// POST-PROCESS over the flat dense ordered pool. It widens the spec's k to the
// groups-op FetchK pool (resolveGroupFetchK — the SAME formula SearchGroups uses,
// so the candidate pool is byte-identical to the standalone oracle), runs the
// UNCHANGED flat dense pipeline (queryFlat with the group fields cleared), then
// collapses the ordered Fused pool by the GroupBy field via GroupDocuments (REUSED
// verbatim — no new grouping algorithm). The pool is read into Documents through the
// SAME local payload accessor (idx.fetchDocs → docForLocked) the GroupCandidates
// path uses, so each id's Metadata[GroupBy] is the identical scalar. FUSION and
// RERANK are both supported: each produces an ordered Fused pool, and grouping is a
// deterministic function of that ordered pool + the per-id group key.
//
// NOTE: this is the engine-level / single-node oracle entry point (called by Query and
// used as the 3-way oracle in tests). The WIRED single-node path (op handler → network)
// goes through handleVectorQuery → QueryGroupedFanOut → groupMergedQueryParts; the op
// handler never calls queryGrouped directly. Do not conflate the two.
func (c *Collection) queryGrouped(spec QuerySpec) (QueryResult, error) {
	// k is the number of GROUPS (mirrors SearchGroups' k); groupSize defaults to 1.
	groupsK := spec.K
	if groupsK <= 0 {
		groupsK = 10 // mirror queryFlat / runQuerySpec's default top-k.
	}
	groupSize := spec.GroupSize
	if groupSize <= 0 {
		groupSize = 1
	}
	// Widen the flat run to the groups-op candidate pool so the ordered Fused pool is
	// large enough to form the requested top-K groups × group_size. Identical formula
	// to SearchGroups ⇒ identical pool ⇒ the oracle equivalence holds.
	fetchK := resolveGroupFetchK(groupsK, groupSize, 0)

	// Run the UNCHANGED flat dense pipeline over the wide pool. Clear the group fields
	// on the inner run so queryFlat takes the plain flat path (no recursion) and
	// validates the dense no-Space rule exactly as a non-grouped query does.
	inner := spec
	inner.GroupBy = ""
	inner.GroupSize = 0
	inner.K = fetchK
	qr, err := c.queryFlat(inner)
	if err != nil {
		return QueryResult{}, err
	}

	// Read each pooled id's stored Metadata LOCALLY via the same accessor the groups
	// op uses (docForLocked through fetchDocs), preserving the Fused order, then fold
	// the ordered pool ONCE via GroupDocuments.
	docs := c.idx.fetchDocs(qr.Fused)
	groups := GroupDocuments(docs, GroupOpts{GroupBy: spec.GroupBy, GroupSize: groupSize}, groupsK)
	return QueryResult{Mode: qr.Mode, Fused: qr.Fused, Lanes: qr.Lanes, Groups: groups}, nil
}

// QueryGroupedFanOut runs the PER-PARTITION side of a coordinator-grouped query
// (spec.GroupBy != ""): it runs the UNCHANGED flat dense pipeline over the SAME wide
// candidate pool the single-node grouped query (queryGrouped) uses, but returns the
// flat QueryResult (UNFUSED Lanes for FUSION, the partition-local reranked Fused for
// RERANK) WITHOUT grouping — plus a per-id group-key map for EVERY id appearing in the
// result. The coordinator unions/fuses/reranks the flat result with its NORMAL merge
// (unchanged, P>1==P1) to get the global ordered pool, then maps id→key and folds via
// GroupDocuments ONCE. Grouping never happens on the partition; it is a deterministic
// post-process on the exact global ordered pool ⇒ P>1==P1 exact for FUSION and RERANK.
//
// The group-key per id is read LOCALLY via the SAME accessor (idx.fetchDocs →
// docForLocked) queryGrouped / GroupCandidates use, so a key is the identical scalar
// the single-node path reads. An id whose Metadata lacks the GroupBy field (or whose
// value is non-scalar) is simply absent from the map — GroupDocuments skips it the
// same way (a hit with no group value is dropped), so the bucketing matches exactly.
func (c *Collection) QueryGroupedFanOut(spec QuerySpec) (QueryResult, map[uint64]Value, error) {
	if spec.GroupBy == "" {
		return QueryResult{}, nil, ErrEmptyGroupBy
	}
	groupsK := spec.K
	if groupsK <= 0 {
		groupsK = 10
	}
	groupSize := spec.GroupSize
	if groupSize <= 0 {
		groupSize = 1
	}
	fetchK := resolveGroupFetchK(groupsK, groupSize, 0)

	// Run the UNCHANGED flat dense pipeline over the wide pool (group fields cleared so
	// queryFlat takes the plain flat path + validates the dense no-Space rule exactly as
	// a non-grouped query). FUSION → Lanes (unfused), RERANK → Fused (partition-local).
	inner := spec
	inner.GroupBy = ""
	inner.GroupSize = 0
	inner.K = fetchK
	qr, err := c.queryFlat(inner)
	if err != nil {
		return QueryResult{}, nil, err
	}

	// Collect the per-id group-key for EVERY id in the result (the union of all lanes
	// for FUSION, the fused list for RERANK), reading each id's Metadata[GroupBy]
	// locally via the SAME accessor the single-node grouped path uses.
	ids := resultIDSet(qr)
	keys := make(map[uint64]Value, len(ids))
	docs := c.idx.fetchDocs(idsToResults(ids))
	for _, d := range docs {
		if d.Metadata == nil {
			continue
		}
		if v, ok := d.Metadata[spec.GroupBy]; ok {
			keys[d.ID] = v
		}
	}
	return qr, keys, nil
}

// resultIDSet collects the distinct ids carried by a flat QueryResult: the Fused
// list for a RERANK result, or the union of all prefetch Lanes for a FUSION result
// (a candidate may appear in several lanes; it is counted once). Order is irrelevant
// — the caller only needs each id's group-key.
func resultIDSet(qr QueryResult) map[uint64]struct{} {
	ids := make(map[uint64]struct{})
	for _, r := range qr.Fused {
		ids[r.ID] = struct{}{}
	}
	for _, lane := range qr.Lanes {
		for _, r := range lane {
			ids[r.ID] = struct{}{}
		}
	}
	return ids
}

// idsToResults wraps a set of ids as bare Results (id only) so they can be passed to
// idx.fetchDocs, which only reads the id to look up each record's stored Metadata.
func idsToResults(ids map[uint64]struct{}) []Result {
	out := make([]Result, 0, len(ids))
	for id := range ids {
		out = append(out, Result{ID: id})
	}
	return out
}

// denseTreeNoSpace enforces the dense-collection no-Space rule across the WHOLE
// query tree (the root leaf + every prefetch source, recursing into nested sub-spec
// sources): a dense collection has no named spaces, so a Space-bearing leaf at ANY
// nesting depth is a request error (ErrQueryDenseLeafHasSpace), fail-loud. A flat
// leaf-source spec walks only the top level — byte-identical to the pre-recursion
// per-leaf check.
func denseTreeNoSpace(spec *QuerySpec) error {
	if spec.Root.Space != "" {
		return ErrQueryDenseLeafHasSpace
	}
	for i := range spec.Prefetch {
		src := spec.Prefetch[i]
		if src.Leaf != nil {
			if src.Leaf.Space != "" {
				return ErrQueryDenseLeafHasSpace
			}
			continue
		}
		if src.Spec != nil {
			if err := denseTreeNoSpace(src.Spec); err != nil {
				return err
			}
		}
	}
	return nil
}

// resolveRecommendLeaves is the single-node RECOMMEND coordinator pre-pass: for
// every recommend leaf in the spec (root + prefetch) it derives the query vector
// from the local index (deriveRecommendVector via the index's vecsForIDs — the
// PQ-drop reconstruct path is handled there) and REWRITES the leaf IN PLACE into a
// dense leaf {Kind:LeafDense, Dense:derived, K, Filter, ScoreDesc:false}, clearing
// the recommend ids. It returns the UNION of all example ids (positive+negative)
// across the rewritten leaves so the caller can exclude them from the result. v1
// is dense-only: a recommend leaf carrying a Space is rejected fail-loud
// (ErrQueryRecommendHasSpace). Non-recommend leaves are left untouched (the dense
// path stays byte-identical). The over-fetch needed so k results remain after the
// exclusion is handled at the lane level (leafLanePool's default pool >= k+examples
// for small k) and, defensively, the rerank/fusion pools already over-fetch.
func (c *Collection) resolveRecommendLeaves(spec *QuerySpec) (map[uint64]struct{}, error) {
	var exclude map[uint64]struct{}
	addExclude := func(positive, negative []uint64) {
		if exclude == nil {
			exclude = make(map[uint64]struct{}, len(positive)+len(negative))
		}
		for _, id := range positive {
			exclude[id] = struct{}{}
		}
		for _, id := range negative {
			exclude[id] = struct{}{}
		}
	}
	rewrite := func(l *QueryLeaf) error {
		if l.Kind != LeafRecommend {
			return nil
		}
		if l.Space != "" {
			return ErrQueryRecommendHasSpace
		}
		if l.Strategy == RecommendBestScore {
			// BEST_SCORE: resolve the example ids → vectors from the LOCAL index and EMBED
			// them into RecPosVecs/RecNegVecs (like discover), then keep the leaf a
			// LeafRecommend the best-score execLeaf runs (NOT a dense rewrite). At least one
			// positive must resolve (the seed pool + max-positive similarity need it);
			// missing negatives are skipped (they only steer). The example ids are excluded
			// from the final result, mirroring AVERAGE_VECTOR.
			//
			// Already-embedded (the cluster coordinator's RewriteRecommendLeavesWith ran
			// first and cleared the ids): leave as-is so the partition handler does NOT
			// re-resolve against its LOCAL index (where cross-partition ids are absent).
			if len(l.RecPosVecs) > 0 && len(l.Positive) == 0 {
				// L2 + negatives is ill-defined (see ErrRecommendBestScoreL2Negatives): the
				// coordinator should already have rejected, but re-check here so a directly-
				// embedded leaf (or a future caller) fails loud the same as the resolve path.
				if c.cfg.Metric == L2 && len(l.RecNegVecs) > 0 {
					return ErrRecommendBestScoreL2Negatives
				}
				l.ScoreDesc = true
				return nil
			}
			if len(l.Positive) == 0 {
				return ErrNoRecommendExamples
			}
			// Reject L2 + negatives BEFORE scoring: the BEST_SCORE -max_neg sign-flip
			// inverts the ranking for the non-positive L2 similarity (fail loud, do not
			// return an inverted ranking). L2 with no negatives stays valid.
			if c.cfg.Metric == L2 && len(l.Negative) > 0 {
				return ErrRecommendBestScoreL2Negatives
			}
			resolved := c.idx.vecsForIDs(append(append([]uint64(nil), l.Positive...), l.Negative...))
			posVecs := make([][]float32, 0, len(l.Positive))
			for _, id := range l.Positive {
				if v, ok := resolved[id]; ok {
					posVecs = append(posVecs, v)
				}
			}
			if len(posVecs) == 0 {
				return ErrIDNotFound
			}
			var negVecs [][]float32
			for _, id := range l.Negative {
				if v, ok := resolved[id]; ok {
					negVecs = append(negVecs, v)
				}
			}
			addExclude(l.Positive, l.Negative)
			// Embed the resolved vectors; clear the K/LaneK so the lane uses the engine's
			// generous default pool (max(runK,50)) and over-fetch survives the exclusion.
			l.RecPosVecs = posVecs
			l.RecNegVecs = negVecs
			l.K = 0
			l.LaneK = 0
			l.ScoreDesc = true
			return nil
		}
		derived, err := deriveRecommendVector(c.cfg.Dim, c.cfg.Metric, c.idx.vecsForIDs, l.Positive, l.Negative)
		if err != nil {
			return err
		}
		addExclude(l.Positive, l.Negative)
		// Rewrite to a plain dense leaf; clear the recommend payload so the leaf is a
		// well-formed dense leaf for the unchanged downstream pipeline. K/LaneK are
		// cleared so the lane uses the engine's generous default pool (max(runK,50)):
		// a recommend leaf's K is the FINAL top-k (spec.K), never a per-lane cap, so
		// keeping it here would starve the over-fetch needed to drop the examples.
		*l = QueryLeaf{
			Kind:      LeafDense,
			Dense:     derived,
			Filter:    l.Filter,
			ScoreDesc: false,
		}
		return nil
	}
	// Tree-walk: rewrite the root + every prefetch source's recommend leaves, RECURSING
	// into nested sub-spec sources so a recommend leaf at ANY depth is derived+rewritten
	// before runQuerySpec executes the tree. accumulate merges each rewritten leaf's
	// example ids into the shared exclude set.
	accumulate := func(sub map[uint64]struct{}) {
		if len(sub) == 0 {
			return
		}
		if exclude == nil {
			exclude = make(map[uint64]struct{}, len(sub))
		}
		for id := range sub {
			exclude[id] = struct{}{}
		}
	}
	if err := rewrite(&spec.Root); err != nil {
		return nil, err
	}
	for i := range spec.Prefetch {
		src := spec.Prefetch[i]
		if src.Leaf != nil {
			if err := rewrite(src.Leaf); err != nil {
				return nil, err
			}
			continue
		}
		if src.Spec != nil {
			subExclude, err := c.resolveRecommendLeaves(src.Spec)
			if err != nil {
				return nil, err
			}
			accumulate(subExclude)
		}
	}
	return exclude, nil
}

// resolveDiscoverLeaves is the single-node DISCOVER coordinator pre-pass: for
// every discover leaf in the spec (root + prefetch) that carries UNRESOLVED ids
// (DiscoverTargetID / DiscoverContextIDs), it resolves those ids to their stored
// vectors via the LOCAL index (vecsForIDs — the PQ-drop reconstruct path is
// handled there) and EMBEDS the resolved vectors into the leaf's DiscoverTarget /
// DiscoverContext fields IN PLACE. Unlike resolveRecommendLeaves it does NOT
// rewrite the leaf to a dense leaf: discover is a custom per-candidate scorer that
// stays a LeafDiscover for the execLeaf (DiscoverVecs). v1 is dense-only: a
// Space-bearing discover leaf is rejected fail-loud (ErrQueryDiscoverHasSpace). It
// fails loud when the target id (id-form) cannot resolve (ErrIDNotFound) or when
// NO context pair resolves (ErrQueryDiscoverNoContext — discover needs at least one
// pair). A leaf already carrying DiscoverContext vectors and no ids is left as-is
// (the resolution is skipped — the client supplied raw vectors).
func (c *Collection) resolveDiscoverLeaves(spec *QuerySpec) error {
	resolve := func(l *QueryLeaf) error {
		if l.Kind != LeafDiscover {
			return nil
		}
		if l.Space != "" {
			return ErrQueryDiscoverHasSpace
		}
		// Discover needs at least one context pair (resolved vectors OR ids) — a leaf
		// with neither cannot score candidates.
		if len(l.DiscoverContext) == 0 && len(l.DiscoverContextIDs) == 0 {
			return ErrQueryDiscoverNoContext
		}
		// Already-resolved (raw context vectors, no ids to resolve): leave as-is. A
		// raw-vector target with no target id is also already embedded.
		if len(l.DiscoverContextIDs) == 0 && len(l.DiscoverTargetID) == 0 {
			return nil
		}
		// Gather every id (the optional target + each context pair's pos/neg) for a
		// single batched resolve.
		ids := make([]uint64, 0, len(l.DiscoverTargetID)+2*len(l.DiscoverContextIDs))
		ids = append(ids, l.DiscoverTargetID...)
		for _, cp := range l.DiscoverContextIDs {
			ids = append(ids, cp.Positive, cp.Negative)
		}
		resolved := c.idx.vecsForIDs(ids)
		// Target: when an id was given it MUST resolve (fail loud); empty target id
		// ⇒ no anchor (seed from the context positives), DiscoverTarget stays nil.
		if len(l.DiscoverTargetID) > 0 {
			tv, ok := resolved[l.DiscoverTargetID[0]]
			if !ok {
				return ErrIDNotFound
			}
			l.DiscoverTarget = tv
		}
		// Context pairs: when context IDS were supplied, resolve them — keep only pairs
		// whose BOTH ids resolve (mirror discover.go, which skips pairs referencing
		// missing ids); at least one must survive. When NO context ids were supplied
		// (target-id-only with raw context vectors), keep the embedded DiscoverContext.
		if len(l.DiscoverContextIDs) > 0 {
			pairs := make([]DiscoverPair, 0, len(l.DiscoverContextIDs))
			for _, cp := range l.DiscoverContextIDs {
				pv, okp := resolved[cp.Positive]
				nv, okn := resolved[cp.Negative]
				if !okp || !okn {
					continue
				}
				pairs = append(pairs, DiscoverPair{Pos: pv, Neg: nv})
			}
			if len(pairs) == 0 {
				return ErrIDNotFound
			}
			l.DiscoverContext = pairs
		}
		return nil
	}
	// Tree-walk: resolve the root + every prefetch source's discover leaves, RECURSING
	// into nested sub-spec sources so a discover leaf at ANY depth is resolved+embedded
	// before runQuerySpec executes the tree.
	if err := resolve(&spec.Root); err != nil {
		return err
	}
	for i := range spec.Prefetch {
		src := spec.Prefetch[i]
		if src.Leaf != nil {
			if err := resolve(src.Leaf); err != nil {
				return err
			}
			continue
		}
		if src.Spec != nil {
			if err := c.resolveDiscoverLeaves(src.Spec); err != nil {
				return err
			}
		}
	}
	return nil
}

// SpecHasRecommendLeaves reports whether the spec carries any LeafRecommend node
// (root or prefetch). The cluster coordinator uses it to decide whether the
// recommend pre-pass (a cluster-wide example-id resolution + derive) is needed
// before fanning out: a spec with no recommend leaves is fanned out verbatim.
func SpecHasRecommendLeaves(spec QuerySpec) bool {
	if spec.Root.Kind == LeafRecommend {
		return true
	}
	for i := range spec.Prefetch {
		src := spec.Prefetch[i]
		if src.Leaf != nil && src.Leaf.Kind == LeafRecommend {
			return true
		}
		// Recurse into nested sub-spec sources: a recommend leaf at ANY depth must be
		// detected so the coordinator pre-pass resolves it before fan-out.
		if src.Spec != nil && SpecHasRecommendLeaves(*src.Spec) {
			return true
		}
	}
	return false
}

// RecommendExampleIDs returns the UNION of every recommend leaf's example ids
// (positive ∪ negative, root + prefetch) in the spec. The cluster coordinator
// batch-gets exactly these ids (they may span partitions), builds the resolved
// id→vector map, and feeds that to RewriteRecommendLeavesWith.
func RecommendExampleIDs(spec QuerySpec) []uint64 {
	var ids []uint64
	collect := func(l QueryLeaf) {
		if l.Kind != LeafRecommend {
			return
		}
		ids = append(ids, l.Positive...)
		ids = append(ids, l.Negative...)
	}
	collect(spec.Root)
	for i := range spec.Prefetch {
		src := spec.Prefetch[i]
		if src.Leaf != nil {
			collect(*src.Leaf)
			continue
		}
		// Recurse into nested sub-spec sources so a nested recommend leaf's example ids
		// are batch-got cluster-wide before fan-out.
		if src.Spec != nil {
			ids = append(ids, RecommendExampleIDs(*src.Spec)...)
		}
	}
	return ids
}

// RewriteRecommendLeavesWith is the CLUSTER coordinator analogue of the single-
// node resolveRecommendLeaves: it rewrites every recommend leaf (root + prefetch)
// IN PLACE into a dense leaf using `derive`, a caller-supplied function that maps
// (positive, negative) → query vector. The coordinator passes a derive backed by
// the cluster-wide resolved id→vector map (DeriveRecommendVector over the batch-get
// result), so the derive runs ONCE on the coordinator and the rewritten dense spec
// is partition-invariant. It returns the union of example ids for post-merge
// exclusion. v1 is dense-only: a recommend leaf carrying a Space is rejected
// fail-loud (ErrQueryRecommendHasSpace). Mirrors resolveRecommendLeaves' rewrite
// (same dense-leaf shape, same K/LaneK clearing) so P>1 matches P==1 exactly —
// including the BEST_SCORE L2-with-negatives reject (ErrRecommendBestScoreL2Negatives),
// for which the metric is passed in (the coordinator knows the collection metric).
func RewriteRecommendLeavesWith(spec *QuerySpec, metric Metric, resolved map[uint64][]float32, derive func(positive, negative []uint64) ([]float32, error)) (map[uint64]struct{}, error) {
	var exclude map[uint64]struct{}
	addExclude := func(positive, negative []uint64) {
		if exclude == nil {
			exclude = make(map[uint64]struct{}, len(positive)+len(negative))
		}
		for _, id := range positive {
			exclude[id] = struct{}{}
		}
		for _, id := range negative {
			exclude[id] = struct{}{}
		}
	}
	rewrite := func(l *QueryLeaf) error {
		if l.Kind != LeafRecommend {
			return nil
		}
		if l.Space != "" {
			return ErrQueryRecommendHasSpace
		}
		if l.Strategy == RecommendBestScore {
			// BEST_SCORE: embed the cluster-wide resolved example VECTORS into RecPosVecs/
			// RecNegVecs (like RewriteDiscoverLeavesWith) and keep the leaf a LeafRecommend
			// the best-score execLeaf runs on each partition — NOT a dense rewrite. Mirrors
			// the single-node resolveRecommendLeaves BEST_SCORE branch so P>1 == P1. At
			// least one positive must resolve cluster-wide; missing negatives are skipped.
			if len(l.Positive) == 0 {
				return ErrNoRecommendExamples
			}
			// Reject L2 + negatives BEFORE embedding/scoring, identically to the single-
			// node resolveRecommendLeaves branch so P>1 fails loud the same as P==1: the
			// BEST_SCORE -max_neg sign-flip inverts the ranking for the non-positive L2
			// similarity (see ErrRecommendBestScoreL2Negatives). L2 with no negatives is fine.
			if metric == L2 && len(l.Negative) > 0 {
				return ErrRecommendBestScoreL2Negatives
			}
			posVecs := make([][]float32, 0, len(l.Positive))
			for _, id := range l.Positive {
				if v, ok := resolved[id]; ok {
					posVecs = append(posVecs, v)
				}
			}
			if len(posVecs) == 0 {
				return ErrIDNotFound
			}
			var negVecs [][]float32
			for _, id := range l.Negative {
				if v, ok := resolved[id]; ok {
					negVecs = append(negVecs, v)
				}
			}
			addExclude(l.Positive, l.Negative)
			l.RecPosVecs = posVecs
			l.RecNegVecs = negVecs
			// Clear the ids so the partition handler's resolveRecommendLeaves treats this
			// leaf as already-resolved (its BEST_SCORE branch resolves from local ids only
			// when RecPosVecs is empty — see the embed-once early return there).
			l.Positive = nil
			l.Negative = nil
			l.K = 0
			l.LaneK = 0
			l.ScoreDesc = true
			return nil
		}
		derived, err := derive(l.Positive, l.Negative)
		if err != nil {
			return err
		}
		addExclude(l.Positive, l.Negative)
		*l = QueryLeaf{
			Kind:      LeafDense,
			Dense:     derived,
			Filter:    l.Filter,
			ScoreDesc: false,
		}
		return nil
	}
	// Tree-walk: rewrite the root + every prefetch source's recommend leaves, RECURSING
	// into nested sub-spec sources (mirroring the single-node resolveRecommendLeaves
	// tree-walk) so a nested recommend leaf is derived+rewritten partition-invariantly
	// on the coordinator before fan-out.
	accumulate := func(sub map[uint64]struct{}) {
		if len(sub) == 0 {
			return
		}
		if exclude == nil {
			exclude = make(map[uint64]struct{}, len(sub))
		}
		for id := range sub {
			exclude[id] = struct{}{}
		}
	}
	if err := rewrite(&spec.Root); err != nil {
		return nil, err
	}
	for i := range spec.Prefetch {
		src := spec.Prefetch[i]
		if src.Leaf != nil {
			if err := rewrite(src.Leaf); err != nil {
				return nil, err
			}
			continue
		}
		if src.Spec != nil {
			subExclude, err := RewriteRecommendLeavesWith(src.Spec, metric, resolved, derive)
			if err != nil {
				return nil, err
			}
			accumulate(subExclude)
		}
	}
	return exclude, nil
}

// SpecHasDiscoverLeaves reports whether the spec carries any LeafDiscover node
// (root or prefetch) that still names UNRESOLVED ids (target id OR context-pair
// ids). The cluster coordinator uses it to decide whether the discover pre-pass
// (a cluster-wide id resolution + embed) is needed before fanning out: a spec with
// no id-bearing discover leaf is fanned out verbatim. A discover leaf already
// carrying raw vectors (no ids) is partition-invariant as-is and needs no pre-pass.
func SpecHasDiscoverLeaves(spec QuerySpec) bool {
	has := func(l QueryLeaf) bool {
		return l.Kind == LeafDiscover && (len(l.DiscoverTargetID) > 0 || len(l.DiscoverContextIDs) > 0)
	}
	if has(spec.Root) {
		return true
	}
	for i := range spec.Prefetch {
		src := spec.Prefetch[i]
		if src.Leaf != nil && has(*src.Leaf) {
			return true
		}
		// Recurse into nested sub-spec sources: an id-bearing discover leaf at ANY depth
		// must be detected so the coordinator pre-pass resolves it before fan-out.
		if src.Spec != nil && SpecHasDiscoverLeaves(*src.Spec) {
			return true
		}
	}
	return false
}

// SpecHasNestedFusion reports whether the spec tree contains ANY nested MULTI-LANE
// FUSION sub-spec (a prefetch QuerySource whose Spec is a FUSION node with ≥2
// prefetch lanes), at ANY depth. It is the PURE function of spec SHAPE that selects
// the cross-partition result codec: a spec with a nested multi-lane FUSION node must
// ship the per-partition UNFUSED tree-lanes (so the coordinator folds each FUSION
// node over the GLOBAL union → P>1==P1 exact), while every other spec (flat FUSION/
// RERANK, nested-RERANK-only, nested SINGLE-lane FUSION) is partition-invariant under
// the existing flat lane codec and encodes BYTE-IDENTICALLY. It is evaluated on BOTH
// sides — the coordinator (from the spec it sends) and the partition (from the spec it
// receives decode the SAME spec) — so both pick the same codec WITHOUT a wire flag.
//
// The walk recurses into EVERY nested spec source (not only multi-lane FUSION ones): a
// deeper multi-lane FUSION buried under a single-lane FUSION or a RERANK prefetch must
// still flip the codec, because the per-partition unfused traversal will expand it.
func SpecHasNestedFusion(spec QuerySpec) bool {
	for i := range spec.Prefetch {
		sub := spec.Prefetch[i].Spec
		if sub == nil {
			continue
		}
		if sub.Mode == ModeFusion && len(sub.Prefetch) >= 2 {
			return true
		}
		if SpecHasNestedFusion(*sub) {
			return true
		}
	}
	return false
}

// DiscoverLeafIDs returns the UNION of every discover leaf's example ids (the
// optional target id ∪ each context pair's positive/negative, root + prefetch) in
// the spec. The cluster coordinator batch-gets exactly these ids (they may span
// partitions), builds the resolved id→vector map, and feeds that to
// RewriteDiscoverLeavesWith.
func DiscoverLeafIDs(spec QuerySpec) []uint64 {
	var ids []uint64
	collect := func(l QueryLeaf) {
		if l.Kind != LeafDiscover {
			return
		}
		ids = append(ids, l.DiscoverTargetID...)
		for _, cp := range l.DiscoverContextIDs {
			ids = append(ids, cp.Positive, cp.Negative)
		}
	}
	collect(spec.Root)
	for i := range spec.Prefetch {
		src := spec.Prefetch[i]
		if src.Leaf != nil {
			collect(*src.Leaf)
			continue
		}
		// Recurse into nested sub-spec sources so a nested discover leaf's ids are
		// batch-got cluster-wide before fan-out.
		if src.Spec != nil {
			ids = append(ids, DiscoverLeafIDs(*src.Spec)...)
		}
	}
	return ids
}

// RewriteDiscoverLeavesWith is the CLUSTER coordinator analogue of the single-node
// resolveDiscoverLeaves: it EMBEDS the resolved target/context VECTORS into every
// id-bearing discover leaf (root + prefetch) IN PLACE, using `resolved`, a
// cluster-wide id→vector map (the VectorGetBatch result), then CLEARS the leaf's id
// fields so each partition handler sees an already-resolved LeafDiscover (no
// per-partition re-resolve — the partition's resolveDiscoverLeaves leaves it as-is).
// Unlike RewriteRecommendLeavesWith it does NOT rewrite the leaf to a dense leaf:
// discover stays a LeafDiscover the execLeaf runs (DiscoverVecs over the embedded
// vectors). It mirrors resolveDiscoverLeaves' fail-loud contract exactly so P>1
// matches P==1: a Space-bearing discover leaf is rejected (ErrQueryDiscoverHasSpace);
// a leaf with neither resolved vectors nor ids is rejected (ErrQueryDiscoverNoContext);
// a target id that does not resolve cluster-wide is ErrIDNotFound; a context-pair
// keeps only pairs whose BOTH ids resolve (skipping the missing, mirroring discover.go)
// and at least one must survive (else ErrIDNotFound). A leaf already carrying raw
// vectors and no ids is left untouched.
func RewriteDiscoverLeavesWith(spec *QuerySpec, resolved map[uint64][]float32) error {
	rewrite := func(l *QueryLeaf) error {
		if l.Kind != LeafDiscover {
			return nil
		}
		if l.Space != "" {
			return ErrQueryDiscoverHasSpace
		}
		if len(l.DiscoverContext) == 0 && len(l.DiscoverContextIDs) == 0 {
			return ErrQueryDiscoverNoContext
		}
		// Already-resolved (raw vectors, no ids): partition-invariant as-is.
		if len(l.DiscoverContextIDs) == 0 && len(l.DiscoverTargetID) == 0 {
			return nil
		}
		// Target: a named id MUST resolve cluster-wide (fail loud); no id ⇒ no anchor.
		if len(l.DiscoverTargetID) > 0 {
			tv, ok := resolved[l.DiscoverTargetID[0]]
			if !ok {
				return ErrIDNotFound
			}
			l.DiscoverTarget = tv
		}
		// Context pairs: resolve ids → vectors, keeping only fully-resolved pairs (skip
		// the missing, mirroring discover.go); at least one must survive.
		if len(l.DiscoverContextIDs) > 0 {
			pairs := make([]DiscoverPair, 0, len(l.DiscoverContextIDs))
			for _, cp := range l.DiscoverContextIDs {
				pv, okp := resolved[cp.Positive]
				nv, okn := resolved[cp.Negative]
				if !okp || !okn {
					continue
				}
				pairs = append(pairs, DiscoverPair{Pos: pv, Neg: nv})
			}
			if len(pairs) == 0 {
				return ErrIDNotFound
			}
			l.DiscoverContext = pairs
		}
		// Clear the ids so the partition handler's resolveDiscoverLeaves treats this leaf
		// as already-resolved (the embedded-vectors-only early return) instead of trying
		// to re-resolve against its LOCAL index (where cross-partition ids are absent).
		l.DiscoverTargetID = nil
		l.DiscoverContextIDs = nil
		return nil
	}
	// Tree-walk: embed the resolved vectors into the root + every prefetch source's
	// discover leaves, RECURSING into nested sub-spec sources (mirroring the single-node
	// resolveDiscoverLeaves tree-walk) so a nested discover leaf is resolved+embedded
	// partition-invariantly on the coordinator before fan-out.
	if err := rewrite(&spec.Root); err != nil {
		return err
	}
	for i := range spec.Prefetch {
		src := spec.Prefetch[i]
		if src.Leaf != nil {
			if err := rewrite(src.Leaf); err != nil {
				return err
			}
			continue
		}
		if src.Spec != nil {
			if err := RewriteDiscoverLeavesWith(src.Spec, resolved); err != nil {
				return err
			}
		}
	}
	return nil
}

// ExcludeExamplesFromResults drops the recommend example ids from a flat VectorResult-
// shaped top-k (the cross-partition coordinator's merged result) and re-truncates to
// k. It is the cluster analogue of excludeExamplesFromResult, which prunes the
// per-collection mode-tagged QueryResult; here the coordinator has already merged
// the per-partition results into a flat ordered list, so the prune+truncate happens
// on that list. Returns the pruned list (caller-supplied slice is reused).
func ExcludeExamplesFromResults[T any](results []T, idOf func(T) uint64, exclude map[uint64]struct{}, k int) []T {
	if len(exclude) == 0 {
		return results
	}
	// Prune the example ids AND dedup by id (see excludeExamplesFromResult): a
	// recommend result is a set of distinct points; deduping keeps the cross-partition
	// merged result == the single-node path even when an over-fetched derived-vector
	// search surfaces a duplicate id on one partition.
	seen := make(map[uint64]struct{}, len(results))
	out := results[:0]
	for _, r := range results {
		id := idOf(r)
		if _, drop := exclude[id]; drop {
			continue
		}
		if _, dup := seen[id]; dup {
			continue
		}
		seen[id] = struct{}{}
		out = append(out, r)
	}
	if k > 0 && len(out) > k {
		out = out[:k]
	}
	return out
}

// excludeExamplesFromResult drops the recommend example ids from a mode-tagged
// QueryResult (both the Fused top-k and every unfused Lane), then re-truncates
// Fused to k — mirroring Recommend's exclude-examples contract. The lanes are
// pruned too so the cross-partition coordinator never re-surfaces an
// example id when it re-fuses the unioned lanes.
func excludeExamplesFromResult(qr *QueryResult, exclude map[uint64]struct{}, k int) {
	// Prune the example ids AND deduplicate by id: the over-fetched derived-vector
	// search can, for some derived query vectors, surface the same point id more than
	// once (an HNSW over-fetch edge case); recommend results must be a set of distinct
	// points, and deduping here ALSO keeps the single-node path == the cross-partition
	// path (where partition-disjoint ids are inherently distinct), so the P>1==P1
	// recommend oracle holds regardless of the engine's per-query search behavior.
	prune := func(in []Result) []Result {
		seen := make(map[uint64]struct{}, len(in))
		out := in[:0]
		for _, r := range in {
			if _, drop := exclude[r.ID]; drop {
				continue
			}
			if _, dup := seen[r.ID]; dup {
				continue
			}
			seen[r.ID] = struct{}{}
			out = append(out, r)
		}
		return out
	}
	qr.Fused = prune(qr.Fused)
	if k > 0 && len(qr.Fused) > k {
		qr.Fused = qr.Fused[:k]
	}
	for i := range qr.Lanes {
		qr.Lanes[i] = prune(qr.Lanes[i])
	}
}

// runQuerySpec is the shared single-collection Query core for BOTH the dense
// (*Collection).Query and the named (*NamedCollection).Query: it validates the
// spec (a non-empty prefetch list, a known leaf kind per leaf, a known mode, a
// non-empty RERANK root), runs each prefetch leaf into a lane via execLeaf, then
// either FUSES the lanes (fuseLanes — the N-lane fold) or RERANKS the union of
// the prefetch candidates via rerankRoot. The two collection families differ
// ONLY in how a leaf executes (dense index vs the named space dispatch) and how
// the rerank root re-scores its candidate set, so those two operations are
// injected as closures; everything else (validation order, fuse fold, candidate
// union, mode tagging) is identical — guaranteeing the dense path stays
// behaviour-identical while the named path reuses it verbatim.
func runQuerySpec(spec QuerySpec, execLeaf func(QueryLeaf, int) ([]Result, error), rerankRoot func(QueryLeaf, []uint64, int) ([]Result, error)) (QueryResult, error) {
	return runQuerySpecAt(spec, execLeaf, rerankRoot, 0)
}

// runQuerySpecAt is the depth-tracked recursive core of runQuerySpec. The top-level
// spec runs at depth 0; each nested QuerySpec prefetch source recurses at depth+1.
// A prefetch source is either a LEAF (execLeaf produces its lane — the unchanged
// 1-level path) or a nested SUB-SPEC (its own fused/reranked top-k IS the lane —
// the recursion). The SAME execLeaf / rerankRoot closures (this index's per-leaf
// executor + rerank scorer) are threaded through every level, so a nested sub-spec
// runs against the SAME index/partition as the parent.
//
// CROSS-PARTITION (P>1==P1) SCOPE — the sub-spec lane carries a PARTITION-INVARIANT
// per-id rank key for: (1) a LEAF source (raw dense Distance / sparse Score — the
// same value regardless of partitioning); (2) a nested RERANK sub-spec (the root
// re-scores the partition-disjoint candidate union with a per-candidate function, so
// each partition produces the globally-correct score for its own ids); (3) a nested
// SINGLE-LANE FUSION sub-spec (the inner leaf's invariant key passes straight through
// — no fold). For these the fan-out (each partition runs the whole tree over its own
// data → a partition-local lane the coordinator unions like a leaf lane) is exact:
// P>1==P1. A nested MULTI-LANE FUSION sub-spec is fused partition-LOCALLY (its fused
// score is rank/normalization derived → partition-dependent), so its cross-partition
// result is exact only when its candidates do not span partitions; the single-node
// (P=1) nested multi-lane FUSION result is always correct (vs the hand oracle).
//
// The per-source EFFECTIVE ORIENTATION drives the parent's fuse fold: a leaf source
// → leaf.ScoreDesc; a spec source → specOrientation(sub) (its root orientation for
// RERANK, its fused orientation for FUSION). lane0's effective orientation picks the
// fold start (Fuse vs FuseScoreLanes) exactly as the flat path does.
//
// A defensive depth bound (maxQueryDepthExec) rejects a malformed in-memory spec that
// would otherwise recurse unbounded (the decode codec bounds the wire depth; this
// guards a directly-built spec).
func runQuerySpecAt(spec QuerySpec, execLeaf func(QueryLeaf, int) ([]Result, error), rerankRoot func(QueryLeaf, []uint64, int) ([]Result, error), depth int) (QueryResult, error) {
	if depth > MaxQueryDepthExec {
		return QueryResult{}, ErrQuerySpecTooDeep
	}
	// Breadth bound (the companion to the depth bound): a node carrying more than
	// MaxPrefetchSources prefetch sources is rejected fail-loud at EVERY nesting level
	// (the recursion below re-checks each sub-spec), so the per-level candidate union is
	// bounded. Structural ⇒ identical regardless of P.
	if len(spec.Prefetch) > MaxPrefetchSources {
		return QueryResult{}, ErrTooManyPrefetchSources
	}
	if len(spec.Prefetch) == 0 {
		return QueryResult{}, ErrQueryNoPrefetch
	}
	// Validate each source up front: a leaf source validates its kind; a spec source
	// must be non-empty (a malformed source with neither arm is fail-loud).
	for i := range spec.Prefetch {
		switch {
		case spec.Prefetch[i].Leaf != nil:
			if err := validateLeafKind(*spec.Prefetch[i].Leaf); err != nil {
				return QueryResult{}, err
			}
		case spec.Prefetch[i].Spec != nil:
			// validated by the recursive run below
		default:
			return QueryResult{}, ErrQueryBadLeafKind
		}
	}
	k := spec.K
	if k <= 0 {
		k = 10
	}

	// Execute every prefetch source into a lane: a leaf → execLeaf (the unchanged
	// lane); a sub-spec → recursively run it (its fused/reranked top-k IS the lane).
	// laneOrient[i] is source i's EFFECTIVE orientation for the parent fuse fold.
	lanes := make([][]Result, len(spec.Prefetch))
	laneOrient := make([]bool, len(spec.Prefetch))
	for i := range spec.Prefetch {
		src := spec.Prefetch[i]
		if src.Leaf != nil {
			lane, err := execLeaf(*src.Leaf, k)
			if err != nil {
				return QueryResult{}, err
			}
			lanes[i] = lane
			laneOrient[i] = src.Leaf.ScoreDesc
			continue
		}
		// Nested sub-spec: run it against the SAME index (same execLeaf/rerankRoot) at
		// depth+1; its fused/reranked top-k becomes this lane. The sub-spec's effective
		// orientation (specOrientation) is the lane's orientation for the parent fuse.
		sub := *src.Spec
		subRes, err := runQuerySpecAt(sub, execLeaf, rerankRoot, depth+1)
		if err != nil {
			return QueryResult{}, err
		}
		lanes[i] = subRes.Fused
		laneOrient[i] = specOrientation(sub)
	}

	switch spec.Mode {
	case ModeFusion:
		// lane0's EFFECTIVE orientation decides the FOLD start: distance-asc (dense/
		// named, or a sub-spec whose fused result is distance-asc) → Fuse (which inverts
		// lane0's distance into a score); score-desc (MV / a sub-spec whose fused result
		// is a score) → FuseScoreLanes (lane0 already a score). For a flat leaf-source
		// spec laneOrient[0] == spec.Prefetch[0].Leaf.ScoreDesc, byte-identical to today.
		fused := fuseLanes(lanes, laneOrient[0], spec.Method, spec.Alpha, spec.RRFK, k)
		return QueryResult{Mode: ModeFusion, Fused: fused, Lanes: lanes}, nil
	case ModeRerank:
		if err := validateLeafKind(spec.Root); err != nil {
			return QueryResult{}, err
		}
		if spec.Root.Kind == LeafDense && len(spec.Root.Dense) == 0 {
			return QueryResult{}, ErrQueryRerankNoRoot
		}
		if spec.Root.Kind == LeafSparse && spec.Root.Sparse.IsZero() {
			return QueryResult{}, ErrQueryRerankNoRoot
		}
		if spec.Root.Kind == LeafDiscover && len(spec.Root.DiscoverContext) == 0 {
			return QueryResult{}, ErrQueryRerankNoRoot
		}
		if spec.Root.Kind == LeafRecommend && len(spec.Root.RecPosVecs) == 0 {
			// A BEST_SCORE recommend root re-scores by the embedded positive/negative
			// example vectors; with no resolved positives it has no scorer (an AVERAGE
			// recommend root would have been rewritten to dense and never reaches here).
			return QueryResult{}, ErrQueryRerankNoRoot
		}
		cands := unionCandidates(lanes)
		reranked, err := rerankRoot(spec.Root, cands, k)
		if err != nil {
			return QueryResult{}, err
		}
		return QueryResult{Mode: ModeRerank, Fused: reranked}, nil
	default:
		return QueryResult{}, ErrQueryBadMode
	}
}

// collectTreeLanesAt is the UNFUSED sibling of runQuerySpecAt: it executes the spec
// tree against the SAME index (the injected execLeaf/rerankRoot closures) but, instead
// of FUSING each multi-lane FUSION node into a single lane, it returns the node's
// children's lanes UNFUSED in a DETERMINISTIC pre-order traversal. The coordinator
// re-walks the IDENTICAL spec tree consuming these lanes in the SAME order and folds
// each FUSION node over the cross-partition GLOBAL union ⇒ P>1==P1 EXACT at EVERY
// FUSION node (the recursive generalization of the top-level union→truncate→fuse-once
// invariant). The per-node rule (mirrored EXACTLY by the coordinator's appendTreeLaneSpec):
//
//   - FUSION node: for each prefetch SOURCE in order, if the source is a nested
//     MULTI-lane FUSION sub-spec → recurse (expand its children's lanes); else → ONE
//     lane (a leaf → execLeaf; a nested RERANK / single-lane FUSION sub-spec → its
//     partition-exact fused/reranked top-k via runQuerySpecAt — partition-invariant, so
//     it unions losslessly into the parent like a leaf lane).
//   - RERANK node: ONE lane — its partition-exact reranked top-k (scores absolute, ids
//     partition-disjoint), exactly as runQuerySpecAt produces it. RERANK children are
//     NEVER expanded (a nested FUSION buried under a RERANK ships fused).
//
// The "nested multi-lane FUSION → expand" predicate is len(sub.Prefetch) >= 2 &&
// sub.Mode == ModeFusion — the SAME predicate SpecHasNestedFusion uses, so the
// codec-selection shape and the traversal shape can never disagree.
func collectTreeLanesAt(spec QuerySpec, execLeaf func(QueryLeaf, int) ([]Result, error), rerankRoot func(QueryLeaf, []uint64, int) ([]Result, error), depth int) ([][]Result, error) {
	if depth > MaxQueryDepthExec {
		return nil, ErrQuerySpecTooDeep
	}
	if len(spec.Prefetch) > MaxPrefetchSources {
		return nil, ErrTooManyPrefetchSources
	}
	if len(spec.Prefetch) == 0 {
		return nil, ErrQueryNoPrefetch
	}
	// A RERANK node ships its single partition-exact fused lane (no expansion): run the
	// node through runQuerySpecAt (which validates the root + reranks) and take its Fused.
	if spec.Mode == ModeRerank {
		res, err := runQuerySpecAt(spec, execLeaf, rerankRoot, depth)
		if err != nil {
			return nil, err
		}
		return [][]Result{res.Fused}, nil
	}
	if spec.Mode != ModeFusion {
		return nil, ErrQueryBadMode
	}
	k := spec.K
	if k <= 0 {
		k = 10
	}
	var lanes [][]Result
	for i := range spec.Prefetch {
		src := spec.Prefetch[i]
		switch {
		case src.Leaf != nil:
			if err := validateLeafKind(*src.Leaf); err != nil {
				return nil, err
			}
			lane, err := execLeaf(*src.Leaf, k)
			if err != nil {
				return nil, err
			}
			lanes = append(lanes, lane)
		case src.Spec != nil:
			sub := *src.Spec
			if sub.Mode == ModeFusion && len(sub.Prefetch) >= 2 {
				// Nested MULTI-lane FUSION: EXPAND — recurse to ship its children's lanes
				// unfused (the coordinator folds them over the global union).
				subLanes, err := collectTreeLanesAt(sub, execLeaf, rerankRoot, depth+1)
				if err != nil {
					return nil, err
				}
				lanes = append(lanes, subLanes...)
				continue
			}
			// Nested RERANK / single-lane FUSION: ONE partition-invariant fused lane.
			subRes, err := runQuerySpecAt(sub, execLeaf, rerankRoot, depth+1)
			if err != nil {
				return nil, err
			}
			lanes = append(lanes, subRes.Fused)
		default:
			return nil, ErrQueryBadLeafKind
		}
	}
	return lanes, nil
}

// specOrientation reports the EFFECTIVE lane orientation of a spec's fused/reranked
// top-k — the orientation the PARENT fuse reads when this spec is a nested prefetch
// source. It mirrors how runQuerySpecAt produces the spec's Fused result:
//
//   - RERANK: Fused is the root leaf's rerank output, oriented by the root's lane
//     orientation (dense root → distance-asc; sparse/discover/MV root → score-desc),
//     so the orientation is spec.Root.ScoreDesc.
//   - FUSION with 2+ lanes: fuseLanes always folds via Fuse/FuseScoreLanes, whose
//     running fused lane is SCORE-descending (Result.Score is the rank key), so the
//     orientation is score-desc (true) regardless of the lane inputs.
//   - FUSION with exactly 1 lane: fuseLanes returns that single lane verbatim (no
//     fold, no score synthesis), so the orientation is lane0's own orientation — a
//     leaf source's ScoreDesc, or recursively the sub-spec's orientation. A 0-lane
//     spec is rejected upstream (ErrQueryNoPrefetch); it defaults to distance-asc.
func specOrientation(spec QuerySpec) bool {
	if spec.Mode == ModeRerank {
		return spec.Root.ScoreDesc
	}
	// FUSION.
	if len(spec.Prefetch) == 1 {
		src := spec.Prefetch[0]
		if src.Leaf != nil {
			return src.Leaf.ScoreDesc
		}
		if src.Spec != nil {
			return specOrientation(*src.Spec)
		}
		return false
	}
	// 2+ lanes: the fold's running fused lane is always score-descending.
	return true
}

// SourceOrientation reports the EFFECTIVE lane orientation of one prefetch SOURCE —
// the orientation the parent fuse fold (and the cross-partition per-lane truncation)
// reads for that source's lane. A leaf source → leaf.ScoreDesc; a nested sub-spec
// source → specOrientation(sub) (its fused/reranked orientation). It is the exported
// accessor the fan-out coordinator (query_fanout.go) uses so the per-partition lane
// merge reads the SAME orientation the single-node engine threaded, keeping P>1==P1
// across nested sources. A malformed (empty) source defaults to distance-asc.
func SourceOrientation(s QuerySource) bool {
	if s.Leaf != nil {
		return s.Leaf.ScoreDesc
	}
	if s.Spec != nil {
		return specOrientation(*s.Spec)
	}
	return false
}

// SourceLanePool reports the per-lane candidate-pool size for one prefetch SOURCE —
// the pool the cross-partition coordinator truncates that source's unioned lane to so
// it matches what a single partition holding all the data produced. A leaf source →
// leafLanePool(leaf, k) (the exact per-leaf pool the engine used). A nested sub-spec
// source → the sub-spec's effective final top-k (its K, else the engine default 10):
// runQuerySpecAt truncates a sub-spec's fused/reranked result to ITS OWN k, so each
// partition's sub-spec lane already holds at most that many — the coordinator unions
// the partition sub-spec lanes and re-truncates to the SAME k.
func SourceLanePool(s QuerySource, k int) int {
	if s.Leaf != nil {
		// leafLanePool already clamps to MaxLanePool — single-node == coordinator.
		return leafLanePool(*s.Leaf, k)
	}
	if s.Spec != nil {
		sk := s.Spec.K
		if sk <= 0 {
			sk = 10
		}
		return min(sk, MaxLanePool)
	}
	if k < 50 {
		return 50
	}
	return min(k, MaxLanePool)
}

// execLeaf runs a single prefetch leaf as a candidate lane. Dense → distance
// ascending (SearchFiltered); sparse → score descending (the sparse lane via
// HybridLanes with an empty dense). The lane POOL is leafLanePool(leaf, k): the
// leaf's LaneK if set, else its K, else the engine default max(k,50) — the SAME
// pool HybridSearch's buildLanes uses (DenseK/SparseK default max(k,50)), so a
// FUSION query with default lane pools fuses byte-for-byte identically to the
// equivalent HybridSearch. The lane is NOT further truncated here: fusion/rerank
// consume the full pool and truncate to the final k.
func (c *Collection) execLeaf(leaf QueryLeaf, k int) ([]Result, error) {
	pool := leafLanePool(leaf, k)
	switch leaf.Kind {
	case LeafDense:
		return c.SearchFiltered(leaf.Dense, pool, leaf.Filter)
	case LeafSparse:
		// HybridLanes with an empty dense vector runs ONLY the sparse lane
		// (score-descending), reusing the exact inverted-index path the hybrid
		// search uses — no new sparse entry point needed.
		opts := HybridOpts{Filter: leaf.Filter, SparseK: pool}
		_, sparseRes, err := c.HybridLanes(nil, leaf.Sparse, pool, opts)
		if err != nil {
			return nil, err
		}
		return sparseRes, nil
	case LeafDiscover:
		// DISCOVER lane: a custom per-candidate context-pair scorer (score-desc,
		// ScoreDesc=true). The resolve pre-pass has embedded the target/context
		// VECTORS, so DiscoverVecs runs the discover scorer over the index candidates
		// (the SAME math as the engine Discover — the equivalence oracle). The lane
		// returns its top-`pool` results; FetchK is left 0 so DiscoverVecs sizes the
		// candidate SCAN pool the same way the engine Discover does (max(4*k,50)) —
		// keeping the leaf == the engine on the same input. fusion/rerank truncate to
		// the final k.
		return c.DiscoverVecs(pool, DiscoverVecsOpts{
			Target:  leaf.DiscoverTarget,
			Context: leaf.DiscoverContext,
			Filter:  leaf.Filter,
		})
	case LeafRecommend:
		// BEST_SCORE recommend lane: a custom per-candidate max-similarity scorer
		// (score-desc, ScoreDesc=true). An AVERAGE_VECTOR recommend leaf never reaches
		// here (validateLeafKind rejects it — it is rewritten to dense by the pre-pass);
		// only a BEST_SCORE leaf executes directly. The resolve pre-pass has embedded the
		// example VECTORS, so RecommendVecs runs the bestScore scorer over the index
		// candidates (the SAME math as the engine — the equivalence oracle). FetchK is
		// left 0 so RecommendVecs sizes the candidate SCAN pool the same way the engine
		// does (max(4*k,50)). fusion/rerank truncate to the final k.
		if leaf.Strategy != RecommendBestScore {
			return nil, ErrQueryBadLeafKind
		}
		return c.RecommendVecs(pool, RecommendVecsOpts{
			Positive: leaf.RecPosVecs,
			Negative: leaf.RecNegVecs,
			Filter:   leaf.Filter,
		})
	default:
		return nil, ErrQueryBadLeafKind
	}
}

// leafLanePool is the per-lane candidate-pool size for a leaf: the leaf's LaneK
// if positive, else its K if positive, else the engine default max(k,50) — the
// exact default HybridSearch/buildLanes applies to DenseK/SparseK, so a query
// with default pools matches the equivalent hybrid search.
func leafLanePool(leaf QueryLeaf, k int) int {
	// The per-lane pool is clamped to MaxLanePool (the structural breadth-of-each-lane
	// bound): a LaneK above the ceiling is capped exactly as if the client had passed
	// LaneK=MaxLanePool (the lane still returns its score-ordered top-N) — deterministic
	// and P-invariant. The single-node and coordinator paths share THIS helper (via
	// SourceLanePool), so single-node == coordinator by construction.
	return min(leafLanePoolRaw(leaf, k), MaxLanePool)
}

// leafLanePoolRaw is the unclamped per-lane pool (the leaf's LaneK if positive, else
// its K if positive, else the engine default max(k,50)); leafLanePool clamps it to
// MaxLanePool. Split out so the clamp lives in exactly one place.
func leafLanePoolRaw(leaf QueryLeaf, k int) int {
	if leaf.LaneK > 0 {
		return leaf.LaneK
	}
	if leaf.K > 0 {
		return leaf.K
	}
	if k < 50 {
		return 50
	}
	return k
}

// fuseLanes folds the prefetch lanes into the fused top-k. The dense-family
// hybrid case (a dense lane + a sparse lane) is the canonical 2-lane Fuse. For
// the general case it folds left-to-right: lane[0] is treated as the "dense"
// (distance-ascending) lane and each subsequent lane is fused in as the "sparse"
// (score-descending) lane via vector.Fuse, re-feeding the running fused result
// (which carries Score desc) as the first argument through FuseScoreLanes for the
// 2nd+ fold so a score-oriented running lane is not inverted. v1 cross-partition
// fan-out only fuses the 2-lane dense+sparse case at the coordinator (>2 lanes is
// a documented v2 follow-up); this single-node fold is exact for any N.
//
// lane0ScoreDesc is lane0's ORIENTATION: false (the dense / named-dense path) →
// the fold starts with Fuse (lane0 is distance-ascending; Fuse inverts it into a
// score), byte-for-byte the dense/named hybrid path; true (the MV path, both
// lanes already score-descending) → the fold starts with FuseScoreLanes (lane0 is
// already a score, NOT inverted). Either way the running fused lane is
// score-descending after the first fold, so every remaining lane folds via
// FuseScoreLanes. A dense/named query (lane0ScoreDesc=false) keeps the EXACT
// pre-orientation behaviour (the Fuse start unchanged); an MV query
// (lane0ScoreDesc=true) correctly treats lane0 as a score.
func fuseLanes(lanes [][]Result, lane0ScoreDesc bool, method FusionMethod, alpha float64, rrfK, k int) []Result {
	switch len(lanes) {
	case 0:
		return nil
	case 1:
		// A single lane: nothing to fuse — truncate to k, preserving its order.
		out := lanes[0]
		if k > 0 && len(out) > k {
			out = out[:k]
		}
		return out
	}
	// 2 lanes: lane0 distance-asc → the canonical dense+sparse Fuse (byte-for-byte
	// the hybrid fusion path); lane0 score-desc → FuseScoreLanes (MV: lane0 is a
	// score, not inverted).
	var fused []Result
	if lane0ScoreDesc {
		fused = FuseScoreLanes(lanes[0], lanes[1], method, alpha, rrfK, k)
	} else {
		fused = Fuse(lanes[0], lanes[1], method, alpha, rrfK, k)
	}
	// 3+ lanes: fold the rest in. The running `fused` is Score-descending, so use
	// FuseScoreLanes (which does NOT invert the first lane) to fold each remaining
	// score-oriented lane in without ranking the worst doc highest.
	for i := 2; i < len(lanes); i++ {
		fused = FuseScoreLanes(fused, lanes[i], method, alpha, rrfK, k)
	}
	return fused
}

// unionCandidates collects the distinct candidate ids across all lanes,
// preserving first-seen order (deterministic). Used by RERANK to form the set
// the root leaf re-scores.
func unionCandidates(lanes [][]Result) []uint64 {
	seen := make(map[uint64]struct{})
	out := make([]uint64, 0)
	for _, lane := range lanes {
		for _, r := range lane {
			if _, ok := seen[r.ID]; ok {
				continue
			}
			seen[r.ID] = struct{}{}
			out = append(out, r.ID)
		}
	}
	return out
}

// rerankByRoot re-scores the candidate id union by the root leaf, restricted to
// that set, returning the top-k. Dense root → filterFirstByID (exact brute-force
// over the candidate set, distance-ascending, using the collection's own
// metadata for the filter re-check). Sparse root → an exhaustive sparse-lane
// scan (the inverted index scores every candidate with a nonzero overlap)
// filtered to the candidate set, score-descending.
func (c *Collection) rerankByRoot(root QueryLeaf, cands []uint64, k int) ([]Result, error) {
	if len(cands) == 0 {
		return nil, nil
	}
	switch root.Kind {
	case LeafDense:
		pred, err := CompileFilter(root.Filter)
		if err != nil {
			return nil, err
		}
		// metaOf == nil → filterFirstByID re-checks the predicate against the
		// collection's OWN arena metadata (the dense family stores metadata
		// inline), which is exactly what a plain dense rerank wants.
		res := c.idx.filterFirstByID(nil, cands, root.Dense, k, pred, nil)
		return res, nil
	case LeafSparse:
		// Run the sparse lane exhaustively (laneK large enough to surface every
		// candidate the inverted index scores), then restrict to the candidate
		// union and take top-k. The sparse lane is exact (an inverted-index dot
		// product, not ANN), so a pool >= the number of distinct scored ids yields
		// the exact restricted scores. Size() bounds that count.
		pool := len(cands)
		if sz := c.Stats().Size; sz > pool {
			pool = sz
		}
		opts := HybridOpts{Filter: root.Filter, SparseK: pool}
		_, sparseRes, err := c.HybridLanes(nil, root.Sparse, pool, opts)
		if err != nil {
			return nil, err
		}
		candSet := make(map[uint64]struct{}, len(cands))
		for _, id := range cands {
			candSet[id] = struct{}{}
		}
		out := make([]Result, 0, len(cands))
		for _, r := range sparseRes {
			if _, ok := candSet[r.ID]; ok {
				out = append(out, r)
			}
		}
		// sparseRes is already score-descending; the filtered subset preserves
		// that order, but sort defensively (ties → lower id) for a stable contract.
		sort.SliceStable(out, func(a, b int) bool {
			if out[a].Score != out[b].Score {
				return out[a].Score > out[b].Score
			}
			return out[a].ID < out[b].ID
		})
		if k > 0 && len(out) > k {
			out = out[:k]
		}
		return out, nil
	case LeafDiscover:
		// DISCOVER rerank root: re-score the candidate union by the discover context-
		// pair scorer (the SAME math as the execLeaf / engine Discover), restricted to
		// the union, score-descending. The candidate vectors are resolved once via
		// vecsForIDs (PQ-drop reconstruct handled there); each is scored by
		// discoverScore over the embedded context pairs. Distance stays 0 (the discover
		// rerank has no pool-distance tiebreak — ties break on lower id for a stable
		// contract, mirroring the sparse rerank).
		resolved := c.idx.vecsForIDs(cands)
		dist := pickDistDim(c.cfg.Metric, c.cfg.Dim)
		out := make([]Result, 0, len(cands))
		for _, id := range cands {
			cv, ok := resolved[id]
			if !ok {
				continue
			}
			out = append(out, Result{ID: id, Score: discoverScore(cv, root.DiscoverContext, dist)})
		}
		sort.SliceStable(out, func(a, b int) bool {
			if out[a].Score != out[b].Score {
				return out[a].Score > out[b].Score
			}
			return out[a].ID < out[b].ID
		})
		if k > 0 && len(out) > k {
			out = out[:k]
		}
		return out, nil
	case LeafRecommend:
		// BEST_SCORE recommend rerank root: re-score the candidate union by the
		// bestScore merge (the SAME math as the execLeaf / RecommendVecs), restricted to
		// the union, score-descending. An AVERAGE recommend root never reaches here (it
		// is rewritten to a dense root). The candidate vectors are resolved once via
		// vecsForIDs (PQ-drop reconstruct handled there); each is scored by bestScore
		// over the embedded positive/negative example vectors. Ties break on lower id for
		// a stable contract (mirrors the discover/sparse rerank).
		if root.Strategy != RecommendBestScore {
			return nil, ErrQueryBadLeafKind
		}
		resolved := c.idx.vecsForIDs(cands)
		dist := pickDistDim(c.cfg.Metric, c.cfg.Dim)
		out := make([]Result, 0, len(cands))
		for _, id := range cands {
			cv, ok := resolved[id]
			if !ok {
				continue
			}
			out = append(out, Result{ID: id, Score: bestScore(cv, root.RecPosVecs, root.RecNegVecs, c.cfg.Metric, dist)})
		}
		sort.SliceStable(out, func(a, b int) bool {
			if out[a].Score != out[b].Score {
				return out[a].Score > out[b].Score
			}
			return out[a].ID < out[b].ID
		})
		if k > 0 && len(out) > k {
			out = out[:k]
		}
		return out, nil
	default:
		return nil, ErrQueryBadLeafKind
	}
}
