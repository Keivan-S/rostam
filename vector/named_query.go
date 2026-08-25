// SPDX-License-Identifier: Apache-2.0

package vector

import (
	"errors"
	"sort"
)

var (
	// ErrQueryNamedRerankRootKind is returned when a RERANK QuerySpec against a
	// NamedCollection carries a root leaf whose Kind is neither LeafDense nor
	// LeafSparse — specifically a LeafDiscover or LeafRecommend root. Those are
	// valid as PREFETCH lanes (where the named engine resolves and scores them in
	// their target space) but not as RERANK roots, because scoreByIDs can only
	// re-score a candidate union by a single dense or sparse distance/similarity;
	// discover and recommend are multi-exemplar scorers that require the per-lane
	// resolve pre-pass, not a post-hoc point-set re-ranking. The caller should
	// restructure the spec: make the discover/recommend a prefetch lane and use a
	// dense or sparse leaf as the root.
	ErrQueryNamedRerankRootKind = errors.New("vector: named rerank root leaf must be dense or sparse (discover/recommend are prefetch-only in named mode)")
)

// --- Unified Query API engine: the NAMED family (multi-space N-lane fusion) ---
//
// This is the named-collection analogue of (*Collection).Query (vector/query.go).
// A dense collection has only its single dense index (+ an optional sparse lane),
// so the dense Query is effectively a 2-lane hybrid; a NAMED collection has MANY
// configured spaces (dense and/or sparse), so a query whose prefetch leaves each
// target a different named SPACE fuses (or reranks) across N>2 spaces — the
// distinctive value the dense Query API could not deliver.
//
// It mirrors the dense path EXACTLY: it reuses the shared runQuerySpec core
// (validation order, the N-lane fuseLanes fold, the candidate union, the
// mode-tagged QueryResult) and only swaps in the per-leaf executor
// (execNamedLeaf → SearchNamed / SearchNamedSparse on the leaf's Space) and the
// rerank scorer (scoreByIDs → the root space's filterFirstByID for a dense root,
// a restricted sparse scan for a sparse root). Single-level prefetch (no
// recursion), dense + sparse leaf kinds, FUSION + RERANK modes — identical to v1.

// Query executes a unified query spec against this single named collection (one
// shard / one partition). Every leaf (root + prefetch) MUST carry a non-empty
// Space naming a configured vector space (fail loud otherwise — a named query
// without a target space is a request error, never silently routed to a default
// space). Each prefetch leaf runs as a lane in its OWN space (dense →
// SearchNamed distance-ascending, sparse → SearchNamedSparse score-descending);
// FUSION folds the N lanes via the SAME fuseLanes the dense path uses (so a
// 2-space dense+sparse FUSION equals the equivalent NamedHybrid), and RERANK
// unions the prefetch candidate ids and re-scores them by the ROOT leaf
// restricted to that set in the root's space. Returns a mode-tagged QueryResult
// identical in shape to (*Collection).Query, so the op handler + cross-partition
// coordinator encode/decode it the same way.
func (nc *NamedCollection) Query(spec QuerySpec) (QueryResult, error) {
	// GROUPED query (group_by) is DENSE-only in v1: the grouping post-process reads
	// each id's Metadata via the dense payload accessor, which the named family does
	// not expose to the shared grouped path yet. Reject fail-loud rather than silently
	// returning an ungrouped result; named grouping is a follow-up.
	if spec.GroupBy != "" {
		return QueryResult{}, ErrQueryGroupNotDense
	}
	// Every leaf (root included) MUST target a named space — at EVERY nesting depth.
	// The named family now supports NESTED prefetch sub-specs (the generic
	// runQuerySpec recursion driver threads the SAME per-space execNamedLeaf /
	// scoreByIDs closures through every level), so the per-leaf Space requirement is
	// enforced recursively by namedTreeRequireSpace (walking each sub-spec's leaves +
	// RERANK root). A flat leaf-source named spec keeps its exact validation surface
	// (namedTreeRequireSpace's top-level pass is byte-identical to the old loop).
	if err := namedTreeRequireSpace(&spec); err != nil {
		return QueryResult{}, err
	}
	// DISCOVER pre-pass (named): resolve each discover leaf's target + context-pair
	// IDS to their stored vectors from the LEAF'S SPACE index and EMBED them into the
	// leaf's vector fields, BEFORE runQuerySpec runs. Mirrors the dense
	// (*Collection).resolveDiscoverLeaves but resolves per-space (indexes[leaf.Space])
	// — the named discover execLeaf scores against the embedded vectors with the
	// SPACE's metric.
	if err := nc.resolveNamedDiscoverLeaves(&spec); err != nil {
		return QueryResult{}, err
	}
	// RECOMMEND pre-pass (named): for each recommend leaf, resolve the example ids →
	// the leaf's SPACE vectors and either DERIVE the AVERAGE_VECTOR query in-space
	// (rewriting the leaf → LeafDense with Space PRESERVED, so the existing
	// LeafDense→SearchNamed path runs unchanged) or EMBED the BEST_SCORE example
	// vectors (the leaf stays a LeafRecommend the named best-score execLeaf runs). It
	// returns the union of example ids to exclude from the final result, mirroring
	// the dense path.
	exclude, err := nc.resolveNamedRecommendLeaves(&spec)
	if err != nil {
		return QueryResult{}, err
	}
	if len(exclude) == 0 {
		return runQuerySpec(spec, nc.execNamedLeaf, nc.scoreByIDs)
	}
	// Over-fetch by the number of example ids so the final top-k still holds k
	// results after the examples are pruned (mirrors the dense Query exclude path).
	wantK := spec.K
	if wantK <= 0 {
		wantK = 10 // runQuerySpec's default
	}
	spec.K = wantK + len(exclude)
	qr, err := runQuerySpec(spec, nc.execNamedLeaf, nc.scoreByIDs)
	if err != nil {
		return QueryResult{}, err
	}
	excludeExamplesFromResult(&qr, exclude, wantK)
	return qr, nil
}

// QueryTreeLanes is the UNFUSED tree-lanes Query variant for a NAMED collection
// whose spec contains a nested MULTI-lane FUSION node (SpecHasNestedFusion == true)
// — the named-family analogue of (*Collection).QueryTreeLanes. Instead of fusing
// each FUSION node into a single lane (Query → runQuerySpecAt), it runs the SAME
// named pipeline (per-leaf Space validation, the discover/recommend resolve
// pre-passes, the per-space execNamedLeaf, the nested RERANK/single-lane-FUSION fold)
// but returns the node-expanded UNFUSED lanes in the deterministic pre-order
// collectTreeLanesAt walks — so the coordinator (and the single-shard path) folds
// EVERY FUSION node over the cross-partition GLOBAL union ⇒ P>1==P1 EXACT at every
// level (multi-space dense/sparse lanes fold via the orientation-aware coordinator
// fold). It uses the SAME GENERIC collectTreeLanesAt the dense path uses, with the
// named per-space closures (nc.execNamedLeaf / nc.scoreByIDs). Query (the flat fold)
// is UNCHANGED — only the nested-multi-lane-FUSION wire path routes here. The
// recommend example ids are kept PRESENT in the lanes (the coordinator/single-shard
// path prunes post-fold, mirroring the dense QueryTreeLanes over-fetch contract).
func (nc *NamedCollection) QueryTreeLanes(spec QuerySpec) ([][]Result, error) {
	if spec.GroupBy != "" {
		return nil, ErrQueryGroupNotDense
	}
	// Per-leaf Space required at EVERY depth (same recursive validation as Query).
	if err := namedTreeRequireSpace(&spec); err != nil {
		return nil, err
	}
	// DISCOVER + RECOMMEND pre-passes (tree-walked): resolve/embed/derive any
	// discover/recommend leaf at any depth in its SPACE BEFORE the tree-lanes emit —
	// exactly as (*NamedCollection).Query does, so the nested-FUSION wire path matches
	// the flat fold for recommend/discover-bearing specs.
	if err := nc.resolveNamedDiscoverLeaves(&spec); err != nil {
		return nil, err
	}
	exclude, err := nc.resolveNamedRecommendLeaves(&spec)
	if err != nil {
		return nil, err
	}
	if len(exclude) > 0 {
		// Over-fetch so wantK results survive the coordinator-side post-fold recommend
		// exclusion (the example ids stay in the lanes so DBSF/Weighted normalization
		// runs over the FULL candidate set, byte-identical to the P>1 path that prunes
		// the merged result POST-fold). Mirrors the dense QueryTreeLanes.
		wantK := spec.K
		if wantK <= 0 {
			wantK = 10
		}
		nExclude := len(exclude)
		spec.K = wantK + nExclude
		// Widen EVERY nested sub-spec's K by nExclude too: collectTreeLanesAt uses each
		// sub-spec's own K as the lane pool size for its direct leaf children, so a
		// recommend leaf buried at any depth (now rewritten to dense/best-score) would
		// otherwise fetch sub.K candidates — no room for the post-fold prune. Widening
		// every sub-spec K is conservative but avoids under-fill at any depth.
		widenNestedSpecsK(&spec, nExclude)
	}
	return collectTreeLanesAt(spec, nc.execNamedLeaf, nc.scoreByIDs, 0)
}

// widenNestedSpecsK adds n to the K field of every NESTED sub-spec (at depth ≥ 1)
// reachable from spec. It does NOT touch spec itself (the caller has already widened
// spec.K). Called only when a recommend pre-pass found exclude ids, so every nested
// sub-spec's lane pool is widened to leave room for the post-fold prune.
func widenNestedSpecsK(spec *QuerySpec, n int) {
	for i := range spec.Prefetch {
		sub := spec.Prefetch[i].Spec
		if sub == nil {
			continue
		}
		wantK := sub.K
		if wantK <= 0 {
			wantK = 10
		}
		sub.K = wantK + n
		widenNestedSpecsK(sub, n)
	}
}

// WidenNestedSpecsK is the exported entry point for widenNestedSpecsK — used by the
// cluster coordinator (named_recommend_discover_fanout.go) to widen nested sub-spec Ks
// when the coordinator's recommend pre-pass found exclude ids.
func WidenNestedSpecsK(spec *QuerySpec, n int) { widenNestedSpecsK(spec, n) }

// resolveNamedRecommendLeaves is the named single-node RECOMMEND coordinator
// pre-pass — the per-space analogue of (*Collection).resolveRecommendLeaves. For
// every recommend leaf in the spec (root + flat prefetch leaves) it resolves the
// example ids → the leaf's SPACE vectors (indexes[leaf.Space].vecsForIDs) and:
//
//   - AVERAGE_VECTOR: derives normalize(mean(pos)−mean(neg)) with the SPACE's
//     metric (cfg[Space].Metric) via the shared DeriveRecommendVector, then rewrites
//     the leaf → a LeafDense keeping Space, so the existing execNamedLeaf
//     LeafDense→SearchNamed path runs unchanged (zero new exec for AVERAGE).
//   - BEST_SCORE: embeds the resolved example vectors into RecPosVecs/RecNegVecs and
//     keeps the leaf a LeafRecommend the named best-score execLeaf runs.
//
// It returns the union of example ids (positive ∪ negative) across the rewritten
// leaves for post-filter exclusion. The named family REQUIRES a Space on every leaf
// (the Query entry already enforced it), so an empty Space is a coordinator bug; the
// space MUST be a configured DENSE space (ErrUnknownVectorName / modality mismatch
// otherwise). The L2+best_score+negatives fail-loud is reused with the SPACE metric.
// v1 named recommend is single-level (flat leaf sources): a nested sub-spec source
// is already rejected by the Query entry (ErrQueryNestedNotSupported).
func (nc *NamedCollection) resolveNamedRecommendLeaves(spec *QuerySpec) (map[uint64]struct{}, error) {
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
		// The space's metric + dim + per-id vector resolver, fail-loud on a missing or
		// modality-mismatched (sparse) space — the same admit contract SearchNamed uses.
		idx, metric, dim, err := nc.denseSpace(l.Space)
		if err != nil {
			return err
		}
		if l.Strategy == RecommendBestScore {
			// Already-embedded (the cluster coordinator's RewriteNamedRecommendLeavesWith
			// ran first and cleared the ids): leave as-is so the partition handler does NOT
			// re-resolve against its LOCAL space index (where cross-partition ids are
			// absent). Mirrors the dense resolveRecommendLeaves embed-once early return so
			// the per-partition named pre-pass is a no-op on an already-resolved leaf
			// (P>1==P1). Re-check L2+negatives with the SPACE metric for a directly-embedded
			// leaf (the coordinator should already have rejected).
			if len(l.RecPosVecs) > 0 && len(l.Positive) == 0 {
				if metric == L2 && len(l.RecNegVecs) > 0 {
					return ErrRecommendBestScoreL2Negatives
				}
				l.ScoreDesc = true
				return nil
			}
			// BEST_SCORE: resolve example ids → vectors from the LEAF'S space and embed.
			// Reject L2 + negatives BEFORE scoring (the -max_neg sign-flip inverts the
			// non-positive L2 similarity ranking) with the SPACE metric.
			if len(l.Positive) == 0 {
				return ErrNoRecommendExamples
			}
			if metric == L2 && len(l.Negative) > 0 {
				return ErrRecommendBestScoreL2Negatives
			}
			resolved := idx.vecsForIDs(append(append([]uint64(nil), l.Positive...), l.Negative...))
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
			// Clear K/LaneK so the lane uses the engine default pool (over-fetch survives
			// the exclusion); Space is PRESERVED (the named best-score execLeaf needs it).
			l.K = 0
			l.LaneK = 0
			l.ScoreDesc = true
			return nil
		}
		// AVERAGE_VECTOR: derive the in-space query vector with the SPACE metric and
		// rewrite the leaf → a dense leaf KEEPING Space, so execNamedLeaf's
		// LeafDense→SearchNamed arm runs it against the right space.
		resolved := idx.vecsForIDs(append(append([]uint64(nil), l.Positive...), l.Negative...))
		derived, derr := DeriveRecommendVector(dim, metric, resolved, l.Positive, l.Negative)
		if derr != nil {
			return derr
		}
		addExclude(l.Positive, l.Negative)
		*l = QueryLeaf{
			Kind:      LeafDense,
			Space:     l.Space,
			Dense:     derived,
			Filter:    l.Filter,
			ScoreDesc: false,
		}
		return nil
	}
	// Tree-walk every leaf (root + prefetch leaves) at EVERY depth, recursing into
	// nested sub-specs, so a recommend leaf buried at any depth is resolved/embedded
	// ONCE before the recursion driver runs. A flat spec walks exactly the old
	// top-level leaves (byte-identical).
	if err := walkSpecLeaves(spec, rewrite); err != nil {
		return nil, err
	}
	return exclude, nil
}

// resolveNamedDiscoverLeaves is the named single-node DISCOVER coordinator pre-pass
// — the per-space analogue of (*Collection).resolveDiscoverLeaves. For every
// discover leaf carrying UNRESOLVED ids it resolves them via the LEAF'S SPACE index
// (indexes[leaf.Space].vecsForIDs) and EMBEDS the resolved vectors into
// DiscoverTarget / DiscoverContext IN PLACE (the leaf stays a LeafDiscover the named
// discover execLeaf scores with the SPACE metric). A leaf already carrying raw
// context vectors and no ids is left as-is. Fail-loud contract mirrors the dense
// path: at least one context pair (ErrQueryDiscoverNoContext), a named target id
// must resolve (ErrIDNotFound), at least one context pair must fully resolve.
func (nc *NamedCollection) resolveNamedDiscoverLeaves(spec *QuerySpec) error {
	resolve := func(l *QueryLeaf) error {
		if l.Kind != LeafDiscover {
			return nil
		}
		idx, _, _, err := nc.denseSpace(l.Space)
		if err != nil {
			return err
		}
		if len(l.DiscoverContext) == 0 && len(l.DiscoverContextIDs) == 0 {
			return ErrQueryDiscoverNoContext
		}
		// Already-resolved (raw context vectors, no ids): leave as-is.
		if len(l.DiscoverContextIDs) == 0 && len(l.DiscoverTargetID) == 0 {
			return nil
		}
		ids := make([]uint64, 0, len(l.DiscoverTargetID)+2*len(l.DiscoverContextIDs))
		ids = append(ids, l.DiscoverTargetID...)
		for _, cp := range l.DiscoverContextIDs {
			ids = append(ids, cp.Positive, cp.Negative)
		}
		resolved := idx.vecsForIDs(ids)
		if len(l.DiscoverTargetID) > 0 {
			tv, ok := resolved[l.DiscoverTargetID[0]]
			if !ok {
				return ErrIDNotFound
			}
			l.DiscoverTarget = tv
		}
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
	// Tree-walk every leaf (root + prefetch leaves) at EVERY depth, recursing into
	// nested sub-specs, so a discover leaf buried at any depth is resolved/embedded
	// before the recursion driver runs. A flat spec walks exactly the old top-level
	// leaves (byte-identical).
	return walkSpecLeaves(spec, resolve)
}

// denseSpace resolves a named DENSE space to its index + metric + dim, fail-loud on
// an unknown space (ErrUnknownVectorName) or a SPARSE space (ErrSpaceModalityMismatch
// — recommend/discover are dense-vector concepts). cfg is immutable after
// construction; indexes is read under nc.mu (read), the same admit gate SearchNamed
// uses. Shared by the named recommend + discover pre-passes and the best-score/
// discover execLeaf arms.
func (nc *NamedCollection) denseSpace(space string) (VectorIndex, Metric, int, error) {
	nc.mu.RLock()
	defer nc.mu.RUnlock()
	idx, ok := nc.indexes[space]
	if !ok {
		if _, isSparse := nc.sparseSpaces[space]; isSparse {
			return nil, 0, 0, ErrSpaceModalityMismatch
		}
		return nil, 0, 0, ErrUnknownVectorName
	}
	p := nc.cfg[space]
	return idx, p.Metric, p.Dim, nil
}

// walkSpecLeaves visits EVERY leaf of a spec tree (each node's RERANK root + each
// leaf prefetch source) at EVERY nesting depth, recursing into nested sub-spec
// sources. It is the shared tree-walk the named recommend/discover pre-passes
// (single-node + coordinator) use so a recommend/discover leaf buried at any depth
// is resolved/embedded exactly once. fn receives a pointer so it can rewrite the
// leaf in place. A flat (1-level) spec visits exactly the top-level root + leaves,
// byte-identical to the old per-pass loops. The traversal order is root, then
// prefetch sources in order, depth-first — deterministic + partition-invariant.
func walkSpecLeaves(spec *QuerySpec, fn func(*QueryLeaf) error) error {
	if err := fn(&spec.Root); err != nil {
		return err
	}
	for i := range spec.Prefetch {
		if l := spec.Prefetch[i].Leaf; l != nil {
			if err := fn(l); err != nil {
				return err
			}
			continue
		}
		if sub := spec.Prefetch[i].Spec; sub != nil {
			if err := walkSpecLeaves(sub, fn); err != nil {
				return err
			}
		}
	}
	return nil
}

// namedTreeRequireSpace enforces the named family's per-leaf Space requirement at
// EVERY nesting depth (the recursion driver threads the per-space execNamedLeaf
// through nested sub-specs, so every leaf at every level must name a configured
// space). Per node: each prefetch leaf MUST carry a Space; a RERANK node's root
// MUST carry a Space; a non-RERANK root carrying a payload MUST carry a Space (a
// payload-less FUSION root is harmless). This mirrors the old flat top-level
// validation EXACTLY for a 1-level spec (the top node's prefetch leaves + root),
// and extends the identical rule to each nested sub-spec node. A malformed source
// (neither leaf nor sub-spec) is left to runQuerySpec's structural validation.
func namedTreeRequireSpace(spec *QuerySpec) error {
	return namedTreeRequireSpaceAt(spec, 0)
}

func namedTreeRequireSpaceAt(spec *QuerySpec, depth int) error {
	// Depth bound mirrors runQuerySpecAt's MaxQueryDepthExec guard so an over-deep
	// directly-built spec fails loud HERE (ErrQuerySpecTooDeep) instead of recursing
	// unbounded before the engine can reject it.
	if depth > MaxQueryDepthExec {
		return ErrQuerySpecTooDeep
	}
	for i := range spec.Prefetch {
		src := spec.Prefetch[i]
		if src.Leaf != nil {
			if src.Leaf.Space == "" {
				return ErrQueryNamedLeafNoSpace
			}
			continue
		}
		if src.Spec != nil {
			if err := namedTreeRequireSpaceAt(src.Spec, depth+1); err != nil {
				return err
			}
		}
	}
	if hasNamedRootPayload(spec.Root) && spec.Root.Space == "" {
		return ErrQueryNamedLeafNoSpace
	}
	// For RERANK the root is consulted; require its Space (a Space-less RERANK root
	// would re-score against no space). Applies at every depth (a nested RERANK
	// sub-spec re-scores in its root's space).
	if spec.Mode == ModeRerank && spec.Root.Space == "" {
		return ErrQueryNamedLeafNoSpace
	}
	return nil
}

// hasNamedRootPayload reports whether a root leaf carries an actual query payload
// (used to decide whether a Space is required on a non-RERANK spec's root — a
// FUSION spec's empty root is harmless and need not carry a space).
func hasNamedRootPayload(l QueryLeaf) bool {
	switch l.Kind {
	case LeafSparse:
		return !l.Sparse.IsZero()
	default: // LeafDense
		return len(l.Dense) > 0
	}
}

// execNamedLeaf runs a single prefetch leaf as a candidate lane in its named
// space: dense → SearchNamed (distance-ascending) on leaf.Space; sparse →
// SearchNamedSparse (score-descending) on leaf.Space. The lane POOL is
// leafLanePool(leaf, k) — the SAME default v1 uses (the leaf's LaneK, else its K,
// else max(k,50)), matching the per-lane pool NamedHybrid's namedHybridK applies,
// so a 2-space named FUSION with default pools fuses byte-for-byte identically to
// the equivalent NamedHybrid. SearchNamed / SearchNamedSparse each take nc.mu
// (read) and apply the shared live-meta/TTL/filter admit gate per leaf.
func (nc *NamedCollection) execNamedLeaf(leaf QueryLeaf, k int) ([]Result, error) {
	pool := leafLanePool(leaf, k)
	switch leaf.Kind {
	case LeafDense:
		return nc.SearchNamed(leaf.Space, leaf.Dense, pool, leaf.Filter)
	case LeafSparse:
		sq := leaf.Sparse
		return nc.SearchNamedSparse(leaf.Space, &sq, pool, leaf.Filter)
	case LeafDiscover:
		// DISCOVER lane (named): seed the candidate pool in the leaf's SPACE, then
		// re-score each candidate by the context-pair scorer with the SPACE metric
		// (the SAME discoverScore math as the dense engine — the equivalence oracle),
		// score-descending. The resolve pre-pass embedded the target/context vectors.
		return nc.discoverNamedLeaf(leaf, pool)
	case LeafRecommend:
		// BEST_SCORE recommend lane (named): an AVERAGE_VECTOR leaf never reaches here
		// (the pre-pass rewrote it to a LeafDense); only a BEST_SCORE leaf executes
		// directly. Seed the pool in the leaf's SPACE and re-score by the bestScore
		// merge with the SPACE metric.
		if leaf.Strategy != RecommendBestScore {
			return nil, ErrQueryBadLeafKind
		}
		return nc.recommendBestNamedLeaf(leaf, pool)
	default:
		return nil, ErrQueryBadLeafKind
	}
}

// discoverNamedLeaf runs a DISCOVER leaf against the leaf's named space: it builds
// the discover SEED query (discoverSeed — target, else the mean of the context
// positives) with the SPACE metric, fetches the candidate pool via SearchNamed
// (distance-ascending — applying the SHARED-payload metaOf filter gate, the named
// family's correct filter path), re-scores each candidate by discoverScore over the
// embedded context pairs with the SPACE metric, and sorts score-descending (pool
// distance tiebreak). This mirrors the dense discoverScoredLocked (seed → pool
// search → per-candidate score → sortDiscover) so a named discover == the dense
// engine discover run IN that space — the equivalence oracle. The candidate scan
// pool is sized discoverFetchK(0, pool) exactly as the engine sizes it.
func (nc *NamedCollection) discoverNamedLeaf(leaf QueryLeaf, pool int) ([]Result, error) {
	if pool <= 0 {
		return nil, nil
	}
	idx, metric, dim, err := nc.denseSpace(leaf.Space)
	if err != nil {
		return nil, err
	}
	if len(leaf.DiscoverContext) == 0 {
		return nil, ErrNoContextPairs
	}
	if leaf.DiscoverTarget != nil && len(leaf.DiscoverTarget) != dim {
		return nil, ErrDimMismatch
	}
	posVecs := make([][]float32, len(leaf.DiscoverContext))
	for i, p := range leaf.DiscoverContext {
		posVecs[i] = p.Pos
	}
	seed := make([]float32, dim)
	discoverSeed(seed, leaf.DiscoverTarget, posVecs, metric)
	fetchK := discoverFetchK(0, pool)
	cands, err := nc.SearchNamed(leaf.Space, seed, fetchK, leaf.Filter)
	if err != nil {
		return nil, err
	}
	if len(cands) == 0 {
		return nil, nil
	}
	dist := pickDistDim(metric, dim)
	resolved := idx.vecsForIDs(resultIDsOf(cands))
	out := cands[:0]
	for _, r := range cands {
		cv, ok := resolved[r.ID]
		if !ok {
			continue
		}
		out = append(out, Result{ID: r.ID, Distance: r.Distance, Score: discoverScore(cv, leaf.DiscoverContext, dist)})
	}
	sortDiscover(out)
	if len(out) > pool {
		out = out[:pool]
	}
	return out, nil
}

// recommendBestNamedLeaf runs a BEST_SCORE recommend leaf against the leaf's named
// space: it builds the best-score SEED query (recommendBestSeed — the positives'
// centroid) with the SPACE metric, fetches the candidate pool via SearchNamed
// (distance-ascending — applying the shared-payload metaOf filter gate), re-scores
// each candidate by the bestScore merge over the embedded positive/negative example
// vectors with the SPACE metric, and sorts score-descending (pool distance
// tiebreak). Mirrors the dense recommendBestLocked so a named best-score recommend
// == the dense engine run IN that space — the equivalence oracle.
func (nc *NamedCollection) recommendBestNamedLeaf(leaf QueryLeaf, pool int) ([]Result, error) {
	if pool <= 0 {
		return nil, nil
	}
	idx, metric, dim, err := nc.denseSpace(leaf.Space)
	if err != nil {
		return nil, err
	}
	if len(leaf.RecPosVecs) == 0 {
		return nil, ErrNoRecommendExamples
	}
	seed := make([]float32, dim)
	recommendBestSeed(seed, leaf.RecPosVecs, metric)
	fetchK := discoverFetchK(0, pool)
	cands, err := nc.SearchNamed(leaf.Space, seed, fetchK, leaf.Filter)
	if err != nil {
		return nil, err
	}
	if len(cands) == 0 {
		return nil, nil
	}
	dist := pickDistDim(metric, dim)
	resolved := idx.vecsForIDs(resultIDsOf(cands))
	out := cands[:0]
	for _, r := range cands {
		cv, ok := resolved[r.ID]
		if !ok {
			continue
		}
		out = append(out, Result{ID: r.ID, Distance: r.Distance, Score: bestScore(cv, leaf.RecPosVecs, leaf.RecNegVecs, metric, dist)})
	}
	sortRecommendBest(out)
	if len(out) > pool {
		out = out[:pool]
	}
	return out, nil
}

// RewriteNamedRecommendLeavesWith is the CLUSTER coordinator analogue of the
// single-node (*NamedCollection).resolveNamedRecommendLeaves: it rewrites every
// recommend leaf (root + flat prefetch leaves) IN PLACE using `resolved`, a
// cluster-wide id→per-space-vector map (the VectorNamedGetBatch result), and
// `spaceMeta`, a per-space (metric, dim) resolver backed by the named collection's
// config available at the coordinator. Each leaf's example ids are resolved to the
// LEAF'S SPACE vectors (resolved[id][leaf.Space]); then:
//
//   - AVERAGE_VECTOR: derives normalize(mean(pos)−mean(neg)) with the SPACE's metric
//     ONCE on the coordinator (DeriveRecommendVector, partition-invariant) and
//     rewrites the leaf → a LeafDense KEEPING Space, so each partition's
//     execNamedLeaf LeafDense→SearchNamed arm runs it against the right space.
//   - BEST_SCORE: embeds the resolved example VECTORS into RecPosVecs/RecNegVecs,
//     CLEARS the ids (so partitions treat the leaf as already-resolved — their
//     resolveNamedRecommendLeaves only resolves from local ids when RecPosVecs is
//     empty), and keeps the leaf a LeafRecommend the named best-score execLeaf runs.
//
// It returns the union of example ids (positive ∪ negative) for post-merge
// exclusion. The named family REQUIRES a Space on every leaf (the Query entry
// enforced it; an empty Space here is ErrQueryNamedLeafNoSpace). The space MUST be a
// configured DENSE space (spaceMeta returns ErrUnknownVectorName /
// ErrSpaceModalityMismatch otherwise). The L2+best_score+negatives fail-loud is
// reused with the SPACE metric. v1 named recommend is single-level (flat leaf
// sources); a nested sub-spec source was already rejected by the Query entry.
func RewriteNamedRecommendLeavesWith(spec *QuerySpec, resolved map[uint64]map[string][]float32, spaceMeta func(space string) (Metric, int, error)) (map[uint64]struct{}, error) {
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
	// spaceVecs picks the leaf's SPACE vectors from the multi-space resolved map.
	spaceVecs := func(space string, ids []uint64) map[uint64][]float32 {
		out := make(map[uint64][]float32, len(ids))
		for _, id := range ids {
			if spaces, ok := resolved[id]; ok {
				if v, ok := spaces[space]; ok && len(v) > 0 {
					out[id] = v
				}
			}
		}
		return out
	}
	rewrite := func(l *QueryLeaf) error {
		if l.Kind != LeafRecommend {
			return nil
		}
		if l.Space == "" {
			return ErrQueryNamedLeafNoSpace
		}
		metric, dim, err := spaceMeta(l.Space)
		if err != nil {
			return err
		}
		if l.Strategy == RecommendBestScore {
			if len(l.Positive) == 0 {
				return ErrNoRecommendExamples
			}
			if metric == L2 && len(l.Negative) > 0 {
				return ErrRecommendBestScoreL2Negatives
			}
			sv := spaceVecs(l.Space, append(append([]uint64(nil), l.Positive...), l.Negative...))
			posVecs := make([][]float32, 0, len(l.Positive))
			for _, id := range l.Positive {
				if v, ok := sv[id]; ok {
					posVecs = append(posVecs, v)
				}
			}
			if len(posVecs) == 0 {
				return ErrIDNotFound
			}
			var negVecs [][]float32
			for _, id := range l.Negative {
				if v, ok := sv[id]; ok {
					negVecs = append(negVecs, v)
				}
			}
			addExclude(l.Positive, l.Negative)
			l.RecPosVecs = posVecs
			l.RecNegVecs = negVecs
			// Clear the ids so the partition handler's resolveNamedRecommendLeaves treats
			// this leaf as already-resolved (its BEST_SCORE branch resolves from local ids
			// only when RecPosVecs is empty). Space PRESERVED (the named best-score execLeaf
			// needs it); clear K/LaneK so the over-fetch survives the exclusion.
			l.Positive = nil
			l.Negative = nil
			l.K = 0
			l.LaneK = 0
			l.ScoreDesc = true
			return nil
		}
		// AVERAGE_VECTOR: derive the in-space query vector with the SPACE metric ONCE on
		// the coordinator and rewrite the leaf → a dense leaf KEEPING Space.
		sv := spaceVecs(l.Space, append(append([]uint64(nil), l.Positive...), l.Negative...))
		derived, derr := DeriveRecommendVector(dim, metric, sv, l.Positive, l.Negative)
		if derr != nil {
			return derr
		}
		addExclude(l.Positive, l.Negative)
		*l = QueryLeaf{
			Kind:      LeafDense,
			Space:     l.Space,
			Dense:     derived,
			Filter:    l.Filter,
			ScoreDesc: false,
		}
		return nil
	}
	// Tree-walk every leaf (root + prefetch leaves) at EVERY depth, recursing into
	// nested sub-specs, so a recommend leaf buried at any depth is resolved/embedded
	// ONCE before the recursion driver runs. A flat spec walks exactly the old
	// top-level leaves (byte-identical).
	if err := walkSpecLeaves(spec, rewrite); err != nil {
		return nil, err
	}
	return exclude, nil
}

// RewriteNamedDiscoverLeavesWith is the CLUSTER coordinator analogue of the single-
// node (*NamedCollection).resolveNamedDiscoverLeaves: it EMBEDS the resolved
// target/context VECTORS into every id-bearing discover leaf (root + flat prefetch
// leaves) IN PLACE, picking each leaf's SPACE vectors from `resolved` (the
// cluster-wide id→per-space-vector map), then CLEARS the leaf's id fields so each
// partition handler sees an already-resolved LeafDiscover (no per-partition
// re-resolve). The leaf stays a LeafDiscover the named discover execLeaf runs
// (discoverNamedLeaf with the SPACE metric). It mirrors resolveNamedDiscoverLeaves'
// fail-loud contract so P>1 matches P==1: a Space-less leaf is rejected
// (ErrQueryNamedLeafNoSpace); a leaf with neither resolved vectors nor ids is
// rejected (ErrQueryDiscoverNoContext); a target id that does not resolve in-space
// cluster-wide is ErrIDNotFound; a context pair keeps only pairs whose BOTH ids
// resolve in-space and at least one must survive (else ErrIDNotFound). A leaf
// already carrying raw vectors and no ids is left untouched. spaceMeta is consulted
// only to validate the space is a configured DENSE space (fail-loud per-space).
func RewriteNamedDiscoverLeavesWith(spec *QuerySpec, resolved map[uint64]map[string][]float32, spaceMeta func(space string) (Metric, int, error)) error {
	spaceVec := func(space string, id uint64) ([]float32, bool) {
		if spaces, ok := resolved[id]; ok {
			if v, ok := spaces[space]; ok && len(v) > 0 {
				return v, true
			}
		}
		return nil, false
	}
	rewrite := func(l *QueryLeaf) error {
		if l.Kind != LeafDiscover {
			return nil
		}
		if l.Space == "" {
			return ErrQueryNamedLeafNoSpace
		}
		if _, _, err := spaceMeta(l.Space); err != nil {
			return err
		}
		if len(l.DiscoverContext) == 0 && len(l.DiscoverContextIDs) == 0 {
			return ErrQueryDiscoverNoContext
		}
		// Already-resolved (raw vectors, no ids): partition-invariant as-is.
		if len(l.DiscoverContextIDs) == 0 && len(l.DiscoverTargetID) == 0 {
			return nil
		}
		if len(l.DiscoverTargetID) > 0 {
			tv, ok := spaceVec(l.Space, l.DiscoverTargetID[0])
			if !ok {
				return ErrIDNotFound
			}
			l.DiscoverTarget = tv
		}
		if len(l.DiscoverContextIDs) > 0 {
			pairs := make([]DiscoverPair, 0, len(l.DiscoverContextIDs))
			for _, cp := range l.DiscoverContextIDs {
				pv, okp := spaceVec(l.Space, cp.Positive)
				nv, okn := spaceVec(l.Space, cp.Negative)
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
		// Clear the ids so the partition handler treats this leaf as already-resolved.
		l.DiscoverTargetID = nil
		l.DiscoverContextIDs = nil
		return nil
	}
	// Tree-walk every leaf (root + prefetch leaves) at EVERY depth, recursing into
	// nested sub-specs, so a discover leaf buried at any depth is embedded cluster-wide
	// before the fan-out. A flat spec walks exactly the old top-level leaves.
	return walkSpecLeaves(spec, rewrite)
}

// resultIDsOf extracts the ids of a Result slice (for a single batched vecsForIDs
// resolve in the named recommend/discover scorers).
func resultIDsOf(res []Result) []uint64 {
	ids := make([]uint64, len(res))
	for i, r := range res {
		ids[i] = r.ID
	}
	return ids
}

// scoreByIDs re-scores the candidate id union by the named root leaf, restricted
// to that set, returning the top-k — the named analogue of (*Collection).
// rerankByRoot. Dense root → the root space's index filterFirstByID (exact
// brute-force over the candidate set, distance-ascending, with the shared-payload
// predicate re-check via nc.metaOf()). Sparse root → a restricted sparse scan
// (SearchNamedSparse pooled large enough to surface every scored candidate, then
// filtered to the candidate set), score-descending. Orientation matches the dense
// rerank: dense distance-ascending, sparse score-descending.
func (nc *NamedCollection) scoreByIDs(root QueryLeaf, cands []uint64, k int) ([]Result, error) {
	if len(cands) == 0 {
		return nil, nil
	}
	switch root.Kind {
	case LeafDense:
		return nc.scoreByIDsDense(root.Space, root.Dense, cands, k, root.Filter)
	case LeafSparse:
		return nc.scoreByIDsSparse(root.Space, root.Sparse, cands, k, root.Filter)
	default:
		return nil, ErrQueryNamedRerankRootKind
	}
}

// scoreByIDsDense brute-force scores the candidate ids against the dense root
// space and returns the distance-ascending top-k. It mirrors the dense
// rerankByRoot: the per-space index's filterFirstByID scores ONLY the candidate
// set, re-checking the compiled filter against the SHARED per-point payload
// (nc.metaOf()) — the same admit gate SearchNamed uses. Fails loud on an unknown/
// modality-mismatched space or a bad filter.
func (nc *NamedCollection) scoreByIDsDense(space string, query []float32, cands []uint64, k int, filter Filter) ([]Result, error) {
	pred, err := CompileFilter(filter)
	if err != nil {
		return nil, err
	}
	nc.mu.RLock()
	defer nc.mu.RUnlock()
	idx, ok := nc.indexes[space]
	if !ok {
		if _, isSparse := nc.sparseSpaces[space]; isSparse {
			return nil, ErrSpaceModalityMismatch // a dense rerank root pointing at a sparse space
		}
		return nil, ErrUnknownVectorName
	}
	return idx.filterFirstByID(nil, cands, query, k, pred, nc.metaOf()), nil
}

// scoreByIDsSparse re-scores the candidate ids against the sparse root space and
// returns the score-descending top-k. It mirrors the dense rerankByRoot sparse
// arm: run the sparse lane with a pool large enough to surface every candidate
// the inverted index scores (the sparse dot product is exact, not ANN, so a pool
// >= the live-point count yields the exact restricted scores), then restrict to
// the candidate union and take top-k (ties → lower id, defensively sorted).
func (nc *NamedCollection) scoreByIDsSparse(space string, query SparseVector, cands []uint64, k int, filter Filter) ([]Result, error) {
	pool := len(cands)
	if n := nc.NumPoints(); n > pool {
		pool = n
	}
	sq := query
	res, err := nc.SearchNamedSparse(space, &sq, pool, filter)
	if err != nil {
		return nil, err
	}
	candSet := make(map[uint64]struct{}, len(cands))
	for _, id := range cands {
		candSet[id] = struct{}{}
	}
	out := make([]Result, 0, len(cands))
	for _, r := range res {
		if _, ok := candSet[r.ID]; ok {
			out = append(out, r)
		}
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
}
