// SPDX-License-Identifier: Apache-2.0

package vector

import (
	"errors"
	"sort"
)

// --- Unified Query API engine: the MV (multi-vector) family ---
//
// This is the multi-vector analogue of (*Collection).Query (vector/query.go) and
// (*NamedCollection).Query (vector/named_query.go). An MV collection holds a
// late-interaction (MaxSim) token index AND an optional doc-level sparse field in
// ONE collection (no named spaces), so an MV query's prefetch leaves are MaxSim
// leaves (LeafMVMaxSim, carrying the token query matrix) and/or the doc sparse
// field (LeafSparse). It reuses the shared runQuerySpec core verbatim, swapping in
// the per-leaf executor (execMVLeafLocked → maxSim / doc-sparse) and the rerank
// scorer (rerankMVLocked → maxSim / restricted sparse over the candidate union).
//
// THE MV-SPECIFIC CRUX: BOTH MV lanes are SCORE-DESCENDING (MaxSim relevance +
// sparse dot-product), unlike the dense/named lane0 which is DISTANCE-ASCENDING.
// Every MV leaf is constructed with ScoreDesc=true, so the shared orientation-
// aware fold (fuseLanes) starts with FuseScoreLanes (lane0 already a score, NOT
// inverted) — making a 2-lane MV FUSION equal the equivalent MVHybrid. The
// dense/named paths (lane0 ScoreDesc=false) keep the Fuse start unchanged.
//
// LOCK/CLOCK DISCIPLINE: like mvHybridLanesLocked, the whole query runs under ONE
// m.mu.RLock acquisition with ONE clock snapshot, so every lane (and the rerank
// re-score) sees a single consistent live/per-key-TTL view. Single-level prefetch
// (no recursion), FUSION + RERANK modes.

// Query API validation errors specific to the MV family (fail-loud at the engine
// edge). The shared dense/named errors (ErrQueryNoPrefetch, ErrQueryBadMode, …)
// are reused.
var (
	// ErrQueryMVLeafHasSpace is returned when a leaf in an MV (*MultiVectorIndex).
	// Query carries a non-empty Space (an MV collection has no named spaces — fail
	// loud rather than silently ignoring the Space).
	ErrQueryMVLeafHasSpace = errors.New("vector: mv query leaf carries a named space")
	// ErrQueryMVMaxSimNoTokens is returned when a MaxSim leaf carries no query
	// token matrix (an empty MaxSim leaf has nothing to score).
	ErrQueryMVMaxSimNoTokens = errors.New("vector: mv maxsim query leaf has no tokens")
	// ErrQueryMVSparseEmpty is returned when an MV sparse leaf carries an empty
	// sparse query (a doc-sparse leaf must carry terms).
	ErrQueryMVSparseEmpty = errors.New("vector: mv sparse query leaf is empty")
	// ErrQueryMVRecommendUnsupported is returned when an MV query carries a recommend
	// leaf: recommend/discover are SINGLE-vector concepts (mean-diff or per-candidate
	// max-similarity over ONE representative vector per point), but an MV doc is a SET
	// of token vectors (late-interaction) with no single representative vector to pool.
	// "mean of all positive tokens − mean of all negative tokens" discards the doc
	// structure and is NOT recommend; Qdrant likewise does not support recommend on
	// multivectors. Fail loud (an inherent semantic limitation, not a wiring gap)
	// rather than ship a meaningless pooled answer.
	ErrQueryMVRecommendUnsupported = errors.New("vector: recommend is not supported on a multi-vector collection (no single-vector pooling semantics)")
	// ErrQueryMVDiscoverUnsupported is the discover analogue of
	// ErrQueryMVRecommendUnsupported: discover's context-pair directional scorer needs
	// ONE representative vector per candidate, which an MV token-set doc does not have.
	// Inherent semantic limitation (Qdrant does not support discover on multivectors),
	// so it is rejected fail-loud rather than pooled into a meaningless score.
	ErrQueryMVDiscoverUnsupported = errors.New("vector: discover is not supported on a multi-vector collection (no single-vector pooling semantics)")
)

// Query executes a unified Query API spec against this multi-vector index (one
// shard / one partition). Prefetch leaves are MaxSim leaves (LeafMVMaxSim, the
// token query matrix) and/or the doc-level sparse field (LeafSparse); NO leaf may
// carry a Space (an MV collection has no named spaces — fail loud). Both lane
// orientations are score-descending, so the shared orientation-aware fold
// (fuseLanes) uses FuseScoreLanes from lane0 (a 2-lane MaxSim+sparse FUSION thus
// equals the equivalent MVHybrid). RERANK unions the prefetch candidate ids and
// re-scores them by the ROOT leaf restricted to that set (MaxSim root → exact
// maxSim over the candidate set; sparse root → a restricted doc-sparse scan).
// Returns a mode-tagged QueryResult identical in shape to the dense/named Query,
// so the op handler + fan-out coordinator encode/decode it the same way. The
// whole query runs under ONE m.mu.RLock with ONE clock snapshot (consistent
// live/per-key-TTL view across all lanes + the rerank re-score).
func (m *MultiVectorIndex) Query(spec QuerySpec) (QueryResult, error) {
	// GROUPED query (group_by) is DENSE-only in v1: the grouping post-process reads
	// each id's Metadata via the dense payload accessor, which the MV family does not
	// expose to the shared grouped path yet. Reject fail-loud rather than silently
	// returning an ungrouped result; MV grouping is a follow-up.
	if spec.GroupBy != "" {
		return QueryResult{}, ErrQueryGroupNotDense
	}
	// An MV collection has NO named spaces and MV-specific per-leaf payloads: at EVERY
	// nesting depth no leaf may carry a Space, every MaxSim leaf MUST carry tokens, and
	// every MV sparse leaf MUST carry terms. The MV family now supports NESTED prefetch
	// sub-specs (the generic runQuerySpec recursion driver threads the SAME
	// lock+snapshot-captured execMVLeafLocked / rerankMVLocked closures through every
	// level — see the single lock acquisition below), so the no-Space + MV-payload
	// checks are enforced recursively by mvTreeValidate (walking each sub-spec's leaves
	// + RERANK root). A flat leaf-source MV spec keeps its exact validation surface
	// (mvTreeValidate's top-level pass is byte-identical to the old loops).
	if err := mvTreeValidate(&spec); err != nil {
		return QueryResult{}, err
	}

	// One lock acquisition + one clock snapshot for the whole query: every lane and
	// the rerank re-score see a single consistent live/per-key-TTL view (mirror
	// mvHybridLanesLocked).
	m.mu.RLock()
	defer m.mu.RUnlock()
	now := m.nowMs()

	execLeaf := func(leaf QueryLeaf, k int) ([]Result, error) {
		return m.execMVLeafLocked(leaf, k, now)
	}
	rerankRoot := func(root QueryLeaf, cands []uint64, k int) ([]Result, error) {
		return m.rerankMVLocked(root, cands, k, now)
	}
	return runQuerySpec(spec, execLeaf, rerankRoot)
}

// mvTreeValidate enforces the MV family's no-Space + per-leaf-payload invariants at
// EVERY nesting depth (the recursion driver threads the MV locked closures through
// nested sub-specs, so every leaf at every level must satisfy the MV contract). Per
// node: each prefetch leaf MUST NOT carry a Space and MUST pass validateMVLeafPayload
// (MaxSim → tokens, sparse → terms, recommend/discover → unsupported); a RERANK
// node's root MUST NOT carry a Space and MUST pass validateMVLeafPayload; a non-RERANK
// root MUST NOT carry a Space. This mirrors the old flat top-level validation EXACTLY
// for a 1-level spec, and extends the identical rule to each nested sub-spec node. A
// malformed source (neither leaf nor sub-spec) is left to runQuerySpec's structural
// validation.
func mvTreeValidate(spec *QuerySpec) error {
	return mvTreeValidateAt(spec, 0)
}

func mvTreeValidateAt(spec *QuerySpec, depth int) error {
	// Depth bound mirrors runQuerySpecAt's MaxQueryDepthExec guard so an over-deep
	// directly-built spec fails loud HERE (ErrQuerySpecTooDeep) instead of recursing
	// unbounded before the engine can reject it.
	if depth > MaxQueryDepthExec {
		return ErrQuerySpecTooDeep
	}
	// Prefetch loop runs FIRST (Space check + payload check) — restoring the old flat
	// validation order (prefetch-before-root) so a multi-fault spec returns the same
	// first error as the pre-nested-support code. Nested sub-specs recurse depth-first.
	for i := range spec.Prefetch {
		src := spec.Prefetch[i]
		if src.Leaf != nil {
			if src.Leaf.Space != "" {
				return ErrQueryMVLeafHasSpace
			}
			if err := validateMVLeafPayload(*src.Leaf); err != nil {
				return err
			}
			continue
		}
		if src.Spec != nil {
			if err := mvTreeValidateAt(src.Spec, depth+1); err != nil {
				return err
			}
		}
	}
	// Root checks AFTER the prefetch loop — matching the old flat order.
	if spec.Root.Space != "" {
		return ErrQueryMVLeafHasSpace
	}
	if spec.Mode == ModeRerank {
		if err := validateMVLeafPayload(spec.Root); err != nil {
			return err
		}
	}
	return nil
}

// QueryTreeLanes is the UNFUSED tree-lanes Query variant for an MV index whose spec
// contains a nested MULTI-lane FUSION node (SpecHasNestedFusion == true) — the
// MV-family analogue of (*Collection).QueryTreeLanes / (*NamedCollection).
// QueryTreeLanes. Instead of fusing each FUSION node into a single lane (Query →
// runQuerySpecAt), it returns the node-expanded UNFUSED lanes in the deterministic
// pre-order collectTreeLanesAt walks — so the coordinator (and the single-shard path)
// folds EVERY FUSION node over the cross-partition GLOBAL union ⇒ P>1==P1 EXACT at
// every level (all MV lanes are score-descending, the orientation-aware coordinator
// fold handles them). It uses the SAME GENERIC collectTreeLanesAt the dense path
// uses, with the MV locked closures (execMVLeafLocked / rerankMVLocked). CRUCIALLY
// the whole recursion runs under ONE m.mu.RLock acquisition with ONE clock snapshot
// (acquired here, BEFORE collectTreeLanesAt recurses — the closures capture `now`),
// exactly like Query, so every lane at every depth + the rerank re-score see a single
// consistent live/per-key-TTL view (no re-lock, no clock skew across depths). Query
// (the flat fold) is UNCHANGED — only the nested-multi-lane-FUSION wire path routes
// here.
func (m *MultiVectorIndex) QueryTreeLanes(spec QuerySpec) ([][]Result, error) {
	if spec.GroupBy != "" {
		return nil, ErrQueryGroupNotDense
	}
	// No-Space + per-leaf MV payload required at EVERY depth (same recursive validation
	// as Query — recommend/discover are MV-unsupported at any depth).
	if err := mvTreeValidate(&spec); err != nil {
		return nil, err
	}
	// One lock + one clock snapshot for the WHOLE nested recursion (mirror Query):
	// the closures below capture `now`, and collectTreeLanesAt threads them through
	// every depth, so no nested path re-acquires the lock or re-reads the clock.
	m.mu.RLock()
	defer m.mu.RUnlock()
	now := m.nowMs()
	execLeaf := func(leaf QueryLeaf, k int) ([]Result, error) {
		return m.execMVLeafLocked(leaf, k, now)
	}
	rerankRoot := func(root QueryLeaf, cands []uint64, k int) ([]Result, error) {
		return m.rerankMVLocked(root, cands, k, now)
	}
	return collectTreeLanesAt(spec, execLeaf, rerankRoot, 0)
}

// validateMVLeafPayload fails loud on an MV leaf whose payload does not match its
// kind: a MaxSim leaf with no tokens, an MV sparse leaf with an empty sparse
// query, or a leaf of an unknown/non-MV kind (a dense leaf in an MV query).
func validateMVLeafPayload(l QueryLeaf) error {
	switch l.Kind {
	case LeafMVMaxSim:
		if len(l.Tokens) == 0 {
			return ErrQueryMVMaxSimNoTokens
		}
		return nil
	case LeafSparse:
		if l.Sparse.IsZero() {
			return ErrQueryMVSparseEmpty
		}
		return nil
	case LeafRecommend:
		// Recommend has no single-vector pooling semantics for an MV token-set doc
		// (see ErrQueryMVRecommendUnsupported) — reject fail-loud, do NOT pool.
		return ErrQueryMVRecommendUnsupported
	case LeafDiscover:
		// Discover's per-candidate directional scorer needs one representative vector,
		// which an MV token-set doc lacks (see ErrQueryMVDiscoverUnsupported).
		return ErrQueryMVDiscoverUnsupported
	default:
		// A dense leaf (or unknown kind) is not a valid MV query node.
		return ErrQueryBadLeafKind
	}
}

// execMVLeafLocked runs a single prefetch leaf as a candidate lane under m.mu
// (read) with the shared clock snapshot. MaxSim leaf → maxSimSearchLockedNow over
// the token matrix (score-descending, converted MultiResult → Result 1:1 on
// Score); MV sparse leaf → the doc-level sparse inverted-index top-k gated by the
// shared live/per-key-TTL/filter admit rule (score-descending). The lane POOL is
// leafLanePool(leaf, k) — the SAME default the dense/named path and mvHybridK use
// (the leaf's LaneK, else its K, else max(k,50)) — so a 2-lane MV FUSION with
// default pools fuses byte-for-byte identically to the equivalent MVHybrid. Caller
// holds m.mu (read).
func (m *MultiVectorIndex) execMVLeafLocked(leaf QueryLeaf, k int, now int64) ([]Result, error) {
	pool := leafLanePool(leaf, k)
	switch leaf.Kind {
	case LeafMVMaxSim:
		for _, q := range leaf.Tokens {
			if len(q) != m.dim {
				return nil, ErrDimMismatch
			}
		}
		pred, err := CompileFilter(leaf.Filter)
		if err != nil {
			return nil, err
		}
		// candidatesPerToken 0 = the standard adaptive over-fetch (matching the MV
		// hybrid's MaxSim lane).
		mr, serr := m.maxSimSearchLockedNow(leaf.Tokens, pool, leaf.Filter, pred, 0, now)
		if serr != nil {
			return nil, serr
		}
		out := make([]Result, len(mr))
		for i, r := range mr {
			// MaxSim Score rides Result.Score (the lane is score-desc); Distance 0.
			out[i] = Result{ID: r.ID, Score: r.Score}
		}
		return out, nil
	case LeafSparse:
		sq := leaf.Sparse
		if verr := sq.Validate(); verr != nil {
			return nil, verr
		}
		pred, err := CompileFilter(leaf.Filter)
		if err != nil {
			return nil, err
		}
		return m.sparseIdx.searchTopK(&sq, pool, m.sparseAdmitLocked(pred, now)), nil
	default:
		return nil, ErrQueryBadLeafKind
	}
}

// rerankMVLocked re-scores the candidate id union by the MV root leaf, restricted
// to that set, returning the score-descending top-k — the MV analogue of
// (*Collection).rerankByRoot / (*NamedCollection).scoreByIDs. MaxSim root → exact
// brute-force maxSim over ONLY the candidate docs (the same view-based MaxSim path
// the engine uses, restricted to cands, with the live/per-key-
// TTL/filter re-check); sparse root → a restricted doc-sparse scan (the inverted
// index is exact, so a pool >= the candidate count surfaces every candidate;
// filter to the union, score-descending). Caller holds m.mu (read).
func (m *MultiVectorIndex) rerankMVLocked(root QueryLeaf, cands []uint64, k int, now int64) ([]Result, error) {
	if len(cands) == 0 {
		return nil, nil
	}
	switch root.Kind {
	case LeafMVMaxSim:
		return m.maxSimRerankLocked(root, cands, k, now)
	case LeafSparse:
		return m.sparseRerankLocked(root, cands, k, now)
	default:
		return nil, ErrQueryBadLeafKind
	}
}

// maxSimRerankLocked scores ONLY the candidate docs by exact MaxSim (no ANN
// gather) and returns the score-descending top-k. It mirrors the Stage-2 rerank
// of maxSimSearchLockedNow but over an EXPLICIT candidate id set: it normalizes
// the query tokens, applies the live/per-key-TTL/filter admit gate per candidate
// (one clock snapshot), and scores each by exact MaxSim against arena views
// (withVecAccess, no per-token-vector copy). Exact over the union. Caller
// holds m.mu (read).
func (m *MultiVectorIndex) maxSimRerankLocked(root QueryLeaf, cands []uint64, k int, now int64) ([]Result, error) {
	for _, q := range root.Tokens {
		if len(q) != m.dim {
			return nil, ErrDimMismatch
		}
	}
	pred, err := CompileFilter(root.Filter)
	if err != nil {
		return nil, err
	}
	// Normalize query tokens once (the stored doc vectors are unit-length, so the
	// per-pair similarity is a plain dot product) — identical to the engine path.
	norm := make([][]float32, len(root.Tokens))
	for i, q := range root.Tokens {
		nq := make([]float32, len(q))
		copy(nq, q)
		normalize(nq)
		norm[i] = nq
	}
	// Restrict to LIVE candidate docs (drop any tombstoned/never-present id in the
	// union) so vecsForIDs resolves only real token sets.
	live := make([]uint64, 0, len(cands))
	for _, doc := range cands {
		if _, ok := m.docTokens[doc]; ok {
			live = append(live, doc)
		}
	}
	if len(live) == 0 {
		return nil, nil
	}
	// Score MaxSim against arena VIEWS under one read lock (no per-token-vector copy),
	// mirroring maxSimSearchLockedNow's Stage-2: withVecAccess lends views, each doc's
	// token views materialize into a reused buffer, scored via the shared maxSimScore.
	// Byte-identical to the former vecsForIDs+maxSimFromMap path.
	scored := make([]Result, 0, len(live))
	m.idx.withVecAccess(func(get func(id uint64) ([]float32, bool)) {
		docVecs := make([][]float32, 0, 64) // reused across docs; views, not copies
		for _, doc := range live {
			meta := liveMetaMap(m.docMeta[doc], m.keyTTL[doc], now)
			if pred != nil && !pred(meta) {
				continue
			}
			docVecs = docVecs[:0]
			for _, tid := range m.docTokens[doc] {
				if v, ok := get(tid); ok {
					docVecs = append(docVecs, v)
				}
			}
			scored = append(scored, Result{ID: doc, Score: maxSimScore(norm, docVecs)})
		}
	})
	sort.SliceStable(scored, func(a, b int) bool {
		if scored[a].Score != scored[b].Score {
			return scored[a].Score > scored[b].Score
		}
		return scored[a].ID < scored[b].ID
	})
	if k > 0 && len(scored) > k {
		scored = scored[:k]
	}
	return scored, nil
}

// sparseRerankLocked re-scores the candidate ids against the doc-level sparse
// field and returns the score-descending top-k. It mirrors the dense/named sparse
// rerank arm: run the sparse lane with a pool large enough to surface every
// candidate the inverted index scores (the dot product is exact, not ANN, so a
// pool >= the live-doc count yields the exact restricted scores), then restrict to
// the candidate union and take top-k (ties → lower id). Caller holds m.mu (read).
func (m *MultiVectorIndex) sparseRerankLocked(root QueryLeaf, cands []uint64, k int, now int64) ([]Result, error) {
	sq := root.Sparse
	if verr := sq.Validate(); verr != nil {
		return nil, verr
	}
	pred, err := CompileFilter(root.Filter)
	if err != nil {
		return nil, err
	}
	pool := len(cands)
	if n := len(m.docTokens); n > pool {
		pool = n
	}
	res := m.sparseIdx.searchTopK(&sq, pool, m.sparseAdmitLocked(pred, now))
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
