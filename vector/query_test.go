// SPDX-License-Identifier: Apache-2.0

package vector

import (
	"sort"
	"testing"
)

// newQueryCorpus builds a small Collection: docs 1-5 cluster near the dense
// origin with a weak shared sparse term; doc 100 is far in dense space but
// carries a strong sparse term 42. Mirrors newHybridCorpus but on a Collection
// (the Query API entry point).
func newQueryCorpus(t *testing.T) *Collection {
	t.Helper()
	c, err := NewCollection("q", Config{Dim: 4, Metric: L2, M: 16, EfConstruction: 200, EfSearch: 64, Seed: 1})
	if err != nil {
		t.Fatalf("NewCollection: %v", err)
	}
	for i := uint64(1); i <= 5; i++ {
		v := []float32{float32(i) * 0.01, 0, 0, 0}
		sv := &SparseVector{Indices: []uint32{1}, Values: []float32{0.1}}
		if err := c.Insert(i, v, 0, nil, sv); err != nil {
			t.Fatalf("Insert %d: %v", i, err)
		}
	}
	if err := c.Insert(100, []float32{9, 9, 9, 9}, 0, nil,
		&SparseVector{Indices: []uint32{42}, Values: []float32{10.0}}); err != nil {
		t.Fatalf("Insert 100: %v", err)
	}
	return c
}

// srcs wraps flat leaves as leaf QuerySources — the 1-level prefetch form. Test
// specs build their prefetch as leaf sources (byte/behaviour-identical to the
// pre-recursion []QueryLeaf prefetch), so the oracle assertions stay unchanged.
func srcs(leaves ...QueryLeaf) []QuerySource {
	out := make([]QuerySource, len(leaves))
	for i := range leaves {
		out[i] = LeafSource(leaves[i])
	}
	return out
}

func queryResultsEqual(a, b []Result) bool {
	if len(a) != len(b) {
		return false
	}
	for i := range a {
		if a[i].ID != b[i].ID || a[i].Distance != b[i].Distance || a[i].Score != b[i].Score {
			return false
		}
	}
	return true
}

// TestQueryFusionEqualsHybridSearch is the unification proof: a FUSION query with
// a dense prefetch lane + a sparse prefetch lane, fused locally, MUST equal the
// equivalent HybridSearch over the same dense+sparse inputs (same method, k,
// lane pools). Tests all three fusion methods.
func TestQueryFusionEqualsHybridSearch(t *testing.T) {
	dense := []float32{0, 0, 0, 0}
	sparse := SparseVector{Indices: []uint32{1, 42}, Values: []float32{0.2, 5.0}}
	k := 4

	for _, tc := range []struct {
		name   string
		method FusionMethod
		alpha  float64
		rrfK   int
	}{
		{"rrf", FusionRRF, 0, 0},
		{"weighted", FusionWeighted, 0.5, 0},
		{"dbsf", FusionDBSF, 0.5, 0},
	} {
		t.Run(tc.name, func(t *testing.T) {
			c := newQueryCorpus(t)
			// laneK must match HybridSearch's default pools (max(k,50)) for an
			// exact cross-check, so leave LaneK=0 (engine default) on both leaves.
			spec := QuerySpec{
				Mode: ModeFusion,
				Prefetch: srcs([]QueryLeaf{
					{Kind: LeafDense, Dense: dense},
					{Kind: LeafSparse, Sparse: sparse},
				}...),
				Method: tc.method,
				Alpha:  tc.alpha,
				RRFK:   tc.rrfK,
				K:      k,
			}
			qr, err := c.Query(spec)
			if err != nil {
				t.Fatalf("Query: %v", err)
			}
			if qr.Mode != ModeFusion {
				t.Fatalf("mode = %d, want fusion", qr.Mode)
			}
			if len(qr.Lanes) != 2 {
				t.Fatalf("lanes = %d, want 2", len(qr.Lanes))
			}

			want, err := c.HybridSearch(dense, sparse, k, HybridOpts{Method: tc.method, Alpha: tc.alpha, RRFK: tc.rrfK})
			if err != nil {
				t.Fatalf("HybridSearch: %v", err)
			}
			if !queryResultsEqual(qr.Fused, want) {
				t.Errorf("FUSION fused != HybridSearch\n got=%+v\nwant=%+v", qr.Fused, want)
			}
		})
	}
}

// TestQueryRerankVsBruteForceOracle proves RERANK exactness: a dense+sparse
// prefetch produces a candidate union; the dense root re-scores that union; the
// result must equal a brute-force exact dense distance over the SAME union,
// top-k.
func TestQueryRerankVsBruteForceOracle(t *testing.T) {
	c := newQueryCorpus(t)
	denseRoot := []float32{0, 0, 0, 0}
	sparse := SparseVector{Indices: []uint32{42}, Values: []float32{5.0}}
	k := 3

	spec := QuerySpec{
		Mode: ModeRerank,
		Root: QueryLeaf{Kind: LeafDense, Dense: denseRoot, K: k},
		Prefetch: srcs([]QueryLeaf{
			{Kind: LeafDense, Dense: denseRoot, K: 4},
			{Kind: LeafSparse, Sparse: sparse, K: 4},
		}...),
		K: k,
	}
	qr, err := c.Query(spec)
	if err != nil {
		t.Fatalf("Query: %v", err)
	}
	if qr.Mode != ModeRerank {
		t.Fatalf("mode = %d, want rerank", qr.Mode)
	}

	// Oracle: recompute the prefetch union, brute-force the exact L2 distance from
	// denseRoot for every union id, take the k closest.
	denseLane, err := c.SearchFiltered(denseRoot, 4, Filter{})
	if err != nil {
		t.Fatal(err)
	}
	_, sparseLane, err := c.HybridLanes(nil, sparse, 4, HybridOpts{SparseK: 4})
	if err != nil {
		t.Fatal(err)
	}
	union := unionCandidates([][]Result{denseLane, sparseLane})
	type idDist struct {
		id   uint64
		dist float32
	}
	oracle := make([]idDist, 0, len(union))
	for _, id := range union {
		vec, _, _, _, _, ok := c.Get(id)
		if !ok {
			t.Fatalf("union id %d not found", id)
		}
		var d float32
		for i := range vec {
			diff := vec[i] - denseRoot[i]
			d += diff * diff
		}
		oracle = append(oracle, idDist{id: id, dist: d})
	}
	sort.SliceStable(oracle, func(a, b int) bool {
		if oracle[a].dist != oracle[b].dist {
			return oracle[a].dist < oracle[b].dist
		}
		return oracle[a].id < oracle[b].id
	})
	if len(oracle) > k {
		oracle = oracle[:k]
	}

	if len(qr.Fused) != len(oracle) {
		t.Fatalf("rerank len %d != oracle len %d", len(qr.Fused), len(oracle))
	}
	for i := range oracle {
		if qr.Fused[i].ID != oracle[i].id {
			t.Errorf("rerank[%d] id=%d, oracle id=%d (dist %v)", i, qr.Fused[i].ID, oracle[i].id, oracle[i].dist)
		}
	}
}

// TestQueryRerankSparseRoot exercises a sparse root re-scoring the prefetch
// union: only ids in the union with a nonzero sparse overlap survive, ranked by
// the sparse score.
func TestQueryRerankSparseRoot(t *testing.T) {
	c := newQueryCorpus(t)
	denseRoot := []float32{0, 0, 0, 0}
	sparseRoot := SparseVector{Indices: []uint32{42}, Values: []float32{5.0}}
	k := 3

	spec := QuerySpec{
		Mode: ModeRerank,
		Root: QueryLeaf{Kind: LeafSparse, Sparse: sparseRoot, K: k},
		Prefetch: srcs([]QueryLeaf{
			{Kind: LeafDense, Dense: denseRoot, K: 6},
			{Kind: LeafSparse, Sparse: sparseRoot, K: 6},
		}...),
		K: k,
	}
	qr, err := c.Query(spec)
	if err != nil {
		t.Fatalf("Query: %v", err)
	}
	// Doc 100 is the only doc carrying term 42, so it must rank first.
	if len(qr.Fused) == 0 || qr.Fused[0].ID != 100 {
		t.Fatalf("sparse-root rerank top should be doc 100; got %+v", qr.Fused)
	}
	// Every returned id must be in the prefetch union (restricted scoring).
	for _, r := range qr.Fused {
		if r.Score <= 0 {
			t.Errorf("sparse rerank result %d has non-positive score %v", r.ID, r.Score)
		}
	}
}

// TestQuerySourceLeafEqualsOldLeafPath proves the 1-level back-compat at the
// engine edge: a spec built with LeafSource(leaf) prefetch sources executes
// byte/behaviour-identically to the same flat leaves — the QuerySource{leaf}
// variant IS the unchanged leaf path. (Construction via srcs() == LeafSource per
// element, so this asserts the leaf-source FUSION result equals the equivalent
// HybridSearch, the same oracle the flat path uses.)
func TestQuerySourceLeafEqualsOldLeafPath(t *testing.T) {
	c := newQueryCorpus(t)
	dense := []float32{0, 0, 0, 0}
	sparse := SparseVector{Indices: []uint32{1, 42}, Values: []float32{0.2, 5.0}}
	k := 4

	spec := QuerySpec{
		Mode: ModeFusion,
		Prefetch: []QuerySource{
			LeafSource(QueryLeaf{Kind: LeafDense, Dense: dense}),
			LeafSource(QueryLeaf{Kind: LeafSparse, Sparse: sparse, ScoreDesc: true}),
		},
		Method: FusionRRF,
		K:      k,
	}
	qr, err := c.Query(spec)
	if err != nil {
		t.Fatalf("Query: %v", err)
	}
	want, err := c.HybridSearch(dense, sparse, k, HybridOpts{Method: FusionRRF})
	if err != nil {
		t.Fatalf("HybridSearch: %v", err)
	}
	if !queryResultsEqual(qr.Fused, want) {
		t.Errorf("leaf-source FUSION != HybridSearch\n got=%+v\nwant=%+v", qr.Fused, want)
	}
}

// TestQueryNestedFusionVsOracle proves the nested-FUSION recursion (single-node):
// a parent FUSION spec whose prefetch is [a dense leaf, a NESTED 2-lane FUSION
// sub-spec] must fuse the sub-spec's own fused top-k as one parent lane. The oracle
// runs the sub-spec INDEPENDENTLY (a separate (*Collection).Query) to get its fused
// lane, runs the parent dense leaf as its own lane, then folds the two lanes with the
// SAME fuseLanes the engine uses (lane0 = the dense leaf, distance-asc → Fuse start).
// The recursion result must equal that hand-computed fold byte-for-byte.
func TestQueryNestedFusionVsOracle(t *testing.T) {
	c := newQueryCorpus(t)
	dense := []float32{0, 0, 0, 0}
	subDense := []float32{0.02, 0, 0, 0}
	subSparse := SparseVector{Indices: []uint32{1, 42}, Values: []float32{0.2, 5.0}}
	k := 4

	// The nested sub-spec: a 2-lane (dense+sparse) FUSION.
	subSpec := QuerySpec{
		Mode: ModeFusion,
		Prefetch: []QuerySource{
			LeafSource(QueryLeaf{Kind: LeafDense, Dense: subDense}),
			LeafSource(QueryLeaf{Kind: LeafSparse, Sparse: subSparse, ScoreDesc: true}),
		},
		Method: FusionRRF,
		K:      k,
	}
	// The parent: [a dense leaf, the nested sub-spec], FUSION.
	parent := QuerySpec{
		Mode: ModeFusion,
		Prefetch: []QuerySource{
			LeafSource(QueryLeaf{Kind: LeafDense, Dense: dense}),
			{Spec: &subSpec},
		},
		Method: FusionRRF,
		K:      k,
	}
	got, err := c.Query(parent)
	if err != nil {
		t.Fatalf("nested Query: %v", err)
	}
	if got.Mode != ModeFusion {
		t.Fatalf("mode = %d, want fusion", got.Mode)
	}
	if len(got.Lanes) != 2 {
		t.Fatalf("parent lanes = %d, want 2", len(got.Lanes))
	}

	// Oracle: run the sub-spec independently for its fused lane; run the parent dense
	// leaf as its own lane; fold [denseLane, subFused] via fuseLanes (lane0 dense,
	// distance-asc → Fuse start) at the parent k.
	subRes, err := c.Query(subSpec)
	if err != nil {
		t.Fatalf("sub Query: %v", err)
	}
	denseLane, err := c.execLeaf(QueryLeaf{Kind: LeafDense, Dense: dense}, k)
	if err != nil {
		t.Fatalf("execLeaf: %v", err)
	}
	oracle := fuseLanes([][]Result{denseLane, subRes.Fused}, false, FusionRRF, 0, 0, k)
	if !queryResultsEqual(got.Fused, oracle) {
		t.Errorf("nested FUSION != oracle\n got=%+v\nwant=%+v", got.Fused, oracle)
	}
	// The sub-spec lane the engine carried (parent Lanes[1]) must equal the sub-spec's
	// independent fused result (the recursion lane IS the sub-spec's fused top-k).
	if !queryResultsEqual(got.Lanes[1], subRes.Fused) {
		t.Errorf("parent lane[1] != sub fused\n got=%+v\nwant=%+v", got.Lanes[1], subRes.Fused)
	}
}

// TestQueryTreeLanesFoldEqualsQuery proves the UNFUSED tree-lanes path reproduces the
// single-node Query fold EXACTLY: for a nested MULTI-lane FUSION spec, QueryTreeLanes
// returns the per-source UNFUSED lanes in the pre-order collectTreeLanesAt walks, and a
// recursive single-partition fold over those lanes (the coordinator's mergeTreeFusionNode
// with a single partition) equals c.Query(spec).Fused. This is the in-package witness
// that the tree-lanes ship-unfused + re-fold round-trip is loss-less at P=1 — the
// foundation of the P>1==P1 invariant the embedded fan-out tests assert.
func TestQueryTreeLanesFoldEqualsQuery(t *testing.T) {
	c := newQueryCorpus(t)
	dense := []float32{0, 0, 0, 0}
	subDense := []float32{0.02, 0, 0, 0}
	subSparse := SparseVector{Indices: []uint32{1, 42}, Values: []float32{0.2, 5.0}}
	k := 4

	subSpec := QuerySpec{
		Mode: ModeFusion,
		Prefetch: []QuerySource{
			LeafSource(QueryLeaf{Kind: LeafDense, Dense: subDense}),
			LeafSource(QueryLeaf{Kind: LeafSparse, Sparse: subSparse, ScoreDesc: true}),
		},
		Method: FusionRRF,
		K:      k,
	}
	parent := QuerySpec{
		Mode: ModeFusion,
		Prefetch: []QuerySource{
			LeafSource(QueryLeaf{Kind: LeafDense, Dense: dense}),
			{Spec: &subSpec},
		},
		Method: FusionRRF,
		K:      k,
	}
	if !SpecHasNestedFusion(parent) {
		t.Fatal("parent should have a nested multi-lane FUSION node")
	}

	want, err := c.Query(parent)
	if err != nil {
		t.Fatalf("Query: %v", err)
	}
	lanes, err := c.QueryTreeLanes(parent)
	if err != nil {
		t.Fatalf("QueryTreeLanes: %v", err)
	}
	cursor := 0
	got := foldTreeLanesSingle(parent, lanes, &cursor)
	if cursor != len(lanes) {
		t.Fatalf("consumed %d of %d tree-lanes (traversal desync)", cursor, len(lanes))
	}
	if !queryResultsEqual(got, want.Fused) {
		t.Errorf("tree-lanes fold != Query\n got=%+v\nwant=%+v", got, want.Fused)
	}
}

// foldTreeLanesSingle is the SINGLE-PARTITION witness of the coordinator's recursive
// tree fold (rostam.mergeTreeFusionNode with one partition, no cross-partition union):
// it re-walks the spec consuming the flat pre-order lanes in the SAME order the partition
// emitted them and folds each FUSION node via fuseLanes — the exact recursion the
// coordinator runs. Used only by TestQueryTreeLanesFoldEqualsQuery.
func foldTreeLanesSingle(spec QuerySpec, lanes [][]Result, cursor *int) []Result {
	nodeK := spec.K
	if nodeK <= 0 {
		nodeK = 10
	}
	srcLanes := make([][]Result, len(spec.Prefetch))
	for i := range spec.Prefetch {
		src := spec.Prefetch[i]
		if src.Spec != nil && src.Spec.Mode == ModeFusion && len(src.Spec.Prefetch) >= 2 {
			srcLanes[i] = foldTreeLanesSingle(*src.Spec, lanes, cursor)
			continue
		}
		srcLanes[i] = lanes[*cursor]
		*cursor++
	}
	return fuseLanes(srcLanes, SourceOrientation(spec.Prefetch[0]), spec.Method, spec.Alpha, spec.RRFK, nodeK)
}

// TestQueryNestedRerankVsOracle proves the nested-RERANK recursion (single-node): a
// parent FUSION spec whose prefetch is [a sparse leaf, a NESTED RERANK sub-spec]
// fuses the sub-spec's reranked top-k as one parent lane. The oracle runs the
// sub-spec (a RERANK) independently and folds its result with the parent sparse lane.
func TestQueryNestedRerankVsOracle(t *testing.T) {
	c := newQueryCorpus(t)
	denseRoot := []float32{0, 0, 0, 0}
	sparse := SparseVector{Indices: []uint32{1, 42}, Values: []float32{0.2, 5.0}}
	k := 3

	// The nested sub-spec: a RERANK (dense+sparse prefetch → dense root rerank).
	subSpec := QuerySpec{
		Mode: ModeRerank,
		Root: QueryLeaf{Kind: LeafDense, Dense: denseRoot, K: k},
		Prefetch: []QuerySource{
			LeafSource(QueryLeaf{Kind: LeafDense, Dense: denseRoot, K: 4}),
			LeafSource(QueryLeaf{Kind: LeafSparse, Sparse: sparse, K: 4, ScoreDesc: true}),
		},
		K: k,
	}
	parent := QuerySpec{
		Mode: ModeFusion,
		Prefetch: []QuerySource{
			LeafSource(QueryLeaf{Kind: LeafSparse, Sparse: sparse, ScoreDesc: true}),
			{Spec: &subSpec},
		},
		Method: FusionRRF,
		K:      k,
	}
	got, err := c.Query(parent)
	if err != nil {
		t.Fatalf("nested rerank Query: %v", err)
	}
	subRes, err := c.Query(subSpec)
	if err != nil {
		t.Fatalf("sub rerank Query: %v", err)
	}
	sparseLane, err := c.execLeaf(QueryLeaf{Kind: LeafSparse, Sparse: sparse, ScoreDesc: true}, k)
	if err != nil {
		t.Fatalf("execLeaf: %v", err)
	}
	// lane0 = sparse (score-desc) → fuseLanes starts with FuseScoreLanes; the sub-spec
	// RERANK fused result is a dense rerank (distance-asc orientation: specOrientation
	// returns spec.Root.ScoreDesc == false). The engine carried laneOrient[1]=false.
	oracle := fuseLanes([][]Result{sparseLane, subRes.Fused}, true, FusionRRF, 0, 0, k)
	if !queryResultsEqual(got.Fused, oracle) {
		t.Errorf("nested RERANK != oracle\n got=%+v\nwant=%+v", got.Fused, oracle)
	}
}

// TestSpecOrientation locks the specOrientation contract (the per-source orientation
// threaded into the parent fold + the fan-out per-lane truncation): a RERANK sub-spec
// → its root orientation; a 2+-lane FUSION sub-spec → score-desc; a 1-lane FUSION
// sub-spec → its single lane's orientation.
func TestSpecOrientation(t *testing.T) {
	denseLeafL := QueryLeaf{Kind: LeafDense, Dense: []float32{0, 0, 0, 0}} // distance-asc
	sparseLeafL := QueryLeaf{Kind: LeafSparse, Sparse: SparseVector{Indices: []uint32{1}, Values: []float32{1}}, ScoreDesc: true}

	// RERANK sub-spec with a dense root → distance-asc.
	rerankDenseRoot := QuerySpec{Mode: ModeRerank, Root: denseLeafL, Prefetch: srcs(denseLeafL), K: 3}
	if specOrientation(rerankDenseRoot) {
		t.Error("RERANK dense-root orientation = score-desc, want distance-asc")
	}
	// RERANK sub-spec with a sparse root → score-desc.
	rerankSparseRoot := QuerySpec{Mode: ModeRerank, Root: sparseLeafL, Prefetch: srcs(denseLeafL), K: 3}
	if !specOrientation(rerankSparseRoot) {
		t.Error("RERANK sparse-root orientation = distance-asc, want score-desc")
	}
	// 2-lane FUSION → score-desc (the fold produces a score).
	fusion2 := QuerySpec{Mode: ModeFusion, Prefetch: srcs(denseLeafL, sparseLeafL), K: 3}
	if !specOrientation(fusion2) {
		t.Error("2-lane FUSION orientation = distance-asc, want score-desc")
	}
	// 1-lane FUSION (dense) → the single lane's orientation (distance-asc).
	fusion1dense := QuerySpec{Mode: ModeFusion, Prefetch: srcs(denseLeafL), K: 3}
	if specOrientation(fusion1dense) {
		t.Error("1-lane dense FUSION orientation = score-desc, want distance-asc")
	}
	// 1-lane FUSION (sparse) → score-desc.
	fusion1sparse := QuerySpec{Mode: ModeFusion, Prefetch: srcs(sparseLeafL), K: 3}
	if !specOrientation(fusion1sparse) {
		t.Error("1-lane sparse FUSION orientation = distance-asc, want score-desc")
	}
}

// TestQueryNestedDepthBound proves the DEFENSIVE execution-side depth check: an
// in-memory spec nested deeper than MaxQueryDepthExec (bypassing the decode bound)
// is rejected fail-loud (ErrQuerySpecTooDeep) — a malformed spec cannot infinite-
// recurse. The decode bound is exercised separately in ops.
func TestQueryNestedDepthBound(t *testing.T) {
	c := newQueryCorpus(t)
	leaf := QueryLeaf{Kind: LeafDense, Dense: []float32{0, 0, 0, 0}}
	// Build a CHAIN (not a cycle) deeper than MaxQueryDepthExec: each level wraps the
	// previous in a single nested FUSION source. Each iteration takes &prev where prev
	// is a freshly-allocated copy (a new heap address per level), so the tree is a
	// finite chain of depth MaxQueryDepthExec+2.
	cur := &QuerySpec{Mode: ModeFusion, Prefetch: srcs(leaf), K: 3}
	for d := 0; d <= MaxQueryDepthExec+1; d++ {
		prev := cur
		cur = &QuerySpec{Mode: ModeFusion, Prefetch: []QuerySource{{Spec: prev}}, K: 3}
	}
	if _, err := c.Query(*cur); err != ErrQuerySpecTooDeep {
		t.Fatalf("over-deep spec: err = %v, want ErrQuerySpecTooDeep", err)
	}
}

// TestQueryNestedRecommendDiscoverTreeWalk proves the recommend/discover pre-pass
// TREE-WALK: a nested sub-spec containing BOTH a recommend leaf AND a discover leaf
// (each carrying UNRESOLVED ids) must have both resolved/derived/embedded by the
// single-node pre-pass before runQuerySpec executes the tree — exactly as if the
// sub-spec ran on its own. The oracle runs the sub-spec INDEPENDENTLY (its own
// pre-pass) and the parent's nested lane must equal it (the tree-walk recursed into
// the sub-spec and resolved both nested leaves).
func TestQueryNestedRecommendDiscoverTreeWalk(t *testing.T) {
	c := newDiscoverQueryCorpus(t, Cosine)
	k := 3

	// The nested sub-spec: a 2-lane FUSION whose lanes are a RECOMMEND leaf (positive
	// id 1 → cluster A) and a DISCOVER leaf (target id 1, context pair 1>4 → steer to
	// cluster A), both id-form (unresolved) — so it exercises BOTH pre-passes nested.
	makeSub := func() QuerySpec {
		return QuerySpec{
			Mode: ModeFusion,
			Prefetch: []QuerySource{
				LeafSource(QueryLeaf{Kind: LeafRecommend, Positive: []uint64{1}, K: k}),
				LeafSource(QueryLeaf{
					Kind:               LeafDiscover,
					DiscoverTargetID:   []uint64{1},
					DiscoverContextIDs: []ContextPair{{Positive: 1, Negative: 4}},
					K:                  k,
					ScoreDesc:          true,
				}),
			},
			Method: FusionRRF,
			K:      k,
		}
	}
	// Parent: [a dense leaf, the nested recommend+discover sub-spec], FUSION.
	subForParent := makeSub()
	parent := QuerySpec{
		Mode: ModeFusion,
		Prefetch: []QuerySource{
			LeafSource(QueryLeaf{Kind: LeafDense, Dense: []float32{0.7, 0.7}}),
			{Spec: &subForParent},
		},
		Method: FusionRRF,
		K:      k,
	}
	got, err := c.Query(parent)
	if err != nil {
		t.Fatalf("nested recommend+discover Query: %v", err)
	}
	if len(got.Lanes) != 2 {
		t.Fatalf("parent lanes = %d, want 2", len(got.Lanes))
	}
	// The nested lane (Lanes[1] = the sub-spec's fused recommend⊕discover) must be
	// non-empty — proving the tree-walk RESOLVED both nested leaves (an un-resolved
	// recommend reaching runQuerySpec is rejected ErrQueryBadLeafKind; an un-resolved
	// discover has no context and fails — so a non-error non-empty lane proves both
	// pre-passes recursed into the sub-spec).
	nested := got.Lanes[1]
	if len(nested) == 0 {
		t.Fatal("nested recommend+discover lane is empty — the pre-pass tree-walk did not resolve")
	}
	// Both nested signals steer to cluster A (ids 1,2,3); id 1 is the recommend example
	// (excluded). The lane must be dominated by cluster-A neighbors 2,3 and must NOT be
	// led by the cluster-B discover negative id 4.
	if nested[0].ID == 4 {
		t.Errorf("nested lane top is the cluster-B negative id 4: %+v", nested)
	}
	for _, r := range nested {
		if r.ID == 1 {
			t.Errorf("nested lane contains the excluded recommend example id 1: %+v", nested)
		}
	}
	// Cross-check: the independent sub-spec (its own pre-pass) produces the SAME top-2
	// resolved cluster-A neighbors, confirming the tree-walk derive/embed matches the
	// standalone pre-pass on the resolved leaves.
	subRes, err := c.Query(makeSub())
	if err != nil {
		t.Fatalf("sub Query: %v", err)
	}
	if len(subRes.Fused) < 2 || subRes.Fused[0].ID != nested[0].ID || subRes.Fused[1].ID != nested[1].ID {
		t.Errorf("nested top-2 %v != independent sub top-2 %v", resultIDs(nested), resultIDs(subRes.Fused))
	}
}

// TestQueryValidation covers the fail-loud paths: no prefetch, unknown mode,
// unknown leaf kind, rerank with an empty root.
func TestQueryValidation(t *testing.T) {
	c := newQueryCorpus(t)
	dense := []float32{0, 0, 0, 0}

	if _, err := c.Query(QuerySpec{Mode: ModeFusion}); err != ErrQueryNoPrefetch {
		t.Errorf("no prefetch: err=%v, want ErrQueryNoPrefetch", err)
	}
	if _, err := c.Query(QuerySpec{
		Mode:     QueryMode(99),
		Prefetch: srcs([]QueryLeaf{{Kind: LeafDense, Dense: dense, K: 3}}...),
	}); err != ErrQueryBadMode {
		t.Errorf("bad mode: err=%v, want ErrQueryBadMode", err)
	}
	if _, err := c.Query(QuerySpec{
		Mode:     ModeFusion,
		Prefetch: srcs([]QueryLeaf{{Kind: LeafKind(7), Dense: dense, K: 3}}...),
	}); err != ErrQueryBadLeafKind {
		t.Errorf("bad leaf kind: err=%v, want ErrQueryBadLeafKind", err)
	}
	if _, err := c.Query(QuerySpec{
		Mode:     ModeRerank,
		Root:     QueryLeaf{Kind: LeafDense}, // empty dense
		Prefetch: srcs([]QueryLeaf{{Kind: LeafDense, Dense: dense, K: 3}}...),
	}); err != ErrQueryRerankNoRoot {
		t.Errorf("rerank empty root: err=%v, want ErrQueryRerankNoRoot", err)
	}
}
