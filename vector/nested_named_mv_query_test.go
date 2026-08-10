// SPDX-License-Identifier: Apache-2.0

package vector

import (
	"errors"
	"testing"
)

// --- NESTED prefetch (Query API recursion) for the NAMED + MV families ---
//
// These tests prove that lifting the v1 named/MV nested-prefetch rejection makes the
// generic runQuerySpec recursion driver "just work" for both families: a nested
// FUSION/RERANK sub-spec executes against the SAME index (named: per-space closures;
// MV: under the single lock+clock snapshot) and equals an independent oracle, and the
// UNFUSED tree-lanes path folds back to the single-node Query result exactly (the P=1
// witness of the P>1==P1 invariant the embedded fan-out tests assert).

// TestNamedNestedFusionVsOracle: a depth-2 named FUSION whose prefetch is [a dense
// "title" leaf, a NESTED 2-lane FUSION sub-spec over (image dense + terms sparse)].
// The nested multi-SPACE sub-spec's own fused top-k IS the parent's second lane. The
// oracle runs the sub-spec independently for its lane, runs the parent dense leaf as a
// lane, and folds the two via fuseLanes (lane0 dense distance-asc → Fuse start).
func TestNamedNestedFusionVsOracle(t *testing.T) {
	nc := newNamedQueryCorpus(t)
	titleQ := []float32{0.9, 0.1, 0, 0}
	imageQ := []float32{0.5, 0.5, 0}
	termsQ := sv([]uint32{2, 5}, []float32{2, 1})
	k := 4

	sub := QuerySpec{
		Mode: ModeFusion,
		Prefetch: srcs([]QueryLeaf{
			{Kind: LeafDense, Space: "image", Dense: imageQ},
			{Kind: LeafSparse, Space: "terms", Sparse: *termsQ, ScoreDesc: true},
		}...),
		Method: FusionRRF,
		K:      k,
	}
	parent := QuerySpec{
		Mode: ModeFusion,
		Prefetch: []QuerySource{
			LeafSource(QueryLeaf{Kind: LeafDense, Space: "title", Dense: titleQ}),
			{Spec: &sub},
		},
		Method: FusionRRF,
		K:      k,
	}
	got, err := nc.Query(parent)
	if err != nil {
		t.Fatalf("nested named Query: %v", err)
	}
	if got.Mode != ModeFusion || len(got.Lanes) != 2 {
		t.Fatalf("mode=%d lanes=%d, want fusion/2", got.Mode, len(got.Lanes))
	}

	subRes, err := nc.Query(sub)
	if err != nil {
		t.Fatalf("sub Query: %v", err)
	}
	denseLane, err := nc.execNamedLeaf(QueryLeaf{Kind: LeafDense, Space: "title", Dense: titleQ}, k)
	if err != nil {
		t.Fatalf("execNamedLeaf: %v", err)
	}
	oracle := fuseLanes([][]Result{denseLane, subRes.Fused}, false, FusionRRF, 0, 0, k)
	if !queryResultsEqual(got.Fused, oracle) {
		t.Errorf("named nested FUSION != oracle\n got=%+v\nwant=%+v", got.Fused, oracle)
	}
	if !queryResultsEqual(got.Lanes[1], subRes.Fused) {
		t.Errorf("parent lane[1] != sub fused")
	}
}

// TestNamedNestedFusionDepth3VsOracle: depth-3 named FUSION. The grandchild is a
// 2-lane FUSION over (image dense + terms sparse); the child wraps it with a title
// dense lane; the parent wraps the child with another title dense lane. Each level's
// fused top-k is the next level's lane — the oracle folds bottom-up independently.
func TestNamedNestedFusionDepth3VsOracle(t *testing.T) {
	nc := newNamedQueryCorpus(t)
	titleA := []float32{0.9, 0.1, 0, 0}
	titleB := []float32{0.2, 0.2, 0.2, 0.2}
	imageQ := []float32{0.5, 0.5, 0}
	termsQ := sv([]uint32{2, 5}, []float32{2, 1})
	k := 5

	grand := QuerySpec{
		Mode: ModeFusion,
		Prefetch: srcs([]QueryLeaf{
			{Kind: LeafDense, Space: "image", Dense: imageQ},
			{Kind: LeafSparse, Space: "terms", Sparse: *termsQ, ScoreDesc: true},
		}...),
		Method: FusionRRF,
		K:      k,
	}
	child := QuerySpec{
		Mode: ModeFusion,
		Prefetch: []QuerySource{
			LeafSource(QueryLeaf{Kind: LeafDense, Space: "title", Dense: titleB}),
			{Spec: &grand},
		},
		Method: FusionRRF,
		K:      k,
	}
	parent := QuerySpec{
		Mode: ModeFusion,
		Prefetch: []QuerySource{
			LeafSource(QueryLeaf{Kind: LeafDense, Space: "title", Dense: titleA}),
			{Spec: &child},
		},
		Method: FusionRRF,
		K:      k,
	}
	got, err := nc.Query(parent)
	if err != nil {
		t.Fatalf("depth-3 named Query: %v", err)
	}

	childRes, err := nc.Query(child)
	if err != nil {
		t.Fatalf("child Query: %v", err)
	}
	parentLane, err := nc.execNamedLeaf(QueryLeaf{Kind: LeafDense, Space: "title", Dense: titleA}, k)
	if err != nil {
		t.Fatalf("execNamedLeaf: %v", err)
	}
	oracle := fuseLanes([][]Result{parentLane, childRes.Fused}, false, FusionRRF, 0, 0, k)
	if !queryResultsEqual(got.Fused, oracle) {
		t.Errorf("named depth-3 FUSION != oracle\n got=%+v\nwant=%+v", got.Fused, oracle)
	}
}

// TestNamedNestedRerankVsOracle: a parent RERANK whose prefetch is a nested 2-space
// FUSION sub-spec, root = a dense "title" leaf. The candidate union is the sub-spec's
// fused ids; the root re-scores them. Oracle: run the sub-spec for its candidate ids,
// then scoreByIDs by the same root over that union.
func TestNamedNestedRerankVsOracle(t *testing.T) {
	nc := newNamedQueryCorpus(t)
	imageQ := []float32{0.5, 0.5, 0}
	termsQ := sv([]uint32{2, 5}, []float32{2, 1})
	rootQ := []float32{0.9, 0.1, 0, 0}
	k := 4

	sub := QuerySpec{
		Mode: ModeFusion,
		Prefetch: srcs([]QueryLeaf{
			{Kind: LeafDense, Space: "image", Dense: imageQ},
			{Kind: LeafSparse, Space: "terms", Sparse: *termsQ, ScoreDesc: true},
		}...),
		Method: FusionRRF,
		K:      k,
	}
	parent := QuerySpec{
		Mode: ModeRerank,
		Root: QueryLeaf{Kind: LeafDense, Space: "title", Dense: rootQ},
		Prefetch: []QuerySource{
			{Spec: &sub},
		},
		K: k,
	}
	got, err := nc.Query(parent)
	if err != nil {
		t.Fatalf("nested named RERANK: %v", err)
	}
	if got.Mode != ModeRerank {
		t.Fatalf("mode = %d, want rerank", got.Mode)
	}

	subRes, err := nc.Query(sub)
	if err != nil {
		t.Fatalf("sub Query: %v", err)
	}
	cands := unionCandidates([][]Result{subRes.Fused})
	oracle, err := nc.scoreByIDs(QueryLeaf{Kind: LeafDense, Space: "title", Dense: rootQ}, cands, k)
	if err != nil {
		t.Fatalf("scoreByIDs: %v", err)
	}
	if !queryResultsEqual(got.Fused, oracle) {
		t.Errorf("named nested RERANK != oracle\n got=%+v\nwant=%+v", got.Fused, oracle)
	}
}

// TestNamedQueryTreeLanesFoldEqualsQuery: the UNFUSED tree-lanes path for a nested
// multi-lane FUSION named spec, folded at a single partition, equals nc.Query(spec).
func TestNamedQueryTreeLanesFoldEqualsQuery(t *testing.T) {
	nc := newNamedQueryCorpus(t)
	titleQ := []float32{0.9, 0.1, 0, 0}
	imageQ := []float32{0.5, 0.5, 0}
	termsQ := sv([]uint32{2, 5}, []float32{2, 1})
	k := 4

	sub := QuerySpec{
		Mode: ModeFusion,
		Prefetch: srcs([]QueryLeaf{
			{Kind: LeafDense, Space: "image", Dense: imageQ},
			{Kind: LeafSparse, Space: "terms", Sparse: *termsQ, ScoreDesc: true},
		}...),
		Method: FusionRRF,
		K:      k,
	}
	parent := QuerySpec{
		Mode: ModeFusion,
		Prefetch: []QuerySource{
			LeafSource(QueryLeaf{Kind: LeafDense, Space: "title", Dense: titleQ}),
			{Spec: &sub},
		},
		Method: FusionRRF,
		K:      k,
	}
	if !SpecHasNestedFusion(parent) {
		t.Fatal("parent should be a nested multi-lane FUSION")
	}
	want, err := nc.Query(parent)
	if err != nil {
		t.Fatalf("Query: %v", err)
	}
	lanes, err := nc.QueryTreeLanes(parent)
	if err != nil {
		t.Fatalf("QueryTreeLanes: %v", err)
	}
	// Single-partition fold over the unfused lanes: rebuild the per-source lanes via the
	// SAME predicate collectTreeLanesAt used (a nested multi-lane FUSION source expands).
	// Here the tree is [leaf, sub(2 lanes)] → 3 unfused lanes; fold the sub's 2 lanes,
	// then fold [leafLane, subFused].
	if len(lanes) != 3 {
		t.Fatalf("tree lanes = %d, want 3", len(lanes))
	}
	subFused := fuseLanes([][]Result{lanes[1], lanes[2]}, false, FusionRRF, 0, 0, k)
	got := fuseLanes([][]Result{lanes[0], subFused}, false, FusionRRF, 0, 0, k)
	if !queryResultsEqual(got, want.Fused) {
		t.Errorf("named tree-lanes fold != Query\n got=%+v\nwant=%+v", got, want.Fused)
	}
}

// TestNamedNestedRejectionGone: a nested named spec used to be ErrQueryNestedNotSupported;
// it must now SUCCEED.
func TestNamedNestedRejectionGone(t *testing.T) {
	nc := newNamedQueryCorpus(t)
	sub := QuerySpec{
		Mode: ModeFusion,
		Prefetch: srcs([]QueryLeaf{
			{Kind: LeafDense, Space: "image", Dense: []float32{0.5, 0.5, 0}},
			{Kind: LeafSparse, Space: "terms", Sparse: *sv([]uint32{2, 5}, []float32{2, 1}), ScoreDesc: true},
		}...),
		Method: FusionRRF,
		K:      4,
	}
	spec := QuerySpec{
		Mode:     ModeFusion,
		Prefetch: []QuerySource{LeafSource(QueryLeaf{Kind: LeafDense, Space: "title", Dense: []float32{0.9, 0.1, 0, 0}}), {Spec: &sub}},
		Method:   FusionRRF,
		K:        4,
	}
	if _, err := nc.Query(spec); err != nil {
		if errors.Is(err, ErrQueryNestedNotSupported) {
			t.Fatal("named nested spec still rejected with ErrQueryNestedNotSupported")
		}
		t.Fatalf("named nested spec failed: %v", err)
	}
}

// TestNamedNestedLeafRequiresSpaceAtDepth: a nested sub-spec leaf missing its Space
// must fail loud (Space required at EVERY depth, recursion does not relax validation).
func TestNamedNestedLeafRequiresSpaceAtDepth(t *testing.T) {
	nc := newNamedQueryCorpus(t)
	sub := QuerySpec{
		Mode: ModeFusion,
		Prefetch: srcs([]QueryLeaf{
			{Kind: LeafDense, Space: "image", Dense: []float32{0.5, 0.5, 0}},
			{Kind: LeafDense, Dense: []float32{1, 0, 0, 0}}, // missing Space at depth 1
		}...),
		Method: FusionRRF,
		K:      4,
	}
	spec := QuerySpec{
		Mode:     ModeFusion,
		Prefetch: []QuerySource{LeafSource(QueryLeaf{Kind: LeafDense, Space: "title", Dense: []float32{0.9, 0.1, 0, 0}}), {Spec: &sub}},
		Method:   FusionRRF,
		K:        4,
	}
	if _, err := nc.Query(spec); !errors.Is(err, ErrQueryNamedLeafNoSpace) {
		t.Fatalf("nested Space-less leaf err = %v, want ErrQueryNamedLeafNoSpace", err)
	}
}

// TestNamedNestedDepthBound: a spec nested past MaxQueryDepthExec must fail loud.
func TestNamedNestedDepthBound(t *testing.T) {
	nc := newNamedQueryCorpus(t)
	// Build a chain deeper than MaxQueryDepthExec, every leaf carrying a Space (so the
	// failure is the DEPTH bound, not the Space check). Each level is a 2-lane FUSION
	// so SpecHasNestedFusion routing + recursion both fire.
	leafA := QueryLeaf{Kind: LeafDense, Space: "title", Dense: []float32{0.9, 0.1, 0, 0}}
	leafB := QueryLeaf{Kind: LeafDense, Space: "image", Dense: []float32{0.5, 0.5, 0}}
	cur := QuerySpec{Mode: ModeFusion, Prefetch: srcs(leafA, leafB), Method: FusionRRF, K: 3}
	for i := 0; i <= MaxQueryDepthExec+1; i++ {
		next := QuerySpec{
			Mode:     ModeFusion,
			Prefetch: []QuerySource{LeafSource(leafA), {Spec: &cur}},
			Method:   FusionRRF,
			K:        3,
		}
		cur = next
	}
	if _, err := nc.Query(cur); !errors.Is(err, ErrQuerySpecTooDeep) {
		t.Fatalf("over-deep named spec err = %v, want ErrQuerySpecTooDeep", err)
	}
}

// TestNamedFlatQueryUnaffected: lifting the rejection must not change the flat path —
// a flat 2-space FUSION still equals NamedHybrid (the existing unification proof,
// re-asserted post-lift).
func TestNamedFlatQueryUnaffected(t *testing.T) {
	nc := newNamedQueryCorpus(t)
	denseQ := []float32{0.9, 0.1, 0, 0}
	sparseQ := sv([]uint32{2, 5}, []float32{2, 1})
	k := 4
	spec := QuerySpec{
		Mode: ModeFusion,
		Prefetch: srcs([]QueryLeaf{
			{Kind: LeafDense, Space: "title", Dense: denseQ},
			{Kind: LeafSparse, Space: "terms", Sparse: *sparseQ},
		}...),
		Method: FusionRRF,
		K:      k,
	}
	if SpecHasNestedFusion(spec) {
		t.Fatal("flat spec must NOT be a nested multi-lane FUSION")
	}
	qr, err := nc.Query(spec)
	if err != nil {
		t.Fatalf("flat Query: %v", err)
	}
	want, err := nc.NamedHybrid("title", denseQ, "terms", sparseQ, k, HybridOpts{Method: FusionRRF})
	if err != nil {
		t.Fatalf("NamedHybrid: %v", err)
	}
	if !queryResultsEqual(qr.Fused, want) {
		t.Errorf("flat named FUSION changed after lift\n got=%+v\nwant=%+v", qr.Fused, want)
	}
}

// --- MV nested ---

// TestMVNestedFusionVsOracle: depth-2 MV FUSION whose prefetch is [a MaxSim leaf, a
// NESTED 2-lane FUSION over (MaxSim + doc-sparse)]. All lanes score-desc.
func TestMVNestedFusionVsOracle(t *testing.T) {
	m, _ := mvHybridFixture(t)
	parentQ := [][]float32{{1, 0, 0}}
	subQ := [][]float32{{0, 1, 0}, {0, 0, 1}}
	subSparse := mvSV(0, 1.0, 2, 1.0, 5, 1.0)
	k := 5

	sub := QuerySpec{
		Mode: ModeFusion,
		Prefetch: srcs([]QueryLeaf{
			{Kind: LeafMVMaxSim, Tokens: subQ, ScoreDesc: true},
			{Kind: LeafSparse, Sparse: *subSparse, ScoreDesc: true},
		}...),
		Method: FusionRRF,
		K:      k,
	}
	parent := QuerySpec{
		Mode: ModeFusion,
		Prefetch: []QuerySource{
			LeafSource(QueryLeaf{Kind: LeafMVMaxSim, Tokens: parentQ, ScoreDesc: true}),
			{Spec: &sub},
		},
		Method: FusionRRF,
		K:      k,
	}
	got, err := m.Query(parent)
	if err != nil {
		t.Fatalf("nested MV Query: %v", err)
	}
	if got.Mode != ModeFusion || len(got.Lanes) != 2 {
		t.Fatalf("mode=%d lanes=%d, want fusion/2", got.Mode, len(got.Lanes))
	}
	subRes, err := m.Query(sub)
	if err != nil {
		t.Fatalf("sub Query: %v", err)
	}
	// Parent lane0 (MaxSim) is score-desc → the fold starts with FuseScoreLanes. Build
	// the parent MaxSim lane via the LOCKED exec to mirror the engine's pool/now.
	parentLane := mvExecLaneOracle(t, m, QueryLeaf{Kind: LeafMVMaxSim, Tokens: parentQ, ScoreDesc: true}, k)
	oracle := fuseLanes([][]Result{parentLane, subRes.Fused}, true, FusionRRF, 0, 0, k)
	if !queryResultsEqual(got.Fused, oracle) {
		t.Errorf("MV nested FUSION != oracle\n got=%+v\nwant=%+v", got.Fused, oracle)
	}
	if !queryResultsEqual(got.Lanes[1], subRes.Fused) {
		t.Errorf("parent lane[1] != sub fused")
	}
}

// TestMVNestedRerankVsOracle: parent RERANK over a nested 2-lane FUSION sub-spec,
// root = a MaxSim leaf. The candidate union is the sub's fused ids; the root re-scores.
func TestMVNestedRerankVsOracle(t *testing.T) {
	m, _ := mvHybridFixture(t)
	subQ := [][]float32{{1, 0, 0}, {0, 1, 0}}
	subSparse := mvSV(0, 1.0, 2, 1.0, 5, 1.0)
	rootQ := [][]float32{{0, 1, 0}, {0, 0, 1}}
	k := 4

	sub := QuerySpec{
		Mode: ModeFusion,
		Prefetch: srcs([]QueryLeaf{
			{Kind: LeafMVMaxSim, Tokens: subQ, ScoreDesc: true},
			{Kind: LeafSparse, Sparse: *subSparse, ScoreDesc: true},
		}...),
		Method: FusionRRF,
		K:      k,
	}
	parent := QuerySpec{
		Mode:     ModeRerank,
		Root:     QueryLeaf{Kind: LeafMVMaxSim, Tokens: rootQ, ScoreDesc: true},
		Prefetch: []QuerySource{{Spec: &sub}},
		K:        k,
	}
	got, err := m.Query(parent)
	if err != nil {
		t.Fatalf("nested MV RERANK: %v", err)
	}
	if got.Mode != ModeRerank {
		t.Fatalf("mode = %d, want rerank", got.Mode)
	}
	subRes, err := m.Query(sub)
	if err != nil {
		t.Fatalf("sub Query: %v", err)
	}
	cands := unionCandidates([][]Result{subRes.Fused})
	oracle := mvRerankOracle(t, m, QueryLeaf{Kind: LeafMVMaxSim, Tokens: rootQ, ScoreDesc: true}, cands, k)
	if !queryResultsEqual(got.Fused, oracle) {
		t.Errorf("MV nested RERANK != oracle\n got=%+v\nwant=%+v", got.Fused, oracle)
	}
}

// TestMVQueryTreeLanesFoldEqualsQuery: the UNFUSED tree-lanes path for a nested MV
// multi-lane FUSION spec, folded at a single partition, equals m.Query(spec).
func TestMVQueryTreeLanesFoldEqualsQuery(t *testing.T) {
	m, _ := mvHybridFixture(t)
	parentQ := [][]float32{{1, 0, 0}}
	subQ := [][]float32{{0, 1, 0}, {0, 0, 1}}
	subSparse := mvSV(0, 1.0, 2, 1.0, 5, 1.0)
	k := 5

	sub := QuerySpec{
		Mode: ModeFusion,
		Prefetch: srcs([]QueryLeaf{
			{Kind: LeafMVMaxSim, Tokens: subQ, ScoreDesc: true},
			{Kind: LeafSparse, Sparse: *subSparse, ScoreDesc: true},
		}...),
		Method: FusionRRF,
		K:      k,
	}
	parent := QuerySpec{
		Mode: ModeFusion,
		Prefetch: []QuerySource{
			LeafSource(QueryLeaf{Kind: LeafMVMaxSim, Tokens: parentQ, ScoreDesc: true}),
			{Spec: &sub},
		},
		Method: FusionRRF,
		K:      k,
	}
	if !SpecHasNestedFusion(parent) {
		t.Fatal("parent should be a nested multi-lane FUSION")
	}
	want, err := m.Query(parent)
	if err != nil {
		t.Fatalf("Query: %v", err)
	}
	lanes, err := m.QueryTreeLanes(parent)
	if err != nil {
		t.Fatalf("QueryTreeLanes: %v", err)
	}
	if len(lanes) != 3 {
		t.Fatalf("tree lanes = %d, want 3", len(lanes))
	}
	// All MV lanes score-desc → folds start with FuseScoreLanes (orientation true).
	subFused := fuseLanes([][]Result{lanes[1], lanes[2]}, true, FusionRRF, 0, 0, k)
	got := fuseLanes([][]Result{lanes[0], subFused}, true, FusionRRF, 0, 0, k)
	if !queryResultsEqual(got, want.Fused) {
		t.Errorf("MV tree-lanes fold != Query\n got=%+v\nwant=%+v", got, want.Fused)
	}
}

// TestMVNestedRejectionGone: a nested MV spec used to be ErrQueryNestedNotSupported;
// it must now SUCCEED.
func TestMVNestedRejectionGone(t *testing.T) {
	m, _ := mvHybridFixture(t)
	sub := QuerySpec{
		Mode: ModeFusion,
		Prefetch: srcs([]QueryLeaf{
			{Kind: LeafMVMaxSim, Tokens: [][]float32{{0, 1, 0}}, ScoreDesc: true},
			{Kind: LeafSparse, Sparse: *mvSV(0, 1.0, 2, 1.0), ScoreDesc: true},
		}...),
		Method: FusionRRF,
		K:      4,
	}
	spec := QuerySpec{
		Mode:     ModeFusion,
		Prefetch: []QuerySource{LeafSource(QueryLeaf{Kind: LeafMVMaxSim, Tokens: [][]float32{{1, 0, 0}}, ScoreDesc: true}), {Spec: &sub}},
		Method:   FusionRRF,
		K:        4,
	}
	if _, err := m.Query(spec); err != nil {
		if errors.Is(err, ErrQueryNestedNotSupported) {
			t.Fatal("MV nested spec still rejected with ErrQueryNestedNotSupported")
		}
		t.Fatalf("MV nested spec failed: %v", err)
	}
}

// TestMVNestedLeafSpaceRejectedAtDepth: a nested MV sub-spec leaf carrying a Space
// must fail loud (no Space at ANY depth).
func TestMVNestedLeafSpaceRejectedAtDepth(t *testing.T) {
	m, _ := mvHybridFixture(t)
	sub := QuerySpec{
		Mode: ModeFusion,
		Prefetch: srcs([]QueryLeaf{
			{Kind: LeafMVMaxSim, Tokens: [][]float32{{0, 1, 0}}, ScoreDesc: true},
			{Kind: LeafSparse, Space: "bad", Sparse: *mvSV(0, 1.0), ScoreDesc: true}, // Space at depth 1
		}...),
		Method: FusionRRF,
		K:      4,
	}
	spec := QuerySpec{
		Mode:     ModeFusion,
		Prefetch: []QuerySource{LeafSource(QueryLeaf{Kind: LeafMVMaxSim, Tokens: [][]float32{{1, 0, 0}}, ScoreDesc: true}), {Spec: &sub}},
		Method:   FusionRRF,
		K:        4,
	}
	if _, err := m.Query(spec); !errors.Is(err, ErrQueryMVLeafHasSpace) {
		t.Fatalf("nested MV Space-bearing leaf err = %v, want ErrQueryMVLeafHasSpace", err)
	}
}

// TestMVNestedDepthBound: an over-deep MV spec must fail loud.
func TestMVNestedDepthBound(t *testing.T) {
	m, _ := mvHybridFixture(t)
	leafA := QueryLeaf{Kind: LeafMVMaxSim, Tokens: [][]float32{{1, 0, 0}}, ScoreDesc: true}
	leafB := QueryLeaf{Kind: LeafMVMaxSim, Tokens: [][]float32{{0, 1, 0}}, ScoreDesc: true}
	cur := QuerySpec{Mode: ModeFusion, Prefetch: srcs(leafA, leafB), Method: FusionRRF, K: 3}
	for i := 0; i <= MaxQueryDepthExec+1; i++ {
		cur = QuerySpec{
			Mode:     ModeFusion,
			Prefetch: []QuerySource{LeafSource(leafA), {Spec: &cur}},
			Method:   FusionRRF,
			K:        3,
		}
	}
	if _, err := m.Query(cur); !errors.Is(err, ErrQuerySpecTooDeep) {
		t.Fatalf("over-deep MV spec err = %v, want ErrQuerySpecTooDeep", err)
	}
}

// mvExecLaneOracle runs a single MV leaf as a lane via the same locked exec the engine
// uses (one lock + clock snapshot), so an oracle lane matches the engine's per-lane
// pool/TTL view.
func mvExecLaneOracle(t *testing.T, m *MultiVectorIndex, leaf QueryLeaf, k int) []Result {
	t.Helper()
	m.mu.RLock()
	defer m.mu.RUnlock()
	now := m.nowMs()
	lane, err := m.execMVLeafLocked(leaf, k, now)
	if err != nil {
		t.Fatalf("execMVLeafLocked: %v", err)
	}
	return lane
}

// mvRerankOracle re-scores a candidate union by the root leaf via the locked rerank.
func mvRerankOracle(t *testing.T, m *MultiVectorIndex, root QueryLeaf, cands []uint64, k int) []Result {
	t.Helper()
	m.mu.RLock()
	defer m.mu.RUnlock()
	now := m.nowMs()
	res, err := m.rerankMVLocked(root, cands, k, now)
	if err != nil {
		t.Fatalf("rerankMVLocked: %v", err)
	}
	return res
}

// TestMVValidationPrecedencePreserved confirms the flat MV validation error PRECEDENCE
// is byte-identical before and after the nested-support refactor (item #3 of the review).
// Old order: prefetch-leaf Space check → prefetch-leaf payload check → root Space check
// → root RERANK payload check. The new mvTreeValidateAt restores this by running the
// prefetch loop BEFORE the root checks.
//
// Multi-fault spec: leaf[0] carries a Space (prefetch error), leaf[1] has no tokens
// (payload error), root carries a Space (root error). The FIRST error returned must be
// the prefetch-level ErrQueryMVLeafHasSpace (leaf[0] Space), not the root Space error.
func TestMVValidationPrecedencePreserved(t *testing.T) {
	m, _ := mvHybridFixture(t)

	// A flat spec with TWO faults: prefetch leaf[0] has Space (first), root has Space (second).
	// Old code hit the prefetch loop first → ErrQueryMVLeafHasSpace from leaf[0].
	// New code must reproduce the same first error.
	spec := QuerySpec{
		Mode: ModeFusion,
		Prefetch: []QuerySource{
			LeafSource(QueryLeaf{Kind: LeafMVMaxSim, Space: "bad", Tokens: [][]float32{{1, 0, 0}}, ScoreDesc: true}), // Space on leaf
			LeafSource(QueryLeaf{Kind: LeafMVMaxSim, Tokens: [][]float32{{0, 1, 0}}, ScoreDesc: true}),
		},
		Root:   QueryLeaf{Space: "also_bad"}, // root Space fault too (must NOT be the first error)
		Method: FusionRRF,
		K:      4,
	}
	if _, err := m.Query(spec); !errors.Is(err, ErrQueryMVLeafHasSpace) {
		t.Fatalf("multi-fault MV spec first error = %v, want ErrQueryMVLeafHasSpace (prefetch-before-root order)", err)
	}

	// Second check: a spec where the prefetch leaf has a valid payload but an empty sparse
	// query (payload error) AND the root also carries a Space. The payload error must fire
	// before the root Space error (prefetch payload check precedes root Space check).
	spec2 := QuerySpec{
		Mode: ModeFusion,
		Prefetch: []QuerySource{
			LeafSource(QueryLeaf{Kind: LeafSparse, Sparse: SparseVector{}, ScoreDesc: true}), // empty sparse = ErrQueryMVSparseEmpty
			LeafSource(QueryLeaf{Kind: LeafMVMaxSim, Tokens: [][]float32{{0, 1, 0}}, ScoreDesc: true}),
		},
		Root:   QueryLeaf{Space: "root_bad"},
		Method: FusionRRF,
		K:      4,
	}
	if _, err := m.Query(spec2); !errors.Is(err, ErrQueryMVSparseEmpty) {
		t.Fatalf("multi-fault MV spec2 first error = %v, want ErrQueryMVSparseEmpty (prefetch payload before root Space)", err)
	}
}
