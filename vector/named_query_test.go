// SPDX-License-Identifier: Apache-2.0

package vector

import (
	"sort"
	"testing"
)

// newNamedQueryCorpus builds a NamedCollection with THREE spaces: "title" (dense
// dim4 cosine), "image" (dense dim3 dot), "terms" (sparse) — the namedSparseTestConfig
// layout. It inserts 6 points across the spaces so a multi-space query has real
// per-space rankings to fuse / rerank.
func newNamedQueryCorpus(t *testing.T) *NamedCollection {
	t.Helper()
	nc, err := NewNamedCollection("default/nq", namedSparseTestConfig())
	if err != nil {
		t.Fatalf("new named: %v", err)
	}
	t.Cleanup(func() { nc.Close() })

	pts := []struct {
		id    uint64
		title []float32
		image []float32
		terms *SparseVector
	}{
		{1, []float32{1, 0, 0, 0}, []float32{1, 0, 0}, sv([]uint32{0, 2}, []float32{1, 0.5})},
		{2, []float32{0, 1, 0, 0}, []float32{0, 1, 0}, sv([]uint32{2, 5}, []float32{2, 1})},
		{3, []float32{0, 0, 1, 0}, []float32{0, 0, 1}, sv([]uint32{5}, []float32{4})},
		{4, []float32{0.9, 0.1, 0, 0}, []float32{0.5, 0.5, 0}, sv([]uint32{0, 5}, []float32{1, 3})},
		{5, []float32{0, 0, 0, 1}, []float32{1, 1, 1}, sv([]uint32{9}, []float32{5})},
		{6, []float32{0.2, 0.2, 0.2, 0.2}, []float32{0.3, 0.3, 0.3}, sv([]uint32{2, 9}, []float32{1, 1})},
	}
	for _, p := range pts {
		dense := map[string][]float32{"title": p.title, "image": p.image}
		sparse := map[string]*SparseVector{"terms": p.terms}
		if err := nc.InsertSparse(p.id, dense, sparse, Metadata{"k": NewInt(int64(p.id))}, 0); err != nil {
			t.Fatalf("insert %d: %v", p.id, err)
		}
	}
	return nc
}

// TestNamedQueryFusionEqualsNamedHybrid is the unification proof: a 2-space named
// FUSION (a dense "title" lane + a sparse "terms" lane) fused locally MUST equal
// the equivalent NamedHybrid over the same two spaces (same method, k, default
// lane pools). Covers all three fusion methods.
func TestNamedQueryFusionEqualsNamedHybrid(t *testing.T) {
	denseQ := []float32{0.9, 0.1, 0, 0}
	sparseQ := sv([]uint32{2, 5}, []float32{2, 1})
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
			nc := newNamedQueryCorpus(t)
			spec := QuerySpec{
				Mode: ModeFusion,
				Prefetch: srcs([]QueryLeaf{
					{Kind: LeafDense, Space: "title", Dense: denseQ},
					{Kind: LeafSparse, Space: "terms", Sparse: *sparseQ},
				}...),
				Method: tc.method,
				Alpha:  tc.alpha,
				RRFK:   tc.rrfK,
				K:      k,
			}
			qr, err := nc.Query(spec)
			if err != nil {
				t.Fatalf("Query: %v", err)
			}
			if qr.Mode != ModeFusion {
				t.Fatalf("mode = %d, want fusion", qr.Mode)
			}
			if len(qr.Lanes) != 2 {
				t.Fatalf("lanes = %d, want 2", len(qr.Lanes))
			}
			want, err := nc.NamedHybrid("title", denseQ, "terms", sparseQ, k, HybridOpts{Method: tc.method, Alpha: tc.alpha, RRFK: tc.rrfK})
			if err != nil {
				t.Fatalf("NamedHybrid: %v", err)
			}
			if !queryResultsEqual(qr.Fused, want) {
				t.Errorf("named FUSION fused != NamedHybrid\n got=%+v\nwant=%+v", qr.Fused, want)
			}
		})
	}
}

// TestNamedQueryThreeSpaceFusionExact runs a FUSION over THREE named spaces
// (title dense + image dense + terms sparse) and checks the local fold equals an
// independent fuseLanes over the three lanes computed directly from the per-space
// searches — proving execNamedLeaf dispatches each leaf to the right space and the
// N-lane fold is reused verbatim.
func TestNamedQueryThreeSpaceFusionExact(t *testing.T) {
	nc := newNamedQueryCorpus(t)
	titleQ := []float32{0.9, 0.1, 0, 0}
	imageQ := []float32{0.5, 0.5, 0}
	termsQ := sv([]uint32{2, 5}, []float32{2, 1})
	k := 5

	spec := QuerySpec{
		Mode: ModeFusion,
		Prefetch: srcs([]QueryLeaf{
			{Kind: LeafDense, Space: "title", Dense: titleQ},
			{Kind: LeafDense, Space: "image", Dense: imageQ},
			{Kind: LeafSparse, Space: "terms", Sparse: *termsQ},
		}...),
		Method: FusionRRF,
		K:      k,
	}
	qr, err := nc.Query(spec)
	if err != nil {
		t.Fatalf("Query: %v", err)
	}
	if len(qr.Lanes) != 3 {
		t.Fatalf("lanes = %d, want 3", len(qr.Lanes))
	}

	// Independent oracle: compute each lane via the per-space search at the same
	// default pool (max(k,50)), then fold via fuseLanes — the exact path Query took.
	pool := k
	if pool < 50 {
		pool = 50
	}
	titleLane, err := nc.SearchNamed("title", titleQ, pool, Filter{})
	if err != nil {
		t.Fatal(err)
	}
	imageLane, err := nc.SearchNamed("image", imageQ, pool, Filter{})
	if err != nil {
		t.Fatal(err)
	}
	termsLane, err := nc.SearchNamedSparse("terms", termsQ, pool, Filter{})
	if err != nil {
		t.Fatal(err)
	}
	// lane0 (titleLane) is a named-dense lane → distance-ascending orientation
	// (false), so the fold starts with Fuse — the exact pre-orientation behaviour.
	oracle := fuseLanes([][]Result{titleLane, imageLane, termsLane}, false, FusionRRF, 0, 0, k)
	if !queryResultsEqual(qr.Fused, oracle) {
		t.Errorf("3-space fused != independent fold\n got=%+v\nwant=%+v", qr.Fused, oracle)
	}
	// Lanes must be returned in prefetch order and match the per-space searches.
	if !queryResultsEqual(qr.Lanes[0], titleLane) || !queryResultsEqual(qr.Lanes[1], imageLane) || !queryResultsEqual(qr.Lanes[2], termsLane) {
		t.Errorf("returned lanes do not match per-space searches")
	}
}

// TestNamedQueryRerankVsBruteOracle proves RERANK exactness: prefetch two spaces,
// union the candidates, re-score by the dense "title" root restricted to the
// union, and compare to a brute-force exact L2 oracle over the same union.
func TestNamedQueryRerankVsBruteOracle(t *testing.T) {
	nc := newNamedQueryCorpus(t)
	titleRoot := []float32{0.9, 0.1, 0, 0}
	imagePref := []float32{0.5, 0.5, 0}
	termsPref := sv([]uint32{0, 5}, []float32{1, 3})
	k := 3

	spec := QuerySpec{
		Mode: ModeRerank,
		Root: QueryLeaf{Kind: LeafDense, Space: "title", Dense: titleRoot, K: k},
		Prefetch: srcs([]QueryLeaf{
			{Kind: LeafDense, Space: "image", Dense: imagePref, K: 6},
			{Kind: LeafSparse, Space: "terms", Sparse: *termsPref, K: 6},
		}...),
		K: k,
	}
	qr, err := nc.Query(spec)
	if err != nil {
		t.Fatalf("Query: %v", err)
	}
	if qr.Mode != ModeRerank {
		t.Fatalf("mode = %d, want rerank", qr.Mode)
	}

	// Oracle: recompute the prefetch union (same pools), brute-force the exact L2
	// distance from titleRoot for every union id in the TITLE space, take k closest.
	imageLane, err := nc.SearchNamed("image", imagePref, 6, Filter{})
	if err != nil {
		t.Fatal(err)
	}
	termsLane, err := nc.SearchNamedSparse("terms", termsPref, 6, Filter{})
	if err != nil {
		t.Fatal(err)
	}
	union := unionCandidates([][]Result{imageLane, termsLane})
	type idDist struct {
		id   uint64
		dist float32
	}
	titleVecs := map[uint64][]float32{
		1: {1, 0, 0, 0}, 2: {0, 1, 0, 0}, 3: {0, 0, 1, 0},
		4: {0.9, 0.1, 0, 0}, 5: {0, 0, 0, 1}, 6: {0.2, 0.2, 0.2, 0.2},
	}
	oracle := make([]idDist, 0, len(union))
	for _, id := range union {
		vec := titleVecs[id]
		var d float32
		for i := range vec {
			diff := vec[i] - titleRoot[i]
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
		t.Fatalf("rerank len %d != oracle len %d (union=%v)", len(qr.Fused), len(oracle), union)
	}
	for i := range oracle {
		if qr.Fused[i].ID != oracle[i].id {
			t.Errorf("rerank[%d] id=%d, oracle id=%d (dist %v)", i, qr.Fused[i].ID, oracle[i].id, oracle[i].dist)
		}
	}
}

// TestNamedQueryRerankSparseRoot exercises a sparse-space root re-scoring the
// prefetch union: only union ids with a nonzero sparse overlap survive, ranked by
// the sparse dot-product score (descending).
func TestNamedQueryRerankSparseRoot(t *testing.T) {
	nc := newNamedQueryCorpus(t)
	titlePref := []float32{0.9, 0.1, 0, 0}
	termsRoot := sv([]uint32{5}, []float32{1})
	k := 4

	spec := QuerySpec{
		Mode: ModeRerank,
		Root: QueryLeaf{Kind: LeafSparse, Space: "terms", Sparse: *termsRoot, K: k},
		Prefetch: srcs([]QueryLeaf{
			{Kind: LeafDense, Space: "title", Dense: titlePref, K: 6},
			{Kind: LeafSparse, Space: "terms", Sparse: *termsRoot, K: 6},
		}...),
		K: k,
	}
	qr, err := nc.Query(spec)
	if err != nil {
		t.Fatalf("Query: %v", err)
	}
	// Doc 3 has the strongest term-5 weight (4.0), so it must rank first among the
	// term-5 holders in the union.
	if len(qr.Fused) == 0 || qr.Fused[0].ID != 3 {
		t.Fatalf("sparse-root rerank top should be doc 3; got %+v", qr.Fused)
	}
	for _, r := range qr.Fused {
		if r.Score <= 0 {
			t.Errorf("sparse rerank result %d has non-positive score %v", r.ID, r.Score)
		}
	}
}

// TestNamedQueryValidation covers the named fail-loud paths: a named query with a
// Space-less prefetch leaf, a Space-less RERANK root, and (the dense side) a dense
// (*Collection).Query rejecting a Space-bearing leaf.
func TestNamedQueryValidation(t *testing.T) {
	nc := newNamedQueryCorpus(t)
	denseQ := []float32{1, 0, 0, 0}

	// Named query, prefetch leaf with no Space → ErrQueryNamedLeafNoSpace.
	if _, err := nc.Query(QuerySpec{
		Mode:     ModeFusion,
		Prefetch: srcs([]QueryLeaf{{Kind: LeafDense, Dense: denseQ, K: 3}}...),
		K:        3,
	}); err != ErrQueryNamedLeafNoSpace {
		t.Errorf("named no-space leaf: err=%v, want ErrQueryNamedLeafNoSpace", err)
	}

	// Named RERANK, root with no Space → ErrQueryNamedLeafNoSpace.
	if _, err := nc.Query(QuerySpec{
		Mode:     ModeRerank,
		Root:     QueryLeaf{Kind: LeafDense, Dense: denseQ, K: 3}, // no Space
		Prefetch: srcs([]QueryLeaf{{Kind: LeafDense, Space: "title", Dense: denseQ, K: 3}}...),
		K:        3,
	}); err != ErrQueryNamedLeafNoSpace {
		t.Errorf("named no-space root: err=%v, want ErrQueryNamedLeafNoSpace", err)
	}

	// Dense collection Query rejecting a Space-bearing leaf → ErrQueryDenseLeafHasSpace.
	c, err := NewCollection("dq", Config{Dim: 4, Metric: L2, M: 16, EfConstruction: 200, EfSearch: 64, Seed: 1})
	if err != nil {
		t.Fatalf("NewCollection: %v", err)
	}
	if _, err := c.Query(QuerySpec{
		Mode:     ModeFusion,
		Prefetch: srcs([]QueryLeaf{{Kind: LeafDense, Space: "title", Dense: denseQ, K: 3}}...),
		K:        3,
	}); err != ErrQueryDenseLeafHasSpace {
		t.Errorf("dense leaf with space: err=%v, want ErrQueryDenseLeafHasSpace", err)
	}
}

// TestNamedQueryUnknownSpace checks a leaf targeting a non-existent / wrong-modality
// space fails loud (the per-space search returns ErrUnknownVectorName /
// ErrSpaceModalityMismatch, propagated out of Query).
func TestNamedQueryUnknownSpace(t *testing.T) {
	nc := newNamedQueryCorpus(t)
	if _, err := nc.Query(QuerySpec{
		Mode:     ModeFusion,
		Prefetch: srcs([]QueryLeaf{{Kind: LeafDense, Space: "nope", Dense: []float32{1, 0, 0, 0}, K: 3}}...),
		K:        3,
	}); err == nil {
		t.Errorf("unknown space: expected a fail-loud error, got nil")
	}
}
