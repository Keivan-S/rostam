// SPDX-License-Identifier: Apache-2.0

package vector

import (
	"errors"
	"sort"
	"testing"
)

// TestMVQueryTwoLaneFusionEqualsMVHybrid is the UNIFICATION cross-check: a 2-lane
// MV FUSION (a MaxSim leaf + a doc-sparse leaf) via (*MultiVectorIndex).Query must
// equal the equivalent MVHybrid EXACTLY. This proves the orientation-aware fold is
// correct — both MV lanes are score-descending, so fuseLanes must start with
// FuseScoreLanes (lane0 ScoreDesc=true), the same fold MVHybrid uses. A regression
// to the dense Fuse start (which inverts lane0's "distance") would diverge.
func TestMVQueryTwoLaneFusionEqualsMVHybrid(t *testing.T) {
	m, _ := mvHybridFixture(t)
	query := [][]float32{{1, 0, 0}, {0, 1, 0}}
	sparseQ := mvSV(0, 1.0, 2, 1.0, 5, 1.0)
	k := 5

	for _, tc := range []struct {
		name string
		opts HybridOpts
	}{
		{"rrf", HybridOpts{Method: FusionRRF}},
		{"weighted_default_alpha", HybridOpts{Method: FusionWeighted}},
		{"weighted_alpha_0.3", HybridOpts{Method: FusionWeighted, Alpha: 0.3}},
		{"rrf_rrfk_10", HybridOpts{Method: FusionRRF, RRFK: 10}},
		{"dbsf", HybridOpts{Method: FusionDBSF}},
	} {
		t.Run(tc.name, func(t *testing.T) {
			want, err := m.MVHybrid(query, sparseQ, k, tc.opts)
			if err != nil {
				t.Fatal(err)
			}
			spec := QuerySpec{
				Mode:   ModeFusion,
				Method: tc.opts.Method,
				Alpha:  tc.opts.Alpha,
				RRFK:   tc.opts.RRFK,
				K:      k,
				Prefetch: srcs([]QueryLeaf{
					{Kind: LeafMVMaxSim, Tokens: query, ScoreDesc: true},
					{Kind: LeafSparse, Sparse: *sparseQ, ScoreDesc: true},
				}...),
			}
			qr, err := m.Query(spec)
			if err != nil {
				t.Fatal(err)
			}
			if qr.Mode != ModeFusion {
				t.Fatalf("mode = %d, want ModeFusion", qr.Mode)
			}
			resultsEqual(t, qr.Fused, want, tc.name)
		})
	}
}

// TestMVQueryFusionSaneTopK checks a 2-lane MV FUSION returns a non-empty, sane
// fused top-k (every result is a live doc; the list is unique).
func TestMVQueryFusionSaneTopK(t *testing.T) {
	m, _ := mvHybridFixture(t)
	spec := QuerySpec{
		Mode:   ModeFusion,
		Method: FusionRRF,
		K:      3,
		Prefetch: srcs([]QueryLeaf{
			{Kind: LeafMVMaxSim, Tokens: [][]float32{{1, 0, 0}}, ScoreDesc: true},
			{Kind: LeafSparse, Sparse: *mvSV(0, 1.0, 2, 1.0), ScoreDesc: true},
		}...),
	}
	qr, err := m.Query(spec)
	if err != nil {
		t.Fatal(err)
	}
	if len(qr.Fused) == 0 {
		t.Fatal("fused result is empty")
	}
	if len(qr.Fused) > 3 {
		t.Fatalf("fused len %d > k=3", len(qr.Fused))
	}
	seen := map[uint64]bool{}
	for _, r := range qr.Fused {
		if seen[r.ID] {
			t.Fatalf("duplicate id %d in fused result", r.ID)
		}
		seen[r.ID] = true
		if !m.Exists(r.ID) {
			t.Fatalf("fused id %d is not a live doc", r.ID)
		}
	}
}

// TestMVQueryRerankMaxSimVsBruteOracle proves RERANK exactness: prefetch a MaxSim
// lane + a sparse lane, then RERANK the UNION of their candidate ids by a MaxSim
// root. The result must equal a brute-force MaxSim over the candidate union,
// score-descending.
func TestMVQueryRerankMaxSimVsBruteOracle(t *testing.T) {
	m, _ := mvHybridFixture(t)
	maxsimQ := [][]float32{{1, 0, 0}, {0, 1, 0}}
	sparseQ := mvSV(0, 1.0, 2, 1.0, 5, 1.0)
	rootQ := [][]float32{{0, 1, 0}, {0, 0, 1}}
	k := 4

	spec := QuerySpec{
		Mode: ModeRerank,
		K:    k,
		Root: QueryLeaf{Kind: LeafMVMaxSim, Tokens: rootQ, ScoreDesc: true},
		Prefetch: srcs([]QueryLeaf{
			{Kind: LeafMVMaxSim, Tokens: maxsimQ, ScoreDesc: true},
			{Kind: LeafSparse, Sparse: *sparseQ, ScoreDesc: true},
		}...),
	}
	qr, err := m.Query(spec)
	if err != nil {
		t.Fatal(err)
	}
	if qr.Mode != ModeRerank {
		t.Fatalf("mode = %d, want ModeRerank", qr.Mode)
	}

	// Oracle: reconstruct the prefetch candidate union exactly the engine would
	// (each lane at the default pool max(k,50)), then brute-force MaxSim over the
	// union by the root query, score-descending.
	pool := 50
	mr, err := m.Search(maxsimQ, pool, MultiSearchOpts{})
	if err != nil {
		t.Fatal(err)
	}
	sparseLane := m.SearchSparse(sparseQ, pool)
	unionSeen := map[uint64]bool{}
	var union []uint64
	for _, r := range mr {
		if !unionSeen[r.ID] {
			unionSeen[r.ID] = true
			union = append(union, r.ID)
		}
	}
	for _, r := range sparseLane {
		if !unionSeen[r.ID] {
			unionSeen[r.ID] = true
			union = append(union, r.ID)
		}
	}
	want := bruteMaxSimOverIDs(t, m, rootQ, union, k)
	resultsEqual(t, qr.Fused, want, "rerank-maxsim")
}

// bruteMaxSim scores ONLY the given candidate ids by exact MaxSim against the root
// query (normalized), score-descending, top-k — an independent oracle for the MV
// RERANK MaxSim root path.
func bruteMaxSimOverIDs(t *testing.T, m *MultiVectorIndex, query [][]float32, cands []uint64, k int) []Result {
	t.Helper()
	norm := make([][]float32, len(query))
	for i, q := range query {
		nq := make([]float32, len(q))
		copy(nq, q)
		normalize(nq)
		norm[i] = nq
	}
	out := make([]Result, 0, len(cands))
	for _, id := range cands {
		tokens, _, _, ok := m.Get(id)
		if !ok {
			continue
		}
		var score float32
		for _, q := range norm {
			var best float32
			for di, d := range tokens {
				s := dotProduct(q, d)
				if di == 0 || s > best {
					best = s
				}
			}
			score += best
		}
		out = append(out, Result{ID: id, Score: score})
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
	return out
}

// TestMVQueryRerankSparseRootVsOracle reranks the prefetch union by a SPARSE root,
// vs a brute restricted sparse scan over the union.
func TestMVQueryRerankSparseRootVsOracle(t *testing.T) {
	m, _ := mvHybridFixture(t)
	maxsimQ := [][]float32{{1, 0, 0}}
	prefetchSparse := mvSV(0, 1.0, 5, 1.0)
	rootSparse := mvSV(0, 1.0, 2, 1.0)
	k := 3

	spec := QuerySpec{
		Mode: ModeRerank,
		K:    k,
		Root: QueryLeaf{Kind: LeafSparse, Sparse: *rootSparse, ScoreDesc: true},
		Prefetch: srcs([]QueryLeaf{
			{Kind: LeafMVMaxSim, Tokens: maxsimQ, ScoreDesc: true},
			{Kind: LeafSparse, Sparse: *prefetchSparse, ScoreDesc: true},
		}...),
	}
	qr, err := m.Query(spec)
	if err != nil {
		t.Fatal(err)
	}

	// Oracle: union of the two prefetch lanes, then a full sparse scan restricted to
	// that union, score-desc.
	pool := 50
	mr, err := m.Search(maxsimQ, pool, MultiSearchOpts{})
	if err != nil {
		t.Fatal(err)
	}
	sparseLane := m.SearchSparse(prefetchSparse, pool)
	unionSeen := map[uint64]bool{}
	for _, r := range mr {
		unionSeen[r.ID] = true
	}
	for _, r := range sparseLane {
		unionSeen[r.ID] = true
	}
	full := m.SearchSparse(rootSparse, m.NumDocs())
	var want []Result
	for _, r := range full {
		if unionSeen[r.ID] {
			want = append(want, r)
		}
	}
	sort.SliceStable(want, func(a, b int) bool {
		if want[a].Score != want[b].Score {
			return want[a].Score > want[b].Score
		}
		return want[a].ID < want[b].ID
	})
	if k > 0 && len(want) > k {
		want = want[:k]
	}
	resultsEqual(t, qr.Fused, want, "rerank-sparse")
}

// TestMVQueryValidationFailLoud covers the MV fail-loud edges: a leaf carrying a
// space, a MaxSim leaf with no tokens, an MV sparse leaf with no terms, a dense
// leaf (not an MV node), no prefetch, and an unknown mode.
func TestMVQueryValidationFailLoud(t *testing.T) {
	m, _ := mvHybridFixture(t)
	query := [][]float32{{1, 0, 0}}

	cases := []struct {
		name string
		spec QuerySpec
		want error
	}{
		{
			name: "leaf_with_space",
			spec: QuerySpec{Mode: ModeFusion, K: 3, Prefetch: srcs([]QueryLeaf{
				{Kind: LeafMVMaxSim, Tokens: query, Space: "title", ScoreDesc: true},
			}...)},
			want: ErrQueryMVLeafHasSpace,
		},
		{
			name: "maxsim_no_tokens",
			spec: QuerySpec{Mode: ModeFusion, K: 3, Prefetch: srcs([]QueryLeaf{
				{Kind: LeafMVMaxSim, ScoreDesc: true},
			}...)},
			want: ErrQueryMVMaxSimNoTokens,
		},
		{
			name: "sparse_empty",
			spec: QuerySpec{Mode: ModeFusion, K: 3, Prefetch: srcs([]QueryLeaf{
				{Kind: LeafSparse, ScoreDesc: true},
			}...)},
			want: ErrQueryMVSparseEmpty,
		},
		{
			name: "dense_leaf_not_mv",
			spec: QuerySpec{Mode: ModeFusion, K: 3, Prefetch: srcs([]QueryLeaf{
				{Kind: LeafDense, Dense: []float32{1, 0, 0}},
			}...)},
			want: ErrQueryBadLeafKind,
		},
		{
			name: "no_prefetch",
			spec: QuerySpec{Mode: ModeFusion, K: 3},
			want: ErrQueryNoPrefetch,
		},
		{
			name: "unknown_mode",
			spec: QuerySpec{Mode: QueryMode(99), K: 3, Prefetch: srcs([]QueryLeaf{
				{Kind: LeafMVMaxSim, Tokens: query, ScoreDesc: true},
			}...)},
			want: ErrQueryBadMode,
		},
		{
			name: "rerank_empty_root",
			spec: QuerySpec{Mode: ModeRerank, K: 3, Prefetch: srcs([]QueryLeaf{
				{Kind: LeafMVMaxSim, Tokens: query, ScoreDesc: true},
			}...), Root: QueryLeaf{Kind: LeafMVMaxSim, ScoreDesc: true}},
			want: ErrQueryMVMaxSimNoTokens,
		},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			_, err := m.Query(tc.spec)
			if !errors.Is(err, tc.want) {
				t.Fatalf("err = %v, want %v", err, tc.want)
			}
		})
	}
}

// TestMVQuerySingleLaneFusion: a 1-lane MV FUSION (MaxSim only) returns the MaxSim
// lane truncated to k (no fusion).
func TestMVQuerySingleLaneFusion(t *testing.T) {
	m, _ := mvHybridFixture(t)
	query := [][]float32{{1, 0, 0}, {0, 1, 0}}
	k := 3
	spec := QuerySpec{Mode: ModeFusion, Method: FusionRRF, K: k, Prefetch: srcs([]QueryLeaf{
		{Kind: LeafMVMaxSim, Tokens: query, ScoreDesc: true},
	}...)}
	qr, err := m.Query(spec)
	if err != nil {
		t.Fatal(err)
	}
	mr, err := m.Search(query, k, MultiSearchOpts{})
	if err != nil {
		t.Fatal(err)
	}
	want := make([]Result, len(mr))
	for i, r := range mr {
		want[i] = Result{ID: r.ID, Score: r.Score}
	}
	resultsEqual(t, qr.Fused, want, "single-lane-maxsim")
}
