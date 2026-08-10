// SPDX-License-Identifier: Apache-2.0

package vector

import (
	"sort"
	"testing"
)

// ownGlobalStats builds a BM25GlobalStats from an index's OWN live corpus stats
// for the given query — i.e. the degenerate "single shard == whole corpus" case.
// Feeding these to the *Global scorers must reproduce the LOCAL scorer exactly.
func ownGlobalStats(h *hnsw, query string) BM25GlobalStats {
	n, tokenTotal, df := h.CorpusStats(query)
	var avgdl float32
	if n > 0 {
		avgdl = float32(tokenTotal) / float32(n)
	}
	return BM25GlobalStats{N: n, Avgdl: avgdl, DF: df}
}

// TestCorpusStatsReader checks corpusStats returns the index's true n/tokenTotal
// and per-query-term df (including an explicit 0 for an absent term).
func TestCorpusStatsReader(t *testing.T) {
	c := newFullTextCollection(t)
	corpus := map[uint64]string{
		1: "the quick brown fox",     // quick(1) brown(1) fox(1) -> 3 tokens
		2: "quick brown dog",         // quick(1) brown(1) dog(1) -> 3 tokens
		3: "lazy cats sleep all day", // lazy cats sleep day -> 4 tokens (stopwords removed)
	}
	for id, content := range corpus {
		mustUpsert(t, c, id, []float32{float32(id), 0, 0, 0}, content)
	}
	h := c.idx.(*hnsw)

	// n and tokenTotal are the index's own.
	wantN := h.bm25.n
	wantTok := h.bm25.tokenTotal

	terms := h.az.Analyze("quick fox missingterm")
	n, tok, df := h.bm25.corpusStats(terms)
	if n != wantN {
		t.Fatalf("n: got %d want %d", n, wantN)
	}
	if tok != wantTok {
		t.Fatalf("tokenTotal: got %d want %d", tok, wantTok)
	}
	// Every query term gets an entry; an absent term is reported as 0.
	if len(df) != len(terms) {
		t.Fatalf("df entries: got %d want %d (one per query term)", len(df), len(terms))
	}
	quick := h.az.Analyze("quick")[0]
	fox := h.az.Analyze("fox")[0]
	missing := h.az.Analyze("missingterm")[0]
	if df[quick] != 2 {
		t.Fatalf("df[quick]: got %d want 2", df[quick])
	}
	if df[fox] != 1 {
		t.Fatalf("df[fox]: got %d want 1", df[fox])
	}
	if v, ok := df[missing]; !ok || v != 0 {
		t.Fatalf("df[missing]: got (%d, present=%v) want (0, present=true)", v, ok)
	}

	// hnsw.CorpusStats (analyze + lock) agrees with the raw reader.
	n2, tok2, df2 := h.CorpusStats("quick fox missingterm")
	if n2 != wantN || tok2 != wantTok || df2[quick] != 2 || df2[fox] != 1 {
		t.Fatalf("CorpusStats mismatch: n=%d tok=%d df[quick]=%d df[fox]=%d", n2, tok2, df2[quick], df2[fox])
	}

	// Full text disabled ⇒ zero/nil cleanly.
	plain, err := NewCollection("plain", Config{Dim: 4, Metric: L2, M: 8, EfConstruction: 50, EfSearch: 32, Seed: 1})
	if err != nil {
		t.Fatal(err)
	}
	if n, tok, df := plain.idx.(*hnsw).CorpusStats("quick"); n != 0 || tok != 0 || df != nil {
		t.Fatalf("disabled CorpusStats: got n=%d tok=%d df=%v want 0/0/nil", n, tok, df)
	}
}

// TestSearchTextGlobalIdentity asserts that feeding an index its OWN stats to the
// global scorer reproduces the local SearchText ids+scores BIT-IDENTICALLY.
func TestSearchTextGlobalIdentity(t *testing.T) {
	c := newFullTextCollection(t)
	corpus := map[uint64]string{
		1: "the quick brown fox jumps over the lazy dog",
		2: "a quick brown dog outpaces a quick fox",
		3: "lazy cats sleep all day long",
		4: "the fox and the hound",
	}
	for id, content := range corpus {
		mustUpsert(t, c, id, []float32{float32(id), 0, 0, 0}, content)
	}
	h := c.idx.(*hnsw)

	for _, q := range []string{"quick fox", "lazy dog", "fox", "quick brown dog"} {
		local := h.SearchText(q, 10, nil)
		global := h.SearchTextGlobal(q, 10, nil, ownGlobalStats(h, q))
		assertExactResults(t, q, global, local)
	}
}

// TestHybridTextLanesGlobalIdentity asserts the global hybrid lanes with the
// index's own stats are bit-identical to the local hybrid lanes (both dense and
// text lanes).
func TestHybridTextLanesGlobalIdentity(t *testing.T) {
	c := newFullTextCollection(t)
	mustUpsert(t, c, 1, []float32{1, 0, 0, 0}, "fox in the henhouse")
	mustUpsert(t, c, 2, []float32{0, 1, 0, 0}, "fox fox fox everywhere")
	mustUpsert(t, c, 3, []float32{0, 0, 1, 0}, "a quiet meadow with a fox")
	h := c.idx.(*hnsw)

	dense := []float32{1, 0, 0, 0}
	query := "fox meadow"

	denseLoc, textLoc, err := h.HybridTextLanes(dense, query, 10, HybridOpts{})
	if err != nil {
		t.Fatalf("local lanes: %v", err)
	}
	denseGlob, textGlob, _, err := h.HybridTextLanesGlobal(dense, query, 10, HybridOpts{}, ownGlobalStats(h, query))
	if err != nil {
		t.Fatalf("global lanes: %v", err)
	}
	assertExactResults(t, "hybrid-dense", denseGlob, denseLoc)
	assertExactResults(t, "hybrid-text", textGlob, textLoc)
}

// TestBM25GlobalDFInvariant is the core correctness proof: a hand-split corpus
// across TWO indexes, each scored with the SUMMED global stats and unioned by
// score, reproduces a THIRD index holding ALL docs scored with plain SearchText —
// ids+scores bit-identical. This is the dfs_query_then_fetch invariant at the
// engine level (per-shard local df differs from the combined df).
func TestBM25GlobalDFInvariant(t *testing.T) {
	// Doc set chosen so per-index df differs from combined df: "fox" lives in BOTH
	// halves, so index1.df[fox] != combined.df[fox].
	docsA := map[uint64]string{
		1: "quick brown fox",
		2: "quick fox runs fast",
		3: "lazy brown dog",
	}
	docsB := map[uint64]string{
		4: "fox hunts at night",
		5: "the brown owl hoots",
	}

	build := func(name string, parts ...map[uint64]string) *Collection {
		c := newFullTextCollection(t)
		for _, p := range parts {
			for id, content := range p {
				mustUpsert(t, c, id, []float32{float32(id), 0, 0, 0}, content)
			}
		}
		_ = name
		return c
	}

	idx1 := build("idx1", docsA)
	idx2 := build("idx2", docsB)
	combined := build("combined", docsA, docsB)

	h1 := idx1.idx.(*hnsw)
	h2 := idx2.idx.(*hnsw)
	hc := combined.idx.(*hnsw)

	for _, query := range []string{"fox", "brown fox", "quick fox dog", "brown"} {
		// Gather per-index stats and SUM into the global stats.
		n1, tt1, df1 := h1.CorpusStats(query)
		n2, tt2, df2 := h2.CorpusStats(query)
		gN := n1 + n2
		gTok := tt1 + tt2
		var gAvgdl float32
		if gN > 0 {
			gAvgdl = float32(gTok) / float32(gN)
		}
		gDF := map[uint32]int{}
		for t, v := range df1 {
			gDF[t] += v
		}
		for t, v := range df2 {
			gDF[t] += v
		}
		g := BM25GlobalStats{N: gN, Avgdl: gAvgdl, DF: gDF}

		// Score each index with the GLOBAL stats; union by score (descending).
		res1 := h1.SearchTextGlobal(query, 50, nil, g)
		res2 := h2.SearchTextGlobal(query, 50, nil, g)
		union := append(append([]Result{}, res1...), res2...)
		sortResults(union)

		// Oracle: the combined index scored with plain (single-node) SearchText.
		want := hc.SearchText(query, 50, nil)
		sortResults(want)

		assertExactResults(t, "dfs-invariant "+query, union, want)
	}
}

// sortResults orders Results descending by Score, ties broken by ID ascending —
// a deterministic merge order for cross-index unions.
func sortResults(rs []Result) {
	sort.SliceStable(rs, func(i, j int) bool {
		if rs[i].Score != rs[j].Score {
			return rs[i].Score > rs[j].Score
		}
		return rs[i].ID < rs[j].ID
	})
}

// assertExactResults asserts two Result slices are bit-identical in id AND score
// (exact float equality — the global path must reproduce the local path exactly).
func assertExactResults(t *testing.T, name string, got, want []Result) {
	t.Helper()
	if len(got) != len(want) {
		t.Fatalf("%s: got %d results, want %d (got=%+v want=%+v)", name, len(got), len(want), got, want)
	}
	for i := range got {
		if got[i].ID != want[i].ID {
			t.Fatalf("%s rank %d: id got %d want %d", name, i, got[i].ID, want[i].ID)
		}
		if got[i].Score != want[i].Score {
			t.Fatalf("%s rank %d (id %d): score got %v want %v (NOT bit-identical)", name, i, got[i].ID, got[i].Score, want[i].Score)
		}
	}
}
