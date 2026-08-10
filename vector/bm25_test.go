// SPDX-License-Identifier: Apache-2.0

package vector

import (
	"math"
	"testing"

	"github.com/rostamlabs/rostam/vector/analysis"
)

// referenceBM25 is a self-contained, hand-derivable BM25 scorer used as the
// oracle for the engine's bm25SearchTopK. It re-uses the SAME analyzer the engine
// uses (so term ids match) but computes idf/saturation independently with float64
// arithmetic, against an explicit corpus.
type refDoc struct {
	slot   uint32
	counts map[uint32]uint32
	length uint64
}

type referenceBM25 struct {
	az     analysis.CountingAnalyzer
	k1, b  float64
	docs   []refDoc
	df     map[uint32]int
	tokTot uint64
}

func newReferenceBM25(k1, b float64) *referenceBM25 {
	return &referenceBM25{
		az: analysis.NewEnglishAnalyzer(),
		k1: k1, b: b,
		df: map[uint32]int{},
	}
}

func (r *referenceBM25) add(slot uint32, content string) {
	counts := r.az.AnalyzeCounts(content)
	var length uint64
	for t, tf := range counts {
		r.df[t]++
		length += uint64(tf)
	}
	r.docs = append(r.docs, refDoc{slot: slot, counts: counts, length: length})
	r.tokTot += length
}

func (r *referenceBM25) avgdl() float64 {
	if len(r.docs) == 0 {
		return 0
	}
	return float64(r.tokTot) / float64(len(r.docs))
}

// score returns the BM25 score of doc d for the query terms (deduped set).
func (r *referenceBM25) score(d refDoc, queryTerms []uint32) float64 {
	n := float64(len(r.docs))
	avgdl := r.avgdl()
	var s float64
	seen := map[uint32]bool{}
	for _, t := range queryTerms {
		if seen[t] {
			continue
		}
		seen[t] = true
		tf := float64(d.counts[t])
		if tf == 0 {
			continue
		}
		df := float64(r.df[t])
		idf := math.Log(1 + (n-df+0.5)/(df+0.5))
		denom := tf + r.k1*(1-r.b+r.b*float64(d.length)/avgdl)
		s += idf * (tf * (r.k1 + 1)) / denom
	}
	return s
}

// newFullTextHNSW builds a bare *hnsw with FullText enabled for unit-level tests
// of the index/scan internals.
func newFullTextHNSW(t *testing.T) *hnsw {
	t.Helper()
	cfg := Config{Dim: 4, Metric: L2, M: 8, EfConstruction: 50, EfSearch: 32, Seed: 1,
		FullText: &FullTextConfig{}}
	h, err := newHNSW(cfg)
	if err != nil {
		t.Fatalf("newHNSW: %v", err)
	}
	return h
}

func mustUpsert(t *testing.T, c *Collection, id uint64, vec []float32, content string) {
	t.Helper()
	if err := c.Upsert(id, vec, content, 0, nil, nil); err != nil {
		t.Fatalf("upsert %d: %v", id, err)
	}
}

func newFullTextCollection(t *testing.T) *Collection {
	t.Helper()
	cfg := Config{Dim: 4, Metric: L2, M: 8, EfConstruction: 50, EfSearch: 32, Seed: 1,
		FullText: &FullTextConfig{}}
	c, err := NewCollection("ft", cfg)
	if err != nil {
		t.Fatalf("NewCollection: %v", err)
	}
	return c
}

// TestBM25ScanMatchesReference inserts a tiny known corpus through the engine and
// asserts bm25SearchTopK's ids+scores equal the independent reference scorer.
func TestBM25ScanMatchesReference(t *testing.T) {
	c := newFullTextCollection(t)
	corpus := map[uint64]string{
		1: "the quick brown fox jumps over the lazy dog",
		2: "a quick brown dog outpaces a quick fox",
		3: "lazy cats sleep all day long",
		4: "the fox and the hound",
	}
	ref := newReferenceBM25(float64(defaultBM25K1), float64(defaultBM25B))
	for id, content := range corpus {
		mustUpsert(t, c, id, []float32{float32(id), 0, 0, 0}, content)
	}
	h := c.idx.(*hnsw)
	// Build the reference with the SAME slot mapping the engine used.
	for id := range corpus {
		slot := h.arena.idMap[id]
		ref.add(slot, corpus[id])
	}

	queries := []string{"quick fox", "lazy dog", "fox"}
	for _, q := range queries {
		terms := h.az.Analyze(q)
		s := layerScratchPool.Get().(*layerScratch)
		got := h.bm25.bm25SearchTopK(s, h.arena.Capacity(), terms, 10, nil)
		layerScratchPool.Put(s)

		// Reference ranking over all docs.
		type sc struct {
			slot  uint32
			score float64
		}
		var want []sc
		for _, d := range ref.docs {
			v := ref.score(d, terms)
			if v != 0 {
				want = append(want, sc{d.slot, v})
			}
		}
		// Sort want desc by score, tie by slot asc (matching topK).
		for i := 0; i < len(want); i++ {
			for j := i + 1; j < len(want); j++ {
				if want[j].score > want[i].score ||
					(want[j].score == want[i].score && want[j].slot < want[i].slot) {
					want[i], want[j] = want[j], want[i]
				}
			}
		}
		if len(got) != len(want) {
			t.Fatalf("query %q: got %d results, want %d", q, len(got), len(want))
		}
		for i := range got {
			if got[i].slot != want[i].slot {
				t.Fatalf("query %q rank %d: slot got %d want %d", q, i, got[i].slot, want[i].slot)
			}
			if math.Abs(float64(got[i].score)-want[i].score) > 1e-4 {
				t.Fatalf("query %q rank %d (slot %d): score got %v want %v",
					q, i, got[i].slot, got[i].score, want[i].score)
			}
		}
	}
}

// TestBM25LengthNormDirection: same tf for a term, longer doc → lower score.
func TestBM25LengthNormDirection(t *testing.T) {
	c := newFullTextCollection(t)
	// Both docs contain "fox" exactly once; doc 2 is much longer.
	mustUpsert(t, c, 1, []float32{1, 0, 0, 0}, "fox")
	mustUpsert(t, c, 2, []float32{2, 0, 0, 0}, "fox runs through the wide green meadow at dawn")
	h := c.idx.(*hnsw)
	terms := h.az.Analyze("fox")
	s := layerScratchPool.Get().(*layerScratch)
	got := h.bm25.bm25SearchTopK(s, h.arena.Capacity(), terms, 10, nil)
	layerScratchPool.Put(s)
	if len(got) != 2 {
		t.Fatalf("got %d results, want 2", len(got))
	}
	short := h.arena.idMap[1]
	if got[0].slot != short {
		t.Fatalf("expected shorter doc (slot %d) ranked first, got slot %d", short, got[0].slot)
	}
	if !(got[0].score > got[1].score) {
		t.Fatalf("shorter doc should score higher: %v vs %v", got[0].score, got[1].score)
	}
}

// TestBM25TFSaturation: more occurrences → higher score, but sub-linearly.
func TestBM25TFSaturation(t *testing.T) {
	c := newFullTextCollection(t)
	mustUpsert(t, c, 1, []float32{1, 0, 0, 0}, "fox")
	mustUpsert(t, c, 2, []float32{2, 0, 0, 0}, "fox fox")
	mustUpsert(t, c, 3, []float32{3, 0, 0, 0}, "fox fox fox fox")
	h := c.idx.(*hnsw)
	terms := h.az.Analyze("fox")
	s := layerScratchPool.Get().(*layerScratch)
	got := h.bm25.bm25SearchTopK(s, h.arena.Capacity(), terms, 10, nil)
	layerScratchPool.Put(s)
	score := map[uint32]float32{}
	for _, r := range got {
		score[r.slot] = r.score
	}
	s1, s2, s3 := score[h.arena.idMap[1]], score[h.arena.idMap[2]], score[h.arena.idMap[3]]
	if !(s3 > s2 && s2 > s1) {
		t.Fatalf("tf monotonicity broken: tf1=%v tf2=%v tf4=%v", s1, s2, s3)
	}
	// Saturation: the jump from 1→2 must exceed the jump from 2→4 (diminishing).
	if !((s2 - s1) > (s3 - s2)) {
		t.Fatalf("expected diminishing returns: d(1->2)=%v d(2->4)=%v", s2-s1, s3-s2)
	}
}

// TestBM25IDFDirection: a rarer term contributes a higher idf, so a doc matching
// a rare term outranks a same-tf doc matching a common term.
func TestBM25IDFDirection(t *testing.T) {
	c := newFullTextCollection(t)
	// "common" appears in many docs; "rare" in one. Doc 1 has both once.
	mustUpsert(t, c, 1, []float32{1, 0, 0, 0}, "common rare")
	mustUpsert(t, c, 2, []float32{2, 0, 0, 0}, "common word")
	mustUpsert(t, c, 3, []float32{3, 0, 0, 0}, "common thing")
	mustUpsert(t, c, 4, []float32{4, 0, 0, 0}, "common item")
	h := c.idx.(*hnsw)
	idf := func(term string) float64 {
		ts := h.az.Analyze(term)
		df := float64(h.bm25.df[ts[0]])
		n := float64(h.bm25.n)
		return math.Log(1 + (n-df+0.5)/(df+0.5))
	}
	if !(idf("rare") > idf("common")) {
		t.Fatalf("rarer term should have higher idf: rare=%v common=%v", idf("rare"), idf("common"))
	}
}

// TestBM25StatsLifecycle: df/n/avgdl stay correct across insert→replace→delete→reclaim.
func TestBM25StatsLifecycle(t *testing.T) {
	c := newFullTextCollection(t)
	h := c.idx.(*hnsw)
	az := analysis.NewEnglishAnalyzer()

	// Insert two docs.
	mustUpsert(t, c, 1, []float32{1, 0, 0, 0}, "alpha beta beta")
	mustUpsert(t, c, 2, []float32{2, 0, 0, 0}, "beta gamma")
	assertStats := func(wantN int, wantTok uint64) {
		t.Helper()
		if h.bm25.n != wantN {
			t.Fatalf("n: got %d want %d", h.bm25.n, wantN)
		}
		if h.bm25.tokenTotal != wantTok {
			t.Fatalf("tokenTotal: got %d want %d", h.bm25.tokenTotal, wantTok)
		}
	}
	// doc1: alpha(1)+beta(2)=3 tokens; doc2: beta(1)+gamma(1)=2 → total 5, n=2.
	assertStats(2, 5)
	betaID := az.Analyze("beta")[0]
	if h.bm25.df[betaID] != 2 {
		t.Fatalf("df[beta]: got %d want 2", h.bm25.df[betaID])
	}

	// Replace doc1 with a shorter, beta-free text. Upsert = delete+insert.
	mustUpsert(t, c, 1, []float32{1, 0, 0, 0}, "delta")
	// doc1: delta(1)=1; doc2: 2 → total 3, n=2.
	assertStats(2, 3)
	if h.bm25.df[betaID] != 1 {
		t.Fatalf("df[beta] after replace: got %d want 1", h.bm25.df[betaID])
	}

	// Delete doc2 (tombstone only — stats stay until reclaim? No: replace path
	// removes; plain Delete is lazy, so n/df are NOT decremented by Delete alone).
	if _, err := h.Delete(2, CASCond{}); err != nil {
		t.Fatalf("delete: %v", err)
	}
	// Delete is lazy: corpus stats still reflect doc2 until Reclaim rebuilds.
	assertStats(2, 3)

	// Reclaim rebuilds from live $content: only doc1 ("delta") survives.
	h.Reclaim()
	assertStats(1, 1)
	if _, ok := h.bm25.df[betaID]; ok {
		t.Fatalf("df[beta] should be gone after reclaim, got %d", h.bm25.df[betaID])
	}
	deltaID := az.Analyze("delta")[0]
	if h.bm25.df[deltaID] != 1 {
		t.Fatalf("df[delta] after reclaim: got %d want 1", h.bm25.df[deltaID])
	}
	if got := h.bm25.avgdl(); got != 1 {
		t.Fatalf("avgdl after reclaim: got %v want 1", got)
	}
}

// TestSearchTextReturnsContentAndRespectsFilter.
func TestSearchTextReturnsContentAndRespectsFilter(t *testing.T) {
	c := newFullTextCollection(t)
	if err := c.Upsert(1, []float32{1, 0, 0, 0}, "quick brown fox", 0, Metadata{"lang": NewString("en")}, nil); err != nil {
		t.Fatal(err)
	}
	if err := c.Upsert(2, []float32{2, 0, 0, 0}, "quick brown fox", 0, Metadata{"lang": NewString("fr")}, nil); err != nil {
		t.Fatal(err)
	}
	docs, err := c.SearchText("fox", 10, Filter{})
	if err != nil {
		t.Fatalf("SearchText: %v", err)
	}
	if len(docs) != 2 {
		t.Fatalf("got %d docs, want 2", len(docs))
	}
	for _, d := range docs {
		if d.Content != "quick brown fox" {
			t.Fatalf("content not returned: %q", d.Content)
		}
		if d.Score <= 0 {
			t.Fatalf("expected positive BM25 score, got %v", d.Score)
		}
	}
	// Filter to lang=en.
	f := Filter{Op: FilterEq, Field: "lang", Value: NewString("en")}
	docs, err = c.SearchText("fox", 10, f)
	if err != nil {
		t.Fatalf("SearchText filtered: %v", err)
	}
	if len(docs) != 1 || docs[0].ID != 1 {
		t.Fatalf("filter not honored: %+v", docs)
	}
}

// TestHybridTextFusion checks RRF + weighted fusion against a hand-fused oracle.
func TestHybridTextFusion(t *testing.T) {
	c := newFullTextCollection(t)
	// Dense vectors crafted so the dense order differs from the text order.
	mustUpsert(t, c, 1, []float32{1, 0, 0, 0}, "fox in the henhouse")
	mustUpsert(t, c, 2, []float32{0, 1, 0, 0}, "fox fox fox everywhere")
	mustUpsert(t, c, 3, []float32{0, 0, 1, 0}, "a quiet meadow")
	h := c.idx.(*hnsw)

	dense := []float32{1, 0, 0, 0} // nearest to doc 1
	query := "fox"

	denseRes, textRes, err := h.HybridTextLanes(dense, query, 10, HybridOpts{})
	if err != nil {
		t.Fatalf("lanes: %v", err)
	}
	if len(denseRes) == 0 || len(textRes) == 0 {
		t.Fatalf("empty lanes: dense=%d text=%d", len(denseRes), len(textRes))
	}

	// RRF oracle.
	wantRRF := Fuse(denseRes, textRes, FusionRRF, 0, 0, 10)
	gotRRF, err := h.HybridText(dense, query, 10, HybridOpts{Method: FusionRRF})
	if err != nil {
		t.Fatal(err)
	}
	assertSameRanking(t, "RRF", gotRRF, wantRRF)

	// Weighted oracle.
	wantW := Fuse(denseRes, textRes, FusionWeighted, 0.5, 0, 10)
	gotW, err := h.HybridText(dense, query, 10, HybridOpts{Method: FusionWeighted, Alpha: 0.5})
	if err != nil {
		t.Fatal(err)
	}
	assertSameRanking(t, "Weighted", gotW, wantW)
}

func assertSameRanking(t *testing.T, name string, got, want []Result) {
	t.Helper()
	if len(got) != len(want) {
		t.Fatalf("%s: got %d results, want %d", name, len(got), len(want))
	}
	for i := range got {
		if got[i].ID != want[i].ID {
			t.Fatalf("%s rank %d: id got %d want %d", name, i, got[i].ID, want[i].ID)
		}
		if math.Abs(float64(got[i].Score-want[i].Score)) > 1e-5 {
			t.Fatalf("%s rank %d (id %d): score got %v want %v", name, i, got[i].ID, got[i].Score, want[i].Score)
		}
	}
}

// TestFullTextDisabledIsIdentical asserts a collection without FullText allocates
// no bm25Index and that the text APIs report disabled.
func TestFullTextDisabledIsIdentical(t *testing.T) {
	cfg := Config{Dim: 4, Metric: L2, M: 8, EfConstruction: 50, EfSearch: 32, Seed: 1}
	c, err := NewCollection("plain", cfg)
	if err != nil {
		t.Fatal(err)
	}
	h := c.idx.(*hnsw)
	if h.bm25 != nil {
		t.Fatalf("bm25 index allocated for a non-full-text collection")
	}
	if h.az != nil {
		t.Fatalf("analyzer resolved for a non-full-text collection")
	}
	if h.FullTextEnabled() {
		t.Fatalf("FullTextEnabled true for a plain collection")
	}
	// Upsert content (RAG path must still work) without any BM25 maintenance.
	if err := c.Upsert(1, []float32{1, 0, 0, 0}, "hello world", 0, nil, nil); err != nil {
		t.Fatalf("upsert on plain collection: %v", err)
	}
	if _, err := c.SearchText("hello", 10, Filter{}); err != ErrFullTextDisabled {
		t.Fatalf("SearchText on plain collection: got %v want ErrFullTextDisabled", err)
	}
	if _, err := c.HybridText([]float32{1, 0, 0, 0}, "hello", 10, HybridOpts{}); err != ErrFullTextDisabled {
		t.Fatalf("HybridText on plain collection: got %v want ErrFullTextDisabled", err)
	}
	// RAG retrieval still returns the content.
	docs, err := c.SearchDocs([]float32{1, 0, 0, 0}, 10, Filter{})
	if err != nil {
		t.Fatal(err)
	}
	if len(docs) != 1 || docs[0].Content != "hello world" {
		t.Fatalf("RAG path broken on plain collection: %+v", docs)
	}
}

// TestFullTextConfigValidation: a FullText config on a non-HNSW index, or with an
// unknown analyzer, fails loud; the default analyzer name is accepted.
func TestFullTextConfigValidation(t *testing.T) {
	bad := Config{Dim: 4, Metric: L2, M: 8, EfConstruction: 50, EfSearch: 32, Seed: 1,
		FullText: &FullTextConfig{Analyzer: "no-such-analyzer"}}
	if err := bad.Validate(); err != ErrInvalidFullText {
		t.Fatalf("unknown analyzer: got %v want ErrInvalidFullText", err)
	}
	ivf := Config{Dim: 4, Metric: L2, M: 8, EfConstruction: 50, EfSearch: 32, Seed: 1,
		IndexType: IndexIVF, FullText: &FullTextConfig{}}
	if err := ivf.Validate(); err != ErrInvalidFullText {
		t.Fatalf("IVF + FullText: got %v want ErrInvalidFullText", err)
	}
	ok := Config{Dim: 4, Metric: L2, M: 8, EfConstruction: 50, EfSearch: 32, Seed: 1,
		FullText: &FullTextConfig{Analyzer: "english", K1: 1.5, B: 0.6}}
	if err := ok.Validate(); err != nil {
		t.Fatalf("valid full-text config rejected: %v", err)
	}
}

// TestBM25CustomKnobs verifies K1/B from the config reach the index.
func TestBM25CustomKnobs(t *testing.T) {
	cfg := Config{Dim: 4, Metric: L2, M: 8, EfConstruction: 50, EfSearch: 32, Seed: 1,
		FullText: &FullTextConfig{K1: 2.0, B: 0.4}}
	h, err := newHNSW(cfg)
	if err != nil {
		t.Fatal(err)
	}
	if h.bm25.k1 != 2.0 || h.bm25.b != 0.4 {
		t.Fatalf("knobs not threaded: k1=%v b=%v", h.bm25.k1, h.bm25.b)
	}
	// Defaults when zero.
	h2 := newFullTextHNSW(t)
	if h2.bm25.k1 != defaultBM25K1 || h2.bm25.b != defaultBM25B {
		t.Fatalf("defaults wrong: k1=%v b=%v", h2.bm25.k1, h2.bm25.b)
	}
}
