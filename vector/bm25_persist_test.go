// SPDX-License-Identifier: Apache-2.0

package vector

import (
	"bytes"
	"math"
	"path/filepath"
	"testing"
)

// bm25Corpus is a tiny, deterministic full-text corpus (id -> content) used by
// the persistence round-trip tests. The terms are chosen so different ids win
// for different queries (distinct df / tf), making a rebuilt-vs-original ranking
// comparison meaningful.
var bm25Corpus = []struct {
	id      uint64
	vec     []float32
	content string
}{
	{1, []float32{1, 0, 0, 0}, "the quick brown fox jumps over the lazy dog"},
	{2, []float32{0, 1, 0, 0}, "quick foxes are clever and quick and quick"},
	{3, []float32{0, 0, 1, 0}, "a lazy dog sleeps in the warm sun"},
	{4, []float32{0, 0, 0, 1}, "brown bears and brown foxes roam the forest"},
	{5, []float32{1, 1, 0, 0}, "the meadow is green and quiet today"},
}

// rankSnapshot captures a SearchText ranking (ids + scores) for equality checks
// across a persistence round-trip.
type rankSnapshot struct {
	ids    []uint64
	scores []float32
}

func searchTextRanking(t *testing.T, c *Collection, query string, k int) rankSnapshot {
	t.Helper()
	docs, err := c.SearchText(query, k, Filter{})
	if err != nil {
		t.Fatalf("SearchText(%q): %v", query, err)
	}
	rs := rankSnapshot{}
	for _, d := range docs {
		rs.ids = append(rs.ids, d.ID)
		rs.scores = append(rs.scores, d.Score)
	}
	return rs
}

func assertSameRank(t *testing.T, what string, got, want rankSnapshot) {
	t.Helper()
	if len(got.ids) != len(want.ids) {
		t.Fatalf("%s: got %d results, want %d (ids got=%v want=%v)", what, len(got.ids), len(want.ids), got.ids, want.ids)
	}
	if len(want.ids) == 0 {
		t.Fatalf("%s: ranking is empty; test cannot distinguish a rebuilt index", what)
	}
	for i := range got.ids {
		if got.ids[i] != want.ids[i] {
			t.Fatalf("%s rank %d: id got %d want %d", what, i, got.ids[i], want.ids[i])
		}
		if math.Abs(float64(got.scores[i]-want.scores[i])) > 1e-5 {
			t.Fatalf("%s rank %d (id %d): score got %v want %v", what, i, got.ids[i], got.scores[i], want.scores[i])
		}
	}
}

// TestBM25SnapshotRestoreRebuildsRanking proves a single-collection snapshot
// save -> Restore re-derives an IDENTICAL BM25 ranking (the bm25Index is never
// serialized; it is rebuilt from each restored slot's $content) and that the
// FullTextConfig (analyzer + k1/b) survives the round-trip.
func TestBM25SnapshotRestoreRebuildsRanking(t *testing.T) {
	cfg := Config{Dim: 4, Metric: L2, M: 8, EfConstruction: 50, EfSearch: 32, Seed: 1,
		FullText: &FullTextConfig{Analyzer: "english", K1: 1.4, B: 0.6}}
	src, err := NewCollection("ft", cfg)
	if err != nil {
		t.Fatal(err)
	}
	for _, d := range bm25Corpus {
		mustUpsert(t, src, d.id, d.vec, d.content)
	}
	const query = "quick brown fox"
	before := searchTextRanking(t, src, query, 10)

	var blob bytes.Buffer
	if err := src.Snapshot(&blob); err != nil {
		t.Fatalf("Snapshot: %v", err)
	}

	// A fresh collection created with the SAME FullText config, restored from the
	// blob, must rebuild the same ranking (mirrors how RestoreAll/WAL reopen work:
	// NewCollection allocates the lane, Restore re-derives the postings).
	dst, err := NewCollection("ft", cfg)
	if err != nil {
		t.Fatal(err)
	}
	if err := dst.Restore(bytes.NewReader(blob.Bytes())); err != nil {
		t.Fatalf("Restore: %v", err)
	}
	after := searchTextRanking(t, dst, query, 10)
	assertSameRank(t, "snapshot restore", after, before)

	// A second query (different winners) to be sure the whole corpus rebuilt, not
	// just the first query's terms.
	assertSameRank(t, "snapshot restore (lazy dog)",
		searchTextRanking(t, dst, "lazy dog", 10),
		searchTextRanking(t, src, "lazy dog", 10))
}

// TestBM25InstantRestartRebuildsRanking proves the instant-restart sidecar path
// (SavePersist -> openPersist) rebuilds the BM25 lane from the persisted $content
// (the bm25Index is not in the sidecar). FullText requires a quantizer-backed
// Persistent index (HNSW-only), so QuantSQ8 + mmap files are used.
func TestBM25InstantRestartRebuildsRanking(t *testing.T) {
	dir := t.TempDir()
	cfg := Config{
		Dim: 4, Metric: Cosine, M: 8, EfConstruction: 100, EfSearch: 64, Seed: 1,
		Quant: QuantSQ8, QuantStorage: QuantMmap, MmapPath: filepath.Join(dir, "v.dat"),
		RescoreFactor: 3, GraphMmapPath: filepath.Join(dir, "g.dat"),
		FullText: &FullTextConfig{Analyzer: "english", K1: 1.3, B: 0.7},
	}
	h, err := newHNSW(cfg)
	if err != nil {
		t.Fatal(err)
	}
	for _, d := range bm25Corpus {
		// Content travels through $content metadata; the engine tokenizes it.
		if _, _, err := h.Insert(d.id, d.vec, 0, WithContent(nil, d.content), nil, nil, CASCond{}); err != nil {
			t.Fatal(err)
		}
	}
	const query = "quick brown fox"
	before := bm25HNSWRanking(t, h, query)

	metaPath := filepath.Join(dir, "meta.bin")
	if err := h.SavePersist(metaPath); err != nil {
		t.Fatalf("SavePersist: %v", err)
	}
	_ = h.Close()

	h2, err := openPersist(cfg, metaPath)
	if err != nil {
		t.Fatalf("openPersist: %v", err)
	}
	defer func() { _ = h2.Close() }()
	if h2.bm25 == nil || h2.az == nil {
		t.Fatal("instant-restart did not reconstruct the bm25 lane / analyzer")
	}
	after := bm25HNSWRanking(t, h2, query)
	assertSameRank(t, "instant restart", after, before)
}

// bm25HNSWRanking runs SearchText on a raw hnsw (admitting all live slots) and
// captures ids+scores.
func bm25HNSWRanking(t *testing.T, h *hnsw, query string) rankSnapshot {
	t.Helper()
	res := h.SearchText(query, 10, func(uint32) bool { return true })
	rs := rankSnapshot{}
	for _, r := range res {
		rs.ids = append(rs.ids, r.ID)
		rs.scores = append(rs.scores, r.Score)
	}
	return rs
}

// TestBM25WALReplayRebuildsRanking proves the store WAL recovery path (open =
// Restore-checkpoint + replay-tail) rebuilds the BM25 lane: the checkpoint's
// Restore re-derives the postings and the replayed tail maintains them
// incrementally (RestoreInsert -> insertLocked -> bm25.add).
func TestBM25WALReplayRebuildsRanking(t *testing.T) {
	dir := t.TempDir()
	cfg := Config{
		Dim: 4, Metric: Cosine, M: 8, EfConstruction: 100, EfSearch: 64, Seed: 1,
		Quant: QuantSQ8, RescoreFactor: 3, WAL: true,
		FullText: &FullTextConfig{Analyzer: "english", K1: 1.2, B: 0.75},
	}
	cs, err := OpenCollectionStore(dir)
	if err != nil {
		t.Fatal(err)
	}
	if err := cs.CreateCollection("docs", cfg); err != nil {
		t.Fatal(err)
	}
	// Insert the first half, Flush (checkpoint), then insert the rest into the WAL
	// tail so recovery exercises BOTH the Restore rebuild and the replay path.
	half := len(bm25Corpus) / 2
	for i, d := range bm25Corpus {
		if err := cs.Upsert("docs", d.id, d.vec, d.content, 0, nil, nil); err != nil {
			t.Fatal(err)
		}
		if i == half-1 {
			if err := cs.Flush("docs"); err != nil {
				t.Fatalf("Flush: %v", err)
			}
		}
	}
	const query = "quick brown fox"
	before, err := cs.SearchText("docs", query, 10, Filter{})
	if err != nil {
		t.Fatal(err)
	}
	_ = cs.Close()

	cs2, err := OpenCollectionStore(dir)
	if err != nil {
		t.Fatalf("reopen: %v", err)
	}
	defer func() { _ = cs2.Close() }()
	after, err := cs2.SearchText("docs", query, 10, Filter{})
	if err != nil {
		t.Fatalf("SearchText after WAL replay: %v", err)
	}
	if len(after) != len(before) || len(after) == 0 {
		t.Fatalf("WAL replay ranking size mismatch: got %d want %d", len(after), len(before))
	}
	for i := range after {
		if after[i].ID != before[i].ID {
			t.Fatalf("WAL replay rank %d: id got %d want %d", i, after[i].ID, before[i].ID)
		}
		if math.Abs(float64(after[i].Score-before[i].Score)) > 1e-5 {
			t.Fatalf("WAL replay rank %d: score got %v want %v", i, after[i].Score, before[i].Score)
		}
	}
	// Config survived the round-trip.
	c2, ok := cs2.Get("docs")
	if !ok {
		t.Fatal("collection missing after WAL reopen")
	}
	assertFullTextConfig(t, "wal reopen", c2.Config().FullText, cfg.FullText)
}

// TestBM25StoreSnapshotRoundTrip proves the Raft-facing SnapshotAll/RestoreAll
// store snapshot persists the FullText config and that the restored collection
// rebuilds the BM25 lane (RestoreAll -> NewCollection allocates the lane ->
// Collection.Restore re-derives the postings from $content).
func TestBM25StoreSnapshotRoundTrip(t *testing.T) {
	cfg := Config{Dim: 4, Metric: L2, M: 8, EfConstruction: 50, EfSearch: 32, Seed: 1,
		FullText: &FullTextConfig{Analyzer: "english", K1: 1.5, B: 0.4}}
	src, err := OpenCollectionStore(t.TempDir())
	if err != nil {
		t.Fatal(err)
	}
	defer src.Close()
	if err := src.CreateCollection("docs", cfg); err != nil {
		t.Fatal(err)
	}
	for _, d := range bm25Corpus {
		if err := src.Upsert("docs", d.id, d.vec, d.content, 0, nil, nil); err != nil {
			t.Fatal(err)
		}
	}
	const query = "quick brown fox"
	before, err := src.SearchText("docs", query, 10, Filter{})
	if err != nil {
		t.Fatal(err)
	}

	var blob bytes.Buffer
	if err := src.SnapshotAll(&blob); err != nil {
		t.Fatalf("SnapshotAll: %v", err)
	}

	dst, err := OpenCollectionStore(t.TempDir())
	if err != nil {
		t.Fatal(err)
	}
	defer dst.Close()
	if err := dst.RestoreAll(bytes.NewReader(blob.Bytes())); err != nil {
		t.Fatalf("RestoreAll: %v", err)
	}
	after, err := dst.SearchText("docs", query, 10, Filter{})
	if err != nil {
		t.Fatalf("SearchText after RestoreAll: %v", err)
	}
	if len(after) != len(before) || len(after) == 0 {
		t.Fatalf("RestoreAll ranking size mismatch: got %d want %d", len(after), len(before))
	}
	for i := range after {
		if after[i].ID != before[i].ID || math.Abs(float64(after[i].Score-before[i].Score)) > 1e-5 {
			t.Fatalf("RestoreAll rank %d: got (%d,%v) want (%d,%v)", i, after[i].ID, after[i].Score, before[i].ID, before[i].Score)
		}
	}
	// HybridText must also work after restore (the lane + analyzer are both live).
	if _, err := dst.HybridText("docs", []float32{1, 0, 0, 0}, query, 5, HybridOpts{}); err != nil {
		t.Fatalf("HybridText after RestoreAll: %v", err)
	}
	c2, ok := dst.Get("docs")
	if !ok {
		t.Fatal("collection missing after RestoreAll")
	}
	assertFullTextConfig(t, "store snapshot", c2.Config().FullText, cfg.FullText)
}

func assertFullTextConfig(t *testing.T, what string, got, want *FullTextConfig) {
	t.Helper()
	if got == nil {
		t.Fatalf("%s: FullText config lost (nil after round-trip)", what)
	}
	if got.Analyzer != want.Analyzer || got.K1 != want.K1 || got.B != want.B {
		t.Fatalf("%s: FullText config = %+v, want %+v", what, *got, *want)
	}
}
