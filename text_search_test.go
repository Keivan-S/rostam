// SPDX-License-Identifier: Apache-2.0

package rostam

import (
	"context"
	"testing"

	"github.com/rostamlabs/rostam/ops"
	"github.com/rostamlabs/rostam/vector"
)

// ftDocs is a tiny, hand-readable corpus where the term "fox" is rare (one doc)
// and "the"/"dog" are common — so BM25 ranks the fox doc highest for a "fox"
// query and the idf direction is obvious by eye.
var ftDocs = map[uint64]string{
	1: "the quick brown fox jumps over the lazy dog",
	2: "the lazy dog sleeps all day in the sun",
	3: "a dog and another dog play in the park",
	4: "the cat sat on the mat near the window",
	5: "brown bears and brown foxes roam the brown forest",
	6: "machine learning models rank documents by relevance",
	7: "the dog barks at the mailman every single morning",
	8: "quick thinking saved the day for the clever team",
}

// denseFor is a deterministic dense vector per id so the dense lane (and hybrid)
// can score every doc; it is independent of the text content.
func denseFor(id uint64) []float32 {
	return []float32{
		float32(id)*0.01 + 0.1,
		float32(id%5)*0.2 + 0.05,
		float32(id%7)*0.13 + 0.02,
		float32(id%3)*0.31 + 0.07,
	}
}

// seedFullTextCollection creates a P-partition (or P==1) full-text collection on
// a single embedded node and upserts ftDocs (dense vector + $content). Each
// physical partition keeps its OWN bm25Index/local corpus, so P>1 exercises the
// per-shard-local-IDF path.
func seedFullTextCollection(t *testing.T, s Store, coll string, P int) {
	t.Helper()
	ctx := context.Background()
	cfg := VectorConfig{
		Dim: 4, M: 8, EfConstruction: 50, EfSearch: 32, Seed: 1, Metric: vector.L2,
		Partitions: P,
		FullText:   &vector.FullTextConfig{Analyzer: "english"},
	}
	if err := s.CreateCollection(ctx, coll, cfg); err != nil {
		t.Fatalf("create full-text collection (P=%d): %v", P, err)
	}
	for id, content := range ftDocs {
		if err := s.VectorUpsert(ctx, coll, id, denseFor(id), content, VectorInsertOpts{}); err != nil {
			t.Fatalf("upsert %d: %v", id, err)
		}
	}
}

func resultIDs(res []VectorResult) []uint64 {
	out := make([]uint64, len(res))
	for i, r := range res {
		out[i] = r.ID
	}
	return out
}

// TestEmbeddedSearchTextDispatch is the single-shard dispatch oracle: SearchText
// over an embedded P=1 collection returns the fox doc (id 1) first for a "fox"
// query (rare term → high idf), each hit carries its $content, and a non-matching
// query returns nothing.
func TestEmbeddedSearchTextDispatch(t *testing.T) {
	s := newSingleEmbedded(t)
	waitLeaderEmbedded(t, s)
	ctx := context.Background()
	seedFullTextCollection(t, s, "ft", 1)

	docs, _, err := s.VectorSearchText(ctx, "ft", "fox", 5, VectorSearchOpts{})
	if err != nil {
		t.Fatalf("SearchText fox: %v", err)
	}
	if len(docs) == 0 {
		t.Fatal("SearchText fox: no results")
	}
	if docs[0].ID != 1 {
		t.Fatalf("SearchText fox: top id = %d, want 1 (the only fox doc)", docs[0].ID)
	}
	if docs[0].Content != ftDocs[1] {
		t.Fatalf("SearchText fox: content = %q, want %q", docs[0].Content, ftDocs[1])
	}
	// BM25 scores are descending (higher = more relevant).
	for i := 1; i < len(docs); i++ {
		if docs[i].Score > docs[i-1].Score {
			t.Fatalf("SearchText fox: scores not descending at %d: %v > %v", i, docs[i].Score, docs[i-1].Score)
		}
	}

	// A query with no indexed term returns no documents (clean empty, not an error).
	none, _, err := s.VectorSearchText(ctx, "ft", "zzzznotaword", 5, VectorSearchOpts{})
	if err != nil {
		t.Fatalf("SearchText miss: %v", err)
	}
	if len(none) != 0 {
		t.Fatalf("SearchText miss: got %d docs, want 0", len(none))
	}
}

// TestEmbeddedHybridTextDispatch: HybridText fuses the dense + BM25 lanes and
// returns the fox doc for a "fox" query+vector over an embedded P=1 collection.
func TestEmbeddedHybridTextDispatch(t *testing.T) {
	s := newSingleEmbedded(t)
	waitLeaderEmbedded(t, s)
	ctx := context.Background()
	seedFullTextCollection(t, s, "ft", 1)

	res, _, err := s.VectorHybridText(ctx, "ft", denseFor(1), "fox", 5, VectorHybridOpts{Method: FusionRRF})
	if err != nil {
		t.Fatalf("HybridText: %v", err)
	}
	if len(res) == 0 {
		t.Fatal("HybridText: no results")
	}
	// id 1 is both the closest dense vector (queried with denseFor(1)) AND the only
	// fox doc, so RRF fusion must place it first.
	if res[0].ID != 1 {
		t.Fatalf("HybridText: top id = %d, want 1", res[0].ID)
	}
}

// TestSearchTextDisabledCollection: SearchText / HybridText on a collection
// created WITHOUT full-text surfaces ErrFullTextDisabled (cleanly, not a panic).
func TestSearchTextDisabledCollection(t *testing.T) {
	s := newSingleEmbedded(t)
	waitLeaderEmbedded(t, s)
	ctx := context.Background()
	cfg := VectorConfig{Dim: 4, M: 8, EfConstruction: 50, EfSearch: 32, Seed: 1, Metric: vector.L2}
	if err := s.CreateCollection(ctx, "plain", cfg); err != nil {
		t.Fatal(err)
	}
	if err := s.VectorUpsert(ctx, "plain", 1, denseFor(1), "hello world", VectorInsertOpts{}); err != nil {
		t.Fatal(err)
	}
	if _, _, err := s.VectorSearchText(ctx, "plain", "hello", 5, VectorSearchOpts{}); err == nil {
		t.Fatal("SearchText on plain collection: want error, got nil")
	}
	if _, _, err := s.VectorHybridText(ctx, "plain", denseFor(1), "hello", 5, VectorHybridOpts{}); err == nil {
		t.Fatal("HybridText on plain collection: want error, got nil")
	}
}

// allInOnePartition returns ids (from ftDocs) that all hash to the SAME partition
// under P partitions — the "single-shard placement" the design-doc caveat names,
// where each partition's local corpus == the global corpus for the touched
// partition, so partitioned BM25 == single-node BM25 EXACTLY.
func allInOnePartition(P int) (part int, ids []uint64) {
	// Pick the partition that the most ftDocs ids land in; that gives the richest
	// single-partition corpus to compare against P=1.
	buckets := map[int][]uint64{}
	for id := range ftDocs {
		p := ops.PartitionOf(id, P)
		buckets[p] = append(buckets[p], id)
	}
	best := -1
	for p, list := range buckets {
		if best < 0 || len(list) > len(buckets[best]) {
			best = p
		}
		_ = list
	}
	return best, buckets[best]
}

// TestSearchTextFanOutSingleShardOracle is the partition-invariance oracle for the
// SINGLE-SHARD placement (the only placement where partitioned BM25 == single-node
// BM25, per the documented per-shard-local-IDF caveat). It seeds a P=1 collection
// and a P=4 collection with the SAME docs, queries a term whose matching docs ALL
// live in ONE partition of the P=4 collection, and asserts the returned ids + the
// global top-k are identical. Because every matching doc shares one partition, that
// partition's local corpus IS the global corpus for those docs, so local IDF ==
// global IDF and the fan-out result is bit-identical to P=1.
func TestSearchTextFanOutSingleShardOracle(t *testing.T) {
	const P = 4
	part, ids := allInOnePartition(P)
	if len(ids) < 2 {
		t.Fatalf("single-partition corpus too small (part %d has %d ids)", part, len(ids))
	}
	ctx := context.Background()

	s1 := newSingleEmbedded(t)
	waitLeaderEmbedded(t, s1)
	seedSubset(t, s1, "ft1", 1, ids)

	sP := newSingleEmbedded(t)
	waitLeaderEmbedded(t, sP)
	seedSubset(t, sP, "ftP", P, ids)

	// Confirm the chosen ids really all land in one partition of the P=4 layout.
	for _, id := range ids {
		if ops.PartitionOf(id, P) != part {
			t.Fatalf("id %d not in partition %d", id, part)
		}
	}

	// Query a term present in the subset corpus.
	q := "dog"
	d1, _, err := s1.VectorSearchText(ctx, "ft1", q, 10, VectorSearchOpts{})
	if err != nil {
		t.Fatalf("P1 SearchText: %v", err)
	}
	dP, fm, err := sP.VectorSearchText(ctx, "ftP", q, 10, VectorSearchOpts{})
	if err != nil {
		t.Fatalf("P4 SearchText: %v", err)
	}
	if fm.Degraded {
		t.Fatalf("P4 SearchText degraded unexpectedly")
	}
	if !equalIDs(docIDs(dP), docIDs(d1)) {
		t.Fatalf("single-shard fan-out != P1:\n P4=%v\n P1=%v", docIDs(dP), docIDs(d1))
	}
	// Scores must match too (local corpus == global corpus for this placement).
	for i := range d1 {
		if dP[i].Score != d1[i].Score {
			t.Fatalf("single-shard score mismatch at %d: P4=%v P1=%v", i, dP[i].Score, d1[i].Score)
		}
	}
}

// seedSubset upserts only the given ids (a subset of ftDocs) into a fresh
// full-text collection.
func seedSubset(t *testing.T, s Store, coll string, P int, ids []uint64) {
	t.Helper()
	ctx := context.Background()
	cfg := VectorConfig{
		Dim: 4, M: 8, EfConstruction: 50, EfSearch: 32, Seed: 1, Metric: vector.L2,
		Partitions: P,
		FullText:   &vector.FullTextConfig{Analyzer: "english"},
	}
	if err := s.CreateCollection(ctx, coll, cfg); err != nil {
		t.Fatalf("create subset collection (P=%d): %v", P, err)
	}
	for _, id := range ids {
		if err := s.VectorUpsert(ctx, coll, id, denseFor(id), ftDocs[id], VectorInsertOpts{}); err != nil {
			t.Fatalf("upsert %d: %v", id, err)
		}
	}
}

func equalIDs(a, b []uint64) bool {
	if len(a) != len(b) {
		return false
	}
	for i := range a {
		if a[i] != b[i] {
			return false
		}
	}
	return true
}

// TestSearchTextFanOutMultiShardSanity exercises the GENUINELY multi-shard
// placement (docs span all 4 partitions). Per the documented per-shard-local-IDF
// caveat, scores are APPROXIMATE here (NOT bit-equal to a single-node corpus), so
// this asserts RANKING SANITY rather than equality: the rare-term query "fox"
// must still return the (only) fox doc, and "dog" (common) must return the
// dog-bearing docs — the right documents come back and the result is non-degraded.
func TestSearchTextFanOutMultiShardSanity(t *testing.T) {
	const P = 4
	ctx := context.Background()
	s := newSingleEmbedded(t)
	waitLeaderEmbedded(t, s)
	seedFullTextCollection(t, s, "ft", P)

	// Confirm the corpus genuinely spans all partitions (multi-shard, not single).
	touched := map[int]bool{}
	for id := range ftDocs {
		touched[ops.PartitionOf(id, P)] = true
	}
	if len(touched) < 2 {
		t.Fatalf("corpus only touches %d partitions; not a multi-shard test", len(touched))
	}

	// Rare term: only doc 1 contains "fox" (doc 5 has "foxes" → stems to "fox" too).
	fox, fm, err := s.VectorSearchText(ctx, "ft", "fox", 10, VectorSearchOpts{})
	if err != nil {
		t.Fatalf("multi-shard SearchText fox: %v", err)
	}
	if fm.Degraded {
		t.Fatalf("multi-shard SearchText degraded unexpectedly")
	}
	gotFox := map[uint64]bool{}
	for _, d := range fox {
		gotFox[d.ID] = true
	}
	if !gotFox[1] {
		t.Fatalf("multi-shard fox: doc 1 missing from %v", docIDs(fox))
	}
	// Only fox-bearing docs (1 and 5) may appear.
	for _, d := range fox {
		if d.ID != 1 && d.ID != 5 {
			t.Fatalf("multi-shard fox: unexpected doc %d in %v", d.ID, docIDs(fox))
		}
	}

	// Common term: the dog-bearing docs (1,2,3,7) must all surface.
	dog, _, err := s.VectorSearchText(ctx, "ft", "dog", 10, VectorSearchOpts{})
	if err != nil {
		t.Fatalf("multi-shard SearchText dog: %v", err)
	}
	for _, want := range []uint64{1, 2, 3, 7} {
		found := false
		for _, d := range dog {
			if d.ID == want {
				found = true
			}
		}
		if !found {
			t.Fatalf("multi-shard dog: doc %d missing from %v", want, docIDs(dog))
		}
	}
}

// TestFullTextAllTransports drives search_text + hybrid_text over ALL THREE
// transports (embedded, direct, and networked client→TCP→server) against a
// freshly seeded full-text collection on each, asserting the fox doc ranks first
// for the rare-term query everywhere. This is the cross-transport consistency
// guard (one engine, three wire paths).
func TestFullTextAllTransports(t *testing.T) {
	ctx := context.Background()

	// Distinct collection names per transport: newSingleDirect / NewDirectServer use
	// an in-memory store whose default catalog can persist across sub-tests in a run,
	// so a shared "ft" name would collide ("collection already exists").
	t.Run("embedded", func(t *testing.T) {
		s := newSingleEmbedded(t)
		waitLeaderEmbedded(t, s)
		seedFullTextCollection(t, s, "ft_emb", 1)
		assertFullText(t, ctx, s, "ft_emb")
	})

	t.Run("direct", func(t *testing.T) {
		// Own DataDir (NOT newSingleDirect's default catalog, which is process-shared
		// and would leak "ft_dir" across runs/tests → "collection already exists").
		reg := ops.NewRegistry()
		if err := ops.RegisterBuiltins(reg); err != nil {
			t.Fatal(err)
		}
		s, err := NewDirect(DirectConfig{DataDir: t.TempDir(), Ops: reg, Cache: CacheConfig{NumShardsPerNode: 1}})
		if err != nil {
			t.Fatalf("NewDirect: %v", err)
		}
		defer s.Close()
		seedFullTextCollection(t, s, "ft_dir", 1)
		assertFullText(t, ctx, s, "ft_dir")
	})

	t.Run("networked", func(t *testing.T) {
		reg := ops.NewRegistry()
		if err := ops.RegisterBuiltins(reg); err != nil {
			t.Fatal(err)
		}
		srv, err := NewDirectServer("127.0.0.1:0", DirectConfig{
			DataDir: t.TempDir(),
			Ops:     reg,
			Cache:   CacheConfig{NumShardsPerNode: 4},
		})
		if err != nil {
			t.Fatal(err)
		}
		defer srv.Close()
		store, err := NewClient(ClientConfig{Servers: []string{srv.Addr()}})
		if err != nil {
			t.Fatal(err)
		}
		defer store.Close()
		seedFullTextCollection(t, store, "ft_net", 1)
		assertFullText(t, ctx, store, "ft_net")
	})
}

// assertFullText runs the shared search_text + hybrid_text assertions used by each
// transport sub-test.
func assertFullText(t *testing.T, ctx context.Context, s Store, coll string) {
	t.Helper()
	docs, _, err := s.VectorSearchText(ctx, coll, "fox", 5, VectorSearchOpts{})
	if err != nil {
		t.Fatalf("SearchText: %v", err)
	}
	if len(docs) == 0 || docs[0].ID != 1 {
		t.Fatalf("SearchText: top = %v, want id 1 first", docIDs(docs))
	}
	if docs[0].Content != ftDocs[1] {
		t.Fatalf("SearchText: content = %q, want %q", docs[0].Content, ftDocs[1])
	}
	res, _, err := s.VectorHybridText(ctx, coll, denseFor(1), "fox", 5, VectorHybridOpts{Method: FusionRRF})
	if err != nil {
		t.Fatalf("HybridText: %v", err)
	}
	if len(res) == 0 || res[0].ID != 1 {
		t.Fatalf("HybridText: top = %v, want id 1 first", resultIDs(res))
	}
}
