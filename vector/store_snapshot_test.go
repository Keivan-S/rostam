// SPDX-License-Identifier: Apache-2.0

package vector

import (
	"bytes"
	"math/rand"
	"testing"
)

// TestStoreSnapshotRoundtrip checks SnapshotAll -> RestoreAll preserves both
// single-vector and multi-vector collections (data + searchability).
func TestStoreSnapshotRoundtrip(t *testing.T) {
	src, err := OpenCollectionStore(t.TempDir())
	if err != nil {
		t.Fatal(err)
	}
	defer src.Close()

	// A single-vector collection with content + metadata.
	if err := src.CreateCollection("docs", Config{Dim: 4, Metric: L2, M: 8, EfConstruction: 50, EfSearch: 32, Seed: 1}); err != nil {
		t.Fatal(err)
	}
	for i := 1; i <= 5; i++ {
		if err := src.Upsert("docs", uint64(i), []float32{float32(i), 0, 0, 0}, "chunk", 0, Metadata{"doc": NewInt(int64(i))}, nil); err != nil {
			t.Fatal(err)
		}
	}

	// A multi-vector collection.
	if err := src.CreateMultiVector("mv", MultiVectorConfig{Dim: 4, Seed: 1}); err != nil {
		t.Fatal(err)
	}
	rng := rand.New(rand.NewSource(3))
	for id := uint64(1); id <= 6; id++ {
		toks := randTokens(rng, 3, 4)
		if err := src.MultiAdd("mv", id, toks, Metadata{"id": NewInt(int64(id))}); err != nil {
			t.Fatal(err)
		}
	}
	mvQuery := randTokens(rng, 2, 4)
	mvBefore, _ := src.MultiSearch("mv", mvQuery, 3, MultiSearchOpts{CandidatesPerToken: 50})

	var buf bytes.Buffer
	if err := src.SnapshotAll(&buf); err != nil {
		t.Fatalf("snapshot: %v", err)
	}

	// Restore into a fresh store.
	dst, err := OpenCollectionStore(t.TempDir())
	if err != nil {
		t.Fatal(err)
	}
	defer dst.Close()
	if err := dst.RestoreAll(bytes.NewReader(buf.Bytes())); err != nil {
		t.Fatalf("restore: %v", err)
	}

	// Single-vector survived with content + metadata.
	docs, err := dst.SearchDocs("docs", []float32{1, 0, 0, 0}, 5, Filter{})
	if err != nil {
		t.Fatal(err)
	}
	if len(docs) != 5 {
		t.Fatalf("restored docs = %d, want 5", len(docs))
	}
	if docs[0].Content != "chunk" || docs[0].Metadata["doc"].Int != int64(docs[0].ID) {
		t.Errorf("restored doc lost content/metadata: %+v", docs[0])
	}

	// Multi-vector survived with identical ranking.
	mv, ok := dst.GetMultiVector("mv")
	if !ok {
		t.Fatal("multi-vector collection missing after restore")
	}
	if mv.NumDocs() != 6 {
		t.Errorf("restored mv NumDocs = %d, want 6", mv.NumDocs())
	}
	mvAfter, _ := dst.MultiSearch("mv", mvQuery, 3, MultiSearchOpts{CandidatesPerToken: 50})
	if len(mvAfter) != len(mvBefore) {
		t.Fatalf("mv search after restore = %d results, want %d", len(mvAfter), len(mvBefore))
	}
	for i := range mvBefore {
		if mvAfter[i].ID != mvBefore[i].ID {
			t.Errorf("mv rank %d: id %d after restore, want %d", i, mvAfter[i].ID, mvBefore[i].ID)
		}
	}
}

// TestStoreSnapshotAllWithNamed checks SnapshotAll -> RestoreAll round-trips ALL
// THREE families: dense, multi-vector, and named-vector collections.
func TestStoreSnapshotAllWithNamed(t *testing.T) {
	src, err := OpenCollectionStore(t.TempDir())
	if err != nil {
		t.Fatal(err)
	}
	defer src.Close()

	// Dense.
	if err := src.CreateCollection("docs", Config{Dim: 4, Metric: L2, M: 8, EfConstruction: 50, EfSearch: 32, Seed: 1}); err != nil {
		t.Fatal(err)
	}
	for i := 1; i <= 5; i++ {
		if err := src.Upsert("docs", uint64(i), []float32{float32(i), 0, 0, 0}, "chunk", 0, Metadata{"doc": NewInt(int64(i))}, nil); err != nil {
			t.Fatal(err)
		}
	}

	// Multi-vector.
	if err := src.CreateMultiVector("mv", MultiVectorConfig{Dim: 4, Seed: 1}); err != nil {
		t.Fatal(err)
	}
	rng := rand.New(rand.NewSource(7))
	for id := uint64(1); id <= 6; id++ {
		if err := src.MultiAdd("mv", id, randTokens(rng, 3, 4), Metadata{"id": NewInt(int64(id))}); err != nil {
			t.Fatal(err)
		}
	}
	mvQuery := randTokens(rng, 2, 4)
	mvBefore, _ := src.MultiSearch("mv", mvQuery, 3, MultiSearchOpts{CandidatesPerToken: 50})

	// Named-vector: two spaces, a point omitting a space, shared payload.
	namedCfg := map[string]NamedVectorParams{
		"title": {Dim: 4, Metric: Cosine},
		"image": {Dim: 3, Metric: DotProduct},
	}
	if err := src.CreateNamed("nv", namedCfg); err != nil {
		t.Fatal(err)
	}
	if err := src.NamedInsert("nv", 1, map[string][]float32{"title": {1, 0, 0, 0}, "image": {1, 0, 0}}, Metadata{"kind": NewString("a")}, 0); err != nil {
		t.Fatal(err)
	}
	if err := src.NamedInsert("nv", 2, map[string][]float32{"title": {0, 1, 0, 0}, "image": {0, 1, 0}}, Metadata{"kind": NewString("b")}, 0); err != nil {
		t.Fatal(err)
	}
	if err := src.NamedInsert("nv", 3, map[string][]float32{"title": {0, 0, 1, 0}}, Metadata{"kind": NewString("a")}, 0); err != nil {
		t.Fatal(err)
	}

	var buf bytes.Buffer
	if err := src.SnapshotAll(&buf); err != nil {
		t.Fatalf("snapshot: %v", err)
	}

	dst, err := OpenCollectionStore(t.TempDir())
	if err != nil {
		t.Fatal(err)
	}
	defer dst.Close()
	if err := dst.RestoreAll(bytes.NewReader(buf.Bytes())); err != nil {
		t.Fatalf("restore: %v", err)
	}

	// Dense survived.
	docs, err := dst.SearchDocs("docs", []float32{1, 0, 0, 0}, 5, Filter{})
	if err != nil {
		t.Fatal(err)
	}
	if len(docs) != 5 || docs[0].Content != "chunk" {
		t.Fatalf("dense restore lost data: %d docs, first=%+v", len(docs), docs[0])
	}

	// Multi-vector survived with identical ranking.
	mvAfter, _ := dst.MultiSearch("mv", mvQuery, 3, MultiSearchOpts{CandidatesPerToken: 50})
	if len(mvAfter) != len(mvBefore) {
		t.Fatalf("mv after restore = %d results, want %d", len(mvAfter), len(mvBefore))
	}
	for i := range mvBefore {
		if mvAfter[i].ID != mvBefore[i].ID {
			t.Errorf("mv rank %d: id %d, want %d", i, mvAfter[i].ID, mvBefore[i].ID)
		}
	}

	// Named survived: per-space search + shared payload (filtered).
	nc, ok := dst.GetNamed("nv")
	if !ok {
		t.Fatal("named collection missing after restore")
	}
	if nc.NumPoints() != 3 {
		t.Fatalf("named NumPoints = %d, want 3", nc.NumPoints())
	}
	titleRes, err := dst.NamedSearch("nv", "title", []float32{1, 0, 0, 0}, 5, Filter{})
	if err != nil {
		t.Fatal(err)
	}
	if len(titleRes) != 3 || titleRes[0].ID != 1 {
		t.Fatalf("named title results = %v, want 3 with id 1 first", resultIDs(titleRes))
	}
	imgRes, err := dst.NamedSearch("nv", "image", []float32{0, 1, 0}, 5, Filter{})
	if err != nil {
		t.Fatal(err)
	}
	if len(imgRes) != 2 {
		t.Fatalf("named image results = %d, want 2 (id 3 omitted image)", len(imgRes))
	}
	// Shared payload preserved → filtered search by kind=a returns ids 1,3.
	filtered, err := dst.NamedSearch("nv", "title", []float32{1, 0, 0, 0}, 10, Filter{Op: FilterEq, Field: "kind", Value: NewString("a")})
	if err != nil {
		t.Fatal(err)
	}
	if len(filtered) != 2 {
		t.Fatalf("named filtered search = %d, want 2 (ids 1,3)", len(filtered))
	}
}

// TestStoreSnapshotBackwardCompatNoNamedSection proves an OLD-format snapshot
// (dense + MV sections only, NO named section appended) restores cleanly on the
// new RestoreAll: no error, dense + MV intact, and zero named collections. The
// old format is simulated by snapshotting a store with no named collections and
// stripping the trailing nNamed=0 section word, leaving the stream ending exactly
// where the pre-named writer ended (right after the MV section).
func TestStoreSnapshotBackwardCompatNoNamedSection(t *testing.T) {
	src, err := OpenCollectionStore(t.TempDir())
	if err != nil {
		t.Fatal(err)
	}
	defer src.Close()

	if err := src.CreateCollection("docs", Config{Dim: 4, Metric: L2, M: 8, EfConstruction: 50, EfSearch: 32, Seed: 1}); err != nil {
		t.Fatal(err)
	}
	for i := 1; i <= 3; i++ {
		if err := src.Upsert("docs", uint64(i), []float32{float32(i), 0, 0, 0}, "chunk", 0, Metadata{"doc": NewInt(int64(i))}, nil); err != nil {
			t.Fatal(err)
		}
	}
	if err := src.CreateMultiVector("mv", MultiVectorConfig{Dim: 4, Seed: 1}); err != nil {
		t.Fatal(err)
	}
	rng := rand.New(rand.NewSource(11))
	for id := uint64(1); id <= 4; id++ {
		if err := src.MultiAdd("mv", id, randTokens(rng, 3, 4), Metadata{"id": NewInt(int64(id))}); err != nil {
			t.Fatal(err)
		}
	}
	mvQuery := randTokens(rng, 2, 4)
	mvBefore, _ := src.MultiSearch("mv", mvQuery, 3, MultiSearchOpts{CandidatesPerToken: 50})

	var buf bytes.Buffer
	if err := src.SnapshotAll(&buf); err != nil {
		t.Fatalf("snapshot: %v", err)
	}
	// Strip the trailing named-section word (nNamed=0, a 4-byte u32). With zero
	// named collections that is the ONLY thing the new writer appended after the
	// MV section, so the truncated stream is byte-identical to an old-format
	// snapshot that never wrote the section.
	raw := buf.Bytes()
	old := raw[:len(raw)-4]

	dst, err := OpenCollectionStore(t.TempDir())
	if err != nil {
		t.Fatal(err)
	}
	defer dst.Close()
	if err := dst.RestoreAll(bytes.NewReader(old)); err != nil {
		t.Fatalf("restore old-format snapshot: %v", err)
	}

	// Dense intact.
	docs, err := dst.SearchDocs("docs", []float32{1, 0, 0, 0}, 3, Filter{})
	if err != nil {
		t.Fatal(err)
	}
	if len(docs) != 3 {
		t.Fatalf("dense after old restore = %d docs, want 3", len(docs))
	}
	// MV intact.
	mvAfter, _ := dst.MultiSearch("mv", mvQuery, 3, MultiSearchOpts{CandidatesPerToken: 50})
	if len(mvAfter) != len(mvBefore) {
		t.Fatalf("mv after old restore = %d, want %d", len(mvAfter), len(mvBefore))
	}
	// Zero named collections.
	if _, ok := dst.GetNamed("anything"); ok {
		t.Fatal("unexpected named collection after old-format restore")
	}

	// A genuinely TRUNCATED stream (1-3 stray bytes in the nNamed word) is NOT the
	// old format — it must error, not silently drop named collections. Stripping
	// only 2 of the 4 nNamed bytes leaves a partial u32 → io.ErrUnexpectedEOF.
	truncated := raw[:len(raw)-2]
	dst2, err := OpenCollectionStore(t.TempDir())
	if err != nil {
		t.Fatal(err)
	}
	defer dst2.Close()
	if err := dst2.RestoreAll(bytes.NewReader(truncated)); err == nil {
		t.Fatal("restore of a truncated snapshot (partial nNamed word) must error, got nil")
	}
}

// TestStoreSnapshotFilterFirstRelativeBP asserts that a collection created with a
// non-zero FilterFirstRelativeBP restores with the same value (the non-zero persist
// path; the omitempty default-off is already covered by field inspection).
// Covers both the dense-collection and multi-vector snapshot paths.
func TestStoreSnapshotFilterFirstRelativeBP(t *testing.T) {
	const bp = 5000
	src, err := OpenCollectionStore(t.TempDir())
	if err != nil {
		t.Fatal(err)
	}
	defer src.Close()

	// Dense collection with bp=5000.
	if err := src.CreateCollection("docs", Config{
		Dim: 4, Metric: L2, M: 8, EfConstruction: 50, EfSearch: 32, Seed: 1,
		FilterFirstRelativeBP: bp,
	}); err != nil {
		t.Fatal(err)
	}

	// Multi-vector collection with bp=5000.
	if err := src.CreateMultiVector("mv", MultiVectorConfig{
		Dim: 4, Seed: 1, FilterFirstRelativeBP: bp,
	}); err != nil {
		t.Fatal(err)
	}

	var buf bytes.Buffer
	if err := src.SnapshotAll(&buf); err != nil {
		t.Fatalf("snapshot: %v", err)
	}

	dst, err := OpenCollectionStore(t.TempDir())
	if err != nil {
		t.Fatal(err)
	}
	defer dst.Close()
	if err := dst.RestoreAll(bytes.NewReader(buf.Bytes())); err != nil {
		t.Fatalf("restore: %v", err)
	}

	// Dense: FilterFirstRelativeBP must survive the round-trip.
	dc, ok := dst.Get("docs")
	if !ok {
		t.Fatal("dense collection missing after restore")
	}
	if dc.Config().FilterFirstRelativeBP != bp {
		t.Errorf("dense FilterFirstRelativeBP = %d after restore, want %d", dc.Config().FilterFirstRelativeBP, bp)
	}

	// Multi-vector: FilterFirstRelativeBP must survive the round-trip.
	mv, ok := dst.GetMultiVector("mv")
	if !ok {
		t.Fatal("multi-vector collection missing after restore")
	}
	if mv.Config().FilterFirstRelativeBP != bp {
		t.Errorf("MV FilterFirstRelativeBP = %d after restore, want %d", mv.Config().FilterFirstRelativeBP, bp)
	}
}
