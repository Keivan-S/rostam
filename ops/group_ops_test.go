// SPDX-License-Identifier: Apache-2.0

package ops

import (
	"testing"

	"github.com/rostamlabs/rostam/cache"
	"github.com/rostamlabs/rostam/vector"
)

func TestGroupSearchArgsRoundtrip(t *testing.T) {
	f := vector.Filter{Op: vector.FilterGte, Field: "doc", Value: vector.NewInt(2)}
	opts := vector.GroupOpts{GroupBy: "doc", GroupSize: 3, FetchK: 128, Filter: f}
	name, k, q, got, err := DecodeGroupSearchArgs(EncodeGroupSearchArgs("acme/docs", 5, []float32{1, 2, 3}, opts))
	if err != nil {
		t.Fatal(err)
	}
	if name != "acme/docs" || k != 5 || len(q) != 3 || q[2] != 3 {
		t.Errorf("scalar roundtrip = %q k=%d q=%v", name, k, q)
	}
	if got.GroupBy != "doc" || got.GroupSize != 3 || got.FetchK != 128 {
		t.Errorf("opts roundtrip = %+v", got)
	}
	if got.Filter.Field != "doc" || got.Filter.Value.Int != 2 {
		t.Errorf("filter roundtrip = %+v", got.Filter)
	}
}

func TestGroupSearchArgsNoFilter(t *testing.T) {
	_, _, _, got, err := DecodeGroupSearchArgs(EncodeGroupSearchArgs("c", 2, []float32{0}, vector.GroupOpts{GroupBy: "g"}))
	if err != nil {
		t.Fatal(err)
	}
	if !got.Filter.IsZero() {
		t.Errorf("expected zero filter, got %+v", got.Filter)
	}
}

func TestGroupsCodecRoundtrip(t *testing.T) {
	groups := []vector.Group{
		{Key: vector.NewInt(1), Hits: []vector.Document{
			{ID: 1, Distance: 0.5, Content: "a", Metadata: vector.Metadata{"doc": vector.NewInt(1)}},
			{ID: 2, Distance: 0.9, Content: "b"},
		}},
		{Key: vector.NewString("xyz"), Hits: []vector.Document{{ID: 3, Distance: 1.5, Content: "c"}}},
	}
	got, err := DecodeGroups(EncodeGroups(groups))
	if err != nil {
		t.Fatal(err)
	}
	if len(got) != 2 {
		t.Fatalf("len = %d, want 2", len(got))
	}
	if got[0].Key.Int != 1 || got[1].Key.Str != "xyz" {
		t.Errorf("keys = %+v, %+v", got[0].Key, got[1].Key)
	}
	if len(got[0].Hits) != 2 || got[0].Hits[1].ID != 2 || got[0].Hits[0].Content != "a" {
		t.Errorf("group0 hits = %+v", got[0].Hits)
	}
	if len(got[1].Hits) != 1 || got[1].Hits[0].ID != 3 {
		t.Errorf("group1 hits = %+v", got[1].Hits)
	}
}

// TestHandleVectorGroupCandidates verifies that handleVectorGroupCandidates
// returns the same top-fetchK candidate documents as Collection.GroupCandidates
// directly, confirming wire encode/decode round-trips correctly.
func TestHandleVectorGroupCandidates(t *testing.T) {
	dir := t.TempDir()
	c, _ := cache.New(cache.DefaultConfig())
	defer c.Close()
	vstore, err := vector.OpenCollectionStore(dir)
	if err != nil {
		t.Fatal(err)
	}
	defer vstore.Close()
	tx := NewTxContextWithVectors(c, vstore)

	cfg := vector.Config{Dim: 3, M: 8, EfConstruction: 50, EfSearch: 32, Seed: 1, Metric: vector.L2}
	if _, err := handleVectorCreateCollection(tx, EncodeCreateCollectionArgs("docs", cfg)); err != nil {
		t.Fatal(err)
	}
	// 6 chunks, two per document, increasing distance from the origin.
	for i := 1; i <= 6; i++ {
		doc := int64((i + 1) / 2)
		args := EncodeVectorUpsertArgs("docs", uint64(i), []float32{float32(i), 0, 0}, "chunk", 0,
			vector.Metadata{"doc": vector.NewInt(doc)}, vector.SparseVector{})
		if _, err := handleVectorUpsert(tx, args); err != nil {
			t.Fatalf("upsert %d: %v", i, err)
		}
	}

	query := []float32{0, 0, 0}
	opts := vector.GroupOpts{GroupBy: "doc", GroupSize: 2, FetchK: 50}
	body, err := handleVectorGroupCandidates(tx, EncodeGroupSearchArgs("docs", 3, query, opts))
	if err != nil {
		t.Fatal(err)
	}
	got, err := DecodeVectorDocs(body)
	if err != nil {
		t.Fatal(err)
	}
	col, ok := tx.vectors.Acquire("docs")
	if !ok {
		t.Fatal("acquire docs")
	}
	defer col.Release()
	want, err := col.GroupCandidates(query, opts)
	if err != nil {
		t.Fatal(err)
	}
	if len(got) != len(want) {
		t.Fatalf("candidates len got %d want %d", len(got), len(want))
	}
	for i := range want {
		if got[i].ID != want[i].ID || got[i].Distance != want[i].Distance {
			t.Fatalf("cand %d: got %+v want %+v", i, got[i], want[i])
		}
	}
	if len(got) == 0 {
		t.Fatal("expected non-empty candidates")
	}
}

// TestGroupSearchViaDispatch drives the full wire path: upsert chunks across
// documents, then vector_search_groups returns the top documents (best chunk
// each) with content.
func TestGroupSearchViaDispatch(t *testing.T) {
	dir := t.TempDir()
	c, _ := cache.New(cache.DefaultConfig())
	defer c.Close()
	vstore, err := vector.OpenCollectionStore(dir)
	if err != nil {
		t.Fatal(err)
	}
	defer vstore.Close()
	tx := NewTxContextWithVectors(c, vstore)

	cfg := vector.Config{Dim: 3, M: 8, EfConstruction: 50, EfSearch: 32, Seed: 1, Metric: vector.L2}
	if _, err := handleVectorCreateCollection(tx, EncodeCreateCollectionArgs("docs", cfg)); err != nil {
		t.Fatal(err)
	}
	// 6 chunks, two per document, increasing distance from the origin.
	for i := 1; i <= 6; i++ {
		doc := int64((i + 1) / 2)
		args := EncodeVectorUpsertArgs("docs", uint64(i), []float32{float32(i), 0, 0}, "chunk", 0,
			vector.Metadata{"doc": vector.NewInt(doc)}, vector.SparseVector{})
		if _, err := handleVectorUpsert(tx, args); err != nil {
			t.Fatalf("upsert %d: %v", i, err)
		}
	}

	body, err := handleVectorSearchGroups(tx, EncodeGroupSearchArgs("docs", 2, []float32{0, 0, 0},
		vector.GroupOpts{GroupBy: "doc", GroupSize: 1}))
	if err != nil {
		t.Fatal(err)
	}
	groups, err := DecodeGroups(body)
	if err != nil {
		t.Fatal(err)
	}
	if len(groups) != 2 {
		t.Fatalf("got %d groups, want 2", len(groups))
	}
	if groups[0].Key.Int != 1 || groups[1].Key.Int != 2 {
		t.Errorf("group keys = %d,%d, want 1,2", groups[0].Key.Int, groups[1].Key.Int)
	}
	if len(groups[0].Hits) != 1 || groups[0].Hits[0].ID != 1 || groups[0].Hits[0].Content != "chunk" {
		t.Errorf("group0 = %+v", groups[0].Hits)
	}
}
