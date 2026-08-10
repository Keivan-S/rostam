// SPDX-License-Identifier: Apache-2.0

package ops

import (
	"testing"

	"github.com/rostamlabs/rostam/cache"
	"github.com/rostamlabs/rostam/vector"
)

func TestVectorDocsCodecRoundtrip(t *testing.T) {
	docs := []vector.Document{
		{ID: 1, Distance: 0.5, Score: 1.5, Content: "hello world", Metadata: vector.Metadata{"src": vector.NewString("a"), "n": vector.NewInt(3)}},
		{ID: 2, Distance: 1.25, Content: "", Metadata: nil},
		{ID: 3, Distance: 2, Content: "a longer chunk of text", Metadata: vector.Metadata{"ok": vector.NewBool(true)}},
	}
	got, err := DecodeVectorDocs(EncodeVectorDocs(docs))
	if err != nil {
		t.Fatal(err)
	}
	if len(got) != len(docs) {
		t.Fatalf("len = %d, want %d", len(got), len(docs))
	}
	for i, d := range got {
		if d.ID != docs[i].ID || d.Content != docs[i].Content || d.Distance != docs[i].Distance || d.Score != docs[i].Score {
			t.Errorf("doc %d = %+v, want %+v", i, d, docs[i])
		}
		if len(d.Metadata) != len(docs[i].Metadata) {
			t.Errorf("doc %d metadata = %v, want %v", i, d.Metadata, docs[i].Metadata)
		}
	}
}

func TestDeleteByFilterArgsRoundtrip(t *testing.T) {
	f := vector.Filter{Op: vector.FilterEq, Field: "doc", Value: vector.NewInt(7)}
	name, got, err := DecodeDeleteByFilterArgs(EncodeDeleteByFilterArgs("acme/docs", f))
	if err != nil {
		t.Fatal(err)
	}
	if name != "acme/docs" || got.Field != "doc" || got.Value.Int != 7 {
		t.Errorf("roundtrip = %q %+v", name, got)
	}
}

// TestRAGOpsViaDispatch drives the full wire path through the op handlers:
// upsert (content folded into metadata), search_docs (returns content), and
// delete_by_filter (purges + returns count).
func TestRAGOpsViaDispatch(t *testing.T) {
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

	for i := 1; i <= 6; i++ {
		args := EncodeVectorUpsertArgs("docs", uint64(i), []float32{float32(i), 0, 0}, "chunk", 0,
			vector.Metadata{"doc": vector.NewInt(int64(i % 2))}, vector.SparseVector{})
		if _, err := handleVectorUpsert(tx, args); err != nil {
			t.Fatalf("upsert %d: %v", i, err)
		}
	}

	// search_docs returns content.
	body, err := handleVectorSearchDocs(tx, EncodeVectorSearchArgs("docs", 3, []float32{1, 0, 0}))
	if err != nil {
		t.Fatal(err)
	}
	docs, err := DecodeVectorDocs(body)
	if err != nil {
		t.Fatal(err)
	}
	if len(docs) == 0 || docs[0].Content != "chunk" {
		t.Errorf("search_docs = %+v", docs)
	}

	// delete_by_filter purges doc==1 (ids 1,3,5) and returns the count.
	df := EncodeDeleteByFilterArgs("docs", vector.Filter{Op: vector.FilterEq, Field: "doc", Value: vector.NewInt(1)})
	cbody, err := handleVectorDeleteByFilter(tx, df)
	if err != nil {
		t.Fatal(err)
	}
	n, err := DecodeDeleteByFilterResult(cbody)
	if err != nil {
		t.Fatal(err)
	}
	if n != 3 {
		t.Errorf("deleted %d, want 3", n)
	}
}
