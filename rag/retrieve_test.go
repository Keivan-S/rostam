package rag

import (
	"context"
	"testing"

	"github.com/rostamlabs/rostam/semcache"
)

func TestRetrieveBM25(t *testing.T) {
	dir := t.TempDir()
	r, err := NewEmbeddedRetriever(dir)
	if err != nil {
		t.Fatal(err)
	}
	defer func() { _ = r.Close() }()
	ctx := context.Background()
	_ = r.EnsureCorpus(ctx, "docs", 0)
	_ = r.Upsert(ctx, "docs", []StoredChunk{
		{ID: 1, Content: "epoll event loop transport", Source: "a.md", Index: 0},
		{ID: 2, Content: "raft shard core count tuning", Source: "b.md", Index: 0},
	})
	hits, err := Retrieve(ctx, r, nil, "docs", "epoll transport", 5)
	if err != nil {
		t.Fatal(err)
	}
	if len(hits) == 0 || hits[0].Source != "a.md" {
		t.Fatalf("got %+v", hits)
	}
}

func TestRetrieveDenseUsesEmbedder(t *testing.T) {
	dir := t.TempDir()
	r, err := NewEmbeddedRetriever(dir)
	if err != nil {
		t.Fatal(err)
	}
	defer func() { _ = r.Close() }()
	ctx := context.Background()
	emb := semcache.NewStubEmbedder("stub", 8)
	_ = r.EnsureCorpus(ctx, "docs", emb.Dim())
	vecs, _ := emb.Embed(ctx, []string{"alpha beta"})
	_ = r.Upsert(ctx, "docs", []StoredChunk{{ID: 1, Content: "alpha beta", Vector: vecs[0], Source: "a.md", Index: 0}})
	hits, err := Retrieve(ctx, r, emb, "docs", "alpha beta", 5)
	if err != nil {
		t.Fatal(err)
	}
	if len(hits) == 0 {
		t.Fatalf("dense retrieve returned nothing")
	}
}
