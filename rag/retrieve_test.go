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
	hits, err := Retrieve(ctx, r, nil, "docs", "epoll transport", 5, false, -1)
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
	hits, err := Retrieve(ctx, r, emb, "docs", "alpha beta", 5, false, -1)
	if err != nil {
		t.Fatal(err)
	}
	if len(hits) == 0 {
		t.Fatalf("dense retrieve returned nothing")
	}
}

type recordingRetriever struct {
	Retriever
	hybridCalled, searchCalled bool
}

func (r *recordingRetriever) HybridSearch(ctx context.Context, c, qt string, qv []float32, k int, a float64) ([]Hit, error) {
	r.hybridCalled = true
	return []Hit{{Content: "x", Source: "s"}}, nil
}

func (r *recordingRetriever) Search(ctx context.Context, c, qt string, qv []float32, k int) ([]Hit, error) {
	r.searchCalled = true
	return []Hit{{Content: "y", Source: "s"}}, nil
}

func TestRetrieveRoutesToHybridWhenEnabled(t *testing.T) {
	rr := &recordingRetriever{}
	emb := semcache.NewStubEmbedder("stub", 8)
	_, _ = Retrieve(context.Background(), rr, emb, "docs", "q", 5, true, -1)
	if !rr.hybridCalled || rr.searchCalled {
		t.Fatalf("hybrid=true+embedder should call HybridSearch only (hybrid=%v search=%v)", rr.hybridCalled, rr.searchCalled)
	}
	rr2 := &recordingRetriever{}
	_, _ = Retrieve(context.Background(), rr2, emb, "docs", "q", 5, false, -1)
	if rr2.hybridCalled || !rr2.searchCalled {
		t.Fatalf("hybrid=false should call Search only")
	}
	rr3 := &recordingRetriever{}
	_, _ = Retrieve(context.Background(), rr3, nil, "docs", "q", 5, true, -1)
	if rr3.hybridCalled || !rr3.searchCalled {
		t.Fatalf("no embedder should call Search (BM25) even if hybrid=true")
	}
}
