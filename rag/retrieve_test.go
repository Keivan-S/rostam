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

// recordingRetriever records each Search call's queryVec nil-ness, since
// HybridSearch no longer exists on Retriever: fusion now drives Retrieve to
// call Search twice (once dense, once BM25) instead of routing to a separate
// method.
type recordingRetriever struct {
	Retriever
	denseVecCalls []bool // one entry per Search call; true = queryVec was non-nil
}

func (r *recordingRetriever) Search(ctx context.Context, c, qt string, qv []float32, k int) ([]Hit, error) {
	r.denseVecCalls = append(r.denseVecCalls, len(qv) > 0)
	return []Hit{{Content: "y", Source: "s"}}, nil
}

func TestRetrieveRoutesToHybridWhenEnabled(t *testing.T) {
	rr := &recordingRetriever{}
	emb := semcache.NewStubEmbedder("stub", 8)
	_, _ = Retrieve(context.Background(), rr, emb, "docs", "q", 5, WithHybrid(-1))
	if len(rr.denseVecCalls) != 2 || rr.denseVecCalls[0] != true || rr.denseVecCalls[1] != false {
		t.Fatalf("WithHybrid+embedder should call Search twice, once dense (vec) then once BM25 (nil): %v", rr.denseVecCalls)
	}

	rr2 := &recordingRetriever{}
	_, _ = Retrieve(context.Background(), rr2, emb, "docs", "q", 5)
	if len(rr2.denseVecCalls) != 1 || rr2.denseVecCalls[0] != true {
		t.Fatalf("no WithHybrid should call Search once, dense: %v", rr2.denseVecCalls)
	}

	rr3 := &recordingRetriever{}
	_, _ = Retrieve(context.Background(), rr3, nil, "docs", "q", 5, WithHybrid(-1))
	if len(rr3.denseVecCalls) != 1 || rr3.denseVecCalls[0] != false {
		t.Fatalf("no embedder should call Search once, BM25, even with WithHybrid: %v", rr3.denseVecCalls)
	}
}
