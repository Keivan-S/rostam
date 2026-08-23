// SPDX-License-Identifier: Apache-2.0
package client

import (
	"context"
	"errors"
	"testing"

	"github.com/rostamlabs/rostam/vtypes"
)

// TestSearchOnMissingCollection confirms Search surfaces the server's
// distinguishable missing-collection error ("ops: unknown collection %q", from
// handleVectorSearch's Acquire) as ErrCollectionNotFound.
func TestSearchOnMissingCollection(t *testing.T) {
	addr, stop := startTestStack(t)
	defer stop()
	c, err := New(Config{Servers: []string{addr}})
	if err != nil {
		t.Fatal(err)
	}
	defer func() { _ = c.Close() }()

	col := c.Collection("never-created")
	_, err = col.Search(context.Background(), SearchRequest{Query: []float32{1, 0, 0, 0}, K: 5})
	if !errors.Is(err, ErrCollectionNotFound) {
		t.Fatalf("Search on never-created collection = %v, want ErrCollectionNotFound", err)
	}
}

// TestSearchDocsOnMissingCollection confirms SearchDocs maps the server's
// missing-collection error onto ErrCollectionNotFound, same as Search.
func TestSearchDocsOnMissingCollection(t *testing.T) {
	addr, stop := startTestStack(t)
	defer stop()
	c, err := New(Config{Servers: []string{addr}})
	if err != nil {
		t.Fatal(err)
	}
	defer func() { _ = c.Close() }()

	col := c.Collection("never-created")
	_, err = col.SearchDocs(context.Background(), SearchDocsRequest{Query: []float32{1, 0, 0, 0}, K: 5})
	if !errors.Is(err, ErrCollectionNotFound) {
		t.Fatalf("SearchDocs on never-created collection = %v, want ErrCollectionNotFound", err)
	}
}

func seedSearchable(t *testing.T) (*Collection, func()) {
	t.Helper()
	addr, stop := startTestStack(t)
	c, err := New(Config{Servers: []string{addr}})
	if err != nil {
		stop()
		t.Fatal(err)
	}
	col := c.Collection("posts")
	ctx := context.Background()
	if err := col.Create(ctx, CreateRequest{
		Dim: 4, Metric: vtypes.Cosine,
		FullText: &vtypes.FullTextConfig{Analyzer: "english"},
	}); err != nil {
		_ = c.Close()
		stop()
		t.Fatal(err)
	}
	pts := []PointInput{
		{ID: 1, Vector: []float32{1, 0, 0, 0}, Content: "golang concurrency",
			Metadata: vtypes.Metadata{"tag": vtypes.NewString("lang")}},
		{ID: 2, Vector: []float32{0, 1, 0, 0}, Content: "vector search engines",
			Metadata: vtypes.Metadata{"tag": vtypes.NewString("search")}},
		{ID: 3, Vector: []float32{0, 0, 1, 0}, Content: "bm25 keyword ranking",
			Metadata: vtypes.Metadata{"tag": vtypes.NewString("search")}},
	}
	if errs := col.UpsertBatch(ctx, pts); len(errs) != 0 {
		_ = c.Close()
		stop()
		t.Fatalf("seed: %+v", errs)
	}
	return col, func() { _ = c.Close(); stop() }
}

func TestSearchAndHybridText(t *testing.T) {
	col, cleanup := seedSearchable(t)
	defer cleanup()
	ctx := context.Background()

	resp, err := col.Search(ctx, SearchRequest{Query: []float32{1, 0, 0, 0}, K: 2})
	if err != nil {
		t.Fatalf("Search: %v", err)
	}
	if len(resp.Results) == 0 || resp.Results[0].ID != 1 {
		t.Fatalf("Search results = %+v, want id 1 first", resp.Results)
	}

	hr, err := col.HybridText(ctx, HybridTextRequest{
		Dense: []float32{0, 0, 1, 0}, Text: "keyword ranking", K: 2,
	})
	if err != nil {
		t.Fatalf("HybridText: %v", err)
	}
	if len(hr.Results) == 0 {
		t.Fatal("HybridText returned no results")
	}
}

func TestSearchDocs(t *testing.T) {
	col, cleanup := seedSearchable(t)
	defer cleanup()
	ctx := context.Background()

	resp, err := col.SearchDocs(ctx, SearchDocsRequest{Query: []float32{1, 0, 0, 0}, K: 1})
	if err != nil {
		t.Fatalf("SearchDocs: %v", err)
	}
	if len(resp.Documents) == 0 {
		t.Fatal("SearchDocs returned no documents")
	}
	if resp.Documents[0].ID != 1 {
		t.Fatalf("SearchDocs top doc id = %d, want 1", resp.Documents[0].ID)
	}
	if resp.Documents[0].Content != "golang concurrency" {
		t.Fatalf("SearchDocs content = %q, want %q", resp.Documents[0].Content, "golang concurrency")
	}
}

func TestSearchGroups(t *testing.T) {
	col, cleanup := seedSearchable(t)
	defer cleanup()
	ctx := context.Background()

	resp, err := col.SearchGroups(ctx, GroupSearchRequest{
		Query: []float32{0, 1, 0, 0}, K: 5, GroupBy: "tag", GroupSize: 2, FetchK: 10,
	})
	if err != nil {
		t.Fatalf("SearchGroups: %v", err)
	}
	seen := map[string]bool{}
	for _, g := range resp.Groups {
		if seen[g.Key] {
			t.Fatalf("duplicate group key %q, groups = %+v", g.Key, resp.Groups)
		}
		seen[g.Key] = true
		if len(g.Hits) == 0 {
			t.Fatalf("group %q has no hits", g.Key)
		}
	}
	if len(resp.Groups) != 2 {
		t.Fatalf("SearchGroups groups = %+v, want 2 distinct tag groups (lang, search)", resp.Groups)
	}
	if !seen["lang"] || !seen["search"] {
		t.Fatalf("SearchGroups keys = %+v, want lang and search", seen)
	}
}
