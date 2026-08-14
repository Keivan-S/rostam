// SPDX-License-Identifier: Apache-2.0

package mcp

import (
	"strings"
	"testing"
)

func TestCreateCollectionUpsertSearchGet(t *testing.T) {
	c := startServer(t, Config{Store: newHeapStore(t)})
	c.initialize()

	var created struct {
		Created string `json:"created"`
	}
	c.callTool("create_collection", map[string]any{"name": "docs", "dim": 4}, &created, false)
	if created.Created != "docs" {
		t.Fatalf("create_collection: got %+v", created)
	}

	c.callTool("upsert", map[string]any{
		"collection": "docs",
		"id":         uint64(1),
		"vector":     []float32{1, 0, 0, 0},
		"content":    "red fox jumps",
		"metadata":   map[string]any{"lang": "en"},
	}, nil, false)
	c.callTool("upsert", map[string]any{
		"collection": "docs",
		"id":         uint64(2),
		"vector":     []float32{0, 1, 0, 0},
		"content":    "blue whale swims",
		"metadata":   map[string]any{"lang": "fr"},
	}, nil, false)

	var textRes struct {
		Hits []struct {
			ID       uint64         `json:"id"`
			Content  string         `json:"content"`
			Metadata map[string]any `json:"metadata"`
		} `json:"hits"`
	}
	c.callTool("search", map[string]any{"collection": "docs", "mode": "text", "query_text": "fox"}, &textRes, false)
	if len(textRes.Hits) != 1 || textRes.Hits[0].ID != 1 {
		t.Fatalf("text search: %+v", textRes.Hits)
	}
	if textRes.Hits[0].Metadata["lang"] != "en" {
		t.Fatalf("text search metadata: %+v", textRes.Hits[0].Metadata)
	}

	var denseRes struct {
		Hits []struct {
			ID uint64 `json:"id"`
		} `json:"hits"`
	}
	c.callTool("search", map[string]any{"collection": "docs", "mode": "dense", "vector": []float32{1, 0, 0, 0}, "k": 1}, &denseRes, false)
	if len(denseRes.Hits) != 1 || denseRes.Hits[0].ID != 1 {
		t.Fatalf("dense search: %+v", denseRes.Hits)
	}

	var got struct {
		Points []struct {
			ID      uint64    `json:"id"`
			Vector  []float32 `json:"vector"`
			Content string    `json:"content"`
		} `json:"points"`
		Missing []uint64 `json:"missing"`
	}
	c.callTool("get", map[string]any{"collection": "docs", "ids": []uint64{1, 999}, "with_vector": true}, &got, false)
	if len(got.Points) != 1 || got.Points[0].ID != 1 || got.Points[0].Content != "red fox jumps" {
		t.Fatalf("get: %+v", got.Points)
	}
	if len(got.Points[0].Vector) != 4 || got.Points[0].Vector[0] != 1 {
		t.Fatalf("get vector: %+v", got.Points[0].Vector)
	}
	if len(got.Missing) != 1 || got.Missing[0] != 999 {
		t.Fatalf("get missing: %+v", got.Missing)
	}
}

// TestSearchResponseHasDistanceKey guards the generic search tool's hit
// shape {id, content, score, distance, metadata}: unlike the memory tools'
// memoryHit, search's searchHit type always carries a distance field, across
// all three modes since they share docsToHits/hybridDocs.
func TestSearchResponseHasDistanceKey(t *testing.T) {
	c := startServer(t, Config{Store: newHeapStore(t), Embedder: fakeEmbedder{}})
	c.initialize()
	c.callTool("create_collection", map[string]any{"name": "docs", "dim": 4}, nil, false)
	c.callTool("upsert", map[string]any{"collection": "docs", "id": uint64(1), "vector": []float32{1, 0, 0, 0}, "content": "a: fox jumps"}, nil, false)

	for _, mode := range []string{"text", "dense", "hybrid"} {
		var res struct {
			Hits []map[string]any `json:"hits"`
		}
		c.callTool("search", map[string]any{"collection": "docs", "mode": mode, "query_text": "fox", "vector": []float32{1, 0, 0, 0}}, &res, false)
		if len(res.Hits) != 1 {
			t.Fatalf("mode %s: expected 1 hit, got %+v", mode, res.Hits)
		}
		if _, ok := res.Hits[0]["distance"]; !ok {
			t.Fatalf("mode %s: search hit should carry a distance key: %+v", mode, res.Hits[0])
		}
	}
}

func TestUpsertRefusesMemoryCollection(t *testing.T) {
	c := startServer(t, Config{Store: newHeapStore(t)})
	c.initialize()
	msg := c.callTool("upsert", map[string]any{"collection": memCollection, "id": uint64(1), "vector": []float32{1, 0, 0, 0}}, nil, true)
	if !strings.Contains(msg, "memory") {
		t.Fatalf("error should mention the memory tools, got %q", msg)
	}
}

func TestCreateCollectionRefusesMemoryName(t *testing.T) {
	c := startServer(t, Config{Store: newHeapStore(t)})
	c.initialize()
	msg := c.callTool("create_collection", map[string]any{"name": memCollection, "dim": 4}, nil, true)
	if !strings.Contains(msg, "memory") {
		t.Fatalf("error should mention the memory tools, got %q", msg)
	}
}

func TestUpsertRequiresVectorOrEmbedder(t *testing.T) {
	c := startServer(t, Config{Store: newHeapStore(t)})
	c.initialize()
	c.callTool("create_collection", map[string]any{"name": "docs", "dim": 4}, nil, false)
	msg := c.callTool("upsert", map[string]any{"collection": "docs", "id": uint64(1), "content": "no vector, no embedder"}, nil, true)
	if !strings.Contains(msg, "embedder") {
		t.Fatalf("error should mention embedder, got %q", msg)
	}
}

func TestSearchFilterTaggedForm(t *testing.T) {
	c := startServer(t, Config{Store: newHeapStore(t)})
	c.initialize()
	c.callTool("create_collection", map[string]any{"name": "docs", "dim": 4}, nil, false)
	c.callTool("upsert", map[string]any{"collection": "docs", "id": uint64(1), "vector": []float32{1, 0, 0, 0}, "content": "english doc", "metadata": map[string]any{"lang": "en"}}, nil, false)
	c.callTool("upsert", map[string]any{"collection": "docs", "id": uint64(2), "vector": []float32{1, 0, 0, 0}, "content": "french doc", "metadata": map[string]any{"lang": "fr"}}, nil, false)

	filter := map[string]any{
		"op":    "eq",
		"field": "lang",
		"value": map[string]any{"kind": "string", "str": "en"},
	}
	var res struct {
		Hits []struct {
			ID uint64 `json:"id"`
		} `json:"hits"`
	}
	c.callTool("search", map[string]any{"collection": "docs", "mode": "dense", "vector": []float32{1, 0, 0, 0}, "k": 10, "filter": filter}, &res, false)
	if len(res.Hits) != 1 || res.Hits[0].ID != 1 {
		t.Fatalf("filtered search: %+v", res.Hits)
	}
}

func TestSearchHybridUsesEmbedder(t *testing.T) {
	c := startServer(t, Config{Store: newHeapStore(t), Embedder: fakeEmbedder{}})
	c.initialize()
	c.callTool("create_collection", map[string]any{"name": "docs", "dim": 4}, nil, false)
	c.callTool("upsert", map[string]any{"collection": "docs", "id": uint64(1), "content": "a: dense-close fact"}, nil, false)
	c.callTool("upsert", map[string]any{"collection": "docs", "id": uint64(2), "content": "b: dense-far fact"}, nil, false)

	var res struct {
		Hits []struct {
			Content string `json:"content"`
		} `json:"hits"`
	}
	// query "a..." shares NO BM25 tokens with either doc; only the dense side ranks it.
	c.callTool("search", map[string]any{"collection": "docs", "mode": "hybrid", "query_text": "a unrelated words", "k": 1}, &res, false)
	if len(res.Hits) != 1 || !strings.HasPrefix(res.Hits[0].Content, "a:") {
		t.Fatalf("hybrid search did not use dense side: %+v", res.Hits)
	}
}

func TestSearchDefaultModeNoEmbedder(t *testing.T) {
	c := startServer(t, Config{Store: newHeapStore(t)})
	c.initialize()
	c.callTool("create_collection", map[string]any{"name": "docs", "dim": 4}, nil, false)
	c.callTool("upsert", map[string]any{"collection": "docs", "id": uint64(1), "vector": []float32{1, 0, 0, 0}, "content": "fox jumps"}, nil, false)

	var res struct {
		Hits []struct {
			ID uint64 `json:"id"`
		} `json:"hits"`
	}
	// no mode specified: no embedder configured, so default mode is "text".
	c.callTool("search", map[string]any{"collection": "docs", "query_text": "fox"}, &res, false)
	if len(res.Hits) != 1 {
		t.Fatalf("default text-mode search: %+v", res.Hits)
	}
}

func TestGetAllowsMemoryCollectionRead(t *testing.T) {
	c := startServer(t, Config{Store: newHeapStore(t)})
	c.initialize()
	var r struct {
		ID uint64 `json:"id"`
	}
	c.callTool("remember", map[string]any{"content": "hello"}, &r, false)

	var got struct {
		Points []struct {
			ID       uint64         `json:"id"`
			Metadata map[string]any `json:"metadata"`
		} `json:"points"`
	}
	c.callTool("get", map[string]any{"collection": memCollection, "ids": []uint64{r.ID}}, &got, false)
	if len(got.Points) != 1 {
		t.Fatalf("get on mcp_memory: %+v", got.Points)
	}
	if _, ok := got.Points[0].Metadata[nsField]; !ok {
		t.Fatalf("expected internal __ns field unstripped on generic get: %+v", got.Points[0].Metadata)
	}
}
