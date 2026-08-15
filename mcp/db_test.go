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

// TestGetRefusesMemoryCollection and TestSearchRefusesMemoryCollection cover
// the reads that used to be allowed through. Neither generic reader applies
// the __ns filter that recall/list_memories do, so both were a way to read
// every namespace's memories at once — get additionally handed back the
// internal __ns metadata naming the namespace each one belongs to. Namespaces
// are the memory subsystem's only isolation boundary; the generic readers must
// not be a hole in it.
func TestGetRefusesMemoryCollection(t *testing.T) {
	c := startServer(t, Config{Store: newHeapStore(t)})
	c.initialize()
	var r struct {
		ID uint64 `json:"id"`
	}
	c.callTool("remember", map[string]any{"content": "hello", "namespace": "private"}, &r, false)

	msg := c.callTool("get", map[string]any{"collection": memCollection, "ids": []uint64{r.ID}}, nil, true)
	if !strings.Contains(msg, "memory") {
		t.Fatalf("error should mention the memory tools, got %q", msg)
	}
	if strings.Contains(msg, "hello") || strings.Contains(msg, nsField) {
		t.Fatalf("rejection must not leak the memory or its namespace field: %q", msg)
	}
}

func TestSearchRefusesMemoryCollection(t *testing.T) {
	c := startServer(t, Config{Store: newHeapStore(t)})
	c.initialize()
	c.callTool("remember", map[string]any{"content": "the deploy password lives in vault", "namespace": "private"}, nil, false)

	msg := c.callTool("search", map[string]any{"collection": memCollection, "query_text": "deploy password"}, nil, true)
	if !strings.Contains(msg, "memory") {
		t.Fatalf("error should mention the memory tools, got %q", msg)
	}
	if strings.Contains(msg, "vault") {
		t.Fatalf("rejection must not leak the memory contents: %q", msg)
	}
}

// TestDestructiveToolsAbsentByDefault guards the registration gate itself:
// delete/delete_by_filter must not appear in tools/list at all (not merely
// refuse when called) unless Config.Destructive is set.
func TestDestructiveToolsAbsentByDefault(t *testing.T) {
	c := startServer(t, Config{Store: newHeapStore(t)})
	c.initialize()
	names := c.toolNames()
	for _, want := range []string{"delete", "delete_by_filter"} {
		for _, n := range names {
			if n == want {
				t.Fatalf("tool %q should be absent when Destructive is false; got %v", want, names)
			}
		}
	}
}

func TestDestructiveToolsPresentWhenEnabled(t *testing.T) {
	c := startServer(t, Config{Store: newHeapStore(t), Destructive: true})
	c.initialize()
	names := c.toolNames()
	for _, want := range []string{"delete", "delete_by_filter"} {
		found := false
		for _, n := range names {
			if n == want {
				found = true
			}
		}
		if !found {
			t.Fatalf("tool %q should be present when Destructive is true; got %v", want, names)
		}
	}
}

func TestDeleteRemovesPoint(t *testing.T) {
	c := startServer(t, Config{Store: newHeapStore(t), Destructive: true})
	c.initialize()
	c.callTool("create_collection", map[string]any{"name": "docs", "dim": 4}, nil, false)
	c.callTool("upsert", map[string]any{"collection": "docs", "id": uint64(1), "vector": []float32{1, 0, 0, 0}, "content": "fox"}, nil, false)
	c.callTool("upsert", map[string]any{"collection": "docs", "id": uint64(2), "vector": []float32{0, 1, 0, 0}, "content": "whale"}, nil, false)

	var res struct {
		Deleted []uint64 `json:"deleted"`
		Missing []uint64 `json:"missing"`
	}
	c.callTool("delete", map[string]any{"collection": "docs", "ids": []uint64{1, 999}}, &res, false)
	if len(res.Deleted) != 1 || res.Deleted[0] != 1 {
		t.Fatalf("delete deleted: %+v", res.Deleted)
	}
	if len(res.Missing) != 1 || res.Missing[0] != 999 {
		t.Fatalf("delete missing: %+v", res.Missing)
	}

	var got struct {
		Points  []struct{ ID uint64 } `json:"points"`
		Missing []uint64              `json:"missing"`
	}
	c.callTool("get", map[string]any{"collection": "docs", "ids": []uint64{1, 2}}, &got, false)
	if len(got.Points) != 1 || got.Points[0].ID != 2 {
		t.Fatalf("expected only id 2 to remain: %+v", got.Points)
	}
	if len(got.Missing) != 1 || got.Missing[0] != 1 {
		t.Fatalf("expected id 1 reported missing after delete: %+v", got.Missing)
	}
}

// TestDeleteReportsPartialFailure guards handleDelete's continue-past-errors
// behavior (mirroring forget's approach in memory.go): a mid-batch
// VectorDelete failure must not discard the outcome of ids processed before
// or after it. Uses the same failDeleteStore wrapper as forget's partial-
// failure test (memory_test.go) to inject a deterministic failure.
func TestDeleteReportsPartialFailure(t *testing.T) {
	failing := &failDeleteStore{Store: newHeapStore(t)}
	c := startServer(t, Config{Store: failing, Destructive: true})
	c.initialize()
	c.callTool("create_collection", map[string]any{"name": "docs", "dim": 4}, nil, false)
	c.callTool("upsert", map[string]any{"collection": "docs", "id": uint64(1), "vector": []float32{1, 0, 0, 0}, "content": "fox"}, nil, false)
	c.callTool("upsert", map[string]any{"collection": "docs", "id": uint64(2), "vector": []float32{0, 1, 0, 0}, "content": "whale"}, nil, false)

	failing.failID = 2

	var res struct {
		Deleted []uint64 `json:"deleted"`
		Missing []uint64 `json:"missing"`
		Errors  []string `json:"errors"`
	}
	c.callTool("delete", map[string]any{"collection": "docs", "ids": []uint64{1, 2}}, &res, false)
	if len(res.Deleted) != 1 || res.Deleted[0] != 1 {
		t.Fatalf("expected id 1 to have deleted despite id 2 failing: %+v", res)
	}
	if len(res.Errors) != 1 || !strings.Contains(res.Errors[0], "2") {
		t.Fatalf("expected an error naming id 2: %+v", res)
	}
}

func TestDeleteByFilterDeletesOnlyMatches(t *testing.T) {
	c := startServer(t, Config{Store: newHeapStore(t), Destructive: true})
	c.initialize()
	c.callTool("create_collection", map[string]any{"name": "docs", "dim": 4}, nil, false)
	c.callTool("upsert", map[string]any{"collection": "docs", "id": uint64(1), "vector": []float32{1, 0, 0, 0}, "content": "english doc", "metadata": map[string]any{"lang": "en"}}, nil, false)
	c.callTool("upsert", map[string]any{"collection": "docs", "id": uint64(2), "vector": []float32{0, 1, 0, 0}, "content": "french doc", "metadata": map[string]any{"lang": "fr"}}, nil, false)

	filter := map[string]any{
		"op":    "eq",
		"field": "lang",
		"value": map[string]any{"kind": "string", "str": "en"},
	}
	var res struct {
		DeletedCount int `json:"deleted_count"`
	}
	c.callTool("delete_by_filter", map[string]any{"collection": "docs", "filter": filter}, &res, false)
	if res.DeletedCount != 1 {
		t.Fatalf("delete_by_filter deleted_count: %+v", res)
	}

	var got struct {
		Points  []struct{ ID uint64 } `json:"points"`
		Missing []uint64              `json:"missing"`
	}
	c.callTool("get", map[string]any{"collection": "docs", "ids": []uint64{1, 2}}, &got, false)
	if len(got.Points) != 1 || got.Points[0].ID != 2 {
		t.Fatalf("expected only id 2 (fr) to remain: %+v", got.Points)
	}
}

func TestDeleteByFilterRefusesMatchAll(t *testing.T) {
	c := startServer(t, Config{Store: newHeapStore(t), Destructive: true})
	c.initialize()
	c.callTool("create_collection", map[string]any{"name": "docs", "dim": 4}, nil, false)
	c.callTool("upsert", map[string]any{"collection": "docs", "id": uint64(1), "vector": []float32{1, 0, 0, 0}, "content": "fox"}, nil, false)

	msg := c.callTool("delete_by_filter", map[string]any{"collection": "docs", "filter": map[string]any{}}, nil, true)
	if !strings.Contains(msg, "match-all") {
		t.Fatalf("error should mention match-all, got %q", msg)
	}
}

func TestDeleteRefusesMemoryCollection(t *testing.T) {
	c := startServer(t, Config{Store: newHeapStore(t), Destructive: true})
	c.initialize()
	msg := c.callTool("delete", map[string]any{"collection": memCollection, "ids": []uint64{1}}, nil, true)
	if !strings.Contains(msg, "memory") {
		t.Fatalf("error should mention the memory tools, got %q", msg)
	}
}

func TestDeleteByFilterRefusesMemoryCollection(t *testing.T) {
	c := startServer(t, Config{Store: newHeapStore(t), Destructive: true})
	c.initialize()
	filter := map[string]any{
		"op":    "eq",
		"field": "lang",
		"value": map[string]any{"kind": "string", "str": "en"},
	}
	msg := c.callTool("delete_by_filter", map[string]any{"collection": memCollection, "filter": filter}, nil, true)
	if !strings.Contains(msg, "memory") {
		t.Fatalf("error should mention the memory tools, got %q", msg)
	}
}
