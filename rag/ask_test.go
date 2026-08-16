// SPDX-License-Identifier: Apache-2.0

package rag

import (
	"context"
	"encoding/json"
	"io"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"
)

func TestBuildPromptNumbersContextAndCites(t *testing.T) {
	hits := []Hit{
		{Content: "epoll loop count = GOMAXPROCS", Source: "a.md", Index: 0},
		{Content: "raft shards ~= cores", Source: "b.md", Index: 1},
	}
	sys, user := BuildPrompt("how many loops?", hits)
	if !strings.Contains(sys, "[n]") && !strings.Contains(sys, "cite") {
		t.Fatalf("system prompt should instruct citation: %q", sys)
	}
	if !strings.Contains(user, "[1]") || !strings.Contains(user, "[2]") {
		t.Fatalf("user prompt should number chunks: %q", user)
	}
	if !strings.Contains(user, "how many loops?") {
		t.Fatalf("user prompt should carry the question: %q", user)
	}
}

func TestAskCallsLLMWithContext(t *testing.T) {
	var gotBody string
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, req *http.Request) {
		b, _ := io.ReadAll(req.Body)
		gotBody = string(b)
		_ = json.NewEncoder(w).Encode(map[string]any{
			"choices": []map[string]any{{"message": map[string]any{"role": "assistant", "content": "42 loops [1]"}}},
		})
	}))
	defer srv.Close()

	dir := t.TempDir()
	r, err := NewEmbeddedRetriever(dir)
	if err != nil {
		t.Fatal(err)
	}
	defer func() { _ = r.Close() }()
	ctx := context.Background()
	_ = r.EnsureCorpus(ctx, "docs", 0)
	_ = r.Upsert(ctx, "docs", []StoredChunk{{ID: 1, Content: "epoll loop count equals gomaxprocs", Source: "a.md", Index: 0}})

	res, err := Ask(ctx, r, nil, LLMConfig{URL: srv.URL, Model: "test"}, "docs", "how many loops?", 5, false, -1)
	if err != nil {
		t.Fatal(err)
	}
	if !strings.Contains(res.Answer, "42 loops") {
		t.Fatalf("answer=%q", res.Answer)
	}
	if !strings.Contains(gotBody, "epoll loop count") {
		t.Fatalf("LLM request should include retrieved context, got %q", gotBody)
	}
	if len(res.Hits) != 1 || res.Hits[0].Source != "a.md" {
		t.Fatalf("hits not returned for source mapping: %+v", res.Hits)
	}
}
