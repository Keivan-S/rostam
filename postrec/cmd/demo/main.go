// SPDX-License-Identifier: Apache-2.0

// Command demo runs the typed-client "next post to read" recommender end-to-end.
//
//	export OPENAI_API_KEY=sk-...
//	export ROSTAM_SERVERS=127.0.0.1:7000   # native protocol addr(s), comma-separated
//	go run ./postrec/cmd/demo
package main

import (
	"context"
	"fmt"
	"log"
	"os"
	"strings"
	"time"

	"github.com/rostamlabs/rostam/postrec"
)

func main() {
	ctx, cancel := context.WithTimeout(context.Background(), 2*time.Minute)
	defer cancel()

	openaiKey := os.Getenv("OPENAI_API_KEY")
	if openaiKey == "" {
		log.Fatal("set OPENAI_API_KEY")
	}
	servers := strings.Split(getenv("ROSTAM_SERVERS", "127.0.0.1:7000"), ",")

	// text-embedding-3-small -> 1536 dims. Ingest and query MUST use the same model.
	embedder := postrec.NewOpenAIEmbedder(openaiKey, "text-embedding-3-small", 1536)
	store, err := postrec.NewStore(servers, os.Getenv("ROSTAM_AUTH_TOKEN"), "posts")
	if err != nil {
		log.Fatalf("connect: %v", err)
	}
	defer store.Close()

	rec := postrec.NewRecommender(store, embedder)

	if err := rec.EnsureCollection(ctx); err != nil {
		log.Printf("create collection (ok if it already exists): %v", err)
	}

	posts := []postrec.Post{
		{ID: 1, Title: "Getting started with Go modules", Body: "How to manage dependencies with go.mod and go.sum.", Tags: []string{"go", "tooling"}},
		{ID: 2, Title: "Goroutines and channels explained", Body: "Concurrency in Go using goroutines, channels and select.", Tags: []string{"go", "concurrency"}},
		{ID: 3, Title: "Building a REST API in Go", Body: "Use net/http and the standard library to serve JSON.", Tags: []string{"go", "web"}},
		{ID: 4, Title: "Vector databases 101", Body: "Nearest-neighbor search, embeddings and ANN indexes.", Tags: []string{"ml", "search"}},
		{ID: 5, Title: "Hybrid search with BM25 and embeddings", Body: "Fuse keyword BM25 scores with dense vector similarity.", Tags: []string{"ml", "search"}},
		{ID: 6, Title: "Sourdough for beginners", Body: "Feeding a starter and baking your first loaf.", Tags: []string{"cooking"}},
	}
	if err := rec.Ingest(ctx, posts...); err != nil {
		log.Fatalf("ingest: %v", err)
	}
	fmt.Printf("ingested %d posts\n\n", len(posts))
	time.Sleep(1 * time.Second) // let the index build in a demo

	titles := map[uint64]string{}
	for _, p := range posts {
		titles[p.ID] = p.Title
	}

	related, err := rec.Related(ctx, posts[3], 3)
	if err != nil {
		log.Fatalf("related: %v", err)
	}
	fmt.Printf("Related to %q:\n", posts[3].Title)
	printRecs(related, titles)

	history := []uint64{1, 2}
	feed, err := rec.ForUser(ctx, history, 3)
	if err != nil {
		log.Fatalf("for user: %v", err)
	}
	fmt.Printf("\nUp next for a reader of Go posts (hybrid):\n")
	printRecs(feed, titles)

	serverFeed, err := rec.ForUserServerSide(ctx, history, 3)
	if err != nil {
		log.Fatalf("for user (server-side): %v", err)
	}
	fmt.Printf("\nUp next (server-side recommend leaf, dense-only):\n")
	printRecs(serverFeed, titles)
}

func printRecs(recs []postrec.Recommendation, titles map[uint64]string) {
	for i, r := range recs {
		fmt.Printf("  %d. [%d] %-45s score=%.4f\n", i+1, r.ID, titles[r.ID], r.Score)
	}
}

func getenv(k, def string) string {
	if v := os.Getenv(k); v != "" {
		return v
	}
	return def
}
