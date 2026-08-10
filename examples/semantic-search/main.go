// SPDX-License-Identifier: Apache-2.0

// Command semantic-search is a minimal-but-real Rostam integration: it connects to
// a running rostam-server over TCP (rostam.NewClient), turns text into vectors with
// a hosted embedding API (OpenAI), upserts a small document set, and runs a
// semantic search — the end-to-end shape a real project uses.
//
// Run:
//
//	# 1. start a server (separate terminal)
//	go run ./cmd/rostam-server -tcp 127.0.0.1:9400 -data ./.rostam-data
//
//	# 2. run this example
//	export OPENAI_API_KEY=sk-...
//	go run ./examples/semantic-search                 # default query
//	go run ./examples/semantic-search "how do I cancel my subscription?"
//
// Env:
//
//	OPENAI_API_KEY   required — your embeddings API key
//	ROSTAM_ADDR      server TCP address (default 127.0.0.1:9400)
package main

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"hash/fnv"
	"log"
	"math"
	"net/http"
	"os"
	"sort"
	"strings"
	"time"
	"unicode"

	"github.com/rostamlabs/rostam"
	"github.com/rostamlabs/rostam/ops"
	"github.com/rostamlabs/rostam/vector"
)

const (
	collection    = "docs"
	embedModel    = "text-embedding-3-small"
	embedDim      = 1536 // text-embedding-3-small output dimension; MUST equal the collection Dim
	embedEndpoint = "https://api.openai.com/v1/embeddings"
)

// sample corpus: a real project loads this from your own data source.
var corpus = []struct {
	text     string
	category string
}{
	{"To cancel your subscription, open Settings → Billing and click Cancel plan.", "billing"},
	{"Reset your password from the login page using the 'Forgot password?' link.", "account"},
	{"We offer a 30-day money-back guarantee on all annual plans.", "billing"},
	{"Export your data as CSV or JSON from the Account → Data page.", "account"},
	{"Our API rate limit is 1000 requests per minute on the Pro tier.", "api"},
	{"Webhooks are delivered at-least-once; make your handlers idempotent.", "api"},
}

func main() {
	apiKey := os.Getenv("OPENAI_API_KEY")
	if apiKey == "" {
		log.Fatal("set OPENAI_API_KEY")
	}
	addr := os.Getenv("ROSTAM_ADDR")
	if addr == "" {
		addr = "127.0.0.1:9400"
	}
	query := "how do I stop paying?"
	if len(os.Args) > 1 {
		query = strings.Join(os.Args[1:], " ")
	}

	ctx, cancel := context.WithTimeout(context.Background(), 60*time.Second)
	defer cancel()

	// The client needs an op registry matching the server's (smart routing).
	reg := ops.NewRegistry()
	if err := ops.RegisterBuiltins(reg); err != nil {
		log.Fatalf("register ops: %v", err)
	}
	store, err := rostam.NewClient(rostam.ClientConfig{Servers: []string{addr}, Ops: reg})
	if err != nil {
		log.Fatalf("connect %s: %v", addr, err)
	}
	defer func() { _ = store.Close() }()

	// Create the collection (idempotent: ignore "already exists" on re-runs).
	// Dim MUST match the embedding model; OpenAI vectors are normalized → Cosine.
	cfg := rostam.VectorConfig{Dim: embedDim, Metric: vector.Cosine, M: 16, EfConstruction: 200, EfSearch: 64}
	if err := store.CreateCollection(ctx, collection, cfg); err != nil && !strings.Contains(strings.ToLower(err.Error()), "exist") {
		log.Fatalf("create collection: %v", err)
	}

	// Embed + upsert the corpus.
	texts := make([]string, len(corpus))
	for i, d := range corpus {
		texts[i] = d.text
	}
	vecs, err := embed(ctx, apiKey, texts)
	if err != nil {
		log.Fatalf("embed corpus: %v", err)
	}
	for i, d := range corpus {
		meta := rostam.VectorMetadata{"category": vector.NewString(d.category)}
		// Attach a lexical sparse vector alongside the dense embedding so the same
		// point can be served by either lane (and fused). Real projects use a
		// learned sparse model (SPLADE) here; this example derives one locally.
		// NB: dense points carry sparse via VectorInsertOpts.Sparse — the embedded
		// WriteOpts.Sparse is a DIFFERENT field used only by MV-add and is silently
		// ignored on dense upserts (which would leave the hybrid lane empty).
		opts := rostam.VectorInsertOpts{Metadata: meta, Sparse: sparseOf(d.text)}
		if err := store.VectorUpsert(ctx, collection, uint64(i+1), vecs[i], d.text, opts); err != nil {
			log.Fatalf("upsert %d: %v", i+1, err)
		}
	}
	fmt.Printf("upserted %d documents into %q\n\n", len(corpus), collection)

	// Embed the query and search.
	qv, err := embed(ctx, apiKey, []string{query})
	if err != nil {
		log.Fatalf("embed query: %v", err)
	}
	fmt.Printf("query: %q\n\n", query)

	// Dense (semantic) search — pure embedding similarity.
	docs, _, err := store.VectorSearchDocs(ctx, collection, qv[0], 3, rostam.VectorSearchOpts{})
	if err != nil {
		log.Fatalf("dense search: %v", err)
	}
	fmt.Println("── dense (semantic) ──")
	for rank, d := range docs {
		fmt.Printf("%d. score=%.4f%s\n   %s\n", rank+1, d.Score, catOf(d.Metadata), d.Content)
	}

	// Hybrid search — fuses the dense lane with a lexical sparse lane via RRF, so a
	// query that shares exact terms with a doc (keywords) is rewarded alongside
	// semantic similarity. Results are (id, score); look the text up from the corpus.
	hits, _, err := store.VectorHybridSearch(ctx, collection, qv[0], 3, rostam.VectorHybridOpts{
		Sparse: sparseOf(query), Method: rostam.FusionRRF,
	})
	if err != nil {
		log.Fatalf("hybrid search: %v", err)
	}
	fmt.Println("\n── hybrid (dense + lexical, RRF) ──")
	for rank, h := range hits {
		fmt.Printf("%d. score=%.4f\n   %s\n", rank+1, h.Score, corpus[h.ID-1].text)
	}
}

// catOf renders the optional category payload for display.
func catOf(m rostam.VectorMetadata) string {
	if v, ok := m["category"]; ok {
		return " [" + v.Str + "]"
	}
	return ""
}

// sparseOf derives a lexical (keyword) sparse vector from text: lowercase, split on
// non-alphanumeric runes, hash each term to a uint32 index, and weight by damped
// term frequency (√tf). Indices MUST be strictly ascending and unique, so term
// counts are aggregated then sorted. This is a dependency-free stand-in for a
// learned sparse model (SPLADE/BM25); swap it without touching the query path.
func sparseOf(text string) rostam.VectorSparse {
	counts := map[uint32]float32{}
	for _, tok := range strings.FieldsFunc(strings.ToLower(text), func(r rune) bool {
		return !unicode.IsLetter(r) && !unicode.IsDigit(r)
	}) {
		if len(tok) < 2 {
			continue // drop single-char noise
		}
		h := fnv.New32a()
		_, _ = h.Write([]byte(tok))
		counts[h.Sum32()]++
	}
	idx := make([]uint32, 0, len(counts))
	for k := range counts {
		idx = append(idx, k)
	}
	sort.Slice(idx, func(i, j int) bool { return idx[i] < idx[j] })
	vals := make([]float32, len(idx))
	for i, k := range idx {
		vals[i] = float32(math.Sqrt(float64(counts[k])))
	}
	return rostam.VectorSparse{Indices: idx, Values: vals}
}

// embed turns texts into vectors via the OpenAI embeddings API. A real project
// swaps this for whatever embedding provider/model it standardizes on — only the
// model name + embedDim (and the collection Dim) change.
func embed(ctx context.Context, apiKey string, texts []string) ([][]float32, error) {
	body, _ := json.Marshal(map[string]any{"model": embedModel, "input": texts})
	req, err := http.NewRequestWithContext(ctx, http.MethodPost, embedEndpoint, bytes.NewReader(body))
	if err != nil {
		return nil, err
	}
	req.Header.Set("Authorization", "Bearer "+apiKey)
	req.Header.Set("Content-Type", "application/json")

	resp, err := http.DefaultClient.Do(req)
	if err != nil {
		return nil, err
	}
	defer func() { _ = resp.Body.Close() }()
	if resp.StatusCode != http.StatusOK {
		var b bytes.Buffer
		_, _ = b.ReadFrom(resp.Body)
		return nil, fmt.Errorf("embeddings API %d: %s", resp.StatusCode, b.String())
	}

	var out struct {
		Data []struct {
			Embedding []float32 `json:"embedding"`
		} `json:"data"`
	}
	if err := json.NewDecoder(resp.Body).Decode(&out); err != nil {
		return nil, err
	}
	if len(out.Data) != len(texts) {
		return nil, fmt.Errorf("embeddings: got %d vectors for %d inputs", len(out.Data), len(texts))
	}
	vecs := make([][]float32, len(out.Data))
	for i := range out.Data {
		if len(out.Data[i].Embedding) != embedDim {
			return nil, fmt.Errorf("embeddings: dim %d != expected %d (fix embedDim + collection Dim)", len(out.Data[i].Embedding), embedDim)
		}
		vecs[i] = out.Data[i].Embedding
	}
	return vecs, nil
}
