// SPDX-License-Identifier: Apache-2.0

package semcache

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"math"
	"net/http"
)

// Embedder turns text into vectors. The SAME embedder (model+version) must be
// used for Store and Lookup — the model id is folded into every cache record's
// scope key, so a model change cannot silently corrupt hit quality.
type Embedder interface {
	// Embed returns one vector per input text, in order.
	Embed(ctx context.Context, texts []string) ([][]float32, error)
	// Model is the embedding model id; stamped into the scope key.
	Model() string
	// Dim is the output dimension; must equal the cache collection's Dim.
	Dim() int
}

// normalizeVec scales v to unit length in place (no-op for the zero vector).
// Cosine distance over normalized vectors collapses to 1 - dot, which the
// cache relies on for its similarity threshold.
func normalizeVec(v []float32) {
	var sum float64
	for _, x := range v {
		sum += float64(x) * float64(x)
	}
	if sum == 0 {
		return
	}
	inv := float32(1.0 / math.Sqrt(sum))
	for i := range v {
		v[i] *= inv
	}
}

// NewStubEmbedder returns a deterministic, dependency-free Embedder for tests
// and local demos: it hashes text into a normalized vector of length dim. NOT
// for production (no semantic meaning) — use OpenAIEmbedder there.
func NewStubEmbedder(model string, dim int) Embedder {
	return stubEmbedder{model: model, dim: dim}
}

type stubEmbedder struct {
	model string
	dim   int
}

func (s stubEmbedder) Model() string { return s.model }
func (s stubEmbedder) Dim() int      { return s.dim }

func (s stubEmbedder) Embed(_ context.Context, texts []string) ([][]float32, error) {
	out := make([][]float32, len(texts))
	for i, t := range texts {
		v := make([]float32, s.dim)
		h := uint32(2166136261)
		for _, b := range []byte(t) {
			h = (h ^ uint32(b)) * 16777619
		}
		for j := 0; j < s.dim; j++ {
			h = (h ^ uint32(j)) * 16777619
			v[j] = float32(int32(h)) / 2147483647.0
		}
		normalizeVec(v)
		out[i] = v
	}
	return out, nil
}

// OpenAIEmbedder is a hosted Embedder using the OpenAI-compatible /embeddings
// API (also served by Azure OpenAI and several open gateways). Outputs are
// L2-normalized so the cache's Cosine = 1 - dot assumption always holds.
type OpenAIEmbedder struct {
	APIKey     string
	ModelID    string
	Endpoint   string // default https://api.openai.com/v1/embeddings
	Dimension  int
	HTTPClient *http.Client // nil => http.DefaultClient
}

// NewOpenAIEmbedder builds an OpenAIEmbedder for model with output dimension
// dim (must equal the model's true output dim and the cache collection's Dim).
func NewOpenAIEmbedder(apiKey, model string, dim int) *OpenAIEmbedder {
	return &OpenAIEmbedder{
		APIKey:    apiKey,
		ModelID:   model,
		Endpoint:  "https://api.openai.com/v1/embeddings",
		Dimension: dim,
	}
}

func (e *OpenAIEmbedder) Model() string { return e.ModelID }
func (e *OpenAIEmbedder) Dim() int      { return e.Dimension }

func (e *OpenAIEmbedder) Embed(ctx context.Context, texts []string) ([][]float32, error) {
	body, _ := json.Marshal(map[string]any{"model": e.ModelID, "input": texts})
	req, err := http.NewRequestWithContext(ctx, http.MethodPost, e.Endpoint, bytes.NewReader(body))
	if err != nil {
		return nil, err
	}
	req.Header.Set("Authorization", "Bearer "+e.APIKey)
	req.Header.Set("Content-Type", "application/json")

	hc := e.HTTPClient
	if hc == nil {
		hc = http.DefaultClient
	}
	resp, err := hc.Do(req)
	if err != nil {
		return nil, err
	}
	defer func() { _ = resp.Body.Close() }()
	if resp.StatusCode != http.StatusOK {
		var b bytes.Buffer
		_, _ = b.ReadFrom(resp.Body)
		return nil, fmt.Errorf("semcache: embeddings API %d: %s", resp.StatusCode, b.String())
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
		return nil, fmt.Errorf("semcache: got %d vectors for %d inputs", len(out.Data), len(texts))
	}
	vecs := make([][]float32, len(out.Data))
	for i := range out.Data {
		if len(out.Data[i].Embedding) != e.Dimension {
			return nil, fmt.Errorf("semcache: embedding dim %d != configured %d", len(out.Data[i].Embedding), e.Dimension)
		}
		v := out.Data[i].Embedding
		normalizeVec(v)
		vecs[i] = v
	}
	return vecs, nil
}
