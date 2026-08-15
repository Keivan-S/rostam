// SPDX-License-Identifier: Apache-2.0

package semcache

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"math"
	"net/http"

	"github.com/cespare/xxhash/v2"
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
//
// Two different texts produce the same vector only if they collide in the
// 64-bit seed hash (collision space 2^64), which is the same budget the cache
// already spends on entry ids. A narrower hash would not do: the proxy's
// "exact" mode treats a vector match as proof the prompts are identical, so a
// collision there is a wrong answer served to a user, not just a wasted slot.
func NewStubEmbedder(model string, dim int) Embedder {
	return stubEmbedder{model: model, dim: dim}
}

type stubEmbedder struct {
	model string
	dim   int
}

func (s stubEmbedder) Model() string { return s.model }
func (s stubEmbedder) Dim() int      { return s.dim }

// splitmix64Gamma and the two multipliers below are the standard splitmix64
// constants: one increment of the golden-ratio gamma plus two xor-shift
// multiply rounds turn a counter into a well-distributed 64-bit value. Used
// here to expand one 64-bit seed into dim per-dimension values without ever
// narrowing the state to 32 bits.
const (
	splitmix64Gamma = 0x9e3779b97f4a7c15
	splitmix64Mix1  = 0xbf58476d1ce4e5b9
	splitmix64Mix2  = 0x94d049bb133111eb
	// int64Scale maps a signed 64-bit value into roughly [-1, 1).
	int64Scale = 1 << 63
)

func (s stubEmbedder) Embed(_ context.Context, texts []string) ([][]float32, error) {
	out := make([][]float32, len(texts))
	for i, t := range texts {
		v := make([]float32, s.dim)
		state := xxhash.Sum64String(t)
		for j := 0; j < s.dim; j++ {
			state += splitmix64Gamma
			z := state
			z = (z ^ (z >> 30)) * splitmix64Mix1
			z = (z ^ (z >> 27)) * splitmix64Mix2
			z ^= z >> 31
			v[j] = float32(float64(int64(z)) / float64(int64Scale))
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
