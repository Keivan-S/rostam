// SPDX-License-Identifier: Apache-2.0
package postrec

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"time"
)

// OpenAIEmbedder implements Embedder against OpenAI's /v1/embeddings endpoint.
// Swap it for any Embedder (local model, Cohere, etc.).
type OpenAIEmbedder struct {
	APIKey string
	Model  string // e.g. "text-embedding-3-small"
	dim    int
	HTTP   *http.Client
}

// NewOpenAIEmbedder returns an OpenAI embedder. dim must match the model:
// text-embedding-3-small = 1536, text-embedding-3-large = 3072.
func NewOpenAIEmbedder(apiKey, model string, dim int) *OpenAIEmbedder {
	return &OpenAIEmbedder{
		APIKey: apiKey,
		Model:  model,
		dim:    dim,
		HTTP:   &http.Client{Timeout: 60 * time.Second},
	}
}

func (e *OpenAIEmbedder) Dim() int { return e.dim }

func (e *OpenAIEmbedder) Embed(ctx context.Context, texts []string) ([][]float32, error) {
	reqBody, _ := json.Marshal(map[string]any{"input": texts, "model": e.Model})
	req, err := http.NewRequestWithContext(ctx, http.MethodPost,
		"https://api.openai.com/v1/embeddings", bytes.NewReader(reqBody))
	if err != nil {
		return nil, err
	}
	req.Header.Set("Content-Type", "application/json")
	req.Header.Set("Authorization", "Bearer "+e.APIKey)

	resp, err := e.HTTP.Do(req)
	if err != nil {
		return nil, err
	}
	defer resp.Body.Close()
	raw, _ := io.ReadAll(resp.Body)
	if resp.StatusCode >= 300 {
		return nil, fmt.Errorf("openai embeddings: status %d: %s", resp.StatusCode, raw)
	}
	var out struct {
		Data []struct {
			Embedding []float32 `json:"embedding"`
		} `json:"data"`
	}
	if err := json.Unmarshal(raw, &out); err != nil {
		return nil, err
	}
	vecs := make([][]float32, len(out.Data))
	for i, d := range out.Data {
		vecs[i] = d.Embedding
	}
	return vecs, nil
}
