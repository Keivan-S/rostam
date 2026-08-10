// SPDX-License-Identifier: Apache-2.0

package semcache

import (
	"context"
	"encoding/json"
	"math"
	"net/http"
	"net/http/httptest"
	"testing"
)

// fakeEmbedder maps text -> a deterministic unit vector of length dim. Used by
// every library test so no network/model is required.
type fakeEmbedder struct {
	model string
	dim   int
}

func (f fakeEmbedder) Model() string { return f.model }
func (f fakeEmbedder) Dim() int      { return f.dim }

func (f fakeEmbedder) Embed(_ context.Context, texts []string) ([][]float32, error) {
	out := make([][]float32, len(texts))
	for i, t := range texts {
		v := make([]float32, f.dim)
		for j := 0; j < f.dim; j++ {
			// stable per (text, j) value; normalized below
			h := uint32(2166136261)
			for _, b := range []byte(t) {
				h = (h ^ uint32(b)) * 16777619
			}
			h ^= uint32(j * 2654435761)
			v[j] = float32(int32(h)) / float32(math.MaxInt32)
		}
		normalizeVec(v)
		out[i] = v
	}
	return out, nil
}

func TestFakeEmbedderIsNormalizedAndDeterministic(t *testing.T) {
	e := fakeEmbedder{model: "fake-1", dim: 8}
	a, err := e.Embed(context.Background(), []string{"hello"})
	if err != nil {
		t.Fatal(err)
	}
	b, _ := e.Embed(context.Background(), []string{"hello"})
	if a[0][0] != b[0][0] {
		t.Fatalf("embedder not deterministic")
	}
	var sum float64
	for _, x := range a[0] {
		sum += float64(x) * float64(x)
	}
	if math.Abs(sum-1.0) > 1e-4 {
		t.Fatalf("not unit-normalized: |v|^2=%v", sum)
	}
}

func TestOpenAIEmbedderShapesRequestAndNormalizes(t *testing.T) {
	var gotModel string
	var gotInputs []string
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.Header.Get("Authorization") != "Bearer test-key" {
			t.Errorf("missing/wrong auth header: %q", r.Header.Get("Authorization"))
		}
		var req struct {
			Model string   `json:"model"`
			Input []string `json:"input"`
		}
		_ = json.NewDecoder(r.Body).Decode(&req)
		gotModel, gotInputs = req.Model, req.Input
		// Return a non-unit vector to prove the embedder normalizes it.
		resp := map[string]any{"data": []map[string]any{{"embedding": []float32{3, 4}}}}
		w.Header().Set("Content-Type", "application/json")
		_ = json.NewEncoder(w).Encode(resp)
	}))
	defer srv.Close()

	e := NewOpenAIEmbedder("test-key", "text-embedding-3-small", 2)
	e.Endpoint = srv.URL
	vecs, err := e.Embed(context.Background(), []string{"hello"})
	if err != nil {
		t.Fatal(err)
	}
	if gotModel != "text-embedding-3-small" || len(gotInputs) != 1 || gotInputs[0] != "hello" {
		t.Fatalf("bad request shape: model=%q inputs=%v", gotModel, gotInputs)
	}
	// (3,4) normalized => (0.6,0.8).
	if math.Abs(float64(vecs[0][0])-0.6) > 1e-4 || math.Abs(float64(vecs[0][1])-0.8) > 1e-4 {
		t.Fatalf("not normalized: %v", vecs[0])
	}
}
