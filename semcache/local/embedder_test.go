// SPDX-License-Identifier: Apache-2.0
//go:build localembed

package local

import (
	"context"
	"math"
	"os"
	"testing"

	"github.com/rostamlabs/rostam/semcache/localcatalog"
)

func TestMeanPoolMasksPadding(t *testing.T) {
	// batch=1, seqLen=3, hdim=2; token 2 is padding (mask 0).
	hidden := []float32{1, 1, 3, 3, 99, 99}
	mask := []int64{1, 1, 0}
	v := meanPool(hidden, mask, 0, 3, 2) // mean of tokens 0,1 => (2,2)
	if v[0] != 2 || v[1] != 2 {
		t.Fatalf("meanPool=%v want [2 2]", v)
	}
}

func TestNormalizeUnitLength(t *testing.T) {
	v := []float32{3, 4}
	normalize(v)
	if l := math.Sqrt(float64(v[0]*v[0] + v[1]*v[1])); math.Abs(l-1) > 1e-6 {
		t.Fatalf("norm=%v len=%f", v, l)
	}
}

// TestEmbedEndToEnd runs a real MiniLM forward pass. It requires ONNX Runtime
// installed (ROSTAM_ONNXRUNTIME_LIB) and network (or a warm cache) for the
// model download, so it self-skips when either is absent.
func TestEmbedEndToEnd(t *testing.T) {
	lib, ok := os.LookupEnv("ROSTAM_ONNXRUNTIME_LIB")
	if !ok {
		t.Skip("set ROSTAM_ONNXRUNTIME_LIB to run the ONNX end-to-end test")
	}
	spec, _ := localcatalog.Lookup("minilm-l6-v2")
	root := t.TempDir()
	e, err := NewEmbedder(context.Background(), spec, root, lib, nil)
	if err != nil {
		t.Fatalf("NewEmbedder: %v", err)
	}
	defer func() { _ = e.Close() }()

	vecs, err := e.Embed(context.Background(), []string{"a cat sits on the mat", "quantum chromodynamics"})
	if err != nil {
		t.Fatalf("Embed: %v", err)
	}
	if len(vecs) != 2 || len(vecs[0]) != spec.Dim {
		t.Fatalf("got %d vecs, dim %d, want 2 x %d", len(vecs), len(vecs[0]), spec.Dim)
	}
	// Normalized: length ~1.
	var sum float64
	for _, x := range vecs[0] {
		sum += float64(x) * float64(x)
	}
	if sum < 0.99 || sum > 1.01 {
		t.Fatalf("not normalized: |v|^2=%f", sum)
	}
	// Determinism: same input twice => identical vector.
	again, _ := e.Embed(context.Background(), []string{"a cat sits on the mat"})
	for d := range again[0] {
		if again[0][d] != vecs[0][d] {
			t.Fatal("non-deterministic embedding")
		}
	}
}
