// SPDX-License-Identifier: Apache-2.0
//go:build localembed

package local

import (
	"context"
	"math"
	"os"
	"path/filepath"
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

// TestEmbedEndToEnd runs a real forward pass for every catalog model. It
// requires ONNX Runtime installed (ROSTAM_ONNXRUNTIME_LIB) and network (or a
// warm cache) for the model downloads, so it self-skips when the lib is absent.
// Each model is a subtest asserting dimension, unit L2 norm, and determinism.
func TestEmbedEndToEnd(t *testing.T) {
	lib, ok := os.LookupEnv("ROSTAM_ONNXRUNTIME_LIB")
	if !ok {
		t.Skip("set ROSTAM_ONNXRUNTIME_LIB to run the ONNX end-to-end test")
	}
	// A shared, persistent cache (ROSTAM_LOCALEMBED_CACHE) lets repeated runs
	// reuse the multi-hundred-MB base-tier downloads instead of re-fetching per
	// subtest; unset, each subtest downloads into its own temp dir. filepath.Clean
	// sanitizes the env-provided path (it is developer-supplied, not attacker
	// input, but keeps gosec's taint analysis quiet).
	var cacheRoot string
	if v := os.Getenv("ROSTAM_LOCALEMBED_CACHE"); v != "" {
		cacheRoot = filepath.Clean(v)
	}
	for _, name := range localcatalog.Names() {
		t.Run(name, func(t *testing.T) {
			spec, ok := localcatalog.Lookup(name)
			if !ok {
				t.Fatalf("Lookup(%q) failed", name)
			}
			root := cacheRoot
			if root == "" {
				root = t.TempDir()
			}
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
			again, err := e.Embed(context.Background(), []string{"a cat sits on the mat"})
			if err != nil {
				t.Fatalf("Embed (second call): %v", err)
			}
			for d := range again[0] {
				if again[0][d] != vecs[0][d] {
					t.Fatal("non-deterministic embedding")
				}
			}
		})
	}
}
