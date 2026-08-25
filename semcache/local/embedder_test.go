// SPDX-License-Identifier: Apache-2.0

package local

import (
	"context"
	"os"
	"testing"

	"github.com/rostamlabs/rostam/semcache/localcatalog"
)

// TestModelStamp checks the scope-key stamp the wrapper adds: Model() must be
// "local:<label>". This is folded into every cache record's scope key, so a
// change here silently invalidates existing caches — it is worth an always-on
// (no-network) guard even though the rest of the pipeline lives in rembed.
func TestModelStamp(t *testing.T) {
	e := &Embedder{label: "minilm-l6-v2"}
	if got, want := e.Model(), "local:minilm-l6-v2"; got != want {
		t.Fatalf("Model()=%q want %q", got, want)
	}
}

// TestEmbedEndToEnd runs a real pure-Go forward pass for every catalog model.
// rembed downloads model weights from the Hugging Face Hub on first use, so this
// test needs network (or a warm cache) and is opt-in: set ROSTAM_LOCALEMBED_E2E=1
// to run it. Each model is a subtest asserting dimension, unit L2 norm, and
// determinism. Point REMBED_CACHE at a persistent dir to reuse downloads across
// runs instead of re-fetching the multi-hundred-MB base-tier weights.
func TestEmbedEndToEnd(t *testing.T) {
	if os.Getenv("ROSTAM_LOCALEMBED_E2E") == "" {
		t.Skip("set ROSTAM_LOCALEMBED_E2E=1 to run the local-embedding end-to-end test (downloads models from the Hugging Face Hub)")
	}
	for _, name := range localcatalog.Names() {
		t.Run(name, func(t *testing.T) {
			spec, ok := localcatalog.Lookup(name)
			if !ok {
				t.Fatalf("Lookup(%q) failed", name)
			}
			e, err := New(spec.HFRepo, spec.Name, spec.Dim)
			if err != nil {
				t.Fatalf("New: %v", err)
			}
			defer func() { _ = e.Close() }()

			if e.Dim() != spec.Dim {
				t.Fatalf("Dim()=%d want %d", e.Dim(), spec.Dim)
			}
			if got, want := e.Model(), "local:"+spec.Name; got != want {
				t.Fatalf("Model()=%q want %q", got, want)
			}

			vecs, err := e.Embed(context.Background(), []string{"a cat sits on the mat", "quantum chromodynamics"})
			if err != nil {
				t.Fatalf("Embed: %v", err)
			}
			if len(vecs) != 2 || len(vecs[0]) != spec.Dim {
				t.Fatalf("got %d vecs, dim %d, want 2 x %d", len(vecs), len(vecs[0]), spec.Dim)
			}
			// rembed L2-normalizes its output: |v| ~ 1.
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
