// SPDX-License-Identifier: Apache-2.0

package vector

import (
	"bytes"
	"testing"
)

// TestRestoredIndexRebuildsPairFnForSnapshotMetric pins the cfg-derived half of
// the cached build-time pair-distance kernel.
//
// initPairFns resolves cfg.Metric/cfg.Dim to a distance kernel and captures it
// by value. RestoreCollection deliberately constructs the target from a
// placeholder Config{Dim: 1, ...} whose zero-valued Metric is COSINE, and lets
// readSnapshot overwrite cfg wholesale from the stream. So unless readSnapshot
// re-runs initPairFns, a restored L2 / DotProduct index keeps a cosine kernel:
// every later Insert runs selectNeighbors' diversity test comparing a cosine
// pf() against candidate distances measured in the restored metric. Nothing
// errors — the graph just quietly degrades on every post-restore write.
//
// The assertion is exact rather than statistical: the restored index's
// buildPairFn must compute the SAME value as the snapshot metric's kernel, and
// (the vacuity guard) a different value from the placeholder's cosine kernel.
func TestRestoredIndexRebuildsPairFnForSnapshotMetric(t *testing.T) {
	const dim = 8
	// Un-normalized, non-collinear vectors so cosine and L2/dot genuinely differ.
	vecs := [][]float32{
		{3, 1, 0, 0, 0, 0, 0, 0},
		{0, 2, 5, 0, 0, 0, 0, 0},
		{1, 1, 1, 4, 0, 0, 0, 0},
		{0, 0, 2, 2, 7, 0, 0, 0},
	}

	for _, metric := range []Metric{L2, DotProduct} {
		t.Run(metricName(metric), func(t *testing.T) {
			src := newCollectionStore(t)
			cfg := Config{Dim: dim, Metric: metric, M: 8, EfConstruction: 32, EfSearch: 16, Seed: 1}
			if err := src.CreateCollection("docs", cfg); err != nil {
				t.Fatalf("CreateCollection: %v", err)
			}
			c, ok := src.Get("docs")
			if !ok {
				t.Fatal("source collection missing")
			}
			for i, v := range vecs {
				if err := c.Insert(uint64(i+1), v, 0, nil, nil); err != nil {
					t.Fatalf("Insert %d: %v", i+1, err)
				}
			}
			var snap bytes.Buffer
			if err := c.Snapshot(&snap); err != nil {
				t.Fatalf("Snapshot: %v", err)
			}

			// The config-LESS restore path: a Cosine, dim-1 placeholder config is
			// used to construct, then readSnapshot swaps in the real geometry.
			dst := newCollectionStore(t)
			if err := dst.RestoreCollection("docs", bytes.NewReader(snap.Bytes())); err != nil {
				t.Fatalf("RestoreCollection: %v", err)
			}
			rc, ok := dst.Get("docs")
			if !ok {
				t.Fatal("restored collection missing")
			}
			h, ok := rc.idx.(*hnsw)
			if !ok {
				t.Fatalf("restored index is %T, want *hnsw", rc.idx)
			}
			if h.cfg.Metric != metric {
				t.Fatalf("restored cfg.Metric = %v, want %v (readSnapshot did not apply the stream config)", h.cfg.Metric, metric)
			}

			// Post-restore inserts are what the stale kernel would corrupt, so run
			// one before sampling: it must go through the same buildPairFn.
			extra := []float32{2, 0, 0, 3, 0, 1, 0, 0}
			if err := rc.Insert(99, extra, 0, nil, nil); err != nil {
				t.Fatalf("post-restore Insert: %v", err)
			}

			pf := h.buildPairFn()
			want := pickDistDim(metric, dim)
			stale := pickDistDim(Cosine, 1) // what the placeholder config resolves to
			var sawDivergence bool
			for a := uint32(0); a < 4; a++ {
				for b := a + 1; b < 4; b++ {
					va, vb := h.arena.Vec(a), h.arena.Vec(b)
					got := pf(a, b)
					if exp := want(va, vb); got != exp {
						t.Fatalf("buildPairFn(%d,%d) = %v, want %v (%v kernel) — the restored index is still using the construction-time kernel",
							a, b, got, exp, metricName(metric))
					}
					if stale(va, vb) != want(va, vb) {
						sawDivergence = true
					}
				}
			}
			if !sawDivergence {
				t.Fatalf("vacuous test: the cosine placeholder kernel agrees with %v on every sampled pair", metric)
			}
		})
	}
}

// newCollectionStore opens a store rooted at a fresh temp dir, closed on
// cleanup.
func newCollectionStore(t *testing.T) *CollectionStore {
	t.Helper()
	cs, err := OpenCollectionStore(t.TempDir())
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() { _ = cs.Close() })
	return cs
}
