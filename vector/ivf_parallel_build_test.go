// SPDX-License-Identifier: Apache-2.0

package vector

import (
	"math/rand"
	"reflect"
	"testing"
)

// buildIVFCentroids builds an IVF index from a fixed corpus with the given worker
// count and returns its trained coarse centroids. Used to assert build
// determinism across worker counts.
func buildIVFCentroids(t *testing.T, cfg Config, ids []uint64, vecs [][]float32, workers int) [][]float32 {
	t.Helper()
	ix, err := newIVF(cfg)
	if err != nil {
		t.Fatal(err)
	}
	if err := ix.BuildConcurrent(ids, vecs, workers); err != nil {
		t.Fatal(err)
	}
	if !ix.trained {
		t.Fatal("index should be trained after BuildConcurrent")
	}
	return ix.centroids
}

// TestIVFBuildDeterministicWorkers asserts the IVF coarse k-means centroids are
// BIT-IDENTICAL whether BuildConcurrent runs with workers=1 or workers=4 (the
// build-determinism guarantee: the parallel assignment + index-ordered reduce
// must not perturb the trained centroids — divergent replicas would break
// snapshots).
func TestIVFBuildDeterministicWorkers(t *testing.T) {
	dim := 32
	rng := rand.New(rand.NewSource(7))
	n := 3000
	vecs := clusteredVecs(rng, n, dim, 50)
	ids := make([]uint64, n)
	for i := range ids {
		ids[i] = uint64(i + 1)
	}
	cfg := ivfTestConfig(dim)

	serial := buildIVFCentroids(t, cfg, ids, vecs, 1)
	par := buildIVFCentroids(t, cfg, ids, vecs, 4)
	if !reflect.DeepEqual(serial, par) {
		t.Fatal("IVF coarse centroids differ between workers=1 and workers=4 (not bit-identical)")
	}
}

// TestIVFPQBuildDeterministicWorkers asserts both the coarse centroids and the
// residual PQ codebooks are bit-identical across worker counts when IVF-PQ is on
// (trainPQ also threads workers through to the per-subspace k-means).
func TestIVFPQBuildDeterministicWorkers(t *testing.T) {
	dim := 32
	rng := rand.New(rand.NewSource(13))
	n := 3000
	vecs := clusteredVecs(rng, n, dim, 50)
	ids := make([]uint64, n)
	for i := range ids {
		ids[i] = uint64(i + 1)
	}
	cfg := ivfTestConfig(dim)
	cfg.IndexType = IndexIVF
	cfg.IVFPQ = true
	cfg.IVFRerank = true // keep floats resident so we can compare centroids cleanly

	build := func(workers int) *ivf {
		ix, err := newIVF(cfg)
		if err != nil {
			t.Fatal(err)
		}
		if err := ix.BuildConcurrent(ids, vecs, workers); err != nil {
			t.Fatal(err)
		}
		if ix.pq == nil {
			t.Fatal("expected IVF-PQ codebooks to be trained")
		}
		return ix
	}

	s := build(1)
	p := build(4)
	if !reflect.DeepEqual(s.centroids, p.centroids) {
		t.Fatal("IVF-PQ coarse centroids differ between workers=1 and workers=4")
	}
	if !reflect.DeepEqual(s.pq.codebooks, p.pq.codebooks) {
		t.Fatal("IVF-PQ residual codebooks differ between workers=1 and workers=4 (not bit-identical)")
	}
}
