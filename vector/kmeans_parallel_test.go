// SPDX-License-Identifier: Apache-2.0

package vector

import (
	"math/rand"
	"reflect"
	"testing"
)

// randomVecs builds n deterministic pseudo-random vectors of the given dim. Used
// to feed kmeans an input large enough to span multiple goroutine chunks so the
// parallel assignment path is actually exercised.
func randomVecs(n, dim int, seed int64) [][]float32 {
	rng := rand.New(rand.NewSource(seed))
	out := make([][]float32, n)
	for i := range out {
		v := make([]float32, dim)
		for d := range v {
			v[d] = rng.Float32()
		}
		out[i] = v
	}
	return out
}

// TestKmeansParallelBitIdentical is THE determinism guarantee: parallelizing the
// assignment step (workers=4) must produce BIT-IDENTICAL centroids to the serial
// path (workers=1) on the same input+seed. n is large enough that the range-split
// spans multiple goroutine chunks. Asserted via reflect.DeepEqual on the
// [][]float32 (exact float equality, no tolerance).
func TestKmeansParallelBitIdentical(t *testing.T) {
	cases := []struct {
		name   string
		metric Metric
		k      int
		seed   int64
	}{
		{"L2/k8", L2, 8, 12345},
		{"L2/k16", L2, 16, 777},
		{"Cosine/k8", Cosine, 8, 4242},
		{"DotProduct/k12", DotProduct, 12, 99},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			// 4000 points, dim 32: with workers<=GOMAXPROCS each chunk holds
			// ~1000 points, so several goroutines run concurrently.
			vecs := randomVecs(4000, 32, tc.seed)
			if tc.metric == Cosine {
				for _, v := range vecs {
					normalize(v)
				}
			}
			serial := kmeans(vecs, tc.k, tc.seed, tc.metric, 1)
			par := kmeans(vecs, tc.k, tc.seed, tc.metric, 4)
			if !reflect.DeepEqual(serial, par) {
				t.Fatalf("workers=4 centroids differ from workers=1 (NOT bit-identical) for %s", tc.name)
			}
			// Sanity: also workers=8 (capped to GOMAXPROCS) must match.
			par8 := kmeans(vecs, tc.k, tc.seed, tc.metric, 8)
			if !reflect.DeepEqual(serial, par8) {
				t.Fatalf("workers=8 centroids differ from workers=1 for %s", tc.name)
			}
		})
	}
}

// TestKmeansParallelAssignmentLabels checks the lower-level invariant: the
// parallel assignment step assigns every point to the same cluster as the serial
// step for a fixed centroid set (same labels). This isolates the assignment
// parallelization from the full Lloyd loop.
func TestKmeansParallelAssignmentLabels(t *testing.T) {
	vecs := randomVecs(3000, 24, 555)
	centroids := randomVecs(10, 24, 111)
	dist := pickDist(L2)
	k := len(centroids)

	serial := make([]int, len(vecs))
	for i := range serial {
		serial[i] = -1
	}
	parallel := make([]int, len(vecs))
	for i := range parallel {
		parallel[i] = -1
	}

	// Serial reference.
	for vi, v := range vecs {
		serial[vi] = nearestCentroidIdx(v, centroids, k, dist)
	}
	// Parallel.
	assignParallel(vecs, centroids, k, dist, parallel, 4)

	if !reflect.DeepEqual(serial, parallel) {
		t.Fatal("parallel assignment labels differ from serial labels")
	}
}

// TestKmeansSerialPathUnchanged guards that workers<=1 takes the verbatim serial
// path and that workers=0 (the zero value / default) behaves identically to
// workers=1.
func TestKmeansSerialPathUnchanged(t *testing.T) {
	vecs := randomVecs(1000, 16, 2024)
	w1 := kmeans(vecs, 6, 2024, L2, 1)
	w0 := kmeans(vecs, 6, 2024, L2, 0)
	wNeg := kmeans(vecs, 6, 2024, L2, -3)
	if !reflect.DeepEqual(w1, w0) {
		t.Fatal("workers=0 must equal workers=1 (serial default)")
	}
	if !reflect.DeepEqual(w1, wNeg) {
		t.Fatal("workers<0 must equal workers=1 (serial default)")
	}
}
