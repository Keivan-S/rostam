// SPDX-License-Identifier: Apache-2.0

package vector

import (
	"math"
	"testing"
)

// blobs builds n points around each of the given 2D-ish blob centers (dim 4),
// with a tiny deterministic jitter so each blob is tight and well-separated.
func blobs(centers [][]float32, perBlob int) [][]float32 {
	out := make([][]float32, 0, len(centers)*perBlob)
	for _, c := range centers {
		for i := 0; i < perBlob; i++ {
			v := make([]float32, len(c))
			j := float32(i%5) * 0.01 // bounded jitter, far smaller than blob gap
			for d := range c {
				v[d] = c[d] + j
			}
			out = append(out, v)
		}
	}
	return out
}

func nearestCentroid(v []float32, centroids [][]float32, dist distFunc) int {
	best := 0
	bd := dist(v, centroids[0])
	for i := 1; i < len(centroids); i++ {
		if d := dist(v, centroids[i]); d < bd {
			bd = d
			best = i
		}
	}
	return best
}

func TestKMeansBlobsCentersAndDeterminism(t *testing.T) {
	centers := [][]float32{
		{0, 0, 0, 0},
		{10, 10, 10, 10},
		{0, 10, 0, 10},
	}
	vecs := blobs(centers, 30)

	c1 := kmeans(vecs, 3, 42, L2, 1)
	c2 := kmeans(vecs, 3, 42, L2, 1)

	if len(c1) != 3 {
		t.Fatalf("expected 3 centroids, got %d", len(c1))
	}

	// Determinism: identical seed -> byte-identical centroids.
	if len(c1) != len(c2) {
		t.Fatalf("non-deterministic centroid count: %d vs %d", len(c1), len(c2))
	}
	for i := range c1 {
		if !vecEqual(c1[i], c2[i]) {
			t.Fatalf("centroid %d not deterministic: %v vs %v", i, c1[i], c2[i])
		}
	}

	// Each true center must have a centroid landing near it.
	dist := pickDist(L2)
	for _, ctr := range centers {
		ci := nearestCentroid(ctr, c1, dist)
		if d := dist(ctr, c1[ci]); d > 1.0 {
			t.Fatalf("no centroid near blob center %v; nearest dist^2=%v centroid=%v", ctr, d, c1[ci])
		}
	}

	// Assignment stability: every point in a blob assigns to one centroid, and
	// the three blobs map to three distinct centroids.
	seen := map[int]bool{}
	for bi, ctr := range centers {
		want := nearestCentroid(ctr, c1, dist)
		seen[want] = true
		for i := 0; i < 30; i++ {
			v := vecs[bi*30+i]
			if got := nearestCentroid(v, c1, dist); got != want {
				t.Fatalf("blob %d point %d assigned to %d, want %d", bi, i, got, want)
			}
		}
	}
	if len(seen) != 3 {
		t.Fatalf("blobs collapsed onto %d centroids, want 3", len(seen))
	}
}

func TestKMeansK1IsGlobalMean(t *testing.T) {
	vecs := [][]float32{
		{0, 0, 0, 0},
		{2, 2, 2, 2},
		{4, 4, 4, 4},
	}
	c := kmeans(vecs, 1, 7, L2, 1)
	if len(c) != 1 {
		t.Fatalf("k=1 expected 1 centroid, got %d", len(c))
	}
	// Global mean is (2,2,2,2).
	for d := 0; d < 4; d++ {
		if math.Abs(float64(c[0][d]-2)) > 1e-4 {
			t.Fatalf("k=1 centroid not global mean: %v", c[0])
		}
	}
}

func TestKMeansKGreaterEqualNReturnsPoints(t *testing.T) {
	vecs := [][]float32{
		{1, 0, 0, 0},
		{0, 1, 0, 0},
		{0, 0, 1, 0},
	}
	// k > n
	c := kmeans(vecs, 5, 1, L2, 1)
	if len(c) != 3 {
		t.Fatalf("k>n expected n=3 centroids, got %d", len(c))
	}
	// Each input must be present as a centroid.
	for _, v := range vecs {
		found := false
		for _, ctr := range c {
			if vecEqual(v, ctr) {
				found = true
				break
			}
		}
		if !found {
			t.Fatalf("input %v missing from centroids %v", v, c)
		}
	}
}

func TestKMeansKEqualNDedups(t *testing.T) {
	// n=4 with one exact duplicate -> dedup leaves 3 distinct centroids even
	// though k==n.
	vecs := [][]float32{
		{1, 0, 0, 0},
		{0, 1, 0, 0},
		{0, 1, 0, 0}, // dup
		{0, 0, 1, 0},
	}
	c := kmeans(vecs, 4, 1, L2, 1)
	if len(c) != 3 {
		t.Fatalf("k==n with one dup expected 3 distinct centroids, got %d", len(c))
	}
}

func TestKMeansEmptyAndGuards(t *testing.T) {
	if c := kmeans(nil, 3, 1, L2, 1); c != nil {
		t.Fatalf("empty input expected nil, got %v", c)
	}
	if c := kmeans([][]float32{}, 3, 1, L2, 1); c != nil {
		t.Fatalf("empty slice expected nil, got %v", c)
	}
	if c := kmeans([][]float32{{1, 2}}, 0, 1, L2, 1); c != nil {
		t.Fatalf("k=0 expected nil, got %v", c)
	}
	if c := kmeans([][]float32{{1, 2}}, -1, 1, L2, 1); c != nil {
		t.Fatalf("k<0 expected nil, got %v", c)
	}
}

func TestKMeansConvergesStable(t *testing.T) {
	// A clearly-clustered set should converge: running once more (more capacity
	// than needed) yields the same result, i.e. it's at a fixed point well
	// within the iteration cap.
	centers := [][]float32{
		{0, 0, 0, 0},
		{20, 0, 0, 0},
		{0, 20, 0, 0},
		{0, 0, 20, 0},
	}
	vecs := blobs(centers, 25)
	a := kmeans(vecs, 4, 99, L2, 1)
	b := kmeans(vecs, 4, 99, L2, 1)
	if len(a) != 4 {
		t.Fatalf("expected 4 centroids, got %d", len(a))
	}
	for i := range a {
		if !vecEqual(a[i], b[i]) {
			t.Fatalf("not stable at centroid %d", i)
		}
	}
	// Every blob center has a tight centroid -> converged correctly.
	dist := pickDist(L2)
	for _, ctr := range centers {
		ci := nearestCentroid(ctr, a, dist)
		if d := dist(ctr, a[ci]); d > 1.0 {
			t.Fatalf("blob %v not converged, nearest dist^2=%v", ctr, d)
		}
	}
}

func TestKMeansCosineClustersByAngle(t *testing.T) {
	// Three angular directions, varying magnitude, then normalized (as a cosine
	// collection stores them). kmeans with Cosine must cluster by angle.
	raw := [][]float32{
		{1, 0, 0, 0}, {2, 0, 0, 0}, {3, 0, 0, 0},
		{0, 1, 0, 0}, {0, 5, 0, 0}, {0, 2, 0, 0},
		{0, 0, 1, 0}, {0, 0, 4, 0}, {0, 0, 7, 0},
	}
	vecs := make([][]float32, len(raw))
	for i, r := range raw {
		v := cloneVec(r)
		normalize(v)
		vecs[i] = v
	}
	c := kmeans(vecs, 3, 5, Cosine, 1)
	if len(c) != 3 {
		t.Fatalf("expected 3 centroids, got %d", len(c))
	}
	dist := pickDist(Cosine)
	// Each group of 3 (same axis) must map to a single centroid, and the three
	// axes to three distinct centroids.
	seen := map[int]bool{}
	for g := 0; g < 3; g++ {
		base := nearestCentroid(vecs[g*3], c, dist)
		seen[base] = true
		for j := 1; j < 3; j++ {
			if got := nearestCentroid(vecs[g*3+j], c, dist); got != base {
				t.Fatalf("axis group %d not coherent: point %d -> %d, want %d", g, j, got, base)
			}
		}
	}
	if len(seen) != 3 {
		t.Fatalf("cosine clustering collapsed onto %d centroids, want 3", len(seen))
	}
}
