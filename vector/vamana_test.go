// SPDX-License-Identifier: Apache-2.0

package vector

import (
	"math/rand"
	"sort"
	"testing"
)

// vamanaClusteredCorpus builds a clustered Cosine corpus + queries (the regime a
// graph index is meant to handle well — real embeddings have directional cluster
// structure). Returns the corpus, the queries, and a brute-force recall@k ground
// truth function over normalized vectors (max dot product).
func vamanaClusteredCorpus(t *testing.T, n, dim, nq, clusters, k int, seed int64, noise float64) (corpus, queries [][]float32, groundTruth func(q []float32) map[uint64]bool) {
	t.Helper()
	rng := rand.New(rand.NewSource(seed))
	centers := make([][]float32, clusters)
	for c := range centers {
		v := make([]float32, dim)
		for j := range v {
			v[j] = float32(rng.NormFloat64())
		}
		normalize(v)
		centers[c] = v
	}
	pointNear := func(c int) []float32 {
		v := make([]float32, dim)
		for j := range v {
			v[j] = centers[c][j] + float32(noise*rng.NormFloat64())
		}
		return v
	}
	corpus = make([][]float32, n)
	for i := range corpus {
		corpus[i] = pointNear(i % clusters)
	}
	queries = make([][]float32, nq)
	for i := range queries {
		queries[i] = pointNear(rng.Intn(clusters))
	}
	groundTruth = func(q []float32) map[uint64]bool {
		qn := append([]float32(nil), q...)
		normalize(qn)
		type pair struct {
			id  uint64
			dot float32
		}
		dists := make([]pair, n)
		for i, v := range corpus {
			vn := append([]float32(nil), v...)
			normalize(vn)
			dists[i] = pair{id: uint64(i + 1), dot: dotScalar(qn, vn)}
		}
		sort.Slice(dists, func(a, b int) bool { return dists[a].dot > dists[b].dot })
		out := make(map[uint64]bool, k)
		for i := 0; i < k; i++ {
			out[dists[i].id] = true
		}
		return out
	}
	return corpus, queries, groundTruth
}

func vamanaRecallOf(t *testing.T, h *hnsw, queries [][]float32, gt func(q []float32) map[uint64]bool, nq, k int) float64 {
	t.Helper()
	var matches int
	for _, q := range queries {
		truth := gt(q)
		results, err := h.Search(q, k)
		if err != nil {
			t.Fatalf("search: %v", err)
		}
		for _, r := range results {
			if truth[r.ID] {
				matches++
			}
		}
	}
	return float64(matches) / float64(nq*k)
}

// TestVamanaRecall is the headline: an IndexVamana graph built by the two-pass
// α-RobustPrune algorithm achieves recall@10 comparable to (here, within a small
// margin of) the HNSW graph over identical clustered data. A correct Vamana graph
// (medoid entry + diverse R-bounded edges) routes greedy search to the true
// nearest neighbors; a broken one would tank recall. Both are exact-float
// (no quantization) so the recall difference is purely graph quality.
func TestVamanaRecall(t *testing.T) {
	if testing.Short() {
		t.Skip("skipping recall test in -short mode")
	}
	const (
		n        = 4_000
		dim      = 128
		nq       = 50
		k        = 10
		seed     = 7
		clusters = 64
		noise    = 0.15
	)
	corpus, queries, gt := vamanaClusteredCorpus(t, n, dim, nq, clusters, k, seed, noise)

	// HNSW baseline (incremental insert).
	hn, err := newHNSW(Config{Dim: dim, M: 16, EfConstruction: 200, EfSearch: 64, Seed: seed, Metric: Cosine})
	if err != nil {
		t.Fatal(err)
	}
	for i, v := range corpus {
		if _, _, err := hn.Insert(uint64(i+1), v, 0, nil, nil, nil, CASCond{}); err != nil {
			t.Fatalf("hnsw insert %d: %v", i, err)
		}
	}

	// Vamana via the two-pass bulk build.
	vm, err := newVamana(Config{Dim: dim, Seed: seed, Metric: Cosine, VamanaR: 64, VamanaL: 100, VamanaAlpha: 1.2})
	if err != nil {
		t.Fatal(err)
	}
	ids := make([]uint64, n)
	for i := range ids {
		ids[i] = uint64(i + 1)
	}
	if err := vm.BuildConcurrent(ids, corpus, 0); err != nil {
		t.Fatalf("vamana build: %v", err)
	}

	hnRecall := vamanaRecallOf(t, hn, queries, gt, nq, k)
	vmRecall := vamanaRecallOf(t, vm, queries, gt, nq, k)
	t.Logf("recall@%d  hnsw=%.3f  vamana=%.3f", k, hnRecall, vmRecall)

	// Vamana on exact floats over clustered data routes to the true NNs; assert a
	// high absolute floor and no large gap below HNSW.
	if vmRecall < 0.95 {
		t.Errorf("Vamana recall@%d = %.3f, want >= 0.95", k, vmRecall)
	}
	if vmRecall < hnRecall-0.05 {
		t.Errorf("Vamana recall@%d = %.3f, want within 0.05 of HNSW (%.3f)", k, vmRecall, hnRecall)
	}
}

// TestVamanaDeterministicBuild verifies the seeded two-pass build is
// deterministic: two builds with the same Seed produce identical graphs (same
// medoid entry point and identical per-slot neighbor lists).
func TestVamanaDeterministicBuild(t *testing.T) {
	const (
		n    = 1_000
		dim  = 32
		seed = 99
	)
	rng := rand.New(rand.NewSource(seed))
	corpus := make([][]float32, n)
	for i := range corpus {
		v := make([]float32, dim)
		for j := range v {
			v[j] = float32(rng.NormFloat64())
		}
		corpus[i] = v
	}
	ids := make([]uint64, n)
	for i := range ids {
		ids[i] = uint64(i + 1)
	}
	build := func() *hnsw {
		h, err := newVamana(Config{Dim: dim, Seed: seed, Metric: Cosine, VamanaR: 48, VamanaL: 80, VamanaAlpha: 1.2})
		if err != nil {
			t.Fatal(err)
		}
		if err := h.BuildConcurrent(ids, corpus, 1); err != nil { // workers=1 for a deterministic link order
			t.Fatalf("build: %v", err)
		}
		return h
	}
	a, b := build(), build()
	if a.entryPoint != b.entryPoint {
		t.Fatalf("entry point differs: %d vs %d", a.entryPoint, b.entryPoint)
	}
	for slot := 0; slot < n; slot++ {
		na, nb := a.nodes[slot], b.nodes[slot]
		if na == nil || nb == nil {
			t.Fatalf("slot %d: nil node", slot)
		}
		la := a.nbrsAt(na, 0)
		lb := b.nbrsAt(nb, 0)
		if len(la) != len(lb) {
			t.Fatalf("slot %d: neighbor count %d vs %d", slot, len(la), len(lb))
		}
		for i := range la {
			if la[i] != lb[i] {
				t.Fatalf("slot %d: neighbor %d differs: %d vs %d", slot, i, la[i], lb[i])
			}
		}
	}
}

// TestVamanaSingleLayerInvariant asserts the Vamana graph is genuinely
// single-layer: maxLevel is 0 and NO node ever carries upper-level edges, after
// both a bulk build and a batch of incremental inserts.
func TestVamanaSingleLayerInvariant(t *testing.T) {
	const (
		n    = 2_000
		dim  = 32
		seed = 11
	)
	rng := rand.New(rand.NewSource(seed))
	corpus := make([][]float32, n)
	for i := range corpus {
		v := make([]float32, dim)
		for j := range v {
			v[j] = float32(rng.NormFloat64())
		}
		corpus[i] = v
	}
	ids := make([]uint64, n)
	for i := range ids {
		ids[i] = uint64(i + 1)
	}
	h, err := newVamana(Config{Dim: dim, Seed: seed, Metric: Cosine, VamanaR: 32, VamanaL: 64, VamanaAlpha: 1.2})
	if err != nil {
		t.Fatal(err)
	}
	if err := h.BuildConcurrent(ids, corpus, 0); err != nil {
		t.Fatalf("build: %v", err)
	}
	// Incremental inserts after build (post-build single-layer Insert).
	for i := 0; i < 200; i++ {
		v := make([]float32, dim)
		for j := range v {
			v[j] = float32(rng.NormFloat64())
		}
		if _, _, err := h.Insert(uint64(n+i+1), v, 0, nil, nil, nil, CASCond{}); err != nil {
			t.Fatalf("insert: %v", err)
		}
	}
	if h.maxLevel != 0 {
		t.Fatalf("maxLevel = %d, want 0 (single-layer)", h.maxLevel)
	}
	for slot, nd := range h.nodes {
		if nd == nil {
			continue
		}
		if nd.level != 0 {
			t.Fatalf("slot %d: level = %d, want 0", slot, nd.level)
		}
		if nd.upper != nil {
			t.Fatalf("slot %d: upper != nil (has upper-level edges)", slot)
		}
	}
}

// TestVamanaInsertKeepsRecall verifies that incremental Inserts after the bulk
// build extend the graph without collapsing recall: half the corpus is bulk-built,
// the other half inserted one at a time, and recall@10 over the FULL set stays
// high.
func TestVamanaInsertKeepsRecall(t *testing.T) {
	if testing.Short() {
		t.Skip("skipping recall test in -short mode")
	}
	const (
		n        = 4_000
		dim      = 128
		nq       = 50
		k        = 10
		seed     = 7
		clusters = 64
		noise    = 0.15
	)
	corpus, queries, gt := vamanaClusteredCorpus(t, n, dim, nq, clusters, k, seed, noise)

	h, err := newVamana(Config{Dim: dim, Seed: seed, Metric: Cosine, VamanaR: 64, VamanaL: 100, VamanaAlpha: 1.2})
	if err != nil {
		t.Fatal(err)
	}
	half := n / 2
	ids := make([]uint64, half)
	for i := range ids {
		ids[i] = uint64(i + 1)
	}
	if err := h.BuildConcurrent(ids, corpus[:half], 0); err != nil {
		t.Fatalf("build: %v", err)
	}
	for i := half; i < n; i++ {
		if _, _, err := h.Insert(uint64(i+1), corpus[i], 0, nil, nil, nil, CASCond{}); err != nil {
			t.Fatalf("insert %d: %v", i, err)
		}
	}
	r := vamanaRecallOf(t, h, queries, gt, nq, k)
	t.Logf("recall@%d after build+incremental insert = %.3f", k, r)
	if r < 0.90 {
		t.Errorf("Vamana build+insert recall@%d = %.3f, want >= 0.90", k, r)
	}
}

// TestVamanaDeleteAndFilter exercises the reused machinery on a Vamana index:
// a metadata equality filter restricts search to matching points, and Delete
// (tombstone) removes a point from results.
func TestVamanaDeleteAndFilter(t *testing.T) {
	const (
		n    = 1_500
		dim  = 32
		k    = 10
		seed = 3
	)
	rng := rand.New(rand.NewSource(seed))
	h, err := newVamana(Config{Dim: dim, Seed: seed, Metric: Cosine, VamanaR: 32, VamanaL: 64, VamanaAlpha: 1.2})
	if err != nil {
		t.Fatal(err)
	}
	for i := 0; i < n; i++ {
		v := make([]float32, dim)
		for j := range v {
			v[j] = float32(rng.NormFloat64())
		}
		bucket := "even"
		if i%2 == 1 {
			bucket = "odd"
		}
		meta := Metadata{"bucket": NewString(bucket)}
		if _, _, err := h.Insert(uint64(i+1), v, 0, meta, nil, nil, CASCond{}); err != nil {
			t.Fatalf("insert %d: %v", i, err)
		}
	}
	// A query vector.
	q := make([]float32, dim)
	for j := range q {
		q[j] = float32(rng.NormFloat64())
	}

	// Metadata equality filter: every result must be in the "odd" bucket. Bucket is
	// "odd" iff i%2==1, and id == i+1, so an "odd"-bucket point has an EVEN id.
	filter := Filter{Op: FilterEq, Field: "bucket", Value: NewString("odd")}
	res, err := h.SearchInto(nil, q, k, filter)
	if err != nil {
		t.Fatalf("filtered search: %v", err)
	}
	if len(res) == 0 {
		t.Fatal("filtered search returned no results")
	}
	for _, r := range res {
		if r.ID%2 != 0 {
			t.Fatalf("filtered result id %d has odd id (=> even bucket), want odd-bucket only", r.ID)
		}
	}

	// Delete the top unfiltered result and confirm it disappears from results.
	top, err := h.Search(q, 1)
	if err != nil {
		t.Fatalf("search: %v", err)
	}
	if len(top) == 0 {
		t.Fatal("search returned no results")
	}
	delID := top[0].ID
	ok, err := h.Delete(delID, CASCond{})
	if err != nil || !ok {
		t.Fatalf("delete %d: ok=%v err=%v", delID, ok, err)
	}
	after, err := h.Search(q, k)
	if err != nil {
		t.Fatalf("search after delete: %v", err)
	}
	for _, r := range after {
		if r.ID == delID {
			t.Fatalf("deleted id %d still present in results", delID)
		}
	}
}

// TestVamanaValidate checks the Validate gate admits IndexVamana with sane params,
// applies defaults for zero fields, and rejects bad geometry.
func TestVamanaValidate(t *testing.T) {
	ok := []Config{
		{Dim: 8, M: 16, EfConstruction: 100, EfSearch: 64, IndexType: IndexVamana},                                             // defaults
		{Dim: 8, M: 16, EfConstruction: 100, EfSearch: 64, IndexType: IndexVamana, VamanaR: 32, VamanaL: 64, VamanaAlpha: 1.5}, // explicit
		{Dim: 8, M: 16, EfConstruction: 100, EfSearch: 64, IndexType: IndexVamana, VamanaR: 32, VamanaL: 32, VamanaAlpha: 1.0}, // L==R, alpha==1
	}
	for i, c := range ok {
		if err := c.Validate(); err != nil {
			t.Errorf("ok[%d]: Validate = %v, want nil", i, err)
		}
	}
	bad := []Config{
		{Dim: 8, M: 16, EfConstruction: 100, EfSearch: 64, IndexType: IndexVamana, VamanaR: -1},                                // R<0
		{Dim: 8, M: 16, EfConstruction: 100, EfSearch: 64, IndexType: IndexVamana, VamanaR: 64, VamanaL: 32},                   // L<R
		{Dim: 8, M: 16, EfConstruction: 100, EfSearch: 64, IndexType: IndexVamana, VamanaR: 32, VamanaL: 64, VamanaAlpha: 0.5}, // alpha<1
		{Dim: 8, M: 16, EfConstruction: 100, EfSearch: 64, IndexType: IndexVamana + 1},                                         // unknown index type
	}
	for i, c := range bad {
		if err := c.Validate(); err == nil {
			t.Errorf("bad[%d]: Validate = nil, want error", i)
		}
	}
}
