// SPDX-License-Identifier: Apache-2.0
//go:build cuda

package vector

// Host-side GPU tests for the -tags cuda exact-KNN index. These REQUIRE a
// working NVIDIA GPU + nvcc-built libknn.a (make -C vector/cuda). They are
// EXCLUDED from the default (!cuda) build and from the memory-capped container.
//
// Run:
//   make -C vector/cuda
//   CGO_ENABLED=1 /usr/local/go/bin/go test -tags cuda ./vector/ -run GPU -count=1
//
// Correctness anchor: the GPU exact KNN is brute force, so its result MUST match
// a CPU exact brute-force reference (ids + distances within float tolerance) for
// every metric. These tests assert that equality, not merely "good recall".

import (
	"bytes"
	"math"
	"math/rand"
	"sort"
	"testing"
	"time"

	"github.com/rostamlabs/rostam/vector/cuda"
)

// cpuExactKNN is the reference: an exact brute-force top-k over the live points,
// computed in plain Go using the package's own distFunc (pickDist), with the
// SAME cosine pre-normalization the index applies. It returns ids ordered
// nearest-first. live maps id -> stored vector (already normalized for Cosine).
func cpuExactKNN(metric Metric, query []float32, live map[uint64][]float32, k int) []Result {
	dist := pickDist(metric)
	q := append([]float32(nil), query...)
	if metric == Cosine {
		normalize(q)
	}
	type idDist struct {
		id uint64
		d  float32
	}
	all := make([]idDist, 0, len(live))
	for id, v := range live {
		all = append(all, idDist{id: id, d: dist(q, v)})
	}
	// Stable order: distance asc, id asc as a deterministic tiebreak.
	sort.Slice(all, func(a, b int) bool {
		if all[a].d != all[b].d {
			return all[a].d < all[b].d
		}
		return all[a].id < all[b].id
	})
	out := make([]Result, 0, k)
	for i := 0; i < len(all) && i < k; i++ {
		out = append(out, Result{ID: all[i].id, Distance: all[i].d})
	}
	return out
}

// buildGPUCorpus inserts n random dim-vectors into a fresh GPU index and returns
// the index plus the live id->storedVector map (normalized for Cosine, matching
// the arena). ids are 1..n.
func buildGPUCorpus(t *testing.T, metric Metric, n, dim int, seed int64) (*gpuIndex, map[uint64][]float32) {
	t.Helper()
	cfg := Config{Dim: dim, M: 16, EfConstruction: 200, EfSearch: 64, Metric: metric, Seed: 1}
	vi, err := newGPUIndex(cfg)
	if err != nil {
		t.Fatalf("newGPUIndex: %v", err)
	}
	g, ok := vi.(*gpuIndex)
	if !ok {
		t.Fatalf("newGPUIndex returned %T, want *gpuIndex", vi)
	}
	rng := rand.New(rand.NewSource(seed))
	live := make(map[uint64][]float32, n)
	for i := 0; i < n; i++ {
		id := uint64(i + 1)
		v := make([]float32, dim)
		for d := range v {
			v[d] = rng.Float32()*2 - 1
		}
		if _, _, err := g.Insert(id, v, 0, nil, nil, nil, CASCond{}); err != nil {
			t.Fatalf("Insert id=%d: %v", id, err)
		}
		stored := append([]float32(nil), v...)
		if metric == Cosine {
			normalize(stored)
		}
		live[id] = stored
	}
	return g, live
}

// assertMatchesCPU checks that gpu results == cpu reference: same ids in order
// (allowing tied-distance reorderings) and distances within tol.
func assertMatchesCPU(t *testing.T, label string, gpu, cpu []Result, tol float32) {
	t.Helper()
	if len(gpu) != len(cpu) {
		t.Fatalf("%s: len(gpu)=%d len(cpu)=%d", label, len(gpu), len(cpu))
	}
	// Compare as a set keyed by id with distance tolerance, then verify the GPU
	// distances are non-decreasing (nearest-first) and equal position-wise within
	// tied-distance groups. We first check the multiset of (id, dist) matches.
	cpuByID := make(map[uint64]float32, len(cpu))
	for _, r := range cpu {
		cpuByID[r.ID] = r.Distance
	}
	for i, r := range gpu {
		cd, ok := cpuByID[r.ID]
		if !ok {
			t.Fatalf("%s: gpu result[%d] id=%d not in cpu reference", label, i, r.ID)
		}
		if math.Abs(float64(r.Distance-cd)) > float64(tol) {
			t.Fatalf("%s: id=%d gpu dist=%g cpu dist=%g (tol %g)", label, r.ID, r.Distance, cd, tol)
		}
	}
	// Verify nearest-first ordering on the GPU side.
	for i := 1; i < len(gpu); i++ {
		if gpu[i].Distance < gpu[i-1].Distance-tol {
			t.Fatalf("%s: gpu results not nearest-first at %d: %g < %g", label, i, gpu[i].Distance, gpu[i-1].Distance)
		}
	}
}

// TestGPUExactMatchAllMetrics is the correctness anchor: GPU exact KNN == CPU
// exact brute force for L2, Cosine, DotProduct, over multiple k and several
// queries.
func TestGPUExactMatchAllMetrics(t *testing.T) {
	const n, dim = 2000, 64
	metrics := []struct {
		name string
		m    Metric
	}{
		{"L2", L2}, {"Cosine", Cosine}, {"DotProduct", DotProduct},
	}
	for _, mc := range metrics {
		mc := mc
		t.Run(mc.name, func(t *testing.T) {
			g, live := buildGPUCorpus(t, mc.m, n, dim, 42)
			defer g.Close()
			rng := rand.New(rand.NewSource(7))
			for _, k := range []int{1, 5, 10, 50, 100} {
				for q := 0; q < 5; q++ {
					query := make([]float32, dim)
					for d := range query {
						query[d] = rng.Float32()*2 - 1
					}
					got, err := g.Search(query, k)
					if err != nil {
						t.Fatalf("Search: %v", err)
					}
					want := cpuExactKNN(mc.m, query, live, k)
					// L2 squared distances can be large; scale tol with dim.
					tol := float32(1e-3)
					if mc.m == L2 {
						tol = 1e-2
					}
					assertMatchesCPU(t, mc.name, got, want, tol)
				}
			}
		})
	}
}

// TestGPUBatchedQueries verifies the batched kernel dispatch returns the same
// per-query exact top-k as the single-query path and the CPU reference.
func TestGPUBatchedQueries(t *testing.T) {
	const n, dim, k = 1500, 32, 20
	g, live := buildGPUCorpus(t, L2, n, dim, 11)
	defer g.Close()
	rng := rand.New(rand.NewSource(99))
	const nq = 16
	queries := make([][]float32, nq)
	for i := range queries {
		qv := make([]float32, dim)
		for d := range qv {
			qv[d] = rng.Float32()*2 - 1
		}
		queries[i] = qv
	}
	batch, err := g.gpuSearchBatch(queries, k)
	if err != nil {
		t.Fatalf("gpuSearchBatch: %v", err)
	}
	if len(batch) != nq {
		t.Fatalf("batch len=%d want %d", len(batch), nq)
	}
	for i, qv := range queries {
		want := cpuExactKNN(L2, qv, live, k)
		assertMatchesCPU(t, "batch", batch[i], want, 1e-2)
		// And the single-query path agrees.
		single, err := g.Search(qv, k)
		if err != nil {
			t.Fatalf("Search: %v", err)
		}
		assertMatchesCPU(t, "single-vs-batch", single, want, 1e-2)
	}
}

// TestGPUFilteredAndDeleted checks that a filtered search returns the exact
// top-k among admitted points (vs a CPU reference restricted to admitted) and
// that tombstoned ids never appear.
func TestGPUFilteredAndDeleted(t *testing.T) {
	const n, dim, k = 1000, 48, 15
	cfg := Config{Dim: dim, M: 16, EfConstruction: 200, EfSearch: 64, Metric: L2, Seed: 1}
	vi, err := newGPUIndex(cfg)
	if err != nil {
		t.Fatalf("newGPUIndex: %v", err)
	}
	g := vi.(*gpuIndex)
	defer g.Close()

	rng := rand.New(rand.NewSource(5))
	live := make(map[uint64][]float32, n)
	for i := 0; i < n; i++ {
		id := uint64(i + 1)
		v := make([]float32, dim)
		for d := range v {
			v[d] = rng.Float32()*2 - 1
		}
		// Half the corpus is group "a", half group "b".
		grp := "a"
		if i%2 == 0 {
			grp = "b"
		}
		meta := Metadata{"grp": NewString(grp)}
		if _, _, err := g.Insert(id, v, 0, meta, nil, nil, CASCond{}); err != nil {
			t.Fatalf("Insert: %v", err)
		}
		live[id] = append([]float32(nil), v...)
	}

	// Delete a chunk of ids; they must never be returned.
	deleted := map[uint64]bool{}
	for id := uint64(1); id <= 100; id++ {
		if _, err := g.Delete(id, CASCond{}); err != nil {
			t.Fatalf("Delete: %v", err)
		}
		deleted[id] = true
	}

	query := make([]float32, dim)
	for d := range query {
		query[d] = rng.Float32()*2 - 1
	}

	// Filtered search: grp == "a", excluding deleted.
	filter := Filter{Op: FilterEq, Field: "grp", Value: NewString("a")}
	got, err := g.SearchFiltered(query, k, filter)
	if err != nil {
		t.Fatalf("SearchFiltered: %v", err)
	}

	// CPU reference restricted to admitted (grp=="a" && not deleted).
	admitted := make(map[uint64][]float32)
	for id, v := range live {
		if deleted[id] {
			continue
		}
		// grp "a" are the odd indices (i%2==1 -> id=i+1 even). Recompute: i = id-1.
		i := int(id - 1)
		if i%2 != 1 {
			continue // not group "a"
		}
		admitted[id] = v
	}
	want := cpuExactKNN(L2, query, admitted, k)
	assertMatchesCPU(t, "filtered", got, want, 1e-2)
	for _, r := range got {
		if deleted[r.ID] {
			t.Fatalf("filtered result returned deleted id %d", r.ID)
		}
	}

	// Unfiltered search must also never return a deleted id.
	gotAll, err := g.Search(query, k)
	if err != nil {
		t.Fatalf("Search: %v", err)
	}
	for _, r := range gotAll {
		if deleted[r.ID] {
			t.Fatalf("unfiltered result returned deleted id %d", r.ID)
		}
	}
}

// TestGPUSnapshotRestore verifies the inherited hnsw Snapshot/Restore round-trips
// and GPU search is still exact after restore (the resident buffer rebuilds
// lazily from the restored arena).
func TestGPUSnapshotRestore(t *testing.T) {
	const n, dim, k = 800, 32, 10
	g, live := buildGPUCorpus(t, Cosine, n, dim, 21)
	defer g.Close()

	// Snapshot via the inherited hnsw method.
	var buf bytes.Buffer
	if err := g.Snapshot(&buf); err != nil {
		t.Fatalf("Snapshot: %v", err)
	}

	// Restore into a fresh GPU index.
	cfg := Config{Dim: dim, M: 16, EfConstruction: 200, EfSearch: 64, Metric: Cosine, Seed: 1}
	vi2, err := newGPUIndex(cfg)
	if err != nil {
		t.Fatalf("newGPUIndex(restore target): %v", err)
	}
	g2 := vi2.(*gpuIndex)
	defer g2.Close()
	if err := g2.Restore(&buf); err != nil {
		t.Fatalf("Restore: %v", err)
	}

	rng := rand.New(rand.NewSource(123))
	for q := 0; q < 8; q++ {
		query := make([]float32, dim)
		for d := range query {
			query[d] = rng.Float32()*2 - 1
		}
		got, err := g2.Search(query, k)
		if err != nil {
			t.Fatalf("Search after restore: %v", err)
		}
		want := cpuExactKNN(Cosine, query, live, k)
		assertMatchesCPU(t, "post-restore", got, want, 1e-3)
	}
}

// TestGPUSelectiveFilterBeyond256 exercises the BLOCKER fix: a highly selective
// filter (~1% admitted) over a corpus far larger than the kernel's MAX_K (256)
// where the admitted top-k lies BEYOND the top-256-by-raw-score. The old code
// over-fetched kfetch=capacity but the kernel only wrote 256 outputs (clamped),
// so the host looped past the written rows reading uninitialized memory ->
// duplicate / garbage / wrong filtered results. With the CPU-exact fallback the
// result must match the CPU reference restricted to the admitted set.
func TestGPUSelectiveFilterBeyond256(t *testing.T) {
	const n, dim, k = 3000, 32, 20
	cfg := Config{Dim: dim, M: 16, EfConstruction: 200, EfSearch: 64, Metric: L2, Seed: 1}
	vi, err := newGPUIndex(cfg)
	if err != nil {
		t.Fatalf("newGPUIndex: %v", err)
	}
	g := vi.(*gpuIndex)
	defer g.Close()

	rng := rand.New(rand.NewSource(77))
	live := make(map[uint64][]float32, n)
	admitted := make(map[uint64][]float32)
	for i := 0; i < n; i++ {
		id := uint64(i + 1)
		v := make([]float32, dim)
		for d := range v {
			v[d] = rng.Float32()*2 - 1
		}
		// ~1% admitted: only every 100th id is group "rare".
		grp := "common"
		if i%100 == 0 {
			grp = "rare"
		}
		meta := Metadata{"grp": NewString(grp)}
		if _, _, err := g.Insert(id, v, 0, meta, nil, nil, CASCond{}); err != nil {
			t.Fatalf("Insert: %v", err)
		}
		live[id] = append([]float32(nil), v...)
		if grp == "rare" {
			admitted[id] = live[id]
		}
	}
	if len(admitted) < k {
		t.Fatalf("test setup: only %d admitted, need >= %d", len(admitted), k)
	}

	rng2 := rand.New(rand.NewSource(303))
	for q := 0; q < 8; q++ {
		query := make([]float32, dim)
		for d := range query {
			query[d] = rng2.Float32()*2 - 1
		}
		filter := Filter{Op: FilterEq, Field: "grp", Value: NewString("rare")}
		got, err := g.SearchFiltered(query, k, filter)
		if err != nil {
			t.Fatalf("SearchFiltered: %v", err)
		}
		want := cpuExactKNN(L2, query, admitted, k)
		assertMatchesCPU(t, "selective-beyond-256", got, want, 1e-2)
	}
}

// TestGPUSearchKAbove256 exercises the BLOCKER fix for plain Search with
// k > MAX_K: the old code silently returned only 256 results (the kernel clamp),
// breaking the exact top-k contract. With the CPU-exact fallback Search(k=300)
// must return the exact top-300 matching the CPU reference.
func TestGPUSearchKAbove256(t *testing.T) {
	const n, dim, k = 2000, 16, 300
	g, live := buildGPUCorpus(t, L2, n, dim, 909)
	defer g.Close()

	rng := rand.New(rand.NewSource(404))
	for q := 0; q < 4; q++ {
		query := make([]float32, dim)
		for d := range query {
			query[d] = rng.Float32()*2 - 1
		}
		got, err := g.Search(query, k)
		if err != nil {
			t.Fatalf("Search(k=%d): %v", k, err)
		}
		if len(got) != k {
			t.Fatalf("Search(k=%d) returned %d results (silent MAX_K truncation?)", k, len(got))
		}
		want := cpuExactKNN(L2, query, live, k)
		assertMatchesCPU(t, "k-above-256", got, want, 1e-2)
	}
}

// TestGPUManyTombstonesBeyond256 exercises the BLOCKER fix for the tombstone
// over-fetch path: with more than MAX_K tombstones the old code set
// kfetch = k + dead > 256, but the kernel only wrote 256 outputs, so the host
// read uninitialized rows. After deleting > 256 ids, an exact small-k search
// must still match the CPU reference over the surviving live set.
func TestGPUManyTombstonesBeyond256(t *testing.T) {
	const n, dim, k = 1200, 24, 15
	g, live := buildGPUCorpus(t, L2, n, dim, 555)
	defer g.Close()

	// Tombstone the first 400 ids (> MAX_K = 256).
	deleted := map[uint64]bool{}
	for id := uint64(1); id <= 400; id++ {
		if _, err := g.Delete(id, CASCond{}); err != nil {
			t.Fatalf("Delete: %v", err)
		}
		deleted[id] = true
		delete(live, id)
	}
	if got := len(deleted); got <= cuda.MaxK {
		t.Fatalf("test setup: %d tombstones, need > %d", got, cuda.MaxK)
	}

	rng := rand.New(rand.NewSource(606))
	for q := 0; q < 6; q++ {
		query := make([]float32, dim)
		for d := range query {
			query[d] = rng.Float32()*2 - 1
		}
		got, err := g.Search(query, k)
		if err != nil {
			t.Fatalf("Search: %v", err)
		}
		for _, r := range got {
			if deleted[r.ID] {
				t.Fatalf("result returned tombstoned id %d", r.ID)
			}
		}
		want := cpuExactKNN(L2, query, live, k)
		assertMatchesCPU(t, "many-tombstones", got, want, 1e-2)
	}
}

// TestGPUThroughputBench reports the GPU-vs-CPU exact-KNN speedup on a larger
// corpus. It is a measurement, not a hard assertion (it only requires the GPU to
// be at least correct); the speedup number is logged.
func TestGPUThroughputBench(t *testing.T) {
	if testing.Short() {
		t.Skip("throughput bench skipped in -short")
	}
	const n, dim, k, nq = 100000, 128, 10, 256
	g, live := buildGPUCorpus(t, L2, n, dim, 1234)
	defer g.Close()

	rng := rand.New(rand.NewSource(2024))
	queries := make([][]float32, nq)
	for i := range queries {
		qv := make([]float32, dim)
		for d := range qv {
			qv[d] = rng.Float32()*2 - 1
		}
		queries[i] = qv
	}

	// Warm the resident buffer (first search triggers the upload).
	if _, err := g.gpuSearchBatch(queries[:1], k); err != nil {
		t.Fatalf("warmup: %v", err)
	}

	// GPU batched timing.
	gpuStart := time.Now()
	gpuRes, err := g.gpuSearchBatch(queries, k)
	if err != nil {
		t.Fatalf("gpuSearchBatch: %v", err)
	}
	gpuDur := time.Since(gpuStart)

	// CPU exact timing (same nq queries, plain Go brute force).
	cpuStart := time.Now()
	cpuRes := make([][]Result, nq)
	for i, qv := range queries {
		cpuRes[i] = cpuExactKNN(L2, qv, live, k)
	}
	cpuDur := time.Since(cpuStart)

	// Correctness check on a sample so the bench also validates parity.
	for i := 0; i < nq; i += 32 {
		assertMatchesCPU(t, "bench-sample", gpuRes[i], cpuRes[i], 1.0)
	}

	speedup := float64(cpuDur) / float64(gpuDur)
	t.Logf("GPU exact KNN N=%d dim=%d k=%d nq=%d: GPU=%v CPU=%v speedup=%.1fx",
		n, dim, k, nq, gpuDur, cpuDur, speedup)
}
