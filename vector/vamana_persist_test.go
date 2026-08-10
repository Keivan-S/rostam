// SPDX-License-Identifier: Apache-2.0

package vector

import (
	"os"
	"path/filepath"
	"runtime"
	"testing"
)

// buildVamanaMmapPQ builds a QuantPQ + QuantMmap IndexVamana over the given corpus
// via the two-pass buildVamana, returning the index plus its effective (path-bearing)
// config so the caller can SavePersist + openPersist. The float32 vectors live in the
// MmapPath file; only the PQ codes (+ graph) are resident.
func buildVamanaMmapPQ(t *testing.T, dir string, corpus [][]float32, dim, r, l int) (*hnsw, Config) {
	t.Helper()
	cfg := Config{
		Dim: dim, Metric: Cosine, Seed: 7, IndexType: IndexVamana,
		VamanaR: r, VamanaL: l, VamanaAlpha: 1.2,
		Quant: QuantPQ, QuantStorage: QuantMmap,
		MmapPath:      filepath.Join(dir, "vecs.dat"),
		GraphMmapPath: filepath.Join(dir, "graph.dat"),
		RescoreFactor: 3,
	}
	h, err := newVamana(cfg)
	if err != nil {
		t.Fatalf("newVamana(mmap-pq): %v", err)
	}
	ids := make([]uint64, len(corpus))
	for i := range ids {
		ids[i] = uint64(i + 1)
	}
	if err := h.buildVamana(ids, corpus, nil, runtime.GOMAXPROCS(0)); err != nil {
		t.Fatalf("buildVamana: %v", err)
	}
	return h, cfg
}

// TestVamanaQuantMmapRecallAndBacking is the disk-native headline: a
// QuantPQ + QuantMmap IndexVamana navigates the single-layer graph on resident PQ
// codes, pages the full float32 vectors from the mmap file for the exact rescore,
// and still hits a healthy recall@10. It also asserts the float32 vectors are
// mmap-backed (the backing file holds n*dim*4 bytes), not heap-resident.
func TestVamanaQuantMmapRecallAndBacking(t *testing.T) {
	if testing.Short() {
		t.Skip("skipping recall test in -short mode")
	}
	const (
		n, dim, nq, clusters, k = 3000, 64, 80, 24, 10
		seed                    = int64(11)
	)
	corpus, queries, gt := vamanaClusteredCorpus(t, n, dim, nq, clusters, k, seed, 0.10)

	dir := t.TempDir()
	h, cfg := buildVamanaMmapPQ(t, dir, corpus, dim, 64, 100)
	defer func() { _ = h.Close() }()

	// The float32 vectors must live in the backing file (n*dim*4 bytes minimum), so
	// only the PQ codes + graph are resident — the disk-native memory profile.
	fi, err := os.Stat(cfg.MmapPath)
	if err != nil {
		t.Fatalf("stat backing file: %v", err)
	}
	if minBytes := int64(n * dim * 4); fi.Size() < minBytes {
		t.Errorf("backing file = %d bytes, want >= %d (n*dim*4) — floats not mmap-backed", fi.Size(), minBytes)
	}
	// The arena must be mmap-backed (a real on-disk region), not a heap slice.
	if h.arena.mmapF == nil {
		t.Errorf("arena.mmapF is nil — floats are heap-resident, not mmap-backed")
	}

	r := vamanaRecallOf(t, h, queries, gt, nq, k)
	t.Logf("vamana pq+mmap recall@%d = %.3f (backing=%d bytes)", k, r, fi.Size())
	// PQ is lossy; with the exact float32 rescore (RescoreFactor=3) reading through the
	// mmap, recall stays healthy. The threshold proves codes-navigate + floats-rescore
	// both engage (an untrained codec or a broken rescore would collapse it).
	if r < 0.60 {
		t.Errorf("vamana pq+mmap recall@%d = %.3f, want >= 0.60", k, r)
	}
}

// TestVamanaPersistInstantRestart is the linchpin test: a QuantPQ + QuantMmap
// IndexVamana is flushed via SavePersist and reopened via openPersist (instant
// restart — map files, no rebuild). The reopen must reproduce IDENTICAL neighbor
// lists (the single-layer slab at stride VamanaR), IDENTICAL PQ codes, the medoid
// entry point, and IDENTICAL search results. This test FAILS if the m0
// reconstruction is wrong (newPersistShell deriving 2*M instead of VamanaR slices
// the level-0 slab at the wrong stride → corrupt neighbor lists → wrong results).
func TestVamanaPersistInstantRestart(t *testing.T) {
	const (
		n, dim, nq, clusters, k = 2500, 48, 60, 20, 10
		seed                    = int64(23)
		// Non-default R so a 2*M (=32 for the default M=16) reconstruction would slice
		// the slab at a DIFFERENT stride than the built one (R=40) — the m0 fix is
		// load-bearing and this test catches a regression of it.
		r = 40
		l = 90
	)
	corpus, queries, gt := vamanaClusteredCorpus(t, n, dim, nq, clusters, k, seed, 0.10)

	dir := t.TempDir()
	h, cfg := buildVamanaMmapPQ(t, dir, corpus, dim, r, l)

	// Sanity: the built slab stride is R, not 2*M.
	if h.m0 != r {
		t.Fatalf("built m0 = %d, want VamanaR = %d", h.m0, r)
	}
	builtMedoid := h.entryPoint

	// Capture pre-flush neighbor lists, codes, and search results.
	beforeNbrs := make(map[uint32][]uint32, n)
	for _, nd := range h.nodes {
		if nd == nil {
			continue
		}
		row := h.nbrsAt(nd, 0)
		beforeNbrs[nd.slot] = append([]uint32(nil), row...)
	}
	beforeCodes := append([]byte(nil), h.arena.codes...)
	before := make([][]uint64, len(queries))
	for i, q := range queries {
		res, err := h.Search(q, k)
		if err != nil {
			t.Fatal(err)
		}
		before[i] = resultIDs(res)
	}

	metaPath := filepath.Join(dir, "meta.bin")
	if err := h.SavePersist(metaPath); err != nil {
		t.Fatalf("SavePersist: %v", err)
	}
	if err := h.Close(); err != nil {
		t.Fatalf("close: %v", err)
	}

	// Instant restart: map the files, read the sidecar — no rebuild. openIndex routes
	// a Persistent Vamana here; call openPersist directly with the path-bearing cfg.
	cfg.Persistent = true
	h2, err := openPersist(cfg, metaPath)
	if err != nil {
		t.Fatalf("openPersist: %v", err)
	}
	defer func() { _ = h2.Close() }()

	// The reopened index must re-pin the single-layer Vamana geometry from cfg.
	if h2.m0 != r {
		t.Fatalf("reopened m0 = %d, want VamanaR = %d (m0 reconstruction bug)", h2.m0, r)
	}
	if h2.mL != 0 {
		t.Errorf("reopened mL = %v, want 0 (single-layer pinning lost)", h2.mL)
	}
	if !h2.vamana {
		t.Errorf("reopened vamana flag = false, want true")
	}
	if h2.entryPoint != builtMedoid {
		t.Errorf("reopened medoid entry point = %d, want %d (entry point not restored)", h2.entryPoint, builtMedoid)
	}

	// Neighbor lists must be byte-for-byte identical (the slab stride matched).
	for _, nd := range h2.nodes {
		if nd == nil {
			continue
		}
		got := h2.nbrsAt(nd, 0)
		want := beforeNbrs[nd.slot]
		if len(got) != len(want) {
			t.Fatalf("slot %d: reopened %d neighbors, want %d (slab stride mismatch)", nd.slot, len(got), len(want))
		}
		for j := range got {
			if got[j] != want[j] {
				t.Fatalf("slot %d nbr[%d] = %d, want %d (corrupt neighbor list on reopen)", nd.slot, j, got[j], want[j])
			}
		}
	}

	// PQ codes must be identical (re-encoded from the mapped floats with the restored
	// trained codebooks).
	if len(h2.arena.codes) != len(beforeCodes) {
		t.Fatalf("reopened codes len = %d, want %d", len(h2.arena.codes), len(beforeCodes))
	}
	for i := range beforeCodes {
		if h2.arena.codes[i] != beforeCodes[i] {
			t.Fatalf("code byte %d = %d, want %d (codes not reproduced on reopen)", i, h2.arena.codes[i], beforeCodes[i])
		}
	}

	// Search results must be identical post-reopen.
	for i, q := range queries {
		res, err := h2.Search(q, k)
		if err != nil {
			t.Fatal(err)
		}
		if !eqUint64(resultIDs(res), before[i]) {
			t.Fatalf("query %d: reopened results %v != original %v", i, resultIDs(res), before[i])
		}
	}
	_ = gt
}

// TestVamanaADCOnly checks the float-dropped (ADC-only) Vamana path returns sane
// results: after building a QuantPQ Vamana and releasing the resident float32
// vectors, the single-layer search navigates and ranks on PQ codes alone (rescore
// skipped) and still finds the true cluster — recall stays well above chance.
func TestVamanaADCOnly(t *testing.T) {
	if testing.Short() {
		t.Skip("skipping recall test in -short mode")
	}
	const (
		n, dim, nq, clusters, k = 3000, 64, 80, 16, 10
		seed                    = int64(31)
	)
	corpus, queries, gt := vamanaClusteredCorpus(t, n, dim, nq, clusters, k, seed, 0.08)

	dir := t.TempDir()
	h, _ := buildVamanaMmapPQ(t, dir, corpus, dim, 64, 100)
	defer func() { _ = h.Close() }()

	// Drop the resident float32 vectors: ADC-only navigation, no exact rescore.
	h.arena.dropVecs()
	if !h.vecsDropped() {
		t.Fatalf("vecsDropped() = false after dropVecs()")
	}

	r := vamanaRecallOf(t, h, queries, gt, nq, k)
	t.Logf("vamana ADC-only (float-dropped) recall@%d = %.3f", k, r)
	// ADC-only loses the exact-rescore refinement, so recall is well below the rescored
	// path — but it must be far above random (k/n ≈ 0.003), proving the single-layer
	// graph + PQ codes alone return sane neighbors (here ~80x chance).
	if r < 0.20 {
		t.Errorf("vamana ADC-only recall@%d = %.3f, want >= 0.20 (sane results, >>chance)", k, r)
	}
}
