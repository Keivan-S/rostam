// SPDX-License-Identifier: Apache-2.0
//go:build cuda

package vector

// Verification for the id-0 sentinel fix: gpu_cuda.go's
// three exact-scan lanes — gpuSearchLocked, cpuExactSearchLocked, and
// gpuSearchBatch — switched their dead-slot guard from `arena.ID(slot) == 0`
// to `arena.Allocated(slot)`. This file exercises the two things that change
// affects:
//
//  1. Point id 0 is now reachable through all three GPU lanes (mirrors
//     id_zero_test.go's own-vector-at-rank-0 assertions, ported to the GPU
//     dense-search entry points).
//
//  2. Allocated() ALSO closes a pre-existing hole the old `id == 0` guard
//     never addressed: a slot freed by Reclaim() keeps a stale, usually
//     NON-ZERO id (Reclaim empties h.tombstoned and arena.Delete removes the
//     idMap entry, but neither touches arena.ids[slot]), so the old guard
//     let it straight through as a "ghost" result whenever that stale id
//     happened to be non-zero — i.e. almost always. These tests reconstruct
//     that exact scenario: they query with the reclaimed point's OWN vector,
//     which makes the ghost score a PERFECT (distance 0) match, so it would
//     rank first under the pre-fix code. They are real regression tests for
//     that pre-existing bug, not smoke tests — they are constructed to FAIL
//     under `arena.ID(slot) == 0` and PASS under `arena.Allocated(slot)`.
//
// A third stale-slot family the code comments call out is covered too:
// reserved-but-unwritten capacity (arena.Reserve pre-sizes storage ahead of a
// bulk/concurrent build that writes into it slot-by-slot) must never be
// scored, including when the "build" only partially completes.
//
// Every lane is exercised on both sides of the GPU kernel's MAX_K (256) fetch
// limit: the FAST PATH (kGPU >= admitted candidates, served straight from the
// GPU's raw top-k) and the CPU-EXACT FALLBACK (k > cuda.MaxK, unconditionally
// routed through cpuExactSearchLocked per gpuSearchLocked's own contract).

import (
	"math/rand"
	"testing"

	"github.com/rostamlabs/rostam/vector/cuda"
)

// gpuReclaimVec returns a deterministic pseudo-random unit-scale vector.
func gpuReclaimVec(dim int, seed int64) []float32 {
	r := rand.New(rand.NewSource(seed)) //nolint:gosec // deterministic test fixture
	v := make([]float32, dim)
	for i := range v {
		v[i] = r.Float32()*2 - 1
	}
	return v
}

// newGPUReclaimIndex builds a fresh, empty GPU index (L2 metric) for these
// tests. The caller must g.Close() it.
func newGPUReclaimIndex(t *testing.T, dim int) *gpuIndex {
	t.Helper()
	cfg := Config{Dim: dim, M: 16, EfConstruction: 200, EfSearch: 64, Metric: L2, Seed: 1}
	vi, err := newGPUIndex(cfg)
	if err != nil {
		t.Fatalf("newGPUIndex: %v", err)
	}
	g, ok := vi.(*gpuIndex)
	if !ok {
		t.Fatalf("newGPUIndex returned %T, want *gpuIndex", vi)
	}
	return g
}

// insertGPUDecoys inserts n points with ids [idStart, idStart+n) and random
// vectors, so a search has plenty of legitimate candidates to satisfy k.
func insertGPUDecoys(t *testing.T, g *gpuIndex, idStart uint64, n, dim int, seed int64) {
	t.Helper()
	rng := rand.New(rand.NewSource(seed)) //nolint:gosec // deterministic test fixture
	for i := 0; i < n; i++ {
		id := idStart + uint64(i)
		v := make([]float32, dim)
		for d := range v {
			v[d] = rng.Float32()*2 - 1
		}
		if _, _, err := g.Insert(id, v, 0, nil, nil, nil, CASCond{}); err != nil {
			t.Fatalf("insert decoy id=%d: %v", id, err)
		}
	}
}

// reclaimGhostSlot inserts ghostID, deletes it, and physically reclaims its
// slot (Reclaim), reproducing the exact "freed slot with a stale non-zero id"
// scenario gpu_cuda.go's comments describe. It returns the point's original
// vector, so the caller can query with it: since the reclaimed slot's floats
// are UNTOUCHED by Delete/Reclaim (only arena.idMap and h.tombstoned are
// cleared — arena.ids[slot] and arena.vecs stay exactly as inserted), a query
// on that vector scores the ghost row at distance 0 if it is not excluded.
func reclaimGhostSlot(t *testing.T, g *gpuIndex, ghostID uint64, dim int, seed int64) []float32 {
	t.Helper()
	v := gpuReclaimVec(dim, seed)
	if _, _, err := g.Insert(ghostID, v, 0, nil, nil, nil, CASCond{}); err != nil {
		t.Fatalf("insert ghost id=%d: %v", ghostID, err)
	}
	if _, err := g.Delete(ghostID, CASCond{}); err != nil {
		t.Fatalf("delete ghost id=%d: %v", ghostID, err)
	}
	if n := g.Reclaim(); n != 1 {
		t.Fatalf("Reclaim() = %d, want 1", n)
	}
	if _, _, _, _, _, ok := g.Get(ghostID); ok {
		t.Fatalf("ghost id %d still Get()-able after Reclaim", ghostID)
	}
	return v
}

// --- Ghost-result regression: gpuSearchLocked (FAST PATH) ---

// TestGPUReclaimGhostNotReturnedFastPath is the core regression test for the
// gpuSearchLocked fast path (k <= cuda.MaxK, no tombstones to widen the
// fetch — Reclaim empties h.tombstoned, so the fetch is NOT over-fetched at
// all). Under the pre-fix `arena.ID(slot) == 0` guard the reclaimed slot's
// stale non-zero id would sail through admission and its untouched vector
// would score a perfect distance-0 match against its own query, landing at
// rank 0 and displacing a legitimate neighbor.
func TestGPUReclaimGhostNotReturnedFastPath(t *testing.T) {
	const dim = 16
	g := newGPUReclaimIndex(t, dim)
	defer g.Close()

	insertGPUDecoys(t, g, 1, 20, dim, 100)
	ghostVec := reclaimGhostSlot(t, g, 99999, dim, 555)

	got, err := g.Search(ghostVec, 5)
	if err != nil {
		t.Fatalf("Search: %v", err)
	}
	if len(got) != 5 {
		t.Fatalf("Search returned %d results, want 5 (20 live decoys should easily cover k)", len(got))
	}
	for _, r := range got {
		if r.ID == 99999 {
			t.Fatalf("ghost id 99999 (reclaimed slot) returned by GPU fast-path search: %v", resultIDs(got))
		}
		if r.Distance == 0 {
			t.Fatalf("a result scored distance 0 against the ghost's own-vector query — looks like an undetected ghost match: %+v", got)
		}
	}
}

// --- Ghost-result regression: cpuExactSearchLocked (forced fallback) ---

// TestGPUReclaimGhostNotReturnedCPUFallback forces the CPU-exact fallback
// unconditionally by requesting k > cuda.MaxK: gpuSearchLocked can then never
// satisfy "added >= k" from the GPU's raw top-kGPU alone (kGPU is clamped to
// MaxK), so it always falls through to cpuExactSearchLocked regardless of how
// many candidates were admitted. That lane has its own copy of the
// Allocated() guard (see gpu_cuda.go's cpuExactSearchLocked), so it needs its
// own regression coverage.
func TestGPUReclaimGhostNotReturnedCPUFallback(t *testing.T) {
	const dim = 16
	g := newGPUReclaimIndex(t, dim)
	defer g.Close()

	const nDecoys = 500
	insertGPUDecoys(t, g, 1, nDecoys, dim, 200)
	ghostVec := reclaimGhostSlot(t, g, 88888, dim, 777)

	k := cuda.MaxK + 44
	got, err := g.Search(ghostVec, k)
	if err != nil {
		t.Fatalf("Search: %v", err)
	}
	if len(got) != k {
		t.Fatalf("Search(k=%d) returned %d results, want %d (%d live decoys should cover k)", k, len(got), k, nDecoys)
	}
	for _, r := range got {
		if r.ID == 88888 {
			t.Fatalf("ghost id 88888 (reclaimed slot) returned by the CPU-exact fallback: %v", resultIDs(got))
		}
	}
}

// --- Ghost-result regression: gpuSearchBatch (fast path + forced fallback) ---

// TestGPUReclaimGhostNotReturnedBatchFastPath is the batch-dispatch analogue
// of the fast-path test: gpuSearchBatch has its OWN copy of the ghost-slot
// comment and Allocated() call (a separate call site from the single-query
// path), so it needs its own regression coverage.
func TestGPUReclaimGhostNotReturnedBatchFastPath(t *testing.T) {
	const dim = 16
	g := newGPUReclaimIndex(t, dim)
	defer g.Close()

	insertGPUDecoys(t, g, 1, 20, dim, 300)
	ghostVec := reclaimGhostSlot(t, g, 77777, dim, 888)
	other := gpuReclaimVec(dim, 42)

	batch, err := g.gpuSearchBatch([][]float32{ghostVec, other}, 5)
	if err != nil {
		t.Fatalf("gpuSearchBatch: %v", err)
	}
	if len(batch) != 2 {
		t.Fatalf("batch len=%d, want 2", len(batch))
	}
	for qi, res := range batch {
		if len(res) != 5 {
			t.Fatalf("query %d: got %d results, want 5", qi, len(res))
		}
		for _, r := range res {
			if r.ID == 77777 {
				t.Fatalf("query %d: ghost id 77777 returned by gpuSearchBatch fast path: %v", qi, resultIDs(res))
			}
		}
	}
}

// TestGPUReclaimGhostNotReturnedBatchCPUFallback forces gpuSearchBatch's own
// internal fallback call to cpuExactSearchLocked (k > cuda.MaxK, same
// reasoning as the single-query forced-fallback test above, but through the
// batch entry point's separate "added < k" branch).
func TestGPUReclaimGhostNotReturnedBatchCPUFallback(t *testing.T) {
	const dim = 16
	g := newGPUReclaimIndex(t, dim)
	defer g.Close()

	const nDecoys = 500
	insertGPUDecoys(t, g, 1, nDecoys, dim, 400)
	ghostVec := reclaimGhostSlot(t, g, 66666, dim, 999)

	k := cuda.MaxK + 44
	batch, err := g.gpuSearchBatch([][]float32{ghostVec}, k)
	if err != nil {
		t.Fatalf("gpuSearchBatch: %v", err)
	}
	if len(batch) != 1 || len(batch[0]) != k {
		gotLen := 0
		if len(batch) == 1 {
			gotLen = len(batch[0])
		}
		t.Fatalf("batch result shape = %d quer(y/ies) x %d results, want 1 x %d", len(batch), gotLen, k)
	}
	for _, r := range batch[0] {
		if r.ID == 66666 {
			t.Fatalf("ghost id 66666 (reclaimed slot) returned by gpuSearchBatch's CPU fallback: %v", resultIDs(batch[0]))
		}
	}
}

// --- id 0 reachability across all three lanes ---

// TestGPUIDZeroFastPath mirrors id_zero_test.go's TestIDZeroSearchable cases
// for the GPU dense-search fast path: point id 0 must be returned at rank 0
// when queried with its own vector, whether it lands on slot 0 or a non-zero
// slot (decoys occupy the earlier slots).
func TestGPUIDZeroFastPath(t *testing.T) {
	const dim = 16
	cases := []struct {
		name  string
		order []uint64
	}{
		{"id0 first among many (slot 0)", idRange(0, 20)},
		{"id0 at nonzero slot", append([]uint64{901, 902, 903}, idRange(0, 20)...)},
	}
	for _, tc := range cases {
		tc := tc
		t.Run(tc.name, func(t *testing.T) {
			g := newGPUReclaimIndex(t, dim)
			defer g.Close()
			for _, id := range tc.order {
				if _, _, err := g.Insert(id, idZeroVec(dim, id), 0, nil, nil, nil, CASCond{}); err != nil {
					t.Fatalf("insert id %d: %v", id, err)
				}
			}
			res, err := g.Search(idZeroVec(dim, 0), 5)
			if err != nil {
				t.Fatal(err)
			}
			requireRankZero(t, res, 0, "GPU fast-path Search")
		})
	}
}

// TestGPUIDZeroCPUFallback is the forced-fallback (k > cuda.MaxK) analogue,
// pinning cpuExactSearchLocked's own copy of the Allocated() gate for id 0.
func TestGPUIDZeroCPUFallback(t *testing.T) {
	const dim = 16
	g := newGPUReclaimIndex(t, dim)
	defer g.Close()

	order := idRange(0, 500) // ids 0..499, corpus > cuda.MaxK
	for _, id := range order {
		if _, _, err := g.Insert(id, idZeroVec(dim, id), 0, nil, nil, nil, CASCond{}); err != nil {
			t.Fatalf("insert id %d: %v", id, err)
		}
	}
	k := cuda.MaxK + 44
	res, err := g.Search(idZeroVec(dim, 0), k)
	if err != nil {
		t.Fatal(err)
	}
	if len(res) != k {
		t.Fatalf("Search(k=%d) returned %d results, want %d", k, len(res), k)
	}
	requireRankZero(t, res, 0, "GPU CPU-exact-fallback Search")
}

// TestGPUIDZeroBatch pins id 0's reachability through gpuSearchBatch, on both
// sides of the fast-path/fallback split.
func TestGPUIDZeroBatch(t *testing.T) {
	const dim = 16

	t.Run("fast path", func(t *testing.T) {
		g := newGPUReclaimIndex(t, dim)
		defer g.Close()
		for _, id := range idRange(0, 20) {
			if _, _, err := g.Insert(id, idZeroVec(dim, id), 0, nil, nil, nil, CASCond{}); err != nil {
				t.Fatalf("insert id %d: %v", id, err)
			}
		}
		batch, err := g.gpuSearchBatch([][]float32{idZeroVec(dim, 0), idZeroVec(dim, 5)}, 5)
		if err != nil {
			t.Fatal(err)
		}
		if len(batch) != 2 {
			t.Fatalf("batch len=%d, want 2", len(batch))
		}
		requireRankZero(t, batch[0], 0, "gpuSearchBatch fast path (query=id0's own vector)")
		requireRankZero(t, batch[1], 5, "gpuSearchBatch fast path (query=id5's own vector, sanity control)")
	})

	t.Run("cpu fallback", func(t *testing.T) {
		g := newGPUReclaimIndex(t, dim)
		defer g.Close()
		for _, id := range idRange(0, 500) {
			if _, _, err := g.Insert(id, idZeroVec(dim, id), 0, nil, nil, nil, CASCond{}); err != nil {
				t.Fatalf("insert id %d: %v", id, err)
			}
		}
		k := cuda.MaxK + 44
		batch, err := g.gpuSearchBatch([][]float32{idZeroVec(dim, 0)}, k)
		if err != nil {
			t.Fatal(err)
		}
		if len(batch) != 1 || len(batch[0]) != k {
			t.Fatalf("batch result shape wrong: got %d queries, first has %d results, want 1 x %d", len(batch), len(batch[0]), k)
		}
		requireRankZero(t, batch[0], 0, "gpuSearchBatch CPU fallback (query=id0's own vector)")
	})
}

// --- Reserved-but-unwritten slots ---

// TestGPUReservedUnwrittenSlotsNotScored covers the third stale-slot family
// gpu_cuda.go's comments call out: arena.Reserve pre-sizes capacity ahead of
// a bulk/concurrent build (see build_concurrent.go, vamana.go, ivf.go, which
// all Reserve(n) then PutAt each slot). If that build is interrupted or only
// partially completes, the untouched tail reads back a.ids[slot] == 0 with no
// idMap entry — Allocated() must reject it (a live id 0 point, if one exists,
// is still told apart correctly via the idMap round-trip; a bare Reserve
// leaves idMap empty so nothing round-trips).
func TestGPUReservedUnwrittenSlotsNotScored(t *testing.T) {
	const dim = 8

	t.Run("pure reserve, nothing written", func(t *testing.T) {
		g := newGPUReclaimIndex(t, dim)
		defer g.Close()
		if err := g.hnsw.arena.Reserve(50); err != nil {
			t.Fatalf("arena.Reserve: %v", err)
		}
		q := gpuReclaimVec(dim, 1)
		got, err := g.Search(q, 10)
		if err != nil {
			t.Fatalf("Search: %v", err)
		}
		if len(got) != 0 {
			t.Fatalf("Search over fully-reserved, fully-unwritten capacity returned %d results, want 0: %v",
				len(got), resultIDs(got))
		}
	})

	t.Run("partial build: written prefix, unwritten tail", func(t *testing.T) {
		g := newGPUReclaimIndex(t, dim)
		defer g.Close()

		const total, written = 50, 30
		if err := g.hnsw.arena.Reserve(total); err != nil {
			t.Fatalf("arena.Reserve: %v", err)
		}
		rng := rand.New(rand.NewSource(9)) //nolint:gosec // deterministic test fixture
		wantIDs := make(map[uint64]bool, written)
		for slot := 0; slot < written; slot++ {
			id := uint64(5000 + slot)
			v := make([]float32, dim)
			for d := range v {
				v[d] = rng.Float32()*2 - 1
			}
			g.hnsw.arena.PutAt(uint32(slot), id, v) //nolint:gosec // slot < total < 2^32
			g.hnsw.arena.idMap[id] = uint32(slot)   //nolint:gosec // slot < total < 2^32
			wantIDs[id] = true
		}
		// slots [written, total) stay exactly as Reserve left them: this is the
		// "Reserve-then-failed-build" case — capacity exists, idMap does not.

		q := gpuReclaimVec(dim, 2)
		got, err := g.Search(q, total) // ask for more than `written` on purpose
		if err != nil {
			t.Fatalf("Search: %v", err)
		}
		if len(got) != written {
			t.Fatalf("Search(k=%d) over a partially-written arena returned %d results, want exactly %d (the written prefix): %v",
				total, len(got), written, resultIDs(got))
		}
		for _, r := range got {
			if !wantIDs[r.ID] {
				t.Fatalf("Search returned id %d, which was never written — a reserved-but-unwritten slot leaked through: %v",
					r.ID, resultIDs(got))
			}
		}
	})
}
