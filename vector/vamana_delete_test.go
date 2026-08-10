// SPDX-License-Identifier: Apache-2.0

package vector

import "testing"

// liveMedoidSlot recomputes the medoid (live point nearest the sample mean) over
// the CURRENTLY LIVE set, treating the given excluded slot as already removed —
// the ground truth the post-deletion entry point should equal. It mirrors
// medoidSlot but skips the excluded slot and any tombstoned slot so the test does
// not depend on whether medoidSlot itself filters tombstones.
func liveMedoidSlot(t *testing.T, h *hnsw, excluded uint32) uint32 {
	t.Helper()
	mean := make([]float32, h.cfg.Dim)
	count := 0
	for _, nd := range h.nodes {
		if nd == nil || nd.slot == excluded || h.tombstoned[nd.slot] {
			continue
		}
		v := h.arena.Vec(nd.slot)
		for i, x := range v {
			mean[i] += x
		}
		count++
	}
	if count == 0 {
		t.Fatal("no live points remain")
	}
	inv := float32(1) / float32(count)
	for i := range mean {
		mean[i] *= inv
	}
	dist := h.metricDist()
	var best uint32
	bestD := float32(1e38)
	found := false
	for _, nd := range h.nodes {
		if nd == nil || nd.slot == excluded || h.tombstoned[nd.slot] {
			continue
		}
		if d := dist(mean, h.arena.Vec(nd.slot)); !found || d < bestD {
			found, bestD, best = true, d, nd.slot
		}
	}
	return best
}

// TestVamanaReElectMedoidOnEntryPointDeletion is the regression for the MAJOR:
// when the Vamana entry point (the medoid) is removed, electEntryPoint must
// recompute the medoid of the remaining live set rather than fall back to the
// HNSW "highest-level node" scan, which on a single-layer Vamana graph picks the
// lowest-indexed live slot — an arbitrary PERIPHERAL node. A peripheral entry
// point makes greedy search start from the graph boundary and degrades recall.
//
// The entry-point assertion (new entry point == recomputed live medoid) FAILS
// pre-fix: the old code would set entryPoint to slot 0 (or the first live slot),
// not the medoid. The recall assertion documents that a central entry point keeps
// recall high.
func TestVamanaReElectMedoidOnEntryPointDeletion(t *testing.T) {
	if testing.Short() {
		t.Skip("skipping recall test in -short mode")
	}
	const (
		n        = 4_000
		dim      = 128
		nq       = 50
		k        = 10
		seed     = 11
		clusters = 64
		noise    = 0.15
	)
	corpus, queries, _ := vamanaClusteredCorpus(t, n, dim, nq, clusters, k, seed, noise)

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

	// The build elected the medoid as the entry point. Capture it, then build the
	// ground-truth set excluding it.
	vm.mu.Lock()
	medoidSlot := vm.entryPoint
	if vm.maxLevel != 0 {
		vm.mu.Unlock()
		t.Fatalf("Vamana maxLevel = %d, want 0 (single layer)", vm.maxLevel)
	}
	wantSlot := liveMedoidSlot(t, vm, medoidSlot)

	// Recall ground truth must EXCLUDE the deleted id, since Search filters it.
	deletedID := vm.arena.ID(medoidSlot)
	vm.mu.Unlock()

	// Recall@k over the remaining live set (recompute ground truth on the fly so the
	// deleted id is never counted as a true neighbor).
	recallExcluding := func() float64 {
		var matches int
		for _, q := range queries {
			qn := append([]float32(nil), q...)
			normalize(qn)
			type pair struct {
				id  uint64
				dot float32
			}
			dists := make([]pair, 0, n)
			for i, v := range corpus {
				id := uint64(i + 1)
				if id == deletedID {
					continue
				}
				vn := append([]float32(nil), v...)
				normalize(vn)
				dists = append(dists, pair{id: id, dot: dotScalar(qn, vn)})
			}
			for i := 1; i < len(dists); i++ { // insertion sort top-k is overkill; full sort
				for j := i; j > 0 && dists[j].dot > dists[j-1].dot; j-- {
					dists[j], dists[j-1] = dists[j-1], dists[j]
				}
			}
			truth := make(map[uint64]bool, k)
			for i := 0; i < k && i < len(dists); i++ {
				truth[dists[i].id] = true
			}
			results, err := vm.Search(q, k)
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

	// ---- Delete the medoid (the entry point) and re-elect ----
	// Plain Delete tombstones the slot; the entry-point HARD-removal path (upsert
	// replace) is what invokes electEntryPoint in production. Drive the same
	// re-election directly here: tombstone the medoid slot and re-elect, exactly the
	// sequence the hard-removal path runs when the entry-point id is replaced.
	if _, err := vm.Delete(deletedID, CASCond{}); err != nil {
		t.Fatalf("delete medoid: %v", err)
	}
	vm.mu.Lock()
	vm.electEntryPoint()
	gotSlot := vm.entryPoint
	gotLevel := vm.maxLevel
	vm.mu.Unlock()

	// (a) The new entry point IS the recomputed live medoid (definitely fails
	// pre-fix, which fell back to the lowest-indexed live slot).
	if gotSlot != wantSlot {
		t.Errorf("re-elected entry point slot = %d, want medoid slot %d", gotSlot, wantSlot)
	}
	if gotLevel != 0 {
		t.Errorf("re-elected maxLevel = %d, want 0", gotLevel)
	}

	// (b) Recall over the remaining live set stays high — a central medoid entry
	// keeps greedy search effective after deletion.
	r := recallExcluding()
	t.Logf("recall@%d after medoid deletion + re-election = %.3f (entry slot %d)", k, r, gotSlot)
	if r < 0.90 {
		t.Errorf("recall@%d after medoid deletion = %.3f, want >= 0.90", k, r)
	}
}
