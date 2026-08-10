// SPDX-License-Identifier: Apache-2.0

package vector

import (
	"math/rand"
	"testing"
)

// TestIVFPQOnlyReclaimNoFloatRead is a regression guard for the PQ-only IVF
// Reclaim panic (fixed in b1e3d9a): rebuildListsLocked used to read arena.Vec
// for every live slot, but a PQ-only build (IVFPQ=true, IVFRerank=false) drops
// the resident floats after encoding (pqDropped, arena.vecs==nil). Calling
// Reclaim() — which re-files the lists after freeing tombstoned slots — then hit
// the dropped float buffer and panicked with "slice bounds out of range ...
// capacity 0". The fix rebuilds the lists from ix.slotCell (the already-recorded
// per-slot cell), reading no floats. This test exercises exactly that path: it
// builds a PQ-only index, deletes points, calls Reclaim() directly, and asserts
// no panic plus a correct post-reclaim Search (survivors found, deleted excluded).
func TestIVFPQOnlyReclaimNoFloatRead(t *testing.T) {
	const (
		dim   = 32
		n     = 1500
		nlist = 32
		m     = 8
		k     = 10
		nprb  = 16
	)
	rng := rand.New(rand.NewSource(7))
	vecs := makeClustered(rng, n, dim, 24, 0.20)
	ids := make([]uint64, n)
	for i := range ids {
		ids[i] = uint64(i + 1)
	}

	ix, err := newIVF(ivfPQConfig(dim, nlist, m, false)) // PQ-only: floats dropped.
	if err != nil {
		t.Fatal(err)
	}
	if err := ix.BuildConcurrent(ids, vecs, 0); err != nil {
		t.Fatal(err)
	}
	ix.nprobe = nprb

	// Precondition for the bug: this is genuinely the PQ-only, floats-dropped path.
	if !ix.pqActive() {
		t.Fatal("PQ-only: pq codec not trained after build")
	}
	if !ix.pqDropped || ix.arena.vecs != nil {
		t.Fatal("PQ-only: resident floats not dropped — test would not exercise the bug")
	}

	// Delete a sizable chunk so Reclaim has real work and the lists must shrink.
	deleted := make(map[uint64]bool)
	for i := 0; i < n; i += 3 { // ~500 deletions
		id := ids[i]
		ok, err := ix.Delete(id, CASCond{})
		if err != nil {
			t.Fatalf("Delete(%d): %v", id, err)
		}
		if !ok {
			t.Fatalf("Delete(%d) returned false", id)
		}
		deleted[id] = true
	}
	if len(ix.tombstoned) != len(deleted) {
		t.Fatalf("tombstoned = %d, want %d", len(ix.tombstoned), len(deleted))
	}

	// The fix under test: pre-fix this panicked ("slice bounds out of range ...
	// capacity 0") because rebuildListsLocked read arena.Vec on the dropped floats.
	removed := ix.Reclaim()
	if removed != len(deleted) {
		t.Fatalf("Reclaim removed %d, want %d", removed, len(deleted))
	}
	if len(ix.tombstoned) != 0 {
		t.Fatalf("post-reclaim tombstoned = %d, want 0", len(ix.tombstoned))
	}
	// Still PQ-only after reclaim (floats never came back).
	if !ix.pqDropped || ix.arena.vecs != nil {
		t.Fatal("post-reclaim: floats unexpectedly resident")
	}

	// Post-reclaim Search must still work and must never return a reclaimed id,
	// while still finding live survivors. Query around each cluster of survivors.
	foundAnyLive := false
	for qi := 0; qi < n; qi += 137 {
		q := vecs[qi]
		got, err := ix.Search(q, k)
		if err != nil {
			t.Fatalf("post-reclaim Search: %v", err)
		}
		for _, r := range got {
			if deleted[r.ID] {
				t.Fatalf("post-reclaim Search returned reclaimed id %d", r.ID)
			}
			if _, ok := ix.arena.Slot(r.ID); !ok {
				t.Fatalf("post-reclaim Search returned id %d absent from arena", r.ID)
			}
			foundAnyLive = true
		}
	}
	if !foundAnyLive {
		t.Fatal("post-reclaim Search returned no live results across all probes")
	}

	// A live (never-deleted) point must still be retrievable.
	var liveID uint64
	for _, id := range ids {
		if !deleted[id] {
			liveID = id
			break
		}
	}
	if _, _, _, _, _, ok := ix.Get(liveID); !ok {
		t.Fatalf("post-reclaim Get(%d) (a survivor) failed", liveID)
	}
}

// TestIVFPQRerankReclaim covers the companion non-dropped path: with IVFRerank
// (floats kept resident), Reclaim takes the normal rebuildListsLocked branch
// that does read arena.Vec — confirming the fix did not regress it.
func TestIVFPQRerankReclaim(t *testing.T) {
	const (
		dim   = 24
		n     = 800
		nlist = 24
		m     = 6
		k     = 10
		nprb  = 12
	)
	rng := rand.New(rand.NewSource(9))
	vecs := makeClustered(rng, n, dim, 16, 0.20)
	ids := make([]uint64, n)
	for i := range ids {
		ids[i] = uint64(i + 1)
	}

	ix, err := newIVF(ivfPQConfig(dim, nlist, m, true)) // rerank: floats kept.
	if err != nil {
		t.Fatal(err)
	}
	if err := ix.BuildConcurrent(ids, vecs, 0); err != nil {
		t.Fatal(err)
	}
	ix.nprobe = nprb
	if ix.pqDropped || ix.arena.vecs == nil {
		t.Fatal("IVFRerank: floats must stay resident")
	}

	deleted := make(map[uint64]bool)
	for i := 0; i < n; i += 4 {
		id := ids[i]
		if ok, err := ix.Delete(id, CASCond{}); err != nil || !ok {
			t.Fatalf("Delete(%d): ok=%v err=%v", id, ok, err)
		}
		deleted[id] = true
	}

	removed := ix.Reclaim()
	if removed != len(deleted) {
		t.Fatalf("Reclaim removed %d, want %d", removed, len(deleted))
	}
	for qi := 0; qi < n; qi += 97 {
		got, err := ix.Search(vecs[qi], k)
		if err != nil {
			t.Fatalf("post-reclaim Search: %v", err)
		}
		for _, r := range got {
			if deleted[r.ID] {
				t.Fatalf("post-reclaim Search returned reclaimed id %d", r.ID)
			}
		}
	}
}
