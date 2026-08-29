// SPDX-License-Identifier: Apache-2.0

package vector

import (
	"bytes"
	"errors"
	"math/rand"
	"path/filepath"
	"testing"
)

// TestGrowFailureRestoresIndex proves the non-terminal grow-failure contract: a
// transient failure of the mmap slab grow (disk-full / address-space exhaustion)
// no longer bricks the collection. growVecMmap restores a mapping of the old
// (append-only, therefore intact) data, the caller rebinds to it, and only the
// single operation that triggered the grow fails — the index stays un-poisoned
// and fully usable, and a later grow succeeds once the transient condition lifts.
//
// The failure is injected deterministically via growVecMmapFailpoint (the arena
// slab is mmap-backed here; the graph is heap-backed, so the failure lands in
// arena.growVecs, which fails BEFORE any side-array/idMap mutation — a clean,
// consistent no-op insert).
func TestGrowFailureRestoresIndex(t *testing.T) {
	t.Cleanup(func() { growVecMmapFailpoint = nil })

	// preN fills the initial mmap capacity exactly, so the next insert (id preN+1)
	// is the first that must grow the slab.
	const (
		dim  = 8
		preN = mmapInitVectors
	)

	// Both slabs are mmap-backed (MmapPath + GraphMmapPath), which SavePersist
	// requires. Within an insert, arena.Insert (and its grow) runs BEFORE
	// setNode/growLevel0, and both slabs share the same mmapInitVectors initial
	// capacity, so on the grow-triggering insert the arena grow fires the failpoint
	// first and Insert bails before the graph grow is reached — the injected failure
	// lands deterministically in arena.growVecs, a clean pre-mutation no-op.
	dir := t.TempDir()
	cfg := Config{
		Dim: dim, Metric: L2, M: 8, EfConstruction: 64, EfSearch: 32, Seed: 1,
		Quant: QuantSQ8, RescoreFactor: 3,
		QuantStorage:  QuantMmap,
		MmapPath:      filepath.Join(dir, "vecs.dat"),
		GraphMmapPath: filepath.Join(dir, "graph.dat"),
	}
	h, err := newHNSW(cfg)
	if err != nil {
		t.Fatalf("newHNSW(mmap): %v", err)
	}

	rng := rand.New(rand.NewSource(7))
	vecs := make([][]float32, preN+1)
	mkVec := func() []float32 {
		v := make([]float32, dim)
		for j := range v {
			v[j] = float32(rng.NormFloat64())
		}
		return v
	}

	// Fill the initial mmap capacity. None of these should grow the slab.
	for i := 0; i < preN; i++ {
		vecs[i] = mkVec()
		if _, _, err := h.Insert(uint64(i+1), vecs[i], 0, nil, nil, nil, CASCond{}); err != nil {
			t.Fatalf("pre-grow insert %d: %v", i+1, err)
		}
	}
	if h.arena.poisoned.Load() {
		t.Fatalf("index poisoned before any grow failure")
	}
	if got := h.arena.Size(); got != preN {
		t.Fatalf("Size after pre-grow inserts = %d, want %d", got, preN)
	}

	// Arm the failpoint and drive the grow: the next insert crosses the initial
	// capacity, so arena.growVecs → growVecMmap fires the failpoint.
	errInjected := errors.New("injected grow failure")
	growVecMmapFailpoint = func() error { return errInjected }

	vecs[preN] = mkVec()
	if _, _, err := h.Insert(uint64(preN+1), vecs[preN], 0, nil, nil, nil, CASCond{}); !errors.Is(err, errInjected) {
		t.Fatalf("grow-triggering insert: got err %v, want errInjected", err)
	}

	// Contract 1: the index must NOT be poisoned — the restore path kept it valid.
	if h.arena.poisoned.Load() {
		t.Fatalf("index poisoned after a RESTORABLE grow failure; want it to remain valid")
	}

	// Contract 2: the failed insert was a clean no-op — the pre-grow points are all
	// still present, and the failed id never landed.
	if got := h.arena.Size(); got != preN {
		t.Fatalf("Size after failed grow = %d, want %d (failed insert must be a no-op)", got, preN)
	}
	if _, ok := h.arena.Slot(uint64(preN + 1)); ok {
		t.Fatalf("failed insert id %d is present in the arena; want it absent", preN+1)
	}

	// Contract 3: Search still returns the pre-grow vectors correctly (reads go
	// through the restored mapping). Query with an exact stored vector; it must
	// come back as the nearest neighbour.
	res, err := h.Search(vecs[0], 5)
	if err != nil {
		t.Fatalf("Search after restored grow failure: %v", err)
	}
	if len(res) == 0 {
		t.Fatalf("Search returned no results after restored grow failure")
	}
	if res[0].ID != 1 {
		t.Fatalf("Search top hit = id %d, want id 1 (exact query vector)", res[0].ID)
	}

	// Contract 4: persistence paths work — the collection is fully serializable.
	var buf bytes.Buffer
	if err := h.Snapshot(&buf); err != nil {
		t.Fatalf("Snapshot after restored grow failure: %v", err)
	}
	if buf.Len() == 0 {
		t.Fatalf("Snapshot wrote 0 bytes after restored grow failure")
	}
	if err := h.SavePersist(filepath.Join(t.TempDir(), "meta")); err != nil {
		t.Fatalf("SavePersist after restored grow failure: %v", err)
	}

	// Contract 5: recovery is complete, not a dead end. Clear the failpoint and
	// retry the same insert — the grow now succeeds and the new vector is
	// searchable, proving the slab genuinely grew.
	growVecMmapFailpoint = nil
	if _, _, err := h.Insert(uint64(preN+1), vecs[preN], 0, nil, nil, nil, CASCond{}); err != nil {
		t.Fatalf("retry insert after clearing failpoint: %v", err)
	}
	if got := h.arena.Size(); got != preN+1 {
		t.Fatalf("Size after successful retry = %d, want %d", got, preN+1)
	}
	res, err = h.Search(vecs[preN], 5)
	if err != nil {
		t.Fatalf("Search for post-recovery vector: %v", err)
	}
	if len(res) == 0 || res[0].ID != uint64(preN+1) {
		t.Fatalf("post-recovery Search top hit = %+v, want id %d", res, preN+1)
	}
}

// TestGraphGrowFailurePoisonsIndex is the counterpart to TestGrowFailureRestoresIndex
// for the OTHER slab: the level-0 GRAPH slab, whose grow runs POST-commit. Here a
// restored grow failure must NOT be treated as recoverable — the point has already
// been committed to idMap/indexes by the time growLevel0 runs (via setNode), so a
// "restored but usable" outcome would leave a torn half-insert. The contract under
// test is therefore the opposite of the arena path: after a graph-grow failure the
// index must be cleanly POISONED and every op must reject with ErrIndexPoisoned
// (never nil-deref h.nodes[slot] / read the unmapped region), which is strictly
// safer than the torn state.
//
// Only the graph slab is mmap-backed (GraphMmapPath); the arena is heap-backed (no
// QuantMmap), so growVecMmap — and thus the failpoint — is reached ONLY from
// growLevel0, never from arena.growVecs. Both slabs start at mmapInitVectors
// capacity, and the arena's heap grow does not go through growVecMmap, so the
// failpoint armed after preN inserts fires precisely on the graph grow at preN+1.
func TestGraphGrowFailurePoisonsIndex(t *testing.T) {
	t.Cleanup(func() { growVecMmapFailpoint = nil })

	const (
		dim  = 8
		preN = mmapInitVectors
	)

	cfg := Config{
		Dim: dim, Metric: L2, M: 8, EfConstruction: 64, EfSearch: 32, Seed: 1,
		// No QuantMmap → heap arena. Graph slab is the only mmap-backed slab.
		GraphMmapPath: filepath.Join(t.TempDir(), "graph.dat"),
	}
	h, err := newHNSW(cfg)
	if err != nil {
		t.Fatalf("newHNSW(graph-mmap): %v", err)
	}

	rng := rand.New(rand.NewSource(11))
	mkVec := func() []float32 {
		v := make([]float32, dim)
		for j := range v {
			v[j] = float32(rng.NormFloat64())
		}
		return v
	}

	for i := 0; i < preN; i++ {
		if _, _, err := h.Insert(uint64(i+1), mkVec(), 0, nil, nil, nil, CASCond{}); err != nil {
			t.Fatalf("pre-grow insert %d: %v", i+1, err)
		}
	}
	if h.arena.poisoned.Load() {
		t.Fatalf("index poisoned before any grow failure")
	}

	// Arm the failpoint and drive the graph grow: the arena's heap grow does not go
	// through growVecMmap, so this fires only in growLevel0, AFTER the point has been
	// committed to the arena.
	errInjected := errors.New("injected graph grow failure")
	growVecMmapFailpoint = func() error { return errInjected }

	if _, _, err := h.Insert(uint64(preN+1), mkVec(), 0, nil, nil, nil, CASCond{}); err == nil {
		t.Fatalf("grow-triggering insert: got nil err, want a failure")
	}

	// Contract: a POST-commit graph-grow failure is terminal. The index must be
	// poisoned — NOT left in a torn/half-committed state.
	if !h.arena.poisoned.Load() {
		t.Fatalf("index NOT poisoned after a post-commit graph-grow failure; a restored-but-usable graph slab would leave a torn insert")
	}

	// Every op must reject cleanly with ErrIndexPoisoned (and never nil-deref the
	// nil graph node / read the region), proving the guard covers the torn point.
	t.Run("Search", func(t *testing.T) {
		defer mustNotPanic(t)
		if _, err := h.Search([]float32{1, 0, 0, 0, 0, 0, 0, 0}, 5); !errors.Is(err, ErrIndexPoisoned) {
			t.Fatalf("Search: got %v, want ErrIndexPoisoned", err)
		}
	})
	t.Run("Insert", func(t *testing.T) {
		defer mustNotPanic(t)
		if _, _, err := h.Insert(uint64(preN+2), []float32{0, 1, 0, 0, 0, 0, 0, 0}, 0, nil, nil, nil, CASCond{}); !errors.Is(err, ErrIndexPoisoned) {
			t.Fatalf("Insert: got %v, want ErrIndexPoisoned", err)
		}
	})
	t.Run("Snapshot", func(t *testing.T) {
		defer mustNotPanic(t)
		var buf bytes.Buffer
		if err := h.Snapshot(&buf); !errors.Is(err, ErrIndexPoisoned) {
			t.Fatalf("Snapshot: got %v, want ErrIndexPoisoned", err)
		}
		if buf.Len() != 0 {
			t.Fatalf("Snapshot wrote %d bytes on a poisoned index; want a clean refusal", buf.Len())
		}
	})
	t.Run("SavePersist", func(t *testing.T) {
		defer mustNotPanic(t)
		if err := h.SavePersist(t.TempDir() + "/meta"); !errors.Is(err, ErrIndexPoisoned) {
			t.Fatalf("SavePersist: got %v, want ErrIndexPoisoned", err)
		}
	})
	t.Run("Get", func(t *testing.T) {
		// No error return; the contract is panic-safety on the torn slot.
		defer mustNotPanic(t)
		_, _, _, _, _, _ = h.Get(1)
	})
}
