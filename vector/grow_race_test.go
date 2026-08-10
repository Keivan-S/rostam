// SPDX-License-Identifier: Apache-2.0

package vector

import (
	"fmt"
	"math/rand"
	"path/filepath"
	"sync"
	"sync/atomic"
	"testing"
)

// THE SLAB-GROWTH INVARIANT.
//
// arena.vecs and the level-0 graph slab (hnsw.level0 / level0Len) grow by
// reallocation — a fresh Go backing array in heap mode, a REMAP of the backing
// file in mmap mode (arena.growVecs, hnsw.growGraphMmap). A remap moves the
// mapping: every slice header derived from it is rebuilt, and the OLD address
// range is unmapped. A reader still walking the old range would take a SIGBUS,
// which no amount of -race instrumentation catches and no recover() survives.
//
// The invariant that makes this safe is stated at graph.go's ensureGraphSlot and
// arena.go's Reserve: EVERY growth site runs under h.mu's WRITE lock, and every
// reader of the slabs holds at least h.mu's READ lock for the whole span in
// which it dereferences them. sync.RWMutex gives mutual exclusion between the
// two, so a remap can never overlap a reader.
//
// These tests are the executable form of that invariant. They hammer searches
// against inserts that cross MANY growth boundaries — the heap doubling ladder
// and, for the mmap variants, dozens of real munmap/mmap cycles — on both
// backings. Under `go test -race` a violation surfaces as a race report on the
// slice headers; without -race a genuine remap-under-reader surfaces as a
// SIGBUS crash. They are deliberately written against the PUBLIC hnsw surface
// (Insert/Search/Get) so they keep their meaning across relocking work: the
// contract they pin is "growth is invisible to a concurrent reader", not any
// particular placement of the lock boundaries inside Insert.
//
// Growth boundaries crossed: the arena starts at capacity 0 and the graph slab
// at mmapInitVectors (1024) slots, both doubling, so a few thousand inserts
// crosses a dozen heap boundaries and, under mmap, several real remaps of each
// region. TestGrowBoundaryRemapCountMmap asserts that count directly rather
// than assuming it.

// growHammer drives `writers` goroutines inserting disjoint id ranges into h
// while `readers` goroutines search it continuously, then verifies every
// inserted id is retrievable. Each writer inserts n/writers vectors, so the
// arena and the graph slab grow repeatedly underneath the live readers.
func growHammer(t *testing.T, h *hnsw, n, dim, writers, readers int) {
	t.Helper()
	rng := rand.New(rand.NewSource(int64(n*31 + dim)))
	vecs := make([][]float32, n)
	for i := range vecs {
		v := make([]float32, dim)
		for d := range v {
			v[d] = rng.Float32()
		}
		vecs[i] = v
	}

	var stop atomic.Bool
	var rwg, wwg sync.WaitGroup
	var searches atomic.Int64

	for r := 0; r < readers; r++ {
		rwg.Add(1)
		go func(seed int64) {
			defer rwg.Done()
			qrng := rand.New(rand.NewSource(seed))
			q := make([]float32, dim)
			var dst []Result
			for !stop.Load() {
				for d := range q {
					q[d] = qrng.Float32()
				}
				var err error
				dst, err = h.SearchInto(dst[:0], q, 10, Filter{})
				if err != nil {
					t.Errorf("search: %v", err)
					return
				}
				searches.Add(1)
				// Get exercises the arena side-arrays (ids/versions/metadata),
				// which grow by append on the same write lock.
				_, _, _, _, _, _ = h.Get(uint64(qrng.Intn(n)))
			}
		}(int64(r) + 1)
	}

	per := n / writers
	for w := 0; w < writers; w++ {
		wwg.Add(1)
		go func(lo, hi int) {
			defer wwg.Done()
			for i := lo; i < hi; i++ {
				if _, _, err := h.Insert(uint64(i), vecs[i], 0, nil, nil, nil, CASCond{}); err != nil {
					t.Errorf("insert %d: %v", i, err)
					return
				}
			}
		}(w*per, (w+1)*per)
	}

	wwg.Wait() // every insert (and its link phase) has returned
	stop.Store(true)
	rwg.Wait()

	if got := searches.Load(); got == 0 {
		t.Fatal("no searches ran concurrently with the inserts — test is not exercising the invariant")
	}
	for i := 0; i < per*writers; i++ {
		if _, _, _, _, _, ok := h.Get(uint64(i)); !ok {
			t.Fatalf("id %d missing after concurrent grow", i)
		}
	}
	t.Logf("inserted=%d searches=%d capacity=%d level0=%d", per*writers, searches.Load(), h.arena.Capacity(), len(h.level0))
}

// TestGrowBoundaryConcurrentHeap crosses the heap doubling ladder for both the
// arena vectors and the level-0 slab with readers live throughout.
func TestGrowBoundaryConcurrentHeap(t *testing.T) {
	cfg := Config{Dim: 16, Metric: L2, M: 8, EfConstruction: 64, EfSearch: 32, Seed: 7}
	h, err := newHNSW(cfg)
	if err != nil {
		t.Fatal(err)
	}
	growHammer(t, h, 4000, 16, 3, 4)
}

// TestGrowBoundaryConcurrentMmap is the same hammer with BOTH slabs backed by
// memory-mapped files, so every growth boundary is a real munmap/mmap that
// MOVES the mapping (growVecMmap). This is the variant that would SIGBUS — not
// merely race — if a reader could observe a remap.
func TestGrowBoundaryConcurrentMmap(t *testing.T) {
	dir := t.TempDir()
	cfg := Config{
		Dim: 16, Metric: L2, M: 8, EfConstruction: 64, EfSearch: 32, Seed: 7,
		Quant:         QuantSQ8,
		QuantStorage:  QuantMmap,
		MmapPath:      filepath.Join(dir, "vecs.dat"),
		GraphMmapPath: filepath.Join(dir, "graph.dat"),
	}
	h, err := newHNSW(cfg)
	if err != nil {
		t.Fatal(err)
	}
	defer func() { _ = h.Close() }()
	growHammer(t, h, 4000, 16, 3, 4)
}

// TestGrowBoundaryRemapCountMmap pins that the mmap variant really does cross
// multiple REMAP boundaries under concurrent readers rather than sizing the
// mapping once: it counts the distinct base addresses the graph region takes
// while a reader hammers the index. A single distinct address would mean the
// test above proves nothing about remapping.
func TestGrowBoundaryRemapCountMmap(t *testing.T) {
	dir := t.TempDir()
	cfg := Config{
		Dim: 8, Metric: L2, M: 8, EfConstruction: 32, EfSearch: 16, Seed: 3,
		Quant:         QuantSQ8,
		QuantStorage:  QuantMmap,
		MmapPath:      filepath.Join(dir, "vecs.dat"),
		GraphMmapPath: filepath.Join(dir, "graph.dat"),
	}
	h, err := newHNSW(cfg)
	if err != nil {
		t.Fatal(err)
	}
	defer func() { _ = h.Close() }()

	const n = 6000
	rng := rand.New(rand.NewSource(11))
	var stop atomic.Bool
	var wg sync.WaitGroup
	wg.Add(1)
	go func() {
		defer wg.Done()
		qrng := rand.New(rand.NewSource(99))
		q := make([]float32, cfg.Dim)
		var dst []Result
		for !stop.Load() {
			for d := range q {
				q[d] = qrng.Float32()
			}
			var err error
			if dst, err = h.SearchInto(dst[:0], q, 5, Filter{}); err != nil {
				t.Errorf("search: %v", err)
				return
			}
		}
	}()

	bases := map[string]bool{}
	v := make([]float32, cfg.Dim)
	for i := 0; i < n; i++ {
		for d := range v {
			v[d] = rng.Float32()
		}
		if _, _, err := h.Insert(uint64(i), v, 0, nil, nil, nil, CASCond{}); err != nil {
			t.Fatalf("insert %d: %v", i, err)
		}
		h.mu.RLock()
		if len(h.graphRegion) > 0 {
			bases[fmt.Sprintf("%p:%d", &h.graphRegion[0], len(h.graphRegion))] = true
		}
		if len(h.arena.mmapRegion) > 0 {
			bases[fmt.Sprintf("v%p:%d", &h.arena.mmapRegion[0], len(h.arena.mmapRegion))] = true
		}
		h.mu.RUnlock()
	}
	stop.Store(true)
	wg.Wait()

	if len(bases) < 4 {
		t.Fatalf("only %d distinct mmap regions observed — the test did not cross enough remap boundaries", len(bases))
	}
	t.Logf("distinct mmap regions observed across %d inserts: %d", n, len(bases))
}
