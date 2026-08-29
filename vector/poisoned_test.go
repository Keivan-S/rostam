// SPDX-License-Identifier: Apache-2.0

package vector

import (
	"bytes"
	"errors"
	"testing"
	"time"
)

// TestPoisonedIndexRejectsOps proves the terminal-state guard: once a slab
// growth has failed (which nils the freed mmap region), every public op must
// reject with ErrIndexPoisoned instead of dereferencing the freed backing.
//
// Test seam: the poisoned flag is an unexported atomic.Bool on the arena, so a
// white-box test in package vector sets it directly (h.arena.poisoned.Store) —
// the same state the real growVecs/growLevel0 failure branches reach. This
// proves the GUARD (no op touches the freed region) without having to simulate
// an ENOSPC on the mmap grow, which is not portably injectable here. The Windows
// mmap path that produces the real failure is exercised by the windows/amd64 CI
// lane; this test locks down the platform-agnostic rejection contract on Linux.
func TestPoisonedIndexRejectsOps(t *testing.T) {
	cfg := Config{Dim: 4, Metric: L2, M: 16, EfConstruction: 100, EfSearch: 32, Seed: 1}
	h, err := newHNSW(cfg)
	if err != nil {
		t.Fatalf("newHNSW: %v", err)
	}

	// Seed one live point so the ops have real work they would otherwise do.
	if _, _, err := h.Insert(1, []float32{1, 0, 0, 0}, 0, nil, nil, nil, CASCond{}); err != nil {
		t.Fatalf("seed Insert: %v", err)
	}

	// Drive the index into the terminal state, exactly as a failed mmap grow does.
	h.arena.poisoned.Store(true)

	t.Run("Search", func(t *testing.T) {
		defer mustNotPanic(t)
		if _, err := h.Search([]float32{1, 0, 0, 0}, 5); !errors.Is(err, ErrIndexPoisoned) {
			t.Fatalf("Search: got %v, want ErrIndexPoisoned", err)
		}
	})

	t.Run("Insert", func(t *testing.T) {
		defer mustNotPanic(t)
		if _, _, err := h.Insert(2, []float32{0, 1, 0, 0}, 0, nil, nil, nil, CASCond{}); !errors.Is(err, ErrIndexPoisoned) {
			t.Fatalf("Insert: got %v, want ErrIndexPoisoned", err)
		}
	})

	t.Run("InsertAt", func(t *testing.T) {
		defer mustNotPanic(t)
		if _, _, err := h.InsertAt(3, []float32{0, 0, 1, 0}, 0, nil, nil, nil, CASCond{}, time.Now().UnixMilli()); !errors.Is(err, ErrIndexPoisoned) {
			t.Fatalf("InsertAt: got %v, want ErrIndexPoisoned", err)
		}
	})

	t.Run("SavePersist", func(t *testing.T) {
		defer mustNotPanic(t)
		if err := h.SavePersist(t.TempDir() + "/meta"); !errors.Is(err, ErrIndexPoisoned) {
			t.Fatalf("SavePersist: got %v, want ErrIndexPoisoned", err)
		}
	})

	t.Run("Snapshot", func(t *testing.T) {
		defer mustNotPanic(t)
		var buf bytes.Buffer
		err := h.Snapshot(&buf)
		if !errors.Is(err, ErrIndexPoisoned) {
			t.Fatalf("Snapshot: got %v, want ErrIndexPoisoned", err)
		}
		// The refusal must happen BEFORE any bytes are written — no silently
		// truncated checkpoint (Capacity()/vecs are 0 after the poison).
		if buf.Len() != 0 {
			t.Fatalf("Snapshot wrote %d bytes on a poisoned index; want a clean refusal with no output", buf.Len())
		}
	})

	t.Run("Get", func(t *testing.T) {
		// Get has no error return; the contract here is panic-safety (vecFor returns
		// nil rather than slicing the freed arena region).
		defer mustNotPanic(t)
		_, _, _, _, _, _ = h.Get(1)
	})

	t.Run("SearchMMR", func(t *testing.T) {
		defer mustNotPanic(t)
		if _, err := h.SearchMMR([]float32{1, 0, 0, 0}, 5, MMROpts{}); !errors.Is(err, ErrIndexPoisoned) {
			t.Fatalf("SearchMMR: got %v, want ErrIndexPoisoned", err)
		}
	})

	t.Run("GroupCandidates", func(t *testing.T) {
		defer mustNotPanic(t)
		// GroupBy must be non-empty to reach the under-lock guard (an empty GroupBy
		// short-circuits to ErrEmptyGroupBy before the lock).
		if _, err := h.GroupCandidates([]float32{1, 0, 0, 0}, GroupOpts{GroupBy: "doc"}); !errors.Is(err, ErrIndexPoisoned) {
			t.Fatalf("GroupCandidates: got %v, want ErrIndexPoisoned", err)
		}
	})

	t.Run("HybridSearch", func(t *testing.T) {
		defer mustNotPanic(t)
		if _, err := h.HybridSearch([]float32{1, 0, 0, 0}, SparseVector{}, 5, HybridOpts{}); !errors.Is(err, ErrIndexPoisoned) {
			t.Fatalf("HybridSearch: got %v, want ErrIndexPoisoned", err)
		}
	})
}

// TestPoisonedIVFRejectsOps is the IVF counterpart of TestPoisonedIndexRejectsOps.
// The *ivf index type shares the same arena (and thus the same poisoned flag) as
// *hnsw and is poisoned via the same arena.growVecs mmap-grow failure, so every
// IVF op must reject or stay panic-safe too. Same white-box seam
// (ix.arena.poisoned.Store). The index is left untrained (a handful of inserts):
// the guards sit above the trained/untrained branch, so training is irrelevant to
// what is under test.
func TestPoisonedIVFRejectsOps(t *testing.T) {
	ix, err := newIVF(ivfTestConfig(4))
	if err != nil {
		t.Fatalf("newIVF: %v", err)
	}
	seed := [][]float32{{1, 0, 0, 0}, {0, 1, 0, 0}, {0, 0, 1, 0}}
	for i, v := range seed {
		if _, _, err := ix.Insert(uint64(i+1), v, 0, nil, nil, nil, CASCond{}); err != nil {
			t.Fatalf("seed Insert %d: %v", i, err)
		}
	}

	ix.arena.poisoned.Store(true)

	q := []float32{1, 0, 0, 0}

	t.Run("Snapshot", func(t *testing.T) {
		defer mustNotPanic(t)
		var buf bytes.Buffer
		if err := ix.Snapshot(&buf); !errors.Is(err, ErrIndexPoisoned) {
			t.Fatalf("Snapshot: got %v, want ErrIndexPoisoned", err)
		}
		if buf.Len() != 0 {
			t.Fatalf("Snapshot wrote %d bytes on a poisoned index; want a clean refusal with no output", buf.Len())
		}
	})

	t.Run("SavePersist", func(t *testing.T) {
		defer mustNotPanic(t)
		if err := ix.SavePersist(t.TempDir() + "/meta"); !errors.Is(err, ErrIndexPoisoned) {
			t.Fatalf("SavePersist: got %v, want ErrIndexPoisoned", err)
		}
	})

	t.Run("Search", func(t *testing.T) {
		defer mustNotPanic(t)
		if _, err := ix.Search(q, 5); !errors.Is(err, ErrIndexPoisoned) {
			t.Fatalf("Search: got %v, want ErrIndexPoisoned", err)
		}
	})

	t.Run("SearchMMR", func(t *testing.T) {
		defer mustNotPanic(t)
		if _, err := ix.SearchMMR(q, 5, MMROpts{}); !errors.Is(err, ErrIndexPoisoned) {
			t.Fatalf("SearchMMR: got %v, want ErrIndexPoisoned", err)
		}
	})

	t.Run("GroupCandidates", func(t *testing.T) {
		defer mustNotPanic(t)
		if _, err := ix.GroupCandidates(q, GroupOpts{GroupBy: "doc"}); !errors.Is(err, ErrIndexPoisoned) {
			t.Fatalf("GroupCandidates: got %v, want ErrIndexPoisoned", err)
		}
	})

	t.Run("HybridSearch", func(t *testing.T) {
		defer mustNotPanic(t)
		if _, err := ix.HybridSearch(q, SparseVector{}, 5, HybridOpts{}); !errors.Is(err, ErrIndexPoisoned) {
			t.Fatalf("HybridSearch: got %v, want ErrIndexPoisoned", err)
		}
	})

	t.Run("Recommend", func(t *testing.T) {
		// Reaches the guard right after RLock, before meanOf dereferences the freed
		// arena floats element-wise.
		defer mustNotPanic(t)
		if _, err := ix.Recommend(5, RecommendOpts{Positive: []uint64{1}}); !errors.Is(err, ErrIndexPoisoned) {
			t.Fatalf("Recommend: got %v, want ErrIndexPoisoned", err)
		}
	})

	t.Run("DiscoverVecs", func(t *testing.T) {
		defer mustNotPanic(t)
		opts := DiscoverVecsOpts{Context: []DiscoverPair{{Pos: q, Neg: []float32{0, 1, 0, 0}}}}
		if _, err := ix.DiscoverVecs(5, opts); !errors.Is(err, ErrIndexPoisoned) {
			t.Fatalf("DiscoverVecs: got %v, want ErrIndexPoisoned", err)
		}
	})

	t.Run("Get", func(t *testing.T) {
		// No error return; the contract is panic-safety (vecFor returns nil rather
		// than slicing the freed arena region).
		defer mustNotPanic(t)
		_, _, _, _, _, _ = ix.Get(1)
	})
}

func mustNotPanic(t *testing.T) {
	t.Helper()
	if r := recover(); r != nil {
		t.Fatalf("op panicked on poisoned index instead of returning ErrIndexPoisoned: %v", r)
	}
}
