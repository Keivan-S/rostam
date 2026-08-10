// SPDX-License-Identifier: Apache-2.0

package vector

import (
	"reflect"
	"testing"
)

// getIntoIndexes builds one small hnsw and one small ivf index, each populated
// with the same points, so the GetInto tests run against every VectorIndex
// implementer through the shared interface.
func getIntoIndexes(t *testing.T) map[string]VectorIndex {
	t.Helper()
	const dim = 4

	h, err := newHNSW(Config{Dim: dim, Metric: L2, M: 16, EfConstruction: 100, EfSearch: 64, Seed: 1})
	if err != nil {
		t.Fatalf("newHNSW: %v", err)
	}
	ivfCfg := Config{Dim: dim, Metric: L2, M: 16, EfConstruction: 100, EfSearch: 64, Seed: 1}
	ivfCfg.IndexType = IndexIVF
	ix, err := newIVF(ivfCfg)
	if err != nil {
		t.Fatalf("newIVF: %v", err)
	}

	for _, idx := range []VectorIndex{h, ix} {
		// id 1: payload + sparse (exercises meta/sparse copy parity).
		if _, _, err := idx.Insert(1, []float32{1, 2, 3, 4}, 0,
			Metadata{"lang": NewString("en"), "n": NewInt(7)},
			&SparseVector{Indices: []uint32{2, 5}, Values: []float32{0.5, 0.25}},
			nil, CASCond{}); err != nil {
			t.Fatalf("Insert 1: %v", err)
		}
		// id 2: bare dense vector, no payload/sparse/ttl — the zero-alloc target.
		if _, _, err := idx.Insert(2, []float32{9, 8, 7, 6}, 0, nil, nil, nil, CASCond{}); err != nil {
			t.Fatalf("Insert 2: %v", err)
		}
	}
	return map[string]VectorIndex{"hnsw": h, "ivf": ix}
}

// TestVectorGetIntoMatchesGet asserts GetInto returns byte-identical
// vec/meta/ttl/version/ok as Get for both a hit and a miss, on every implementer.
func TestVectorGetIntoMatchesGet(t *testing.T) {
	for name, idx := range getIntoIndexes(t) {
		t.Run(name, func(t *testing.T) {
			for _, id := range []uint64{1, 2, 999 /* miss */} {
				wVec, wMeta, wTTL, wSparse, wVer, wOK := idx.Get(id)
				gVec, gMeta, gTTL, gSparse, gVer, gOK := idx.GetInto(make([]float32, 0, 4), id)

				if gOK != wOK {
					t.Fatalf("id %d: ok = %v, want %v", id, gOK, wOK)
				}
				if !reflect.DeepEqual(gVec, wVec) {
					t.Errorf("id %d: vec = %v, want %v", id, gVec, wVec)
				}
				if !reflect.DeepEqual(gMeta, wMeta) {
					t.Errorf("id %d: meta = %v, want %v", id, gMeta, wMeta)
				}
				if gTTL != wTTL {
					t.Errorf("id %d: ttl = %v, want %v", id, gTTL, wTTL)
				}
				if !reflect.DeepEqual(gSparse, wSparse) {
					t.Errorf("id %d: sparse = %v, want %v", id, gSparse, wSparse)
				}
				if gVer != wVer {
					t.Errorf("id %d: version = %d, want %d", id, gVer, wVer)
				}
			}
		})
	}
}

// TestVectorGetIntoAllocFree confirms that reading a bare dense point (no
// payload/sparse/ttl) through GetInto with a REUSED buffer adds zero allocations
// for the vector copy, while the allocating Get shows at least one (the fresh
// []float32 it returns each call).
func TestVectorGetIntoAllocFree(t *testing.T) {
	for name, idx := range getIntoIndexes(t) {
		t.Run(name, func(t *testing.T) {
			dst := make([]float32, 0, 4) // cap >= dim, reused across runs
			into := testing.AllocsPerRun(100, func() {
				dst, _, _, _, _, _ = idx.GetInto(dst[:0], 2)
			})
			if into != 0 {
				t.Errorf("GetInto with reused buffer = %.1f allocs/op, want 0", into)
			}

			get := testing.AllocsPerRun(100, func() {
				_, _, _, _, _, _ = idx.Get(2)
			})
			if get == 0 {
				t.Errorf("Get = %.1f allocs/op, want > 0 (the per-call []float32 GetInto elides)", get)
			}
		})
	}
}
