// SPDX-License-Identifier: Apache-2.0

package ops

import (
	"reflect"
	"testing"

	"github.com/rostamlabs/rostam/cache"
	"github.com/rostamlabs/rostam/vector"
)

// newVectorTxContext builds a TxContext backed by a real cache + CollectionStore
// with a single dense collection, so the vector op handlers can be driven
// end-to-end (not just their codecs).
func newVectorTxContext(t *testing.T, name string, dim int) *TxContext {
	t.Helper()
	cfg := cache.DefaultConfig()
	cfg.NumShards = 1
	c, err := cache.New(cfg)
	if err != nil {
		t.Fatalf("cache.New: %v", err)
	}
	t.Cleanup(func() { _ = c.Close() })

	store, err := vector.OpenCollectionStore(t.TempDir())
	if err != nil {
		t.Fatalf("OpenCollectionStore: %v", err)
	}
	vcfg := vector.Config{Dim: dim, M: 8, EfConstruction: 50, EfSearch: 16, Seed: 1, Metric: vector.L2}
	if err := store.CreateCollection(name, vcfg); err != nil {
		t.Fatalf("CreateCollection: %v", err)
	}
	return NewTxContextWithVectors(c, store)
}

// distinctVec builds a dim-length vector whose contents are a deterministic
// function of id, so every point's payload is unique and any cross-op leakage
// of the shared pooled scratch shows up as a mismatch.
func distinctVec(id uint64, dim int) []float32 {
	v := make([]float32, dim)
	for i := range v {
		v[i] = float32(id)*1000 + float32(i)
	}
	return v
}

// TestVectorHandlerPoolNoCrossContamination drives handleVectorInsert and
// handleVectorGet through a real TxContext and proves the shared global
// vectorDenseBufPool scratch — reused across consecutive ops on the same
// goroutine — never leaks one op's dense bytes into another's result:
//
//   - hit -> hit: each get returns EXACTLY its own inserted vector even though
//     the previous get recycled the same backing for a different point.
//   - miss -> hit: a get that misses (vec==nil, buffer left intact by the
//     `if vec != nil` guard) is followed by a hit that must still be correct,
//     not stale bytes from before the miss.
//   - no aliasing into collection storage: a final full re-read after all the
//     churn returns every original vector unchanged, so the retained pooled
//     backing never aliased the collection's internal storage.
func TestVectorHandlerPoolNoCrossContamination(t *testing.T) {
	const (
		name = "docs"
		dim  = 8
		K    = 32
	)
	tx := newVectorTxContext(t, name, dim)

	want := make(map[uint64][]float32, K)
	for id := uint64(1); id <= K; id++ {
		v := distinctVec(id, dim)
		want[id] = v
		if _, err := handleVectorInsert(tx, EncodeVectorInsertArgs(name, id, v)); err != nil {
			t.Fatalf("handleVectorInsert id=%d: %v", id, err)
		}
	}

	get := func(id uint64) (bool, []float32) {
		t.Helper()
		body, err := handleVectorGet(tx, EncodeVectorGetArgs(name, id, GetFlagWithVector))
		if err != nil {
			t.Fatalf("handleVectorGet id=%d: %v", id, err)
		}
		found, vec, _, _, _, _, err := DecodeVectorGetResultV(body)
		if err != nil {
			t.Fatalf("DecodeVectorGetResultV id=%d: %v", id, err)
		}
		return found, vec
	}

	// Interleave hits with misses so the recycled scratch is reused across both
	// the present-id (overwrite) and missing-id (guard) paths between hits.
	for id := uint64(1); id <= K; id++ {
		found, vec := get(id)
		if !found {
			t.Fatalf("get id=%d: not found, want hit", id)
		}
		if !reflect.DeepEqual(vec, want[id]) {
			t.Fatalf("get id=%d (after prior op reused the pool) = %v, want %v", id, vec, want[id])
		}

		// A miss for an absent id between two hits: must report not-found and
		// leave the pooled buffer such that the NEXT hit is still correct.
		if found, _ := get(1_000_000 + id); found {
			t.Fatalf("get missing id=%d reported found", 1_000_000+id)
		}
		if id < K {
			nf, nv := get(id + 1)
			if !nf || !reflect.DeepEqual(nv, want[id+1]) {
				t.Fatalf("hit after miss: get id=%d found=%v vec=%v, want %v", id+1, nf, nv, want[id+1])
			}
		}
	}

	// Final full re-read: every original vector must be byte-identical, proving
	// no op's retained pooled backing ever aliased + corrupted collection storage.
	for id := uint64(1); id <= K; id++ {
		found, vec := get(id)
		if !found || !reflect.DeepEqual(vec, want[id]) {
			t.Fatalf("final re-read id=%d found=%v vec=%v, want %v", id, found, vec, want[id])
		}
	}
}

// TestVectorHandlerGetPoolReusesScratch asserts the pooled get path keeps the
// per-call dense-buffer allocation out of the hot path: a steady-state
// handleVectorGet (after the pool is warm) allocates strictly fewer times than
// the equivalent UNPOOLED get, which reintroduces one `make([]float32, dim)` per
// call via GetPointVersion. The comparison is RELATIVE (mirroring the sibling
// TestDecodeVectorInsertArgsIntoAllocFree) rather than a hand-picked absolute
// ceiling: both closures share the same arg-decode and response-encode, so any
// allocator/encode shift cancels and only the elided dense buffer remains as the
// measured difference. This is exactly the regression the comment names — revert
// handleVectorGet to GetPointVersion and the pooled measurement equals the
// unpooled one, tripping the guard.
func TestVectorHandlerGetPoolReusesScratch(t *testing.T) {
	const (
		name = "docs"
		dim  = 128
		id   = uint64(7)
	)
	tx := newVectorTxContext(t, name, dim)
	if _, err := handleVectorInsert(tx, EncodeVectorInsertArgs(name, id, distinctVec(id, dim))); err != nil {
		t.Fatalf("handleVectorInsert: %v", err)
	}
	args := EncodeVectorGetArgs(name, id, GetFlagWithVector)

	// unpooledGet mirrors handleVectorGet exactly but uses the ALLOCATING
	// GetPointVersion (a fresh dim-length []float32 per get) in place of the
	// pooled GetPointVersionInto — i.e. the precise regression the pool guards
	// against. Everything else (arg decode, response encode) is identical, so the
	// only allocation difference between the two paths is the dense buffer.
	unpooledGet := func() []byte {
		gname, gid, flags, err := DecodeVectorGetArgs(args)
		if err != nil {
			t.Fatalf("DecodeVectorGetArgs: %v", err)
		}
		withVec := flags&GetFlagWithVector != 0
		withPayload := flags&GetFlagWithPayload != 0
		vec, meta, ttl, sparse, version, ok, err := tx.vectors.GetPointVersion(gname, gid)
		if err != nil {
			t.Fatalf("GetPointVersion: %v", err)
		}
		return EncodeVectorGetResultV(ok, vec, meta, ttl, sparse, withVec, withPayload, version)
	}

	// Warm the pool, then measure the pooled hot path against the unpooled path.
	if _, err := handleVectorGet(tx, args); err != nil {
		t.Fatalf("warm handleVectorGet: %v", err)
	}
	pooled := testing.AllocsPerRun(500, func() {
		body, err := handleVectorGet(tx, args)
		if err != nil || len(body) == 0 {
			t.Fatalf("handleVectorGet: body=%d err=%v", len(body), err)
		}
	})
	unpooled := testing.AllocsPerRun(500, func() {
		if body := unpooledGet(); len(body) == 0 {
			t.Fatalf("unpooledGet: empty body")
		}
	})
	t.Logf("handleVectorGet allocs/op (dim=%d): pooled = %.1f, unpooled = %.1f", dim, pooled, unpooled)

	if pooled >= unpooled {
		t.Fatalf("pooled get allocs/op = %.1f, unpooled = %.1f; want pooled strictly fewer (dense scratch elided)", pooled, unpooled)
	}
	if unpooled-pooled < 1 {
		t.Fatalf("expected the pool to save at least the dense []float32 alloc; saved %.1f", unpooled-pooled)
	}
}
