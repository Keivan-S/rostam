// SPDX-License-Identifier: Apache-2.0

package ops

import (
	"testing"
)

// TestHandleVectorGetBatchProjectionSkipsDenseCopy covers finding 022: with
// with_vector=false the batch handler must acquire the collection ONCE and skip the
// per-id dense-vector copy entirely, instead of the old path which re-Acquired and
// deep-copied every point's vector (then discarded it). The assertion is RELATIVE
// (mirroring TestVectorHandlerGetPoolReusesScratch): the projection path is measured
// against a faithful reconstruction of the pre-fix per-id GetPointVersion loop, so
// any shared arg-decode/encode cost cancels and only the eliminated per-id dense
// allocations remain. Revert handleVectorGetBatch to the per-id copy and the gap
// collapses, tripping the guard.
func TestHandleVectorGetBatchProjectionSkipsDenseCopy(t *testing.T) {
	const (
		name = "docs"
		dim  = 128
		N    = 256
	)
	tx := newVectorTxContext(t, name, dim)
	ids := make([]uint64, N)
	for i := range ids {
		id := uint64(i + 1)
		ids[i] = id
		if _, err := handleVectorInsert(tx, EncodeVectorInsertArgs(name, id, distinctVec(id, dim))); err != nil {
			t.Fatalf("handleVectorInsert id=%d: %v", id, err)
		}
	}

	// with_vector=false / with_payload=false: no projection is retained, so the fixed
	// path copies nothing per id.
	args := EncodeVectorGetBatchArgs(name, ids, 0)

	// oldGetBatch reconstructs the pre-fix loop: one GetPointVersion per id, which
	// re-Acquires the collection AND deep-copies the dense vector for every point
	// regardless of the (empty) projection. Everything else matches the handler.
	oldGetBatch := func() []byte {
		gname, gids, flags, err := DecodeVectorGetBatchArgs(args)
		if err != nil {
			t.Fatalf("DecodeVectorGetBatchArgs: %v", err)
		}
		withVec := flags&getFlagWithVector != 0
		withPayload := flags&getFlagWithPayload != 0
		rows := make([]GetBatchRow, 0, len(gids))
		for _, id := range gids {
			vec, meta, ttl, sparse, version, ok, err := tx.vectors.GetPointVersion(gname, id)
			if err != nil {
				t.Fatalf("GetPointVersion id=%d: %v", id, err)
			}
			row := GetBatchRow{ID: id, Found: ok}
			if ok {
				if withVec {
					row.Vec = vec
				}
				if withPayload {
					row.Meta = meta
					row.Sparse = sparse
				}
				row.TTLMs = uint64(ttl.Milliseconds()) //nolint:gosec // TTL >= 0
				row.Version = version
			}
			rows = append(rows, row)
		}
		return EncodeVectorGetBatchResult(rows)
	}

	// Warm, then measure.
	if _, err := handleVectorGetBatch(tx, args); err != nil {
		t.Fatalf("warm handleVectorGetBatch: %v", err)
	}
	projected := testing.AllocsPerRun(50, func() {
		body, err := handleVectorGetBatch(tx, args)
		if err != nil || len(body) == 0 {
			t.Fatalf("handleVectorGetBatch: body=%d err=%v", len(body), err)
		}
	})
	old := testing.AllocsPerRun(50, func() {
		if body := oldGetBatch(); len(body) == 0 {
			t.Fatalf("oldGetBatch: empty body")
		}
	})
	t.Logf("vector_get_batch allocs/op (N=%d, dim=%d, with_vector=false): projected = %.0f, old = %.0f", N, dim, projected, old)

	if projected >= old {
		t.Fatalf("projected allocs/op = %.0f, old = %.0f; want projected strictly fewer (per-id dense copy elided)", projected, old)
	}
	// The old path allocates ~one dense []float32 per id; the fixed path allocates
	// none. The saving must scale with the batch size, not be a constant fudge.
	if old-projected < float64(N)/2 {
		t.Fatalf("expected the projection to save ~one dense alloc per id (N=%d); saved only %.0f", N, old-projected)
	}

	// Correctness: the fixed path still returns a found row per id with its version,
	// but no vector (with_vector=false).
	body, err := handleVectorGetBatch(tx, args)
	if err != nil {
		t.Fatalf("handleVectorGetBatch: %v", err)
	}
	rows, err := DecodeVectorGetBatchResult(body)
	if err != nil {
		t.Fatalf("DecodeVectorGetBatchResult: %v", err)
	}
	if len(rows) != N {
		t.Fatalf("rows = %d, want %d", len(rows), N)
	}
	for i, r := range rows {
		if r.ID != ids[i] || !r.Found || r.Vec != nil || r.Version < 1 {
			t.Fatalf("row %d: %+v (want id=%d found no-vec version>=1)", i, r, ids[i])
		}
	}
}
