// SPDX-License-Identifier: Apache-2.0

package ops

import (
	"errors"
	"fmt"

	"github.com/rostamlabs/rostam/sdk/wire"
	"github.com/rostamlabs/rostam/vector"
)

func handleMVCreate(tx *TxContext, args []byte) ([]byte, error) {
	if tx.vectors == nil {
		return nil, ErrVectorsNotAvailable
	}
	name, cfg, err := wire.DecodeMVCreateArgs(args)
	if err != nil {
		return nil, err
	}
	return nil, tx.vectors.CreateMultiVector(name, cfg)
}

func handleMVDrop(tx *TxContext, args []byte) ([]byte, error) {
	if tx.vectors == nil {
		return nil, ErrVectorsNotAvailable
	}
	name, _, err := wire.DecodeMVDeleteArgs(args)
	if err != nil {
		return nil, err
	}
	// Idempotent drop, mirroring dense vector_drop_collection: dropping an
	// absent collection is a no-op. Dense achieves this in the engine
	// (DropCollection returns nil on a missing name); MV's DropMultiVector
	// instead reports vector.ErrNoMultiVector, so we swallow ONLY that specific
	// case here at the handler. This keeps mvDropCollectionFanout retry-safe: a
	// retry after a partial failure won't error on already-dropped partitions.
	if err := tx.vectors.DropMultiVector(name); err != nil && !errors.Is(err, vector.ErrNoMultiVector) {
		return nil, err
	}
	return nil, nil
}

func handleMVAdd(tx *TxContext, args []byte) ([]byte, error) {
	if tx.vectors == nil {
		return nil, ErrVectorsNotAvailable
	}
	name, docID, tokens, meta, expected, hasExpected, keyTTLMs, sparse, err := wire.DecodeMVAddArgsCASKeyTTLSparse(args)
	if err != nil {
		return nil, err
	}
	// A CAS mismatch returns ErrVersionConflict — propagates as a clean op error
	// (mapped to FailedPrecondition/409 at the edge). keyTTLMs (relative ms) → the
	// engine computes the absolute deadline now+ttl at add and the WAL logs it
	// (replay restores verbatim). Empty/nil = no per-key TTL (zero-overhead). The
	// optional doc-level sparse (nil when absent ⇒ dense-only) is stored + indexed.
	cas := vector.CASCond{Expected: expected, Has: hasExpected}
	if ms, stamped := tx.applyStamp(); stamped {
		_, err = tx.vectors.MultiAddCASKeyTTLSparseAt(name, docID, tokens, meta, keyTTLMs, sparse, cas, ms)
	} else {
		_, err = tx.vectors.MultiAddCASKeyTTLSparse(name, docID, tokens, meta, keyTTLMs, sparse, cas)
	}
	return nil, err
}

// handleMVAddIfAbsent runs the atomic MV add-if-absent engine op (reuses the
// add-args wire shape). OpReadWrite: Raft serialization closes Race A for the MV
// path. Result: [inserted:u8].
func handleMVAddIfAbsent(tx *TxContext, args []byte) ([]byte, error) {
	if tx.vectors == nil {
		return nil, ErrVectorsNotAvailable
	}
	// Version-aware decode (old wire.EncodeMVAddArgs wire ⇒ version 0). version!=0 →
	// version-PRESERVING if-absent (the online MV reshard copy pass): carry the
	// copied document's exact per-document CAS version instead of resetting to 1,
	// while still never clobbering a concurrent live dual-write (Race A). version==0
	// is the plain if-absent (byte-identical to the legacy path).
	name, docID, tokens, meta, version, keyExpires, sparse, err := wire.DecodeMVAddArgsVersionedKeyExpiresSparse(args)
	if err != nil {
		return nil, err
	}
	// keyExpires is the copied doc's ABSOLUTE per-key payload deadline map (from the
	// scan trailer), set VERBATIM on a real add (NOT recomputed now+ttl) so resharded
	// per-key TTLs survive the ONLINE copy time-stable; nil otherwise. The optional
	// doc sparse rides verbatim too (nil when absent ⇒ dense-only).
	inserted, err := tx.vectors.MultiAddIfAbsentVersionSparse(name, docID, tokens, meta, keyExpires, version, sparse)
	if err != nil {
		return nil, err
	}
	return wire.EncodeIfAbsentResult(inserted), nil
}

// handleMVAddVersioned runs a verbatim-version MV replace-add (the OFFLINE MV
// resplit backfill primitive): it sets the document's per-document CAS version
// EXACTLY to the wire version (not bumped), via MultiRestoreAdd → restoreAdd.
// version==0 is the normal bump (a fresh add → 1). OpReadWrite. Mirrors the dense
// version-preserving reinsert (wire.EncodeVectorInsertArgsVersioned). Result: nil.
func handleMVAddVersioned(tx *TxContext, args []byte) ([]byte, error) {
	if tx.vectors == nil {
		return nil, ErrVectorsNotAvailable
	}
	name, docID, tokens, meta, version, keyExpires, sparse, err := wire.DecodeMVAddArgsVersionedKeyExpiresSparse(args)
	if err != nil {
		return nil, err
	}
	// keyExpires is the copied doc's ABSOLUTE per-key payload deadline map (from the
	// scan trailer), applied VERBATIM by MultiRestoreAdd → restoreAdd (NOT recomputed)
	// so resharded per-key TTLs survive the OFFLINE copy time-stable; nil otherwise.
	// The optional doc sparse rides verbatim too (nil when absent ⇒ dense-only).
	return nil, tx.vectors.MultiRestoreAddSparse(name, docID, tokens, meta, keyExpires, version, sparse)
}

// handleMVExists is the cheap MV liveness probe (OpReadOnly) for the
// resurrection guard (Race B). Result: [exists:u8].
func handleMVExists(tx *TxContext, args []byte) ([]byte, error) {
	if tx.vectors == nil {
		return nil, ErrVectorsNotAvailable
	}
	name, docID, err := wire.DecodeMVExistsArgs(args)
	if err != nil {
		return nil, err
	}
	exists, err := tx.vectors.MultiExists(name, docID)
	if err != nil {
		return nil, err
	}
	return wire.EncodeExistsResult(exists), nil
}

func handleMVSearch(tx *TxContext, args []byte) ([]byte, error) {
	if tx.vectors == nil {
		return nil, ErrVectorsNotAvailable
	}
	// Opts-aware decode so the payload filter (carried in the self-delimiting
	// trailer) reaches the engine. rc/opa are cross-shard routing knobs the
	// single-node engine ignores (the fan-out coordinator consumes them before
	// this handler runs), so they are decoded-and-dropped here.
	name, query, k, candPerToken, _, _, filter, _, err := wire.DecodeMVSearchArgsOptsFilter(args)
	if err != nil {
		return nil, err
	}
	res, err := tx.vectors.MultiSearch(name, query, k, vector.MultiSearchOpts{
		CandidatesPerToken: candPerToken,
		Filter:             filter,
	})
	if err != nil {
		return nil, err
	}
	return wire.EncodeMVResults(res), nil
}

// handleMVHybridSearch fuses an MV collection's MaxSim lane and its doc-level sparse
// lane into the top-k. It decodes the shared MV-hybrid wire, calls
// CollectionStore.MultiHybrid (which collapses the single-lane degradation and fuses
// via FuseScoreLanes — both lanes score-desc), and encodes the FUSED results with
// wire.EncodeHybridResults (id + distance + SCORE — the fused score must survive the
// shard barrier for a Linearizable MV hybrid). rc/opa are decoded-and-dropped (the
// fan-out coordinator consumes them before this handler runs).
func handleMVHybridSearch(tx *TxContext, args []byte) ([]byte, error) {
	if tx.vectors == nil {
		return nil, ErrVectorsNotAvailable
	}
	name, query, sparseQ, k, opts, _, _, _, err := wire.DecodeMVHybridArgs(args)
	if err != nil {
		return nil, err
	}
	var sq *vector.SparseVector
	if !sparseQ.IsZero() {
		sq = &sparseQ
	}
	res, err := tx.vectors.MultiHybrid(name, query, sq, k, opts)
	if err != nil {
		return nil, err
	}
	return wire.EncodeHybridResults(res), nil
}

// handleMVHybridLanes is the partition-exact fan-out LEAF for the MV hybrid: it
// returns the UNFUSED MaxSim + sparse lanes so the cross-partition coordinator
// (mvHybridFanOut) can union the per-partition lanes and fuse ONCE globally (exact
// fan-out). It shares the wire.EncodeMVHybridArgs wire with vector_mv_hybrid_search; the
// lanes op returns the two pre-fusion lanes via wire.EncodeHybridLanesResult instead of
// one fused list. rc arms the shard barrier (consumed by the fan-out before this
// runs). BOTH returned lanes are descending by Score (MaxSim is a desc score, not a
// distance).
func handleMVHybridLanes(tx *TxContext, args []byte) ([]byte, error) {
	if tx.vectors == nil {
		return nil, ErrVectorsNotAvailable
	}
	name, query, sparseQ, k, opts, _, _, _, err := wire.DecodeMVHybridArgs(args)
	if err != nil {
		return nil, err
	}
	var sq *vector.SparseVector
	if !sparseQ.IsZero() {
		sq = &sparseQ
	}
	dense, sparse, err := tx.vectors.MultiHybridLanes(name, query, sq, k, opts)
	if err != nil {
		return nil, err
	}
	return wire.EncodeHybridLanesResult(dense, sparse), nil
}

// handleMVGetConfig returns a multi-vector collection's config (so an offline
// resplit can re-create new-generation partitions with the same configuration),
// doubling as a read-only existence probe. On a missing collection it returns
// the SAME "unknown collection" error the dense handleVectorGetConfig returns,
// so callers treat error==absent uniformly across both families. The body is the
// wire.EncodeMVCreateArgs wire form (decodable via wire.DecodeMVCreateArgs); callers that
// only need existence simply discard the body and check err==nil.
func handleMVGetConfig(tx *TxContext, args []byte) ([]byte, error) {
	if tx.vectors == nil {
		return nil, ErrVectorsNotAvailable
	}
	name, err := wire.DecodeMVGetConfigArgs(args)
	if err != nil {
		return nil, err
	}
	idx, ok := tx.vectors.GetMultiVector(name)
	if !ok {
		return nil, fmt.Errorf("ops: unknown collection %q", name)
	}
	return wire.EncodeMVCreateArgs(name, idx.Config()), nil
}

// handleMVScanVectors enumerates every live document of a (physical partition)
// multi-vector collection — the read primitive an offline MV resplit uses to
// re-insert each document into a re-hashed generation. Read-only.
func handleMVScanVectors(tx *TxContext, args []byte) ([]byte, error) {
	if tx.vectors == nil {
		return nil, ErrVectorsNotAvailable
	}
	name, err := wire.DecodeMVScanArgs(args)
	if err != nil {
		return nil, err
	}
	recs, err := tx.vectors.MultiScanDocuments(name)
	if err != nil {
		return nil, err
	}
	return wire.EncodeMVScanResult(recs), nil
}

// handleMVScroll pages a multi-vector collection's live documents (id + payload),
// id-ASCENDING, resuming strictly after the cursor's afterID when present, up to
// limit, applying the payload filter. Returns the shared scroll doc wire
// (wire.EncodeVectorDocs — id + payload, no score, no token vectors; vector_mv_get is
// the path for tokens). Mirrors handleNamedScroll. rc/opa are cross-shard routing
// knobs the single-node engine ignores (the fan-out coordinator consumes them
// before this handler runs), so they are decoded-and-dropped here. Read-only.
func handleMVScroll(tx *TxContext, args []byte) ([]byte, error) {
	if tx.vectors == nil {
		return nil, ErrVectorsNotAvailable
	}
	col, filter, limit, _, _, afterID, hasAfter, order, err := wire.DecodeMVScrollArgsOrder(args)
	if err != nil {
		return nil, err
	}
	// MVScrollPage{,Order} Compile()s the filter and propagates a Compile error.
	// The fan-out coordinator passes the SAME global afterID to every partition and
	// derives next_cursor from the merged ids.
	if order != nil {
		ob := wire.ScrollOrderToOrderBy(order)
		var afterKey float64
		if order.HasResume {
			afterKey = order.ResumeKey
		}
		docs, _, _, derr := tx.vectors.MVScrollPageOrder(col, filter, ob, afterID, afterKey, hasAfter, limit)
		if derr != nil {
			return nil, derr
		}
		return wire.EncodeVectorDocs(docs), nil
	}
	docs, _, _, err := tx.vectors.MVScrollPage(col, filter, afterID, hasAfter, limit)
	if err != nil {
		return nil, err
	}
	return wire.EncodeVectorDocs(docs), nil
}

func handleMVDelete(tx *TxContext, args []byte) ([]byte, error) {
	if tx.vectors == nil {
		return nil, ErrVectorsNotAvailable
	}
	name, docID, expected, hasExpected, err := wire.DecodeMVDeleteArgsCAS(args)
	if err != nil {
		return nil, err
	}
	ok, _, err := tx.vectors.MultiDeleteCAS(name, docID, vector.CASCond{Expected: expected, Has: hasExpected})
	if err != nil {
		return nil, err // incl. ErrVersionConflict
	}
	if ok {
		return []byte{1}, nil
	}
	return []byte{0}, nil
}

// handleMVGet retrieves a multi-vector document by id: its token matrix + payload,
// gated by the with_vector/with_payload flags. A missing document returns the
// found=0 FLAG (NEVER an op error). Read-only.
func handleMVGet(tx *TxContext, args []byte) ([]byte, error) {
	if tx.vectors == nil {
		return nil, ErrVectorsNotAvailable
	}
	name, docID, flags, err := wire.DecodeVectorGetArgs(args)
	if err != nil {
		return nil, err
	}
	withVec := flags&wire.GetFlagWithVector != 0
	withPayload := flags&wire.GetFlagWithPayload != 0
	tokens, payload, version, ok, err := tx.vectors.MVGetVersion(name, docID)
	if err != nil {
		return nil, err
	}
	return wire.EncodeMVGetResultV(ok, tokens, payload, withVec, withPayload, version), nil
}

// handleMVGetBatch retrieves a subset of multi-vector documents by id in one op:
// for each requested id it runs the same MVGet lookup as handleMVGet and emits a
// row. A missing id is a Found=false row (NEVER an op error) so the coordinator
// can derive the global missing set from absent ids. Rows preserve the input id
// order (this is the per-partition handler — it returns rows for ITS id-subset in
// the order given). The with_vector/with_payload flags gate the token-matrix and
// the payload projections, applied here at fetch time exactly as single mv get.
// Read-only. Mirrors handleNamedGetBatch (no ttl).
func handleMVGetBatch(tx *TxContext, args []byte) ([]byte, error) {
	if tx.vectors == nil {
		return nil, ErrVectorsNotAvailable
	}
	col, ids, flags, err := wire.DecodeVectorGetBatchArgs(args)
	if err != nil {
		return nil, err
	}
	withVec := flags&wire.GetFlagWithVector != 0
	withPayload := flags&wire.GetFlagWithPayload != 0
	rows := make([]wire.MVGetBatchRow, 0, len(ids))
	for _, id := range ids {
		tokens, payload, version, ok, err := tx.vectors.MVGetVersion(col, id)
		if err != nil {
			return nil, err
		}
		row := wire.MVGetBatchRow{ID: id, Found: ok}
		if ok {
			if withVec {
				row.Tokens = tokens
			}
			if withPayload {
				row.Meta = payload
			}
			row.Version = version
		}
		rows = append(rows, row)
	}
	return wire.EncodeMVGetBatchResult(rows), nil
}

// handleMVSetPayload merges the provided payload into docID's payload (no reindex,
// no WAL). applied=0 for a missing document (not-found flag); bad JSON is a hard
// error. Read-write.
func handleMVSetPayload(tx *TxContext, args []byte) ([]byte, error) {
	if tx.vectors == nil {
		return nil, ErrVectorsNotAvailable
	}
	name, docID, meta, keyTTLMs, expected, hasExpected, err := wire.DecodeSetPayloadArgsCAS(args)
	if err != nil {
		return nil, err
	}
	cas := vector.CASCond{Expected: expected, Has: hasExpected}
	var applied bool
	if ms, stamped := tx.applyStamp(); stamped {
		applied, _, err = tx.vectors.MVSetPayloadCASAt(name, docID, meta, keyTTLMs, cas, ms)
	} else {
		applied, _, err = tx.vectors.MVSetPayloadCAS(name, docID, meta, keyTTLMs, cas)
	}
	if err != nil {
		return nil, err // incl. ErrVersionConflict
	}
	return wire.EncodePayloadResult(applied), nil
}

// handleMVOverwritePayload replaces docID's entire payload. applied=0 for a missing
// document; bad JSON is a hard error. Read-write.
func handleMVOverwritePayload(tx *TxContext, args []byte) ([]byte, error) {
	if tx.vectors == nil {
		return nil, ErrVectorsNotAvailable
	}
	name, docID, meta, keyTTLMs, expected, hasExpected, err := wire.DecodeSetPayloadArgsCAS(args)
	if err != nil {
		return nil, err
	}
	cas := vector.CASCond{Expected: expected, Has: hasExpected}
	var applied bool
	if ms, stamped := tx.applyStamp(); stamped {
		applied, _, err = tx.vectors.MVOverwritePayloadCASAt(name, docID, meta, keyTTLMs, cas, ms)
	} else {
		applied, _, err = tx.vectors.MVOverwritePayloadCAS(name, docID, meta, keyTTLMs, cas)
	}
	if err != nil {
		return nil, err
	}
	return wire.EncodePayloadResult(applied), nil
}

// handleMVDeletePayloadKeys removes the listed keys from docID's payload.
// applied=0 for a missing document. Read-write.
func handleMVDeletePayloadKeys(tx *TxContext, args []byte) ([]byte, error) {
	if tx.vectors == nil {
		return nil, ErrVectorsNotAvailable
	}
	name, docID, keys, expected, hasExpected, err := wire.DecodeDeletePayloadKeysArgsCAS(args)
	if err != nil {
		return nil, err
	}
	applied, _, err := tx.vectors.MVDeletePayloadKeysCAS(name, docID, keys, vector.CASCond{Expected: expected, Has: hasExpected})
	if err != nil {
		return nil, err
	}
	return wire.EncodePayloadResult(applied), nil
}

// handleMVClearPayload clears docID's payload. applied=0 for a missing document.
// Read-write.
func handleMVClearPayload(tx *TxContext, args []byte) ([]byte, error) {
	if tx.vectors == nil {
		return nil, ErrVectorsNotAvailable
	}
	name, docID, expected, hasExpected, err := wire.DecodeClearPayloadArgsCAS(args)
	if err != nil {
		return nil, err
	}
	applied, _, err := tx.vectors.MVClearPayloadCAS(name, docID, vector.CASCond{Expected: expected, Has: hasExpected})
	if err != nil {
		return nil, err
	}
	return wire.EncodePayloadResult(applied), nil
}
