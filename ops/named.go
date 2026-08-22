// SPDX-License-Identifier: Apache-2.0

package ops

import (
	"errors"

	"github.com/rostamlabs/rostam/ops/wire"
	"github.com/rostamlabs/rostam/vector"
)

func handleNamedCreate(tx *TxContext, args []byte) ([]byte, error) {
	if tx.vectors == nil {
		return nil, ErrVectorsNotAvailable
	}
	// partitions is a fan-out routing concern resolved by the dispatcher/embedded
	// coordinator (which re-expands into physical single-partition named
	// collections); the single-collection handler always creates ONE plain named
	// collection, so it is decoded but not consulted here.
	col, cfg, _, err := wire.DecodeNamedCreateArgs(args)
	if err != nil {
		return nil, err
	}
	return nil, tx.vectors.CreateNamed(col, cfg)
}

func handleNamedDrop(tx *TxContext, args []byte) ([]byte, error) {
	if tx.vectors == nil {
		return nil, ErrVectorsNotAvailable
	}
	col, err := wire.DecodeNamedNameArgs(args)
	if err != nil {
		return nil, err
	}
	// Idempotent drop, mirroring handleMVDrop: dropping an absent named
	// collection is a no-op (swallow ONLY the ErrNoNamed sentinel) so a fan-out
	// retry after a partial failure won't error on already-dropped partitions.
	if err := tx.vectors.DropNamed(col); err != nil && !errors.Is(err, vector.ErrNoNamed) {
		return nil, err
	}
	return nil, nil
}

func handleNamedInsert(tx *TxContext, args []byte) ([]byte, error) {
	if tx.vectors == nil {
		return nil, ErrVectorsNotAvailable
	}
	col, id, vectors, sparseVectors, payload, ttl, expected, hasExpected, keyTTLMs, err := wire.DecodeNamedInsertArgsSparseKeyTTL(args)
	if err != nil {
		return nil, err
	}
	// Insert validates names/dims/MODALITY and fails loud (ErrUnknownVectorName /
	// ErrDimMismatch / ErrSpaceModalityMismatch); a CAS mismatch returns
	// ErrVersionConflict — all propagate as clean op errors (mapped to
	// InvalidArgument/400 or FailedPrecondition/409 at the edge). sparseVectors is
	// nil for a dense-only insert (the wire is byte-identical). keyTTLMs (relative
	// ms) → the engine computes the absolute deadline now+ttl at insert and the WAL
	// logs it (replay restores verbatim). Empty/nil = no per-key TTL (zero-overhead).
	cas := vector.CASCond{Expected: expected, Has: hasExpected}
	if ms, stamped := tx.applyStamp(); stamped {
		_, err = tx.vectors.NamedInsertSparseCASKeyTTLAt(col, id, vectors, sparseVectors, payload, ttl, keyTTLMs, cas, ms)
	} else {
		_, err = tx.vectors.NamedInsertSparseCASKeyTTL(col, id, vectors, sparseVectors, payload, ttl, keyTTLMs, cas)
	}
	return nil, err
}

func handleNamedSearch(tx *TxContext, args []byte) ([]byte, error) {
	if tx.vectors == nil {
		return nil, ErrVectorsNotAvailable
	}
	col, vecName, query, k, filter, err := wire.DecodeNamedSearchArgs(args)
	if err != nil {
		return nil, err
	}
	// NamedSearch -> NamedCollection.SearchNamed -> hnsw.SearchFilteredWith,
	// which Compile()s the filter and propagates a Compile error (fail-loud);
	// an unknown vector name returns ErrUnknownVectorName.
	res, err := tx.vectors.NamedSearch(col, vecName, query, k, filter)
	if err != nil {
		return nil, err
	}
	return wire.EncodeVectorSearchResults(res), nil
}

// handleNamedSparseSearch runs a sparse-dot-product top-k against a SPARSE named
// space. It decodes the sparse query + filter + rc/opa, calls
// CollectionStore.NamedSearchSparse (which fails loud on an unknown space /
// modality mismatch / bad filter), and encodes the results with wire.EncodeHybridResults
// (id+distance+SCORE) — NOT wire.EncodeVectorSearchResults, which carries only
// distance: the sparse lane ranks by SCORE (the dot product), so the per-partition
// score MUST ride the wire for the coordinator's score-descending fan-out merge to
// be correct. rc is consulted by wire.ReadConsistencyOf (it arms the shard data barrier
// for a Linearizable sparse read).
func handleNamedSparseSearch(tx *TxContext, args []byte) ([]byte, error) {
	if tx.vectors == nil {
		return nil, ErrVectorsNotAvailable
	}
	col, space, query, k, filter, _, _, _, err := wire.DecodeNamedSparseSearchArgsOpts(args)
	if err != nil {
		return nil, err
	}
	res, err := tx.vectors.NamedSearchSparse(col, space, &query, k, filter)
	if err != nil {
		return nil, err
	}
	return wire.EncodeHybridResults(res), nil
}

// handleNamedHybridSearch fuses a DENSE named space and a SPARSE named space into
// the top-k (single-partition path). It decodes both space names, the dense + sparse
// queries, the fusion opts, the filter and rc/opa, calls
// CollectionStore.NamedHybrid (which fails loud on an unknown space / modality
// mismatch / bad filter, and degrades single-lane internally), and encodes the
// FUSED results with wire.EncodeHybridResults (id+distance+SCORE) — the fused score must
// ride the wire. rc is consulted by wire.ReadConsistencyOf (it arms the shard data
// barrier for a Linearizable named hybrid).
func handleNamedHybridSearch(tx *TxContext, args []byte) ([]byte, error) {
	if tx.vectors == nil {
		return nil, ErrVectorsNotAvailable
	}
	col, denseSpace, denseQ, sparseSpace, sparseQ, k, opts, _, _, _, err := wire.DecodeNamedHybridArgs(args)
	if err != nil {
		return nil, err
	}
	var sq *vector.SparseVector
	if !sparseQ.IsZero() {
		sq = &sparseQ
	}
	res, err := tx.vectors.NamedHybrid(col, denseSpace, denseQ, sparseSpace, sq, k, opts)
	if err != nil {
		return nil, err
	}
	return wire.EncodeHybridResults(res), nil
}

// handleNamedHybridLanes is the partition-exact fan-out LEAF for named hybrid: it
// returns the two UNFUSED candidate lanes (dense + sparse) for ITS partition so the
// coordinator (namedHybridFanOut) can union the per-partition lanes and fuse ONCE
// globally — reproducing the single-partition oracle. It shares the
// wire.EncodeNamedHybridArgs wire with vector_named_hybrid_search (the lanes op carries
// the same dense/sparse queries + opts + filter + rc); only the RESULT differs (two
// lanes via wire.EncodeHybridLanesResult instead of one fused list). rc arms the shard
// barrier via wire.ReadConsistencyOf.
func handleNamedHybridLanes(tx *TxContext, args []byte) ([]byte, error) {
	if tx.vectors == nil {
		return nil, ErrVectorsNotAvailable
	}
	col, denseSpace, denseQ, sparseSpace, sparseQ, k, opts, _, _, _, err := wire.DecodeNamedHybridArgs(args)
	if err != nil {
		return nil, err
	}
	var sq *vector.SparseVector
	if !sparseQ.IsZero() {
		sq = &sparseQ
	}
	dense, sparse, err := tx.vectors.NamedHybridLanes(col, denseSpace, denseQ, sparseSpace, sq, k, opts)
	if err != nil {
		return nil, err
	}
	return wire.EncodeHybridLanesResult(dense, sparse), nil
}

func handleNamedSearchDocs(tx *TxContext, args []byte) ([]byte, error) {
	if tx.vectors == nil {
		return nil, ErrVectorsNotAvailable
	}
	col, vecName, query, k, filter, err := wire.DecodeNamedSearchArgs(args)
	if err != nil {
		return nil, err
	}
	docs, err := tx.vectors.NamedSearchDocs(col, vecName, query, k, filter)
	if err != nil {
		return nil, err
	}
	return wire.EncodeVectorDocs(docs), nil
}

func handleNamedDelete(tx *TxContext, args []byte) ([]byte, error) {
	if tx.vectors == nil {
		return nil, ErrVectorsNotAvailable
	}
	col, id, expected, hasExpected, err := wire.DecodeNamedDeleteArgsCAS(args)
	if err != nil {
		return nil, err
	}
	ok, _, err := tx.vectors.NamedDeleteCAS(col, id, vector.CASCond{Expected: expected, Has: hasExpected})
	if err != nil {
		return nil, err // incl. ErrVersionConflict (mapped to FailedPrecondition/409)
	}
	if ok {
		return []byte{1}, nil
	}
	return []byte{0}, nil
}

// handleNamedGet retrieves a named-vector point by id: its per-space vectors +
// shared payload + remaining TTL, gated by the with_vector/with_payload flags. A
// missing point returns the found=0 FLAG (NEVER an op error). Read-only.
func handleNamedGet(tx *TxContext, args []byte) ([]byte, error) {
	if tx.vectors == nil {
		return nil, ErrVectorsNotAvailable
	}
	col, id, flags, err := wire.DecodeVectorGetArgs(args)
	if err != nil {
		return nil, err
	}
	withVec := flags&wire.GetFlagWithVector != 0
	withPayload := flags&wire.GetFlagWithPayload != 0
	vectors, payload, ttl, version, ok, err := tx.vectors.NamedGetVersion(col, id)
	if err != nil {
		return nil, err
	}
	return wire.EncodeNamedGetResultV(ok, vectors, payload, ttl, withVec, withPayload, version), nil
}

// handleNamedGetBatch retrieves a subset of named-vector points by id in one op:
// for each requested id it runs the same NamedGet lookup as handleNamedGet and
// emits a row. A missing id is a Found=false row (NEVER an op error) so the
// coordinator can derive the global missing set from absent ids. Rows preserve
// the input id order (this is the per-partition handler — it returns rows for ITS
// id-subset in the order given). The with_vector/with_payload flags gate the
// vectors-map and the payload projections, applied here at fetch time exactly as
// single named get. Read-only. Mirrors handleVectorGetBatch.
func handleNamedGetBatch(tx *TxContext, args []byte) ([]byte, error) {
	if tx.vectors == nil {
		return nil, ErrVectorsNotAvailable
	}
	col, ids, flags, err := wire.DecodeVectorGetBatchArgs(args)
	if err != nil {
		return nil, err
	}
	withVec := flags&wire.GetFlagWithVector != 0
	withPayload := flags&wire.GetFlagWithPayload != 0
	rows := make([]wire.NamedGetBatchRow, 0, len(ids))
	for _, id := range ids {
		vectors, payload, ttl, version, ok, err := tx.vectors.NamedGetVersion(col, id)
		if err != nil {
			return nil, err
		}
		row := wire.NamedGetBatchRow{ID: id, Found: ok}
		if ok {
			if withVec {
				row.Vectors = vectors
			}
			if withPayload {
				row.Meta = payload
			}
			row.TTLMs = uint64(ttl.Milliseconds()) //nolint:gosec // TTL >= 0
			row.Version = version
		}
		rows = append(rows, row)
	}
	return wire.EncodeNamedGetBatchResult(rows), nil
}

// handleNamedSetPayload merges the provided payload into id's shared payload (no
// reindex, no WAL). applied=0 for a missing point (not-found flag); bad JSON is a
// hard error. Read-write.
func handleNamedSetPayload(tx *TxContext, args []byte) ([]byte, error) {
	if tx.vectors == nil {
		return nil, ErrVectorsNotAvailable
	}
	col, id, meta, keyTTLMs, expected, hasExpected, err := wire.DecodeSetPayloadArgsCAS(args)
	if err != nil {
		return nil, err
	}
	cas := vector.CASCond{Expected: expected, Has: hasExpected}
	var applied bool
	if ms, stamped := tx.applyStamp(); stamped {
		applied, _, err = tx.vectors.NamedSetPayloadCASAt(col, id, meta, keyTTLMs, cas, ms)
	} else {
		applied, _, err = tx.vectors.NamedSetPayloadCAS(col, id, meta, keyTTLMs, cas)
	}
	if err != nil {
		return nil, err // incl. ErrVersionConflict
	}
	return wire.EncodePayloadResult(applied), nil
}

// handleNamedOverwritePayload replaces id's entire shared payload. applied=0 for a
// missing point; bad JSON is a hard error. Read-write.
func handleNamedOverwritePayload(tx *TxContext, args []byte) ([]byte, error) {
	if tx.vectors == nil {
		return nil, ErrVectorsNotAvailable
	}
	col, id, meta, keyTTLMs, expected, hasExpected, err := wire.DecodeSetPayloadArgsCAS(args)
	if err != nil {
		return nil, err
	}
	cas := vector.CASCond{Expected: expected, Has: hasExpected}
	var applied bool
	if ms, stamped := tx.applyStamp(); stamped {
		applied, _, err = tx.vectors.NamedOverwritePayloadCASAt(col, id, meta, keyTTLMs, cas, ms)
	} else {
		applied, _, err = tx.vectors.NamedOverwritePayloadCAS(col, id, meta, keyTTLMs, cas)
	}
	if err != nil {
		return nil, err
	}
	return wire.EncodePayloadResult(applied), nil
}

// handleNamedDeletePayloadKeys removes the listed keys from id's shared payload.
// applied=0 for a missing point. Read-write.
func handleNamedDeletePayloadKeys(tx *TxContext, args []byte) ([]byte, error) {
	if tx.vectors == nil {
		return nil, ErrVectorsNotAvailable
	}
	col, id, keys, expected, hasExpected, err := wire.DecodeDeletePayloadKeysArgsCAS(args)
	if err != nil {
		return nil, err
	}
	cas := vector.CASCond{Expected: expected, Has: hasExpected}
	var applied bool
	if ms, stamped := tx.applyStamp(); stamped {
		applied, _, err = tx.vectors.NamedDeletePayloadKeysCASAt(col, id, keys, cas, ms)
	} else {
		applied, _, err = tx.vectors.NamedDeletePayloadKeysCAS(col, id, keys, cas)
	}
	if err != nil {
		return nil, err
	}
	return wire.EncodePayloadResult(applied), nil
}

// handleNamedClearPayload clears id's shared payload. applied=0 for a missing
// point. Read-write.
func handleNamedClearPayload(tx *TxContext, args []byte) ([]byte, error) {
	if tx.vectors == nil {
		return nil, ErrVectorsNotAvailable
	}
	col, id, expected, hasExpected, err := wire.DecodeClearPayloadArgsCAS(args)
	if err != nil {
		return nil, err
	}
	cas := vector.CASCond{Expected: expected, Has: hasExpected}
	var applied bool
	if ms, stamped := tx.applyStamp(); stamped {
		applied, _, err = tx.vectors.NamedClearPayloadCASAt(col, id, cas, ms)
	} else {
		applied, _, err = tx.vectors.NamedClearPayloadCAS(col, id, cas)
	}
	if err != nil {
		return nil, err
	}
	return wire.EncodePayloadResult(applied), nil
}

func handleNamedScroll(tx *TxContext, args []byte) ([]byte, error) {
	if tx.vectors == nil {
		return nil, ErrVectorsNotAvailable
	}
	col, filter, limit, afterID, hasAfter, _, _, order, err := wire.DecodeNamedScrollArgsOrder(args)
	if err != nil {
		return nil, err
	}
	// Cursor-aware page (the named partition fan-out passes the SAME global afterID
	// to every partition; the coordinator derives next_cursor from the merged ids).
	// NamedScrollPage{,Order} Compile()s the filter and propagates a Compile error.
	if order != nil {
		ob := wire.ScrollOrderToOrderBy(order)
		var afterKey float64
		if order.HasResume {
			afterKey = order.ResumeKey
		}
		docs, _, _, derr := tx.vectors.NamedScrollPageOrder(col, filter, ob, afterID, afterKey, hasAfter, limit)
		if derr != nil {
			return nil, derr
		}
		return wire.EncodeVectorDocs(docs), nil
	}
	docs, _, _, err := tx.vectors.NamedScrollPage(col, filter, afterID, hasAfter, limit)
	if err != nil {
		return nil, err
	}
	return wire.EncodeVectorDocs(docs), nil
}

func handleNamedGetConfig(tx *TxContext, args []byte) ([]byte, error) {
	if tx.vectors == nil {
		return nil, ErrVectorsNotAvailable
	}
	col, err := wire.DecodeNamedNameArgs(args)
	if err != nil {
		return nil, err
	}
	cfg, err := tx.vectors.NamedConfig(col)
	if err != nil {
		return nil, err
	}
	return wire.EncodeNamedConfigResult(cfg), nil
}
