// SPDX-License-Identifier: Apache-2.0

package ops

import "github.com/rostamlabs/rostam/ops/wire"

// handleMVAddBatch applies a batch of version/deadline-preserving MV restore-adds
// in one op. Each record is applied via MultiRestoreAddSparse exactly as the
// per-record vector_mv_add_versioned op does, so the result is byte-identical to
// the unbatched offline copy — only the Raft commit count changes (one per batch
// instead of one per document).
func handleMVAddBatch(tx *TxContext, args []byte) ([]byte, error) {
	if tx.vectors == nil {
		return nil, ErrVectorsNotAvailable
	}
	name, recs, err := wire.DecodeMVAddBatchArgs(args)
	if err != nil {
		return nil, err
	}
	// Fast path: an EMPTY target (the offline MV-resplit case — fresh partitions)
	// builds the whole batch's token graph concurrently in one pass. Falls through
	// to the per-record restore-add when the target already has documents.
	built, err := tx.vectors.MultiBulkBuild(name, recs, 0)
	if err != nil {
		return nil, err
	}
	if built {
		return nil, nil
	}
	for _, r := range recs {
		if err := tx.vectors.MultiRestoreAddSparse(name, r.ID, r.Tokens, r.Metadata, r.KeyExpires, r.Version, r.Sparse); err != nil {
			return nil, err
		}
	}
	return nil, nil
}
