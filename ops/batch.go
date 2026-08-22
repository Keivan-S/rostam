// SPDX-License-Identifier: Apache-2.0

package ops

import "github.com/rostamlabs/rostam/ops/wire"

// handlePutBatch applies every put in the batch within one TxContext — one Raft
// log entry / fsync / round-trip / apply for the whole batch.
//
// It decodes the ENTIRE buffer first, so a malformed (truncated) batch applies
// nothing (structural errors are atomic). It then applies each entry
// independently, CONTINUING past a per-entry tx.Put failure and returning the
// first such error. This makes a put_batch state-equivalent to applying the same
// keys as N sequential single puts: a capacity/quota rejection on one entry
// affects only that entry (as it would for a lone put), never the rest of the
// batch — so put_batch cannot amplify a replica-local capacity decision into a
// whole-tail divergence. tx.Put normally only errors on validation/quota.
func handlePutBatch(tx *TxContext, args []byte) ([]byte, error) {
	entries, err := wire.DecodePutBatchArgs(args)
	if err != nil {
		return nil, err
	}
	var firstErr error
	for _, e := range entries {
		if perr := tx.Put(e.Key, e.Val, e.TTL); perr != nil && firstErr == nil {
			firstErr = perr
		}
	}
	if firstErr != nil {
		return nil, firstErr
	}
	return wire.EncodePutBatchResult(len(entries)), nil
}
