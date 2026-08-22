// SPDX-License-Identifier: Apache-2.0

package wire

import (
	"errors"

	"github.com/rostamlabs/rostam/vector"
)

// errMVAddBatchTruncated is returned when vector_mv_add_batch args are shorter
// than the [nameLen:u8][name] header requires (fail-loud, mirroring the other
// truncation guards).
var errMVAddBatchTruncated = errors.New("ops: mv add-batch args truncated")

// EncodeMVAddBatchArgs serializes a version/deadline-preserving BATCH of
// multi-vector documents destined for ONE physical partition. Wire:
//
//	[nameLen:u8][name] || EncodeMVScanResult(recs)
//
// The body reuses the MV scan-result codec verbatim (records carry id, tokens,
// metadata, per-doc CAS version, per-key TTL deadlines, and the optional doc
// sparse), and the u8-length name prefix matches VectorKeyColAt1 so the op routes
// to that partition's shard. handleMVAddBatch applies the whole batch in ONE op
// (one Raft commit), so an offline MV resplit copies a partition's docs in a
// single commit instead of one commit per document.
func EncodeMVAddBatchArgs(collection string, recs []vector.MultiScanRecord) []byte {
	body := EncodeMVScanResult(recs)
	buf := make([]byte, 0, 1+len(collection)+len(body))
	buf = append(buf, byte(len(collection)))
	buf = append(buf, collection...)
	buf = append(buf, body...)
	return buf
}

// DecodeMVAddBatchArgs reads args produced by EncodeMVAddBatchArgs.
func DecodeMVAddBatchArgs(args []byte) (string, []vector.MultiScanRecord, error) {
	if len(args) < 1 {
		return "", nil, errMVAddBatchTruncated
	}
	n := int(args[0])
	if len(args) < 1+n {
		return "", nil, errMVAddBatchTruncated
	}
	name := string(args[1 : 1+n])
	recs, err := DecodeMVScanResult(args[1+n:])
	if err != nil {
		return "", nil, err
	}
	return name, recs, nil
}
