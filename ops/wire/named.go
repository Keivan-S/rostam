// SPDX-License-Identifier: Apache-2.0

package wire

import (
	"encoding/binary"
	"encoding/json"
	"fmt"
	"math"
	"slices"
	"time"

	"github.com/rostamlabs/rostam/vtypes"
)

// EncodeNamedCreateArgs serializes a vector_named_create_collection request.
// Wire: [colLen:u8][col][cfgLen:u32][cfgJSON][partitions:u32] where cfgJSON is
// the map[string]vector.NamedVectorParams config (the named spaces) and
// partitions is the collection-level partition count (0 or 1 = single-partition;
// >1 drives cross-shard fan-out, like dense/MV's cfg.Partitions). The trailing
// partitions word is APPENDED (older 0-partition payloads end after cfgJSON), so
// DecodeNamedCreateArgs tolerates its absence (defaults to 0) — backward
// compatible with any pre-fan-out encoder.
func EncodeNamedCreateArgs(col string, cfg map[string]vtypes.NamedVectorParams, partitions int) []byte {
	cfgJSON, _ := json.Marshal(cfg)
	buf := make([]byte, 0, 1+len(col)+4+len(cfgJSON)+4)
	buf = append(buf, byte(len(col)))
	buf = append(buf, col...)
	var u32 [4]byte
	binary.BigEndian.PutUint32(u32[:], uint32(len(cfgJSON))) //nolint:gosec
	buf = append(buf, u32[:]...)
	buf = append(buf, cfgJSON...)
	binary.BigEndian.PutUint32(u32[:], uint32(partitions)) //nolint:gosec
	buf = append(buf, u32[:]...)
	return buf
}

// DecodeNamedCreateArgs reads args produced by EncodeNamedCreateArgs. The
// trailing partitions word is optional: a payload that ends right after cfgJSON
// (the legacy encoding) decodes with partitions=0 (single-partition).
func DecodeNamedCreateArgs(args []byte) (string, map[string]vtypes.NamedVectorParams, int, error) {
	if len(args) < 1 {
		return "", nil, 0, ErrVectorArgsTruncated
	}
	colLen := int(args[0])
	if len(args) < 1+colLen+4 {
		return "", nil, 0, ErrVectorArgsTruncated
	}
	col := string(args[1 : 1+colLen])
	off := 1 + colLen
	cfgLen := int(binary.BigEndian.Uint32(args[off:]))
	off += 4
	if len(args) < off+cfgLen {
		return "", nil, 0, ErrVectorArgsTruncated
	}
	var cfg map[string]vtypes.NamedVectorParams
	if cfgLen > 0 {
		if err := json.Unmarshal(args[off:off+cfgLen], &cfg); err != nil {
			return "", nil, 0, fmt.Errorf("ops: decode named config: %w", err)
		}
	}
	off += cfgLen
	partitions := 0
	if len(args) >= off+4 {
		partitions = int(binary.BigEndian.Uint32(args[off:])) //nolint:gosec
	}
	return col, cfg, partitions, nil
}

// EncodeNamedInsertArgs serializes a vector_named_insert (upsert) request. Wire:
//
//	[colLen:u8][col][id:u64][numVecs:u32]
//	  per vec: [nameLen:u16][name][dim:u32][dim*f32]
//	[ttlMs:u64][metaLen:u32][metaJSON]
//
// The named vectors are the per-point map of named spaces; the shared payload +
// ttl are point-level. ttlMs is the TTL in milliseconds (0 = no expiry).
func EncodeNamedInsertArgs(col string, id uint64, vectors map[string][]float32, payload vtypes.Metadata, ttl time.Duration) []byte {
	var metaJSON []byte
	if len(payload) > 0 {
		metaJSON, _ = json.Marshal(payload)
	}
	buf := make([]byte, 0, 1+len(col)+8+4+len(metaJSON)+16)
	buf = append(buf, byte(len(col)))
	buf = append(buf, col...)
	var u64 [8]byte
	binary.BigEndian.PutUint64(u64[:], id)
	buf = append(buf, u64[:]...)
	var u32 [4]byte
	binary.BigEndian.PutUint32(u32[:], uint32(len(vectors))) //nolint:gosec
	buf = append(buf, u32[:]...)
	for name, vec := range vectors {
		var u16 [2]byte
		binary.BigEndian.PutUint16(u16[:], uint16(len(name))) //nolint:gosec
		buf = append(buf, u16[:]...)
		buf = append(buf, name...)
		binary.BigEndian.PutUint32(u32[:], uint32(len(vec))) //nolint:gosec
		buf = append(buf, u32[:]...)
		for _, f := range vec {
			binary.BigEndian.PutUint32(u32[:], math.Float32bits(f))
			buf = append(buf, u32[:]...)
		}
	}
	binary.BigEndian.PutUint64(u64[:], uint64(ttl.Milliseconds())) //nolint:gosec
	buf = append(buf, u64[:]...)
	binary.BigEndian.PutUint32(u32[:], uint32(len(metaJSON))) //nolint:gosec
	buf = append(buf, u32[:]...)
	buf = append(buf, metaJSON...)
	return buf
}

// namedInsertSparseMarker is the per-space SPARSE sub-block marker. It rides at
// the byte offset immediately after the named-insert base block — the SAME offset
// the optional keyTTL present byte (0/1) would occupy. The legacy keyTTL present
// byte is only ever 0 or 1, so the value 2 unambiguously means "a sparse-spaces
// sub-block follows here, then the keyTTL/CAS trailers resume after it". When a
// point has NO sparse values the marker is NOT emitted, so an all-dense insert is
// BYTE-IDENTICAL to EncodeNamedInsertArgs and its keyTTL/CAS trailers sit at the
// exact same offsets as before this feature (a dense space is framed unchanged in
// the base block; the sparse block is purely additive and absent when empty).
//
// Sub-block layout after the marker:
//
//	[marker:u8=2][numSparse:u32]
//	  per space: [nameLen:u16][name][sparse frame: nIdx:u32, idx*u32, nVal:u32, val*f32]
const namedInsertSparseMarker uint8 = 2

// appendNamedSparseBlock appends the per-space SPARSE sub-block to buf and returns
// the grown slice. When sparseVectors is empty NOTHING is appended (all-dense
// byte-identical). The sparse frame reuses the shared SparseVector wire shape
// (ops.writeSparse). A nil/zero entry for a space is skipped (it carries no terms).
func appendNamedSparseBlock(buf []byte, sparseVectors map[string]*vtypes.SparseVector) []byte {
	// Count only non-nil entries so an all-nil map appends nothing.
	n := 0
	for _, sv := range sparseVectors {
		if sv != nil {
			n++
		}
	}
	if n == 0 {
		return buf
	}
	out := append(buf, namedInsertSparseMarker)
	var u32 [4]byte
	binary.BigEndian.PutUint32(u32[:], uint32(n)) //nolint:gosec
	out = append(out, u32[:]...)
	var u16 [2]byte
	for name, sv := range sparseVectors {
		if sv == nil {
			continue
		}
		binary.BigEndian.PutUint16(u16[:], uint16(len(name))) //nolint:gosec
		out = append(out, u16[:]...)
		out = append(out, name...)
		out = writeSparseAppend(out, *sv)
	}
	return out
}

// readNamedSparseBlock reads the OPTIONAL per-space SPARSE sub-block at offset off
// in args, returning the decoded map (nil when absent), the offset just past the
// block, and whether the block was present. A byte != namedInsertSparseMarker at
// off (incl. the legacy keyTTL present byte 0/1, or end-of-args) means "no sparse
// block": (nil, off, false, nil) — the caller then reads keyTTL/CAS at off
// unchanged (full back-compat). A present-but-truncated block is fail-loud.
func readNamedSparseBlock(args []byte, off int) (sparseVectors map[string]*vtypes.SparseVector, next int, present bool, err error) {
	if off >= len(args) || args[off] != namedInsertSparseMarker {
		return nil, off, false, nil
	}
	off++ // consume the marker
	if len(args) < off+4 {
		return nil, off, false, ErrVectorArgsTruncated
	}
	n := int(binary.BigEndian.Uint32(args[off:]))
	off += 4
	// An entry costs >= 6 bytes ([nameLen:u16] with an empty name + the sparse
	// block's own [nnz:u32] header).
	if !CountFitsIn(n, len(args)-off, 6) {
		return nil, off, false, ErrVectorArgsTruncated
	}
	sv := make(map[string]*vtypes.SparseVector, n)
	for i := 0; i < n; i++ {
		if len(args) < off+2 {
			return nil, off, false, ErrVectorArgsTruncated
		}
		nameLen := int(binary.BigEndian.Uint16(args[off:]))
		off += 2
		if len(args) < off+nameLen {
			return nil, off, false, ErrVectorArgsTruncated
		}
		name := string(args[off : off+nameLen])
		off += nameLen
		s, noff, rerr := readSparse(args, off)
		if rerr != nil {
			return nil, off, false, rerr
		}
		off = noff
		scopy := s // own the slices (readSparse returns a fresh value)
		sv[name] = &scopy
	}
	return sv, off, true, nil
}

// DecodeNamedInsertArgs reads args produced by EncodeNamedInsertArgs. Trailing
// bytes beyond the base block (e.g. a CAS trailer from EncodeNamedInsertArgsCAS)
// are ignored, so a single-shard non-CAS handler stays backward-compatible.
func DecodeNamedInsertArgs(args []byte) (col string, id uint64, vectors map[string][]float32, payload vtypes.Metadata, ttl time.Duration, err error) {
	col, id, vectors, payload, ttl, _, err = decodeNamedInsertArgsN(args)
	return col, id, vectors, payload, ttl, err
}

// decodeNamedInsertArgsN decodes the named-insert base block and returns the
// number of bytes consumed, so DecodeNamedInsertArgsCAS can read a trailing CAS
// block (self-delimiting behind a present byte).
func decodeNamedInsertArgsN(args []byte) (col string, id uint64, vectors map[string][]float32, payload vtypes.Metadata, ttl time.Duration, n int, err error) {
	if len(args) < 1 {
		return "", 0, nil, nil, 0, 0, ErrVectorArgsTruncated
	}
	colLen := int(args[0])
	if len(args) < 1+colLen+8+4 {
		return "", 0, nil, nil, 0, 0, ErrVectorArgsTruncated
	}
	col = string(args[1 : 1+colLen])
	off := 1 + colLen
	id = binary.BigEndian.Uint64(args[off:])
	off += 8
	numVecs := int(binary.BigEndian.Uint32(args[off:]))
	off += 4
	if numVecs > 0 {
		vectors = make(map[string][]float32, numVecs)
	}
	for i := 0; i < numVecs; i++ {
		if len(args) < off+2 {
			return "", 0, nil, nil, 0, 0, ErrVectorArgsTruncated
		}
		nameLen := int(binary.BigEndian.Uint16(args[off:]))
		off += 2
		if len(args) < off+nameLen+4 {
			return "", 0, nil, nil, 0, 0, ErrVectorArgsTruncated
		}
		name := string(args[off : off+nameLen])
		off += nameLen
		dim := int(binary.BigEndian.Uint32(args[off:]))
		off += 4
		if len(args) < off+4*dim {
			return "", 0, nil, nil, 0, 0, ErrVectorArgsTruncated
		}
		vec := make([]float32, dim)
		for j := 0; j < dim; j++ {
			vec[j] = math.Float32frombits(binary.BigEndian.Uint32(args[off:]))
			off += 4
		}
		vectors[name] = vec
	}
	if len(args) < off+8+4 {
		return "", 0, nil, nil, 0, 0, ErrVectorArgsTruncated
	}
	ttlMs := binary.BigEndian.Uint64(args[off:])
	off += 8
	ttl = time.Duration(ttlMs) * time.Millisecond //nolint:gosec
	mlen := int(binary.BigEndian.Uint32(args[off:]))
	off += 4
	if len(args) < off+mlen {
		return "", 0, nil, nil, 0, 0, ErrVectorArgsTruncated
	}
	if mlen > 0 {
		m := make(vtypes.Metadata)
		if err := json.Unmarshal(args[off:off+mlen], &m); err != nil {
			return "", 0, nil, nil, 0, 0, fmt.Errorf("ops: decode named payload: %w", err)
		}
		payload = m
	}
	off += mlen
	return col, id, vectors, payload, ttl, off, nil
}

// EncodeNamedInsertArgsCAS serializes a vector_named_insert request carrying an
// optimistic-CAS precondition: a trailing [casPresent:u8][?expectedVersion:u64]
// block after the base EncodeNamedInsertArgs payload. When hasExpected is false
// the output is BYTE-IDENTICAL to EncodeNamedInsertArgs (no CAS block; the legacy
// DecodeNamedInsertArgs reads the base and ignores trailing bytes). When present
// the handler turns it into CASCond{Expected: expectedVersion, Has: true}
// (expectedVersion 0 = expect-absent).
func EncodeNamedInsertArgsCAS(col string, id uint64, vectors map[string][]float32, payload vtypes.Metadata, ttl time.Duration, expectedVersion uint64, hasExpected bool) []byte {
	return EncodeNamedInsertArgsCASKeyTTL(col, id, vectors, payload, ttl, expectedVersion, hasExpected, nil)
}

// EncodeNamedInsertArgsKeyTTL serializes a vector_named_insert request carrying an
// OPTIONAL per-key payload TTL map (key -> RELATIVE ms; the engine computes the
// ABSOLUTE deadline now+ttl at insert, mirroring set_payload). When keyTTLMs is
// empty/nil the output is BYTE-IDENTICAL to EncodeNamedInsertArgs (no trailing
// keyTTL block). The keyTTL block rides AFTER the base block (before any CAS
// trailer), self-delimiting behind a present byte — exactly like set_payload.
func EncodeNamedInsertArgsKeyTTL(col string, id uint64, vectors map[string][]float32, payload vtypes.Metadata, ttl time.Duration, keyTTLMs map[string]int64) []byte {
	return EncodeNamedInsertArgsCASKeyTTL(col, id, vectors, payload, ttl, 0, false, keyTTLMs)
}

// EncodeNamedInsertArgsCASKeyTTL is EncodeNamedInsertArgsCAS plus an OPTIONAL
// per-key payload TTL map (key -> RELATIVE ms). The keyTTL block rides AFTER the
// base block and BEFORE the CAS block, so the two trailers coexist. To keep the
// CAS block at a deterministic offset when both are present, the keyTTL present
// byte is ALWAYS emitted when a CAS block follows (0 when no map) — exactly the
// set_payload-CAS interpose. When keyTTLMs is empty AND hasExpected is false the
// output is BYTE-IDENTICAL to EncodeNamedInsertArgs (no trailing bytes at all).
func EncodeNamedInsertArgsCASKeyTTL(col string, id uint64, vectors map[string][]float32, payload vtypes.Metadata, ttl time.Duration, expectedVersion uint64, hasExpected bool, keyTTLMs map[string]int64) []byte {
	return EncodeNamedInsertArgsSparseCASKeyTTL(col, id, vectors, nil, payload, ttl, expectedVersion, hasExpected, keyTTLMs)
}

// EncodeNamedInsertArgsSparseCASKeyTTL is EncodeNamedInsertArgsCASKeyTTL carrying
// per-space SPARSE values (sparseVectors[space] is the *SparseVector for a sparse
// space) in an ADDITIVE sub-block that rides AFTER the dense base block and BEFORE
// the keyTTL/CAS trailers. When sparseVectors is empty the sub-block is NOT emitted
// and the output is BYTE-IDENTICAL to EncodeNamedInsertArgsCASKeyTTL (and, with no
// keyTTL/CAS, to EncodeNamedInsertArgs) — a dense-only insert is unchanged on the
// wire. A space entry is dense XOR sparse: a dense value rides the base block's
// vectors framing, a sparse value rides this sub-block; the modality is validated
// engine-side (ErrSpaceModalityMismatch). See namedInsertSparseMarker.
func EncodeNamedInsertArgsSparseCASKeyTTL(col string, id uint64, vectors map[string][]float32, sparseVectors map[string]*vtypes.SparseVector, payload vtypes.Metadata, ttl time.Duration, expectedVersion uint64, hasExpected bool, keyTTLMs map[string]int64) []byte {
	base := EncodeNamedInsertArgs(col, id, vectors, payload, ttl)
	base = appendNamedSparseBlock(base, sparseVectors) // additive; nothing when all-dense
	out := appendKeyTTLBlock(base, keyTTLMs, hasExpected)
	if !hasExpected {
		return out
	}
	out = append(out, 1)
	var u64 [8]byte
	binary.BigEndian.PutUint64(u64[:], expectedVersion)
	return append(out, u64[:]...)
}

// DecodeNamedInsertArgsCAS reads args produced by EncodeNamedInsertArgsCAS (or the
// legacy EncodeNamedInsertArgs — the CAS block is optional). hasExpected reports
// whether a CAS precondition trailer was present; expectedVersion is its value (0 =
// expect-absent). A legacy blob (no CAS block) decodes to hasExpected=false; a
// present-but-truncated CAS block is fail-loud.
func DecodeNamedInsertArgsCAS(args []byte) (col string, id uint64, vectors map[string][]float32, payload vtypes.Metadata, ttl time.Duration, expectedVersion uint64, hasExpected bool, err error) {
	col, id, vectors, payload, ttl, expectedVersion, hasExpected, _, err = DecodeNamedInsertArgsKeyTTL(args)
	return col, id, vectors, payload, ttl, expectedVersion, hasExpected, err
}

// DecodeNamedInsertArgsKeyTTL reads args produced by EncodeNamedInsertArgsKeyTTL /
// EncodeNamedInsertArgsCASKeyTTL (or the legacy EncodeNamedInsertArgs/CAS — both
// trailers are optional). It decodes the base block, then the OPTIONAL
// self-delimiting per-key TTL block (key -> RELATIVE ms; nil when absent), then the
// OPTIONAL [casPresent:u8][expectedVersion:u64] block. A legacy blob (no trailers)
// decodes to keyTTLMs=nil, hasExpected=false. A present-but-truncated trailer is
// fail-loud. The handler turns keyTTLMs into absolute deadlines at insert.
func DecodeNamedInsertArgsKeyTTL(args []byte) (col string, id uint64, vectors map[string][]float32, payload vtypes.Metadata, ttl time.Duration, expectedVersion uint64, hasExpected bool, keyTTLMs map[string]int64, err error) {
	col, id, vectors, _, payload, ttl, expectedVersion, hasExpected, keyTTLMs, err = DecodeNamedInsertArgsSparseKeyTTL(args)
	return col, id, vectors, payload, ttl, expectedVersion, hasExpected, keyTTLMs, err
}

// DecodeNamedInsertArgsSparseKeyTTL is DecodeNamedInsertArgsKeyTTL plus the per-
// space SPARSE values carried in the additive sparse sub-block (nil when absent,
// i.e. a dense-only / legacy insert). Wire-compatible with every prior named-insert
// encoder: the sparse sub-block rides behind the namedInsertSparseMarker at the
// offset just past the dense base block; a legacy blob has no marker there (it has
// the keyTTL present byte 0/1 or nothing), so sparseVectors is nil. The keyTTL/CAS
// trailers are read AFTER the (optional) sparse block, at their natural offsets.
func DecodeNamedInsertArgsSparseKeyTTL(args []byte) (col string, id uint64, vectors map[string][]float32, sparseVectors map[string]*vtypes.SparseVector, payload vtypes.Metadata, ttl time.Duration, expectedVersion uint64, hasExpected bool, keyTTLMs map[string]int64, err error) {
	col, id, vectors, payload, ttl, n, err := decodeNamedInsertArgsN(args)
	if err != nil {
		return "", 0, nil, nil, nil, 0, 0, false, nil, err
	}
	sparseVectors, n, _, err = readNamedSparseBlock(args, n)
	if err != nil {
		return "", 0, nil, nil, nil, 0, 0, false, nil, err
	}
	keyTTLMs, off, err := readKeyTTLBlock(args, n)
	if err != nil {
		return "", 0, nil, nil, nil, 0, 0, false, nil, err
	}
	if off >= len(args) || args[off] == 0 {
		return col, id, vectors, sparseVectors, payload, ttl, 0, false, keyTTLMs, nil
	}
	off++
	if len(args) < off+8 {
		return "", 0, nil, nil, nil, 0, 0, false, nil, ErrVectorArgsTruncated
	}
	expectedVersion = binary.BigEndian.Uint64(args[off:])
	return col, id, vectors, sparseVectors, payload, ttl, expectedVersion, true, keyTTLMs, nil
}

// EncodeNamedSearchArgs serializes a vector_named_search / search_docs request.
// Wire: [colLen:u8][col][nameLen:u16][vecName][k:u32][dim:u32][dim*f32]
//
//	[filterLen:u32][filterJSON]
//
// vecName names which configured space to query; filterLen 0 = no filter.
func EncodeNamedSearchArgs(col, vecName string, query []float32, k int, filter vtypes.Filter) []byte {
	var filterJSON []byte
	if !filter.IsZero() {
		filterJSON, _ = json.Marshal(filter)
	}
	buf := make([]byte, 0, 1+len(col)+2+len(vecName)+4+4+4*len(query)+4+len(filterJSON))
	buf = append(buf, byte(len(col)))
	buf = append(buf, col...)
	var u16 [2]byte
	binary.BigEndian.PutUint16(u16[:], uint16(len(vecName))) //nolint:gosec
	buf = append(buf, u16[:]...)
	buf = append(buf, vecName...)
	var u32 [4]byte
	binary.BigEndian.PutUint32(u32[:], uint32(k)) //nolint:gosec
	buf = append(buf, u32[:]...)
	binary.BigEndian.PutUint32(u32[:], uint32(len(query))) //nolint:gosec
	buf = append(buf, u32[:]...)
	for _, f := range query {
		binary.BigEndian.PutUint32(u32[:], math.Float32bits(f))
		buf = append(buf, u32[:]...)
	}
	binary.BigEndian.PutUint32(u32[:], uint32(len(filterJSON))) //nolint:gosec
	buf = append(buf, u32[:]...)
	buf = append(buf, filterJSON...)
	return buf
}

// DecodeNamedSearchArgs reads args produced by EncodeNamedSearchArgs. Trailing
// bytes beyond the base block (e.g. an opts trailer from
// EncodeNamedSearchArgsOpts) are ignored, so a single-shard handler stays
// backward-compatible with rc-carrying args.
func DecodeNamedSearchArgs(args []byte) (col, vecName string, query []float32, k int, filter vtypes.Filter, err error) {
	col, vecName, query, k, _, filter, err = decodeNamedSearchArgsN(args)
	return col, vecName, query, k, filter, err
}

// decodeNamedSearchArgsN decodes the named-search base block and returns the
// number of bytes consumed, so DecodeNamedSearchArgsOpts can read a trailing
// opts block (self-delimiting behind a marker byte).
func decodeNamedSearchArgsN(args []byte) (col, vecName string, query []float32, k, n int, filter vtypes.Filter, err error) {
	if len(args) < 1 {
		return "", "", nil, 0, 0, vtypes.Filter{}, ErrVectorArgsTruncated
	}
	colLen := int(args[0])
	if len(args) < 1+colLen+2 {
		return "", "", nil, 0, 0, vtypes.Filter{}, ErrVectorArgsTruncated
	}
	col = string(args[1 : 1+colLen])
	off := 1 + colLen
	nameLen := int(binary.BigEndian.Uint16(args[off:]))
	off += 2
	if len(args) < off+nameLen+4+4 {
		return "", "", nil, 0, 0, vtypes.Filter{}, ErrVectorArgsTruncated
	}
	vecName = string(args[off : off+nameLen])
	off += nameLen
	k = int(binary.BigEndian.Uint32(args[off:]))
	off += 4
	dim := int(binary.BigEndian.Uint32(args[off:]))
	off += 4
	if len(args) < off+4*dim+4 {
		return "", "", nil, 0, 0, vtypes.Filter{}, ErrVectorArgsTruncated
	}
	query = make([]float32, dim)
	for j := 0; j < dim; j++ {
		query[j] = math.Float32frombits(binary.BigEndian.Uint32(args[off:]))
		off += 4
	}
	flen := int(binary.BigEndian.Uint32(args[off:]))
	off += 4
	if len(args) < off+flen {
		return "", "", nil, 0, 0, vtypes.Filter{}, ErrVectorArgsTruncated
	}
	if flen > 0 {
		if err := json.Unmarshal(args[off:off+flen], &filter); err != nil {
			return "", "", nil, 0, 0, vtypes.Filter{}, fmt.Errorf("ops: decode named filter: %w", err)
		}
	}
	off += flen
	return col, vecName, query, k, off, filter, nil
}

// Named-search opts-trailer marker bit. Named search args already carry a filter
// in the base block (length-prefixed JSON), so — unlike MV — the trailer only
// needs to carry the cross-shard consistency opts pair. The marker is still a
// BITFIELD (mirroring mvTrailerOpts) so the trailer is forward-extensible and
// self-describing: bit0 (NamedTrailerOpts) means [rc:u8][opa:u8] follow the
// marker. A zero marker is never emitted (no trailer at all), which is how the
// no-rc/no-opa case stays BYTE-IDENTICAL to EncodeNamedSearchArgs.
const NamedTrailerOpts uint8 = 1 << 0

// namedTrailerStaleness is the second trailer marker bit: when set (ONLY for a
// ConsistencyBoundedStaleness read) an 8-byte big-endian staleness bound follows
// the [rc][opa] pair. Additive: rc∈{0,1,2} never set it, so those trailers are
// byte-identical to the pre-bounded-staleness form.
const namedTrailerStaleness uint8 = 1 << 1

// EncodeNamedSearchArgsOpts serializes a vector_named_search / _search_docs
// request (both share this codec) plus an optional self-delimiting trailer
// carrying the cross-shard consistency opts pair. The base block (incl. the
// length-prefixed filter) is UNCHANGED — the filter stays in the base; only the
// rc/opa trailer is appended. Wire when the trailer is present:
//
//	<base block from EncodeNamedSearchArgs (incl. filter)>
//	  [marker:u8]
//	  [rc:u8][opa:u8]   ← present when marker&NamedTrailerOpts
//
// When readConsistency==0 AND onPartitionUnavailable==0 the trailer is omitted
// entirely and the output is BYTE-IDENTICAL to EncodeNamedSearchArgs
// (backward-compatible); the plain DecodeNamedSearchArgs (single-shard handler)
// ignores any trailing bytes, so an old decoder reads the same base.
func EncodeNamedSearchArgsOpts(col, vecName string, query []float32, k int, filter vtypes.Filter, readConsistency, onPartitionUnavailable uint8, bound uint64) []byte {
	base := EncodeNamedSearchArgs(col, vecName, query, k, filter)
	var marker uint8
	if readConsistency != 0 || onPartitionUnavailable != 0 {
		marker |= NamedTrailerOpts
	}
	if readConsistency == ConsistencyBoundedStaleness {
		marker |= namedTrailerStaleness
	}
	if marker == 0 {
		return base // byte-identical to the legacy / no-trailer form
	}
	out := append(base, marker)
	if marker&NamedTrailerOpts != 0 {
		out = append(out, readConsistency, onPartitionUnavailable)
	}
	if marker&namedTrailerStaleness != 0 {
		var b [8]byte
		binary.BigEndian.PutUint64(b[:], bound)
		out = append(out, b[:]...)
	}
	return out
}

// DecodeNamedSearchArgsOpts decodes a vector_named_search / _search_docs request
// that may carry the self-delimiting opts trailer (rc/opa). Backward-compatible:
// legacy args (no trailer) decode with readConsistency=0, onPartitionUnavailable=0.
// A present marker with a truncated rc/opa block is corruption — fail loud (never
// a silent drop, so a Linearizable read never silently degrades to stale).
func DecodeNamedSearchArgsOpts(args []byte) (col, vecName string, query []float32, k int, filter vtypes.Filter, readConsistency, onPartitionUnavailable uint8, bound uint64, err error) {
	col, vecName, query, k, n, filter, err := decodeNamedSearchArgsN(args)
	if err != nil {
		return "", "", nil, 0, vtypes.Filter{}, 0, 0, 0, err
	}
	if len(args) <= n || args[n] == 0 {
		// No trailer (legacy form). A zero marker is never emitted, so treat a zero
		// byte here as "no trailer" too (trailing-bytes tolerance), matching
		// DecodeNamedSearchArgs's contract.
		return col, vecName, query, k, filter, 0, 0, 0, nil
	}
	marker := args[n]
	off := n + 1
	if marker&NamedTrailerOpts != 0 {
		if len(args) < off+2 {
			return "", "", nil, 0, vtypes.Filter{}, 0, 0, 0, ErrVectorArgsTruncated
		}
		readConsistency = args[off]
		onPartitionUnavailable = args[off+1]
		off += 2
	}
	if marker&namedTrailerStaleness != 0 {
		if len(args) < off+8 {
			return "", "", nil, 0, vtypes.Filter{}, 0, 0, 0, ErrVectorArgsTruncated
		}
		bound = binary.BigEndian.Uint64(args[off : off+8])
	}
	return col, vecName, query, k, filter, readConsistency, onPartitionUnavailable, bound, nil
}

// EncodeNamedSparseSearchArgs serializes a vector_named_sparse_search request.
// Wire: [colLen:u8][col][spaceLen:u16][space][k:u32][sparse frame][filterLen:u32]
// [filterJSON] where the sparse frame is the shared [nnz:u32]{[dim:u32][value:f32]}
// shape (writeSparse). space names the SPARSE space to query; filterLen 0 = no
// filter. This is the sparse-lane analogue of EncodeNamedSearchArgs (whose query
// is a dense [dim][floats] block); here the query is a sparse frame instead.
func EncodeNamedSparseSearchArgs(col, space string, query vtypes.SparseVector, k int, filter vtypes.Filter) []byte {
	var filterJSON []byte
	if !filter.IsZero() {
		filterJSON, _ = json.Marshal(filter)
	}
	buf := make([]byte, 0, 1+len(col)+2+len(space)+4+4+8*len(query.Indices)+4+len(filterJSON))
	buf = append(buf, byte(len(col)))
	buf = append(buf, col...)
	var u16 [2]byte
	binary.BigEndian.PutUint16(u16[:], uint16(len(space))) //nolint:gosec
	buf = append(buf, u16[:]...)
	buf = append(buf, space...)
	var u32 [4]byte
	binary.BigEndian.PutUint32(u32[:], uint32(k)) //nolint:gosec
	buf = append(buf, u32[:]...)
	buf = writeSparseAppend(buf, query)
	binary.BigEndian.PutUint32(u32[:], uint32(len(filterJSON))) //nolint:gosec
	buf = append(buf, u32[:]...)
	buf = append(buf, filterJSON...)
	return buf
}

// DecodeNamedSparseSearchArgs reads args produced by EncodeNamedSparseSearchArgs.
// Trailing bytes beyond the base block (the rc/opa opts trailer) are ignored, so a
// single-shard handler stays backward-compatible with rc-carrying args.
func DecodeNamedSparseSearchArgs(args []byte) (col, space string, query vtypes.SparseVector, k int, filter vtypes.Filter, err error) {
	col, space, query, k, _, filter, err = decodeNamedSparseSearchArgsN(args)
	return col, space, query, k, filter, err
}

// decodeNamedSparseSearchArgsN decodes the named-sparse-search base block and
// returns the bytes consumed, so DecodeNamedSparseSearchArgsOpts can read a
// trailing opts block (self-delimiting behind a marker byte, like named search).
func decodeNamedSparseSearchArgsN(args []byte) (col, space string, query vtypes.SparseVector, k, n int, filter vtypes.Filter, err error) {
	if len(args) < 1 {
		return "", "", vtypes.SparseVector{}, 0, 0, vtypes.Filter{}, ErrVectorArgsTruncated
	}
	colLen := int(args[0])
	if len(args) < 1+colLen+2 {
		return "", "", vtypes.SparseVector{}, 0, 0, vtypes.Filter{}, ErrVectorArgsTruncated
	}
	col = string(args[1 : 1+colLen])
	off := 1 + colLen
	nameLen := int(binary.BigEndian.Uint16(args[off:]))
	off += 2
	if len(args) < off+nameLen+4 {
		return "", "", vtypes.SparseVector{}, 0, 0, vtypes.Filter{}, ErrVectorArgsTruncated
	}
	space = string(args[off : off+nameLen])
	off += nameLen
	k = int(binary.BigEndian.Uint32(args[off:]))
	off += 4
	query, off, err = readSparse(args, off)
	if err != nil {
		return "", "", vtypes.SparseVector{}, 0, 0, vtypes.Filter{}, err
	}
	if len(args) < off+4 {
		return "", "", vtypes.SparseVector{}, 0, 0, vtypes.Filter{}, ErrVectorArgsTruncated
	}
	flen := int(binary.BigEndian.Uint32(args[off:]))
	off += 4
	if len(args) < off+flen {
		return "", "", vtypes.SparseVector{}, 0, 0, vtypes.Filter{}, ErrVectorArgsTruncated
	}
	if flen > 0 {
		if uerr := json.Unmarshal(args[off:off+flen], &filter); uerr != nil {
			return "", "", vtypes.SparseVector{}, 0, 0, vtypes.Filter{}, fmt.Errorf("ops: decode named sparse filter: %w", uerr)
		}
	}
	off += flen
	return col, space, query, k, off, filter, nil
}

// EncodeNamedSparseSearchArgsOpts serializes a vector_named_sparse_search request
// plus an optional self-delimiting rc/opa opts trailer, mirroring
// EncodeNamedSearchArgsOpts exactly (the same NamedTrailerOpts marker bit). When
// readConsistency==0 AND onPartitionUnavailable==0 the trailer is omitted and the
// output is BYTE-IDENTICAL to EncodeNamedSparseSearchArgs.
func EncodeNamedSparseSearchArgsOpts(col, space string, query vtypes.SparseVector, k int, filter vtypes.Filter, readConsistency, onPartitionUnavailable uint8, bound uint64) []byte {
	base := EncodeNamedSparseSearchArgs(col, space, query, k, filter)
	var marker uint8
	if readConsistency != 0 || onPartitionUnavailable != 0 {
		marker |= NamedTrailerOpts
	}
	if readConsistency == ConsistencyBoundedStaleness {
		marker |= namedTrailerStaleness
	}
	if marker == 0 {
		return base
	}
	out := append(base, marker)
	if marker&NamedTrailerOpts != 0 {
		out = append(out, readConsistency, onPartitionUnavailable)
	}
	if marker&namedTrailerStaleness != 0 {
		var b [8]byte
		binary.BigEndian.PutUint64(b[:], bound)
		out = append(out, b[:]...)
	}
	return out
}

// DecodeNamedSparseSearchArgsOpts decodes a vector_named_sparse_search request that
// may carry the self-delimiting rc/opa opts trailer. Backward-compatible: legacy
// args (no trailer) decode with rc=0/opa=0. A present marker with a truncated
// rc/opa block is fail-loud (so a Linearizable sparse read never silently degrades).
func DecodeNamedSparseSearchArgsOpts(args []byte) (col, space string, query vtypes.SparseVector, k int, filter vtypes.Filter, readConsistency, onPartitionUnavailable uint8, bound uint64, err error) {
	col, space, query, k, n, filter, err := decodeNamedSparseSearchArgsN(args)
	if err != nil {
		return "", "", vtypes.SparseVector{}, 0, vtypes.Filter{}, 0, 0, 0, err
	}
	if len(args) <= n || args[n] == 0 {
		return col, space, query, k, filter, 0, 0, 0, nil
	}
	marker := args[n]
	off := n + 1
	if marker&NamedTrailerOpts != 0 {
		if len(args) < off+2 {
			return "", "", vtypes.SparseVector{}, 0, vtypes.Filter{}, 0, 0, 0, ErrVectorArgsTruncated
		}
		readConsistency = args[off]
		onPartitionUnavailable = args[off+1]
		off += 2
	}
	if marker&namedTrailerStaleness != 0 {
		if len(args) < off+8 {
			return "", "", vtypes.SparseVector{}, 0, vtypes.Filter{}, 0, 0, 0, ErrVectorArgsTruncated
		}
		bound = binary.BigEndian.Uint64(args[off : off+8])
	}
	return col, space, query, k, filter, readConsistency, onPartitionUnavailable, bound, nil
}

// namedHybridFlag* are the named-hybrid arg flag bits. The named-hybrid wire is
// the cross-space analogue of the dense hybrid wire (EncodeHybridSearchArgs): it
// carries TWO space names (a dense + a sparse) instead of one collection's single
// space, a dense query AND a sparse query, the fusion opts, an optional filter, and
// an optional rc/opa trailer. The flags byte advertises the optional blocks so an
// all-default request stays compact and the decode is unambiguous.
const (
	namedHybridFlagFilter uint8 = 1 << 0 // [filterLen:u32][filterJSON] present
	namedHybridFlagSparse uint8 = 1 << 1 // the sparse query frame carries terms (non-zero)
	NamedHybridFlagOpts   uint8 = 1 << 2 // [rc:u8][opa:u8] trailer present
)

// EncodeNamedHybridArgs serializes a vector_named_hybrid_search /
// vector_named_hybrid_lanes request (both share this codec). Wire:
//
//	[flags:u8]                       bit0=HAS_FILTER, bit1=HAS_SPARSE, bit2=HAS_OPTS
//	[colLen:u8][col]
//	[denseSpaceLen:u16][denseSpace]
//	[sparseSpaceLen:u16][sparseSpace]
//	[k:u32]
//	[method:u8][alpha:f64][rrfK:u32][denseK:u32][sparseK:u32]
//	[dim:u32][dense: f32×dim]
//	[sparse frame: nnz:u32, {dim:u32, value:f32}×nnz]   ← always written; nnz 0 = empty
//	if HAS_FILTER: [filterLen:u32][filterJSON]
//	if HAS_OPTS:   [rc:u8][opa:u8]
//
// The dense query rides as a dim-prefixed float block (dim 0 = sparse-only); the
// sparse query rides as the shared sparse frame (writeSparseAppend; nnz 0 =
// dense-only). HAS_SPARSE is set iff the sparse query carries terms, so the
// degradation cases are self-describing. When rc==0 && opa==0 the opts trailer is
// omitted and HAS_OPTS is clear (byte-identical trailer).
func EncodeNamedHybridArgs(col, denseSpace string, denseQ []float32, sparseSpace string, sparseQ vtypes.SparseVector, k int, opts vtypes.HybridOpts, readConsistency, onPartitionUnavailable uint8, bound uint64) []byte {
	var flags uint8
	var filterJSON []byte
	if !opts.Filter.IsZero() {
		flags |= namedHybridFlagFilter
		filterJSON, _ = json.Marshal(opts.Filter)
	}
	if !sparseQ.IsZero() {
		flags |= namedHybridFlagSparse
	}
	if readConsistency != 0 || onPartitionUnavailable != 0 {
		flags |= NamedHybridFlagOpts
	}
	buf := make([]byte, 0, 1+1+len(col)+2+len(denseSpace)+2+len(sparseSpace)+4+(1+8+4+4+4)+4+4*len(denseQ)+4+8*len(sparseQ.Indices)+4+len(filterJSON)+2)
	buf = append(buf, flags)
	buf = append(buf, byte(len(col)))
	buf = append(buf, col...)
	var u16 [2]byte
	var u32 [4]byte
	var u64 [8]byte
	binary.BigEndian.PutUint16(u16[:], uint16(len(denseSpace))) //nolint:gosec
	buf = append(buf, u16[:]...)
	buf = append(buf, denseSpace...)
	binary.BigEndian.PutUint16(u16[:], uint16(len(sparseSpace))) //nolint:gosec
	buf = append(buf, u16[:]...)
	buf = append(buf, sparseSpace...)
	binary.BigEndian.PutUint32(u32[:], uint32(k)) //nolint:gosec
	buf = append(buf, u32[:]...)
	buf = append(buf, byte(opts.Method))
	binary.BigEndian.PutUint64(u64[:], math.Float64bits(opts.Alpha))
	buf = append(buf, u64[:]...)
	binary.BigEndian.PutUint32(u32[:], uint32(opts.RRFK)) //nolint:gosec
	buf = append(buf, u32[:]...)
	binary.BigEndian.PutUint32(u32[:], uint32(opts.DenseK)) //nolint:gosec
	buf = append(buf, u32[:]...)
	binary.BigEndian.PutUint32(u32[:], uint32(opts.SparseK)) //nolint:gosec
	buf = append(buf, u32[:]...)
	binary.BigEndian.PutUint32(u32[:], uint32(len(denseQ))) //nolint:gosec
	buf = append(buf, u32[:]...)
	for _, f := range denseQ {
		binary.BigEndian.PutUint32(u32[:], math.Float32bits(f))
		buf = append(buf, u32[:]...)
	}
	buf = writeSparseAppend(buf, sparseQ) // always: nnz 0 = empty (dense-only)
	if flags&namedHybridFlagFilter != 0 {
		binary.BigEndian.PutUint32(u32[:], uint32(len(filterJSON))) //nolint:gosec
		buf = append(buf, u32[:]...)
		buf = append(buf, filterJSON...)
	}
	if flags&NamedHybridFlagOpts != 0 {
		buf = append(buf, readConsistency, onPartitionUnavailable)
		buf = appendBoundTail(buf, readConsistency, bound) // 8 bound bytes ride ONLY when rc==BoundedStaleness
	}
	return buf
}

// DecodeNamedHybridArgs reads args produced by EncodeNamedHybridArgs. opts.Filter
// is the zero filter when absent; sparseQ is the zero SparseVector when absent; rc/
// opa are 0 when the opts trailer is absent. A present flag with a truncated block
// is fail-loud (so a Linearizable named hybrid never silently degrades to stale).
func DecodeNamedHybridArgs(args []byte) (col, denseSpace string, denseQ []float32, sparseSpace string, sparseQ vtypes.SparseVector, k int, opts vtypes.HybridOpts, readConsistency, onPartitionUnavailable uint8, bound uint64, err error) {
	if len(args) < 2 {
		return "", "", nil, "", sparseQ, 0, opts, 0, 0, 0, ErrVectorArgsTruncated
	}
	flags := args[0]
	colLen := int(args[1])
	off := 2
	if len(args) < off+colLen+2 {
		return "", "", nil, "", sparseQ, 0, opts, 0, 0, 0, ErrVectorArgsTruncated
	}
	col = string(args[off : off+colLen])
	off += colLen
	dsLen := int(binary.BigEndian.Uint16(args[off:]))
	off += 2
	if len(args) < off+dsLen+2 {
		return "", "", nil, "", sparseQ, 0, opts, 0, 0, 0, ErrVectorArgsTruncated
	}
	denseSpace = string(args[off : off+dsLen])
	off += dsLen
	ssLen := int(binary.BigEndian.Uint16(args[off:]))
	off += 2
	if len(args) < off+ssLen+4+(1+8+4+4+4)+4 {
		return "", "", nil, "", sparseQ, 0, opts, 0, 0, 0, ErrVectorArgsTruncated
	}
	sparseSpace = string(args[off : off+ssLen])
	off += ssLen
	k = int(binary.BigEndian.Uint32(args[off:]))
	off += 4
	opts.Method = vtypes.FusionMethod(args[off])
	off++
	opts.Alpha = math.Float64frombits(binary.BigEndian.Uint64(args[off:]))
	off += 8
	opts.RRFK = int(binary.BigEndian.Uint32(args[off:]))
	off += 4
	opts.DenseK = int(binary.BigEndian.Uint32(args[off:]))
	off += 4
	opts.SparseK = int(binary.BigEndian.Uint32(args[off:]))
	off += 4
	dim := int(binary.BigEndian.Uint32(args[off:]))
	off += 4
	if len(args) < off+4*dim {
		return "", "", nil, "", sparseQ, 0, opts, 0, 0, 0, ErrVectorArgsTruncated
	}
	if dim > 0 {
		denseQ = make([]float32, dim)
		for i := 0; i < dim; i++ {
			denseQ[i] = math.Float32frombits(binary.BigEndian.Uint32(args[off:]))
			off += 4
		}
	}
	sparseQ, off, err = readSparse(args, off)
	if err != nil {
		return "", "", nil, "", vtypes.SparseVector{}, 0, opts, 0, 0, 0, err
	}
	if flags&namedHybridFlagFilter != 0 {
		if len(args) < off+4 {
			return "", "", nil, "", vtypes.SparseVector{}, 0, opts, 0, 0, 0, ErrVectorArgsTruncated
		}
		flen := int(binary.BigEndian.Uint32(args[off:]))
		off += 4
		if len(args) < off+flen {
			return "", "", nil, "", vtypes.SparseVector{}, 0, opts, 0, 0, 0, ErrVectorArgsTruncated
		}
		if uerr := json.Unmarshal(args[off:off+flen], &opts.Filter); uerr != nil {
			return "", "", nil, "", vtypes.SparseVector{}, 0, opts, 0, 0, 0, fmt.Errorf("ops: decode named hybrid filter: %w", uerr)
		}
		off += flen
	}
	if flags&NamedHybridFlagOpts != 0 {
		if len(args) < off+2 {
			return "", "", nil, "", vtypes.SparseVector{}, 0, opts, 0, 0, 0, ErrVectorArgsTruncated
		}
		readConsistency = args[off]
		onPartitionUnavailable = args[off+1]
		off += 2
		bound, _, err = readBoundTail(args, off, readConsistency)
		if err != nil {
			return "", "", nil, "", vtypes.SparseVector{}, 0, opts, 0, 0, 0, err
		}
	}
	return col, denseSpace, denseQ, sparseSpace, sparseQ, k, opts, readConsistency, onPartitionUnavailable, bound, nil
}

// EncodeNamedDeleteArgs serializes a vector_named_delete request.
// Wire: [colLen:u8][col][id:u64].
func EncodeNamedDeleteArgs(col string, id uint64) []byte {
	buf := make([]byte, 1+len(col)+8)
	buf[0] = byte(len(col))
	off := 1 + copy(buf[1:], col)
	binary.BigEndian.PutUint64(buf[off:], id)
	return buf
}

// DecodeNamedDeleteArgs reads args produced by EncodeNamedDeleteArgs. Trailing
// bytes beyond the base block (a CAS trailer) are ignored for the non-CAS path.
func DecodeNamedDeleteArgs(args []byte) (string, uint64, error) {
	col, id, _, _, err := DecodeNamedDeleteArgsCAS(args)
	return col, id, err
}

// EncodeNamedDeleteArgsCAS serializes a vector_named_delete request with an
// optional optimistic-CAS precondition. When hasExpected is false the output is
// BYTE-IDENTICAL to EncodeNamedDeleteArgs (no trailer). When present the trailing
// [1][expectedVersion:u64] is the CAS guard (expectedVersion 0 = expect-absent).
func EncodeNamedDeleteArgsCAS(col string, id uint64, expectedVersion uint64, hasExpected bool) []byte {
	base := EncodeNamedDeleteArgs(col, id)
	if !hasExpected {
		return base
	}
	out := append(base, 1)
	var u64 [8]byte
	binary.BigEndian.PutUint64(u64[:], expectedVersion)
	return append(out, u64[:]...)
}

// DecodeNamedDeleteArgsCAS reads args produced by EncodeNamedDeleteArgsCAS (or the
// legacy EncodeNamedDeleteArgs — the CAS block is optional). hasExpected reports
// whether a CAS precondition trailer was present.
func DecodeNamedDeleteArgsCAS(args []byte) (col string, id uint64, expectedVersion uint64, hasExpected bool, err error) {
	if len(args) < 1 {
		return "", 0, 0, false, ErrVectorArgsTruncated
	}
	colLen := int(args[0])
	if len(args) < 1+colLen+8 {
		return "", 0, 0, false, ErrVectorArgsTruncated
	}
	col = string(args[1 : 1+colLen])
	id = binary.BigEndian.Uint64(args[1+colLen:])
	off := 1 + colLen + 8
	if off >= len(args) || args[off] == 0 {
		return col, id, 0, false, nil
	}
	off++
	if len(args) < off+8 {
		return "", 0, 0, false, ErrVectorArgsTruncated
	}
	expectedVersion = binary.BigEndian.Uint64(args[off:])
	return col, id, expectedVersion, true, nil
}

// EncodeNamedScrollArgs serializes a vector_named_scroll request.
// Wire: [colLen:u8][col][limit:u32][filterLen:u32][filterJSON] (filterLen 0 = no filter).
func EncodeNamedScrollArgs(col string, filter vtypes.Filter, limit int) []byte {
	var filterJSON []byte
	if !filter.IsZero() {
		filterJSON, _ = json.Marshal(filter)
	}
	buf := make([]byte, 0, 1+len(col)+4+4+len(filterJSON))
	buf = append(buf, byte(len(col)))
	buf = append(buf, col...)
	var u32 [4]byte
	binary.BigEndian.PutUint32(u32[:], uint32(limit)) //nolint:gosec
	buf = append(buf, u32[:]...)
	binary.BigEndian.PutUint32(u32[:], uint32(len(filterJSON))) //nolint:gosec
	buf = append(buf, u32[:]...)
	buf = append(buf, filterJSON...)
	return buf
}

// DecodeNamedScrollArgs reads args produced by EncodeNamedScrollArgs. Trailing
// bytes beyond the base block (e.g. a cursor trailer from
// EncodeNamedScrollArgsCursor) are ignored, so a single-shard handler stays
// backward-compatible with cursor-carrying args.
func DecodeNamedScrollArgs(args []byte) (col string, filter vtypes.Filter, limit int, err error) {
	col, filter, limit, _, err = decodeNamedScrollArgsN(args)
	return col, filter, limit, err
}

// decodeNamedScrollArgsN decodes the named-scroll base block and returns the
// number of bytes consumed, so DecodeNamedScrollArgsCursor can read a trailing
// cursor block (self-delimiting behind a present byte).
func decodeNamedScrollArgsN(args []byte) (col string, filter vtypes.Filter, limit int, n int, err error) {
	if len(args) < 1 {
		return "", vtypes.Filter{}, 0, 0, ErrVectorArgsTruncated
	}
	colLen := int(args[0])
	if len(args) < 1+colLen+4+4 {
		return "", vtypes.Filter{}, 0, 0, ErrVectorArgsTruncated
	}
	col = string(args[1 : 1+colLen])
	off := 1 + colLen
	limit = int(binary.BigEndian.Uint32(args[off:]))
	off += 4
	flen := int(binary.BigEndian.Uint32(args[off:]))
	off += 4
	if len(args) < off+flen {
		return "", vtypes.Filter{}, 0, 0, ErrVectorArgsTruncated
	}
	if flen > 0 {
		if err := json.Unmarshal(args[off:off+flen], &filter); err != nil {
			return "", vtypes.Filter{}, 0, 0, fmt.Errorf("ops: decode named scroll filter: %w", err)
		}
	}
	off += flen
	return col, filter, limit, off, nil
}

// Named-scroll trailer marker bits. The trailer rides behind a single marker
// byte immediately after the base block and is a BITFIELD so a resume-after-id
// cursor and the cross-shard consistency opts pair can be carried independently:
//
//	bit0 (NamedScrollCursor): [afterID:u64 BE] follows the marker.
//	bit1 (NamedScrollOpts):   [rc:u8][opa:u8] follow (after the afterID, when present).
//
// CRUCIAL backward-compat: the legacy cursor trailer was "[1][afterID:u64]" —
// that is EXACTLY marker==NamedScrollCursor with the afterID block, BYTE-IDENTICAL,
// so old encoders/decoders interoperate. A zero marker is never emitted (no
// trailer at all), which is how the no-cursor/no-opts case stays byte-identical to
// EncodeNamedScrollArgs (the base block, with NOTHING appended).
//
//	bit2 (namedScrollOrder):  the shared order_by block (appendScrollOrderBlock)
//	  follows AFTER the cursor + opts blocks.
const (
	NamedScrollCursor    uint8 = 1 << 0 // [afterID:u64] present
	NamedScrollOpts      uint8 = 1 << 1 // [rc:u8][opa:u8] present
	namedScrollOrder     uint8 = 1 << 2 // shared order_by block present
	namedScrollStaleness uint8 = 1 << 3 // 8-byte big-endian bound present (after rc/opa, before order)
)

// EncodeNamedScrollArgsCursor serializes a vector_named_scroll request with an
// optional resume-after-id cursor (the Task-2 named fan-out passes the SAME
// global afterID to every partition). It appends a self-delimiting trailer:
//
//	[marker:u8=NamedScrollCursor][afterID:u64 BE]
//
// appended ONLY when hasAfter — so the no-cursor default is byte-identical to
// EncodeNamedScrollArgs (backward-compatible: the plain DecodeNamedScrollArgs
// ignores any trailing bytes). See DecodeNamedScrollArgsCursor for the read side.
func EncodeNamedScrollArgsCursor(col string, filter vtypes.Filter, limit int, afterID uint64, hasAfter bool) []byte {
	if !hasAfter {
		return EncodeNamedScrollArgs(col, filter, limit) // byte-identical to the legacy form
	}
	return EncodeNamedScrollArgsOpts(col, filter, limit, afterID, hasAfter, 0, 0)
}

// EncodeNamedScrollArgsOpts serializes a vector_named_scroll request carrying an
// optional cursor AND/OR the cross-shard consistency opts pair behind a single
// marker bitfield. Wire when the trailer is present:
//
//	<base block from EncodeNamedScrollArgs (incl. filter)>
//	  [marker:u8]
//	  [afterID:u64]   ← present when marker&NamedScrollCursor
//	  [rc:u8][opa:u8] ← present when marker&NamedScrollOpts
//
// When neither a cursor nor non-zero rc/opa is present the trailer is omitted
// entirely and the output is BYTE-IDENTICAL to EncodeNamedScrollArgs. When only
// the cursor is present the trailer is "[NamedScrollCursor][afterID]" —
// byte-identical to the legacy cursor encoding.
func EncodeNamedScrollArgsOpts(col string, filter vtypes.Filter, limit int, afterID uint64, hasAfter bool, readConsistency, onPartitionUnavailable uint8) []byte {
	return EncodeNamedScrollArgsOptsBounded(col, filter, limit, afterID, hasAfter, readConsistency, onPartitionUnavailable, 0)
}

// EncodeNamedScrollArgsOptsBounded is EncodeNamedScrollArgsOpts plus the optional
// 8-byte staleness bound, which rides behind the namedScrollStaleness marker bit
// (after [rc][opa], before any order block) ONLY when rc==ConsistencyBoundedStaleness.
// Byte-identical for rc∈{0,1,2}.
func EncodeNamedScrollArgsOptsBounded(col string, filter vtypes.Filter, limit int, afterID uint64, hasAfter bool, readConsistency, onPartitionUnavailable uint8, bound uint64) []byte {
	base := EncodeNamedScrollArgs(col, filter, limit)
	var marker uint8
	if hasAfter {
		marker |= NamedScrollCursor
	}
	if readConsistency != 0 || onPartitionUnavailable != 0 {
		marker |= NamedScrollOpts
	}
	if readConsistency == ConsistencyBoundedStaleness {
		marker |= namedScrollStaleness
	}
	if marker == 0 {
		return base // byte-identical to the legacy / no-trailer form
	}
	out := append(base, marker)
	if marker&NamedScrollCursor != 0 {
		var idb [8]byte
		binary.BigEndian.PutUint64(idb[:], afterID)
		out = append(out, idb[:]...)
	}
	if marker&NamedScrollOpts != 0 {
		out = append(out, readConsistency, onPartitionUnavailable)
	}
	if marker&namedScrollStaleness != 0 {
		var b [8]byte
		binary.BigEndian.PutUint64(b[:], bound)
		out = append(out, b[:]...)
	}
	return out
}

// EncodeNamedScrollArgsOrder is EncodeNamedScrollArgsOpts with an ADDITIVE order_by
// block (the SAME shared block the dense codec uses, appendScrollOrderBlock). When
// order == nil it is byte-identical to EncodeNamedScrollArgsOpts (the no-order_by wire
// is zero-overhead). When order != nil the namedScrollOrder marker bit is set and the
// order block is appended after the cursor + opts blocks. The cursor (afterID) +
// opts ride the existing marker bits. Mirrors EncodeScrollArgsOrder.
func EncodeNamedScrollArgsOrder(col string, filter vtypes.Filter, limit int, afterID uint64, hasAfter bool, readConsistency, onPartitionUnavailable uint8, order *ScrollOrder) []byte {
	return EncodeNamedScrollArgsOrderBounded(col, filter, limit, afterID, hasAfter, readConsistency, onPartitionUnavailable, order, 0)
}

// EncodeNamedScrollArgsOrderBounded is EncodeNamedScrollArgsOrder plus the optional
// 8-byte staleness bound, which rides behind the namedScrollStaleness marker bit
// (after [rc][opa], BEFORE the order block) ONLY when rc==ConsistencyBoundedStaleness.
func EncodeNamedScrollArgsOrderBounded(col string, filter vtypes.Filter, limit int, afterID uint64, hasAfter bool, readConsistency, onPartitionUnavailable uint8, order *ScrollOrder, bound uint64) []byte {
	if order == nil {
		return EncodeNamedScrollArgsOptsBounded(col, filter, limit, afterID, hasAfter, readConsistency, onPartitionUnavailable, bound)
	}
	base := EncodeNamedScrollArgs(col, filter, limit)
	marker := namedScrollOrder
	if hasAfter {
		marker |= NamedScrollCursor
	}
	if readConsistency != 0 || onPartitionUnavailable != 0 {
		marker |= NamedScrollOpts
	}
	if readConsistency == ConsistencyBoundedStaleness {
		marker |= namedScrollStaleness
	}
	out := append(base, marker)
	if marker&NamedScrollCursor != 0 {
		var idb [8]byte
		binary.BigEndian.PutUint64(idb[:], afterID)
		out = append(out, idb[:]...)
	}
	if marker&NamedScrollOpts != 0 {
		out = append(out, readConsistency, onPartitionUnavailable)
	}
	if marker&namedScrollStaleness != 0 {
		var b [8]byte
		binary.BigEndian.PutUint64(b[:], bound)
		out = append(out, b[:]...)
	}
	return appendScrollOrderBlock(out, order)
}

// DecodeNamedScrollArgsOrder decodes a vector_named_scroll request that MAY carry the
// additive order_by block written by EncodeNamedScrollArgsOrder. It is a superset of
// DecodeNamedScrollArgsOpts: the same base + marker (cursor/opts) trailer, then (if
// the namedScrollOrder bit is set) the shared order block. order is nil for legacy
// args / order==nil at encode time. Mirrors DecodeScrollArgsOrder.
func DecodeNamedScrollArgsOrder(args []byte) (col string, filter vtypes.Filter, limit int, afterID uint64, hasAfter bool, readConsistency, onPartitionUnavailable uint8, order *ScrollOrder, err error) {
	col, filter, limit, n, err := decodeNamedScrollArgsN(args)
	if err != nil {
		return "", vtypes.Filter{}, 0, 0, false, 0, 0, nil, err
	}
	if len(args) <= n || args[n] == 0 {
		return col, filter, limit, 0, false, 0, 0, nil, nil
	}
	marker := args[n]
	off := n + 1
	if marker&NamedScrollCursor != 0 {
		if len(args) < off+8 {
			return "", vtypes.Filter{}, 0, 0, false, 0, 0, nil, ErrVectorArgsTruncated
		}
		afterID = binary.BigEndian.Uint64(args[off:])
		hasAfter = true
		off += 8
	}
	if marker&NamedScrollOpts != 0 {
		if len(args) < off+2 {
			return "", vtypes.Filter{}, 0, 0, false, 0, 0, nil, ErrVectorArgsTruncated
		}
		readConsistency = args[off]
		onPartitionUnavailable = args[off+1]
		off += 2
	}
	if marker&namedScrollStaleness != 0 {
		// Consume the 8-byte bound (after rc/opa, before the order block) so the
		// order block decode is correctly positioned.
		if len(args) < off+8 {
			return "", vtypes.Filter{}, 0, 0, false, 0, 0, nil, ErrVectorArgsTruncated
		}
		off += 8
	}
	if marker&namedScrollOrder != 0 {
		order, _, err = readScrollOrderBlock(args, off)
		if err != nil {
			return "", vtypes.Filter{}, 0, 0, false, 0, 0, nil, err
		}
		if order == nil {
			// The marker bit promised an order block; a missing one is corruption.
			return "", vtypes.Filter{}, 0, 0, false, 0, 0, nil, ErrVectorArgsTruncated
		}
	}
	return col, filter, limit, afterID, hasAfter, readConsistency, onPartitionUnavailable, order, nil
}

// DecodeNamedScrollArgsCursor decodes a vector_named_scroll request that may
// carry the optional cursor trailer written by EncodeNamedScrollArgsCursor.
// Backward-compatible: legacy args (no trailer) decode with hasAfter=false. A
// present-flag with a missing/truncated afterID is corruption — fail loud.
func DecodeNamedScrollArgsCursor(args []byte) (col string, filter vtypes.Filter, limit int, afterID uint64, hasAfter bool, err error) {
	col, filter, limit, afterID, hasAfter, _, _, _, err = DecodeNamedScrollArgsOpts(args)
	return col, filter, limit, afterID, hasAfter, err
}

// DecodeNamedScrollArgsOpts decodes a vector_named_scroll request that may carry
// the cursor and/or opts trailer written by EncodeNamedScrollArgsOpts.
// Backward-compatible: legacy args (no trailer) decode with hasAfter=false,
// readConsistency=0, onPartitionUnavailable=0. The legacy "[1][afterID]" cursor
// form decodes identically (marker==NamedScrollCursor). A present marker with a
// missing/truncated block is corruption — fail loud.
func DecodeNamedScrollArgsOpts(args []byte) (col string, filter vtypes.Filter, limit int, afterID uint64, hasAfter bool, readConsistency, onPartitionUnavailable uint8, bound uint64, err error) {
	col, filter, limit, n, err := decodeNamedScrollArgsN(args)
	if err != nil {
		return "", vtypes.Filter{}, 0, 0, false, 0, 0, 0, err
	}
	if len(args) <= n || args[n] == 0 {
		// No trailer (legacy / no-cursor no-opts form). A zero marker is never
		// emitted, so treat a zero byte here as "no trailer" too.
		return col, filter, limit, 0, false, 0, 0, 0, nil
	}
	marker := args[n]
	off := n + 1
	if marker&NamedScrollCursor != 0 {
		if len(args) < off+8 {
			return "", vtypes.Filter{}, 0, 0, false, 0, 0, 0, ErrVectorArgsTruncated
		}
		afterID = binary.BigEndian.Uint64(args[off:])
		hasAfter = true
		off += 8
	}
	if marker&NamedScrollOpts != 0 {
		if len(args) < off+2 {
			return "", vtypes.Filter{}, 0, 0, false, 0, 0, 0, ErrVectorArgsTruncated
		}
		readConsistency = args[off]
		onPartitionUnavailable = args[off+1]
		off += 2
	}
	if marker&namedScrollStaleness != 0 {
		if len(args) < off+8 {
			return "", vtypes.Filter{}, 0, 0, false, 0, 0, 0, ErrVectorArgsTruncated
		}
		bound = binary.BigEndian.Uint64(args[off : off+8])
	}
	return col, filter, limit, afterID, hasAfter, readConsistency, onPartitionUnavailable, bound, nil
}

// EncodeNamedNameArgs serializes a name-only request (drop / get_config).
// Wire: [colLen:u8][col]. Mirrors EncodeMVGetConfigArgs.
func EncodeNamedNameArgs(col string) []byte {
	buf := make([]byte, 1+len(col))
	buf[0] = byte(len(col))
	copy(buf[1:], col)
	return buf
}

// DecodeNamedNameArgs reads args produced by EncodeNamedNameArgs. Trailing bytes
// beyond the base [colLen][col] block (the rc/opa opts trailer added by
// EncodeNamedNameArgsOpts) are IGNORED, so the single-shard handler stays
// backward-compatible with rc-carrying args.
func DecodeNamedNameArgs(args []byte) (string, error) {
	col, _, err := decodeNameArgsN(args)
	return col, err
}

// EncodeNamedNameArgsOpts is EncodeNamedNameArgs plus the self-delimiting
// [marker][rc][opa] opts trailer. Byte-identical to EncodeNamedNameArgs when
// rc==0 && opa==0 (AnyReplica default unchanged); a non-zero rc rides the trailer
// so a Linearizable named get_config arms the shard barrier (via ReadConsistencyOf).
func EncodeNamedNameArgsOpts(col string, readConsistency, onPartitionUnavailable uint8, bound uint64) []byte {
	return AppendReadOptsTrailerBounded(EncodeNamedNameArgs(col), readConsistency, onPartitionUnavailable, bound)
}

// DecodeNamedNameArgsOpts decodes a named-name request that may carry the rc/opa
// (+ bound) opts trailer. Backward-compatible (legacy args ⇒ rc=0,opa=0,bound=0); a
// present marker with a truncated block is corruption — fail loud.
func DecodeNamedNameArgsOpts(args []byte) (col string, readConsistency, onPartitionUnavailable uint8, bound uint64, err error) {
	col, n, err := decodeNameArgsN(args)
	if err != nil {
		return "", 0, 0, 0, err
	}
	readConsistency, onPartitionUnavailable, bound, err = DecodeReadOptsTrailerBounded(args, n)
	if err != nil {
		return "", 0, 0, 0, err
	}
	return col, readConsistency, onPartitionUnavailable, bound, nil
}

// EncodeNamedConfigResult serializes a named collection's config (the
// map[string]NamedVectorParams) as JSON. Wire: [cfgLen:u32][cfgJSON].
func EncodeNamedConfigResult(cfg map[string]vtypes.NamedVectorParams) []byte {
	cfgJSON, _ := json.Marshal(cfg)
	buf := make([]byte, 4+len(cfgJSON))
	binary.BigEndian.PutUint32(buf[0:4], uint32(len(cfgJSON))) //nolint:gosec
	copy(buf[4:], cfgJSON)
	return buf
}

// DecodeNamedConfigResult reads a result produced by EncodeNamedConfigResult.
func DecodeNamedConfigResult(body []byte) (map[string]vtypes.NamedVectorParams, error) {
	if len(body) < 4 {
		return nil, ErrVectorArgsTruncated
	}
	cfgLen := int(binary.BigEndian.Uint32(body[0:4]))
	// A negative cfgLen (32-bit widening) satisfies `len(body) < 4+cfgLen` and
	// then makes body[4:4+cfgLen] a backwards slice.
	if !CountFitsIn(cfgLen, len(body)-4, 1) {
		return nil, ErrVectorArgsTruncated
	}
	var cfg map[string]vtypes.NamedVectorParams
	if cfgLen > 0 {
		if err := json.Unmarshal(body[4:4+cfgLen], &cfg); err != nil {
			return nil, fmt.Errorf("ops: decode named config result: %w", err)
		}
	}
	return cfg, nil
}

// EncodeNamedGetResult serializes a vector_named_get result. Wire:
//
//	[found:u8] then if found==1:
//	  [numSpaces:u32]{[nameLen:u16][name][dim:u32][vec f32×dim]}  ← empty when with_vector off
//	  [ttlMs:u64]
//	  [metaPresent:u8][?metaLen:u32][?metaJSON]   ← metaPresent 0 when no payload / payload off
//	  [verPresent:u8][?version:u64]   ← OPTIONAL trailing CAS version (byte-identical when 0)
//
// not-found is the found=0 FLAG (NEVER an op error). withVector gates the named
// spaces map; withPayload gates the shared payload. Mirrors the dense get result.
// The trailing per-point CAS version block rides behind a present byte (0 ⇒ no
// version field, byte-identical to the pre-version encoding); a legacy decoder
// that stops after the payload tolerates the extra bytes.
func EncodeNamedGetResult(found bool, vectors map[string][]float32, payload vtypes.Metadata, ttl time.Duration, withVector, withPayload bool) []byte {
	return EncodeNamedGetResultV(found, vectors, payload, ttl, withVector, withPayload, 0)
}

// EncodeNamedGetResultV is EncodeNamedGetResult plus the point's per-point CAS
// version. A 0 version writes verPresent=0 and NO version field (byte-identical to
// the pre-version EncodeNamedGetResult); a live version (>=1) writes verPresent=1
// + the u64.
func EncodeNamedGetResultV(found bool, vectors map[string][]float32, payload vtypes.Metadata, ttl time.Duration, withVector, withPayload bool, version uint64) []byte {
	return appendNamedGetResultV(nil, found, vectors, payload, ttl, withVector, withPayload, version)
}

// appendNamedGetResultV appends a named-get-result record (same wire layout as
// EncodeNamedGetResultV) to dst and returns the grown slice. It sizes the record
// up front (the prior single-get encoder under-presized — its capacity hint
// excluded the vector floats, so every named vector forced reallocations) and
// grows dst ONCE, so it is allocation-free when dst already has the capacity —
// what lets the batch encoder serialize many rows into one buffer without a
// throwaway slice per row. EncodeNamedGetResultV is the nil-dst single-get wrapper.
func appendNamedGetResultV(dst []byte, found bool, vectors map[string][]float32, payload vtypes.Metadata, ttl time.Duration, withVector, withPayload bool, version uint64) []byte {
	if !found {
		return append(dst, 0)
	}
	var metaJSON []byte
	if withPayload && len(payload) > 0 {
		metaJSON, _ = json.Marshal(payload)
	}
	n := 1 + 4 // found + numVectors
	if withVector {
		for name, vec := range vectors {
			n += 2 + len(name) + 4 + 4*len(vec)
		}
	}
	n += 8 + 1 // ttl + metaPresent
	if len(metaJSON) > 0 {
		n += 4 + len(metaJSON)
	}
	n++ // verPresent
	if version != 0 {
		n += 8
	}
	dst = slices.Grow(dst, n) // single growth; appends below never reallocate
	dst = append(dst, 1)
	var u32 [4]byte
	var u16 [2]byte
	var u64 [8]byte
	if withVector {
		binary.BigEndian.PutUint32(u32[:], uint32(len(vectors))) //nolint:gosec
		dst = append(dst, u32[:]...)
		for name, vec := range vectors {
			binary.BigEndian.PutUint16(u16[:], uint16(len(name))) //nolint:gosec
			dst = append(dst, u16[:]...)
			dst = append(dst, name...)
			binary.BigEndian.PutUint32(u32[:], uint32(len(vec))) //nolint:gosec
			dst = append(dst, u32[:]...)
			for _, f := range vec {
				binary.BigEndian.PutUint32(u32[:], math.Float32bits(f))
				dst = append(dst, u32[:]...)
			}
		}
	} else {
		binary.BigEndian.PutUint32(u32[:], 0)
		dst = append(dst, u32[:]...)
	}
	binary.BigEndian.PutUint64(u64[:], uint64(ttl.Milliseconds())) //nolint:gosec // TTL >= 0
	dst = append(dst, u64[:]...)
	if len(metaJSON) > 0 {
		dst = append(dst, 1)
		binary.BigEndian.PutUint32(u32[:], uint32(len(metaJSON))) //nolint:gosec
		dst = append(dst, u32[:]...)
		dst = append(dst, metaJSON...)
	} else {
		dst = append(dst, 0)
	}
	// Trailing per-point CAS version block (byte-identical when 0).
	if version != 0 {
		dst = append(dst, 1)
		binary.BigEndian.PutUint64(u64[:], version)
		dst = append(dst, u64[:]...)
	} else {
		dst = append(dst, 0)
	}
	return dst
}

// DecodeNamedGetResult reads a result produced by EncodeNamedGetResult. found is
// false for an absent/expired point (the not-found flag).
func DecodeNamedGetResult(body []byte) (found bool, vectors map[string][]float32, payload vtypes.Metadata, ttl time.Duration, err error) {
	found, vectors, payload, ttlMs, _, _, err := DecodeNamedGetResultAt(body, 0)
	return found, vectors, payload, time.Duration(ttlMs) * time.Millisecond, err
}

// DecodeNamedGetResultV is DecodeNamedGetResult plus the point's per-point CAS
// version (0 for an absent point or a legacy result with no version block).
func DecodeNamedGetResultV(body []byte) (found bool, vectors map[string][]float32, payload vtypes.Metadata, ttl time.Duration, version uint64, err error) {
	found, vectors, payload, ttlMs, version, _, err := DecodeNamedGetResultAt(body, 0)
	return found, vectors, payload, time.Duration(ttlMs) * time.Millisecond, version, err
}

// DecodeNamedGetResultAt decodes a single named-get-result record (the
// [found:u8]+body shape produced by EncodeNamedGetResult) starting at body[off],
// returning the decoded fields, the raw ttl in milliseconds, and the offset just
// past the record. Shared by DecodeNamedGetResult (the single-get path) and the
// named batch result decoder (which reads one such record per row after the
// row's id) so the per-row wire layout stays defined in one place. Fails loud on
// truncation, exactly like the original single-get decoder.
func DecodeNamedGetResultAt(body []byte, off int) (found bool, vectors map[string][]float32, meta vtypes.Metadata, ttlMs uint64, version uint64, next int, err error) {
	if len(body) < off+1 {
		return false, nil, nil, 0, 0, off, ErrVectorArgsTruncated
	}
	if body[off] == 0 {
		return false, nil, nil, 0, 0, off + 1, nil
	}
	off++
	if len(body) < off+4 {
		return false, nil, nil, 0, 0, off, ErrVectorArgsTruncated
	}
	numSpaces := int(binary.BigEndian.Uint32(body[off:]))
	off += 4
	// Bound the DECLARED space count before it sizes the map reservation: the
	// smallest a space can encode is 6 bytes ([nameLen:u16] with an empty name +
	// [dim:u32] with dim 0, per the loop below), so no body of this length carries
	// more than (remaining)/6 spaces. Without the bound a hostile count reserves a
	// map the body cannot justify, and the per-space truncation checks run only
	// AFTER the reservation.
	if !CountFitsIn(numSpaces, len(body)-off, 6) {
		return false, nil, nil, 0, 0, off, ErrVectorArgsTruncated
	}
	if numSpaces > 0 {
		vectors = make(map[string][]float32, numSpaces)
	}
	for i := 0; i < numSpaces; i++ {
		if len(body) < off+2 {
			return false, nil, nil, 0, 0, off, ErrVectorArgsTruncated
		}
		nameLen := int(binary.BigEndian.Uint16(body[off:]))
		off += 2
		if len(body) < off+nameLen+4 {
			return false, nil, nil, 0, 0, off, ErrVectorArgsTruncated
		}
		name := string(body[off : off+nameLen])
		off += nameLen
		dim := int(binary.BigEndian.Uint32(body[off:]))
		off += 4
		if len(body) < off+4*dim {
			return false, nil, nil, 0, 0, off, ErrVectorArgsTruncated
		}
		vec := make([]float32, dim)
		for j := 0; j < dim; j++ {
			vec[j] = math.Float32frombits(binary.BigEndian.Uint32(body[off:]))
			off += 4
		}
		vectors[name] = vec
	}
	if len(body) < off+8+1 {
		return false, nil, nil, 0, 0, off, ErrVectorArgsTruncated
	}
	ttlMs = binary.BigEndian.Uint64(body[off:])
	off += 8
	metaPresent := body[off]
	off++
	if metaPresent == 1 {
		if len(body) < off+4 {
			return false, nil, nil, 0, 0, off, ErrVectorArgsTruncated
		}
		mlen := int(binary.BigEndian.Uint32(body[off:]))
		off += 4
		if len(body) < off+mlen {
			return false, nil, nil, 0, 0, off, ErrVectorArgsTruncated
		}
		m := make(vtypes.Metadata)
		if err := json.Unmarshal(body[off:off+mlen], &m); err != nil {
			return false, nil, nil, 0, 0, off, fmt.Errorf("ops: decode named get payload: %w", err)
		}
		meta = m
		off += mlen
	}
	// Optional trailing per-point CAS version block. A legacy result (no version
	// block) ends right after the payload; treat the absence of the present byte as
	// "no version" (back-compat). A present byte == 1 carries the u64.
	if off < len(body) {
		verPresent := body[off]
		off++
		if verPresent == 1 {
			if len(body) < off+8 {
				return false, nil, nil, 0, 0, off, ErrVectorArgsTruncated
			}
			version = binary.BigEndian.Uint64(body[off:])
			off += 8
		}
	}
	return true, vectors, meta, ttlMs, version, off, nil
}

// NamedGetBatchRow is one row of a vector_named_get_batch result: the requested
// id plus the same projected fields a single vector_named_get carries (Found is
// the not-found FLAG, never an error). For a not-found id only ID/Found are
// meaningful. Vectors/Meta follow the with_vector/with_payload projection applied
// at fetch time; TTLMs is the remaining TTL in milliseconds. Mirrors the dense
// GetBatchRow — the named row carries a per-space vectors MAP (+ ttl) instead of
// a single dense vector (and has no sparse lane).
type NamedGetBatchRow struct {
	ID      uint64
	Found   bool
	Vectors map[string][]float32
	Meta    vtypes.Metadata
	TTLMs   uint64
	Version uint64 // per-point CAS version (>=1 for a found point; 0 = absent/unknown)
}

// EncodeNamedGetBatchResult serializes a per-partition vector_named_get_batch
// result. Wire: [n:u32] then for each row [id:u64] followed by the SAME
// [found:u8]+body record EncodeNamedGetResult produces (so a batch row is just id
// + a single named-get result). Rows preserve the order the handler was given. A
// not-found id is a found=0 record (NEVER an op error) — the coordinator derives
// the global missing set from absent ids. Mirrors EncodeVectorGetBatchResult.
func EncodeNamedGetBatchResult(rows []NamedGetBatchRow) []byte {
	// Presize: header + per-row (id + record); meta JSON is the only unsized field
	// (small). Collapses the old ~log(rows) reallocations + a throwaway per-row
	// slice into one allocation in the common case.
	est := 4
	for i := range rows {
		est += 8 + estimateNamedGetRowSize(&rows[i])
	}
	buf := slices.Grow([]byte(nil), est)
	buf = append(buf, 0, 0, 0, 0)
	binary.BigEndian.PutUint32(buf, uint32(len(rows))) //nolint:gosec
	var idbuf [8]byte
	for i := range rows {
		binary.BigEndian.PutUint64(idbuf[:], rows[i].ID)
		buf = append(buf, idbuf[:]...)
		// EncodeNamedGetResultV always frames the version present byte, so the
		// record self-delimits; append it directly (no intermediate slice).
		buf = appendNamedGetResultV(buf, rows[i].Found, rows[i].Vectors, rows[i].Meta,
			time.Duration(rows[i].TTLMs)*time.Millisecond, true, true, rows[i].Version)
	}
	return buf
}

// estimateNamedGetRowSize returns a near-exact estimate of a batch row's encoded
// size (every field except the meta JSON, unknown without marshaling). Presize only.
func estimateNamedGetRowSize(r *NamedGetBatchRow) int {
	if !r.Found {
		return 1
	}
	n := 1 + 4 + 8 + 1 + 1 + 8 // found, numVectors, ttl, metaPresent, verPresent, version
	for name, vec := range r.Vectors {
		n += 2 + len(name) + 4 + 4*len(vec)
	}
	return n
}

// DecodeNamedGetBatchResult reads a result produced by EncodeNamedGetBatchResult.
// Fails loud on truncation or a declared row count that overruns the buffer. A
// zero-row result yields an empty (non-nil) slice. Each row's projected fields are
// exactly what the encoder carried. Mirrors DecodeVectorGetBatchResult.
func DecodeNamedGetBatchResult(body []byte) ([]NamedGetBatchRow, error) {
	if len(body) < 4 {
		return nil, ErrVectorArgsTruncated
	}
	n := int(binary.BigEndian.Uint32(body))
	off := 4
	// Bound the declared count before allocating: each row carries at least an
	// [id:u64] = 8 bytes, so a truncated/hostile body cannot force an oversized
	// capacity reservation ahead of the per-row truncation checks below.
	if n < 0 || n > (len(body)-4)/8 {
		return nil, ErrVectorArgsTruncated
	}
	rows := make([]NamedGetBatchRow, 0, n)
	for i := 0; i < n; i++ {
		if len(body) < off+8 {
			return nil, ErrVectorArgsTruncated
		}
		id := binary.BigEndian.Uint64(body[off:])
		off += 8
		found, vectors, meta, ttlMs, version, next, err := DecodeNamedGetResultAt(body, off)
		if err != nil {
			return nil, err
		}
		off = next
		rows = append(rows, NamedGetBatchRow{
			ID:      id,
			Found:   found,
			Vectors: vectors,
			Meta:    meta,
			TTLMs:   ttlMs,
			Version: version,
		})
	}
	return rows, nil
}
