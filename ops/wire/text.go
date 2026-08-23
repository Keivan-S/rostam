// SPDX-License-Identifier: Apache-2.0

package wire

import (
	"encoding/binary"
	"encoding/json"
	"fmt"
	"math"
	"sort"

	"github.com/rostamlabs/rostam/vtypes"
)

// Full-text (BM25) search op codecs. These mirror the dense/hybrid search wire
// (ops/vector.go) but carry the RAW query TEXT (a string the server tokenizes +
// BM25-scores) instead of a query vector / sparse query. The SDK sends NO tokens:
// the analyzer lives server-side, so the client ships the raw query text and the
// shard analyzes it. Two ops:
//
//   - vector_search_text:        [flags][colLen][col][k][queryLen][query](+filter)(+opts)
//   - vector_hybrid_text(_lanes): dense + raw text, fused server-side.
//
// The collection name sits at offset 1 (behind the flags byte) in BOTH layouts —
// the At2 routing layout (VectorKeyColAt2), IDENTICAL to vector_hybrid_search — so
// the CollectionNameFor offset table and the routing key extractor are reused
// verbatim. DOUBLE-CHECKED: flags is args[0], colLen is args[1], col is
// args[2:2+colLen] — exactly what VectorKeyColAt2 reads.
//
// SHARDED-IDF / global-DF: by default each shard scores the text query against ITS
// OWN local corpus stats (n/df/avgdl), so partitioned BM25 scores are APPROXIMATE vs
// a single-node corpus — exactly Elasticsearch's default query_then_fetch. The
// opt-in global-DF two-phase (dfs_query_then_fetch) removes that approximation: the
// caller sets textFlagGlobalIDF and the coordinator ships per-query global stats in
// the textFlagGlobalStats block (see appendGlobalStatsBlock / text_fanout.go).

const (
	textFlagFilter      uint8 = 1 << 0 // filter JSON present
	textFlagOpts        uint8 = 1 << 1 // consistency opts trailer present
	textFlagGlobalIDF   uint8 = 1 << 2 // REQUEST flag: caller wants global-DF (dfs) two-phase scoring
	textFlagGlobalStats uint8 = 1 << 3 // PHASE-1 block: coordinator-supplied global stats follow the opts trailer
)

// Shared wire-layout widths for the corpus-stats codecs, so the encode / size /
// decode sites cannot drift. A df pair is [termID:u32, df:u32]. The global-stats
// block header is [gN:i64, gAvgdl:f32, dfCount:u32]; the bm25_stats result header is
// [n:i64, tokenTotal:u64, dfCount:u32].
const (
	dfPairLen          = 4 + 4     // termID:u32 + df:u32
	globalStatsHdrLen  = 8 + 4 + 4 // gN:i64 + gAvgdl:f32 + dfCount:u32
	bm25StatsResultHdr = 8 + 8 + 4 // n:i64 + tokenTotal:u64 + dfCount:u32
)

// appendGlobalStatsBlock appends the optional global-stats block to buf and sets
// textFlagGlobalStats in *flags when g is non-nil. The block rides AFTER the
// rc/opa/bound opts trailer so ReadConsistencyOf's byte-peek (which decodes the
// base + opts trailer at offset n) is unaffected. Layout:
//
//	[gN:i64][gAvgdl:f32][dfCount:u32]{termID:u32, gDF:u32}*
//
// When g==nil the buffer is returned unchanged and the flag bit is NOT set, so a
// no-stats encoding is byte-identical to the pre-global form.
func appendGlobalStatsBlock(buf []byte, flags *uint8, g *vtypes.BM25GlobalStats) []byte {
	if g == nil {
		return buf
	}
	*flags |= textFlagGlobalStats
	buf = binary.BigEndian.AppendUint64(buf, uint64(int64(g.N)))
	buf = binary.BigEndian.AppendUint32(buf, math.Float32bits(g.Avgdl))
	buf = binary.BigEndian.AppendUint32(buf, uint32(len(g.DF)))
	// Serialize in ascending term-id order so the block is byte-canonical (Go map
	// iteration is randomized); the decoder is order-independent regardless.
	for _, term := range sortedDFKeys(g.DF) {
		buf = binary.BigEndian.AppendUint32(buf, term)
		buf = binary.BigEndian.AppendUint32(buf, uint32(g.DF[term]))
	}
	return buf
}

// sortedDFKeys returns df's term ids in ascending order, so the corpus-stats
// codecs emit a deterministic (byte-canonical) wire form independent of Go's
// randomized map iteration.
func sortedDFKeys(df map[uint32]int) []uint32 {
	keys := make([]uint32, 0, len(df))
	for term := range df {
		keys = append(keys, term)
	}
	sort.Slice(keys, func(i, j int) bool { return keys[i] < keys[j] })
	return keys
}

// globalStatsBlockLen returns the byte length the global-stats block for g
// occupies (0 when g==nil) so encoders can pre-size their buffers.
func globalStatsBlockLen(g *vtypes.BM25GlobalStats) int {
	if g == nil {
		return 0
	}
	return globalStatsHdrLen + dfPairLen*len(g.DF)
}

// readGlobalStatsBlock decodes the optional global-stats block at offset off when
// flags has textFlagGlobalStats set. It returns nil (and the unchanged offset)
// when the bit is clear, so a no-stats body decodes to a nil *BM25GlobalStats. A
// set bit with a truncated block is corruption — fail loud.
func readGlobalStatsBlock(args []byte, off int, flags uint8) (g *vtypes.BM25GlobalStats, newOff int, err error) {
	if flags&textFlagGlobalStats == 0 {
		return nil, off, nil
	}
	if len(args) < off+globalStatsHdrLen {
		return nil, off, ErrVectorArgsTruncated
	}
	n := int(int64(binary.BigEndian.Uint64(args[off:])))
	off += 8
	avgdl := math.Float32frombits(binary.BigEndian.Uint32(args[off:]))
	off += 4
	cnt := int(binary.BigEndian.Uint32(args[off:]))
	off += 4
	// dfPairLen*cnt is safe on a 64-bit int (one wire factor times a constant), but
	// express it as the shared bound anyway: the same reasoning in one place, and no
	// reliance on int being 64 bits.
	if !CountFitsIn(cnt, len(args)-off, dfPairLen) {
		return nil, off, ErrVectorArgsTruncated
	}
	df := make(map[uint32]int, cnt)
	for i := 0; i < cnt; i++ {
		term := binary.BigEndian.Uint32(args[off:])
		off += 4
		d := int(binary.BigEndian.Uint32(args[off:]))
		off += 4
		df[term] = d
	}
	return &vtypes.BM25GlobalStats{N: n, Avgdl: avgdl, DF: df}, off, nil
}

// EncodeSearchTextArgs serializes vector_search_text args. Wire:
//
//	[flags:u8]                  bit0=HAS_FILTER, bit1=HAS_OPTS
//	[colLen:u8][col]
//	[k:u32]
//	[queryLen:u32][query: raw UTF-8 text]
//	if HAS_FILTER: [filterLen:u32][filterJSON]
func EncodeSearchTextArgs(collection string, query string, k int, filter vtypes.Filter) []byte {
	return EncodeSearchTextArgsOpts(collection, query, k, filter, 0, 0, 0)
}

// EncodeSearchTextArgsOpts serializes vector_search_text args with an optional
// cross-shard consistency opts trailer. When readConsistency and
// onPartitionUnavailable are both zero the trailer is omitted (byte-identical to
// the no-opts form). The 8-byte staleness bound rides ONLY when
// readConsistency==ConsistencyBoundedStaleness.
func EncodeSearchTextArgsOpts(collection string, query string, k int, filter vtypes.Filter, readConsistency, onPartitionUnavailable uint8, bound uint64) []byte {
	return EncodeSearchTextArgsGlobal(collection, query, k, filter, readConsistency, onPartitionUnavailable, bound, false, nil)
}

// EncodeSearchTextArgsGlobal serializes vector_search_text args with the optional
// rc/opa/bound trailer PLUS two additive global-DF (dfs) extensions, mirroring the
// additive-trailer pattern so the wire is byte-identical when both are absent:
//
//   - globalIDF (REQUEST flag): set by the client/coordinator request to ask the
//     server-side coordinator to run the two-phase global-DF search. It rides as a
//     single flags bit (textFlagGlobalIDF) — NO extra bytes — so a false flag is
//     byte-identical to the pre-global form.
//   - g (PHASE-1 stats): when non-nil the coordinator's phase-1 fan-out appends the
//     global-stats block (gN/gAvgdl/gDF) AFTER the rc/opa/bound trailer, behind
//     textFlagGlobalStats. The decode handler then routes scoring through the
//     *Global path. When nil the block is absent (byte-identical).
//
// The two are independent: a request carries globalIDF=true with g=nil; a phase-1
// scatter carries g!=nil (and may leave globalIDF unset — the shard only needs the
// stats). Placing the stats block AFTER the opts trailer keeps ReadConsistencyOf's
// byte-peek (which decodes base+opts at offset n) correct.
func EncodeSearchTextArgsGlobal(collection string, query string, k int, filter vtypes.Filter, readConsistency, onPartitionUnavailable uint8, bound uint64, globalIDF bool, g *vtypes.BM25GlobalStats) []byte {
	var flags uint8
	var filterJSON []byte
	if !filter.IsZero() {
		flags |= textFlagFilter
		filterJSON, _ = json.Marshal(filter)
	}
	hasOpts := readConsistency != 0 || onPartitionUnavailable != 0
	if hasOpts {
		flags |= textFlagOpts
	}
	if globalIDF {
		flags |= textFlagGlobalIDF
	}

	n := 1 + 1 + len(collection) + 4 + 4 + len(query)
	if flags&textFlagFilter != 0 {
		n += 4 + len(filterJSON)
	}
	if hasOpts {
		n += 2
	}
	n += globalStatsBlockLen(g)
	buf := make([]byte, 0, n)
	buf = append(buf, flags)
	buf = append(buf, byte(len(collection)))
	buf = append(buf, collection...)
	buf = binary.BigEndian.AppendUint32(buf, uint32(k))
	buf = binary.BigEndian.AppendUint32(buf, uint32(len(query)))
	buf = append(buf, query...)
	if flags&textFlagFilter != 0 {
		buf = binary.BigEndian.AppendUint32(buf, uint32(len(filterJSON)))
		buf = append(buf, filterJSON...)
	}
	if hasOpts {
		buf = append(buf, readConsistency, onPartitionUnavailable)
		buf = appendBoundTail(buf, readConsistency, bound)
	}
	buf = appendGlobalStatsBlock(buf, &flags, g)
	buf[0] = flags
	return buf
}

// DecodeSearchTextArgs reads vector_search_text args (ignoring any opts trailer).
func DecodeSearchTextArgs(args []byte) (collection string, query string, k int, filter vtypes.Filter, err error) {
	collection, query, k, filter, _, err = decodeSearchTextArgsN(args)
	return collection, query, k, filter, err
}

// decodeSearchTextArgsN decodes the base block and returns the number of bytes
// consumed so DecodeSearchTextArgsOpts can read a trailing opts block.
func decodeSearchTextArgsN(args []byte) (collection string, query string, k int, filter vtypes.Filter, n int, err error) {
	if len(args) < 2 {
		return "", "", 0, filter, 0, ErrVectorArgsTruncated
	}
	flags := args[0]
	colLen := int(args[1])
	if len(args) < 2+colLen+4+4 {
		return "", "", 0, filter, 0, ErrVectorArgsTruncated
	}
	collection = string(args[2 : 2+colLen])
	off := 2 + colLen
	k = int(binary.BigEndian.Uint32(args[off:]))
	off += 4
	qLen := int(binary.BigEndian.Uint32(args[off:]))
	off += 4
	// A negative qLen (32-bit widening) satisfies this comparison and then makes
	// args[off : off+qLen] a backwards slice. CountFitsIn rejects the sign; the
	// element floor is 1 because a query is a byte string.
	if !CountFitsIn(qLen, len(args)-off, 1) {
		return "", "", 0, filter, 0, ErrVectorArgsTruncated
	}
	query = string(args[off : off+qLen])
	off += qLen
	if flags&textFlagFilter != 0 {
		if len(args) < off+4 {
			return "", "", 0, filter, 0, ErrVectorArgsTruncated
		}
		flen := int(binary.BigEndian.Uint32(args[off:]))
		off += 4
		if len(args) < off+flen {
			return "", "", 0, filter, 0, ErrVectorArgsTruncated
		}
		if uerr := json.Unmarshal(args[off:off+flen], &filter); uerr != nil {
			return "", "", 0, filter, 0, fmt.Errorf("ops: decode filter: %w", uerr)
		}
		off += flen
	}
	return collection, query, k, filter, off, nil
}

// DecodeSearchTextArgsOpts decodes vector_search_text args that may carry a
// cross-shard consistency opts trailer (textFlagOpts). Backward-compatible: args
// without the trailer decode with readConsistency=0, onPartitionUnavailable=0.
func DecodeSearchTextArgsOpts(args []byte) (collection string, query string, k int, filter vtypes.Filter, readConsistency, onPartitionUnavailable uint8, bound uint64, err error) {
	collection, query, k, filter, readConsistency, onPartitionUnavailable, bound, _, _, err = DecodeSearchTextArgsGlobal(args)
	return collection, query, k, filter, readConsistency, onPartitionUnavailable, bound, err
}

// DecodeSearchTextArgsGlobal decodes vector_search_text args including the optional
// rc/opa/bound trailer AND the global-DF extensions: globalIDF (the request flag,
// textFlagGlobalIDF) and g (the phase-1 global stats, nil when absent). Backward-
// compatible: a pre-global body yields globalIDF=false, g=nil.
func DecodeSearchTextArgsGlobal(args []byte) (collection string, query string, k int, filter vtypes.Filter, readConsistency, onPartitionUnavailable uint8, bound uint64, globalIDF bool, g *vtypes.BM25GlobalStats, err error) {
	collection, query, k, filter, n, err := decodeSearchTextArgsN(args)
	if err != nil {
		return "", "", 0, filter, 0, 0, 0, false, nil, err
	}
	flags := byte(0)
	if len(args) > 0 {
		flags = args[0]
	}
	off := n
	if flags&textFlagOpts != 0 {
		if len(args) < n+2 {
			return "", "", 0, filter, 0, 0, 0, false, nil, ErrVectorArgsTruncated
		}
		readConsistency = args[n]
		onPartitionUnavailable = args[n+1]
		bound, off, err = readBoundTail(args, n+2, readConsistency)
		if err != nil {
			return "", "", 0, filter, 0, 0, 0, false, nil, err
		}
	}
	globalIDF = flags&textFlagGlobalIDF != 0
	g, _, err = readGlobalStatsBlock(args, off, flags)
	if err != nil {
		return "", "", 0, filter, 0, 0, 0, false, nil, err
	}
	return collection, query, k, filter, readConsistency, onPartitionUnavailable, bound, globalIDF, g, nil
}

// EncodeHybridTextArgs serializes vector_hybrid_text / vector_hybrid_text_lanes
// args. Wire:
//
//	[flags:u8]                  bit0=HAS_FILTER, bit1=HAS_OPTS
//	[colLen:u8][col]
//	[k:u32]
//	[method:u8][alpha:f64][rrfK:u32][denseK:u32][sparseK:u32]
//	[dim:u32][dense: f32×dim]
//	[queryLen:u32][query: raw UTF-8 text]
//	if HAS_FILTER: [filterLen:u32][filterJSON]
//
// The text lane is BM25 (no sparse query vector rides the wire: the server
// analyzes the raw text). opts carries fusion knobs identical to HybridOpts.
func EncodeHybridTextArgs(collection string, dense []float32, query string, k int, opts vtypes.HybridOpts) []byte {
	return EncodeHybridTextArgsOpts(collection, dense, query, k, opts, 0, 0, 0)
}

// EncodeHybridTextArgsOpts serializes vector_hybrid_text args with an optional
// cross-shard consistency opts trailer (byte-identical to EncodeHybridTextArgs
// when readConsistency and onPartitionUnavailable are both zero).
func EncodeHybridTextArgsOpts(collection string, dense []float32, query string, k int, opts vtypes.HybridOpts, readConsistency, onPartitionUnavailable uint8, bound uint64) []byte {
	return EncodeHybridTextArgsGlobal(collection, dense, query, k, opts, readConsistency, onPartitionUnavailable, bound, false, nil)
}

// EncodeHybridTextArgsGlobal is EncodeHybridTextArgsOpts with the two additive
// global-DF (dfs) extensions (byte-identical to the no-global form when globalIDF
// is false and g is nil): the request flag textFlagGlobalIDF, and the phase-1
// global-stats block (behind textFlagGlobalStats) appended AFTER the rc/opa/bound
// trailer. See EncodeSearchTextArgsGlobal for the rationale.
func EncodeHybridTextArgsGlobal(collection string, dense []float32, query string, k int, opts vtypes.HybridOpts, readConsistency, onPartitionUnavailable uint8, bound uint64, globalIDF bool, g *vtypes.BM25GlobalStats) []byte {
	var flags uint8
	var filterJSON []byte
	if !opts.Filter.IsZero() {
		flags |= textFlagFilter
		filterJSON, _ = json.Marshal(opts.Filter)
	}
	hasOpts := readConsistency != 0 || onPartitionUnavailable != 0
	if hasOpts {
		flags |= textFlagOpts
	}
	if globalIDF {
		flags |= textFlagGlobalIDF
	}

	n := 1 + 1 + len(collection) + 4 + (1 + 8 + 4 + 4 + 4) + 4 + 4*len(dense) + 4 + len(query)
	if flags&textFlagFilter != 0 {
		n += 4 + len(filterJSON)
	}
	if hasOpts {
		n += 2
	}
	n += globalStatsBlockLen(g)
	buf := make([]byte, 0, n)
	buf = append(buf, flags)
	buf = append(buf, byte(len(collection)))
	buf = append(buf, collection...)
	buf = binary.BigEndian.AppendUint32(buf, uint32(k))
	buf = append(buf, byte(opts.Method))
	buf = binary.BigEndian.AppendUint64(buf, math.Float64bits(opts.Alpha))
	buf = binary.BigEndian.AppendUint32(buf, uint32(opts.RRFK))
	buf = binary.BigEndian.AppendUint32(buf, uint32(opts.DenseK))
	buf = binary.BigEndian.AppendUint32(buf, uint32(opts.SparseK))
	buf = binary.BigEndian.AppendUint32(buf, uint32(len(dense)))
	for _, f := range dense {
		buf = binary.BigEndian.AppendUint32(buf, math.Float32bits(f))
	}
	buf = binary.BigEndian.AppendUint32(buf, uint32(len(query)))
	buf = append(buf, query...)
	if flags&textFlagFilter != 0 {
		buf = binary.BigEndian.AppendUint32(buf, uint32(len(filterJSON)))
		buf = append(buf, filterJSON...)
	}
	if hasOpts {
		buf = append(buf, readConsistency, onPartitionUnavailable)
		buf = appendBoundTail(buf, readConsistency, bound)
	}
	buf = appendGlobalStatsBlock(buf, &flags, g)
	buf[0] = flags
	return buf
}

// DecodeHybridTextArgs reads vector_hybrid_text args (ignoring any opts trailer).
func DecodeHybridTextArgs(args []byte) (collection string, dense []float32, query string, k int, opts vtypes.HybridOpts, err error) {
	collection, dense, query, k, opts, _, err = decodeHybridTextArgsN(args)
	return collection, dense, query, k, opts, err
}

func decodeHybridTextArgsN(args []byte) (collection string, dense []float32, query string, k int, opts vtypes.HybridOpts, n int, err error) {
	if len(args) < 2 {
		return "", nil, "", 0, opts, 0, ErrVectorArgsTruncated
	}
	flags := args[0]
	colLen := int(args[1])
	// fixed: flags(1)+colLen(1)+col+k(4)+method(1)+alpha(8)+rrfK(4)+denseK(4)+sparseK(4)+dim(4)
	if len(args) < 2+colLen+4+1+8+4+4+4+4 {
		return "", nil, "", 0, opts, 0, ErrVectorArgsTruncated
	}
	collection = string(args[2 : 2+colLen])
	off := 2 + colLen
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
	// See DecodeVectorInsertArgs: 4*dim overflows to 0 for a negative dim, and
	// make([]float32, dim) then panics rather than erroring.
	if !CountFitsIn(dim, len(args)-off, 4) {
		return "", nil, "", 0, opts, 0, ErrVectorArgsTruncated
	}
	dense = make([]float32, dim)
	for i := 0; i < dim; i++ {
		dense[i] = math.Float32frombits(binary.BigEndian.Uint32(args[off:]))
		off += 4
	}
	if len(args) < off+4 {
		return "", nil, "", 0, opts, 0, ErrVectorArgsTruncated
	}
	qLen := int(binary.BigEndian.Uint32(args[off:]))
	off += 4
	if len(args) < off+qLen {
		return "", nil, "", 0, opts, 0, ErrVectorArgsTruncated
	}
	query = string(args[off : off+qLen])
	off += qLen
	if flags&textFlagFilter != 0 {
		if len(args) < off+4 {
			return "", nil, "", 0, opts, 0, ErrVectorArgsTruncated
		}
		flen := int(binary.BigEndian.Uint32(args[off:]))
		off += 4
		if len(args) < off+flen {
			return "", nil, "", 0, opts, 0, ErrVectorArgsTruncated
		}
		if uerr := json.Unmarshal(args[off:off+flen], &opts.Filter); uerr != nil {
			return "", nil, "", 0, opts, 0, fmt.Errorf("ops: decode filter: %w", uerr)
		}
		off += flen
	}
	return collection, dense, query, k, opts, off, nil
}

// DecodeHybridTextArgsOpts decodes vector_hybrid_text args that may carry a
// cross-shard consistency opts trailer (textFlagOpts). Backward-compatible with
// the no-opts form (readConsistency=0, onPartitionUnavailable=0).
func DecodeHybridTextArgsOpts(args []byte) (collection string, dense []float32, query string, k int, opts vtypes.HybridOpts, readConsistency, onPartitionUnavailable uint8, bound uint64, err error) {
	collection, dense, query, k, opts, readConsistency, onPartitionUnavailable, bound, _, _, err = DecodeHybridTextArgsGlobal(args)
	return collection, dense, query, k, opts, readConsistency, onPartitionUnavailable, bound, err
}

// DecodeHybridTextArgsGlobal decodes vector_hybrid_text args including the optional
// rc/opa/bound trailer AND the global-DF extensions: globalIDF (request flag) and g
// (phase-1 global stats, nil when absent). Backward-compatible with the pre-global
// form (globalIDF=false, g=nil).
func DecodeHybridTextArgsGlobal(args []byte) (collection string, dense []float32, query string, k int, opts vtypes.HybridOpts, readConsistency, onPartitionUnavailable uint8, bound uint64, globalIDF bool, g *vtypes.BM25GlobalStats, err error) {
	collection, dense, query, k, opts, n, err := decodeHybridTextArgsN(args)
	if err != nil {
		return "", nil, "", 0, opts, 0, 0, 0, false, nil, err
	}
	flags := byte(0)
	if len(args) > 0 {
		flags = args[0]
	}
	off := n
	if flags&textFlagOpts != 0 {
		if len(args) < n+2 {
			return "", nil, "", 0, opts, 0, 0, 0, false, nil, ErrVectorArgsTruncated
		}
		readConsistency = args[n]
		onPartitionUnavailable = args[n+1]
		bound, off, err = readBoundTail(args, n+2, readConsistency)
		if err != nil {
			return "", nil, "", 0, opts, 0, 0, 0, false, nil, err
		}
	}
	globalIDF = flags&textFlagGlobalIDF != 0
	g, _, err = readGlobalStatsBlock(args, off, flags)
	if err != nil {
		return "", nil, "", 0, opts, 0, 0, 0, false, nil, err
	}
	return collection, dense, query, k, opts, readConsistency, onPartitionUnavailable, bound, globalIDF, g, nil
}

// --- vector_bm25_stats: phase-0 corpus-stats op of the global-DF fan-out ---
//
// vector_bm25_stats gathers ONE partition's corpus-wide BM25 stats for a query's
// terms (n, tokenTotal, and per-term df). The coordinator fans it to every
// partition and SUMS the results into the global N/avgdl/df injected into phase 1.
//
// Args wire (collection at offset 0 — the At1 routing layout, VectorKeyColAt1,
// NOT the flags-first At2 of the scoring ops):
//
//	[colLen:u8][col]
//	[queryLen:u32][query: raw UTF-8 text]
//	(optional self-delimiting [marker][rc][opa](+bound) read-opts trailer)
//
// The rc/opa/bound trailer uses the shared self-delimiting marker codec
// (AppendReadOptsTrailerBounded) so a no-opts blob is byte-identical to the bare
// [col][query] form and ReadConsistencyOf can peek the rc for the Linearizable
// barrier on phase 0.

// EncodeBM25StatsArgs serializes vector_bm25_stats args with an optional read-opts
// trailer (byte-identical to the no-opts form when rc==0 && opa==0).
func EncodeBM25StatsArgs(collection string, query string, readConsistency, onPartitionUnavailable uint8, bound uint64) []byte {
	n := 1 + len(collection) + 4 + len(query)
	buf := make([]byte, 0, n+11)
	buf = append(buf, byte(len(collection)))
	buf = append(buf, collection...)
	buf = binary.BigEndian.AppendUint32(buf, uint32(len(query)))
	buf = append(buf, query...)
	buf = AppendReadOptsTrailerBounded(buf, readConsistency, onPartitionUnavailable, bound)
	return buf
}

// decodeBM25StatsArgsN decodes the base [col][query] block, returning bytes
// consumed so the opts trailer can be read behind it.
func decodeBM25StatsArgsN(args []byte) (collection string, query string, n int, err error) {
	if len(args) < 1 {
		return "", "", 0, ErrVectorArgsTruncated
	}
	colLen := int(args[0])
	if len(args) < 1+colLen+4 {
		return "", "", 0, ErrVectorArgsTruncated
	}
	collection = string(args[1 : 1+colLen])
	off := 1 + colLen
	qLen := int(binary.BigEndian.Uint32(args[off:]))
	off += 4
	if len(args) < off+qLen {
		return "", "", 0, ErrVectorArgsTruncated
	}
	query = string(args[off : off+qLen])
	off += qLen
	return collection, query, off, nil
}

// DecodeBM25StatsArgs decodes vector_bm25_stats args, including the optional
// read-opts trailer (rc=0, opa=0, bound=0 when absent).
func DecodeBM25StatsArgs(args []byte) (collection string, query string, readConsistency, onPartitionUnavailable uint8, bound uint64, err error) {
	collection, query, n, err := decodeBM25StatsArgsN(args)
	if err != nil {
		return "", "", 0, 0, 0, err
	}
	readConsistency, onPartitionUnavailable, bound, err = DecodeReadOptsTrailerBounded(args, n)
	if err != nil {
		return "", "", 0, 0, 0, err
	}
	return collection, query, readConsistency, onPartitionUnavailable, bound, nil
}

// EncodeBM25StatsResult serializes a partition's corpus stats reply. Wire:
//
//	[n:i64][tokenTotal:u64][dfCount:u32]{termID:u32, df:u32}*
func EncodeBM25StatsResult(n int, tokenTotal uint64, df map[uint32]int) []byte {
	buf := make([]byte, 0, bm25StatsResultHdr+dfPairLen*len(df))
	buf = binary.BigEndian.AppendUint64(buf, uint64(int64(n)))
	buf = binary.BigEndian.AppendUint64(buf, tokenTotal)
	buf = binary.BigEndian.AppendUint32(buf, uint32(len(df)))
	// Ascending term-id order so the reply is byte-canonical (see sortedDFKeys).
	for _, term := range sortedDFKeys(df) {
		buf = binary.BigEndian.AppendUint32(buf, term)
		buf = binary.BigEndian.AppendUint32(buf, uint32(df[term]))
	}
	return buf
}

// DecodeBM25StatsResult decodes a partition's corpus stats reply.
func DecodeBM25StatsResult(body []byte) (n int, tokenTotal uint64, df map[uint32]int, err error) {
	if len(body) < bm25StatsResultHdr {
		return 0, 0, nil, ErrVectorArgsTruncated
	}
	n = int(int64(binary.BigEndian.Uint64(body[0:])))
	tokenTotal = binary.BigEndian.Uint64(body[8:])
	cnt := int(binary.BigEndian.Uint32(body[16:]))
	off := bm25StatsResultHdr
	// See the args-side twin: one wire factor times a constant, expressed as the
	// shared bound so the reasoning lives in one place.
	if !CountFitsIn(cnt, len(body)-off, dfPairLen) {
		return 0, 0, nil, ErrVectorArgsTruncated
	}
	if cnt > 0 {
		df = make(map[uint32]int, cnt)
	}
	for i := 0; i < cnt; i++ {
		term := binary.BigEndian.Uint32(body[off:])
		off += 4
		d := int(binary.BigEndian.Uint32(body[off:]))
		off += 4
		df[term] = d
	}
	return n, tokenTotal, df, nil
}
