// SPDX-License-Identifier: Apache-2.0

package wire

import (
	"encoding/binary"
	"encoding/json"
	"fmt"
	"math"
	"slices"

	"github.com/rostamlabs/rostam/sdk/vtypes"
)

// encodeMatrix serializes a [][]float32 as [rows:u32]{[dim:u32][floats]}.
func encodeMatrix(buf []byte, rows [][]float32) []byte {
	var tmp [4]byte
	binary.BigEndian.PutUint32(tmp[:], uint32(len(rows))) //nolint:gosec
	buf = append(buf, tmp[:]...)
	for _, r := range rows {
		binary.BigEndian.PutUint32(tmp[:], uint32(len(r))) //nolint:gosec
		buf = append(buf, tmp[:]...)
		for _, f := range r {
			binary.BigEndian.PutUint32(tmp[:], math.Float32bits(f))
			buf = append(buf, tmp[:]...)
		}
	}
	return buf
}

// DecodeMatrix reads a matrix written by encodeMatrix, returning it and the
// number of bytes consumed.
func DecodeMatrix(b []byte) ([][]float32, int, error) {
	if len(b) < 4 {
		return nil, 0, ErrVectorArgsTruncated
	}
	rows := int(binary.BigEndian.Uint32(b))
	off := 4
	// A row costs >= 4 bytes ([dim:u32], dim 0) — see encodeMatrix.
	if !CountFitsIn(rows, len(b)-off, 4) {
		return nil, 0, ErrVectorArgsTruncated
	}
	out := make([][]float32, rows)
	for i := 0; i < rows; i++ {
		if len(b) < off+4 {
			return nil, 0, ErrVectorArgsTruncated
		}
		dim := int(binary.BigEndian.Uint32(b[off:]))
		off += 4
		if len(b) < off+4*dim {
			return nil, 0, ErrVectorArgsTruncated
		}
		row := make([]float32, dim)
		for j := 0; j < dim; j++ {
			row[j] = math.Float32frombits(binary.BigEndian.Uint32(b[off:]))
			off += 4
		}
		out[i] = row
	}
	return out, off, nil
}

// EncodeMVCreateArgs serializes a vector_mv_create_collection request. Wire:
//
//	[nameLen:u8][name][dim:u32][m:u32][efc:u32][efs:u32][seed:i64]
//	[quant:u8][rescore:u32][persistent:u8][partitions:u32]
//	  [indexType:1][ivfNlist:4][ivfNprobe:4]              (IVF block, optional)
//	  [ivfPQ:1][ivfPQM:4][ivfRerank:1]                    (IVF-PQ sub-block, optional)
//	  [opq:1]                                             (OPQ flag, optional)
//	  [ivfTrainThreshold:4]                               (train threshold, optional)
//	  [pqDropVecs:1]                                       (HNSW-PQ float-drop, optional)
//
// The quant/rescore/persistent tail is appended backward-compatibly: a decoder
// that finds it absent leaves those fields zero (QuantNone, non-persistent). The
// trailing partitions u32 is likewise optional — an old payload without it
// decodes to Partitions=0.
//
// The IVF extension mirrors the dense create wire (EncodeCreateCollectionArgs).
// It is written ONLY when non-default (IndexType != HNSW || any IVF knob set), so
// a plain HNSW MV create stays BYTE-IDENTICAL to the pre-IVF encoder. Each
// sub-block (IVF-PQ, OPQ, train-threshold) is itself appended only when needed,
// so an IVF-Flat create omits the PQ/OPQ bytes.
func EncodeMVCreateArgs(name string, cfg vtypes.MultiVectorConfig) []byte {
	ivfpq := cfg.IVFPQ || cfg.IVFRerank
	// Drift-retrain trailer (ivfDriftRetrain:1 + ivfDriftGrowthFactor:8 +
	// ivfDriftFactor:8 = 17 bytes) rides at the VERY END, after the PQDropVecs byte.
	// A non-default drift config forces every preceding optional slot present (OPQ,
	// the threshold word, and the PQDropVecs byte) so the trailing block is anchored.
	// Byte-identical when all three drift fields are zero/false.
	// FilterFirstRelativeBP (a 4-byte word) rides at the VERY END, after the drift
	// block. A non-zero relativeBP FORCES the drift block (which forces the OPQ slot,
	// the threshold word, and the PQDropVecs byte), so the trailing 4-byte read is
	// anchored. Byte-identical when FilterFirstRelativeBP==0 (forces nothing).
	// OPQIters (a 4-byte word) rides at the VERY END, AFTER the FilterFirstRelativeBP
	// word. A non-zero OPQIters FORCES the relativeBP word (which forces the drift
	// block, the OPQ slot, the threshold word, and the PQDropVecs byte), so the
	// trailing 4-byte read is anchored. Byte-identical when OPQIters==0 (forces
	// nothing — incl. the default 0→1 v1 behavior).
	opqIters := cfg.OPQIters != 0
	relBP := cfg.FilterFirstRelativeBP != 0 || opqIters
	drift := cfg.IVFDriftRetrain || cfg.IVFDriftGrowthFactor != 0 || cfg.IVFDriftFactor != 0 || relBP
	ivf := cfg.IndexType != vtypes.IndexHNSW || cfg.IVFNlist != 0 || cfg.IVFNprobe != 0 || ivfpq || cfg.OPQ || cfg.IVFTrainThreshold != 0 || drift
	n := 1 + len(name) + 4 + 4 + 4 + 4 + 8 + 1 + 4 + 1 + 4
	if ivf {
		n += 1 + 4 + 4 // indexType + ivfNlist + ivfNprobe
	}
	if ivfpq {
		n += 1 + 4 + 1 // ivfPQ + ivfPQM + ivfRerank
	}
	// OPQ flag appended after the PQ sub-block ONLY when true, mirroring the dense
	// wire. A non-zero IVFTrainThreshold OR a PQDropVecs flag (OR a drift config)
	// FORCES the OPQ slot too, so the 1-byte OPQ read anchors the 4-byte threshold
	// word that follows it (which in turn anchors the trailing PQDropVecs byte).
	if cfg.OPQ || cfg.IVFTrainThreshold != 0 || cfg.PQDropVecs || drift {
		n++
	}
	// IVFTrainThreshold (4 bytes) appended after the OPQ slot when non-zero OR when
	// PQDropVecs/drift is set (so the trailing bytes have their 4-byte anchor; a
	// PQDropVecs-only create writes a zero threshold word == engine default).
	if cfg.IVFTrainThreshold != 0 || cfg.PQDropVecs || drift {
		n += 4
	}
	// PQDropVecs (HNSW-PQ float-drop) appended after the threshold word ONLY when true
	// OR when drift follows (so the drift block has its 1-byte anchor).
	if cfg.PQDropVecs || drift {
		n++
	}
	if drift {
		n += 1 + 8 + 8 // ivfDriftRetrain:1 + ivfDriftGrowthFactor:8 + ivfDriftFactor:8
	}
	if relBP {
		n += 4 // FilterFirstRelativeBP word
	}
	if opqIters {
		n += 4 // OPQIters word
	}
	buf := make([]byte, n)
	buf[0] = byte(len(name))
	off := 1 + copy(buf[1:], name)
	binary.BigEndian.PutUint32(buf[off:], uint32(cfg.Dim)) //nolint:gosec
	off += 4
	binary.BigEndian.PutUint32(buf[off:], uint32(cfg.M)) //nolint:gosec
	off += 4
	binary.BigEndian.PutUint32(buf[off:], uint32(cfg.EfConstruction)) //nolint:gosec
	off += 4
	binary.BigEndian.PutUint32(buf[off:], uint32(cfg.EfSearch)) //nolint:gosec
	off += 4
	binary.BigEndian.PutUint64(buf[off:], uint64(cfg.Seed)) //nolint:gosec
	off += 8
	buf[off] = byte(cfg.Quant)
	off++
	binary.BigEndian.PutUint32(buf[off:], uint32(cfg.RescoreFactor)) //nolint:gosec
	off += 4
	if cfg.Persistent {
		buf[off] = 1
	}
	off++
	binary.BigEndian.PutUint32(buf[off:], uint32(cfg.Partitions)) //nolint:gosec
	off += 4
	if ivf {
		buf[off] = byte(cfg.IndexType)
		off++
		binary.BigEndian.PutUint32(buf[off:], uint32(cfg.IVFNlist)) //nolint:gosec
		off += 4
		binary.BigEndian.PutUint32(buf[off:], uint32(cfg.IVFNprobe)) //nolint:gosec
		off += 4
	}
	if ivfpq {
		if cfg.IVFPQ {
			buf[off] = 1
		}
		off++
		binary.BigEndian.PutUint32(buf[off:], uint32(cfg.IVFPQM)) //nolint:gosec
		off += 4
		if cfg.IVFRerank {
			buf[off] = 1
		}
		off++
	}
	if cfg.OPQ || cfg.IVFTrainThreshold != 0 || cfg.PQDropVecs || drift {
		if cfg.OPQ {
			buf[off] = 1
		}
		off++
	}
	if cfg.IVFTrainThreshold != 0 || cfg.PQDropVecs || drift {
		binary.BigEndian.PutUint32(buf[off:], uint32(cfg.IVFTrainThreshold)) //nolint:gosec
		off += 4
	}
	if cfg.PQDropVecs || drift {
		if cfg.PQDropVecs {
			buf[off] = 1
		}
		off++
	}
	if drift {
		if cfg.IVFDriftRetrain {
			buf[off] = 1
		}
		off++
		binary.BigEndian.PutUint64(buf[off:], math.Float64bits(cfg.IVFDriftGrowthFactor))
		off += 8
		binary.BigEndian.PutUint64(buf[off:], math.Float64bits(cfg.IVFDriftFactor))
		off += 8
	}
	if relBP {
		binary.BigEndian.PutUint32(buf[off:], uint32(cfg.FilterFirstRelativeBP)) //nolint:gosec
		off += 4
	}
	if opqIters {
		binary.BigEndian.PutUint32(buf[off:], uint32(cfg.OPQIters)) //nolint:gosec
	}
	return buf
}

// DecodeMVCreateArgs reads args produced by EncodeMVCreateArgs.
func DecodeMVCreateArgs(args []byte) (string, vtypes.MultiVectorConfig, error) {
	if len(args) < 1 {
		return "", vtypes.MultiVectorConfig{}, ErrVectorArgsTruncated
	}
	nameLen := int(args[0])
	if len(args) < 1+nameLen+4+4+4+4+8 {
		return "", vtypes.MultiVectorConfig{}, ErrVectorArgsTruncated
	}
	name := string(args[1 : 1+nameLen])
	off := 1 + nameLen
	cfg := vtypes.MultiVectorConfig{}
	cfg.Dim = int(binary.BigEndian.Uint32(args[off:]))
	off += 4
	cfg.M = int(binary.BigEndian.Uint32(args[off:]))
	off += 4
	cfg.EfConstruction = int(binary.BigEndian.Uint32(args[off:]))
	off += 4
	cfg.EfSearch = int(binary.BigEndian.Uint32(args[off:]))
	off += 4
	cfg.Seed = int64(binary.BigEndian.Uint64(args[off:])) //nolint:gosec
	off += 8
	// Quant/rescore/persistent extension — present iff the encoder included it.
	if len(args) >= off+1+4+1 {
		cfg.Quant = vtypes.QuantMode(args[off])
		off++
		cfg.RescoreFactor = int(binary.BigEndian.Uint32(args[off:]))
		off += 4
		cfg.Persistent = args[off] == 1
		off++
		// Partitions — present iff the encoder included it.
		if len(args) >= off+4 {
			cfg.Partitions = int(binary.BigEndian.Uint32(args[off:]))
			off += 4
			// IVF block — present iff the encoder included it (written only for a
			// non-default IVF config), mirroring the dense decode.
			if len(args) >= off+1+4+4 {
				cfg.IndexType = vtypes.IndexType(args[off])
				off++
				cfg.IVFNlist = int(binary.BigEndian.Uint32(args[off:]))
				off += 4
				cfg.IVFNprobe = int(binary.BigEndian.Uint32(args[off:]))
				off += 4
				// IVF-PQ sub-block — present iff requested (IVFPQ/IVFRerank), which
				// requires an IVF index. Gate the read on IndexType == IndexIVF so a
				// non-IVF (HNSW-PQ) index's trailing [OPQ][threshold][pqDropVecs] block
				// (also 6 bytes) is NOT misread as a PQ sub-block. The encoder writes
				// this sub-block only for IVFPQ/IVFRerank, both IVF-only.
				if cfg.IndexType == vtypes.IndexIVF && len(args) >= off+1+4+1 {
					cfg.IVFPQ = args[off] == 1
					off++
					cfg.IVFPQM = int(binary.BigEndian.Uint32(args[off:]))
					off += 4
					cfg.IVFRerank = args[off] == 1
					off++
				}
			}
			// OPQ flag — appended after the PQ sub-block iff true (so OPQ=false
			// never wrote it). A lone trailing byte here is unambiguously OPQ; with
			// the IVFTrainThreshold word also present the 4-byte read below consumes
			// it after this 1-byte slot.
			if len(args) >= off+1 {
				cfg.OPQ = args[off] == 1
				off++
				// IVFTrainThreshold — a 4-byte word appended after the OPQ slot iff
				// non-zero OR iff PQDropVecs follows (which needs its anchor). The OPQ
				// slot above anchors it.
				if len(args) >= off+4 {
					cfg.IVFTrainThreshold = int(binary.BigEndian.Uint32(args[off:]))
					off += 4
					// PQDropVecs — a trailing 1-byte flag appended at the very end iff
					// true (the HNSW-PQ float-drop). The IVFTrainThreshold word anchors
					// it; absent => false (byte-identical to the pre-PQDropVecs wire).
					if len(args) >= off+1 {
						cfg.PQDropVecs = args[off] == 1
						off++
						// Drift-retrain trailer (ivfDriftRetrain:1 + ivfDriftGrowthFactor:8
						// + ivfDriftFactor:8 = 17 bytes) at the very end iff any drift field
						// was non-default. A non-default drift config forces the OPQ slot,
						// threshold word, and PQDropVecs byte above, so the bytes remaining
						// here are unambiguously the drift block. Absent => zero/false
						// (byte-identical to the pre-drift wire).
						if len(args) >= off+1+8+8 {
							cfg.IVFDriftRetrain = args[off] == 1
							off++
							cfg.IVFDriftGrowthFactor = math.Float64frombits(binary.BigEndian.Uint64(args[off:]))
							off += 8
							cfg.IVFDriftFactor = math.Float64frombits(binary.BigEndian.Uint64(args[off:]))
							off += 8
							// FilterFirstRelativeBP — a 4-byte word at the very end iff
							// non-zero. A non-zero relativeBP forces the drift block above
							// (which forces every upstream slot), so any 4 bytes remaining
							// here are unambiguously the relativeBP. Absent => 0 (off,
							// byte-identical to the pre-feature wire).
							if len(args) >= off+4 {
								cfg.FilterFirstRelativeBP = int(binary.BigEndian.Uint32(args[off:]))
								off += 4
								// OPQIters — a 4-byte word at the very end iff non-zero. A
								// non-zero OPQIters forces the relativeBP word above (which
								// forces every upstream slot), so any 4 bytes remaining here
								// are unambiguously OPQIters. Absent => 0 (= 1 = v1 behavior,
								// byte-identical to the pre-feature wire).
								if len(args) >= off+4 {
									cfg.OPQIters = int(binary.BigEndian.Uint32(args[off:]))
								}
							}
						}
					}
				}
			}
		}
	}
	return name, cfg, nil
}

// EncodeMVAddArgs serializes a vector_mv_add request.
// Wire: [nameLen:u8][name][docID:u64][matrix][metaLen:u32][metaJSON].
func EncodeMVAddArgs(name string, docID uint64, tokens [][]float32, meta vtypes.Metadata) []byte {
	var metaJSON []byte
	if len(meta) > 0 {
		metaJSON, _ = json.Marshal(meta)
	}
	buf := make([]byte, 0, 1+len(name)+8+4+len(metaJSON)+16)
	buf = append(buf, byte(len(name)))
	buf = append(buf, name...)
	var u64 [8]byte
	binary.BigEndian.PutUint64(u64[:], docID)
	buf = append(buf, u64[:]...)
	buf = encodeMatrix(buf, tokens)
	var u32 [4]byte
	binary.BigEndian.PutUint32(u32[:], uint32(len(metaJSON))) //nolint:gosec
	buf = append(buf, u32[:]...)
	buf = append(buf, metaJSON...)
	return buf
}

// DecodeMVAddArgs reads args produced by EncodeMVAddArgs. Trailing bytes beyond
// the base block (a CAS trailer from EncodeMVAddArgsCAS) are ignored, so a
// non-CAS handler stays backward-compatible.
func DecodeMVAddArgs(args []byte) (string, uint64, [][]float32, vtypes.Metadata, error) {
	name, docID, tokens, meta, _, err := decodeMVAddArgsN(args)
	return name, docID, tokens, meta, err
}

// decodeMVAddArgsN decodes the mv-add base block and returns the number of bytes
// consumed, so DecodeMVAddArgsCAS can read a trailing CAS block.
func decodeMVAddArgsN(args []byte) (name string, docID uint64, tokens [][]float32, meta vtypes.Metadata, n int, err error) {
	if len(args) < 1 {
		return "", 0, nil, nil, 0, ErrVectorArgsTruncated
	}
	nameLen := int(args[0])
	if len(args) < 1+nameLen+8 {
		return "", 0, nil, nil, 0, ErrVectorArgsTruncated
	}
	name = string(args[1 : 1+nameLen])
	off := 1 + nameLen
	docID = binary.BigEndian.Uint64(args[off:])
	off += 8
	tokens, m, err := DecodeMatrix(args[off:])
	if err != nil {
		return "", 0, nil, nil, 0, err
	}
	off += m
	if len(args) < off+4 {
		return "", 0, nil, nil, 0, ErrVectorArgsTruncated
	}
	mlen := int(binary.BigEndian.Uint32(args[off:]))
	off += 4
	if len(args) < off+mlen {
		return "", 0, nil, nil, 0, ErrVectorArgsTruncated
	}
	if mlen > 0 {
		mm := make(vtypes.Metadata)
		if err := json.Unmarshal(args[off:off+mlen], &mm); err != nil {
			return "", 0, nil, nil, 0, fmt.Errorf("ops: decode mv metadata: %w", err)
		}
		meta = mm
	}
	off += mlen
	return name, docID, tokens, meta, off, nil
}

// appendMVSparseTrailer APPENDS the OPTIONAL doc-level sparse trailer to out and
// returns the grown slice. To keep a dense-only MV add byte-identical it appends
// NOTHING when sparse is nil/zero (no flag byte at all — the EOF-tolerant decoder
// covers the absence). Otherwise it appends a present flag (1) then
// [nnz:u32]{[dim:u32][value:f32]} (writeSparseAppend). It MUST be the LAST trailer
// on the wire (it self-delimits behind a single flag byte at the very end).
func appendMVSparseTrailer(out []byte, sparse *vtypes.SparseVector) []byte {
	if sparse == nil || sparse.IsZero() {
		return out // dense-only: byte-identical, no trailing block
	}
	out = append(out, 1)
	return writeSparseAppend(out, *sparse)
}

// readMVSparseTrailer reads the OPTIONAL doc-level sparse trailer at args[off:],
// returning the sparse vector (nil when absent) and the new offset. An old/dense
// blob ends before the flag byte ⇒ (nil, off, nil). A present flag (1) with a
// truncated body is fail-loud. flag 0 is treated as "absent" (defensive; the
// encoder omits the trailer rather than writing 0).
func readMVSparseTrailer(args []byte, off int) (*vtypes.SparseVector, int, error) {
	if off >= len(args) {
		return nil, off, nil // old / dense-only blob: no sparse trailer
	}
	if args[off] == 0 {
		return nil, off + 1, nil
	}
	off++
	sv, noff, err := readSparse(args, off)
	if err != nil {
		return nil, off, err
	}
	scopy := sv // own the slices (readSparse returns a fresh value)
	return &scopy, noff, nil
}

// EncodeMVAddArgsCAS serializes a vector_mv_add request carrying an optional
// optimistic-CAS precondition: a trailing [casPresent:u8][?expectedVersion:u64]
// block. When hasExpected is false the output is BYTE-IDENTICAL to EncodeMVAddArgs.
// The handler turns a present block into CASCond{Expected: expectedVersion,
// Has: true} (expectedVersion 0 = expect-absent / add-if-absent).
func EncodeMVAddArgsCAS(name string, docID uint64, tokens [][]float32, meta vtypes.Metadata, expectedVersion uint64, hasExpected bool) []byte {
	return EncodeMVAddArgsCASKeyTTL(name, docID, tokens, meta, expectedVersion, hasExpected, nil)
}

// EncodeMVAddArgsKeyTTL serializes a vector_mv_add request carrying an OPTIONAL
// per-key payload TTL map (key -> RELATIVE ms; the engine computes the ABSOLUTE
// deadline now+ttl at add, mirroring set_payload). When keyTTLMs is empty/nil the
// output is BYTE-IDENTICAL to EncodeMVAddArgs. The keyTTL block rides AFTER the
// base block (before any CAS/version trailer), self-delimiting behind a present
// byte — exactly like set_payload.
func EncodeMVAddArgsKeyTTL(name string, docID uint64, tokens [][]float32, meta vtypes.Metadata, keyTTLMs map[string]int64) []byte {
	return EncodeMVAddArgsCASKeyTTL(name, docID, tokens, meta, 0, false, keyTTLMs)
}

// EncodeMVAddArgsCASKeyTTL is EncodeMVAddArgsCAS plus an OPTIONAL per-key payload
// TTL map (key -> RELATIVE ms). The keyTTL block rides AFTER the base block and
// BEFORE the CAS block; to keep the CAS block at a deterministic offset when both
// are present the keyTTL present byte is ALWAYS emitted when a CAS block follows (0
// when no map) — the set_payload-CAS interpose. When keyTTLMs is empty AND
// hasExpected is false the output is BYTE-IDENTICAL to EncodeMVAddArgs.
func EncodeMVAddArgsCASKeyTTL(name string, docID uint64, tokens [][]float32, meta vtypes.Metadata, expectedVersion uint64, hasExpected bool, keyTTLMs map[string]int64) []byte {
	return EncodeMVAddArgsCASKeyTTLSparse(name, docID, tokens, meta, expectedVersion, hasExpected, keyTTLMs, nil)
}

// EncodeMVAddArgsCASKeyTTLSparse is EncodeMVAddArgsCASKeyTTL plus an OPTIONAL
// doc-level sparse vector carried by a trailing block AFTER the CAS block. When
// sparse is nil/zero the output is BYTE-IDENTICAL to EncodeMVAddArgsCASKeyTTL (no
// trailing block) — so a dense-only MV add stays byte-identical on the wire. The
// sparse trailer is ALWAYS last (it self-delimits behind a single flag byte at the
// very end, decoded EOF-tolerantly).
func EncodeMVAddArgsCASKeyTTLSparse(name string, docID uint64, tokens [][]float32, meta vtypes.Metadata, expectedVersion uint64, hasExpected bool, keyTTLMs map[string]int64, sparse *vtypes.SparseVector) []byte {
	sparsePresent := sparse != nil && !sparse.IsZero()
	base := EncodeMVAddArgs(name, docID, tokens, meta)
	// A trailer (CAS and/or sparse) follows ⇒ force the keyTTL present byte so the CAS
	// block (and behind it the sparse trailer) ride at a deterministic offset.
	out := appendKeyTTLBlock(base, keyTTLMs, hasExpected || sparsePresent)
	switch {
	case hasExpected:
		out = append(out, 1)
		var u64 [8]byte
		binary.BigEndian.PutUint64(u64[:], expectedVersion)
		out = append(out, u64[:]...)
	case sparsePresent:
		// No CAS but a sparse trailer follows: emit the CAS-absent marker (0) so the
		// sparse trailer rides at a deterministic offset (the decoder consumes this
		// byte, then reads the sparse trailer).
		out = append(out, 0)
	}
	return appendMVSparseTrailer(out, sparse)
}

// DecodeMVAddArgsCAS reads args produced by EncodeMVAddArgsCAS (or the legacy
// EncodeMVAddArgs — the CAS block is optional). hasExpected reports whether a CAS
// precondition trailer was present.
func DecodeMVAddArgsCAS(args []byte) (name string, docID uint64, tokens [][]float32, meta vtypes.Metadata, expectedVersion uint64, hasExpected bool, err error) {
	name, docID, tokens, meta, expectedVersion, hasExpected, _, err = DecodeMVAddArgsCASKeyTTL(args)
	return name, docID, tokens, meta, expectedVersion, hasExpected, err
}

// DecodeMVAddArgsCASKeyTTL reads args produced by EncodeMVAddArgsKeyTTL /
// EncodeMVAddArgsCASKeyTTL (or the legacy EncodeMVAddArgs/CAS — both trailers are
// optional). It decodes the base block, then the OPTIONAL self-delimiting per-key
// TTL block (key -> RELATIVE ms; nil when absent), then the OPTIONAL
// [casPresent:u8][expectedVersion:u64] block. A legacy blob decodes to
// keyTTLMs=nil, hasExpected=false. A present-but-truncated trailer is fail-loud.
func DecodeMVAddArgsCASKeyTTL(args []byte) (name string, docID uint64, tokens [][]float32, meta vtypes.Metadata, expectedVersion uint64, hasExpected bool, keyTTLMs map[string]int64, err error) {
	name, docID, tokens, meta, expectedVersion, hasExpected, keyTTLMs, _, err = DecodeMVAddArgsCASKeyTTLSparse(args)
	return name, docID, tokens, meta, expectedVersion, hasExpected, keyTTLMs, err
}

// DecodeMVAddArgsCASKeyTTLSparse reads args produced by EncodeMVAddArgsCASKeyTTLSparse
// (or any legacy MV add wire — every trailer is optional). It returns the base
// fields, the per-key TTL map, the CAS precondition, AND the OPTIONAL doc-level
// sparse vector (nil when absent). A legacy/dense blob decodes to keyTTLMs=nil,
// hasExpected=false, sparse=nil. The CAS marker (0 = absent, 1 = present+u64) is
// consumed whenever a trailer block exists so the sparse trailer rides at a
// deterministic offset. A present-but-truncated trailer is fail-loud.
func DecodeMVAddArgsCASKeyTTLSparse(args []byte) (name string, docID uint64, tokens [][]float32, meta vtypes.Metadata, expectedVersion uint64, hasExpected bool, keyTTLMs map[string]int64, sparse *vtypes.SparseVector, err error) {
	name, docID, tokens, meta, n, err := decodeMVAddArgsN(args)
	if err != nil {
		return "", 0, nil, nil, 0, false, nil, nil, err
	}
	keyTTLMs, off, err := readKeyTTLBlock(args, n)
	if err != nil {
		return "", 0, nil, nil, 0, false, nil, nil, err
	}
	if off >= len(args) {
		return name, docID, tokens, meta, 0, false, keyTTLMs, nil, nil
	}
	// CAS marker: consume it (0 = absent, 1 = present + u64 version). A standalone 0
	// is only emitted when a sparse trailer follows (the deterministic-offset
	// interpose), so consuming it then reading the sparse trailer is unambiguous.
	casMarker := args[off]
	off++
	if casMarker == 1 {
		if len(args) < off+8 {
			return "", 0, nil, nil, 0, false, nil, nil, ErrVectorArgsTruncated
		}
		expectedVersion = binary.BigEndian.Uint64(args[off:])
		hasExpected = true
		off += 8
	}
	sparse, _, err = readMVSparseTrailer(args, off)
	if err != nil {
		return "", 0, nil, nil, 0, false, nil, nil, err
	}
	return name, docID, tokens, meta, expectedVersion, hasExpected, keyTTLMs, sparse, nil
}

// EncodeMVAddArgsVersioned serializes a vector_mv_add request carrying an optional
// VERSION-PRESERVING block: a trailing [verPresent:u8][?version:u64] that requests
// the handler set the document's per-document version VERBATIM (not bumped to 1).
// When version==0 the output is BYTE-IDENTICAL to EncodeMVAddArgs (no trailer).
// Used by the MV reshard copy passes so copied documents keep their CAS version:
// the online if-absent path (vector_mv_add_if_absent) and the offline verbatim
// replace path (vector_mv_add_versioned) both decode this trailer. Mirrors the
// dense vecFlagVersion block and the additive EncodeMVAddArgsCAS trailer.
func EncodeMVAddArgsVersioned(name string, docID uint64, tokens [][]float32, meta vtypes.Metadata, version uint64) []byte {
	base := EncodeMVAddArgs(name, docID, tokens, meta)
	if version == 0 {
		return base // byte-identical to the no-version wire
	}
	out := append(base, 1)
	var u64 [8]byte
	binary.BigEndian.PutUint64(u64[:], version)
	return append(out, u64[:]...)
}

// EncodeMVAddArgsVersionedKeyExpires is EncodeMVAddArgsVersioned plus an OPTIONAL
// ABSOLUTE per-key payload TTL map (key -> ABSOLUTE unix-millis deadline) carried
// by a trailing keyExpires block that rides AFTER the version trailer:
//
//	<base block from EncodeMVAddArgs>
//	  [verPresent:u8][version:u64]                           ← version trailer
//	  [kePresent:u8=1]{[n:u32]{[kLen:u32][k][deadline:u64]×n}} ← absolute keyExpires
//
// When keyExpires is empty the output is BYTE-IDENTICAL to EncodeMVAddArgsVersioned
// (no kePresent byte, no trailing block) — so the no-key-TTL reshard path stays
// zero-overhead and old decoders interoperate. When BOTH version==0 AND keyExpires
// is empty it is BYTE-IDENTICAL to EncodeMVAddArgs. The keyExpires block is emitted
// ONLY when keyExpires is non-empty; to keep its offset deterministic the version
// trailer is then ALWAYS emitted (verPresent=1, version possibly 0). The keyExpires
// deadlines are ABSOLUTE, applied VERBATIM (NOT recomputed now+ttl) — DISTINCT from
// the relative MV keyTTL block (EncodeMVAddArgsKeyTTL). Used ONLY by the MV
// reshard/resplit copy passes so a copied document keeps BOTH its per-document CAS
// version AND its original absolute key deadlines time-stable.
func EncodeMVAddArgsVersionedKeyExpires(name string, docID uint64, tokens [][]float32, meta vtypes.Metadata, version uint64, keyExpires map[string]uint64) []byte {
	return EncodeMVAddArgsVersionedKeyExpiresSparse(name, docID, tokens, meta, version, keyExpires, nil)
}

// EncodeMVAddArgsVersionedKeyExpiresSparse is EncodeMVAddArgsVersionedKeyExpires plus
// an OPTIONAL doc-level sparse vector carried by a trailing block AFTER the
// keyExpires block. When sparse is nil/zero the output is BYTE-IDENTICAL to
// EncodeMVAddArgsVersionedKeyExpires (so a dense-only reshard copy stays
// byte-identical). When sparse is present the version trailer is forced (verPresent=1,
// version possibly 0) and the kePresent byte is forced (0 when no keyExpires) so the
// sparse trailer rides at a deterministic offset. The sparse trailer is ALWAYS last.
func EncodeMVAddArgsVersionedKeyExpiresSparse(name string, docID uint64, tokens [][]float32, meta vtypes.Metadata, version uint64, keyExpires map[string]uint64, sparse *vtypes.SparseVector) []byte {
	sparsePresent := sparse != nil && !sparse.IsZero()
	if len(keyExpires) == 0 && !sparsePresent {
		// Zero-overhead: byte-identical to the version-only wire (the version trailer's
		// EOF tolerance covers a missing kePresent byte on decode).
		return EncodeMVAddArgsVersioned(name, docID, tokens, meta, version)
	}
	// A trailer (keyExpires and/or sparse) follows: force the version trailer (so the
	// kePresent byte rides at a deterministic offset even when version==0), then the
	// kePresent byte (0 when no keyExpires so the sparse trailer is deterministic),
	// then the ABSOLUTE keyExpires block, then the sparse trailer.
	base := EncodeMVAddArgs(name, docID, tokens, meta)
	out := append(base, 1) // verPresent=1
	var u64 [8]byte
	binary.BigEndian.PutUint64(u64[:], version)
	out = append(out, u64[:]...)
	if len(keyExpires) == 0 {
		out = append(out, 0) // kePresent=0 (sparse trailer follows at a fixed offset)
	} else {
		out = append(out, 1) // kePresent=1
		var u32 [4]byte
		binary.BigEndian.PutUint32(u32[:], uint32(len(keyExpires))) //nolint:gosec
		out = append(out, u32[:]...)
		for k, deadline := range keyExpires {
			binary.BigEndian.PutUint32(u32[:], uint32(len(k))) //nolint:gosec
			out = append(out, u32[:]...)
			out = append(out, k...)
			binary.BigEndian.PutUint64(u64[:], deadline)
			out = append(out, u64[:]...)
		}
	}
	return appendMVSparseTrailer(out, sparse)
}

// DecodeMVAddArgsVersioned reads args produced by EncodeMVAddArgsVersioned (or the
// legacy EncodeMVAddArgs — the version block is optional). version 0 / a missing
// block ⇒ no version-preservation (the plain bump/if-absent semantics).
func DecodeMVAddArgsVersioned(args []byte) (name string, docID uint64, tokens [][]float32, meta vtypes.Metadata, version uint64, err error) {
	name, docID, tokens, meta, version, _, err = DecodeMVAddArgsVersionedKeyExpires(args)
	return name, docID, tokens, meta, version, err
}

// DecodeMVAddArgsVersionedKeyExpiresSparse reads args produced by
// EncodeMVAddArgsVersionedKeyExpiresSparse (or any legacy MV add wire — every
// trailer is optional). It returns the version, the OPTIONAL ABSOLUTE keyExpires
// map, AND the OPTIONAL doc-level sparse vector (nil when absent). A legacy/dense
// blob decodes to version 0, keyExpires nil, sparse nil. The kePresent byte (0 =
// absent) is consumed whenever a trailer exists so the sparse trailer rides at a
// deterministic offset. A present-but-truncated trailer is fail-loud.
func DecodeMVAddArgsVersionedKeyExpiresSparse(args []byte) (name string, docID uint64, tokens [][]float32, meta vtypes.Metadata, version uint64, keyExpires map[string]uint64, sparse *vtypes.SparseVector, err error) {
	name, docID, tokens, meta, n, err := decodeMVAddArgsN(args)
	if err != nil {
		return "", 0, nil, nil, 0, nil, nil, err
	}
	if n >= len(args) || args[n] == 0 {
		return name, docID, tokens, meta, 0, nil, nil, nil
	}
	off := n + 1
	if len(args) < off+8 {
		return "", 0, nil, nil, 0, nil, nil, ErrVectorArgsTruncated
	}
	version = binary.BigEndian.Uint64(args[off:])
	off += 8
	// keyExpires block (kePresent byte). When absent (no byte at all) ⇒ no ke, no
	// sparse (old version-only wire). A present byte (0 = none, 1 = block) is consumed
	// so the sparse trailer rides at a deterministic offset.
	if off >= len(args) {
		return name, docID, tokens, meta, version, nil, nil, nil
	}
	kePresent := args[off]
	off++
	if kePresent == 1 {
		if len(args) < off+4 {
			return "", 0, nil, nil, 0, nil, nil, ErrVectorArgsTruncated
		}
		cnt := int(binary.BigEndian.Uint32(args[off:]))
		off += 4
		// An entry costs >= 12 bytes ([klen:u32] with an empty key + [ttl:u64]).
		if !CountFitsIn(cnt, len(args)-off, 12) {
			return "", 0, nil, nil, 0, nil, nil, ErrVectorArgsTruncated
		}
		ke := make(map[string]uint64, cnt)
		for j := 0; j < cnt; j++ {
			if len(args) < off+4 {
				return "", 0, nil, nil, 0, nil, nil, ErrVectorArgsTruncated
			}
			kl := int(binary.BigEndian.Uint32(args[off:]))
			off += 4
			if len(args) < off+kl+8 {
				return "", 0, nil, nil, 0, nil, nil, ErrVectorArgsTruncated
			}
			key := string(args[off : off+kl])
			off += kl
			ke[key] = binary.BigEndian.Uint64(args[off:])
			off += 8
		}
		if len(ke) > 0 {
			keyExpires = ke
		}
	}
	sparse, _, err = readMVSparseTrailer(args, off)
	if err != nil {
		return "", 0, nil, nil, 0, nil, nil, err
	}
	return name, docID, tokens, meta, version, keyExpires, sparse, nil
}

// DecodeMVAddArgsVersionedKeyExpires reads args produced by
// EncodeMVAddArgsVersionedKeyExpires (or the legacy EncodeMVAddArgs/Versioned —
// both trailers are optional). It returns the version AND the OPTIONAL ABSOLUTE
// per-key payload TTL map (nil when the kePresent byte is 0 or absent), applied
// VERBATIM by the reshard reinsert handlers. A legacy blob (no trailers) decodes
// to version 0, keyExpires nil. A present-but-truncated trailer is fail-loud.
func DecodeMVAddArgsVersionedKeyExpires(args []byte) (name string, docID uint64, tokens [][]float32, meta vtypes.Metadata, version uint64, keyExpires map[string]uint64, err error) {
	name, docID, tokens, meta, version, keyExpires, _, err = DecodeMVAddArgsVersionedKeyExpiresSparse(args)
	return name, docID, tokens, meta, version, keyExpires, err
}

// EncodeMVSearchArgs serializes a vector_mv_search request.
// Wire: [nameLen:u8][name][k:u32][candPerToken:u32][matrix].
func EncodeMVSearchArgs(name string, query [][]float32, k, candPerToken int) []byte {
	buf := make([]byte, 0, 1+len(name)+8+16)
	buf = append(buf, byte(len(name)))
	buf = append(buf, name...)
	var u32 [4]byte
	binary.BigEndian.PutUint32(u32[:], uint32(k)) //nolint:gosec
	buf = append(buf, u32[:]...)
	binary.BigEndian.PutUint32(u32[:], uint32(candPerToken)) //nolint:gosec
	buf = append(buf, u32[:]...)
	buf = encodeMatrix(buf, query)
	return buf
}

// DecodeMVSearchArgs reads args produced by EncodeMVSearchArgs. Trailing bytes
// beyond the base block (e.g. an opts trailer from EncodeMVSearchArgsOpts) are
// ignored, so the single-shard handler stays backward-compatible with
// opts-carrying args.
func DecodeMVSearchArgs(args []byte) (string, [][]float32, int, int, error) {
	name, query, k, candPerToken, _, err := decodeMVSearchArgsN(args)
	return name, query, k, candPerToken, err
}

// decodeMVSearchArgsN decodes the MV-search base block and returns the number of
// bytes consumed (so DecodeMVSearchArgsOpts can read a trailing opts block). MV
// args carry no flags byte, so the Opts trailer is self-delimiting (see
// EncodeMVSearchArgsOpts).
func decodeMVSearchArgsN(args []byte) (name string, query [][]float32, k, candPerToken, n int, err error) {
	if len(args) < 1 {
		return "", nil, 0, 0, 0, ErrVectorArgsTruncated
	}
	nameLen := int(args[0])
	if len(args) < 1+nameLen+8 {
		return "", nil, 0, 0, 0, ErrVectorArgsTruncated
	}
	name = string(args[1 : 1+nameLen])
	off := 1 + nameLen
	k = int(binary.BigEndian.Uint32(args[off:]))
	off += 4
	candPerToken = int(binary.BigEndian.Uint32(args[off:]))
	off += 4
	query, mn, err := DecodeMatrix(args[off:])
	if err != nil {
		return "", nil, 0, 0, 0, err
	}
	off += mn
	return name, query, k, candPerToken, off, nil
}

// MV-search opts-trailer marker bits. MV args carry no leading flags byte, so the
// optional trailer self-describes its contents via a single marker byte that rides
// immediately after the base block. The marker is a BITFIELD (not a bare "present"
// flag) so the trailer can carry rc/opa and/or a payload filter independently:
//
//	bit0 (mvTrailerOpts):   [rc:u8][opa:u8] follow the marker.
//	bit1 (MVTrailerFilter): [filterLen:u32][filterJSON] follow (after the rc/opa
//	                        block, when present).
//
// Crucially, the legacy "[1][rc][opa]" trailer (the only form ever emitted before
// the filter was added) is exactly marker==mvTrailerOpts with the rc/opa block —
// BYTE-IDENTICAL — so old encoders/decoders interoperate. A zero marker is never
// emitted (no trailer at all), which is how the no-filter/no-rc/no-opa case stays
// byte-identical to EncodeMVSearchArgs.
const (
	mvTrailerOpts      uint8 = 1 << 0 // [rc:u8][opa:u8] present
	MVTrailerFilter    uint8 = 1 << 1 // [filterLen:u32][filterJSON] present
	mvTrailerStaleness uint8 = 1 << 2 // 8-byte big-endian bound present (after rc/opa, before filter)
)

// EncodeMVSearchArgsOpts serializes a vector_mv_search request plus an optional
// self-delimiting trailer carrying a cross-shard consistency opts pair AND/OR a
// payload filter. MV args carry no flags byte, so the trailer self-describes via a
// marker byte (see mvTrailerOpts/MVTrailerFilter). Wire when the trailer is present:
//
//	<base block from EncodeMVSearchArgs>
//	  [marker:u8]
//	  [rc:u8][opa:u8]                ← present when marker&mvTrailerOpts
//	  [filterLen:u32][filterJSON]    ← present when marker&MVTrailerFilter
//
// When the filter is zero AND readConsistency==0 AND onPartitionUnavailable==0 the
// trailer is omitted entirely and the output is BYTE-IDENTICAL to EncodeMVSearchArgs
// (backward-compatible); the plain DecodeMVSearchArgs (single-shard handler) ignores
// any trailing bytes. When only rc/opa are non-zero the trailer is the legacy
// "[mvTrailerOpts][rc][opa]" form, byte-identical to the pre-filter encoder.
func EncodeMVSearchArgsOpts(name string, query [][]float32, k, candPerToken int, readConsistency, onPartitionUnavailable uint8, bound uint64) []byte {
	return encodeMVSearchArgsOptsFilter(name, query, k, candPerToken, readConsistency, onPartitionUnavailable, vtypes.Filter{}, bound)
}

// EncodeMVSearchArgsOptsFilter is EncodeMVSearchArgsOpts plus a payload filter
// carried on the wire (length-prefixed JSON, mirroring the dense
// EncodeVectorSearchArgsExt serialization). See EncodeMVSearchArgsOpts for the
// trailer layout and the backward-compat guarantees.
func EncodeMVSearchArgsOptsFilter(name string, query [][]float32, k, candPerToken int, readConsistency, onPartitionUnavailable uint8, filter vtypes.Filter, bound uint64) []byte {
	return encodeMVSearchArgsOptsFilter(name, query, k, candPerToken, readConsistency, onPartitionUnavailable, filter, bound)
}

func encodeMVSearchArgsOptsFilter(name string, query [][]float32, k, candPerToken int, readConsistency, onPartitionUnavailable uint8, filter vtypes.Filter, bound uint64) []byte {
	base := EncodeMVSearchArgs(name, query, k, candPerToken)

	var marker uint8
	if readConsistency != 0 || onPartitionUnavailable != 0 {
		marker |= mvTrailerOpts
	}
	if readConsistency == ConsistencyBoundedStaleness {
		marker |= mvTrailerStaleness
	}
	var filterJSON []byte
	if !filter.IsZero() {
		marker |= MVTrailerFilter
		filterJSON, _ = json.Marshal(filter) // same marshal dense uses (EncodeVectorSearchArgsExt)
	}
	if marker == 0 {
		return base // byte-identical to the legacy / no-trailer form
	}

	out := append(base, marker)
	if marker&mvTrailerOpts != 0 {
		out = append(out, readConsistency, onPartitionUnavailable)
	}
	if marker&mvTrailerStaleness != 0 {
		var b [8]byte
		binary.BigEndian.PutUint64(b[:], bound)
		out = append(out, b[:]...)
	}
	if marker&MVTrailerFilter != 0 {
		var u32 [4]byte
		binary.BigEndian.PutUint32(u32[:], uint32(len(filterJSON))) //nolint:gosec
		out = append(out, u32[:]...)
		out = append(out, filterJSON...)
	}
	return out
}

// DecodeMVSearchArgsOpts decodes a vector_mv_search request that may carry a
// self-delimiting opts trailer (rc/opa). Backward-compatible: legacy args (no
// trailer) decode with readConsistency=0, onPartitionUnavailable=0. A filter
// trailer block, if present, is correctly skipped but not returned; callers that
// need the filter use DecodeMVSearchArgsOptsFilter.
func DecodeMVSearchArgsOpts(args []byte) (name string, query [][]float32, k, candPerToken int, readConsistency, onPartitionUnavailable uint8, bound uint64, err error) {
	name, query, k, candPerToken, readConsistency, onPartitionUnavailable, _, bound, err = DecodeMVSearchArgsOptsFilter(args)
	return name, query, k, candPerToken, readConsistency, onPartitionUnavailable, bound, err
}

// DecodeMVSearchArgsOptsFilter decodes a vector_mv_search request including any
// optional opts trailer (rc/opa) and payload filter. Backward-compatible: legacy
// args (no trailer) decode with readConsistency=0, onPartitionUnavailable=0 and a
// zero vector.Filter. A malformed filter JSON is a fail-loud error (NOT a silent
// drop) so a corrupt/over-permissive filter never weakens a search.
func DecodeMVSearchArgsOptsFilter(args []byte) (name string, query [][]float32, k, candPerToken int, readConsistency, onPartitionUnavailable uint8, filter vtypes.Filter, bound uint64, err error) {
	name, query, k, candPerToken, n, err := decodeMVSearchArgsN(args)
	if err != nil {
		return "", nil, 0, 0, 0, 0, vtypes.Filter{}, 0, err
	}
	if len(args) <= n || args[n] == 0 {
		// No trailer (legacy / no-filter no-rc no-opa form). A zero marker is never
		// emitted, so treat a zero byte here as "no trailer" too (trailing-bytes
		// tolerance), matching DecodeMVSearchArgs's contract.
		return name, query, k, candPerToken, 0, 0, vtypes.Filter{}, 0, nil
	}
	marker := args[n]
	off := n + 1
	if marker&mvTrailerOpts != 0 {
		if len(args) < off+2 {
			return "", nil, 0, 0, 0, 0, vtypes.Filter{}, 0, ErrVectorArgsTruncated
		}
		readConsistency = args[off]
		onPartitionUnavailable = args[off+1]
		off += 2
	}
	if marker&mvTrailerStaleness != 0 {
		if len(args) < off+8 {
			return "", nil, 0, 0, 0, 0, vtypes.Filter{}, 0, ErrVectorArgsTruncated
		}
		bound = binary.BigEndian.Uint64(args[off : off+8])
		off += 8
	}
	if marker&MVTrailerFilter != 0 {
		if len(args) < off+4 {
			return "", nil, 0, 0, 0, 0, vtypes.Filter{}, 0, ErrVectorArgsTruncated
		}
		flen := int(binary.BigEndian.Uint32(args[off:]))
		off += 4
		if len(args) < off+flen {
			return "", nil, 0, 0, 0, 0, vtypes.Filter{}, 0, ErrVectorArgsTruncated
		}
		if uerr := json.Unmarshal(args[off:off+flen], &filter); uerr != nil {
			return "", nil, 0, 0, 0, 0, vtypes.Filter{}, 0, fmt.Errorf("ops: decode mv filter: %w", uerr)
		}
	}
	return name, query, k, candPerToken, readConsistency, onPartitionUnavailable, filter, bound, nil
}

// mvHybridFlag* are the MV-hybrid arg flag bits. The MV-hybrid wire is the
// cross-modality analogue of the named-hybrid wire (EncodeNamedHybridArgs): it
// carries ONE collection name (an MV index holds its MaxSim tokens AND an optional
// doc-level sparse vector in the same collection — no two space names), the MV token
// QUERY MATRIX (not a single dense vector), a sparse query, the fusion opts, an
// optional filter, and an optional rc/opa trailer. The collection name sits at
// offset 1 (behind the flags byte) — the At2 layout, IDENTICAL to the named-hybrid
// and dense-hybrid ops — so vector_mv_hybrid_search / vector_mv_hybrid_lanes are
// registered with VectorKeyColAt2 and listed in the At2 routing arm.
const (
	mvHybridFlagFilter uint8 = 1 << 0 // [filterLen:u32][filterJSON] present
	mvHybridFlagSparse uint8 = 1 << 1 // the sparse query frame carries terms (non-zero)
	mvHybridFlagOpts   uint8 = 1 << 2 // [rc:u8][opa:u8] trailer present
)

// EncodeMVHybridArgs serializes a vector_mv_hybrid_search / vector_mv_hybrid_lanes
// request (both share this codec). Wire (At2 — collection name at offset 1):
//
//	[flags:u8]                       bit0=HAS_FILTER, bit1=HAS_SPARSE, bit2=HAS_OPTS
//	[colLen:u8][col]
//	[k:u32]
//	[method:u8][alpha:f64][rrfK:u32][denseK:u32][sparseK:u32]
//	[token matrix: rows:u32, {dim:u32, f32×dim}×rows]   ← encodeMatrix; rows 0 = sparse-only
//	[sparse frame: nnz:u32, {dim:u32, value:f32}×nnz]   ← writeSparseAppend; nnz 0 = MaxSim-only
//	if HAS_FILTER: [filterLen:u32][filterJSON]
//	if HAS_OPTS:   [rc:u8][opa:u8]
//
// The MV token query rides as the shared matrix block (rows 0 = sparse-only); the
// sparse query rides as the shared sparse frame (nnz 0 = MaxSim-only). HAS_SPARSE is
// set iff the sparse query carries terms, so the degradation cases are
// self-describing. When rc==0 && opa==0 the opts trailer is omitted and HAS_OPTS is
// clear (byte-identical trailer), mirroring EncodeNamedHybridArgs.
func EncodeMVHybridArgs(name string, query [][]float32, sparseQ vtypes.SparseVector, k int, opts vtypes.HybridOpts, readConsistency, onPartitionUnavailable uint8, bound uint64) []byte {
	var flags uint8
	var filterJSON []byte
	if !opts.Filter.IsZero() {
		flags |= mvHybridFlagFilter
		filterJSON, _ = json.Marshal(opts.Filter)
	}
	if !sparseQ.IsZero() {
		flags |= mvHybridFlagSparse
	}
	if readConsistency != 0 || onPartitionUnavailable != 0 {
		flags |= mvHybridFlagOpts
	}
	buf := make([]byte, 0, 1+1+len(name)+4+(1+8+4+4+4)+4+4+len(filterJSON)+2)
	buf = append(buf, flags)
	buf = append(buf, byte(len(name)))
	buf = append(buf, name...)
	var u32 [4]byte
	var u64 [8]byte
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
	buf = encodeMatrix(buf, query)        // token query matrix (rows 0 = sparse-only)
	buf = writeSparseAppend(buf, sparseQ) // always: nnz 0 = empty (MaxSim-only)
	if flags&mvHybridFlagFilter != 0 {
		binary.BigEndian.PutUint32(u32[:], uint32(len(filterJSON))) //nolint:gosec
		buf = append(buf, u32[:]...)
		buf = append(buf, filterJSON...)
	}
	if flags&mvHybridFlagOpts != 0 {
		buf = append(buf, readConsistency, onPartitionUnavailable)
		buf = appendBoundTail(buf, readConsistency, bound) // 8 bound bytes ride ONLY when rc==BoundedStaleness
	}
	return buf
}

// DecodeMVHybridArgs reads args produced by EncodeMVHybridArgs. opts.Filter is the
// zero filter when absent; sparseQ is the zero SparseVector when absent; rc/opa are
// 0 when the opts trailer is absent. A present flag with a truncated block is
// fail-loud (so a Linearizable MV hybrid never silently degrades to stale).
func DecodeMVHybridArgs(args []byte) (name string, query [][]float32, sparseQ vtypes.SparseVector, k int, opts vtypes.HybridOpts, readConsistency, onPartitionUnavailable uint8, bound uint64, err error) {
	if len(args) < 2 {
		return "", nil, sparseQ, 0, opts, 0, 0, 0, ErrVectorArgsTruncated
	}
	flags := args[0]
	colLen := int(args[1])
	off := 2
	if len(args) < off+colLen+4+(1+8+4+4+4) {
		return "", nil, sparseQ, 0, opts, 0, 0, 0, ErrVectorArgsTruncated
	}
	name = string(args[off : off+colLen])
	off += colLen
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
	mtx, mn, merr := DecodeMatrix(args[off:])
	if merr != nil {
		return "", nil, vtypes.SparseVector{}, 0, opts, 0, 0, 0, merr
	}
	query = mtx
	off += mn
	sparseQ, off, err = readSparse(args, off)
	if err != nil {
		return "", nil, vtypes.SparseVector{}, 0, opts, 0, 0, 0, err
	}
	if flags&mvHybridFlagFilter != 0 {
		if len(args) < off+4 {
			return "", nil, vtypes.SparseVector{}, 0, opts, 0, 0, 0, ErrVectorArgsTruncated
		}
		flen := int(binary.BigEndian.Uint32(args[off:]))
		off += 4
		if len(args) < off+flen {
			return "", nil, vtypes.SparseVector{}, 0, opts, 0, 0, 0, ErrVectorArgsTruncated
		}
		if uerr := json.Unmarshal(args[off:off+flen], &opts.Filter); uerr != nil {
			return "", nil, vtypes.SparseVector{}, 0, opts, 0, 0, 0, fmt.Errorf("ops: decode mv hybrid filter: %w", uerr)
		}
		off += flen
	}
	if flags&mvHybridFlagOpts != 0 {
		if len(args) < off+2 {
			return "", nil, vtypes.SparseVector{}, 0, opts, 0, 0, 0, ErrVectorArgsTruncated
		}
		readConsistency = args[off]
		onPartitionUnavailable = args[off+1]
		off += 2
		bound, _, err = readBoundTail(args, off, readConsistency)
		if err != nil {
			return "", nil, vtypes.SparseVector{}, 0, opts, 0, 0, 0, err
		}
	}
	return name, query, sparseQ, k, opts, readConsistency, onPartitionUnavailable, bound, nil
}

// MV-scroll trailer marker bits. Like the MV-search trailer, vector_mv_scroll
// args carry no leading flags byte, so the optional trailer self-describes its
// contents via a single marker byte riding immediately after the scroll base
// block. The marker is a BITFIELD so a resume-after-id cursor and the cross-shard
// consistency opts pair can be carried independently. The filter rides INSIDE the
// base block (like named scroll), so there is no separate filter bit here:
//
//	bit0 (MVScrollCursor): [afterID:u64 BE] follows the marker.
//	bit1 (MVScrollOpts):   [rc:u8][opa:u8] follow (after the afterID, when present).
//
// A zero marker is never emitted (no trailer at all), which is how the
// no-cursor/no-rc/no-opa case stays byte-identical to EncodeMVScrollArgs.
//
//	bit2 (mvScrollOrder):  the shared order_by block (appendScrollOrderBlock) follows
//	  AFTER the cursor + opts blocks.
const (
	MVScrollCursor    uint8 = 1 << 0 // [afterID:u64] present
	MVScrollOpts      uint8 = 1 << 1 // [rc:u8][opa:u8] present
	mvScrollOrder     uint8 = 1 << 2 // shared order_by block present
	mvScrollStaleness uint8 = 1 << 3 // 8-byte big-endian bound present (after rc/opa, before order)
)

// EncodeMVScrollArgs serializes a vector_mv_scroll request (no cursor, no opts).
// Wire: [colLen:u8][col][limit:u32][filterLen:u32][filterJSON] (filterLen 0 = no
// filter) — the SAME base shape as EncodeNamedScrollArgs, so the MV scroll wire
// mirrors named scroll. The filter rides in the base block.
func EncodeMVScrollArgs(col string, filter vtypes.Filter, limit int) []byte {
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

// EncodeMVScrollArgsOpts serializes a vector_mv_scroll request carrying an
// optional resume-after-id cursor AND/OR the cross-shard consistency opts pair
// behind a single marker bitfield. The cursor carried here is the RAW afterID:u64
// — the base64 Cursor string (ops.EncodeScrollCursor) is an embedded-layer
// concern decoded to this raw afterID before the codec runs (mirroring named
// scroll, which also carries the raw u64 on the wire). Wire when the trailer is
// present:
//
//	<base block from EncodeMVScrollArgs (incl. filter)>
//	  [marker:u8]
//	  [afterID:u64]   ← present when marker&MVScrollCursor
//	  [rc:u8][opa:u8] ← present when marker&MVScrollOpts
//
// When neither a cursor nor non-zero rc/opa is present the trailer is omitted
// entirely and the output is BYTE-IDENTICAL to EncodeMVScrollArgs. Mirrors
// EncodeNamedScrollArgsOpts.
func EncodeMVScrollArgsOpts(col string, filter vtypes.Filter, limit int, readConsistency, onPartitionUnavailable uint8, afterID uint64, hasAfter bool) []byte {
	return EncodeMVScrollArgsOptsBounded(col, filter, limit, readConsistency, onPartitionUnavailable, afterID, hasAfter, 0)
}

// EncodeMVScrollArgsOptsBounded is EncodeMVScrollArgsOpts plus the optional 8-byte
// staleness bound behind the mvScrollStaleness marker bit (after rc/opa, before any
// order block) ONLY when rc==ConsistencyBoundedStaleness. Byte-identical for rc∈{0,1,2}.
func EncodeMVScrollArgsOptsBounded(col string, filter vtypes.Filter, limit int, readConsistency, onPartitionUnavailable uint8, afterID uint64, hasAfter bool, bound uint64) []byte {
	base := EncodeMVScrollArgs(col, filter, limit)
	var marker uint8
	if hasAfter {
		marker |= MVScrollCursor
	}
	if readConsistency != 0 || onPartitionUnavailable != 0 {
		marker |= MVScrollOpts
	}
	if readConsistency == ConsistencyBoundedStaleness {
		marker |= mvScrollStaleness
	}
	if marker == 0 {
		return base // byte-identical to the no-trailer form
	}
	out := append(base, marker)
	if marker&MVScrollCursor != 0 {
		var idb [8]byte
		binary.BigEndian.PutUint64(idb[:], afterID)
		out = append(out, idb[:]...)
	}
	if marker&MVScrollOpts != 0 {
		out = append(out, readConsistency, onPartitionUnavailable)
	}
	if marker&mvScrollStaleness != 0 {
		var b [8]byte
		binary.BigEndian.PutUint64(b[:], bound)
		out = append(out, b[:]...)
	}
	return out
}

// EncodeMVScrollArgsOrder is EncodeMVScrollArgsOpts with an ADDITIVE order_by block
// (the SAME shared block the dense codec uses, appendScrollOrderBlock). When order ==
// nil it is byte-identical to EncodeMVScrollArgsOpts (zero-overhead no-order_by wire).
// When order != nil the mvScrollOrder marker bit is set and the order block is appended
// after the cursor + opts blocks. Mirrors EncodeNamedScrollArgsOrder / EncodeScrollArgsOrder.
func EncodeMVScrollArgsOrder(col string, filter vtypes.Filter, limit int, readConsistency, onPartitionUnavailable uint8, afterID uint64, hasAfter bool, order *ScrollOrder) []byte {
	return EncodeMVScrollArgsOrderBounded(col, filter, limit, readConsistency, onPartitionUnavailable, afterID, hasAfter, order, 0)
}

// EncodeMVScrollArgsOrderBounded is EncodeMVScrollArgsOrder plus the optional 8-byte
// staleness bound behind the mvScrollStaleness marker bit (after rc/opa, BEFORE the
// order block) ONLY when rc==ConsistencyBoundedStaleness.
func EncodeMVScrollArgsOrderBounded(col string, filter vtypes.Filter, limit int, readConsistency, onPartitionUnavailable uint8, afterID uint64, hasAfter bool, order *ScrollOrder, bound uint64) []byte {
	if order == nil {
		return EncodeMVScrollArgsOptsBounded(col, filter, limit, readConsistency, onPartitionUnavailable, afterID, hasAfter, bound)
	}
	base := EncodeMVScrollArgs(col, filter, limit)
	marker := mvScrollOrder
	if hasAfter {
		marker |= MVScrollCursor
	}
	if readConsistency != 0 || onPartitionUnavailable != 0 {
		marker |= MVScrollOpts
	}
	if readConsistency == ConsistencyBoundedStaleness {
		marker |= mvScrollStaleness
	}
	out := append(base, marker)
	if marker&MVScrollCursor != 0 {
		var idb [8]byte
		binary.BigEndian.PutUint64(idb[:], afterID)
		out = append(out, idb[:]...)
	}
	if marker&MVScrollOpts != 0 {
		out = append(out, readConsistency, onPartitionUnavailable)
	}
	if marker&mvScrollStaleness != 0 {
		var b [8]byte
		binary.BigEndian.PutUint64(b[:], bound)
		out = append(out, b[:]...)
	}
	return appendScrollOrderBlock(out, order)
}

// DecodeMVScrollArgsOrder decodes a vector_mv_scroll request that MAY carry the
// additive order_by block written by EncodeMVScrollArgsOrder. Superset of
// DecodeMVScrollArgsOpts: same base + marker (cursor/opts) trailer, then (if the
// mvScrollOrder bit is set) the shared order block. order is nil for legacy args /
// order==nil at encode time. Mirrors DecodeNamedScrollArgsOrder / DecodeScrollArgsOrder.
func DecodeMVScrollArgsOrder(args []byte) (col string, filter vtypes.Filter, limit int, readConsistency, onPartitionUnavailable uint8, afterID uint64, hasAfter bool, order *ScrollOrder, err error) {
	col, filter, limit, n, err := decodeMVScrollArgsN(args)
	if err != nil {
		return "", vtypes.Filter{}, 0, 0, 0, 0, false, nil, err
	}
	if len(args) <= n || args[n] == 0 {
		return col, filter, limit, 0, 0, 0, false, nil, nil
	}
	marker := args[n]
	off := n + 1
	if marker&MVScrollCursor != 0 {
		if len(args) < off+8 {
			return "", vtypes.Filter{}, 0, 0, 0, 0, false, nil, ErrVectorArgsTruncated
		}
		afterID = binary.BigEndian.Uint64(args[off:])
		hasAfter = true
		off += 8
	}
	if marker&MVScrollOpts != 0 {
		if len(args) < off+2 {
			return "", vtypes.Filter{}, 0, 0, 0, 0, false, nil, ErrVectorArgsTruncated
		}
		readConsistency = args[off]
		onPartitionUnavailable = args[off+1]
		off += 2
	}
	if marker&mvScrollStaleness != 0 {
		// Consume the 8-byte bound (after rc/opa, before the order block).
		if len(args) < off+8 {
			return "", vtypes.Filter{}, 0, 0, 0, 0, false, nil, ErrVectorArgsTruncated
		}
		off += 8
	}
	if marker&mvScrollOrder != 0 {
		order, _, err = readScrollOrderBlock(args, off)
		if err != nil {
			return "", vtypes.Filter{}, 0, 0, 0, 0, false, nil, err
		}
		if order == nil {
			return "", vtypes.Filter{}, 0, 0, 0, 0, false, nil, ErrVectorArgsTruncated
		}
	}
	return col, filter, limit, readConsistency, onPartitionUnavailable, afterID, hasAfter, order, nil
}

// DecodeMVScrollArgsOpts decodes a vector_mv_scroll request that may carry the
// cursor and/or opts trailer written by EncodeMVScrollArgsOpts. Backward/forward
// compatible: args with no trailer decode with hasAfter=false, readConsistency=0,
// onPartitionUnavailable=0. A present marker with a missing/truncated block is
// corruption — fail loud. A malformed filter JSON in the base block is also
// fail-loud (never a silent drop). Mirrors DecodeNamedScrollArgsOpts.
func DecodeMVScrollArgsOpts(args []byte) (col string, filter vtypes.Filter, limit int, readConsistency, onPartitionUnavailable uint8, afterID uint64, hasAfter bool, bound uint64, err error) {
	col, filter, limit, n, err := decodeMVScrollArgsN(args)
	if err != nil {
		return "", vtypes.Filter{}, 0, 0, 0, 0, false, 0, err
	}
	if len(args) <= n || args[n] == 0 {
		// No trailer (no-cursor no-rc no-opa form). A zero marker is never emitted,
		// so a zero byte here means "no trailer" too (trailing-bytes tolerance).
		return col, filter, limit, 0, 0, 0, false, 0, nil
	}
	marker := args[n]
	off := n + 1
	if marker&MVScrollCursor != 0 {
		if len(args) < off+8 {
			return "", vtypes.Filter{}, 0, 0, 0, 0, false, 0, ErrVectorArgsTruncated
		}
		afterID = binary.BigEndian.Uint64(args[off:])
		hasAfter = true
		off += 8
	}
	if marker&MVScrollOpts != 0 {
		if len(args) < off+2 {
			return "", vtypes.Filter{}, 0, 0, 0, 0, false, 0, ErrVectorArgsTruncated
		}
		readConsistency = args[off]
		onPartitionUnavailable = args[off+1]
		off += 2
	}
	if marker&mvScrollStaleness != 0 {
		if len(args) < off+8 {
			return "", vtypes.Filter{}, 0, 0, 0, 0, false, 0, ErrVectorArgsTruncated
		}
		bound = binary.BigEndian.Uint64(args[off : off+8])
	}
	return col, filter, limit, readConsistency, onPartitionUnavailable, afterID, hasAfter, bound, nil
}

// decodeMVScrollArgsN decodes the MV-scroll base block (the same shape as the
// named-scroll base: [colLen:u8][col][limit:u32][filterLen:u32][filterJSON]) and
// returns the number of bytes consumed so DecodeMVScrollArgsOpts can read the
// self-delimiting trailer. A malformed filter JSON is fail-loud.
func decodeMVScrollArgsN(args []byte) (col string, filter vtypes.Filter, limit int, n int, err error) {
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
		if uerr := json.Unmarshal(args[off:off+flen], &filter); uerr != nil {
			return "", vtypes.Filter{}, 0, 0, fmt.Errorf("ops: decode mv scroll filter: %w", uerr)
		}
	}
	off += flen
	return col, filter, limit, off, nil
}

// EncodeMVResults serializes []vector.MultiResult.
// Wire: [count:u32]{[id:u64][score:f32][metaLen:u32][metaJSON]}.
func EncodeMVResults(results []vtypes.MultiResult) []byte {
	metas := make([][]byte, len(results))
	n := 4
	for i, r := range results {
		if len(r.Metadata) > 0 {
			metas[i], _ = json.Marshal(r.Metadata)
		}
		n += 8 + 4 + 4 + len(metas[i])
	}
	buf := make([]byte, n)
	binary.BigEndian.PutUint32(buf[0:4], uint32(len(results))) //nolint:gosec
	off := 4
	for i, r := range results {
		binary.BigEndian.PutUint64(buf[off:], r.ID)
		off += 8
		binary.BigEndian.PutUint32(buf[off:], math.Float32bits(r.Score))
		off += 4
		binary.BigEndian.PutUint32(buf[off:], uint32(len(metas[i]))) //nolint:gosec
		off += 4
		off += copy(buf[off:], metas[i])
	}
	return buf
}

// DecodeMVResults reads results produced by EncodeMVResults.
func DecodeMVResults(body []byte) ([]vtypes.MultiResult, error) {
	out, _, err := decodeMVResultsN(body)
	return out, err
}

// decodeMVResultsN decodes one MV-results block and returns the results plus the
// number of bytes consumed (so callers can read a trailing degraded block).
func decodeMVResultsN(body []byte) ([]vtypes.MultiResult, int, error) {
	if len(body) < 4 {
		return nil, 0, ErrVectorArgsTruncated
	}
	count := int(binary.BigEndian.Uint32(body[0:4]))
	off := 4
	// A result costs >= 16 bytes ([id:u64][score:u32][numTokens:u32]).
	if !CountFitsIn(count, len(body)-off, 16) {
		return nil, 0, ErrVectorArgsTruncated
	}
	out := make([]vtypes.MultiResult, 0, count)
	for i := 0; i < count; i++ {
		if len(body) < off+8+4+4 {
			return nil, 0, ErrVectorArgsTruncated
		}
		var r vtypes.MultiResult
		r.ID = binary.BigEndian.Uint64(body[off:])
		off += 8
		r.Score = math.Float32frombits(binary.BigEndian.Uint32(body[off:]))
		off += 4
		mlen := int(binary.BigEndian.Uint32(body[off:]))
		off += 4
		if len(body) < off+mlen {
			return nil, 0, ErrVectorArgsTruncated
		}
		if mlen > 0 {
			m := make(vtypes.Metadata)
			if err := json.Unmarshal(body[off:off+mlen], &m); err != nil {
				return nil, 0, fmt.Errorf("ops: decode mv result metadata: %w", err)
			}
			r.Metadata = m
			off += mlen
		}
		out = append(out, r)
	}
	return out, off, nil
}

// EncodeMVResultsDegraded encodes MV results with an optional degraded-partition
// trailer (same wire format as the dense search trailer). When degraded is false
// and missing is empty the output is byte-identical to EncodeMVResults.
func EncodeMVResultsDegraded(results []vtypes.MultiResult, degraded bool, missing []uint16) []byte {
	return appendDegradedTrailer(EncodeMVResults(results), degraded, missing)
}

// DecodeMVResultsDegraded decodes MV results and the optional degraded trailer.
// Backward-compatible with legacy EncodeMVResults bytes.
func DecodeMVResultsDegraded(body []byte) (results []vtypes.MultiResult, degraded bool, missing []uint16, err error) {
	results, off, err := decodeMVResultsN(body)
	if err != nil {
		return nil, false, nil, err
	}
	degraded, missing, err = readDegradedTrailer(body, off)
	return results, degraded, missing, err
}

// EncodeMVDeleteArgs serializes a vector_mv_delete (or drop, name-only) request.
// Wire: [nameLen:u8][name][docID:u64]. For drop, docID is ignored.
func EncodeMVDeleteArgs(name string, docID uint64) []byte {
	buf := make([]byte, 1+len(name)+8)
	buf[0] = byte(len(name))
	off := 1 + copy(buf[1:], name)
	binary.BigEndian.PutUint64(buf[off:], docID)
	return buf
}

// DecodeMVDeleteArgs reads args produced by EncodeMVDeleteArgs. Trailing bytes
// beyond the base block (a CAS trailer) are ignored for the non-CAS path.
func DecodeMVDeleteArgs(args []byte) (string, uint64, error) {
	name, docID, _, _, err := DecodeMVDeleteArgsCAS(args)
	return name, docID, err
}

// EncodeMVDeleteArgsCAS serializes a vector_mv_delete request with an optional
// optimistic-CAS precondition. When hasExpected is false the output is
// BYTE-IDENTICAL to EncodeMVDeleteArgs (no trailer). When present the trailing
// [1][expectedVersion:u64] is the CAS guard (expectedVersion 0 = expect-absent).
func EncodeMVDeleteArgsCAS(name string, docID uint64, expectedVersion uint64, hasExpected bool) []byte {
	base := EncodeMVDeleteArgs(name, docID)
	if !hasExpected {
		return base
	}
	out := append(base, 1)
	var u64 [8]byte
	binary.BigEndian.PutUint64(u64[:], expectedVersion)
	return append(out, u64[:]...)
}

// DecodeMVDeleteArgsCAS reads args produced by EncodeMVDeleteArgsCAS (or the
// legacy EncodeMVDeleteArgs — the CAS block is optional). hasExpected reports
// whether a CAS precondition trailer was present.
func DecodeMVDeleteArgsCAS(args []byte) (name string, docID uint64, expectedVersion uint64, hasExpected bool, err error) {
	if len(args) < 1 {
		return "", 0, 0, false, ErrVectorArgsTruncated
	}
	nameLen := int(args[0])
	if len(args) < 1+nameLen+8 {
		return "", 0, 0, false, ErrVectorArgsTruncated
	}
	name = string(args[1 : 1+nameLen])
	docID = binary.BigEndian.Uint64(args[1+nameLen:])
	off := 1 + nameLen + 8
	if off >= len(args) || args[off] == 0 {
		return name, docID, 0, false, nil
	}
	off++
	if len(args) < off+8 {
		return "", 0, 0, false, ErrVectorArgsTruncated
	}
	expectedVersion = binary.BigEndian.Uint64(args[off:])
	return name, docID, expectedVersion, true, nil
}

// EncodeMVExistsArgs serializes a vector_mv_exists request. Wire:
// [nameLen:u8][name][docID:u64] — byte-identical to the MV delete-args shape; the
// result is the shared exists byte (EncodeExistsResult). The MV add-if-absent op
// reuses EncodeMVAddArgs for its write side; its result is EncodeIfAbsentResult.
func EncodeMVExistsArgs(name string, docID uint64) []byte {
	return EncodeMVDeleteArgs(name, docID)
}

// DecodeMVExistsArgs reads args produced by EncodeMVExistsArgs.
func DecodeMVExistsArgs(args []byte) (string, uint64, error) {
	return DecodeMVDeleteArgs(args)
}

// EncodeMVGetConfigArgs serializes a vector_mv_get_config request (name-only).
// Wire: [nameLen:u8][name]. Mirrors EncodeGetConfigArgs for the dense path.
func EncodeMVGetConfigArgs(name string) []byte {
	buf := make([]byte, 1+len(name))
	buf[0] = byte(len(name))
	copy(buf[1:], name)
	return buf
}

// DecodeMVGetConfigArgs reads args produced by EncodeMVGetConfigArgs. Trailing
// bytes beyond the base [nameLen][name] block (the rc/opa opts trailer added by
// EncodeMVGetConfigArgsOpts) are IGNORED, so the single-shard handler stays
// backward-compatible with rc-carrying args.
func DecodeMVGetConfigArgs(args []byte) (string, error) {
	name, _, err := decodeNameArgsN(args)
	return name, err
}

// EncodeMVGetConfigArgsOpts is EncodeMVGetConfigArgs plus the self-delimiting
// [marker][rc][opa] opts trailer. Byte-identical to EncodeMVGetConfigArgs when
// rc==0 && opa==0 (AnyReplica default unchanged); a non-zero rc rides the trailer
// so a Linearizable MV get_config arms the shard barrier (via ReadConsistencyOf).
func EncodeMVGetConfigArgsOpts(name string, readConsistency, onPartitionUnavailable uint8, bound uint64) []byte {
	return AppendReadOptsTrailerBounded(EncodeMVGetConfigArgs(name), readConsistency, onPartitionUnavailable, bound)
}

// DecodeMVGetConfigArgsOpts decodes an MV get_config request that may carry the
// rc/opa (+ bound) opts trailer. Backward-compatible (legacy args ⇒ rc=0,opa=0,bound=0);
// a present marker with a truncated block is corruption — fail loud.
func DecodeMVGetConfigArgsOpts(args []byte) (name string, readConsistency, onPartitionUnavailable uint8, bound uint64, err error) {
	name, n, err := decodeNameArgsN(args)
	if err != nil {
		return "", 0, 0, 0, err
	}
	readConsistency, onPartitionUnavailable, bound, err = DecodeReadOptsTrailerBounded(args, n)
	if err != nil {
		return "", 0, 0, 0, err
	}
	return name, readConsistency, onPartitionUnavailable, bound, nil
}

// EncodeMVScanArgs serializes a vector_mv_scan_vectors request (name-only).
// Wire: [nameLen:u8][name] — identical to the MVGetConfig args codec.
func EncodeMVScanArgs(name string) []byte {
	return EncodeMVGetConfigArgs(name)
}

// DecodeMVScanArgs reads args produced by EncodeMVScanArgs.
func DecodeMVScanArgs(args []byte) (string, error) {
	return DecodeMVGetConfigArgs(args)
}

// EncodeMVScanResult serializes live MultiScanRecords (the MV resplit read
// primitive). Wire:
//
//	[count:u32] then per record:
//	  [id:u64][numTokens:u32][dim:u32] then numTokens*(dim*f32) [metaLen:u32][metaJSON][version:u64][keyExpires]
//
// dim is the per-record token dimensionality (0 if numTokens==0); every token of
// a record shares it. metaLen 0 = no metadata. version is the document's
// per-document CAS version, carried so the MV reshard backfill reinserts
// version-preserving (mirror dense EncodeScanVectorsResult). It is a TRAILING
// u64 per record, ALWAYS written (MV scan results are transient — never a stored
// artifact — so the decoder requires it, exactly like the dense scan codec).
//
// keyExpires is the per-record ABSOLUTE per-key payload TTL trailer riding AFTER
// the version: [present:u8]{[n:u32][kLen:u32 k deadline:u64]×n} (present=0 ⇒ no
// per-key TTL, no further bytes). It carries ABSOLUTE unix-millis deadlines
// VERBATIM (NOT relative ms; NOT recomputed) so the MV reshard backfill restores
// the doc's original key deadlines time-stable. A decoder reading an OLD blob (no
// keyExpires byte at all) tolerates its absence per-record (keyExpires → nil),
// mirroring the dense scan codec's EOF tolerance — MV scans are transient (never a
// stored artifact), so the new encoder always writes the present byte.
func EncodeMVScanResult(recs []vtypes.MultiScanRecord) []byte {
	metas := make([][]byte, len(recs))
	n := 4
	for i, r := range recs {
		if len(r.Metadata) > 0 {
			metas[i], _ = json.Marshal(r.Metadata)
		}
		dim := 0
		if len(r.Tokens) > 0 {
			dim = len(r.Tokens[0])
		}
		n += 8 + 4 + 4 + len(r.Tokens)*dim*4 + 4 + len(metas[i]) + 8 // +8: trailing version
		n++                                                          // +1: keyExpires present byte
		if len(r.KeyExpires) > 0 {
			n += 4 // n:u32
			for k := range r.KeyExpires {
				n += 4 + len(k) + 8 // kLen:u32 + key + deadline:u64
			}
		}
		n++ // +1: sparse present byte (always written; mirrors keyExpires)
		if r.Sparse != nil && !r.Sparse.IsZero() {
			n += 4 + len(r.Sparse.Indices)*8 // nnz:u32 + {dim:u32,value:f32}×nnz
		}
	}
	buf := make([]byte, n)
	binary.BigEndian.PutUint32(buf[0:4], uint32(len(recs))) //nolint:gosec // count >= 0
	off := 4
	for i, r := range recs {
		binary.BigEndian.PutUint64(buf[off:], r.ID)
		off += 8
		dim := 0
		if len(r.Tokens) > 0 {
			dim = len(r.Tokens[0])
		}
		binary.BigEndian.PutUint32(buf[off:], uint32(len(r.Tokens))) //nolint:gosec
		off += 4
		binary.BigEndian.PutUint32(buf[off:], uint32(dim)) //nolint:gosec
		off += 4
		for _, tok := range r.Tokens {
			for _, f := range tok {
				binary.BigEndian.PutUint32(buf[off:], math.Float32bits(f))
				off += 4
			}
		}
		binary.BigEndian.PutUint32(buf[off:], uint32(len(metas[i]))) //nolint:gosec
		off += 4
		off += copy(buf[off:], metas[i])
		binary.BigEndian.PutUint64(buf[off:], r.Version) // trailing per-document CAS version
		off += 8
		// ABSOLUTE per-key payload TTL trailer (present byte gated). present=0 when the
		// doc has no per-key TTL — no further bytes. Deadlines are ABSOLUTE unix-ms,
		// written verbatim so the reshard reinsert restores them time-stable.
		if len(r.KeyExpires) > 0 {
			buf[off] = 1
			off++
			binary.BigEndian.PutUint32(buf[off:], uint32(len(r.KeyExpires))) //nolint:gosec
			off += 4
			for k, deadline := range r.KeyExpires {
				binary.BigEndian.PutUint32(buf[off:], uint32(len(k))) //nolint:gosec
				off += 4
				off += copy(buf[off:], k)
				binary.BigEndian.PutUint64(buf[off:], deadline)
				off += 8
			}
		} else {
			buf[off] = 0
			off++
		}
		// OPTIONAL doc-level sparse trailer (present byte gated). present=0 when the doc
		// has no sparse vector — no further bytes (BYTE-IDENTICAL tail to the pre-sparse
		// encoding once the keyExpires byte rules are unchanged). present=1 ⇒ a sparse
		// frame [nnz:u32]{[dim:u32][value:f32]} follows. Carried verbatim so the reshard
		// reinsert restores it; an OLD blob (no byte at all) tolerates absence on decode.
		if r.Sparse != nil && !r.Sparse.IsZero() {
			buf[off] = 1
			off++
			binary.BigEndian.PutUint32(buf[off:], uint32(len(r.Sparse.Indices))) //nolint:gosec
			off += 4
			for i, dim := range r.Sparse.Indices {
				binary.BigEndian.PutUint32(buf[off:], dim)
				off += 4
				binary.BigEndian.PutUint32(buf[off:], math.Float32bits(r.Sparse.Values[i]))
				off += 4
			}
		} else {
			buf[off] = 0
			off++
		}
	}
	return buf
}

// DecodeMVScanResult reads records produced by EncodeMVScanResult. Token floats
// are deep-copied (owned by the result). The trailing per-record version is
// REQUIRED (the encoder always writes it; scan results are transient), mirroring
// the dense DecodeScanVectorsResult — a truncated trailer is ErrVectorArgsTruncated.
func DecodeMVScanResult(body []byte) ([]vtypes.MultiScanRecord, error) {
	if len(body) < 4 {
		return nil, ErrVectorArgsTruncated
	}
	count := int(binary.BigEndian.Uint32(body[0:4]))
	off := 4
	// A record costs >= 16 bytes ([id:u64][numTokens:u32][dim:u32]).
	if !CountFitsIn(count, len(body)-off, 16) {
		return nil, ErrVectorArgsTruncated
	}
	recs := make([]vtypes.MultiScanRecord, 0, count)
	for i := 0; i < count; i++ {
		if len(body) < off+8+4+4 {
			return nil, ErrVectorArgsTruncated
		}
		var r vtypes.MultiScanRecord
		r.ID = binary.BigEndian.Uint64(body[off:])
		off += 8
		numTokens := int(binary.BigEndian.Uint32(body[off:]))
		off += 4
		dim := int(binary.BigEndian.Uint32(body[off:]))
		off += 4
		// Same overflow as DecodeMVGetResultAt (see there for the full argument):
		// numTokens*dim*4 wraps for wire-sized factors — numTokens = dim = 2^31
		// wraps the product to exactly 0 — which makes this check a no-op and lets
		// a tiny body reserve tens of GB. Bound each factor against the bytes that
		// remain, then test the product by division rather than computing it.
		rem := len(body) - off
		if numTokens < 0 || dim < 0 || numTokens > rem {
			return nil, ErrVectorArgsTruncated
		}
		need := 0
		if numTokens > 0 && dim > 0 {
			if dim > rem/4 {
				return nil, ErrVectorArgsTruncated
			}
			perToken := dim * 4 // <= rem: cannot overflow
			if numTokens > (rem-4)/perToken {
				return nil, ErrVectorArgsTruncated
			}
			need = numTokens * perToken // <= rem-4: cannot overflow
		}
		if len(body) < off+need+4 {
			return nil, ErrVectorArgsTruncated
		}
		if numTokens > 0 {
			r.Tokens = make([][]float32, numTokens)
			for ti := 0; ti < numTokens; ti++ {
				tok := make([]float32, dim)
				for j := 0; j < dim; j++ {
					tok[j] = math.Float32frombits(binary.BigEndian.Uint32(body[off:]))
					off += 4
				}
				r.Tokens[ti] = tok
			}
		}
		mlen := int(binary.BigEndian.Uint32(body[off:]))
		off += 4
		if len(body) < off+mlen {
			return nil, ErrVectorArgsTruncated
		}
		if mlen > 0 {
			m := make(vtypes.Metadata)
			if err := json.Unmarshal(body[off:off+mlen], &m); err != nil {
				return nil, fmt.Errorf("ops: decode mv scan metadata: %w", err)
			}
			r.Metadata = m
			off += mlen
		}
		// Trailing per-document CAS version. It is ALWAYS written by the current
		// encoder (MV scan results are transient — a live reshard scan, never a
		// stored artifact), so it is REQUIRED, exactly like the dense
		// DecodeScanVectorsResult precedent. Treating it as optional/per-record
		// "old-blob-tolerant" would be unsafe for a multi-record blob: a missing
		// trailer on record N would consume record N+1's [id:u64] as N's version
		// and corrupt every following record. Since the encoder and decoder are
		// always versioned together, a required trailer is both correct and the
		// faithful mirror of the dense contract.
		if len(body) < off+8 {
			return nil, ErrVectorArgsTruncated
		}
		r.Version = binary.BigEndian.Uint64(body[off:])
		off += 8
		// ABSOLUTE per-key payload TTL trailer (present byte gated). The current encoder
		// always writes the present byte; an OLD blob (no byte at all) tolerates its
		// absence per-record (keyExpires → nil), mirroring the dense scan codec's EOF
		// tolerance (only ever hit at the very tail, since the encoder writes it for
		// every record). present!=0 ⇒ a [n:u32]{kLen,k,deadline:u64} block follows.
		if off < len(body) {
			present := body[off]
			off++
			if present != 0 {
				if len(body) < off+4 {
					return nil, ErrVectorArgsTruncated
				}
				cnt := int(binary.BigEndian.Uint32(body[off:]))
				off += 4
				// An entry costs >= 12 bytes ([klen:u32] empty key + [ttl:u64]).
				if !CountFitsIn(cnt, len(body)-off, 12) {
					return nil, ErrVectorArgsTruncated
				}
				ke := make(map[string]uint64, cnt)
				for j := 0; j < cnt; j++ {
					if len(body) < off+4 {
						return nil, ErrVectorArgsTruncated
					}
					klen := int(binary.BigEndian.Uint32(body[off:]))
					off += 4
					if len(body) < off+klen+8 {
						return nil, ErrVectorArgsTruncated
					}
					key := string(body[off : off+klen])
					off += klen
					ke[key] = binary.BigEndian.Uint64(body[off:])
					off += 8
				}
				if len(ke) > 0 {
					r.KeyExpires = ke
				}
			}
			// OPTIONAL doc-level sparse trailer (present byte gated), riding AFTER the
			// keyExpires block. The current encoder always writes the present byte; an OLD
			// blob (no byte) tolerates absence (sparse → nil) via the off<len guard.
			// present!=0 ⇒ a [nnz:u32]{dim:u32,value:f32} frame follows (readSparse).
			if off < len(body) {
				sp := body[off]
				off++
				if sp != 0 {
					sv, noff, serr := readSparse(body, off)
					if serr != nil {
						return nil, serr
					}
					off = noff
					scopy := sv
					r.Sparse = &scopy
				}
			}
		}
		recs = append(recs, r)
	}
	return recs, nil
}

// EncodeMVGetResult serializes a vector_mv_get result. Wire:
//
//	[found:u8] then if found==1:
//	  [numTokens:u32][dim:u32]{[tok f32×dim]}     ← numTokens 0 when with_vector off
//	  [metaPresent:u8][?metaLen:u32][?metaJSON]   ← metaPresent 0 when no payload / payload off
//
// dim is the per-doc token dimensionality (0 if numTokens==0); every token shares
// it. not-found is the found=0 FLAG (NEVER an op error). withVector gates the token
// matrix; withPayload gates the doc payload. Mirrors EncodeMVScanResult's record.
// A trailing [verPresent:u8][?version:u64] CAS version block rides at the tail
// (byte-identical when 0); see EncodeMVGetResultV.
func EncodeMVGetResult(found bool, tokens [][]float32, payload vtypes.Metadata, withVector, withPayload bool) []byte {
	return EncodeMVGetResultV(found, tokens, payload, withVector, withPayload, 0)
}

// EncodeMVGetResultV is EncodeMVGetResult plus the document's per-document CAS
// version. A 0 version writes verPresent=0 and NO version field (byte-identical to
// the pre-version encoding); a live version (>=1) writes verPresent=1 + the u64.
func EncodeMVGetResultV(found bool, tokens [][]float32, payload vtypes.Metadata, withVector, withPayload bool, version uint64) []byte {
	return appendMVGetResultV(nil, found, tokens, payload, withVector, withPayload, version)
}

// appendMVGetResultV appends an mv-get-result record (same wire layout as
// EncodeMVGetResultV) to dst and returns the grown slice. It sizes the record up
// front and grows dst ONCE, so it is allocation-free when dst already has the
// capacity — what lets the batch encoder serialize many token-matrix rows into one
// buffer without a throwaway slice per row. EncodeMVGetResultV is the nil-dst
// single-get wrapper (one presized alloc, byte-identical to before).
func appendMVGetResultV(dst []byte, found bool, tokens [][]float32, payload vtypes.Metadata, withVector, withPayload bool, version uint64) []byte {
	if !found {
		return append(dst, 0)
	}
	var metaJSON []byte
	if withPayload && len(payload) > 0 {
		metaJSON, _ = json.Marshal(payload)
	}
	dim := 0
	includeTokens := withVector && len(tokens) > 0
	if includeTokens {
		dim = len(tokens[0])
	}
	n := 1 + 4 + 4 + 1 + 1 // found + numTokens + dim + metaPresent + verPresent
	if len(metaJSON) > 0 {
		n += 4 + len(metaJSON) // metaLen + metaJSON (only written when present)
	}
	if version != 0 {
		n += 8
	}
	if includeTokens {
		n += len(tokens) * dim * 4
	}
	start := len(dst)
	dst = slices.Grow(dst, n)[:start+n]
	buf := dst[start:]
	buf[0] = 1
	off := 1
	if includeTokens {
		binary.BigEndian.PutUint32(buf[off:], uint32(len(tokens))) //nolint:gosec
		off += 4
		binary.BigEndian.PutUint32(buf[off:], uint32(dim)) //nolint:gosec
		off += 4
		for _, tok := range tokens {
			for _, f := range tok {
				binary.BigEndian.PutUint32(buf[off:], math.Float32bits(f))
				off += 4
			}
		}
	} else {
		binary.BigEndian.PutUint32(buf[off:], 0)
		off += 4
		binary.BigEndian.PutUint32(buf[off:], 0)
		off += 4
	}
	if len(metaJSON) > 0 {
		buf[off] = 1
		off++
		binary.BigEndian.PutUint32(buf[off:], uint32(len(metaJSON))) //nolint:gosec
		off += 4
		off += copy(buf[off:], metaJSON)
	} else {
		buf[off] = 0
		off++
	}
	// Trailing per-document CAS version block (byte-identical when 0).
	if version != 0 {
		buf[off] = 1
		off++
		binary.BigEndian.PutUint64(buf[off:], version)
	} else {
		buf[off] = 0
	}
	return dst
}

// DecodeMVGetResult reads a result produced by EncodeMVGetResult. found is false
// for an absent document (the not-found flag). Token floats are deep-copied.
func DecodeMVGetResult(body []byte) (found bool, tokens [][]float32, payload vtypes.Metadata, err error) {
	found, tokens, payload, _, _, err = DecodeMVGetResultAt(body, 0)
	return found, tokens, payload, err
}

// DecodeMVGetResultV is DecodeMVGetResult plus the document's per-document CAS
// version (0 for an absent document or a legacy result with no version block).
func DecodeMVGetResultV(body []byte) (found bool, tokens [][]float32, payload vtypes.Metadata, version uint64, err error) {
	found, tokens, payload, version, _, err = DecodeMVGetResultAt(body, 0)
	return found, tokens, payload, version, err
}

// DecodeMVGetResultAt decodes a single mv-get-result record (the [found:u8]+body
// shape produced by EncodeMVGetResult) starting at body[off], returning the
// decoded fields and the offset just past the record. Shared by DecodeMVGetResult
// (the single-get path) and the MV batch result decoder (which reads one such
// record per row after the row's id) so the per-row wire layout stays defined in
// one place. MV has NO ttl. Token floats are deep-copied. Fails loud on
// truncation, exactly like the original single-get decoder.
func DecodeMVGetResultAt(body []byte, off int) (found bool, tokens [][]float32, meta vtypes.Metadata, version uint64, next int, err error) {
	if len(body) < off+1 {
		return false, nil, nil, 0, off, ErrVectorArgsTruncated
	}
	if body[off] == 0 {
		return false, nil, nil, 0, off + 1, nil
	}
	off++
	if len(body) < off+4+4 {
		return false, nil, nil, 0, off, ErrVectorArgsTruncated
	}
	numTokens := int(binary.BigEndian.Uint32(body[off:]))
	off += 4
	dim := int(binary.BigEndian.Uint32(body[off:]))
	off += 4
	// numTokens*dim*4 OVERFLOWS for wire-sized factors: numTokens = dim = 2^31
	// wraps the product to exactly 0, which turned the byte check below into a
	// no-op and let a 22-byte body reach make([][]float32, 2^31) — a 51.5GB
	// reservation, and an out-of-memory abort is not an error a caller can reject
	// the frame on. So bound each factor against the bytes that actually REMAIN
	// before multiplying, and test the product by division instead of computing it.
	//
	// A token costs dim*4 bytes, so for dim > 0 the body caps numTokens directly.
	// For dim == 0 a token carries NO bytes and the body cannot bound the count at
	// all, so a count above the remaining byte count is refused on its face: such a
	// record (zero-width token vectors) is degenerate and no encoder emits it —
	// appendMVGetResultV only writes a non-zero dim when it writes tokens.
	rem := len(body) - off
	if numTokens < 0 || dim < 0 || numTokens > rem {
		return false, nil, nil, 0, off, ErrVectorArgsTruncated
	}
	need := 0
	if numTokens > 0 && dim > 0 {
		if dim > rem/4 { // one token's floats alone overrun the body
			return false, nil, nil, 0, off, ErrVectorArgsTruncated
		}
		perToken := dim * 4 // <= rem: cannot overflow
		if numTokens > (rem-1)/perToken {
			return false, nil, nil, 0, off, ErrVectorArgsTruncated
		}
		need = numTokens * perToken // <= rem-1: cannot overflow
	}
	if len(body) < off+need+1 {
		return false, nil, nil, 0, off, ErrVectorArgsTruncated
	}
	if numTokens > 0 {
		tokens = make([][]float32, numTokens)
		for ti := 0; ti < numTokens; ti++ {
			tok := make([]float32, dim)
			for j := 0; j < dim; j++ {
				tok[j] = math.Float32frombits(binary.BigEndian.Uint32(body[off:]))
				off += 4
			}
			tokens[ti] = tok
		}
	}
	metaPresent := body[off]
	off++
	if metaPresent == 1 {
		if len(body) < off+4 {
			return false, nil, nil, 0, off, ErrVectorArgsTruncated
		}
		mlen := int(binary.BigEndian.Uint32(body[off:]))
		off += 4
		if len(body) < off+mlen {
			return false, nil, nil, 0, off, ErrVectorArgsTruncated
		}
		m := make(vtypes.Metadata)
		if err := json.Unmarshal(body[off:off+mlen], &m); err != nil {
			return false, nil, nil, 0, off, fmt.Errorf("ops: decode mv get payload: %w", err)
		}
		meta = m
		off += mlen
	}
	// Optional trailing per-document CAS version block. A legacy result (no version
	// block) ends right after the payload; treat the absence of the present byte as
	// "no version" (back-compat). A present byte == 1 carries the u64.
	if off < len(body) {
		verPresent := body[off]
		off++
		if verPresent == 1 {
			if len(body) < off+8 {
				return false, nil, nil, 0, off, ErrVectorArgsTruncated
			}
			version = binary.BigEndian.Uint64(body[off:])
			off += 8
		}
	}
	return true, tokens, meta, version, off, nil
}

// MVGetBatchRow is one row of a vector_mv_get_batch result: the requested id plus
// the same projected fields a single vector_mv_get carries (Found is the
// not-found FLAG, never an error). For a not-found id only ID/Found are
// meaningful. Tokens/Meta follow the with_vector/with_payload projection applied
// at fetch time. Mirrors the named GetBatchRow — the MV row carries a token
// matrix (Tokens [][]float32) instead of a per-space vectors map, and has NO ttl
// (multi-vector documents have none).
type MVGetBatchRow struct {
	ID      uint64
	Found   bool
	Tokens  [][]float32
	Meta    vtypes.Metadata
	Version uint64 // per-document CAS version (>=1 for a found doc; 0 = absent/unknown)
}

// EncodeMVGetBatchResult serializes a per-partition vector_mv_get_batch result.
// Wire: [n:u32] then for each row [id:u64] followed by the SAME [found:u8]+body
// record EncodeMVGetResult produces (so a batch row is just id + a single mv-get
// result). Rows preserve the order the handler was given. A not-found id is a
// found=0 record (NEVER an op error) — the coordinator derives the global missing
// set from absent ids. Mirrors EncodeNamedGetBatchResult (no ttl).
func EncodeMVGetBatchResult(rows []MVGetBatchRow) []byte {
	// Presize: count header + per-row (id + token-matrix record). The estimate is
	// exact except the meta JSON (small, unknown without marshaling); any shortfall
	// just triggers a normal append regrowth. This collapses the old ~log(rows)
	// buffer reallocations + a throwaway per-row slice into a single allocation in
	// the common case — the token matrices make those copies expensive.
	est := 4
	for i := range rows {
		est += 8 + estimateMVGetRowSize(&rows[i])
	}
	buf := slices.Grow([]byte(nil), est)
	buf = append(buf, 0, 0, 0, 0)
	binary.BigEndian.PutUint32(buf, uint32(len(rows))) //nolint:gosec
	var idbuf [8]byte
	for i := range rows {
		binary.BigEndian.PutUint64(idbuf[:], rows[i].ID)
		buf = append(buf, idbuf[:]...)
		// EncodeMVGetResultV always frames the version present byte, so the batch
		// record self-delimits; append it directly (no intermediate slice).
		buf = appendMVGetResultV(buf, rows[i].Found, rows[i].Tokens, rows[i].Meta, true, true, rows[i].Version)
	}
	return buf
}

// estimateMVGetRowSize returns a near-exact estimate of a batch row's encoded size
// (every field except the meta JSON, unknown without marshaling). Presize only.
func estimateMVGetRowSize(r *MVGetBatchRow) int {
	if !r.Found {
		return 1
	}
	n := 1 + 4 + 4 + 1 + 1 + 8 // found, numTokens, dim, metaPresent, verPresent, version
	if len(r.Tokens) > 0 {
		n += len(r.Tokens) * len(r.Tokens[0]) * 4
	}
	return n
}

// DecodeMVGetBatchResult reads a result produced by EncodeMVGetBatchResult. Fails
// loud on truncation or a declared row count that overruns the buffer. A zero-row
// result yields an empty (non-nil) slice. Each row's projected fields are exactly
// what the encoder carried. Mirrors DecodeNamedGetBatchResult (no ttl).
func DecodeMVGetBatchResult(body []byte) ([]MVGetBatchRow, error) {
	if len(body) < 4 {
		return nil, ErrVectorArgsTruncated
	}
	n := int(binary.BigEndian.Uint32(body))
	off := 4
	// Bound the DECLARED row count before reserving for it: the smallest a row can
	// encode is 9 bytes ([id:u64] + a not-found [found=0:u8], per
	// EncodeMVGetBatchResult/appendMVGetResultV), so no body of this length holds
	// more than (len(body)-4)/9 rows. Without the bound a hostile count reserves
	// memory the body cannot justify, and the per-row truncation checks are too
	// late — they run after the reservation. Mirrors DecodeVectorGetBatchResult.
	if !CountFitsIn(n, len(body)-off, 9) {
		return nil, ErrVectorArgsTruncated
	}
	rows := make([]MVGetBatchRow, 0, n)
	for i := 0; i < n; i++ {
		if len(body) < off+8 {
			return nil, ErrVectorArgsTruncated
		}
		id := binary.BigEndian.Uint64(body[off:])
		off += 8
		found, tokens, meta, version, next, err := DecodeMVGetResultAt(body, off)
		if err != nil {
			return nil, err
		}
		off = next
		rows = append(rows, MVGetBatchRow{
			ID:      id,
			Found:   found,
			Tokens:  tokens,
			Meta:    meta,
			Version: version,
		})
	}
	return rows, nil
}
