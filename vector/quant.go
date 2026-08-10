// SPDX-License-Identifier: Apache-2.0

package vector

import (
	"encoding/binary"
	"math"
	"math/bits"
)

// QuantMode selects the vector quantization scheme for an index.
type QuantMode uint8

const (
	// QuantNone stores full-precision float32 vectors with no quantization.
	// This is the default.
	QuantNone QuantMode = iota
	// QuantSQ8 stores scalar int8 codes (4× smaller than float32) and rescores
	// the over-collected candidate set on full-precision float32. Cosine-scope
	// for v1 (see the quantization design spec).
	QuantSQ8
	// QuantBQ1 stores 1-bit-per-dimension sign codes (32× smaller than float32),
	// navigates the graph by Hamming distance, and rescores on float32. Best for
	// high-dimensional normalized embeddings. Cosine-scope for v1.
	QuantBQ1
	// QuantPQ stores product-quantization codes (m bytes/vector, nbits=8). Used by
	// the IVF-PQ index mode: the residual codebooks + per-cell ADC LUT live on the
	// pq codec and are driven by the IVF (see pq.go). The newQuantizer adapter for
	// this mode exists so the arena sizes the codes side-array to CodeLen()==m and
	// can Encode on insert; the residual LUT scoring path is IVF-driven, not via
	// the centroid-agnostic Distance below.
	QuantPQ
	// QuantSQ is the TRAINED, metric-agnostic scalar quantizer (trainedSQ). Unlike
	// the legacy fixed-scale QuantSQ8 (a 1/127 Cosine-only fast-path), it learns a
	// per-dimension [min,max] range from a build sample and scores asymmetrically
	// under the index's ACTUAL metric (Cosine/L2/DotProduct). SQBits selects the
	// bit-depth (8-bit only). Added AFTER QuantPQ so existing on-disk
	// QuantNone/SQ8/BQ1/PQ enum values are unchanged. See sq.go.
	QuantSQ
	// QuantPRQ is PRODUCT-RESIDUAL quantization: a stack of PRQLayers (default 2)
	// product-quantizer layers where each layer quantizes the RESIDUAL of the
	// previous one. Code = PRQLayers·m bytes; reconstruction = Σ layer
	// reconstructions; ADC = sum of the per-layer LUTs (the additive approximation,
	// since HNSW's exact float rescore fixes the final ranking). Strictly higher
	// accuracy than plain PQ at the same m. Added AFTER QuantSQ so existing on-disk
	// QuantNone/SQ8/BQ1/PQ/SQ enum values are unchanged. See prq.go.
	QuantPRQ
)

// defaultPQM derives a sensible sub-quantizer count for dim when the IVF config
// does not specify one: the largest power-of-two m that divides dim and keeps each
// sub-vector roughly 4–16 dims (favouring ~8). Config wiring may override
// it; this keeps the arena-sizing path well-defined for QuantPQ. Falls back to a
// divisor of dim (or 1) when no power-of-two fits.
func defaultPQM(dim int) int {
	if dim <= 0 {
		return 1
	}
	for _, m := range []int{dim / 8, 16, 8, 4, 2, 1} {
		if m > 0 && dim%m == 0 {
			return m
		}
	}
	return 1
}

// QuantStore selects where the full-precision float32 vectors live when
// quantization is enabled. Codes always stay resident in RAM.
type QuantStore uint8

const (
	// QuantInRAM keeps float32 vectors on the Go heap alongside the codes.
	// Fastest rescore, but uses more memory than no quantization. Default.
	QuantInRAM QuantStore = iota
	// QuantMmap stores float32 vectors in a memory-mapped file so only the
	// int8 codes are resident; the rescore stage pages vectors in on demand.
	// This is the configuration that reduces resident memory.
	QuantMmap
)

// defaultRescoreFactor is the candidate over-collection multiple used when
// Config.RescoreFactor is 0: the exact rescore stage re-ranks RescoreFactor*k
// candidates.
const defaultRescoreFactor = 3

// queryCode is an opaque, per-search query handle produced by a quantizer's
// PrepareQuery and consumed by its Distance. Each quantizer stores its own
// concrete type (e.g. sq8 keeps the float32 query for asymmetric distance).
type queryCode = any

// quantizer compresses a float32 vector to a fixed-length code and computes an
// approximate distance between a prepared query and a stored code. Smaller
// distances mean "more similar", matching distFunc semantics. The approximate
// distance only orders graph traversal; the final ranking is always rescored
// on exact float32 (see hnsw.rescore).
type quantizer interface {
	Encode(dst []byte, vec []float32) // float32 -> code; len(dst) == CodeLen()
	CodeLen() int                     // bytes per code
	PrepareQuery(q []float32) queryCode
	Distance(qc queryCode, code []byte) float32
	// CodeDistance is the symmetric approximate distance between two stored
	// codes (no float side). Used by the quantized build path so neighbor
	// selection reads only codes (4x less memory traffic than float32 vectors).
	CodeDistance(a, b []byte) float32
}

// newQuantizer returns the quantizer for mode, or nil for QuantNone. For QuantPQ
// (the dense/HNSW PQ mode) m is the configured sub-quantizer count (0 ⇒
// defaultPQM(dim)) and metric drives the ADC LUT orientation/offset. For QuantSQ
// (the trained metric-agnostic scalar quantizer) bits selects the bit-depth (0 ⇒
// 8) and metric drives the asymmetric distance; the returned trainedSQ starts
// UNTRAINED (nil ranges) and the build-time auto-train learns + swaps ranges in.
// m/bits/layers/metric are ignored for the modes that do not use them. The PQ and
// PRQ adapters also start UNTRAINED: BuildConcurrent (or the incremental auto-train)
// trains them and re-encodes every slot. For QuantPRQ, m is the per-layer
// sub-quantizer count (0 ⇒ defaultPQM(dim)) and layers the PRQ layer count
// (0 ⇒ defaultPRQLayers); metric drives the per-layer ADC LUT orientation.
// pqNBits is the per-subspace PQ code width for QuantPQ: 0/8 ⇒ the 8-bit codec
// (byte-identical), 4 ⇒ the 4-bit LUT16 fast-scan codec. Ignored for non-PQ modes.
func newQuantizer(mode QuantMode, dim, m, bits, layers, pqNBits int, metric Metric) quantizer {
	switch mode {
	case QuantSQ8:
		return newSQ8(dim)
	case QuantBQ1:
		return newBQ1(dim)
	case QuantPQ:
		if m <= 0 {
			m = defaultPQM(dim)
		}
		return newPQQuantizer(dim, m, pqNBits, metric)
	case QuantSQ:
		return newTrainedSQ(dim, bits, metric)
	case QuantPRQ:
		if m <= 0 {
			m = defaultPQM(dim)
		}
		if layers <= 0 {
			layers = defaultPRQLayers
		}
		return newPRQQuantizer(dim, m, layers, metric)
	default:
		return nil
	}
}

// Compile-time assertion that the quantizers satisfy quantizer.
var (
	_ quantizer = (*sq8)(nil)
	_ quantizer = (*bq1)(nil)
	_ quantizer = (*pqQuantizer)(nil)
	_ quantizer = (*trainedSQ)(nil)
	_ quantizer = (*prqQuantizer)(nil)
)

// pqQuantizer adapts the residual-aware pq codec to the centroid-agnostic
// quantizer interface so the arena can size the codes side-array (CodeLen()==m)
// and Encode on insert. The IVF-PQ index drives the residual train/encode/LUT
// path directly on the embedded *pq (see pq.go); this adapter's
// Distance/CodeDistance treat the codec as a plain (non-residual) ADC quantizer,
// which is only exercised if a QuantPQ index is built outside the IVF residual
// path. The codec starts untrained (nil codebooks); the owner trains it via
// trainPQ and swaps the result in before any Encode/Distance call.
type pqQuantizer struct {
	codec *pq
	dim   int
	m     int

	// nbits is the per-subspace code width: 0/8 ⇒ the 8-bit codec (byte-identical),
	// 4 ⇒ the 4-bit LUT16 fast-scan codec. When 4, codec4 wraps codec (after
	// training) and Encode/CodeLen/PrepareQuery/Distance/reconstruct route through
	// the nibble-packed 4-bit path; the 8-bit path is untouched when nbits is 0/8.
	nbits  int
	codec4 *pq4 // non-nil only when nbits==4 AND trained
}

// newPQQuantizer returns an untrained PQ-backed quantizer for dim/m/nbits/metric.
// The codec's codebooks are populated later (BuildConcurrent trains them and swaps
// the result in); until then only CodeLen() is meaningful. nbits 0/8 ⇒ the 8-bit
// codec, 4 ⇒ the 4-bit LUT16 codec (16 sub-centroids/subspace, nibble-packed).
func newPQQuantizer(dim, m, nbits int, metric Metric) *pqQuantizer {
	return &pqQuantizer{
		codec: &pq{m: m, dsub: dim / m, dim: dim, metric: metric, nbits: nbits},
		dim:   dim,
		m:     m,
		nbits: nbits,
	}
}

// CodeLen returns the per-slot code length the arena sizes its codes side-array
// to: m bytes for the 8-bit codec, ceil(m/2) for the 4-bit (nibble-packed) codec.
func (q *pqQuantizer) CodeLen() int {
	if q.nbits == 4 {
		return pq4CodeLen(q.m)
	}
	return q.m
}

// trained reports whether the codec's codebooks have been populated (via
// setCodec after trainPQ). An untrained codec cannot Encode or score a code; the
// HNSW owner (build/search) checks this and uses EXACT float distance until the
// codebooks exist (mirrors IVF's untrained=exact policy). See newPQQuantizer.
func (q *pqQuantizer) trained() bool { return q.codec != nil && q.codec.codebooks != nil }

// setCodec swaps the trained *pq (from trainPQ) into the adapter. Called once by
// BuildConcurrent after training; subsequent Encode/Distance calls use it. For the
// 4-bit codec it also wraps the *pq in a *pq4 so the nibble-packed Encode + the
// quantized fast-scan LUT engage.
func (q *pqQuantizer) setCodec(p *pq) {
	q.codec = p
	if q.nbits == 4 {
		q.codec4 = &pq4{codec: p}
	}
}

// codebooks returns the trained codec's codebooks ([m][≤256][dsub]) for snapshot
// serialization, or nil when untrained. The slices alias the codec; callers must
// only read them.
func (q *pqQuantizer) codebooks() [][][]float32 {
	if q.codec == nil {
		return nil
	}
	return q.codec.codebooks
}

// rotation returns the trained codec's OPQ rotation R (dim×dim flat, row-major)
// for snapshot serialization, or nil when OPQ is off / untrained. The slice
// aliases the codec; callers must only read it.
func (q *pqQuantizer) rotation() []float32 {
	if q.codec == nil {
		return nil
	}
	return q.codec.rotation
}

// loadCodebooks rebuilds the codec from serialized codebooks (the inverse of
// codebooks(), used by snapshot/persist restore). dim/metric come from the
// adapter; m is inferred from len(cb). After this trained() is true and the
// caller's re-encode-from-vecs produces the same codes as before the snapshot.
// r is the OPQ rotation restored VERBATIM (nil when OPQ was off) so the restored
// codec rotates/un-rotates bit-identically — re-encoded codes match the originals.
func (q *pqQuantizer) loadCodebooks(cb [][][]float32, r []float32) {
	m := len(cb)
	q.codec = &pq{
		m:         m,
		dsub:      q.dim / m,
		dim:       q.dim,
		metric:    q.codec.metric,
		nbits:     q.nbits,
		codebooks: cb,
		rotation:  r,
	}
	q.m = m
	if q.nbits == 4 {
		q.codec4 = &pq4{codec: q.codec}
	}
}

// Encode writes the m-byte PQ code for vec into dst (len(dst) == CodeLen()).
// SAFE before training: an UNTRAINED codec (nil codebooks) leaves dst untouched
// (the arena's auto-encode on PutAt/Insert thus writes a zero placeholder, which
// is never scored — the scorer uses exact float distance until trained). Once
// trained (post-BuildConcurrent), it writes the real PQ code.
func (q *pqQuantizer) Encode(dst []byte, vec []float32) {
	if !q.trained() {
		return
	}
	if q.codec4 != nil {
		q.codec4.encodeInto(dst, vec)
		return
	}
	q.codec.encodeInto(dst, vec)
}

// PrepareQuery builds the ADC query handle for q. For the 8-bit codec this is the
// flat float ADC LUT; for the 4-bit codec it is the quantized uint8 fast-scan LUT
// (*lut16). For the IVF-PQ residual path the IVF builds the per-cell LUT directly
// via codec.queryLUT(q − centroid); this adapter path treats q as already
// centroid-adjusted.
func (q *pqQuantizer) PrepareQuery(query []float32) queryCode {
	if q.codec4 != nil {
		return q.codec4.buildLUT16(query)
	}
	return q.codec.queryLUT(query)
}

// Distance returns the ADC distance between the prepared query handle and a code.
// Lower = nearer, matching distFunc semantics. The 4-bit path uses the scalar LUT16
// ADC (adcScalar); the bulk fast-scan (block) kernel is exercised by batch scoring,
// while HNSW's graph traversal scores one neighbor at a time here.
func (q *pqQuantizer) Distance(qc queryCode, code []byte) float32 {
	if l, ok := qc.(*lut16); ok {
		return l.adcScalar(code)
	}
	return q.codec.adc(qc.([]float32), code)
}

// reconstruct returns the APPROXIMATE float32 vector for a PQ code: it
// concatenates the m selected sub-centroids and un-rotates (Rᵀ) when OPQ is on,
// landing back in the ORIGINAL space. Used by hnsw.vecFor on the float-drop path
// (PQDropVecs) to serve Get/MMR/Recommend/Discover/reshard without resident
// floats. The returned slice is freshly allocated (codec.reconstruct allocates).
// SAFE only when trained() — the float-drop path is post-build, so it always is.
func (q *pqQuantizer) reconstruct(code []byte) []float32 {
	if q.codec4 != nil {
		return q.codec4.reconstruct(code)
	}
	return q.codec.reconstruct(code)
}

// validateCodes verifies that a verbatim-restored codes buffer (float-drop
// snapshot/sidecar) is self-consistent with the loaded codec before any slot is
// scored: the per-slot code length must equal CodeLen(), the buffer must be an
// exact multiple of it, and every sub-code index must fall inside its subspace's
// codebook. Without this a corrupt code byte would index codebooks[s] out of
// bounds in reconstruct()/adc() and panic at query time. Returns nil when the
// codec is untrained (nothing to validate against) or codes is empty.
func (q *pqQuantizer) validateCodes(codes []byte, codeLen int) error {
	if !q.trained() || len(codes) == 0 {
		return nil
	}
	if codeLen != q.CodeLen() || codeLen <= 0 || len(codes)%codeLen != 0 {
		return ErrSnapshotFormat
	}
	cb := q.codec.codebooks
	m := q.codec.m
	for off := 0; off+codeLen <= len(codes); off += codeLen {
		code := codes[off : off+codeLen]
		for s := 0; s < m; s++ {
			var idx int
			if q.nbits == 4 {
				idx = subCodeAt(code, s)
			} else {
				idx = int(code[s])
			}
			if idx >= len(cb[s]) {
				return ErrSnapshotFormat
			}
		}
	}
	return nil
}

// CodeDistance is the symmetric approximate distance between two codes: it
// reconstructs the first code's sub-centroids into a query handle, then ADC-scores
// the second. The 4-bit path reconstructs from nibbles and scores with the LUT16
// scalar ADC; the 8-bit path uses the float ADC LUT.
func (q *pqQuantizer) CodeDistance(a, b []byte) float32 {
	if q.codec4 != nil {
		return q.codec4.buildLUT16(q.codec4.reconstruct(a)).adcScalar(b)
	}
	return q.codec.adc(q.codec.queryLUT(q.codec.reconstruct(a)), b)
}

// prqQuantizer adapts the product-residual prq codec to the quantizer interface,
// mirroring pqQuantizer. The arena sizes the per-slot codes side-array to
// CodeLen() == L*m and Encode-on-insert writes the concatenated layer sub-codes.
// The codec starts UNTRAINED (nil codec); the owner trains it via trainPRQ and
// swaps the result in (setCodec) before any Encode/Distance call. Until then
// trained() is false and the HNSW owner navigates on EXACT float32.
type prqQuantizer struct {
	codec  *prq
	dim    int
	m      int
	layers int
	metric Metric
}

// newPRQQuantizer returns an UNTRAINED PRQ-backed quantizer for dim/m/layers/metric.
// CodeLen() is meaningful immediately (L*m); the codebooks are populated later by
// the build-time trainPRQ + setCodec.
func newPRQQuantizer(dim, m, layers int, metric Metric) *prqQuantizer {
	return &prqQuantizer{dim: dim, m: m, layers: layers, metric: metric}
}

// CodeLen returns L*m: one sub-code byte per subspace per layer. Meaningful before
// training so the arena sizes the codes side-array.
func (q *prqQuantizer) CodeLen() int { return q.layers * q.m }

// trained reports whether the codec has been populated (via setCodec after
// trainPRQ). An untrained codec cannot Encode or score; the HNSW owner checks this
// and uses EXACT float distance until the codebooks exist.
func (q *prqQuantizer) trained() bool { return q.codec != nil && len(q.codec.layers) > 0 }

// setCodec swaps the trained *prq (from trainPRQ) into the adapter. Called once by
// the build path after training.
func (q *prqQuantizer) setCodec(p *prq) { q.codec = p }

// codebooks returns all L layers' codebooks ([L][m][≤256][dsub]) for snapshot
// serialization, or nil when untrained. The slices alias the codec; callers must
// only read them.
func (q *prqQuantizer) codebooks() [][][][]float32 {
	if !q.trained() {
		return nil
	}
	out := make([][][][]float32, len(q.codec.layers))
	for l := range q.codec.layers {
		out[l] = q.codec.layers[l].codebooks
	}
	return out
}

// rotation returns the codec's OPQ rotation R (layer 0's, dim×dim flat row-major)
// for snapshot serialization, or nil when OPQ is off / untrained.
func (q *prqQuantizer) rotation() []float32 {
	if !q.trained() {
		return nil
	}
	return q.codec.rotation()
}

// loadCodebooks rebuilds the codec from serialized per-layer codebooks (the inverse
// of codebooks(), used by snapshot/persist restore). dim/m/metric come from the
// adapter; the layer count is inferred from len(cb). r is the OPQ rotation restored
// VERBATIM into layer 0 (nil when OPQ was off, nil for layers 1..L-1) so the
// restored codec rotates/un-rotates bit-identically — re-encoded codes match.
func (q *prqQuantizer) loadCodebooks(cb [][][][]float32, r []float32) {
	l := len(cb)
	layers := make([]*pq, l)
	for i := 0; i < l; i++ {
		var rot []float32
		if i == 0 {
			rot = r // only layer 0 carries the rotation (applied once)
		}
		m := len(cb[i])
		layers[i] = &pq{
			m:         m,
			dsub:      q.dim / m,
			dim:       q.dim,
			metric:    q.metric,
			codebooks: cb[i],
			rotation:  rot,
		}
	}
	q.codec = &prq{layers: layers, l: l, m: len(cb[0]), dim: q.dim, metric: q.metric}
	q.layers = l
	q.m = len(cb[0])
}

// Encode writes the L*m-byte PRQ code for vec into dst. SAFE before training: an
// UNTRAINED codec leaves dst untouched (the arena's auto-encode writes a zero
// placeholder, never scored — the scorer uses exact float distance until trained).
func (q *prqQuantizer) Encode(dst []byte, vec []float32) {
	if !q.trained() {
		return
	}
	q.codec.encodeInto(dst, vec)
}

// PrepareQuery builds the per-layer ADC LUTs for query q (rotated once inside).
func (q *prqQuantizer) PrepareQuery(query []float32) queryCode { return q.codec.PrepareQuery(query) }

// Distance returns the summed-LUT ADC distance between the prepared query and a
// code. Lower = nearer, matching distFunc semantics.
func (q *prqQuantizer) Distance(qc queryCode, code []byte) float32 {
	return q.codec.Distance(qc.(*prqLUTs), code)
}

// CodeDistance is the symmetric approximate distance between two PRQ codes,
// delegating to the codec (reconstruct-a + summed-LUT ADC over b), consistent with
// Distance so the quantized build graph aligns with the search traversal.
func (q *prqQuantizer) CodeDistance(a, b []byte) float32 { return q.codec.CodeDistance(a, b) }

// sq8 is a scalar int8 quantizer. Each float32 component is mapped through a
// symmetric global scale (1/127) and rounded to the nearest int8, giving a 4×
// reduction over float32 storage. It targets Cosine collections, whose vectors
// are pre-normalized so components lie in [-1, 1]; values outside that range
// are clamped. Codes are stored one int8 per dimension, packed into a []byte.
type sq8 struct {
	dim int
}

// sq8InvScale dequantizes an int8 code component back toward [-1, 1].
const sq8InvScale = float32(1.0 / 127.0)

// newSQ8 returns a scalar int8 quantizer for dim-dimensional vectors.
func newSQ8(dim int) *sq8 { return &sq8{dim: dim} }

// CodeLen returns the encoded length in bytes: one int8 per dimension.
func (q *sq8) CodeLen() int { return q.dim }

// Encode writes the int8 code for vec into dst (len(dst) must be CodeLen()).
// Each component is clamped to [-1, 1], scaled by 127, and rounded to the
// nearest integer, then stored as an int8 reinterpreted into the byte slot.
func (q *sq8) Encode(dst []byte, vec []float32) {
	for i, x := range vec {
		if x > 1 {
			x = 1
		} else if x < -1 {
			x = -1
		}
		//nolint:gosec // x is clamped to [-1,1], so x*127 ∈ [-127,127] fits int8; the byte cast reinterprets the int8 for storage.
		dst[i] = byte(int8(math.Round(float64(x) * 127)))
	}
}

// PrepareQuery returns the per-search query handle. SQ8 uses asymmetric
// distance — the query stays in full float32 precision — so the handle is the
// query slice itself. The caller must not mutate it for the search's duration.
func (q *sq8) PrepareQuery(query []float32) queryCode { return query }

// Distance returns the approximate Cosine distance (1 - dot) between the
// prepared float32 query and an int8 code. The code side is dequantized by the
// 1/127 scale; the query side keeps full precision. Smaller = more similar,
// matching distFunc semantics. Cosine-scope for v1 (see quantization spec).
func (q *sq8) Distance(qc queryCode, code []byte) float32 {
	return 1 - sq8Dot(qc.([]float32), code)*sq8InvScale
}

// CodeDistance returns the symmetric approximate Cosine distance between two
// int8 codes: both sides are dequantized by 1/127, so the dot is scaled by
// 1/127². Used by the quantized build path (neighbor selection over codes).
func (q *sq8) CodeDistance(a, b []byte) float32 {
	return 1 - float32(sq8CodeDot(a, b))*sq8InvScale*sq8InvScale
}

// sq8Dot computes the asymmetric dot product Σ query[i]*float32(int8(code[i])).
// Defaults to the portable scalar implementation; an init() in
// distance_amd64.go swaps it for an AVX2 kernel on capable CPUs.
var sq8Dot = sq8DotScalar

// sq8CodeDot computes the symmetric integer dot Σ int8(a[i])*int8(b[i]) of two
// codes. Scalar by default; swapped for an AVX2 kernel in distance_amd64.go.
var sq8CodeDot = sq8CodeDotScalar

// sq8CodeDotScalar is the portable Σ int8(a[i])*int8(b[i]) kernel.
func sq8CodeDotScalar(a, b []byte) int32 {
	var dot int32
	for i := range a {
		//nolint:gosec // a[i],b[i] hold int8 codes stored in bytes; the int8 casts reinterpret them.
		dot += int32(int8(a[i])) * int32(int8(b[i]))
	}
	return dot
}

// sq8DotScalar is the portable Σ query[i]*float32(int8(code[i])) kernel.
// len(query) must equal len(code).
func sq8DotScalar(query []float32, code []byte) float32 {
	var dot float32
	for i, c := range code {
		//nolint:gosec // c holds an int8 code stored in a byte; the int8 cast reinterprets it back to its signed value.
		dot += query[i] * float32(int8(c))
	}
	return dot
}

// bq1 is a binary (1-bit) quantizer. Each component contributes one sign bit
// (set when positive), packed LSB-first into ceil(dim/8) bytes — a 32× size
// reduction over float32. Graph traversal uses symmetric Hamming distance
// (popcount of the XOR); the exact float32 rescore stage recovers the final
// ranking. Cosine-scope for v1.
type bq1 struct {
	dim     int
	codeLen int
}

// newBQ1 returns a binary quantizer for dim-dimensional vectors.
func newBQ1(dim int) *bq1 { return &bq1{dim: dim, codeLen: (dim + 7) / 8} }

// CodeLen returns the encoded length in bytes: ceil(dim/8).
func (q *bq1) CodeLen() int { return q.codeLen }

// Encode writes the sign-bit code for vec into dst (len(dst) must be CodeLen()).
// Bit i is set when vec[i] > 0, stored at dst[i/8] bit (i%8), LSB-first.
func (q *bq1) Encode(dst []byte, vec []float32) {
	for i := range dst {
		dst[i] = 0
	}
	for i, x := range vec {
		if x > 0 {
			dst[i>>3] |= 1 << (uint(i) & 7)
		}
	}
}

// PrepareQuery packs the query into the same sign-bit form so Distance is a
// symmetric Hamming comparison. The returned handle is the packed query bits.
func (q *bq1) PrepareQuery(query []float32) queryCode {
	bitsBuf := make([]byte, q.codeLen)
	q.Encode(bitsBuf, query)
	return bitsBuf
}

// Distance returns the Hamming distance between the prepared query bits and a
// code: the number of differing sign bits (popcount of the XOR). Smaller =
// more similar, matching distFunc semantics.
func (q *bq1) Distance(qc queryCode, code []byte) float32 {
	return float32(bq1Hamming(qc.([]byte), code))
}

// CodeDistance returns the symmetric Hamming distance between two sign-bit
// codes — the same metric bq1 already navigates with, so the build path needs no
// special case beyond reading both sides as codes.
func (q *bq1) CodeDistance(a, b []byte) float32 {
	return float32(bq1Hamming(a, b))
}

// bq1Hamming returns the number of differing bits between qbits and code (both
// CodeLen bytes). It folds 8 bytes at a time into a uint64 popcount (which the
// compiler lowers to POPCNT on amd64), with a byte tail for the remainder.
func bq1Hamming(qbits, code []byte) int {
	var diff int
	i, n := 0, len(code)
	for ; i+8 <= n; i += 8 {
		diff += bits.OnesCount64(binary.LittleEndian.Uint64(qbits[i:]) ^ binary.LittleEndian.Uint64(code[i:]))
	}
	for ; i < n; i++ {
		diff += bits.OnesCount8(qbits[i] ^ code[i])
	}
	return diff
}
