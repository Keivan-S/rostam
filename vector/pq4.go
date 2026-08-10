// SPDX-License-Identifier: Apache-2.0

package vector

import "math"

// 4-bit LUT16 Product Quantization with ScaNN/FAISS-style in-register fast-scan.
//
// A 4-bit PQ uses 16 sub-centroids per subspace (nbits=4) instead of the 8-bit
// codec's 256 (pq.go). Two sub-codes pack into one byte (the low nibble is the
// even subspace, the high nibble the odd one), so a code is CodeLen = ceil(m/2) =
// (m+1)/2 bytes — half the 8-bit code at the cost of a coarser quantization the
// exact float32 rescore recovers from. 4-bit PQ is OPT-IN via Config.PQNBits=4;
// PQNBits 0/8 is the existing 8-bit codec, byte-identical.
//
// CANDIDATE-GEN ADC: the 4-bit fast-scan is the candidate-generation ADC for the
// IVF-PQ batched-list query path (ivf.gatherADCLocked — see ivf.go: for each probed
// cell it builds a uint8 lut16 on the residual q−centroid and scores the cell's
// slot list with fastScanBlockInto, on-the-fly transposing the slots' nibble codes
// into blocks of 32); the existing exact float32 rescore (IVFRerank) is the
// score-aware reorder stage. So 4-bit trades ADC accuracy for ADC SPEED — the
// in-register VPSHUFB table lookup scores 32 database vectors per subspace per
// instruction (fastscan), and the rescore fixes the final ranking.
//
// The codec reuses the 8-bit codec's training (trainCodebooks with k=16, which
// composes with the anisotropic η loss) and the same per-subspace L2 reconstruction
// objective; only the code WIDTH (4 bits, nibble-packed) and the LUT (16 entries
// per subspace, quantized to uint8 for the SIMD path) differ.

// pqCodebookSize4 is the number of sub-centroids per subspace for nbits=4 (LUT16).
const pqCodebookSize4 = 16

// fastScanBlock is the number of database vectors whose codes are transposed into
// one fast-scan block. 32 matches the AVX2 register width: a YMM holds 32 bytes,
// so one VPSHUFB looks up 32 nibbles (one per database vector) against the
// broadcast 16-entry LUT in a single instruction. Codes that do not fill a whole
// block are handled by the scalar tail (fastScanScalar over the unpacked nibbles).
const fastScanBlock = 32

// pq4 is a 4-bit (LUT16) product quantizer. It embeds an 8-bit-style pq codec but
// with codebooks of pqCodebookSize4 (16) sub-centroids per subspace; pq4 owns the
// nibble-packing of codes and the quantized fast-scan LUT. The embedded *pq holds
// the trained codebooks, metric, dim, dsub, m, and (optional) OPQ rotation, so
// encode/queryLUT reuse the exact 8-bit machinery up to the code/LUT width.
type pq4 struct {
	codec *pq // codebooks[s] is the s-th subspace's <=16 sub-centroids
}

// pq4CodeLen returns the packed code length for an m-subspace 4-bit PQ:
// ceil(m/2) — two 4-bit sub-codes per byte (low nibble = even subspace s,
// high nibble = odd subspace s+1; an odd final subspace leaves the high nibble 0).
func pq4CodeLen(m int) int { return (m + 1) / 2 }

// CodeLen returns the packed (nibble) code length in bytes: ceil(m/2).
func (p *pq4) CodeLen() int { return pq4CodeLen(p.codec.m) }

// encodeInto writes the packed 4-bit PQ code for vec into dst (len(dst) ==
// CodeLen()). For each subspace it finds the nearest sub-centroid (L2 over the
// dsub-length sub-vector, the minimal-reconstruction choice — see pq.go header)
// and packs its 4-bit index: even subspaces into the low nibble of byte s/2, odd
// subspaces into the high nibble. OPQ rotation (if any) is applied by reusing the
// 8-bit nibbleCodes path's rotation handling.
func (p *pq4) encodeInto(dst []byte, vec []float32) {
	for i := range dst {
		dst[i] = 0
	}
	p.codec.forEachSubCode(vec, func(s, idx int) {
		// idx ∈ [0,15] by construction (codebook size <= 16).
		nib := byte(idx & 0x0f)
		if s&1 == 0 {
			dst[s>>1] |= nib
		} else {
			dst[s>>1] |= nib << 4
		}
	})
}

// Encode writes the packed 4-bit code into a freshly allocated slice.
func (p *pq4) Encode(vec []float32) []byte {
	dst := make([]byte, p.CodeLen())
	p.encodeInto(dst, vec)
	return dst
}

// reconstruct concatenates the m sub-centroids selected by a PACKED 4-bit code
// into a dim-length vector (un-rotating by Rᵀ when OPQ is on), mirroring
// pq.reconstruct but unpacking nibbles. Used by the float-drop path and
// CodeDistance.
func (p *pq4) reconstruct(code []byte) []float32 {
	c := p.codec
	out := make([]float32, c.dim)
	for s := 0; s < c.m; s++ {
		sub := c.codebooks[s][subCodeAt(code, s)]
		copy(out[s*c.dsub:], sub)
	}
	if c.rotation != nil {
		out = rotateT(c.rotation, out)
	}
	return out
}

// subCodeAt unpacks the 4-bit sub-code for subspace s from a packed code.
func subCodeAt(code []byte, s int) int {
	b := code[s>>1]
	if s&1 == 0 {
		return int(b & 0x0f)
	}
	return int(b >> 4)
}

// lut16 is a quantized fast-scan LUT for a 4-bit PQ query. tbl is a flat
// [m*16]uint8 table: tbl[s*16+c] is the quantized metric contribution of the
// query's s-th sub-vector against sub-centroid c. The float ADC value is
// recovered as bias + scale*Σ tbl[s*16+code[s]] (see lut16.dequant). The
// quantization maps the full float LUT's [min,max] range linearly onto the uint8
// range [0,255] (per the whole LUT, not per subspace, so the scale is shared and
// the integer accumulation is a single affine map of the true ADC sum).
type lut16 struct {
	tbl   []uint8 // [m*16] quantized per-subspace contributions
	m     int
	bias  float32 // dequant offset: lut.min*m + pqMetricOffset(metric)
	scale float32 // dequant slope: (lut.max-lut.min)/255
}

// quantErr is the worst-case absolute ADC error introduced by the uint8 LUT
// quantization, expressed as a multiple of the per-entry quantization step. Each
// of the m subspace contributions is rounded to the nearest of 256 levels, so the
// per-entry error is at most scale/2 and the total over m subspaces is at most
// m*scale/2. Tests assert the scalar 4-bit ADC matches the float reference within
// this bound (see pq4_test.go). Documented here as the LUT-quantization tolerance.

// buildLUT16 quantizes the float ADC LUT for query q into a uint8 fast-scan LUT.
// It first builds the full float [m*16] LUT (reusing the 8-bit queryLUT inner
// product / L2 contributions, but over 16 entries), finds its [min,max] across all
// written entries, and maps each entry linearly to [0,255] via round-to-nearest.
// The shared (whole-LUT) scale means the integer fast-scan accumulation Σ tbl is
// an affine image of the true ADC sum, so a single (bias,scale) dequant recovers
// the float distance. Entries for c >= len(codebook[s]) (small-n training) sit at
// the LUT min so they never spuriously win.
func (p *pq4) buildLUT16(q []float32) *lut16 {
	c := p.codec
	m := c.m
	flut := make([]float32, m*pqCodebookSize4)
	c.queryLUT16Into(flut, q)
	// Range across the WRITTEN entries (per-subspace c < len(cb[s])); unwritten
	// trailing entries are set to the running min below so they never win.
	minV := float32(math.Inf(1))
	maxV := float32(math.Inf(-1))
	for s := 0; s < m; s++ {
		nc := len(c.codebooks[s])
		for cc := 0; cc < nc; cc++ {
			v := flut[s*pqCodebookSize4+cc]
			if v < minV {
				minV = v
			}
			if v > maxV {
				maxV = v
			}
		}
	}
	if math.IsInf(float64(minV), 1) { // empty / untrained: degenerate flat LUT
		minV, maxV = 0, 0
	}
	span := maxV - minV
	scale := span / 255
	var invScale float32
	if span > 0 {
		invScale = 255 / span
	}
	tbl := make([]uint8, m*pqCodebookSize4)
	for s := 0; s < m; s++ {
		nc := len(c.codebooks[s])
		base := s * pqCodebookSize4
		for cc := 0; cc < pqCodebookSize4; cc++ {
			v := minV // default (unwritten / never-encoded slots map to the min)
			if cc < nc {
				v = flut[base+cc]
			}
			q := int32(0)
			if span > 0 {
				q = int32(math.Round(float64((v - minV) * invScale)))
			}
			if q < 0 {
				q = 0
			} else if q > 255 {
				q = 255
			}
			tbl[base+cc] = uint8(q)
		}
	}
	return &lut16{tbl: tbl, m: m, bias: minV*float32(m) + pqMetricOffset(c.metric), scale: scale}
}

// dequant maps a fast-scan integer accumulator (Σ tbl[s*16+code[s]] over m
// subspaces) back to the float ADC distance: bias + scale*acc. This is exact up
// to the per-entry round-to-nearest quantization (<= m*scale/2 absolute error).
func (l *lut16) dequant(acc uint32) float32 {
	return l.bias + l.scale*float32(acc)
}

// adcScalar is the SCALAR 4-bit ADC reference: for a single packed code it sums
// the quantized per-subspace LUT entries and dequantizes. This is the correctness
// reference the AVX2 fast-scan must match (within the integer accumulation, it is
// EXACT against fastScanScalar; both go through the same uint8 tbl). It is also the
// portable scoring path on non-AVX2 CPUs.
func (l *lut16) adcScalar(code []byte) float32 {
	var acc uint32
	for s := 0; s < l.m; s++ {
		acc += uint32(l.tbl[s*pqCodebookSize4+subCodeAt(code, s)])
	}
	return l.dequant(acc)
}

// queryLUT16Into fills dst[:m*16] with the FLOAT ADC contributions for query q
// over the 16-entry 4-bit codebooks. It mirrors pq.queryLUTInto exactly (same OPQ
// rotation, same metric orientation: −dot for IP/Cosine, squared-L2 for L2) but
// strides by 16 instead of 256. Only entries [base, base+len(cb[s])) are written;
// the buildLUT16 caller fills the rest with the LUT min.
func (p *pq) queryLUT16Into(dst []float32, q []float32) {
	var scratch *[]float32
	if p.rotation != nil {
		scratch = getRotScratch(len(q))
		rotateInto(*scratch, p.rotation, q)
		q = *scratch
	}
	negDot := p.metric != L2
	for s := 0; s < p.m; s++ {
		sub := q[s*p.dsub : s*p.dsub+p.dsub]
		cb := p.codebooks[s]
		base := s * pqCodebookSize4
		for c := 0; c < len(cb); c++ {
			if negDot {
				dst[base+c] = -dotProduct(sub, cb[c])
			} else {
				dst[base+c] = l2Squared(sub, cb[c])
			}
		}
	}
	if scratch != nil {
		putRotScratch(scratch)
	}
}

// forEachSubCode finds, for each subspace s, the nearest sub-centroid index (by L2
// over the dsub sub-vector, applying the OPQ rotation once up front) and calls fn(s,
// idx). It is the shared encode core for both the 8-bit byte code and the 4-bit
// nibble code, factored out of encodeInto so the two widths pack the same indices
// differently. idx ∈ [0,len(cb[s])).
func (p *pq) forEachSubCode(vec []float32, fn func(s, idx int)) {
	var scratch *[]float32
	if p.rotation != nil {
		scratch = getRotScratch(len(vec))
		rotateInto(*scratch, p.rotation, vec)
		vec = *scratch
	}
	dist := pickDistDim(L2, p.dsub)
	for s := 0; s < p.m; s++ {
		sub := vec[s*p.dsub : s*p.dsub+p.dsub]
		cb := p.codebooks[s]
		best := 0
		bestD := dist(sub, cb[0])
		for c := 1; c < len(cb); c++ {
			if d := dist(sub, cb[c]); d < bestD {
				bestD = d
				best = c
			}
		}
		fn(s, best)
	}
	if scratch != nil {
		putRotScratch(scratch)
	}
}
