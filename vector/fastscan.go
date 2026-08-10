// SPDX-License-Identifier: Apache-2.0

package vector

import "sync/atomic"

// fastScanBlocksScored counts blocks scored through fastScanBlockInto. It exists so
// tests can ASSERT the IVF-PQ query path actually took the fast-scan branch (not
// just that a 4-bit collection works) — see ivf_pq4_test.go. A single relaxed
// atomic add per block is negligible on the scoring hot path.
var fastScanBlocksScored atomic.Uint64

// Fast-scan: ScaNN/FAISS in-register LUT16 ADC over a TRANSPOSED block of database
// vectors. The scalar reference here defines correctness; the AVX2 (VPSHUFB) and
// arm64 (TBL) kernels in fastscan_amd64.* / fastscan_arm64.* must match it within
// the integer accumulation (they are EXACT against it — same uint8 LUT, same
// per-subspace uint16 sums).
//
// LAYOUT. A block holds up to fastScanBlock (32) database vectors. Their 4-bit
// codes are TRANSPOSED so that, for each subspace s, the 32 nibbles (one per
// database vector) are contiguous: blockCodes is m rows of 32 bytes, row s holding
// the s-th sub-code of all 32 vectors (one nibble per byte, low nibble used). This
// is the layout the in-register table lookup consumes: load the 16-entry uint8 LUT
// for subspace s, VPSHUFB it by the 32-byte row → 32 looked-up contributions (one
// per database vector lane), accumulate into 32 uint16 lanes. After m subspaces
// each lane holds Σ_s tbl[s*16 + code_lane[s]] — the same integer accumulator the
// scalar adcScalar computes, one per database vector.
//
// OVERFLOW. Each tbl entry is a uint8 (<=255); m subspaces sum to <= 255*m. The
// uint16 lane accumulator holds up to 65535, so m <= 257 is safe — every realistic
// PQ m (typically 8..64, at most dim/2). fastScanBlockInto guards m and falls back
// to the scalar path if a pathological m would overflow.

// fastScanBlockInto scores a transposed block of nVec (<= fastScanBlock) database
// vectors against the prepared LUT, writing one uint16 accumulator per vector into
// out[:nVec]. blockCodes is the m×fastScanBlock transposed nibble layout (only the
// low nibble of each byte is read; rows are fastScanBlock-strided regardless of
// nVec). The active kernel (scalar default, AVX2/NEON swapped in by init()) is
// fastScanKernel. The caller dequantizes via lut.dequant(uint32(out[i])).
var fastScanKernel = fastScanScalar

// fastScanScalar is the portable reference: for each of the nVec lanes it sums the
// per-subspace uint8 LUT entries indexed by that lane's nibble in each row. This is
// the byte-for-byte definition the SIMD kernels reproduce.
func fastScanScalar(tbl []uint8, m int, blockCodes []byte, out []uint16, nVec int) {
	for i := 0; i < nVec; i++ {
		out[i] = 0
	}
	for s := 0; s < m; s++ {
		row := blockCodes[s*fastScanBlock : s*fastScanBlock+nVec]
		base := s * pqCodebookSize4
		for i := 0; i < nVec; i++ {
			out[i] += uint16(tbl[base+int(row[i]&0x0f)])
		}
	}
}

// fastScanOverflowSafe reports whether m subspaces of uint8 LUT entries sum within
// the uint16 lane accumulator (255*m <= 65535). True for every realistic PQ m.
func fastScanOverflowSafe(m int) bool { return m <= 257 }

// transposeCodes packs nVec contiguous PACKED 4-bit codes (each pq4CodeLen(m)
// bytes) into the m×fastScanBlock transposed block layout fastScanBlockInto
// consumes. dst must be m*fastScanBlock bytes; rows beyond nVec are left as-is
// (the kernel only reads [:nVec] per row). Each byte in dst holds one nibble in its
// low 4 bits (high nibble 0), so the SIMD VPSHUFB index is just the byte.
func transposeCodes(dst []byte, codes []byte, m, nVec int) {
	cl := pq4CodeLen(m)
	for i := 0; i < nVec; i++ {
		code := codes[i*cl : i*cl+cl]
		for s := 0; s < m; s++ {
			dst[s*fastScanBlock+i] = byte(subCodeAt(code, s))
		}
	}
}

// fastScanBlockInto is the high-level entry: it scores nVec database vectors whose
// PACKED codes sit contiguously in codes (pq4CodeLen(m) bytes each) against lut,
// writing the dequantized float ADC distance into dists[:nVec]. It transposes the
// codes into a scratch block, runs the active kernel, and dequantizes. scratch must
// be >= m*fastScanBlock bytes and acc >= fastScanBlock uint16; pass nil to let it
// allocate. nVec must be <= fastScanBlock; the caller loops blocks of fastScanBlock
// and handles the final partial block (nVec < fastScanBlock) here (the kernels read
// only [:nVec] per row, so a partial block is the scalar-tail-safe path).
func (l *lut16) fastScanBlockInto(dists []float32, codes []byte, nVec int, scratch []byte, acc []uint16) {
	if nVec <= 0 {
		return
	}
	fastScanBlocksScored.Add(1)
	if scratch == nil {
		scratch = make([]byte, l.m*fastScanBlock)
	}
	if acc == nil {
		acc = make([]uint16, fastScanBlock)
	}
	transposeCodes(scratch, codes, l.m, nVec)
	if fastScanOverflowSafe(l.m) {
		fastScanKernel(l.tbl, l.m, scratch, acc, nVec)
		for i := 0; i < nVec; i++ {
			dists[i] = l.dequant(uint32(acc[i]))
		}
		return
	}
	// Pathological m (> 257): the uint16 lanes could overflow, so fall back to the
	// per-code scalar ADC (uint32 accumulator) for correctness.
	for i := 0; i < nVec; i++ {
		dists[i] = l.adcScalar(codes[i*pq4CodeLen(l.m) : i*pq4CodeLen(l.m)+pq4CodeLen(l.m)])
	}
}
