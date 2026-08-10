// SPDX-License-Identifier: Apache-2.0
//go:build amd64

package vector

// fastScanAVX2 is the AVX2 (VPSHUFB) in-register LUT16 fast-scan kernel: it scores
// a transposed block of nVec (<= fastScanBlock=32) database vectors against the
// quantized LUT, writing one uint16 accumulator per vector into out[:nVec]. It
// matches fastScanScalar exactly (same uint8 table, same per-subspace uint16 sums).
// Implemented in fastscan_amd64.s. Callers must ensure m >= 1, 0 < nVec <= 32, and
// valid (non-nil) pointers. Selected over the scalar default in init() only when
// detectAVX2() holds.
func fastScanAVX2(tbl *uint8, m int, blockCodes *byte, out *uint16, nVec int)

// fastScanAVX2Adapter bridges the slice-signature kernel var to the pointer kernel.
// blockCodes is the m×fastScanBlock transposed layout; only [:nVec] of each row is
// meaningful (the kernel processes a full 32-lane block but the caller reads only
// out[:nVec], and transposeCodes leaves unused lanes at their previous value — the
// AVX2 kernel zero-extends all 32 lanes but writes only out[:nVec], so stale lanes
// never leak). m, nVec bounds are guaranteed by fastScanBlockInto.
func fastScanAVX2Adapter(tbl []uint8, m int, blockCodes []byte, out []uint16, nVec int) {
	if nVec <= 0 || m <= 0 {
		return
	}
	fastScanAVX2(&tbl[0], m, &blockCodes[0], &out[0], nVec)
}

func init() {
	if avx2Enabled {
		fastScanKernel = fastScanAVX2Adapter
	}
}
