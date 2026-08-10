// SPDX-License-Identifier: Apache-2.0
//go:build arm64

package vector

// ARM64 always provides Advanced SIMD (NEON), so the fast-scan kernel is installed
// unconditionally (no runtime feature detection), mirroring distance_arm64.go.
//
// The NEON kernel uses TBL (a 16-byte in-register table lookup) as the arm64
// analogue of AVX2's VPSHUFB: it looks up each subspace's 16-entry uint8 LUT by the
// block's code nibbles, widens the looked-up bytes to uint16, and accumulates per
// lane. It is correctness-verified against fastScanScalar on amd64 hosts via QEMU
// (CGO_ENABLED=0 GOARCH=arm64 go test -exec=qemu-aarch64-static ./vector/) by the
// equivalence tests in pq4_test.go (which run on every arch). Its speedup is
// not benchmarked here (no native arm64 hardware in CI).

// fastScanNEON is the NEON (TBL) LUT16 fast-scan kernel. Same contract as
// fastScanAVX2: scores a transposed block of nVec (<=32) vectors against tbl,
// writing out[:nVec]. Implemented in fastscan_arm64.s. Callers ensure m>=1,
// 0<nVec<=32, valid pointers.
func fastScanNEON(tbl *uint8, m int, blockCodes *byte, out *uint16, nVec int)

func fastScanNEONAdapter(tbl []uint8, m int, blockCodes []byte, out []uint16, nVec int) {
	if nVec <= 0 || m <= 0 {
		return
	}
	fastScanNEON(&tbl[0], m, &blockCodes[0], &out[0], nVec)
}

func init() {
	fastScanKernel = fastScanNEONAdapter
}
