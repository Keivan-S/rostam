// SPDX-License-Identifier: Apache-2.0
//go:build arm64

package vector

import "unsafe"

// ARM64 always provides Advanced SIMD (NEON), so unlike the amd64 path there is
// no runtime feature detection — the kernels are installed unconditionally.
//
// These kernels are correctness-verified on amd64 hosts via QEMU
// (CGO_ENABLED=0 GOARCH=arm64 go test -exec=qemu-aarch64-static ./vector/);
// the equivalence tests in distance_arm64_test.go compare them against the
// scalar reference. Their speedup is not benchmarked here (no native arm64
// hardware in CI).
//
// Scope: NEON accelerates the float32 kernels — dotProduct (Cosine/DotProduct)
// and l2Squared (L2), which cover every metric on the unquantized path. The
// SQ8 asymmetric int8 dot stays on the scalar fallback (sq8DotScalar): Go's
// arm64 assembler exposes neither a signed vector widen (only unsigned VUSHLL)
// nor a vector int->float convert (only scalar SCVTF), so a NEON SQ8 kernel
// would degrade to per-lane scalar work anyway. Quantized search still works on
// arm64 — it just uses the portable distance kernel.

// dotProductNEON computes the dot product of a[:n] and b[:n] using NEON
// (4 float32 lanes per iteration) with a scalar tail. Implemented in
// distance_arm64.s. Callers must ensure n >= 0 and valid pointers when n > 0.
func dotProductNEON(a, b *float32, n int) float32

// l2SquaredNEON computes the squared L2 distance of a[:n] and b[:n] using NEON.
// Implemented in distance_arm64.s. Same calling contract as dotProductNEON.
func l2SquaredNEON(a, b *float32, n int) float32

func dotNEONSlice(a, b []float32) float32 {
	if len(a) == 0 {
		return 0
	}
	return dotProductNEON(&a[0], &b[0], len(a))
}

func l2SquaredNEONSlice(a, b []float32) float32 {
	if len(a) == 0 {
		return 0
	}
	return l2SquaredNEON(&a[0], &b[0], len(a))
}

func init() {
	dotProduct = dotNEONSlice
	l2Squared = l2SquaredNEONSlice
}

// prefetchRange issues nlines PRFM PLDL1KEEP hints from p, one per 64-byte line,
// covering a whole per-slot record. Implemented in distance_arm64.s.
func prefetchRange(p unsafe.Pointer, nlines int)
