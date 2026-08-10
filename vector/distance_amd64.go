// SPDX-License-Identifier: Apache-2.0
//go:build amd64

package vector

import "unsafe"

// cpuid and xgetbv are thin wrappers over the CPUID and XGETBV instructions
// (implemented in distance_amd64.s) used for runtime AVX2 detection — keeping
// the package dependency-free instead of pulling golang.org/x/sys/cpu.
func cpuid(eaxArg, ecxArg uint32) (eax, ebx, ecx, edx uint32)
func xgetbv() (eax, edx uint32)

// prefetch issues a PREFETCHT0 for the cache line at p (implemented in
// distance_amd64.s). A no-op on non-amd64 (see distance_prefetch_other.go).
func prefetch(p unsafe.Pointer)

// prefetchRange issues nlines consecutive PREFETCHT0s from p, covering a whole
// per-slot record rather than only its first 64 bytes (implemented in
// distance_amd64.s). A no-op on non-amd64.
func prefetchRange(p unsafe.Pointer, nlines int)

// dotProductAVX2 computes the dot product of a[:n] and b[:n] using AVX2
// (8 float32 lanes per iteration) with a scalar tail. Implemented in
// distance_amd64.s. Callers must ensure n >= 0 and that the pointers are
// valid when n > 0.
func dotProductAVX2(a, b *float32, n int) float32

// l2SquaredAVX2 computes the squared L2 distance of a[:n] and b[:n] using AVX2.
// Implemented in distance_amd64.s. Same calling contract as dotProductAVX2.
func l2SquaredAVX2(a, b *float32, n int) float32

// sq8DotProductAVX2 computes the asymmetric SQ8 dot product
// Σ query[i]*float32(int8(code[i])) over n elements using AVX2 (8 lanes per
// iteration) with a scalar tail. Implemented in distance_amd64.s. Callers must
// ensure n >= 0 and that the pointers are valid when n > 0.
func sq8DotProductAVX2(query *float32, code *byte, n int) float32

// dotProductAVX512 and l2SquaredAVX512 are the 16-lane (ZMM) counterparts of the
// AVX2 kernels, used when the CPU has AVX512F and the OS saves ZMM state. Same
// calling contract. Implemented in distance_avx512_amd64.s.
func dotProductAVX512(a, b *float32, n int) float32
func l2SquaredAVX512(a, b *float32, n int) float32

// sq8CodeDotProductAVX2 computes the symmetric integer dot Σ int8(a[i])*int8(b[i])
// of two SQ8 codes using AVX2 (16 int8/iteration) with a scalar tail. Implemented
// in distance_amd64.s.
func sq8CodeDotProductAVX2(a, b *byte, n int) int32

// sq8CodeDotAVX2 adapts the pointer kernel to the sq8CodeDot signature.
func sq8CodeDotAVX2(a, b []byte) int32 {
	if len(a) == 0 {
		return 0
	}
	return sq8CodeDotProductAVX2(&a[0], &b[0], len(a))
}

// sq8CodeDotProductVNNI computes the symmetric integer dot Σ int8(a[i])*int8(b[i])
// of two SQ8 codes using AVX-512-VNNI (VPDPBUSD, 64 int8/iteration) with an
// unsigned-offset correction and a scalar tail. Implemented in distance_amd64.s.
func sq8CodeDotProductVNNI(a, b *byte, n int) int32

// sq8CodeDotVNNI adapts the VNNI pointer kernel to the sq8CodeDot signature.
func sq8CodeDotVNNI(a, b []byte) int32 {
	if len(a) == 0 {
		return 0
	}
	return sq8CodeDotProductVNNI(&a[0], &b[0], len(a))
}

// avx2Enabled reports whether the AVX2 dot kernel is active on this CPU.
var avx2Enabled = detectAVX2()

// avx512Enabled reports whether the AVX-512 kernels are active on this CPU.
var avx512Enabled = detectAVX512F()

// avx512VNNIEnabled reports whether the AVX-512-VNNI code kernel is active.
var avx512VNNIEnabled = detectAVX512VNNI()

// detectAVX512VNNI reports whether the CPU supports AVX-512-VNNI (VPDPBUSD) and
// the OS saves ZMM state. It requires everything detectAVX512F checks plus
// AVX512_VNNI (CPUID leaf 7 sub-leaf 0, ECX bit 11).
func detectAVX512VNNI() bool {
	if !detectAVX512F() {
		return false
	}
	const avx512vnni = 1 << 11
	_, _, ecx7, _ := cpuid(7, 0)
	return ecx7&avx512vnni != 0
}

// detectAVX512F checks, in order: CPUID is new enough; the CPU advertises AVX +
// OSXSAVE; the OS saves both YMM state (XCR0 bits 1,2) and full ZMM/opmask state
// (XCR0 bits 5,6,7 — opmask, ZMM_Hi256, Hi16_ZMM); and the CPU advertises
// AVX512F (CPUID leaf 7, EBX bit 16). All must hold to issue AVX-512 safely.
func detectAVX512F() bool {
	maxLeaf, _, _, _ := cpuid(0, 0)
	if maxLeaf < 7 {
		return false
	}
	const osxsave = 1 << 27
	const avx = 1 << 28
	_, _, ecx1, _ := cpuid(1, 0)
	if ecx1&(osxsave|avx) != (osxsave | avx) {
		return false
	}
	xcr0lo, _ := xgetbv()
	// Bits 1,2 = XMM,YMM; bits 5,6,7 = opmask, ZMM_Hi256, Hi16_ZMM.
	const need = 1<<1 | 1<<2 | 1<<5 | 1<<6 | 1<<7
	if xcr0lo&need != need {
		return false
	}
	const avx512f = 1 << 16
	_, ebx7, _, _ := cpuid(7, 0)
	return ebx7&avx512f != 0
}

// dotAVX512 / l2SquaredAVX512Slice adapt the pointer kernels to distFunc.
func dotAVX512(a, b []float32) float32 {
	if len(a) == 0 {
		return 0
	}
	return dotProductAVX512(&a[0], &b[0], len(a))
}

func l2SquaredAVX512Slice(a, b []float32) float32 {
	if len(a) == 0 {
		return 0
	}
	return l2SquaredAVX512(&a[0], &b[0], len(a))
}

// detectAVX2 checks, in order: CPUID is new enough; the CPU advertises AVX +
// OSXSAVE; the OS actually saves YMM state (XCR0 bits 1,2 via XGETBV); and the
// CPU advertises AVX2 (CPUID leaf 7, EBX bit 5). All four must hold to issue
// AVX2 instructions safely.
func detectAVX2() bool {
	maxLeaf, _, _, _ := cpuid(0, 0)
	if maxLeaf < 7 {
		return false
	}
	const osxsave = 1 << 27
	const avx = 1 << 28
	const fma = 1 << 12 // the kernels use VFMADD231PS (FMA3); every AVX2 CPU has it
	_, _, ecx1, _ := cpuid(1, 0)
	if ecx1&(osxsave|avx|fma) != (osxsave | avx | fma) {
		return false
	}
	xcr0lo, _ := xgetbv()
	const xmmAndYmm = 1<<1 | 1<<2
	if xcr0lo&xmmAndYmm != xmmAndYmm {
		return false
	}
	const avx2 = 1 << 5
	_, ebx7, _, _ := cpuid(7, 0)
	return ebx7&avx2 != 0
}

// dotAVX2 adapts the pointer-based kernel to the distFunc slice signature.
func dotAVX2(a, b []float32) float32 {
	if len(a) == 0 {
		return 0
	}
	return dotProductAVX2(&a[0], &b[0], len(a))
}

// l2SquaredAVX2Slice adapts the pointer-based L2 kernel to the slice signature.
func l2SquaredAVX2Slice(a, b []float32) float32 {
	if len(a) == 0 {
		return 0
	}
	return l2SquaredAVX2(&a[0], &b[0], len(a))
}

// sq8DotAVX2 adapts the pointer-based SQ8 kernel to the sq8Dot signature.
// len(query) must equal len(code).
func sq8DotAVX2(query []float32, code []byte) float32 {
	if len(code) == 0 {
		return 0
	}
	return sq8DotProductAVX2(&query[0], &code[0], len(code))
}

func init() {
	if avx2Enabled {
		dotProduct = dotAVX2
		l2Squared = l2SquaredAVX2Slice
		sq8Dot = sq8DotAVX2
		sq8CodeDot = sq8CodeDotAVX2
	}
	// AVX-512-VNNI does the symmetric int8 code dot at 64 int8/op (vs AVX2's 16),
	// so it supersedes the AVX2 code kernel for the quantized build path. Gated on
	// VPDPBUSD support; the float/asymmetric kernels are unaffected.
	if avx512VNNIEnabled {
		sq8CodeDot = sq8CodeDotVNNI
	}
	// AVX-512 (ZMM) kernels are enabled only at high dimension (>= avx512MinDim),
	// via a per-collection dispatch (pickDistDim → metricDist). On AMD Zen 4 they
	// double-pump 512-bit over a 256-bit datapath, so they tie AVX2+FMA at low/mid
	// dim and only win at large dim (~14% on dot at 1536) — exactly the high-dim
	// build/search regime. Below the threshold the default AVX2 kernels stay in
	// force, so small/mid-dim workloads are unchanged.
	if avx512Enabled {
		avx512DistDim = func(m Metric, dim int) distFunc {
			if dim < avx512MinDim {
				return nil
			}
			switch m {
			case Cosine:
				return func(a, b []float32) float32 { return 1 - dotAVX512(a, b) }
			case DotProduct:
				return func(a, b []float32) float32 { return -dotAVX512(a, b) }
			case L2:
				return l2SquaredAVX512Slice
			}
			return nil
		}
	}
}

// avx512MinDim is the dimension at or above which the AVX-512 kernels are
// preferred over AVX2. Set above the dims where Zen 4's double-pumped AVX-512
// merely ties AVX2 (<= 768) so it only engages where it measurably wins. A var
// (not const) so benchmarks can force the AVX2 path for A/B timing.
var avx512MinDim = 1024
