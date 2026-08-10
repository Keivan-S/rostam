#include "textflag.h"

// func prefetch(p unsafe.Pointer)
//
// Issues a PREFETCHT0 (fetch into all cache levels) for the line at p. Used to
// pull an upcoming candidate vector toward the core while the current distance
// is computed, hiding the cache-miss latency that dominates graph traversal.
TEXT ·prefetch(SB), NOSPLIT, $0-8
	MOVQ p+0(FP), AX
	PREFETCHT0 (AX)
	RET

// func prefetchRange(p unsafe.Pointer, nlines int)
//
// Issues nlines consecutive PREFETCHT0s starting at p, one per 64-byte cache
// line. A single prefetch covers 64 bytes, which is an eighth of a 128-dim
// float32 vector — the search's lookahead only helped for the first line and
// the rest of the vector still missed. Unrolling the whole record here also
// amortizes the CALL over every line instead of paying one per line.
//
// PREFETCHT0 never faults, but callers still pass only in-bounds addresses so
// the walk stays inside the arena's backing array.
TEXT ·prefetchRange(SB), NOSPLIT, $0-16
	MOVQ p+0(FP), AX
	MOVQ nlines+8(FP), CX
	TESTQ CX, CX
	JLE  done

loop:
	PREFETCHT0 (AX)
	ADDQ $64, AX
	DECQ CX
	JNZ  loop

done:
	RET

// func cpuid(eaxArg, ecxArg uint32) (eax, ebx, ecx, edx uint32)
TEXT ·cpuid(SB), NOSPLIT, $0-24
	MOVL eaxArg+0(FP), AX
	MOVL ecxArg+4(FP), CX
	CPUID
	MOVL AX, eax+8(FP)
	MOVL BX, ebx+12(FP)
	MOVL CX, ecx+16(FP)
	MOVL DX, edx+20(FP)
	RET

// func xgetbv() (eax, edx uint32)
TEXT ·xgetbv(SB), NOSPLIT, $0-8
	MOVL $0, CX
	XGETBV
	MOVL AX, eax+0(FP)
	MOVL DX, edx+4(FP)
	RET

// func dotProductAVX2(a, b *float32, n int) float32
//
// Four independent FMA accumulators (Y0..Y3), 32 floats per iteration. Multiple
// accumulators break the loop-carried dependency on a single sum, so the FMAs
// pipeline (throughput-bound rather than latency-bound); FMA fuses the multiply
// and add. Falls back to an 8-wide loop, then a scalar tail.
TEXT ·dotProductAVX2(SB), NOSPLIT, $0-28
	MOVQ a+0(FP), AX
	MOVQ b+8(FP), BX
	MOVQ n+16(FP), CX
	VXORPS Y0, Y0, Y0
	VXORPS Y1, Y1, Y1
	VXORPS Y2, Y2, Y2
	VXORPS Y3, Y3, Y3

loop32:
	CMPQ CX, $32
	JL   combine
	VMOVUPS 0(AX), Y4
	VMOVUPS 0(BX), Y5
	VFMADD231PS Y4, Y5, Y0
	VMOVUPS 32(AX), Y6
	VMOVUPS 32(BX), Y7
	VFMADD231PS Y6, Y7, Y1
	VMOVUPS 64(AX), Y8
	VMOVUPS 64(BX), Y9
	VFMADD231PS Y8, Y9, Y2
	VMOVUPS 96(AX), Y10
	VMOVUPS 96(BX), Y11
	VFMADD231PS Y10, Y11, Y3
	ADDQ $128, AX
	ADDQ $128, BX
	SUBQ $32, CX
	JMP  loop32

combine:
	VADDPS Y1, Y0, Y0
	VADDPS Y3, Y2, Y2
	VADDPS Y2, Y0, Y0

loop8:
	CMPQ CX, $8
	JL   reduce
	VMOVUPS (AX), Y4
	VMOVUPS (BX), Y5
	VFMADD231PS Y4, Y5, Y0
	ADDQ $32, AX
	ADDQ $32, BX
	SUBQ $8, CX
	JMP  loop8

reduce:
	VEXTRACTF128 $1, Y0, X1
	VADDPS       X1, X0, X0
	VHADDPS      X0, X0, X0
	VHADDPS      X0, X0, X0
	VZEROUPPER

scalarloop:
	CMPQ CX, $0
	JE   done
	MOVSS (AX), X1
	MULSS (BX), X1
	ADDSS X1, X0
	ADDQ  $4, AX
	ADDQ  $4, BX
	DECQ  CX
	JMP   scalarloop

done:
	MOVSS X0, ret+24(FP)
	RET

// func sq8DotProductAVX2(query *float32, code *byte, n int) float32
//
// Asymmetric SQ8 dot: accumulates Σ query[i]*float32(int8(code[i])). Each
// iteration sign-extends 8 int8 codes to int32 (VPMOVSXBD), converts to float32
// (VCVTDQ2PS), multiplies by 8 query floats, and adds into Y0; then reduces to a
// scalar and handles the n%8 tail with SSE. query advances 32 bytes/iter (8
// floats), code advances 8 bytes/iter (8 int8).
TEXT ·sq8DotProductAVX2(SB), NOSPLIT, $0-28
	MOVQ query+0(FP), AX
	MOVQ code+8(FP), BX
	MOVQ n+16(FP), CX
	VXORPS Y0, Y0, Y0

sq8simd:
	CMPQ CX, $8
	JL   sq8reduce
	VPMOVSXBD (BX), Y1
	VCVTDQ2PS Y1, Y1
	VMOVUPS   (AX), Y2
	VMULPS    Y1, Y2, Y1
	VADDPS    Y1, Y0, Y0
	ADDQ      $32, AX
	ADDQ      $8, BX
	SUBQ      $8, CX
	JMP       sq8simd

sq8reduce:
	VEXTRACTF128 $1, Y0, X1
	VADDPS       X1, X0, X0
	VHADDPS      X0, X0, X0
	VHADDPS      X0, X0, X0
	VZEROUPPER

sq8scalar:
	CMPQ CX, $0
	JE   sq8done
	MOVBLSX  (BX), DX
	CVTSL2SS DX, X1
	MULSS    (AX), X1
	ADDSS    X1, X0
	ADDQ     $4, AX
	ADDQ     $1, BX
	DECQ     CX
	JMP      sq8scalar

sq8done:
	MOVSS X0, ret+24(FP)
	RET

// func l2SquaredAVX2(a, b *float32, n int) float32
//
// Accumulates 8 lanes of (a-b)^2 per iteration into Y0, reduces to a scalar,
// then handles the n%8 tail with SSE. The square makes the result independent
// of subtraction operand order.
TEXT ·l2SquaredAVX2(SB), NOSPLIT, $0-28
	MOVQ a+0(FP), AX
	MOVQ b+8(FP), BX
	MOVQ n+16(FP), CX
	VXORPS Y0, Y0, Y0
	VXORPS Y1, Y1, Y1
	VXORPS Y2, Y2, Y2
	VXORPS Y3, Y3, Y3

l2loop32:
	CMPQ CX, $32
	JL   l2combine
	VMOVUPS 0(AX), Y4
	VMOVUPS 0(BX), Y5
	VSUBPS  Y5, Y4, Y4
	VFMADD231PS Y4, Y4, Y0
	VMOVUPS 32(AX), Y6
	VMOVUPS 32(BX), Y7
	VSUBPS  Y7, Y6, Y6
	VFMADD231PS Y6, Y6, Y1
	VMOVUPS 64(AX), Y8
	VMOVUPS 64(BX), Y9
	VSUBPS  Y9, Y8, Y8
	VFMADD231PS Y8, Y8, Y2
	VMOVUPS 96(AX), Y10
	VMOVUPS 96(BX), Y11
	VSUBPS  Y11, Y10, Y10
	VFMADD231PS Y10, Y10, Y3
	ADDQ $128, AX
	ADDQ $128, BX
	SUBQ $32, CX
	JMP  l2loop32

l2combine:
	VADDPS Y1, Y0, Y0
	VADDPS Y3, Y2, Y2
	VADDPS Y2, Y0, Y0

l2loop8:
	CMPQ CX, $8
	JL   l2reduce
	VMOVUPS (AX), Y4
	VMOVUPS (BX), Y5
	VSUBPS  Y5, Y4, Y4
	VFMADD231PS Y4, Y4, Y0
	ADDQ $32, AX
	ADDQ $32, BX
	SUBQ $8, CX
	JMP  l2loop8

l2reduce:
	VEXTRACTF128 $1, Y0, X1
	VADDPS       X1, X0, X0
	VHADDPS      X0, X0, X0
	VHADDPS      X0, X0, X0
	VZEROUPPER

l2scalar:
	CMPQ CX, $0
	JE   l2done
	MOVSS (AX), X1
	SUBSS (BX), X1
	MULSS X1, X1
	ADDSS X1, X0
	ADDQ  $4, AX
	ADDQ  $4, BX
	DECQ  CX
	JMP   l2scalar

l2done:
	MOVSS X0, ret+24(FP)
	RET

// func sq8CodeDotProductAVX2(a, b *byte, n int) int32
//
// Symmetric integer dot Σ int8(a[i])*int8(b[i]) of two SQ8 codes. Each iteration
// sign-extends 16 int8 to int16 (VPMOVSXBW) for both sides, multiplies and
// horizontally adds adjacent pairs into 8 int32 (VPMADDWD), accumulating in Y0;
// then reduces to a scalar int32 and handles the n%16 tail with a scalar loop.
// Used by the quantized build path (neighbor selection over codes).
TEXT ·sq8CodeDotProductAVX2(SB), NOSPLIT, $0-28
	MOVQ a+0(FP), AX
	MOVQ b+8(FP), BX
	MOVQ n+16(FP), CX
	VPXOR Y0, Y0, Y0

codeloop16:
	CMPQ CX, $16
	JL   codereduce
	VPMOVSXBW (AX), Y1
	VPMOVSXBW (BX), Y2
	VPMADDWD  Y2, Y1, Y1
	VPADDD    Y1, Y0, Y0
	ADDQ $16, AX
	ADDQ $16, BX
	SUBQ $16, CX
	JMP  codeloop16

codereduce:
	VEXTRACTI128 $1, Y0, X1
	VPADDD       X1, X0, X0
	VPHADDD      X0, X0, X0
	VPHADDD      X0, X0, X0
	MOVD         X0, DX
	VZEROUPPER

codescalar:
	CMPQ CX, $0
	JE   codedone
	MOVBLSX (AX), R8
	MOVBLSX (BX), R9
	IMULL   R9, R8
	ADDL    R8, DX
	INCQ    AX
	INCQ    BX
	DECQ    CX
	JMP     codescalar

codedone:
	MOVL DX, ret+24(FP)
	RET

// func sq8CodeDotProductVNNI(a, b *byte, n int) int32
//
// Symmetric integer dot Σ int8(a[i])*int8(b[i]) of two SQ8 codes using
// AVX-512-VNNI (VPDPBUSD: 64 int8/op vs the AVX2 kernel's 16). VPDPBUSD computes
// an UNSIGNED×SIGNED byte dot, but both codes are signed, so we offset the a
// side to unsigned and correct: with a' = a + 128 (== a XOR 0x80 in two's
// complement),
//     Σ a[i]*b[i] = Σ (a'[i]-128)*b[i] = Σ a'[i]*b[i] - 128*Σ b[i].
// Two VNNI accumulators run per chunk — dot chains accumulate Σ a'[i]*b[i]
// (a' unsigned, b signed) and bsum chains accumulate Σ b[i] (a constant-1
// unsigned vector × b signed) — then result = dotSum - 128*bSum. The
// whole-vector correction is exact and order-independent. 128 bytes/iter over 4
// independent EVEX chains, a 64-byte step, and a signed scalar n%64 tail.
// Selected at init() only when detectAVX512VNNI() holds (see distance_amd64.go);
// used by the quantized build path (neighbor selection over codes). Measured on
// EPYC 9454P (Zen 4): kernel ~3.3x the AVX2 code kernel at dim 1536 (17.5 vs
// ~59 ns); end-to-end 200k x 1536 quantized build ~1.42x (154 -> 108 s, the
// build being only partly kernel-bound), recall@100 unchanged at 0.994.
TEXT ·sq8CodeDotProductVNNI(SB), NOSPLIT, $0-28
	MOVQ a+0(FP), AX
	MOVQ b+8(FP), BX
	MOVQ n+16(FP), CX
	VPXORQ Z0, Z0, Z0  // dot chain 0  (Σ a'·b)
	VPXORQ Z1, Z1, Z1  // dot chain 1
	VPXORQ Z2, Z2, Z2  // bsum chain 0 (Σ b)
	VPXORQ Z3, Z3, Z3  // bsum chain 1
	VPBROADCASTD mask80<>(SB), Z14  // 0x80 in every byte (a -> unsigned)
	VPBROADCASTD onesb<>(SB), Z15   // 0x01 in every byte (Σ b multiplier)

vnniloop128:
	CMPQ CX, $128
	JL   vnnitail64
	VMOVDQU64 0(AX), Z4
	VMOVDQU64 0(BX), Z5
	VPXORD    Z14, Z4, Z4
	VPDPBUSD  Z5, Z4, Z0
	VPDPBUSD  Z5, Z15, Z2
	VMOVDQU64 64(AX), Z6
	VMOVDQU64 64(BX), Z7
	VPXORD    Z14, Z6, Z6
	VPDPBUSD  Z7, Z6, Z1
	VPDPBUSD  Z7, Z15, Z3
	ADDQ $128, AX
	ADDQ $128, BX
	SUBQ $128, CX
	JMP  vnniloop128

vnnitail64:
	CMPQ CX, $64
	JL   vnnicombine
	VMOVDQU64 0(AX), Z4
	VMOVDQU64 0(BX), Z5
	VPXORD    Z14, Z4, Z4
	VPDPBUSD  Z5, Z4, Z0
	VPDPBUSD  Z5, Z15, Z2
	ADDQ $64, AX
	ADDQ $64, BX
	SUBQ $64, CX

vnnicombine:
	VPADDD Z1, Z0, Z0  // fold dot chains
	VPADDD Z3, Z2, Z2  // fold bsum chains

	// reduce Z0 (dotSum) -> R8
	VEXTRACTI64X4 $1, Z0, Y1
	VPADDD        Y1, Y0, Y0
	VEXTRACTI128  $1, Y0, X1
	VPADDD        X1, X0, X0
	VPHADDD       X0, X0, X0
	VPHADDD       X0, X0, X0
	MOVD          X0, R8

	// reduce Z2 (bSum) -> R9
	VEXTRACTI64X4 $1, Z2, Y1
	VPADDD        Y1, Y2, Y2
	VEXTRACTI128  $1, Y2, X1
	VPADDD        X1, X2, X2
	VPHADDD       X2, X2, X2
	VPHADDD       X2, X2, X2
	MOVD          X2, R9
	VZEROUPPER

	SHLL $7, R9    // 128 * bSum
	SUBL R9, R8    // dotSum - 128*bSum

vnniscalar:
	CMPQ CX, $0
	JE   vnnidone
	MOVBLSX (AX), R10
	MOVBLSX (BX), R11
	IMULL   R11, R10
	ADDL    R10, R8
	INCQ AX
	INCQ BX
	DECQ CX
	JMP  vnniscalar

vnnidone:
	MOVL R8, ret+24(FP)
	RET

// Broadcast-source dwords for sq8CodeDotProductVNNI: 0x80 in every byte (the
// unsigned offset applied via XOR) and 0x01 in every byte (the Σ b multiplier).
DATA mask80<>+0(SB)/4, $0x80808080
GLOBL mask80<>(SB), RODATA|NOPTR, $4
DATA onesb<>+0(SB)/4, $0x01010101
GLOBL onesb<>(SB), RODATA|NOPTR, $4
