#include "textflag.h"

// Batched AVX-512 (ZMM) "one query vs N slots" kernels — the 16-lane
// counterparts of the AVX2 kernels in distance_ny_amd64.s, selected at
// dim >= avx512MinDim on CPUs where the per-pair path would also have chosen
// AVX-512 (see distance_ny_amd64.go's init).
//
// As with the AVX2 pair, the accumulator layout is a deliberate duplicate of the
// per-pair kernel in distance_avx512_amd64.s — four ZMM accumulators per
// candidate, 64 floats per iteration, the same combine tree, 16-wide follow-on
// loop, two-stage reduce and scalar tail — so results are bit-identical to what
// the per-pair AVX-512 kernel produces. Two candidates are interleaved against
// one set of query registers (Z8..Z11).
//
// The caller (nyDispatch) guarantees n is even and every slot in range.

// func l2SquaredNyAVX512(q, base *float32, dim int, slots *uint32, n int, out *float32)
TEXT ·l2SquaredNyAVX512(SB), NOSPLIT, $0-48
	MOVQ base+8(FP), BX
	MOVQ slots+24(FP), SI
	MOVQ n+32(FP), CX
	MOVQ out+40(FP), DI

l2block:
	CMPQ CX, $2
	JL   l2ret

	// Warm four candidates ahead; see the AVX2 kernel for why the prefetch
	// lives inside the block loop rather than in the Go caller.
	CMPQ  CX, $6
	JL    l2nopf
	MOVQ  dim+16(FP), R10
	LEAQ  15(R10), R11
	SHRQ  $4, R11         // lines = (dim*4 + 63)/64
	MOVL  16(SI), R8
	IMULQ R10, R8
	LEAQ  (BX)(R8*4), R8
	MOVL  20(SI), R9
	IMULQ R10, R9
	LEAQ  (BX)(R9*4), R9

l2pf:
	PREFETCHT0 (R8)
	PREFETCHT0 (R9)
	ADDQ       $64, R8
	ADDQ       $64, R9
	DECQ       R11
	JNZ        l2pf

l2nopf:
	MOVQ  q+0(FP), AX
	MOVQ  dim+16(FP), R12
	MOVL  0(SI), R13
	IMULQ R12, R13
	LEAQ  (BX)(R13*4), R8
	MOVL  4(SI), R13
	IMULQ R12, R13
	LEAQ  (BX)(R13*4), R9

	VXORPS Z0, Z0, Z0
	VXORPS Z1, Z1, Z1
	VXORPS Z2, Z2, Z2
	VXORPS Z3, Z3, Z3
	VXORPS Z4, Z4, Z4
	VXORPS Z5, Z5, Z5
	VXORPS Z6, Z6, Z6
	VXORPS Z7, Z7, Z7

l2loop64:
	CMPQ R12, $64
	JL   l2combine

	VMOVUPS 0(AX), Z8
	VMOVUPS 64(AX), Z9
	VMOVUPS 128(AX), Z10
	VMOVUPS 192(AX), Z11

	VMOVUPS     0(R8), Z12
	VMOVUPS     64(R8), Z13
	VMOVUPS     128(R8), Z14
	VMOVUPS     192(R8), Z15
	VSUBPS      Z8, Z12, Z12
	VSUBPS      Z9, Z13, Z13
	VSUBPS      Z10, Z14, Z14
	VSUBPS      Z11, Z15, Z15
	VFMADD231PS Z12, Z12, Z0
	VFMADD231PS Z13, Z13, Z1
	VFMADD231PS Z14, Z14, Z2
	VFMADD231PS Z15, Z15, Z3

	VMOVUPS     0(R9), Z12
	VMOVUPS     64(R9), Z13
	VMOVUPS     128(R9), Z14
	VMOVUPS     192(R9), Z15
	VSUBPS      Z8, Z12, Z12
	VSUBPS      Z9, Z13, Z13
	VSUBPS      Z10, Z14, Z14
	VSUBPS      Z11, Z15, Z15
	VFMADD231PS Z12, Z12, Z4
	VFMADD231PS Z13, Z13, Z5
	VFMADD231PS Z14, Z14, Z6
	VFMADD231PS Z15, Z15, Z7

	ADDQ $256, AX
	ADDQ $256, R8
	ADDQ $256, R9
	SUBQ $64, R12
	JMP  l2loop64

l2combine:
	VADDPS Z1, Z0, Z0
	VADDPS Z3, Z2, Z2
	VADDPS Z2, Z0, Z0
	VADDPS Z5, Z4, Z4
	VADDPS Z7, Z6, Z6
	VADDPS Z6, Z4, Z4

l2loop16:
	CMPQ R12, $16
	JL   l2reduce
	VMOVUPS     (AX), Z8
	VMOVUPS     (R8), Z12
	VSUBPS      Z8, Z12, Z12
	VFMADD231PS Z12, Z12, Z0
	VMOVUPS     (R9), Z13
	VSUBPS      Z8, Z13, Z13
	VFMADD231PS Z13, Z13, Z4
	ADDQ        $64, AX
	ADDQ        $64, R8
	ADDQ        $64, R9
	SUBQ        $16, R12
	JMP         l2loop16

l2reduce:
	VEXTRACTF64X4 $1, Z0, Y12
	VADDPS        Y12, Y0, Y0
	VEXTRACTF128  $1, Y0, X12
	VADDPS        X12, X0, X0
	VHADDPS       X0, X0, X0
	VHADDPS       X0, X0, X0
	VEXTRACTF64X4 $1, Z4, Y13
	VADDPS        Y13, Y4, Y4
	VEXTRACTF128  $1, Y4, X13
	VADDPS        X13, X4, X4
	VHADDPS       X4, X4, X4
	VHADDPS       X4, X4, X4
	VZEROUPPER

l2scalar:
	CMPQ  R12, $0
	JE    l2store
	MOVSS (AX), X1
	MOVSS (R8), X2
	SUBSS X1, X2
	MULSS X2, X2
	ADDSS X2, X0
	MOVSS (R9), X3
	SUBSS X1, X3
	MULSS X3, X3
	ADDSS X3, X4
	ADDQ  $4, AX
	ADDQ  $4, R8
	ADDQ  $4, R9
	DECQ  R12
	JMP   l2scalar

l2store:
	MOVSS X0, 0(DI)
	MOVSS X4, 4(DI)
	ADDQ  $8, SI
	ADDQ  $8, DI
	SUBQ  $2, CX
	JMP   l2block

l2ret:
	RET

// func dotNyAVX512(q, base *float32, dim int, slots *uint32, n int, out *float32)
TEXT ·dotNyAVX512(SB), NOSPLIT, $0-48
	MOVQ base+8(FP), BX
	MOVQ slots+24(FP), SI
	MOVQ n+32(FP), CX
	MOVQ out+40(FP), DI

dotblock:
	CMPQ CX, $2
	JL   dotret

	// Warm four candidates ahead; see the AVX2 kernel.
	CMPQ  CX, $6
	JL    dotnopf
	MOVQ  dim+16(FP), R10
	LEAQ  15(R10), R11
	SHRQ  $4, R11
	MOVL  16(SI), R8
	IMULQ R10, R8
	LEAQ  (BX)(R8*4), R8
	MOVL  20(SI), R9
	IMULQ R10, R9
	LEAQ  (BX)(R9*4), R9

dotpf:
	PREFETCHT0 (R8)
	PREFETCHT0 (R9)
	ADDQ       $64, R8
	ADDQ       $64, R9
	DECQ       R11
	JNZ        dotpf

dotnopf:
	MOVQ  q+0(FP), AX
	MOVQ  dim+16(FP), R12
	MOVL  0(SI), R13
	IMULQ R12, R13
	LEAQ  (BX)(R13*4), R8
	MOVL  4(SI), R13
	IMULQ R12, R13
	LEAQ  (BX)(R13*4), R9

	VXORPS Z0, Z0, Z0
	VXORPS Z1, Z1, Z1
	VXORPS Z2, Z2, Z2
	VXORPS Z3, Z3, Z3
	VXORPS Z4, Z4, Z4
	VXORPS Z5, Z5, Z5
	VXORPS Z6, Z6, Z6
	VXORPS Z7, Z7, Z7

dotloop64:
	CMPQ R12, $64
	JL   dotcombine

	VMOVUPS 0(AX), Z8
	VMOVUPS 64(AX), Z9
	VMOVUPS 128(AX), Z10
	VMOVUPS 192(AX), Z11

	VMOVUPS     0(R8), Z12
	VMOVUPS     64(R8), Z13
	VMOVUPS     128(R8), Z14
	VMOVUPS     192(R8), Z15
	VFMADD231PS Z8, Z12, Z0
	VFMADD231PS Z9, Z13, Z1
	VFMADD231PS Z10, Z14, Z2
	VFMADD231PS Z11, Z15, Z3

	VMOVUPS     0(R9), Z12
	VMOVUPS     64(R9), Z13
	VMOVUPS     128(R9), Z14
	VMOVUPS     192(R9), Z15
	VFMADD231PS Z8, Z12, Z4
	VFMADD231PS Z9, Z13, Z5
	VFMADD231PS Z10, Z14, Z6
	VFMADD231PS Z11, Z15, Z7

	ADDQ $256, AX
	ADDQ $256, R8
	ADDQ $256, R9
	SUBQ $64, R12
	JMP  dotloop64

dotcombine:
	VADDPS Z1, Z0, Z0
	VADDPS Z3, Z2, Z2
	VADDPS Z2, Z0, Z0
	VADDPS Z5, Z4, Z4
	VADDPS Z7, Z6, Z6
	VADDPS Z6, Z4, Z4

dotloop16:
	CMPQ R12, $16
	JL   dotreduce
	VMOVUPS     (AX), Z8
	VMOVUPS     (R8), Z12
	VFMADD231PS Z8, Z12, Z0
	VMOVUPS     (R9), Z13
	VFMADD231PS Z8, Z13, Z4
	ADDQ        $64, AX
	ADDQ        $64, R8
	ADDQ        $64, R9
	SUBQ        $16, R12
	JMP         dotloop16

dotreduce:
	VEXTRACTF64X4 $1, Z0, Y12
	VADDPS        Y12, Y0, Y0
	VEXTRACTF128  $1, Y0, X12
	VADDPS        X12, X0, X0
	VHADDPS       X0, X0, X0
	VHADDPS       X0, X0, X0
	VEXTRACTF64X4 $1, Z4, Y13
	VADDPS        Y13, Y4, Y4
	VEXTRACTF128  $1, Y4, X13
	VADDPS        X13, X4, X4
	VHADDPS       X4, X4, X4
	VHADDPS       X4, X4, X4
	VZEROUPPER

dotscalar:
	CMPQ  R12, $0
	JE    dotstore
	MOVSS (AX), X1
	MOVSS (R8), X2
	MULSS X1, X2
	ADDSS X2, X0
	MOVSS (R9), X3
	MULSS X1, X3
	ADDSS X3, X4
	ADDQ  $4, AX
	ADDQ  $4, R8
	ADDQ  $4, R9
	DECQ  R12
	JMP   dotscalar

dotstore:
	MOVSS X0, 0(DI)
	MOVSS X4, 4(DI)
	ADDQ  $8, SI
	ADDQ  $8, DI
	SUBQ  $2, CX
	JMP   dotblock

dotret:
	RET
