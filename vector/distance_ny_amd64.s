#include "textflag.h"

// Batched AVX2 "one query vs N slots" kernels (the faiss fvec_L2sqr_ny shape).
//
// Each kernel walks the slot list two candidates at a time. Both candidates are
// scored against the SAME query registers (Y8..Y11), so one 32-float query load
// serves two candidates instead of one — and the whole block costs a single CALL
// instead of one per candidate.
//
// The accumulator layout is a deliberate duplicate of the per-pair kernels in
// distance_amd64.s: four accumulators per candidate (Y0..Y3 for the first,
// Y4..Y7 for the second), chunk i into accumulator i, the same combine tree, the
// same 8-wide follow-on loop and the same scalar tail. That makes the summation
// order — and therefore the last bit of every result — identical to what the
// per-pair kernel would have produced for the same pair, which is what the
// differential tests assert. See distance_ny.go for why that matters more than
// the extra ~12% a four-candidate interleave would have bought (four candidates
// times four accumulators leaves no register for the query).
//
// The kernels also own their PREFETCHING, warming the block four candidates
// ahead from inside the loop. That placement is the whole reason the batched
// path can take the entire neighbor list in one call: prefetch depth and call
// granularity become independent, so the list is amortized over one call while
// the warming window stays bounded and — unlike a Go-side burst issued before
// the call — is overlapped with the FMA work of the preceding blocks. Four
// candidates matches prefetchDistance, the depth the per-pair path was tuned to.
//
// The caller (nyDispatch) guarantees n is even and every slot in range.

// func l2SquaredNyAVX2(q, base *float32, dim int, slots *uint32, n int, out *float32)
TEXT ·l2SquaredNyAVX2(SB), NOSPLIT, $0-48
	MOVQ base+8(FP), BX
	MOVQ slots+24(FP), SI
	MOVQ n+32(FP), CX
	MOVQ out+40(FP), DI

l2block:
	CMPQ CX, $2
	JL   l2ret

	// Warm the block four candidates ahead (slots[i+4], slots[i+5]), whole
	// record each, interleaved. Needs six candidates left for both to exist.
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
	// Resolve this block's two candidate vectors: addr = base + slot*dim*4.
	// The arena is a flat slab, so the slot index scales directly.
	MOVQ  q+0(FP), AX
	MOVQ  dim+16(FP), R12
	MOVL  0(SI), R13
	IMULQ R12, R13
	LEAQ  (BX)(R13*4), R8
	MOVL  4(SI), R13
	IMULQ R12, R13
	LEAQ  (BX)(R13*4), R9

	VXORPS Y0, Y0, Y0
	VXORPS Y1, Y1, Y1
	VXORPS Y2, Y2, Y2
	VXORPS Y3, Y3, Y3
	VXORPS Y4, Y4, Y4
	VXORPS Y5, Y5, Y5
	VXORPS Y6, Y6, Y6
	VXORPS Y7, Y7, Y7

l2loop32:
	CMPQ R12, $32
	JL   l2combine

	VMOVUPS 0(AX), Y8
	VMOVUPS 32(AX), Y9
	VMOVUPS 64(AX), Y10
	VMOVUPS 96(AX), Y11

	VMOVUPS     0(R8), Y12
	VMOVUPS     32(R8), Y13
	VMOVUPS     64(R8), Y14
	VMOVUPS     96(R8), Y15
	VSUBPS      Y8, Y12, Y12
	VSUBPS      Y9, Y13, Y13
	VSUBPS      Y10, Y14, Y14
	VSUBPS      Y11, Y15, Y15
	VFMADD231PS Y12, Y12, Y0
	VFMADD231PS Y13, Y13, Y1
	VFMADD231PS Y14, Y14, Y2
	VFMADD231PS Y15, Y15, Y3

	VMOVUPS     0(R9), Y12
	VMOVUPS     32(R9), Y13
	VMOVUPS     64(R9), Y14
	VMOVUPS     96(R9), Y15
	VSUBPS      Y8, Y12, Y12
	VSUBPS      Y9, Y13, Y13
	VSUBPS      Y10, Y14, Y14
	VSUBPS      Y11, Y15, Y15
	VFMADD231PS Y12, Y12, Y4
	VFMADD231PS Y13, Y13, Y5
	VFMADD231PS Y14, Y14, Y6
	VFMADD231PS Y15, Y15, Y7

	ADDQ $128, AX
	ADDQ $128, R8
	ADDQ $128, R9
	SUBQ $32, R12
	JMP  l2loop32

l2combine:
	VADDPS Y1, Y0, Y0
	VADDPS Y3, Y2, Y2
	VADDPS Y2, Y0, Y0
	VADDPS Y5, Y4, Y4
	VADDPS Y7, Y6, Y6
	VADDPS Y6, Y4, Y4

l2loop8:
	CMPQ R12, $8
	JL   l2reduce
	VMOVUPS     (AX), Y8
	VMOVUPS     (R8), Y12
	VSUBPS      Y8, Y12, Y12
	VFMADD231PS Y12, Y12, Y0
	VMOVUPS     (R9), Y13
	VSUBPS      Y8, Y13, Y13
	VFMADD231PS Y13, Y13, Y4
	ADDQ        $32, AX
	ADDQ        $32, R8
	ADDQ        $32, R9
	SUBQ        $8, R12
	JMP         l2loop8

l2reduce:
	VEXTRACTF128 $1, Y0, X12
	VADDPS       X12, X0, X0
	VHADDPS      X0, X0, X0
	VHADDPS      X0, X0, X0
	VEXTRACTF128 $1, Y4, X13
	VADDPS       X13, X4, X4
	VHADDPS      X4, X4, X4
	VHADDPS      X4, X4, X4
	VZEROUPPER

	// Scalar tail. The subtraction runs candidate-minus-query here and
	// query-minus-candidate in the per-pair kernel; the square makes the two
	// bit-identical, so the tail matches as well.
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

// func dotNyAVX2(q, base *float32, dim int, slots *uint32, n int, out *float32)
//
// Identical structure to l2SquaredNyAVX2 with the subtraction dropped: the FMA
// multiplies the query chunk by the candidate chunk directly. Multiplication is
// commutative, so operand order costs nothing against the per-pair kernel.
TEXT ·dotNyAVX2(SB), NOSPLIT, $0-48
	MOVQ base+8(FP), BX
	MOVQ slots+24(FP), SI
	MOVQ n+32(FP), CX
	MOVQ out+40(FP), DI

dotblock:
	CMPQ CX, $2
	JL   dotret

	// Warm four candidates ahead; see l2SquaredNyAVX2.
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

	VXORPS Y0, Y0, Y0
	VXORPS Y1, Y1, Y1
	VXORPS Y2, Y2, Y2
	VXORPS Y3, Y3, Y3
	VXORPS Y4, Y4, Y4
	VXORPS Y5, Y5, Y5
	VXORPS Y6, Y6, Y6
	VXORPS Y7, Y7, Y7

dotloop32:
	CMPQ R12, $32
	JL   dotcombine

	VMOVUPS 0(AX), Y8
	VMOVUPS 32(AX), Y9
	VMOVUPS 64(AX), Y10
	VMOVUPS 96(AX), Y11

	VMOVUPS     0(R8), Y12
	VMOVUPS     32(R8), Y13
	VMOVUPS     64(R8), Y14
	VMOVUPS     96(R8), Y15
	VFMADD231PS Y8, Y12, Y0
	VFMADD231PS Y9, Y13, Y1
	VFMADD231PS Y10, Y14, Y2
	VFMADD231PS Y11, Y15, Y3

	VMOVUPS     0(R9), Y12
	VMOVUPS     32(R9), Y13
	VMOVUPS     64(R9), Y14
	VMOVUPS     96(R9), Y15
	VFMADD231PS Y8, Y12, Y4
	VFMADD231PS Y9, Y13, Y5
	VFMADD231PS Y10, Y14, Y6
	VFMADD231PS Y11, Y15, Y7

	ADDQ $128, AX
	ADDQ $128, R8
	ADDQ $128, R9
	SUBQ $32, R12
	JMP  dotloop32

dotcombine:
	VADDPS Y1, Y0, Y0
	VADDPS Y3, Y2, Y2
	VADDPS Y2, Y0, Y0
	VADDPS Y5, Y4, Y4
	VADDPS Y7, Y6, Y6
	VADDPS Y6, Y4, Y4

dotloop8:
	CMPQ R12, $8
	JL   dotreduce
	VMOVUPS     (AX), Y8
	VMOVUPS     (R8), Y12
	VFMADD231PS Y8, Y12, Y0
	VMOVUPS     (R9), Y13
	VFMADD231PS Y8, Y13, Y4
	ADDQ        $32, AX
	ADDQ        $32, R8
	ADDQ        $32, R9
	SUBQ        $8, R12
	JMP         dotloop8

dotreduce:
	VEXTRACTF128 $1, Y0, X12
	VADDPS       X12, X0, X0
	VHADDPS      X0, X0, X0
	VHADDPS      X0, X0, X0
	VEXTRACTF128 $1, Y4, X13
	VADDPS       X13, X4, X4
	VHADDPS      X4, X4, X4
	VHADDPS      X4, X4, X4
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
