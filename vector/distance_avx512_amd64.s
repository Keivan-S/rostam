#include "textflag.h"

// AVX-512 distance kernels (16 float32 lanes per ZMM register vs 8 for AVX2).
// Same contract as the AVX2 kernels in distance_amd64.s: pointer+length in, a
// float32 out, scalar tail for the n%16 remainder. Selected at init() only when
// detectAVX512F() reports AVX512F + OS ZMM state (see distance_amd64.go); on any
// CPU without it the AVX2 (or scalar) path stays in force.
//
// On AMD Zen 4 (the validation box) 512-bit ops are executed over a 256-bit
// datapath, so the win over AVX2 is instruction/load density (one 64-byte load
// + one FMA per 16 lanes instead of two of each), largest at high dims.

// func dotProductAVX512(a, b *float32, n int) float32
//
// Four independent ZMM FMA accumulators (Z0..Z3), 64 floats per iteration, then
// a 16-wide loop and a scalar tail. Multiple accumulators hide FMA latency.
TEXT ·dotProductAVX512(SB), NOSPLIT, $0-28
	MOVQ a+0(FP), AX
	MOVQ b+8(FP), BX
	MOVQ n+16(FP), CX
	VXORPS Z0, Z0, Z0
	VXORPS Z1, Z1, Z1
	VXORPS Z2, Z2, Z2
	VXORPS Z3, Z3, Z3

loop64:
	CMPQ CX, $64
	JL   combine
	VMOVUPS 0(AX), Z4
	VMOVUPS 0(BX), Z5
	VFMADD231PS Z4, Z5, Z0
	VMOVUPS 64(AX), Z6
	VMOVUPS 64(BX), Z7
	VFMADD231PS Z6, Z7, Z1
	VMOVUPS 128(AX), Z8
	VMOVUPS 128(BX), Z9
	VFMADD231PS Z8, Z9, Z2
	VMOVUPS 192(AX), Z10
	VMOVUPS 192(BX), Z11
	VFMADD231PS Z10, Z11, Z3
	ADDQ $256, AX
	ADDQ $256, BX
	SUBQ $64, CX
	JMP  loop64

combine:
	VADDPS Z1, Z0, Z0
	VADDPS Z3, Z2, Z2
	VADDPS Z2, Z0, Z0

loop16:
	CMPQ CX, $16
	JL   reduce
	VMOVUPS (AX), Z4
	VMOVUPS (BX), Z5
	VFMADD231PS Z4, Z5, Z0
	ADDQ $64, AX
	ADDQ $64, BX
	SUBQ $16, CX
	JMP  loop16

reduce:
	VEXTRACTF64X4 $1, Z0, Y1
	VADDPS       Y1, Y0, Y0
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

// func l2SquaredAVX512(a, b *float32, n int) float32
//
// Accumulates (a-b)^2 in four ZMM accumulators, 64 floats per iteration, then a
// 16-wide loop and a scalar tail. The square makes the result independent of
// VSUBPS operand order.
TEXT ·l2SquaredAVX512(SB), NOSPLIT, $0-28
	MOVQ a+0(FP), AX
	MOVQ b+8(FP), BX
	MOVQ n+16(FP), CX
	VXORPS Z0, Z0, Z0
	VXORPS Z1, Z1, Z1
	VXORPS Z2, Z2, Z2
	VXORPS Z3, Z3, Z3

l2loop64:
	CMPQ CX, $64
	JL   l2combine
	VMOVUPS 0(AX), Z4
	VMOVUPS 0(BX), Z5
	VSUBPS  Z5, Z4, Z4
	VFMADD231PS Z4, Z4, Z0
	VMOVUPS 64(AX), Z6
	VMOVUPS 64(BX), Z7
	VSUBPS  Z7, Z6, Z6
	VFMADD231PS Z6, Z6, Z1
	VMOVUPS 128(AX), Z8
	VMOVUPS 128(BX), Z9
	VSUBPS  Z9, Z8, Z8
	VFMADD231PS Z8, Z8, Z2
	VMOVUPS 192(AX), Z10
	VMOVUPS 192(BX), Z11
	VSUBPS  Z11, Z10, Z10
	VFMADD231PS Z10, Z10, Z3
	ADDQ $256, AX
	ADDQ $256, BX
	SUBQ $64, CX
	JMP  l2loop64

l2combine:
	VADDPS Z1, Z0, Z0
	VADDPS Z3, Z2, Z2
	VADDPS Z2, Z0, Z0

l2loop16:
	CMPQ CX, $16
	JL   l2reduce
	VMOVUPS (AX), Z4
	VMOVUPS (BX), Z5
	VSUBPS  Z5, Z4, Z4
	VFMADD231PS Z4, Z4, Z0
	ADDQ $64, AX
	ADDQ $64, BX
	SUBQ $16, CX
	JMP  l2loop16

l2reduce:
	VEXTRACTF64X4 $1, Z0, Y1
	VADDPS       Y1, Y0, Y0
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
