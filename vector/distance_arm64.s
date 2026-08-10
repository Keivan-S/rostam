#include "textflag.h"

// func dotProductNEON(a, b *float32, n int) float32
//
// Accumulates 4 float32 lanes per iteration into V0 via fused multiply-add,
// reduces the 4 partial sums to a scalar (through a stack slot), then adds the
// n%4 tail with scalar ops.
TEXT ·dotProductNEON(SB), NOSPLIT, $16-28
	MOVD a+0(FP), R0
	MOVD b+8(FP), R1
	MOVD n+16(FP), R2
	VEOR V0.B16, V0.B16, V0.B16

loop4:
	CMP  $4, R2
	BLT  reduce
	VLD1.P 16(R0), [V1.S4]
	VLD1.P 16(R1), [V2.S4]
	VFMLA  V2.S4, V1.S4, V0.S4 // V0 += V1 * V2
	SUB  $4, R2
	B    loop4

reduce:
	MOVD   RSP, R3
	VST1   [V0.S4], (R3)
	FMOVS  (R3), F0
	FMOVS  4(R3), F1
	FADDS  F1, F0, F0
	FMOVS  8(R3), F1
	FADDS  F1, F0, F0
	FMOVS  12(R3), F1
	FADDS  F1, F0, F0

tail:
	CBZ   R2, done
	FMOVS.P 4(R0), F1
	FMOVS.P 4(R1), F2
	FMULS F2, F1, F1
	FADDS F1, F0, F0
	SUB   $1, R2
	B     tail

done:
	FMOVS F0, ret+24(FP)
	RET

// func l2SquaredNEON(a, b *float32, n int) float32
//
// Accumulates 4 lanes of (a-b)^2 per iteration into V0, reduces to a scalar,
// then handles the n%4 tail with scalar ops.
TEXT ·l2SquaredNEON(SB), NOSPLIT, $16-28
	MOVD a+0(FP), R0
	MOVD b+8(FP), R1
	MOVD n+16(FP), R2
	VEOR V0.B16, V0.B16, V0.B16

	// Go's arm64 assembler has no vector float subtract, only VFMLA/VFMLS.
	// Build a ones vector (float32 1.0 = 0x3F800000) so VFMLS computes a-b as
	// a -= b*1.
	MOVW $0x3F800000, R4
	VDUP R4, V4.S4

l2loop4:
	CMP  $4, R2
	BLT  l2reduce
	VLD1.P 16(R0), [V1.S4]
	VLD1.P 16(R1), [V2.S4]
	VFMLS  V4.S4, V2.S4, V1.S4 // V1 -= V2 * V4(=1) = a - b
	VFMLA  V1.S4, V1.S4, V0.S4 // V0 += (a-b)^2
	SUB  $4, R2
	B    l2loop4

l2reduce:
	MOVD   RSP, R3
	VST1   [V0.S4], (R3)
	FMOVS  (R3), F0
	FMOVS  4(R3), F1
	FADDS  F1, F0, F0
	FMOVS  8(R3), F1
	FADDS  F1, F0, F0
	FMOVS  12(R3), F1
	FADDS  F1, F0, F0

l2tail:
	CBZ   R2, l2done
	FMOVS.P 4(R0), F1
	FMOVS.P 4(R1), F2
	FSUBS F2, F1, F1
	FMULS F1, F1, F1
	FADDS F1, F0, F0
	SUB   $1, R2
	B     l2tail

l2done:
	FMOVS F0, ret+24(FP)
	RET

// func prefetchRange(p unsafe.Pointer, nlines int)
//
// Issues nlines consecutive PRFM PLDL1KEEP hints starting at p, one per 64-byte
// cache line (the arm64 counterpart of the amd64 PREFETCHT0 loop): load, into
// L1, retained. Like PREFETCHT0 a PRFM never faults and never traps.
TEXT ·prefetchRange(SB), NOSPLIT, $0-16
	MOVD p+0(FP), R0
	MOVD nlines+8(FP), R1
	CMP  $0, R1
	BLE  prefdone

prefloop:
	PRFM (R0), PLDL1KEEP
	ADD  $64, R0
	SUB  $1, R1
	CBNZ R1, prefloop

prefdone:
	RET
