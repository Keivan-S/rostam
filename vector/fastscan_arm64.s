#include "textflag.h"

// func fastScanNEON(tbl *uint8, m int, blockCodes *byte, out *uint16, nVec int)
//
// NEON in-register LUT16 fast-scan, the arm64 analogue of the AVX2 VPSHUFB kernel.
// blockCodes is m rows of 32 bytes (row s = subspace s's nibble for all 32 lanes).
// For each subspace s: load the 16-byte LUT tbl[s*16:] as a TBL table, look it up
// by the low 16 and high 16 code bytes (TBL zeroes any index >= 16; nibbles are
// 0..15 so every lookup is in-table), widen the looked-up bytes to uint16, and add
// into four .H8 accumulators (V16..V19 = lanes 0..7, 8..15, 16..23, 24..31). After
// m subspaces each lane holds Σ_s tbl[s*16 + code_lane[s]] — identical to
// fastScanScalar.
//
// The accumulators are spilled to a 64-byte stack buffer, then only out[:nVec] is
// copied, so a partial block (nVec<32) never writes past out[nVec-1]. Reading a
// full 32-byte row is always in bounds (a 32-byte slice of the m*32 scratch block).
//
// Stack: $64 spill buffer.
TEXT ·fastScanNEON(SB), NOSPLIT, $64-40
	MOVD tbl+0(FP), R0         // R0 walks per-subspace LUT (tbl + s*16)
	MOVD m+8(FP), R1           // R1 = m
	MOVD blockCodes+16(FP), R2 // R2 walks per-subspace code row (+32 each)
	MOVD out+24(FP), R3        // R3 = &out[0]
	MOVD nVec+32(FP), R4       // R4 = nVec

	// Zero the four uint16 lane accumulators.
	VEOR V16.B16, V16.B16, V16.B16 // lanes 0..7
	VEOR V17.B16, V17.B16, V17.B16 // lanes 8..15
	VEOR V18.B16, V18.B16, V18.B16 // lanes 16..23
	VEOR V19.B16, V19.B16, V19.B16 // lanes 24..31

	MOVD R1, R5                // R5 = remaining subspaces

fs_subspace:
	CBZ  R5, fs_reduce
	VLD1 (R0), [V0.B16]        // V0 = 16-byte LUT for this subspace
	VLD1 (R2), [V1.B16, V2.B16] // V1 = code lanes 0..15, V2 = lanes 16..31
	VTBL V1.B16, [V0.B16], V3.B16 // V3 = looked-up bytes, lanes 0..15
	VTBL V2.B16, [V0.B16], V4.B16 // V4 = looked-up bytes, lanes 16..31
	// widen V3 (16 uint8) -> two .H8 (lanes 0..7 in V5, 8..15 in V6)
	VUSHLL  $0, V3.B8, V5.H8       // lanes 0..7
	VUSHLL2 $0, V3.B16, V6.H8      // lanes 8..15
	VADD    V5.H8, V16.H8, V16.H8
	VADD    V6.H8, V17.H8, V17.H8
	// widen V4 -> lanes 16..23 (V7), 24..31 (V8)
	VUSHLL  $0, V4.B8, V7.H8       // lanes 16..23
	VUSHLL2 $0, V4.B16, V8.H8      // lanes 24..31
	VADD    V7.H8, V18.H8, V18.H8
	VADD    V8.H8, V19.H8, V19.H8
	ADD  $16, R0               // next LUT
	ADD  $32, R2               // next code row
	SUB  $1, R5
	B    fs_subspace

fs_reduce:
	// Spill the four .H8 accumulators (32 uint16 = 64 bytes) to the stack buffer.
	MOVD RSP, R6
	VST1 [V16.H8, V17.H8, V18.H8, V19.H8], (R6)

fs_copy:
	CBZ   R4, fs_done
	MOVHU.P 2(R6), R7          // load one uint16 lane, post-inc R6
	MOVH.P  R7, 2(R3)          // store into out[i], post-inc R3
	SUB   $1, R4
	B     fs_copy

fs_done:
	RET
