#include "textflag.h"

// func fastScanAVX2(tbl *uint8, m int, blockCodes *byte, out *uint16, nVec int)
//
// AVX2 in-register LUT16 fast-scan. Scores a TRANSPOSED block of nVec (<=32)
// database vectors: blockCodes is m rows of 32 bytes (row s = subspace s's nibble
// for all 32 lanes, low 4 bits of each byte). For each subspace s it broadcasts
// the 16-entry uint8 LUT tbl[s*16:s*16+16] into both 128-bit halves of a YMM,
// VPSHUFB-looks it up by the 32 code bytes (32 contributions, one per lane), then
// zero-extends the low/high 16 bytes to uint16 and adds into two uint16 lane
// accumulators (Y0 = lanes 0..15, Y1 = lanes 16..31). After m subspaces each lane
// holds Σ_s tbl[s*16 + code_lane[s]] — identical to fastScanScalar.
//
// The accumulators are spilled to a 64-byte stack buffer, then only out[:nVec] is
// copied out, so a partial block (nVec<32) never writes past out[nVec-1]. Reading
// a full 32-byte row is always in bounds (the row is a 32-byte slice of the
// m*32-byte scratch block). VPSHUFB zeroes any lane whose index has bit 7 set;
// nibbles are 0..15 so every lookup is in-table.
//
// Stack: $64 for the spill buffer.
TEXT ·fastScanAVX2(SB), NOSPLIT, $64-40
	MOVQ tbl+0(FP), AX        // AX = &tbl[0]
	MOVQ m+8(FP), CX          // CX = m (subspace count)
	MOVQ blockCodes+16(FP), BX // BX = &blockCodes[0]
	MOVQ out+24(FP), DI       // DI = &out[0]
	MOVQ nVec+32(FP), SI      // SI = nVec

	VPXOR Y0, Y0, Y0          // lane accumulators 0..15 (uint16)
	VPXOR Y1, Y1, Y1          // lane accumulators 16..31 (uint16)

	MOVQ AX, R8               // R8 walks the per-subspace LUT (tbl + s*16)
	MOVQ BX, R9               // R9 walks the per-subspace code row (+32 each)
	MOVQ CX, R10              // R10 = remaining subspaces

fs_subspace:
	CMPQ R10, $0
	JE   fs_reduce
	VBROADCASTI128 (R8), Y2   // Y2 = LUT[s] in both 128-bit lanes
	VMOVDQU        (R9), Y3   // Y3 = 32 code bytes (lanes 0..31)
	VPSHUFB        Y3, Y2, Y4 // Y4 = 32 looked-up uint8 contributions
	// zero-extend low 16 bytes (lanes 0..15) -> uint16, add into Y0
	VEXTRACTI128 $0, Y4, X5
	VPMOVZXBW    X5, Y5
	VPADDW       Y5, Y0, Y0
	// zero-extend high 16 bytes (lanes 16..31) -> uint16, add into Y1
	VEXTRACTI128 $1, Y4, X6
	VPMOVZXBW    X6, Y6
	VPADDW       Y6, Y1, Y1
	ADDQ $16, R8              // next LUT (16 entries)
	ADDQ $32, R9             // next code row (32 lanes)
	DECQ R10
	JMP  fs_subspace

fs_reduce:
	// Spill the two accumulators to the 64-byte stack buffer, then copy out[:nVec].
	MOVQ  SP, DX
	VMOVDQU Y0, 0(DX)        // lanes 0..15  (16 uint16 = 32 bytes)
	VMOVDQU Y1, 32(DX)       // lanes 16..31 (16 uint16 = 32 bytes)
	VZEROUPPER

fs_copy:
	CMPQ SI, $0
	JE   fs_done
	MOVWLZX (DX), R11        // load one uint16 lane
	MOVW    R11, (DI)        // store into out[i]
	ADDQ    $2, DX
	ADDQ    $2, DI
	DECQ    SI
	JMP     fs_copy

fs_done:
	RET
