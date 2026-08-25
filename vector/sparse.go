// SPDX-License-Identifier: Apache-2.0

package vector

import (
	"bytes"
	"io"
)

// SparseVector, its ErrSparse* sentinels, and its pure IsZero/Validate/Clone
// methods now live in the engine-free vtypes leaf package and are re-exported
// via vtypes_aliases.go. The framing/dot helpers below stay here because they
// depend on the engine wire helpers (writeU32/readU32/…), not just the type.

// cloneSparse deep-copies sv (nil-safe), returning nil for a nil or zero vector
// so callers can store "no sparse value for this space" uniformly. Used by the
// named id-keyed sparse path (which holds *SparseVector, unlike the dense arena's
// value-receiver Clone).
func cloneSparse(sv *SparseVector) *SparseVector {
	if sv == nil {
		return nil
	}
	return sv.Clone()
}

// writeSparseVecFrame encodes sv as the shared sparse-vector framing used by both
// the named WAL and the named snapshot: [nIdx:u32]{idx:u32...}[nVal:u32]{val:f32...}.
// sv MUST be non-nil and validated (sorted/unique/equal lengths). nIdx == nVal
// always (kept distinct so the reader validates length agreement).
func writeSparseVecFrame(buf *bytes.Buffer, sv *SparseVector) {
	_ = writeU32(buf, uint32(len(sv.Indices))) //nolint:gosec
	for _, idx := range sv.Indices {
		_ = writeU32(buf, idx)
	}
	_ = writeU32(buf, uint32(len(sv.Values))) //nolint:gosec
	for _, v := range sv.Values {
		_ = writeF32(buf, v)
	}
}

// readSparseVecFrame is the inverse of writeSparseVecFrame. It returns (sv, true)
// on a well-formed frame, or (nil, false) on a truncated/inconsistent frame
// (mismatched index/value counts) so a WAL replayer stops at the durability
// boundary. A zero-length frame decodes to a zero (empty) SparseVector pointer.
func readSparseVecFrame(r io.Reader) (*SparseVector, bool) {
	nIdx, err := readU32(r)
	if err != nil {
		return nil, false
	}
	idx := make([]uint32, nIdx)
	for i := range idx {
		if idx[i], err = readU32(r); err != nil {
			return nil, false
		}
	}
	nVal, err := readU32(r)
	if err != nil || nVal != nIdx {
		return nil, false
	}
	val := make([]float32, nVal)
	for i := range val {
		if val[i], err = readF32(r); err != nil {
			return nil, false
		}
	}
	return &SparseVector{Indices: idx, Values: val}, true
}

// sparseDot computes the dot product of two sparse vectors via a sorted-merge
// walk over their indices — O(|a|+|b|). Both must be sorted ascending (the
// arena and query paths guarantee this via Validate).
func sparseDot(a, b SparseVector) float32 {
	var sum float32
	i, j := 0, 0
	for i < len(a.Indices) && j < len(b.Indices) {
		switch {
		case a.Indices[i] == b.Indices[j]:
			sum += a.Values[i] * b.Values[j]
			i++
			j++
		case a.Indices[i] < b.Indices[j]:
			i++
		default:
			j++
		}
	}
	return sum
}
