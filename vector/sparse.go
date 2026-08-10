// SPDX-License-Identifier: Apache-2.0

package vector

import (
	"bytes"
	"errors"
	"io"
)

// SparseVector is a sparse representation of a document or query: Indices are
// dimension ids (sorted ascending, unique) and Values are the corresponding
// weights. len(Indices) must equal len(Values). The zero value (no indices)
// means "no sparse lane".
//
// Typical producers are learned-sparse models (SPLADE) or client-side BM25 —
// Rostam treats the vector as opaque numeric data and never tokenizes text.
type SparseVector struct {
	Indices []uint32  `json:"indices"`
	Values  []float32 `json:"values"`
}

// ErrSparseMismatch is returned when Indices and Values have different lengths.
var ErrSparseMismatch = errors.New("vector: sparse Indices/Values length mismatch")

// ErrSparseUnsorted is returned when Indices are not strictly ascending (which
// also rejects duplicates).
var ErrSparseUnsorted = errors.New("vector: sparse Indices must be strictly ascending and unique")

// IsZero reports whether the sparse vector carries no terms.
func (s SparseVector) IsZero() bool { return len(s.Indices) == 0 }

// Validate checks the invariants: equal lengths and strictly ascending indices.
// A zero SparseVector is valid (no sparse lane).
func (s SparseVector) Validate() error {
	if len(s.Indices) != len(s.Values) {
		return ErrSparseMismatch
	}
	for i := 1; i < len(s.Indices); i++ {
		if s.Indices[i] <= s.Indices[i-1] {
			return ErrSparseUnsorted
		}
	}
	return nil
}

// clone returns a deep copy so the arena does not alias caller-owned slices.
func (s SparseVector) clone() *SparseVector {
	if s.IsZero() {
		return nil
	}
	idx := make([]uint32, len(s.Indices))
	val := make([]float32, len(s.Values))
	copy(idx, s.Indices)
	copy(val, s.Values)
	return &SparseVector{Indices: idx, Values: val}
}

// cloneSparse deep-copies sv (nil-safe), returning nil for a nil or zero vector
// so callers can store "no sparse value for this space" uniformly. Used by the
// named id-keyed sparse path (which holds *SparseVector, unlike the dense arena's
// value-receiver clone).
func cloneSparse(sv *SparseVector) *SparseVector {
	if sv == nil {
		return nil
	}
	return sv.clone()
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
