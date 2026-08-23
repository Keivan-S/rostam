// SPDX-License-Identifier: Apache-2.0

package vtypes

import (
	"errors"
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

// Clone returns a deep copy so callers (e.g. the arena) do not alias
// caller-owned slices. It returns nil for a zero (empty) sparse vector.
func (s SparseVector) Clone() *SparseVector {
	if s.IsZero() {
		return nil
	}
	idx := make([]uint32, len(s.Indices))
	val := make([]float32, len(s.Values))
	copy(idx, s.Indices)
	copy(val, s.Values)
	return &SparseVector{Indices: idx, Values: val}
}
