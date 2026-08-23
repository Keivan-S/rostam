// SPDX-License-Identifier: Apache-2.0

package vector

import "github.com/rostamlabs/rostam/vtypes"

// This file re-exports the engine-free data types that now live in the vtypes
// leaf package, so the engine and every existing caller compile unchanged while
// the wire codec and network client can depend on vtypes without pulling in the
// engine. See package vtypes.

// Moved types.
type (
	ValueKind      = vtypes.ValueKind
	Value          = vtypes.Value
	Metadata       = vtypes.Metadata
	SparseVector   = vtypes.SparseVector
	Metric         = vtypes.Metric
	IndexType      = vtypes.IndexType
	QuantMode      = vtypes.QuantMode
	FullTextConfig = vtypes.FullTextConfig
)

// Moved constants (values preserved verbatim, so wire/snapshot encodings are
// byte-identical).
const (
	ValueNone    = vtypes.ValueNone
	ValueString  = vtypes.ValueString
	ValueInt     = vtypes.ValueInt
	ValueFloat   = vtypes.ValueFloat
	ValueBool    = vtypes.ValueBool
	ValueStrings = vtypes.ValueStrings
	ValueInts    = vtypes.ValueInts
	ValueFloats  = vtypes.ValueFloats
	ValueGeo     = vtypes.ValueGeo

	Cosine     = vtypes.Cosine
	L2         = vtypes.L2
	DotProduct = vtypes.DotProduct

	IndexHNSW   = vtypes.IndexHNSW
	IndexIVF    = vtypes.IndexIVF
	IndexVamana = vtypes.IndexVamana
	IndexGPU    = vtypes.IndexGPU

	QuantNone = vtypes.QuantNone
	QuantSQ8  = vtypes.QuantSQ8
	QuantBQ1  = vtypes.QuantBQ1
	QuantPQ   = vtypes.QuantPQ
	QuantSQ   = vtypes.QuantSQ
	QuantPRQ  = vtypes.QuantPRQ
)

// Moved error sentinels (value aliases preserve errors.Is identity) and the
// Value constructors (re-exported so every existing call site is unchanged).
var (
	ErrSparseMismatch = vtypes.ErrSparseMismatch
	ErrSparseUnsorted = vtypes.ErrSparseUnsorted

	NewString  = vtypes.NewString
	NewInt     = vtypes.NewInt
	NewFloat   = vtypes.NewFloat
	NewBool    = vtypes.NewBool
	NewStrings = vtypes.NewStrings
	NewInts    = vtypes.NewInts
	NewFloats  = vtypes.NewFloats
	NewGeo     = vtypes.NewGeo
)
