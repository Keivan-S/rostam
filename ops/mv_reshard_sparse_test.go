// SPDX-License-Identifier: Apache-2.0

package ops

import (
	"bytes"
	"testing"

	"github.com/rostamlabs/rostam/vector"
)

// TestMVScanResultSparseRoundTrip: the MV scan codec carries the per-record
// OPTIONAL doc-level sparse vector verbatim, alongside tokens/meta/version/
// keyExpires. Indices/values are decoded byte-for-byte; a record with no sparse
// decodes to Sparse==nil.
func TestMVScanResultSparseRoundTrip(t *testing.T) {
	recs := []vector.MultiScanRecord{
		{
			ID:         7,
			Tokens:     [][]float32{{1, 2, 3, 4}, {5, 6, 7, 8}},
			Metadata:   vector.Metadata{"src": vector.NewString("a")},
			Version:    3,
			KeyExpires: map[string]uint64{"temp": 1_700_000_123_456},
			Sparse:     &vector.SparseVector{Indices: []uint32{1, 9, 42}, Values: []float32{0.5, -1.25, 3.0}},
		},
		{ID: 8, Tokens: [][]float32{{0, 0, 0, 0}}, Version: 1}, // no sparse, no key TTL
	}
	got, err := DecodeMVScanResult(EncodeMVScanResult(recs))
	if err != nil {
		t.Fatal(err)
	}
	if len(got) != 2 {
		t.Fatalf("decoded %d records, want 2", len(got))
	}
	if got[0].Sparse == nil {
		t.Fatal("record 0 Sparse = nil, want carried")
	}
	wantIdx := []uint32{1, 9, 42}
	wantVal := []float32{0.5, -1.25, 3.0}
	if len(got[0].Sparse.Indices) != 3 {
		t.Fatalf("record 0 sparse nnz = %d, want 3", len(got[0].Sparse.Indices))
	}
	for i := range wantIdx {
		if got[0].Sparse.Indices[i] != wantIdx[i] || got[0].Sparse.Values[i] != wantVal[i] {
			t.Fatalf("record 0 sparse[%d] = (%d,%v), want (%d,%v)", i, got[0].Sparse.Indices[i], got[0].Sparse.Values[i], wantIdx[i], wantVal[i])
		}
	}
	// keyExpires still round-trips alongside the sparse trailer.
	if got[0].KeyExpires["temp"] != 1_700_000_123_456 {
		t.Fatalf("record 0 KeyExpires = %v, want temp=1700000123456", got[0].KeyExpires)
	}
	if got[1].Sparse != nil {
		t.Fatalf("record 1 Sparse = %v, want nil (no sparse vector)", got[1].Sparse)
	}
}

// TestMVScanResultSparseEmptyByteIdentical: a dense-only MV scan (no sparse, no
// key TTL) encodes BYTE-IDENTICALLY whether the records carry a nil Sparse field
// or not — the sparse trailer contributes only a single present=0 byte (mirroring
// keyExpires), and an explicitly-zero SparseVector is treated as absent.
func TestMVScanResultSparseEmptyByteIdentical(t *testing.T) {
	nilSparse := []vector.MultiScanRecord{{ID: 5, Tokens: [][]float32{{1, 2, 3, 4}}, Version: 2}}
	zeroSparse := []vector.MultiScanRecord{{ID: 5, Tokens: [][]float32{{1, 2, 3, 4}}, Version: 2, Sparse: &vector.SparseVector{}}}
	a := EncodeMVScanResult(nilSparse)
	b := EncodeMVScanResult(zeroSparse)
	if !bytes.Equal(a, b) {
		t.Fatalf("nil vs zero sparse not byte-identical:\n nil =%x\n zero=%x", a, b)
	}
	// The final byte is the sparse present=0 marker.
	if a[len(a)-1] != 0 {
		t.Fatalf("empty-sparse record final byte = %d, want present=0", a[len(a)-1])
	}
	got, err := DecodeMVScanResult(a)
	if err != nil {
		t.Fatal(err)
	}
	if got[0].Sparse != nil {
		t.Fatalf("Sparse = %v, want nil", got[0].Sparse)
	}
}

// TestMVScanResultOldBlobNoSparseTrailerDecodesNil: an OLD MV scan blob (version +
// keyExpires-present byte but NO sparse present byte) decodes per-record to
// Sparse==nil, mirroring the EOF tolerance of the keyExpires trailer. We
// synthesize an old blob by stripping the trailing sparse present=0 byte.
func TestMVScanResultOldBlobNoSparseTrailerDecodesNil(t *testing.T) {
	recs := []vector.MultiScanRecord{{ID: 9, Tokens: [][]float32{{1, 2, 3, 4}}, Version: 4}}
	blob := EncodeMVScanResult(recs)
	old := blob[:len(blob)-1] // drop the sparse present byte → pre-sparse layout
	got, err := DecodeMVScanResult(old)
	if err != nil {
		t.Fatalf("decode old blob: %v", err)
	}
	if len(got) != 1 || got[0].ID != 9 || got[0].Version != 4 || got[0].Sparse != nil {
		t.Fatalf("old blob decode = %+v, want id=9 version=4 Sparse=nil", got[0])
	}
}
