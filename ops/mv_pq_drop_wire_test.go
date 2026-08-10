// SPDX-License-Identifier: Apache-2.0

package ops

import (
	"bytes"
	"testing"

	"github.com/rostamlabs/rostam/vector"
)

// TestMVCreateArgsPQDropVecsRoundtrip proves the HNSW-PQ float-drop flag rides the
// MV create wire: a QuantPQ + PQDropVecs config (with a low IVFTrainThreshold)
// round-trips through Encode/Decode unchanged.
func TestMVCreateArgsPQDropVecsRoundtrip(t *testing.T) {
	cfg := vector.MultiVectorConfig{
		Dim: 32, M: 16, EfConstruction: 200, EfSearch: 64, Seed: 9,
		Quant: vector.QuantPQ, RescoreFactor: 3, Partitions: 2,
		IVFTrainThreshold: 500, PQDropVecs: true,
	}
	name, got, err := DecodeMVCreateArgs(EncodeMVCreateArgs("acme/mv", cfg))
	if err != nil {
		t.Fatal(err)
	}
	if name != "acme/mv" || got != cfg {
		t.Fatalf("PQDropVecs roundtrip = %q %+v, want %+v", name, got, cfg)
	}
	if !got.PQDropVecs || got.IVFTrainThreshold != 500 || got.Quant != vector.QuantPQ {
		t.Fatalf("PQDropVecs fields not carried: %+v", got)
	}
}

// TestMVCreateArgsPQDropVecsAnchoredWithoutThreshold proves a PQDropVecs-only
// create (no explicit IVFTrainThreshold) still round-trips: the encoder writes a
// zero threshold word as the anchor for the trailing drop byte, and the decoder
// reads PQDropVecs=true with IVFTrainThreshold=0 (engine default).
func TestMVCreateArgsPQDropVecsAnchoredWithoutThreshold(t *testing.T) {
	cfg := vector.MultiVectorConfig{
		Dim: 16, M: 8, EfConstruction: 100, EfSearch: 50, Seed: 7,
		Quant: vector.QuantPQ, PQDropVecs: true,
	}
	_, got, err := DecodeMVCreateArgs(EncodeMVCreateArgs("docs", cfg))
	if err != nil {
		t.Fatal(err)
	}
	if !got.PQDropVecs {
		t.Fatalf("PQDropVecs lost without explicit threshold: %+v", got)
	}
	if got.IVFTrainThreshold != 0 {
		t.Fatalf("IVFTrainThreshold = %d, want 0 (default) for a PQDropVecs-only create", got.IVFTrainThreshold)
	}
}

// TestMVCreateArgsPQDropVecsByteIdentical proves the default (PQDropVecs=false)
// MV create wire is BYTE-IDENTICAL to the pre-PQDropVecs encoder: the trailing
// drop byte is appended ONLY when true, so an unset config (even one that already
// carried an IVFTrainThreshold) encodes exactly as before. The trailing byte adds
// exactly one byte when set.
func TestMVCreateArgsPQDropVecsByteIdentical(t *testing.T) {
	// A QuantPQ config WITHOUT PQDropVecs but WITH a threshold: this is the
	// pre-PQDropVecs trailing layout (OPQ slot + threshold word, no drop byte).
	base := vector.MultiVectorConfig{
		Dim: 32, M: 16, EfConstruction: 200, EfSearch: 64, Seed: 9,
		Quant: vector.QuantPQ, IVFTrainThreshold: 500,
	}
	baseWire := EncodeMVCreateArgs("docs", base)

	drop := base
	drop.PQDropVecs = true
	dropWire := EncodeMVCreateArgs("docs", drop)

	if len(dropWire) != len(baseWire)+1 {
		t.Fatalf("PQDropVecs encode length = %d, want %d (+1 trailing byte)", len(dropWire), len(baseWire)+1)
	}
	if !bytes.Equal(dropWire[:len(baseWire)], baseWire) {
		t.Fatal("PQDropVecs wire prefix is not byte-identical to the no-drop wire")
	}
	if dropWire[len(baseWire)] != 1 {
		t.Fatalf("trailing PQDropVecs byte = %d, want 1", dropWire[len(baseWire)])
	}

	// A fully default HNSW config (no quant, no threshold, no drop) is unchanged:
	// the trailing block is absent entirely, byte-identical to the pre-PQDropVecs
	// encoder (the fixed-length base wire).
	plain := vector.MultiVectorConfig{
		Dim: 16, M: 8, EfConstruction: 100, EfSearch: 50, Seed: 7,
		Quant: vector.QuantBQ1, RescoreFactor: 4, Persistent: true, Partitions: 8,
	}
	wantLen := 1 + len("docs") + 4 + 4 + 4 + 4 + 8 + 1 + 4 + 1 + 4
	if got := len(EncodeMVCreateArgs("docs", plain)); got != wantLen {
		t.Fatalf("plain HNSW encode length = %d, want %d (no trailing block)", got, wantLen)
	}

	// An old-format payload (the no-drop wire) decodes to PQDropVecs=false: the
	// decoder tolerates the absent trailing byte (backward compatibility).
	_, gotOld, err := DecodeMVCreateArgs(baseWire)
	if err != nil {
		t.Fatal(err)
	}
	if gotOld.PQDropVecs {
		t.Fatal("old-format (no drop byte) wire decoded PQDropVecs=true, want false")
	}
}
