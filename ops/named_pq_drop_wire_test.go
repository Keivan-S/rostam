// SPDX-License-Identifier: Apache-2.0

package ops

import (
	"bytes"
	"reflect"
	"testing"

	"github.com/rostamlabs/rostam/vector"
)

// TestNamedCreateArgsPQDropVecsRoundtrip proves the HNSW-PQ float-drop flag rides
// the named create wire (which carries per-space params as JSON): a QuantPQ +
// PQDropVecs space round-trips through Encode/Decode unchanged.
func TestNamedCreateArgsPQDropVecsRoundtrip(t *testing.T) {
	cfg := map[string]vector.NamedVectorParams{
		"plain": {Dim: 32, Metric: vector.Cosine, M: 16, EfConstruction: 200},
		"pq": {
			Dim: 32, Metric: vector.L2,
			Quant: vector.QuantPQ, IVFTrainThreshold: 500, PQDropVecs: true,
		},
	}
	col, got, partitions, err := DecodeNamedCreateArgs(EncodeNamedCreateArgs("acme/named", cfg, 4))
	if err != nil {
		t.Fatal(err)
	}
	if col != "acme/named" || partitions != 4 || !reflect.DeepEqual(got, cfg) {
		t.Fatalf("PQDropVecs roundtrip: col=%q partitions=%d cfg=%+v want %+v", col, partitions, got, cfg)
	}
	sp := got["pq"]
	if !sp.PQDropVecs || sp.IVFTrainThreshold != 500 || sp.Quant != vector.QuantPQ {
		t.Fatalf("PQDropVecs space fields not carried: %+v", sp)
	}
}

// TestNamedCreateArgsPQDropVecsByteIdentical proves a named config WITHOUT
// PQDropVecs encodes with no pq_drop_vecs JSON key (the field is omitempty), so
// the wire is byte-identical to the pre-PQDropVecs encoder for an unset config.
func TestNamedCreateArgsPQDropVecsByteIdentical(t *testing.T) {
	cfg := map[string]vector.NamedVectorParams{
		"title": {Dim: 4, Metric: vector.Cosine, M: 8, EfConstruction: 100},
		"pq":    {Dim: 8, Metric: vector.L2, Quant: vector.QuantPQ},
	}
	wire := EncodeNamedCreateArgs("docs", cfg, 0)
	if bytes.Contains(wire, []byte("pq_drop_vecs")) {
		t.Fatal("HNSW named wire unexpectedly contains pq_drop_vecs key for an unset config")
	}
}
