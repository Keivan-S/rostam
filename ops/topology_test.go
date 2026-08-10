// SPDX-License-Identifier: Apache-2.0

package ops

import (
	"errors"
	"reflect"
	"testing"
)

func TestTopologyEncodeDecodeRoundtrip(t *testing.T) {
	in := Topology{
		NumShards: 4,
		Members: []TopologyMember{
			{NodeID: "n1", ServerAddr: "10.0.0.1:7001"},
			{NodeID: "n2", ServerAddr: "10.0.0.2:7001"},
		},
		Leaders: []string{"10.0.0.1:7001", "10.0.0.2:7001", "10.0.0.1:7001", ""},
	}
	b, err := EncodeTopology(in)
	if err != nil {
		t.Fatal(err)
	}
	out, err := DecodeTopology(b)
	if err != nil {
		t.Fatal(err)
	}
	if !reflect.DeepEqual(in, out) {
		t.Errorf("roundtrip mismatch:\n in=%+v\nout=%+v", in, out)
	}
}

func TestTopologyDecodeEmpty(t *testing.T) {
	if _, err := DecodeTopology(nil); err == nil {
		t.Error("expected error decoding nil")
	}
	if _, err := DecodeTopology([]byte{}); err == nil {
		t.Error("expected error decoding empty")
	}
}

func TestRegisterTopologyAddsShardlessOp(t *testing.T) {
	r := NewRegistry()
	want := Topology{
		NumShards: 2,
		Members:   []TopologyMember{{NodeID: "n1", ServerAddr: "a:1"}},
		Leaders:   []string{"a:1", ""},
	}
	src := func() (Topology, error) { return want, nil }
	if err := RegisterTopology(r, src); err != nil {
		t.Fatal(err)
	}
	fn, kind, ke, ok := r.Lookup("__topology__")
	if !ok {
		t.Fatal("__topology__ not registered")
	}
	if kind != OpReadOnly {
		t.Errorf("kind = %v, want OpReadOnly", kind)
	}
	if ke != nil {
		t.Error("__topology__ should be shardless (nil KeyExtractor)")
	}
	result, err := fn(nil, nil)
	if err != nil {
		t.Fatal(err)
	}
	got, err := DecodeTopology(result)
	if err != nil {
		t.Fatal(err)
	}
	if !reflect.DeepEqual(want, got) {
		t.Errorf("handler returned mismatched topology")
	}
}

func TestRegisterTopologySourceError(t *testing.T) {
	r := NewRegistry()
	wantErr := errors.New("boom")
	src := func() (Topology, error) { return Topology{}, wantErr }
	if err := RegisterTopology(r, src); err != nil {
		t.Fatal(err)
	}
	fn, _, _, _ := r.Lookup("__topology__")
	if _, err := fn(nil, nil); err == nil {
		t.Fatal("expected source error to propagate")
	}
}
