// SPDX-License-Identifier: Apache-2.0

package ops

import (
	"testing"

	"google.golang.org/protobuf/proto"

	"github.com/rostamlabs/rostam/grpcapi/pb"
	"github.com/rostamlabs/rostam/vector"
)

// denseLeafPB builds a minimal dense proto QueryLeaf for the codec tests.
func denseLeafPB(vec []float32) *pb.QueryLeaf {
	return &pb.QueryLeaf{Leaf: &pb.QueryLeaf_Dense{Dense: &pb.DenseLeaf{Dense: vec}}}
}

// TestQuerySourceFlatBackCompat: a v1 QuerySpec carrying the flat `prefetch`
// (repeated QueryLeaf) field — every existing dense/named/MV/recommend/discover
// query — decodes into []QuerySource{leaf} (each leaf lifted into a leaf source),
// byte/behaviour-identical to the pre-recursion path. The additive
// `prefetch_sources` field is absent, so the flat field is the source of truth.
func TestQuerySourceFlatBackCompat(t *testing.T) {
	p := &pb.QuerySpec{
		Mode:     pb.QueryMode_QUERY_MODE_FUSION,
		Prefetch: []*pb.QueryLeaf{denseLeafPB([]float32{1, 0}), denseLeafPB([]float32{0, 1})},
		K:        5,
	}
	spec, err := querySpecFromProto(p, 0)
	if err != nil {
		t.Fatalf("decode flat spec: %v", err)
	}
	if len(spec.Prefetch) != 2 {
		t.Fatalf("prefetch len = %d, want 2", len(spec.Prefetch))
	}
	for i := range spec.Prefetch {
		if !spec.Prefetch[i].IsLeaf() || spec.Prefetch[i].Spec != nil {
			t.Fatalf("flat prefetch[%d] is not a leaf source: %+v", i, spec.Prefetch[i])
		}
		if spec.Prefetch[i].Leaf.Kind != vector.LeafDense {
			t.Fatalf("flat prefetch[%d] kind = %d, want dense", i, spec.Prefetch[i].Leaf.Kind)
		}
	}
}

// TestQuerySourceLeafOnlyEncodesFlat: a leaf-only engine spec MUST encode to the
// v1 flat `prefetch` field (NOT `prefetch_sources`) so the wire stays byte-
// identical to the pre-recursion proto for every existing query.
func TestQuerySourceLeafOnlyEncodesFlat(t *testing.T) {
	spec := vector.QuerySpec{
		Mode: vector.ModeFusion,
		Prefetch: srcs(
			vector.QueryLeaf{Kind: vector.LeafDense, Dense: []float32{1, 0}},
			vector.QueryLeaf{Kind: vector.LeafSparse, Sparse: vector.SparseVector{Indices: []uint32{1}, Values: []float32{0.5}}, ScoreDesc: true},
		),
		K: 3,
	}
	p, err := querySpecToProto(spec)
	if err != nil {
		t.Fatalf("encode leaf-only spec: %v", err)
	}
	if len(p.GetPrefetch()) != 2 {
		t.Fatalf("flat prefetch len = %d, want 2 (leaf-only must use the v1 field)", len(p.GetPrefetch()))
	}
	if len(p.GetPrefetchSources()) != 0 {
		t.Fatalf("prefetch_sources len = %d, want 0 (leaf-only must NOT use the nested field)", len(p.GetPrefetchSources()))
	}
}

// TestQuerySourceNestedRoundTrip: a spec whose prefetch contains a NESTED
// QuerySpec source survives the codec (To then From) intact — the nested sub-spec
// (its own prefetch + mode + k) decodes back identically, riding the additive
// `prefetch_sources` field. Proves the recursion-capable wire.
func TestQuerySourceNestedRoundTrip(t *testing.T) {
	sub := vector.QuerySpec{
		Mode: vector.ModeFusion,
		Prefetch: srcs(
			vector.QueryLeaf{Kind: vector.LeafDense, Dense: []float32{1, 0, 0}},
			vector.QueryLeaf{Kind: vector.LeafDense, Dense: []float32{0, 1, 0}},
		),
		Method: vector.FusionRRF,
		K:      7,
	}
	spec := vector.QuerySpec{
		Mode: vector.ModeFusion,
		Prefetch: []vector.QuerySource{
			vector.LeafSource(vector.QueryLeaf{Kind: vector.LeafDense, Dense: []float32{1, 1, 1}}),
			{Spec: &sub},
		},
		K: 4,
	}

	p, err := querySpecToProto(spec)
	if err != nil {
		t.Fatalf("encode nested spec: %v", err)
	}
	// A nested spec MUST ride prefetch_sources (not the flat field).
	if len(p.GetPrefetchSources()) != 2 || len(p.GetPrefetch()) != 0 {
		t.Fatalf("nested encode: prefetch_sources=%d prefetch=%d, want 2/0", len(p.GetPrefetchSources()), len(p.GetPrefetch()))
	}

	// Marshal + unmarshal so the round-trip is the real wire, then decode.
	blob, merr := proto.Marshal(p)
	if merr != nil {
		t.Fatalf("marshal: %v", merr)
	}
	var p2 pb.QuerySpec
	if uerr := proto.Unmarshal(blob, &p2); uerr != nil {
		t.Fatalf("unmarshal: %v", uerr)
	}
	got, derr := querySpecFromProto(&p2, 0)
	if derr != nil {
		t.Fatalf("decode nested spec: %v", derr)
	}
	if len(got.Prefetch) != 2 {
		t.Fatalf("decoded prefetch len = %d, want 2", len(got.Prefetch))
	}
	if !got.Prefetch[0].IsLeaf() {
		t.Fatalf("decoded prefetch[0] should be a leaf source: %+v", got.Prefetch[0])
	}
	if got.Prefetch[1].Spec == nil {
		t.Fatalf("decoded prefetch[1] should be a nested spec source: %+v", got.Prefetch[1])
	}
	gotSub := got.Prefetch[1].Spec
	if gotSub.K != 7 || len(gotSub.Prefetch) != 2 || gotSub.Method != vector.FusionRRF {
		t.Fatalf("nested sub-spec lost: k=%d prefetch=%d method=%d", gotSub.K, len(gotSub.Prefetch), gotSub.Method)
	}
	for i := range gotSub.Prefetch {
		if !gotSub.Prefetch[i].IsLeaf() || gotSub.Prefetch[i].Leaf.Kind != vector.LeafDense {
			t.Fatalf("nested sub-spec prefetch[%d] not a dense leaf: %+v", i, gotSub.Prefetch[i])
		}
	}
}

// nestProto builds a proto QuerySpec nested `depth` spec-levels deep: the
// innermost spec carries a flat dense prefetch; each wrapping level carries a
// single QuerySource{spec} pointing at the level below. depth==0 is the flat leaf
// spec (no nesting).
func nestProto(depth int) *pb.QuerySpec {
	inner := &pb.QuerySpec{
		Mode:     pb.QueryMode_QUERY_MODE_FUSION,
		Prefetch: []*pb.QueryLeaf{denseLeafPB([]float32{1, 0})},
		K:        3,
	}
	for i := 0; i < depth; i++ {
		inner = &pb.QuerySpec{
			Mode: pb.QueryMode_QUERY_MODE_FUSION,
			PrefetchSources: []*pb.QuerySource{
				{Source: &pb.QuerySource_Spec{Spec: inner}},
			},
			K: 3,
		}
	}
	return inner
}

// TestQuerySourceDepthBound: the decode-time depth bound (maxQueryDepth) rejects a
// nested tree deeper than the bound fail-loud with ErrQuerySpecTooDeep BEFORE any
// engine work (anti-DoS); a tree AT the bound decodes fine.
func TestQuerySourceDepthBound(t *testing.T) {
	// At the bound: maxQueryDepth nested spec-levels (root at depth 0, the deepest
	// spec source decoded at depth maxQueryDepth) must decode without error.
	atBound := nestProto(maxQueryDepth)
	if _, err := querySpecFromProto(atBound, 0); err != nil {
		t.Fatalf("spec nested to the bound (%d) should decode, got: %v", maxQueryDepth, err)
	}

	// One past the bound: ErrQuerySpecTooDeep, fail-loud at decode.
	tooDeep := nestProto(maxQueryDepth + 1)
	_, err := querySpecFromProto(tooDeep, 0)
	if err != vector.ErrQuerySpecTooDeep {
		t.Fatalf("over-deep spec err = %v, want ErrQuerySpecTooDeep", err)
	}
}
