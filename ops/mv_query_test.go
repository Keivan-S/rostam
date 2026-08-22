// SPDX-License-Identifier: Apache-2.0

package ops

import (
	"testing"

	"google.golang.org/protobuf/proto"

	"github.com/rostamlabs/rostam/grpcapi/pb"
	"github.com/rostamlabs/rostam/vector"
)

// buildMVQueryCollection creates an MV collection (MaxSim tokens + doc-level
// sparse) via the op handlers and returns a ready tx. Some docs omit the sparse
// vector so they contribute only to the MaxSim lane.
func buildMVQueryCollection(t *testing.T) *TxContext {
	t.Helper()
	tx := newNamedTx(t)
	cfg := vector.MultiVectorConfig{Dim: 3, M: 8, EfConstruction: 50, EfSearch: 32, Seed: 7}
	if _, err := handleMVCreate(tx, EncodeMVCreateArgs("mv", cfg)); err != nil {
		t.Fatalf("create: %v", err)
	}
	type doc struct {
		id     uint64
		tokens [][]float32
		sparse *vector.SparseVector
	}
	docs := []doc{
		{1, [][]float32{{1, 0, 0}, {0, 1, 0}}, &vector.SparseVector{Indices: []uint32{0, 2}, Values: []float32{1, 2}}},
		{2, [][]float32{{0, 1, 0}}, &vector.SparseVector{Indices: []uint32{1, 2}, Values: []float32{3, 1}}},
		{3, [][]float32{{0, 0, 1}, {1, 0, 0}}, &vector.SparseVector{Indices: []uint32{0, 5}, Values: []float32{0.2, 4}}},
		{4, [][]float32{{1, 1, 0}}, nil}, // MaxSim-only doc
		{5, [][]float32{{0, 1, 1}}, &vector.SparseVector{Indices: []uint32{2}, Values: []float32{5}}},
	}
	for _, d := range docs {
		args := EncodeMVAddArgsCASKeyTTLSparse("mv", d.id, d.tokens, nil, 0, false, nil, d.sparse)
		if _, err := handleMVAdd(tx, args); err != nil {
			t.Fatalf("add %d: %v", d.id, err)
		}
	}
	return tx
}

// mvQuerySpecBlob marshals an MV QuerySpec (built from the engine struct via the
// ops to-proto direction) into the op's spec-blob bytes.
func mvQuerySpecBlob(t *testing.T, spec vector.QuerySpec) []byte {
	t.Helper()
	blob, err := MarshalEngineQuerySpec(spec)
	if err != nil {
		t.Fatalf("MarshalEngineQuerySpec: %v", err)
	}
	return blob
}

// TestMVQueryLeafRoundTrip checks an MV-MaxSim leaf spec survives the proto↔struct
// conversion (QuerySpecToProto → QuerySpecFromProto), landing in the mv_maxsim
// oneof arm, with the token matrix and the doc-sparse leaf intact.
func TestMVQueryLeafRoundTrip(t *testing.T) {
	spec := vector.QuerySpec{
		Mode: vector.ModeRerank,
		Root: vector.QueryLeaf{Kind: vector.LeafMVMaxSim, Tokens: [][]float32{{1, 0, 0}, {0, 1, 0}}, K: 5, LaneK: 7, ScoreDesc: true},
		Prefetch: srcs([]vector.QueryLeaf{
			{Kind: vector.LeafMVMaxSim, Tokens: [][]float32{{0, 0, 1}}, K: 6, ScoreDesc: true},
			{Kind: vector.LeafSparse, Sparse: vector.SparseVector{Indices: []uint32{0, 2}, Values: []float32{1, 2}}, K: 6, ScoreDesc: true},
		}...),
		Method: vector.FusionRRF,
		K:      5,
	}
	p, err := QuerySpecToProto(spec)
	if err != nil {
		t.Fatalf("QuerySpecToProto: %v", err)
	}
	if _, ok := p.GetRoot().GetLeaf().(*pb.QueryLeaf_MvMaxsim); !ok {
		t.Fatalf("root not encoded as MvMaxsim: %T", p.GetRoot().GetLeaf())
	}
	if _, ok := p.GetPrefetch()[0].GetLeaf().(*pb.QueryLeaf_MvMaxsim); !ok {
		t.Fatalf("prefetch[0] not encoded as MvMaxsim: %T", p.GetPrefetch()[0].GetLeaf())
	}
	if _, ok := p.GetPrefetch()[1].GetLeaf().(*pb.QueryLeaf_Sparse); !ok {
		t.Fatalf("prefetch[1] not encoded as Sparse: %T", p.GetPrefetch()[1].GetLeaf())
	}

	got, err := QuerySpecFromProto(p, 0)
	if err != nil {
		t.Fatalf("QuerySpecFromProto: %v", err)
	}
	if got.Root.Kind != vector.LeafMVMaxSim || len(got.Root.Tokens) != 2 || len(got.Root.Tokens[0]) != 3 {
		t.Fatalf("root maxsim tokens lost: %+v", got.Root)
	}
	if !got.Root.ScoreDesc || !got.Prefetch[0].Leaf.ScoreDesc || !got.Prefetch[1].Leaf.ScoreDesc {
		t.Fatalf("MV leaves must be ScoreDesc after round-trip: %+v", got)
	}
	if got.Prefetch[1].Leaf.Kind != vector.LeafSparse || len(got.Prefetch[1].Leaf.Sparse.Indices) != 2 {
		t.Fatalf("sparse leaf lost: %+v", got.Prefetch[1])
	}

	// Full arg round-trip (EncodeQueryArgs + proto marshal) survives.
	blob := mvQuerySpecBlob(t, spec)
	wire := EncodeQueryArgs("mv", blob, 0, 0, 0)
	col, gotBlob, _, _, _, err := DecodeQueryArgs(wire)
	if err != nil || col != "mv" {
		t.Fatalf("DecodeQueryArgs: col=%q err=%v", col, err)
	}
	var pbSpec pb.QuerySpec
	if err := proto.Unmarshal(gotBlob, &pbSpec); err != nil {
		t.Fatalf("unmarshal spec blob: %v", err)
	}
	rt, err := QuerySpecFromProto(&pbSpec, 0)
	if err != nil || rt.Root.Kind != vector.LeafMVMaxSim || len(rt.Root.Tokens) != 2 {
		t.Fatalf("blob round-trip lost maxsim: %+v err=%v", rt, err)
	}
}

// TestHandleMVQueryFusion drives a FUSION MV query end-to-end through the op
// handler and checks the unfused MaxSim + sparse lanes come back.
func TestHandleMVQueryFusion(t *testing.T) {
	tx := buildMVQueryCollection(t)
	spec := vector.QuerySpec{
		Mode: vector.ModeFusion,
		Prefetch: srcs([]vector.QueryLeaf{
			{Kind: vector.LeafMVMaxSim, Tokens: [][]float32{{1, 0, 0}, {0, 1, 0}}, ScoreDesc: true},
			{Kind: vector.LeafSparse, Sparse: vector.SparseVector{Indices: []uint32{0, 2}, Values: []float32{1, 1}}, ScoreDesc: true},
		}...),
		Method: vector.FusionRRF,
		K:      4,
	}
	blob := mvQuerySpecBlob(t, spec)
	body, err := handleMVQuery(tx, EncodeQueryArgs("mv", blob, 0, 0, 0))
	if err != nil {
		t.Fatalf("handleMVQuery: %v", err)
	}
	qr, err := DecodeQueryResult(body)
	if err != nil {
		t.Fatalf("DecodeQueryResult: %v", err)
	}
	if qr.Mode != vector.ModeFusion {
		t.Fatalf("mode = %d, want fusion", qr.Mode)
	}
	if len(qr.Lanes) != 2 {
		t.Fatalf("lanes = %d, want 2 (maxsim + sparse)", len(qr.Lanes))
	}
	if len(qr.Lanes[0]) == 0 {
		t.Fatalf("maxsim lane empty")
	}
}

// TestHandleMVQueryRerank drives a RERANK MV query end-to-end and checks the
// reranked flat top-k comes back (mode rerank, non-empty).
func TestHandleMVQueryRerank(t *testing.T) {
	tx := buildMVQueryCollection(t)
	spec := vector.QuerySpec{
		Mode: vector.ModeRerank,
		Root: vector.QueryLeaf{Kind: vector.LeafMVMaxSim, Tokens: [][]float32{{0, 1, 0}}, K: 3, ScoreDesc: true},
		Prefetch: srcs([]vector.QueryLeaf{
			{Kind: vector.LeafMVMaxSim, Tokens: [][]float32{{1, 0, 0}}, K: 6, ScoreDesc: true},
			{Kind: vector.LeafSparse, Sparse: vector.SparseVector{Indices: []uint32{0, 2}, Values: []float32{1, 1}}, K: 6, ScoreDesc: true},
		}...),
		K: 3,
	}
	blob := mvQuerySpecBlob(t, spec)
	body, err := handleMVQuery(tx, EncodeQueryArgs("mv", blob, 0, 0, 0))
	if err != nil {
		t.Fatalf("handleMVQuery: %v", err)
	}
	qr, err := DecodeQueryResult(body)
	if err != nil {
		t.Fatalf("DecodeQueryResult: %v", err)
	}
	if qr.Mode != vector.ModeRerank {
		t.Fatalf("mode = %d, want rerank", qr.Mode)
	}
	if len(qr.Fused) == 0 {
		t.Fatalf("rerank fused is empty")
	}
}

// TestHandleMVQueryFailLoud checks the handler propagates the engine's fail-loud
// validation: a MaxSim leaf carrying a Space errors (an MV collection has no named
// spaces). The Space rides via a NamedDense arm, which the MV engine rejects.
func TestHandleMVQueryFailLoud(t *testing.T) {
	tx := buildMVQueryCollection(t)
	// A Space-bearing leaf encodes as a NamedDense arm; the MV engine's Query
	// rejects a dense leaf (LeafDense is not a valid MV node) → fail loud.
	spec := vector.QuerySpec{
		Mode:     vector.ModeFusion,
		Prefetch: srcs([]vector.QueryLeaf{{Kind: vector.LeafDense, Space: "title", Dense: []float32{1, 0, 0}}}...),
		K:        3,
	}
	blob := mvQuerySpecBlob(t, spec)
	if _, err := handleMVQuery(tx, EncodeQueryArgs("mv", blob, 0, 0, 0)); err == nil {
		t.Fatalf("expected fail-loud for a Space-bearing MV-query leaf, got nil")
	}
}

// TestReadConsistencyOfMVQuery checks vector_mv_query is recognized by
// ReadConsistencyOf / ReadStalenessOf (so a Linearizable / bounded MV query arms
// the shard barrier — anti-silent-stale).
func TestReadConsistencyOfMVQuery(t *testing.T) {
	spec := &pb.QuerySpec{
		Mode: pb.QueryMode_QUERY_MODE_FUSION,
		Prefetch: []*pb.QueryLeaf{{Leaf: &pb.QueryLeaf_MvMaxsim{MvMaxsim: &pb.MVMaxSimLeaf{
			Query: []*pb.TokenVector{{Values: []float32{1, 0, 0}}},
		}}}},
	}
	blob, _ := proto.Marshal(spec)

	args := EncodeQueryArgs("mv", blob, ConsistencyLinearizable, 0, 0)
	rc, ok := ReadConsistencyOf("vector_mv_query", args)
	if !ok || rc != ConsistencyLinearizable {
		t.Fatalf("ReadConsistencyOf = (%d,%v), want (Linearizable,true)", rc, ok)
	}

	argsB := EncodeQueryArgs("mv", blob, ConsistencyBoundedStaleness, 0, 555)
	b, okB := ReadStalenessOf("vector_mv_query", argsB)
	if !okB || b != 555 {
		t.Fatalf("ReadStalenessOf = (%d,%v), want (555,true)", b, okB)
	}
}
