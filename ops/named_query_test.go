// SPDX-License-Identifier: Apache-2.0

package ops

import (
	"testing"

	"google.golang.org/protobuf/proto"

	"github.com/rostamlabs/rostam/grpcapi/pb"
	"github.com/rostamlabs/rostam/vector"
)

// buildNamedQueryCollection creates a 3-space (title dense, image dense, terms
// sparse) named collection via the op handlers and returns a ready tx.
func buildNamedQueryCollection(t *testing.T) *TxContext {
	t.Helper()
	tx := newNamedTx(t)
	cfg := map[string]vector.NamedVectorParams{
		"title": {Dim: 4, Metric: vector.Cosine},
		"image": {Dim: 3, Metric: vector.DotProduct},
		"terms": {Sparse: true},
	}
	if _, err := handleNamedCreate(tx, EncodeNamedCreateArgs("docs", cfg, 0)); err != nil {
		t.Fatalf("create: %v", err)
	}
	pts := []struct {
		id    uint64
		title []float32
		image []float32
		tIdx  []uint32
		tVal  []float32
	}{
		{1, []float32{1, 0, 0, 0}, []float32{1, 0, 0}, []uint32{0, 2}, []float32{1, 0.5}},
		{2, []float32{0, 1, 0, 0}, []float32{0, 1, 0}, []uint32{2, 5}, []float32{2, 1}},
		{3, []float32{0, 0, 1, 0}, []float32{0, 0, 1}, []uint32{5}, []float32{4}},
		{4, []float32{0.9, 0.1, 0, 0}, []float32{0.5, 0.5, 0}, []uint32{0, 5}, []float32{1, 3}},
	}
	for _, p := range pts {
		dense := map[string][]float32{"title": p.title, "image": p.image}
		sparse := map[string]*vector.SparseVector{"terms": {Indices: p.tIdx, Values: p.tVal}}
		args := EncodeNamedInsertArgsSparseCASKeyTTL("docs", p.id, dense, sparse, nil, 0, 0, false, nil)
		if _, err := handleNamedInsert(tx, args); err != nil {
			t.Fatalf("insert %d: %v", p.id, err)
		}
	}
	return tx
}

// namedQuerySpecBlob marshals a named QuerySpec (built from the engine struct via
// the ops to-proto direction) into the op's spec-blob bytes.
func namedQuerySpecBlob(t *testing.T, spec vector.QuerySpec) []byte {
	t.Helper()
	blob, err := MarshalEngineQuerySpec(spec)
	if err != nil {
		t.Fatalf("MarshalEngineQuerySpec: %v", err)
	}
	return blob
}

// TestNamedQueryLeafRoundTrip checks a named-leaf spec survives the proto↔struct
// conversion (QuerySpecToProto → QuerySpecFromProto) carrying the Space, and that
// the leaf lands in the NamedDense / NamedSparse oneof arm (not the dense arm).
func TestNamedQueryLeafRoundTrip(t *testing.T) {
	spec := vector.QuerySpec{
		Mode: vector.ModeRerank,
		Root: vector.QueryLeaf{Kind: vector.LeafDense, Space: "title", Dense: []float32{1, 0, 0, 0}, K: 5, LaneK: 7},
		Prefetch: srcs([]vector.QueryLeaf{
			{Kind: vector.LeafDense, Space: "image", Dense: []float32{0, 1, 0}, K: 6},
			{Kind: vector.LeafSparse, Space: "terms", Sparse: vector.SparseVector{Indices: []uint32{2, 5}, Values: []float32{2, 1}}, K: 6},
		}...),
		Method: vector.FusionRRF,
		K:      5,
	}
	p, err := QuerySpecToProto(spec)
	if err != nil {
		t.Fatalf("QuerySpecToProto: %v", err)
	}
	// The root + prefetch[0] must be NamedDense arms; prefetch[1] a NamedSparse arm.
	if _, ok := p.GetRoot().GetLeaf().(*pb.QueryLeaf_NamedDense); !ok {
		t.Fatalf("root not encoded as NamedDense: %T", p.GetRoot().GetLeaf())
	}
	if _, ok := p.GetPrefetch()[1].GetLeaf().(*pb.QueryLeaf_NamedSparse); !ok {
		t.Fatalf("prefetch[1] not encoded as NamedSparse: %T", p.GetPrefetch()[1].GetLeaf())
	}

	got, err := QuerySpecFromProto(p, 0)
	if err != nil {
		t.Fatalf("QuerySpecFromProto: %v", err)
	}
	if got.Root.Space != "title" || got.Prefetch[0].Leaf.Space != "image" || got.Prefetch[1].Leaf.Space != "terms" {
		t.Fatalf("spaces lost in round-trip: %+v", got)
	}
	if got.Prefetch[1].Leaf.Kind != vector.LeafSparse || len(got.Prefetch[1].Leaf.Sparse.Indices) != 2 {
		t.Fatalf("sparse leaf lost: %+v", got.Prefetch[1])
	}

	// And the full arg round-trip (EncodeQueryArgs + proto marshal) survives.
	blob := namedQuerySpecBlob(t, spec)
	wire := EncodeQueryArgs("docs", blob, 0, 0, 0)
	col, gotBlob, _, _, _, err := DecodeQueryArgs(wire)
	if err != nil || col != "docs" {
		t.Fatalf("DecodeQueryArgs: col=%q err=%v", col, err)
	}
	var pbSpec pb.QuerySpec
	if err := proto.Unmarshal(gotBlob, &pbSpec); err != nil {
		t.Fatalf("unmarshal spec blob: %v", err)
	}
	rt, err := QuerySpecFromProto(&pbSpec, 0)
	if err != nil || rt.Root.Space != "title" {
		t.Fatalf("blob round-trip lost space: %+v err=%v", rt, err)
	}
}

// TestHandleNamedQueryFusion drives the FUSION named query end-to-end through the
// op handler and checks the unfused multi-space lanes come back.
func TestHandleNamedQueryFusion(t *testing.T) {
	tx := buildNamedQueryCollection(t)
	spec := vector.QuerySpec{
		Mode: vector.ModeFusion,
		Prefetch: srcs([]vector.QueryLeaf{
			{Kind: vector.LeafDense, Space: "title", Dense: []float32{0.9, 0.1, 0, 0}},
			{Kind: vector.LeafDense, Space: "image", Dense: []float32{0.5, 0.5, 0}},
			{Kind: vector.LeafSparse, Space: "terms", Sparse: vector.SparseVector{Indices: []uint32{2, 5}, Values: []float32{2, 1}}},
		}...),
		Method: vector.FusionRRF,
		K:      4,
	}
	blob := namedQuerySpecBlob(t, spec)
	body, err := handleNamedQuery(tx, EncodeQueryArgs("docs", blob, 0, 0, 0))
	if err != nil {
		t.Fatalf("handleNamedQuery: %v", err)
	}
	qr, err := DecodeQueryResult(body)
	if err != nil {
		t.Fatalf("DecodeQueryResult: %v", err)
	}
	if qr.Mode != vector.ModeFusion {
		t.Fatalf("mode = %d, want fusion", qr.Mode)
	}
	if len(qr.Lanes) != 3 {
		t.Fatalf("lanes = %d, want 3 (multi-space)", len(qr.Lanes))
	}
	// title lane near (0.9,0.1,..) should rank id 4 (0.9,0.1,..) or 1 high.
	if len(qr.Lanes[0]) == 0 {
		t.Fatalf("title lane empty")
	}
}

// TestHandleNamedQueryRerank drives a RERANK named query end-to-end and checks the
// reranked flat top-k comes back (mode rerank, non-empty).
func TestHandleNamedQueryRerank(t *testing.T) {
	tx := buildNamedQueryCollection(t)
	spec := vector.QuerySpec{
		Mode: vector.ModeRerank,
		Root: vector.QueryLeaf{Kind: vector.LeafDense, Space: "title", Dense: []float32{0.9, 0.1, 0, 0}, K: 3},
		Prefetch: srcs([]vector.QueryLeaf{
			{Kind: vector.LeafDense, Space: "image", Dense: []float32{0.5, 0.5, 0}, K: 6},
			{Kind: vector.LeafSparse, Space: "terms", Sparse: vector.SparseVector{Indices: []uint32{0, 5}, Values: []float32{1, 3}}, K: 6},
		}...),
		K: 3,
	}
	blob := namedQuerySpecBlob(t, spec)
	body, err := handleNamedQuery(tx, EncodeQueryArgs("docs", blob, 0, 0, 0))
	if err != nil {
		t.Fatalf("handleNamedQuery: %v", err)
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

// TestHandleNamedQueryFailLoud checks the handler propagates the engine's
// fail-loud validation: a named query with a Space-less leaf errors.
func TestHandleNamedQueryFailLoud(t *testing.T) {
	tx := buildNamedQueryCollection(t)
	// A Space-less prefetch leaf: QuerySpecToProto encodes it as a dense arm (no
	// space), so the named engine's Query rejects it with ErrQueryNamedLeafNoSpace.
	spec := vector.QuerySpec{
		Mode:     vector.ModeFusion,
		Prefetch: srcs([]vector.QueryLeaf{{Kind: vector.LeafDense, Dense: []float32{1, 0, 0, 0}}}...),
		K:        3,
	}
	blob := namedQuerySpecBlob(t, spec)
	if _, err := handleNamedQuery(tx, EncodeQueryArgs("docs", blob, 0, 0, 0)); err == nil {
		t.Fatalf("expected fail-loud for a Space-less named-query leaf, got nil")
	}
}

// TestReadConsistencyOfNamedQuery checks vector_named_query is recognized by
// ReadConsistencyOf / ReadStalenessOf (so a Linearizable / bounded named query
// arms the shard barrier — anti-silent-stale).
func TestReadConsistencyOfNamedQuery(t *testing.T) {
	spec := &pb.QuerySpec{
		Mode: pb.QueryMode_QUERY_MODE_FUSION,
		Prefetch: []*pb.QueryLeaf{{Leaf: &pb.QueryLeaf_NamedDense{NamedDense: &pb.NamedDenseLeaf{
			Space: "title", Dense: []float32{1, 0, 0, 0},
		}}}},
	}
	blob, _ := proto.Marshal(spec)

	args := EncodeQueryArgs("docs", blob, ConsistencyLinearizable, 0, 0)
	rc, ok := ReadConsistencyOf("vector_named_query", args)
	if !ok || rc != ConsistencyLinearizable {
		t.Fatalf("ReadConsistencyOf = (%d,%v), want (Linearizable,true)", rc, ok)
	}

	argsB := EncodeQueryArgs("docs", blob, ConsistencyBoundedStaleness, 0, 777)
	b, okB := ReadStalenessOf("vector_named_query", argsB)
	if !okB || b != 777 {
		t.Fatalf("ReadStalenessOf = (%d,%v), want (777,true)", b, okB)
	}
}
