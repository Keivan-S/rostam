// SPDX-License-Identifier: Apache-2.0

package ops

import (
	"bytes"
	"testing"

	"google.golang.org/protobuf/proto"

	"github.com/rostamlabs/rostam/cache"
	"github.com/rostamlabs/rostam/grpcapi/pb"
	"github.com/rostamlabs/rostam/vector"
)

// srcs wraps flat leaves as leaf QuerySources (the 1-level prefetch form).
func srcs(leaves ...vector.QueryLeaf) []vector.QuerySource {
	out := make([]vector.QuerySource, len(leaves))
	for i := range leaves {
		out[i] = vector.LeafSource(leaves[i])
	}
	return out
}

// TestQueryArgsRoundTrip verifies EncodeQueryArgs/DecodeQueryArgs round-trips the
// collection + spec blob, and that the opts trailer is byte-clean (omitted) when
// rc==0 && opa==0, but present (and decoded) otherwise.
func TestQueryArgsRoundTrip(t *testing.T) {
	specBytes := []byte{0x01, 0x02, 0x03, 0xff, 0x00, 0x10}

	// rc==0 && opa==0 → no trailer (byte-clean).
	clean := EncodeQueryArgs("docs", specBytes, 0, 0, 0)
	wantLen := 1 + len("docs") + 4 + len(specBytes)
	if len(clean) != wantLen {
		t.Fatalf("clean args len %d, want %d (no trailer)", len(clean), wantLen)
	}
	col, gotSpec, rc, opa, bound, err := DecodeQueryArgs(clean)
	if err != nil {
		t.Fatalf("DecodeQueryArgs clean: %v", err)
	}
	if col != "docs" || !bytes.Equal(gotSpec, specBytes) || rc != 0 || opa != 0 || bound != 0 {
		t.Fatalf("clean decode mismatch: col=%q spec=%x rc=%d opa=%d bound=%d", col, gotSpec, rc, opa, bound)
	}

	// rc==Linearizable → trailer present, rc survives.
	withRC := EncodeQueryArgs("docs", specBytes, ConsistencyLinearizable, 1, 0)
	if bytes.HasPrefix(withRC, clean) && len(withRC) == len(clean) {
		t.Fatalf("rc trailer not appended")
	}
	col, gotSpec, rc, opa, _, err = DecodeQueryArgs(withRC)
	if err != nil {
		t.Fatalf("DecodeQueryArgs withRC: %v", err)
	}
	if col != "docs" || !bytes.Equal(gotSpec, specBytes) || rc != ConsistencyLinearizable || opa != 1 {
		t.Fatalf("withRC decode mismatch: col=%q rc=%d opa=%d", col, rc, opa)
	}
	// The clean form must be a strict prefix of the with-trailer form (the
	// trailer is purely additive), proving the base bytes are byte-identical.
	if !bytes.Equal(withRC[:len(clean)], clean) {
		t.Fatalf("with-trailer base bytes differ from clean base")
	}

	// bounded staleness → 8 bound bytes ride.
	withBound := EncodeQueryArgs("docs", specBytes, ConsistencyBoundedStaleness, 0, 4242)
	_, _, rc, _, bound, err = DecodeQueryArgs(withBound)
	if err != nil {
		t.Fatalf("DecodeQueryArgs withBound: %v", err)
	}
	if rc != ConsistencyBoundedStaleness || bound != 4242 {
		t.Fatalf("withBound decode: rc=%d bound=%d", rc, bound)
	}
}

// TestQueryResultRoundTrip checks the mode-tagged result encode/decode for both
// modes.
func TestQueryResultRoundTrip(t *testing.T) {
	// RERANK: a flat scored list.
	rerank := vector.QueryResult{
		Mode:  vector.ModeRerank,
		Fused: []vector.Result{{ID: 7, Distance: 1.5, Score: 0.9}, {ID: 3, Distance: 2.0, Score: 0.4}},
	}
	got, err := DecodeQueryResult(EncodeQueryResult(rerank))
	if err != nil {
		t.Fatalf("rerank decode: %v", err)
	}
	if got.Mode != vector.ModeRerank || len(got.Fused) != 2 || got.Fused[0].ID != 7 || got.Fused[1].ID != 3 {
		t.Fatalf("rerank round-trip mismatch: %+v", got)
	}
	if got.Fused[0].Score != 0.9 || got.Fused[0].Distance != 1.5 {
		t.Fatalf("rerank score/distance lost: %+v", got.Fused[0])
	}

	// FUSION: N unfused lanes.
	fusion := vector.QueryResult{
		Mode: vector.ModeFusion,
		Lanes: [][]vector.Result{
			{{ID: 1, Distance: 0.1}, {ID: 2, Distance: 0.2}},
			{{ID: 100, Score: 5.0}},
		},
	}
	got, err = DecodeQueryResult(EncodeQueryResult(fusion))
	if err != nil {
		t.Fatalf("fusion decode: %v", err)
	}
	if got.Mode != vector.ModeFusion || len(got.Lanes) != 2 {
		t.Fatalf("fusion lanes mismatch: %+v", got)
	}
	if len(got.Lanes[0]) != 2 || got.Lanes[0][1].ID != 2 || got.Lanes[1][0].ID != 100 || got.Lanes[1][0].Score != 5.0 {
		t.Fatalf("fusion lane content mismatch: %+v", got.Lanes)
	}
}

// buildQueryTestCollection creates a small dense+sparse collection via the op
// handlers and returns a ready tx.
func buildQueryTestCollection(t *testing.T) *TxContext {
	t.Helper()
	dir := t.TempDir()
	c, _ := cache.New(cache.DefaultConfig())
	t.Cleanup(func() { c.Close() })
	vstore, err := vector.OpenCollectionStore(dir)
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() { vstore.Close() })
	tx := NewTxContextWithVectors(c, vstore)

	cfg := vector.Config{Dim: 4, M: 16, EfConstruction: 200, EfSearch: 64, Seed: 1, Metric: vector.L2}
	if _, err := handleVectorCreateCollection(tx, EncodeCreateCollectionArgs("docs", cfg)); err != nil {
		t.Fatal(err)
	}
	for i := 1; i <= 5; i++ {
		args := EncodeVectorUpsertArgs("docs", uint64(i), []float32{float32(i) * 0.01, 0, 0, 0}, "", 0,
			nil, vector.SparseVector{Indices: []uint32{1}, Values: []float32{0.1}})
		if _, err := handleVectorUpsert(tx, args); err != nil {
			t.Fatalf("upsert %d: %v", i, err)
		}
	}
	args := EncodeVectorUpsertArgs("docs", 100, []float32{9, 9, 9, 9}, "", 0,
		nil, vector.SparseVector{Indices: []uint32{42}, Values: []float32{10.0}})
	if _, err := handleVectorUpsert(tx, args); err != nil {
		t.Fatalf("upsert 100: %v", err)
	}
	return tx
}

// TestHandleVectorQueryFusion drives the FUSION query end-to-end through the
// handler and checks the unfused lanes come back.
func TestHandleVectorQueryFusion(t *testing.T) {
	tx := buildQueryTestCollection(t)
	spec := &pb.QuerySpec{
		Mode: pb.QueryMode_QUERY_MODE_FUSION,
		Prefetch: []*pb.QueryLeaf{
			{Leaf: &pb.QueryLeaf_Dense{Dense: &pb.DenseLeaf{Dense: []float32{0, 0, 0, 0}}}},
			{Leaf: &pb.QueryLeaf_Sparse{Sparse: &pb.SparseLeaf{Indices: []uint32{42}, Values: []float32{5.0}}}},
		},
		FusionMethod: "rrf",
		K:            4,
	}
	blob, err := proto.Marshal(spec)
	if err != nil {
		t.Fatal(err)
	}
	body, err := handleVectorQuery(tx, EncodeQueryArgs("docs", blob, 0, 0, 0))
	if err != nil {
		t.Fatalf("handleVectorQuery fusion: %v", err)
	}
	qr, err := DecodeQueryResult(body)
	if err != nil {
		t.Fatalf("decode result: %v", err)
	}
	if qr.Mode != vector.ModeFusion || len(qr.Lanes) != 2 {
		t.Fatalf("fusion result: mode=%d lanes=%d", qr.Mode, len(qr.Lanes))
	}
	// Doc 100 must be present in the sparse lane (term 42).
	found := false
	for _, r := range qr.Lanes[1] {
		if r.ID == 100 {
			found = true
		}
	}
	if !found {
		t.Errorf("sparse lane should carry doc 100; got %+v", qr.Lanes[1])
	}
}

// TestHandleVectorQueryRerank drives the RERANK query end-to-end through the
// handler.
func TestHandleVectorQueryRerank(t *testing.T) {
	tx := buildQueryTestCollection(t)
	spec := &pb.QuerySpec{
		Mode: pb.QueryMode_QUERY_MODE_RERANK,
		Root: &pb.QueryLeaf{Leaf: &pb.QueryLeaf_Dense{Dense: &pb.DenseLeaf{Dense: []float32{0, 0, 0, 0}}}},
		Prefetch: []*pb.QueryLeaf{
			{Leaf: &pb.QueryLeaf_Dense{Dense: &pb.DenseLeaf{Dense: []float32{0, 0, 0, 0}, K: 4}}},
			{Leaf: &pb.QueryLeaf_Sparse{Sparse: &pb.SparseLeaf{Indices: []uint32{42}, Values: []float32{5.0}, K: 4}}},
		},
		K: 3,
	}
	blob, err := proto.Marshal(spec)
	if err != nil {
		t.Fatal(err)
	}
	body, err := handleVectorQuery(tx, EncodeQueryArgs("docs", blob, 0, 0, 0))
	if err != nil {
		t.Fatalf("handleVectorQuery rerank: %v", err)
	}
	qr, err := DecodeQueryResult(body)
	if err != nil {
		t.Fatalf("decode result: %v", err)
	}
	if qr.Mode != vector.ModeRerank {
		t.Fatalf("rerank result mode=%d", qr.Mode)
	}
	if len(qr.Fused) == 0 {
		t.Fatal("rerank returned no results")
	}
	// Dense root from the origin → the closest dense doc (id 1) ranks first.
	if qr.Fused[0].ID != 1 {
		t.Errorf("rerank top should be doc 1 (closest dense); got %+v", qr.Fused)
	}
}

// TestHandleVectorQueryFailLoud checks fail-loud paths through the handler: an
// unknown fusion method, an empty prefetch, and an empty leaf oneof.
func TestHandleVectorQueryFailLoud(t *testing.T) {
	tx := buildQueryTestCollection(t)

	// Unknown fusion method.
	bad := &pb.QuerySpec{
		Mode:         pb.QueryMode_QUERY_MODE_FUSION,
		FusionMethod: "bogus",
		Prefetch:     []*pb.QueryLeaf{{Leaf: &pb.QueryLeaf_Dense{Dense: &pb.DenseLeaf{Dense: []float32{0, 0, 0, 0}}}}},
	}
	blob, _ := proto.Marshal(bad)
	if _, err := handleVectorQuery(tx, EncodeQueryArgs("docs", blob, 0, 0, 0)); err == nil {
		t.Error("unknown fusion method should fail loud")
	}

	// No prefetch.
	noPrefetch := &pb.QuerySpec{Mode: pb.QueryMode_QUERY_MODE_FUSION}
	blob, _ = proto.Marshal(noPrefetch)
	if _, err := handleVectorQuery(tx, EncodeQueryArgs("docs", blob, 0, 0, 0)); err == nil {
		t.Error("empty prefetch should fail loud")
	}

	// Empty leaf oneof.
	emptyLeaf := &pb.QuerySpec{
		Mode:     pb.QueryMode_QUERY_MODE_FUSION,
		Prefetch: []*pb.QueryLeaf{{}},
	}
	blob, _ = proto.Marshal(emptyLeaf)
	if _, err := handleVectorQuery(tx, EncodeQueryArgs("docs", blob, 0, 0, 0)); err == nil {
		t.Error("empty leaf oneof should fail loud")
	}
}

// TestReadConsistencyOfVectorQuery proves a Linearizable query's rc is recovered
// by ReadConsistencyOf (so the shard barrier arms).
func TestReadConsistencyOfVectorQuery(t *testing.T) {
	spec := &pb.QuerySpec{
		Mode:     pb.QueryMode_QUERY_MODE_FUSION,
		Prefetch: []*pb.QueryLeaf{{Leaf: &pb.QueryLeaf_Dense{Dense: &pb.DenseLeaf{Dense: []float32{0, 0, 0, 0}}}}},
	}
	blob, _ := proto.Marshal(spec)

	args := EncodeQueryArgs("docs", blob, ConsistencyLinearizable, 0, 0)
	rc, ok := ReadConsistencyOf("vector_query", args)
	if !ok || rc != ConsistencyLinearizable {
		t.Fatalf("ReadConsistencyOf = (%d,%v), want (Linearizable,true)", rc, ok)
	}

	// AnyReplica (rc==0) → still recognized as a query op, rc 0.
	args0 := EncodeQueryArgs("docs", blob, 0, 0, 0)
	rc, ok = ReadConsistencyOf("vector_query", args0)
	if !ok || rc != 0 {
		t.Fatalf("ReadConsistencyOf rc0 = (%d,%v), want (0,true)", rc, ok)
	}

	// Bounded staleness bound recovered.
	argsB := EncodeQueryArgs("docs", blob, ConsistencyBoundedStaleness, 0, 999)
	b, okB := ReadStalenessOf("vector_query", argsB)
	if !okB || b != 999 {
		t.Fatalf("ReadStalenessOf = (%d,%v), want (999,true)", b, okB)
	}
}

// TestEncodeQueryResultFlatByteIdentical asserts the FLAT result wire (FUSION lanes /
// RERANK fused) is BYTE-IDENTICAL to the pre-tree-lanes encoding for any spec WITHOUT a
// nested multi-lane FUSION node (SpecHasNestedFusion == false): the additive tree-lanes
// codec must NOT have perturbed the flat path. The expected bytes are reconstructed from
// the documented flat wire format ([mode][nLanes:u32]{block} / [mode][block]) so the
// test pins the exact bytes, not just round-trip equality.
func TestEncodeQueryResultFlatByteIdentical(t *testing.T) {
	laneA := []vector.Result{{ID: 7, Distance: 1.5}, {ID: 9, Distance: 2.5}}
	laneB := []vector.Result{{ID: 7, Score: 0.9}}

	// FUSION: [mode=0][nLanes:u32]{EncodeHybridResults(lane)}…
	gotFusion := EncodeQueryResult(vector.QueryResult{Mode: vector.ModeFusion, Lanes: [][]vector.Result{laneA, laneB}})
	wantFusion := []byte{queryResultModeFusion, 0, 0, 0, 2}
	wantFusion = append(wantFusion, EncodeHybridResults(laneA)...)
	wantFusion = append(wantFusion, EncodeHybridResults(laneB)...)
	if !bytes.Equal(gotFusion, wantFusion) {
		t.Fatalf("FUSION flat encoding changed:\n got=%v\nwant=%v", gotFusion, wantFusion)
	}

	// RERANK: [mode=1]EncodeHybridResults(fused)
	gotRerank := EncodeQueryResult(vector.QueryResult{Mode: vector.ModeRerank, Fused: laneA})
	wantRerank := append([]byte{queryResultModeRerank}, EncodeHybridResults(laneA)...)
	if !bytes.Equal(gotRerank, wantRerank) {
		t.Fatalf("RERANK flat encoding changed:\n got=%v\nwant=%v", gotRerank, wantRerank)
	}

	// A flat FUSION spec and a nested-RERANK-only spec must both report no nested FUSION,
	// so the partition handler takes the flat EncodeQueryResult path (byte-identical).
	flat := vector.QuerySpec{Mode: vector.ModeFusion, Prefetch: srcs(
		vector.QueryLeaf{Kind: vector.LeafDense, Dense: []float32{1}},
		vector.QueryLeaf{Kind: vector.LeafSparse, ScoreDesc: true},
	)}
	if vector.SpecHasNestedFusion(flat) {
		t.Fatal("flat FUSION spec must report no nested FUSION")
	}
	rerankSub := vector.QuerySpec{Mode: vector.ModeRerank, Root: vector.QueryLeaf{Kind: vector.LeafDense, Dense: []float32{1}}, Prefetch: srcs(
		vector.QueryLeaf{Kind: vector.LeafDense, Dense: []float32{1}},
		vector.QueryLeaf{Kind: vector.LeafSparse, ScoreDesc: true},
	)}
	nestedRerankOnly := vector.QuerySpec{Mode: vector.ModeFusion, Prefetch: []vector.QuerySource{
		vector.LeafSource(vector.QueryLeaf{Kind: vector.LeafDense, Dense: []float32{1}}),
		{Spec: &rerankSub},
	}}
	if vector.SpecHasNestedFusion(nestedRerankOnly) {
		t.Fatal("nested-RERANK-only spec must report no nested FUSION")
	}
}

// TestQueryTreeLanesRoundTrip asserts the tree-lanes codec round-trips the flat
// pre-order lane list and is fail-loud on a wrong tag.
func TestQueryTreeLanesRoundTrip(t *testing.T) {
	lanes := [][]vector.Result{
		{{ID: 1, Distance: 0.1}, {ID: 2, Distance: 0.2}},
		{{ID: 1, Score: 0.5}},
		{},
	}
	enc := EncodeQueryTreeLanes(lanes)
	if enc[0] != queryResultModeTreeLanes {
		t.Fatalf("tree-lanes tag = %d, want %d", enc[0], queryResultModeTreeLanes)
	}
	got, err := DecodeQueryTreeLanes(enc)
	if err != nil {
		t.Fatalf("DecodeQueryTreeLanes: %v", err)
	}
	if len(got) != len(lanes) {
		t.Fatalf("lane count = %d, want %d", len(got), len(lanes))
	}
	for i := range lanes {
		if len(got[i]) != len(lanes[i]) {
			t.Fatalf("lane %d len = %d, want %d", i, len(got[i]), len(lanes[i]))
		}
		for j := range lanes[i] {
			if got[i][j] != lanes[i][j] {
				t.Fatalf("lane %d[%d] = %+v, want %+v", i, j, got[i][j], lanes[i][j])
			}
		}
	}
	// DecodeQueryResult routes the tree-lanes tag into Lanes (same field as FUSION).
	qr, err := DecodeQueryResult(enc)
	if err != nil {
		t.Fatalf("DecodeQueryResult(tree-lanes): %v", err)
	}
	if len(qr.Lanes) != len(lanes) {
		t.Fatalf("DecodeQueryResult lanes = %d, want %d", len(qr.Lanes), len(lanes))
	}
	// Fail-loud on a wrong tag.
	if _, err := DecodeQueryTreeLanes([]byte{queryResultModeFusion, 0, 0, 0, 0}); err == nil {
		t.Fatal("DecodeQueryTreeLanes accepted a non-tree-lanes tag")
	}
}
