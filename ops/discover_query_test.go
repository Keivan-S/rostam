// SPDX-License-Identifier: Apache-2.0

package ops

import (
	"testing"

	"google.golang.org/protobuf/proto"

	"github.com/rostamlabs/rostam/cache"
	"github.com/rostamlabs/rostam/grpcapi/pb"
	"github.com/rostamlabs/rostam/vector"
)

// TestDiscoverQueryLeafRoundTrip checks a discover-leaf spec survives the
// proto↔struct conversion (QuerySpecToProto → QuerySpecFromProto) carrying BOTH
// the resolved target/context VECTORS and the unresolved target/context IDS, and
// that the leaf lands in the QueryLeaf_Discover oneof arm with ScoreDesc=true.
func TestDiscoverQueryLeafRoundTrip(t *testing.T) {
	spec := vector.QuerySpec{
		Mode: vector.ModeRerank,
		Root: vector.QueryLeaf{
			Kind:               vector.LeafDiscover,
			DiscoverTarget:     []float32{0.1, 0.2, 0.3, 0.4},
			DiscoverContext:    []vector.DiscoverPair{{Pos: []float32{1, 0, 0, 0}, Neg: []float32{0, 1, 0, 0}}},
			DiscoverTargetID:   []uint64{2},
			DiscoverContextIDs: []vector.ContextPair{{Positive: 1, Negative: 4}},
			K:                  5,
			Filter:             vector.Filter{Op: vector.FilterEq, Field: "kind", Value: vector.NewString("doc")},
			ScoreDesc:          true,
		},
		Prefetch: srcs([]vector.QueryLeaf{
			{
				Kind:               vector.LeafDiscover,
				DiscoverContextIDs: []vector.ContextPair{{Positive: 3, Negative: 9}},
				K:                  6,
				ScoreDesc:          true,
			},
			{Kind: vector.LeafDense, Dense: []float32{1, 0, 0, 0}, K: 6},
		}...),
		Method: vector.FusionRRF,
		K:      5,
	}
	p, err := QuerySpecToProto(spec)
	if err != nil {
		t.Fatalf("QuerySpecToProto: %v", err)
	}
	// The root + prefetch[0] must land in the Discover arm.
	rootD, ok := p.GetRoot().GetLeaf().(*pb.QueryLeaf_Discover)
	if !ok {
		t.Fatalf("root not encoded as Discover: %T", p.GetRoot().GetLeaf())
	}
	if len(rootD.Discover.GetTarget()) != 4 || len(rootD.Discover.GetContext()) != 1 {
		t.Fatalf("root discover vectors lost: %+v", rootD.Discover)
	}
	if len(rootD.Discover.GetTargetId()) != 1 || len(rootD.Discover.GetContextIds()) != 1 {
		t.Fatalf("root discover ids lost: %+v", rootD.Discover)
	}
	if _, ok := p.GetPrefetch()[0].GetLeaf().(*pb.QueryLeaf_Discover); !ok {
		t.Fatalf("prefetch[0] not encoded as Discover: %T", p.GetPrefetch()[0].GetLeaf())
	}

	got, err := QuerySpecFromProto(p, 0)
	if err != nil {
		t.Fatalf("QuerySpecFromProto: %v", err)
	}
	if got.Root.Kind != vector.LeafDiscover {
		t.Fatalf("root kind = %d, want LeafDiscover", got.Root.Kind)
	}
	if len(got.Root.DiscoverTarget) != 4 || got.Root.DiscoverTarget[3] != 0.4 {
		t.Fatalf("root target vector lost: %v", got.Root.DiscoverTarget)
	}
	if len(got.Root.DiscoverContext) != 1 || got.Root.DiscoverContext[0].Pos[0] != 1 {
		t.Fatalf("root context pair lost: %+v", got.Root.DiscoverContext)
	}
	if len(got.Root.DiscoverTargetID) != 1 || got.Root.DiscoverTargetID[0] != 2 {
		t.Fatalf("root target id lost: %v", got.Root.DiscoverTargetID)
	}
	if len(got.Root.DiscoverContextIDs) != 1 || got.Root.DiscoverContextIDs[0].Positive != 1 || got.Root.DiscoverContextIDs[0].Negative != 4 {
		t.Fatalf("root context ids lost: %+v", got.Root.DiscoverContextIDs)
	}
	if got.Root.K != 5 || got.Root.Filter.Field != "kind" {
		t.Fatalf("root k/filter lost: k=%d filter=%+v", got.Root.K, got.Root.Filter)
	}
	// A discover leaf is a custom per-candidate scorer → score-descending.
	if !got.Root.ScoreDesc {
		t.Fatalf("discover leaf should be score-descending (ScoreDesc=true)")
	}
	if got.Prefetch[0].Leaf.Kind != vector.LeafDiscover || len(got.Prefetch[0].Leaf.DiscoverContextIDs) != 1 {
		t.Fatalf("prefetch discover leaf lost: %+v", got.Prefetch[0])
	}

	// Full arg round-trip (EncodeQueryArgs + proto marshal) survives.
	blob, err := MarshalEngineQuerySpec(spec)
	if err != nil {
		t.Fatalf("MarshalEngineQuerySpec: %v", err)
	}
	wire := EncodeQueryArgs("disc", blob, 0, 0, 0)
	col, gotBlob, _, _, _, err := DecodeQueryArgs(wire)
	if err != nil || col != "disc" {
		t.Fatalf("DecodeQueryArgs: col=%q err=%v", col, err)
	}
	var pbSpec pb.QuerySpec
	if err := proto.Unmarshal(gotBlob, &pbSpec); err != nil {
		t.Fatalf("unmarshal spec blob: %v", err)
	}
	rt, err := QuerySpecFromProto(&pbSpec, 0)
	if err != nil {
		t.Fatalf("QuerySpecFromProto (blob): %v", err)
	}
	if rt.Root.Kind != vector.LeafDiscover || len(rt.Root.DiscoverContextIDs) != 1 || len(rt.Root.DiscoverTarget) != 4 {
		t.Fatalf("blob round-trip lost discover payload: %+v", rt.Root)
	}
}

// TestHandleVectorQueryDiscover drives a discover leaf end-to-end through the
// vector_query op handler: the engine resolve pre-pass resolves the context-pair
// ids → vectors on the LOCAL collection (two clusters), and the discover scorer
// steers the top-k toward the positive-side cluster.
func TestHandleVectorQueryDiscover(t *testing.T) {
	tx := buildDiscoverTestCollection(t)

	// target between the clusters; context pair steers to cluster A (ids 1-3).
	spec := vector.QuerySpec{
		Mode: vector.ModeFusion,
		Prefetch: srcs([]vector.QueryLeaf{
			{
				Kind:               vector.LeafDiscover,
				DiscoverTarget:     []float32{0.7, 0.7},
				DiscoverContextIDs: []vector.ContextPair{{Positive: 1, Negative: 4}},
				K:                  3,
				ScoreDesc:          true,
			},
		}...),
		K: 3,
	}
	blob, err := MarshalEngineQuerySpec(spec)
	if err != nil {
		t.Fatalf("MarshalEngineQuerySpec: %v", err)
	}
	out, err := handleVectorQuery(tx, EncodeQueryArgs("disc", blob, 0, 0, 0))
	if err != nil {
		t.Fatalf("handleVectorQuery: %v", err)
	}
	qr, err := DecodeQueryResult(out)
	if err != nil {
		t.Fatalf("DecodeQueryResult: %v", err)
	}
	if qr.Mode != vector.ModeFusion || len(qr.Lanes) != 1 {
		t.Fatalf("mode=%d lanes=%d, want fusion + 1 lane", qr.Mode, len(qr.Lanes))
	}
	if len(qr.Lanes[0]) == 0 {
		t.Fatal("discover lane returned no results")
	}
	// The discover scorer steers to cluster A: every result with a positive context
	// score (closer to the positive than the negative) must be a cluster-A id; the
	// top result must be a cluster-A id.
	if top := qr.Lanes[0][0].ID; top >= 4 {
		t.Errorf("discover top = %d, want a cluster-A id (1-3): lane=%+v", top, qr.Lanes[0])
	}
}

// buildDiscoverTestCollection builds a TxContext with one dense cosine collection
// holding two clusters (ids 1-3 near [1,0], ids 4-6 near [0,1]) — the corpus the
// discover scorer steers over.
func buildDiscoverTestCollection(t *testing.T) *TxContext {
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

	cfg := vector.Config{Dim: 2, M: 16, EfConstruction: 200, EfSearch: 64, Seed: 1, Metric: vector.Cosine}
	if _, err := handleVectorCreateCollection(tx, EncodeCreateCollectionArgs("disc", cfg)); err != nil {
		t.Fatal(err)
	}
	corpus := map[uint64][]float32{
		1: {1, 0.02}, 2: {1, -0.02}, 3: {0.98, 0.05},
		4: {0.02, 1}, 5: {-0.02, 1}, 6: {0.05, 0.98},
	}
	for id, v := range corpus {
		args := EncodeVectorUpsertArgs("disc", id, v, "", 0, nil, vector.SparseVector{})
		if _, err := handleVectorUpsert(tx, args); err != nil {
			t.Fatalf("upsert %d: %v", id, err)
		}
	}
	return tx
}
