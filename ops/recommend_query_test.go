// SPDX-License-Identifier: Apache-2.0

package ops

import (
	"testing"

	"google.golang.org/protobuf/proto"

	"github.com/rostamlabs/rostam/grpcapi/pb"
	"github.com/rostamlabs/rostam/vector"
)

// TestRecommendQueryLeafRoundTrip checks a recommend-leaf spec survives the
// proto↔struct conversion (querySpecToProto → querySpecFromProto) carrying the
// example point-ids, and that the leaf lands in the QueryLeaf_Recommend oneof arm.
// The derive happens in the engine (NOT the codec), so the codec must transport
// the positive/negative ids + k + filter verbatim.
func TestRecommendQueryLeafRoundTrip(t *testing.T) {
	spec := vector.QuerySpec{
		Mode: vector.ModeRerank,
		Root: vector.QueryLeaf{
			Kind:     vector.LeafRecommend,
			Positive: []uint64{1, 7, 42},
			Negative: []uint64{9},
			K:        5,
			Filter:   vector.Filter{Op: vector.FilterEq, Field: "kind", Value: vector.NewString("doc")},
		},
		Prefetch: srcs([]vector.QueryLeaf{
			{Kind: vector.LeafRecommend, Positive: []uint64{3, 4}, K: 6},
			{Kind: vector.LeafDense, Dense: []float32{1, 0, 0, 0}, K: 6},
		}...),
		Method: vector.FusionRRF,
		K:      5,
	}
	p, err := querySpecToProto(spec)
	if err != nil {
		t.Fatalf("querySpecToProto: %v", err)
	}
	// The root + prefetch[0] must land in the Recommend arm.
	rootR, ok := p.GetRoot().GetLeaf().(*pb.QueryLeaf_Recommend)
	if !ok {
		t.Fatalf("root not encoded as Recommend: %T", p.GetRoot().GetLeaf())
	}
	if len(rootR.Recommend.GetPositive()) != 3 || len(rootR.Recommend.GetNegative()) != 1 {
		t.Fatalf("root recommend ids lost: %+v", rootR.Recommend)
	}
	if _, ok := p.GetPrefetch()[0].GetLeaf().(*pb.QueryLeaf_Recommend); !ok {
		t.Fatalf("prefetch[0] not encoded as Recommend: %T", p.GetPrefetch()[0].GetLeaf())
	}

	got, err := querySpecFromProto(p, 0)
	if err != nil {
		t.Fatalf("querySpecFromProto: %v", err)
	}
	if got.Root.Kind != vector.LeafRecommend {
		t.Fatalf("root kind = %d, want LeafRecommend", got.Root.Kind)
	}
	if len(got.Root.Positive) != 3 || got.Root.Positive[0] != 1 || got.Root.Positive[2] != 42 {
		t.Fatalf("root positive ids lost: %v", got.Root.Positive)
	}
	if len(got.Root.Negative) != 1 || got.Root.Negative[0] != 9 {
		t.Fatalf("root negative ids lost: %v", got.Root.Negative)
	}
	if got.Root.K != 5 || got.Root.Filter.Field != "kind" {
		t.Fatalf("root k/filter lost: k=%d filter=%+v", got.Root.K, got.Root.Filter)
	}
	if got.Prefetch[0].Leaf.Kind != vector.LeafRecommend || len(got.Prefetch[0].Leaf.Positive) != 2 {
		t.Fatalf("prefetch recommend leaf lost: %+v", got.Prefetch[0])
	}
	// A recommend leaf is distance-ascending (it rewrites to a dense lane).
	if got.Root.ScoreDesc {
		t.Fatalf("recommend leaf should be distance-ascending (ScoreDesc=false)")
	}

	// Full arg round-trip (EncodeQueryArgs + proto marshal) survives.
	blob, err := MarshalEngineQuerySpec(spec)
	if err != nil {
		t.Fatalf("MarshalEngineQuerySpec: %v", err)
	}
	wire := EncodeQueryArgs("reco", blob, 0, 0, 0)
	col, gotBlob, _, _, _, err := DecodeQueryArgs(wire)
	if err != nil || col != "reco" {
		t.Fatalf("DecodeQueryArgs: col=%q err=%v", col, err)
	}
	var pbSpec pb.QuerySpec
	if err := proto.Unmarshal(gotBlob, &pbSpec); err != nil {
		t.Fatalf("unmarshal spec blob: %v", err)
	}
	rt, err := querySpecFromProto(&pbSpec, 0)
	if err != nil {
		t.Fatalf("querySpecFromProto (blob): %v", err)
	}
	if rt.Root.Kind != vector.LeafRecommend || len(rt.Root.Positive) != 3 {
		t.Fatalf("blob round-trip lost recommend ids: %+v", rt.Root)
	}
}

// TestRecommendBestScoreQueryLeafRoundTrip checks a BEST_SCORE recommend-leaf spec
// survives the proto↔struct conversion carrying the strategy enum + the embedded
// example VECTORS (RecPosVecs/RecNegVecs → best_pos/best_neg), and that a BEST_SCORE
// leaf is SCORE-descending (a custom scorer, unlike AVERAGE_VECTOR's dense rewrite).
func TestRecommendBestScoreQueryLeafRoundTrip(t *testing.T) {
	spec := vector.QuerySpec{
		Mode: vector.ModeFusion,
		Prefetch: srcs([]vector.QueryLeaf{
			{
				Kind:       vector.LeafRecommend,
				Strategy:   vector.RecommendBestScore,
				ScoreDesc:  true,
				Positive:   []uint64{1, 7},
				Negative:   []uint64{9},
				RecPosVecs: [][]float32{{1, 0, 0, 0}, {0, 1, 0, 0}},
				RecNegVecs: [][]float32{{0, 0, 1, 0}},
				K:          5,
			},
		}...),
		K: 5,
	}
	p, err := querySpecToProto(spec)
	if err != nil {
		t.Fatalf("querySpecToProto: %v", err)
	}
	r, ok := p.GetPrefetch()[0].GetLeaf().(*pb.QueryLeaf_Recommend)
	if !ok {
		t.Fatalf("leaf not encoded as Recommend: %T", p.GetPrefetch()[0].GetLeaf())
	}
	if r.Recommend.GetStrategy() != pb.RecommendStrategy_RECOMMEND_BEST_SCORE {
		t.Fatalf("strategy lost: %v", r.Recommend.GetStrategy())
	}
	if len(r.Recommend.GetBestPos()) != 2 || len(r.Recommend.GetBestNeg()) != 1 {
		t.Fatalf("best vectors lost: pos=%d neg=%d", len(r.Recommend.GetBestPos()), len(r.Recommend.GetBestNeg()))
	}
	if len(r.Recommend.GetBestPos()[1].GetValues()) != 4 || r.Recommend.GetBestPos()[1].GetValues()[1] != 1 {
		t.Fatalf("best_pos[1] corrupted: %v", r.Recommend.GetBestPos()[1].GetValues())
	}

	got, err := querySpecFromProto(p, 0)
	if err != nil {
		t.Fatalf("querySpecFromProto: %v", err)
	}
	gl := got.Prefetch[0].Leaf
	if gl.Kind != vector.LeafRecommend || gl.Strategy != vector.RecommendBestScore {
		t.Fatalf("strategy round-trip lost: kind=%d strategy=%d", gl.Kind, gl.Strategy)
	}
	if !gl.ScoreDesc {
		t.Fatalf("BEST_SCORE leaf must be score-descending")
	}
	if len(gl.RecPosVecs) != 2 || len(gl.RecNegVecs) != 1 {
		t.Fatalf("best vectors round-trip lost: pos=%d neg=%d", len(gl.RecPosVecs), len(gl.RecNegVecs))
	}
	if len(gl.Positive) != 2 || gl.Positive[1] != 7 {
		t.Fatalf("positive ids round-trip lost: %v", gl.Positive)
	}

	// Full blob round-trip.
	blob, err := MarshalEngineQuerySpec(spec)
	if err != nil {
		t.Fatalf("MarshalEngineQuerySpec: %v", err)
	}
	var pbSpec pb.QuerySpec
	if err := proto.Unmarshal(blob, &pbSpec); err != nil {
		t.Fatalf("unmarshal blob: %v", err)
	}
	rt, err := querySpecFromProto(&pbSpec, 0)
	if err != nil {
		t.Fatalf("querySpecFromProto (blob): %v", err)
	}
	if rt.Prefetch[0].Leaf.Strategy != vector.RecommendBestScore || len(rt.Prefetch[0].Leaf.RecPosVecs) != 2 {
		t.Fatalf("blob round-trip lost BEST_SCORE payload: %+v", rt.Prefetch[0].Leaf)
	}
}

// TestHandleVectorQueryRecommendBestScore drives a BEST_SCORE recommend leaf
// end-to-end through the vector_query op handler: the engine resolves the example
// ids → vectors from the LOCAL collection, runs the best-score scorer, and returns
// the score-descending lane with the example id excluded.
func TestHandleVectorQueryRecommendBestScore(t *testing.T) {
	tx := buildQueryTestCollection(t)
	spec := &pb.QuerySpec{
		Mode: pb.QueryMode_QUERY_MODE_FUSION,
		Prefetch: []*pb.QueryLeaf{
			{Leaf: &pb.QueryLeaf_Recommend{Recommend: &pb.RecommendLeaf{
				Positive: []uint64{1},
				Strategy: pb.RecommendStrategy_RECOMMEND_BEST_SCORE,
			}}},
		},
		K: 2,
	}
	blob, err := proto.Marshal(spec)
	if err != nil {
		t.Fatal(err)
	}
	body, err := handleVectorQuery(tx, EncodeQueryArgs("docs", blob, 0, 0, 0))
	if err != nil {
		t.Fatalf("handleVectorQuery BEST_SCORE: %v", err)
	}
	qr, err := DecodeQueryResult(body)
	if err != nil {
		t.Fatalf("decode result: %v", err)
	}
	if len(qr.Lanes) != 1 {
		t.Fatalf("BEST_SCORE fusion lanes = %d, want 1", len(qr.Lanes))
	}
	lane := qr.Lanes[0]
	if len(lane) == 0 {
		t.Fatal("BEST_SCORE(+1) lane returned no results")
	}
	for _, r := range lane {
		if r.ID == 1 {
			t.Errorf("example id 1 leaked into BEST_SCORE handler lane %+v", lane)
		}
	}
	// The nearest doc to doc 1 is doc 2; BEST_SCORE ranks it first.
	if lane[0].ID != 2 {
		t.Errorf("BEST_SCORE(+1) top = %d, want doc 2; lane=%+v", lane[0].ID, lane)
	}
}

// TestHandleVectorQueryRecommend drives a recommend leaf end-to-end through the
// vector_query op handler: the engine coordinator pre-pass derives the query
// vector from the LOCAL collection (docs 1-5 on a line near the origin) and
// rewrites it to a dense leaf, then the dense pipeline runs. recommend(+1) must
// surface the near-neighbors (2, 3) with the example id 1 excluded.
func TestHandleVectorQueryRecommend(t *testing.T) {
	tx := buildQueryTestCollection(t)
	spec := &pb.QuerySpec{
		Mode: pb.QueryMode_QUERY_MODE_FUSION,
		Prefetch: []*pb.QueryLeaf{
			{Leaf: &pb.QueryLeaf_Recommend{Recommend: &pb.RecommendLeaf{Positive: []uint64{1}}}},
		},
		K: 2,
	}
	blob, err := proto.Marshal(spec)
	if err != nil {
		t.Fatal(err)
	}
	body, err := handleVectorQuery(tx, EncodeQueryArgs("docs", blob, 0, 0, 0))
	if err != nil {
		t.Fatalf("handleVectorQuery recommend: %v", err)
	}
	qr, err := DecodeQueryResult(body)
	if err != nil {
		t.Fatalf("decode result: %v", err)
	}
	if qr.Mode != vector.ModeFusion {
		t.Fatalf("recommend result mode=%d, want fusion", qr.Mode)
	}
	// The op handler encodes the UNFUSED lanes for a FUSION result (the coordinator
	// fuses); the single recommend lane carries the derived dense search, with the
	// example id pruned. Check lane 0.
	if len(qr.Lanes) != 1 {
		t.Fatalf("recommend fusion lanes = %d, want 1", len(qr.Lanes))
	}
	lane := qr.Lanes[0]
	for _, r := range lane {
		if r.ID == 1 {
			t.Errorf("example id 1 leaked into recommend handler lane %+v", lane)
		}
	}
	if len(lane) == 0 {
		t.Fatal("recommend(+1) lane returned no near-neighbors")
	}
	// The nearest doc to doc 1 (on the line near origin) is doc 2; the far doc 100
	// must rank LAST, never as a near-neighbor.
	if lane[0].ID != 2 {
		t.Errorf("recommend(+1) nearest = %d, want doc 2 (closest line neighbor); lane=%+v", lane[0].ID, lane)
	}
	if lane[len(lane)-1].ID != 100 {
		t.Errorf("far doc 100 should rank last; lane=%+v", lane)
	}
}

// TestHandleVectorQueryRecommendFailLoud checks the recommend fail-loud edges
// through the handler: no positives, and all-positives-missing.
func TestHandleVectorQueryRecommendFailLoud(t *testing.T) {
	tx := buildQueryTestCollection(t)
	mk := func(positive []uint64) []byte {
		spec := &pb.QuerySpec{
			Mode: pb.QueryMode_QUERY_MODE_FUSION,
			Prefetch: []*pb.QueryLeaf{
				{Leaf: &pb.QueryLeaf_Recommend{Recommend: &pb.RecommendLeaf{Positive: positive}}},
			},
			K: 3,
		}
		blob, _ := proto.Marshal(spec)
		return EncodeQueryArgs("docs", blob, 0, 0, 0)
	}
	if _, err := handleVectorQuery(tx, mk(nil)); err == nil {
		t.Error("recommend with no positives should fail loud through the handler")
	}
	if _, err := handleVectorQuery(tx, mk([]uint64{9998, 9999})); err == nil {
		t.Error("recommend with all positives missing should fail loud through the handler")
	}
}
