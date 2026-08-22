// SPDX-License-Identifier: Apache-2.0

package ops

import (
	"testing"

	"google.golang.org/protobuf/proto"

	"github.com/rostamlabs/rostam/grpcapi/pb"
	"github.com/rostamlabs/rostam/vector"
)

// TestQuerySpecGroupRoundTrip checks group_by / group_size survive the engine→proto
// →marshal→unmarshal→engine round-trip, and that an empty group_by decodes to a flat
// spec (the byte-identical no-group path).
func TestQuerySpecGroupRoundTrip(t *testing.T) {
	spec := vector.QuerySpec{
		Mode:      vector.ModeFusion,
		Method:    vector.FusionRRF,
		Prefetch:  srcs(vector.QueryLeaf{Kind: vector.LeafDense, Dense: []float32{0, 0, 0, 0}}),
		K:         3,
		GroupBy:   "doc_id",
		GroupSize: 4,
	}
	blob, err := MarshalEngineQuerySpec(spec)
	if err != nil {
		t.Fatalf("MarshalEngineQuerySpec: %v", err)
	}
	var p pb.QuerySpec
	if err := proto.Unmarshal(blob, &p); err != nil {
		t.Fatalf("unmarshal: %v", err)
	}
	if p.GetGroupBy() != "doc_id" || p.GetGroupSize() != 4 {
		t.Fatalf("proto group fields lost: group_by=%q group_size=%d", p.GetGroupBy(), p.GetGroupSize())
	}
	got, err := QuerySpecFromProto(&p, 0)
	if err != nil {
		t.Fatalf("QuerySpecFromProto: %v", err)
	}
	if got.GroupBy != "doc_id" || got.GroupSize != 4 {
		t.Fatalf("engine group fields lost: GroupBy=%q GroupSize=%d", got.GroupBy, got.GroupSize)
	}

	// Empty group_by ⇒ flat spec (no-group path).
	flat := spec
	flat.GroupBy = ""
	flat.GroupSize = 0
	fblob, _ := MarshalEngineQuerySpec(flat)
	var fp pb.QuerySpec
	if err := proto.Unmarshal(fblob, &fp); err != nil {
		t.Fatalf("flat unmarshal: %v", err)
	}
	if fp.GetGroupBy() != "" || fp.GetGroupSize() != 0 {
		t.Fatalf("flat spec carries group fields: %q %d", fp.GetGroupBy(), fp.GetGroupSize())
	}
}

// TestQueryResultGroupedRoundTrip checks the grouped-result encode/decode (the new
// queryResultModeGrouped tag wrapping EncodeGroups).
func TestQueryResultGroupedRoundTrip(t *testing.T) {
	qr := vector.QueryResult{
		Mode: vector.ModeRerank,
		Groups: []vector.Group{
			{Key: vector.NewInt(1), Hits: []vector.Document{{ID: 1, Distance: 0.1}, {ID: 2, Distance: 0.2}}},
			{Key: vector.NewInt(2), Hits: []vector.Document{{ID: 3, Distance: 0.3}}},
		},
	}
	got, err := DecodeQueryResult(EncodeQueryResult(qr))
	if err != nil {
		t.Fatalf("grouped decode: %v", err)
	}
	if len(got.Groups) != 2 {
		t.Fatalf("want 2 groups, got %d: %+v", len(got.Groups), got.Groups)
	}
	if got.Groups[0].Key.Int != 1 || len(got.Groups[0].Hits) != 2 || got.Groups[0].Hits[0].ID != 1 {
		t.Fatalf("group0 mismatch: %+v", got.Groups[0])
	}
	if got.Groups[1].Key.Int != 2 || len(got.Groups[1].Hits) != 1 || got.Groups[1].Hits[0].ID != 3 {
		t.Fatalf("group1 mismatch: %+v", got.Groups[1])
	}
	// A flat result (Groups nil) must NOT take the grouped tag.
	flat := vector.QueryResult{Mode: vector.ModeRerank, Fused: []vector.Result{{ID: 7, Distance: 1.5}}}
	if EncodeQueryResult(flat)[0] != QueryResultModeRerank {
		t.Fatal("flat result took a non-rerank mode tag")
	}
}

// TestQueryResultGroupedFanOutRoundTrip checks the PER-PARTITION grouped fan-out
// codec: the flat (FUSION lanes / RERANK fused) result + per-id group-key map survive
// the encode/decode the coordinator consumes. Both modes are exercised.
func TestQueryResultGroupedFanOutRoundTrip(t *testing.T) {
	keys := map[uint64]vector.Value{1: vector.NewInt(10), 2: vector.NewString("x"), 3: vector.NewInt(20)}

	t.Run("rerank", func(t *testing.T) {
		qr := vector.QueryResult{Mode: vector.ModeRerank, Fused: []vector.Result{
			{ID: 1, Distance: 0.1, Score: 0.9}, {ID: 2, Distance: 0.2, Score: 0.5}, {ID: 3, Distance: 0.3},
		}}
		gotQR, gotKeys, err := DecodeQueryResultGroupedFanOut(EncodeQueryResultGroupedFanOut(qr, keys))
		if err != nil {
			t.Fatalf("decode: %v", err)
		}
		if gotQR.Mode != vector.ModeRerank || len(gotQR.Fused) != 3 || gotQR.Fused[0].ID != 1 {
			t.Fatalf("flat rerank mismatch: %+v", gotQR)
		}
		assertKeys(t, gotKeys, keys)
	})

	t.Run("fusion", func(t *testing.T) {
		qr := vector.QueryResult{Mode: vector.ModeFusion, Lanes: [][]vector.Result{
			{{ID: 1, Distance: 0.1}, {ID: 3, Distance: 0.3}},
			{{ID: 2, Score: 0.5}, {ID: 1, Score: 0.4}},
		}}
		gotQR, gotKeys, err := DecodeQueryResultGroupedFanOut(EncodeQueryResultGroupedFanOut(qr, keys))
		if err != nil {
			t.Fatalf("decode: %v", err)
		}
		if gotQR.Mode != vector.ModeFusion || len(gotQR.Lanes) != 2 {
			t.Fatalf("flat fusion mismatch: %+v", gotQR)
		}
		if len(gotQR.Lanes[0]) != 2 || gotQR.Lanes[0][0].ID != 1 || gotQR.Lanes[1][0].ID != 2 {
			t.Fatalf("fusion lanes mismatch: %+v", gotQR.Lanes)
		}
		assertKeys(t, gotKeys, keys)
	})

	// A non-grouped-fan-out body (a plain flat result) is fail-loud here.
	if _, _, err := DecodeQueryResultGroupedFanOut(EncodeQueryResultFused([]vector.Result{{ID: 1}})); err == nil {
		t.Fatal("decoding a plain flat result as grouped fan-out should fail")
	}
}

func assertKeys(t *testing.T, got, want map[uint64]vector.Value) {
	t.Helper()
	if len(got) != len(want) {
		t.Fatalf("key count %d != %d", len(got), len(want))
	}
	for id, v := range want {
		gv, ok := got[id]
		if !ok {
			t.Fatalf("id %d missing from keys", id)
		}
		if gv.Kind != v.Kind || gv.Int != v.Int || gv.Str != v.Str {
			t.Fatalf("id %d key %+v != %+v", id, gv, v)
		}
	}
}
