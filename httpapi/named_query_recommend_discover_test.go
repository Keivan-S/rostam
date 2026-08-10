// SPDX-License-Identifier: Apache-2.0

package httpapi

import (
	"net/http"
	"testing"

	"github.com/rostamlabs/rostam/ops"
	"github.com/rostamlabs/rostam/vector"
)

// TestHTTPNamedQueryRecommendSpaceReachesWire: a /named/{name}/query body with a
// recommend leaf carrying a "space" decodes on the wire into a LeafRecommend whose
// Space survives end-to-end (the per-space coordinator pre-pass needs it). Proves the
// HTTP named recommend parse → proto RecommendLeaf{space} → engine QueryLeaf round-trip.
func TestHTTPNamedQueryRecommendSpaceReachesWire(t *testing.T) {
	disp := &recordingDispatcher{result: ops.EncodeQueryResultFusedDegraded(nil, false, nil)}
	h := Handler(disp, Options{})

	body := `{"prefetch":[{"space":"title","recommend":{"positive":[1,2],"negative":[9]},"k":3}],"k":3}`
	rec := do(t, h, "POST", "/v1/named/docs/query", body, nil)
	if rec.Code != http.StatusOK {
		t.Fatalf("named recommend query = %d, want 200 (%s)", rec.Code, rec.Body)
	}
	if len(disp.calls) != 1 || disp.calls[0].name != "vector_named_query" {
		t.Fatalf("calls = %+v, want one vector_named_query", disp.calls)
	}
	_, _, spec, _, _, _, err := ops.DecodeQuerySpecArgs(disp.calls[0].args)
	if err != nil {
		t.Fatalf("decode query spec: %v", err)
	}
	if len(spec.Prefetch) != 1 {
		t.Fatalf("prefetch lanes = %d, want 1", len(spec.Prefetch))
	}
	l := spec.Prefetch[0].Leaf
	if l.Kind != vector.LeafRecommend {
		t.Fatalf("leaf kind = %d, want LeafRecommend", l.Kind)
	}
	if l.Space != "title" {
		t.Fatalf("recommend leaf space = %q, want %q (space dropped end-to-end)", l.Space, "title")
	}
	if len(l.Positive) != 2 || l.Positive[0] != 1 || len(l.Negative) != 1 || l.Negative[0] != 9 {
		t.Fatalf("recommend example ids = pos%v neg%v, want pos[1 2] neg[9]", l.Positive, l.Negative)
	}
	if l.Strategy != vector.RecommendAverageVector {
		t.Fatalf("strategy = %d, want AVERAGE_VECTOR (default)", l.Strategy)
	}
}

// TestHTTPNamedQueryRecommendBestSpaceReachesWire: a named recommend leaf with
// strategy "best_score" + a space decodes into a LeafRecommend (BEST_SCORE,
// ScoreDesc=true) carrying the Space.
func TestHTTPNamedQueryRecommendBestSpaceReachesWire(t *testing.T) {
	disp := &recordingDispatcher{result: ops.EncodeQueryResultFusedDegraded(nil, false, nil)}
	h := Handler(disp, Options{})

	body := `{"prefetch":[{"space":"image","recommend":{"positive":[3,4],"strategy":"best_score"},"k":3}],"k":3}`
	rec := do(t, h, "POST", "/v1/named/docs/query", body, nil)
	if rec.Code != http.StatusOK {
		t.Fatalf("named best_score recommend query = %d, want 200 (%s)", rec.Code, rec.Body)
	}
	_, _, spec, _, _, _, err := ops.DecodeQuerySpecArgs(disp.calls[0].args)
	if err != nil {
		t.Fatalf("decode query spec: %v", err)
	}
	l := spec.Prefetch[0].Leaf
	if l.Kind != vector.LeafRecommend || l.Space != "image" {
		t.Fatalf("leaf = {kind:%d space:%q}, want {LeafRecommend image}", l.Kind, l.Space)
	}
	if l.Strategy != vector.RecommendBestScore || !l.ScoreDesc {
		t.Fatalf("best_score leaf = {strategy:%d scoreDesc:%v}, want {BEST_SCORE true}", l.Strategy, l.ScoreDesc)
	}
}

// TestHTTPNamedQueryDiscoverSpaceReachesWire: a /named/{name}/query body with a
// discover leaf (ID form) + a space decodes into a LeafDiscover (ScoreDesc=true)
// carrying the unresolved target/context ids AND the Space (end-to-end).
func TestHTTPNamedQueryDiscoverSpaceReachesWire(t *testing.T) {
	disp := &recordingDispatcher{result: ops.EncodeQueryResultFusedDegraded(nil, false, nil)}
	h := Handler(disp, Options{})

	body := `{"prefetch":[{"space":"title","discover":{"target":5,"context":[{"positive":1,"negative":9}]},"k":3}],"k":3}`
	rec := do(t, h, "POST", "/v1/named/docs/query", body, nil)
	if rec.Code != http.StatusOK {
		t.Fatalf("named discover query = %d, want 200 (%s)", rec.Code, rec.Body)
	}
	_, _, spec, _, _, _, err := ops.DecodeQuerySpecArgs(disp.calls[0].args)
	if err != nil {
		t.Fatalf("decode query spec: %v", err)
	}
	l := spec.Prefetch[0].Leaf
	if l.Kind != vector.LeafDiscover {
		t.Fatalf("leaf kind = %d, want LeafDiscover", l.Kind)
	}
	if l.Space != "title" {
		t.Fatalf("discover leaf space = %q, want %q (space dropped end-to-end)", l.Space, "title")
	}
	if !l.ScoreDesc {
		t.Fatal("discover leaf ScoreDesc = false, want true")
	}
	if len(l.DiscoverTargetID) != 1 || l.DiscoverTargetID[0] != 5 {
		t.Fatalf("target id = %v, want [5]", l.DiscoverTargetID)
	}
	if len(l.DiscoverContextIDs) != 1 || l.DiscoverContextIDs[0] != (vector.ContextPair{Positive: 1, Negative: 9}) {
		t.Fatalf("context ids = %v, want [{1 9}]", l.DiscoverContextIDs)
	}
}
