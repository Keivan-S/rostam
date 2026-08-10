// SPDX-License-Identifier: Apache-2.0

package httpapi

import (
	"net/http"
	"testing"

	"github.com/rostamlabs/rostam/ops"
	"github.com/rostamlabs/rostam/vector"
)

// TestHTTPQueryDiscoverIDFormReachesWire: a /query body with a discover leaf in the
// ID form (target + context positive/negative point-ids) decodes on the wire into a
// LeafDiscover carrying the unresolved target/context IDS (the coordinator resolves
// them cluster-wide) with ScoreDesc=true. Proves the HTTP discover parse → proto
// DiscoverLeaf → engine QueryLeaf round-trip.
func TestHTTPQueryDiscoverIDFormReachesWire(t *testing.T) {
	disp := &recordingDispatcher{result: ops.EncodeQueryResultFusedDegraded(nil, false, nil)}
	h := Handler(disp, Options{})

	body := `{"prefetch":[{"discover":{"target":5,"context":[{"positive":1,"negative":9},{"positive":2,"negative":8}]},"k":3}],"k":3}`
	rec := do(t, h, "POST", "/v1/collections/docs/query", body, nil)
	if rec.Code != http.StatusOK {
		t.Fatalf("discover query = %d, want 200 (%s)", rec.Code, rec.Body)
	}
	if len(disp.calls) != 1 || disp.calls[0].name != "vector_query" {
		t.Fatalf("calls = %+v, want one vector_query", disp.calls)
	}
	_, _, spec, _, _, _, err := ops.DecodeQuerySpecArgs(disp.calls[0].args)
	if err != nil {
		t.Fatalf("decode query spec: %v", err)
	}
	if len(spec.Prefetch) != 1 {
		t.Fatalf("prefetch lanes = %d, want 1", len(spec.Prefetch))
	}
	l := spec.Prefetch[0].Leaf
	if l.Kind != vector.LeafDiscover {
		t.Fatalf("leaf kind = %d, want LeafDiscover", l.Kind)
	}
	if !l.ScoreDesc {
		t.Fatal("discover leaf ScoreDesc = false, want true (score-descending lane)")
	}
	if len(l.DiscoverTargetID) != 1 || l.DiscoverTargetID[0] != 5 {
		t.Fatalf("target id = %v, want [5]", l.DiscoverTargetID)
	}
	want := []vector.ContextPair{{Positive: 1, Negative: 9}, {Positive: 2, Negative: 8}}
	if len(l.DiscoverContextIDs) != len(want) {
		t.Fatalf("context ids = %v, want %v", l.DiscoverContextIDs, want)
	}
	for i, cp := range want {
		if l.DiscoverContextIDs[i] != cp {
			t.Fatalf("context id[%d] = %v, want %v", i, l.DiscoverContextIDs[i], cp)
		}
	}
}

// TestHTTPQueryDiscoverTargetIDZero: point id 0 is a legal anchor. The request
// DTO used a plain uint64 for "target", so an explicit "target":0 was
// indistinguishable from an omitted field and the anchor was silently dropped —
// the query still succeeded, but answered from the context positives instead of
// the requested point. Distinguishing the two cases is the whole fix, so both
// are asserted here.
func TestHTTPQueryDiscoverTargetIDZero(t *testing.T) {
	decode := func(t *testing.T, body string) *vector.QueryLeaf {
		t.Helper()
		disp := &recordingDispatcher{result: ops.EncodeQueryResultFusedDegraded(nil, false, nil)}
		h := Handler(disp, Options{})
		rec := do(t, h, "POST", "/v1/collections/docs/query", body, nil)
		if rec.Code != http.StatusOK {
			t.Fatalf("discover query = %d, want 200 (%s)", rec.Code, rec.Body)
		}
		_, _, spec, _, _, _, err := ops.DecodeQuerySpecArgs(disp.calls[0].args)
		if err != nil {
			t.Fatalf("decode query spec: %v", err)
		}
		return spec.Prefetch[0].Leaf
	}

	// "target":0 is an ANCHOR on point id 0, not an absent anchor.
	l := decode(t, `{"prefetch":[{"discover":{"target":0,"context":[{"positive":1,"negative":9}]},"k":3}],"k":3}`)
	if len(l.DiscoverTargetID) != 1 || l.DiscoverTargetID[0] != 0 {
		t.Errorf(`"target":0 -> DiscoverTargetID = %v, want [0] (anchor on point id 0 was dropped)`, l.DiscoverTargetID)
	}

	// A NON-ZERO target still works, unchanged.
	l = decode(t, `{"prefetch":[{"discover":{"target":7,"context":[{"positive":1,"negative":9}]},"k":3}],"k":3}`)
	if len(l.DiscoverTargetID) != 1 || l.DiscoverTargetID[0] != 7 {
		t.Errorf(`"target":7 -> DiscoverTargetID = %v, want [7]`, l.DiscoverTargetID)
	}

	// An OMITTED target means "no anchor" — seed from the context positives.
	l = decode(t, `{"prefetch":[{"discover":{"context":[{"positive":1,"negative":9}]},"k":3}],"k":3}`)
	if len(l.DiscoverTargetID) != 0 {
		t.Errorf("omitted target -> DiscoverTargetID = %v, want empty", l.DiscoverTargetID)
	}

	// An explicit JSON null is absent too — a *uint64 decodes null to nil, which
	// must not be confused with id 0.
	l = decode(t, `{"prefetch":[{"discover":{"target":null,"context":[{"positive":1,"negative":9}]},"k":3}],"k":3}`)
	if len(l.DiscoverTargetID) != 0 {
		t.Errorf("null target -> DiscoverTargetID = %v, want empty", l.DiscoverTargetID)
	}

	// A context pair anchored on id 0 must survive too.
	l = decode(t, `{"prefetch":[{"discover":{"target":5,"context":[{"positive":0,"negative":9}]},"k":3}],"k":3}`)
	want := vector.ContextPair{Positive: 0, Negative: 9}
	if len(l.DiscoverContextIDs) != 1 || l.DiscoverContextIDs[0] != want {
		t.Errorf("context ids = %v, want [%v]", l.DiscoverContextIDs, want)
	}
}

// TestHTTPQueryDiscoverPairFormRejected: a context pair must be WHOLLY the id
// form or WHOLLY the vector form. The dispatch was `both vecs present ? vector :
// id`, so a half-specified pair fell through to the id form and read its missing
// side as point id 0. That was invisible while id 0 was unsearchable; now that
// id 0 is a real point, guessing would silently anchor discover on someone's
// actual data. Reject at the edge instead.
func TestHTTPQueryDiscoverPairFormRejected(t *testing.T) {
	post := func(t *testing.T, body string) int {
		t.Helper()
		disp := &recordingDispatcher{result: ops.EncodeQueryResultFusedDegraded(nil, false, nil)}
		h := Handler(disp, Options{})
		return do(t, h, "POST", "/v1/collections/docs/query", body, nil).Code
	}

	for _, tc := range []struct {
		name string
		pair string
		want int
	}{
		// Malformed: each mixes the two forms or omits half of one.
		{"vec positive, id negative", `{"positive_vec":[1,2],"negative":5}`, http.StatusBadRequest},
		{"id positive, vec negative", `{"positive":5,"negative_vec":[1,2]}`, http.StatusBadRequest},
		{"only positive id", `{"positive":5}`, http.StatusBadRequest},
		{"only negative id", `{"negative":5}`, http.StatusBadRequest},
		{"only positive vec", `{"positive_vec":[1,2]}`, http.StatusBadRequest},
		{"empty pair", `{}`, http.StatusBadRequest},
		{"all four given", `{"positive":1,"negative":2,"positive_vec":[1,2],"negative_vec":[3,4]}`, http.StatusBadRequest},
		{"nulled ids with vecs", `{"positive":null,"negative":null,"positive_vec":[1,2],"negative_vec":[3,4]}`, http.StatusOK},
		// Well-formed: both complete shapes, including id 0 on either side.
		{"both ids", `{"positive":1,"negative":9}`, http.StatusOK},
		{"both ids, positive 0", `{"positive":0,"negative":9}`, http.StatusOK},
		{"both ids, negative 0", `{"positive":1,"negative":0}`, http.StatusOK},
		{"both vecs", `{"positive_vec":[1,2],"negative_vec":[3,4]}`, http.StatusOK},
	} {
		t.Run(tc.name, func(t *testing.T) {
			body := `{"prefetch":[{"discover":{"context":[` + tc.pair + `]},"k":3}],"k":3}`
			if got := post(t, body); got != tc.want {
				t.Errorf("pair %s -> %d, want %d", tc.pair, got, tc.want)
			}
		})
	}
}

// TestHTTPQueryDiscoverVecFormReachesWire: a /query body with a discover leaf in the
// VEC form (target_vec + per-pair positive_vec/negative_vec) decodes into a
// LeafDiscover carrying the already-embedded RESOLVED vectors (no ids — the
// coordinator pre-pass leaves it as-is). Proves the raw-vector discover form.
func TestHTTPQueryDiscoverVecFormReachesWire(t *testing.T) {
	disp := &recordingDispatcher{result: ops.EncodeQueryResultFusedDegraded(nil, false, nil)}
	h := Handler(disp, Options{})

	body := `{"prefetch":[{"discover":{"target_vec":[0.7,0.7],"context":[{"positive_vec":[1,0],"negative_vec":[0,1]}]},"k":3}],"k":3}`
	rec := do(t, h, "POST", "/v1/collections/docs/query", body, nil)
	if rec.Code != http.StatusOK {
		t.Fatalf("discover vec query = %d, want 200 (%s)", rec.Code, rec.Body)
	}
	_, _, spec, _, _, _, err := ops.DecodeQuerySpecArgs(disp.calls[0].args)
	if err != nil {
		t.Fatalf("decode query spec: %v", err)
	}
	l := spec.Prefetch[0].Leaf
	if l.Kind != vector.LeafDiscover {
		t.Fatalf("leaf kind = %d, want LeafDiscover", l.Kind)
	}
	if len(l.DiscoverTargetID) != 0 || len(l.DiscoverContextIDs) != 0 {
		t.Fatalf("vec-form leaf carries ids (target=%v ctx=%v), want none", l.DiscoverTargetID, l.DiscoverContextIDs)
	}
	if len(l.DiscoverTarget) != 2 || l.DiscoverTarget[0] != 0.7 {
		t.Fatalf("target vec = %v, want [0.7 0.7]", l.DiscoverTarget)
	}
	if len(l.DiscoverContext) != 1 {
		t.Fatalf("context pairs = %d, want 1", len(l.DiscoverContext))
	}
	p := l.DiscoverContext[0]
	if len(p.Pos) != 2 || p.Pos[0] != 1 || len(p.Neg) != 2 || p.Neg[1] != 1 {
		t.Fatalf("context pair = {Pos:%v Neg:%v}, want {Pos:[1 0] Neg:[0 1]}", p.Pos, p.Neg)
	}
}
