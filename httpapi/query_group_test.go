// SPDX-License-Identifier: Apache-2.0

package httpapi

import (
	"net/http"
	"testing"

	"github.com/rostamlabs/rostam/ops"
	"github.com/rostamlabs/rostam/vector"
)

// TestHTTPQueryGroupedRoundTrip proves the /query endpoint, when group_by is set:
// (1) marshals the group fields into the dispatched QuerySpec (group_by/group_size
// survive the wire), and (2) decodes the coordinator's GROUPED result into the
// standalone /groups response shape {groups:[{key,hits:[]}]} (NOT the flat {results}
// shape). The dispatcher is canned with EncodeGroupsDegraded — mirroring how the flat
// /query round-trip test cans EncodeQueryResultFusedDegraded — since the real
// coordinator grouping (P>1==P1) is covered by the root-package oracle tests.
func TestHTTPQueryGroupedRoundTrip(t *testing.T) {
	want := []vector.Group{
		{Key: vector.NewInt(1), Hits: []vector.Document{{ID: 1}, {ID: 2}}},
		{Key: vector.NewInt(2), Hits: []vector.Document{{ID: 3}, {ID: 4}}},
	}
	disp := &recordingDispatcher{result: ops.EncodeGroupsDegraded(want, true, []uint16{2})}
	h := Handler(disp, Options{})

	var out struct {
		Groups   []vector.Group  `json:"groups"`
		Results  []vector.Result `json:"results"`
		Degraded bool            `json:"degraded"`
		Missing  []int           `json:"missing"`
	}
	rec := do(t, h, "POST", "/v1/collections/docs/query",
		`{"mode":"fusion","k":2,"group_by":"doc","group_size":2,"prefetch":[{"dense":[1,2],"k":50}]}`,
		&out)
	if rec.Code != http.StatusOK {
		t.Fatalf("grouped query = %d, want 200 (%s)", rec.Code, rec.Body)
	}
	if len(disp.calls) != 1 || disp.calls[0].name != "vector_query" {
		t.Fatalf("calls = %+v, want one vector_query", disp.calls)
	}
	// The group fields must survive into the dispatched spec.
	_, _, spec, _, _, _, err := ops.DecodeQuerySpecArgs(disp.calls[0].args)
	if err != nil {
		t.Fatalf("DecodeQuerySpecArgs: %v", err)
	}
	if spec.GroupBy != "doc" || spec.GroupSize != 2 {
		t.Fatalf("dispatched spec group fields = (%q,%d), want (doc,2)", spec.GroupBy, spec.GroupSize)
	}
	// The response is the grouped shape, NOT flat results.
	if out.Results != nil {
		t.Fatalf("grouped response unexpectedly carried flat results: %+v", out.Results)
	}
	if len(out.Groups) != 2 || out.Groups[0].Key.Int != 1 || out.Groups[1].Key.Int != 2 {
		t.Fatalf("groups = %+v, want keys 1,2", out.Groups)
	}
	if len(out.Groups[0].Hits) != 2 || out.Groups[0].Hits[0].ID != 1 || out.Groups[0].Hits[1].ID != 2 {
		t.Fatalf("group0 hits = %+v, want [1,2]", out.Groups[0].Hits)
	}
	if !out.Degraded || len(out.Missing) != 1 || out.Missing[0] != 2 {
		t.Fatalf("degraded/missing = %v/%v, want true/[2]", out.Degraded, out.Missing)
	}
}

// TestHTTPQueryNoGroupByStaysFlat is the no-break companion: a /query body WITHOUT
// group_by decodes the flat {results} shape (the grouped branch never engages), so
// the additive group fields default empty and the flat path is unchanged.
func TestHTTPQueryNoGroupByStaysFlat(t *testing.T) {
	want := []vector.Result{{ID: 7, Distance: 0.1, Score: 0.9}, {ID: 3, Distance: 0.2, Score: 0.5}}
	disp := &recordingDispatcher{result: ops.EncodeQueryResultFusedDegraded(want, false, nil)}
	h := Handler(disp, Options{})

	var out struct {
		Results []vector.Result `json:"results"`
		Groups  []vector.Group  `json:"groups"`
	}
	rec := do(t, h, "POST", "/v1/collections/docs/query",
		`{"mode":"fusion","k":2,"prefetch":[{"dense":[1,2],"k":10}]}`, &out)
	if rec.Code != http.StatusOK {
		t.Fatalf("flat query = %d, want 200 (%s)", rec.Code, rec.Body)
	}
	if out.Groups != nil {
		t.Fatalf("flat query unexpectedly returned groups: %+v", out.Groups)
	}
	if len(out.Results) != 2 || out.Results[0].ID != 7 {
		t.Fatalf("flat results = %+v, want [7,3]", out.Results)
	}
}
