// SPDX-License-Identifier: Apache-2.0

package httpapi

import (
	"net/http"
	"testing"

	"github.com/rostamlabs/rostam/ops"
	"github.com/rostamlabs/rostam/vector"
)

// TestHTTPHybridDBSFReachesWire: "dbsf" on the dense hybrid edge maps to
// vector.FusionDBSF in the wire args.
func TestHTTPHybridDBSFReachesWire(t *testing.T) {
	disp := &recordingDispatcher{result: ops.EncodeHybridResults(nil)}
	h := Handler(disp, Options{})

	rec := do(t, h, "POST", "/v1/collections/docs/points/search/hybrid",
		`{"dense":[1,2],"k":1,"method":"dbsf","alpha":0.3}`, nil)
	if rec.Code != http.StatusOK {
		t.Fatalf("hybrid dbsf = %d, want 200 (%s)", rec.Code, rec.Body)
	}
	if len(disp.calls) != 1 || disp.calls[0].name != "vector_hybrid_search" {
		t.Fatalf("calls = %+v, want one vector_hybrid_search", disp.calls)
	}
	_, _, _, _, opts, _, _, _, err := ops.DecodeHybridSearchArgsOpts(disp.calls[0].args)
	if err != nil {
		t.Fatalf("decode args: %v", err)
	}
	if opts.Method != vector.FusionDBSF {
		t.Fatalf("decoded method = %v, want FusionDBSF", opts.Method)
	}
}

// TestHTTPNamedHybridDBSFReachesWire: "dbsf" on the named hybrid edge maps to
// vector.FusionDBSF in the wire args.
func TestHTTPNamedHybridDBSFReachesWire(t *testing.T) {
	disp := &recordingDispatcher{result: ops.EncodeHybridResults(nil)}
	h := Handler(disp, Options{})

	rec := do(t, h, "POST", "/v1/named/coll/hybrid-search",
		`{"dense_space":"title","dense":[1,2],"sparse_space":"terms","k":1,"method":"dbsf","alpha":0.3}`, nil)
	if rec.Code != http.StatusOK {
		t.Fatalf("named hybrid dbsf = %d, want 200 (%s)", rec.Code, rec.Body)
	}
	if len(disp.calls) != 1 || disp.calls[0].name != "vector_named_hybrid_search" {
		t.Fatalf("calls = %+v, want one vector_named_hybrid_search", disp.calls)
	}
	_, _, _, _, _, _, opts, _, _, _, err := ops.DecodeNamedHybridArgs(disp.calls[0].args)
	if err != nil {
		t.Fatalf("decode args: %v", err)
	}
	if opts.Method != vector.FusionDBSF {
		t.Fatalf("decoded named method = %v, want FusionDBSF", opts.Method)
	}
}

// TestHTTPMVHybridDBSFReachesWire: "dbsf" on the MV hybrid edge maps to
// vector.FusionDBSF in the wire args.
func TestHTTPMVHybridDBSFReachesWire(t *testing.T) {
	disp := &recordingDispatcher{result: ops.EncodeMVResults(nil)}
	h := Handler(disp, Options{})

	rec := do(t, h, "POST", "/v1/multivector/mv/hybrid-search",
		`{"query":[[1,2]],"k":1,"method":"dbsf","alpha":0.3}`, nil)
	if rec.Code != http.StatusOK {
		t.Fatalf("mv hybrid dbsf = %d, want 200 (%s)", rec.Code, rec.Body)
	}
	if len(disp.calls) != 1 || disp.calls[0].name != "vector_mv_hybrid_search" {
		t.Fatalf("calls = %+v, want one vector_mv_hybrid_search", disp.calls)
	}
	_, _, _, _, opts, _, _, _, err := ops.DecodeMVHybridArgs(disp.calls[0].args)
	if err != nil {
		t.Fatalf("decode args: %v", err)
	}
	if opts.Method != vector.FusionDBSF {
		t.Fatalf("decoded mv method = %v, want FusionDBSF", opts.Method)
	}
}

// TestHTTPHybridUnknownMethodFailsLoud: an unknown fusion method is rejected with
// a 400 on all three hybrid edges and never reaches the dispatcher.
func TestHTTPHybridUnknownMethodFailsLoud(t *testing.T) {
	cases := []struct {
		name, path, body string
	}{
		{"dense", "/v1/collections/docs/points/search/hybrid", `{"dense":[1,2],"k":1,"method":"bogus"}`},
		{"named", "/v1/named/coll/hybrid-search", `{"dense_space":"title","dense":[1,2],"sparse_space":"terms","k":1,"method":"bogus"}`},
		{"mv", "/v1/multivector/mv/hybrid-search", `{"query":[[1,2]],"k":1,"method":"bogus"}`},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			disp := &recordingDispatcher{}
			h := Handler(disp, Options{})
			rec := do(t, h, "POST", tc.path, tc.body, nil)
			if rec.Code != http.StatusBadRequest {
				t.Fatalf("unknown method rc = %d, want 400 (%s)", rec.Code, rec.Body)
			}
			if len(disp.calls) != 0 {
				t.Fatalf("dispatcher called %d times on unknown method, want 0", len(disp.calls))
			}
		})
	}
}
