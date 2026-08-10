// SPDX-License-Identifier: Apache-2.0

package httpapi

import (
	"net/http"
	"testing"

	"github.com/rostamlabs/rostam/ops"
)

// TestHTTPSearchTextThreadsGlobalIDF proves "global_idf": true in the
// /points/search/text body rides into the dispatched vector_search_text args as
// the request flag (textFlagGlobalIDF) so the fan-out coordinator runs the
// two-phase global-DF (dfs) path. Default/absent ⇒ flag off (per-shard-local).
func TestHTTPSearchTextThreadsGlobalIDF(t *testing.T) {
	cases := []struct {
		name string
		body string
		want bool
	}{
		{"absent", `{"text":"quick fox","k":5}`, false},
		{"false", `{"text":"quick fox","k":5,"global_idf":false}`, false},
		{"true", `{"text":"quick fox","k":5,"global_idf":true}`, true},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			disp := &recordingDispatcher{result: ops.EncodeVectorDocsDegraded(nil, false, nil)}
			h := Handler(disp, Options{})
			rec := do(t, h, "POST", "/v1/collections/docs/points/search/text", tc.body, nil)
			if rec.Code != http.StatusOK {
				t.Fatalf("search/text = %d, want 200 (%s)", rec.Code, rec.Body)
			}
			if len(disp.calls) != 1 || disp.calls[0].name != "vector_search_text" {
				t.Fatalf("dispatched %d ops (%v), want 1 vector_search_text", len(disp.calls), disp.calls)
			}
			_, _, _, _, _, _, _, gotIDF, g, err := ops.DecodeSearchTextArgsGlobal(disp.calls[0].args)
			if err != nil {
				t.Fatalf("decode dispatched args: %v", err)
			}
			if gotIDF != tc.want {
				t.Fatalf("dispatched globalIDF = %v, want %v", gotIDF, tc.want)
			}
			if g != nil {
				t.Fatalf("dispatched phase-1 stats = %v, want nil (edge sets only the request flag)", g)
			}
		})
	}
}

// TestHTTPHybridTextThreadsGlobalIDF is the hybrid-text mirror.
func TestHTTPHybridTextThreadsGlobalIDF(t *testing.T) {
	cases := []struct {
		name string
		body string
		want bool
	}{
		{"absent", `{"vector":[1,0],"text":"quick fox","k":5,"method":"rrf"}`, false},
		{"true", `{"vector":[1,0],"text":"quick fox","k":5,"method":"rrf","global_idf":true}`, true},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			disp := &recordingDispatcher{result: ops.EncodeHybridResultsDegraded(nil, false, nil)}
			h := Handler(disp, Options{})
			rec := do(t, h, "POST", "/v1/collections/docs/points/search/hybrid-text", tc.body, nil)
			if rec.Code != http.StatusOK {
				t.Fatalf("search/hybrid-text = %d, want 200 (%s)", rec.Code, rec.Body)
			}
			if len(disp.calls) != 1 || disp.calls[0].name != "vector_hybrid_text" {
				t.Fatalf("dispatched %d ops (%v), want 1 vector_hybrid_text", len(disp.calls), disp.calls)
			}
			_, _, _, _, _, _, _, _, gotIDF, g, err := ops.DecodeHybridTextArgsGlobal(disp.calls[0].args)
			if err != nil {
				t.Fatalf("decode dispatched args: %v", err)
			}
			if gotIDF != tc.want {
				t.Fatalf("dispatched globalIDF = %v, want %v", gotIDF, tc.want)
			}
			if g != nil {
				t.Fatalf("dispatched phase-1 stats = %v, want nil (edge sets only the request flag)", g)
			}
		})
	}
}
