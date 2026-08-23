// SPDX-License-Identifier: Apache-2.0

package grpcapi

import (
	"context"
	"testing"

	"github.com/rostamlabs/rostam/sdk/pb"
	"github.com/rostamlabs/rostam/ops"
)

// TestTextSearchThreadsGlobalIDFIntoArgs proves GetGlobalIdf() rides into the
// dispatched vector_search_text args (the request flag textFlagGlobalIDF) so the
// fan-out coordinator runs the two-phase global-DF (dfs) path. Default false ⇒
// the flag is absent (per-shard-local fast path).
func TestTextSearchThreadsGlobalIDFIntoArgs(t *testing.T) {
	for _, want := range []bool{false, true} {
		disp := &stubDispatcher{}
		s := NewServer(disp, nil)
		_, _ = s.TextSearch(context.Background(), &pb.TextSearchRequest{
			Collection: "c", Text: "quick fox", K: 5, GlobalIdf: want,
		})
		if disp.lastOp != "vector_search_text" {
			t.Fatalf("op = %q, want vector_search_text", disp.lastOp)
		}
		_, _, _, _, _, _, _, gotIDF, g, err := ops.DecodeSearchTextArgsGlobal(disp.lastArg)
		if err != nil {
			t.Fatalf("decode dispatched args: %v", err)
		}
		if gotIDF != want {
			t.Fatalf("dispatched globalIDF = %v, want %v", gotIDF, want)
		}
		// The edge sets ONLY the request flag; it never injects phase-1 stats.
		if g != nil {
			t.Fatalf("dispatched phase-1 stats = %v, want nil (edge sets only the request flag)", g)
		}
	}
}

// TestHybridTextSearchThreadsGlobalIDFIntoArgs is the hybrid-text mirror: the flag
// rides into the dispatched vector_hybrid_text args.
func TestHybridTextSearchThreadsGlobalIDFIntoArgs(t *testing.T) {
	for _, want := range []bool{false, true} {
		disp := &stubDispatcher{}
		s := NewServer(disp, nil)
		_, _ = s.HybridTextSearch(context.Background(), &pb.HybridTextRequest{
			Collection: "c", Dense: []float32{1, 0}, Text: "quick fox", K: 5, Method: "rrf", GlobalIdf: want,
		})
		if disp.lastOp != "vector_hybrid_text" {
			t.Fatalf("op = %q, want vector_hybrid_text", disp.lastOp)
		}
		_, _, _, _, _, _, _, _, gotIDF, g, err := ops.DecodeHybridTextArgsGlobal(disp.lastArg)
		if err != nil {
			t.Fatalf("decode dispatched args: %v", err)
		}
		if gotIDF != want {
			t.Fatalf("dispatched globalIDF = %v, want %v", gotIDF, want)
		}
		if g != nil {
			t.Fatalf("dispatched phase-1 stats = %v, want nil (edge sets only the request flag)", g)
		}
	}
}
