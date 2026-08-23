// SPDX-License-Identifier: Apache-2.0

package grpcapi

import (
	"context"
	"testing"

	"github.com/rostamlabs/rostam/ops"
	"github.com/rostamlabs/rostam/sdk/pb"
	"github.com/rostamlabs/rostam/vector"
)

// TestVectorQueryGroupedRoundTrip proves the gRPC VectorQuery RPC, when the spec
// carries group_by, marshals the group fields into the dispatched QuerySpec and
// decodes the coordinator's GROUPED result into SearchResponse.groups (NOT the flat
// results field) — mirroring SearchGroups. The dispatcher is canned with
// EncodeGroupsDegraded, matching how the flat VectorQuery test cans the fused body.
func TestVectorQueryGroupedRoundTrip(t *testing.T) {
	want := []vector.Group{
		{Key: vector.NewInt(1), Hits: []vector.Document{{ID: 1}, {ID: 2}}},
		{Key: vector.NewInt(2), Hits: []vector.Document{{ID: 3}}},
	}
	disp := &countingDispatcher{body: ops.EncodeGroupsDegraded(want, true, []uint16{2})}
	s := NewServer(disp, nil)
	resp, err := s.VectorQuery(context.Background(), &pb.VectorQueryRequest{
		Collection: "docs",
		Spec: &pb.QuerySpec{
			Mode:      pb.QueryMode_QUERY_MODE_FUSION,
			K:         2,
			GroupBy:   "doc",
			GroupSize: 2,
			Prefetch: []*pb.QueryLeaf{
				{Leaf: &pb.QueryLeaf_Dense{Dense: &pb.DenseLeaf{Dense: []float32{1, 2}, K: 50}}},
			},
		},
	})
	if err != nil {
		t.Fatalf("VectorQuery grouped: %v", err)
	}
	if !disp.called || disp.lastOp != "vector_query" {
		t.Fatalf("dispatch op = %q, want vector_query", disp.lastOp)
	}
	_, _, spec, _, _, _, derr := ops.DecodeQuerySpecArgs(disp.lastArg)
	if derr != nil {
		t.Fatalf("DecodeQuerySpecArgs: %v", derr)
	}
	if spec.GroupBy != "doc" || spec.GroupSize != 2 {
		t.Fatalf("dispatched spec group fields = (%q,%d), want (doc,2)", spec.GroupBy, spec.GroupSize)
	}
	if len(resp.GetResults()) != 0 {
		t.Fatalf("grouped response unexpectedly carried flat results: %v", resp.GetResults())
	}
	groups := resp.GetGroups()
	if len(groups) != 2 {
		t.Fatalf("groups = %v, want 2", groups)
	}
	// KeyJson is the JSON-marshaled vector.Value (same shape SearchGroups emits).
	if groups[0].GetKeyJson() != `{"kind":"int","int":1}` || groups[1].GetKeyJson() != `{"kind":"int","int":2}` {
		t.Fatalf("group keys = %q,%q", groups[0].GetKeyJson(), groups[1].GetKeyJson())
	}
	if len(groups[0].GetHits()) != 2 || groups[0].GetHits()[0].GetId() != 1 {
		t.Fatalf("group0 hits = %v, want [1,2]", groups[0].GetHits())
	}
	if !resp.GetDegraded() || len(resp.GetMissing()) != 1 || resp.GetMissing()[0] != 2 {
		t.Fatalf("degraded/missing = %v/%v, want true/[2]", resp.GetDegraded(), resp.GetMissing())
	}
}
