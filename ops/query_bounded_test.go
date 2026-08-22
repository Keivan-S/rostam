// SPDX-License-Identifier: Apache-2.0

package ops

import (
	"testing"

	"github.com/rostamlabs/rostam/grpcapi/pb"
	"github.com/rostamlabs/rostam/vector"
)

// flatLeavesPB builds n flat dense proto prefetch leaves.
func flatLeavesPB(n int) []*pb.QueryLeaf {
	out := make([]*pb.QueryLeaf, n)
	for i := range out {
		out[i] = denseLeafPB([]float32{1, 0})
	}
	return out
}

// leafSourcesPB builds n dense leaf proto QuerySources (the additive prefetch_sources form).
func leafSourcesPB(n int) []*pb.QuerySource {
	out := make([]*pb.QuerySource, n)
	for i := range out {
		out[i] = &pb.QuerySource{Source: &pb.QuerySource_Leaf{Leaf: denseLeafPB([]float32{1, 0})}}
	}
	return out
}

// TestQuerySourceBreadthBound: the decode-time breadth bound (MaxPrefetchSources)
// rejects a spec node carrying more than MaxPrefetchSources prefetch sources fail-loud
// (ErrTooManyPrefetchSources) at DECODE — the breadth companion to the depth bound. A
// node AT the bound decodes fine; the check fires on BOTH the flat `prefetch` and the
// additive `prefetch_sources` field, and at a NESTED node (recursion).
func TestQuerySourceBreadthBound(t *testing.T) {
	// Flat prefetch AT the bound decodes.
	atBoundFlat := &pb.QuerySpec{
		Mode:     pb.QueryMode_QUERY_MODE_FUSION,
		Prefetch: flatLeavesPB(vector.MaxPrefetchSources),
		K:        3,
	}
	if _, err := QuerySpecFromProto(atBoundFlat, 0); err != nil {
		t.Fatalf("flat spec at the bound (%d) should decode, got: %v", vector.MaxPrefetchSources, err)
	}

	// Flat prefetch ONE PAST the bound: ErrTooManyPrefetchSources, fail-loud.
	tooManyFlat := &pb.QuerySpec{
		Mode:     pb.QueryMode_QUERY_MODE_FUSION,
		Prefetch: flatLeavesPB(vector.MaxPrefetchSources + 1),
		K:        3,
	}
	if _, err := QuerySpecFromProto(tooManyFlat, 0); err != vector.ErrTooManyPrefetchSources {
		t.Fatalf("over-wide flat spec err = %v, want ErrTooManyPrefetchSources", err)
	}

	// Additive prefetch_sources ONE PAST the bound: also rejected.
	tooManySources := &pb.QuerySpec{
		Mode:            pb.QueryMode_QUERY_MODE_FUSION,
		PrefetchSources: leafSourcesPB(vector.MaxPrefetchSources + 1),
		K:               3,
	}
	if _, err := QuerySpecFromProto(tooManySources, 0); err != vector.ErrTooManyPrefetchSources {
		t.Fatalf("over-wide prefetch_sources err = %v, want ErrTooManyPrefetchSources", err)
	}

	// NESTED node over the bound: the root is a valid 1-source spec whose single source
	// is an over-wide sub-spec; the recursive decode (querySourceFromProto →
	// QuerySpecFromProto) must reject the nested node.
	overWideSub := &pb.QuerySpec{
		Mode:     pb.QueryMode_QUERY_MODE_FUSION,
		Prefetch: flatLeavesPB(vector.MaxPrefetchSources + 1),
		K:        3,
	}
	parent := &pb.QuerySpec{
		Mode: pb.QueryMode_QUERY_MODE_FUSION,
		PrefetchSources: []*pb.QuerySource{
			{Source: &pb.QuerySource_Spec{Spec: overWideSub}},
		},
		K: 3,
	}
	if _, err := QuerySpecFromProto(parent, 0); err != vector.ErrTooManyPrefetchSources {
		t.Fatalf("nested over-wide node err = %v, want ErrTooManyPrefetchSources (recursion)", err)
	}
}
