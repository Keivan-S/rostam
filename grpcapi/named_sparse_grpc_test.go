// SPDX-License-Identifier: Apache-2.0

package grpcapi

import (
	"context"
	"testing"

	"google.golang.org/grpc/codes"
	"google.golang.org/grpc/status"

	"github.com/rostamlabs/rostam/sdk/pb"
)

// namedSparseConfigJSON is a mixed dense+sparse named config: dense "title" (dim4)
// + sparse "terms".
const namedSparseConfigJSON = `{"title":{"dim":4},"terms":{"sparse":true}}`

// nsv builds a sparse NamedVectorList (the sparse lane of the per-space upsert map).
func nsv(idx []uint32, val []float32) *pb.NamedVectorList {
	return &pb.NamedVectorList{SparseIndices: idx, SparseValues: val}
}

// TestGRPCNamedSparseLifecycle drives create/upsert(dense+sparse)/sparse-search
// end-to-end over the new NamedSparseSearch RPC against a real engine, and asserts
// the modality-mismatch fail-loud paths return InvalidArgument.
func TestGRPCNamedSparseLifecycle(t *testing.T) {
	s := newRealServer(t)
	ctx := context.Background()

	if _, err := s.NamedCreate(ctx, &pb.NamedCreateRequest{Name: "sp", ConfigJson: namedSparseConfigJSON}); err != nil {
		t.Fatalf("NamedCreate: %v", err)
	}

	// Upsert points: a dense "title" value + a sparse "terms" value per point.
	upserts := []*pb.NamedUpsertRequest{
		{Name: "sp", Id: 1, Vectors: map[string]*pb.NamedVectorList{"title": nvl(1, 0, 0, 0), "terms": nsv([]uint32{0, 2}, []float32{1, 3})}},
		{Name: "sp", Id: 2, Vectors: map[string]*pb.NamedVectorList{"title": nvl(0, 1, 0, 0), "terms": nsv([]uint32{0, 2}, []float32{5, 1})}},
		{Name: "sp", Id: 3, Vectors: map[string]*pb.NamedVectorList{"terms": nsv([]uint32{2}, []float32{2})}},
	}
	for _, u := range upserts {
		if _, err := s.NamedUpsert(ctx, u); err != nil {
			t.Fatalf("NamedUpsert id=%d: %v", u.GetId(), err)
		}
	}

	// Sparse search over "terms" with query {0:1, 2:1}: dot products are
	// 1=1*1+1*3=4, 2=1*5+1*1=6, 3=1*2=2 ⇒ ranked 2,1,3.
	res, err := s.NamedSparseSearch(ctx, &pb.NamedSparseSearchRequest{
		Name: "sp", VectorName: "terms", SparseIndices: []uint32{0, 2}, SparseValues: []float32{1, 1}, K: 3,
	})
	if err != nil {
		t.Fatalf("NamedSparseSearch: %v", err)
	}
	got := res.GetResults()
	if len(got) != 3 || got[0].GetId() != 2 || got[1].GetId() != 1 || got[2].GetId() != 3 {
		t.Fatalf("sparse search ranking = %+v, want ids [2 1 3]", got)
	}
	if got[0].GetScore() != 6 {
		t.Fatalf("top score = %v, want 6", got[0].GetScore())
	}

	// Modality mismatch: a sparse query against the DENSE "title" space ⇒ InvalidArgument.
	if _, err := s.NamedSparseSearch(ctx, &pb.NamedSparseSearchRequest{
		Name: "sp", VectorName: "title", SparseIndices: []uint32{0}, SparseValues: []float32{1}, K: 3,
	}); status.Code(err) != codes.InvalidArgument {
		t.Fatalf("sparse search on dense space: code = %v, want InvalidArgument", status.Code(err))
	}

	// Modality mismatch: a SPARSE value upserted to the DENSE "title" space ⇒ InvalidArgument.
	if _, err := s.NamedUpsert(ctx, &pb.NamedUpsertRequest{
		Name: "sp", Id: 9, Vectors: map[string]*pb.NamedVectorList{"title": nsv([]uint32{0}, []float32{1})},
	}); status.Code(err) != codes.InvalidArgument {
		t.Fatalf("sparse value to dense space: code = %v, want InvalidArgument", status.Code(err))
	}
}
