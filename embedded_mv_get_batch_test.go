// SPDX-License-Identifier: Apache-2.0

package rostam

import (
	"context"
	"reflect"
	"sort"
	"testing"

	"github.com/rostamlabs/rostam/ops"
	"github.com/rostamlabs/rostam/vector"
)

// seedMVBatchCollection creates an unpartitioned (P==1) multi-vector collection
// named coll and adds docs 1..n, each with a 2-token matrix
// {{float32(id),0,0,0},{0,float32(id),0,0}} and a shared payload {"id": id}.
// Returns the Store. The MV clone of seedNamedBatchCollection (no ttl, single
// partition — multi-partition scatter correctness is covered elsewhere).
func seedMVBatchCollection(t *testing.T, coll string, n int) Store {
	t.Helper()
	s := newSingleEmbedded(t)
	waitLeaderEmbedded(t, s)
	ctx := context.Background()
	if err := s.VectorMVCreateCollection(ctx, coll, MultiVectorConfig{Dim: 4}); err != nil {
		t.Fatalf("VectorMVCreateCollection %q: %v", coll, err)
	}
	for id := uint64(1); id <= uint64(n); id++ {
		tokens := [][]float32{
			{float32(id), 0, 0, 0},
			{0, float32(id), 0, 0},
		}
		meta := VectorMetadata{"id": vector.NewInt(int64(id))}
		if err := s.VectorMVAdd(ctx, coll, id, tokens, meta); err != nil {
			t.Fatalf("mv add %d: %v", id, err)
		}
	}
	return s
}

func mvPointIDs(points []MVBatchGetPoint) []uint64 {
	out := make([]uint64, len(points))
	for i, p := range points {
		out[i] = p.ID
	}
	return out
}

// seedMVBatchCollectionP creates a P-partition multi-vector collection and adds
// docs 1..n (2-token matrix + shared payload). The partitioned variant of
// seedMVBatchCollection, for the scatter-across-partitions test.
func seedMVBatchCollectionP(t *testing.T, coll string, P, n int) Store {
	t.Helper()
	s := newSingleEmbedded(t)
	waitLeaderEmbedded(t, s)
	ctx := context.Background()
	if err := s.VectorMVCreateCollection(ctx, coll, MultiVectorConfig{Dim: 4, Partitions: P}); err != nil {
		t.Fatalf("VectorMVCreateCollection %q (P=%d): %v", coll, P, err)
	}
	for id := uint64(1); id <= uint64(n); id++ {
		tokens := [][]float32{{float32(id), 0, 0, 0}, {0, float32(id), 0, 0}}
		meta := VectorMetadata{"id": vector.NewInt(int64(id))}
		if err := s.VectorMVAdd(ctx, coll, id, tokens, meta); err != nil {
			t.Fatalf("mv add %d: %v", id, err)
		}
	}
	return s
}

// TestMVGetBatchScatterSpansAllPartitions: an MV batch over a P>1 collection with
// ids spanning ALL partitions returns every present doc (correct token matrix +
// meta) and EXACTLY the absent ids in missing, vs an independent single-get
// ground truth. The MV clone of TestNamedGetBatchScatterSpansAllPartitions.
func TestMVGetBatchScatterSpansAllPartitions(t *testing.T) {
	const P, n = 4, 200
	s := seedMVBatchCollectionP(t, "mv", P, n)
	ctx := context.Background()

	var ids []uint64
	for id := uint64(1); id <= uint64(n); id++ {
		ids = append(ids, id)
	}
	var wantMissing []uint64
	for id := uint64(n + 1); id <= uint64(n+20); id++ {
		ids = append(ids, id)
		wantMissing = append(wantMissing, id)
	}
	touched := map[int]bool{}
	for _, id := range ids {
		touched[ops.PartitionOf(id, P)] = true
	}
	if len(touched) != P {
		t.Fatalf("test ids only touch %d/%d partitions", len(touched), P)
	}

	points, missing, err := s.VectorMVGetBatch(ctx, "mv", ids, true, true)
	if err != nil {
		t.Fatalf("VectorMVGetBatch: %v", err)
	}

	// Ground truth via independent single MV gets.
	var gtPoints []MVBatchGetPoint
	var gtMissing []uint64
	for id := uint64(1); id <= uint64(n+20); id++ {
		found, tokens, meta, gerr := s.VectorMVGet(ctx, "mv", id, true, true)
		if gerr != nil {
			t.Fatalf("VectorMVGet %d: %v", id, gerr)
		}
		if found {
			// Each doc is added exactly once, so its version is 1
			// (VectorMVGet's signature doesn't return version, so set it here).
			gtPoints = append(gtPoints, MVBatchGetPoint{ID: id, Tokens: tokens, Meta: meta, Version: 1})
		} else {
			gtMissing = append(gtMissing, id)
		}
	}

	if !reflect.DeepEqual(missing, wantMissing) {
		t.Fatalf("missing = %v, want %v", missing, wantMissing)
	}
	if !reflect.DeepEqual(missing, gtMissing) {
		t.Fatalf("missing != ground truth: %v vs %v", missing, gtMissing)
	}
	if len(points) != len(gtPoints) {
		t.Fatalf("got %d points, ground truth %d", len(points), len(gtPoints))
	}
	for i := range points {
		if !reflect.DeepEqual(points[i], gtPoints[i]) {
			t.Fatalf("point %d mismatch:\n got %+v\nwant %+v", i, points[i], gtPoints[i])
		}
	}
}

// TestMVGetBatchP1FastPath exercises the unpartitioned (P<=1) single-call fast
// path of embedded.VectorMVGetBatch: a mixed present/absent batch returns the
// present points (id-sorted, token matrix + payload) and exactly the absent ids
// in missing. The MV clone of TestNamedGetBatchP1FastPath (no ttl). Multi-
// partition scatter correctness is covered elsewhere.
func TestMVGetBatchP1FastPath(t *testing.T) {
	const n = 50
	s := seedMVBatchCollection(t, "mv1", n)
	ctx := context.Background()

	ids := []uint64{5, 1, 50, 25, 999} // 999 absent
	points, missing, err := s.VectorMVGetBatch(ctx, "mv1", ids, true, true)
	if err != nil {
		t.Fatalf("VectorMVGetBatch: %v", err)
	}
	if !reflect.DeepEqual(mvPointIDs(points), []uint64{1, 5, 25, 50}) {
		t.Fatalf("points ids = %v, want sorted [1 5 25 50]", mvPointIDs(points))
	}
	if !reflect.DeepEqual(missing, []uint64{999}) {
		t.Fatalf("missing = %v, want [999]", missing)
	}
	// token-matrix projection sanity on a present point (id 1 → tokens of dim 4).
	if len(points[0].Tokens) != 2 || len(points[0].Tokens[0]) != 4 {
		t.Fatalf("point 1 tokens = %+v, want 2x4", points[0].Tokens)
	}
	if points[0].Meta["id"].Int != 1 {
		t.Fatalf("point 1 meta = %+v, want id=1", points[0].Meta)
	}

	// dedup + all-absent + empty (single-call path).
	pts, miss, err := s.VectorMVGetBatch(ctx, "mv1", []uint64{7, 7, 7, 42, 42, 7}, false, false)
	if err != nil {
		t.Fatalf("VectorMVGetBatch dedup: %v", err)
	}
	if len(miss) != 0 || !reflect.DeepEqual(mvPointIDs(pts), []uint64{7, 42}) {
		t.Fatalf("dedup: points=%v missing=%v, want [7 42] / none", mvPointIDs(pts), miss)
	}

	absent := []uint64{1000, 1001, 1002}
	pts, miss, err = s.VectorMVGetBatch(ctx, "mv1", absent, true, true)
	if err != nil {
		t.Fatalf("VectorMVGetBatch all-absent: %v", err)
	}
	want := append([]uint64(nil), absent...)
	sort.Slice(want, func(i, j int) bool { return want[i] < want[j] })
	if len(pts) != 0 || !reflect.DeepEqual(miss, want) {
		t.Fatalf("all-absent: points=%v missing=%v, want none / %v", pts, miss, want)
	}

	pts, miss, err = s.VectorMVGetBatch(ctx, "mv1", nil, true, true)
	if err != nil {
		t.Fatalf("VectorMVGetBatch empty: %v", err)
	}
	if len(pts) != 0 || len(miss) != 0 {
		t.Fatalf("empty ids => points=%v missing=%v, want both empty", pts, miss)
	}
}
