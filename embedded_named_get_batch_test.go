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

// seedNamedBatchCollection creates a P-partition (or P==1) named collection named
// coll with two spaces ("title","image") and inserts ids 1..n, each with
// title={float32(id),0,0,0}, image={0,float32(id),0,0} and a shared payload
// {"id": id}. Returns the Store. The named clone of seedBatchCollection.
func seedNamedBatchCollection(t *testing.T, coll string, P, n int) Store {
	t.Helper()
	s := newSingleEmbedded(t)
	waitLeaderEmbedded(t, s)
	ctx := context.Background()
	cfg := map[string]NamedVectorParams{
		"title": {Dim: 4, Metric: vector.L2},
		"image": {Dim: 4, Metric: vector.L2},
	}
	if err := s.VectorNamedCreateCollection(ctx, coll, cfg, P); err != nil {
		t.Fatalf("VectorNamedCreateCollection %q (P=%d): %v", coll, P, err)
	}
	for id := uint64(1); id <= uint64(n); id++ {
		vecs := map[string][]float32{
			"title": {float32(id), 0, 0, 0},
			"image": {0, float32(id), 0, 0},
		}
		meta := VectorMetadata{"id": vector.NewInt(int64(id))}
		if err := s.VectorNamedInsert(ctx, coll, id, vecs, meta, 0); err != nil {
			t.Fatalf("named insert %d: %v", id, err)
		}
	}
	return s
}

func namedPointIDs(points []NamedBatchGetPoint) []uint64 {
	out := make([]uint64, len(points))
	for i, p := range points {
		out[i] = p.ID
	}
	return out
}

// TestNamedGetBatchScatterSpansAllPartitions: a named batch over a P>1 collection
// with ids spanning ALL partitions returns every present point (correct
// vectors-map/meta) and EXACTLY the absent ids in missing, vs an independent
// single-get ground truth. The named clone of the dense scatter test.
func TestNamedGetBatchScatterSpansAllPartitions(t *testing.T) {
	const P, n = 4, 200
	s := seedNamedBatchCollection(t, "named", P, n)
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

	points, missing, err := s.VectorNamedGetBatch(ctx, "named", ids, true, true)
	if err != nil {
		t.Fatalf("VectorNamedGetBatch: %v", err)
	}

	// Ground truth via independent single named gets.
	var gtPoints []NamedBatchGetPoint
	var gtMissing []uint64
	for id := uint64(1); id <= uint64(n+20); id++ {
		found, vecs, meta, ttl, gerr := s.VectorNamedGet(ctx, "named", id, true, true)
		if gerr != nil {
			t.Fatalf("VectorNamedGet %d: %v", id, gerr)
		}
		if found {
			// Each point is inserted exactly once, so its version is 1
			// (VectorNamedGet's signature doesn't return version, so set it here).
			gtPoints = append(gtPoints, NamedBatchGetPoint{ID: id, Vectors: vecs, Meta: meta, TTL: ttl, Version: 1})
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

// TestNamedGetBatchP1FastPath: P==1 collection — one call, all ids returned.
func TestNamedGetBatchP1FastPath(t *testing.T) {
	const n = 50
	s := seedNamedBatchCollection(t, "named1", 1, n)
	ctx := context.Background()

	ids := []uint64{5, 1, 50, 25, 999} // 999 absent
	points, missing, err := s.VectorNamedGetBatch(ctx, "named1", ids, true, true)
	if err != nil {
		t.Fatalf("VectorNamedGetBatch: %v", err)
	}
	if !reflect.DeepEqual(namedPointIDs(points), []uint64{1, 5, 25, 50}) {
		t.Fatalf("points ids = %v, want sorted [1 5 25 50]", namedPointIDs(points))
	}
	if !reflect.DeepEqual(missing, []uint64{999}) {
		t.Fatalf("missing = %v, want [999]", missing)
	}
	// vectors-map projection sanity on a present point.
	if len(points[0].Vectors["title"]) != 4 || len(points[0].Vectors["image"]) != 4 {
		t.Fatalf("point 1 vectors-map = %+v, want title+image of dim 4", points[0].Vectors)
	}
}

// TestNamedGetBatchDedupAllAbsentEmpty: a duplicated id is fetched once; an
// all-absent batch yields all ids in missing; empty ids yields empty both.
func TestNamedGetBatchDedupAllAbsentEmpty(t *testing.T) {
	const P, n = 4, 100
	s := seedNamedBatchCollection(t, "named", P, n)
	ctx := context.Background()

	// dedup
	points, missing, err := s.VectorNamedGetBatch(ctx, "named", []uint64{7, 7, 7, 42, 42, 7}, false, false)
	if err != nil {
		t.Fatalf("VectorNamedGetBatch dedup: %v", err)
	}
	if len(missing) != 0 || !reflect.DeepEqual(namedPointIDs(points), []uint64{7, 42}) {
		t.Fatalf("dedup: points=%v missing=%v, want [7 42] / none", namedPointIDs(points), missing)
	}

	// all-absent
	absent := []uint64{1000, 1001, 1002, 1003}
	pts, miss, err := s.VectorNamedGetBatch(ctx, "named", absent, true, true)
	if err != nil {
		t.Fatalf("VectorNamedGetBatch all-absent: %v", err)
	}
	want := append([]uint64(nil), absent...)
	sort.Slice(want, func(i, j int) bool { return want[i] < want[j] })
	if len(pts) != 0 || !reflect.DeepEqual(miss, want) {
		t.Fatalf("all-absent: points=%v missing=%v, want none / %v", pts, miss, want)
	}

	// empty
	pts, miss, err = s.VectorNamedGetBatch(ctx, "named", nil, true, true)
	if err != nil {
		t.Fatalf("VectorNamedGetBatch empty: %v", err)
	}
	if len(pts) != 0 || len(miss) != 0 {
		t.Fatalf("empty ids => points=%v missing=%v, want both empty", pts, miss)
	}
}
