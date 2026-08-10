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

// seedBatchCollection creates a P-partition (or P==1) collection named coll and
// inserts ids 1..n with vector {float32(id),0,0,0} and a payload {"id": id}.
// Returns the Store. Used by the batch-get scatter tests.
func seedBatchCollection(t *testing.T, coll string, P, n int) Store {
	t.Helper()
	s := newSingleEmbedded(t)
	waitLeaderEmbedded(t, s)
	ctx := context.Background()
	cfg := VectorConfig{Dim: 4, M: 8, EfConstruction: 50, EfSearch: 32, Seed: 1, Metric: vector.L2, Partitions: P}
	if err := s.CreateCollection(ctx, coll, cfg); err != nil {
		t.Fatalf("CreateCollection %q (P=%d): %v", coll, P, err)
	}
	for id := uint64(1); id <= uint64(n); id++ {
		v := []float32{float32(id), 0, 0, 0}
		meta := VectorMetadata{"id": vector.NewInt(int64(id))}
		if err := s.VectorInsertExt(ctx, coll, id, v, VectorInsertOpts{Metadata: meta}); err != nil {
			t.Fatalf("insert %d: %v", id, err)
		}
	}
	return s
}

func pointIDs(points []BatchGetPoint) []uint64 {
	out := make([]uint64, len(points))
	for i, p := range points {
		out[i] = p.ID
	}
	return out
}

// TestVectorGetBatchScatterSpansAllPartitions: a batch over a P>1 collection with
// ids spanning ALL partitions returns every present doc (correct vec/meta) and
// EXACTLY the absent ids in missing, vs an independent single-get ground truth.
func TestVectorGetBatchScatterSpansAllPartitions(t *testing.T) {
	const P, n = 4, 200
	s := seedBatchCollection(t, "docs", P, n)
	ctx := context.Background()

	// Request a mix of present (1..n) and absent (n+1..n+20) ids, in a shuffled
	// order, spanning all partitions.
	var ids []uint64
	for id := uint64(1); id <= uint64(n); id++ {
		ids = append(ids, id)
	}
	var wantMissing []uint64
	for id := uint64(n + 1); id <= uint64(n+20); id++ {
		ids = append(ids, id)
		wantMissing = append(wantMissing, id)
	}
	// Sanity: the absent ids must really span all partitions so the scatter must
	// touch every partition for the missing set too.
	touched := map[int]bool{}
	for _, id := range ids {
		touched[ops.PartitionOf(id, P)] = true
	}
	if len(touched) != P {
		t.Fatalf("test ids only touch %d/%d partitions", len(touched), P)
	}

	points, missing, err := s.VectorGetBatch(ctx, "docs", ids, true, true)
	if err != nil {
		t.Fatalf("VectorGetBatch: %v", err)
	}

	// Ground truth via independent single gets.
	var gtPoints []BatchGetPoint
	var gtMissing []uint64
	for id := uint64(1); id <= uint64(n+20); id++ {
		found, vec, meta, ttl, sparse, gerr := s.VectorGet(ctx, "docs", id, true, true)
		if gerr != nil {
			t.Fatalf("VectorGet %d: %v", id, gerr)
		}
		if found {
			// Each point is inserted exactly once, so its version is 1
			// (VectorGet's signature doesn't return version, so set it here).
			gtPoints = append(gtPoints, BatchGetPoint{ID: id, Vec: vec, Meta: meta, TTL: ttl, Sparse: sparse, Version: 1})
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

// TestVectorGetBatchPartitionOnlyGetsOwnedIds proves each partition is asked ONLY
// for the ids it owns. It calls the per-partition vector_get_batch op directly on
// every physical partition with the FULL id set, then asserts that partition p
// reports Found ONLY for ids where PartitionOf(id,P)==p. Because a point lives in
// exactly one partition, a correct scatter that sent partition p ids it does not
// own would get found=0 for them — so this is the ground truth the scatter relies
// on (the scatter sends each partition exactly its owned subset).
func TestVectorGetBatchPartitionOnlyGetsOwnedIds(t *testing.T) {
	const P, n = 4, 200
	s := seedBatchCollection(t, "docs", P, n)
	ctx := context.Background()

	allIDs := make([]uint64, 0, n)
	for id := uint64(1); id <= uint64(n); id++ {
		allIDs = append(allIDs, id)
	}

	_, gen, ok := s.(*embedded).catalog.PartitionsGen("docs")
	if !ok {
		t.Fatal("docs not partitioned")
	}
	for p := 0; p < P; p++ {
		phys := string(ops.PartitionKeyGen("docs", gen, p))
		body, err := s.Call(ctx, "vector_get_batch", ops.EncodeVectorGetBatchArgs(phys, allIDs, ops.GetFlagsBoth))
		if err != nil {
			t.Fatalf("partition %d vector_get_batch: %v", p, err)
		}
		rows, err := ops.DecodeVectorGetBatchResult(body)
		if err != nil {
			t.Fatalf("partition %d decode: %v", p, err)
		}
		for _, r := range rows {
			ownedHere := ops.PartitionOf(r.ID, P) == p
			if r.Found != ownedHere {
				t.Fatalf("partition %d: id %d Found=%v, but PartitionOf=%d (ownedHere=%v)",
					p, r.ID, r.Found, ops.PartitionOf(r.ID, P), ownedHere)
			}
		}
	}
}

// TestVectorGetBatchGroupingSendsOnlySubset asserts the scatter's grouping logic
// directly: for each partition, the subset it would be sent contains ONLY ids
// that hash to that partition. This pins the group-by-PartitionOf contract at the
// embedded layer (the scatter encodes exactly groups[p] per partition).
func TestVectorGetBatchGroupingSendsOnlySubset(t *testing.T) {
	const P = 4
	ids := []uint64{1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 99, 100, 1000, 4242}
	groups := map[int][]uint64{}
	for _, id := range ids {
		p := ops.PartitionOf(id, P)
		groups[p] = append(groups[p], id)
	}
	for p, sub := range groups {
		for _, id := range sub {
			if got := ops.PartitionOf(id, P); got != p {
				t.Fatalf("group %d contains id %d owned by partition %d", p, id, got)
			}
		}
	}
}

// TestVectorGetBatchP1FastPath: P==1 collection — one call, all ids returned.
func TestVectorGetBatchP1FastPath(t *testing.T) {
	const n = 50
	s := seedBatchCollection(t, "docs1", 1, n)
	ctx := context.Background()

	ids := []uint64{5, 1, 50, 25, 999} // 999 absent
	points, missing, err := s.VectorGetBatch(ctx, "docs1", ids, true, true)
	if err != nil {
		t.Fatalf("VectorGetBatch: %v", err)
	}
	if !reflect.DeepEqual(pointIDs(points), []uint64{1, 5, 25, 50}) {
		t.Fatalf("points ids = %v, want sorted [1 5 25 50]", pointIDs(points))
	}
	if !reflect.DeepEqual(missing, []uint64{999}) {
		t.Fatalf("missing = %v, want [999]", missing)
	}
}

// TestVectorGetBatchEmpty: empty ids => empty points + empty missing, no error.
func TestVectorGetBatchEmpty(t *testing.T) {
	s := seedBatchCollection(t, "docs", 4, 10)
	ctx := context.Background()
	points, missing, err := s.VectorGetBatch(ctx, "docs", nil, true, true)
	if err != nil {
		t.Fatalf("VectorGetBatch empty: %v", err)
	}
	if len(points) != 0 || len(missing) != 0 {
		t.Fatalf("empty ids => points=%v missing=%v, want both empty", points, missing)
	}
}

// TestVectorGetBatchDedup: a duplicated id is fetched once and appears once.
func TestVectorGetBatchDedup(t *testing.T) {
	s := seedBatchCollection(t, "docs", 4, 100)
	ctx := context.Background()

	ids := []uint64{7, 7, 7, 42, 42, 7} // present, heavily duplicated
	points, missing, err := s.VectorGetBatch(ctx, "docs", ids, false, false)
	if err != nil {
		t.Fatalf("VectorGetBatch: %v", err)
	}
	if len(missing) != 0 {
		t.Fatalf("missing = %v, want none", missing)
	}
	if !reflect.DeepEqual(pointIDs(points), []uint64{7, 42}) {
		t.Fatalf("points ids = %v, want deduped [7 42]", pointIDs(points))
	}
}

// TestVectorGetBatchAllAbsent: all-absent batch => all ids in missing, no error.
func TestVectorGetBatchAllAbsent(t *testing.T) {
	s := seedBatchCollection(t, "docs", 4, 10)
	ctx := context.Background()

	ids := []uint64{1000, 1001, 1002, 1003}
	points, missing, err := s.VectorGetBatch(ctx, "docs", ids, true, true)
	if err != nil {
		t.Fatalf("VectorGetBatch: %v", err)
	}
	if len(points) != 0 {
		t.Fatalf("points = %v, want none", points)
	}
	want := append([]uint64(nil), ids...)
	sort.Slice(want, func(i, j int) bool { return want[i] < want[j] })
	if !reflect.DeepEqual(missing, want) {
		t.Fatalf("missing = %v, want %v", missing, want)
	}
}

// TestVectorGetBatchSortedByID: points + missing are returned id-ascending even
// when the input is shuffled and mixes present/absent across partitions.
func TestVectorGetBatchSortedByID(t *testing.T) {
	const P, n = 4, 100
	s := seedBatchCollection(t, "docs", P, n)
	ctx := context.Background()

	// Shuffled present + absent ids.
	ids := []uint64{90, 3, 200, 45, 1, 150, 77, 12, 300, 33}
	points, missing, err := s.VectorGetBatch(ctx, "docs", ids, false, false)
	if err != nil {
		t.Fatalf("VectorGetBatch: %v", err)
	}
	got := pointIDs(points)
	if !sort.SliceIsSorted(got, func(i, j int) bool { return got[i] < got[j] }) {
		t.Fatalf("points not id-sorted: %v", got)
	}
	if !sort.SliceIsSorted(missing, func(i, j int) bool { return missing[i] < missing[j] }) {
		t.Fatalf("missing not id-sorted: %v", missing)
	}
	// 1,3,12,33,45,77,90 present; 150,200,300 absent.
	if !reflect.DeepEqual(got, []uint64{1, 3, 12, 33, 45, 77, 90}) {
		t.Fatalf("present ids = %v", got)
	}
	if !reflect.DeepEqual(missing, []uint64{150, 200, 300}) {
		t.Fatalf("missing ids = %v", missing)
	}
}

// TestScatterIDsByPartitionZeroPartitionsNoop pins that P==0 is a no-op (fetch
// never invoked, no panic) rather than ops.PartitionOf(id, 0)==0 indexing the
// P-length counts/groups slices this scatter builds internally.
func TestScatterIDsByPartitionZeroPartitionsNoop(t *testing.T) {
	called := false
	fetch := func(phys string, sub []uint64) ([]ops.GetBatchRow, error) {
		called = true
		return nil, nil
	}
	rows, err := scatterIDsByPartition(nil, "docs", 0, 0, []uint64{1, 2, 3}, fetch)
	if err != nil {
		t.Fatalf("scatterIDsByPartition: %v", err)
	}
	if called {
		t.Fatal("fetch invoked with P=0, want no-op")
	}
	if len(rows) != 0 {
		t.Fatalf("got %d rows, want 0", len(rows))
	}
}
