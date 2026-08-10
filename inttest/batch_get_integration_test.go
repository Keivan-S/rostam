// SPDX-License-Identifier: Apache-2.0

package inttest

import (
	"context"
	"fmt"
	"reflect"
	"sort"
	"testing"
	"time"

	"github.com/rostamlabs/rostam"
	"github.com/rostamlabs/rostam/ops"
	"github.com/rostamlabs/rostam/vector"
)

// seedBatchGetCollection creates a P-partition dense collection `name` on the
// creating coordinator and upserts ids 1..N, each with a DISTINCT vector and a
// DISTINCT payload, retrying through replication/election jitter (every transient
// error, per id, within a generous budget — mirroring seedDocs). It returns the
// ground-truth maps (id -> vector, id -> payload) so the batch-get reads can be
// checked against an independent oracle the test itself constructed.
//
// The vector is {id, id*2, id*3, id mod 7} and the payload is {"n": id, "tag":
// "p<id mod 4>"} so that (a) every id's vector is unique (the vec equality in the
// assertions is genuinely load-bearing), (b) the payload varies per id AND has a
// field that correlates with the partition spread, and (c) nothing is tie-prone.
func seedBatchGetCollection(t *testing.T, ctx context.Context, coord rostam.Store, name string, P, N int) (map[uint64][]float32, map[uint64]rostam.VectorMetadata) {
	t.Helper()
	createCollectionTolerant(t, ctx, coord, name, rostam.VectorConfig{
		Dim: 4, Metric: vector.L2, M: 8, EfConstruction: 50, EfSearch: 64, Seed: 1, Partitions: P,
	})
	wantVec := make(map[uint64][]float32, N)
	wantMeta := make(map[uint64]rostam.VectorMetadata, N)
	for id := uint64(1); id <= uint64(N); id++ {
		v := []float32{float32(id), float32(id) * 2, float32(id) * 3, float32(id % 7)}
		md := rostam.VectorMetadata{
			"n":   vector.NewInt(int64(id)),
			"tag": vector.NewString(fmt.Sprintf("p%d", id%4)),
		}
		wantVec[id] = v
		wantMeta[id] = md
		deadline := time.Now().Add(90 * time.Second)
		var lastErr error
		ok := false
		for time.Now().Before(deadline) {
			if err := coord.VectorInsertExt(ctx, name, id, v, rostam.VectorInsertOpts{Metadata: md}); err != nil {
				lastErr = err
				time.Sleep(50 * time.Millisecond)
				continue
			}
			ok = true
			break
		}
		if !ok {
			t.Fatalf("seed insert id=%d never succeeded: %v", id, lastErr)
		}
	}
	return wantVec, wantMeta
}

// pointIDs extracts the present-point ids in order (used to assert sortedness +
// the exact present set).
func pointIDs(pts []rostam.BatchGetPoint) []uint64 {
	out := make([]uint64, len(pts))
	for i, p := range pts {
		out[i] = p.ID
	}
	return out
}

// isSortedAsc reports whether ids is strictly ascending (so it doubles as a
// no-duplicates check: a dup would break strict ascending).
func isSortedAsc(ids []uint64) bool {
	for i := 1; i < len(ids); i++ {
		if ids[i] <= ids[i-1] {
			return false
		}
	}
	return true
}

// TestGetBatchScatterAcrossPartitions is the headline batch-get integration
// proof: on a real 3-node cluster, a dense batch get scatters a list of ids to
// their owning partitions, and the merged result returns EXACTLY the present
// requested ids (with the correct vector + payload vs the ground truth the test
// seeded) plus EXACTLY the absent requested ids in `missing`. All sub-cases are
// driven from a NON-creating coordinator (node 1), exercising the durable
// meta-Raft catalog convergence + the cross-node CallPhysical scatter over the
// wire.
//
// Correctness == correct grouping: because batch get groups ids by
// ops.PartitionOf(id, P) and asks each partition ONLY for its owned subset, if
// the grouping were wrong (a partition asked for an id it does not own) that id
// would come back NOT FOUND (each partition only stores its own ids) and would
// land in `missing` instead of `points`. So an exact present/missing match on a
// batch spanning ALL partitions is a direct proof that every id was routed to the
// partition that actually holds it. There is no clean per-partition CallPhysical
// counter seam for batch get, so this exact-correctness is the per-partition
// subset assertion (documented in the plan as the fallback when no counter hook
// exists). We additionally assert below that the requested present ids genuinely
// span every partition, so the proof is not vacuous.
func TestGetBatchScatterAcrossPartitions(t *testing.T) {
	stores := sharedInmemEmbeddedCluster(t, 3, 8)
	ctx := context.Background()

	const (
		P = 4   // P>1: a real scatter across partitions
		N = 200 // seeded ids 1..200
	)
	// Seed via node 0 (the creating coordinator); read via node 1 (NON-creating).
	wantVec, wantMeta := seedBatchGetCollection(t, ctx, stores[0], "bg", P, N)
	read := stores[1]
	waitEmbeddedCatalogGen(t, read.(*rostam.Embedded), "bg", P, 0, 20*time.Second)

	// Build a spread of PRESENT ids that provably spans ALL P partitions, plus a
	// block of ABSENT ids (5000..5010, never inserted). The present spread is a
	// stride across 1..200 so consecutive ids land in different partitions; we
	// then verify the chosen present set actually covers every partition before
	// relying on the "spans all partitions" claim.
	presentSet := map[uint64]bool{}
	for id := uint64(7); id <= N; id += 13 { // 7,20,33,...,196 — a scattered spread
		presentSet[id] = true
	}
	// Ensure a few specific edge ids are present too (first + last seeded id).
	presentSet[1] = true
	presentSet[uint64(N)] = true
	present := make([]uint64, 0, len(presentSet))
	for id := range presentSet {
		present = append(present, id)
	}
	sort.Slice(present, func(i, j int) bool { return present[i] < present[j] })

	// Verify the present request genuinely spans every partition (non-vacuous proof).
	coveredParts := map[int]bool{}
	for _, id := range present {
		coveredParts[ops.PartitionOf(id, P)] = true
	}
	if len(coveredParts) != P {
		t.Fatalf("present request spans %d/%d partitions (%v); test would not prove cross-partition scatter — adjust the spread",
			len(coveredParts), P, coveredParts)
	}

	absent := []uint64{5000, 5001, 5002, 5003, 5004, 5005, 5006, 5007, 5008, 5009, 5010}

	// assertBatch checks a (points, missing) result against the expected present +
	// absent id sets and the seeded ground truth, honoring the projection flags.
	assertBatch := func(t *testing.T, label string, pts []rostam.BatchGetPoint, missing []uint64,
		wantPresent, wantAbsent []uint64, withVector, withPayload bool) {
		t.Helper()

		// points: EXACT present set, sorted ascending, no dups.
		gotIDs := pointIDs(pts)
		if !isSortedAsc(gotIDs) {
			t.Fatalf("%s: points ids not strictly ascending (unsorted or duplicated): %v", label, gotIDs)
		}
		if !reflect.DeepEqual(gotIDs, wantPresent) {
			t.Fatalf("%s: points ids = %v, want EXACTLY %v (no extras, no missing, no dups)", label, gotIDs, wantPresent)
		}

		// missing: EXACT absent set, sorted ascending, no dups.
		if !isSortedAsc(missing) {
			t.Fatalf("%s: missing ids not strictly ascending (unsorted or duplicated): %v", label, missing)
		}
		if !reflect.DeepEqual(missing, wantAbsent) {
			t.Fatalf("%s: missing = %v, want EXACTLY %v", label, missing, wantAbsent)
		}

		// len(points)+len(missing) == #distinct requested ids.
		if len(pts)+len(missing) != len(wantPresent)+len(wantAbsent) {
			t.Fatalf("%s: len(points)=%d + len(missing)=%d = %d, want %d (every distinct requested id is accounted for exactly once)",
				label, len(pts), len(missing), len(pts)+len(missing), len(wantPresent)+len(wantAbsent))
		}

		// Per-point ground-truth: vector + payload, honoring the projection.
		for _, p := range pts {
			if withVector {
				if !reflect.DeepEqual(p.Vec, wantVec[p.ID]) {
					t.Fatalf("%s: id=%d vec = %v, want %v (scatter routed this id to the WRONG partition or corrupted the row)",
						label, p.ID, p.Vec, wantVec[p.ID])
				}
			} else if len(p.Vec) != 0 {
				t.Fatalf("%s: id=%d withVector=false but Vec=%v (projection leaked the vector)", label, p.ID, p.Vec)
			}
			if withPayload {
				if !reflect.DeepEqual(p.Meta, wantMeta[p.ID]) {
					t.Fatalf("%s: id=%d meta = %v, want %v", label, p.ID, p.Meta, wantMeta[p.ID])
				}
			} else if len(p.Meta) != 0 {
				t.Fatalf("%s: id=%d withPayload=false but Meta=%v (projection leaked the payload)", label, p.ID, p.Meta)
			}
		}
	}

	// (1) Mixed present + absent, full projection (vec + payload). The request
	// interleaves present + absent ids; the result must split them exactly.
	t.Run("MixedSpanningAllPartitions", func(t *testing.T) {
		req := append(append([]uint64{}, present...), absent...)
		pts, missing, err := read.VectorGetBatch(ctx, "bg", req, true, true)
		if err != nil {
			t.Fatalf("VectorGetBatch (mixed): %v", err)
		}
		assertBatch(t, "mixed", pts, missing, present, absent, true, true)
	})

	// (2) Projection variant: withVector=false, withPayload=false. Same id split,
	// but points carry NO vec + NO payload (id-only projection).
	t.Run("ProjectionIdOnly", func(t *testing.T) {
		req := append(append([]uint64{}, present...), absent...)
		pts, missing, err := read.VectorGetBatch(ctx, "bg", req, false, false)
		if err != nil {
			t.Fatalf("VectorGetBatch (id-only): %v", err)
		}
		assertBatch(t, "id-only", pts, missing, present, absent, false, false)
	})

	// (2b) Projection variant: withVector=true, withPayload=false (vec only).
	t.Run("ProjectionVecOnly", func(t *testing.T) {
		req := append(append([]uint64{}, present...), absent...)
		pts, missing, err := read.VectorGetBatch(ctx, "bg", req, true, false)
		if err != nil {
			t.Fatalf("VectorGetBatch (vec-only): %v", err)
		}
		assertBatch(t, "vec-only", pts, missing, present, absent, true, false)
	})

	// (3) All-present batch: EVERY seeded id (1..N) returns, none missing. This is
	// the densest scatter — every partition is asked for its full owned subset.
	t.Run("AllPresent", func(t *testing.T) {
		all := make([]uint64, 0, N)
		for id := uint64(1); id <= uint64(N); id++ {
			all = append(all, id)
		}
		pts, missing, err := read.VectorGetBatch(ctx, "bg", all, true, true)
		if err != nil {
			t.Fatalf("VectorGetBatch (all-present): %v", err)
		}
		assertBatch(t, "all-present", pts, missing, all, nil, true, true)
	})

	// (4) All-absent batch: empty points + EVERY requested id in missing, NO error
	// (a full miss is normal, never an op error).
	t.Run("AllAbsent", func(t *testing.T) {
		req := append([]uint64{}, absent...)
		pts, missing, err := read.VectorGetBatch(ctx, "bg", req, true, true)
		if err != nil {
			t.Fatalf("VectorGetBatch (all-absent): %v (a full miss must NOT be an error)", err)
		}
		if len(pts) != 0 {
			t.Fatalf("all-absent: points = %v, want empty", pointIDs(pts))
		}
		if !reflect.DeepEqual(missing, absent) {
			t.Fatalf("all-absent: missing = %v, want EXACTLY %v", missing, absent)
		}
	})
}

// TestGetBatchSinglePartitionFastPath proves the P==1 fast path (one Call with
// all ids, no scatter) returns the same exact present/missing split as the
// partitioned path — driven from a non-creating coordinator on a real cluster.
func TestGetBatchSinglePartitionFastPath(t *testing.T) {
	stores := sharedInmemEmbeddedCluster(t, 3, 8)
	ctx := context.Background()

	const N = 60
	wantVec, wantMeta := seedBatchGetCollection(t, ctx, stores[0], "bg1", 1, N)
	read := stores[1]
	// P==1 registers NO partitioned catalog entry (routes by the plain logical
	// name), so we cannot wait on PartitionsGen; the create already converged via
	// createCollectionTolerant on node 0. Give node 1's catalog a beat and rely on
	// the read's own forwarding to the owning shard.

	present := []uint64{1, 7, 23, 42, 60}
	absent := []uint64{7000, 7001, 7002}
	req := append(append([]uint64{}, present...), absent...)

	pts, missing, err := read.VectorGetBatch(ctx, "bg1", req, true, true)
	if err != nil {
		t.Fatalf("VectorGetBatch (P==1): %v", err)
	}
	if got := pointIDs(pts); !isSortedAsc(got) || !reflect.DeepEqual(got, present) {
		t.Fatalf("P==1: points ids = %v, want EXACTLY %v sorted", got, present)
	}
	if !isSortedAsc(missing) || !reflect.DeepEqual(missing, absent) {
		t.Fatalf("P==1: missing = %v, want EXACTLY %v sorted", missing, absent)
	}
	for _, p := range pts {
		if !reflect.DeepEqual(p.Vec, wantVec[p.ID]) {
			t.Fatalf("P==1: id=%d vec = %v, want %v", p.ID, p.Vec, wantVec[p.ID])
		}
		if !reflect.DeepEqual(p.Meta, wantMeta[p.ID]) {
			t.Fatalf("P==1: id=%d meta = %v, want %v", p.ID, p.Meta, wantMeta[p.ID])
		}
	}
}
