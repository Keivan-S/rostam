// SPDX-License-Identifier: Apache-2.0

package inttest

import (
	"context"
	"errors"
	"fmt"
	"math"
	"sync"
	"testing"
	"time"

	"github.com/rostamlabs/rostam"
	"github.com/rostamlabs/rostam/vector"
)

// Cluster correctness + integration for optimistic CAS (per-point
// monotonic version).
//
// Why payload mutations (not upserts) drive the CAS-determinism test: a dense
// Upsert is delete-then-insert (HNSW has no in-place vector update), so a
// re-upsert RESETS the version to 1 rather than bumping it — two CAS upserts with
// the same expected version would therefore BOTH pass (each sees the reset value).
// A payload mutation (set_payload) is a true in-place mutate that bumps the
// version current+1 under the engine lock, so it is the primitive that genuinely
// exercises "exactly one of two concurrent CAS writes wins". (The plan's version
// semantics: a payload mutation of an existing point → current+1.)
//
// RF>1 determinism note: the version check + bump are performed
// at the ENGINE MUTATOR under the engine lock — the SAME handler that runs inside
// Raft FSM Apply (shard/fsm.go → handler → mutator) for every OpReadWrite op. The
// bump is a PURE counter (current+1, or 1 for a fresh insert, or verbatim on a
// restore) with NO wall-clock and NO RNG input, so it is a deterministic function
// of (applied state, op bytes). Therefore every replica that applies the same
// Raft log entry computes the IDENTICAL resulting version — a follower's restored
// version is byte-for-byte the leader's. This determinism is STRUCTURAL (it falls
// out of "check+bump under the engine lock at FSM Apply, pure counter") rather
// than something a multi-node read-back could disprove without reproducing the
// structure, so we cover it via the single-node Raft-serialized coverage here
// (every embedded write is proposed to and applied from the Raft log) plus the
// dense/named/MV engine tests, and document the structural
// argument rather than standing up a flaky 3-node read-after-replication race.
// The two-concurrent-CAS test below directly exercises the serialization point:
// both writes are proposed to the Raft log and applied serially by the FSM, so
// exactly one can match the shared expected_version.

// readVersion returns id's current per-point version via the batch-get path (the
// public read that surfaces the version), or 0 if absent.
func readVersion(t *testing.T, s rostam.Store, coll string, id uint64) uint64 {
	t.Helper()
	pts, _, err := s.VectorGetBatch(context.Background(), coll, []uint64{id}, false, false)
	if err != nil {
		t.Fatalf("VectorGetBatch %s id=%d: %v", coll, id, err)
	}
	for _, p := range pts {
		if p.ID == id {
			return p.Version
		}
	}
	return 0
}

// TestCASConflictAtRaftSerializationPoint is the core Task-4 determinism gate.
// Two concurrent CAS payload mutations carrying the SAME expected_version race;
// because the version check+bump happens at FSM Apply (the Raft serialization
// point), the log orders them and exactly ONE matches the expected version while
// the other sees the already-bumped version and returns ErrVersionConflict. This
// proves the check is evaluated at the engine mutator under FSM Apply, NOT at the
// coordinator (a coordinator-side check would let both reads observe the same
// version and both "win").
func TestCASConflictAtRaftSerializationPoint(t *testing.T) {
	s := newSingleEmbedded(t)
	waitLeaderEmbedded(t, s)
	ctx := context.Background()

	must(t, s.CreateCollection(ctx, "cas", rostam.VectorConfig{Dim: 4, Metric: vector.L2, M: 8, EfConstruction: 50, EfSearch: 64, Seed: 1}))

	const id = uint64(1)
	// Seed the point (fresh insert → version 1).
	must(t, s.VectorUpsert(ctx, "cas", id, []float32{1, 0, 0, 0}, "v1", rostam.VectorInsertOpts{}))
	v := readVersion(t, s, "cas", id)
	if v != 1 {
		t.Fatalf("seeded version = %d, want 1", v)
	}

	// A CAS payload mutation with the WRONG expected_version must conflict and NOT
	// mutate (the check is at the mutator, so a stale expected is rejected).
	wrong := v + 99
	_, err := s.VectorSetPayload(ctx, "cas", id, vector.Metadata{"k": vector.NewInt(1)}, nil, rostam.WriteOpts{ExpectedVersion: &wrong})
	if !errors.Is(err, vector.ErrVersionConflict) {
		t.Fatalf("CAS set_payload with wrong expected_version: err = %v, want ErrVersionConflict", err)
	}
	if got := readVersion(t, s, "cas", id); got != v {
		t.Fatalf("version after rejected CAS = %d, want unchanged %d (no mutation on conflict)", got, v)
	}

	// The crux: TWO concurrent CAS payload mutations with the SAME (correct)
	// expected version. Exactly one must succeed; the other must conflict.
	expected := v
	var wg sync.WaitGroup
	errs := make([]error, 2)
	start := make(chan struct{})
	for i := 0; i < 2; i++ {
		wg.Add(1)
		go func(i int) {
			defer wg.Done()
			exp := expected
			<-start // release both as close together as possible
			_, errs[i] = s.VectorSetPayload(ctx, "cas", id, vector.Metadata{"racer": vector.NewInt(int64(i))}, nil, rostam.WriteOpts{ExpectedVersion: &exp})
		}(i)
	}
	close(start)
	wg.Wait()

	wins, conflicts := 0, 0
	for i, e := range errs {
		switch {
		case e == nil:
			wins++
		case errors.Is(e, vector.ErrVersionConflict):
			conflicts++
		default:
			t.Fatalf("racer %d: unexpected error %v (want nil or ErrVersionConflict)", i, e)
		}
	}
	if wins != 1 || conflicts != 1 {
		t.Fatalf("concurrent CAS: wins=%d conflicts=%d, want exactly 1 each (Raft serialization must let exactly one match the shared expected_version)", wins, conflicts)
	}

	// The single winner bumped the version exactly once: 1 -> 2.
	if got := readVersion(t, s, "cas", id); got != expected+1 {
		t.Fatalf("version after concurrent CAS = %d, want %d (exactly one bump)", got, expected+1)
	}
}

// TestCASInsertIfAbsentAtRaftSerializationPoint covers the expected_version==0
// (insert-if-absent CAS) variant under a concurrent race: two writers each assert
// the point is ABSENT (expected_version 0) via a CAS upsert. The log serializes
// them, so the first creates it (version 1) and the second sees version 1 !=
// expected 0 and conflicts — proving expect-absent is also evaluated at FSM Apply.
func TestCASInsertIfAbsentAtRaftSerializationPoint(t *testing.T) {
	s := newSingleEmbedded(t)
	waitLeaderEmbedded(t, s)
	ctx := context.Background()
	must(t, s.CreateCollection(ctx, "casia", rostam.VectorConfig{Dim: 4, Metric: vector.L2, M: 8, EfConstruction: 50, EfSearch: 64, Seed: 1}))

	const id = uint64(7)
	var wg sync.WaitGroup
	errs := make([]error, 2)
	start := make(chan struct{})
	for i := 0; i < 2; i++ {
		wg.Add(1)
		go func(i int) {
			defer wg.Done()
			exp := uint64(0)
			<-start
			errs[i] = s.VectorUpsert(ctx, "casia", id, []float32{float32(i + 1), 0, 0, 0}, fmt.Sprintf("ia-%d", i), rostam.VectorInsertOpts{WriteOpts: rostam.WriteOpts{ExpectedVersion: &exp}})
		}(i)
	}
	close(start)
	wg.Wait()

	wins, conflicts := 0, 0
	for i, e := range errs {
		switch {
		case e == nil:
			wins++
		case errors.Is(e, vector.ErrVersionConflict):
			conflicts++
		default:
			t.Fatalf("ia racer %d: unexpected error %v", i, e)
		}
	}
	if wins != 1 || conflicts != 1 {
		t.Fatalf("concurrent insert-if-absent CAS: wins=%d conflicts=%d, want exactly 1 each", wins, conflicts)
	}
	if got := readVersion(t, s, "casia", id); got != 1 {
		t.Fatalf("version after insert-if-absent CAS = %d, want 1 (one fresh insert)", got)
	}
}

// TestOnlineReshardPreservesVersion is the version-preservation gate for the
// ONLINE reshard path (VectorReshard → reshardCopyPass). Before the Task-4 fix,
// reshardCopyPass copied points via vector_insert_if_absent WITHOUT carrying the
// version, so every copied point's version reset to 1 — silently dropping CAS
// history on a live reshard. The fix routes the copy through
// EncodeVectorInsertArgsVersioned (the version-preserving if-absent), so a
// resharded point keeps its exact per-point version. Here we seed points and bump
// some of them past version 1 with payload mutations (the in-place bump path),
// reshard online, and assert every point's post-reshard version equals its
// pre-reshard version.
func TestOnlineReshardPreservesVersion(t *testing.T) {
	// Speed up the online reshard's drain grace (setup-only, mirrors the existing
	// online-reshard integration tests).
	defer rostam.SetReshardDrainGrace(20 * time.Millisecond)()

	s := newSingleEmbedded(t)
	waitLeaderEmbedded(t, s)
	ctx := context.Background()

	const (
		oldP = 2
		newP = 4
		N    = 24
	)
	must(t, s.CreateCollection(ctx, "rev", rostam.VectorConfig{Dim: 4, Metric: vector.L2, M: 8, EfConstruction: 50, EfSearch: 64, Seed: 1, Partitions: oldP}))

	// Seed N points, then bump id i an extra (i % 3) times via payload mutations so
	// versions span 1, 2, 3 across ids — proving the copy carries the EXACT version,
	// not a constant.
	want := make(map[uint64]uint64, N)
	for id := uint64(1); id <= N; id++ {
		must(t, s.VectorUpsert(ctx, "rev", id, []float32{float32(id), 0, 0, 0}, fmt.Sprintf("doc-%d", id), rostam.VectorInsertOpts{}))
		extra := int(id % 3)
		for b := 0; b < extra; b++ {
			applied, err := s.VectorSetPayload(ctx, "rev", id, vector.Metadata{"bump": vector.NewInt(int64(b))}, nil)
			must(t, err)
			if !applied {
				t.Fatalf("set_payload id=%d bump=%d not applied", id, b)
			}
		}
		want[id] = uint64(1 + extra)
	}
	// Sanity: pre-reshard versions match expectations AND we DO have versions > 1.
	sawGT1 := false
	for id := uint64(1); id <= N; id++ {
		got := readVersion(t, s, "rev", id)
		if got != want[id] {
			t.Fatalf("pre-reshard version id=%d = %d, want %d", id, got, want[id])
		}
		if got > 1 {
			sawGT1 = true
		}
	}
	if !sawGT1 {
		t.Fatal("test setup bug: no point with version > 1 (would not exercise the gap)")
	}

	// ONLINE reshard 2 -> 4.
	must(t, s.VectorReshard(ctx, "rev", newP))

	// Every point's version survives the online copy VERBATIM.
	for id := uint64(1); id <= N; id++ {
		got := readVersion(t, s, "rev", id)
		if got != want[id] {
			t.Fatalf("post-reshard version id=%d = %d, want %d (online reshard must preserve version)", id, got, want[id])
		}
	}
}

// mvCasTokenAt produces a strictly-separated unit token (distinct cosine
// distances) so the seeded MV docs are unambiguous — mirrors mvTokenAt.
func mvCasTokenAt(i int) []float32 {
	theta := float64(i) * (math.Pi / 2 / 64)
	return []float32{float32(math.Cos(theta)), float32(math.Sin(theta)), 0, 0}
}

// readMVVersion returns id's current per-document MV version via the batch-get
// path (the public read that surfaces the version), or 0 if absent.
func readMVVersion(t *testing.T, s rostam.Store, coll string, id uint64) uint64 {
	t.Helper()
	pts, _, err := s.VectorMVGetBatch(context.Background(), coll, []uint64{id}, false, false)
	if err != nil {
		t.Fatalf("VectorMVGetBatch %s id=%d: %v", coll, id, err)
	}
	for _, p := range pts {
		if p.ID == id {
			return p.Version
		}
	}
	return 0
}

// seedMVVersions creates a partitioned MV collection with N docs, then bumps doc
// i an extra (i % 3) times via in-place payload mutations so versions span 1, 2,
// 3 across ids — the SAME bump pattern as the dense TestOnlineReshardPreservesVersion.
// It asserts the pre-reshard versions match (incl. at least one > 1, so the test
// genuinely exercises the gap) and returns the expected version map.
func seedMVVersions(t *testing.T, s rostam.Store, coll string, oldP, N int) map[uint64]uint64 {
	t.Helper()
	ctx := context.Background()
	must(t, s.VectorMVCreateCollection(ctx, coll, rostam.MultiVectorConfig{Dim: 4, Partitions: oldP}))

	want := make(map[uint64]uint64, N)
	for id := uint64(1); id <= uint64(N); id++ {
		must(t, s.VectorMVAdd(ctx, coll, id, [][]float32{mvCasTokenAt(int(id))}, nil))
		extra := int(id % 3)
		for b := 0; b < extra; b++ {
			applied, err := s.VectorMVSetPayload(ctx, coll, id, vector.Metadata{"bump": vector.NewInt(int64(b))}, nil)
			must(t, err)
			if !applied {
				t.Fatalf("mv set_payload id=%d bump=%d not applied", id, b)
			}
		}
		want[id] = uint64(1 + extra)
	}

	sawGT1 := false
	for id := uint64(1); id <= uint64(N); id++ {
		got := readMVVersion(t, s, coll, id)
		if got != want[id] {
			t.Fatalf("pre-reshard MV version id=%d = %d, want %d", id, got, want[id])
		}
		if got > 1 {
			sawGT1 = true
		}
	}
	if !sawGT1 {
		t.Fatal("test setup bug: no MV doc with version > 1 (would not exercise the gap)")
	}
	return want
}

// TestMVOnlineReshardPreservesVersion is the MV analog of
// TestOnlineReshardPreservesVersion for the ONLINE MV path (VectorMVReshard →
// mvReshardCopyPass). Before the fix, mvReshardCopyPass copied docs via
// vector_mv_add_if_absent + EncodeMVAddArgs (no version → AddIfAbsent hardcodes
// version 1), silently dropping CAS history on a live MV reshard. The fix routes
// the copy through EncodeMVAddArgsVersioned carrying r.Version (the scan record's
// version), so a resharded doc keeps its exact per-document version while the
// copy stays if-absent (dual-write Race-A protection intact). This test FAILS if
// the encoder swap at embedded.go's mvReshardCopyPass is reverted (the > 1 docs
// would read back as 1).
func TestMVOnlineReshardPreservesVersion(t *testing.T) {
	// Speed up the online reshard's drain grace (setup-only, mirrors the dense
	// online-reshard integration tests).
	defer rostam.SetReshardDrainGrace(20 * time.Millisecond)()

	s := newSingleEmbedded(t)
	waitLeaderEmbedded(t, s)
	ctx := context.Background()

	const (
		coll = "mvrev"
		oldP = 2
		newP = 4
		N    = 24
	)
	want := seedMVVersions(t, s, coll, oldP, N)

	// ONLINE MV reshard 2 -> 4.
	must(t, s.VectorMVReshard(ctx, coll, newP))

	// Every doc's version survives the online copy VERBATIM.
	for id := uint64(1); id <= N; id++ {
		got := readMVVersion(t, s, coll, id)
		if got != want[id] {
			t.Fatalf("post-online-reshard MV version id=%d = %d, want %d (online MV reshard must preserve version)", id, got, want[id])
		}
	}
}

// TestMVOfflineResplitPreservesVersion is the MV analog for the OFFLINE MV path
// (VectorMVResplit). Before the fix, the resplit backfill copied docs via
// vector_mv_add + EncodeMVAddArgs (no version → addLocked BUMPS to 1), dropping
// CAS history. The fix routes the copy through the new op vector_mv_add_versioned
// + EncodeMVAddArgs Versioned carrying r.Version → handleMVAddVersioned →
// MultiRestoreAdd → restoreAdd (verbatim version, NOT bumped). This test FAILS if
// the encoder/op swap at embedded.go's VectorMVResplit is reverted. Offline path:
// no concurrent writes, so no drain-grace tuning is needed.
func TestMVOfflineResplitPreservesVersion(t *testing.T) {
	s := newSingleEmbedded(t)
	waitLeaderEmbedded(t, s)
	ctx := context.Background()

	const (
		coll = "mvrsp"
		oldP = 2
		newP = 4
		N    = 24
	)
	want := seedMVVersions(t, s, coll, oldP, N)

	// OFFLINE MV resplit 2 -> 4.
	must(t, s.VectorMVResplit(ctx, coll, newP))

	// Every doc's version survives the offline copy VERBATIM.
	for id := uint64(1); id <= N; id++ {
		got := readMVVersion(t, s, coll, id)
		if got != want[id] {
			t.Fatalf("post-offline-resplit MV version id=%d = %d, want %d (offline MV resplit must preserve version)", id, got, want[id])
		}
	}
}
