// SPDX-License-Identifier: Apache-2.0

package inttest

import (
	"context"
	"sync/atomic"
	"testing"

	"github.com/rostamlabs/rostam"
	"github.com/rostamlabs/rostam/ops"
	"github.com/rostamlabs/rostam/shard"
	"github.com/rostamlabs/rostam/vector"
)

// TestLinearizableBarrierRunsOnRealReadPath is the regression test for the
// stale-serve hole: a strict-Linearizable read MUST actually enter the shard's
// readIndex barrier (shard.Store.verifyLeaderAndCatchUp) on the REAL production
// read path — the embedded VectorSearchExt → denseFanOut → CallPhysical → shard
// path for a partitioned (P>1) collection, AND the embedded VectorSearchExt →
// callReadLeader → CallPhysical → shard path for an unpartitioned (P==1)
// collection.
//
// Before the fix, the fan-out Encode closures re-encoded the per-partition args
// with NON-rc-carrying encoders (EncodeVectorSearchArgsExt), so the shard's
// ops.ReadConsistencyOf peek returned (0,false) and the barrier was NEVER
// entered — a Linearizable read silently degraded to a best-effort
// (stale-capable) read. The P==1 path used e.Call (plain node.Call) with no rc
// byte AND no leader routing, so it too skipped the barrier. This test would
// FAIL pre-fix: barrierEntries would stay 0 for the Linearizable reads, because
// the barrier hook can only fire when ReadConsistencyOf returns
// ConsistencyLinearizable, which requires the rc byte to reach the shard in the
// per-op args.
//
// The contrast cases (AnyReplica) assert the default path adds ZERO barrier
// cost — the hook must NOT fire for rc=0.
func TestLinearizableBarrierRunsOnRealReadPath(t *testing.T) {
	s := newSingleEmbedded(t)
	waitLeaderEmbedded(t, s)
	ctx := context.Background()

	// Observe every readIndex-barrier entry (process-wide hook; the embedded
	// engine runs in this process). Reset between phases by reading the delta.
	var barrierEntries atomic.Int64
	shard.SetBarrierEnteredHook(func() { barrierEntries.Add(1) })
	t.Cleanup(func() { shard.SetBarrierEnteredHook(nil) })
	delta := func() int64 {
		// Swap to 0 and return the prior value so each phase measures its own
		// barrier entries independently.
		return barrierEntries.Swap(0)
	}

	cfg := func(p int) rostam.VectorConfig {
		return rostam.VectorConfig{Dim: 4, M: 8, EfConstruction: 50, EfSearch: 64, Seed: 1, Metric: vector.L2, Partitions: p}
	}
	// Partitioned (P>1): a Linearizable search fans out and each partition's
	// leader read must enter the barrier.
	const P = 4
	if err := s.CreateCollection(ctx, "parts", cfg(P)); err != nil {
		t.Fatalf("CreateCollection parts (P=%d): %v", P, err)
	}
	// Unpartitioned (P==1): a Linearizable search routes to the leader and must
	// enter the barrier exactly once.
	if err := s.CreateCollection(ctx, "single", cfg(1)); err != nil {
		t.Fatalf("CreateCollection single (P=1): %v", err)
	}
	for id := uint64(1); id <= 80; id++ {
		v := []float32{float32(id), 0, 0, 0}
		if err := s.VectorInsert(ctx, "parts", id, v); err != nil {
			t.Fatalf("insert parts %d: %v", id, err)
		}
		if err := s.VectorInsert(ctx, "single", id, v); err != nil {
			t.Fatalf("insert single %d: %v", id, err)
		}
	}
	// Drain any barrier entries incurred during setup (writes don't enter it, but
	// be defensive so the assertions below measure only the reads under test).
	delta()

	query := []float32{1, 0, 0, 0}
	lin := rostam.VectorSearchOpts{ReadConsistency: ops.ConsistencyLinearizable}
	any := rostam.VectorSearchOpts{ReadConsistency: ops.ConsistencyAnyReplica}

	// (a) PARTITIONED Linearizable: barrier runs on every partition's leader.
	res, _, err := s.VectorSearchExt(ctx, "parts", query, 5, lin)
	if err != nil {
		t.Fatalf("linearizable VectorSearchExt parts: %v", err)
	}
	if len(res) == 0 || res[0].ID != 1 {
		t.Fatalf("linearizable parts results = %+v, want nearest id=1 first", res)
	}
	// >= 1 (not >= P): the per-partition fan-out reads are concurrent on this Store
	// pool, so partitions co-hosted on one shard.Store COALESCE into a shared readIndex
	// barrier (single-flight readindex coalescing — arrival<=capture safe). The
	// anti-stale guard is the FIRE itself: 0 ⇒ the fan-out re-encode dropped the rc
	// byte ⇒ STALE-SERVE HOLE. Coalescing only reduces the count, never to 0.
	if n := delta(); n < 1 {
		t.Fatalf("PARTITIONED linearizable search entered the barrier %d times, want >= 1 "+
			"(coalesced across co-hosted partitions). 0 ⇒ the fan-out re-encode dropped the rc byte ⇒ STALE-SERVE HOLE", n)
	}

	// (b) UNPARTITIONED (P==1) Linearizable: routes to the leader, barrier runs.
	res, _, err = s.VectorSearchExt(ctx, "single", query, 5, lin)
	if err != nil {
		t.Fatalf("linearizable VectorSearchExt single: %v", err)
	}
	if len(res) == 0 || res[0].ID != 1 {
		t.Fatalf("linearizable single results = %+v, want nearest id=1 first", res)
	}
	if n := delta(); n < 1 {
		t.Fatalf("UNPARTITIONED (P==1) linearizable search entered the barrier %d times, want >= 1 "+
			"(the P<=1 path must route to the leader WITH rc). 0 ⇒ the P==1 read skipped the barrier ⇒ STALE-SERVE HOLE", n)
	}

	// (c) CONTRAST: AnyReplica on the SAME collections must NOT enter the barrier
	// (zero added consensus cost on the default path).
	if _, _, err := s.VectorSearchExt(ctx, "parts", query, 5, any); err != nil {
		t.Fatalf("anyreplica VectorSearchExt parts: %v", err)
	}
	if n := delta(); n != 0 {
		t.Fatalf("PARTITIONED AnyReplica search entered the barrier %d times, want 0 (default path must be barrier-free)", n)
	}
	if _, _, err := s.VectorSearchExt(ctx, "single", query, 5, any); err != nil {
		t.Fatalf("anyreplica VectorSearchExt single: %v", err)
	}
	if n := delta(); n != 0 {
		t.Fatalf("UNPARTITIONED AnyReplica search entered the barrier %d times, want 0 (default path must be barrier-free)", n)
	}
}

// TestLinearizableBarrierRunsAllReadFamilies extends the real-path barrier proof
// to every Linearizable read family that fans out at the shard — docs, hybrid
// (vector_hybrid_lanes), groups (vector_group_candidates), scroll, and MV
// (vector_mv_search) — on a PARTITIONED collection. Pre-fix, each family's
// fan-out Encode closure dropped the rc byte (or ReadConsistencyOf lacked the
// real lanes/candidates op cases), so the barrier never ran and the read was
// stale-capable. This asserts the barrier fires at least once per family (the
// MUST-HAVE: the seam the prior tests bypassed by hand-encoding args).
func TestLinearizableBarrierRunsAllReadFamilies(t *testing.T) {
	s := newSingleEmbedded(t)
	waitLeaderEmbedded(t, s)
	ctx := context.Background()

	var barrierEntries atomic.Int64
	shard.SetBarrierEnteredHook(func() { barrierEntries.Add(1) })
	t.Cleanup(func() { shard.SetBarrierEnteredHook(nil) })
	delta := func() int64 { return barrierEntries.Swap(0) }

	const P = 4
	if err := s.CreateCollection(ctx, "docs", rostam.VectorConfig{
		Dim: 4, M: 8, EfConstruction: 50, EfSearch: 64, Seed: 1, Metric: vector.L2, Partitions: P,
	}); err != nil {
		t.Fatalf("CreateCollection docs: %v", err)
	}
	for id := uint64(1); id <= 80; id++ {
		v := []float32{float32(id), 0, 0, 0}
		if err := s.VectorUpsert(ctx, "docs", id, v, "chunk", rostam.VectorInsertOpts{
			Metadata: rostam.VectorMetadata{"grp": vector.NewInt(int64(id % 8))},
		}); err != nil {
			t.Fatalf("upsert docs %d: %v", id, err)
		}
	}
	delta()

	query := []float32{1, 0, 0, 0}

	// docs (vector_search_docs)
	if _, _, err := s.VectorSearchDocs(ctx, "docs", query, 5, rostam.VectorSearchOpts{ReadConsistency: ops.ConsistencyLinearizable}); err != nil {
		t.Fatalf("linearizable VectorSearchDocs: %v", err)
	}
	if n := delta(); n < 1 {
		t.Fatalf("linearizable docs fan-out entered barrier %d times, want >= 1 (coalesced; 0 ⇒ vector_search_docs rc dropped ⇒ STALE)", n)
	}

	// hybrid (vector_hybrid_lanes) — the per-partition op shares vector_hybrid_search wire.
	if _, _, err := s.VectorHybridSearch(ctx, "docs", query, 5, rostam.VectorHybridOpts{ReadConsistency: ops.ConsistencyLinearizable}); err != nil {
		t.Fatalf("linearizable VectorHybridSearch: %v", err)
	}
	if n := delta(); n < 1 {
		t.Fatalf("linearizable hybrid fan-out entered barrier %d times, want >= 1 (coalesced; 0 ⇒ "+
			"vector_hybrid_lanes rc dropped or ReadConsistencyOf missing the lanes case ⇒ STALE)", n)
	}

	// groups (vector_group_candidates) — the per-partition op shares vector_search_groups wire.
	if _, _, err := s.VectorSearchGroups(ctx, "docs", query, 3, rostam.VectorGroupOpts{
		GroupBy: "grp", GroupSize: 2, ReadConsistency: ops.ConsistencyLinearizable,
	}); err != nil {
		t.Fatalf("linearizable VectorSearchGroups: %v", err)
	}
	if n := delta(); n < 1 {
		t.Fatalf("linearizable groups fan-out entered barrier %d times, want >= 1 (coalesced; 0 ⇒ "+
			"vector_group_candidates rc dropped or ReadConsistencyOf missing the candidates case ⇒ STALE)", n)
	}

	// scroll (vector_scroll)
	if _, _, _, err := s.VectorScroll(ctx, "docs", rostam.VectorFilter{}, 10, rostam.VectorScrollOpts{ReadConsistency: ops.ConsistencyLinearizable}); err != nil {
		t.Fatalf("linearizable VectorScroll: %v", err)
	}
	if n := delta(); n < 1 {
		t.Fatalf("linearizable scroll fan-out entered barrier %d times, want >= 1 (coalesced; 0 ⇒ vector_scroll rc dropped ⇒ STALE)", n)
	}

	// Contrast: AnyReplica across the same families must not enter the barrier.
	if _, _, err := s.VectorSearchDocs(ctx, "docs", query, 5, rostam.VectorSearchOpts{}); err != nil {
		t.Fatalf("anyreplica VectorSearchDocs: %v", err)
	}
	if _, _, err := s.VectorHybridSearch(ctx, "docs", query, 5, rostam.VectorHybridOpts{}); err != nil {
		t.Fatalf("anyreplica VectorHybridSearch: %v", err)
	}
	if _, _, err := s.VectorSearchGroups(ctx, "docs", query, 3, rostam.VectorGroupOpts{GroupBy: "grp", GroupSize: 2}); err != nil {
		t.Fatalf("anyreplica VectorSearchGroups: %v", err)
	}
	if _, _, _, err := s.VectorScroll(ctx, "docs", rostam.VectorFilter{}, 10, rostam.VectorScrollOpts{}); err != nil {
		t.Fatalf("anyreplica VectorScroll: %v", err)
	}
	if n := delta(); n != 0 {
		t.Fatalf("AnyReplica reads across all families entered the barrier %d times, want 0 (default path must be barrier-free)", n)
	}
}

// TestLinearizableBarrierRunsMVRealPath proves the MV (multi-vector) search
// family enters the barrier on both the partitioned fan-out (vector_mv_search
// per partition) and the P==1 leader-routed path. MV uses a distinct create/
// search surface, so it gets its own collection.
func TestLinearizableBarrierRunsMVRealPath(t *testing.T) {
	s := newSingleEmbedded(t)
	waitLeaderEmbedded(t, s)
	ctx := context.Background()

	var barrierEntries atomic.Int64
	shard.SetBarrierEnteredHook(func() { barrierEntries.Add(1) })
	t.Cleanup(func() { shard.SetBarrierEnteredHook(nil) })
	delta := func() int64 { return barrierEntries.Swap(0) }

	const P = 4
	if err := s.VectorMVCreateCollection(ctx, "mvparts", rostam.MultiVectorConfig{Dim: 4, Partitions: P}); err != nil {
		t.Fatalf("VectorMVCreateCollection mvparts: %v", err)
	}
	if err := s.VectorMVCreateCollection(ctx, "mvsingle", rostam.MultiVectorConfig{Dim: 4, Partitions: 1}); err != nil {
		t.Fatalf("VectorMVCreateCollection mvsingle: %v", err)
	}
	doc := [][]float32{{1, 0, 0, 0}, {0, 1, 0, 0}}
	for id := uint64(1); id <= 40; id++ {
		if err := s.VectorMVAdd(ctx, "mvparts", id, doc, nil); err != nil {
			t.Fatalf("mv add parts %d: %v", id, err)
		}
		if err := s.VectorMVAdd(ctx, "mvsingle", id, doc, nil); err != nil {
			t.Fatalf("mv add single %d: %v", id, err)
		}
	}
	delta()

	q := [][]float32{{1, 0, 0, 0}}
	lin := rostam.MultiSearchOpts{ReadConsistency: ops.ConsistencyLinearizable}

	if _, _, err := s.VectorMVSearch(ctx, "mvparts", q, 5, lin); err != nil {
		t.Fatalf("linearizable VectorMVSearch parts: %v", err)
	}
	if n := delta(); n < 1 {
		t.Fatalf("PARTITIONED linearizable MV search entered barrier %d times, want >= 1 (coalesced; 0 ⇒ vector_mv_search rc dropped ⇒ STALE)", n)
	}

	if _, _, err := s.VectorMVSearch(ctx, "mvsingle", q, 5, lin); err != nil {
		t.Fatalf("linearizable VectorMVSearch single: %v", err)
	}
	if n := delta(); n < 1 {
		t.Fatalf("UNPARTITIONED (P==1) linearizable MV search entered barrier %d times, want >= 1 (P==1 MV skipped barrier ⇒ STALE)", n)
	}

	if _, _, err := s.VectorMVSearch(ctx, "mvparts", q, 5, rostam.MultiSearchOpts{}); err != nil {
		t.Fatalf("anyreplica VectorMVSearch parts: %v", err)
	}
	if _, _, err := s.VectorMVSearch(ctx, "mvsingle", q, 5, rostam.MultiSearchOpts{}); err != nil {
		t.Fatalf("anyreplica VectorMVSearch single: %v", err)
	}
	if n := delta(); n != 0 {
		t.Fatalf("AnyReplica MV searches entered the barrier %d times, want 0 (default path must be barrier-free)", n)
	}
}
