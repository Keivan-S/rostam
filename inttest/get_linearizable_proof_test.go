// SPDX-License-Identifier: Apache-2.0

package inttest

import (
	"context"
	"sync/atomic"
	"testing"

	rostam "github.com/rostamlabs/rostam"
	"github.com/rostamlabs/rostam/ops"
	"github.com/rostamlabs/rostam/shard"
	"github.com/rostamlabs/rostam/vector"
)

// TestGetLinearizableShardBarrierRealPath is the get-family analogue of
// TestNamedLinearizableShardBarrierRealPath: a Linearizable POINT-GET (dense /
// named / MV) and a Linearizable GET_CONFIG genuinely enter the SHARD readIndex
// barrier (shard.Store.verifyLeaderAndCatchUp) — i.e. vector_get / vector_named_get
// / vector_mv_get / vector_named_get_config being in ops.ReadConsistencyOf actually
// arms the barrier, and the embedded route carries the rc byte to the owning
// partition's leader (P>1 route-by-id) / the leader (P==1).
//
// THIS would FAIL if any get op were dropped from ReadConsistencyOf (the silent-
// degrade hole) or the embedded fan-out dropped the rc byte: the shard's peek would
// return (0,false), the barrier would never run, and a Linearizable get would
// silently serve stale — barrierEntries would stay 0. The AnyReplica contrast
// asserts the default path adds ZERO barrier cost.
func TestGetLinearizableShardBarrierRealPath(t *testing.T) {
	s := newSingleEmbedded(t)
	waitLeaderEmbedded(t, s)
	ctx := context.Background()

	var barrierEntries atomic.Int64
	shard.SetBarrierEnteredHook(func() { barrierEntries.Add(1) })
	t.Cleanup(func() { shard.SetBarrierEnteredHook(nil) })
	delta := func() int64 { return barrierEntries.Swap(0) }

	cfg := rostam.VectorConfig{Dim: 3, Metric: vector.L2, M: 8, EfConstruction: 50, EfSearch: 32, Seed: 1}
	// Partitioned (P>1): a point-get routes BY ID to one partition's leader; the
	// barrier runs on that leader. Unpartitioned (P==1): callReadLeader routes to
	// the single leader.
	cfgP := cfg
	cfgP.Partitions = 4
	if err := s.CreateCollection(ctx, "parts", cfgP); err != nil {
		t.Fatalf("CreateCollection parts (P=4): %v", err)
	}
	if err := s.CreateCollection(ctx, "single", cfg); err != nil {
		t.Fatalf("CreateCollection single (P=1): %v", err)
	}
	for id := 1; id <= 40; id++ {
		v := []float32{float32(id), 0, 0}
		if err := s.VectorUpsert(ctx, "parts", uint64(id), v, "x", rostam.VectorInsertOpts{}); err != nil {
			t.Fatalf("upsert parts %d: %v", id, err)
		}
		if err := s.VectorUpsert(ctx, "single", uint64(id), v, "x", rostam.VectorInsertOpts{}); err != nil {
			t.Fatalf("upsert single %d: %v", id, err)
		}
	}
	delta() // drain any setup entries

	lin := rostam.ReadOpts{ReadConsistency: ops.ConsistencyLinearizable}
	any := rostam.ReadOpts{ReadConsistency: ops.ConsistencyAnyReplica}

	// (a) PARTITIONED Linearizable dense get: routes by id to the owning partition's
	// leader, barrier runs (>= 1). 0 ⇒ vector_get dropped from ReadConsistencyOf or
	// the fan-out dropped rc ⇒ STALE-SERVE.
	found, _, _, _, _, err := s.VectorGetExt(ctx, "parts", 7, true, true, lin)
	if err != nil {
		t.Fatalf("linearizable VectorGetExt parts: %v", err)
	}
	if !found {
		t.Fatalf("linearizable dense get parts id=7 not found (read-your-writes failed)")
	}
	if n := delta(); n < 1 {
		t.Fatalf("PARTITIONED linearizable dense get entered the shard barrier %d times, want >= 1 "+
			"(0 ⇒ vector_get not in ReadConsistencyOf or rc dropped on the route ⇒ STALE-SERVE HOLE)", n)
	}

	// (b) UNPARTITIONED (P==1) Linearizable dense get: routes to the leader via
	// callReadLeader, barrier runs (>= 1).
	if found, _, _, _, _, err := s.VectorGetExt(ctx, "single", 7, true, true, lin); err != nil || !found {
		t.Fatalf("linearizable VectorGetExt single: found=%v err=%v", found, err)
	}
	if n := delta(); n < 1 {
		t.Fatalf("UNPARTITIONED (P==1) linearizable dense get entered the shard barrier %d times, want >= 1 "+
			"(the P<=1 path must route to the leader WITH rc) ⇒ STALE-SERVE HOLE)", n)
	}

	// (c) Linearizable get_config (named): the catalog read arms the barrier (>= 1).
	if err := s.VectorNamedCreateCollection(ctx, "named", map[string]rostam.NamedVectorParams{
		"title": {Dim: 3, Metric: vector.Cosine, M: 8, EfConstruction: 50, EfSearch: 32},
	}, 1); err != nil {
		t.Fatalf("VectorNamedCreateCollection named: %v", err)
	}
	delta()
	if _, err := s.VectorNamedGetConfigExt(ctx, "named", lin); err != nil {
		t.Fatalf("linearizable VectorNamedGetConfigExt: %v", err)
	}
	if n := delta(); n < 1 {
		t.Fatalf("linearizable named get_config entered the shard barrier %d times, want >= 1 "+
			"(0 ⇒ vector_named_get_config not in ReadConsistencyOf or rc dropped ⇒ STALE-SERVE HOLE)", n)
	}

	// (d) CONTRAST: AnyReplica gets on the SAME collections must NOT enter the
	// barrier (zero added consensus cost on the default path).
	if _, _, _, _, _, err := s.VectorGetExt(ctx, "parts", 7, true, true, any); err != nil {
		t.Fatalf("anyreplica VectorGetExt parts: %v", err)
	}
	if _, _, _, _, _, err := s.VectorGetExt(ctx, "single", 7, true, true, any); err != nil {
		t.Fatalf("anyreplica VectorGetExt single: %v", err)
	}
	if _, err := s.VectorNamedGetConfigExt(ctx, "named", any); err != nil {
		t.Fatalf("anyreplica VectorNamedGetConfigExt: %v", err)
	}
	// The plain (non-Ext) convenience forms must also be barrier-free (they delegate
	// with a zero ReadOpts).
	if _, _, _, _, _, err := s.VectorGet(ctx, "parts", 7, true, true); err != nil {
		t.Fatalf("VectorGet parts: %v", err)
	}
	if n := delta(); n != 0 {
		t.Fatalf("AnyReplica gets entered the shard barrier %d times, want 0 (default path must be barrier-free)", n)
	}
}
