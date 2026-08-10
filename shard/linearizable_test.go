// SPDX-License-Identifier: Apache-2.0

package shard

import (
	"bytes"
	"errors"
	"sync/atomic"
	"testing"
	"time"

	"github.com/rostamlabs/rostam/ops"
	"github.com/rostamlabs/rostam/vector"
)

// newSingleNodeVectorStore boots a single-node leader with a 3-dim "docs"
// collection populated with a handful of points, ready for search.
func newSingleNodeVectorStore(t *testing.T) *Store {
	t.Helper()
	s := newSingleNodeStore(t)
	colCfg := vector.Config{Dim: 3, Metric: vector.L2, M: 8, EfConstruction: 50, EfSearch: 32, Seed: 1}
	if _, err := s.Call("vector_create_collection", ops.EncodeCreateCollectionArgs("docs", colCfg)); err != nil {
		t.Fatalf("create collection: %v", err)
	}
	for i := 1; i <= 8; i++ {
		args := ops.EncodeVectorUpsertArgs("docs", uint64(i), []float32{float32(i), 0, 0}, "chunk", 0, nil, vector.SparseVector{})
		if _, err := s.Call("vector_upsert", args); err != nil {
			t.Fatalf("upsert %d: %v", i, err)
		}
	}
	return s
}

// TestVerifyLeaderAndCatchUpSingleNodeLeader: on a single-node leader the barrier
// (VerifyLeader immediate, catch-up trivially satisfied) returns nil promptly.
func TestVerifyLeaderAndCatchUpSingleNodeLeader(t *testing.T) {
	s := newSingleNodeStore(t)
	done := make(chan error, 1)
	go func() { done <- s.verifyLeaderAndCatchUp(time.Now().Add(linearizableReadTimeout)) }()
	select {
	case err := <-done:
		if err != nil {
			t.Fatalf("verifyLeaderAndCatchUp on single-node leader = %v, want nil", err)
		}
	case <-time.After(3 * time.Second):
		t.Fatal("verifyLeaderAndCatchUp hung on a single-node leader")
	}
}

// TestVerifyLeaderAndCatchUpAfterWrites: after committing writes the barrier still
// resolves (the FSM applied index has reached the committed index).
func TestVerifyLeaderAndCatchUpAfterWrites(t *testing.T) {
	s := newSingleNodeStore(t)
	for i := 0; i < 5; i++ {
		if err := s.Put([]byte{byte('a' + i)}, []byte("v"), 0); err != nil {
			t.Fatalf("Put: %v", err)
		}
	}
	if err := s.verifyLeaderAndCatchUp(time.Now().Add(linearizableReadTimeout)); err != nil {
		t.Fatalf("verifyLeaderAndCatchUp after writes = %v, want nil", err)
	}
}

// TestCallLinearizableSearchServes: a Linearizable-encoded search on a single-node
// leader passes the barrier and returns the correct results (byte-identical to a
// default search on the same data).
func TestCallLinearizableSearchServes(t *testing.T) {
	s := newSingleNodeVectorStore(t)
	query := []float32{1, 0, 0}

	linArgs := ops.EncodeVectorSearchArgsOpts("docs", 5, query, vector.Filter{}, ops.ConsistencyLinearizable, 0, 0)
	res, err := s.Call("vector_search", linArgs)
	if err != nil {
		t.Fatalf("linearizable search: %v", err)
	}
	linHits, err := ops.DecodeVectorSearchResults(res)
	if err != nil {
		t.Fatalf("decode linearizable hits: %v", err)
	}
	if len(linHits) != 5 {
		t.Fatalf("linearizable search returned %d hits, want 5", len(linHits))
	}

	// Same query at AnyReplica must return the same set/order (the barrier is a
	// no-op on data correctness, it only adds the freshness guarantee).
	anyArgs := ops.EncodeVectorSearchArgs("docs", 5, query)
	res2, err := s.Call("vector_search", anyArgs)
	if err != nil {
		t.Fatalf("anyreplica search: %v", err)
	}
	anyHits, err := ops.DecodeVectorSearchResults(res2)
	if err != nil {
		t.Fatalf("decode anyreplica hits: %v", err)
	}
	if len(anyHits) != len(linHits) {
		t.Fatalf("hit count mismatch: lin=%d any=%d", len(linHits), len(anyHits))
	}
	for i := range linHits {
		if linHits[i].ID != anyHits[i].ID {
			t.Fatalf("hit[%d] id mismatch: lin=%d any=%d", i, linHits[i].ID, anyHits[i].ID)
		}
	}
}

// TestCallDefaultPathZeroBarrierCost is the perf-correctness proof: AnyReplica and
// LeaderOnly reads NEVER enter the readIndex barrier (no VerifyLeader cost); only a
// Linearizable read does. Observed via SetBarrierEnteredHook.
func TestCallDefaultPathZeroBarrierCost(t *testing.T) {
	s := newSingleNodeVectorStore(t)
	query := []float32{1, 0, 0}

	var barrierEntries atomic.Int64
	SetBarrierEnteredHook(func() { barrierEntries.Add(1) })
	t.Cleanup(func() { SetBarrierEnteredHook(nil) })

	// AnyReplica (default, legacy encoding) ⇒ no barrier.
	if _, err := s.Call("vector_search", ops.EncodeVectorSearchArgs("docs", 5, query)); err != nil {
		t.Fatalf("anyreplica search: %v", err)
	}
	// AnyReplica via the opts encoder (rc=0) ⇒ no barrier.
	if _, err := s.Call("vector_search", ops.EncodeVectorSearchArgsOpts("docs", 5, query, vector.Filter{}, ops.ConsistencyAnyReplica, 0, 0)); err != nil {
		t.Fatalf("anyreplica(opts) search: %v", err)
	}
	// LeaderOnly ⇒ no barrier.
	if _, err := s.Call("vector_search", ops.EncodeVectorSearchArgsOpts("docs", 5, query, vector.Filter{}, ops.ConsistencyLeaderOnly, 0, 0)); err != nil {
		t.Fatalf("leaderonly search: %v", err)
	}
	if n := barrierEntries.Load(); n != 0 {
		t.Fatalf("default-path reads entered the barrier %d times, want 0", n)
	}

	// Now a Linearizable read MUST enter the barrier exactly once.
	if _, err := s.Call("vector_search", ops.EncodeVectorSearchArgsOpts("docs", 5, query, vector.Filter{}, ops.ConsistencyLinearizable, 0, 0)); err != nil {
		t.Fatalf("linearizable search: %v", err)
	}
	if n := barrierEntries.Load(); n != 1 {
		t.Fatalf("linearizable read entered the barrier %d times, want 1", n)
	}
}

// TestCallLinearizableGetArmsBarrier is the get-family analogue of
// TestCallDefaultPathZeroBarrierCost: a Linearizable point-get MUST enter the
// readIndex barrier exactly once (read-your-writes), while AnyReplica / LeaderOnly
// gets NEVER enter it (zero added cost). It also covers vector_get_config. This is
// the anti-silent-drop teeth: if vector_get / vector_get_config were removed from
// ops.ReadConsistencyOf (or the rc dropped from the encoder), the barrier-entry
// count for the Linearizable get would drop to 0 and this test goes RED.
func TestCallLinearizableGetArmsBarrier(t *testing.T) {
	s := newSingleNodeVectorStore(t)

	var barrierEntries atomic.Int64
	SetBarrierEnteredHook(func() { barrierEntries.Add(1) })
	t.Cleanup(func() { SetBarrierEnteredHook(nil) })

	getArgs := func(rc uint8) []byte {
		return ops.EncodeVectorGetArgsOpts("docs", 3, ops.GetFlagWithVector|ops.GetFlagWithPayload, rc, 0, 0)
	}

	// AnyReplica (legacy encoding) ⇒ no barrier.
	if _, err := s.Call("vector_get", ops.EncodeVectorGetArgs("docs", 3, ops.GetFlagWithVector)); err != nil {
		t.Fatalf("anyreplica get (legacy): %v", err)
	}
	// AnyReplica via the opts encoder (rc=0) ⇒ no barrier.
	if _, err := s.Call("vector_get", getArgs(ops.ConsistencyAnyReplica)); err != nil {
		t.Fatalf("anyreplica get (opts): %v", err)
	}
	// LeaderOnly ⇒ no barrier.
	if _, err := s.Call("vector_get", getArgs(ops.ConsistencyLeaderOnly)); err != nil {
		t.Fatalf("leaderonly get: %v", err)
	}
	if n := barrierEntries.Load(); n != 0 {
		t.Fatalf("default-path gets entered the barrier %d times, want 0", n)
	}

	// A Linearizable get MUST enter the barrier exactly once.
	if _, err := s.Call("vector_get", getArgs(ops.ConsistencyLinearizable)); err != nil {
		t.Fatalf("linearizable get: %v", err)
	}
	if n := barrierEntries.Load(); n != 1 {
		t.Fatalf("linearizable get entered the barrier %d times, want 1 "+
			"(anti-silent-drop: is vector_get in ops.ReadConsistencyOf?)", n)
	}

	// A Linearizable get_config also arms the barrier (the catalog read).
	if _, err := s.Call("vector_get_config", ops.EncodeGetConfigArgsOpts("docs", ops.ConsistencyLinearizable, 0, 0)); err != nil {
		t.Fatalf("linearizable get_config: %v", err)
	}
	if n := barrierEntries.Load(); n != 2 {
		t.Fatalf("linearizable get_config entered the barrier total=%d, want 2 "+
			"(anti-silent-drop: is vector_get_config in ops.ReadConsistencyOf?)", n)
	}
}

// TestLinearizableGetServesByteIdentical proves a Linearizable get returns the
// SAME point as an AnyReplica get on the same data — the barrier adds the
// freshness guarantee, never changes the served bytes.
func TestLinearizableGetServesByteIdentical(t *testing.T) {
	s := newSingleNodeVectorStore(t)
	flags := ops.GetFlagWithVector | ops.GetFlagWithPayload

	anyBody, err := s.Call("vector_get", ops.EncodeVectorGetArgs("docs", 5, flags))
	if err != nil {
		t.Fatalf("anyreplica get: %v", err)
	}
	linBody, err := s.Call("vector_get", ops.EncodeVectorGetArgsOpts("docs", 5, flags, ops.ConsistencyLinearizable, 0, 0))
	if err != nil {
		t.Fatalf("linearizable get: %v", err)
	}
	if !bytes.Equal(anyBody, linBody) {
		t.Fatalf("linearizable get served different bytes than anyreplica:\n any=%v\n lin=%v", anyBody, linBody)
	}
}

// TestLinearizableTimeoutError documents the typed timeout sentinel exists and is
// distinct from NotLeaderError (fail-loud, never a silent stale serve).
func TestLinearizableTimeoutError(t *testing.T) {
	if errors.Is(ErrLinearizableTimeout, ErrNotLeader) {
		t.Fatal("ErrLinearizableTimeout must not alias ErrNotLeader")
	}
}

// TestVerifyLeaderAndCatchUpWaitsForSlowInflightApply is the linearizability
// regression for the readIndex barrier: a COMMAND committed before the read but
// still MID-APPLY (a slow Apply) must be applied before the barrier returns. The
// earlier "stability fixpoint" slow-path could serve while such a command was
// still applying (its index <= ci, FSM frontier held constant during the Apply,
// raft already past ci) — a stale serve. The Barrier slow path waits correctly.
func TestVerifyLeaderAndCatchUpWaitsForSlowInflightApply(t *testing.T) {
	reg := ops.NewRegistry()
	if err := ops.RegisterBuiltins(reg); err != nil {
		t.Fatal(err)
	}
	const applyDelay = 200 * time.Millisecond
	var applied atomic.Bool
	// A write op (goes through Raft) whose Apply is deliberately slow.
	if err := reg.Register("slow_noop", ops.OpReadWrite, func(_ *ops.TxContext, _ []byte) ([]byte, error) {
		time.Sleep(applyDelay)
		applied.Store(true)
		return nil, nil
	}); err != nil {
		t.Fatal(err)
	}
	cfg := DefaultConfig(t.TempDir(), "node1", reg)
	cfg.Bootstrap = true
	cfg.RaftHeartbeatMs = 50
	cfg.RaftElectionMs = 100
	cfg.NoSync = true
	s, err := New(cfg)
	if err != nil {
		t.Fatalf("shard.New: %v", err)
	}
	t.Cleanup(func() { _ = s.Close() })
	waitLeader(t, s)

	preCommit := s.raft.CommitIndex()

	// Submit the slow write async: it COMMITS quickly (quorum=1) but its FSM Apply
	// blocks for applyDelay, so it is committed-but-not-yet-applied during the window.
	errc := make(chan error, 1)
	go func() { _, e := s.Call("slow_noop", nil); errc <- e }()

	// Wait until slow_noop has COMMITTED (CommitIndex advanced past its pre-submit
	// value) while its Apply has NOT finished (applied still false) — the
	// committed-but-mid-Apply window the bug lived in. (Note: fsm.AppliedIndex <
	// CommitIndex is ALWAYS true on an idle leader because of the election no-op gap,
	// so we must gate on CommitIndex ADVANCING, not on the fsm/commit delta.)
	deadline := time.Now().Add(3 * time.Second)
	for s.raft.CommitIndex() <= preCommit || applied.Load() {
		if time.Now().After(deadline) {
			t.Fatal("slow_noop never reached the committed-but-mid-Apply window")
		}
		time.Sleep(2 * time.Millisecond)
	}

	// The barrier MUST NOT return before the in-flight committed command applies.
	if err := s.verifyLeaderAndCatchUp(time.Now().Add(linearizableReadTimeout)); err != nil {
		t.Fatalf("verifyLeaderAndCatchUp: %v", err)
	}
	if !applied.Load() {
		t.Fatal("barrier returned BEFORE the committed in-flight command applied — linearizability violation")
	}
	if e := <-errc; e != nil {
		t.Fatalf("slow_noop apply: %v", e)
	}
}
