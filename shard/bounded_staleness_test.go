// SPDX-License-Identifier: Apache-2.0

package shard

import (
	"errors"
	"sync/atomic"
	"testing"
	"time"

	"github.com/rostamlabs/rostam/ops"
)

// boundedGetArgs builds a bounded-staleness vector_get carrying the given staleness
// bound on the wire (the real bound is threaded through EncodeVectorGetArgsOpts).
// The within-bound cases use lag 0 (serve) and the out-of-bound case uses a frontier
// far beyond applied+bound (route to leader).
func boundedGetArgs(id uint64, bound uint64) []byte {
	return ops.EncodeVectorGetArgsOpts("docs", id, 0, ops.ConsistencyBoundedStaleness, 0, bound)
}

// TestBoundedStalenessLeaderWithinBoundServesNoBarrier: on a single-node leader the
// lag (CommitIndex - AppliedIndex) is 0, so any bound>=0 serves LOCALLY without the
// readIndex barrier. The served hook fires true; the barrier hook never fires.
func TestBoundedStalenessLeaderWithinBoundServesNoBarrier(t *testing.T) {
	s := newSingleNodeVectorStore(t)

	var barriers atomic.Int64
	SetBarrierEnteredHook(func() { barriers.Add(1) })
	t.Cleanup(func() { SetBarrierEnteredHook(nil) })

	var served atomic.Int64
	var lastServedLocal atomic.Bool
	SetBoundedStalenessServedHook(func(local bool) {
		served.Add(1)
		lastServedLocal.Store(local)
	})
	t.Cleanup(func() { SetBoundedStalenessServedHook(nil) })

	if _, err := s.Call("vector_get", boundedGetArgs(3, 5)); err != nil {
		t.Fatalf("bounded get on leader: %v", err)
	}
	if served.Load() != 1 || !lastServedLocal.Load() {
		t.Fatalf("served=%d local=%v, want 1 local serve", served.Load(), lastServedLocal.Load())
	}
	if barriers.Load() != 0 {
		t.Fatalf("barrier fired %d times on a within-bound leader, want 0", barriers.Load())
	}
}

// TestBoundedStalenessFollowerWithinBoundServesLocal: a follower with an injected
// leaderFrontierFn that reports a frontier within bound of its applied index serves
// LOCALLY (hook=true), with NO barrier.
func TestBoundedStalenessFollowerWithinBoundServesLocal(t *testing.T) {
	cluster := newVectorCluster(t, 3)
	var follower *Store
	for _, s := range cluster {
		if !s.IsLeader() {
			follower = s
			break
		}
	}
	if follower == nil {
		t.Fatal("no follower in 3-node cluster")
	}

	var barriers atomic.Int64
	SetBarrierEnteredHook(func() { barriers.Add(1) })
	t.Cleanup(func() { SetBarrierEnteredHook(nil) })

	var servedLocal atomic.Bool
	SetBoundedStalenessServedHook(func(local bool) { servedLocal.Store(local) })
	t.Cleanup(func() { SetBoundedStalenessServedHook(nil) })

	// Frontier == this follower's applied index ⇒ lag 0 ⇒ within ANY bound.
	follower.SetLeaderFrontierFn(func(time.Time) (uint64, error) {
		return follower.fsm.AppliedIndex(), nil
	})

	if _, err := follower.Call("vector_get", boundedGetArgs(3, 5)); err != nil {
		t.Fatalf("bounded get within bound on follower: %v", err)
	}
	if !servedLocal.Load() {
		t.Fatal("served hook did not report a local serve")
	}
	if barriers.Load() != 0 {
		t.Fatalf("barrier fired %d times on a within-bound follower, want 0", barriers.Load())
	}
}

// TestBoundedStalenessFollowerOutOfBoundRoutesToLeader: a follower whose injected
// frontier is FAR beyond applied+bound returns NotLeaderError (route to leader), and
// the served hook reports a non-local (upgrade) decision.
func TestBoundedStalenessFollowerOutOfBoundRoutesToLeader(t *testing.T) {
	cluster := newVectorCluster(t, 3)
	var follower *Store
	for _, s := range cluster {
		if !s.IsLeader() {
			follower = s
			break
		}
	}
	if follower == nil {
		t.Fatal("no follower in 3-node cluster")
	}

	var servedLocal atomic.Bool
	servedLocal.Store(true)
	SetBoundedStalenessServedHook(func(local bool) { servedLocal.Store(local) })
	t.Cleanup(func() { SetBoundedStalenessServedHook(nil) })

	// Frontier far beyond applied+bound ⇒ too stale ⇒ route to leader.
	follower.SetLeaderFrontierFn(func(time.Time) (uint64, error) {
		return follower.fsm.AppliedIndex() + 1_000_000, nil
	})

	_, err := follower.Call("vector_get", boundedGetArgs(3, 1))
	var nle *NotLeaderError
	if !errors.As(err, &nle) {
		t.Fatalf("out-of-bound follower err = %v, want *NotLeaderError", err)
	}
	if servedLocal.Load() {
		t.Fatal("served hook reported a local serve for an out-of-bound read")
	}
}

// TestBoundedStalenessFollowerFrontierErrorFailsClosed: when the injected
// leaderFrontierFn errors (an unreachable / partitioned leader), the read fails
// CLOSED — NotLeaderError to route to the leader, never a possibly-stale local serve.
func TestBoundedStalenessFollowerFrontierErrorFailsClosed(t *testing.T) {
	cluster := newVectorCluster(t, 3)
	var follower *Store
	for _, s := range cluster {
		if !s.IsLeader() {
			follower = s
			break
		}
	}
	if follower == nil {
		t.Fatal("no follower in 3-node cluster")
	}

	var servedLocal atomic.Bool
	servedLocal.Store(true)
	SetBoundedStalenessServedHook(func(local bool) { servedLocal.Store(local) })
	t.Cleanup(func() { SetBoundedStalenessServedHook(nil) })

	follower.SetLeaderFrontierFn(func(time.Time) (uint64, error) {
		return 0, errors.New("leader unreachable (partition)")
	})

	_, err := follower.Call("vector_get", boundedGetArgs(3, 5))
	var nle *NotLeaderError
	if !errors.As(err, &nle) {
		t.Fatalf("frontier-error follower err = %v, want *NotLeaderError (fail closed)", err)
	}
	if servedLocal.Load() {
		t.Fatal("served hook reported a local serve despite a frontier error")
	}
}

// TestBoundedStalenessFollowerNoFnUpgradesToBarrier: a follower with NO injected
// leaderFrontierFn falls back to the Linearizable barrier; VerifyLeader on a follower
// yields NotLeaderError (route to leader) and the barrier hook fires (upgrade).
func TestBoundedStalenessFollowerNoFnUpgradesToBarrier(t *testing.T) {
	cluster := newVectorCluster(t, 3)
	var follower *Store
	for _, s := range cluster {
		if !s.IsLeader() {
			follower = s
			break
		}
	}
	if follower == nil {
		t.Fatal("no follower in 3-node cluster")
	}
	// Ensure no fn is wired (inMemCluster does not set one).
	follower.SetLeaderFrontierFn(nil)

	var barriers atomic.Int64
	SetBarrierEnteredHook(func() { barriers.Add(1) })
	t.Cleanup(func() { SetBarrierEnteredHook(nil) })

	_, err := follower.Call("vector_get", boundedGetArgs(3, 5))
	var nle *NotLeaderError
	if !errors.As(err, &nle) {
		t.Fatalf("no-fn follower err = %v, want *NotLeaderError (barrier upgrade)", err)
	}
	if barriers.Load() == 0 {
		t.Fatal("barrier hook did not fire — no-fn follower must upgrade to the barrier")
	}
}
