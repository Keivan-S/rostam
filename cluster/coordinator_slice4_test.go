// SPDX-License-Identifier: Apache-2.0

package cluster

import (
	"context"
	"errors"
	"testing"
	"time"
)

// TestSlice4ResumeAndCancel exercises the coordinator's resumability and
// cancellation (plus the progress metrics). It runs half a rebalance plan, then
// re-runs Rebalance and asserts it recomputes only the remaining diff and
// converges — the design's "placement is the state, resume by recompute" claim.
// It also asserts a pre-cancelled context dispatches no moves.
func TestSlice4ResumeAndCancel(t *testing.T) {
	const numShards = 6
	tc := newTestCluster(t, 3, numShards, 1)
	nodes := map[string]*Node{"n1": tc.nodes[0], "n2": tc.nodes[1], "n3": tc.nodes[2]}
	mc := MigrationClusterFromPeers(nodes, tc.peers)
	coord := &Coordinator{MC: mc, MaxParallel: 1, StepTimeout: 20 * time.Second}
	ctx := context.Background()

	// --- Cancel: a pre-cancelled context dispatches nothing. ---
	cancelled, cancel := context.WithCancel(ctx)
	cancel()
	plan0, err := coord.Rebalance(cancelled, tc.peers, numShards, 2)
	if !errors.Is(err, context.Canceled) {
		t.Fatalf("cancelled rebalance: err=%v, want context.Canceled", err)
	}
	if len(plan0.Moves) != numShards {
		t.Fatalf("cancelled rebalance still planned %d moves, want %d", len(plan0.Moves), numShards)
	}
	if s := coord.Stats(); s.Done != 0 {
		t.Fatalf("cancelled rebalance did %d moves, want 0 (stats=%+v)", s.Done, s)
	}

	// --- Partial: execute only the first half of the RF1->2 plan. ---
	current := mc.currentPlacement(numShards)
	full := PlanRebalance(current, tc.peers, numShards, 2)
	if len(full.Moves) != numShards {
		t.Fatalf("RF1->2 full plan: %d moves, want %d", len(full.Moves), numShards)
	}
	half := len(full.Moves) / 2
	if err := coord.Execute(ctx, RebalancePlan{Moves: full.Moves[:half]}); err != nil {
		t.Fatalf("partial execute: %v", err)
	}
	if s := coord.Stats(); s.Done != half || s.Failed != 0 {
		t.Fatalf("partial execute stats=%+v, want Done=%d Failed=0", s, half)
	}

	// --- Resume: re-running recomputes ONLY the remaining diff and converges. ---
	plan2, err := coord.Rebalance(ctx, tc.peers, numShards, 2)
	if err != nil {
		t.Fatalf("resume rebalance: %v", err)
	}
	if len(plan2.Moves) != numShards-half {
		t.Fatalf("resume recomputed %d moves, want %d (the remainder)", len(plan2.Moves), numShards-half)
	}
	assertPlacement(t, mc, computePlacement(tc.peers, numShards, 2))
	if s := coord.Stats(); s.Done != numShards-half {
		t.Fatalf("resume stats=%+v, want Done=%d", s, numShards-half)
	}

	// --- Converged: a third run is a no-op. ---
	plan3, err := coord.Rebalance(ctx, tc.peers, numShards, 2)
	if err != nil || len(plan3.Moves) != 0 {
		t.Fatalf("converged re-run: moves=%d err=%v, want 0/nil", len(plan3.Moves), err)
	}
}
