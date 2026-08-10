// SPDX-License-Identifier: Apache-2.0

package cluster

import (
	"context"
	"errors"
	"fmt"
	"sort"
	"sync"
	"sync/atomic"
	"time"
)

// Online-rebalancing slice 3: the coordinator. On a membership change (or an RF
// change) it computes the target placement, diffs it against the current
// placement, and drives one MigrateShard per changed shard with bounded
// concurrency. Placement advances shard-by-shard (each MigrateShard commits its
// own shard), so the operation is idempotent and resumable: re-running
// recomputes the remaining diff from live placement and continues.

// ShardMove is one shard's owner-set change in a rebalance plan.
type ShardMove struct {
	ShardID int
	From    []string
	To      []string
}

// RebalancePlan is the set of shards whose owner set must change to reach the
// target placement.
type RebalancePlan struct {
	Moves []ShardMove
}

// PlanRebalance computes the target placement for (members, numShards, rf) and
// diffs it against current, returning the shards that must move. A shard with an
// unchanged owner set (order-independent) is omitted. The plan is a pure
// function of (current, members, numShards, rf).
func PlanRebalance(current [][]string, members []Peer, numShards, rf int) RebalancePlan {
	target := computePlacement(members, numShards, rf)
	var moves []ShardMove
	for s := 0; s < numShards; s++ {
		var cur []string
		if s < len(current) {
			cur = current[s]
		}
		if sameSet(cur, target[s]) {
			continue
		}
		moves = append(moves, ShardMove{ShardID: s, From: cur, To: target[s]})
	}
	return RebalancePlan{Moves: moves}
}

// Coordinator drives a rebalance to completion over a cluster view (in-process
// *Node handles or network-backed remoteNodes). The operator surface
// (__rebalance__ op / rostam-server reconfigure) triggers it on the meta leader;
// it is also callable directly via the Go API. Methods take a pointer receiver
// so progress counters accumulate on the caller's value.
type Coordinator struct {
	MC MigrationCluster
	// MaxParallel bounds how many shards migrate concurrently (<=0 → 1). This is
	// the throttle that keeps rebalancing from saturating the cluster; a
	// per-stream byte/sec cap on the Raft snapshot transfer is intentionally out
	// of scope (it would mean wrapping hashicorp/raft's transport internals).
	MaxParallel int
	// StepTimeout bounds each single-shard migration (passed to MigrateShard).
	StepTimeout time.Duration

	// progress counters (live during Execute; read via Stats).
	total, pending, inflight, done, failed atomic.Int64
}

// RebalanceStats is a point-in-time snapshot of a rebalance's progress —
// exported for metrics (rebalance_shards_pending and friends).
type RebalanceStats struct {
	Total    int // shards the current/last plan must move
	Pending  int // not yet started
	InFlight int // migrating now
	Done     int // completed successfully
	Failed   int // completed with error
}

// Stats returns the current progress counters.
func (c *Coordinator) Stats() RebalanceStats {
	return RebalanceStats{
		Total:    int(c.total.Load()),
		Pending:  int(c.pending.Load()),
		InFlight: int(c.inflight.Load()),
		Done:     int(c.done.Load()),
		Failed:   int(c.failed.Load()),
	}
}

// Rebalance moves the cluster to the placement implied by (members, numShards,
// rf): it reads the current placement from a live node, plans the diff, and
// executes every move with bounded concurrency. Returns the joined errors of any
// failed shard migrations; shards that succeeded stay committed. Idempotent and
// resumable: a re-run recomputes the remaining diff from live placement and
// continues (a no-op once converged). ctx cancellation stops dispatching new
// moves (in-flight moves run to their StepTimeout).
func (c *Coordinator) Rebalance(ctx context.Context, members []Peer, numShards, rf int) (RebalancePlan, error) {
	current := c.MC.currentPlacement(numShards)
	plan := PlanRebalance(current, members, numShards, rf)
	return plan, c.Execute(ctx, plan)
}

// Execute runs the plan's moves with bounded concurrency, updating progress
// counters as it goes. New moves stop being dispatched once ctx is cancelled.
func (c *Coordinator) Execute(ctx context.Context, plan RebalancePlan) error {
	c.total.Store(int64(len(plan.Moves)))
	c.pending.Store(int64(len(plan.Moves)))
	c.inflight.Store(0)
	c.done.Store(0)
	c.failed.Store(0)
	if len(plan.Moves) == 0 {
		return nil
	}
	timeout := c.StepTimeout
	if timeout <= 0 {
		timeout = 30 * time.Second
	}
	parallel := c.MaxParallel
	if parallel <= 0 {
		parallel = 1
	}

	sem := make(chan struct{}, parallel)
	var (
		wg   sync.WaitGroup
		mu   sync.Mutex
		errs []error
	)
	for _, mv := range plan.Moves {
		if err := ctx.Err(); err != nil {
			mu.Lock()
			errs = append(errs, err)
			mu.Unlock()
			break // cancelled: stop launching new migrations
		}
		sem <- struct{}{}
		wg.Add(1)
		go func(mv ShardMove) {
			defer wg.Done()
			defer func() { <-sem }()
			c.pending.Add(-1)
			c.inflight.Add(1)
			defer c.inflight.Add(-1)
			if err := c.MC.MigrateShard(mv.ShardID, mv.To, timeout); err != nil {
				c.failed.Add(1)
				mu.Lock()
				errs = append(errs, fmt.Errorf("shard %d %v->%v: %w", mv.ShardID, mv.From, mv.To, err))
				mu.Unlock()
				return
			}
			c.done.Add(1)
		}(mv)
	}
	wg.Wait()
	return errors.Join(errs...)
}

// currentPlacement returns the cluster's current placement (first numShards
// entries) from a live node's routing view (local-preferred); all members hold
// the same replicated copy.
func (mc MigrationCluster) currentPlacement(numShards int) [][]string {
	for _, nd := range mc.orderedNodes() {
		full := nd.placementCopy()
		if len(full) >= numShards {
			return full[:numShards]
		}
		if len(full) > 0 {
			return full
		}
	}
	return nil
}

// sameSet reports whether a and b contain the same elements, ignoring order and
// duplicates.
func sameSet(a, b []string) bool {
	as := append([]string(nil), a...)
	bs := append([]string(nil), b...)
	sort.Strings(as)
	sort.Strings(bs)
	// Collapse adjacent duplicates so [n1,n1] == [n1].
	as = dedupSorted(as)
	bs = dedupSorted(bs)
	if len(as) != len(bs) {
		return false
	}
	for i := range as {
		if as[i] != bs[i] {
			return false
		}
	}
	return true
}

// dedupSorted removes adjacent duplicates from a sorted slice.
func dedupSorted(s []string) []string {
	if len(s) < 2 {
		return s
	}
	out := s[:1]
	for _, x := range s[1:] {
		if x != out[len(out)-1] {
			out = append(out, x)
		}
	}
	return out
}
