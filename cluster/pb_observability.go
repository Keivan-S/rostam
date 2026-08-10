// SPDX-License-Identifier: Apache-2.0

package cluster

import "sync"

// ISR grow/shrink observability. Before this, a grow/shrink abort
// (cluster/pb_grow.go, cluster/pb_shrink.go) was silently discarded: no log
// line, no counter, nothing on any health surface. A shard could sit
// indefinitely under-replicated — e.g. wedged on pbisr.ErrGrowRingEvicted,
// which needs a full snapshot transfer to clear — with every surface
// reporting green. This file adds the shared plumbing both drivers use to
// make an abort visible.
//
// Snapshot transfer ships that snapshot transfer, so ring-cold and diverged aborts
// now self-heal. That does not make this plumbing less useful, it changes
// how to read it: the counters became a RATE signal (is the shard still
// failing to repair?) rather than a wedge detector. growAbortReason in
// cluster/pb_grow.go carries the per-bucket severity.

// pbAbortTracker rate-limits grow/shrink abort logging to the REASON
// TRANSITION (the first failure, or a change of reason) rather than every
// tick. Both drivers tick sub-second to few-seconds (pbShrinkTickInterval,
// pbGrowTickInterval), so a permanently-stuck shard would otherwise flood the
// log forever. Counts are cumulative per (shard, reason) regardless of
// whether a given tick's abort was the one that got logged, so the
// /v1/replication counters (cluster/repl_metrics.go) never undercount.
type pbAbortTracker struct {
	mu sync.Mutex
	// lastReason is keyed by (shard, target), NOT by shard alone.
	//
	// One tick calls record() once per grow TARGET, and a shard commonly has
	// more than one candidate outside its ISR. Keying on the shard alone made
	// two targets failing for DIFFERENT reasons overwrite each other's state
	// every call, so each one always observed a "changed" reason and every
	// abort logged: 100 lines for 100 aborts, ~2 WARN/s/shard at the default
	// tick, unbounded and growing with the number of distinct reasons.
	//
	// Per-target keying is what the suppression actually means: this target
	// keeps failing the same way, say it once.
	lastReason map[pbAbortKey]string
	// counts stay shard-scoped — an operator reads them per shard, and the
	// per-target detail is in the log line.
	counts map[int]map[string]uint64
}

// pbAbortKey identifies the (shard, target) pair a suppression decision is
// about. The empty target is used by drivers whose aborts are shard-scoped
// rather than per-candidate (the shrink submit path).
type pbAbortKey struct {
	shard  int
	target string
}

func newPBAbortTracker() *pbAbortTracker {
	return &pbAbortTracker{
		lastReason: make(map[pbAbortKey]string),
		counts:     make(map[int]map[string]uint64),
	}
}

// record notes one abort for (shardID, target) at reason and returns true
// exactly when the caller should log it — the first time this reason is seen
// for that TARGET since its last clear (a prior success, or a different
// reason for the same target). The count is always incremented.
//
// Pass an empty target for a shard-scoped abort that is not about a particular
// candidate.
func (t *pbAbortTracker) record(shardID int, target, reason string) bool {
	if t == nil {
		return false
	}
	t.mu.Lock()
	defer t.mu.Unlock()
	if t.counts[shardID] == nil {
		t.counts[shardID] = make(map[string]uint64)
	}
	t.counts[shardID][reason]++
	k := pbAbortKey{shard: shardID, target: target}
	shouldLog := t.lastReason[k] != reason
	t.lastReason[k] = reason
	return shouldLog
}

// clear resets the logged-reason state for every target of shardID (called
// wherever a driver observes a tick with no abort for that shard), so a LATER
// recurrence of the same reason is treated as a new transition and logged
// again instead of staying silent forever. It does not touch the cumulative
// counts.
//
// Clearing the whole shard is deliberate: the call sites fire when the shard
// as a whole had nothing to do or succeeded, which invalidates every target's
// suppression state, not just one candidate's.
func (t *pbAbortTracker) clear(shardID int) {
	if t == nil {
		return
	}
	t.mu.Lock()
	defer t.mu.Unlock()
	for k := range t.lastReason {
		if k.shard == shardID {
			delete(t.lastReason, k)
		}
	}
}

// snapshot returns a copy of shardID's cumulative abort counts by reason, or
// nil if none have been recorded. Safe to call from any goroutine (used by
// the /v1/replication handler, which runs on the request-handling goroutine,
// not the driver's own tick goroutine).
func (t *pbAbortTracker) snapshot(shardID int) map[string]uint64 {
	if t == nil {
		return nil
	}
	t.mu.Lock()
	defer t.mu.Unlock()
	c := t.counts[shardID]
	if len(c) == 0 {
		return nil
	}
	out := make(map[string]uint64, len(c))
	for k, v := range c {
		out[k] = v
	}
	return out
}
