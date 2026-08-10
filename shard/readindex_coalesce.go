// SPDX-License-Identifier: Apache-2.0

package shard

import (
	"sync"
	"time"
)

// readindexCoalescer coalesces concurrent Linearizable-read readIndex barriers on
// ONE Store into shared VerifyLeader+CommitIndex+Barrier round-trips, while
// PRESERVING the linearizability invariant that no reader is ever served a leader
// frontier captured BEFORE that reader arrived.
//
// THE CORRECTNESS RULE (why naive singleflight is UNSAFE here): a Linearizable
// read R must reflect every write committed before R started. A readIndex flight
// captures the leader CommitIndex ci at flight-start time tc; that ci includes
// everything committed before tc but NOT a write committed in (tc, R.arrival]. So a
// flight is SAFE for R IFF R.arrival <= tc. A naive singleflight.Group.Do lets a
// LATECOMER R2 (arriving at t2 > tc) join the in-flight call and receive ci
// captured at tc < t2 — R2 could MISS a write committed in (tc, t2) and serve
// stale. That is a silent linearizability violation.
//
// THE PATTERN — batch-then-capture (plan pattern (a)), realized as "batch while
// busy": at most ONE flight runs at a time; readers arriving WHILE a flight is in
// progress accumulate into a PENDING batch that becomes the NEXT flight.
//
//   - A reader registers into the PENDING batch under c.mu (its arrival is ordered
//     before any later mutation of the batch).
//   - If no flight is running, the registrant becomes the LEADER: under c.mu it marks
//     the flight running and DETACHES the pending batch (installs a fresh empty one),
//     THEN releases c.mu and runs the barrier. Detach-before-capture is the crux: the
//     set of readers in the batch is frozen BEFORE ci is captured, so every one of
//     them arrived at or before the capture.
//   - If a flight IS running, the registrant is a FOLLOWER of the pending batch: it
//     waits on that batch's result. The pending batch is run (one shared barrier) by
//     the current flight's leader once the current flight finishes (drain loop).
//   - Therefore for every reader R in any batch: R.arrival < that batch's detach <=
//     its ci capture, i.e. R.arrival <= tc. The captured frontier reflects every
//     write committed before R arrived. SAFE.
//   - A reader arriving while a flight runs is NEVER handed the in-flight,
//     pre-its-arrival frontier; it waits for the NEXT flight (captured strictly after
//     it arrived). That is exactly the coalescing window — a burst of arrivals during
//     one flight shares ONE follow-up barrier.
//
// THE GUARANTEE: every caller of do() receives the result of a flight whose ci was
// captured AT OR AFTER the caller arrived. No reader is ever served a pre-arrival
// frontier; coalescing only ever shares a barrier among readers all of whom arrived
// before that barrier's capture.
//
// Non-Linearizable reads never call this (Call gates on ConsistencyLinearizable), so
// AnyReplica/LeaderOnly pay zero coalescer cost. A lone Linearizable read with no
// flight running runs the barrier immediately — identical to the un-coalesced path.
type readindexCoalescer struct {
	mu      sync.Mutex
	running bool            // a flight is currently capturing/barriering
	pending *readindexBatch // the next batch accumulating arrivals (nil until an arrival)
}

// readindexBatch is one coalesced flight. done is closed by the flight leader once
// err holds the shared barrier result; followers block on done then read err.
// deadline is the MAX deadline of the readers in the batch (set under c.mu as each
// reader registers) so the shared barrier is given the most generous budget any
// member granted — a member with a shorter deadline still bails on its own wait, but
// the flight itself is never cut short below a waiting member's budget.
type readindexBatch struct {
	done     chan struct{}
	err      error
	deadline time.Time
	fn       func(time.Time) error // the barrier body; set by the batch's first registrant
}

// do runs (or joins) a coalesced readIndex barrier for one Linearizable read. fn is
// the barrier body (VerifyLeader+capture+Barrier). It is invoked at most once per
// flight, by a leader, AFTER the batch is detached — so every reader sharing an fn
// result arrived before fn captured its frontier. deadline bounds the leader's
// barrier and how long a follower waits for the shared result.
func (c *readindexCoalescer) do(deadline time.Time, fn func(time.Time) error) error {
	c.mu.Lock()
	if c.pending == nil {
		// First registrant of this batch fixes its barrier body. On a single Store fn is
		// always the SAME function (verifyLeaderAndCatchUpBody), so any member's fn is
		// equivalent; we use the first's so a batch is self-contained (the drain loop can
		// run it without referencing some other reader's closure).
		c.pending = &readindexBatch{done: make(chan struct{}), fn: fn}
	}
	b := c.pending
	if deadline.After(b.deadline) {
		b.deadline = deadline // most-generous budget among the batch's readers
	}

	if c.running {
		// A flight is already capturing; we accumulate into the pending batch (the NEXT
		// flight, captured strictly after we arrived). Wait for its shared result; its
		// barrier is run by the current flight's leader in the drain loop below.
		c.mu.Unlock()
		return c.wait(b, deadline)
	}

	// No flight running ⇒ we are the leader. Mark running and detach the batch BEFORE
	// releasing c.mu, so the batch membership is frozen before we capture ci.
	c.running = true
	c.pending = nil
	c.mu.Unlock()

	// Run our batch's barrier, then drain any batches that accumulated while we ran.
	// Each batch is a SEPARATE flight with its own capture (strictly after that
	// batch's readers arrived), so the arrival<=capture rule holds per batch.
	err := c.runBatch(b)
	c.drain()
	return err
}

// runBatch runs one flight's barrier (the batch's own fn with the batch's own
// most-generous deadline) and publishes the shared result to followers.
func (c *readindexCoalescer) runBatch(b *readindexBatch) error {
	err := b.fn(b.deadline)
	b.err = err
	close(b.done)
	return err
}

// drain runs queued pending batches after a flight completes: while readers have
// accumulated a pending batch, detach and run it (one barrier shared by all of its
// readers), repeating until none remain, then release the flight slot. Each detach
// happens strictly after every reader in that batch arrived (they arrived while the
// PRIOR flight was running). This is what coalesces a burst into few barriers. The
// drain runs on the FIRST leader's goroutine; followers only ever block on done.
func (c *readindexCoalescer) drain() {
	for {
		c.mu.Lock()
		if c.pending == nil {
			c.running = false // no queued readers; free the flight slot
			c.mu.Unlock()
			return
		}
		b := c.pending
		c.pending = nil // detach: freeze membership BEFORE this batch's capture
		// running stays true: we keep the single-flight slot while running this batch.
		c.mu.Unlock()

		_ = c.runBatch(b)
	}
}

// wait blocks a follower on its batch's result, bounded by its own deadline. The
// deadline elapsing only bounds the WAIT; it never serves stale (fail-loud timeout).
func (c *readindexCoalescer) wait(b *readindexBatch, deadline time.Time) error {
	select {
	case <-b.done:
		return b.err
	case <-time.After(time.Until(deadline)):
		return ErrLinearizableTimeout
	}
}
