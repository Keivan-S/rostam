// SPDX-License-Identifier: Apache-2.0

package cluster

import (
	"sync"
	"time"
)

// metaFrontierCoalescer coalesces concurrent follower __meta_readindex__ forwards
// on ONE Node into shared leader RTTs, while PRESERVING the linearizability
// invariant that no Linearizable read accepts a leader frontier captured BEFORE the
// read arrived. It mirrors shard.readindexCoalescer (batch-while-busy); see that
// type for the full proof. The same arrival<=capture rule applies: a forward
// captures the leader's FSM command frontier f at the leader's verify/capture time
// tc, and f is SAFE for a read R iff R.arrival <= tc.
//
// THE PATTERN — batch-while-busy: at most ONE forward runs at a time; followers
// arriving WHILE a forward is in progress accumulate into a PENDING batch that
// becomes the NEXT forward (captured strictly after they arrived). The flight leader
// detaches its batch (freezing membership) BEFORE forwarding/capturing, so every
// batched reader arrived at or before the capture. A reader arriving during an
// in-flight forward is NEVER handed that forward's (pre-its-arrival) frontier.
//
// THE GUARANTEE: every caller of do() receives a frontier captured at or after the
// caller arrived. No read is served a pre-arrival catalog frontier.
//
// This sits in front of fetchMetaLeaderFrontier, which the follower poll loop calls
// once per tick. A failed forward fails its whole batch; each caller then retries on
// its own next tick (a fresh batch / flight) — identical retry semantics to the
// un-coalesced loop, just sharing the in-flight attempt.
type metaFrontierCoalescer struct {
	mu      sync.Mutex
	running bool
	pending *metaFrontierBatch
}

// metaFrontierBatch is one coalesced forward. done is closed by the flight leader
// once (frontier, err) hold the shared result; followers block on done then read
// them. deadline is the MAX deadline among the batch's readers; fn is fixed by the
// batch's first registrant (always the same fetchMetaLeaderFrontier on one Node).
type metaFrontierBatch struct {
	done     chan struct{}
	frontier uint64
	err      error
	deadline time.Time
	fn       func(time.Time) (uint64, error)
}

// do runs (or joins) a coalesced __meta_readindex__ forward. fn is
// fetchMetaLeaderFrontier; it runs at most once per flight, by a leader, AFTER the
// batch is detached — so every reader sharing fn's result arrived before fn captured
// the leader frontier. A follower's deadline bounds only how long it waits.
func (c *metaFrontierCoalescer) do(deadline time.Time, fn func(time.Time) (uint64, error)) (uint64, error) {
	c.mu.Lock()
	if c.pending == nil {
		c.pending = &metaFrontierBatch{done: make(chan struct{}), fn: fn}
	}
	b := c.pending
	if deadline.After(b.deadline) {
		b.deadline = deadline
	}

	if c.running {
		// A forward is already in flight; we accumulate into the pending batch (the NEXT
		// forward, captured strictly after we arrived) and wait for its shared result.
		c.mu.Unlock()
		return c.wait(b, deadline)
	}

	// Leader: mark running and detach the batch BEFORE releasing c.mu, freezing
	// membership before the capture.
	c.running = true
	c.pending = nil
	c.mu.Unlock()

	f, err := c.runBatch(b)
	c.drain()
	return f, err
}

// runBatch runs one forward and publishes the shared (frontier, err) to followers.
func (c *metaFrontierCoalescer) runBatch(b *metaFrontierBatch) (uint64, error) {
	f, err := b.fn(b.deadline)
	b.frontier = f
	b.err = err
	close(b.done)
	return f, err
}

// drain runs forwards that accumulated while a prior forward ran, until none remain,
// then frees the flight slot. Each detach happens strictly after every reader in
// that batch arrived. Runs on the first leader's goroutine; followers only wait.
func (c *metaFrontierCoalescer) drain() {
	for {
		c.mu.Lock()
		if c.pending == nil {
			c.running = false
			c.mu.Unlock()
			return
		}
		b := c.pending
		c.pending = nil
		c.mu.Unlock()

		_, _ = c.runBatch(b)
	}
}

// wait blocks a follower on its batch's result, bounded by its own deadline. On its
// deadline it returns the not-leader sentinel so the follower loop retries (it never
// serves stale on this path).
func (c *metaFrontierCoalescer) wait(b *metaFrontierBatch, deadline time.Time) (uint64, error) {
	select {
	case <-b.done:
		return b.frontier, b.err
	case <-time.After(time.Until(deadline)):
		return 0, errMetaNotLeader
	}
}
