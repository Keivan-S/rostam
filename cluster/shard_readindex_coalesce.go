// SPDX-License-Identifier: Apache-2.0

package cluster

import (
	"sync"
	"time"
)

// shardFrontierCoalescer coalesces concurrent follower __shard_readindex__ forwards
// on ONE Node into shared leader RTTs, KEYED PER SHARD: N concurrent
// ConsistencyBoundedStaleness reads for the SAME shard on a follower share ONE leader
// frontier ping, while reads for different shards proceed independently. It is the
// per-shard analogue of metaFrontierCoalescer (which is global to the meta-Raft);
// each shard gets its own embedded batch-while-busy coalescer (shardFrontierFlight),
// guarded by a mutex over the per-shard map. The same arrival<=capture rule that
// makes the meta coalescer safe applies per shard: a forward captures the shard
// leader's committed frontier at the leader's verify/capture time tc, and that
// frontier is safe for a read R iff R.arrival <= tc; followers arriving while a
// forward is in flight accumulate into the NEXT flight (captured strictly after they
// arrived). See meta_readindex_coalesce.go for the full proof.
type shardFrontierCoalescer struct {
	mu      sync.Mutex
	flights map[int]*shardFrontierFlight
}

// shardFrontierFlight is the per-shard batch-while-busy coalescer state, mirroring
// metaFrontierCoalescer's (running, pending) pair for a single shard.
type shardFrontierFlight struct {
	mu      sync.Mutex
	running bool
	pending *shardFrontierBatch
}

// shardFrontierBatch is one coalesced forward (mirrors metaFrontierBatch). done is
// closed by the flight leader once (frontier, err) hold the shared result; followers
// block on done then read them. deadline is the MAX deadline among the batch's
// readers; fn is the per-shard fetch closure fixed by the batch's first registrant.
type shardFrontierBatch struct {
	done     chan struct{}
	frontier uint64
	err      error
	deadline time.Time
	fn       func(time.Time) (uint64, error)
}

// flightFor returns (creating if needed) the per-shard flight state, guarded by the
// coalescer mutex.
func (c *shardFrontierCoalescer) flightFor(idx int) *shardFrontierFlight {
	c.mu.Lock()
	defer c.mu.Unlock()
	if c.flights == nil {
		c.flights = make(map[int]*shardFrontierFlight)
	}
	f := c.flights[idx]
	if f == nil {
		f = &shardFrontierFlight{}
		c.flights[idx] = f
	}
	return f
}

// do runs (or joins) a coalesced __shard_readindex__ forward for shard idx. fn is
// fetchShardLeaderFrontier bound to idx; it runs at most once per flight, by a flight
// leader, AFTER the batch is detached — so every reader sharing fn's result arrived
// before fn captured the leader frontier. A follower's deadline bounds only how long
// it waits.
func (c *shardFrontierCoalescer) do(idx int, deadline time.Time, fn func(time.Time) (uint64, error)) (uint64, error) {
	return c.flightFor(idx).do(deadline, fn)
}

func (f *shardFrontierFlight) do(deadline time.Time, fn func(time.Time) (uint64, error)) (uint64, error) {
	f.mu.Lock()
	if f.pending == nil {
		f.pending = &shardFrontierBatch{done: make(chan struct{}), fn: fn}
	}
	b := f.pending
	if deadline.After(b.deadline) {
		b.deadline = deadline
	}

	if f.running {
		// A forward is already in flight; accumulate into the pending batch (the NEXT
		// forward, captured strictly after we arrived) and wait for its shared result.
		f.mu.Unlock()
		return f.wait(b, deadline)
	}

	// Flight leader: mark running and detach the batch BEFORE releasing the lock,
	// freezing membership before the capture.
	f.running = true
	f.pending = nil
	f.mu.Unlock()

	res, err := f.runBatch(b)
	f.drain()
	return res, err
}

// runBatch runs one forward and publishes the shared (frontier, err) to followers.
func (f *shardFrontierFlight) runBatch(b *shardFrontierBatch) (uint64, error) {
	res, err := b.fn(b.deadline)
	b.frontier = res
	b.err = err
	close(b.done)
	return res, err
}

// drain runs forwards that accumulated while a prior forward ran, until none remain,
// then frees the flight slot. Each detach happens strictly after every reader in that
// batch arrived. Runs on the first flight leader's goroutine; followers only wait.
func (f *shardFrontierFlight) drain() {
	for {
		f.mu.Lock()
		if f.pending == nil {
			f.running = false
			f.mu.Unlock()
			return
		}
		b := f.pending
		f.pending = nil
		f.mu.Unlock()

		_, _ = f.runBatch(b)
	}
}

// wait blocks a follower on its batch's result, bounded by its own deadline. On its
// deadline it returns the not-shard-leader sentinel so the caller fails closed (the
// shard Store maps it to NotLeaderError ⇒ route to leader); it never serves stale on
// this path.
func (f *shardFrontierFlight) wait(b *shardFrontierBatch, deadline time.Time) (uint64, error) {
	select {
	case <-b.done:
		return b.frontier, b.err
	case <-time.After(time.Until(deadline)):
		return 0, errNotShardLeader
	}
}
