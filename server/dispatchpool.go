// SPDX-License-Identifier: Apache-2.0

package server

import "sync/atomic"

// dispatchPool reuses the goroutines that run pipelined requests, so their
// stacks stay grown instead of being regrown on every request.
//
// WHY THIS EXISTS. handleConn's pipelined path used to do `go func(req){...}`
// per request. A fresh goroutine starts with a small stack, and the replicated
// write path (dispatch -> keysDispatcher -> fanoutDispatcher -> Node.Call ->
// callHostedShard -> Store.Call -> applyOpIndexed -> pbReplicator.ApplyIndexed
// -> pbisr.Engine.ProposeDeadline -> ...) is deep enough to overflow it, so
// every request paid a runtime.copystack. A CPU profile of a 3-node PB RF=2
// cluster under 128-conn PUT load measured runtime.newstack at **11.89%
// cumulative**, ~98% of the time attributed to cluster.(*MetaFSM).ShardPrimary
// — which does nothing but an RLock and a map read (0.11s of the 6.19s). That
// function was not slow; it was merely the frame whose prologue sat at the
// stack boundary. Optimising it would have moved the growth to its neighbour.
//
// WHY NOT A FIXED-SIZE POOL. The dispatch this runs BLOCKS: a replicated write
// waits for its ISR acks inside Propose. A pool capped near GOMAXPROCS would
// cap in-flight writes and serialise the very thing pipelining exists to
// overlap. So this pool never makes a submitter wait for a worker — it hands
// the job to an idle worker if one exists and spawns a fresh goroutine
// otherwise. Concurrency semantics are therefore IDENTICAL to the previous
// code; only stack reuse changes.
//
// The idle set is bounded: a worker that finishes when the idle channel is
// full exits and lets its stack be collected, so a load spike does not leave
// thousands of parked goroutines behind forever.
//
// SHARDED PER CONNECTION, not per server. A single server-wide idle channel was
// measured and rejected: with GET load (short, non-blocking requests) every one
// of 128 connections hit the same channel on every request, and while the CPU
// cost roughly cancelled the stack-growth saving (chansend 3.12% -> 6.21%,
// newstack 8.02% -> gone), read QPS fell 6.3% and run-to-run spread went from
// 0.6% to 36.4%. One pool per connection keeps the channel private to that
// connection's reader goroutine and its own workers, so submissions never
// contend across connections. The idle bound is therefore per connection and
// sized to the pipeline window — a connection cannot have more than that many
// requests in flight, so it cannot use more workers than that.
type dispatchPool struct {
	idle   chan *dispatchWorker
	stop   chan struct{}
	closed atomic.Bool
}

type dispatchWorker struct {
	jobs chan func() // buffered 1; the submitter must never block on hand-off
	pool *dispatchPool
}

// maxIdleDispatchWorkers bounds one connection's parked-worker set. A
// connection cannot have more than connPipelineWindow requests in flight, so it
// can never need more workers than that; anything larger would only park idle
// goroutines.
const maxIdleDispatchWorkers = connPipelineWindow

func newDispatchPool() *dispatchPool {
	return &dispatchPool{
		idle: make(chan *dispatchWorker, maxIdleDispatchWorkers),
		stop: make(chan struct{}),
	}
}

// run executes fn on a pooled goroutine, or a new one if none is idle. It never
// blocks waiting for a worker.
func (p *dispatchPool) run(fn func()) {
	select {
	case w := <-p.idle:
		w.jobs <- fn // buffered(1) and w is parked on recv, so this cannot block
		return
	default:
	}
	w := &dispatchWorker{jobs: make(chan func(), 1), pool: p}
	w.jobs <- fn
	go w.loop()
}

func (w *dispatchWorker) loop() {
	for {
		var fn func()
		select {
		case fn = <-w.jobs:
		case <-w.pool.stop:
			return
		}
		fn()
		// Park for reuse, or retire if the idle set is full or the pool closed.
		if w.pool.closed.Load() {
			return
		}
		select {
		case w.pool.idle <- w:
		default:
			return // idle set full: let this goroutine (and its stack) go
		}
	}
}

// close retires the pool's parked workers. Idempotent.
func (p *dispatchPool) close() {
	if p.closed.CompareAndSwap(false, true) {
		close(p.stop)
	}
}
