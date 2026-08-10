//go:build cgo

// SPDX-License-Identifier: Apache-2.0

package wasm

import (
	"fmt"
	"strings"
	"sync"
	"testing"
	"time"

	"github.com/rostamlabs/rostam/cache"
	"github.com/rostamlabs/rostam/ops"
)

// hugeFuel makes the epoch deadline (not fuel) the limiter for the busy-loop
// modules below, so these tests exercise the wall-clock/epoch path specifically.
const hugeFuel = uint64(1) << 60

// spinForeverWAT loops forever; only the epoch deadline can stop it.
const spinForeverWAT = `(module
  (memory (export "memory") 1)
  (func (export "apply") (param i32 i32) (result i32)
    (loop $l (br $l))
    (i32.const 0)))`

// watCountLoop is a bounded pure-compute loop of iters iterations that then
// returns 0. Large iters keeps it running for tens of ms so a foreign epoch bump
// can be timed to land while it executes.
func watCountLoop(iters int64) string {
	return fmt.Sprintf(`(module
  (memory (export "memory") 1)
  (func (export "apply") (param i32 i32) (result i32)
    (local $i i64)
    (block $done
      (loop $l
        (br_if $done (i64.ge_u (local.get $i) (i64.const %d)))
        (local.set $i (i64.add (local.get $i) (i64.const 1)))
        (br $l)))
    (i32.const 0)))`, iters)
}

func newTxCache(t *testing.T) *ops.TxContext {
	t.Helper()
	c, err := cache.New(cache.DefaultConfig())
	if err != nil {
		t.Fatalf("cache.New: %v", err)
	}
	t.Cleanup(func() { _ = c.Close() })
	return ops.NewTxContext(c)
}

// A single epoch increment models ANOTHER module's invoke timing out and bumping
// the shared engine epoch. It must NOT trap a fresh, in-budget invoke of a
// different module. Under the old code (SetEpochDeadline(1) + a per-invoke timer
// that IncrementEpoch()s the shared engine) that one bump trapped every store
// holding the identical engineEpoch+1 deadline — this test reproduces exactly
// that cross-talk and asserts it no longer happens.
func TestEpochIncrementDoesNotTrapConcurrentInvoke(t *testing.T) {
	// Budget >= 2 ticks and a dormant ticker (period far longer than the test),
	// so the ONLY epoch bump during the test is the one we trigger by hand.
	rt, err := newRuntimeWithTiming(1000*time.Second, 1000*time.Second)
	if err != nil {
		t.Fatalf("newRuntimeWithTiming: %v", err)
	}
	defer func() { _ = rt.Close() }()
	if rt.epochTicks < 2 {
		t.Fatalf("epochTicks=%d, want >= 2 so a single foreign bump cannot trip an invoke", rt.epochTicks)
	}

	m := compileWAT(t, watCountLoop(800_000_000))
	id, err := rt.AddModule(m, "apply", hugeFuel)
	if err != nil {
		t.Fatalf("AddModule: %v", err)
	}
	tx := newTxCache(t)

	var wg sync.WaitGroup
	var invErr error
	wg.Add(1)
	go func() {
		defer wg.Done()
		_, invErr = rt.Invoke(id, tx, nil)
	}()

	// Let the invoke enter its loop and arm its epoch deadline, then bump the
	// shared engine epoch once (simulating a different module timing out).
	time.Sleep(50 * time.Millisecond)
	rt.engine.IncrementEpoch()

	wg.Wait()
	if invErr != nil {
		t.Fatalf("a single foreign epoch increment trapped an in-budget invoke: %v", invErr)
	}
}

// The ticker must still bound a runaway: an infinite loop is trapped at ~the
// wall-clock budget, so removing the per-invoke timer did not lose the deadline.
func TestEpochTickerBoundsRunaway(t *testing.T) {
	rt, err := newRuntimeWithTiming(80*time.Millisecond, 10*time.Millisecond)
	if err != nil {
		t.Fatalf("newRuntimeWithTiming: %v", err)
	}
	defer func() { _ = rt.Close() }()

	m := compileWAT(t, spinForeverWAT)
	id, err := rt.AddModule(m, "apply", hugeFuel)
	if err != nil {
		t.Fatalf("AddModule: %v", err)
	}
	tx := newTxCache(t)

	start := time.Now()
	_, err = rt.Invoke(id, tx, nil)
	elapsed := time.Since(start)
	if err == nil {
		t.Fatal("infinite-loop op was not trapped by the epoch deadline")
	}
	// The epoch-deadline trap surfaces through wasmtime as an "interrupt" trap.
	if !strings.Contains(err.Error(), "interrupt") {
		t.Fatalf("err = %v, want an epoch-interruption (deadline) trap", err)
	}
	if elapsed > 5*time.Second {
		t.Fatalf("runaway took %v to trap; the wall-clock bound looks broken", elapsed)
	}
}

// While one module runs long (trapped by the ticker), repeated concurrent
// invokes of a short in-budget module must all succeed — no cross-talk.
func TestConcurrentInvokesNotTrappedByOtherTimeout(t *testing.T) {
	// Generous budget (100 ticks) so the short op is comfortably in-budget even
	// under -race / CPU contention, while the infinite op is still trapped.
	rt, err := newRuntimeWithTiming(2*time.Second, 20*time.Millisecond)
	if err != nil {
		t.Fatalf("newRuntimeWithTiming: %v", err)
	}
	defer func() { _ = rt.Close() }()

	slow := compileWAT(t, spinForeverWAT)
	slowID, err := rt.AddModule(slow, "apply", hugeFuel)
	if err != nil {
		t.Fatalf("AddModule slow: %v", err)
	}
	quick := compileWAT(t, watCountLoop(20_000_000))
	quickID, err := rt.AddModule(quick, "apply", hugeFuel)
	if err != nil {
		t.Fatalf("AddModule quick: %v", err)
	}

	// Fire the slow (infinite) op; it will trap ~2s later. Its tx is built on the
	// test goroutine (newTxCache may call t.Fatalf, which must not run elsewhere).
	slowTx := newTxCache(t)
	var slowWG sync.WaitGroup
	slowWG.Add(1)
	go func() {
		defer slowWG.Done()
		_, _ = rt.Invoke(slowID, slowTx, nil)
	}()

	// Hammer the quick op concurrently; every invocation must succeed.
	const workers = 4
	const perWorker = 20
	tx := newTxCache(t)
	var wg sync.WaitGroup
	errs := make(chan error, workers*perWorker)
	for w := 0; w < workers; w++ {
		wg.Add(1)
		go func() {
			defer wg.Done()
			for i := 0; i < perWorker; i++ {
				if _, err := rt.Invoke(quickID, tx, nil); err != nil {
					errs <- err
				}
			}
		}()
	}
	wg.Wait()
	close(errs)
	for err := range errs {
		t.Fatalf("a short in-budget invoke was trapped while another op ran long: %v", err)
	}
	// Draining the slow op is not required for correctness; Close stops the
	// ticker and the trapped invoke returns on its own.
	slowWG.Wait()
}
