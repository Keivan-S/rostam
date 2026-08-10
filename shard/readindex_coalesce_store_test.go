// SPDX-License-Identifier: Apache-2.0

package shard

import (
	"sync"
	"sync/atomic"
	"testing"
	"time"

	"github.com/rostamlabs/rostam/ops"
	"github.com/rostamlabs/rostam/vector"
)

// TestStoreLinearizableReadsCoalesce is the COALESCING proof on the real Store read
// path: N concurrent Linearizable searches on ONE Store fire FEWER than N actual
// readIndex barriers (VerifyLeader RTTs), because they coalesce into shared flights.
// We gate the barrier body (via the per-flight barrierEnteredHook) so the first
// flight stays open long enough for the rest to register and join it.
func TestStoreLinearizableReadsCoalesce(t *testing.T) {
	s := newSingleNodeVectorStore(t)
	query := []float32{1, 0, 0}
	linArgs := ops.EncodeVectorSearchArgsOpts("docs", 5, query, vector.Filter{}, ops.ConsistencyLinearizable, 0, 0)

	const n = 12
	var barrierEntries atomic.Int64
	release := make(chan struct{})
	var gateOnce sync.Once
	gated := make(chan struct{})
	// The hook fires ONCE per actual flight (inside verifyLeaderAndCatchUpBody). The
	// first flight blocks on release so concurrent readers pile into the open batch
	// (or the next one) instead of each running its own serialized barrier.
	SetBarrierEnteredHook(func() {
		barrierEntries.Add(1)
		gateOnce.Do(func() { close(gated) })
		<-release
	})
	t.Cleanup(func() { SetBarrierEnteredHook(nil) })

	var wg sync.WaitGroup
	for i := 0; i < n; i++ {
		wg.Add(1)
		go func() {
			defer wg.Done()
			if _, err := s.Call("vector_search", linArgs); err != nil {
				t.Errorf("linearizable search: %v", err)
			}
		}()
	}

	<-gated                            // first flight is in the barrier body, blocked
	time.Sleep(100 * time.Millisecond) // let the rest register and coalesce
	close(release)
	wg.Wait()

	if got := barrierEntries.Load(); got >= n {
		t.Fatalf("%d concurrent Linearizable reads fired %d barriers, want < %d (must coalesce)", n, got, n)
	}
	if barrierEntries.Load() == 0 {
		t.Fatal("no barrier fired — Linearizable reads did not run the readIndex barrier at all")
	}
	t.Logf("coalescing: %d concurrent Linearizable reads fired %d barriers (< %d)", n, barrierEntries.Load(), n)
}

// TestStoreLatecomerLivenessAndFreshness is the real-Store liveness+freshness check
// for a LATECOMER to an in-flight coalesced flight: a read R2 that arrives while R1's
// flight is open does NOT deadlock/stall — it waits for R1's flight, then runs its own
// (drained) flight and serves the latest applied state.
//
// NOTE on what this can and cannot prove: the strict arrival>=capture NO-STALENESS
// invariant (a latecomer is never served a pre-arrival frontier) is proven directly,
// without a real Raft, by TestReadindexCoalescerLatecomerGetsFreshFlight (R2's capture
// sequence is strictly after R1's), and end-to-end against a LAGGING FOLLOWER by the
// linearizable inttests (block-then-serve-fresh). It is NOT demonstrable on a
// single-node leader: the barrier is a liveness GATE (it returns once the FSM is
// caught up to the captured commit index), not a snapshot, and a single-node leader's
// FSM is always current — so the subsequent handler read always observes the latest
// applied state regardless of which frontier the gate captured. This test therefore
// asserts the property that IS meaningful here: a latecomer completes (no deadlock,
// no spurious FSM-catch-up timeout) and sees the post-write data.
//
// Construction (must NOT deadlock): R1 is gated inside the barrier body; R2 is launched
// as a goroutine so it registers as a latecomer to R1's in-flight flight; THEN R1 is
// released so its flight finishes and the drain runs R2's flight. (Releasing R1 only
// after R2 returns — as a naive version would — deadlocks, since R2 waits for R1.)
func TestStoreLatecomerLivenessAndFreshness(t *testing.T) {
	s := newSingleNodeVectorStore(t)
	query := []float32{1000, 0, 0} // id=999 (below) will be the unique nearest
	linArgs := ops.EncodeVectorSearchArgsOpts("docs", 1, query, vector.Filter{}, ops.ConsistencyLinearizable, 0, 0)

	// Gate ONLY the first flight (R1) inside the barrier body.
	firstFlight := make(chan struct{})
	releaseFirst := make(chan struct{})
	var gatedOnce sync.Once
	SetBarrierEnteredHook(func() {
		gatedOnce.Do(func() {
			close(firstFlight)
			<-releaseFirst // hold R1's flight open (captured, batch closed)
		})
	})
	t.Cleanup(func() { SetBarrierEnteredHook(nil) })

	// Launch R1; it will block inside the barrier body.
	r1Done := make(chan struct{})
	go func() {
		_, _ = s.Call("vector_search", linArgs)
		close(r1Done)
	}()
	<-firstFlight // R1 has captured its frontier and CLOSED its batch.

	// Commit the post-capture write: id=999 at the query point.
	up := ops.EncodeVectorUpsertArgs("docs", 999, []float32{1000, 0, 0}, "chunk", 0, nil, vector.SparseVector{})
	if _, err := s.Call("vector_upsert", up); err != nil {
		t.Fatalf("upsert id=999 (post-capture): %v", err)
	}

	// R2 arrives NOW (as a goroutine) — a latecomer to R1's in-flight flight. It joins
	// the pending NEXT batch and must wait for R1's flight to finish, not join it.
	type r2result struct {
		res []byte
		err error
	}
	r2ch := make(chan r2result, 1)
	go func() {
		res, err := s.Call("vector_search", linArgs)
		r2ch <- r2result{res, err}
	}()

	// Give R2 time to register as a latecomer behind R1, THEN release R1 so its flight
	// completes and the drain runs R2's flight. (Without this ordering R2 would wait on
	// R1 forever — a self-inflicted deadlock, not a coalescer bug.)
	time.Sleep(200 * time.Millisecond)
	close(releaseFirst)
	<-r1Done

	got := <-r2ch
	if got.err != nil {
		t.Fatalf("latecomer R2 linearizable search: %v (a stalled/timed-out latecomer ⇒ coalescer liveness bug)", got.err)
	}
	hits, err := ops.DecodeVectorSearchResults(got.res)
	if err != nil {
		t.Fatalf("decode R2 hits: %v", err)
	}
	if len(hits) == 0 || hits[0].ID != 999 {
		t.Fatalf("latecomer R2 = %+v, want nearest id=999 (it must complete and see the post-write state)", hits)
	}
}
