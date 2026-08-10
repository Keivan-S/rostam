// SPDX-License-Identifier: Apache-2.0

package cluster

import (
	"sync"
	"sync/atomic"
	"testing"
	"time"
)

// TestMetaFrontierCoalescerSharesInFlight: follower forwards that arrive while one
// forward is in flight coalesce into ONE follow-up forward (fewer leader RTTs).
func TestMetaFrontierCoalescerSharesInFlight(t *testing.T) {
	var c metaFrontierCoalescer
	const n = 8

	var calls atomic.Int64
	release := make(chan struct{})
	registered := make(chan struct{}, n)

	var wg sync.WaitGroup
	for i := 0; i < n; i++ {
		wg.Add(1)
		go func() {
			defer wg.Done()
			registered <- struct{}{}
			_, _ = c.do(time.Now().Add(2*time.Second), func(time.Time) (uint64, error) {
				calls.Add(1)
				<-release
				return 7, nil
			})
		}()
	}
	for i := 0; i < n; i++ {
		<-registered
	}
	time.Sleep(50 * time.Millisecond)
	close(release)
	wg.Wait()

	if got := calls.Load(); got >= n {
		t.Fatalf("coalescer ran %d forwards for %d concurrent followers, want < %d", got, n, n)
	}
	if calls.Load() == 0 {
		t.Fatal("coalescer never ran the forward")
	}
}

// TestMetaFrontierCoalescerLatecomerGetsFreshFlight is the NO-STALENESS proof: a
// follower arriving AFTER an in-flight forward captured its frontier must get a
// FRESH forward (capture strictly later), never the in-flight pre-arrival frontier.
func TestMetaFrontierCoalescerLatecomerGetsFreshFlight(t *testing.T) {
	var c metaFrontierCoalescer

	var captureSeq atomic.Int64
	flight1Capturing := make(chan struct{})
	flight1Release := make(chan struct{})
	var captureOnce sync.Once

	fnFor := func(name string, slot *int64) func(time.Time) (uint64, error) {
		return func(time.Time) (uint64, error) {
			s := captureSeq.Add(1)
			*slot = s
			if name == "R1" {
				captureOnce.Do(func() { close(flight1Capturing) })
				<-flight1Release
			}
			return uint64(s), nil
		}
	}

	var r1Seq, r2Seq int64
	r1Done := make(chan struct{})
	go func() {
		_, _ = c.do(time.Now().Add(5*time.Second), fnFor("R1", &r1Seq))
		close(r1Done)
	}()
	<-flight1Capturing

	r2Done := make(chan struct{})
	go func() {
		_, _ = c.do(time.Now().Add(5*time.Second), fnFor("R2", &r2Seq))
		close(r2Done)
	}()

	time.Sleep(50 * time.Millisecond)
	close(flight1Release)
	<-r1Done
	<-r2Done

	if r1Seq != 1 {
		t.Fatalf("R1 capture seq = %d, want 1", r1Seq)
	}
	if r2Seq <= r1Seq {
		t.Fatalf("latecomer R2 capture seq = %d, must be > R1 seq %d (no pre-arrival frontier)", r2Seq, r1Seq)
	}
}

// TestMetaFrontierCoalescerSingleReaderRunsForward: a lone follower runs the forward
// directly (identical to the un-coalesced path) and gets its result.
func TestMetaFrontierCoalescerSingleReaderRunsForward(t *testing.T) {
	var c metaFrontierCoalescer
	f, err := c.do(time.Now().Add(time.Second), func(time.Time) (uint64, error) {
		return 42, nil
	})
	if err != nil {
		t.Fatalf("do: %v", err)
	}
	if f != 42 {
		t.Fatalf("frontier = %d, want 42", f)
	}
}
