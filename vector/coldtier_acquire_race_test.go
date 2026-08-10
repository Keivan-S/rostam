// SPDX-License-Identifier: Apache-2.0

package vector

import (
	"sync"
	"sync/atomic"
	"testing"
	"time"

	"github.com/rostamlabs/rostam/objstore"
)

// TestAcquireConcurrentLastAccessStamp exercises the cold-tier last-access stamp
// on the Acquire read/write hot path under concurrency. With cold tiering armed
// (an injected clock), many goroutines resolve the SAME hot collection at once;
// each Acquire stamps the collection's last-access under the SHARED read lock.
//
// Regression guard for the concurrent-map-write crash: the stamp used to be a
// write into a shared map (s.access[canonical] = ...) performed under s.mu.RLock.
// Two concurrent Acquire calls both hold the read lock simultaneously, so both hit
// mapassign at once and the runtime aborts the process with "fatal error:
// concurrent map writes". The fix stamps a per-collection atomic instead, which is
// lock-free and race-clean. Run under `go test -race`: this must stay green (and
// would fatal/report on the old map-write code).
func TestAcquireConcurrentLastAccessStamp(t *testing.T) {
	s := newColdStore(t)

	// Arm cold tiering: inject a clock so Acquire stamps last-access on resolve.
	var nowNanos atomic.Int64
	nowNanos.Store(time.Date(2026, 6, 23, 0, 0, 0, 0, time.UTC).UnixNano())
	s.SetClock(func() time.Time { return time.Unix(0, nowNanos.Load()).UTC() })

	seedCollection(t, s, "hot", 1, 2, 3)

	// Mirror production wiring: run one idle sweep first (the driver calls SweepCold
	// on a ticker), which seeds last-access and is what originally "armed" the crash.
	obj := objstore.NewMemStore()
	if _, err := s.SweepCold(time.Unix(0, nowNanos.Load()).UTC(), time.Hour, obj, "acme"); err != nil {
		t.Fatalf("sweep: %v", err)
	}

	// Many goroutines resolve the same hot collection concurrently. Each Acquire
	// takes the shared read lock and stamps last-access; the race detector flags any
	// unsynchronized shared-state mutation.
	const goroutines, iters = 32, 200
	var wg sync.WaitGroup
	wg.Add(goroutines)
	for g := 0; g < goroutines; g++ {
		go func() {
			defer wg.Done()
			for i := 0; i < iters; i++ {
				c, ok := s.Acquire("hot")
				if !ok {
					t.Errorf("Acquire(hot) missed a hot collection")
					return
				}
				c.Release()
			}
		}()
	}
	wg.Wait()
}
