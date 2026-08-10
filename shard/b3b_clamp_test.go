// SPDX-License-Identifier: Apache-2.0

package shard

import (
	"testing"
	"time"

	"github.com/rostamlabs/rostam/ops"
)

// TestB3bLeaderStampMonotonicClamp proves the leader-side monotonic clamp in
// applyOpIndexed (#4 Phase B / B3b): a committed write's baked stamp is
// max(wall clock, cache.LastAppliedStampMs()), so it NEVER regresses below any
// apply-stamp already folded into the logical clock — the invariant that makes the
// sweeper-vs-later-write race provably safe.
//
// Testing "the wall clock went backwards" without a wall-clock seam in the stamp
// site: instead we push the LOGICAL clock ABOVE the real wall clock (seed it far in
// the future), then observe that the next committed write's stamp tracks the logical
// clock, not the smaller wall clock. That is exactly the regression the clamp must
// prevent.
func TestB3bLeaderStampMonotonicClamp(t *testing.T) {
	reg := ops.NewRegistry()
	if err := ops.RegisterBuiltins(reg); err != nil {
		t.Fatal(err)
	}
	cfg := DefaultConfig(t.TempDir(), "node1", reg)
	cfg.Bootstrap = true
	cfg.RaftHeartbeatMs = 50
	cfg.RaftElectionMs = 100
	cfg.NoSync = true
	cfg.EnableApplyStamp = true // turn on the stamped apply path
	s, err := New(cfg)
	if err != nil {
		t.Fatalf("shard.New: %v", err)
	}
	t.Cleanup(func() { _ = s.Close() })
	waitLeader(t, s)

	// Seed the shard's logical clock FAR above the real wall clock (~11.5 days
	// ahead) by folding a high stamp in directly (a stamped apply would do this in
	// production). We use PutAt on the underlying cache purely to advance the clock.
	future := uint64(time.Now().UnixMilli()) + 1_000_000_000
	if err := s.cache.PutAt([]byte("__seed__"), []byte("x"), 0, future); err != nil {
		t.Fatal(err)
	}
	if got := s.cache.LastAppliedStampMs(); got < future {
		t.Fatalf("seed failed: logical clock=%d, want >= %d", got, future)
	}

	// A committed write with a 1s TTL. Its stamp goes through the real clamp in
	// applyOpIndexed. If the clamp holds, stamp = future (> wall clock), so
	// exp = future + 1000 and the key is LIVE at `future`. If the stamp had regressed
	// to the wall clock, exp = wallNow + 1000 << future and the key would read as
	// already expired at `future`.
	if err := s.Put([]byte("k"), []byte("v"), time.Second); err != nil {
		t.Fatalf("Put: %v", err)
	}
	if _, err := s.cache.GetAt([]byte("k"), future); err != nil {
		t.Fatalf("GetAt(k, future) = %v, want hit — the leader stamp regressed below the logical clock (clamp failed)", err)
	}
	// Sanity: at future + 2000 (past future+1000) the key is expired, confirming the
	// expiry really was computed from the clamped stamp `future`, not some larger value.
	if _, err := s.cache.GetAt([]byte("k"), future+2_000); err == nil {
		t.Fatal("GetAt(k, future+2000) unexpectedly hit — expiry not anchored at the clamped stamp")
	}
	// The logical clock never regressed.
	if got := s.cache.LastAppliedStampMs(); got < future {
		t.Fatalf("logical clock regressed to %d, want >= %d (must be monotonic)", got, future)
	}
}
