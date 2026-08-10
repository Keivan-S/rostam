// SPDX-License-Identifier: Apache-2.0

package cluster

import (
	"bytes"
	"log/slog"
	"strings"
	"testing"
	"time"

	"github.com/rostamlabs/rostam/ops"
)

// TestGrowAbortLoggedAndRateLimited is the end-to-end payoff: a grow
// that can never succeed must (a) emit at least one log line and a non-zero
// abort counter — closing the audited gap where cluster/pb_grow.go silently
// discarded every catch-up error — and (b) NOT flood the log with one line
// per tick: the grow driver ticks sub-second (PBGrowTickMs below), so over
// many ticks against the same stuck target the log-line count must stay far
// below the abort count.
//
// The cluster is built with newShrinkablePBTestCluster, whose PB-transport
// wrapper implements only the plain Replicate contract (see pbDroppingTransport
// in pb_shrink_test.go) — NOT pbisr.CatchupTransport/GroupTransport. So on
// this harness Engine.StartLearnerCatchup deterministically fails at the
// handshake step with pbisr.ErrGrowNoCatchupTransport for EVERY target, on
// EVERY tick: a stuck grow with no dependence on real node liveness, network
// timing, or write-history races (unlike engineering a genuinely
// unreachable/lagging peer, which risks tripping unrelated backfill-path
// behavior — out of scope for this observation-only stage to touch).
// Ordinary writes are unaffected (Replicate still forwards to the base
// transport), so establishing a primary and the pre-reset baseline write
// works exactly as in the other pb_*_test.go harness tests.
func TestGrowAbortLoggedAndRateLimited(t *testing.T) {
	const numShards = 1
	const sh = 0
	tc, _ := newShrinkablePBTestCluster(t, 3, numShards, 2, func(c *Config) {
		c.PBAutoFailover = true
		c.PBGrowTickMs = 150 // fast ticking so the test accumulates many aborts quickly
	})

	key := []byte("grow-abort-key")
	putArgs := ops.EncodePutArgs(key, []byte("v"), 0)
	primaryIdx := findShardPrimaryNode(t, tc, sh, putArgs)
	primary := tc.nodes[primaryIdx]

	// Capture slog output for the duration of the grow attempts. No cluster
	// test in this package uses t.Parallel(), so swapping the process-wide
	// default logger is safe here.
	var logBuf bytes.Buffer
	origLogger := slog.Default()
	slog.SetDefault(slog.New(slog.NewTextHandler(&logBuf, nil)))
	t.Cleanup(func() { slog.SetDefault(origLogger) })

	// ESTABLISH THE TEST'S PREMISE BEFORE BUMPING THE EPOCH: shard `sh` must have
	// a VISIBLE placement in the watched node's MetaFSM.
	//
	// The grow driver reads its re-add candidates from MetaState.Placement
	// (decidePBGrow(placement, isr)). That table is normally filled by the
	// bootstrap OpSetMembers entry — but that entry can be LOST to bootstrap
	// leadership churn (appended on a leader that steps down before it commits),
	// which is why MetaFSM.Apply carries an explicit OpSetPlacement self-heal path.
	// In a static test cluster nothing ever triggers that self-heal, so Placement
	// stays permanently empty in exactly those runs: decidePBGrow returns no
	// targets, the driver takes its "nothing to mutate" early-continue, never
	// reaches the catch-up step, and records ZERO aborts. The assertions below then
	// fail against a premise that silently does not hold — a ~25% flake under
	// -race, with the PRODUCT behaviour (refuse to grow against an unknown
	// placement) entirely correct.
	//
	// So: wait briefly for the bootstrap entry, and if it was lost, drive the same
	// OpSetPlacement self-heal production would. Deterministic, and it strengthens
	// nothing away — every assertion below is untouched.
	placementVisible := func() bool {
		got := primary.meta.FSM.State().Placement
		return sh < len(got) && len(got[sh]) > 0
	}
	healDeadline := time.Now().Add(5 * time.Second)
	for !placementVisible() && time.Now().Before(healDeadline) {
		time.Sleep(50 * time.Millisecond)
	}
	if !placementVisible() {
		// The bootstrap entry was lost. Commit this shard's owners explicitly (the
		// node's OWN computed placement, which is what the seed was derived from).
		owners := append([]string(nil), primary.shardOwners(sh)...)
		if len(owners) == 0 {
			t.Fatalf("shard %d has no placement owners on the primary node", sh)
		}
		healed := false
		for _, nd := range tc.nodes {
			if err := nd.meta.ApplySetPlacement(sh, numShards, owners, 5*time.Second); err == nil {
				healed = true
				break
			}
		}
		if !healed {
			t.Fatal("could not self-heal the placement table on any node (no meta leader?)")
		}
		healDeadline = time.Now().Add(10 * time.Second)
		for !placementVisible() && time.Now().Before(healDeadline) {
			time.Sleep(50 * time.Millisecond)
		}
		if !placementVisible() {
			t.Fatalf("shard %d placement still not visible after an explicit OpSetPlacement "+
				"— the grow driver would have no re-add candidates, so this test cannot assert anything", sh)
		}
	}

	// Reset the ISR to {primary} via the election-reset op (same technique as
	// TestReadinessFailsOnUnderReplicatedShard) — exactly how an
	// under-replicated ISR actually arises (a failover). Both other placement
	// owners become grow re-add candidates, and — on this harness — every
	// attempt on either aborts with ErrGrowNoCatchupTransport.
	epoch := primary.meta.FSM.ShardEpoch(sh)
	primaryID := primary.meta.FSM.ShardPrimary(sh)
	var applied bool
	for _, nd := range tc.nodes {
		if err := nd.meta.ApplySetShardEpoch(sh, epoch+1, primaryID, 5*time.Second); err == nil {
			applied = true
			break
		}
	}
	if !applied {
		t.Fatal("could not apply epoch bump on any node (no meta leader?)")
	}

	// Give the grow driver many ticks (150ms interval; the window is well over 10
	// ticks even accounting for the growStableTicks settle window and a loaded
	// host). wantAborts is deliberately well ABOVE the log-line bound asserted
	// below, so "one line per abort" and "one line per target" are separated by a
	// wide margin rather than by one or two counts.
	const wantAborts = 24
	deadline := time.Now().Add(20 * time.Second)
	var total uint64
	for time.Now().Before(deadline) {
		total = 0
		for _, c := range primary.pbGrow.AbortCounts(sh) {
			total += c
		}
		if total >= wantAborts {
			break
		}
		time.Sleep(100 * time.Millisecond)
	}
	if total < wantAborts {
		t.Fatalf("grow-abort count = %d after 20s, want >= %d (every tick should abort against the capability-less transport)", total, wantAborts)
	}
	counts := primary.pbGrow.AbortCounts(sh)
	if counts["no_catchup_transport"] == 0 {
		t.Fatalf("grow-abort counts = %v, want a non-zero no_catchup_transport count", counts)
	}

	logLines := strings.Count(logBuf.String(), "pb: grow catch-up aborted")
	if logLines == 0 {
		t.Fatalf("no grow-catch-up-aborted log line emitted despite %d recorded aborts; log=%s", total, logBuf.String())
	}
	// The core rate-limiting assertion, now stated as the bound the tracker
	// actually promises (M4): logging is bounded by the number of stuck TARGETS
	// and their reason transitions, NOT by the number of ticks. Both grow
	// candidates are permanently stuck on the same reason here, so the steady
	// state is exactly one line each; the slack of 2x absorbs a re-transition
	// after a clear() (a tick that found nothing to grow) without admitting
	// anything resembling a per-tick flood.
	//
	// The old bound (lines*2 < total) was satisfied by 2 lines for 5 aborts, which
	// is only barely distinguishable from flooding. This one is not: a per-tick
	// flood would put logLines at ~total.
	candidates := len(primary.meta.FSM.State().Placement[sh]) - 1 // every owner but the primary
	maxLines := 2 * candidates
	if candidates <= 0 || logLines > maxLines {
		t.Fatalf("rate limiting failed: %d log lines for %d aborts across %d stuck targets (want <= %d — bounded by targets, not by ticks)",
			logLines, total, candidates, maxLines)
	}
	t.Logf("grow-abort count=%d, log lines=%d (rate-limited as expected)", total, logLines)
}
