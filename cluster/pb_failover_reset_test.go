// SPDX-License-Identifier: Apache-2.0

package cluster

import (
	"reflect"
	"testing"
)

// pbState builds a one-shard control-plane State for the failover-reset tests.
func pbState(shard int, epoch uint64, primary string, isr []string) State {
	return State{
		ShardEpoch:   map[int]uint64{shard: epoch},
		ShardPrimary: map[int]string{shard: primary},
		ShardISR:     map[int][]string{shard: isr},
	}
}

// TestPBFailoverHonorRuleSelfFenceBeforePromotion is the timing-relationship guard
// for the no-DOUBLE-primary property. The crash-stop gate (TestPBFailoverNoAckedLoss)
// proves no-acked-LOSS but stops the old primary instantly, so it never exercises
// the window where a partitioned-but-ALIVE primary must self-fence BEFORE its
// replacement is named. That window is a pure consequence of the RESOLVED timings:
//
//	earliest promotion  = τ + failoverTimeout − renewInterval   (lastRenewNs can be
//	                                                              a full interval old)
//	old-primary fence   = τ + metaContactStaleness + pbLeaseTTL  (lease lapses; see
//	                                                              TestOH1StalePrimary…)
//
// Safety (promotion strictly after fence) ⟺
// failoverTimeout − renewInterval > pbLeaseTTL + metaContactStaleness. We assert it
// for the SHIPPED defaults AND the gate test's shrunk timings. That a non-renewed
// primary actually DOES fence by pbLeaseTTL is proven by
// TestOH1StalePrimarySelfFencesOnLeaseExpiry (shard/pbisr). A full network-partition
// e2e test remains the explicit REMAINING gate item before PBAutoFailover goes
// default-on (see shard/pbisr/DESIGN.md) — this property test + that
// self-fence unit test + the crash-stop no-loss gate are the interim proof, NOT an
// e2e partition.
func TestPBFailoverHonorRuleSelfFenceBeforePromotion(t *testing.T) {
	cases := []struct {
		name string
		mut  func(*Config)
	}{
		{"shipped defaults", func(c *Config) {}},
		{"gate-test shrunk timings", func(c *Config) {
			c.PBLeaseTTLMs = 1000
			c.PBMetaContactStalenessMs = 500
			c.PBFailoverTimeoutMs = 3000
			c.PBRenewIntervalMs = 300
		}},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			var c Config
			tc.mut(&c)
			leaseTTL, metaStale, failTO, renew := c.pbEffectiveTimings()
			if failTO-renew <= leaseTTL+metaStale {
				t.Fatalf("no-double-primary window violated: failoverTimeout(%s) − renewInterval(%s) = %s must strictly exceed pbLeaseTTL(%s) + metaContactStaleness(%s) = %s",
					failTO, renew, failTO-renew, leaseTTL, metaStale, leaseTTL+metaStale)
			}
		})
	}
}

// TestPBFailoverElectionFloor is Stage-3 case (a): the mass-promote guard. A
// freshly-(re)elected meta leader must NOT treat pre-leadership silence as failover
// evidence. With the election floor (tracker reset #1), a shard whose last beacon
// is ANCIENT is treated as renewed at the takeover instant, so no promotion fires
// until a full failoverTimeout of ACTUAL post-election silence — but the floor
// DELAYS, it never MASKS: past that window the silent primary is still promoted.
func TestPBFailoverElectionFloor(t *testing.T) {
	const timeout = int64(2_000_000_000) // 2s
	var clk int64
	tr := newPBFailoverTracker(func() int64 { return clk })

	// The shard beaconed at its current epoch, but long ago.
	clk = 1_000
	tr.observeRenew(0, 1)
	clk = 1_000 + 100*timeout // ancient beacon

	// This node just won the meta election.
	tr.onBecomeLeader()
	leaderSince := clk

	st := pbState(0, 1, "n1", []string{"n1", "n2", "n3"})

	// Immediately after takeover: floored to now ⇒ no promotion.
	if got := pbFailoverDecisions(st, tr, clk, timeout, nil); got != nil {
		t.Fatalf("promotion the instant leadership was won: %+v (mass-promote guard failed)", got)
	}
	// Exactly at failoverTimeout past takeover: still live (decidePBPromotions uses <=).
	clk = leaderSince + timeout
	if got := pbFailoverDecisions(st, tr, clk, timeout, nil); got != nil {
		t.Fatalf("promotion at exactly failoverTimeout past takeover: %+v", got)
	}
	// Just past the window with no fresh beacon: the floor delayed but did not mask.
	clk = leaderSince + timeout + 1
	want := []pbPromotion{{shardID: 0, newEpoch: 2, newPrimary: "n2"}}
	if got := pbFailoverDecisions(st, tr, clk, timeout, nil); !reflect.DeepEqual(got, want) {
		t.Fatalf("after the floor window: got %+v, want %+v", got, want)
	}
}

// TestPBFailoverEpochAdvanceReset is Stage-3 case (b): after a promotion bumps a
// shard to E+1, the ticker must NOT re-promote (to E+2, E+3, …) in the window
// before the new primary Q beacons. The epoch-advance grace (tracker reset #2)
// treats a shard whose committed epoch is newer than any stamped beacon as fresh.
// Once Q beacons and later goes silent, normal failover resumes for the new epoch.
func TestPBFailoverEpochAdvanceReset(t *testing.T) {
	const timeout = int64(2_000_000_000)
	var clk int64
	tr := newPBFailoverTracker(func() int64 { return clk })
	tr.onBecomeLeader() // leaderSince = 0 (ancient — the floor is not what we test here)

	// shard0 (n1, epoch1) beaconed then went silent ⇒ promote to n2, epoch2.
	clk = 1_000
	tr.observeRenew(0, 1)
	clk = 1_000 + timeout + 1
	got := pbFailoverDecisions(pbState(0, 1, "n1", []string{"n1", "n2", "n3"}), tr, clk, timeout, nil)
	if want := []pbPromotion{{0, 2, "n2"}}; !reflect.DeepEqual(got, want) {
		t.Fatalf("initial promotion: got %+v, want %+v", got, want)
	}

	// The ticker committed the bump: epoch2, primary n2, ISR reset to {n2}. The
	// tracker still holds the OLD (P,E)=(n1,1) beacon stamp. Far in the future, the
	// shard must NOT be re-promoted — Q has not beaconed yet (epoch-advance grace).
	bumped := pbState(0, 2, "n2", []string{"n2"})
	clk += 100 * timeout
	if got := pbFailoverDecisions(bumped, tr, clk, timeout, nil); got != nil {
		t.Fatalf("re-promoted after epoch bump before the new primary beaconed: %+v", got)
	}

	// New primary n2 beacons (ISR repopulated to {n2,n3}). The grace ends: it is
	// fresh now, so no promotion; then n2 goes silent past the timeout ⇒ promote n3.
	tr.observeRenew(0, 2)
	grown := pbState(0, 2, "n2", []string{"n2", "n3"})
	if got := pbFailoverDecisions(grown, tr, clk, timeout, nil); got != nil {
		t.Fatalf("promoted immediately after the new primary beaconed: %+v", got)
	}
	clk += timeout + 1
	if want := []pbPromotion{{0, 3, "n3"}}; !reflect.DeepEqual(pbFailoverDecisions(grown, tr, clk, timeout, nil), want) {
		t.Fatalf("failover did not resume for the new epoch after Q went silent")
	}
}

// TestPBFailoverSilentPrimaryPromotesLowestISRSurvivor is Stage-3 case (c): a
// primary silent past the timeout is failed over to the LOWEST-nodeID ISR member
// other than itself (a survivor guaranteed by full-ISR commit to hold every acked
// write). Exercised through the tracker (not decidePBPromotions directly) so the
// two resets are shown NOT to interfere with a genuine promotion.
func TestPBFailoverSilentPrimaryPromotesLowestISRSurvivor(t *testing.T) {
	const timeout = int64(2_000_000_000)
	var clk int64
	tr := newPBFailoverTracker(func() int64 { return clk })
	tr.onBecomeLeader() // ancient floor

	clk = 1_000
	tr.observeRenew(0, 1)
	clk = 1_000 + timeout + 1
	// ISR deliberately out of order {n1,n3,n2}; the survivor must be n2 (lowest id != n1).
	got := pbFailoverDecisions(pbState(0, 1, "n1", []string{"n1", "n3", "n2"}), tr, clk, timeout, nil)
	if want := []pbPromotion{{0, 2, "n2"}}; !reflect.DeepEqual(got, want) {
		t.Fatalf("silent-primary promotion: got %+v, want %+v", got, want)
	}
}
