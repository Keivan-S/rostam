// SPDX-License-Identifier: Apache-2.0

package cluster

import "testing"

// THE FAILOVER GATE RANKS ON THE APPLIED WATERMARK, NOT THE FRONTIER.
//
// Log matching split one number into two. AppliedSeq is how much of the COMMITTED
// tail a candidate received AS A REPLICA; the frontier additionally counts
// writes the candidate proposed ITSELF while primary at an older epoch —
// exactly the possibly-uncommitted ones a promotion must not reward. Ranking on
// the frontier can hand the epoch to a node whose lead consists of writes no
// other member ever acked.
//
// Review found this correct at the source and entirely untested: mutating
// pbCandidateHighWater to return the frontier — undoing the split the stage
// exists for — passed the whole cluster package. Every engine the suite builds
// has the two numbers equal, so nothing could tell them apart.
//
// The SOURCE semantic — that an ex-primary's two watermarks actually differ, and
// that CatchupInfo reports them separately — is pinned in the engine package
// (shard/pbisr, TestCatchupInfoSeparatesAppliedFromFrontier). This file pins the
// CONSUMER: the promotion decision ranks on the applied watermark it is handed
// and refuses an unverifiable candidate.

// TestDecidePBPromotionsPrefersHigherAppliedWatermark pins the consumer: given
// the gate's values it promotes the candidate holding more of the committed
// tail, and refuses an unverifiable one outright.
func TestDecidePBPromotionsPrefersHigherAppliedWatermark(t *testing.T) {
	shards := []pbShardLiveness{{
		shardID:     0,
		primary:     "n1",
		isr:         []string{"n1", "n2", "n3"},
		lastRenewNs: 0,
	}}
	const now, timeout = int64(1_000_000_000), int64(1000)

	got := decidePBPromotions(shards, now, timeout, func(_ int, c string) (uint64, bool) {
		switch c {
		case "n2":
			return 5, true
		case "n3":
			return 9, true // holds more of the committed tail
		}
		return 0, false
	})
	if len(got) != 1 || got[0].newPrimary != "n3" {
		t.Fatalf("promoted %+v, want n3 — the candidate with the higher applied watermark", got)
	}

	// Unverifiable candidates are excluded even when they would otherwise win.
	got = decidePBPromotions(shards, now, timeout, func(_ int, c string) (uint64, bool) {
		if c == "n3" {
			return 0, false
		}
		return 5, true
	})
	if len(got) != 1 || got[0].newPrimary != "n2" {
		t.Errorf("with n3 unverifiable, promoted %+v, want n2 — an unreachable or poison-fenced "+
			"candidate is never promotable", got)
	}
}
