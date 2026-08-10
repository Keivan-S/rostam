// SPDX-License-Identifier: Apache-2.0

package cluster

import (
	"encoding/json"
	"fmt"
)

// Replication-observability surface (#6). The __repl_metrics__ admin op renders
// THIS node's per-hosted-shard replication state as JSON: the mode, the shard's
// primary/leader, the ISR set size vs the configured min-ISR floor, whether the
// shard is under-replicated (ISR < min-ISR), and — in PB mode, where the engine
// exposes it — each backup's replication lag (assigned seq behind the primary).
// Like __ready__ it is shardless/node-local (Node.Call dispatches it off
// n.adminOps before key routing) so it reports the node it is sent to.
//
// Three PB-mode fields make ISR under-replication visible
// even when the shard still meets the durability floor: PlacementSize (the
// intended replica count), BelowPlacement (ISRSize < PlacementSize — a lost-
// redundancy signal DISTINCT from UnderReplicated's durability-floor breach),
// and GrowAborts (cumulative Plan-4c grow-abort counts by reason, so a
// transient retry is distinguishable from a permanent wedge).
//
// THESE ARE NOT EDGE-CASE TELEMETRY. shard/pbisr/DESIGN.md
// claims failover re-opens a minISR>=2 shard "in the common — quiesced /
// non-divergent — case". Falsifying experiments against the log-matching check confirm that claim is technically true but practically
// empty: catch-up only succeeds when the survivor's high-water exactly ties the
// promoted primary's (ringDeltaLocked's `to < from` short-circuit at
// shard/pbisr/pb_grow.go:120-122 skips the ring rather than transferring
// anything) — i.e. only when there was NOTHING to transfer. On any ordinary
// failover under live write traffic the survivor and the newly-promoted
// primary have each usually advanced past the other's last agreed point, so
// the grow aborts with pbisr.ErrCatchupDiverged ("diverged" in GrowAborts).
// So at minISR>=2, below_placement=true plus a "diverged" count is the NORMAL
// transient of a non-quiesced failover, not an exotic one.
//
// STAGE 4.3 CHANGED THE SEVERITY OF THAT, AND THE FIELD DOCS BELOW REFLECT THE
// AFTER SNAPSHOT TRANSFER. "diverged" and "ring_evicted" were PERMANENT when
// this surface was written — catch-up is log append, and append cannot repair a fork —
// and they accompanied a permanent write outage. Snapshot transfer now ships
// transfer, which pbisr routes both into automatically, so both are now
// REPAIRABLE and the counts are expected to STOP CLIMBING once repair lands.
// A count that keeps climbing is the signal; a count that plateaued is a healed
// shard. (Reporting a recovered condition as unrecoverable would be worse than
// not reporting it at all.)
//
// What has NOT changed: the outage, however long it lasts, is one __ready__
// reports as healthy, because readiness deliberately stays floor-only (see
// handleReady's doc comment in cluster/admin_ops.go). Do not read a quiet
// under_replicated/ready pair as "the shard recovered": check below_placement
// and grow_aborts.

// replPeerLag is one backup's replication lag as seen by this primary.
type replPeerLag struct {
	Node  string `json:"node"`      // backup node id
	Acked uint64 `json:"acked_seq"` // highest seq this backup has acked
	Lag   uint64 `json:"lag"`       // last_seq - acked_seq (writes behind)
}

// replShardMetrics is one hosted shard's replication view.
type replShardMetrics struct {
	Shard     int    `json:"shard"`
	Mode      string `json:"mode"`       // "pb" | "raft"
	IsPrimary bool   `json:"is_primary"` // this node is the shard's primary/leader
	// Primary is the primary (PB) or leader-address (raft, when known) node id.
	Primary string `json:"primary,omitempty"`
	Epoch   uint64 `json:"epoch,omitempty"` // PB leadership epoch (0 in raft mode)
	// ISRSize/MinISR/UnderReplicated are PB-mode durability signals. UnderReplicated
	// (ISR < min-ISR) is the same condition that fails readiness (#3 linkage). Its
	// MEANING is unchanged — it stays the durability-FLOOR breach only.
	ISRSize         int  `json:"isr_size"`
	MinISR          int  `json:"min_isr"`
	UnderReplicated bool `json:"under_replicated"`
	// PlacementSize is the shard's INTENDED replica count, read
	// from the same placement table the grow driver reads (st.Placement[shardID]
	// in cluster/pb_grow.go). BelowPlacement (ISRSize < PlacementSize) is a
	// SEPARATE, genuinely different signal from UnderReplicated: a shard can have
	// lost redundancy (an owner missing from the ISR) while still meeting the raw
	// min-ISR floor — e.g. min_isr=1, placement 3, ISR={primary} reports
	// under_replicated=false but below_placement=true. Previously that case
	// was invisible on every surface. Both fields are zero/false outside PB mode.
	//
	// below_placement=true is the EXPECTED state, not a rare one, right after any
	// non-quiesced failover at minISR>=2 — see the package doc above. Since Stage
	// 4.3 it is also expected to CLEAR on its own: snapshot transfer repairs the
	// post-failover fork without an operator. below_placement that persists across
	// many grow-driver ticks is the thing to alert on, not its mere presence.
	PlacementSize  int  `json:"placement_size,omitempty"`
	BelowPlacement bool `json:"below_placement"`
	// GrowAborts is this shard's cumulative grow-abort
	// count by reason (e.g. "ring_evicted", "diverged", "submit_failed") since
	// this node's grow driver started, omitted when nothing has been recorded.
	// Lets an operator tell the causes apart, which now differ in
	// SEVERITY as well as in cause — growAbortReason's doc comment in
	// cluster/pb_grow.go is the authority and groups every bucket as retryable,
	// repairable-by-snapshot, needs-re-snapshotting, or configuration.
	//
	// Read the counts as RATES, not as presence. "diverged" and "ring_evicted" are
	// the EXPECTED abort reasons on any ordinary non-quiesced failover and are now
	// self-healing via snapshot transfer, so a handful of them is a shard
	// recovering normally. What is actionable is a count that keeps climbing, or
	// any count in the configuration buckets ("no_snapshot_store",
	// "no_snapshot_transport") — those mean the repair path cannot run at all and
	// the shard really is wedged until an operator intervenes. "unverifiable"
	// (pbisr.ErrCatchupUnverifiable) names a poison-fenced target that is
	// currently REFUSING TO SERVE and needs re-snapshotting. See
	// cluster/pb_grow.go's growAbortReason and pbGrowDriver.AbortCounts.
	GrowAborts map[string]uint64 `json:"grow_aborts,omitempty"`
	// LastSeq/Committed/Backups are the PB primary-side progress (omitted for a
	// raft shard and for a PB backup that has proposed nothing).
	LastSeq   uint64        `json:"last_seq,omitempty"`
	Committed uint64        `json:"committed,omitempty"`
	Backups   []replPeerLag `json:"backups,omitempty"`
}

// replMetrics is the __repl_metrics__ JSON body: this node id and one entry per
// hosted shard.
type replMetrics struct {
	Node   string             `json:"node"`
	Shards []replShardMetrics `json:"shards"`
}

// handleReplMetrics serves the __repl_metrics__ op: it renders THIS node's
// per-hosted-shard replication state (see replShardMetrics). It reuses the same
// hosted-shard discovery as handleReady (getShard != nil) and, in PB mode, reads
// the authoritative (epoch, primary, ISR, min-ISR) from n.pbControl and the
// per-backup lag from the shard's *pbisr.Engine.ReplicationStatus(). In raft
// mode it reports the coarse leader signal only (no ISR/lag — raft does not
// expose a per-follower replication high-water here). It is a pure read; it
// never blocks on Raft or the network.
func (n *Node) handleReplMetrics(_ []byte) ([]byte, error) {
	out := replMetrics{Node: n.cfg.NodeID, Shards: []replShardMetrics{}}
	pb := n.cfg.ReplicationMode == ReplicationModePB
	// Placement: the same source cluster/pb_grow.go's grow
	// driver reads (st.Placement[shardID]) — a pure, non-blocking FSM read.
	var placement [][]string
	if pb && n.meta != nil {
		placement = n.meta.FSM.State().Placement
	}
	for i := 0; i < n.cfg.NumShards; i++ {
		s := n.getShard(i)
		if s == nil {
			continue // not hosted here — not this node's concern
		}
		sm := replShardMetrics{Shard: i}
		if pb && n.pbControl != nil {
			sm.Mode = ReplicationModePB
			sm.Primary = n.pbControl.Primary(i)
			sm.Epoch = n.pbControl.Epoch(i)
			isr := n.pbControl.ISR(i)
			sm.ISRSize = len(isr)
			sm.MinISR = n.pbControl.MinISR(i)
			sm.UnderReplicated = sm.ISRSize < sm.MinISR
			sm.IsPrimary = sm.Primary == n.cfg.NodeID
			if i < len(placement) {
				sm.PlacementSize = len(placement[i])
				sm.BelowPlacement = sm.ISRSize < sm.PlacementSize
			}
			sm.GrowAborts = n.pbGrow.AbortCounts(i)
			// Per-backup lag is a primary-side view: only the primary ships to and
			// hears acks from backups, so only its engine holds the high-water. A
			// backup's engine reports LastSeq==0 / no peers, which we omit.
			if eng := n.pbEngines[i]; eng != nil {
				st := eng.ReplicationStatus()
				sm.LastSeq = st.LastSeq
				sm.Committed = st.Committed
				// Render backups from the CURRENT ISR, not the engine's ack history.
				// The engine's per-peer high-water map is never pruned when a backup
				// leaves the ISR, so a departed peer would otherwise linger here with a
				// frozen acked_seq and an ever-growing (fake) lag; symmetrically a
				// freshly-added ISR member that hasn't acked yet must still appear
				// (acked 0, lag == LastSeq). Cross-referencing the ISR gives the live
				// view: one entry per current backup (ISR minus the primary), its
				// acked seq looked up from the snapshot (0 if it has not acked).
				acked := make(map[string]uint64, len(st.Peers))
				for _, p := range st.Peers {
					acked[p.Peer] = p.Acked
				}
				for _, node := range isr {
					if node == sm.Primary {
						continue // the primary is not its own backup
					}
					a := acked[node]
					var lag uint64
					if st.LastSeq > a {
						lag = st.LastSeq - a
					}
					sm.Backups = append(sm.Backups, replPeerLag{Node: node, Acked: a, Lag: lag})
				}
			}
		} else {
			// Raft mode: coarse leadership only. The ISR/lag fields stay zero — raft
			// replication health is not exposed per-follower on this surface.
			sm.Mode = ReplicationModeRaft
			sm.IsPrimary = s.IsLeader()
			if addr := n.raftToServerAddr(s.LeaderAddr()); addr != "" {
				sm.Primary = addr
			}
		}
		out.Shards = append(out.Shards, sm)
	}
	body, err := json.Marshal(out)
	if err != nil {
		return nil, fmt.Errorf("cluster: __repl_metrics__ marshal: %w", err)
	}
	return body, nil
}
