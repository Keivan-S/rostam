// SPDX-License-Identifier: Apache-2.0

package cluster

import (
	"fmt"
	"log/slog"
	"time"

	hraft "github.com/hashicorp/raft"
)

// SHARD-GROUP FORMATION AS A CONTROL-PLANE DECISION.
//
// A shard's Raft group has to be CREATED by exactly one of its owners
// (hashicorp raft's BootstrapCluster); the others join by replaying that
// configuration from the leader. The node-level `-bootstrap` flag used to be the
// authority for that decision, and it is the wrong one: a node hosts only the
// shards it OWNS, so as soon as ReplicationFactor < len(members) some shards have
// an owner set that EXCLUDES the bootstrap node. Every owner of such a shard then
// ran with Bootstrap=false, nobody ever called BootstrapCluster, and the group sat
// configuration-less and leaderless FOREVER — writes to any key hashing there hung,
// and since a uniform keyspace touches every shard the whole cluster was unusable.
// RF == len(members) hid it completely, because then every shard includes the
// bootstrap node.
//
// The fix moves the decision into the meta log: the meta leader designates one
// former per shard (State.ShardFormer, write-once), and every node acts on the
// REPLICATED designation for the shards it owns. Two properties come from routing
// it through Raft rather than deciding locally:
//
//   - Exactly one former per shard, agreed by the cluster — not a local guess that
//     two nodes could make simultaneously.
//   - A fresh-disk node rejoining an ESTABLISHED cluster finds the shard already
//     designated (write-once), so it never forms a rival group; it comes up empty
//     and replays from the existing leader, which is the normal join path.

const (
	// shardFormationRetry paces both the designation seeder and the driver. Formation
	// is a once-per-cluster-lifetime event, so this only has to be fast relative to a
	// human noticing startup, not to the write path.
	shardFormationRetry = 500 * time.Millisecond
	// shardFormationDeadline bounds the whole effort so a node can never leak these
	// goroutines. Generous: it must cover a rolling start where the last owner of some
	// shard is minutes behind the first.
	shardFormationDeadline = 5 * time.Minute
)

// seedShardFormers designates one former per shard in the meta log. Runs on EVERY
// node but only commits while this node is the meta leader (the Apply helper
// returns ErrNotLeader otherwise), so whichever node holds meta leadership during
// the formation window does the work and the rest are harmless no-ops. Retries
// until every shard has a designation, then returns.
//
// The former is placement-derived (owners[0]) and therefore deterministic, so
// which node happens to seed does not change the outcome. The FSM's write-once
// rule makes a concurrent or repeated seed idempotent.
func (n *Node) seedShardFormers(p shardFormerProposer, placement [][]string, stop <-chan struct{}) {
	if len(placement) == 0 {
		return
	}
	ticker := time.NewTicker(shardFormationRetry)
	defer ticker.Stop()
	deadline := time.After(shardFormationDeadline)
	for {
		if n.allShardFormersDesignated(placement) {
			return
		}
		for shardID, owners := range placement {
			if len(owners) == 0 {
				continue // unplaced shard: nothing to form
			}
			if n.meta.FSM.ShardFormer(shardID) != "" {
				continue // already designated
			}
			// owners[0] — deterministic over the same membership, and computePlacement
			// gives every node the same owner order.
			if _, err := p.ApplySetShardFormer(shardID, owners[0], 5*time.Second); err != nil {
				break // not the leader (or meta busy): retry the whole sweep later
			}
		}
		if n.allShardFormersDesignated(placement) {
			return
		}
		select {
		case <-stop:
			return
		case <-deadline:
			return
		case <-ticker.C:
		}
	}
}

// allShardFormersDesignated reports whether every PLACED shard has a former in the
// local (replicated) meta state.
func (n *Node) allShardFormersDesignated(placement [][]string) bool {
	for shardID, owners := range placement {
		if len(owners) == 0 {
			continue
		}
		if n.meta.FSM.ShardFormer(shardID) == "" {
			return false
		}
	}
	return true
}

// driveShardFormation forms the Raft groups this node is designated to form. It
// reads the designation from the local replicated meta FSM — never proposes — so
// it works identically on a leader and a follower.
//
// Only ever forms a group that has NO leader: a group the cluster already brought
// up needs nothing, and raft.Node.BootstrapGroup additionally no-ops on existing
// persisted state. Returns once every owned shard has a leader, so on a healthy
// cluster (including every RF == len(members) cluster, which behaves exactly as
// before this existed) it exits after one pass without touching anything.
func (n *Node) driveShardFormation(stop <-chan struct{}) {
	ticker := time.NewTicker(shardFormationRetry)
	defer ticker.Stop()
	deadline := time.After(shardFormationDeadline)
	for {
		if n.formAssignedShards() {
			return
		}
		select {
		case <-stop:
			return
		case <-deadline:
			// Loud: a shard still without a leader here is the failure this whole
			// mechanism exists to prevent, and it is invisible on every other surface
			// until a client's write hangs on it.
			for shardID, s := range n.snapshotShards() {
				if s != nil && s.LeaderAddr() == "" {
					slog.Warn("shard still has no leader after formation deadline; writes to it will hang",
						"component", "cluster", "shard", shardID,
						"former", n.meta.FSM.ShardFormer(shardID), "node", n.cfg.NodeID)
				}
			}
			return
		case <-ticker.C:
		}
	}
}

// formAssignedShards makes one pass over this node's shards, forming any that it
// is designated to form and that still have no leader. Returns whether every
// hosted shard now reports a leader.
func (n *Node) formAssignedShards() bool {
	allLed := true
	// snapshotShards, not n.shards: online rebalancing mutates the slice's slots at
	// runtime (AddShardOwner/RemoveShardOwner in cluster/rebalance.go write them
	// under n.shardMu), so a raw range here is a data race against any live
	// migration. Every *shard.Store call below happens AFTER the lock is released,
	// which is the discipline the rest of the file's background readers follow.
	for shardID, s := range n.snapshotShards() {
		if s == nil {
			continue // not an owner of this shard (or just removed by a rebalance)
		}
		if s.LeaderAddr() != "" {
			continue // already up
		}
		allLed = false
		if n.meta.FSM.ShardFormer(shardID) != n.cfg.NodeID {
			continue // someone else forms this one; wait for their config to replicate
		}
		servers := n.shardRaftServers(shardID)
		if len(servers) == 0 {
			continue // placement not visible yet
		}
		if err := s.BootstrapGroup(servers); err != nil {
			slog.Warn("shard group formation failed; will retry",
				"component", "cluster", "shard", shardID, "err", err)
			continue
		}
		slog.Info("formed shard Raft group",
			"component", "cluster", "shard", shardID, "voters", len(servers), "node", n.cfg.NodeID)
	}
	return allLed
}

// shardRaftServers builds the shard's Raft voter list — the same per-shard-suffixed
// server set buildShardConfig installs as RaftPeers, so a group formed here gets
// byte-identical configuration to one formed at construction.
func (n *Node) shardRaftServers(shardID int) []hraft.Server {
	// ownersFor takes n.shardMu and returns a COPY — n.placement is mutated at
	// runtime by SetShardPlacement during a migration, so reading it directly here
	// would race. Must not be called while already holding shardMu (the lock is
	// not reentrant-safe for this composition).
	owners := n.ownersFor(shardID)
	if len(owners) == 0 {
		return nil
	}
	// Same suffix buildShardConfig uses, so the configuration installed here is
	// byte-identical to a construction-time one.
	return toRaftServers(peersForOwners(n.cfg.Peers, owners), fmt.Sprintf("-shard-%04d", shardID))
}

// shardFormerProposer is the meta-Raft surface the seeder needs, narrowed to one
// method so tests can drive formation without a live meta group.
type shardFormerProposer interface {
	ApplySetShardFormer(shardID int, node string, timeout time.Duration) (bool, error)
}
