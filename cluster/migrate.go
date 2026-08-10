// SPDX-License-Identifier: Apache-2.0

package cluster

import (
	"errors"
	"fmt"
	"time"

	hraft "github.com/hashicorp/raft"
)

// Online-rebalancing slice 2: migrate a single shard from its current owner set
// to a target owner set as one coordinated gain-then-release operation, keeping
// the shard served throughout and never dropping an owner before its
// replacement has caught up (design §4). Slice 3's coordinator drives many of
// these from a membership-change diff.

// MigrationCluster is the in-process view a single-shard migration needs: every
// member node by id, plus each node's Raft transport address. The migration
// drives membership changes on the shard's leader and dynamic shard lifecycle on
// the gained/lost nodes through these handles. A future multi-process
// coordinator (slice 4 operator surface) will back the same sequence with
// remote admin calls instead of direct *Node handles.
type MigrationCluster struct {
	Nodes     map[string]shardAdmin // nodeID -> control surface (local *Node or remoteNode)
	RaftAddr  map[string]string     // nodeID -> Raft transport addr
	NumShards int                   // cluster shard count (immutable)
	// Local is the node id of the in-process *Node (the coordinator's own node),
	// preferred for placement reads to avoid a network round-trip. "" = no
	// preference (all-local, in-process embeddings).
	Local string
}

// MigrationClusterFromPeers builds an in-process MigrationCluster from a node
// id->*Node map and a peer list (for the Raft addresses). Used by tests and any
// single-process embedding; the network-driven coordinator builds its Nodes map
// from a local *Node plus remoteNode peers instead.
func MigrationClusterFromPeers(nodes map[string]*Node, peers []Peer) MigrationCluster {
	admins := make(map[string]shardAdmin, len(nodes))
	numShards := 0
	for id, nd := range nodes {
		admins[id] = nd
		if nd != nil && nd.cfg.NumShards > numShards {
			numShards = nd.cfg.NumShards
		}
	}
	raftAddr := make(map[string]string, len(peers))
	for _, p := range peers {
		raftAddr[p.NodeID] = p.RaftAddr
	}
	return MigrationCluster{Nodes: admins, RaftAddr: raftAddr, NumShards: numShards}
}

// MigrateShard moves shardID's owner set to target. It runs the gain-then-release
// sequence: pre-create the gained replicas and add them as voters, wait for them
// to catch up via Raft, advance placement (add new owners, then drop old), and
// finally remove the lost replicas from the group and tear down their stores.
// At every step the shard retains a caught-up owner, so reads/writes continue.
//
// timeout bounds each blocking sub-step (leader discovery, catch-up, placement
// commit) independently. The operation is safe to retry: it recomputes the
// current owner set from placement each call, so a partially-applied migration
// converges on a re-run.
func (mc MigrationCluster) MigrateShard(shardID int, target []string, timeout time.Duration) error {
	if len(target) == 0 {
		return errors.New("cluster: migrate shard: empty target owner set")
	}
	for _, id := range target {
		if mc.Nodes[id] == nil {
			return fmt.Errorf("cluster: migrate shard %d: target node %q not in cluster", shardID, id)
		}
	}

	current := mc.currentOwners(shardID)
	gained := subtract(target, current)
	lost := subtract(current, target)
	if len(gained) == 0 && len(lost) == 0 {
		// Already at target; ensure placement reflects it and return.
		return mc.commitPlacement(shardID, target, timeout)
	}

	// --- Gain: pre-create joining stores, then the leader adds them as voters. ---
	leaderID, err := mc.findLeader(shardID, current, timeout)
	if err != nil {
		return err
	}
	for _, b := range gained {
		if err := mc.Nodes[b].addOwner(shardID); err != nil {
			return fmt.Errorf("cluster: migrate shard %d: add owner %q: %w", shardID, b, err)
		}
		if err := mc.Nodes[leaderID].addVoter(shardID, b, mc.RaftAddr[b]); err != nil {
			return fmt.Errorf("cluster: migrate shard %d: add voter %q: %w", shardID, b, err)
		}
	}

	// --- Catch up: each gained replica must reach the leader's log tail before
	// we count it as live and start removing old owners. ---
	if err := mc.waitCaughtUp(shardID, leaderID, gained, timeout); err != nil {
		return err
	}

	// --- Advance placement: add the new owners first (union), so routing can
	// reach them; then commit the final target, dropping the lost owners from
	// routing before they leave the group. ---
	if err := mc.commitPlacement(shardID, union(current, gained), timeout); err != nil {
		return err
	}
	if err := mc.commitPlacement(shardID, target, timeout); err != nil {
		return err
	}

	// --- Release: remove each lost owner from the Raft group, then tear down its
	// store. If a lost node is the current leader, move leadership to a surviving
	// target owner first so no node ever has to remove itself. ---
	for _, a := range lost {
		if mc.Nodes[a].status(shardID).IsLeader {
			dst := target[0] // a caught-up surviving owner
			if err := mc.Nodes[a].transferLeadership(shardID, dst, mc.RaftAddr[dst]); err != nil {
				return fmt.Errorf("cluster: migrate shard %d: transfer leadership %q->%q: %w", shardID, a, dst, err)
			}
		}
		newLeader, err := mc.findLeader(shardID, target, timeout)
		if err != nil {
			return err
		}
		if err := mc.Nodes[newLeader].removeVoter(shardID, a); err != nil {
			return fmt.Errorf("cluster: migrate shard %d: remove voter %q: %w", shardID, a, err)
		}
		if err := mc.Nodes[a].removeOwner(shardID); err != nil {
			return fmt.Errorf("cluster: migrate shard %d: remove owner %q: %w", shardID, a, err)
		}
	}
	return nil
}

// numShards returns the cluster's (immutable) shard count.
func (mc MigrationCluster) numShards() int { return mc.NumShards }

// currentOwners reads shardID's owner set from a member's routing view (all
// members hold the same replicated placement). Prefers the local node to avoid
// a network round-trip when the coordinator runs in-cluster.
func (mc MigrationCluster) currentOwners(shardID int) []string {
	for _, nd := range mc.orderedNodes() {
		p := nd.placementCopy()
		if shardID < len(p) && len(p[shardID]) > 0 {
			return append([]string(nil), p[shardID]...)
		}
	}
	return nil
}

// orderedNodes returns the member admins with the local node (if set) first, so
// reads prefer the in-process node over a remote round-trip.
func (mc MigrationCluster) orderedNodes() []shardAdmin {
	out := make([]shardAdmin, 0, len(mc.Nodes))
	if mc.Local != "" {
		if nd := mc.Nodes[mc.Local]; nd != nil {
			out = append(out, nd)
		}
	}
	for id, nd := range mc.Nodes {
		if nd == nil || id == mc.Local {
			continue
		}
		out = append(out, nd)
	}
	return out
}

// findLeader polls until one of candidates leads shardID, or timeout elapses.
func (mc MigrationCluster) findLeader(shardID int, candidates []string, timeout time.Duration) (string, error) {
	deadline := time.Now().Add(timeout)
	for {
		for _, id := range candidates {
			if nd := mc.Nodes[id]; nd != nil && nd.status(shardID).IsLeader {
				return id, nil
			}
		}
		if !time.Now().Before(deadline) {
			return "", fmt.Errorf("cluster: migrate shard %d: no leader among %v within %s", shardID, candidates, timeout)
		}
		time.Sleep(20 * time.Millisecond)
	}
}

// waitCaughtUp blocks until every gained replica's applied Raft index reaches
// the leader's last log index (captured once, after the voters were added).
func (mc MigrationCluster) waitCaughtUp(shardID int, leaderID string, gained []string, timeout time.Duration) error {
	target := mc.Nodes[leaderID].status(shardID).LastIndex
	deadline := time.Now().Add(timeout)
	for {
		caught := true
		for _, b := range gained {
			if mc.Nodes[b].status(shardID).AppliedIndex < target {
				caught = false
				break
			}
		}
		if caught {
			return nil
		}
		if !time.Now().Before(deadline) {
			return fmt.Errorf("cluster: migrate shard %d: replicas %v did not catch up to index %d within %s", shardID, gained, target, timeout)
		}
		time.Sleep(20 * time.Millisecond)
	}
}

// commitPlacement commits owners for shardID to meta-Raft (on whichever node is
// the meta leader) and updates every node's local routing view. It retries
// across leadership churn within timeout: ErrNotLeader rotates to the next node,
// and a transient apply failure (leadership lost mid-apply, or a freshly elected
// leader whose FSM has not yet replayed the membership entry) is retried after a
// short backoff rather than aborting the migration.
func (mc MigrationCluster) commitPlacement(shardID int, owners []string, timeout time.Duration) error {
	numShards := mc.numShards()
	deadline := time.Now().Add(timeout)
	committed := false
	metaExists := false
	var lastErr = errNoMetaLeader
	for {
		for _, nd := range mc.Nodes {
			if nd == nil {
				continue
			}
			err := nd.commitMetaPlacement(shardID, numShards, owners, timeout)
			switch {
			case err == nil:
				committed = true
				metaExists = true
			case errors.Is(err, errNoMeta):
				// This node has no meta-Raft; ignore it.
			case errors.Is(err, hraft.ErrNotLeader):
				metaExists = true // meta exists, just not the leader here
			default:
				metaExists = true
				lastErr = err // a real apply error (record for the timeout msg)
			}
			if committed {
				break
			}
		}
		// Done when committed, or when no node has meta at all (single-node /
		// no-meta deployment: routing-only update below).
		if committed || !metaExists {
			break
		}
		if !time.Now().Before(deadline) {
			return fmt.Errorf("cluster: migrate shard %d: commit placement: %w", shardID, lastErr)
		}
		time.Sleep(20 * time.Millisecond)
	}
	// Mirror to every node's hot-path routing view (also the only step in a
	// single-node / no-meta deployment, so the operation stays meaningful).
	for _, nd := range mc.Nodes {
		if nd != nil {
			_ = nd.setPlacement(shardID, owners) //nolint:errcheck // best-effort routing push
		}
	}
	return nil
}

// errNoMetaLeader is the placeholder cause when a placement commit never found a
// meta leader within its deadline.
var errNoMetaLeader = errors.New("no meta-Raft leader accepted the commit")

// subtract returns the elements of a not present in b (set difference a\b).
func subtract(a, b []string) []string {
	in := make(map[string]struct{}, len(b))
	for _, x := range b {
		in[x] = struct{}{}
	}
	var out []string
	for _, x := range a {
		if _, ok := in[x]; !ok {
			out = append(out, x)
		}
	}
	return out
}

// union returns the unique elements of a and b, preserving a's order then b's.
func union(a, b []string) []string {
	seen := make(map[string]struct{}, len(a)+len(b))
	out := make([]string, 0, len(a)+len(b))
	for _, x := range append(append([]string(nil), a...), b...) {
		if _, ok := seen[x]; ok {
			continue
		}
		seen[x] = struct{}{}
		out = append(out, x)
	}
	return out
}
