// SPDX-License-Identifier: Apache-2.0

package cluster

import (
	"context"
	"fmt"
	"strings"
	"time"

	hraft "github.com/hashicorp/raft"

	"github.com/rostamlabs/rostam/ops"
)

// catalogGenTick is the poll interval of the all-nodes-applied cutover gate,
// matching the write-consistency / rebalance catch-up loops' 20ms cadence.
const catalogGenTick = 20 * time.Millisecond

// ErrCutoverGateTimeout reports that waitAllNodesCatalogGen reached its deadline
// before EVERY node's local catalog reported the wanted generation. It names the
// nodes that did NOT confirm (errored/unreachable, or still reporting an older
// gen) so the reshard coordinator can LOG which nodes lagged: the gate is
// best-effort (the coordinator proceeds), but the residual is observable, not
// silent. The "cluster: cutover gate" prefix is stable/greppable like the other
// cluster sentinels.
type ErrCutoverGateTimeout struct {
	Collection  string
	WantGen     uint32
	Timeout     time.Duration
	Unconfirmed []string // node IDs that did not report WantGen by the deadline
}

func (e *ErrCutoverGateTimeout) Error() string {
	return fmt.Sprintf(
		"cluster: cutover gate for %q gen %d timed out after %s; nodes not confirmed: %s",
		e.Collection, e.WantGen, e.Timeout, strings.Join(e.Unconfirmed, ","),
	)
}

// CollectionPartitions returns the partition count for a collection from this
// node's local meta-Raft FSM (a lock-guarded in-memory map read: no network, no
// consensus). Consistent once the setting log entry has been applied locally.
// Returns (0, false) in single-node mode (no meta-Raft) or when the collection
// has no catalog entry (i.e. single-partition).
//
// Reads are eventually consistent across nodes: a node that has not yet applied
// a recent catalog write (a follower mid-replication, or a freshly (re)joined
// node still replaying) returns (0, false) for a collection that is actually
// partitioned. Callers that just created the collection are covered by
// SetCollectionPartitions' read-your-writes wait (waitLocalCatalog); other nodes
// converge within one replication round-trip. During that window, embedded
// routing treats the collection as single-partition — searches hit the empty
// logical collection and return empty results with no error.
func (n *Node) CollectionPartitions(collection string) (uint32, bool) {
	if n.meta == nil {
		return 0, false
	}
	return n.meta.FSM.CatalogLookup(collection)
}

// CollectionPartitionsGen is CollectionPartitions plus the collection's partition
// generation (gen-aware routing reads this). Returns (0,0,false) in single-node
// mode (no meta-Raft) or when the collection has no catalog entry. A partitioned
// collection that has never been resplit reports generation 0.
func (n *Node) CollectionPartitionsGen(collection string) (uint32, uint32, bool) {
	if n.meta == nil {
		return 0, 0, false
	}
	return n.meta.FSM.CatalogLookupGen(collection)
}

// WaitAllNodesCatalogGen is the exported entry point for the all-nodes-applied
// cutover gate, invoked by the embedded reshard coordinator (a different package)
// after the Phase-4 catalog flip and before retiring the old generation. It simply
// delegates to waitAllNodesCatalogGen; on timeout it returns the typed
// *ErrCutoverGateTimeout (its Error() carries the unconfirmed node IDs) so the
// coordinator can log the residual and proceed. See waitAllNodesCatalogGen.
func (n *Node) WaitAllNodesCatalogGen(collection string, wantGen uint32, timeout time.Duration) error {
	return n.waitAllNodesCatalogGen(collection, wantGen, timeout)
}

// waitAllNodesCatalogGen blocks until EVERY node in the cluster reports wantGen
// as the local catalog generation for collection, or timeout elapses. It is the
// all-nodes-applied cutover gate: the reshard coordinator calls it after the
// catalog flip so it can confirm no node still routes to the old generation
// before retiring it. It is NOT on any per-read/write path — only the reshard
// coordinator invokes it (rare), so its poll cost never taxes serving.
//
// The LOCAL node is read directly via CollectionPartitionsGen (no round-trip);
// each REMOTE peer is polled via the __catalog_gen__ admin op through a cached
// peer-forwarding client. A peer that errors / is unreachable / still reports an
// older gen is "not yet confirmed" and the loop simply retries (mirroring the
// write-consistency / rebalance catch-up loops). When EVERY node reports wantGen
// (and OK) it returns nil. At the deadline it returns a typed
// *ErrCutoverGateTimeout naming the unconfirmed nodes so the coordinator can log
// the residual.
//
// Single-node / len(Peers) <= 1: the local node IS all nodes, so the gate is
// trivially satisfied — it returns nil immediately without polling.
//
// collection is canonicalized with ops.CanonicalName (the same key
// CollectionPartitionsGen / the meta-FSM catalog use), so the gen comparison
// matches the catalog exactly; a mismatch would make the gate never satisfy.
func (n *Node) waitAllNodesCatalogGen(collection string, wantGen uint32, timeout time.Duration) error {
	canon := ops.CanonicalName(collection)
	// Single-node / no-peers: the local node is the whole cluster ⇒ trivially
	// satisfied (no remote node could disagree).
	if len(n.cfg.Peers) <= 1 {
		return nil
	}
	deadline := time.Now().Add(timeout)
	for {
		unconfirmed := n.unconfirmedCatalogGen(canon, wantGen)
		if len(unconfirmed) == 0 {
			return nil
		}
		if !time.Now().Before(deadline) {
			return &ErrCutoverGateTimeout{
				Collection:  collection,
				WantGen:     wantGen,
				Timeout:     timeout,
				Unconfirmed: unconfirmed,
			}
		}
		time.Sleep(catalogGenTick)
	}
}

// unconfirmedCatalogGen returns the node IDs (from cfg.Peers) whose local
// catalog does NOT currently report wantGen (with OK) for the canonical
// collection. The local node is read directly; each remote peer via
// __catalog_gen__. An errored/unreachable peer is treated as not-confirmed (it
// is included). collection is already canonical.
func (n *Node) unconfirmedCatalogGen(collection string, wantGen uint32) []string {
	var unconfirmed []string
	for _, p := range n.cfg.Peers {
		var (
			gen uint32
			ok  bool
		)
		if p.NodeID == n.cfg.NodeID {
			_, gen, ok = n.CollectionPartitionsGen(collection)
		} else {
			gen, ok = n.remoteCatalogGen(p.NodeID, collection)
		}
		if !ok || gen != wantGen {
			unconfirmed = append(unconfirmed, p.NodeID)
		}
	}
	return unconfirmed
}

// remoteCatalogGen fetches a peer's LOCAL catalog generation for the canonical
// collection over the network via the __catalog_gen__ admin op, reusing the
// cached peer-forwarding client. Returns (0, false) on any transport / decode
// error or unresolved address (treated as not-confirmed by the caller, which
// then retries) — mirroring remoteOwnerStatus's degrade-to-zero behaviour.
func (n *Node) remoteCatalogGen(nodeID, collection string) (uint32, bool) {
	addr := n.serverAddrFor(nodeID)
	if addr == "" {
		return 0, false
	}
	cl, err := n.peerClient(addr)
	if err != nil {
		return 0, false
	}
	args, err := gobEncode(catalogGenReq{Collection: collection})
	if err != nil {
		return 0, false
	}
	ctx, cancel := context.WithTimeout(context.Background(), catalogGenTick)
	defer cancel()
	raw, err := cl.Call(ctx, opCatalogGenName, args)
	if err != nil {
		return 0, false
	}
	var reply catalogGenReply
	if err := gobDecode(raw, &reply); err != nil {
		return 0, false
	}
	return reply.Gen, reply.OK
}

// SetCollectionPartitions durably records a collection's partition count in the
// meta-Raft catalog. It can be called on any node: a non-leader forwards the
// write to the meta-Raft leader via the __set_catalog__ admin op (mirroring the
// __rebalance__ trigger), so the write reaches consensus regardless of which
// node issued it. Returns errNoMeta in single-node mode.
func (n *Node) SetCollectionPartitions(collection string, p, gen uint32, timeout time.Duration) error {
	if n.meta == nil {
		return errNoMeta
	}
	// Forward to the meta leader if we are not it. The leader's handler applies
	// the entry locally (ApplySetCatalogEntry), never re-entering this branch, so
	// there is no forwarding loop.
	if n.meta.Raft.State() != hraft.Leader {
		addr := n.metaLeaderServerAddr()
		if addr == "" || addr == n.serverAddrFor(n.cfg.NodeID) {
			return fmt.Errorf("cluster: SetCollectionPartitions: no meta-Raft leader yet")
		}
		cl, err := n.peerClient(addr)
		if err != nil {
			return err
		}
		args, err := gobEncode(setCatalogReq{Collection: collection, Partitions: p, Generation: gen})
		if err != nil {
			return err
		}
		ctx, cancel := context.WithTimeout(context.Background(), timeout)
		defer cancel()
		if _, err = cl.Call(ctx, opSetCatalogName, args); err != nil {
			return err
		}
	} else {
		if err := n.meta.ApplySetCatalogEntry(collection, p, gen, timeout); err != nil {
			return err
		}
	}
	// Read-your-writes: ensure THIS node's local meta-FSM reflects the write before
	// returning, so a subsequent local CollectionPartitions read (e.g. the next
	// VectorInsert routing) sees it. Immediate on the leader (the entry is applied
	// before ApplySetCatalogEntry returns); brief replication wait on a follower
	// whose forwarded write was acked by the leader before local apply. Bounded by
	// timeout.
	return n.waitLocalCatalog(collection, p, gen, timeout)
}

// CollectionReshard returns a collection's online-reshard state from this node's
// local meta-Raft FSM (a lock-guarded in-memory map read: no network, no
// consensus). ok=true iff the collection is actively resharding (Status!=0).
// Returns (zero, false) in single-node mode (no meta-Raft) or for a Stable /
// absent collection. Eventual-consistency caveats mirror CollectionPartitions.
func (n *Node) CollectionReshard(collection string) (ReshardEntry, bool) {
	if n.meta == nil {
		return ReshardEntry{}, false
	}
	return n.meta.FSM.CatalogReshardLookup(collection)
}

// SetCollectionReshard durably records a collection's online-reshard state in the
// meta-Raft catalog. Like SetCollectionPartitions it can be called on any node: a
// non-leader forwards the write to the meta-Raft leader via the __set_reshard__
// admin op, so the write reaches consensus regardless of which node issued it.
// On return THIS node's local FSM already reflects the write (read-your-writes).
// Returns errNoMeta in single-node mode.
func (n *Node) SetCollectionReshard(collection string, e ReshardEntry, timeout time.Duration) error {
	if n.meta == nil {
		return errNoMeta
	}
	if n.meta.Raft.State() != hraft.Leader {
		addr := n.metaLeaderServerAddr()
		if addr == "" || addr == n.serverAddrFor(n.cfg.NodeID) {
			return fmt.Errorf("cluster: SetCollectionReshard: no meta-Raft leader yet")
		}
		cl, err := n.peerClient(addr)
		if err != nil {
			return err
		}
		args, err := gobEncode(setReshardReq{
			Collection: collection,
			Status:     e.Status,
			TargetP:    e.TargetP,
			TargetGen:  e.TargetGen,
			SourceP:    e.SourceP,
			SourceGen:  e.SourceGen,
		})
		if err != nil {
			return err
		}
		ctx, cancel := context.WithTimeout(context.Background(), timeout)
		defer cancel()
		if _, err = cl.Call(ctx, opSetReshardName, args); err != nil {
			return err
		}
	} else {
		if err := n.meta.ApplySetCatalogReshard(collection, e, timeout); err != nil {
			return err
		}
	}
	return n.waitLocalReshard(collection, e, timeout)
}

// waitLocalReshard blocks until this node's local meta-FSM reflects the reshard
// state just written for collection, or timeout elapses. Gives read-your-writes
// semantics. A Stable write (Status 0) is satisfied once the lookup reports a
// non-resharding (absent / Status-0) entry.
func (n *Node) waitLocalReshard(collection string, want ReshardEntry, timeout time.Duration) error {
	deadline := time.Now().Add(timeout)
	for {
		got, ok := n.meta.FSM.CatalogReshardLookup(collection)
		if want.Status == 0 {
			if !ok {
				return nil
			}
		} else if ok && got == want {
			return nil
		}
		if time.Now().After(deadline) {
			return fmt.Errorf("cluster: reshard write for %q not locally applied within %s", collection, timeout)
		}
		time.Sleep(5 * time.Millisecond)
	}
}

// ResolveAlias returns the canonical target collection an alias resolves to from
// this node's local meta-Raft FSM (a lock-guarded in-memory map read: no network,
// no consensus). ok=false means the name is not an alias. Returns ("", false) in
// single-node mode (no meta-Raft). Eventual-consistency caveats mirror
// CollectionPartitions.
func (n *Node) ResolveAlias(name string) (string, bool) {
	if n.meta == nil {
		return "", false
	}
	return n.meta.FSM.AliasLookup(name)
}

// ListAliases returns a snapshot copy of this node's local alias map
// (alias→target). Local read (no Raft). Returns an empty map in single-node mode.
func (n *Node) ListAliases() map[string]string {
	if n.meta == nil {
		return map[string]string{}
	}
	return n.meta.FSM.AliasSnapshot()
}

// SetAliases durably applies a batch of alias mutations to the meta-Raft catalog
// as one atomic log entry. Like SetCollectionReshard it can be called on any
// node: a non-leader forwards the write to the meta-Raft leader via the
// __set_aliases__ admin op, so the write reaches consensus regardless of which
// node issued it. On return THIS node's local FSM already reflects the batch
// (read-your-writes). Returns errNoMeta in single-node mode.
func (n *Node) SetAliases(actions []AliasAction, timeout time.Duration) error {
	if n.meta == nil {
		return errNoMeta
	}
	if n.meta.Raft.State() != hraft.Leader {
		addr := n.metaLeaderServerAddr()
		if addr == "" || addr == n.serverAddrFor(n.cfg.NodeID) {
			return fmt.Errorf("cluster: SetAliases: no meta-Raft leader yet")
		}
		cl, err := n.peerClient(addr)
		if err != nil {
			return err
		}
		args, err := gobEncode(setAliasReq{Actions: actions})
		if err != nil {
			return err
		}
		ctx, cancel := context.WithTimeout(context.Background(), timeout)
		defer cancel()
		if _, err = cl.Call(ctx, opSetAliasesName, args); err != nil {
			return err
		}
	} else {
		if err := n.meta.ApplySetAliasBatch(actions, timeout); err != nil {
			return err
		}
	}
	return n.waitLocalAliases(actions, timeout)
}

// waitLocalAliases blocks until this node's local meta-FSM reflects EACH action
// in the batch just written, or timeout elapses. Gives read-your-writes
// semantics for the SPECIFIC value written: a create is satisfied once
// AliasLookup(alias) reports exactly the written canonical (not merely "any
// value" — a slow follower could otherwise return on a stale prior target); a
// delete is satisfied once AliasLookup(alias) reports a miss.
func (n *Node) waitLocalAliases(actions []AliasAction, timeout time.Duration) error {
	deadline := time.Now().Add(timeout)
	for {
		if aliasBatchApplied(n.meta.FSM, actions) {
			return nil
		}
		if time.Now().After(deadline) {
			return fmt.Errorf("cluster: alias batch not locally applied within %s", timeout)
		}
		time.Sleep(5 * time.Millisecond)
	}
}

// aliasBatchApplied reports whether every action's effect is visible in the FSM.
// For a batch that mutates the same alias twice (e.g. a swap), only the LAST
// action for that alias is the authoritative final state — but iterating all
// actions still converges, because the FSM holds the post-batch state and the
// final action for each alias is the one that matches.
func aliasBatchApplied(fsm *MetaFSM, actions []AliasAction) bool {
	// Compute the final intended state per alias (last action wins), then verify.
	final := make(map[string]AliasAction, len(actions))
	for _, a := range actions {
		final[a.Alias] = a
	}
	for alias, a := range final {
		got, ok := fsm.AliasLookup(alias)
		if a.Delete {
			if ok {
				return false
			}
		} else if !ok || got != a.Canonical {
			return false
		}
	}
	return true
}

// waitLocalCatalog blocks until this node's local meta-FSM shows the given
// partition count for collection, or timeout elapses. Polling for ok && got == p
// gives read-your-writes semantics for the value just written.
func (n *Node) waitLocalCatalog(collection string, p, gen uint32, timeout time.Duration) error {
	deadline := time.Now().Add(timeout)
	for {
		if got, g, ok := n.meta.FSM.CatalogLookupGen(collection); ok && got == p && g == gen {
			return nil
		}
		if time.Now().After(deadline) {
			return fmt.Errorf("cluster: catalog write for %q not locally applied within %s", collection, timeout)
		}
		time.Sleep(5 * time.Millisecond)
	}
}
