// SPDX-License-Identifier: Apache-2.0

package cluster

import (
	"context"
	"fmt"
	"sync"
	"time"

	hraft "github.com/hashicorp/raft"
)

// The primary-liveness BEACON (producer side).
//
// Each node, when PBAutoFailover is on, runs a pbBeacon: every renewInterval it
// commits an OpShardLeaseRenew carrying every shard it currently primaries. The
// meta leader observes these commits (via the FSM leaseRenewObserver → its
// pbFailoverTracker) and treats a shard whose beacon has gone silent past the
// failover timeout as a dead primary. COMMITTING the beacon IS the quorum-
// connection proof: a node partitioned from the meta quorum cannot land the entry,
// so it stops proving liveness and is (correctly) failed over — no separate
// confirmMetaView gate is needed on this path.

// submitShardLeaseRenew commits (or forwards) a batched primary-liveness beacon.
// On the meta leader it applies locally via ApplyShardLeaseRenew; on a follower it
// forwards to the leader over the __pb_lease_renew__ admin op — mirroring
// SetCollectionPartitions exactly, so a follower-hosted primary's beacon still
// reaches consensus. The leader's handler applies it locally (never re-entering
// this branch), so there is no forwarding loop. An empty batch is a no-op.
func (n *Node) submitShardLeaseRenew(renews []ShardEpochPair, timeout time.Duration) error {
	if n.meta == nil {
		return errNoMeta
	}
	if len(renews) == 0 {
		return nil
	}
	if n.meta.Raft.State() != hraft.Leader {
		addr := n.metaLeaderServerAddr()
		if addr == "" || addr == n.serverAddrFor(n.cfg.NodeID) {
			return fmt.Errorf("cluster: submitShardLeaseRenew: no meta-Raft leader yet")
		}
		cl, err := n.peerClient(addr)
		if err != nil {
			return err
		}
		args, err := gobEncode(pbLeaseRenewReq{Node: n.cfg.NodeID, Renews: renews})
		if err != nil {
			return err
		}
		ctx, cancel := context.WithTimeout(context.Background(), timeout)
		defer cancel()
		_, err = cl.Call(ctx, opPBLeaseRenewName, args)
		return err
	}
	return n.meta.ApplyShardLeaseRenew(n.cfg.NodeID, renews, timeout)
}

// pbBeacon is the per-node primary-liveness beacon goroutine. It mirrors
// leaseKeeper's lifecycle (start/stop/run/tick). It is started ONLY when
// PBAutoFailover is on — when off, no beacon runs and the meta-Raft log carries no
// OpShardLeaseRenew entries (byte-identical static cluster).
type pbBeacon struct {
	node     *Node
	interval time.Duration
	timeout  time.Duration // per-beacon submit bound (well under the lease TTL)

	mu      sync.Mutex
	started bool
	done    chan struct{}
	stopped chan struct{}
}

// newPBBeacon constructs a beacon for node n committing every interval.
func newPBBeacon(n *Node, interval, timeout time.Duration) *pbBeacon {
	return &pbBeacon{node: n, interval: interval, timeout: timeout}
}

// start spawns the beacon goroutine (no-op if already started). Mirrors
// leaseKeeper.start.
func (b *pbBeacon) start() {
	b.mu.Lock()
	defer b.mu.Unlock()
	if b.started {
		return
	}
	b.started = true
	b.done = make(chan struct{})
	b.stopped = make(chan struct{})
	go b.run()
}

// stop signals the goroutine to exit and blocks until it has. Safe without a
// prior start and safe to call repeatedly. Mirrors leaseKeeper.stop.
func (b *pbBeacon) stop() {
	b.mu.Lock()
	if !b.started {
		b.mu.Unlock()
		return
	}
	done := b.done
	stopped := b.stopped
	b.started = false
	b.mu.Unlock()

	select {
	case <-done:
	default:
		close(done)
	}
	<-stopped
}

func (b *pbBeacon) run() {
	defer close(b.stopped)
	ticker := time.NewTicker(b.interval)
	defer ticker.Stop()
	for {
		select {
		case <-b.done:
			return
		case <-ticker.C:
			b.tick()
		}
	}
}

// tick builds this node's current primary set from the local (replicated) MetaFSM
// and submits it as one beacon. A best-effort submit: a transient failure (no meta
// leader yet, a slow forward) simply skips this interval — the NEXT beacon renews,
// and a SUSTAINED failure is exactly the silence that should trigger failover.
func (b *pbBeacon) tick() {
	n := b.node
	nodeID := n.cfg.NodeID
	// Read the primary set under the FSM's own lock via the accessors (a full
	// State() deep-copy is unnecessary here — we only need primary+epoch per shard).
	st := n.meta.FSM.State()
	var renews []ShardEpochPair
	for shardID, primary := range st.ShardPrimary {
		if primary != nodeID {
			continue
		}
		renews = append(renews, ShardEpochPair{ShardID: shardID, Epoch: st.ShardEpoch[shardID]})
	}
	if len(renews) == 0 {
		return
	}
	_ = n.submitShardLeaseRenew(renews, b.timeout) //nolint:errcheck,gosec // best-effort; sustained failure is the failover signal
}
