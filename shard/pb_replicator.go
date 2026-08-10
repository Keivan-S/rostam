// SPDX-License-Identifier: Apache-2.0

package shard

import (
	"errors"
	"strconv"
	"time"

	"github.com/rostamlabs/rostam/raft"
	"github.com/rostamlabs/rostam/shard/pbisr"
)

// pbReplicator adapts a primary-backup/ISR pbisr.Engine to the shard.replicator
// seam, so a Store can run on primary-backup replication exactly where it would
// otherwise hold a *raft.Node. Raft-shaped methods map onto PB semantics:
// Propose→ApplyIndexed, primary/lease→IsLeader/VerifyLeader, Committed→CommitIndex.
// Control-plane-driven membership changes are ErrPBUnimplemented for now.
type pbReplicator struct {
	nodeID string
	shard  int
	e      *pbisr.Engine
	ctrl   pbisr.Control
}

func newPBReplicator(nodeID string, shard int, e *pbisr.Engine, ctrl pbisr.Control) *pbReplicator {
	return &pbReplicator{nodeID: nodeID, shard: shard, e: e, ctrl: ctrl}
}

var _ replicator = (*pbReplicator)(nil)

func (r *pbReplicator) ApplyIndexed(data []byte, timeout time.Duration) (any, uint64, error) {
	// ProposeDeadline, not Propose(ctx): this seam has no cancellation source —
	// only a timeout — and the deadline variant keeps the per-write context +
	// timer allocations off the hot path (see pbisr.ProposeDeadline).
	result, seq, err := r.e.ProposeDeadline(data, timeout)
	if err != nil {
		// A fenced/non-primary write maps onto the seam's NotLeader signal so the
		// Store redirects exactly as it does for Raft.
		if errors.Is(err, pbisr.ErrNotPrimary) || errors.Is(err, pbisr.ErrLeaseExpired) {
			return nil, 0, raft.ErrNotLeader
		}
		// Everything else is a non-fenced, non-durable outcome — the write may have
		// applied locally but did not durably commit: ErrReplicationTimeout (the
		// full ISR did not ack in time), or ErrPipelineStalled (a wedged pipeline
		// refused admission below full ISR). These are retryable/unknown, NOT a
		// leadership change, so surface them UNCHANGED; the Store must not redirect
		// the writer on them the way it does for ErrNotLeader.
		return nil, 0, err
	}
	return decodePBResult(result), seq, nil
}

func (r *pbReplicator) IsLeader() bool { return r.ctrl.Primary(r.shard) == r.nodeID }

func (r *pbReplicator) LeaderAddr() string { return r.ctrl.Primary(r.shard) }

func (r *pbReplicator) CommitIndex() uint64  { return r.e.Committed() }
func (r *pbReplicator) AppliedIndex() uint64 { return r.e.Committed() }

func (r *pbReplicator) LastIndex() uint64 {
	s, a := r.e.LastSeq(), r.e.LastApplied()
	if s >= a {
		return s
	}
	return a
}

func (r *pbReplicator) VerifyLeader() error {
	if r.ctrl.Primary(r.shard) == r.nodeID && r.e.LeaseValid() {
		return nil
	}
	return raft.ErrNotLeader
}

func (r *pbReplicator) Barrier(timeout time.Duration) error {
	// The primary applies each write locally BEFORE acking, so its local state is
	// at least as fresh as Committed — confirming still-primary suffices.
	return r.VerifyLeader()
}

func (r *pbReplicator) AddVoter(id, addr string, prevIndex uint64, timeout time.Duration) error {
	return pbisr.ErrPBUnimplemented
}

func (r *pbReplicator) RemoveServer(id string, prevIndex uint64, timeout time.Duration) error {
	return pbisr.ErrPBUnimplemented
}

func (r *pbReplicator) LeadershipTransferToServer(id, addr string) error {
	return pbisr.ErrPBUnimplemented
}

func (r *pbReplicator) Stats() map[string]string {
	// The snapshot-transfer gauges are reported here because the write
	// STALL is a real operational cost of this design (SnapshotFSM runs under the
	// engine quiesce) and a cost that is not measurable is a cost that gets
	// discovered in production. snapshotStallMaxNs is the number to alert on — it
	// includes this deployment's vector state, which no key-count estimate does.
	// snapshotPending surfaces the poison fence, i.e. "this shard is refusing to
	// serve and needs a fresh snapshot".
	st := r.e.SnapshotStats()
	return map[string]string{
		"mode":                "pb",
		"epoch":               strconv.FormatUint(r.e.Epoch(), 10),
		"committed":           strconv.FormatUint(r.e.Committed(), 10),
		"lastApplied":         strconv.FormatUint(r.e.LastApplied(), 10),
		"primary":             r.ctrl.Primary(r.shard),
		"snapshotsTaken":      strconv.FormatUint(st.Taken, 10),
		"snapshotsInstalled":  strconv.FormatUint(st.Installed, 10),
		"snapshotStallLastNs": strconv.FormatInt(st.StallLastNs, 10),
		"snapshotStallMaxNs":  strconv.FormatInt(st.StallMaxNs, 10),
		"snapshotPending":     strconv.FormatBool(st.Poisoned),
	}
}

func (r *pbReplicator) Shutdown() error {
	// Stop the engine's per-peer sender goroutines. The engine
	// does not own the Transport, so closing it stays with the transport's owner.
	r.e.Shutdown()
	return nil
}
