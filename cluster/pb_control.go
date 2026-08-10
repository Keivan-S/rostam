// SPDX-License-Identifier: Apache-2.0

package cluster

import (
	"time"

	"github.com/rostamlabs/rostam/shard/pbisr"
)

// metaControl is the MetaRaft-authoritative implementation of pbisr.Control: it
// reads a shard's (epoch, primary, ISR) from the replicated MetaFSM. It is the
// production Control the primary-backup engine consults (tests inject a fake).
//
// minISR is the durability floor the engine enforces (Propose refuses to ack
// when the current ISR is below it). In cluster-level PB mode it is one value
// for every shard; a MetaRaft-ratified per-shard floor is a later refinement
// (shard/pbisr/DESIGN.md H3).
//
// This adapter is a pure read view — it holds NO lease and drives NO failover.
// A static primary stays valid via the Plan-1 self-lease; lease renewal and
// automatic promotion are deferred to a post-gate plan.
type metaControl struct {
	fsm    *MetaFSM
	minISR int
}

func newMetaControl(fsm *MetaFSM, minISR int) *metaControl {
	return &metaControl{fsm: fsm, minISR: minISR}
}

var _ pbisr.Control = (*metaControl)(nil)

func (c *metaControl) Epoch(shard int) uint64   { return c.fsm.ShardEpoch(shard) }
func (c *metaControl) Primary(shard int) string { return c.fsm.ShardPrimary(shard) }
func (c *metaControl) ISR(shard int) []string   { return c.fsm.ShardISR(shard) }
func (c *metaControl) MinISR(shard int) int     { return c.minISR }

// pbShardSeed is one shard's initial primary-backup control state.
type pbShardSeed struct {
	ShardID int
	Primary string
	ISR     []string
}

// pbShardControlSeeds derives each shard's initial (primary, ISR) from the
// placement table: the first owner is the primary, all owners form the ISR.
// Shards with no owners are skipped. Pure — no side effects.
func pbShardControlSeeds(placement [][]string) []pbShardSeed {
	seeds := make([]pbShardSeed, 0, len(placement))
	for shardID, owners := range placement {
		if len(owners) == 0 {
			continue
		}
		seeds = append(seeds, pbShardSeed{
			ShardID: shardID,
			Primary: owners[0],
			ISR:     append([]string(nil), owners...),
		})
	}
	return seeds
}

// shardControlProposer is the leader-only meta-Raft surface bootstrap needs;
// *MetaRaft satisfies it.
type shardControlProposer interface {
	ApplySetShardSeed(shardID int, epoch uint64, primary string, isr []string, timeout time.Duration) error
}

var _ shardControlProposer = (*MetaRaft)(nil)

// bootstrapPBShardControl seeds the primary-backup control plane for a set of
// shards: ONE atomic meta-Raft entry per shard carrying (epoch, primary, full ISR).
//
// It used to commit two sequential entries — OpSetShardEpoch then OpSetShardISR —
// and that was an ACKED-WRITE-LOSS defect, not a style wart. OpSetShardEpoch resets
// the ISR to {primary}, so between the two entries the cluster's COMMITTED state
// held a SINGLETON ISR. A primary that had applied the first entry but not yet the
// second read ISR=[self] from its local FSM, passed the MinISR floor, computed an
// EMPTY peer set, and committed writes on itself alone while the cluster had
// already committed the full ISR. Kill that primary and every one of those acked
// writes is gone — the failover high-water gate promotes a survivor that never saw
// them, and it is right to: the "full-ISR commit ⇒ every in-sync member holds every
// acked write" precondition was simply false. One entry, one apply: the
// intermediate state does not exist, so no read of it can exist either.
//
// Returns the first error — a follower's ErrNotLeader aborts so the caller retries
// on the current meta leader.
func bootstrapPBShardControl(p shardControlProposer, seeds []pbShardSeed, epoch uint64, timeout time.Duration) error {
	for _, s := range seeds {
		if err := p.ApplySetShardSeed(s.ShardID, epoch, s.Primary, s.ISR, timeout); err != nil {
			return err
		}
	}
	return nil
}
