// SPDX-License-Identifier: Apache-2.0

package cluster

import (
	"errors"
	"testing"
	"time"

	"github.com/rostamlabs/rostam/shard/pbisr"
)

// The leaseKeeper must gate every renewal on a quorum-confirmed meta
// read. These tests drive tick() directly (deterministic, no goroutine) so the
// barrier decision is observable via engine.LeaseValid().

func barrierEngine(t *testing.T, clock *fakeClock) *pbisr.Engine {
	t.Helper()
	ctrl := newMetaControl(NewMetaFSM(), 1)
	return pbisr.New("n1", 0, ctrl, nil, nil, pbisr.WithClock(clock.now))
}

func seededFSM(t *testing.T, primary string) *MetaFSM {
	t.Helper()
	f := NewMetaFSM()
	applyMeta(t, f, LogEntry{Op: OpSetShardEpoch, ShardID: 0, Epoch: 1, Primary: primary})
	applyMeta(t, f, LogEntry{Op: OpSetShardISR, ShardID: 0, Epoch: 1, ISR: []string{primary}})
	return f
}

// TestLeaseBarrierFailBlocksRenewal: a failing barrier (partitioned from the
// meta quorum) must renew NO lease, so a fenced-but-still-local-primary node
// self-fences instead of double-priming.
func TestLeaseBarrierFailBlocksRenewal(t *testing.T) {
	clock := newFakeClock(0)
	eng := barrierEngine(t, clock)
	f := seededFSM(t, "n1") // local FSM still says n1 is primary

	barrierErr := errors.New("no meta quorum")
	k := newLeaseKeeper(f, "n1", map[int]*pbisr.Engine{0: eng},
		50*time.Millisecond, 5*time.Millisecond, clock.now,
		func(time.Time) error { return barrierErr }, nil, time.Second)

	k.tick()
	if eng.LeaseValid() {
		t.Fatal("lease granted despite a FAILED barrier — the OH1 double-primary window is open")
	}
}

// TestLeaseBarrierSuccessButDeposedBlocksRenewal: the barrier succeeds (we can
// reach the quorum) but the fresh view shows a DIFFERENT primary — a handoff we
// have now seen. We must not renew our own lease.
func TestLeaseBarrierSuccessButDeposedBlocksRenewal(t *testing.T) {
	clock := newFakeClock(0)
	eng := barrierEngine(t, clock)
	f := seededFSM(t, "n2") // quorum-confirmed: n2 is primary now, not us

	k := newLeaseKeeper(f, "n1", map[int]*pbisr.Engine{0: eng},
		50*time.Millisecond, 5*time.Millisecond, clock.now,
		func(time.Time) error { return nil }, nil, time.Second)

	k.tick()
	if eng.LeaseValid() {
		t.Fatal("lease granted for a shard we no longer primary — deposed-node renewal not blocked")
	}
}

// TestLeaseBarrierSuccessAndPrimaryGrants: barrier succeeds and the confirmed
// view still names us primary → renew (the happy path stays working).
func TestLeaseBarrierSuccessAndPrimaryGrants(t *testing.T) {
	clock := newFakeClock(0)
	eng := barrierEngine(t, clock)
	f := seededFSM(t, "n1")

	var calls int
	k := newLeaseKeeper(f, "n1", map[int]*pbisr.Engine{0: eng},
		50*time.Millisecond, 5*time.Millisecond, clock.now,
		func(time.Time) error { calls++; return nil }, nil, time.Second)

	k.tick()
	if calls != 1 {
		t.Fatalf("barrier called %d times, want exactly 1 per tick", calls)
	}
	if !eng.LeaseValid() {
		t.Fatal("healthy primary with a passing barrier was not granted a lease")
	}
}

// TestLeaseReadBarrierOncePerEpoch pins the COST and the PLACEMENT of the
// per-epoch read-index barrier: exactly one call per (shard, epoch) transition, and
// none at all while an epoch is merely being renewed.
//
// Placement matters because a lease is a licence to ACK, and the engine builds each
// write's peer set from an unbarriered read of this same local MetaFSM. Barriering
// before the epoch's first grant makes that FSM provably current at the moment
// acking becomes legal; the FSM's applied index only advances afterwards, so every
// later read is at least as fresh. Cost matters because doing this per TICK (or per
// write) would reintroduce the follower-forward starvation confirmMetaView was
// written to avoid — a regression to per-tick barriering must fail here.
func TestLeaseReadBarrierOncePerEpoch(t *testing.T) {
	clock := newFakeClock(0)
	eng := barrierEngine(t, clock)
	f := seededFSM(t, "n1")

	var reads int
	k := newLeaseKeeper(f, "n1", map[int]*pbisr.Engine{0: eng},
		50*time.Millisecond, 5*time.Millisecond, clock.now,
		func(time.Time) error { return nil },
		func(time.Time) error { reads++; return nil }, time.Second)

	for i := 0; i < 5; i++ {
		k.tick()
	}
	if reads != 1 {
		t.Fatalf("read barrier called %d times over 5 ticks at ONE epoch, want exactly 1 "+
			"(per EPOCH TRANSITION, not per tick)", reads)
	}
	if !eng.LeaseValid() {
		t.Fatal("primary was not granted a lease after a passing read barrier")
	}

	// A new epoch (same primary) is a new licence to ack: barrier again, once.
	applyMeta(t, f, LogEntry{Op: OpSetShardEpoch, ShardID: 0, Epoch: 2, Primary: "n1"})
	for i := 0; i < 3; i++ {
		k.tick()
	}
	if reads != 2 {
		t.Fatalf("read barrier called %d times total, want 2 (one per epoch)", reads)
	}
	if eng.Epoch() != 2 {
		t.Fatalf("engine epoch = %d, want 2 (the barriered promotion was not applied)", eng.Epoch())
	}
}

// TestLeaseReadBarrierFailBlocksNewEpochGrant: if freshness cannot be confirmed,
// the new epoch is NOT granted — the node would otherwise be licensed to ack while
// its local ISR view could be narrower than the committed one. It must retry, not
// proceed, and it must recover once the barrier heals.
func TestLeaseReadBarrierFailBlocksNewEpochGrant(t *testing.T) {
	clock := newFakeClock(0)
	eng := barrierEngine(t, clock)
	f := seededFSM(t, "n1")

	healthy := false
	k := newLeaseKeeper(f, "n1", map[int]*pbisr.Engine{0: eng},
		50*time.Millisecond, 5*time.Millisecond, clock.now,
		func(time.Time) error { return nil },
		func(time.Time) error {
			if healthy {
				return nil
			}
			return errors.New("meta frontier unreachable")
		}, time.Second)

	k.tick()
	if eng.LeaseValid() {
		t.Fatal("lease granted despite a FAILED read barrier — the node may ack against a stale ISR")
	}

	healthy = true
	k.tick()
	if !eng.LeaseValid() {
		t.Fatal("lease not granted after the read barrier healed — the block must be transient")
	}
}

// TestLeaseBarrierRecoveryReenablesRenewal: once the barrier heals (partition
// ends), a subsequent tick renews again — the fence is transient, not terminal.
func TestLeaseBarrierRecoveryReenablesRenewal(t *testing.T) {
	clock := newFakeClock(0)
	eng := barrierEngine(t, clock)
	f := seededFSM(t, "n1")

	healthy := false
	k := newLeaseKeeper(f, "n1", map[int]*pbisr.Engine{0: eng},
		50*time.Millisecond, 5*time.Millisecond, clock.now,
		func(time.Time) error {
			if healthy {
				return nil
			}
			return errors.New("partitioned")
		}, nil, time.Second)

	k.tick()
	if eng.LeaseValid() {
		t.Fatal("granted while partitioned")
	}
	healthy = true
	k.tick()
	if !eng.LeaseValid() {
		t.Fatal("did not re-grant after the barrier recovered")
	}
}

// applierFunc adapts a func to pbisr.Applier for the promotion test's backup.
type applierFunc func([]byte) ([]byte, error)

func (f applierFunc) Apply(data []byte) ([]byte, error) { return f(data) }

// TestLeaseKeeperPromotesBackupOnEpochBump (failover consumer side): when the
// MetaFSM names this node the primary at an epoch HIGHER than its engine holds
// — a failover — the keeper must PROMOTE (continue seq assignment from the
// applied high-water), not plain AdoptEpoch (which would leave lastSeq at 0 and
// gap-reject the first post-promotion write).
func TestLeaseKeeperPromotesBackupOnEpochBump(t *testing.T) {
	clock := newFakeClock(0)
	ctrl := newMetaControl(NewMetaFSM(), 1)
	// "n2" as a BACKUP: feed it 3 replicated writes so lastApplied == 3, engine
	// epoch == 1 (adopted via Receive), but it is NOT the primary.
	eng := pbisr.New("n2", 0, ctrl, nil, applierFunc(func(d []byte) ([]byte, error) { return d, nil }), pbisr.WithClock(clock.now))
	// Each frame names its predecessor's (seq, epoch) — the chain link.
	// Seq 1 extends genesis (0, 0); the rest extend the epoch-1 write before them.
	for i := uint64(1); i <= 3; i++ {
		var prevEpoch uint64
		if i > 1 {
			prevEpoch = 1
		}
		ack := eng.Receive(pbisr.ReplicateMsg{Epoch: 1, Seq: i, PrevSeq: i - 1, PrevEpoch: prevEpoch, Data: []byte("w")})
		if !ack.OK {
			t.Fatalf("backup Receive seq %d not OK", i)
		}
	}
	if eng.LastApplied() != 3 {
		t.Fatalf("LastApplied = %d, want 3", eng.LastApplied())
	}

	// Failover: MetaRaft bumps to epoch 2 and names n2 primary.
	f := NewMetaFSM()
	applyMeta(t, f, LogEntry{Op: OpSetShardEpoch, ShardID: 0, Epoch: 2, Primary: "n2"})
	applyMeta(t, f, LogEntry{Op: OpSetShardISR, ShardID: 0, Epoch: 2, ISR: []string{"n2"}})

	k := newLeaseKeeper(f, "n2", map[int]*pbisr.Engine{0: eng},
		50*time.Millisecond, 5*time.Millisecond, clock.now, nil, nil, 0)
	k.tick()

	if eng.Epoch() != 2 {
		t.Fatalf("engine epoch = %d, want 2 after promotion", eng.Epoch())
	}
	if eng.LastSeq() != 3 {
		t.Fatalf("LastSeq = %d, want 3 (promotion must continue from applied high-water, not 0)", eng.LastSeq())
	}
	if !eng.LeaseValid() {
		t.Fatal("promoted primary holds no valid lease")
	}
}

// TestConfirmMetaViewSingleNode: with no meta / no peers there is no quorum to
// lose, so the connection check always confirms (nil).
func TestConfirmMetaViewSingleNode(t *testing.T) {
	n := &Node{} // meta == nil, no peers
	if err := n.confirmMetaView(time.Now().Add(time.Second)); err != nil {
		t.Fatalf("single-node confirmMetaView = %v, want nil", err)
	}
}

// TestLeaseBarrierNilUnconditional: a nil barrier (single-node / no meta peers)
// renews unconditionally — the pre-4a behavior for a cluster with no quorum to
// confirm.
func TestLeaseBarrierNilUnconditional(t *testing.T) {
	clock := newFakeClock(0)
	eng := barrierEngine(t, clock)
	f := seededFSM(t, "n1")

	k := newLeaseKeeper(f, "n1", map[int]*pbisr.Engine{0: eng},
		50*time.Millisecond, 5*time.Millisecond, clock.now, nil, nil, 0)
	k.tick()
	if !eng.LeaseValid() {
		t.Fatal("nil-barrier keeper failed to grant (single-node path regressed)")
	}
}
