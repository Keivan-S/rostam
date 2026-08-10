// SPDX-License-Identifier: Apache-2.0

package pbisr

import (
	"context"
	"errors"
	"sync"
	"testing"
	"time"
)

const testShard = 0

// Injected-clock constants (monotonic ns). Tests start the clock at t0 and grant
// the primary a lease expiring at t0+leaseDur, then advance the clock to model a
// partitioned primary whose lease lapses without renewal.
const (
	t0       = int64(1_000_000_000)     // arbitrary monotonic start
	leaseDur = int64(1_000_000_000_000) // far-future lease window (~1000s)
)

// fakeClock is a test-controllable monotonic clock injected via WithClock. The
// engine reads it INSTEAD of time.Now so the lease fence is deterministic.
type fakeClock struct {
	mu sync.Mutex
	t  int64
}

func (c *fakeClock) now() int64 {
	c.mu.Lock()
	defer c.mu.Unlock()
	return c.t
}

func (c *fakeClock) set(t int64) {
	c.mu.Lock()
	c.t = t
	c.mu.Unlock()
}

// fakeControl is an in-memory, mutable view of one shard's MetaRaft-authoritative
// control-plane state (epoch / primary / ISR / min-ISR). All engines in a test
// share one instance — these are global cluster facts.
type fakeControl struct {
	mu      sync.Mutex
	epoch   uint64
	primary string
	isr     []string
	minISR  int
}

func (c *fakeControl) Epoch(int) uint64 {
	c.mu.Lock()
	defer c.mu.Unlock()
	return c.epoch
}

func (c *fakeControl) Primary(int) string {
	c.mu.Lock()
	defer c.mu.Unlock()
	return c.primary
}

func (c *fakeControl) ISR(int) []string {
	c.mu.Lock()
	defer c.mu.Unlock()
	return append([]string(nil), c.isr...)
}

func (c *fakeControl) MinISR(int) int {
	c.mu.Lock()
	defer c.mu.Unlock()
	return c.minISR
}

func (c *fakeControl) setISR(isr []string) {
	c.mu.Lock()
	c.isr = isr
	c.mu.Unlock()
}

// fakeApplier records, in order, every op it applies.
type fakeApplier struct {
	mu      sync.Mutex
	applied [][]byte
	fail    bool
}

func (a *fakeApplier) Apply(data []byte) ([]byte, error) {
	a.mu.Lock()
	defer a.mu.Unlock()
	if a.fail {
		return nil, errors.New("apply failed")
	}
	a.applied = append(a.applied, append([]byte(nil), data...))
	return []byte("ok"), nil
}

func (a *fakeApplier) count() int {
	a.mu.Lock()
	defer a.mu.Unlock()
	return len(a.applied)
}

func (a *fakeApplier) setFail(v bool) {
	a.mu.Lock()
	a.fail = v
	a.mu.Unlock()
}

// cluster bundles a set of engines sharing one control plane and transport.
type cluster struct {
	ctrl     *fakeControl
	tr       *inMemTransport
	clk      *fakeClock
	engines  map[string]*Engine
	appliers map[string]*fakeApplier
}

// newCluster builds one engine per node ID over a shared control plane, in-memory
// transport, and a single injected clock (started at t0). The primary is granted
// a far-future lease for the current epoch (models MetaRaft leasing it); backups
// hold no lease. Advancing c.clk past t0+leaseDur expires the primary's lease.
func newCluster(nodeIDs []string, primary string, epoch uint64, isr []string, minISR int) *cluster {
	ctrl := &fakeControl{epoch: epoch, primary: primary, isr: isr, minISR: minISR}
	tr := newInMemTransport()
	clk := &fakeClock{t: t0}
	c := &cluster{
		ctrl:     ctrl,
		tr:       tr,
		clk:      clk,
		engines:  make(map[string]*Engine),
		appliers: make(map[string]*fakeApplier),
	}
	for _, id := range nodeIDs {
		ap := &fakeApplier{}
		eng := New(id, testShard, ctrl, tr, ap, WithClock(clk.now))
		c.engines[id] = eng
		c.appliers[id] = ap
		tr.register(id, eng)
	}
	// Grant the primary a far-future lease for the current epoch. This is what
	// lets it pass the Propose lease fence; a partitioned primary loses it by
	// clock advance, not by any control-plane observation (OH1).
	c.engines[primary].GrantLease(epoch, t0+leaseDur)
	return c
}

func ctxWithTimeout(t *testing.T, d time.Duration) context.Context {
	t.Helper()
	ctx, cancel := context.WithTimeout(context.Background(), d)
	t.Cleanup(cancel)
	return ctx
}

// eventually polls cond until true or the deadline, then fails with msg. Used
// for backups that apply asynchronously after Propose has already met its floor.
func eventually(t *testing.T, cond func() bool, msg string) {
	t.Helper()
	deadline := time.Now().Add(2 * time.Second)
	for time.Now().Before(deadline) {
		if cond() {
			return
		}
		time.Sleep(time.Millisecond)
	}
	t.Fatalf("condition not met within deadline: %s", msg)
}

// --- Happy path -------------------------------------------------------------

func TestHappyPathReplicatesAndAcks(t *testing.T) {
	c := newCluster([]string{"n1", "n2", "n3"}, "n1", 1, []string{"n1", "n2", "n3"}, 2)
	primary := c.engines["n1"]

	res, seq, err := primary.Propose(ctxWithTimeout(t, time.Second), []byte("w1"))
	if err != nil {
		t.Fatalf("propose: %v", err)
	}
	if seq != 1 {
		t.Fatalf("seq = %d, want 1", seq)
	}
	if string(res) != "ok" {
		t.Fatalf("result = %q, want ok", res)
	}
	// Both backups apply the write in seq order (n3 may lag Propose's return,
	// which stops once the min-ISR floor of 2 is met — hence the poll).
	for _, id := range []string{"n2", "n3"} {

		eventually(t, func() bool { return c.engines[id].LastApplied() == 1 },
			id+" lastApplied should reach 1")
		if got := c.appliers[id].count(); got != 1 {
			t.Fatalf("%s applied count = %d, want 1", id, got)
		}
	}
	// Primary applied locally too.
	if got := c.appliers["n1"].count(); got != 1 {
		t.Fatalf("primary applied count = %d, want 1", got)
	}

	// A second write gets the next monotonic seq.
	_, seq2, err := primary.Propose(ctxWithTimeout(t, time.Second), []byte("w2"))
	if err != nil {
		t.Fatalf("propose 2: %v", err)
	}
	if seq2 != 2 {
		t.Fatalf("seq2 = %d, want 2", seq2)
	}
}

// --- Primary apply-failure must not burn a seq (shard-wedge regression) ------

func TestPrimaryApplyFailureDoesNotBurnSeq(t *testing.T) {
	c := newCluster([]string{"n1", "n2", "n3"}, "n1", 1, []string{"n1", "n2", "n3"}, 2)
	primary := c.engines["n1"]

	if _, seq, err := primary.Propose(ctxWithTimeout(t, time.Second), []byte("w1")); err != nil || seq != 1 {
		t.Fatalf("w1: seq=%d err=%v", seq, err)
	}

	// The primary's local apply fails: the write must error AND must not advance
	// the seq (a phantom seq would gap-reject every future write at the backups).
	c.appliers["n1"].setFail(true)
	if _, _, err := primary.Propose(ctxWithTimeout(t, time.Second), []byte("bad")); err == nil {
		t.Fatal("expected apply-failure error")
	}
	c.appliers["n1"].setFail(false)

	// The next write must still replicate and commit — no wedge — and take the
	// next DENSE seq (2, not 3: nothing was burned).
	_, seq, err := primary.Propose(ctxWithTimeout(t, time.Second), []byte("w2"))
	if err != nil {
		t.Fatalf("post-recovery propose wedged: %v", err)
	}
	if seq != 2 {
		t.Fatalf("seq = %d, want 2 (a phantom seq was burned)", seq)
	}
	if got := primary.Committed(); got != 2 {
		t.Fatalf("committed = %d, want 2", got)
	}
	// Backups applied exactly the two real writes, gap-free.
	eventually(t, func() bool { return c.engines["n2"].LastApplied() == 2 }, "n2 reaches seq 2")
	if got := c.appliers["n2"].count(); got != 2 {
		t.Fatalf("n2 applied %d, want 2 (w1,w2)", got)
	}
}

// --- H3: min-ISR floor ------------------------------------------------------

func TestH3BelowMinISRRejects(t *testing.T) {
	c := newCluster([]string{"n1", "n2", "n3"}, "n1", 1, []string{"n1", "n2", "n3"}, 2)
	// Shrink ISR below the floor (2).
	c.ctrl.setISR([]string{"n1"})

	_, _, err := c.engines["n1"].Propose(ctxWithTimeout(t, time.Second), []byte("w"))
	if !errors.Is(err, ErrBelowMinISR) {
		t.Fatalf("err = %v, want ErrBelowMinISR", err)
	}
	// Nothing was applied anywhere — the write is refused before local apply.
	for _, id := range []string{"n1", "n2", "n3"} {
		if got := c.appliers[id].count(); got != 0 {
			t.Fatalf("%s applied %d ops, want 0 (nothing acked below floor)", id, got)
		}
	}
}

// --- H1/H5: epoch fence at the backup ---------------------------------------

func TestH1FenceStalePrimaryCannotAck(t *testing.T) {
	// Control still names n1 primary at epoch 1, but n2 has already adopted the
	// newer epoch 2 (e.g. it heard from the new primary). n1 is stale.
	c := newCluster([]string{"n1", "n2"}, "n1", 1, []string{"n1", "n2"}, 2)
	c.engines["n2"].AdoptEpoch(2)

	_, _, err := c.engines["n1"].Propose(ctxWithTimeout(t, 500*time.Millisecond), []byte("w"))
	if !errors.Is(err, ErrReplicationTimeout) {
		t.Fatalf("err = %v, want ErrReplicationTimeout (backup fenced the stale primary)", err)
	}
	// The fenced backup rejected the epoch-1 write: it applied nothing.
	if got := c.appliers["n2"].count(); got != 0 {
		t.Fatalf("fenced backup applied %d ops, want 0", got)
	}
}

// --- H6: acks are matched per exact (epoch, seq) ----------------------------

func TestH6WrongSeqAckDoesNotCount(t *testing.T) {
	c := newCluster([]string{"n1", "n2"}, "n1", 1, []string{"n1", "n2"}, 2)
	// n2 forges an OK ack, but for a DIFFERENT seq (a lagging/old-seq ack).
	c.tr.setFault("n2", peerFault{
		ackOverride: func(msg ReplicateMsg) AckMsg {
			return AckMsg{Epoch: msg.Epoch, Seq: msg.Seq + 100, OK: true}
		},
	})

	_, _, err := c.engines["n1"].Propose(ctxWithTimeout(t, 500*time.Millisecond), []byte("w"))
	if !errors.Is(err, ErrReplicationTimeout) {
		t.Fatalf("err = %v, want ErrReplicationTimeout (wrong-seq ack must not count)", err)
	}
}

func TestH6LivenessOnlyAckDoesNotCount(t *testing.T) {
	c := newCluster([]string{"n1", "n2"}, "n1", 1, []string{"n1", "n2"}, 2)
	// n2 returns a liveness-only signal (right seq, but OK:false).
	c.tr.setFault("n2", peerFault{
		ackOverride: func(msg ReplicateMsg) AckMsg {
			return AckMsg{Epoch: msg.Epoch, Seq: msg.Seq, OK: false}
		},
	})

	_, _, err := c.engines["n1"].Propose(ctxWithTimeout(t, 500*time.Millisecond), []byte("w"))
	if !errors.Is(err, ErrReplicationTimeout) {
		t.Fatalf("err = %v, want ErrReplicationTimeout (liveness ack must not count)", err)
	}
}

// --- Gap check at the backup ------------------------------------------------

func TestGapCheckRejectsOutOfOrder(t *testing.T) {
	c := newCluster([]string{"n1", "n2"}, "n1", 1, []string{"n1", "n2"}, 2)
	backup := c.engines["n2"]

	// Drive the backup to lastApplied = 5 with a gap-free run. Each frame names its
	// predecessor's (seq, epoch): seq 1 extends genesis (0, 0), the rest extend the
	// epoch-1 write before them.
	for s := uint64(1); s <= 5; s++ {
		var prevEpoch uint64
		if s > 1 {
			prevEpoch = 1
		}
		ack := backup.Receive(ReplicateMsg{Epoch: 1, Seq: s, PrevSeq: s - 1, PrevEpoch: prevEpoch, Data: []byte("d")})
		if !ack.OK {
			t.Fatalf("seq %d: ack OK = false, want true", s)
		}
	}
	if got := backup.LastApplied(); got != 5 {
		t.Fatalf("lastApplied = %d, want 5", got)
	}
	appliedBefore := c.appliers["n2"].count()

	// A message whose PrevSeq (7) does not match lastApplied (5) is rejected.
	ack := backup.Receive(ReplicateMsg{Epoch: 1, Seq: 8, PrevSeq: 7, PrevEpoch: 1, Data: []byte("gap")})
	if ack.OK {
		t.Fatalf("gapped ack OK = true, want false")
	}
	if got := backup.LastApplied(); got != 5 {
		t.Fatalf("lastApplied after gap = %d, want 5 (unchanged)", got)
	}
	if got := c.appliers["n2"].count(); got != appliedBefore {
		t.Fatalf("applied count changed on gap: %d -> %d", appliedBefore, got)
	}
}

// --- Not primary ------------------------------------------------------------

func TestProposeOnNonPrimaryRejected(t *testing.T) {
	c := newCluster([]string{"n1", "n2", "n3"}, "n1", 1, []string{"n1", "n2", "n3"}, 2)

	// A backup node is not the primary.
	_, _, err := c.engines["n2"].Propose(ctxWithTimeout(t, time.Second), []byte("w"))
	if !errors.Is(err, ErrNotPrimary) {
		t.Fatalf("err = %v, want ErrNotPrimary", err)
	}
}

func TestProposeWithoutLeaseRejected(t *testing.T) {
	// n1 is the named primary, but holds NO lease (never granted one). Under the
	// lease model, authority to write comes from the lease, not from being named
	// or from a cached epoch — so Propose must fail closed.
	ctrl := &fakeControl{epoch: 2, primary: "n1", isr: []string{"n1", "n2"}, minISR: 2}
	tr := newInMemTransport()
	clk := &fakeClock{t: t0}
	ap := &fakeApplier{}
	noLease := New("n1", testShard, ctrl, tr, ap, WithClock(clk.now)) // leaseEpoch 0, epoch 0
	tr.register("n1", noLease)

	_, _, err := noLease.Propose(ctxWithTimeout(t, time.Second), []byte("w"))
	if !errors.Is(err, ErrLeaseExpired) {
		t.Fatalf("err = %v, want ErrLeaseExpired (no lease held)", err)
	}
	if got := ap.count(); got != 0 {
		t.Fatalf("applied %d ops, want 0 (rejected before local apply)", got)
	}
}

// --- OH1: a stale primary self-fences the instant its lease lapses -----------

// TestOH1StalePrimarySelfFencesOnLeaseExpiry is the OH1 write-path closure. A
// primary holds epoch 1 over ISR {P,B1,B2}, minISR 2, with a lease expiring at T.
// The injected clock advances past T with NO renewal (models P partitioned from
// MetaRaft). Even though the control plane STILL (stale-ly) reports P primary at
// epoch 1, Propose must reject with ErrLeaseExpired and neither commit nor ack:
// the stale primary self-fences purely on its own monotonic clock.
func TestOH1StalePrimarySelfFencesOnLeaseExpiry(t *testing.T) {
	c := newCluster([]string{"n1", "n2", "n3"}, "n1", 1, []string{"n1", "n2", "n3"}, 2)
	primary := c.engines["n1"]

	// Partition from MetaRaft = no renewal. Advance the clock past the lease.
	c.clk.set(t0 + leaseDur + 1)

	_, _, err := primary.Propose(ctxWithTimeout(t, time.Second), []byte("w"))
	if !errors.Is(err, ErrLeaseExpired) {
		t.Fatalf("err = %v, want ErrLeaseExpired (stale primary must self-fence on lease expiry)", err)
	}
	// The control plane is still stale: it names n1 primary at epoch 1. That the
	// fence fired regardless is exactly the OH1 fix — no fresh state was observed.
	if got := c.ctrl.Primary(testShard); got != "n1" {
		t.Fatalf("control primary = %q, want n1 (still stale)", got)
	}
	if got := c.ctrl.Epoch(testShard); got != 1 {
		t.Fatalf("control epoch = %d, want 1 (still stale)", got)
	}
	// No node applied the write (fence precedes local apply), nothing committed.
	for _, id := range []string{"n1", "n2", "n3"} {
		if got := c.appliers[id].count(); got != 0 {
			t.Fatalf("%s applied %d ops, want 0 (fenced primary must not write)", id, got)
		}
	}
	if got := primary.Committed(); got != 0 {
		t.Fatalf("committed = %d, want 0", got)
	}
}

// TestLeaseRenewalReenablesPropose shows a lapsed lease is recoverable: after
// expiry a fresh GrantLease for the same epoch with a later expiry re-enables
// the write path.
func TestLeaseRenewalReenablesPropose(t *testing.T) {
	c := newCluster([]string{"n1", "n2"}, "n1", 1, []string{"n1", "n2"}, 2)
	primary := c.engines["n1"]

	// Expire the lease.
	c.clk.set(t0 + leaseDur + 1)
	if _, _, err := primary.Propose(ctxWithTimeout(t, time.Second), []byte("w")); !errors.Is(err, ErrLeaseExpired) {
		t.Fatalf("pre-renewal err = %v, want ErrLeaseExpired", err)
	}

	// MetaRaft renews the lease (same epoch, later expiry).
	primary.GrantLease(1, c.clk.now()+leaseDur)

	_, seq, err := primary.Propose(ctxWithTimeout(t, time.Second), []byte("w"))
	if err != nil {
		t.Fatalf("post-renewal propose: %v", err)
	}
	if seq != 1 {
		t.Fatalf("seq = %d, want 1", seq)
	}
}

// --- Full-ISR commit: every current ISR member must ack ----------------------

// TestFullISRCommitRequiresEveryMember shows commit needs the FULL current ISR,
// not just a minISR subset. With ISR {n1,n2,n3} and minISR 2, a partitioned n3
// makes Propose time out even though {n1,n2} alone would satisfy the old floor.
// Shrinking the ISR to {n1,n2} (a MetaRaft op, faked here) makes the full ISR
// reachable again and the write commits.
func TestFullISRCommitRequiresEveryMember(t *testing.T) {
	c := newCluster([]string{"n1", "n2", "n3"}, "n1", 1, []string{"n1", "n2", "n3"}, 2)
	primary := c.engines["n1"]

	// n3 is partitioned. Full-ISR commit REQUIRES its ack, so Propose times out
	// despite {n1,n2} being >= minISR.
	c.tr.setFault("n3", peerFault{partition: true})
	_, _, err := primary.Propose(ctxWithTimeout(t, 200*time.Millisecond), []byte("w1"))
	if !errors.Is(err, ErrReplicationTimeout) {
		t.Fatalf("err = %v, want ErrReplicationTimeout (full-ISR commit needs n3)", err)
	}
	if got := primary.Committed(); got != 0 {
		t.Fatalf("committed = %d, want 0 (full ISR did not ack)", got)
	}
	// n2 (reachable) did apply w1; n3 (partitioned) did not.
	eventually(t, func() bool { return c.engines["n2"].LastApplied() == 1 }, "n2 applies w1")
	if got := c.appliers["n3"].count(); got != 0 {
		t.Fatalf("partitioned n3 applied %d, want 0", got)
	}

	// MetaRaft shrinks the ISR to drop the straggler: len(ISR)=2 >= minISR, and
	// the full (smaller) ISR is now reachable, so the next write commits.
	c.ctrl.setISR([]string{"n1", "n2"})
	_, seq, err := primary.Propose(ctxWithTimeout(t, time.Second), []byte("w2"))
	if err != nil {
		t.Fatalf("propose after ISR shrink: %v", err)
	}
	if seq != 2 {
		t.Fatalf("seq = %d, want 2 (w1 assigned seq 1 uncommitted)", seq)
	}
	if got := primary.Committed(); got != 2 {
		t.Fatalf("committed = %d, want 2 (full smaller ISR acked)", got)
	}
}

// --- Hole class (ii): dead backup stalls, ISR shrink recovers (P7/§3(ii)) ----

// TestDeadBackupStallsThenRecoversOnISRShrink covers hole class (ii): a
// PARTITIONED (silent) backup — one that delivers no completion at all — makes
// every full-ISR write time out. Each write applies locally and the reachable
// backup acks, but the record can never commit (the dead peer never acks), so a
// stalled tail of uncommitted seqs accumulates with committed pinned at 0. The
// pipeline is admission-bounded at W (proven by the windowWait/Propose stall
// tests; here the few writes we issue stay well under W, so the ctx deadline —
// not admission — is what returns each write). When MetaRaft shrinks the ISR to
// drop the dead peer, the full (smaller) ISR {n1,n2} becomes reachable: the next
// Propose commits AND committed sweeps TRANSITIVELY past the earlier stalled
// seqs (P7) — the §3(ii) recovery.
func TestDeadBackupStallsThenRecoversOnISRShrink(t *testing.T) {
	c := newCluster([]string{"n1", "n2", "n3"}, "n1", 1, []string{"n1", "n2", "n3"}, 2)
	primary := c.engines["n1"]
	defer primary.Shutdown()

	// n3 is partitioned: silent, never acks. Full-ISR commit REQUIRES n3, so each
	// write applies locally (n2 also applies) then times out — a stalled tail.
	c.tr.setFault("n3", peerFault{partition: true})

	const stalled = 3
	for i := 0; i < stalled; i++ {
		_, seq, err := primary.Propose(ctxWithTimeout(t, 200*time.Millisecond), []byte("w"))
		if !errors.Is(err, ErrReplicationTimeout) {
			t.Fatalf("stalled write %d: err = %v, want ErrReplicationTimeout (n3 silent)", i, err)
		}
		if seq != uint64(i+1) {
			t.Fatalf("stalled write %d: seq = %d, want %d (dense)", i, seq, i+1)
		}
	}
	// Nothing committed: the full ISR never acked any of the stalled writes.
	if got := primary.Committed(); got != 0 {
		t.Fatalf("committed = %d, want 0 (dead backup blocks every commit)", got)
	}
	// The reachable backup applied the whole uncommitted tail; the dead one did not.
	eventually(t, func() bool { return c.engines["n2"].LastApplied() == stalled },
		"n2 applies the stalled tail")
	if got := c.appliers["n3"].count(); got != 0 {
		t.Fatalf("partitioned n3 applied %d, want 0", got)
	}

	// MetaRaft shrinks the ISR to drop the dead peer. The full (smaller) ISR is
	// reachable now, so the next write commits — and committed sweeps transitively
	// past the stalled seqs 1..stalled, exposing them as durable (P7/P10).
	c.ctrl.setISR([]string{"n1", "n2"})
	_, seq, err := primary.Propose(ctxWithTimeout(t, 2*time.Second), []byte("recover"))
	if err != nil {
		t.Fatalf("post-shrink propose: %v", err)
	}
	if seq != uint64(stalled+1) {
		t.Fatalf("recovery seq = %d, want %d (dense, one past the stalled tail)", seq, stalled+1)
	}
	if got := primary.Committed(); got != uint64(stalled+1) {
		t.Fatalf("committed = %d, want %d (transitive commit swept past the holes)", got, stalled+1)
	}
}

// --- Timeout ----------------------------------------------------------------

func TestProposeTimesOutWhenBackupUnreachable(t *testing.T) {
	c := newCluster([]string{"n1", "n2"}, "n1", 1, []string{"n1", "n2"}, 2)
	// Partition n2: replication to it blocks until the context deadline.
	c.tr.setFault("n2", peerFault{partition: true})

	start := time.Now()
	_, _, err := c.engines["n1"].Propose(ctxWithTimeout(t, 100*time.Millisecond), []byte("w"))
	elapsed := time.Since(start)
	if !errors.Is(err, ErrReplicationTimeout) {
		t.Fatalf("err = %v, want ErrReplicationTimeout", err)
	}
	if elapsed > 2*time.Second {
		t.Fatalf("propose took %v, expected to return near the 100ms deadline", elapsed)
	}
	// The write was applied locally even though it was not durably acked.
	if got := c.appliers["n1"].count(); got != 1 {
		t.Fatalf("primary applied count = %d, want 1", got)
	}
}

// --- Fail-fast on a definitive negative ack (no ctx-deadline wait) -----------

// TestProposeFailsFastOnNegativeAck proves the restored fail-fast failure path:
// a backup that returns a DEFINITIVE negative ack (drop fault → immediate
// OK:false) can never satisfy full-ISR for this (epoch,seq), so Propose must
// resolve the record FAILED on that transport signal and return promptly — NOT
// wait out its (deliberately long) ctx deadline the way the buggy version did.
// The timing assertion is the whole point: a 5s ctx, but the write must fail in
// well under 500ms.
func TestProposeFailsFastOnNegativeAck(t *testing.T) {
	c := newCluster([]string{"n1", "n2"}, "n1", 1, []string{"n1", "n2"}, 2)
	defer c.engines["n1"].Shutdown()
	c.tr.setFault("n2", peerFault{drop: true})

	start := time.Now()
	_, _, err := c.engines["n1"].Propose(ctxWithTimeout(t, 5*time.Second), []byte("w"))
	elapsed := time.Since(start)
	if !errors.Is(err, ErrReplicationTimeout) {
		t.Fatalf("err = %v, want ErrReplicationTimeout", err)
	}
	if elapsed > 500*time.Millisecond {
		t.Fatalf("Propose took %v; a definitive negative ack must fail the write promptly, not wait out the 5s ctx", elapsed)
	}
	// Applied locally (uncommitted tail) but never committed.
	if got := c.appliers["n1"].count(); got != 1 {
		t.Fatalf("primary applied count = %d, want 1", got)
	}
	if got := c.engines["n1"].Committed(); got != 0 {
		t.Fatalf("committed = %d, want 0 (negative ack is not durable)", got)
	}
}

// --- Monotonic / in-order under concurrency ---------------------------------

func TestConcurrentProposesAreMonotonicAndOrdered(t *testing.T) {
	// RF2 minISR2: every write REQUIRES the single backup to apply it, so the
	// backup reaching lastApplied == N proves gap-free, in-seq-order application.
	c := newCluster([]string{"n1", "n2"}, "n1", 1, []string{"n1", "n2"}, 2)
	primary := c.engines["n1"]

	const n = 20
	var (
		mu    sync.Mutex
		seqs  []uint64
		wg    sync.WaitGroup
		fails []error
	)
	for i := 0; i < n; i++ {
		wg.Add(1)
		go func() {
			defer wg.Done()
			_, seq, err := primary.Propose(ctxWithTimeout(t, 5*time.Second), []byte("w"))
			mu.Lock()
			if err != nil {
				fails = append(fails, err)
			} else {
				seqs = append(seqs, seq)
			}
			mu.Unlock()
		}()
	}
	wg.Wait()

	if len(fails) != 0 {
		t.Fatalf("got %d failed proposes, first: %v", len(fails), fails[0])
	}
	if len(seqs) != n {
		t.Fatalf("got %d seqs, want %d", len(seqs), n)
	}
	// Every seq in 1..n appears exactly once.
	seen := make(map[uint64]bool, n)
	for _, s := range seqs {
		if s < 1 || s > n {
			t.Fatalf("seq %d out of range 1..%d", s, n)
		}
		if seen[s] {
			t.Fatalf("duplicate seq %d", s)
		}
		seen[s] = true
	}
	// Backup applied all n writes gap-free and in order.
	if got := c.engines["n2"].LastApplied(); got != n {
		t.Fatalf("backup lastApplied = %d, want %d (in-order application)", got, n)
	}
	if got := c.appliers["n2"].count(); got != n {
		t.Fatalf("backup applied count = %d, want %d", got, n)
	}
}

// --- Ring backlog: bounded, drops oldest ------------------------------------

func TestRingDropsOldestOnOverflow(t *testing.T) {
	r := newRing(3)
	for i := 0; i < 5; i++ {
		r.append(ringEntry{seq: uint64(i)})
	}
	if r.len() != 3 {
		t.Fatalf("ring len = %d, want 3", r.len())
	}
	// Oldest two (seq 0,1) dropped; 2,3,4 retained in order.
	want := []uint64{2, 3, 4}
	for i, w := range want {
		got := r.buf[(r.head+i)%len(r.buf)].seq
		if got != w {
			t.Fatalf("ring[%d] = %d, want %d", i, got, w)
		}
	}
}

// --- Pipelining: concurrent callers overlap in-flight (redesign core) --------

// blockingTransport implements the async Transport but does NOT complete a
// submission until release() is called: each Replicate stores its done callback
// and returns immediately (non-blocking, as the contract requires), so multiple
// submissions to a peer can be in flight at once. It records the peak number of
// concurrently-stored (submitted-but-not-yet-acked) callbacks — the direct
// measurement of pipeline depth. release() then fires every stored callback with
// an OK ack for its exact (epoch,seq), letting every waiting Propose commit.
type blockingTransport struct {
	mu       sync.Mutex
	inflight int
	peak     int
	pending  []pendingSubmit
	released bool
}

type pendingSubmit struct {
	done func(AckMsg, error)
	msg  ReplicateMsg
}

func (b *blockingTransport) Replicate(peer string, msg ReplicateMsg, done func(AckMsg, error)) error {
	b.mu.Lock()
	if b.released {
		// Post-release submissions (none expected in this test) ack immediately.
		b.mu.Unlock()
		done(AckMsg{Epoch: msg.Epoch, Seq: msg.Seq, OK: true}, nil)
		return nil
	}
	b.inflight++
	if b.inflight > b.peak {
		b.peak = b.inflight
	}
	b.pending = append(b.pending, pendingSubmit{done: done, msg: msg})
	b.mu.Unlock()
	return nil
}

// release fires every stored callback with a matching OK ack.
func (b *blockingTransport) release() {
	b.mu.Lock()
	b.released = true
	pend := b.pending
	b.pending = nil
	b.mu.Unlock()
	for _, p := range pend {
		p.done(AckMsg{Epoch: p.msg.Epoch, Seq: p.msg.Seq, OK: true}, nil)
		b.mu.Lock()
		b.inflight--
		b.mu.Unlock()
	}
}

func (b *blockingTransport) inflightNow() int {
	b.mu.Lock()
	defer b.mu.Unlock()
	return b.inflight
}

func (b *blockingTransport) peakSeen() int {
	b.mu.Lock()
	defer b.mu.Unlock()
	return b.peak
}

// TestProposePipelinesConcurrentCallers proves the redesign's central claim:
// with W >= 8 and 8 concurrent Propose callers, more than one write is in flight
// to a peer at once (the old ping-pong path held writeMu across the network wait,
// so peak was always exactly 1). A transport that blocks each submission until
// released lets all 8 pile up; peak must exceed 1. Then release everything and
// assert every Propose commits, in dense seq order.
func TestProposePipelinesConcurrentCallers(t *testing.T) {
	const n = 8 // <= pipelineWindow (W = 256), so admission never gates here
	ctrl := &fakeControl{epoch: 1, primary: "n1", isr: []string{"n1", "n2"}, minISR: 2}
	bt := &blockingTransport{}
	clk := &fakeClock{t: t0}
	ap := &fakeApplier{}
	eng := New("n1", testShard, ctrl, bt, ap, WithClock(clk.now))
	eng.GrantLease(1, t0+leaseDur)
	defer eng.Shutdown()

	var wg sync.WaitGroup
	errs := make([]error, n)
	seqs := make([]uint64, n)
	for i := 0; i < n; i++ {
		wg.Add(1)
		go func(i int) {
			defer wg.Done()
			_, seq, err := eng.Propose(ctxWithTimeout(t, 10*time.Second), []byte("w"))
			seqs[i] = seq
			errs[i] = err
		}(i)
	}

	// Wait until all n submissions are simultaneously in flight to the peer (the
	// single sender submits each as its frame is enqueued; none complete until we
	// release). If the path were still ping-pong, inflight would never exceed 1
	// and this would time out — a real failure signal.
	eventually(t, func() bool { return bt.inflightNow() >= n },
		"all 8 proposes should be in flight to the peer concurrently")

	bt.release()
	wg.Wait()

	peak := bt.peakSeen()
	t.Logf("pipelining depth: peak concurrent in-flight submissions to peer = %d (callers = %d, W = %d)", peak, n, pipelineWindow)
	if peak <= 1 {
		t.Fatalf("peak in-flight submissions = %d, want > 1 (pipelining; old ping-pong == 1)", peak)
	}
	for i := 0; i < n; i++ {
		if errs[i] != nil {
			t.Fatalf("propose %d failed after release: %v", i, errs[i])
		}
	}
	// Every seq in 1..n appears exactly once, and the full ISR commit reached n.
	seen := make(map[uint64]bool, n)
	for _, s := range seqs {
		if s < 1 || s > n || seen[s] {
			t.Fatalf("bad/duplicate seq %d in %v", s, seqs)
		}
		seen[s] = true
	}
	if got := eng.Committed(); got != n {
		t.Fatalf("committed = %d, want %d (all pipelined writes commit)", got, n)
	}
}

func TestProposeRetainsInBacklog(t *testing.T) {
	c := newCluster([]string{"n1", "n2"}, "n1", 1, []string{"n1", "n2"}, 2)
	primary := c.engines["n1"]
	for i := 0; i < 3; i++ {
		if _, _, err := primary.Propose(ctxWithTimeout(t, time.Second), []byte("w")); err != nil {
			t.Fatalf("propose %d: %v", i, err)
		}
	}
	if got := primary.backlogLen(); got != 3 {
		t.Fatalf("backlog len = %d, want 3", got)
	}
}
