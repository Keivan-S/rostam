// SPDX-License-Identifier: Apache-2.0

package pbisr

import (
	"bytes"
	"encoding/binary"
	"errors"
	"fmt"
	"sync"
	"sync/atomic"
	"testing"
	"time"
)

// ============================================================================
// SNAPSHOT TRANSFER.
//
// The adversarial invariants under test:
//
//	(1) A ring-cold or diverged member is REPAIRED, not merely refused, and ends
//	    byte-identical to the primary.
//	(2) A minISR>=2 shard that a failover left write-dead regains writability with
//	    NO operator action. This is the case that is permanently dead without this
//	    stage, and it is the headline.
//	(3) A partial install is UNUSABLE, never half-installed-but-in-ISR: the poisoned
//	    node refuses to serve, refuses to receive, and is un-promotable.
//	(4) An abort mid-TRANSFER leaves the target PRISTINE (nothing installed), which
//	    is stronger than "nothing left half-installed".
// ============================================================================

// --- fake snapshot store ------------------------------------------------------

// fakeSnapStore is a pbisr.SnapshotStore over a fakeApplier: the "FSM" is the
// applier's ordered op list, so a snapshot is that list and an install REPLACES
// it wholesale (wipe-then-install, exactly like restoreSnapshot's key-set wipe).
// That makes "byte-identical FSM" a literal slice comparison.
//
// The durable poison fence is modelled as a plain field that SURVIVES engine
// reconstruction — the store outlives the Engine in these tests exactly as the
// on-disk fence file outlives a process.
type fakeSnapStore struct {
	ap *fakeApplier

	mu         sync.Mutex
	fenceUp    bool
	fenceSeq   uint64
	fenceEpoch uint64

	// failInstall makes InstallFSM WIPE the FSM and then fail — the half-wiped
	// state the poison fence exists for.
	failInstall bool

	installs atomic.Int64
	begins   atomic.Int64
	commits  atomic.Int64
	aborts   atomic.Int64
}

var _ SnapshotStore = (*fakeSnapStore)(nil)

func newFakeSnapStore(ap *fakeApplier) *fakeSnapStore { return &fakeSnapStore{ap: ap} }

func (s *fakeSnapStore) SnapshotFSM(appliedIndex uint64) ([]byte, error) {
	s.ap.mu.Lock()
	defer s.ap.mu.Unlock()
	var buf bytes.Buffer
	var n [8]byte
	binary.BigEndian.PutUint64(n[:], appliedIndex)
	buf.Write(n[:])
	binary.BigEndian.PutUint64(n[:], uint64(len(s.ap.applied)))
	buf.Write(n[:])
	for _, op := range s.ap.applied {
		var l [4]byte
		binary.BigEndian.PutUint32(l[:], uint32(len(op)))
		buf.Write(l[:])
		buf.Write(op)
	}
	return buf.Bytes(), nil
}

func (s *fakeSnapStore) InstallFSM(blob []byte) error {
	s.ap.mu.Lock()
	defer s.ap.mu.Unlock()
	// WIPE FIRST — the same delete-then-put shape restoreSnapshot has, so an
	// injected failure below leaves a genuinely half-installed FSM.
	s.ap.applied = nil
	s.mu.Lock()
	fail := s.failInstall
	s.mu.Unlock()
	if fail {
		return errors.New("injected install failure after wipe")
	}
	if len(blob) < 16 {
		return errors.New("short blob")
	}
	count := binary.BigEndian.Uint64(blob[8:16])
	b := blob[16:]
	for i := uint64(0); i < count; i++ {
		if len(b) < 4 {
			return errors.New("truncated blob")
		}
		l := binary.BigEndian.Uint32(b[0:4])
		b = b[4:]
		if uint32(len(b)) < l {
			return errors.New("truncated blob data")
		}
		s.ap.applied = append(s.ap.applied, append([]byte(nil), b[:l]...))
		b = b[l:]
	}
	s.installs.Add(1)
	return nil
}

func (s *fakeSnapStore) BeginInstall(seq, epoch uint64) error {
	s.begins.Add(1)
	s.mu.Lock()
	s.fenceUp, s.fenceSeq, s.fenceEpoch = true, seq, epoch
	s.mu.Unlock()
	return nil
}

func (s *fakeSnapStore) CommitInstall(seq, epoch uint64) error {
	s.commits.Add(1)
	s.mu.Lock()
	s.fenceUp = false
	s.mu.Unlock()
	return nil
}

func (s *fakeSnapStore) AbortInstall() error {
	s.aborts.Add(1)
	s.mu.Lock()
	s.fenceUp = false
	s.mu.Unlock()
	return nil
}

func (s *fakeSnapStore) InstallPending() bool {
	s.mu.Lock()
	defer s.mu.Unlock()
	return s.fenceUp
}

func (s *fakeSnapStore) setFailInstall(v bool) {
	s.mu.Lock()
	s.failInstall = v
	s.mu.Unlock()
}

// --- snapshot-capable test cluster --------------------------------------------

// snapCluster is `cluster` with a SnapshotStore wired into every engine, so both
// ends of a transfer are capable. The stores are kept addressable so a test can
// inject an install failure or model a restart.
type snapCluster struct {
	*cluster
	stores map[string]*fakeSnapStore
}

func newSnapCluster(nodeIDs []string, primary string, epoch uint64, isr []string, minISR int) *snapCluster {
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
	sc := &snapCluster{cluster: c, stores: make(map[string]*fakeSnapStore)}
	for _, id := range nodeIDs {
		ap := &fakeApplier{}
		ss := newFakeSnapStore(ap)
		eng := New(id, testShard, ctrl, tr, ap, WithClock(clk.now), WithSnapshotStore(ss))
		c.engines[id] = eng
		c.appliers[id] = ap
		sc.stores[id] = ss
		tr.register(id, eng)
	}
	c.engines[primary].GrantLease(epoch, t0+leaseDur)
	return sc
}

// restart models a process restart of node id: a FRESH Engine over the SAME
// applier and the SAME (durable) snapshot store. This is how the fence's
// crash-survival is tested — the new engine seeds its poisoned latch from
// InstallPending, exactly as the production one seeds it from the fence file.
func (sc *snapCluster) restart(id string) *Engine {
	sc.engines[id].Shutdown()
	eng := New(id, testShard, sc.ctrl, sc.tr, sc.appliers[id], WithClock(sc.clk.now),
		WithSnapshotStore(sc.stores[id]))
	sc.engines[id] = eng
	sc.tr.register(id, eng)
	return eng
}

// appliedOps returns a node's FSM contents as a comparable value.
func (sc *snapCluster) appliedOps(id string) [][]byte {
	a := sc.appliers[id]
	a.mu.Lock()
	defer a.mu.Unlock()
	return append([][]byte(nil), a.applied...)
}

// requireIdenticalFSM asserts two nodes' FSMs are BYTE-IDENTICAL — the whole
// point of a state transfer, and the assertion the delta path could never make
// for a ring-cold or diverged member because it never got to run.
func requireIdenticalFSM(t *testing.T, sc *snapCluster, a, b string) {
	t.Helper()
	oa, ob := sc.appliedOps(a), sc.appliedOps(b)
	if len(oa) != len(ob) {
		t.Fatalf("FSM length mismatch: %s has %d ops, %s has %d", a, len(oa), b, len(ob))
	}
	for i := range oa {
		if !bytes.Equal(oa[i], ob[i]) {
			t.Fatalf("FSM byte mismatch at op %d: %s=%q %s=%q", i, a, oa[i], b, ob[i])
		}
	}
}

// --- (1) ring-cold survivor ---------------------------------------------------

// TestSnapshotRepairsRingColdSurvivor pins the state the brief MEASURED as
// unrecoverable: after a failover the promoted primary's ring is EMPTY (Promote
// leaves the backlog untouched and the sole append site is proposeSequenced), so
// a survivor lagging by even ONE write needs a delta starting before the ring
// origin. Before this stage that was ErrGrowRingEvicted, permanently.
//
// The test deliberately uses a lag of exactly ONE, because that is the case the
// old code could not serve while still being the case where "just re-ship it"
// feels obviously possible.
func TestSnapshotRepairsRingColdSurvivor(t *testing.T) {
	sc := newSnapCluster([]string{"n1", "n2", "n3"}, "n1", 1, []string{"n1", "n2", "n3"}, 1)
	n1 := sc.engines["n1"]
	defer n1.Shutdown()

	// Build shared history on all three.
	for i := 0; i < 20; i++ {
		if _, _, err := n1.Propose(ctxWithTimeout(t, 5*time.Second), []byte{byte(i)}); err != nil {
			t.Fatalf("propose %d: %v", i, err)
		}
	}
	eventually(t, func() bool { return sc.engines["n3"].LastApplied() == 20 }, "n3 receives the shared history")

	// One more write that n3 misses (partitioned), then n2 is promoted: the classic
	// "survivor lags by one" shape.
	sc.tr.setFault("n3", peerFault{partition: true})
	sc.ctrl.setISR([]string{"n1", "n2"})
	if _, _, err := n1.Propose(ctxWithTimeout(t, 5*time.Second), []byte{0xFF}); err != nil {
		t.Fatalf("propose the write n3 misses: %v", err)
	}
	sc.tr.clearFault("n3")
	sc.tr.setFault("n1", peerFault{partition: true})

	// FAILOVER to n2 at epoch 2. Its ring is empty by construction.
	n2 := sc.engines["n2"]
	defer n2.Shutdown()
	sc.ctrl.mu.Lock()
	sc.ctrl.epoch, sc.ctrl.primary, sc.ctrl.isr = 2, "n2", []string{"n2"}
	sc.ctrl.mu.Unlock()
	n2.Promote(2, t0+leaseDur)
	if n2.backlogLen() != 0 {
		t.Fatalf("promoted primary's ring must be empty (the premise of this case), got %d", n2.backlogLen())
	}

	// n3 lags n2 by exactly one write.
	if got, want := sc.engines["n3"].LastApplied(), n2.LastSeq()-1; got != want {
		t.Fatalf("n3 lag setup: LastApplied=%d, want %d (exactly one behind)", got, want)
	}

	if err := n2.StartLearnerCatchup(ctxWithTimeout(t, 10*time.Second), "n3"); err != nil {
		t.Fatalf("StartLearnerCatchup over an EMPTY ring must now succeed via snapshot, got: %v", err)
	}
	if sc.stores["n3"].installs.Load() != 1 {
		t.Fatalf("expected exactly one snapshot install on n3, got %d", sc.stores["n3"].installs.Load())
	}
	if !n2.IsLearner("n3") {
		t.Fatal("n3 must be a learner after the post-snapshot flip")
	}
	requireIdenticalFSM(t, sc, "n2", "n3")

	// And the repaired member's log identity matches the primary's exactly, which
	// is what makes the next write's log-matching check pass.
	fs, fe := sc.engines["n3"].AppliedFrontier()
	ps, pe := n2.AppliedFrontier()
	if fs != ps || fe != pe {
		t.Fatalf("post-install frontier %v/%v != primary %v/%v", fs, fe, ps, pe)
	}
}

// --- (2) diverged node --------------------------------------------------------

// TestSnapshotRepairsDivergedNode: a target holding a FORK (a write at a shared
// seq assigned under a DIFFERENT epoch) is refused by the log matching and
// was then terminal. Snapshot repairs it, and the divergent write is GONE — not
// merged, not interleaved, DISCARDED. The discard is safe because the target is a
// learner, in no in-flight record's required set.
func TestSnapshotRepairsDivergedNode(t *testing.T) {
	sc := newSnapCluster([]string{"n1", "n2", "n3"}, "n1", 2, []string{"n1", "n2"}, 1)
	n1 := sc.engines["n1"]
	defer n1.Shutdown()

	// n3 holds a fork: seq 1 under epoch 1, a write n1's history never had.
	n3 := sc.engines["n3"]
	if ack := n3.Receive(ReplicateMsg{Epoch: 1, Seq: 1, PrevSeq: 0, PrevEpoch: 0, Data: []byte("FORK")}); !ack.OK {
		t.Fatalf("seeding the fork on n3 failed: %+v", ack)
	}

	// n1's own history at epoch 2 occupies the same seqs with different content.
	for i := 0; i < 10; i++ {
		if _, _, err := n1.Propose(ctxWithTimeout(t, 5*time.Second), []byte{0xE0, byte(i)}); err != nil {
			t.Fatalf("propose %d: %v", i, err)
		}
	}

	// Confirm the premise: without a snapshot store this is ErrCatchupDiverged.
	if err := n1.catchupOnce(ctxWithTimeout(t, 5*time.Second), "n3"); !errors.Is(err, ErrCatchupDiverged) {
		t.Fatalf("premise: the delta path must report divergence, got %v", err)
	}

	if err := n1.StartLearnerCatchup(ctxWithTimeout(t, 10*time.Second), "n3"); err != nil {
		t.Fatalf("StartLearnerCatchup must repair a diverged target via snapshot, got: %v", err)
	}
	requireIdenticalFSM(t, sc, "n1", "n3")
	for _, op := range sc.appliedOps("n3") {
		if bytes.Equal(op, []byte("FORK")) {
			t.Fatal("the divergent write survived the repair: it must be DISCARDED, not merged")
		}
	}
}

// --- (3) THE HEADLINE: a write-dead shard heals itself ------------------------

// TestMinISRFailoverRegainsWritabilityWithoutOperator is the case that is
// PERMANENTLY DEAD without this stage, and it is dead in a way that compounds:
//
//	OpSetShardEpoch resets the ISR to {newPrimary}
//	  -> Propose returns ErrBelowMinISR (minISR=2)
//	    -> the primary cannot write
//	      -> its ring never fills
//	        -> the ring-cold grow can never be served
//	          -> the ISR can never widen -> back to the top.
//
// The primary is killed UNDER LOAD (concurrent, non-quiesced proposers), so the
// survivor's lag is arbitrary rather than a tidy fixture value. The assertion is
// end-to-end: writability returns with NO operator action, and both surviving
// FSMs end byte-identical.
func TestMinISRFailoverRegainsWritabilityWithoutOperator(t *testing.T) {
	sc := newSnapCluster([]string{"n1", "n2", "n3"}, "n1", 1, []string{"n1", "n2", "n3"}, 2)
	n1 := sc.engines["n1"]

	// LOAD: concurrent proposers, killed mid-flight. Nothing is quiesced.
	var wg sync.WaitGroup
	stop := make(chan struct{})
	for w := 0; w < 4; w++ {
		wg.Add(1)
		go func(w int) {
			defer wg.Done()
			for i := 0; ; i++ {
				select {
				case <-stop:
					return
				default:
				}
				//nolint:errcheck,gosec // failures after the kill are expected and are the point
				_, _, _ = n1.Propose(ctxWithTimeout(t, 2*time.Second), []byte(fmt.Sprintf("w%d-%d", w, i)))
			}
		}(w)
	}
	time.Sleep(60 * time.Millisecond)
	// KILL the primary mid-flight: partition it from every peer, then stop it.
	sc.tr.setFault("n1", peerFault{partition: true})
	close(stop)
	wg.Wait()
	n1.Shutdown()

	// FAILOVER (4b): MetaRaft names n2 primary at epoch 2 and RESETS the ISR to
	// {n2}. This is the step that kills the shard.
	n2 := sc.engines["n2"]
	defer n2.Shutdown()
	sc.ctrl.mu.Lock()
	sc.ctrl.epoch, sc.ctrl.primary, sc.ctrl.isr = 2, "n2", []string{"n2"}
	sc.ctrl.mu.Unlock()
	n2.Promote(2, t0+leaseDur)

	// PREMISE 1 — the shard is write-dead.
	if _, _, err := n2.Propose(ctxWithTimeout(t, time.Second), []byte("post-failover")); !errors.Is(err, ErrBelowMinISR) {
		t.Fatalf("premise: a minISR=2 shard with ISR={n2} must refuse writes, got %v", err)
	}
	// PREMISE 2 — and its ring is empty, so no delta can ever serve the grow.
	if n2.backlogLen() != 0 {
		t.Fatalf("premise: a write-blocked promoted primary's ring must be empty, got %d", n2.backlogLen())
	}

	// RECOVERY, with NO operator action: the grow driver's catch-up now falls back
	// to a full state transfer. Note this runs entirely off the ISR floor —
	// RunExclusive and flipLearner require the LEASE, which a below-floor primary
	// still holds.
	if err := n2.StartLearnerCatchup(ctxWithTimeout(t, 15*time.Second), "n3"); err != nil {
		t.Fatalf("recovery catch-up must succeed via snapshot, got: %v", err)
	}
	// The driver then commits the widened ISR and reconciles it.
	sc.ctrl.setISR([]string{"n2", "n3"})
	n2.GrowISR(2, []string{"n2", "n3"})

	// WRITABILITY RESTORED.
	for i := 0; i < 25; i++ {
		if _, _, err := n2.Propose(ctxWithTimeout(t, 5*time.Second), []byte{0xB0, byte(i)}); err != nil {
			t.Fatalf("post-recovery propose %d must commit on the restored ISR, got: %v", i, err)
		}
	}
	eventually(t, func() bool {
		return len(sc.appliedOps("n3")) == len(sc.appliedOps("n2"))
	}, "n3 tracks the primary as a full ISR member")
	requireIdenticalFSM(t, sc, "n2", "n3")
}

// --- (4) abort mid-INSTALL: unusable and un-promotable ------------------------

// TestSnapshotAbortMidInstallLeavesTargetUnusable injects a failure AFTER the
// wipe — the genuinely half-installed state. The target must be UNUSABLE, not
// merely stale: it refuses writes, refuses replicated frames, reports a not-OK
// catch-up identity (so the failover gate treats it as unverifiable), and stays
// that way ACROSS A RESTART. Anything less lets a half-wiped node into the ISR.
func TestSnapshotAbortMidInstallLeavesTargetUnusable(t *testing.T) {
	// Same divergence shape as TestSnapshotRepairsDivergedNode — a fork at an
	// OLDER epoch, which is what log matching can actually detect. (A fork at the
	// SAME epoch and seq is by construction indistinguishable: the chain link is
	// (seq, epoch), not a content hash. That is a known, accepted property of the
	// protocol, not something this stage changes.)
	sc := newSnapCluster([]string{"n1", "n2", "n3"}, "n1", 2, []string{"n1", "n2"}, 1)
	n1 := sc.engines["n1"]
	defer n1.Shutdown()
	n3 := sc.engines["n3"]
	if ack := n3.Receive(ReplicateMsg{Epoch: 1, Seq: 1, PrevSeq: 0, PrevEpoch: 0, Data: []byte("FORK")}); !ack.OK {
		t.Fatalf("seeding the fork on n3: %+v", ack)
	}
	for i := 0; i < 12; i++ {
		if _, _, err := n1.Propose(ctxWithTimeout(t, 5*time.Second), []byte{byte(i)}); err != nil {
			t.Fatalf("propose %d: %v", i, err)
		}
	}
	// n3 now needs a snapshot; make that snapshot fail AFTER the wipe.
	sc.stores["n3"].setFailInstall(true)

	if err := n1.StartLearnerCatchup(ctxWithTimeout(t, 10*time.Second), "n3"); err == nil {
		t.Fatal("a failed install must abort the grow, not report success")
	}

	// (a) POISONED, and the durable fence is still raised.
	if !n3.Poisoned() {
		t.Fatal("a target whose install failed after the wipe must be POISONED")
	}
	if !sc.stores["n3"].InstallPending() {
		t.Fatal("the durable poison fence must remain RAISED after a failed install")
	}
	// (b) UN-PROMOTABLE: the catch-up identity is not-OK AND carries no watermark
	// a caller could misuse.
	info := n3.CatchupInfo()
	if info.OK {
		t.Fatal("a poisoned node must report CatchupInfo{OK:false} so the failover gate never promotes it")
	}
	if info.AppliedSeq != 0 || info.FrontierSeq != 0 || info.FrontierEpoch != 0 {
		t.Fatalf("a poisoned node must not advertise watermarks it cannot back: %+v", info)
	}
	// (c) REFUSES TO SERVE and REFUSES TO RECEIVE.
	if n3.LeaseValid() {
		t.Fatal("a poisoned node must report no valid lease (the seam's refuse-to-serve gate)")
	}
	if ack := n3.Receive(ReplicateMsg{Epoch: 2, Seq: 13, PrevSeq: 12, PrevEpoch: 2, Data: []byte("x")}); ack.OK {
		t.Fatal("a poisoned node must nack replicated writes")
	}
	// (d) Even if the control plane named it primary, it cannot write.
	sc.ctrl.mu.Lock()
	sc.ctrl.primary = "n3"
	sc.ctrl.mu.Unlock()
	n3.GrantLease(2, t0+leaseDur)
	if _, _, err := n3.Propose(ctxWithTimeout(t, time.Second), []byte("nope")); !errors.Is(err, ErrSnapshotPending) {
		t.Fatalf("a poisoned node must refuse to propose with ErrSnapshotPending, got %v", err)
	}
	// (e) And Promote refuses it too (defence in depth behind the failover gate).
	n3.Promote(9, t0+leaseDur)
	if n3.Epoch() == 9 {
		t.Fatal("a poisoned node must refuse promotion")
	}
	sc.ctrl.mu.Lock()
	sc.ctrl.primary = "n1"
	sc.ctrl.mu.Unlock()

	// (f) The poison SURVIVES A RESTART — the whole reason the fence is durable.
	n3b := sc.restart("n3")
	if !n3b.Poisoned() {
		t.Fatal("a node that crashed mid-install must come back POISONED")
	}
	if n3b.CatchupInfo().OK {
		t.Fatal("a restarted poisoned node must still report an unverifiable identity")
	}

	// (g) RE-SNAPSHOTABLE: that is the only exit, and it works.
	sc.stores["n3"].setFailInstall(false)
	if err := n1.StartLearnerCatchup(ctxWithTimeout(t, 10*time.Second), "n3"); err != nil {
		t.Fatalf("a poisoned node must be repairable by a fresh snapshot, got: %v", err)
	}
	if n3b.Poisoned() {
		t.Fatal("a successful install must lower the poison fence")
	}
	requireIdenticalFSM(t, sc, "n1", "n3")
}

// --- (5) primary changes mid-transfer -----------------------------------------

// TestSnapshotEpochChangeMidTransferInstallsNothing: the growing primary loses its
// epoch between raising the fence and taking the install locks. Nothing may be
// installed, and — because the FSM is provably untouched on that path — the fence
// must be LOWERED again rather than leaving a healthy node poisoned.
func TestSnapshotEpochChangeMidTransferInstallsNothing(t *testing.T) {
	sc := newSnapCluster([]string{"n1", "n2"}, "n1", 1, []string{"n1"}, 1)
	n1 := sc.engines["n1"]
	defer n1.Shutdown()
	for i := 0; i < 5; i++ {
		if _, _, err := n1.Propose(ctxWithTimeout(t, 5*time.Second), []byte{byte(i)}); err != nil {
			t.Fatalf("propose %d: %v", i, err)
		}
	}
	n2 := sc.engines["n2"]
	before := sc.appliedOps("n2")

	// n2 has already adopted a HIGHER epoch than the shipping primary's.
	n2.AdoptEpoch(7)
	blob, fseq, fepoch, err := n1.takeSnapshot(1)
	if err != nil {
		t.Fatalf("takeSnapshot: %v", err)
	}
	ack := n2.ReceiveSnapshotChunk(SnapshotChunk{
		Epoch: 1, FrontierSeq: fseq, FrontierEpoch: fepoch,
		Offset: 0, Total: uint64(len(blob)), Final: true, Data: blob,
	})
	if ack.OK {
		t.Fatal("a target at a HIGHER epoch must reject a stale primary's snapshot")
	}
	if n2.Poisoned() {
		t.Fatal("a snapshot rejected at the epoch fence must not poison the target")
	}
	if got := sc.appliedOps("n2"); len(got) != len(before) {
		t.Fatalf("nothing may be installed on a fenced rejection: FSM went %d -> %d ops", len(before), len(got))
	}
	if sc.stores["n2"].installs.Load() != 0 {
		t.Fatal("InstallFSM must never run for a fence-rejected snapshot")
	}
}

// --- (6) abort mid-TRANSFER leaves the target pristine ------------------------

// TestSnapshotPartialTransferLeavesTargetPristine pins the property that makes
// this design's abort story stronger than the brief anticipated: chunks accumulate
// in a MEMORY-ONLY staging buffer, and the poison fence is raised only when the
// final chunk triggers the actual install. So an abort at any point mid-transfer —
// a dropped conn, a bad offset, a hostile stream — leaves the target EXACTLY as it
// was, with no fence to clear and nothing to repair.
func TestSnapshotPartialTransferLeavesTargetPristine(t *testing.T) {
	sc := newSnapCluster([]string{"n1", "n2"}, "n1", 1, []string{"n1"}, 1)
	n1 := sc.engines["n1"]
	defer n1.Shutdown()
	for i := 0; i < 8; i++ {
		if _, _, err := n1.Propose(ctxWithTimeout(t, 5*time.Second), []byte{byte(i)}); err != nil {
			t.Fatalf("propose %d: %v", i, err)
		}
	}
	n2 := sc.engines["n2"]
	before := len(sc.appliedOps("n2"))

	blob, fseq, fepoch := mustSnapshot(t, n1)
	half := uint64(len(blob) / 2)
	base := SnapshotChunk{Epoch: 1, FrontierSeq: fseq, FrontierEpoch: fepoch, Total: uint64(len(blob))}

	c0 := base
	c0.Offset, c0.Data = 0, blob[:half]
	if ack := n2.ReceiveSnapshotChunk(c0); !ack.OK {
		t.Fatalf("first chunk must be accepted: %+v", ack)
	}
	// Abandon here — nothing durable, nothing installed, nothing fenced.
	if n2.Poisoned() || sc.stores["n2"].InstallPending() || sc.stores["n2"].begins.Load() != 0 {
		t.Fatal("a mid-transfer abort must not raise the poison fence: staging is memory-only")
	}
	if got := len(sc.appliedOps("n2")); got != before {
		t.Fatalf("a mid-transfer abort must leave the FSM untouched: %d -> %d ops", before, got)
	}

	// A NON-CONTIGUOUS continuation is rejected and discards the staging buffer, so
	// two streams can never be merged into a Frankenstein blob.
	cBad := base
	cBad.Offset, cBad.Data, cBad.Final = half+7, blob[half:], true
	if ack := n2.ReceiveSnapshotChunk(cBad); ack.OK {
		t.Fatal("a non-contiguous chunk must be rejected")
	}
	if sc.stores["n2"].installs.Load() != 0 {
		t.Fatal("no install may run from a discarded staging buffer")
	}

	// A clean restart of the transfer still works.
	c0.Offset, c0.Data, c0.Final = 0, blob[:half], false
	if ack := n2.ReceiveSnapshotChunk(c0); !ack.OK {
		t.Fatalf("restarted transfer chunk 0: %+v", ack)
	}
	c1 := base
	c1.Offset, c1.Data, c1.Final = half, blob[half:], true
	if ack := n2.ReceiveSnapshotChunk(c1); !ack.OK {
		t.Fatalf("restarted transfer final chunk: %+v", ack)
	}
	requireIdenticalFSM(t, sc, "n1", "n2")
}

// --- (7) ring origin passes F+1 mid-transfer ----------------------------------

// TestSnapshotRingOriginPassesFrontierConverges: the primary writes MORE than a
// full ring capacity between the snapshot and the flip, so F+1 has fallen out of
// the ring and the retry is ring-cold AGAIN. The loop must re-snapshot from the
// newer frontier and converge rather than give up.
//
// A tiny ring (the minimum the engine permits) makes "more than a ring capacity"
// cheap to produce; the mechanism under test is identical at 4096.
func TestSnapshotRingOriginPassesFrontierConverges(t *testing.T) {
	ctrl := &fakeControl{epoch: 1, primary: "n1", isr: []string{"n1"}, minISR: 1}
	tr := newInMemTransport()
	clk := &fakeClock{t: t0}
	sc := &snapCluster{
		cluster: &cluster{ctrl: ctrl, tr: tr, clk: clk,
			engines: map[string]*Engine{}, appliers: map[string]*fakeApplier{}},
		stores: map[string]*fakeSnapStore{},
	}
	for _, id := range []string{"n1", "n2"} {
		ap := &fakeApplier{}
		ss := newFakeSnapStore(ap)
		eng := NewWithRingCapacity(id, testShard, ctrl, tr, ap, pipelineWindow,
			WithClock(clk.now), WithSnapshotStore(ss))
		sc.engines[id], sc.appliers[id], sc.stores[id] = eng, ap, ss
		tr.register(id, eng)
	}
	n1 := sc.engines["n1"]
	n1.GrantLease(1, t0+leaseDur)
	defer n1.Shutdown()

	burst := func(tag byte, n int) {
		t.Helper()
		for i := 0; i < n; i++ {
			if _, _, err := n1.Propose(ctxWithTimeout(t, 5*time.Second), []byte{tag, byte(i), byte(i >> 8)}); err != nil {
				t.Fatalf("propose %c%d: %v", tag, i, err)
			}
		}
	}

	// Over-fill the ring so n2 (at genesis) is ring-cold from the start.
	burst(0xA0, pipelineWindow+44)
	if err := n1.catchupOnce(ctxWithTimeout(t, 5*time.Second), "n2"); !errors.Is(err, ErrGrowRingEvicted) {
		t.Fatalf("premise: n2 must be ring-cold, got %v", err)
	}

	// ROUND 1 transfer only (no flip), establishing frontier F on n2.
	f1, _, err := n1.snapshotCatchup(ctxWithTimeout(t, 10*time.Second), "n2", 1)
	if err != nil {
		t.Fatalf("round-1 snapshot: %v", err)
	}
	if got := sc.stores["n2"].installs.Load(); got != 1 {
		t.Fatalf("round-1 installs = %d, want 1", got)
	}

	// THE HAZARD: the primary writes MORE than a whole ring capacity before the
	// flip could run, so F+1 has fallen out of the ring and the delta is unservable
	// again. This is exactly "the ring origin passed F+1 mid-transfer".
	burst(0xB0, pipelineWindow+44)
	n1.mu.Lock()
	oldest, _, ok := n1.backlog.span()
	n1.mu.Unlock()
	if !ok || oldest <= f1+1 {
		t.Fatalf("premise: the ring origin (%d) must have passed F+1 (%d)", oldest, f1+1)
	}
	if err := n1.catchupOnce(ctxWithTimeout(t, 5*time.Second), "n2"); !errors.Is(err, ErrGrowRingEvicted) {
		t.Fatalf("premise: the retry must be ring-cold again, got %v", err)
	}

	// CONVERGENCE: the loop re-snapshots from the newer frontier and completes.
	if err := n1.StartLearnerCatchup(ctxWithTimeout(t, 20*time.Second), "n2"); err != nil {
		t.Fatalf("the snapshot loop must converge after the ring origin passes F+1, got: %v", err)
	}
	if got := sc.stores["n2"].installs.Load(); got < 2 {
		t.Fatalf("expected a RE-snapshot after the ring origin passed F+1 (installs=%d)", got)
	}
	if !n1.IsLearner("n2") {
		t.Fatal("n2 must be a learner after the converged flip")
	}
	requireIdenticalFSM(t, sc, "n1", "n2")
}

func mustSnapshot(t *testing.T, e *Engine) ([]byte, uint64, uint64) {
	t.Helper()
	blob, fseq, fepoch, err := e.takeSnapshot(e.Epoch())
	if err != nil {
		t.Fatalf("takeSnapshot: %v", err)
	}
	return blob, fseq, fepoch
}

// --- wire codec ---------------------------------------------------------------

func TestSnapshotChunkCodecRoundTrip(t *testing.T) {
	in := SnapshotChunk{
		Epoch: 7, FrontierSeq: 4242, FrontierEpoch: 6,
		Offset: 1024, Total: 999999, Final: true,
		Data: []byte("hello snapshot"),
	}
	out, err := decodeSnapshotChunk(encodeSnapshotChunk(nil, in))
	if err != nil {
		t.Fatalf("decode: %v", err)
	}
	if out.Epoch != in.Epoch || out.FrontierSeq != in.FrontierSeq || out.FrontierEpoch != in.FrontierEpoch ||
		out.Offset != in.Offset || out.Total != in.Total || out.Final != in.Final || !bytes.Equal(out.Data, in.Data) {
		t.Fatalf("round trip mismatch:\n in=%+v\nout=%+v", in, out)
	}
	// A length that disagrees with the body must be REJECTED, not trusted — a
	// wire-declared count never sizes anything here.
	b := encodeSnapshotChunk(nil, in)
	binary.BigEndian.PutUint32(b[41:45], 1<<20)
	if _, err := decodeSnapshotChunk(b); err == nil {
		t.Fatal("a data-length mismatch must be rejected")
	}
}

// TestSnapshotChunkBoundsHostileTotal: an absurd declared Total must be refused
// before it can influence anything, and must never size a reservation.
func TestSnapshotChunkBoundsHostileTotal(t *testing.T) {
	sc := newSnapCluster([]string{"n1", "n2"}, "n1", 1, []string{"n1"}, 1)
	defer sc.engines["n1"].Shutdown()
	n2 := sc.engines["n2"]
	if ack := n2.ReceiveSnapshotChunk(SnapshotChunk{
		Epoch: 1, FrontierSeq: 1, FrontierEpoch: 1,
		Offset: 0, Total: pbSnapshotMaxBytes + 1, Final: false, Data: []byte("x"),
	}); ack.OK {
		t.Fatal("a Total beyond the bound must be rejected")
	}
	if ack := n2.ReceiveSnapshotChunk(SnapshotChunk{
		Epoch: 1, FrontierSeq: 1, FrontierEpoch: 1,
		Offset: 0, Total: 4, Final: true, Data: []byte("far too much data"),
	}); ack.OK {
		t.Fatal("data exceeding the declared Total must be rejected")
	}
}

// --- stall metric -------------------------------------------------------------

// TestSnapshotStallIsMeasured: the write-path freeze is the named cost of this
// stage's flow-control choice, so it must be OBSERVABLE. An operator sizes shards
// against StallMaxNs; a stall that is not reported is a stall that is hidden.
func TestSnapshotStallIsMeasured(t *testing.T) {
	sc := newSnapCluster([]string{"n1", "n2"}, "n1", 1, []string{"n1"}, 1)
	n1 := sc.engines["n1"]
	defer n1.Shutdown()
	for i := 0; i < 10; i++ {
		if _, _, err := n1.Propose(ctxWithTimeout(t, 5*time.Second), []byte{byte(i)}); err != nil {
			t.Fatalf("propose %d: %v", i, err)
		}
	}
	if st := n1.SnapshotStats(); st.Taken != 0 {
		t.Fatalf("no snapshot taken yet: %+v", st)
	}
	if _, _, err := n1.snapshotCatchup(ctxWithTimeout(t, 5*time.Second), "n2", 1); err != nil {
		t.Fatalf("snapshotCatchup: %v", err)
	}
	st := n1.SnapshotStats()
	if st.Taken != 1 {
		t.Fatalf("Taken = %d, want 1", st.Taken)
	}
	if st.StallMaxNs < st.StallLastNs {
		t.Fatalf("StallMax (%d) must dominate StallLast (%d)", st.StallMaxNs, st.StallLastNs)
	}
	if got := sc.engines["n2"].SnapshotStats().Installed; got != 1 {
		t.Fatalf("target Installed = %d, want 1", got)
	}
}
