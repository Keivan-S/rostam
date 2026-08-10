// SPDX-License-Identifier: Apache-2.0

package pbisr

import (
	"sync"
	"sync/atomic"
	"testing"
	"time"
)

// ============================================================================
// ISR GROW — catch-up transport + learner flip,
// non-blocking learner send, GrowISR).
//
// The adversarial invariant under test throughout: catching a lagging survivor
// up and re-adding it NEVER opens a seq gap on it and NEVER loses / falsely
// commits a write. A learner is shipped every write but gates NO commit and is
// added to NO in-flight record's pending set (the no-widen-in-flight rule).
// ============================================================================

// --- (a) catch-up + flip catches a lagging survivor to high-water ---------------

// TestGrowCatchupToHighWater: n1 (primary) proposes N committed on the writable
// ISR {n1,n2}; n3 is a lagging out-of-ISR survivor that received none of them.
// StartLearnerCatchup handshakes n3's high-water, backfills within a window, and
// FLIPS n3 into the learner ship-set with the final delta pre-loaded on its
// fresh sender. n3 then catches to the exact high-water, gap-free.
func TestGrowCatchupToHighWater(t *testing.T) {
	c := newCluster([]string{"n1", "n2", "n3"}, "n1", 1, []string{"n1", "n2"}, 2)
	primary := c.engines["n1"]
	defer primary.Shutdown()

	const n = 50
	for i := 0; i < n; i++ {
		if _, _, err := primary.Propose(ctxWithTimeout(t, 5*time.Second), []byte{byte(i)}); err != nil {
			t.Fatalf("propose %d: %v", i, err)
		}
	}
	if got := c.engines["n3"].LastApplied(); got != 0 {
		t.Fatalf("n3 LastApplied before catch-up = %d, want 0", got)
	}

	if err := primary.StartLearnerCatchup(ctxWithTimeout(t, 5*time.Second), "n3"); err != nil {
		t.Fatalf("StartLearnerCatchup: %v", err)
	}
	if !primary.IsLearner("n3") {
		t.Fatal("n3 must be a learner after the flip")
	}
	// The pre-loaded fresh sender ships [1..50] to n3 asynchronously.
	eventually(t, func() bool { return c.engines["n3"].LastApplied() == n },
		"n3 catches to the exact high-water via its pre-loaded learner sender")
	if got := c.appliers["n3"].count(); got != n {
		t.Fatalf("n3 applied %d ops, want %d (gap-free)", got, n)
	}
}

// --- (b) re-add restores full-ISR durability -----------------------------------

// TestGrowReAddRestoresFullISR is the core Stage-2 correctness case. ISR {n1,n2}
// is writable; n3 lags at 0. We catch n3, then SIMULATE the driver (observe the
// committed widen → GrowISR) by widening ctrl.ISR to {n1,n2,n3} and calling
// GrowISR. Every write proposed AFTER the grow requires n3, lands on all three,
// and commits — proof the re-add is gap-free and durability is restored.
func TestGrowReAddRestoresFullISR(t *testing.T) {
	c := newCluster([]string{"n1", "n2", "n3"}, "n1", 1, []string{"n1", "n2"}, 2)
	primary := c.engines["n1"]
	defer primary.Shutdown()

	const pre = 100
	for i := 0; i < pre; i++ {
		if _, _, err := primary.Propose(ctxWithTimeout(t, 5*time.Second), []byte{byte(i)}); err != nil {
			t.Fatalf("pre-grow propose %d: %v", i, err)
		}
	}

	if err := primary.StartLearnerCatchup(ctxWithTimeout(t, 5*time.Second), "n3"); err != nil {
		t.Fatalf("StartLearnerCatchup: %v", err)
	}
	eventually(t, func() bool { return c.engines["n3"].LastApplied() == pre },
		"n3 catches up as a learner")

	// Simulate the driver observing OpSetShardISR(1,{n1,n2,n3}) committed.
	c.ctrl.setISR([]string{"n1", "n2", "n3"})
	primary.GrowISR(1, []string{"n1", "n2", "n3"})

	const post = 60
	for i := 0; i < post; i++ {
		if _, _, err := primary.Propose(ctxWithTimeout(t, 5*time.Second), []byte{0xA0, byte(i)}); err != nil {
			t.Fatalf("post-grow propose %d: %v (a gap-reject would surface here)", i, err)
		}
	}

	total := uint64(pre + post)
	if got := primary.Committed(); got != total {
		t.Fatalf("committed = %d, want %d (post-grow writes require n3 and must commit)", got, total)
	}
	eventually(t, func() bool { return c.engines["n3"].LastApplied() == total },
		"n3 has every write, gap-free, as a full voter")
	// Every acked write lands on all three replicas.
	for _, id := range []string{"n1", "n2", "n3"} {
		if got := c.appliers[id].count(); got != int(total) {
			t.Fatalf("%s applied %d, want %d (every acked write on all 3)", id, got, total)
		}
	}
}

// --- (c) boundary: writes proposed ACROSS the flip -----------------------------

// TestGrowBoundaryConcurrentProposes runs a proposer continuously (the tail
// MOVES) while StartLearnerCatchup catches n3 and re-adds it. The flip happens
// amid live seq assignment; the invariant is that no write ever gap-rejects on
// n3 and, once quiesced, all three replicas hold an identical, continuous log.
func TestGrowBoundaryConcurrentProposes(t *testing.T) {
	c := newCluster([]string{"n1", "n2", "n3"}, "n1", 1, []string{"n1", "n2"}, 2)
	primary := c.engines["n1"]
	defer primary.Shutdown()

	var stop atomic.Bool
	var proposeErr atomic.Pointer[error]
	var wg sync.WaitGroup
	wg.Add(1)
	go func() {
		defer wg.Done()
		i := 0
		for !stop.Load() {
			if _, _, err := primary.Propose(ctxWithTimeout(t, 5*time.Second), []byte{byte(i)}); err != nil {
				e := err
				proposeErr.Store(&e)
				return
			}
			i++
			time.Sleep(200 * time.Microsecond) // keep the tail moving without starving catch-up
		}
	}()

	// Let the tail move, then catch n3 up amid live proposes (the flip races writes).
	time.Sleep(20 * time.Millisecond)
	if err := primary.StartLearnerCatchup(ctxWithTimeout(t, 5*time.Second), "n3"); err != nil {
		stop.Store(true)
		wg.Wait()
		t.Fatalf("StartLearnerCatchup across a moving tail: %v", err)
	}
	// Re-add (simulate the driver): from here every new write requires n3.
	c.ctrl.setISR([]string{"n1", "n2", "n3"})
	primary.GrowISR(1, []string{"n1", "n2", "n3"})

	// Keep proposing across/after the grow, then quiesce.
	time.Sleep(20 * time.Millisecond)
	stop.Store(true)
	wg.Wait()
	if ep := proposeErr.Load(); ep != nil {
		t.Fatalf("a propose failed across the flip/grow: %v (gap-reject / wedge)", *ep)
	}

	// Once quiesced, n3 (now a full voter) must converge to the exact frontier and
	// all three logs must be identical and continuous.
	last := primary.LastSeq()
	eventually(t, func() bool { return primary.Committed() == last }, "all writes commit under full ISR {n1,n2,n3}")
	eventually(t, func() bool { return c.engines["n3"].LastApplied() == last }, "n3 converges to the frontier gap-free")
	if a, b, cc := c.appliers["n1"].count(), c.appliers["n2"].count(), c.appliers["n3"].count(); a != b || b != cc {
		t.Fatalf("replica apply counts diverge: n1=%d n2=%d n3=%d", a, b, cc)
	}
}

// --- (d') stalled-learner RE-ADD: abandon → compensation un-wedges -------------

// blockableTransport delivers to registered peer engines like the inmem transport
// (Replicate / ReplicateGroup / CatchupRequest) but can BLOCK all replication to a
// chosen peer — modelling a degraded link that stalls that peer's sender goroutine
// so its bounded channel fills. The catch-up HANDSHAKE is never blocked (a grow
// must still be able to learn a stalled peer's high-water).
type blockableTransport struct {
	mu    sync.Mutex
	peers map[string]*Engine
	gate  map[string]chan struct{} // peer -> an OPEN channel means "block on it"
}

func newBlockableTransport() *blockableTransport {
	return &blockableTransport{peers: make(map[string]*Engine), gate: make(map[string]chan struct{})}
}
func (t *blockableTransport) register(id string, e *Engine) {
	t.mu.Lock()
	t.peers[id] = e
	t.mu.Unlock()
}
func (t *blockableTransport) block(peer string) {
	t.mu.Lock()
	if t.gate[peer] == nil {
		t.gate[peer] = make(chan struct{})
	}
	t.mu.Unlock()
}
func (t *blockableTransport) unblock(peer string) {
	t.mu.Lock()
	g := t.gate[peer]
	delete(t.gate, peer)
	t.mu.Unlock()
	if g != nil {
		close(g)
	}
}
func (t *blockableTransport) waitGate(peer string) {
	t.mu.Lock()
	g := t.gate[peer]
	t.mu.Unlock()
	if g != nil {
		<-g // blocks until unblock() closes it (or returns at once if already closed)
	}
}
func (t *blockableTransport) Replicate(peer string, msg ReplicateMsg, done func(AckMsg, error)) error {
	t.waitGate(peer)
	t.mu.Lock()
	e := t.peers[peer]
	t.mu.Unlock()
	if e == nil {
		done(AckMsg{Epoch: msg.Epoch, Seq: msg.Seq, OK: false}, nil)
		return nil
	}
	done(e.Receive(msg), nil)
	return nil
}
func (t *blockableTransport) ReplicateGroup(peer string, msgs []ReplicateMsg, done func(AckMsg, error)) error {
	t.waitGate(peer)
	t.mu.Lock()
	e := t.peers[peer]
	t.mu.Unlock()
	if e == nil {
		done(AckMsg{Epoch: msgs[0].Epoch, Seq: msgs[0].Seq - 1, OK: false}, nil)
		return nil
	}
	done(e.ReceiveGroup(msgs), nil)
	return nil
}
func (t *blockableTransport) CatchupRequest(peer string, epoch uint64) (CatchupInfoMsg, error) {
	t.mu.Lock() // NOT gated: the handshake must work even on a degraded link
	e := t.peers[peer]
	t.mu.Unlock()
	if e == nil {
		return CatchupInfoMsg{}, errInMemPeerUnreachable
	}
	return e.CatchupInfo(), nil
}

// TestGrowStalledLearnerReAddCompensationUnwedges drives the CRITICAL abandon race
// and its fix: a learner is caught up and flipped, then its link DEGRADES so its
// channel fills and the grow ABANDONS it (recording the signal). If the driver
// then (buggily, unconditionally) widened the ISR to include the gapped member,
// every write requiring it would gap-reject FOREVER under CommitFullISR (the wedge
// grow claims to close). The fix: the abandon is observable (LearnerAbandoned),
// and the driver's COMPENSATION — a re-narrow back to S (modelled here by the same
// ShrinkISR the driver issues) — un-wedges the pipeline. The invariant: the shard
// does NOT permanently wedge, and committed advances.
func TestGrowStalledLearnerReAddCompensationUnwedges(t *testing.T) {
	ctrl := &fakeControl{epoch: 1, primary: "n1", isr: []string{"n1", "n2"}, minISR: 2}
	clk := &fakeClock{t: t0}
	tr := newBlockableTransport()
	n1 := New("n1", testShard, ctrl, tr, &fakeApplier{}, WithClock(clk.now))
	n1.GrantLease(1, t0+leaseDur)
	tr.register("n1", n1)
	defer func() { tr.unblock("n3"); n1.Shutdown() }()

	n2 := New("n2", testShard, ctrl, tr, &fakeApplier{}, WithClock(clk.now))
	tr.register("n2", n2)
	defer n2.Shutdown()
	n3 := New("n3", testShard, ctrl, tr, &fakeApplier{}, WithClock(clk.now))
	tr.register("n3", n3)
	defer n3.Shutdown()

	// Commit a few writes on the writable ISR {n1,n2} (n3 is out of the ISR).
	const pre = 5
	for i := 0; i < pre; i++ {
		if _, _, err := n1.ProposeDeadline([]byte{byte(i)}, 5*time.Second); err != nil {
			t.Fatalf("pre put %d: %v", i, err)
		}
	}

	// Catch n3 up and flip it into the learner ship-set (link healthy during catch-up).
	if err := n1.StartLearnerCatchup(ctxWithTimeout(t, 5*time.Second), "n3"); err != nil {
		t.Fatalf("StartLearnerCatchup: %v", err)
	}
	eventually(t, func() bool { return n3.LastApplied() == pre }, "n3 catches up as a learner")

	// DEGRADE n3's link: its sender now stalls, so learner sends pile up in its
	// bounded channel. A burst of writes (ISR still {n1,n2}, so they COMMIT without
	// n3) fills the channel and ABANDONS the grow.
	tr.block("n3")
	for i := 0; i < pipelineWindow+30; i++ {
		if _, _, err := n1.ProposeDeadline([]byte{byte(i)}, 5*time.Second); err != nil {
			t.Fatalf("burst put %d must still commit on {n1,n2}: %v", i, err)
		}
	}
	if !n1.LearnerAbandoned("n3") {
		t.Fatal("a stalled learner must be recorded as abandoned (the driver's coordination signal)")
	}
	if n1.IsLearner("n3") {
		t.Fatal("an abandoned learner must be dropped from the ship-set")
	}

	// The BUGGY driver widens the ISR to include the gapped n3 (it did not observe
	// the abandon). Now every new write REQUIRES n3, which is gapped + blocked.
	ctrl.setISR([]string{"n1", "n2", "n3"})
	n1.GrowISR(1, []string{"n1", "n2", "n3"})

	committedBefore := n1.Committed()
	// Fire a write that now requires the gapped n3: under CommitFullISR it cannot
	// commit (n3 gap-rejects / is blocked) — it would wedge WITHOUT the compensation.
	wedge := make(chan error, 1)
	go func() {
		_, _, err := n1.ProposeDeadline([]byte("wedged"), 5*time.Second)
		wedge <- err
	}()
	// Let the write register and stall on n3.
	eventually(t, func() bool { return n1.LastSeq() > committedBefore }, "the n3-required write registers")

	// COMPENSATION (what the grow driver does on seeing LearnerAbandoned("n3")):
	// re-narrow the committed ISR back to {n1,n2} so the gapped voter is removed and
	// the pipeline un-wedges.
	ctrl.setISR([]string{"n1", "n2"})
	n1.ShrinkISR(1, []string{"n1", "n2"})

	select {
	case err := <-wedge:
		if err != nil {
			t.Fatalf("the write did not commit after the compensating re-narrow: %v (permanent wedge)", err)
		}
	case <-time.After(3 * time.Second):
		t.Fatal("the shard stayed WEDGED after the compensating re-narrow — the fix failed")
	}
	if n1.Committed() <= committedBefore {
		t.Fatalf("committed did not advance after compensation: %d <= %d (wedge)", n1.Committed(), committedBefore)
	}
}

// --- (d) non-blocking learner send abandons a hopeless grow -------------------

// TestGrowLearnerAbandonedOnFullChannel: a learner whose sender is stalled (its
// channel fills) must be ABANDONED by the write path, never block it. We install
// a learner over a gated (blocking) transport, then propose past the channel
// buffer: the write path drops the learner and keeps committing on the required
// set (here the single-node ISR {n1}).
func TestGrowLearnerAbandonedOnFullChannel(t *testing.T) {
	ctrl := &fakeControl{epoch: 1, primary: "n1", isr: []string{"n1"}, minISR: 1}
	clk := &fakeClock{t: t0}
	g := newGatedTransport() // blocks in Replicate; the learner sender stalls
	e := New("n1", testShard, ctrl, g, &fakeApplier{}, WithClock(clk.now))
	e.GrantLease(1, t0+leaseDur)
	defer func() { g.release(); e.Shutdown() }()

	// Install n3 as a learner with a fresh (empty) sender; the sender goroutine
	// blocks in the first gated Replicate as soon as it gets a frame.
	e.writeMu.Lock()
	e.mu.Lock()
	e.createLearnerSenderLocked("n3", nil)
	e.learners = map[string]bool{"n3": true}
	e.mu.Unlock()
	e.writeMu.Unlock()

	// Propose past the learner channel buffer. Each write ships to n3 as a learner
	// (single-node ISR ⇒ peers empty, so commit never waits on n3). Once the
	// channel fills, submitLearnerLocked abandons n3 instead of blocking.
	const n = pipelineWindow + 20
	for i := 0; i < n; i++ {
		if _, _, err := e.ProposeDeadline([]byte{byte(i)}, 2*time.Second); err != nil {
			t.Fatalf("propose %d must not block/fail on a stalled learner: %v", i, err)
		}
	}
	if e.IsLearner("n3") {
		t.Fatal("a stalled learner must be abandoned once its channel fills")
	}
	if got := e.Committed(); got != n {
		t.Fatalf("committed = %d, want %d (the write path proceeds despite the abandoned grow)", got, n)
	}
}

// --- (e) an epoch bump mid-catch-up aborts the grow ---------------------------

// epochBumpOnGroup wraps an inMemTransport and, on the FIRST group frame shipped
// (deterministically, the first backfill round), bumps the primary's epoch — so
// the NEXT backfill round's re-fence must observe the superseded epoch and abort.
type epochBumpOnGroup struct {
	*inMemTransport
	primary *Engine
	bumpTo  uint64
	once    sync.Once
}

func (w *epochBumpOnGroup) ReplicateGroup(peer string, msgs []ReplicateMsg, done func(AckMsg, error)) error {
	return w.inMemTransport.ReplicateGroup(peer, msgs, func(ack AckMsg, err error) {
		w.once.Do(func() { w.primary.AdoptEpoch(w.bumpTo) })
		done(ack, err)
	})
}

// TestGrowEpochBumpAbortsCatchup: if the engine's epoch advances during the
// backfill (a failover), the catch-up aborts (ErrGrowEpochChanged) rather than
// re-adding a member under a superseded epoch.
func TestGrowEpochBumpAbortsCatchup(t *testing.T) {
	ctrl := &fakeControl{epoch: 1, primary: "n1", isr: []string{"n1"}, minISR: 1}
	clk := &fakeClock{t: t0}
	inner := newInMemTransport()
	tr := &epochBumpOnGroup{inMemTransport: inner, bumpTo: 2}
	primary := New("n1", testShard, ctrl, tr, &fakeApplier{}, WithClock(clk.now))
	tr.primary = primary
	primary.GrantLease(1, t0+leaseDur)
	inner.register("n1", primary)
	defer primary.Shutdown()

	lag := New("n3", testShard, ctrl, inner, &fakeApplier{}, WithClock(clk.now))
	inner.register("n3", lag)
	defer lag.Shutdown()

	// Enough writes that catching n3 to within a window takes MORE than one round,
	// so the second round's re-fence runs after the first-round epoch bump.
	for i := 0; i < 3*growBackfillBatch; i++ {
		if _, _, err := primary.ProposeDeadline([]byte{byte(i)}, 2*time.Second); err != nil {
			t.Fatalf("propose %d: %v", i, err)
		}
	}

	err := primary.StartLearnerCatchup(ctxWithTimeout(t, 3*time.Second), "n3")
	if err != ErrGrowEpochChanged {
		t.Fatalf("epoch-bump catch-up: err = %v, want ErrGrowEpochChanged (grow must not succeed at the stale epoch)", err)
	}
	if primary.IsLearner("n3") {
		t.Fatal("a grow aborted by an epoch bump must not leave a learner installed")
	}
}

// --- catch-up aborts (retained) ------------------------------------------------

func TestGrowCatchupAbortsWhenPeerAhead(t *testing.T) {
	c := newCluster([]string{"n1", "n2", "n3"}, "n1", 1, []string{"n1", "n2"}, 2)
	primary := c.engines["n1"]
	defer primary.Shutdown()

	c.engines["n3"].AdoptEpoch(5) // a newer generation the grower has not caught up to
	if err := primary.StartLearnerCatchup(ctxWithTimeout(t, 2*time.Second), "n3"); err != ErrGrowPeerAhead {
		t.Fatalf("catch-up of an ahead peer: err = %v, want ErrGrowPeerAhead", err)
	}
}

func TestGrowCatchupRingEvicted(t *testing.T) {
	ctrl := &fakeControl{epoch: 1, primary: "n1", isr: []string{"n1"}, minISR: 1}
	clk := &fakeClock{t: t0}
	tr := newInMemTransport()
	primary := NewWithRingCapacity("n1", testShard, ctrl, tr, &fakeApplier{}, pipelineWindow, WithClock(clk.now))
	primary.GrantLease(1, t0+leaseDur)
	tr.register("n1", primary)
	defer primary.Shutdown()

	lag := New("n3", testShard, ctrl, tr, &fakeApplier{}, WithClock(clk.now))
	tr.register("n3", lag)
	defer lag.Shutdown()

	// Overflow the ring so seq 1 (which n3 at 0 still needs) is evicted.
	for i := 0; i < pipelineWindow+10; i++ {
		if _, _, err := primary.ProposeDeadline([]byte{byte(i)}, 2*time.Second); err != nil {
			t.Fatalf("propose %d: %v", i, err)
		}
	}
	if err := primary.StartLearnerCatchup(ctxWithTimeout(t, 2*time.Second), "n3"); err != ErrGrowRingEvicted {
		t.Fatalf("catch-up of a ring-evicted survivor: err = %v, want ErrGrowRingEvicted", err)
	}
}

// --- Hardening: ring-capacity floor panics at construction ----------------------

func TestNewBelowRingFloorPanics(t *testing.T) {
	defer func() {
		if recover() == nil {
			t.Fatal("NewWithRingCapacity(ringCapacity < pipelineWindow) must panic")
		}
	}()
	ctrl := &fakeControl{epoch: 1, primary: "n1", isr: []string{"n1"}, minISR: 1}
	_ = NewWithRingCapacity("n1", testShard, ctrl, newInMemTransport(), &fakeApplier{}, pipelineWindow-1)
}
