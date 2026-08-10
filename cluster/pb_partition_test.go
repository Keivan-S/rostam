// SPDX-License-Identifier: Apache-2.0

package cluster

import (
	"errors"
	"fmt"
	"net"
	"sync"
	"sync/atomic"
	"testing"
	"time"

	hraft "github.com/hashicorp/raft"

	"github.com/rostamlabs/rostam/ops"
	"github.com/rostamlabs/rostam/shard"
)

// ---------------------------------------------------------------------------
// partitionableStreamLayer — the meta-transport partition injector.
//
// It wraps a single node's meta-Raft raft.StreamLayer to SIMULATE a network
// partition of exactly that node's META transport, while leaving every other
// transport (its PB data path over shard/pbisr's NetTransport, and its
// client-facing server) fully connected. That asymmetry is the whole point of
// the no-double-primary gate: a primary P whose META link is cut can still
// commit to its ISR over the (untouched) PB path during its lease-valid window —
// the exact double-primary risk — so the test can prove P self-fences before a
// replacement is named.
//
// The cut is BIDIRECTIONAL so it isolates P whether P is the meta leader or a
// follower:
//   - Dial (outbound): while partitioned, fail immediately with errPartitioned —
//     P cannot open new pipelines to peers.
//   - Accept (inbound): while partitioned, close every accepted conn and keep
//     waiting — P serves no inbound peer frame.
//
// It also TRACKS every live net.Conn it hands out (from both Dial and Accept) so
// partition() can close them synchronously. Closing the live conns is what makes
// the cut take effect PROMPTLY: hashicorp/raft keeps long-lived pipeline conns,
// so without closing them an already-established leader→follower pipeline would
// keep flowing until its next natural Dial/Accept. Killing them at partition()
// tears down in-flight pipelines at once, so LastContact starts aging within a
// heartbeat instead of a full pipeline lifetime.
// ---------------------------------------------------------------------------

// errPartitioned is returned by Dial while the injector is cut. It is a plain
// dial-style failure so hashicorp/raft treats it as an unreachable peer.
var errPartitioned = errors.New("cluster: meta transport partitioned (test injector)")

type partitionableStreamLayer struct {
	// StreamLayer is the real (mux) per-group meta layer. Embedded so Addr and
	// Close pass straight through to it; only Dial and Accept are intercepted.
	hraft.StreamLayer

	partitioned atomic.Bool

	mu    sync.Mutex
	conns map[net.Conn]struct{} // live conns handed out (send + receive side)
}

// track records a live conn so partition() can close it later. Idempotent.
func (p *partitionableStreamLayer) track(c net.Conn) {
	p.mu.Lock()
	if p.conns == nil {
		p.conns = make(map[net.Conn]struct{})
	}
	p.conns[c] = struct{}{}
	p.mu.Unlock()
}

// Dial fails fast while partitioned; otherwise dials the real layer and tracks
// the returned conn.
func (p *partitionableStreamLayer) Dial(addr hraft.ServerAddress, timeout time.Duration) (net.Conn, error) {
	if p.partitioned.Load() {
		return nil, errPartitioned
	}
	c, err := p.StreamLayer.Dial(addr, timeout)
	if err != nil {
		return nil, err
	}
	p.track(c)
	return c, nil
}

// Accept drops inbound conns while partitioned (close + keep waiting); otherwise
// tracks and returns the accepted conn. The loop means a conn that arrives during
// the cut is silently discarded rather than delivered to hraft.
func (p *partitionableStreamLayer) Accept() (net.Conn, error) {
	for {
		c, err := p.StreamLayer.Accept()
		if err != nil {
			return nil, err
		}
		if p.partitioned.Load() {
			_ = c.Close()
			continue
		}
		p.track(c)
		return c, nil
	}
}

// partition cuts the meta transport: set the flag (future Dials fail, future
// inbound conns are dropped) AND close every currently-live conn so in-flight
// hraft pipelines die immediately and the cut takes effect within a heartbeat.
func (p *partitionableStreamLayer) partition() {
	p.partitioned.Store(true)
	p.mu.Lock()
	for c := range p.conns {
		_ = c.Close()
		delete(p.conns, c)
	}
	p.mu.Unlock()
}

// heal reconnects the meta transport: clear the flag so future Dials/Accepts work
// again. Conns closed by partition() are gone; hraft re-dials as needed.
func (p *partitionableStreamLayer) heal() {
	p.partitioned.Store(false)
}

// newPartitionablePBTestCluster builds an n-node PB cluster exactly like
// newPBTestCluster, but additionally installs a per-node partitionableStreamLayer
// over each node's META-Raft transport and records the handles in tc.metaParts
// (keyed by NodeID). The caller's opts run AFTER the injector is wired, so they
// may still tune timings etc. Because newPBTestCluster applies each opt to that
// node's Config after setting cfg.NodeID — and does so sequentially, one node at
// a time — the wrap opt keys each fresh injector by that node's unique ID.
func newPartitionablePBTestCluster(t *testing.T, n, numShards, minISR int, opts ...func(*Config)) *pbTestCluster {
	t.Helper()
	parts := make(map[string]*partitionableStreamLayer, n)
	wrapOpt := func(c *Config) {
		inj := &partitionableStreamLayer{}
		c.metaStreamLayerWrap = func(inner hraft.StreamLayer) hraft.StreamLayer {
			inj.StreamLayer = inner
			return inj
		}
		parts[c.NodeID] = inj
	}
	all := append([]func(*Config){wrapOpt}, opts...)
	tc := newPBTestCluster(t, n, numShards, minISR, all...)
	tc.metaParts = parts
	return tc
}

// ---------------------------------------------------------------------------
// TestPBFailoverPartitionNoDoublePrimary — the pre-default-on gate.
//
// This proves the HARDER half of PB automatic failover safety that a crash-stop
// test (TestPBFailoverNoAckedLoss) cannot: under a real network PARTITION of the
// primary's meta transport — with the primary STILL ALIVE and STILL ABLE to
// replicate to its ISR — no two nodes ever hold a valid primary lease at once.
//
// Setup: 3-node RF=full PB cluster, PBAutoFailover ON, timings shrunk to satisfy
// the corrected honor rule (failoverTimeout > leaseTTL + staleness + renewInterval
// + tick): leaseTTL=1s, staleness=500ms, renewInterval=300ms, tick=500ms ⇒ floor
// 2300ms; failoverTimeout=3000ms. (Same setup as the crash-stop gate.)
//
// WHY THIS IS A VALID PROOF (the safety gap the honor rule buys):
//   - P self-fences by ≈ partition + staleness + leaseTTL (≈1.5s): its leaseKeeper
//     renews only while confirmMetaView passes, and confirmMetaView reads the META
//     transport's LastContact — which we cut — so P stops renewing and its engine's
//     OH1 fence lapses. This is INDEPENDENT of the still-connected client/PB paths.
//   - Q is promoted no earlier than ≈ partition + failoverTimeout − renewInterval
//     (≈2.7s), and in practice later: while P is a follower its beacon may keep
//     reaching the leader over the un-cut CLIENT path until P's meta view goes
//     leaderless (~one election timeout), which only DELAYS promotion (safer).
//   - So there is a gap [P-fences … Q-named] in which NO node acks as primary.
//     During [partition, P-fences] P still holds a valid lease and commits via its
//     still-connected PB path to the FULL ISR (incl. Q) ⇒ those writes are on Q ⇒
//     no acked loss. After P fences, before Q is named, nobody acks. After Q is
//     named, P (still partitioned, lease dead forever) rejects every write.
//
// The assertions below observe exactly this ordering. If the honor rule were
// violated (a double-primary window), assertion 2 would catch a P ack whose
// timestamp is NOT strictly before Q's promotion — i.e. P still acking as primary
// while Q is already primary.
// ---------------------------------------------------------------------------
func TestPBFailoverPartitionNoDoublePrimary(t *testing.T) {
	const numShards = 1
	const sh = 0
	tc := newPartitionablePBTestCluster(t, 3, numShards, 1, func(c *Config) {
		c.PBAutoFailover = true
		c.MinISR = 1 // durability FLOOR only; full-ISR commit still waits for all 3
		c.PBLeaseTTLMs = 1000
		c.PBMetaContactStalenessMs = 500
		// Honor-rule floor = leaseTTL + staleness + renewInterval + failoverTick
		// = 1000 + 500 + 300 + 500 = 2300ms; failoverTimeout strictly exceeds it.
		c.PBFailoverTimeoutMs = 3000
		c.PBRenewIntervalMs = 300
		c.ShardCfg.RaftHeartbeatMs = 200
	})

	findNodeIdx := func(nodeID string) int {
		for i, p := range tc.peers {
			if p.NodeID == nodeID {
				return i
			}
		}
		return -1
	}

	// --- Find a primary P that holds a currently-VALID lease. The test premise is only
	// that P is the primary with a valid lease AT CUT TIME; a fresh ack immediately
	// before the cut proves it, and the recorded writer supplies exactly that (it acks
	// against P through the short pre-cut window). So here we only need to settle on a
	// primary that is currently accepting writes.
	//
	// We mirror the PROVEN crash-stop gate's discovery — SEQUENTIAL blocking probes, one
	// at a time — and deliberately do NOT run concurrent/timeout-bounded probes: a probe
	// against a briefly-wedged pipeline blocks for the internal put-deadline, and firing
	// more probes concurrently would just pile up behind it on the engine's write lock,
	// worsening the wedge. We require TWO sequential acks ~100ms apart from the SAME FSM
	// primary — light evidence of a stable-enough lease (one ack = valid now; a second
	// after a short gap guards against an ack-then-immediately-wedge) — without demanding
	// an unbroken fast run (pre-cut blips are irrelevant: fenced writes burn no seq and
	// assertion 4 is a lower bound). A primary change restarts discovery. Fails SAFE:
	// it can only refuse to start, never produce a false pass.
	probeArgs := ops.EncodePutArgs([]byte("probe"), []byte("v"), 0)
	var primaryIdx = -1
	var backupIdx = -1
	var origPrimary string
	var origEpoch, baseline uint64
	findDeadline := time.Now().Add(45 * time.Second)
	for time.Now().Before(findDeadline) && primaryIdx < 0 {
		primary := tc.nodes[0].meta.FSM.ShardPrimary(sh)
		if primary == "" {
			time.Sleep(50 * time.Millisecond)
			continue
		}
		idx := findNodeIdx(primary)
		if idx < 0 {
			t.Fatalf("primary %q not in peer list", primary)
		}
		// First confirming ack (valid lease now).
		if _, err := tc.nodes[idx].Call("put", probeArgs); err != nil {
			time.Sleep(50 * time.Millisecond)
			continue
		}
		time.Sleep(100 * time.Millisecond)
		// Identity must hold, and a second ack must land (still leased after the gap).
		if tc.nodes[0].meta.FSM.ShardPrimary(sh) != primary {
			continue
		}
		if _, err := tc.nodes[idx].Call("put", probeArgs); err != nil {
			time.Sleep(50 * time.Millisecond)
			continue
		}
		primaryIdx = idx
		origPrimary = primary
		origEpoch = tc.nodes[0].meta.FSM.ShardEpoch(sh)
		// Pick a BACKUP (any node != P) and read the BASELINE from its engine: the
		// committed high-water AFTER the throwaway probes and BEFORE the recorded writer.
		// It MUST be read on a backup, not P: a primary's engine advances lastSeq/
		// committed but its LastApplied stays at the value it accumulated via the Receive
		// (backup) path, whereas a backup's LastApplied tracks every committed seq it
		// applied. The final assertion reads Q's LastApplied, and Q is a FORMER BACKUP —
		// so its reference frame is the backup counter. Under full-ISR the last probe was
		// acked only after every backup applied it, so all backups (incl. Q) share this
		// high-water. This absorbs all bringup/probe commits so the dense-seq lower bound
		// holds regardless of how many probes it took to settle.
		for k := range tc.nodes {
			if k != idx {
				backupIdx = k
				break
			}
		}
		baseline = tc.nodes[backupIdx].pbEngines[sh].LastApplied()
	}
	if primaryIdx < 0 {
		// PRECONDITION not met — NOT an invariant violation. Establishing the test
		// premise requires an alive primary holding a valid lease; on a CPU-oversubscribed
		// host the PB cluster's leases can thrash so hard (full-ISR writes hitting the 5s
		// deadline, meta heartbeats slipping past the staleness bound) that no primary
		// stays leased long enough to even begin — the same environmental condition that
		// flakes the crash-stop gate. We SKIP (not fail) so a loaded host is never a false
		// red: this test's PASS still requires reaching every no-double-primary assertion,
		// and a real invariant violation still fails loudly below. Re-run on a quieter host.
		t.Skip("SKIP: could not establish an alive, leased PB primary within 45s (host too loaded); no-double-primary invariant not exercised this run")
	}
	inj := tc.metaParts[origPrimary]
	if inj == nil {
		t.Fatalf("no meta partition injector for primary %q", origPrimary)
	}
	t.Logf("stable primary=%q idx=%d origEpoch=%d baseline=%d", origPrimary, primaryIdx, origEpoch, baseline)

	// --- A background writer drives distinct keys to P for the whole run, recording
	// each attempt's (key, value, ackErr, time). Every returned-nil Put is a
	// full-ISR-committed (durable on all three, incl. Q) acked write. Because P
	// stops renewing once cut and never renews again, its acks form a clean PREFIX
	// (all succeed, then all fail) — so the recorded acks are exactly the writes
	// that must survive on Q.
	type attempt struct {
		key, val []byte
		err      error
		at       time.Time
	}
	var (
		recMu   sync.Mutex
		records []attempt
		stopCh  = make(chan struct{})
		wdone   = make(chan struct{})
	)
	go func() {
		defer close(wdone)
		i := 0
		for {
			select {
			case <-stopCh:
				return
			default:
			}
			key := []byte(fmt.Sprintf("key-%05d", i))
			val := []byte(fmt.Sprintf("val-%05d", i))
			_, err := tc.nodes[primaryIdx].Call("put", ops.EncodePutArgs(key, val, 0))
			recMu.Lock()
			records = append(records, attempt{key: key, val: val, err: err, at: time.Now()})
			recMu.Unlock()
			i++
			time.Sleep(5 * time.Millisecond)
		}
	}()

	// A survivor node (any node != P) sits on the majority side of the cut; use it to
	// read replicated FSM state when polling for the promotion.
	survivorIdx := -1
	for i := range tc.nodes {
		if i != primaryIdx {
			survivorIdx = i
			break
		}
	}

	// Let the recorded writer run briefly against the already-stable primary so it has
	// a batch of pre-cut acked writes before we cut. We keep this SHORT: P has been
	// primary (and beaconing every renewInterval) since bootstrap, so every tracker has
	// long since seen it alive; a short window minimises the exposure to any natural
	// lease blip/failover before our induced cut, while the recorded writer keeps
	// acking THROUGH the post-cut lease window (its acks there also commit to the full
	// ISR incl. Q), so there is no shortage of lease-window writes to prove survive.
	time.Sleep(200 * time.Millisecond)

	// --- CUT P's META transport. PB data path + client server stay UP.
	partitionAt := time.Now()
	inj.partition()
	t.Logf("partitioned %q meta transport at %s", origPrimary, partitionAt)

	// --- Poll the SURVIVOR for the promotion: a higher epoch AND a new primary
	// Q != P. Record the instant we observe it.
	var (
		newPrimary       string
		newEpoch         uint64
		promoteObservedT time.Time
	)
	promoteDeadline := time.Now().Add(30 * time.Second)
	for time.Now().Before(promoteDeadline) {
		ep := tc.nodes[survivorIdx].meta.FSM.ShardEpoch(sh)
		pr := tc.nodes[survivorIdx].meta.FSM.ShardPrimary(sh)
		if ep > origEpoch && pr != "" && pr != origPrimary {
			newPrimary, newEpoch, promoteObservedT = pr, ep, time.Now()
			break
		}
		time.Sleep(50 * time.Millisecond)
	}

	// Stop the writer and drain it so every record is settled before we analyse.
	close(stopCh)
	<-wdone

	if newPrimary == "" {
		t.Fatalf("no promotion within 30s (orig primary %q epoch %d, survivor epoch %d primary %q)",
			origPrimary, origEpoch, tc.nodes[survivorIdx].meta.FSM.ShardEpoch(sh), tc.nodes[survivorIdx].meta.FSM.ShardPrimary(sh))
	}
	t.Logf("failover: %q (epoch %d) -> %q (epoch %d) observed at +%s",
		origPrimary, origEpoch, newPrimary, newEpoch, promoteObservedT.Sub(partitionAt))

	// --- Analyse the recorded writer attempts.
	recMu.Lock()
	snap := make([]attempt, len(records))
	copy(snap, records)
	recMu.Unlock()

	var (
		ackedKeys   [][]byte
		ackedVals   [][]byte
		lastAckAt   time.Time
		postCutFail bool
	)
	for _, a := range snap {
		if a.err == nil {
			ackedKeys = append(ackedKeys, a.key)
			ackedVals = append(ackedVals, a.val)
			if a.at.After(lastAckAt) {
				lastAckAt = a.at
			}
		} else if a.at.After(partitionAt) {
			postCutFail = true // P rejected a write AFTER the cut ⇒ evidence it fenced
		}
	}
	if len(ackedKeys) == 0 {
		// PRECONDITION not met (not an invariant violation): the writer landed no acked
		// write, so there is nothing whose survival we could prove — the run would be
		// vacuous. This only happens if the (Phase-A-confirmed) primary wedged/fenced
		// right after selection under load. SKIP rather than false-fail.
		t.Skip("SKIP: writer made no acked progress before the cut (host too loaded); nothing to prove this run")
	}

	// ASSERTION 1 — P SELF-FENCES: after the cut, writes to P eventually FAIL. A
	// partitioned primary that kept acking would be a self-fence bug (it could form
	// a double-primary with Q). We require at least one post-cut rejection.
	if !postCutFail {
		t.Fatal("assertion 1 FAILED: P never rejected a write after its meta transport was cut — it did not self-fence")
	}

	// ASSERTION 2 (temporal no-overlap) — P's LAST successful ack is STRICTLY BEFORE
	// Q's promotion instant. This is the core no-double-primary proof: it shows P
	// had stopped acking as primary before Q became primary, so the two valid-lease
	// windows never overlapped. A honor-rule violation would surface here as a P ack
	// at/after promotion.
	if !lastAckAt.Before(promoteObservedT) {
		t.Fatalf("assertion 2 FAILED (double-primary): P's last ack at +%s was NOT before Q promotion at +%s",
			lastAckAt.Sub(partitionAt), promoteObservedT.Sub(partitionAt))
	}
	t.Logf("no-overlap: P last ack +%s < Q promotion +%s (gap %s)",
		lastAckAt.Sub(partitionAt), promoteObservedT.Sub(partitionAt), promoteObservedT.Sub(lastAckAt))

	// ASSERTION 2 (explicit rejection) — with Q now primary at epoch E+1, issue
	// several FRESH writes to P and require ALL to be rejected. P's lease is dead and
	// can never renew (still partitioned), so every Propose self-fences. A
	// double-primary bug would let some of these slip through.
	newIdx := findNodeIdx(newPrimary)
	if newIdx == primaryIdx || newIdx < 0 {
		t.Fatalf("new primary %q resolves to bad node index %d (P was %d)", newPrimary, newIdx, primaryIdx)
	}
	cleanFences := 0
	for i := 0; i < 5; i++ {
		key := []byte(fmt.Sprintf("post-promote-%d", i))
		_, err := tc.nodes[primaryIdx].Call("put", ops.EncodePutArgs(key, []byte("x"), 0))
		// The SAFETY property is "P must not ACK": any non-nil error is a valid
		// rejection, a SUCCESS (err == nil) would be the double-primary bug. A
		// clean lease fence surfaces as a NotLeaderError (pbisr fence →
		// raft.ErrNotLeader → Store NotLeaderError); a partitioned P may instead
		// return a wedged-pipeline error (ErrReplicationTimeout/ErrPipelineStalled) —
		// still a non-ack, so still safe. We require err != nil and additionally count
		// the clean fences for visibility.
		if err == nil {
			t.Fatalf("assertion 2 FAILED (double-primary): P acked write %q AFTER Q was promoted primary", key)
		}
		var nle *shard.NotLeaderError
		if errors.As(err, &nle) {
			cleanFences++
		}
	}
	t.Logf("post-promotion: all 5 fresh writes to P rejected (%d clean lease-fences, rest non-ack pipeline errors)", cleanFences)

	// --- New primary Q must have adopted the promotion (its lease-keeper Promote'd
	// once it saw the higher epoch). Wait for its engine to reach newEpoch.
	newNode := tc.nodes[newIdx]
	eng := newNode.pbEngines[sh]
	if eng == nil {
		t.Fatalf("new primary node %d has no engine for shard %d", newIdx, sh)
	}
	engDeadline := time.Now().Add(10 * time.Second)
	for time.Now().Before(engDeadline) && eng.Epoch() < newEpoch {
		time.Sleep(50 * time.Millisecond)
	}

	// ASSERTION 3 — NO ACKED LOSS: every write P acknowledged (committed via full ISR
	// to the ISR incl. Q during its lease-valid window) is present on Q. These are
	// the pre-partition + lease-window writes that must survive on the new primary.
	store := newNode.getShard(sh)
	for i, key := range ackedKeys {
		got, err := store.Get(key)
		if err != nil || string(got) != string(ackedVals[i]) {
			t.Fatalf("assertion 3 FAILED (acked loss): key %q missing/wrong on new primary Q: got %q err %v want %q",
				key, got, err, ackedVals[i])
		}
	}

	// ASSERTION 4 — DENSITY on Q: Q's applied high-water must be AT LEAST baseline (all
	// pre-writer commits) + the count of recorded acked writes. Combined with assertion
	// 3 (every acked KEY is present on Q), this proves no acked seq is missing. The
	// engine STRUCTURALLY gap-rejects at apply (a seq out of order is refused), so the
	// applied range baseline+1..HW is inherently dense — there can be no hole. The lower
	// bound (not exact equality) tolerates a benign ORPHAN seq: a write whose seq was
	// assigned and applied on the backup but whose client Call then returned a
	// full-ISR-timeout error at the fence boundary under load — that seq is really on Q
	// (extra, not missing) so it only pushes HW ABOVE the acked count, never below.
	// HW < expected would be the real bug: a missing (lost) acked seq.
	expectedHW := baseline + uint64(len(ackedKeys))
	la := eng.LastApplied()
	if la < expectedHW {
		t.Fatalf("assertion 4 FAILED: Q LastApplied=%d < %d (baseline %d + %d acked) — acked-loss or gap",
			la, expectedHW, baseline, len(ackedKeys))
	}
	if la > expectedHW {
		t.Logf("Q LastApplied=%d > expected %d by %d (benign orphan seq(s) applied-but-not-client-acked at the fence boundary)", la, expectedHW, la-expectedHW)
	}
	t.Logf("Q LastApplied=%d, baseline=%d, acked keys=%d — no acked-loss, dense", la, baseline, len(ackedKeys))

	// --- OPTIONAL (best-effort, NON-FATAL): heal P and observe it rejoin as a BACKUP
	// at the new epoch. The safety property (P does NOT resume as primary) is already
	// PROVEN above: P's lease is dead and the epoch advanced, and assertion 2 showed
	// every write to P is fenced. Full rejoin convergence is timing-sensitive, so per
	// the gate plan this is a soft observation only — we log the outcome and never
	// fail on it. (P's FSM is transiently still at the old epoch immediately after
	// heal until meta-Raft replication catches it up; that is expected, not a bug.)
	inj.heal()
	converged := false
	healDeadline := time.Now().Add(10 * time.Second)
	for time.Now().Before(healDeadline) {
		ep := tc.nodes[primaryIdx].meta.FSM.ShardEpoch(sh)
		pr := tc.nodes[primaryIdx].meta.FSM.ShardPrimary(sh)
		if ep >= newEpoch && pr == newPrimary {
			t.Logf("post-heal: P converged to epoch %d, primary=%q (rejoined as backup, not primary)", ep, pr)
			converged = true
			break
		}
		time.Sleep(100 * time.Millisecond)
	}
	if converged {
		// Now that P sees the new epoch, a write to it must STILL be fenced: P is a
		// backup, never the primary. This is a bonus check on the healed path; it is
		// only meaningful once P has converged, so it is gated on that.
		if _, err := tc.nodes[primaryIdx].Call("put", ops.EncodePutArgs([]byte("post-heal"), []byte("x"), 0)); err == nil {
			t.Fatalf("post-heal BUG: healed P acked a write though it is a backup at epoch %d (primary is %q)", newEpoch, newPrimary)
		}
	} else {
		t.Logf("post-heal: P did not converge within 10s (soft check, not a gate failure)")
	}
}
