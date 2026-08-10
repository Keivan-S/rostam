// SPDX-License-Identifier: Apache-2.0

package cluster

import (
	"context"
	"crypto/sha256"
	"errors"
	"fmt"
	"log/slog"
	"sort"
	"strings"
	"sync"
	"time"

	"github.com/rostamlabs/rostam/ops"
)

// WASM BLOB FETCH — how a node that is BLOCKED obtains the bytes it is blocked
// on, and how that block is made visible.
//
// The transport (__wasm_blob_put__ / __wasm_blob_get__) and the push move bytes;
// a thin marker turns "this node does not hold the
// bytes" from a curiosity into a REACHABLE APPLY STATE: a committed entry can
// name a module version this node has never fetched. This file is the two halves
// of the answer to that:
//
//	the FETCHER — one background loop per FINGERPRINT, deduplicated, retrying
//	forever, that turns a missing blob into a present one;
//	the BLOCK REGISTRY — the lock-free published record of every (group, op)
//	pair currently parked, how long for, how many attempts, and why.
//
// ############ THE SECOND HALF IS NOT OPTIONAL, AND IT IS NOT DECORATION ######
//
// The design trades a HALT for a client-visible retryable error plus a SILENT
// WAIT. A halt announces itself: the process is gone, every dashboard notices. A
// wait announces nothing at all — the node is up, healthy, serving every other
// group, and one group has simply stopped. Worse, the wait has a second-order
// cost with no natural alarm: hashicorp/raft runs Snapshot on the same goroutine
// as Apply, so a blocked group CANNOT SNAPSHOT and therefore cannot compact its
// Raft log. Disk grows for as long as the block lasts. That is why the gauge that
// matters is the DURATION of the longest current block and not the count of them:
// a hundred blocks that clear in 20ms are the system working, and one block that
// has lasted an hour is a disk-consumption incident.

// wasmBlobFetchInitialBackoff / wasmBlobFetchMaxBackoff bound the retry cadence
// of one fingerprint's fetch loop.
//
// The first sweep over the sources happens IMMEDIATELY (the loop tries before it
// ever sleeps), because the overwhelmingly common case is a member that was
// merely restarting during the push and is reachable now. Everything after that
// backs off, because the second most common case is a member that will be
// unreachable for a long time, and a tight loop against it would be one failed
// dial per source per iteration, forever, on a node that is otherwise idle.
const (
	wasmBlobFetchInitialBackoff = 250 * time.Millisecond
	wasmBlobFetchMaxBackoff     = 15 * time.Second
)

// wasmBlobGetTimeout bounds ONE peer's leg of a fetch, for the reason
// wasmBlobPushTimeout and wasmBroadcastGroupTimeout exist: the sweep over
// sources is sequential, so one unreachable-but-not-erroring peer would
// otherwise stall the whole fetch — and behind it, a shard group's apply loop —
// indefinitely.
const wasmBlobGetTimeout = 10 * time.Second

// wasmBlockWarnAfter / wasmBlockErrorEvery are the escalating log schedule for a
// blocked apply.
//
// A block under wasmBlockWarnAfter is not logged at all. That is deliberate and
// it is the difference between a useful signal and a discarded one: the normal,
// expected block is the few-millisecond race between a marker applying and the
// push that carries its bytes finishing, and it happens once per registration per
// node. Logging those would put the interesting case — a block that is not going
// to clear — in the middle of routine noise.
const (
	wasmBlockWarnAfter  = 5 * time.Second
	wasmBlockErrorEvery = 30 * time.Second
)

// wasmBlobFetch is one in-flight fetch. It exists to be the DEDUP KEY's value:
// its presence in Node.wasmFetching is what makes a second request for the same
// fingerprint a no-op.
type wasmBlobFetch struct {
	started time.Time

	mu       sync.Mutex
	attempts int
	lastErr  error
	// tried is every server address this fetch has asked, in first-asked order.
	// It is what the escalating ERROR log names, because "the blob has not
	// arrived" is not actionable and "asked A, B and C; all refused" is.
	tried []string
}

// blockKey identifies one parked (group, op) pair. It is NOT keyed on the
// fingerprint: a group blocks on ONE entry at a time (raft applies a log in
// order on one goroutine), so (group, op) already identifies the block, and
// keying on the fingerprint too would leave a stale record behind every time a
// group's binding moved forward while it was parked.
type blockKey struct {
	group int
	op    string
}

// blockRecord is one live block. Only the FSM apply goroutine of `group` writes
// it, and it writes under Node.wasmBlockMu, so the published snapshot is built
// from a consistent read.
type blockRecord struct {
	fingerprint string
	since       time.Time
	attempts    int
	lastErr     string
	warned      bool
	lastErrLog  time.Time
}

// WASMBlockedApply is one entry of WASMBlockStats.Blocked: a committed entry a
// shard group cannot apply yet because the module version it names has not
// arrived on this node.
//
// EVERY FIELD IS THERE TO ANSWER A DIFFERENT OPERATOR QUESTION, and none of them
// is redundant:
//
//   - Group and Op say WHAT IS STOPPED. A blocked group serves no writes for any
//     key that routes to it, and cannot snapshot or compact;
//   - Fingerprint says WHAT TO DO ABOUT IT. It is the exact argument of
//     __wasm_blob_put__, which is the escape hatch (see WASMBlockStats);
//   - Since is the one to alert on. See WASMBlockStats.LongestBlock;
//   - Attempts distinguishes "waiting on a backoff" from "trying hard and
//     failing";
//   - LastErr is the fetch's most recent refusal, which is where a module that
//     does not compile on THIS node shows up — the one block shape that will
//     never clear on its own.
type WASMBlockedApply struct {
	Group       int
	Op          string
	Fingerprint string
	Since       time.Duration
	Attempts    int
	LastErr     string
}

// WASMBlockStats makes the classRetry block observable. See the file header for
// why that is load-bearing rather than a nice-to-have.
//
// ############ THE OPERATOR ESCAPE HATCH IS __wasm_blob_put__ ############
//
// A block named here clears the instant this node holds the bytes, and an
// operator can supply them BY HAND with no restart, no failover and no data
// movement:
//
//	rostam call __wasm_blob_put__ <fingerprint-hex><module-bytes>
//
// against the blocked node, with the module the fingerprint names. The put
// verifies the hash and compiles the module before it acks, so a wrong file is
// refused rather than accepted; on success the blocked group's very next re-run
// applies. Fingerprint above is exactly the hex to use.
//
// That is a strictly better runbook entry than anything in wasmRecoveryAdvice,
// and it is worth saying why rather than leaving the comparison implicit:
// wasmRecoveryAdvice's remedy is "wipe this node's data dir and rejoin", which is
// expensive, carries two caveats (config modules are not recovered; it is
// unqualified data loss if this node is the last healthy replica of any group it
// hosts), and takes as long as a full catch-up. This one is a single admin call
// that moves one file.
type WASMBlockStats struct {
	// Blocked is every (group, op) pair currently parked, sorted by group then
	// op so successive Stats() calls are diffable. Freshly allocated per call.
	Blocked []WASMBlockedApply

	// Total counts blocks ENTERED since process start, across all groups and ops.
	// A rising Total with an empty Blocked is the healthy shape: registrations are
	// racing their pushes and winning. A flat Total with a non-empty Blocked is
	// one block that is not clearing.
	Total uint64

	// LongestBlock is how long the OLDEST currently-blocked entry has been
	// parked, or zero when nothing is blocked.
	//
	// ALERT ON THIS, NOT ON len(Blocked). A block is not merely a stalled group:
	// hashicorp/raft calls Snapshot on the same goroutine as Apply, so a blocked
	// group cannot snapshot and therefore cannot compact its Raft log, and cannot
	// accept an InstallSnapshot either. The disk cost is a function of DURATION ×
	// that group's write rate and is unbounded, while the count says nothing about
	// it — sixty-four groups blocked for 20ms cost nothing, one group blocked for
	// an hour is an incident.
	LongestBlock time.Duration
}

// prefetchWASMBlob asks for a fingerprint's bytes WITHOUT WAITING for them, and
// deduplicates: N groups blocked on one fingerprint issue ONE fetch.
//
// ############ IT MUST NEVER BLOCK. EVERY CALLER HOLDS AN APPLY LOCK. ##########
//
// It is called from applyWASMRegistration and from the disk-reload path, both of
// which run under wasmApplyMu, and from the FSM's classRetry hook, which runs on
// a shard's apply goroutine. wasmApplyMu is also taken by snapshotWASMState and
// restoreWASMState, so a version of this that waited for the bytes would stall
// EVERY group's snapshot on this node behind one group's missing module — turning
// a group-local condition into a node-global one, which is the exact failure this
// whole stage is built to avoid.
//
// THE DEDUP IS PER FINGERPRINT, NOT PER (GROUP, OP), and that is the only key
// that makes sense: the thing being fetched is a file addressed by its content,
// and a registration broadcast to 64 groups produces 64 markers naming ONE blob.
// Keying on the blocked pair instead would issue 64 identical fetches of the same
// file against the same peers.
func (n *Node) prefetchWASMBlob(fp [sha256.Size]byte) {
	n.wasmFetchMu.Lock()
	if n.wasmFetching == nil {
		n.wasmFetching = make(map[[sha256.Size]byte]*wasmBlobFetch, 2)
	}
	if _, inflight := n.wasmFetching[fp]; inflight {
		n.wasmFetchMu.Unlock()
		return
	}
	f := &wasmBlobFetch{started: time.Now()}
	n.wasmFetching[fp] = f
	n.wasmFetchMu.Unlock()
	n.wasmFetchStarts.Add(1)
	go n.runWASMBlobFetch(fp, f)
}

// runWASMBlobFetch retries FOREVER — until the bytes land or the node closes.
//
// There is no attempt cap and no give-up, for the same reason the apply-side
// block is unbounded: giving up would leave the group parked with nothing left
// trying to unpark it, which is strictly worse than continuing to ask. The cost
// of asking is bounded by the backoff (one sweep every 15s at the cap), and the
// loop exits the moment the fetch succeeds or the node shuts down.
func (n *Node) runWASMBlobFetch(fp [sha256.Size]byte, f *wasmBlobFetch) {
	defer func() {
		n.wasmFetchMu.Lock()
		delete(n.wasmFetching, fp)
		n.wasmFetchMu.Unlock()
	}()
	backoff := wasmBlobFetchInitialBackoff
	for {
		if err := n.fetchWASMBlobOnce(fp, f); err == nil {
			n.wasmFetchOKs.Add(1)
			slog.Info("fetched wasm module bytes; any group blocked on this version unblocks on its next apply retry",
				"component", "cluster", "blob", ops.WASMBlobHex(fp), "attempts", f.attemptCount(), "took", time.Since(f.started))
			return
		}
		select {
		case <-n.wasmFetchStop:
			return
		case <-time.After(backoff):
		}
		if backoff *= 2; backoff > wasmBlobFetchMaxBackoff {
			backoff = wasmBlobFetchMaxBackoff
		}
	}
}

// fetchWASMBlobOnce sweeps the sources once, in order, and stops at the first
// one that serves bytes this node accepts.
//
// SOURCE ORDER, and the one place the recorded design had to be adapted rather
// than followed. It says: (a) the group's own Raft peers, (b) any cluster member.
// The FETCH IS DEDUPLICATED PER FINGERPRINT, though, and a fingerprint is not
// attached to a group — one registration broadcast to 64 groups names one blob,
// and several different groups may be blocked on it at once. "The group's own
// peers" is therefore not a well-defined set for this loop. What IS well defined,
// and preserves the intent exactly, is the union of the owners of every group
// THIS NODE HOSTS: those are the nodes that replicate the logs this node applies,
// so they are both the most likely holders and the ones a block actually depends
// on. They are tried first, then the wider member table.
//
//	(a) owners of the shard groups this node hosts (meta placement), which is
//	    the cluster-native form of "my groups' Raft peers" — and it is resolved
//	    through the same serverAddrFor/raftToServerAddr mapping every other
//	    peer-directed call uses, so a Raft transport address is never dialled as
//	    if it were a client port;
//	(b) every cluster member from meta Members, plus the static cfg.Peers, which
//	    is the same union wasmBlobPushTargets pushes to. The DURABILITY FLOOR is
//	    stated over exactly this set: a majority of it held the bytes when the
//	    marker was proposed, so any majority of it that is reachable now contains
//	    a holder.
//
// A source that does not have the blob answers with an ordinary error and the
// sweep moves on — "ask someone else" is the correct answer, not a failure.
//
// ############ SOURCE ZERO IS THIS NODE'S OWN DISK ############
//
// It looks redundant — a fetch runs because the bytes are missing — and it is
// not, because what makes a group block is RUNTIME residency, not disk. The
// resolver asks the runtime (resolveModuleForInvoke runs on an apply goroutine
// and must never do I/O), so a blob that reached disk but was never compiled
// into the runtime blocks exactly like an absent one. That state is reachable by
// design: applyWASMRegistration treats a materializeWASMBlob failure as a
// residency condition and continues, and a reload can bind a group to a version
// it did not instantiate. Without this leg, the only ways out are a peer serving
// bytes this node already has, or an operator — so a node whose data dir holds
// the exact file it is waiting for could stay blocked permanently while every
// peer that ever held it churned away.
//
// It goes through the SAME acceptance rule as a network source, deliberately.
// readWASMBlob hash-verifies against the filename, and storeWASMBlobVerified
// re-checks the hash and COMPILES before accepting — which is the part that
// matters here: a local file that does not compile on this node must be reported
// as the block reason and fall through to the peers rather than be declared a
// success that ends the fetch loop. The rewrite it performs is a no-op on
// identical content (writeWASMBlob is content-addressed and idempotent).
func (n *Node) fetchWASMBlobOnce(fp [sha256.Size]byte, f *wasmBlobFetch) error {
	hexFP := ops.WASMBlobHex(fp)
	var localErr error
	if b, err := readWASMBlob(n.cfg.DataDir, hexFP); err != nil {
		localErr = fmt.Errorf("local store: %w", err)
	} else if err := storeWASMBlobVerified(n.cfg.DataDir, hexFP, b); err != nil {
		localErr = fmt.Errorf("local store: %w", err)
	} else {
		n.installArrivedWASMBlob(fp)
		return nil
	}
	// noteErr but NOT noteAttempt: `tried` is the peer list the escalating ERROR
	// log names, and reading our own disk is not a peer that refused us.
	f.noteErr(localErr)

	payload := []byte(hexFP)
	var lastErr error
	for _, addr := range n.wasmBlobFetchSources() {
		f.noteAttempt(addr)
		b, err := n.getWASMBlobFromPeer(addr, payload)
		if err != nil {
			lastErr = fmt.Errorf("%s: %w", addr, err)
			f.noteErr(lastErr)
			continue
		}
		// The SAME acceptance rule every other route into the blob store uses:
		// the bytes must hash to the fingerprint we asked for AND compile here.
		// Reusing it is what keeps a fetched blob from being the one blob in the
		// store nobody verified — and the compile check is what turns "this module
		// does not work on this node" into a named, reported block reason instead
		// of a mysterious one.
		if err := storeWASMBlobVerified(n.cfg.DataDir, hexFP, b); err != nil {
			lastErr = fmt.Errorf("%s: %w", addr, err)
			f.noteErr(lastErr)
			continue
		}
		n.installArrivedWASMBlob(fp)
		return nil
	}
	if lastErr == nil {
		// No peer was even asked. The local leg's refusal is then the ONLY thing
		// that happened this sweep, and it is the more informative half — carry it
		// rather than replace it with a bare "nobody answered".
		lastErr = fmt.Errorf("%w; and no peer or cluster member could be reached", localErr)
		f.noteErr(lastErr)
	}
	return lastErr
}

// wasmBlobFetchSources builds the ordered, deduplicated source list described on
// fetchWASMBlobOnce. Self is excluded: this node is fetching precisely because it
// does not have the blob.
func (n *Node) wasmBlobFetchSources() []string {
	out := make([]string, 0, 8)
	seen := make(map[string]struct{}, 8)
	add := func(addr string) {
		if addr == "" {
			return
		}
		if _, dup := seen[addr]; dup {
			return
		}
		seen[addr] = struct{}{}
		out = append(out, addr)
	}
	// (a) owners of the groups this node hosts.
	n.shardMu.RLock()
	hosted := make([]int, 0, len(n.shards))
	for i, s := range n.shards {
		if s != nil {
			hosted = append(hosted, i)
		}
	}
	n.shardMu.RUnlock()
	for _, idx := range hosted {
		for _, ownerID := range n.ownersFor(idx) {
			if ownerID == n.cfg.NodeID {
				continue
			}
			add(n.serverAddrFor(ownerID))
		}
	}
	// (b) the wider member table — the set the durability floor is stated over.
	for _, m := range n.wasmBlobPushTargets() {
		add(m.serverAddr)
	}
	return out
}

// getWASMBlobFromPeer performs one __wasm_blob_get__ against one server address,
// bounded by wasmBlobGetTimeout.
//
// It addresses the member DIRECTLY rather than going through forwardTimeout, for
// the reason putWASMBlobOnPeer does: forwardTimeout rotates over a SHARD's
// owners, and a blob is not about a shard — every member is a candidate holder,
// including one that owns no group.
func (n *Node) getWASMBlobFromPeer(addr string, payload []byte) ([]byte, error) {
	cl, err := n.peerClient(addr)
	if err != nil {
		return nil, err
	}
	ctx, cancel := context.WithTimeout(context.Background(), wasmBlobGetTimeout)
	defer cancel()
	return cl.Call(ctx, opWASMBlobGetName, payload)
}

func (f *wasmBlobFetch) noteAttempt(addr string) {
	f.mu.Lock()
	defer f.mu.Unlock()
	f.attempts++
	for _, a := range f.tried {
		if a == addr {
			return
		}
	}
	f.tried = append(f.tried, addr)
}

func (f *wasmBlobFetch) noteErr(err error) {
	f.mu.Lock()
	defer f.mu.Unlock()
	f.lastErr = err
}

func (f *wasmBlobFetch) attemptCount() int {
	f.mu.Lock()
	defer f.mu.Unlock()
	return f.attempts
}

// report renders the fetch's current state for a block's log line and stats.
func (f *wasmBlobFetch) report() (attempts int, lastErr string, tried []string) {
	f.mu.Lock()
	defer f.mu.Unlock()
	if f.lastErr != nil {
		lastErr = f.lastErr.Error()
	}
	return f.attempts, lastErr, append([]string(nil), f.tried...)
}

// onShardApplyRetry is shard.Config.OnApplyRetry: the FSM has decided a committed
// entry cannot be applied yet, and is about to wait.
//
// ############ IT RESOLVES UNDER THE LOCK, RELEASES, THEN RETURNS #############
//
// It runs ON a shard's FSM apply goroutine, so it must not block and must not
// still hold wasmApplyMu when it returns. The sequence is exactly the one the
// design prescribes: take wasmApplyMu only long enough to turn the (op, group) in
// the error into the fingerprint the binding names, release it, and then kick a
// fire-and-forget fetch. Nothing in this function waits for a network round trip,
// a lock an apply holds, or the bytes themselves.
//
// WHAT WOULD REINTRODUCE THE HAZARD, since each of these reads as an
// improvement: fetching inline "because we are going to wait anyway"; holding
// wasmApplyMu across the prefetch "to keep the lookup and the fetch consistent";
// or resolving the fingerprint by asking the runtime rather than wasmState, which
// would take the Runtime's own RWMutex — the lock AddModule holds from inside an
// apply.
func (n *Node) onShardApplyRetry(err error, attempt int, blockedFor time.Duration) {
	var nre *ops.WASMNotResidentError
	if !errors.As(err, &nre) {
		// Some other classRetry sentinel: there is nothing to fetch, but the block
		// is still real and must still be visible.
		n.recordWASMBlock(blockKey{group: ops.NoShardIndex, op: "?"}, "", attempt, blockedFor, err.Error(), nil)
		return
	}
	fp, known := n.lookupWASMBlobFor(nre.Op, nre.Group)
	if !known {
		// The op has no install on this node at all. That should not happen — the
		// resolver found a binding to produce this error — but reporting a block
		// with no fingerprint is far better than dropping it, because a block with
		// nothing fetching it is precisely the one an operator has to see.
		n.recordWASMBlock(blockKey{group: nre.Group, op: nre.Op}, "", attempt, blockedFor, err.Error(), nil)
		return
	}
	// Fire-and-forget, deduplicated per fingerprint. Cheap enough to call on every
	// retry: after the first, it is one map lookup under a mutex nothing else
	// contends for.
	n.prefetchWASMBlob(fp)
	n.wasmFetchMu.Lock()
	f := n.wasmFetching[fp]
	n.wasmFetchMu.Unlock()
	var (
		fetchErr string
		tried    []string
	)
	if f != nil {
		_, fetchErr, tried = f.report()
	}
	if fetchErr == "" {
		fetchErr = err.Error()
	}
	n.recordWASMBlock(blockKey{group: nre.Group, op: nre.Op}, ops.WASMBlobHex(fp), attempt, blockedFor, fetchErr, tried)
}

// onShardApplyRetryCleared is shard.Config.OnApplyRetryCleared: the entry finally
// applied. It removes the record and republishes.
func (n *Node) onShardApplyRetryCleared(err error, attempts int, blockedFor time.Duration) {
	key := blockKey{group: ops.NoShardIndex, op: "?"}
	var nre *ops.WASMNotResidentError
	if errors.As(err, &nre) {
		key = blockKey{group: nre.Group, op: nre.Op}
	}
	n.wasmBlockMu.Lock()
	rec, had := n.wasmBlockLive[key]
	delete(n.wasmBlockLive, key)
	n.publishWASMBlocksLocked()
	n.wasmBlockMu.Unlock()
	if had && rec.warned {
		// Only the blocks that were loud enough to WARN get a matching "cleared"
		// line. A block that never crossed wasmBlockWarnAfter was never reported,
		// and announcing the end of something nobody was told about is noise.
		slog.Warn("wasm apply block cleared",
			"component", "cluster", "shard", key.group, "op", key.op,
			"blob", rec.fingerprint, "blocked_for", blockedFor, "attempts", attempts)
	}
}

// lookupWASMBlobFor resolves (op, group) to the blob fingerprint the binding
// names. group == ops.NoShardIndex asks for the node-wide install, which is what
// the resolver's three node-wide fallbacks use.
//
// It takes wasmApplyMu and RELEASES IT BEFORE RETURNING — see onShardApplyRetry.
func (n *Node) lookupWASMBlobFor(op string, group int) ([sha256.Size]byte, bool) {
	n.wasmApplyMu.Lock()
	defer n.wasmApplyMu.Unlock()
	in, have := n.wasmState.installed[op]
	if !have {
		return ops.ZeroWASMBlob, false
	}
	if group != ops.NoShardIndex {
		if b, bound := in.groups[group]; bound {
			return b.reg.Blob, true
		}
	}
	if in.reg.Blob == ops.ZeroWASMBlob {
		return ops.ZeroWASMBlob, false
	}
	return in.reg.Blob, true
}

// recordWASMBlock updates (or creates) the live record for one blocked pair,
// republishes the lock-free snapshot, and drives the escalating log schedule.
//
// THE LOG SCHEDULE IS DELIBERATELY SPARSE AT THE START AND LOUD LATER: nothing
// under wasmBlockWarnAfter, one WARN at that threshold, then an ERROR every
// wasmBlockErrorEvery naming the group, the op, the fingerprint and EVERY peer
// the fetch has tried. The peer list is the actionable part — "the blob has not
// arrived" tells an operator nothing they cannot see in Stats, while "asked A, B
// and C; C said the module does not compile on this node" names the fix.
func (n *Node) recordWASMBlock(key blockKey, fingerprint string, attempt int, blockedFor time.Duration, lastErr string, tried []string) {
	now := time.Now()
	n.wasmBlockMu.Lock()
	if n.wasmBlockLive == nil {
		n.wasmBlockLive = make(map[blockKey]*blockRecord, 2)
	}
	rec, had := n.wasmBlockLive[key]
	if !had {
		rec = &blockRecord{since: now.Add(-blockedFor)}
		n.wasmBlockLive[key] = rec
		n.wasmBlockTotal.Add(1)
	}
	rec.fingerprint = fingerprint
	rec.attempts = attempt
	rec.lastErr = lastErr
	logWarn := false
	logError := false
	if blockedFor >= wasmBlockWarnAfter {
		if !rec.warned {
			rec.warned = true
			rec.lastErrLog = now
			logWarn = true
		} else if now.Sub(rec.lastErrLog) >= wasmBlockErrorEvery {
			rec.lastErrLog = now
			logError = true
		}
	}
	n.publishWASMBlocksLocked()
	n.wasmBlockMu.Unlock()

	// Logged OUTSIDE the mutex: slog handlers are caller-supplied and may do
	// anything, including I/O, and this runs on a shard's apply goroutine.
	switch {
	case logWarn:
		slog.Warn("shard group is BLOCKED waiting for wasm module bytes; it applies nothing and cannot snapshot or compact its log until they arrive (push them by hand with __wasm_blob_put__ to unblock immediately)",
			"component", "cluster", "shard", key.group, "op", key.op, "blob", fingerprint,
			"blocked_for", blockedFor, "attempts", attempt, "last_err", lastErr)
	case logError:
		slog.Error("shard group STILL BLOCKED waiting for wasm module bytes; its Raft log is growing uncompacted (push them by hand with __wasm_blob_put__ to unblock immediately)",
			"component", "cluster", "shard", key.group, "op", key.op, "blob", fingerprint,
			"blocked_for", blockedFor, "attempts", attempt, "last_err", lastErr,
			"peers_tried", strings.Join(tried, ", "))
	}
}

// publishWASMBlocksLocked rebuilds and stores the immutable blocked-set snapshot.
// Callers hold wasmBlockMu.
//
// It is published by copy-on-write into an atomic.Pointer for the same reason the
// route-gate snapshot is: Stats() must be answerable while a group is blocked,
// and the whole point of the exercise is that it is answerable WITHOUT touching
// anything an apply holds. A Stats() that took wasmBlockMu would still be fine
// today (nothing holds it across a wait), but it would be one edit away from not
// being — and "can I still see the block" is the question that must never depend
// on the block.
func (n *Node) publishWASMBlocksLocked() {
	out := make([]WASMBlockedApply, 0, len(n.wasmBlockLive))
	for k, rec := range n.wasmBlockLive {
		out = append(out, WASMBlockedApply{
			Group:       k.group,
			Op:          k.op,
			Fingerprint: rec.fingerprint,
			Since:       time.Since(rec.since),
			Attempts:    rec.attempts,
			LastErr:     rec.lastErr,
		})
	}
	sort.Slice(out, func(i, j int) bool {
		if out[i].Group != out[j].Group {
			return out[i].Group < out[j].Group
		}
		return out[i].Op < out[j].Op
	})
	n.wasmBlocks.Store(&out)
}

// wasmBlockStats renders the block registry for Node.Stats.
//
// It reads the PUBLISHED snapshot with one atomic load and takes NO LOCK, which
// is what makes "Node.Stats still answers while a group is blocked" a structural
// property rather than a timing accident. Since is recomputed from the record's
// start instant at publish time, so an entry that has been parked for an hour
// reports an hour even if nothing republished in between — the republish happens
// on every retry, which is at worst once a second.
func (n *Node) wasmBlockStats() WASMBlockStats {
	out := WASMBlockStats{Total: n.wasmBlockTotal.Load()}
	snap := n.wasmBlocks.Load()
	if snap == nil || len(*snap) == 0 {
		return out
	}
	out.Blocked = append([]WASMBlockedApply(nil), (*snap)...)
	for _, b := range out.Blocked {
		if b.Since > out.LongestBlock {
			out.LongestBlock = b.Since
		}
	}
	return out
}

// installArrivedWASMBlob is what actually UNBLOCKS a parked group: it
// instantiates a newly-arrived blob into the runtime for every op version that
// names it.
//
// ############ WRITING THE FILE IS NOT ENOUGH, AND THAT IS EASY TO MISS ########
//
// resolveModuleForInvoke asks the RUNTIME whether the version is resident, not
// the filesystem — it runs on an apply goroutine and must never do I/O. So a blob
// that reaches disk but is never compiled leaves the group parked forever on a
// module it now has. Every route by which bytes can arrive AFTER the marker has
// been applied therefore ends here:
//
//   - a fetch completing (fetchWASMBlobOnce);
//   - an operator's __wasm_blob_put__, which is the documented escape hatch and
//     whose whole promise is that the group unblocks IMMEDIATELY, with no
//     restart. That promise is this function.
//
// The pre-registration PUSH does not need it: it lands before the marker, so
// applyWASMRegistration finds the blob already on disk and materializes it there.
// This is a no-op on that path, which is why it is cheap to call unconditionally.
//
// IT TAKES wasmApplyMu, AND THAT IS SAFE — but only because of where it runs, so
// state the argument rather than leaving it to be re-derived. It runs on a fetch
// goroutine or on a __wasm_blob_put__ handler goroutine, NEVER on an apply
// goroutine. A parked apply holds NO lock at all (see Node.onShardApplyRetry: it
// resolves under wasmApplyMu, releases, and only then waits), and everything that
// does hold wasmApplyMu — applyWASMRegistration, snapshotWASMState,
// restoreWASMState — completes without waiting on anything external. So there is
// no cycle: this can wait for an apply, and no apply ever waits for this.
//
// What would break that is making __wasm_blob_get__ take a lock, which is the
// contract opWASMBlobGetName exists to protect: a get is served WHILE a peer is
// parked in an apply, so a lock there is the one edge that closes the cycle.
//
// ###### IT SCANS UNDER THE LOCK AND MATERIALIZES OUTSIDE IT ######
//
// Not a deadlock either way — nothing under wasmApplyMu waits externally — but
// materializing means a file read and a wasm.Compile, and a multi-megabyte module
// compiles in the tens to hundreds of milliseconds. snapshotWASMState and
// restoreWASMState take this same mutex, so holding it across the compile would
// stall EVERY group's snapshot on this node for that long. That is precisely the
// node-global-from-group-local escalation the block path is built to avoid (see
// prefetchWASMBlob and fsm.onApplyRetry), and it would be perverse for the code
// that CLEARS a block to do what the block itself is careful not to.
//
// Dropping the lock before materializing needs nothing from the runtime's side:
// wasm.Runtime guards its own module table (its mu), and AddModule is
// content-addressed and idempotent, so a concurrent applyWASMRegistration
// installing the same slot is a no-op rather than a conflict. The worst a racing
// registration can do is add a slot after our scan — and that path materializes
// its own blob, which is on disk by the time we are called.
func (n *Node) installArrivedWASMBlob(fp [sha256.Size]byte) {
	// Every (export, fuel) pairing this fingerprint is referenced under. A blob is
	// content-addressed and op-agnostic, so one file can back several ops and
	// several versions, and each pairing is a DIFFERENT runtime slot (see
	// wasm.ModuleID). Materializing only the first would leave the others parked.
	type slot struct {
		op     string
		export string
		fuel   uint64
	}
	var (
		seen  = make(map[slot]struct{}, 2)
		slots = make([]slot, 0, 2)
	)

	n.wasmApplyMu.Lock()
	rt := n.wasmRT
	if rt != nil {
		consider := func(r ops.WASMRegistration) {
			if r.Blob != fp {
				return
			}
			k := slot{op: r.Name, export: r.ExportName, fuel: r.MaxFuel}
			dedup := slot{export: r.ExportName, fuel: r.MaxFuel}
			if _, dup := seen[dedup]; dup {
				return
			}
			seen[dedup] = struct{}{}
			slots = append(slots, k)
		}
		for _, in := range n.wasmState.installed {
			consider(in.reg)
			for _, b := range in.groups {
				consider(b.reg)
			}
		}
	}
	n.wasmApplyMu.Unlock()

	if rt == nil {
		return // pre-construction; finishWASMSetup materializes from disk.
	}
	for _, s := range slots {
		if err := materializeWASMBlob(n.cfg.DataDir, rt, fp, s.export, s.fuel); err != nil {
			slog.Error("wasm module bytes arrived but could not be instantiated; any group blocked on this version STAYS blocked",
				"component", "cluster", "op", s.op, "blob", ops.WASMBlobHex(fp), "err", err)
		}
	}
}
