// SPDX-License-Identifier: Apache-2.0

package cluster

import (
	"crypto/sha256"
	"errors"
	"fmt"
	"log/slog"
	"os"
	"path/filepath"
	"strings"
	"time"

	"github.com/rostamlabs/rostam/ops"
)

// WASM BLOB RETIREMENT — how a node reclaims the disk a superseded module
// version, or a put-only orphan, is holding.
//
// Module bytes are content-addressed, the executing version is bound per shard
// group, __wasm_blob_put__/__wasm_blob_get__ and the pre-registration push move
// them, and a registration marker names its module rather than carrying it — so
// an apply that cannot resolve its module BLOCKS, unbounded, until the bytes
// arrive. Nothing so
// far ever removes a blob: every version any group has ever bound stays on disk
// forever, and so does every blob an operator ever put by hand.
//
// This file is the only thing that removes one, and it is OFF unless
// Config.WASMBlobRetention is set.
//
// ############ READ THIS BEFORE CHANGING THE RULE ############
//
// THE RECORDED DESIGN'S JUSTIFICATION FOR RETIREMENT IS WRONG. It said a node
// may drop blob fp when no hosted group's PERSISTED binding names it, because
// "a durable restart resumes above the persisted applied index, and replay from
// there only encounters markers — which name their own fp and trigger a fetch",
// and it offered a sharper form: a superseded version stops being needed once
// every hosted group's LOG has compacted past the marker naming it, because a
// replica behind the compaction point is caught up by InstallSnapshot and the
// snapshot carries the group's CURRENT binding rather than the old markers.
//
// Both arguments are about THIS node. Neither survives contact with a LAGGING
// REPLICA ELSEWHERE, and the compaction form fails for two independent reasons
// that are worth stating separately because each one alone is fatal:
//
//	(1) LOG COMPACTION IS PER-NODE, AND IT IS THE WRONG NODE'S. Node X's
//	    compaction point is a fact about X's own log. The replica R that still
//	    needs version V has its OWN log, and R has NOT compacted past the marker:
//	    hashicorp/raft runs Snapshot on the same goroutine as Apply, so a replica
//	    that has not APPLIED past index i has not snapshotted past i and has not
//	    truncated past i either. R therefore replays those entries out of its own
//	    log and never asks for an InstallSnapshot at all — InstallSnapshot is
//	    chosen by the LEADER, from R's matchIndex (what R has RECEIVED), not from
//	    R's applied index (what R has EXECUTED). A replica that received
//	    everything and applied nothing is caught up by neither mechanism: it just
//	    applies, and needs V to do it.
//
//	(2) THE NODE THAT STILL NEEDS V IS EXACTLY THE NODE THAT DOES NOT HAVE V.
//	    R never applied the marker, so R never prefetched V — the prefetch is
//	    issued from applyWASMRegistration and from the classRetry hook, both of
//	    which are downstream of applying. The holders of V are precisely the nodes
//	    that DID apply the marker and have moved past it, which are precisely the
//	    nodes whose logs compact past it first. So the compaction rule concentrates
//	    deletion on the only holders, at the moment the lagging replica's need is
//	    still outstanding. It does not merely fail to protect R; it targets R's
//	    only sources.
//
// It is ALSO undecidable from local state, which is the lesser problem but worth
// recording so nobody tries to implement it: nothing anywhere stores the log
// index at which a group's binding was established (the register hook receives
// (shardIdx, registration) and no index), and nothing exposes a group's first
// log index — there is no FirstIndex bookkeeping in shard/ or cluster/ at all.
//
// ############ WHAT IS SHIPPED INSTEAD, AND WHAT IT DOES NOT CLAIM ############
//
// No purely LOCAL rule can be safe against a lagging replica. "Will any replica
// of any group still apply an entry under V" is not a function of anything this
// node holds, and making it one is cluster-wide GC — explicitly out of scope,
// and correctly so, since dropping a blob everywhere needs every group's binding
// cluster-wide.
//
// So this does not claim safety. It offers a BOUNDED, OPT-IN, OPERATOR-STATED
// exposure:
//
//	a blob is retired when nothing on this node has referenced it for
//	Config.WASMBlobRetention.
//
// The window is the operator's assertion that no replica of any group this node
// hosts will be more than that far behind the supersession of a module version.
// That is a statement about their operations — how long a node may be down,
// partitioned, or lagging — which an operator can know and this process cannot.
// Zero (the default) asserts nothing and retires nothing, so the shipped
// behaviour is byte-identical to retention being off.
//
// WHY THE TRADE IS ACCEPTABLE AT ALL, and it is only acceptable because of this:
// if the operator's assertion is wrong, the failure is the ordinary blocked
// apply. It is named in WASMBlockStats with the exact fingerprint, it is logged
// with escalating severity, and it is fixed by ONE admin call —
// __wasm_blob_put__ against the blocked node, no restart, no failover, no data
// movement. A loud, self-announcing, one-call-recoverable failure is a thing an
// operator may reasonably choose to risk for bounded disk. A silent one would
// not be.
//
// ############ THE SYMMETRY THAT MAKES THIS LESS ALARMING THAN IT SOUNDS #######
//
// Retirement drops exactly the set that a NEW NODE NEVER ACQUIRES. A shard
// snapshot carries the node-wide installs and the snapshotted group's CURRENT
// bindings (wasmSnapshotBlob) and nothing else, so a node joining today fetches
// the live versions and never fetches a superseded one. The superseded set
// therefore already decays out of the cluster by attrition, as nodes are
// replaced, with no rule and no bound. This makes that decay explicit, uniform
// and operator-controlled instead of accidental.
//
// ############ WHAT IS DELETED IS THE FILE, AND ONLY THE FILE ############
//
// The instantiated module is left in the runtime. That is not an omission, it is
// what makes the "grace period for invocations currently executing under it"
// that the recorded design asks for unnecessary rather than merely satisfied:
//
//   - wasm.Runtime.resolveModuleForInvoke asks the RUNTIME whether a version is
//     resident, never the filesystem (it runs on an apply goroutine and must not
//     do I/O). An invocation already executing under a retired version, and any
//     invocation a still-bound group issues under it, is completely unaffected by
//     the file going away;
//   - there is no safe eviction available anyway. rt.Invoke holds no reference
//     this package can see, so evicting a slot would be a race against every
//     in-flight call under it, for a memory saving nothing has asked for.
//
// Disk is the thing that actually grows without bound (one file per distinct
// registration per name, forever), and disk is what this reclaims.
//
// ############ THE DELETION PATH IS THE MOST DANGEROUS JOIN IN THE PACKAGE #####
//
// A fingerprint→path join that ends in os.Remove is strictly worse than one that
// ends in a read or a write, so retirableWASMBlob below is deliberately
// paranoid, and every clause of it is load-bearing:
//
//   - the candidate name comes from os.ReadDir, so it can never itself contain a
//     path separator, but that is a property of the SOURCE and the guarantee must
//     be attached to the INPUT (the same argument parseWASMBlobRef makes);
//   - the path is RE-DERIVED from the fingerprint through wasmBlobPath — the one
//     validator every fingerprint→path join in this package goes through — which
//     refuses anything that is not exactly a sha256 in lower-case hex. A
//     traversal is therefore unrepresentable, not merely rejected;
//   - the derived path must equal the entry that was listed. That pins the two
//     directions together: the file we delete is the file we looked at, and it is
//     also a file whose name is provably 64 hex characters;
//   - anything that fails either check is SKIPPED, never deleted. A blobs
//     directory holding a file this build cannot name is a file this build has no
//     business removing;
//   - os.Remove does not follow symlinks. A symlink planted in the blobs
//     directory (under a valid-looking fingerprint name) removes the LINK, never
//     its target, so the deletion cannot be steered outside the directory even by
//     an attacker with write access to it.

// wasmBlobRetireSweepCap / wasmBlobRetireSweepFloor bound how often the sweeper
// runs, relative to the configured retention.
//
// The sweep is a directory listing plus three map builds — cheap enough that the
// cadence is chosen for RESPONSIVENESS rather than cost, but there is no reason
// to do it more than once a minute for a retention measured in hours. The floor
// exists so a short retention (a test, or a deployment that genuinely wants
// aggressive reclamation) still gets a sweep inside its own window: a sweeper
// that ticks less often than the retention would make the effective window the
// tick, not the setting.
const (
	wasmBlobRetireSweepCap   = time.Minute
	wasmBlobRetireSweepFloor = 100 * time.Millisecond
)

// wasmBlobRetireInterval is the sweep cadence for a given retention. A quarter
// of the window means a blob is retired between one and one-and-a-quarter
// windows after it goes unreferenced — the overshoot is in the safe direction.
func wasmBlobRetireInterval(retention time.Duration) time.Duration {
	d := retention / 4
	if d > wasmBlobRetireSweepCap {
		d = wasmBlobRetireSweepCap
	}
	if d < wasmBlobRetireSweepFloor {
		d = wasmBlobRetireSweepFloor
	}
	return d
}

// startWASMBlobRetirement launches the sweeper, or does nothing when retirement
// is off.
//
// IT IS A NO-OP AT ZERO, AND THAT IS THE WHOLE DEFAULT. No goroutine, no timer,
// no directory listing, no bookkeeping map — a node with the flag unset behaves
// exactly as it did before this file existed, which is the property
// TestWASMBlobRetirementIsOffByDefault pins.
//
// It reuses wasmFetchStop rather than adding a second lifetime channel: that
// channel already means "this node is closing, stop the background WASM work",
// Close already closes it exactly once (closeOnce), and a sweeper that outlived
// Close would be deleting files out of a data directory the process no longer
// owns.
func (n *Node) startWASMBlobRetirement() {
	if n.cfg.WASMBlobRetention <= 0 {
		return
	}
	go n.runWASMBlobRetirement(wasmBlobRetireInterval(n.cfg.WASMBlobRetention))
}

// runWASMBlobRetirement is the sweeper loop. It sleeps FIRST: a blob that is
// unreferenced at process start must still wait out the window, and starting
// with a sweep would only ever record it as newly-unreferenced anyway.
func (n *Node) runWASMBlobRetirement(every time.Duration) {
	t := time.NewTicker(every)
	defer t.Stop()
	for {
		select {
		case <-n.wasmFetchStop:
			return
		case <-t.C:
		}
		retired, err := n.sweepWASMBlobsAt(time.Now())
		if err != nil {
			slog.Warn("wasm blob retirement sweep failed; nothing was removed",
				"component", "cluster", "err", err)
			continue
		}
		if len(retired) > 0 {
			slog.Info("retired unreferenced wasm module blobs",
				"component", "cluster", "count", len(retired), "blobs", strings.Join(retired, ","),
				"retention", n.cfg.WASMBlobRetention)
		}
	}
}

// sweepWASMBlobsAt is ONE retirement pass, with the clock passed in so the rule
// is testable without sleeping through a real window.
//
// It returns the fingerprints it removed. The bookkeeping map is REBUILT from
// the directory listing on every pass, which is what keeps it bounded: a
// fingerprint that has been deleted, or that became referenced again, simply
// does not appear in the new map.
//
// ###### A NODE THAT HOSTS NO SHARD GROUP RETIRES NOTHING ######
//
// This is the sharpest orphan case and it is easy to miss. Markers are applied
// per shard group, so a node hosting NO group applies no marker, records no
// sidecar, and therefore has an empty live set — every blob it holds looks like
// a put-only orphan. But those blobs are not garbage: they are the
// pre-registration PUSH's residue, and that node is one of the members the
// durability floor counted when it let the marker be proposed. Retiring them
// would delete the floor's holdings on exactly the nodes that hold nothing else,
// which is the membership-erosion failure already named as the open
// limit. "Unreferenced" carries no information on such a node, so the rule
// declines to act on it.
func (n *Node) sweepWASMBlobsAt(now time.Time) ([]string, error) {
	if n.cfg.WASMBlobRetention <= 0 {
		return nil, nil
	}
	if !n.hostsAnyShardGroup() {
		return nil, nil
	}
	dir := filepath.Join(n.cfg.DataDir, "wasm", wasmBlobsSubdir)
	entries, err := os.ReadDir(dir)
	if err != nil {
		if errors.Is(err, os.ErrNotExist) {
			return nil, nil // no blob store yet: nothing to retire
		}
		return nil, fmt.Errorf("cluster: wasm blob retirement: readdir %s: %w", dir, err)
	}

	keep := n.wasmBlobsInUse()

	n.wasmRetireMu.Lock()
	defer n.wasmRetireMu.Unlock()
	next := make(map[string]time.Time, len(entries))
	var retired []string
	for _, e := range entries {
		if e.IsDir() {
			continue
		}
		fp, path, ok := retirableWASMBlob(n.cfg.DataDir, dir, e.Name())
		if !ok {
			continue
		}
		if _, inUse := keep[fp]; inUse {
			continue
		}
		since, seen := n.wasmRetireUnrefSince[fp]
		if !seen {
			since = now
		}
		if now.Sub(since) < n.cfg.WASMBlobRetention {
			next[fp] = since
			continue
		}
		if err := os.Remove(path); err != nil && !errors.Is(err, os.ErrNotExist) {
			// Keep the clock running rather than restarting it: a permission or I/O
			// problem is not a reason to grant the blob a fresh window every sweep.
			next[fp] = since
			slog.Warn("could not retire an unreferenced wasm module blob",
				"component", "cluster", "blob", fp, "err", err)
			continue
		}
		retired = append(retired, fp)
	}
	n.wasmRetireUnrefSince = next
	n.wasmRetirePending.Store(int64(len(next)))
	n.wasmRetireSweeps.Add(1)
	if len(retired) > 0 {
		n.wasmBlobsRetired.Add(uint64(len(retired)))
	}
	return retired, nil
}

// retirableWASMBlob decides whether a directory entry NAMES A BLOB THIS BUILD
// MAY DELETE, and returns the fingerprint and the path to delete.
//
// ############ THIS IS THE TRAVERSAL GATE ON THE DELETE ############
//
// See the file header for the full argument. In short: the path is re-derived
// from the fingerprint through wasmBlobPath (which makes a traversal
// unrepresentable, not merely rejected) and must then equal the entry that was
// listed, so the file removed is provably `<dataDir>/wasm/blobs/<64 lower-case
// hex>.wasm` and provably the one this pass examined. Anything else — a
// stray file, a temp file atomicWriteFile left behind, an upper-case name, a
// short name, a name with a separator in it — is skipped, because a file this
// build cannot name is a file it has no business removing.
func retirableWASMBlob(dataDir, blobsDir, entryName string) (fp, path string, ok bool) {
	fp = strings.TrimSuffix(entryName, ".wasm")
	if fp == entryName {
		return "", "", false // not a blob file at all
	}
	path, err := wasmBlobPath(dataDir, fp)
	if err != nil {
		return "", "", false
	}
	if path != filepath.Join(blobsDir, entryName) {
		return "", "", false
	}
	return fp, path, true
}

// wasmBlobsInUse is the set of hex fingerprints this node MUST NOT retire,
// whatever their age. Three sources, and each answers a different question:
//
//	REFERENCED — named by any op's node-wide install, or by any hosted group's
//	binding. This is the persisted state (wasmState is what the sidecars hold),
//	and it covers the two cases the brief separates: a version a group still
//	executes, and a version this node installed node-wide. It is also what makes
//	a put-only ORPHAN detectable at all — an orphan is, by construction, a blob
//	no registration references, so one condition covers both classes;
//
//	BLOCKED ON — named by a live block record. A group parked waiting for these
//	exact bytes must not have them deleted from under it while they are in
//	flight to it. This cannot arise from the referenced set (a block implies a
//	binding), but it CAN arise from a fetch that has already landed the file
//	while the group has not yet re-run its apply;
//
//	BEING FETCHED — an in-flight fetch's fingerprint. Same reason, one step
//	earlier: a node already scrambling for a blob is a node in the
//	scarce-blob state, and taking a copy away then is the worst possible moment.
//
// The three locks are taken SEQUENTIALLY, never nested, and none of them is held
// across the sweep's file I/O.
func (n *Node) wasmBlobsInUse() map[string]struct{} {
	out := make(map[string]struct{}, 4)
	add := func(b [sha256.Size]byte) {
		if b == ops.ZeroWASMBlob {
			return
		}
		out[ops.WASMBlobHex(b)] = struct{}{}
	}

	n.wasmApplyMu.Lock()
	for _, in := range n.wasmState.installed {
		add(in.reg.Blob)
		for _, b := range in.groups {
			add(b.reg.Blob)
		}
	}
	n.wasmApplyMu.Unlock()

	n.wasmBlockMu.Lock()
	for _, rec := range n.wasmBlockLive {
		if rec.fingerprint != "" {
			out[rec.fingerprint] = struct{}{}
		}
	}
	n.wasmBlockMu.Unlock()

	n.wasmFetchMu.Lock()
	for fp := range n.wasmFetching {
		add(fp)
	}
	n.wasmFetchMu.Unlock()

	return out
}

// hostsAnyShardGroup reports whether this node hosts at least one shard group.
// See sweepWASMBlobsAt for why a node hosting none retires nothing.
func (n *Node) hostsAnyShardGroup() bool {
	n.shardMu.RLock()
	defer n.shardMu.RUnlock()
	for _, s := range n.shards {
		if s != nil {
			return true
		}
	}
	return false
}

// wasmBlobRetireStats renders the retirement sweeper for Node.Stats.
func (n *Node) wasmBlobRetireStats() WASMBlobRetireStats {
	return WASMBlobRetireStats{
		Retention: n.cfg.WASMBlobRetention,
		Sweeps:    n.wasmRetireSweeps.Load(),
		Retired:   n.wasmBlobsRetired.Load(),
		Pending:   n.wasmRetirePending.Load(),
	}
}
