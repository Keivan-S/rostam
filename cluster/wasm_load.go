// SPDX-License-Identifier: Apache-2.0

package cluster

import (
	"bytes"
	"crypto/sha256"
	"encoding/binary"
	"encoding/hex"
	"encoding/json"
	"errors"
	"fmt"
	"log/slog"
	"os"
	"path/filepath"
	"sort"
	"strings"

	"github.com/rostamlabs/rostam/ops"
	"github.com/rostamlabs/rostam/wasm"
)

// loadedModule is returned by loadWASMModules for each successfully
// loaded entry. The caller is responsible for Module.Close().
type loadedModule struct {
	Name     string
	Module   *wasm.Module
	ModuleID wasm.ModuleID
	BlobPath string
	MetaPath string
	// Registration is the canonical form of what was loaded, so the caller can
	// record it in wasmState without re-reading the file it just wrote.
	Registration ops.WASMRegistration
}

// wasmMeta is the JSON sidecar <dataDir>/wasm/<name>.json. It is the node's
// entire per-op record: every field of the registration EXCEPT the module bytes,
// plus a reference to the content-addressed blob that holds them. A restart
// rebuilds the exact ops.WASMRegistration from it — including the Epoch, without
// which a reloaded module could not be compared against an incoming registration
// and the convergence rule would restart from scratch on every process.
//
// IT IS ALSO THE COMMIT POINT for an install, and that is what makes the pair of
// writes safe without cross-file atomicity. The blob is written FIRST and is
// named by its own content, so it can never collide with, or overwrite, the blob
// of any other registration; only after it is durable is the sidecar renamed into
// place. A crash in between therefore leaves the PREVIOUS sidecar intact,
// pointing at the PREVIOUS blob, and the new blob merely unreferenced — the node
// restarts on the old registration, which is a state it was already in.
//
// Under the old layout, where the bytes lived at <name>.wasm, the same crash
// left the new bytes sitting under a sidecar describing the old Kind, extractor
// and Epoch, and a restart loaded that pair happily and ran new code under the
// old contract. Content addressing is what removed that, and it is why
// applyWASMRegistration no longer unlinks anything on the sidecar-failure path.
type wasmMeta struct {
	Kind       ops.OpKind `json:"kind"`
	ExportName string     `json:"export_name"`
	MaxFuel    uint64     `json:"max_fuel"`
	Epoch      uint64     `json:"epoch"`
	// Source records how this module got here. It is the ONLY thing that
	// distinguishes an operator-configured module from one installed by a
	// replicated __register_wasm__ entry once both have been written to disk,
	// and that distinction is load-bearing twice over:
	//
	//   - a config module must never win the (Epoch, fingerprint) comparison
	//     against a replicated registration. Its content is node-LOCAL, so if it
	//     could win, a node that happens to have it configured would install a
	//     different module than a node that does not — divergence produced by the
	//     very rule meant to prevent it. A replicated registration therefore
	//     always overrides a config module;
	//   - only replicated registrations are carried in the shard snapshot
	//     (see wasmSnapshotBytes). Operator config stays operator config.
	Source string `json:"source"`

	// Version is the sidecar format version. It is REQUIRED and must equal
	// wasmMetaVersion; anything else is refused at load time (see readWASMSidecar).
	//
	// It exists because the per-group binding table replaced the old Groups []int
	// field, and the replacement is not detectable by absence: a replicated op can
	// legitimately have an EMPTY binding set (a snapshot record the snapshotting
	// node did not attribute to its group installs the module and binds nothing),
	// so "no bindings key" cannot be distinguished from "a sidecar written before
	// bindings existed". Reading such a sidecar silently would leave every group
	// unbound while the op stayed registered — the op would be routable nowhere and
	// any entry that did reach it would halt the node. An explicit version is the
	// only thing that makes that a loud refusal.
	Version int `json:"version"`

	// Bindings is the PER-GROUP VERSION BINDING for this op: for each shard group
	// whose Raft log this node knows carries a registration of it, the module
	// version that group's log has committed (see installedWASM.groups). Sorted by
	// Group, so the sidecar is byte-stable across rewrites.
	//
	// It replaces the old Groups []int. The KEY SET is the same route-gate evidence
	// Groups was — the gate asks only "is group g proven?" — with the executing
	// version attached to it, so the gate and the resolver cannot drift apart.
	//
	// IT IS PERSISTED BECAUSE IT IS NOT RE-DERIVABLE. In durable mode fsm.Apply
	// SKIPS every entry at or below the recorded applied index, so the
	// __register_wasm__ entries that established these bindings are never applied
	// again. Without this field a restart would leave every group unbound: the op
	// would be unroutable on this node (route gate shut) AND any committed
	// invocation still to replay out of a group's log would fail with
	// ops.ErrWASMNoGroupBinding, which is classFatal.
	//
	// Each binding carries its OWN blob and module parameters rather than
	// referencing the top-level ones, because a group's bound version may differ
	// from the node-wide installed one — that is the entire point. The blobs are
	// content-addressed and shared, so the common case (every group on one version)
	// costs one file.
	Bindings []wasmMetaBinding `json:"bindings,omitempty"`

	// Blob is the hex ops.WASMBlobFingerprint of this op's module bytes, i.e.
	// the basename of <dataDir>/wasm/blobs/<Blob>.wasm.
	//
	// It is REQUIRED. A sidecar without it describes an op whose bytes cannot be
	// found, and there is no defensible fallback: the pre-blob layout kept them
	// at <name>.wasm, so silently reading that file back would resurrect exactly
	// the "bytes and metadata drift apart" failure the blob store removes. A
	// missing or empty Blob is refused at load time (see readWASMSidecar), which
	// on a data dir written by an older build means the node refuses to start —
	// see the ON-DISK BREAK note there.
	Blob string `json:"blob"`
}

// wasmMetaVersion is the current wasmMeta format. Version 2 replaced the
// route-gate's Groups []int with the per-group version binding table.
//
// VERSION 3 REMOVED NO FIELD AND ADDED NONE — it exists because a version-2
// sidecar's key_extractor_handle could say "raw", or the empty string that
// silently meant SHARDLESS, and this build must not act on either. Nothing about
// such a file's SHAPE says so: it parses cleanly and every field is populated.
// Left unversioned, a data dir written before ops.WASMKeyExtractorHandle would
// reload a "raw" op and put that node back on a different routing rule from its
// peers — the divergence the pin exists to make unrepresentable. The version byte
// is the only thing that turns that into a loud refusal instead of a silent one.
//
// DROPPING THE FIELD FROM wasmMeta DID NOT NEED A FURTHER BUMP, and the asymmetry
// is the rule this constant follows rather than an oversight. A version byte
// moves when an artefact would be MISREAD. Every version-3 sidecar was written
// with the handle already validated to the one legal value, so this build — which
// ignores the now-unknown JSON key and uses that extractor unconditionally —
// reads exactly what a version-3 build meant. There is nothing to refuse.
// Contrast wasmSnapshotBlobVersion, which DID move: its records are positional
// bytes, so removing the field changes what the same bytes decode to.
//
// ON-DISK BREAK, deliberate and unmigrated (the app is unreleased): a data dir
// written by a version-1 or version-2 build refuses to load. The remedy is the
// one in wasmRecoveryAdvice — wipe the node's data dir and rejoin, subject to its
// two caveats — or, for a config-only deployment, delete <dataDir>/wasm and let
// cfg.WASMModules reinstall.
const wasmMetaVersion = 3

// wasmMetaBinding is one entry of wasmMeta.Bindings: everything needed to rebuild
// the module version shard group Group's log has committed, without consulting
// the sidecar's top-level (node-wide) fields — which may describe a DIFFERENT
// version.
//
// Kind is carried even though it is frozen at first registration, because the
// apply-time freeze is enforced BY COMPARISON against it: applyWASMRegistration
// refuses a registration whose Kind differs from what this group's binding
// already records, and that comparison has to survive a restart or the refusal
// silently stops firing.
type wasmMetaBinding struct {
	Group      int        `json:"group"`
	Kind       ops.OpKind `json:"kind"`
	ExportName string     `json:"export_name"`
	MaxFuel    uint64     `json:"max_fuel"`
	Epoch      uint64     `json:"epoch"`
	Blob       string     `json:"blob"`
}

// wasmBlobsSubdir is the directory, under <dataDir>/wasm, holding the
// content-addressed module blobs. It is a SUBDIRECTORY rather than a filename
// convention so reloadWASMModulesFromDisk's scan over the sidecars cannot
// collide with it, whatever an op is called.
const wasmBlobsSubdir = "blobs"

// errWASMBlob marks a blob that cannot be used: absent, unreadable, or holding
// contents that do not hash to its own filename.
//
// THE HASH CHECK IS THE POINT OF THE LAYOUT. A blob is valid iff
// sha256(contents) == the hex in its name, so every read re-derives the identity
// of what it just loaded instead of trusting the path it came from. Bit rot, a
// truncated write on a filesystem that does not honour the fsync, or a
// hand-edited file are all caught at load rather than compiled and run. The
// previous <name>.wasm layout could not check anything: the file's name carried
// no claim about its contents.
var errWASMBlob = errors.New("cluster: wasm module blob is missing or corrupt; " + wasmBlobRecoveryAdvice)

// wasmBlobRecoveryAdvice is the operator instruction for a MISSING OR CORRUPT
// BLOB, as distinct from a missing or unreadable sidecar (wasmRecoveryAdvice).
//
// IT LEADS WITH __wasm_blob_put__ RATHER THAN WITH THE WIPE, and the difference
// is not stylistic. This error is what a blocked group's reported reason is made
// of, an operator reads it mid-incident, and the two remedies are not remotely
// comparable:
//
//   - a blob put moves ONE content-addressed file to ONE node. It hash-verifies
//     and compiles before it acks, so a wrong file is refused rather than
//     accepted, and the blocked group's very next apply retry succeeds. No
//     restart, no failover, no data movement, and nothing to undo if it fails;
//   - "wipe the data dir and rejoin" is unqualified DATA LOSS when this node is
//     the last healthy replica of a group it hosts (see wasmRecoveryAdvice's
//     second caveat), and it takes a full catch-up even when it is safe.
//
// Leading with the wipe therefore put the destructive remedy first for a problem
// whose non-destructive remedy is a single admin call. A blob is content-
// addressed and immutable, which is exactly what makes the cheap fix total: any
// copy of the right bytes, from anywhere, is THE right bytes.
//
// The wipe is still named, because a blob put needs the module file to hand and
// an operator may not have it.
const wasmBlobRecoveryAdvice = "a blob is content-addressed, so ANY copy of the right bytes fixes this: " +
	"run `rostam call __wasm_blob_put__ <fingerprint-hex><module-bytes>` against THIS node with the module the fingerprint names " +
	"(Stats().WASM.Blocked prints the exact hex for every blocked group). The put verifies the hash and compiles the module before it acks, " +
	"so a wrong file is refused rather than accepted, and any group blocked on this version unblocks on its very next apply retry — " +
	"no restart, no failover, no data movement. The node also fetches the blob from its peers on its own and retries forever, " +
	"so this is only needed when no reachable member still holds it. " +
	"DO NOT delete the <name>.json sidecar to force a re-apply: in durable mode fsm.Apply skips every entry at or below the persisted applied index, " +
	"so the registration is never re-applied from the Raft log, the node comes up CLEAN with the op silently absent, " +
	"and it omits the op from every snapshot it serves — replicas joining from this node inherit the same gap. " +
	"Wiping this node's data dir and rejoining also recovers the module, but it is a LAST RESORT: " +
	"it is unqualified DATA LOSS if this node is the last healthy replica of any group it hosts, or a PB primary with no in-sync replica, " +
	"and it does not recover a module installed from this node's config (that comes back only from cfg.WASMModules, so keep that config)"

// wasmBlobPath resolves the blob file for a hex fingerprint, refusing anything
// that is not exactly a sha256 in lower-case hex.
//
// The fingerprint reaches this function from a JSON sidecar, which is node-local
// operator-adjacent state rather than wire input — but it is joined onto a path,
// and filepath.Join RESOLVES a traversal rather than failing on it, so a
// hand-edited or corrupted sidecar could otherwise name a file anywhere on the
// host. The format check is free and makes traversal unrepresentable.
func wasmBlobPath(dataDir, fp string) (string, error) {
	raw, err := hex.DecodeString(fp)
	if err != nil || len(raw) != sha256.Size {
		return "", fmt.Errorf("%w: %q is not a sha256 hex fingerprint", errWASMBlob, fp)
	}
	if fp != strings.ToLower(fp) {
		return "", fmt.Errorf("%w: fingerprint %q is not lower-case hex", errWASMBlob, fp)
	}
	return filepath.Join(dataDir, "wasm", wasmBlobsSubdir, fp+".wasm"), nil
}

// writeWASMBlob stores b at its content address and returns the hex
// fingerprint. It is idempotent by construction: the same bytes always land on
// the same path with the same contents, so two ops sharing a module share one
// file and a re-registration rewrites a file that is already identical.
//
// It uses atomicWriteFile, so a crash never leaves a SHORT blob — which matters
// even though the reader hash-checks, because a short blob that a reader then
// refuses is a node that will not start.
func writeWASMBlob(dataDir string, b []byte) (string, error) {
	sum := ops.WASMBlobFingerprint(b)
	fp := hex.EncodeToString(sum[:])
	path, err := wasmBlobPath(dataDir, fp)
	if err != nil {
		return "", err
	}
	if err := os.MkdirAll(filepath.Dir(path), 0o750); err != nil {
		return "", fmt.Errorf("cluster: mkdir %s: %w", filepath.Dir(path), err)
	}
	if err := atomicWriteFile(path, b); err != nil {
		return "", err
	}
	return fp, nil
}

// readWASMBlob loads the blob named by fp and verifies it against its own name.
func readWASMBlob(dataDir, fp string) ([]byte, error) {
	path, err := wasmBlobPath(dataDir, fp)
	if err != nil {
		return nil, err
	}
	b, err := os.ReadFile(path) //nolint:gosec // path is <dataDir>/wasm/blobs/<validated hex>.wasm
	if err != nil {
		return nil, fmt.Errorf("%w: read %s: %v", errWASMBlob, path, err)
	}
	if got := ops.WASMBlobFingerprint(b); hex.EncodeToString(got[:]) != fp {
		return nil, fmt.Errorf("%w: %s holds %d bytes hashing to %s, not to its own name",
			errWASMBlob, path, len(b), hex.EncodeToString(got[:]))
	}
	return b, nil
}

const (
	wasmSourceConfig     = "config"
	wasmSourceReplicated = "replicated"
)

// wasmGroupBinding is the module version ONE shard group's Raft log has
// committed for ONE op name: the registration that won the fold over that
// group's log prefix, plus the wasm.ModuleID it resolves to.
//
// reg carries the full registration rather than just the ModuleID because three
// consumers need the rest of it: the apply-time freeze compares Kind against it,
// the fold compares (Epoch, fingerprint) against
// it, and the sidecar/snapshot must be able to rebuild the version from it. Bytes
// is a slice header shared with the installed registration, so a binding costs
// no module memory.
type wasmGroupBinding struct {
	reg ops.WASMRegistration
	id  wasm.ModuleID
}

// newWASMGroupBinding derives the binding r would establish.
//
// It needs NO module bytes, which is what makes thin markers work at all: the
// ModuleID is derived from the marker's content address (wasm.ModuleIDForBlob),
// so a node that has not fetched the module still computes the exact slot the
// module will occupy once it arrives, and can therefore bind, gate, persist and
// publish it in the meantime. Whether that slot is currently OCCUPIED is a
// separate question asked at execution time, once per invocation, by
// wasm.Runtime.resolveModuleForInvoke.
func newWASMGroupBinding(r ops.WASMRegistration) wasmGroupBinding {
	return wasmGroupBinding{reg: r, id: wasm.ModuleIDForBlob(r.Blob, r.ExportName, r.MaxFuel)}
}

// installedWASM is one entry of wasmState.
type installedWASM struct {
	// reg is the NODE-WIDE installed registration: the (Epoch, fingerprint)
	// maximum over everything this node has applied for this name, from any group.
	// It is what the ops.Registry entry was installed from, and therefore what
	// every call with no group to bind against executes (see
	// wasm.Runtime.resolveModuleForInvoke's fallback cases).
	reg ops.WASMRegistration
	// replicated is true when this install came from a committed
	// __register_wasm__ entry (or a snapshot of one) rather than from operator
	// config. See wasmMeta.Source.
	replicated bool

	// groups is the PER-GROUP VERSION BINDING TABLE for this op name: for each
	// shard group whose Raft LOG this node knows carries a registration of it, the
	// version that group's log has committed.
	//
	// IT IS ONE STRUCTURE SERVING TWO CONSUMERS, deliberately. Its KEY SET is the
	// route gate's evidence (see checkWASMRouteGate) — "group g's log carries a
	// registration for this name" — which is exactly what it was before per-group
	// binding, when it was a map[int]struct{}. Its VALUES are what
	// wasm.Runtime.resolveModuleForInvoke executes. Keeping them in one map is what
	// makes it impossible for the gate to consider a group proven while the
	// resolver finds no version for it.
	//
	// THE VALUE IS A FOLD OVER THAT GROUP'S ORDERED LOG PREFIX, and the property
	// that holds is PREFIX-DETERMINISM, not permutation-invariance. The difference
	// is not pedantry — it is the whole safety argument, so state it exactly:
	//
	//	group g's binding for a name is a pure function of the SEQUENCE of
	//	registrations of that name in g's log prefix.
	//
	// Every replica of g has applied the SAME sequence — that is what a Raft log
	// is — so every replica of g derives the SAME binding, and which OTHER groups
	// the applying node hosts, and how far along them it is, cannot affect it.
	// That is the invariant per-group binding exists to establish.
	//
	// IT IS NOT PERMUTATION-INVARIANT, and an earlier version of this comment
	// wrongly said it was. bindWith ALONE is a (Epoch, fingerprint) MAXIMUM
	// (ops.WASMRegistrationNewer, the same total order the node-wide install uses)
	// and a maximum is commutative. But the fold that actually runs in production
	// is bindWith COMPOSED WITH checkWASMGroupContract (see
	// applyWASMRegistration), and the contract check fires ONLY when the group
	// already has a binding to compare against. Hence:
	//
	//	FIRST-REGISTRATION-PINS-THE-CONTRACT. For every non-empty prefix of g's
	//	log, the contract (Kind) of g's binding equals that of
	//	the FIRST registration of that name in g's log. By induction: the first
	//	one establishes the binding with no check, and every later one moves the
	//	binding only after checkWASMGroupContract has proved its contract EQUAL to
	//	the incumbent's.
	//
	// So the fold is a maximum WITHIN one contract and first-wins ACROSS contracts.
	// Worked counterexample, with A = (OpReadWrite, Epoch 1) and
	// B = (OpReadOnly, Epoch 5) in one group's log: the order A,B binds A
	// (B is refused — it changes the Kind), and the order B,A binds B (B is
	// unchecked, then A loses the maximum). Same SET, two answers. That is not a
	// defect, because a log is a sequence and every replica of g sees the same
	// one; it is only a defect if a test asserts the wrong property. See
	// TestWASMBindingIsPrefixDeterministic.
	//
	// EVERY CATCH-UP ROUTE PRESERVES IT.
	//   - LOG REPLAY: by construction — every replica of g applies one identical
	//     ordered prefix.
	//   - SNAPSHOT: wasmSnapshotBlob carries the fold's RESULT, not its input
	//     sequence, so the restorer inherits the binding rather than recomputing
	//     it from an arrival order it never saw.
	//   - SNAPSHOT ONTO A REPLICA THAT ALREADY HOLDS A BINDING FOR g: here the
	//     contract check DOES fire, and it PASSES — by the lemma. The snapshot's
	//     binding and this replica's binding are folds over two prefixes of the
	//     SAME log (g's), and every non-empty prefix of a log shares its first
	//     element, so both contracts are the first registration's and are equal. A
	//     restore can therefore move a group's binding forward, but can never be
	//     refused into leaving it stale. See
	//     TestWASMSnapshotOntoAnExistingBindingIsNotRefused.
	//
	// See bindWith.
	//
	// WHAT THIS REPLACED AND WHY THAT WAS NOT ENOUGH. As a bare group SET it
	// established exactly one thing: every replica of g can LOOK THE NAME UP by the
	// time it reaches an invocation, so none of them meets
	// shard.ErrOpNotRegistered and halts. It did NOT establish that they all run
	// the same VERSION, because the ops.Registry entry that bound name → version
	// was node-wide and registrations arrive through per-group logs that commit at
	// independent times. Re-keying the SET on (name, fingerprint) could not close
	// that either — the node still running the old version holds correct, complete
	// evidence for the version IT runs, so its gate stays open and it keeps
	// proposing, and it is that proposal the already-updated node applies with
	// different bytes. Nothing available at PROPOSE time can see a peer's version.
	// Attaching the version here, and resolving from it at APPLY time, is what
	// closes it.
	//
	// MEMBERSHIP IS MONOTONIC and never has to be retracted — a Raft log only ever
	// grows, and snapshots carry the binding forward — so a placement change, a
	// restart, or a superseding registration all only ever ADD to it or move a
	// group's binding FORWARD under the (Epoch, fingerprint) order.
	//
	// nil is the empty table (no group is known to carry it), which is the
	// conservative state: this node will not propose an invocation into any
	// group it hosts.
	groups map[int]wasmGroupBinding
}

// bindWith folds r into group idx's binding and reports whether the binding
// CHANGED.
//
// ########################## THIS IS THE FOLD ##########################
//
// It is half the per-group-binding rule in one place: group idx binds the
// (Epoch, fingerprint) MAXIMUM of the registrations of this name in idx's log
// prefix.
//
// READ THIS METHOD'S PROPERTY AND THE PRODUCTION FOLD'S PROPERTY SEPARATELY —
// they are not the same, and only one of them is what safety rests on. bindWith
// ON ITS OWN is a maximum, hence commutative and associative, hence
// permutation-invariant. The fold that actually runs is bindWith COMPOSED WITH
// checkWASMGroupContract (applyWASMRegistration), and that composition is
// order-DEPENDENT: the contract check has nothing to compare a group's FIRST
// registration against, so that first registration pins the group's contract and
// every later contract-differing one is refused. What the design needs, and the
// only thing it claims, is PREFIX-DETERMINISM: the same ordered log prefix
// yields the same binding, on every replica of the group. See
// installedWASM.groups for the lemma, the worked counterexample, and why every
// catch-up route preserves it.
//
// idx == ops.NoShardIndex binds nothing: there is no group to attribute the
// registration to, so recording it against group 0 would be a false claim about
// a log this registration may never have entered. The install still happens —
// see applyWASMRegistration.
//
// The guard is EQUALITY against the sentinel rather than `idx < 0`, to match
// wasm.Runtime.resolveModuleForInvoke's case (3) exactly. The two are the write
// and read ends of one table and they must agree on what counts as "no group":
// under the old `idx < 0` a hypothetical index of, say, -5 would bind NOTHING
// here while the resolver treated it as a real group, missed, and halted the node
// under classFatal ops.ErrWASMNoGroupBinding. No caller produces such an index
// today (a dispatcher reports 0..NumShards-1, and every other path reports the
// sentinel), so this is a guard against a future divergence, not a live bug.
//
// The result is always a FRESH map when it differs, so a caller holding the old
// one (a published binding/gate snapshot, read without a lock by every shard
// group's apply goroutine) never observes a mutation.
func (in installedWASM) bindWith(idx int, r ops.WASMRegistration) (map[int]wasmGroupBinding, bool) {
	if idx == ops.NoShardIndex {
		return in.groups, false
	}
	if cur, have := in.groups[idx]; have && !ops.WASMRegistrationNewer(r, cur.reg) {
		return in.groups, false
	}
	out := make(map[int]wasmGroupBinding, len(in.groups)+1)
	for g, b := range in.groups {
		out[g] = b
	}
	out[idx] = newWASMGroupBinding(r)
	return out, true
}

// sortedGroups renders a group-keyed map's key set in ascending order, for the
// JSON sidecar and for Stats. It is generic over the value so the one helper
// serves both the binding table and any plain group set.
func sortedGroups[V any](groups map[int]V) []int {
	if len(groups) == 0 {
		return nil
	}
	out := make([]int, 0, len(groups))
	for g := range groups {
		out = append(out, g)
	}
	sort.Ints(out)
	return out
}

// sidecarBindings renders a binding table for the JSON sidecar, sorted by group.
func sidecarBindings(groups map[int]wasmGroupBinding) []wasmMetaBinding {
	if len(groups) == 0 {
		return nil
	}
	out := make([]wasmMetaBinding, 0, len(groups))
	for _, g := range sortedGroups(groups) {
		b := groups[g]
		out = append(out, wasmMetaBinding{
			Group:      g,
			Kind:       b.reg.Kind,
			ExportName: b.reg.ExportName,
			MaxFuel:    b.reg.MaxFuel,
			Epoch:      b.reg.Epoch,
			Blob:       ops.WASMBlobHex(b.reg.Blob),
		})
	}
	return out
}

// pendingWASMRestore is one buffered snapshot WASM section plus the shard group
// whose snapshot delivered it (see Node.wasmRestorePending). The group is kept
// because it is the route-gate provenance for the records the blob flags as
// carried by that group, and by drain time there is no other way to recover it.
type pendingWASMRestore struct {
	group int
	blob  []byte
}

// wasmState is the node's record of WHICH registration is currently installed
// under each op name. It is the reference point for the order-independent
// convergence rule (ops.WASMRegistrationNewer): without it, "is this incoming
// registration newer than what I have?" would have to be answered from the
// wasm.Runtime, which only fingerprints (bytes, export, fuel) and knows nothing
// about Epoch, Kind or the key-extractor handle.
//
// It is NOT independently locked: every mutator runs under Node.wasmApplyMu,
// which already has to serialize the per-group FSM apply goroutines against each
// other (they share the files, the runtime and the ops registry).
type wasmState struct {
	installed map[string]installedWASM
}

func newWASMState() *wasmState {
	return &wasmState{installed: make(map[string]installedWASM, 4)}
}

// replicatedRegs returns the installs that arrived through replication, in
// deterministic (name) order. Config modules are excluded — see wasmMeta.Source.
func (s *wasmState) replicatedRegs() []installedWASM {
	if s == nil {
		return nil
	}
	names := make([]string, 0, len(s.installed))
	for name, in := range s.installed {
		if in.replicated {
			names = append(names, name)
		}
	}
	sort.Strings(names)
	out := make([]installedWASM, 0, len(names))
	for _, name := range names {
		out = append(out, s.installed[name])
	}
	return out
}

// loadWASMModules persists each module from cfg to <dataDir>/wasm/,
// compiles it, and optionally adds it to rt (when rt is non-nil).
//
// The duplicate-name check runs first so no side effects happen before
// all names are validated.
// st, when non-nil, records each loaded module as a CONFIG install so a later
// replicated registration for the same name is known to override it rather than
// compete with it on (Epoch, fingerprint).
//
// A CONFIG MODULE WHOSE NAME ALREADY HAS A REPLICATED SIDECAR IS SKIPPED
// ENTIRELY — not written, not compiled, not added to the runtime, not recorded
// in st — and is reported in the returned skipped set. applyWASMRegistration
// documents that a replicated registration always overrides a config module, and
// at apply time it does; without this check the RESTART path reversed it, because
// loadWASMModules runs BEFORE reloadWASMModulesFromDisk and used to overwrite
// both files unconditionally. The damage was permanent and silent: the node ran
// config bytes while its peers ran replicated bytes (divergence); the sidecar's
// Source flipped to "config", which drops the op out of the route-gate snapshot
// (publishWASMGateLocked gates only replicated installs) so the node proposed
// invocations into groups with no evidence at all; the op vanished from every
// snapshot the node took; and the durable Groups/Epoch were erased, neither of
// which is re-derivable — a durable restart never replays the __register_wasm__
// entries that established them.
//
// SKIP rather than fail construction. The replicated registration is the
// cluster-wide truth and is already installed on the peers, so honouring it is
// the outcome that keeps this node in agreement. Failing construction would be
// loud but far worse in practice: operator config is typically IDENTICAL on
// every node, so one stale entry naming a since-replicated module would stop the
// WHOLE cluster from starting.
//
// Callers must not register the skipped names from cfg — see the returned set.
func loadWASMModules(dataDir string, cfg []WASMModuleConfig, rt *wasm.Runtime, st *wasmState) ([]loadedModule, error) {
	// First pass: reject duplicate and unsafe names before any side effects.
	//
	// The name check matters as much here as on the wire path, for the same
	// reason: c.Name is joined onto <dataDir>/wasm to build two file paths, and
	// filepath.Join RESOLVES a traversal rather than failing on it. Operator config
	// is trusted, but a typo'd or templated name containing a separator would write
	// outside the data dir on every node that carries the config, and the resulting
	// pair would then be invisible to reloadWASMModulesFromDisk (which globs only
	// <dataDir>/wasm/*.wasm). Failing construction is the right answer for config:
	// unlike a replicated registration there is no cluster-wide truth to honour,
	// and the operator is present to fix it.
	seen := make(map[string]struct{}, len(cfg))
	for _, c := range cfg {
		if err := ops.ValidateWASMOpName(c.Name); err != nil {
			return nil, fmt.Errorf("cluster: loadWASMModules: %w", err)
		}
		if _, dup := seen[c.Name]; dup {
			return nil, fmt.Errorf("cluster: loadWASMModules: duplicate module name %q", c.Name)
		}
		seen[c.Name] = struct{}{}
	}

	wasmDir := filepath.Join(dataDir, "wasm")
	if err := os.MkdirAll(wasmDir, 0o750); err != nil {
		return nil, fmt.Errorf("cluster: loadWASMModules: mkdir %s: %w", wasmDir, err)
	}

	loaded := make([]loadedModule, 0, len(cfg))
	for _, c := range cfg {
		meta, have, err := readWASMSidecar(dataDir, c.Name)
		if err != nil {
			return nil, fmt.Errorf("cluster: loadWASMModules: module %q: %w", c.Name, err)
		}
		if have && meta.Source == wasmSourceReplicated {
			// Leave the sidecar and its blob untouched; reloadWASMModulesFromDisk
			// installs the replicated module, its Epoch, and its proven-group set.
			continue
		}
		// NO SIDECAR MEANS NO REPLICATED REGISTRATION, and unlike the old layout
		// that is now the whole truth rather than a guess. Under <name>.wasm the
		// bytes of a replicated module sat under the op's own name, so a missing
		// sidecar left a file that LOOKED like provenance and had to be refused
		// (errOrphanWASM) in case a config module was about to clobber it.
		//
		// A blob is named by its content and says nothing about which op — or how
		// many ops — reference it, so that inference is not merely dropped here, it
		// is unavailable in principle. WHAT IT COSTS, stated plainly: if an operator
		// DELETES a replicated sidecar by hand while the same op name is also in
		// cfg.WASMModules, the config module now takes the name silently, and in
		// durable mode the replicated registration is never re-applied — the node
		// runs config bytes while its peers run replicated bytes. What it buys is
		// that the CRASH which used to produce that same half-state cannot happen at
		// all: the sidecar is written last and is the only per-op artifact, so a
		// crash mid-install leaves the previous sidecar, not a sidecar-less module.
		// The remaining trigger is manual deletion, and the mirror of it — a sidecar
		// whose blob is gone — is now a loud refusal (errWASMBlob) where the old
		// layout dropped the op silently and halted on the first invocation.
		//
		// THE COSTLIER HALF, which has NO config module involved at all: deleting the
		// sidecar of a replicated op that is NOT in cfg.WASMModules. The old layout
		// globbed *.wasm and refused to start on the leftover module file; the scan is
		// now over *.json, so there is no leftover to find and the node starts CLEAN
		// with the op simply absent. It is missing from ops.Registry, so the first
		// committed invocation any peer proposes hits shard.ErrOpNotRegistered →
		// classFatal → halt, with no replay path in durable mode. And it is missing
		// from wasmState.installed, so replicatedRegs omits it from wasmSnapshotBlob:
		// every replica that InstallSnapshots from this node inherits the loss and
		// halts the same way. The old startup refusal stopped this node serving at
		// all, which contained it. This is a real regression in blast radius and it is
		// the reason wasmRecoveryAdvice tells operators never to delete a sidecar.
		lm, err := loadOneModule(dataDir, wasmDir, c, rt)
		if err != nil {
			return nil, fmt.Errorf("cluster: loadWASMModules: module %q: %w", c.Name, err)
		}
		loaded = append(loaded, lm)
		if st != nil {
			st.installed[c.Name] = installedWASM{reg: lm.Registration, replicated: false}
		}
	}
	return loaded, nil
}

// loadOneModule validates a config module COMPLETELY before it writes anything.
//
// Order is the whole point. Compiling and running the OpReadOnly/writes-state
// guard first means a module that cannot be registered leaves NO files behind.
// Writing first (the previous order) was unrecoverable at cluster scale: a
// module declared OpReadOnly that imports cache_put had its .wasm and sidecar
// written on EVERY node and only then failed the guard, and on the next restart
// reloadWASMModulesFromDisk re-registered it, hit the same non-ErrDuplicateOp
// failure, and failed node construction — so no node could start until someone
// deleted the file by hand.
func loadOneModule(dataDir, wasmDir string, c WASMModuleConfig, rt *wasm.Runtime) (loadedModule, error) {
	b, err := resolveBytes(c)
	if err != nil {
		return loadedModule{}, err
	}

	mod, err := wasm.Compile(b)
	if err != nil {
		return loadedModule{}, fmt.Errorf("compile: %w", err)
	}
	// The guard that decides whether this module is registrable at all, run
	// against the compiled artifact rather than against the runtime slot — so it
	// fires before anything is written.
	if err := wasm.ValidateModuleKind(c.Name, c.Kind, mod); err != nil {
		_ = mod.Close() //nolint:errcheck,gosec
		return loadedModule{}, err
	}

	// Blob first, sidecar second: the sidecar is the commit point, so it must
	// never name a blob that is not already durable.
	blobFP, err := writeWASMBlob(dataDir, b)
	if err != nil {
		_ = mod.Close() //nolint:errcheck,gosec
		return loadedModule{}, err
	}
	blobPath, err := wasmBlobPath(dataDir, blobFP)
	if err != nil {
		_ = mod.Close() //nolint:errcheck,gosec
		return loadedModule{}, err
	}

	meta := wasmMeta{
		Kind:       c.Kind,
		ExportName: c.ExportName,
		MaxFuel:    c.MaxFuel,
		Source:     wasmSourceConfig,
		Version:    wasmMetaVersion,
		Blob:       blobFP,
	}
	metaBytes, err := json.Marshal(meta)
	if err != nil {
		_ = mod.Close() //nolint:errcheck,gosec
		return loadedModule{}, fmt.Errorf("marshal meta: %w", err)
	}
	metaPath := filepath.Join(wasmDir, c.Name+".json")
	if err := atomicWriteFile(metaPath, metaBytes); err != nil {
		_ = mod.Close() //nolint:errcheck,gosec
		return loadedModule{}, fmt.Errorf("write meta: %w", err)
	}

	var id wasm.ModuleID
	if rt != nil {
		if id, err = rt.AddModule(mod, c.ExportName, c.MaxFuel); err != nil {
			_ = mod.Close() //nolint:errcheck,gosec
			return loadedModule{}, fmt.Errorf("add to runtime: %w", err)
		}
	} else {
		id = wasm.ModuleIDFor(b, c.ExportName, c.MaxFuel)
	}

	return loadedModule{
		Name:     c.Name,
		Module:   mod,
		ModuleID: id,
		BlobPath: blobPath,
		MetaPath: metaPath,
		Registration: ops.WASMRegistration{
			Name:       c.Name,
			Kind:       c.Kind,
			Blob:       ops.WASMBlobFingerprint(b),
			ExportName: c.ExportName,
			MaxFuel:    c.MaxFuel,
		},
	}, nil
}

// resolveBytes returns the WASM bytes from c. Exactly one of Bytes or Path
// must be set; both set or both empty is an error.
func resolveBytes(c WASMModuleConfig) ([]byte, error) {
	hasBytes := len(c.Bytes) > 0
	hasPath := c.Path != ""
	switch {
	case hasBytes && hasPath:
		return nil, errors.New("bytes and Path are mutually exclusive")
	case hasBytes:
		return c.Bytes, nil
	case hasPath:
		b, err := os.ReadFile(c.Path) //nolint:gosec // path comes from trusted operator config
		if err != nil {
			return nil, fmt.Errorf("read %s: %w", c.Path, err)
		}
		return b, nil
	default:
		return nil, errors.New("one of Bytes or Path must be set")
	}
}

// reloadWASMModulesFromDisk scans <dataDir>/wasm/*.json — the SIDECARS — and
// reloads each op's module into rt from the content-addressed blob the sidecar
// names, then registers the op in reg.
//
// THE SCAN IS OVER SIDECARS, NOT OVER MODULE FILES, and that inversion is the
// invariant "a module present with no metadata must not load silently" expressed
// structurally instead of as a check. A blob nothing references is simply not
// reachable from this loop: it has no op name, no Kind, no key extractor and no
// provenance, so there is nothing it could be loaded AS. The old layout globbed
// *.wasm and had to catch the sidecar-less case explicitly (errOrphanWASM),
// because a file called <name>.wasm looked like it named an op.
//
// The opposite half — a sidecar whose blob is missing or corrupt — is a hard
// error (errWASMBlob). Under the old layout that direction was the silent one: a
// deleted <name>.wasm left the .json unread, the op vanished from the registry
// with no complaint, and the node halted under classFatal on the first committed
// invocation of it.
//
// Config-loaded modules win on conflict: a name already present in st was just
// installed from cfg.WASMModules in this process, so the on-disk copy (written
// by a PREVIOUS process) is stale and skipped. That check replaces the old
// "register and swallow ErrDuplicateOp" idiom, which no longer works now that
// the install path REPLACES rather than refusing duplicates.
//
// Everything it does install is recorded in st with the Source AND the proven
// shard-group set recorded in the sidecar, so the epoch comparison, the snapshot
// serializer and the route gate all survive a restart with the same view they
// had before it.
//
// It DOES register each reloaded op in reg, unconditionally and before any group
// has been re-proven. That is deliberate. Registry presence is what the FSM needs
// to APPLY an entry; withholding it would recreate the classFatal
// ErrOpNotRegistered halt on the first invocation replayed out of a group's log.
// What is withheld until a group is proven is the right to PROPOSE an invocation
// into that group — see checkWASMRouteGate.
func reloadWASMModulesFromDisk(dataDir string, rt *wasm.Runtime, reg *ops.Registry, st *wasmState, prefetch wasmPrefetchFn) error {
	wasmDir := filepath.Join(dataDir, "wasm")
	entries, err := os.ReadDir(wasmDir)
	if err != nil {
		if errors.Is(err, os.ErrNotExist) {
			return nil
		}
		return fmt.Errorf("cluster: reloadWASMModulesFromDisk: readdir %s: %w", wasmDir, err)
	}

	for _, entry := range entries {
		// The blobs/ subdirectory is skipped by IsDir; every remaining .json is a
		// per-op sidecar this node wrote.
		if entry.IsDir() || !strings.HasSuffix(entry.Name(), ".json") {
			continue
		}
		name := strings.TrimSuffix(entry.Name(), ".json")
		if st != nil {
			if _, fromConfig := st.installed[name]; fromConfig {
				continue // config wins over the previous process's on-disk copy
			}
		}

		// A sidecar that exists but does not parse is a HARD ERROR (see
		// readWASMSidecar), and so is one whose Blob is absent. Falling back to the
		// zero wasmMeta would re-register the op as OpReadOnly, shardless, with no
		// Epoch, no proven groups and Source "" — which drops it out of
		// publishWASMGateLocked (the route gate goes OFF for it) and out of
		// replicatedRegs (it is omitted from every snapshot this node takes, so a
		// replica joining from one halts on the first invocation). None of that
		// surfaces: ExportName "" is not an error either, since the backend defaults
		// it to "apply", so a typical module LOADS and simply behaves wrongly
		// forever.
		meta, have, err := readWASMSidecar(dataDir, name)
		if err != nil {
			return err
		}
		if !have {
			// ReadDir just listed it; a disappearance between the two is a real I/O
			// anomaly, not a normal state.
			return fmt.Errorf("cluster: reloadWASMModulesFromDisk: sidecar %s vanished between listing and read",
				filepath.Join(wasmDir, entry.Name()))
		}

		blobFP, err := parseWASMBlobRef(meta.Blob)
		if err != nil {
			return fmt.Errorf("cluster: reloadWASMModulesFromDisk: op %q: %w", name, err)
		}

		ke := ops.WASMKeyExtractor()
		kind := ops.OpKind(meta.Kind)
		// A MISSING OR UNUSABLE BLOB NO LONGER FAILS NODE CONSTRUCTION, and that
		// reversal is the restart-time face of thin markers. It used to be a hard
		// error because a sidecar was the only record of an op and a blob was
		// guaranteed to be beside it: the registration entry had carried the bytes,
		// so a node that had applied it had them. That guarantee is gone — a marker
		// names its module — so "the sidecar is here and the blob is not" is now an
		// ORDINARY, EXPECTED state: this node was unreachable during the push, or was
		// restarted between applying the marker and finishing the fetch.
		//
		// Refusing to start on it would be the worst available answer. It converts a
		// self-healing condition (fetch the blob) into an outage of every group this
		// node hosts, on a node that is otherwise completely healthy, and it does so
		// at exactly the moment an operator restarted the node hoping to fix
		// something. So the op is registered, its bindings are restored, and the
		// bytes are fetched in the background; an invocation that arrives first
		// blocks (classRetry) and says so, which is bounded to the ops that actually
		// need the missing module.
		id := wasm.ModuleIDForBlob(blobFP, meta.ExportName, meta.MaxFuel)
		if err := materializeWASMBlob(dataDir, rt, blobFP, meta.ExportName, meta.MaxFuel); err != nil {
			slog.Warn("wasm module bytes are not usable at startup; the op is registered and its bytes will be fetched, and invocations of it block until they arrive",
				"component", "cluster", "op", name, "blob", meta.Blob, "err", err)
			prefetchWASMBlob(prefetch, blobFP)
		}
		// Plain Register, NOT the replacing variant: names skipped above are the
		// ones this process already installed from config, so an ErrDuplicateOp
		// here means some OTHER subsystem owns the name (a builtin, say). A stale
		// get.json left in the data dir must not take over "get".
		if err := wasm.RegisterModule(reg, rt, name, id, kind, ke); err != nil {
			if errors.Is(err, ops.ErrDuplicateOp) {
				continue
			}
			return fmt.Errorf("cluster: register reloaded %s: %w", name, err)
		}
		if st != nil {
			// Restore the PER-GROUP BINDING TABLE the previous process recorded. This
			// is the ONLY way it survives a durable restart: fsm.Apply skips every
			// entry at or below the persisted applied index, so the
			// __register_wasm__ entries that established these bindings are never
			// replayed. Losing them would shut the route gate on every group AND
			// leave any committed invocation still to replay out of a group's log
			// with no version to resolve (ops.ErrWASMNoGroupBinding, classFatal).
			//
			// Each binding's module is loaded from ITS OWN blob, because a group may
			// be bound to a version other than the node-wide install. The common
			// case — every group on the node-wide version — costs no extra compile:
			// rt.HasModule short-circuits it.
			//
			// It also cannot deadlock the gate: nothing here WAITS for a group. A
			// group bound before the restart is bound again immediately; a group not
			// in the table becomes bound the moment it applies or restores a
			// registration, and if it never does, that is the correct answer (its
			// log genuinely has no registration) and shows up as a client-visible
			// error, not a stall.
			groups, err := loadWASMBindings(dataDir, rt, name, meta, prefetch)
			if err != nil {
				return err
			}
			st.installed[name] = installedWASM{
				reg: ops.WASMRegistration{
					Name:       name,
					Kind:       kind,
					Blob:       blobFP,
					ExportName: meta.ExportName,
					MaxFuel:    meta.MaxFuel,
					Epoch:      meta.Epoch,
				},
				// A sidecar with no recorded source was written by the config path;
				// treating it as non-replicated is the safe default — it only means a
				// replicated registration for that name overrides it unconditionally.
				replicated: meta.Source == wasmSourceReplicated,
				groups:     groups,
			}
		}
	}
	return nil
}

// wasmPrefetchFn is how the install paths ask for a module's bytes without
// waiting for them. It is ALWAYS fire-and-forget — see Node.prefetchWASMBlob —
// because every one of its callers holds wasmApplyMu, which cluster's
// snapshotWASMState and restoreWASMState also take: blocking here would stall
// every group's snapshot on this node behind one group's missing module.
//
// nil is fine and means "nothing to ask" — a bare wasmState in a test, or a node
// with no peers to ask.
type wasmPrefetchFn func(fp [sha256.Size]byte)

// prefetchWASMBlob is the nil-safe call.
func prefetchWASMBlob(fn wasmPrefetchFn, fp [sha256.Size]byte) {
	if fn != nil {
		fn(fp)
	}
}

// parseWASMBlobRef converts a sidecar's hex blob reference to a fingerprint,
// routing it through wasmBlobPath so the ONE traversal/format gate applies to
// this input like it does to every other fingerprint→path join.
//
// The path itself is discarded: this is a format check, and the traversal
// argument in wasmBlobPath is precisely that the guarantee must be attached to
// the INPUT rather than to whichever call site happens to be safe.
func parseWASMBlobRef(fp string) ([sha256.Size]byte, error) {
	var out [sha256.Size]byte
	if _, err := wasmBlobPath("", fp); err != nil {
		return out, err
	}
	raw, err := hex.DecodeString(fp)
	if err != nil {
		return out, fmt.Errorf("%w: %q is not a sha256 hex fingerprint", errWASMBlob, fp)
	}
	copy(out[:], raw)
	return out, nil
}

// materializeWASMBlob makes one module version EXECUTABLE on this node: read the
// blob (which verifies it against its own name), compile it, instantiate it under
// the ModuleID the marker derives.
//
// ############ ITS FAILURES ARE NOT REGISTRATION FAILURES ############
//
// Every caller treats an error from here as "not resident yet", never as a reason
// to refuse an install, and that is the single most important consequence of thin
// markers. The reason is DETERMINISM ACROSS REPLICAS, and it is easy to get
// backwards:
//
//	A marker is a pure function of the log. Whether THIS node holds the bytes is
//	not. So any apply-time decision that consults the bytes makes the outcome
//	depend on residency, and two replicas of one group would then disagree about
//	whether the op is installed at all.
//
// Concretely, with the old eager behaviour carried forward: node A has the bytes,
// they fail to compile on its wasmtime, so A refuses the registration and does not
// register the op. Node B has not fetched them, so B installs the marker, opens
// its route gate, and proposes an invocation into the group. A applies that
// invocation, cannot look the op up, and gets shard.ErrOpNotRegistered — which is
// classFatal. One node's compile disagreement would HALT a different node. The
// same argument applies verbatim to the OpReadOnly/writes-state guard, which is
// why that guard moved to wasm.Runtime.resolveModuleForInvoke, where it is asked
// once per invocation on every node and therefore reaches the same verdict on all
// of them.
//
// So the line this file now draws without exception is: an apply-time REFUSAL
// must be a pure function of the entry; anything that needs the bytes is a
// RESIDENCY condition, reported and retried rather than decided.
//
// The push's compile-verify is what keeps that from being a silent trap: a module
// that does not compile is refused at REGISTRATION time by every member that
// answers (cluster.storeWASMBlobVerified), so the residue is confined to members
// the push could not reach — and those are named in the push report, counted in
// WASMBlobPushStats.Skips, and, if they later block on it, named in
// WASMBlockStats with this error as the reason.
func materializeWASMBlob(dataDir string, rt *wasm.Runtime, fp [sha256.Size]byte, exportName string, maxFuel uint64) error {
	id := wasm.ModuleIDForBlob(fp, exportName, maxFuel)
	if rt == nil || rt.HasModule(id) {
		return nil
	}
	b, err := readWASMBlob(dataDir, ops.WASMBlobHex(fp))
	if err != nil {
		return err
	}
	m, err := wasm.Compile(b)
	if err != nil {
		return fmt.Errorf("cluster: compile module %s: %w", ops.WASMBlobHex(fp), err)
	}
	if _, err := rt.AddModule(m, exportName, maxFuel); err != nil {
		_ = m.Close() //nolint:errcheck,gosec
		return fmt.Errorf("cluster: add module %s: %w", ops.WASMBlobHex(fp), err)
	}
	return nil
}

// loadWASMBindings rebuilds one op's per-group binding table from its sidecar and
// makes each bound version executable when its blob is present.
//
// It considers every distinct blob the bindings name, not just the node-wide one,
// because a group's bound version may be older than the node-wide install and the
// group's applies must still be able to execute it.
//
// A BINDING WHOSE MODULE CANNOT BE LOADED NO LONGER FAILS NODE CONSTRUCTION. The
// binding itself is derived from the sidecar alone (newWASMGroupBinding needs no
// bytes), so it is restored exactly as before — the route gate stays open, the
// group stays bound, and every committed invocation still resolves to the right
// version. The only thing missing is the module, which is a residency condition:
// the bytes are fetched, and an invocation that gets there first blocks and says
// so. See materializeWASMBlob for why refusing instead would be unsafe as well as
// unhelpful.
func loadWASMBindings(dataDir string, rt *wasm.Runtime, name string, meta wasmMeta, prefetch wasmPrefetchFn) (map[int]wasmGroupBinding, error) {
	if len(meta.Bindings) == 0 {
		return nil, nil
	}
	out := make(map[int]wasmGroupBinding, len(meta.Bindings))
	for _, mb := range meta.Bindings {
		if _, dup := out[mb.Group]; dup {
			return nil, fmt.Errorf("cluster: reloadWASMModulesFromDisk: op %q: sidecar binds shard group %d twice", name, mb.Group)
		}
		fp, err := parseWASMBlobRef(mb.Blob)
		if err != nil {
			return nil, fmt.Errorf("cluster: reloadWASMModulesFromDisk: op %q shard group %d: %w", name, mb.Group, err)
		}
		reg := ops.WASMRegistration{
			Name:       name,
			Kind:       mb.Kind,
			Blob:       fp,
			ExportName: mb.ExportName,
			MaxFuel:    mb.MaxFuel,
			Epoch:      mb.Epoch,
		}
		if err := materializeWASMBlob(dataDir, rt, fp, mb.ExportName, mb.MaxFuel); err != nil {
			slog.Warn("wasm binding's module bytes are not usable at startup; the binding is restored and its bytes will be fetched, and invocations in this group block until they arrive",
				"component", "cluster", "op", name, "group", mb.Group, "blob", mb.Blob, "err", err)
			prefetchWASMBlob(prefetch, fp)
		}
		out[mb.Group] = newWASMGroupBinding(reg)
	}
	return out, nil
}

// wasmSnapshotSections is a decoded snapshot WASM section: the node-wide
// registrations to INSTALL, and the SNAPSHOTTED GROUP's per-group bindings.
type wasmSnapshotSections struct {
	// installs is every replicated registration the snapshotting node held. It is
	// a SUPERSET on purpose (see wasmSnapshotBlob) and carries no group claim.
	installs []ops.WASMRegistration
	// bindings is EXACTLY the snapshotting node's binding table for the
	// snapshotted group: one registration per bound op name, being the version
	// that group's log had committed at the snapshot point.
	bindings []ops.WASMRegistration
}

// wasmSnapshotBlobVersion is the leading byte of the encoding below.
//
// Version 1 was [count u32]{[inGroup u8][len u32][reg]}* — one record per
// replicated registration with a boolean "this group's log carries it". That
// cannot express per-group binding, because the version a group is bound to may
// DIFFER from the node-wide installed one, and version 1 had only the node-wide
// registration to attach the flag to. The format break is unmigrated: a v1 blob's
// first byte is the high byte of a small count (0), which is not 2, so it is
// refused loudly rather than misread. The app is unreleased.
//
// Version 3 changed no framing at all — the section layout was byte-for-byte
// version 2's. What changed was the RECORDS: ops.EncodeWASMRegistration replaced
// the inline [bytes_len u32][bytes] pair with a fixed 32-byte content address, so
// a v2 blob's records decode to different registrations rather than failing. That
// is precisely why the version byte had to move: this is the one format break in
// the set that a reader could not otherwise detect, and reading a v2 snapshot as
// v3 would install ops whose module fingerprints are the first 32 bytes of a WASM
// binary.
//
// VERSION 4 IS THE SAME SHAPE OF BREAK, one field later. The record lost its
// [handle_len u16][handle] pair when WASMRegistration lost KeyExtractorHandle
// (see ops.WASMKeyExtractorHandle), and the encoding is POSITIONAL: a v3 record
// read as v4 takes the handle's length prefix as the first two bytes of MaxFuel
// and every later field slides. It decodes — to nonsense — rather than failing,
// which is exactly the case the comment above says must move the version byte.
// Same unmigrated posture, same reason (the app is unreleased); a v3 blob is
// refused loudly.
//
// A SNAPSHOT SECTION IS NOW ~40 BYTES PER REGISTRATION instead of ~40 bytes plus
// the module, and it is carried in EVERY group's snapshot. That, and the matching
// shrink of every group's Raft log, is the entire point of thin markers.
const wasmSnapshotBlobVersion = 4

// wasmSnapshotBlob encodes st's WASM state for group groupIdx's shard snapshot:
//
//	[version u8]
//	[installs u32]{[len u32][ops.EncodeWASMRegistration]}*
//	[bindings u32]{[len u32][ops.EncodeWASMRegistration]}*
//
// THE TWO SECTIONS ANSWER TWO DIFFERENT QUESTIONS and their exactness
// requirements are opposite, which is why they are separate lists rather than
// one list with a flag.
//
// INSTALLS IS A SUPERSET, deliberately: EVERY replicated registration is carried
// regardless of groupIdx. A replica brought up by this snapshot must be able to
// LOOK UP every op that groupIdx's remaining log may invoke, and withholding one
// would hand it the classFatal shard.ErrOpNotRegistered halt that snapshot
// carriage exists to prevent. Carrying a superset can never cause a halt;
// carrying a subset can.
//
// BINDINGS IS EXACT: exactly the ops bound in groupIdx, each at exactly the
// version groupIdx's log committed. It is both route-gate evidence and the
// executing version, and both directions of inexactness are bugs. Too WIDE lets
// the restorer propose invocations into a group whose log never got the
// registration (the failure the route gate exists to close) and would execute
// entries with a version that group's log never named. Too NARROW leaves a
// committed invocation in groupIdx's log with no version to resolve, which is
// ops.ErrWASMNoGroupBinding — classFatal.
//
// A binding's registration may differ from the same op's entry in installs, and
// that is the point: the restorer installs the binding's module too (it is
// applied through applyWASMRegistration like any other registration, so its blob
// is written and its module added), then binds it to groupIdx while the node-wide
// install stays the (Epoch, fingerprint) maximum.
//
// Returns nil when there is nothing to carry, so the snapshot omits the section.
func wasmSnapshotBlob(st *wasmState, groupIdx int) []byte {
	installs := st.replicatedRegs()
	if len(installs) == 0 {
		return nil
	}
	var buf bytes.Buffer
	buf.WriteByte(wasmSnapshotBlobVersion)
	writeRegs := func(regs []ops.WASMRegistration) {
		var lenBuf [4]byte
		binary.BigEndian.PutUint32(lenBuf[:], uint32(len(regs))) //nolint:gosec // bounded by the registered op count
		buf.Write(lenBuf[:])
		for _, r := range regs {
			enc := ops.EncodeWASMRegistration(r)
			binary.BigEndian.PutUint32(lenBuf[:], uint32(len(enc))) //nolint:gosec // bounded by maxDynamicWASMBytes
			buf.Write(lenBuf[:])
			buf.Write(enc)
		}
	}
	nodeWide := make([]ops.WASMRegistration, 0, len(installs))
	bound := make([]ops.WASMRegistration, 0, len(installs))
	for _, in := range installs {
		nodeWide = append(nodeWide, in.reg)
		if b, ok := in.groups[groupIdx]; ok {
			bound = append(bound, b.reg)
		}
	}
	writeRegs(nodeWide)
	writeRegs(bound)
	return buf.Bytes()
}

// decodeWASMSnapshotBlob parses what wasmSnapshotBlob produced.
func decodeWASMSnapshotBlob(b []byte) (wasmSnapshotSections, error) {
	var out wasmSnapshotSections
	if len(b) < 1 {
		return out, fmt.Errorf("cluster: wasm snapshot blob too short (%d bytes)", len(b))
	}
	if b[0] != wasmSnapshotBlobVersion {
		return out, fmt.Errorf("cluster: wasm snapshot blob version %d is not %d (a snapshot written before the key extractor was pinned cannot be read by this build)",
			b[0], wasmSnapshotBlobVersion)
	}
	off := 1
	readRegs := func(section string) ([]ops.WASMRegistration, error) {
		if off+4 > len(b) {
			return nil, fmt.Errorf("cluster: wasm snapshot blob truncated at %s count", section)
		}
		count := int(binary.BigEndian.Uint32(b[off : off+4]))
		off += 4
		// Clamp the PRE-ALLOCATION against the buffer that actually has to supply
		// the records. count is an untrusted u32, and each record needs at least 4
		// header bytes, so a corrupt or hostile length would otherwise have us
		// reserve up to a billion structs before reading the first one. The loop
		// still validates every record; this only bounds the allocation.
		if maxRecords := (len(b) - off) / 4; count > maxRecords {
			return nil, fmt.Errorf("cluster: wasm snapshot blob %s claims %d records but only %d can fit in %d bytes", section, count, maxRecords, len(b))
		}
		regs := make([]ops.WASMRegistration, 0, count)
		for i := 0; i < count; i++ {
			if off+4 > len(b) {
				return nil, fmt.Errorf("cluster: wasm snapshot blob truncated at %s record %d header", section, i)
			}
			n := int(binary.BigEndian.Uint32(b[off : off+4]))
			off += 4
			if n < 0 || off+n > len(b) {
				return nil, fmt.Errorf("cluster: wasm snapshot blob truncated at %s record %d body", section, i)
			}
			r, err := ops.DecodeWASMRegistration(b[off : off+n])
			if err != nil {
				return nil, fmt.Errorf("cluster: wasm snapshot blob %s record %d: %w", section, i, err)
			}
			// n is an untrusted u32 and the decoder does not assert it CONSUMED its
			// input, so a record whose declared length overstates its encoding decodes
			// cleanly and `off += n` swallows the difference — the blob-level trailing
			// check below still balances and the padding is accepted. Requiring the
			// record to be exactly the canonical encoding of what it decoded to is the
			// only thing that sees it. Defence in depth: the only writer of this blob
			// is wasmSnapshotBlob and the only reader path is a peer's InstallSnapshot,
			// so today the input is trusted.
			if enc := len(ops.EncodeWASMRegistration(r)); enc != n {
				return nil, fmt.Errorf("cluster: wasm snapshot blob %s record %d: declared %d bytes but the registration encodes to %d", section, i, n, enc)
			}
			off += n
			regs = append(regs, r)
		}
		return regs, nil
	}
	var err error
	if out.installs, err = readRegs("installs"); err != nil {
		return wasmSnapshotSections{}, err
	}
	if out.bindings, err = readRegs("bindings"); err != nil {
		return wasmSnapshotSections{}, err
	}
	if off != len(b) {
		return wasmSnapshotSections{}, fmt.Errorf("cluster: wasm snapshot blob has %d trailing bytes", len(b)-off)
	}
	return out, nil
}

// snapshotWASMState is the shard.Config.WASMSnapshot hook for group groupIdx.
// hashicorp/raft calls it on the shard's FSM goroutine, which is a DIFFERENT
// goroutine per group and concurrent with other groups' applies, so it takes the
// same mutex those applies hold.
func (n *Node) snapshotWASMState(groupIdx int) []byte {
	n.wasmApplyMu.Lock()
	defer n.wasmApplyMu.Unlock()
	return wasmSnapshotBlob(n.wasmState, groupIdx)
}

// restoreWASMState is the shard.Config.WASMRestore hook for group groupIdx: it
// installs the registrations a snapshot carried, through the SAME
// applyWASMRegistration path (and therefore the same epoch comparison) that a
// log entry would take. That is what keeps a snapshot-installed replica from
// disagreeing with a log-replaying one: neither can clobber a strictly newer
// registration, whichever arrives first, and the disk reload on the next restart
// reads back exactly what the winner wrote.
//
// groupIdx is also the route-gate provenance, but ONLY for the records the
// snapshot flagged as carried by that group (see wasmSnapshotBlob): installing a
// module is unconditional, attributing it to a group's log is not.
//
// Before the node's WASM runtime exists there is nowhere to install to, so the
// blob is buffered (see Node.wasmRestorePending) and drained by finishWASMSetup.
func (n *Node) restoreWASMState(groupIdx int, b []byte) error {
	n.wasmApplyMu.Lock()
	defer n.wasmApplyMu.Unlock()
	if n.wasmRT == nil {
		cp := make([]byte, len(b))
		copy(cp, b)
		n.wasmRestorePending = append(n.wasmRestorePending, pendingWASMRestore{group: groupIdx, blob: cp})
		return nil
	}
	return n.installWASMSnapshotBlobLocked(groupIdx, b)
}

// installWASMSnapshotBlobLocked decodes and applies one snapshot blob taken from
// group groupIdx. Callers hold wasmApplyMu.
//
// The INSTALLS section is applied with NO group provenance: it is a superset of
// what groupIdx's log carries (see wasmSnapshotBlob), so attributing it to
// groupIdx would inflate both the route-gate evidence and the binding table with
// claims about a log that may never have carried the registration.
//
// The BINDINGS section is applied WITH groupIdx, and it is applied SECOND so
// that whichever of the two carries the newer registration for a name, the fold
// lands on the same result. The two sections commute HERE for a narrow and
// checkable reason rather than "the fold is a maximum": the installs section
// carries ops.NoShardIndex, so it cannot touch a binding or reach the contract
// check at all, and the bindings section holds at most ONE record per name. Only
// the node-wide fold sees both, and that one IS a bare maximum. Applying
// bindings last additionally keeps the common case (the two sections carry the
// same registration) on the cheap no-op path.
//
// A BINDING RECORD CAN LAND ON A GROUP THIS REPLICA IS ALREADY BOUND IN, which
// is the one place the contract check runs on a restore. It passes: both
// bindings are folds over prefixes of the SAME log (groupIdx's), which share a
// first registration, so their contracts are equal. See the catch-up argument in
// installedWASM.groups.
func (n *Node) installWASMSnapshotBlobLocked(groupIdx int, b []byte) error {
	secs, err := decodeWASMSnapshotBlob(b)
	if err != nil {
		return err
	}
	for _, r := range secs.installs {
		if err := applyWASMRegistration(n.cfg.DataDir, n.wasmRT, n.cfg.Ops, n.wasmState, r, ops.NoShardIndex, n.prefetchWASMBlob); err != nil {
			return fmt.Errorf("cluster: restore wasm registration %q: %w", r.Name, err)
		}
	}
	for _, r := range secs.bindings {
		if err := applyWASMRegistration(n.cfg.DataDir, n.wasmRT, n.cfg.Ops, n.wasmState, r, groupIdx, n.prefetchWASMBlob); err != nil {
			return fmt.Errorf("cluster: restore wasm binding %q for shard group %d: %w", r.Name, groupIdx, err)
		}
	}
	n.publishWASMGateLocked()
	return nil
}

// setWASMRuntime publishes the node's WASM runtime. It is a separate,
// mutex-taking setter rather than a plain field assignment because the shard FSM
// apply loops are already running by the time the runtime is built (shard.New
// starts them), and restoreWASMState reads the field from those goroutines.
func (n *Node) setWASMRuntime(rt *wasm.Runtime) {
	n.wasmApplyMu.Lock()
	defer n.wasmApplyMu.Unlock()
	n.wasmRT = rt
}

// finishWASMSetup completes node-construction WASM wiring: it reloads the
// modules a previous process registered from disk, then installs any snapshot
// blobs that arrived while the runtime was still being built. The epoch
// comparison makes the order between those two (and the config modules loaded
// before them) irrelevant to the final state.
//
// It holds wasmApplyMu for the WHOLE sequence. That is load-bearing: the shard
// FSM apply loops are ALREADY RUNNING at this point, and applyWASMRegistration
// touches the same files, the same wasm.Runtime and the same ops.Registry. An
// unlocked reload could interleave with a committed __register_wasm__ entry
// applying concurrently and fail node construction on a torn read.
func (n *Node) finishWASMSetup(dataDir string) error {
	n.wasmApplyMu.Lock()
	defer n.wasmApplyMu.Unlock()
	if err := reloadWASMModulesFromDisk(dataDir, n.wasmRT, n.cfg.Ops, n.wasmState, n.prefetchWASMBlob); err != nil {
		return err
	}
	pending := n.wasmRestorePending
	n.wasmRestorePending = nil
	for _, p := range pending {
		if err := n.installWASMSnapshotBlobLocked(p.group, p.blob); err != nil {
			return err
		}
	}
	n.publishWASMGateLocked()
	return nil
}

// applyWASMRegistration persists a WASM module and its metadata to disk,
// compiles it into rt, and registers its op in reg. Called by the
// __register_wasm__ FSM hook on every node that applies the log entry, and by
// the snapshot-restore path for a replica that received the registration as
// snapshot state rather than as a log entry.
//
// fromGroup is the shard group whose log carried the entry (ops.NoShardIndex
// when there is no group provenance — a direct handler call in a test, or a
// snapshot's install section, which is a superset of what the snapshotted group
// carries). It is what the entry is BOUND to in installedWASM.groups and
// persisted as in the sidecar; it is both the route gate's evidence and the
// executing version. The NODE-WIDE install is not conditional on it: the module
// is compiled, written and registered regardless, because registry presence is
// what the FSM needs to apply an entry at all and gating THAT would manufacture
// the very halt this machinery exists to prevent.
//
// TWO FOLDS RUN HERE, over the same total order and for different consumers.
//
//	PER GROUP (installedWASM.groups, via bindWith): the binding for fromGroup
//	becomes the (Epoch, fingerprint) MAXIMUM over the registrations of this name
//	in fromGroup's log prefix. This is the one that decides what EXECUTES: it is a
//	pure function of that group's prefix, so every replica of the group derives
//	the same version for the same entry.
//
//	NODE-WIDE (installedWASM.reg + the ops.Registry entry): the maximum over
//	everything this node has applied from ANY group. This decides Kind, the key
//	extractor and the routing the PROPOSE side reads, and it is the version used by
//	calls with no group to bind against (read-only ops, config modules, Direct
//	mode) — see wasm.Runtime.resolveModuleForInvoke.
//
// THE TWO FOLDS DO NOT HAVE THE SAME PROPERTY, and conflating them is the
// easiest mistake available in this file.
//
// The NODE-WIDE fold is a BARE MAXIMUM: commutative, associative, and therefore
// order-independent over the SET of registrations this node received. That is
// what settles a race between two first-time registrations of one name on any
// node that received both, and it is what lets a snapshot and a log replay
// deliver two registrations for one name in either order. What it does NOT
// settle is two nodes receiving DIFFERENT sets, which the forwarded-leg gate
// makes possible on purpose — see cluster/wasm_gate.go's "STATE THE REST OF THAT
// COST" and the WHAT REMAINS paragraph in shard/apply_class.go.
//
// The PER-GROUP fold is that maximum COMPOSED WITH the contract freeze below,
// and the composition is order-DEPENDENT: the freeze compares against
// fromGroup's EXISTING binding, so it cannot fire on the first registration in
// that group's log, and that first registration therefore PINS the group's
// contract. The per-group binding is a pure function of the group's ORDERED log
// prefix — which is exactly what safety requires, since every replica of a group
// applies one identical sequence — and it is NOT a function of the unordered
// set. installedWASM.groups carries the lemma, a worked counterexample, and the
// argument that every catch-up route (log replay, snapshot, snapshot onto an
// existing binding) preserves it.
//
// AN IN-PLACE BYTES UPDATE IS NOW SAFE and is no longer refused. It is the
// per-group fold that made it so: a node that has applied the update on group 3
// and not on group 7 executes group 7's entries with group 7's committed version,
// exactly as a node that has applied neither does. What is still refused is a
// registration that changes Kind — below, per group, and at
// propose time (checkWASMUpdateGate).
//
// THE PER-GROUP REFUSAL IS SAFE AT APPLY TIME, and this is the biggest single
// consequence of the design. "WHY NOT reject a re-registration if the name
// already exists" below rejects a NODE-WIDE state-dependent refusal, and that
// argument is still correct for node-wide state: node A holds the op and refuses,
// node B does not and accepts, and the two diverge. It stops holding for
// PER-GROUP state. A refusal keyed on group g's own binding is a pure function of
// g's log prefix, so every replica of g — which by definition has applied the same
// prefix — reaches the SAME verdict on the SAME entry. It is as deterministic as
// the Kind range check, and it is therefore classAdvance like one.
//
// WHY NOT "reject a re-registration if the name already exists". That rule is
// not order-independent at all — it is state-dependent. A replica that has
// already installed the op REJECTS the second registration while a replica that
// has not yet installed anything ACCEPTS it. The two then hold different
// modules under the same name, which is precisely the divergence being fixed.
//
// A CONFIG-installed module (Source=config) never participates in the
// comparison: it is node-local, so letting it win would make the outcome depend
// on which node happened to have it configured. A replicated registration always
// overrides it.
//
// IDEMPOTENT BY CONTRACT. On a node hosting k groups this runs k times for one
// client-visible registration — plus again on any client retry. The comparison
// makes the repeats cheap no-ops (an identical registration is not strictly
// newer than itself), which also skips the wasm.Compile Cranelift pass that the
// old rt.HasModule fast path existed to avoid. The state check is strictly
// stronger than that fast path was: HasModule fingerprints only
// (bytes, export, fuel), so it could not see a change in Kind — the field that
// decides whether the op bypasses Raft. Gating the sidecar rewrite on it let two
// nodes end up with DIFFERENT sidecars for the same module and, after a restart,
// register the op with a different contract.
//
// Concurrency: the k per-group FSM apply loops are independent goroutines, so
// the caller (the hook installed in the Node constructors) serializes them on
// Node.wasmApplyMu, which also guards st. Without that the file writes and the
// load/register pair would interleave across groups.
func applyWASMRegistration(dataDir string, rt *wasm.Runtime, reg *ops.Registry, st *wasmState, r ops.WASMRegistration, fromGroup int, prefetch wasmPrefetchFn) error {
	// r.Name becomes two file paths below, so it is validated BEFORE anything
	// looks at it. This is not redundant with the check in
	// ops.RegisterWASMRegisterOp: the snapshot-restore path
	// (installWASMSnapshotBlobLocked) reaches this function WITHOUT going through
	// that op at all, so a registration decoded out of a peer's snapshot — or out
	// of an object-store backup — would otherwise arrive here unvalidated. Being a
	// pure function of the entry, the refusal is identical on every replica.
	if err := ops.ValidateWASMOpName(r.Name); err != nil {
		return fmt.Errorf("cluster: register op: %w", err)
	}
	cur, mine := st.installed[r.Name]

	groups, rebound := cur.bindWith(fromGroup, r)

	// THE PER-GROUP CONTRACT FREEZE, decided from the entry and from fromGroup's
	// OWN binding and nothing else — so every replica of fromGroup, which has
	// applied the same prefix, refuses or accepts identically. See the
	// "THE PER-GROUP REFUSAL IS SAFE AT APPLY TIME" paragraph above and
	// checkWASMGroupContract.
	//
	// IT SCOPES TO THE BINDING, NOT TO THE WHOLE APPLY, and that scoping is
	// load-bearing rather than tidy. Aborting the apply here would take the
	// NODE-WIDE fold down with it, and the node-wide fold must stay a function of
	// the SET of registrations a node received — never of which shard groups it
	// happens to host. Concretely: registrations A and B of one name race, with
	// different key extractors; group 0's log orders them A,B and group 7's orders
	// them B,A. A node hosting only group 0 would refuse B and keep A, a node
	// hosting only group 7 would refuse A and keep B, and the two would then ROUTE
	// the op's invocations to different shard groups forever — divergence
	// manufactured by the check meant to prevent it, and precisely the
	// order-independence that ops.WASMRegistrationNewer exists to guarantee.
	//
	// So the group keeps the binding it has, the node-wide fold runs untouched, and
	// the refusal is REPORTED (every replica of fromGroup reports it identically,
	// on the same entry, so classAdvance stays correct). The client learns that the
	// contract change did not take in that group; it does not silently take.
	var groupRefusal error
	if rebound {
		if prev, had := cur.groups[fromGroup]; had {
			if err := checkWASMGroupContract(prev, r, fromGroup); err != nil {
				groupRefusal, groups, rebound = err, cur.groups, false
			}
		}
	}

	// installNodeWide is the node-wide fold's verdict. A CONFIG-installed module
	// (Source=config) never participates: it is node-local, so letting it win would
	// make the outcome depend on which node happened to have it configured, and a
	// replicated registration therefore always overrides it.
	installNodeWide := !mine || !cur.replicated || ops.WASMRegistrationNewer(r, cur.reg)
	if !installNodeWide && !rebound {
		// Nothing references this registration: it lost the node-wide fold AND it
		// did not become fromGroup's binding. This is the idempotent-repeat path —
		// on a node hosting k groups the hook runs k times for one client-visible
		// registration, plus again on every client retry — so it must stay a cheap
		// no-op with no compile, no write and no registry touch.
		return groupRefusal
	}
	if !mine {
		// The name is not one this node installed as a WASM module, so a registry
		// entry under it belongs to someone else — a builtin, or another
		// subsystem's op. The install below REPLACES the registry entry (it has
		// to: a winning registration may change Kind or the key extractor), so
		// without this guard an admin-gated registration named "get" would shadow
		// the builtin cluster-wide. Refusing is a deterministic, entry-derived
		// error, so every replica refuses identically and no divergence follows.
		if _, _, _, exists := reg.Lookup(r.Name); exists {
			return fmt.Errorf("cluster: register op %q: name is already an existing op and cannot be shadowed by a WASM module", r.Name)
		}
	}
	// EVERY ENTRY-DERIVED REJECTION RUNS BEFORE ANY SIDE EFFECT. That is the
	// guarantee — not "everything that can reject this registration", which is not
	// achievable here: rt.AddModule and the registry install below can still fail
	// on node-local state. What IS guaranteed is that nothing decidable from the
	// entry alone — the name, the Kind byte, the module bytes' compilability, and
	// the OpReadOnly/writes-state pairing of the two — can fail after a file has
	// been written.
	//
	// It has to be that way because a failure AFTER the writes is not a failure at
	// all in practice: both writes succeed, the sidecar-failure unlink below never
	// fires, classifyApplyErr treats the returned error as classAdvance (no halt, no
	// metric), and the files persist — so the next restart's
	// reloadWASMModulesFromDisk finds the pair, compiles it, and re-derives the same
	// rejection out of ops.Registry.Replace, which is NOT ErrDuplicateOp and so
	// FAILS NODE CONSTRUCTION on every node at once. One wire-controlled
	// registration, every node bricked at its next start, recoverable only by
	// deleting the files by hand.
	//
	// That is why the Kind byte is range-checked here rather than being left to
	// validateEntry (27 lines below, past both writes), and why the two registry
	// NAME rules — length and the protocol-v2 length-2 collision — moved into
	// ops.ValidateWASMOpName above. wasm.ValidateModuleKind does NOT cover the
	// range: it short-circuits on `kind != ops.OpReadOnly`, so every out-of-range
	// value sails through it.
	//
	// The previous order was worse still: it wrote the module and the sidecar
	// first, then compiled, then swapped the runtime, and only then ran the
	// read-only guard, so (1) applied to a rejected module as well, and (2) when
	// the guard fired, rt.AddModule had already replaced the runtime module while
	// the registry entry and st still described the previous one — the handler
	// executed the NEW bytes under the OLD Kind and key extractor.
	if err := validateWASMRegistrationKind(r.Name, r.Kind); err != nil {
		return err
	}
	// THE MODULE IS MADE RESOLVABLE WHENEVER ANYTHING WILL REFERENCE IT, which
	// since per-group binding is no longer the same as "it won the node-wide fold".
	// A group may bind a version that is OLDER than this node's node-wide install
	// (its log simply has not seen the newer one), and that group's applies must
	// still be able to execute it — so the module goes into the runtime even when
	// the registry entry is left alone below.
	//
	// ############ AND IT IS BEST-EFFORT NOW, WHICH IT WAS NOT BEFORE ############
	//
	// The marker does not carry the module, so this step depends on whether THIS
	// node holds the blob — which is not a function of the log. A failure here is
	// therefore a RESIDENCY condition, never a refusal: the marker installs, the
	// binding is recorded, the route gate opens, and the bytes are fetched in the
	// background. An invocation that arrives before they do blocks (classRetry) and
	// is reported; it does not halt anything and it does not diverge anything.
	//
	// Refusing instead would manufacture the divergence this file exists to
	// prevent — a node with the bytes could refuse a registration a node without
	// them installs, and the second would then propose invocations the first halts
	// on with classFatal ErrOpNotRegistered. materializeWASMBlob states that
	// argument in full. It is also why the OpReadOnly/writes-state guard no longer
	// runs here: it needs the bytes, so it moved to
	// wasm.Runtime.resolveModuleForInvoke, which asks it once per invocation on
	// EVERY node and so reaches one verdict everywhere.
	id := wasm.ModuleIDForBlob(r.Blob, r.ExportName, r.MaxFuel)
	residencyErr := materializeWASMBlob(dataDir, rt, r.Blob, r.ExportName, r.MaxFuel)

	// The sidecar is written next, so a crash between here and the registry install
	// leaves DISK holding the winning registration — which is what the next restart
	// reloads.
	//
	// THERE IS NO BLOB WRITE HERE ANY MORE. The marker carries no bytes to write,
	// and the blob it names is already durable on every node that has it: it was
	// put there by the pre-registration push (storeWASMBlobVerified), by a fetch, or
	// by an operator's __wasm_blob_put__. The sidecar is now the ONLY artifact this
	// path writes, which strengthens rather than weakens the crash argument the old
	// blob-then-sidecar ordering made: there is no longer a pair of writes that
	// could be interrupted between, and a sidecar naming a blob this node does not
	// hold is a state the reload path is now built to handle (see
	// reloadWASMModulesFromDisk).
	//
	// The sidecar's TOP-LEVEL fields describe the NODE-WIDE install, which is not
	// necessarily r: a registration that only became some group's binding must not
	// rewrite them, or a restart would reload the loser as this node's registry
	// entry.
	nodeReg := r
	if !installNodeWide {
		nodeReg = cur.reg
	}
	if err := writeWASMSidecar(dataDir, nodeReg, groups); err != nil {
		return err
	}
	if residencyErr != nil {
		// Fire-and-forget, and it MUST stay that way: this runs under wasmApplyMu,
		// which snapshotWASMState and restoreWASMState also take, so waiting for the
		// bytes here would stall every group's snapshot on this node.
		prefetchWASMBlob(prefetch, r.Blob)
		slog.Warn("wasm registration installed without its module bytes; they will be fetched, and invocations of this version block until they arrive",
			"component", "cluster", "op", r.Name, "group", fromGroup, "blob", ops.WASMBlobHex(r.Blob), "err", residencyErr)
	}

	if !installNodeWide {
		// r is some group's binding but not this node's registry entry. The runtime
		// holds it (above) and the sidecar records it (above), and the resolver finds
		// it by (op, group) — the registry entry is not consulted for that lookup, so
		// there is deliberately nothing more to do.
		st.installed[r.Name] = installedWASM{reg: cur.reg, replicated: true, groups: groups}
		return groupRefusal
	}
	ke := ops.WASMKeyExtractor()
	// NO RUNTIME ROLLBACK, because there is nothing left to roll back. The
	// runtime is content addressed (wasm.ModuleID), so AddModule ADDED this
	// version alongside whatever was already there rather than displacing it, and
	// the ops.Registry entry — which still names the previous version's ModuleID
	// alongside its Kind and key extractor — is the only thing that decides what
	// executes. A failed Replace therefore leaves this node's IN-MEMORY state
	// exactly as it was: old bytes under the old contract, consistent with st.
	//
	// It is NOT consistent with disk, and an earlier version of this comment
	// wrongly said it was. The sidecar was rewritten with the new registration
	// above, before this call, so on a failed Replace disk describes r while st
	// and the registry describe the previous version. That is the SAME deliberate
	// direction the write ordering already commits to (see the comment there): a
	// crash between the sidecar write and the registry install leaves disk holding
	// the winning registration, and the next restart converges FORWARD onto it.
	// The divergence is with this process's memory, not between replicas, and it
	// closes on restart — it is not a state anything needs to repair.
	//
	// The "runtime holds the new bytes under the old registry contract" hazard,
	// and the recompile-the-previous-module repair it needed, are both gone with
	// the per-name slot.
	if err := wasm.RegisterOrReplaceModule(reg, rt, r.Name, id, r.Kind, ke); err != nil {
		return fmt.Errorf("cluster: register op %q: %w", r.Name, err)
	}
	st.installed[r.Name] = installedWASM{reg: r, replicated: true, groups: groups}
	// groupRefusal, when set, means the node-wide fold accepted this registration
	// while fromGroup's binding refused to move to it (a contract change). Both
	// outcomes are deterministic on every replica of fromGroup, so reporting the
	// refusal after the node-wide install is consistent rather than half-done: the
	// state is what every replica of that group holds, and the caller is told the
	// contract change did not take there.
	return groupRefusal
}

// checkWASMGroupContract enforces the FROZEN half of a registration against one
// shard group's own binding: an update may change the module BYTES (and the
// export name and fuel cap that ride with them), and may change NOTHING ELSE.
//
// WHY KIND CANNOT BE PER-GROUP, which is what makes freezing it the only option
// rather than a simplification. It is read on the PROPOSE side, before any group
// is known: Kind decides whether there is a replicated proposal at all (an
// ops.OpReadOnly op is served locally and never enters a log). So a per-group Kind
// would have to be looked up before the group is known, and the group cannot be
// known before it is looked up. Only the module bytes are consumed exclusively at
// APPLY time, where tx.ShardIndex() already names the group — so only the bytes
// can be bound to a group's log prefix.
//
// THE KEY EXTRACTOR USED TO BE THE OTHER HALF OF THIS CONTRACT and is not any
// more: WASMRegistration has no field for it, because it is what COMPUTES the
// group index and a per-node disagreement about it re-routed entries SILENTLY.
// See ops.WASMKeyExtractorHandle.
//
// WHY THIS IS SAFE AT APPLY TIME even though a node-wide equivalent would not be.
// The comparison target is fromGroup's OWN binding, which is a fold over
// fromGroup's log prefix. Every replica of fromGroup that reaches this entry has
// applied that same prefix, so every one of them holds the same `prev` and
// reaches the same verdict on the same entry. The refusal is therefore as
// deterministic as a decode error: classifyApplyErr's classAdvance treatment is
// correct for it, and no replica skips an entry another executed. Contrast the
// NODE-WIDE version of the same idea, which cluster/wasm_gate.go rejects and
// which is genuinely unsafe: a node holding the op refuses while a node that does
// not accepts, and the two then diverge.
//
// It wraps ErrWASMUpdateUnsupported so the message survives the stringifying
// Raft/RPC boundary unredacted — see that sentinel and ops.WASMUpdateUnsupportedMsg.
func checkWASMGroupContract(prev wasmGroupBinding, r ops.WASMRegistration, group int) error {
	if r.Kind == prev.reg.Kind {
		return nil
	}
	return fmt.Errorf("%w: op %q is bound in shard group %d as kind %d, and this registration declares kind %d; only the module bytes may change in place — register the new contract under a NEW op name",
		ErrWASMUpdateUnsupported, r.Name, group, uint8(prev.reg.Kind), uint8(r.Kind))
}

// validateWASMRegistrationKind rejects a Kind byte outside the two values OpKind
// actually defines.
//
// Kind is a wire-controlled u8 (ops.DecodeWASMRegistration reads it straight out
// of the frame), and NOTHING between the wire and the two file writes looked at
// its range. wasm.ValidateModuleKind is not that check — it returns nil
// immediately for anything that is not OpReadOnly, so it passes 2..255 by
// construction. The only range check in the codebase is in ops.validateEntry,
// which applyWASMRegistration does not reach until after both writes; see the
// side-effect-ordering block there for why a rejection at that point bricks every
// node's next startup rather than refusing anything.
//
// Being a pure function of the entry, this is safe at apply time: every replica
// refuses the same bytes identically, so the refusal is deterministic and
// classifyApplyErr's classAdvance treatment stays correct.
func validateWASMRegistrationKind(name string, kind ops.OpKind) error {
	if kind != ops.OpReadOnly && kind != ops.OpReadWrite {
		return fmt.Errorf("%w: register op %q: Kind %d is not a valid OpKind (%d = read-only, %d = read-write)",
			ErrWASMRegistrationRefused, name, uint8(kind), uint8(ops.OpReadOnly), uint8(ops.OpReadWrite))
	}
	return nil
}

// writeWASMSidecar rewrites <dataDir>/wasm/<name>.json for a REPLICATED install.
//
// r describes the NODE-WIDE install; groups is the per-group binding table. The
// two are written together and are allowed to differ — a group may be bound to a
// version this node did not install node-wide — which is why the bindings carry
// their own blob and module parameters (see wasmMeta.Bindings).
//
// The blob address is READ OFF THE REGISTRATION rather than passed alongside it.
// It used to be a separate argument because the caller had just written the blob
// and held the fingerprint the write returned; now the marker carries it, and
// taking it from anywhere else would reintroduce the possibility of a sidecar
// whose metadata and blob reference describe different registrations.
//
// It is called both when a registration installs and when an already-installed
// one merely gains or moves a group binding, because the binding table is durable
// state in its own right and is not re-derivable after a restart.
func writeWASMSidecar(dataDir string, r ops.WASMRegistration, groups map[int]wasmGroupBinding) error {
	meta := wasmMeta{
		Kind:       r.Kind,
		ExportName: r.ExportName,
		MaxFuel:    r.MaxFuel,
		Epoch:      r.Epoch,
		Source:     wasmSourceReplicated,
		Version:    wasmMetaVersion,
		Bindings:   sidecarBindings(groups),
		Blob:       ops.WASMBlobHex(r.Blob),
	}
	mbytes, err := json.Marshal(meta)
	if err != nil {
		return fmt.Errorf("cluster: marshal meta %q: %w", r.Name, err)
	}
	dir := filepath.Join(dataDir, "wasm")
	if err := os.MkdirAll(dir, 0o750); err != nil {
		return fmt.Errorf("cluster: mkdir %s: %w", dir, err)
	}
	return atomicWriteFile(filepath.Join(dir, r.Name+".json"), mbytes)
}

// atomicWriteFile writes b to path so a crash leaves EITHER the previous
// contents or the new ones, never a prefix of the new ones: temp file in the
// same directory, fsync the data, rename over the target, fsync the directory
// so the rename itself is durable.
//
// A bare os.WriteFile truncates in place, and every byte between the truncate
// and the last write is a window in which a crash leaves a SHORT file. That
// matters far more here than the usual "you lose the last write": a truncated
// .wasm sidecar unmarshals to the ZERO wasmMeta, whose every field is the worst
// possible value — Source "" (⇒ not replicated ⇒ the op silently drops out of
// the route-gate snapshot, so the gate stops guarding it), Groups nil (⇒ the
// durable, non-re-derivable route-gate evidence is gone) and Kind 0 (⇒ OpReadOnly,
// which is the CRITICAL-2 apply-time skew). The read side refuses to consume such a file
// (see readWASMSidecar), and this write side makes sure one is never produced:
// on a node hosting k groups the sidecar is rewritten up to k times per
// registration, so the window is not narrow.
func atomicWriteFile(path string, b []byte) error {
	dir := filepath.Dir(path)
	tmp, err := os.CreateTemp(dir, "."+filepath.Base(path)+".tmp-*")
	if err != nil {
		return fmt.Errorf("cluster: create temp for %s: %w", path, err)
	}
	tmpName := tmp.Name()
	// Any failure past this point must not leave the temp file behind.
	defer func() { _ = os.Remove(tmpName) }() //nolint:errcheck // best-effort cleanup; a no-op after a successful rename
	if err := tmp.Chmod(0o600); err != nil {
		_ = tmp.Close() //nolint:errcheck,gosec
		return fmt.Errorf("cluster: chmod temp for %s: %w", path, err)
	}
	if _, err := tmp.Write(b); err != nil {
		_ = tmp.Close() //nolint:errcheck,gosec
		return fmt.Errorf("cluster: write temp for %s: %w", path, err)
	}
	if err := tmp.Sync(); err != nil {
		_ = tmp.Close() //nolint:errcheck,gosec
		return fmt.Errorf("cluster: fsync temp for %s: %w", path, err)
	}
	if err := tmp.Close(); err != nil {
		return fmt.Errorf("cluster: close temp for %s: %w", path, err)
	}
	if err := os.Rename(tmpName, path); err != nil {
		return fmt.Errorf("cluster: rename temp onto %s: %w", path, err)
	}
	// fsync the DIRECTORY: without it the rename can be lost on a crash even
	// though the file data was synced, leaving the old (or no) entry.
	d, err := os.Open(dir) //nolint:gosec // dir is derived from dataDir
	if err != nil {
		return fmt.Errorf("cluster: open dir %s: %w", dir, err)
	}
	if err := d.Sync(); err != nil {
		_ = d.Close() //nolint:errcheck,gosec
		return fmt.Errorf("cluster: fsync dir %s: %w", dir, err)
	}
	return d.Close()
}

// wasmRecoveryAdvice is the operator instruction attached to every unreadable- or
// missing-sidecar refusal.
//
// IT DELIBERATELY DOES NOT SAY "DELETE THE PAIR AND LET THE NODE RE-RECEIVE IT",
// which is what it used to say and which is FALSE in durable mode. The claim
// rested on "the registration is still in every group's Raft log", which is true,
// and on the node re-applying it, which is not: fsm.Apply skips every entry at or
// below the persisted applied index (shard/fsm.go), so a __register_wasm__ entry
// the node already applied is never applied again. Deleting the pair on a durable
// node therefore destroys the op on that node permanently, and the first
// committed invocation halts it under classFatal with no replay path — turning a
// recoverable local file problem into an unrecoverable one by following the
// documentation.
//
// What DOES work is wiping the node's data dir and rejoining: that discards the
// applied index along with everything else, so the node catches up by
// InstallSnapshot, and snapshots carry the registrations (wasmSnapshotBlob) plus
// their per-group route-gate evidence. It is expensive and it is the honest
// answer.
//
// IT IS ALSO NOT UNCONDITIONALLY SAFE, and the string says so, because an
// operator reads it mid-incident and follows it literally. Two qualifications ride
// with it:
//
//   - the snapshot carries only Source == "replicated" installs (see
//     replicatedRegs), so a wipe does NOT recover a CONFIG module. Those come back
//     from cfg.WASMModules on the next start, which means the wipe is only harmless
//     for them if the config is still in place — and if the broken pair WAS a config
//     module, rejoining recovers nothing about it;
//   - "wipe and rejoin" is unqualified DATA LOSS if this node is the last healthy
//     replica of any group it hosts, or a PB primary with no ISR. There is nothing
//     to catch up FROM in that case; the wipe is the deletion. Confirm another
//     healthy replica of every group on this node first.
//
// THE CONTENT-ADDRESSED BLOB STORE DID NOT MAKE THIS ADVICE OBSOLETE, and it is
// worth being exact about which half it fixed. It removed the way the broken
// state used to ARISE from our own writes — the sidecar is now the single
// per-op artifact and the single commit point, so no crash and no failed write
// can leave a module's bytes and its metadata describing different registrations.
// It does nothing about the durable-mode replay gap, which is what makes deletion
// lossy, and deletion is still the operator's instinct when a blob fails its hash
// check or a sidecar goes missing. So the advice stands unchanged.
const wasmRecoveryAdvice = "recovery is NOT as simple as deleting the files: in durable mode fsm.Apply skips every entry at or below the persisted applied index, so a deleted module is never re-applied from the Raft log and the first committed invocation halts this node. NEVER delete a <name>.json sidecar: the reload scans sidecars, so removing one does not fail startup — the node comes up CLEAN with the op silently absent, halts on the first committed invocation of it, and omits it from every snapshot it serves, so replicas joining from this node inherit the same halt. Wipe this node's data dir and rejoin the cluster instead, which recovers the module and its route-gate evidence from a peer's snapshot. TWO CAVEATS BEFORE YOU WIPE: (1) snapshots carry only replicated registrations, so a module installed from this node's config is NOT recovered by the rejoin — it comes back only from cfg.WASMModules, so keep that config; (2) wiping is unqualified DATA LOSS if this node is the last healthy replica of any group it hosts, or a PB primary with no in-sync replica, because there is then nothing to catch up from — confirm another healthy replica of every group on this node BEFORE wiping"

// readWASMSidecar reads <dataDir>/wasm/<name>.json.
//
// It returns ok=false ONLY when the sidecar genuinely does not exist. That is
// the state "this node has no record of an op by this name", which the config
// path treats as permission to install one and the reload path never reaches
// (it scans the sidecars themselves).
//
// A sidecar that exists but cannot be parsed is a HARD ERROR here, not a
// fall-back to defaults: "swallow the unmarshal error and use defaults" converts
// a torn file into a silently mis-registered op and a disarmed route gate (see
// atomicWriteFile for what the zero wasmMeta costs). Failing loudly is correct;
// what the failure must NOT do is misstate the remedy, which is why it carries
// wasmRecoveryAdvice rather than the "delete the pair and let it re-arrive"
// instruction it used to carry.
//
// A PARSEABLE SIDECAR WITH NO Blob IS ALSO A HARD ERROR, for the same reason one
// notch further in: the module bytes would be unfindable, and the only fallback
// available — reading the pre-blob-store <name>.wasm — is precisely the
// name-addressed layout whose drift this store exists to end.
//
// ON-DISK BREAK. A data dir written by a build older than the blob store has
// <name>.wasm files and Blob-less sidecars, and this refusal means such a node
// REFUSES TO START rather than silently losing its WASM ops. That is deliberate
// and the app is unreleased, so no migration is offered: wipe the node's data dir
// and rejoin (subject to the two caveats in wasmRecoveryAdvice), or delete the
// stale <dataDir>/wasm contents if the ops were config-only and will come back
// from cfg.WASMModules.
func readWASMSidecar(dataDir, name string) (wasmMeta, bool, error) {
	var meta wasmMeta
	path := filepath.Join(dataDir, "wasm", name+".json")
	b, err := os.ReadFile(path) //nolint:gosec // path constructed from dataDir + trusted filename
	if err != nil {
		if errors.Is(err, os.ErrNotExist) {
			return meta, false, nil
		}
		return meta, false, fmt.Errorf("cluster: read wasm sidecar %s: %w", path, err)
	}
	if err := json.Unmarshal(b, &meta); err != nil {
		return wasmMeta{}, false, fmt.Errorf("cluster: wasm sidecar %s is unreadable (%w); it is not safe to fall back to defaults — %s", path, err, wasmRecoveryAdvice)
	}
	// This check is LOAD-BEARING, not just a nicer message. For a REPLICATED
	// sidecar it only improves the wording — wasmBlobPath rejects "" anyway. But
	// for a CONFIG module it is the only thing that makes the on-disk break loud:
	// without it a pre-blob-store sidecar flows past the config skip into
	// loadOneModule, which rewrites both artifacts and starts the node normally —
	// a silent in-place upgrade of a data dir this build cannot actually read.
	// Deleting it would quietly falsify the "refuses to start on an old data dir"
	// guarantee for exactly the case an operator is most likely to hit.
	if meta.Blob == "" {
		return wasmMeta{}, false, fmt.Errorf("%w: sidecar %s names no blob (a data dir written before the content-addressed blob store is not readable by this build)",
			errWASMBlob, path)
	}
	// THE VERSION CHECK IS THE SECOND ON-DISK BREAK, and it is load-bearing for the
	// same reason the Blob check is. A version-1 sidecar records the route gate's
	// evidence as a bare group SET ("groups": [...]), which this build does not
	// read: it needs a per-group VERSION. Accepting such a file would leave every
	// group unbound while the op stayed registered — unroutable everywhere (the
	// route gate shut) and, for any committed invocation still to replay out of a
	// group's log, a classFatal ops.ErrWASMNoGroupBinding halt. Absence of bindings
	// cannot be used as the signal instead: a replicated op legitimately has an
	// empty binding set when its only source was a snapshot install section.
	if meta.Version != wasmMetaVersion {
		return wasmMeta{}, false, fmt.Errorf("cluster: wasm sidecar %s is format version %d, not %d (a data dir written before the key extractor was pinned to %q is not readable by this build) — %s",
			path, meta.Version, wasmMetaVersion, ops.WASMKeyExtractorHandle, wasmRecoveryAdvice)
	}
	return meta, true, nil
}
