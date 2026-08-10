// SPDX-License-Identifier: Apache-2.0

package ops

import (
	"errors"
	"fmt"
	"strings"
)

// WASMOpNameUnsafeMsg and WASMUpdateUnsupportedMsg are the two WASM refusal
// message fragments that transport layers OUTSIDE this package have to
// recognise by TEXT.
//
// They live here because the refusals originate in `cluster` (or in this
// package) and are then stringified across the Raft/RPC boundary, so by the time
// server.clientFacingErr and httpapi.statusForError see them there is no sentinel
// left to errors.Is against — only the message. Both of those classifiers must
// keep the message UNREDACTED (and answer 4xx rather than 500), because the
// message is the only place the caller learns the remedy.
//
// Exporting them as consts is what makes that coupling compile-time. They used to
// be seven independent string literals across cluster, server, httpapi, three
// test files and the docs, with a comment claiming they were linked; a rewording
// in one place would have silently started redacting the refusal to "internal
// error" everywhere else. Reword the const and every user follows; reword a
// literal and nothing follows, which is why there should be no literals left.
const (
	// WASMOpNameUnsafeMsg is the fragment carried by ErrWASMOpNameUnsafe.
	WASMOpNameUnsafeMsg = "wasm op name is not a safe filename"
	// WASMUpdateUnsupportedMsg is the fragment carried by
	// cluster.ErrWASMUpdateUnsupported. It is defined here, in a leaf package both
	// server and httpapi can import, because cluster is not importable from either.
	//
	// ITS SCOPE NARROWED WHEN PER-GROUP VERSION BINDING LANDED, and the text was
	// rewritten rather than the const replaced, precisely so the server/httpapi
	// substring wiring did not have to be unpicked. Updating a live WASM module's
	// BYTES is now a supported operation: the version that executes a committed
	// entry in shard group g is resolved from g's own log prefix, so every replica
	// of g agrees on it (see wasm.Runtime.resolveModuleForInvoke). What remains
	// refused is a registration that changes the op's Kind,
	// because those two are read on the PROPOSE side to decide whether there is a
	// proposal at all and which group it routes to — and the group index is what
	// the key extractor computes, so they cannot be resolved per group without
	// knowing the group first. They are FROZEN at first registration; changing
	// either stays a new-op-name operation.
	WASMUpdateUnsupportedMsg = "changing a live WASM op's kind or key extractor is unsupported"
	// WASMRegistrationRefusedMsg is the fragment carried by
	// cluster.ErrWASMRegistrationRefused: the propose-time refusals of a
	// __register_wasm__ payload that are about the PAYLOAD rather than about its
	// name or an attempted update — an over-cap or undecodable encoded frame, and
	// an out-of-range Kind byte. Same reason it lives here as
	// WASMUpdateUnsupportedMsg: cluster is not importable from server or httpapi.
	WASMRegistrationRefusedMsg = "wasm registration refused"
)

// ErrWASMOpNameUnsafe rejects a WASM op name that cannot be used as a bare
// filename. See ValidateWASMOpName.
var ErrWASMOpNameUnsafe = errors.New("ops: " + WASMOpNameUnsafeMsg)

// ValidateWASMOpName rejects any op name that is not a single, literal path
// component.
//
// THE NAME IS A FILESYSTEM PATH COMPONENT, AND IT COMES OFF THE WIRE. Every node
// that applies a __register_wasm__ entry persists the module as
// <dataDir>/wasm/<Name>.wasm with a <Name>.json sidecar beside it, both built
// with filepath.Join. filepath.Join CLEANS its result, so a Name of
// "../../../../etc/cron.d/x" does not fail — it resolves, and the node writes
// attacker-chosen bytes to an attacker-chosen absolute path as the server user,
// on every replica at once. The atomic-write helper makes it worse rather than
// better: it creates its temp file in filepath.Dir(path), i.e. inside the
// traversed directory, so even a failed write lands there.
//
// The second-order damage is a cluster halt. A traversed pair is invisible to the
// restart scan, which globs only <dataDir>/wasm/*.wasm, so after a restart the op
// is simply GONE on that node while invocation entries for it sit in every
// group's Raft log. Those applies fail with shard.ErrOpNotRegistered, which is
// classFatal — the node stops.
//
// The rule is the strictest one that still admits every sane op name: the name
// must not be "." or "..", must contain no separator of either flavour, no volume
// separator, no NUL, and no ".." anywhere.
//
// IT ALSO CARRIES THE TWO REGISTRY NAME RULES — length <= maxOpNameLen and
// length != 2 (the protocol-v2 version-byte collision). Those are also enforced
// by validateEntry, but validateEntry is reached only from Registry.Register /
// Registry.Replace, which cluster.applyWASMRegistration calls 27 lines AFTER it
// has written <Name>.wasm and its sidecar. A registration failing only there
// therefore left both files on every node while installing nothing, and the next
// restart's reloadWASMModulesFromDisk re-derived the same failure — which is NOT
// ErrDuplicateOp, so it FAILED NODE CONSTRUCTION cluster-wide until an operator
// deleted the files by hand. Checking them here, on the entry-derived path both
// the propose side and the apply side already run, is what keeps them from ever
// reaching a write. The messages wrap ErrWASMOpNameUnsafe so the existing
// client-facing plumbing (server.clientFacingErr, httpapi.statusForError) carries
// them unredacted.
//
// The length checks come LAST so that a genuinely unsafe name still gets the
// message that describes why it is unsafe: "/x" is a separator problem, not a
// protocol-v2 collision.
//
// IT IS FREE OF GOOS. It used to end with `filepath.Base(name) != name`, which is
// dead on Linux (the separator clause has already fired for anything Base would
// strip) but LIVE and DIFFERENT on Windows, where Base also strips a drive-volume
// prefix: "a:b" would be accepted by a Linux replica and refused by a Windows one.
// That is a build-target-dependent verdict on the AUTHORITATIVE apply-time check,
// i.e. exactly the divergence this function's determinism claim rules out. There
// is no Windows target today, so nothing is being supported here — the GOOS input
// is simply removed, by rejecting ':' explicitly and by dropping the redundant
// os.PathSeparator term (it is always '/' or '\\', both already listed).
//
// It is a PURE FUNCTION OF THE ENTRY, which is what makes it safe to enforce at
// apply time: every replica evaluates it on the same bytes and refuses
// identically, so a refusal is deterministic (classAdvance) and cannot diverge
// replicas the way a state-dependent apply-time rejection would.
func ValidateWASMOpName(name string) error {
	switch {
	case name == "":
		return fmt.Errorf("%w: the name is empty", ErrWASMOpNameUnsafe)
	case name == "." || name == "..":
		return fmt.Errorf("%w: %q is a directory reference", ErrWASMOpNameUnsafe, name)
	case strings.ContainsAny(name, `/\:`+"\x00"):
		return fmt.Errorf("%w: %q contains a path or volume separator or a NUL byte", ErrWASMOpNameUnsafe, name)
	case strings.Contains(name, ".."):
		return fmt.Errorf("%w: %q contains %q, which filepath.Join would resolve into a parent directory", ErrWASMOpNameUnsafe, name, "..")
	case len(name) > maxOpNameLen:
		return fmt.Errorf("%w: name length %d exceeds max %d", ErrWASMOpNameUnsafe, len(name), maxOpNameLen)
	case len(name) == 2:
		return fmt.Errorf("%w: %q has length 2 which collides with the protocol v2 version byte; use 3+ chars", ErrWASMOpNameUnsafe, name)
	}
	return nil
}

// WASMRegisterHook is called on every replica when a "__register_wasm__" op
// is applied through the Raft FSM. Implementations should load the WASM module
// into the node-local wasm.Runtime and wire it into the ops.Registry.
//
// shardIdx names the shard GROUP whose log carried the entry being applied, or
// NoShardIndex when the call has no dispatcher behind it. It is load-bearing,
// not diagnostic: the registration is replicated to every group, but whether an
// INVOCATION of the registered op may be proposed into a given group's log
// depends on whether THAT group's log carries the registration. Only the applying
// dispatcher knows which group that is, so it has to be threaded through here.
// See cluster.checkWASMRouteGate for the full argument.
type WASMRegisterHook func(shardIdx int, r WASMRegistration) error

// noopWASMHook is the zero-value hook installed when the caller passes nil.
// It returns an error so that accidental invocations surface clearly rather
// than silently doing nothing.
func noopWASMHook(_ int, _ WASMRegistration) error {
	return errors.New("ops: no WASMRegisterHook installed")
}

// RegisterWASMRegisterOp registers the shardless "__register_wasm__" op
// (OpReadWrite) on reg. When the FSM applies the op it:
//  1. Decodes the args as a WASMRegistration — the THIN MARKER, which names the
//     module by content address and does not carry it.
//  2. Rejects payloads with a zero Blob or an unsafe Name.
//  3. Calls hook with the decoded registration and the shard group the entry
//     was applied from (tx.ShardIndex).
//
// THE ZERO-Blob CHECK REPLACED THE EMPTY-Bytes ONE, and it is the same check in
// the new representation: a marker that names no module is one nothing can ever
// resolve, so installing it would register an op whose every invocation blocks
// forever waiting for bytes that do not exist. It is a pure function of the
// entry, so every replica refuses it identically — classAdvance stays correct.
//
// WHAT IS NO LONGER CHECKED HERE, deliberately: that the module COMPILES, or that
// an OpReadOnly declaration matches a module that does not mutate state. Neither
// is decidable from the entry any more, because the entry no longer contains the
// module. Both moved to the points that do have the bytes — the push's compile
// verdict at registration time (cluster.storeWASMBlobVerified, on the coordinator
// and on every member that acks) and the kind guard at the moment a fetched
// module is resolved for execution (wasm.Runtime.resolveModuleForInvoke). Both
// remain pure functions of (marker, blob), and the blob is self-verifying against
// the marker's fingerprint, so both verdicts stay identical on every replica —
// they are just reached at different TIMES on a node that had to fetch.
//
// If hook is nil, a default hook that returns an error is used.
//
// THE NAME CHECK IS AUTHORITATIVE HERE, not merely advisory, even though the
// propose path also runs it. This is the last point before the hook turns Name
// into a filesystem path, and it is the only point that EVERY route into the hook
// passes through — a client Call, a peer's forwarded shard-scoped leg, and a log
// entry replayed on a replica that never saw the propose-time check. Being a pure
// function of the entry (see ValidateWASMOpName) it is safe here: every replica
// refuses the same entry identically, so the refusal is deterministic and
// classifyApplyErr's classAdvance treatment stays correct.
func RegisterWASMRegisterOp(reg *Registry, hook WASMRegisterHook) error {
	if hook == nil {
		hook = noopWASMHook
	}
	return reg.Register("__register_wasm__", OpReadWrite, func(tx *TxContext, args []byte) ([]byte, error) {
		r, err := DecodeWASMRegistration(args)
		if err != nil {
			return nil, err
		}
		if err := ValidateWASMOpName(r.Name); err != nil {
			return nil, err
		}
		if r.Blob == ZeroWASMBlob {
			return nil, errors.New("ops: wasm registration names no module blob")
		}
		if err := hook(tx.ShardIndex(), r); err != nil {
			return nil, err
		}
		return nil, nil
	})
}
