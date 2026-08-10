// SPDX-License-Identifier: Apache-2.0

package cluster

import (
	"bytes"
	"encoding/binary"
	"errors"
	"fmt"
	"strings"
	"time"

	"github.com/rostamlabs/rostam/ops"
	"github.com/rostamlabs/rostam/shard"
)

// wasmRegisterOpName is the shardless op that carries a dynamic WASM module
// registration (ops/wasm_register.go). Node.Call intercepts it and broadcasts.
const wasmRegisterOpName = "__register_wasm__"

// opRegisterWASMShardName is the INTERNAL, node-local wrapper op used to drive a
// WASM registration into ONE named shard group on a peer. It dispatches off
// n.adminOps (Node.Call) — before op-registry routing — so the receiving node
// proposes the entry to exactly the requested group and does NOT re-broadcast.
//
// Sending wasmRegisterOpName itself over the wire would re-enter Node.Call on
// the peer and fan out to every group IT hosts, which in a cluster whose nodes
// host overlapping-but-unequal shard subsets can bounce between nodes without
// terminating. The wrapper makes the remote step a leaf.
//
// It is not in authz's admin allowlist by name and does not need to be: an op
// that matches no allowlist and is absent from the ops registry falls through
// to actionFor's deny-by-default "admin" bucket, which is exactly the privilege
// __register_wasm__ itself requires. Inter-node hops carry the internal service
// token, so they pass; an external client presenting a non-admin key does not.
const opRegisterWASMShardName = "__register_wasm_shard__"

// encodeShardScopedWASM prefixes a WASMRegistration payload with the target
// shard index (4-byte big-endian), the wire form of opRegisterWASMShardName.
func encodeShardScopedWASM(shardIdx int, reg []byte) []byte {
	out := make([]byte, 4+len(reg))
	binary.BigEndian.PutUint32(out[:4], uint32(shardIdx)) //nolint:gosec // bounded by NumShards
	copy(out[4:], reg)
	return out
}

// decodeShardScopedWASM splits the wire form written by encodeShardScopedWASM.
func decodeShardScopedWASM(args []byte) (int, []byte, error) {
	if len(args) < 4 {
		return 0, nil, fmt.Errorf("cluster: %s: short args (%d bytes)", opRegisterWASMShardName, len(args))
	}
	return int(binary.BigEndian.Uint32(args[:4])), args[4:], nil
}

// checkWASMRegistrationPayload holds every PROPOSE-TIME refusal for a decoded
// registration, in one place, because there are TWO ways an entry reaches a
// group's Raft log and both must apply the same set:
//
//   - Node.Call's __register_wasm__ intercept, which broadcasts to every group;
//   - handleRegisterWASMShard, the shard-scoped wrapper a broadcast leg forwards
//     to a peer.
//
// The second one used to apply NONE of these. It is dispatched off n.adminOps at
// the very top of Node.Call, before the __register_wasm__ intercept is reached,
// so it walked around the size cap and the update gate completely — and it is
// reachable by any admin-authenticated EXTERNAL client, not just by peers (see
// opRegisterWASMShardName's authz note). That gave an admin client two primitives
// this cluster is otherwise built to deny: drive a DIFFERING registration into
// exactly ONE group's log (one group on v2 while every other group stays on v1 —
// the maximally divergent shape), and push a module up to the 16 MiB frame limit
// into a Raft log the 4 MiB cap exists to protect.
//
// The three checks differ in strength and it is worth being precise about which
// is which:
//
//   - the NAME check is absolute. It is a pure function of the entry, so it is
//     also enforced authoritatively at apply time (ops.RegisterWASMRegisterOp);
//     running it here only keeps a doomed entry out of the log;
//   - the SIZE cap is absolute and entry-derived, but it is deliberately NOT an
//     apply-time check: rejecting at apply time is too late (the entry is already
//     committed and replicated on every node) and the frame layer is far too
//     permissive;
//   - the KIND check is absolute and entry-derived, exactly like the name check,
//     and is likewise enforced authoritatively at apply time
//     (validateWASMRegistrationKind, called by applyWASMRegistration before any
//     write). Running it here only keeps a doomed entry out of the log;
//   - the UPDATE gate is BEST-EFFORT, because it is judged against this node's
//     own install state, which is exactly the state allowed to lag. See
//     checkWASMUpdateGate for why it must not move to apply time.
func (n *Node) checkWASMRegistrationPayload(r ops.WASMRegistration) error {
	if err := ops.ValidateWASMOpName(r.Name); err != nil {
		return fmt.Errorf("cluster: register wasm: %w", err)
	}
	if err := validateWASMRegistrationKind(r.Name, r.Kind); err != nil {
		return err
	}
	if r.Blob == ops.ZeroWASMBlob {
		return fmt.Errorf("%w: register wasm %q: the marker names no module blob; nothing could ever resolve it",
			ErrWASMRegistrationRefused, r.Name)
	}
	// THE MODULE SIZE CAP IS NO LONGER CHECKED HERE, because there is no module
	// here to check. A marker names its module by content address, so the size the
	// broadcast amplifies is now a few dozen bytes regardless of what the module
	// weighs — which is the entire point of thin markers. The cap still exists and
	// still matters, but it now guards the transport that actually carries the
	// bytes: maxWASMBlobPutFrame, enforced on the client-edge registration request
	// (Node.Call) and on every __wasm_blob_put__ leg (handleWASMBlobPut). See
	// maxDynamicWASMBytes.
	return n.checkWASMUpdateGate(r)
}

// checkWASMRegistrationArgs is checkWASMRegistrationPayload plus the two refusals
// that can only be made on the RAW, still-encoded frame. It is the single entry
// both propose-time callers use (Node.Call's __register_wasm__ intercept and
// handleRegisterWASMShard), because both used to share the same two holes.
//
// THE ENCODED SIZE CAP RUNS FIRST, BEFORE THE DECODE. The decoded cap on r.Bytes
// cannot see a frame that does not decode, and nothing below the client edge
// bounds one: server.MaxFrameSize admits 16 MiB and neither shard nor raft
// imposes a propose-side entry-size limit. So 16 MiB of garbage was accepted here
// and appended to EVERY group's log — NumShards is 64 by default and validates up
// to 65536, i.e. ~1 GiB of replicated Raft log per attempt, repeatable, all of it
// discarded again at apply time as a classAdvance decode error. Capping len(args)
// is the only check that can stop that, and it has to happen before the decode is
// even attempted.
//
// IT IS NOT SUBSUMED BY THE DECODE REFUSAL BELOW, and neither is it a substitute
// for the CANONICALITY refusal that follows it. ops.DecodeWASMRegistration does
// not assert that it CONSUMED its input: it reads the fixed tail at whatever
// offset it has reached and returns. So a valid registration followed by trailing
// junk decodes CLEANLY, and its r.Bytes is whatever the real module was — under
// the module cap. Neither the decode refusal nor the module cap sees that frame,
// and broadcastWASMRegistration proposes the RAW args rather than a re-encode, so
// every junk byte would be replicated into every group's log behind a registration
// that then applies successfully: no error, no metric, nothing to observe but the
// log growth. Worse, checkWASMUpdateGate fingerprints the DECODED registration, so
// padding changes no field: a padded re-send of an already-installed module
// compares EQUAL, takes the idempotent-retry branch, and is broadcast again with
// fresh padding — repeatable without bound.
//
// The frame cap BOUNDS that (one attempt cannot exceed maxWASMRegistrationFrame
// per group) but does not close it, because padding under the cap still rides
// along. What closes it is the check that args must be byte-for-byte what
// re-encoding the decoded registration produces: there is then nowhere for a
// padding byte to hide. The frame cap still has to run FIRST — it is the only
// check that can bound a frame which never decodes at all.
//
// THE CANONICALITY REFUSAL IS PROPOSE-SIDE ONLY, deliberately. Tightening
// ops.DecodeWASMRegistration itself would be a far larger move than it looks:
// ops/wasm_register.go calls that decoder at APPLY time, on bytes already
// committed to every group's log. A strict decoder would change the verdict on
// entries that are already durable, so during a rolling upgrade an old and a new
// binary would disagree about whether a committed entry applies — replica
// divergence, the exact failure class this file exists to prevent. A propose-time
// refusal has no cross-version verdict: it only ever decides what is allowed to
// ENTER a log, and every legitimate producer already builds args with
// ops.EncodeWASMRegistration.
//
// AN UNDECODABLE PAYLOAD IS REFUSED RATHER THAN FORWARDED. Both callers used to
// run the checks only `if err == nil` on the decode and otherwise fall through,
// deferring to the apply-time decode error — which is a strange thing to defer,
// since it is a pure function of the bytes the caller just sent and can only ever
// come back as a failure. Refusing here costs the client nothing (it gets the same
// error, sooner and once instead of NumShards times) and keeps a frame that cannot
// possibly apply out of every group's log.
//
// IT RETURNS THE DECODED REGISTRATION so its callers do not decode a second
// time. That is not tidiness: ops.DecodeWASMRegistration COPIES r.Bytes, so the
// push phase (pushWASMBlob), which needs the module bytes the moment this check
// passes, would otherwise copy up to maxDynamicWASMBytes again for every
// registration.
func (n *Node) checkWASMRegistrationArgs(args []byte) (ops.WASMRegistration, error) {
	var zero ops.WASMRegistration
	if len(args) > maxWASMRegistrationFrame {
		return zero, fmt.Errorf("%w: encoded registration is %d bytes, over the %d-byte limit (it would be written to all %d shard groups' logs)",
			ErrWASMRegistrationRefused, len(args), maxWASMRegistrationFrame, n.cfg.NumShards)
	}
	r, err := ops.DecodeWASMRegistration(args)
	if err != nil {
		return zero, fmt.Errorf("%w: payload does not decode as a WASM registration (%v); it could never have applied on any replica", ErrWASMRegistrationRefused, err)
	}
	if enc := ops.EncodeWASMRegistration(r); !bytes.Equal(enc, args) {
		return zero, fmt.Errorf("%w: encoded registration is not canonical (%d bytes decode to a %d-byte registration)",
			ErrWASMRegistrationRefused, len(args), len(enc))
	}
	if err := n.checkWASMRegistrationPayload(r); err != nil {
		return zero, err
	}
	return r, nil
}

// checkWASMRegistrationRequest is the CLIENT-EDGE entry point: it validates the
// one payload shape that still carries the module and returns the thin marker
// plus the bytes.
//
// It is separate from checkWASMRegistrationArgs — which validates a bare MARKER —
// because the two frames are genuinely different and conflating them is how the
// module would find its way back into every group's log. This one is accepted
// only by Node.Call's __register_wasm__ intercept, from the client that is
// registering; checkWASMRegistrationArgs is what guards everything that enters a
// Raft log, including the shard-scoped leg a broadcast forwards to a peer.
//
// The order is the same one every other frame check in this package uses, and for
// the same reasons: the FRAME cap first, before any decode, because it is the only
// check that can bound a payload which never decodes at all; then the decode,
// which here also PROVES the module hashes to the fingerprint the marker names
// (ops.DecodeWASMRegistrationRequest) — the one place both halves exist together
// and therefore the only place that can be checked; then the module cap, which is
// now a bound on the blob transport rather than on any Raft log; then every
// propose-time refusal the marker itself is subject to.
func (n *Node) checkWASMRegistrationRequest(args []byte) (ops.WASMRegistration, []byte, error) {
	var zero ops.WASMRegistration
	if len(args) > maxWASMRegistrationRequestFrame {
		return zero, nil, fmt.Errorf("%w: encoded registration request is %d bytes, over the %d-byte limit",
			ErrWASMRegistrationRefused, len(args), maxWASMRegistrationRequestFrame)
	}
	r, module, err := ops.DecodeWASMRegistrationRequest(args)
	if err != nil {
		return zero, nil, fmt.Errorf("%w: %v", ErrWASMRegistrationRefused, err)
	}
	if len(module) == 0 {
		return zero, nil, fmt.Errorf("%w: register wasm %q: the request carries no module bytes", ErrWASMRegistrationRefused, r.Name)
	}
	if len(module) > maxDynamicWASMBytes {
		return zero, nil, fmt.Errorf("%w: register wasm %q: module is %d bytes, over the %d-byte limit",
			ErrWASMRegistrationRefused, r.Name, len(module), maxDynamicWASMBytes)
	}
	if err := n.checkWASMRegistrationPayload(r); err != nil {
		return zero, nil, err
	}
	return r, module, nil
}

// handleRegisterWASMShard is the node-local handler for opRegisterWASMShardName.
// It proposes the registration to the ONE group named in the payload — never to
// any other group, and never onward to another node — so the broadcast fan-out
// stays flat. Returning ErrNoShardOwner when this node does not host the group
// lets the sender's forward() loop move on to the next owner.
//
// It applies the SAME propose-time refusals Node.Call's broadcast intercept does
// (see checkWASMRegistrationArgs). Without them this handler was a complete
// bypass of both the size cap and the update gate, reachable by any
// admin-authenticated client.
//
// IT DOES NOT RUN THE BLOB PUSH, and must not. The push is a property of the
// REGISTRATION, not of one group's leg of it: the node that accepted the
// registration has already delivered the bytes to every member it could reach
// (see pushWASMBlob), so repeating it here would issue NumShards × Members
// pushes of the same module for one client-visible registration — and would do so
// from a node the client never addressed, whose skip report nothing reads.
func (n *Node) handleRegisterWASMShard(args []byte) ([]byte, error) {
	idx, reg, err := decodeShardScopedWASM(args)
	if err != nil {
		return nil, err
	}
	if idx < 0 || idx >= n.cfg.NumShards {
		return nil, fmt.Errorf("cluster: %s: shard %d out of range [0,%d)", opRegisterWASMShardName, idx, n.cfg.NumShards)
	}
	if _, err := n.checkWASMRegistrationArgs(reg); err != nil {
		return nil, err
	}
	s := n.getShard(idx)
	if s == nil {
		return nil, ErrNoShardOwner
	}
	return n.callHostedShard(s, wasmRegisterOpName, reg)
}

// broadcastWASMRegistration proposes a __register_wasm__ entry to EVERY shard
// Raft group instead of only shard 0's.
//
// WHY (silent cross-replica divergence). __register_wasm__ is a shardless op,
// so Call would route it to shard 0 alone. But a WASM module is always ROUTABLE
// (wasm.RegisterModule → RegisterRoutableCrossShard), so its invocations are logged in
// shardOf(key, NumShards)'s group — a different, independently ordered Raft
// log. On a replica caught up on group j but lagging on group 0, group j's FSM
// applies an invocation whose op is not yet in the node-wide registry; the
// apply fails with shard.ErrOpNotRegistered, which classifyApplyErr treats as
// deterministic (classAdvance) and SKIPS while peers that already applied the
// registration EXECUTE it. That is permanent, silent divergence with no halt,
// no error and no metric. Any client that can register a WASM op can reach it.
//
// WHAT THE BROADCAST DOES AND DOES NOT ESTABLISH. It gets the registration into
// every group's log, which is what makes the op USABLE on every group. It does
// NOT establish the ordering that makes it SAFE, and an earlier version of this
// comment claiming otherwise was wrong. The loop below is SEQUENTIAL while the
// ops registry is node-WIDE, so nothing here stops a node whose group 0 has
// applied from routing an invocation into group j before the loop reaches j.
//
// Ordering is established by the ROUTE GATE (checkWASMRouteGate): a node will
// not propose an invocation into a group it hosts until it knows that group's
// log already carries the registration. That is what puts REG below INV in every
// group's log, and hence what guarantees every replica of a group has the op
// registered before it applies an invocation of it. The broadcast is what makes
// the gate OPENABLE at all — a group that never receives the registration is a
// group the gate keeps permanently shut — and the snapshot carriage in
// wasm_load.go is what carries both the module and the per-group proof to
// replicas that catch up by InstallSnapshot rather than by log replay.
//
// WHAT MAY BE BROADCAST. Only a FIRST registration of a name, or an exact
// re-send of one. Node.Call refuses a registration that would change a live
// module before calling this (checkWASMUpdateGate), because nothing here or in
// the gate makes an in-place update safe: the effective module version is
// node-wide while these logs commit at independent times.
//
// PARTIAL FAILURE. Every group is attempted even after one fails (a transient
// election on group 3 must not deny groups 4..N the registration), and a
// non-empty failure set is returned as an error so the caller can retry.
// Retrying is safe and cheap: applyWASMRegistration is idempotent, so the
// groups that already accepted the entry simply re-apply it to no effect.
// Retrying is also how the op becomes usable on the starved group at all: under
// the route gate a group left without the registration serves nothing (every
// invocation routed to it fails with a client-visible error) until a retry
// lands the entry in its log. Before the gate, such an invocation entered the
// starved group's log and halted every replica of it.
//
// WRITE AMPLIFICATION — LARGELY RETIRED, and it is worth recording what it used
// to be. The module bytes were appended to NumShards Raft logs on every node —
// NumShards × the module size for one client-visible registration, plus the same
// again in each group's snapshot once it compacted, i.e. up to 64 × 4 MiB of
// replicated log per registration at the default shard count and module cap. What
// this function broadcasts now is a THIN MARKER: a name, a contract, an epoch and
// a 32-byte content address, so the amplification is NumShards × a few dozen
// bytes and is independent of the module's size. The bytes travel once, out of
// band, to each member (cluster.pushWASMBlob), and are fetched on demand by
// anyone the push missed.
//
// maxDynamicWASMBytes therefore no longer protects the Raft logs; it protects the
// blob transport, which is the thing that now carries the module.
func (n *Node) broadcastWASMRegistration(args []byte) ([]byte, error) {
	var failures []string
	for i := 0; i < n.cfg.NumShards; i++ {
		if err := n.proposeWASMRegistration(i, args); err != nil {
			failures = append(failures, fmt.Sprintf("shard %d: %v", i, err))
		}
	}
	if len(failures) > 0 {
		return nil, fmt.Errorf("cluster: register wasm: %d of %d shard groups rejected the registration (safe to retry, registration is idempotent): %s",
			len(failures), n.cfg.NumShards, strings.Join(failures, "; "))
	}
	return nil, nil
}

// maxDynamicWASMBytes caps the module a registration may carry.
//
// ITS JUSTIFICATION CHANGED WITH THIN MARKERS, and the number did not. It used to
// exist because the broadcast wrote the bytes into EVERY shard group's log —
// NumShards defaults to 64 and validates up to 65536 — so an unbounded module
// multiplied into gigabytes of Raft log and, after compaction, of per-group
// snapshot. That amplification is gone: a marker names its module.
//
// What the cap bounds now is the BLOB TRANSPORT and the blob store: one
// __wasm_blob_put__ frame per member per registration (maxWASMBlobPutFrame is
// this plus a fingerprint), the client-edge registration request that carries the
// module to the coordinator, and the on-disk blob every member keeps forever
// (retirement is separate). Those are linear in the module size rather than
// NumShards × it, so the cap is far less load-bearing than it was — but a 16 MiB
// server.MaxFrameSize payload replicated to every member is still not something
// to accept by default, and the limit constrains no realistic UDF: compiled WASM
// handlers are tens to hundreds of KiB.
const maxDynamicWASMBytes = 4 << 20 // 4 MiB

// maxWASMRegistrationFrame caps the ENCODED __register_wasm__ payload, i.e. the
// bytes that actually get appended to every group's log.
//
// It exists because maxDynamicWASMBytes is checked against a DECODED field, so it
// protects nothing when the frame does not decode — and the whole point of the cap
// is to bound what the broadcast amplifies, which happens to the encoded bytes
// whether or not they mean anything. See checkWASMRegistrationArgs.
//
// It applies to the MARKER, which no longer contains a module: the wire format
// (ops.EncodeWASMRegistration) is a 32-byte content address plus two
// u16-length-prefixed strings (Name, ExportName) and 21 bytes of fixed fields, so
// no legitimate marker comes close. The generous bound is
// kept rather than tightened to the true maximum because it costs nothing and
// because the check's job is to stop a hostile frame before the decode, not to
// audit a legitimate one.
//
// THE CLIENT-EDGE REGISTRATION REQUEST IS A DIFFERENT FRAME AND IS CAPPED
// SEPARATELY. That one does carry the module (ops.EncodeWASMRegistrationRequest),
// so it is bounded by maxWASMRegistrationRequestFrame; this cap governs only what
// enters a Raft log.
const maxWASMRegistrationFrame = maxDynamicWASMBytes + 1<<18 // generous marker bound

// maxWASMRegistrationRequestFrame caps the CLIENT-EDGE __register_wasm__ payload
// — the one leg that still carries the module, from the registering client to the
// node that will push it.
//
// It is checked BEFORE ANY DECODE, for the reason maxWASMRegistrationFrame and
// maxWASMBlobPutFrame are: nothing below the client edge bounds a frame
// (server.MaxFrameSize admits 16 MiB), and every check that could bound the
// MODULE has to look past a header a hostile frame may not have.
const maxWASMRegistrationRequestFrame = maxWASMRegistrationFrame + maxDynamicWASMBytes

// wasmBroadcastGroupTimeout bounds ONE group's leg of the broadcast.
//
// Without it a single unreachable-but-not-erroring peer stalls the whole
// registration — and the client connection behind it — indefinitely, because
// the loop is sequential over up to NumShards groups and forward() used an
// unbounded context. A group that times out is reported as a failure like any
// other, and the client retries.
const wasmBroadcastGroupTimeout = 10 * time.Second

// proposeWASMRegistration lands the registration in shard idx's Raft log,
// wherever this node sits relative to that group:
//
//   - hosted and led here: propose straight into the local store;
//   - hosted but led elsewhere (the normal case for all but one group in a
//     multi-shard cluster): shard.Store.Call answers NotLeaderError, so hop;
//   - not hosted at all (partitioned cluster): hop.
//
// The hop reuses forward(), whose per-owner client follows NotLeader to the
// group's leader — but carries the shard-scoped wrapper op so the peer handles
// this ONE group (see opRegisterWASMShardName).
//
// Unlike a normal op, a NotLeaderError must not escape to the client here: a
// broadcast spans groups with DIFFERENT leaders, so "retry at node X" is not a
// hint any single node can satisfy, and following it would bounce the client
// between nodes that each lead a different subset. The text is preserved for
// diagnostics; the error type is not.
func (n *Node) proposeWASMRegistration(idx int, args []byte) error {
	var hostedErr error
	if s := n.getShard(idx); s != nil {
		_, err := n.callHostedShard(s, wasmRegisterOpName, args)
		if err == nil {
			return nil
		}
		var nle *shard.NotLeaderError
		if !errors.As(err, &nle) {
			// A genuine propose/apply failure for this group (not a routing
			// miss); another owner would fail the same way.
			return err
		}
		hostedErr = err
	}
	if _, err := n.forwardTimeout(idx, opRegisterWASMShardName, encodeShardScopedWASM(idx, args), wasmBroadcastGroupTimeout); err != nil {
		if hostedErr != nil {
			return fmt.Errorf("%v; leader hop: %v", hostedErr, err)
		}
		return err
	}
	return nil
}
