// SPDX-License-Identifier: Apache-2.0

package cluster

import (
	"context"
	"crypto/sha256"
	"encoding/hex"
	"errors"
	"fmt"
	"strings"
	"time"

	"github.com/rostamlabs/rostam/client"
	"github.com/rostamlabs/rostam/ops"
	"github.com/rostamlabs/rostam/wasm"
)

// WASM BLOB TRANSPORT — how a node that lacks a module's bytes obtains them.
//
// Module bytes are content-addressed and self-verifying
// (<dataDir>/wasm/blobs/<sha256>.wasm; see wasmBlobPath / readWASMBlob). The
// bound the executing version per shard group. What neither established is that
// the bytes can MOVE: today they move only because every __register_wasm__ entry
// carries them inline in every shard group's log, which costs NumShards × the
// module size per node, in the Raft log and again in each group's snapshot.
//
// This file is the replacement channel, and it is the ONLY thing this stage
// builds. Two node-local admin ops:
//
//	__wasm_blob_put__(fp, bytes) — verify, compile, store, ack;
//	__wasm_blob_get__(fp)        — read the local store, return bytes.
//
// THE INVARIANT IT ESTABLISHES: any node can obtain the bytes for a fingerprint
// from any node that has them, and a received blob is accepted only if it hashes
// to its fingerprint AND compiles locally.
//
// Both ops are dispatched off n.adminOps (Node.Call), before op-registry routing,
// exactly like opRegisterWASMShardName — so the receiving node handles them
// itself and never re-broadcasts. Both are admin-gated by an EXPLICIT entry in
// authz.adminOps rather than by absence from the ops registry; see that map and
// authz/wasm_blob_authz_test.go for why the absence-based classification is not
// good enough.

// opWASMBlobPutName is the INTERNAL, node-local op that delivers one module's
// bytes to a peer's content-addressed blob store.
//
// THE ACK IS A COMPILE VERDICT, NOT A DELIVERY RECEIPT, and that is the reason
// this op exists in the shape it does rather than as a plain file copy. The
// handler compiles the bytes with THIS node's wasmtime before it acks, so a
// fleet whose nodes disagree about the module — a wasmtime version that rejects
// an instruction another accepts, a build without cgo, a determinism-gate
// refusal on a banned import — says so HERE, at registration time, to the client
// that is trying to register.
//
// The alternative is to discover it at apply time on a peer, which is a strictly
// worse place for it to surface: the entry is already committed to every group's
// log, every replica has to reach the same verdict or diverge, and once the
// marker no longer carries the bytes the failing node cannot even
// re-derive them from the entry. Refusing at registration is the difference
// between a 400 the operator can act on and a group that will not move.
const opWASMBlobPutName = "__wasm_blob_put__" //nolint:gosec // G101 false positive: an internal OP NAME, not a credential.

// opWASMBlobGetName is the INTERNAL, node-local op that serves one module's
// bytes out of this node's content-addressed blob store.
//
// ############ IT TOUCHES NOTHING BUT THE FILESYSTEM. THAT IS A CONTRACT. ######
//
// handleWASMBlobGet reads n.cfg.DataDir — immutable after construction — and
// calls readWASMBlob. It does NOT take wasmApplyMu, does NOT read n.wasmState or
// n.wasmRT, does NOT consult a shard FSM, the route gate, the meta FSM, or Raft.
// It acquires NO lock that any apply path can hold, and no lock at all.
//
// WHY THAT IS WORTH PROTECTING. A node fetches a blob because it is trying to
// APPLY something that needs it. The fetch is issued from
// exactly that position. If serving a get required a lock an apply holds, then
// node A stalled in an apply waiting on a fetch from node B, while node B is
// stalled in an apply holding the lock that would serve it, is a two-node
// deadlock with no timeout that resolves it into progress — both sides are
// waiting for the other's apply to finish. Keeping the get free of every such
// lock makes that analysis trivial instead of delicate: there is no edge to
// draw, so there is no cycle to find, and the argument does not have to be
// re-derived every time a lock is added elsewhere.
//
// WHAT WOULD BREAK IT, concretely, because each of these looks locally
// reasonable. The two locks an apply actually holds are named, since that is what
// decides whether a given addition is a hazard at all:
//
//   - answering "which op is this blob for?" by consulting n.wasmState — needs
//     wasmApplyMu, which every applyWASMRegistration holds for its whole
//     compile-write-install;
//   - short-circuiting on rt.HasModule to skip the disk read — takes the
//     Runtime's own RWMutex for reading, and AddModule takes it for WRITING from
//     inside an apply. THIS IS THE SHARPEST EDGE of the four: it is the one that
//     reads as a pure optimisation, and the lock it takes is not one this file
//     mentions anywhere else;
//   - serving from a memory cache guarded by any mutex an apply also takes;
//   - proving freshness through Raft or the meta FSM before answering — no lock
//     needed to make this fatal: it makes the get wait on an apply LOOP, which is
//     the same cycle by a different mechanism.
//
// The ROUTE GATE is deliberately NOT on that list, though an earlier version of
// this comment put it there. n.wasmGate is an atomic.Pointer and
// checkWASMRouteGate is a lock-free load, so gating the read on it would be
// WRONG — a blob has no op name and no group — without being a deadlock hazard.
// Listing a non-hazard among the hazards is how the real ones stop being read.
//
// None of them is needed. The blob store is content-addressed and self-verifying,
// so the FILE alone answers the question completely and correctly: a blob is the
// bytes hashing to its own name, or it is refused.
//
// TestWASMBlobGetTouchesNoApplyLock defends the LOCK half of this mechanically:
// it holds both wasmApplyMu and the Runtime's write lock — the state an apply is
// in at the moment it installs a module — and requires the get to complete
// anyway. The Raft/meta-FSM case is not mechanically covered; nothing there is a
// lock this test can hold.
const opWASMBlobGetName = "__wasm_blob_get__" //nolint:gosec // G101 false positive: an internal OP NAME, not a credential.

// wasmBlobFPHexLen is the length of a blob fingerprint on the wire: a sha256 in
// lower-case hex, the same form wasmBlobPath accepts and the same form the
// sidecar stores. It is FIXED-WIDTH, which is what lets both wire formats below
// dispense with a length prefix — and, more usefully, what makes them CANONICAL
// for free: with no variable-length header there is nowhere for a padding byte
// to hide, so the trailing-junk hole that checkWASMRegistrationArgs has to close
// with an explicit re-encode comparison cannot arise here. A put's trailing junk
// is module bytes, and the hash check rejects it; a get's trailing junk changes
// the payload length, and the length check rejects it.
const wasmBlobFPHexLen = 2 * sha256.Size

// maxWASMBlobPutFrame caps the ENCODED __wasm_blob_put__ payload.
//
// It is checked BEFORE ANY DECODE, for the same reason maxWASMRegistrationFrame
// is: nothing below the client edge bounds a frame (server.MaxFrameSize admits
// 16 MiB), and every check that could bound the MODULE has to look past a header
// that a hostile frame may not have. Refusing on length costs one comparison and
// stops a 16 MiB payload before this node hashes it, compiles it, or writes it.
//
// UNLIKE THE REGISTRATION FRAME, THERE IS NO SEPARATE DECODED MODULE CAP, and
// that is a property of the format rather than an omission. The registration
// envelope is variable-length (three u16-prefixed strings), so its frame cap has
// to carry slack and cannot decide the module's size; a payload here is exactly
// wasmBlobFPHexLen + the module, so this ONE check is exactly the
// maxDynamicWASMBytes check. The message names both numbers so the operator sees
// the module limit, which is the one they can act on.
const maxWASMBlobPutFrame = wasmBlobFPHexLen + maxDynamicWASMBytes

// wasmBlobPushTimeout bounds ONE member's leg of the pre-registration push.
//
// It exists for the reason wasmBroadcastGroupTimeout does: the push is
// sequential over members and a single unreachable-but-not-erroring peer would
// otherwise stall the registration, and the client connection behind it,
// indefinitely. A member that times out is UNREACHABLE (see pushWASMBlob), so it
// is skipped rather than failing the registration.
const wasmBlobPushTimeout = 10 * time.Second

// ErrWASMBlobRefused carries the __wasm_blob_put__ refusals: an oversized or
// short frame, a fingerprint that is not a sha256 in lower-case hex, bytes that
// do not hash to the claimed fingerprint, and bytes that do not compile here.
//
// It wraps ops.WASMRegistrationRefusedMsg — the SAME substring the registration
// refusals use — deliberately. Every one of these reaches the registering client
// only as a STRING: the refusal is raised on a PEER, returned as a
// client.RemoteError, and folded into the error Node.Call returns, at which point
// no sentinel identity survives. server.clientFacingErr and
// httpapi.statusForError recognise that substring and keep the message
// unredacted (and a 400 rather than a 500). Without it, an operator whose module
// one node's wasmtime rejects is told "internal error" for a payload they can
// fix. See ErrWASMRegistrationRefused.
var ErrWASMBlobRefused = errors.New("cluster: wasm blob: " + ops.WASMRegistrationRefusedMsg)

// wasmBlobUnknownOpMsg is how a peer that does not KNOW __wasm_blob_put__ says
// so, seen from here: the tail of cluster.ErrUnknownOp's message, arriving as a
// bare string inside a client.RemoteError. wasmBlobPeerRefused matches it to keep
// a mixed-version peer out of the refusal class.
//
// It is a literal because the identity is gone by the time this node sees it (the
// error crossed the RPC boundary as text) and because the sentinel lives in
// another concern; TestWASMBlobUnknownOpMsgTracksErrUnknownOp pins the two
// together so rewording the sentinel fails a test rather than silently
// reclassifying every mixed-version peer as a refusal.
const wasmBlobUnknownOpMsg = "op not registered"

// encodeWASMBlobPut builds the __wasm_blob_put__ wire form:
//
//	[fingerprint: wasmBlobFPHexLen bytes of lower-case hex][module bytes]
func encodeWASMBlobPut(fp string, b []byte) []byte {
	out := make([]byte, 0, len(fp)+len(b))
	out = append(out, fp...)
	return append(out, b...)
}

// handleWASMBlobPut stores one module's bytes in this node's content-addressed
// blob store, but only after proving BOTH halves of the acceptance rule:
// sha256(bytes) == the claimed fingerprint, AND the bytes compile here.
//
// ORDER IS THE POINT, and it is the same order loadOneModule uses: every refusal
// that can be decided from the payload runs before anything is written, so a
// module this node cannot accept leaves NO file behind. Cheapest and most
// bounding first:
//
//  1. the FRAME cap, before any decode — the only check that can bound a payload
//     which never decodes at all;
//  2. the frame's minimum length, so the header slice cannot panic;
//  3. the fingerprint FORMAT, via wasmBlobPath — before the sha256 of up to
//     4 MiB, so a malformed fingerprint costs nothing, and (see below) so the
//     traversal check is the one that decides it;
//  4. the CONTENT check: the bytes must hash to the fingerprint they claim;
//  5. the COMPILE check, which is the ack's whole value (see opWASMBlobPutName);
//  6. only then, the write.
//
// STEP 3 IS THE PATH-TRAVERSAL GATE and it is not optional even though step 4
// looks like it subsumes it. A fingerprint now arrives over the WIRE and is
// joined onto a filesystem path, which is the same class of admin-reachable
// arbitrary-file-write as the op-name traversal fixed earlier — and filepath.Join
// RESOLVES a traversal rather than failing on it. It happens that the write below
// goes through writeWASMBlob, which addresses the file by the hash it COMPUTES
// rather than by the claimed string, so a traversing fingerprint could not reach
// a path even if this check were absent; that is a property of one call site, not
// of the op, and the next person to add a second write here would not inherit it.
// Routing every wire fingerprint through the one validator is what keeps the
// guarantee attached to the input instead of to the caller.
//
// It takes NO lock. Two concurrent puts of the same fingerprint write identical
// content through atomicWriteFile's temp-then-rename, and two puts of different
// fingerprints address different files, so the content addressing makes the
// store safe for concurrent writers by construction.
func (n *Node) handleWASMBlobPut(args []byte) ([]byte, error) {
	if len(args) > maxWASMBlobPutFrame {
		return nil, fmt.Errorf("%w: put frame is %d bytes, over the %d-byte limit (a %d-byte fingerprint plus the %d-byte module cap)",
			ErrWASMBlobRefused, len(args), maxWASMBlobPutFrame, wasmBlobFPHexLen, maxDynamicWASMBytes)
	}
	if len(args) < wasmBlobFPHexLen {
		return nil, fmt.Errorf("%w: put frame is %d bytes, too short to carry a %d-byte fingerprint",
			ErrWASMBlobRefused, len(args), wasmBlobFPHexLen)
	}
	fp, body := string(args[:wasmBlobFPHexLen]), args[wasmBlobFPHexLen:]
	// Called for its VALIDATION, not for the path: wasmBlobPath is the single
	// place that decides what a fingerprint may look like, and every path that
	// joins one goes through it. See the traversal note above.
	if _, err := wasmBlobPath(n.cfg.DataDir, fp); err != nil {
		return nil, fmt.Errorf("%w: %v", ErrWASMBlobRefused, err)
	}
	if err := storeWASMBlobVerified(n.cfg.DataDir, fp, body); err != nil {
		return nil, err
	}
	// THE ESCAPE HATCH'S SECOND HALF. Storing the file is not what unblocks a
	// parked group — resolveModuleForInvoke asks the RUNTIME, not the filesystem,
	// because it runs on an apply goroutine and must not do I/O. So a put that
	// arrives AFTER the marker (the operator-remedy case this op is documented
	// for) has to instantiate the module too, or the group would sit parked on a
	// module the node now holds. On the pre-registration push path this is a
	// no-op: the marker has not applied yet, so nothing references the blob and
	// applyWASMRegistration materializes it in a moment.
	var sum [sha256.Size]byte
	copy(sum[:], mustDecodeWASMBlobFP(fp))
	n.installArrivedWASMBlob(sum)
	return nil, nil
}

// mustDecodeWASMBlobFP converts an ALREADY-VALIDATED lower-case hex fingerprint
// to bytes. Callers must have passed it through wasmBlobPath first — which every
// wire path does, as the traversal gate — so a decode failure here would mean
// that gate had been bypassed, and returning a zero fingerprint (which matches no
// registration) is the safe answer rather than a panic on a wire input.
func mustDecodeWASMBlobFP(fp string) []byte {
	raw, err := hex.DecodeString(fp)
	if err != nil || len(raw) != sha256.Size {
		return make([]byte, sha256.Size)
	}
	return raw
}

// storeWASMBlobVerified is the acceptance rule for a blob whose fingerprint has
// already been format-checked: it must hash to fp and it must compile here,
// and only then is it written.
//
// It is shared by handleWASMBlobPut and by the registering node's own local
// store (pushWASMBlob), so the node that ACCEPTS a registration holds itself to
// exactly the acceptance rule it imposes on every peer. Without the sharing, the
// coordinator would be the one node in the cluster whose wasmtime never had to
// agree that the module compiles.
func storeWASMBlobVerified(dataDir, fp string, b []byte) error {
	sum := ops.WASMBlobFingerprint(b)
	if got := hex.EncodeToString(sum[:]); got != fp {
		return fmt.Errorf("%w: %d bytes hash to %s, not to the claimed fingerprint %s",
			ErrWASMBlobRefused, len(b), got, fp)
	}
	// THE COMPILE IS A VERIFICATION, NOT AN INSTALL. The module is closed
	// immediately: nothing is added to the runtime and no op is registered here,
	// because a blob carries no op name, Kind or key extractor and there is
	// nothing it could be installed AS. Those arrive with the registration, whose
	// apply path (applyWASMRegistration) does the installing.
	m, err := wasm.Compile(b)
	if err != nil {
		return fmt.Errorf("%w: module %s does not compile on this node: %v", ErrWASMBlobRefused, fp, err)
	}
	_ = m.Close() //nolint:errcheck,gosec
	if _, err := writeWASMBlob(dataDir, b); err != nil {
		return err
	}
	return nil
}

// handleWASMBlobGet serves the bytes for one fingerprint out of this node's
// content-addressed blob store.
//
// READ opWASMBlobGetName's CONTRACT BEFORE CHANGING THIS FUNCTION. Its entire
// body is a length check, a string conversion, and readWASMBlob — no lock, no
// wasmState, no runtime, no shard, no Raft — and that is a deliberate property
// this stage exists to establish, not an accident of it being small.
//
// The length check is the frame bound: a get payload is exactly a fingerprint,
// so anything else is refused before any work. It is also what makes the format
// canonical (see wasmBlobFPHexLen). readWASMBlob then applies the two checks that
// matter: wasmBlobPath refuses any fingerprint that is not exactly 32 lower-case
// hex bytes — the traversal gate, now on a WIRE input — and the blob's contents
// are re-hashed against its own name, so this node can never serve bytes that
// are not the bytes the caller asked for.
//
// A fingerprint this node does not hold is an ordinary error (errWASMBlob),
// which is the correct answer and not a failure of anything: it means "ask
// someone else".
func (n *Node) handleWASMBlobGet(args []byte) ([]byte, error) {
	if len(args) != wasmBlobFPHexLen {
		return nil, fmt.Errorf("%w: get payload is %d bytes, not a %d-byte fingerprint",
			ErrWASMBlobRefused, len(args), wasmBlobFPHexLen)
	}
	b, err := readWASMBlob(n.cfg.DataDir, string(args))
	if err != nil {
		return nil, fmt.Errorf("cluster: %s: %w", opWASMBlobGetName, err)
	}
	return b, nil
}

// wasmBlobPushMember is one target of the pre-registration push.
type wasmBlobPushMember struct {
	nodeID     string
	serverAddr string
}

// wasmBlobPushTargets is the member set the push must reach: every node in the
// cluster EXCEPT this one.
//
// IT IS THE UNION of meta Members and the static cfg.Peers, deduplicated by node
// id, not a preference of one over the other. Preferring Members and consulting
// cfg.Peers only when Members is EMPTY — which is what this did first — leaves a
// node that config knows about but that has not yet been published in Members
// neither pushed to NOR named in the skip report. That is invisibility, which is
// the one failure shape this must not have: a registration may
// whose bytes went nowhere be the thing every group's log points at, so a member
// the push never considered has to be impossible, not merely unlikely. Members
// being empty outright (the bootstrap OpSetMembers commit lost to early election
// churn, the case raftToServerAddr already documents) is then just the degenerate
// end of the same union.
//
// Members WINS ON THE ADDRESS where both know a node, because it is the live
// table and cfg.Peers is frozen at startup.
//
// STATE THE COST OF THE UNION, since it is not free: a node REMOVED from the
// cluster leaves Members but stays in this process's cfg.Peers until it is
// restarted with new config, so every registration keeps pushing to it and names
// it in the skip report once it is gone. That is noise, and it is the direction
// to be wrong in — a stale name in a report is diagnosable, a member silently
// missing every module's bytes is not.
//
// A member with no resolvable ServerAddr is NOT dropped silently; it is returned
// with an empty serverAddr so pushWASMBlob reports it as skipped by name.
func (n *Node) wasmBlobPushTargets() []wasmBlobPushMember {
	var members []Peer
	if n.meta != nil {
		members = n.meta.FSM.State().Members
	}
	out := make([]wasmBlobPushMember, 0, len(members)+len(n.cfg.Peers))
	seen := make(map[string]struct{}, len(members)+len(n.cfg.Peers))
	// Members first so its (live) ServerAddr is the one recorded for any node
	// both tables carry.
	for _, src := range [][]Peer{members, n.cfg.Peers} {
		for _, p := range src {
			if p.NodeID == n.cfg.NodeID {
				continue
			}
			if _, dup := seen[p.NodeID]; dup {
				continue
			}
			seen[p.NodeID] = struct{}{}
			out = append(out, wasmBlobPushMember{nodeID: p.NodeID, serverAddr: p.ServerAddr})
		}
	}
	return out
}

// pushWASMBlob is THE PUSH PHASE: before a registration is broadcast into any
// group's log, this node stores the module's bytes locally and delivers them to
// every member of the cluster it can reach.
//
// It runs on the PROPOSE side (Node.Call's __register_wasm__ intercept), never
// on an apply path, and it takes no lock.
//
// WHAT "REACHABLE" MEANS HERE — stated exactly, because the whole tolerance rule
// turns on it. A member is REACHED iff this node obtained a VERDICT ON THE MODULE
// from it within wasmBlobPushTimeout. A completed response frame is necessary for
// that and is not sufficient: a peer on an older build answers promptly with
// "op not registered" without the handler that renders the verdict ever running,
// and that is a skip, not a refusal. Everything that leaves this node without a
// verdict — no server address, client construction, dial refused, TLS failure,
// timeout, EOF mid-response, a peer that does not know the op — is UNREACHABLE.
// See wasmBlobPeerRefused, which draws the line.
//
// The line is drawn at "did the peer render a verdict", not at "did the call
// succeed", and that is the only place it can be drawn:
//
//   - a REACHED-AND-REFUSED member FAILS THE REGISTRATION. This is the
//     compile-verify doing its job: a node whose wasmtime rejects the module has
//     told us so, and letting the registration proceed would convert that into
//     an apply-time discovery on that node — a group that cannot
//     move. The client is told which member refused and why;
//   - an UNREACHABLE member DOES NOT block the registration. It has rendered no
//     verdict, so there is nothing to honour, and refusing would mean any node
//     being restarted could stop the cluster from registering a module. It is
//     reported as skipped and fetches the blob on demand later.
//
// STATE THE RESIDUE PLAINLY: an unreachable member's wasmtime has NOT agreed
// that the module compiles, so for that member the disagreement is still an
// apply-time discovery. The push converts registration-time refusal from
// impossible to available; it cannot make it universal, because a node that
// cannot be asked cannot answer. What it does guarantee is that the failure is
// confined to members this node could not reach at all, and that they are NAMED
// in the reply rather than assumed healthy.
//
// ############ THE DURABILITY FLOOR — A STAGE 4 CORRECTION ############
//
// The recorded design said a blocked replica could always fetch from its OWN
// GROUP'S Raft peers, because "those provably have the bytes — they applied the
// same marker". THAT REASONING DIED WITH THIN MARKERS. Once the marker carries no
// module, applying it no longer implies holding one, so a group peer is exactly
// as likely to be missing the blob as the blocked node is.
//
// the own review had already named the consequence and deferred it: the
// push has NO FLOOR, so if every member is unreachable the registration proceeds
// anyway with the bytes on exactly ONE node. That was harmless while the marker
// carried the module — every replica could re-derive it from the entry. Under
// thin markers it is the blocking condition, and if that one node dies the blob
// is unreachable, every group blocks forever, and there is no source anywhere in
// the cluster for the fetch to succeed from. A registration that cannot ever be
// executed would have been accepted, replicated and made permanent.
//
// SO: a MAJORITY OF CLUSTER MEMBERS must hold the bytes before any marker is
// proposed, and a registration that cannot reach the floor FAILS LOUDLY.
//
// WHY A MAJORITY OF MEMBERS, and why not the alternatives:
//
//   - it is EXACTLY the liveness assumption the cluster already makes. Two
//     majorities of one set intersect, so if a majority holds the blob and a
//     majority is reachable, some reachable node holds it. "A majority of members
//     is reachable" is what meta-Raft needs to commit anything at all, so this
//     adds no new availability requirement: whenever the cluster can serve, a
//     fetch can succeed;
//   - PER-GROUP VOTER QUORUM was the obvious alternative and is worse in both
//     directions. A registration is broadcast to EVERY group, so the floor would
//     have to be a majority of every group's voters — which, with overlapping
//     placements, is close to "every node" and far stricter than what is needed,
//     while also making the floor depend on placement state that changes under
//     it. The fetch, meanwhile, is deduplicated per FINGERPRINT and is therefore
//     not attached to a group at all (see fetchWASMBlobOnce), so a per-group
//     floor would not even be the set the fetcher searches;
//   - ONE ACK (any peer) is not enough: it survives no failure at all.
//
// STATE THE FLOOR'S LIMIT, because it is real and it is not closed here.
// MEMBERSHIP CHURN ERODES IT. The floor is a majority of the member set AS IT WAS
// when the marker was proposed. Push to {A,B} of {A,B,C}, then grow to
// {A,B,C,D,E}, and the majority {C,D,E} contains no holder — so a blocked node
// that can reach only {C,D,E} has no source. Raft solves the analogous problem
// for its log with joint consensus; blobs have no such mechanism, and building
// one belongs with retirement/GC, which is where the set of live blobs
// first has to be tracked at all. Until then the mitigation is operational: a
// node joining an existing cluster fetches on demand and, if it cannot, says so
// loudly through WASMBlockStats — and __wasm_blob_put__ moves the file by hand.
//
// The MEMBER SET is the meta Members table when it is populated, read BEHIND A
// META READ BARRIER (see wasmMemberSet) so "authoritative" is a property and not
// an assumption — an unbarriered local read is only as current as this node's
// apply lag, and a short denominator is a short floor. When the barrier cannot
// complete the DENOMINATOR widens to the conservative cfg.Peers union while the
// set itself stays strict; the registration is never refused for the barrier's
// own sake, because that would make a rolling restart a registration outage. Only
// when the table is empty — the bootstrap window wasmBlobPushTargets already
// documents — does the floor fall back to cfg.Peers wholesale, because with no
// live table there is nothing better to count.
//
// The returned string is the human-readable skip report — empty when every
// member acked. Node.Call returns it as the registration's reply payload.
func (n *Node) pushWASMBlob(r ops.WASMRegistration, module []byte) (string, error) {
	fp := ops.WASMBlobHex(r.Blob)

	// LOCAL FIRST, and it is held to the same rule every peer is (see
	// storeWASMBlobVerified). A registration whose module this node cannot compile
	// must fail here, before a single peer is contacted and before anything enters
	// a log. It also makes this node the first HOLDER, which is what the floor
	// below counts as one.
	if err := storeWASMBlobVerified(n.cfg.DataDir, fp, module); err != nil {
		return "", fmt.Errorf("cluster: register wasm %q: storing module %s locally: %w", r.Name, fp, err)
	}

	// The member set is resolved BEFORE the targets, and behind a read barrier, so
	// both reads of the meta FSM see the same membership. An unbarriered read is
	// what makes the denominator gameable by nothing more than lag: 3 of a real 5
	// makes the floor 2, and a marker gets proposed with 2 of 5 holding — a floor
	// that no longer means what it says. `total` is NOT len(members) when the
	// barrier could not complete; see wasmMemberSet.
	members, total := n.wasmMemberSet()

	targets := n.wasmBlobPushTargets()
	if len(targets) == 0 {
		return "", nil // single-node, or a cluster of one member: self IS the majority.
	}
	payload := encodeWASMBlobPut(fp, module)

	holders := 1 // self, stored and compile-verified above
	var skipped []string
	for _, m := range targets {
		err := n.putWASMBlobOnPeer(m, payload)
		if err == nil {
			n.wasmBlobPushAcks.Add(1)
			// Only acks from nodes in the AUTHORITATIVE member set count toward the
			// floor. A target that is in this process's stale cfg.Peers but not in
			// Members is a node the cluster has removed; pushing to it is harmless
			// (and the union exists so it is never silently skipped), but counting
			// its ack would let a removed node satisfy a floor stated over live
			// members.
			if _, isMember := members[m.nodeID]; isMember {
				holders++
			}
			continue
		}
		if wasmBlobPeerRefused(err) {
			return "", fmt.Errorf("%w: register wasm %q: member %s refused module %s: %v",
				ErrWASMBlobRefused, r.Name, m.nodeID, fp, err)
		}
		n.wasmBlobPushSkips.Add(1)
		skipped = append(skipped, fmt.Sprintf("%s: %v", m.nodeID, err))
	}

	// THE FLOOR. Strictly greater than half, i.e. a genuine majority — on an
	// even-sized cluster exactly half is NOT enough, because two halves need not
	// intersect and the whole argument is an intersection argument.
	if holders*2 <= total {
		n.wasmBlobFloorFailures.Add(1)
		return "", fmt.Errorf("%w: register wasm %q: module %s reached only %d of %d cluster members, short of the majority required before a registration may be proposed (a marker carries no module, so a version no majority holds could become unfetchable and block every shard group forever); unreachable: %s",
			ErrWASMBlobRefused, r.Name, fp, holders, total, strings.Join(skipped, "; "))
	}

	if len(skipped) == 0 {
		return "", nil
	}
	return fmt.Sprintf("cluster: register wasm %q: module %s pushed to %d of %d members; unreachable (they will fetch it on demand): %s",
		r.Name, fp, len(targets)-len(skipped), len(targets), strings.Join(skipped, "; ")), nil
}

// wasmMemberSet is the AUTHORITATIVE cluster membership the durability floor is
// counted against, as a set of node ids INCLUDING this node.
//
// It is the meta Members table when that is populated, and the cfg.Peers union
// only when it is not. That asymmetry is deliberate and is the opposite of
// wasmBlobPushTargets' union rule, for a reason worth stating: the push wants to
// reach EVERYONE it might need to, so a superset is right there; the floor is a
// claim about a majority, and a superset there would inflate the denominator with
// nodes the cluster has removed — making a healthy cluster unable to register a
// module because a long-dead entry in this process's static config is counted as
// a member that must be reached.
//
// ###### IT BARRIERS FIRST, BECAUSE THE DENOMINATOR IS THE WHOLE CLAIM ######
//
// State().Members is a LOCAL read of this node's meta FSM, and a local read of a
// replicated FSM is only as current as this node's apply lag. A coordinator whose
// MetaFSM is three entries behind sees 3 members of a real 5, computes a floor of
// 2, and proposes a marker with 2 of 5 holding the bytes — the arithmetic is
// sound, the set it is stated over is not, and the resulting registration is
// exactly the unfetchable-forever one the floor exists to prevent. metaReadBarrier
// is what turns "authoritative" from an aspiration into a property.
//
// ###### IT DOES NOT FAIL CLOSED ON THE BARRIER, AND THAT WAS MEASURED ######
//
// Refusing the registration when the barrier fails is the obvious reading of
// "consistency over availability", and it is wrong here — not as a judgement call
// but because it breaks a property this package already gates. The barrier reaches
// the meta leader over its CLIENT-FACING port, so a coordinator that is a meta
// follower cannot complete it while the meta leader's server is down. Failing
// closed therefore turns any rolling restart into a registration outage, which is
// exactly what TestWASMBlobPushToleratesAnUnreachableMember exists to prevent, and
// it does so nondeterministically — it depends on which node happens to lead the
// meta group. That test caught it.
//
// So the barrier is used for what it is good for and nothing more: when it
// succeeds, Members is confirmed fresh and the denominator is exactly len(Members).
// When it fails, we do not know how much larger the true membership is, and the
// denominator becomes max(len(Members), len(cfg.Peers ∪ self)) — the most
// conservative statement available from purely local information.
//
// THE NUMERATOR STAYS STRICT EITHER WAY. cfg.Peers widens the DENOMINATOR only; it
// never joins the ack-eligibility set, so a node the cluster removed still cannot
// satisfy a floor stated over live members (see the isMember check in
// pushWASMBlob). Conservative below the line, strict above it — the direction that
// errs toward durability on both halves.
//
// The inflation this function's own asymmetry rule warns about — a node removed
// months ago, still in one operator's static config, raising the denominator on
// that node alone — is therefore confined to the window in which the meta leader
// is ALSO unreachable, instead of applying always. That is a real cost, and it is
// the smaller of the two.
//
// It returns the ack-eligibility SET and the denominator separately because after
// a failed barrier they are deliberately not the same size.
func (n *Node) wasmMemberSet() (members map[string]struct{}, total int) {
	barriered := n.metaReadBarrier(time.Now().Add(wasmMemberSetBarrierTimeout)) == nil

	out := make(map[string]struct{}, 4)
	out[n.cfg.NodeID] = struct{}{}
	if n.meta != nil {
		for _, p := range n.meta.FSM.State().Members {
			out[p.NodeID] = struct{}{}
		}
	}
	if len(out) == 1 {
		// No live table (the bootstrap window raftToServerAddr documents): fall back
		// to static config, which is the only membership statement available. This is
		// the one case where cfg.Peers DOES join the ack-eligibility set, because
		// there is no authoritative set to be strict about.
		for _, p := range n.cfg.Peers {
			out[p.NodeID] = struct{}{}
		}
		return out, len(out)
	}
	if barriered {
		return out, len(out)
	}
	// Unconfirmed: widen the denominator, not the set.
	conf := map[string]struct{}{n.cfg.NodeID: {}}
	for _, p := range n.cfg.Peers {
		conf[p.NodeID] = struct{}{}
	}
	if len(conf) > len(out) {
		return out, len(conf)
	}
	return out, len(out)
}

// wasmMemberSetBarrierTimeout bounds the meta read barrier the durability floor
// takes. It is generous relative to pbLeaseBarrierTimeout because nothing is
// waiting on the answer: this runs once per __register_wasm__, on an admin path
// that is about to push a whole module to every member anyway.
//
// It is also the delay a registration pays when the meta leader is unreachable,
// since the fallback above is only reached once this elapses — which is the other
// reason not to make it much larger.
const wasmMemberSetBarrierTimeout = 5 * time.Second

// putWASMBlobOnPeer delivers the payload to one member, bounded by
// wasmBlobPushTimeout.
//
// It addresses the member DIRECTLY by its server address rather than going
// through forwardTimeout, because forwardTimeout rotates over a SHARD's owners
// and this push is not about a shard at all — every member needs the bytes,
// including one that owns no group. Reaching a different node than the one named
// would defeat the point: the skip report would name a member that was never
// asked.
func (n *Node) putWASMBlobOnPeer(m wasmBlobPushMember, payload []byte) error {
	if m.serverAddr == "" {
		return errors.New("no server address in the member table")
	}
	cl, err := n.peerClient(m.serverAddr)
	if err != nil {
		return err
	}
	ctx, cancel := context.WithTimeout(context.Background(), wasmBlobPushTimeout)
	defer cancel()
	_, err = cl.Call(ctx, opWASMBlobPutName, payload)
	return err
}

// wasmBlobPeerRefused reports whether err is a VERDICT ON THE MODULE from the
// peer rather than a failure to obtain one — the distinction pushWASMBlob's
// tolerance rule turns on.
//
// THE QUESTION IS "DID THE PEER JUDGE THESE BYTES", NOT "DID A RESPONSE FRAME
// COME BACK", and an earlier version of this function conflated the two. A
// completed response frame is NECESSARY for a verdict and is not sufficient for
// one: a peer can answer, promptly and correctly, without the handler that
// renders the verdict having run at all.
//
// The enumeration below is of the cases known to be REACHABLE, and it is not
// claimed to be exhaustive — the case that made this function wrong was one an
// earlier "the three cases are exactly ..." enumeration had left out.
//
// REFUSALS — the peer judged the module and said no:
//
//   - RemoteError from handleWASMBlobPut: a bad hash, a module that does not
//     compile there, an oversized frame. This is the one the compile-verify
//     exists to produce, and keeping it fatal to the registration is the whole
//     point of the push (see pushWASMBlob);
//   - ErrUnauthorized: the peer's authorizer refused the internal identity. The
//     module was not judged, but this is a cluster misconfiguration that would
//     otherwise make every registration silently push to NOBODY while reporting
//     members as merely unreachable, so it fails the registration deliberately;
//   - ErrNotFound: not produced by this op today (it is dispatched off adminOps,
//     which never answers StatusNotFound), but an unexpected verdict is
//     classified as a refusal so the rule fails closed.
//
// NOT A REFUSAL — the peer answered, but never reached the handler:
//
//   - A PEER ON AN OLDER BUILD, which is the reachable case the enumeration used
//     to omit and the one that matters most in production. Such a peer has no
//     __wasm_blob_put__ in adminOps and none in the ops registry, so Node.Call
//     answers ErrUnknownOp; server.clientFacingErr keeps that message unredacted
//     and returns StatusError, which the client maps to a *client.RemoteError.
//     Read as a refusal, that turns EVERY rolling upgrade through the commit that
//     introduced this file into a cluster-wide registration outage — precisely
//     the "any node being restarted could stop the cluster from registering a
//     module" failure the tolerance rule exists to prevent — and reports a
//     compile refusal for what is version skew. The peer rendered no verdict, so
//     it is skipped and NAMED, exactly like an unreachable one.
//
// It is matched by SUBSTRING, for the reason server.clientFacingErr and
// httpapi.statusForError match the same substring: the refusal is raised on the
// peer and reaches this node as a STRING, so no sentinel identity survives. The
// other two errors carrying that substring (shard.ErrOpNotRegistered,
// ErrWASMOpNotInThisGroup) cannot arrive on THIS op's reply — it never enters a
// registry lookup that could produce them — and if one somehow did, it would mean
// the same thing: the handler did not run, so no verdict exists to honour.
//
// The refusal marker is checked FIRST because the substring is otherwise
// FORGEABLE BY THE PAYLOAD. A compile failure's text quotes module-controlled
// names, so a module crafted to fail on one node with "op not registered" in the
// message would downgrade that node's genuine refusal to a skip. Every real
// refusal wraps ErrWASMBlobRefused and therefore carries
// ops.WASMRegistrationRefusedMsg, which nothing about an unknown op does.
//
// Everything else — dial refused, TLS failure, context deadline, EOF mid-response
// — leaves this node with NO verdict, and is therefore unreachable.
func wasmBlobPeerRefused(err error) bool {
	var re *client.RemoteError
	if errors.As(err, &re) &&
		!strings.Contains(re.Msg, ops.WASMRegistrationRefusedMsg) &&
		strings.Contains(re.Msg, wasmBlobUnknownOpMsg) {
		return false
	}
	return errors.As(err, &re) ||
		errors.Is(err, client.ErrUnauthorized) ||
		errors.Is(err, client.ErrNotFound)
}
