// SPDX-License-Identifier: Apache-2.0

package cluster

import (
	"errors"
	"fmt"

	"github.com/rostamlabs/rostam/ops"
	"github.com/rostamlabs/rostam/wasm"
)

// ErrWASMOpNotInThisGroup is returned by Call when a dynamically registered WASM
// op is invoked against a shard group THIS NODE HOSTS whose Raft log is not
// known to carry the op's registration. It is a client-visible, retryable
// condition — the registration may still be in flight to that group — not an
// internal error, and emphatically not a reason to halt anything.
var ErrWASMOpNotInThisGroup = errors.New("cluster: op not registered in this shard group yet")

// ErrWASMUpdateUnsupported is returned when a __register_wasm__ names an op that
// is already registered and CHANGES ITS CONTRACT — its Kind or its
// Kind. Changing only the module bytes is supported and is not
// this error.
//
// It is raised in two places, with different strengths: at PROPOSE time against
// this node's node-wide install state (checkWASMUpdateGate, best-effort — that
// state is allowed to lag), and at APPLY time against one shard group's own
// binding (checkWASMGroupContract, exact and identical on every replica of that
// group).
//
// The sentinel's text is load-bearing: server.clientFacingErr and
// httpapi.statusForError key off it to keep the message unredacted (and a 400
// rather than a 500) across the stringifying Raft/RPC boundary, where no sentinel
// identity survives. That coupling is now a COMPILE-TIME one — the text lives in
// ops.WASMUpdateUnsupportedMsg and every user references the const — so rewording
// it in one place cannot silently start redacting the refusal in another.
var ErrWASMUpdateUnsupported = errors.New("cluster: " + ops.WASMUpdateUnsupportedMsg)

// ErrWASMRegistrationRefused carries the __register_wasm__ refusals that are
// about the PAYLOAD rather than about its name (ops.ErrWASMOpNameUnsafe) or about
// an attempted update (ErrWASMUpdateUnsupported): an encoded frame over
// maxWASMRegistrationFrame, a frame that does not decode at all, a module over
// maxDynamicWASMBytes, and a Kind byte outside {OpReadOnly, OpReadWrite}.
//
// Its text is load-bearing for the same reason ErrWASMUpdateUnsupported's is:
// every one of these is a caller mistake with an actionable remedy, and every one
// of them reaches the client only as a STRING (the refusal may be raised on a peer
// and stringified across the Raft/RPC boundary), so server.clientFacingErr and
// httpapi.statusForError can only recognise it by substring. The substring is the
// const ops.WASMRegistrationRefusedMsg, which makes the coupling compile-time.
// Without it these refusals fall into the catch-all and the caller is told
// "internal error" for a payload it can fix.
var ErrWASMRegistrationRefused = errors.New("cluster: " + ops.WASMRegistrationRefusedMsg)

// checkWASMUpdateGate refuses, at PROPOSE time, a __register_wasm__ that would
// change the CONTRACT of a name this node already has installed from replication:
// its Kind. A registration that changes only the module
// (its bytes, export symbol or fuel cap) is ALLOWED.
//
// WHY BYTES UPDATES ARE NOW SAFE. They used not to be, because the effective
// module version was NODE-WIDE: the ops.Registry entry is keyed on the op NAME
// and its handler closure captured ONE ModuleID, replaced the instant ANY group
// this node hosts applied a superseding registration — while registrations arrive
// through PER-GROUP logs that commit at independent times. A node whose local
// state was entirely self-consistent would therefore propose an invocation that a
// peer which had already applied the update executed with DIFFERENT bytes, with
// both applies succeeding, no error to classify and no halt. Nothing checkable at
// propose time could see a peer's version, which is why this gate could only ever
// be best-effort, and why re-keying the route gate's evidence on the content
// fingerprint was tried and did not close it either.
//
// What closed it is that the version is no longer node-wide. The module executing
// a committed entry in group g is resolved from g's OWN binding
// (wasm.Runtime.resolveModuleForInvoke, fed by installedWASM.groups), which is a
// fold over g's log prefix — so every replica of g picks the same version for the
// same entry, whatever else it hosts and however far along it is. A lagging node
// proposing into g is then harmless: the entry executes with g's committed
// version on every replica of g, including the proposer.
//
// WHY KIND STILL CANNOT CHANGE. It is read on the PROPOSE side, before any group
// is known — it decides whether there is a replicated proposal at all. So it
// cannot be resolved per group without knowing the group first, and the group
// cannot be known without resolving it. Only the module is consumed exclusively
// at apply time, where tx.ShardIndex() already names the group. Kind is therefore
// frozen at first registration; a registration that changes it stays a
// new-op-name operation. See checkWASMGroupContract.
//
// The key extractor was the other half of this argument — it COMPUTES the group
// index, so it is even less resolvable per group — and it is now CONSTANT rather
// than frozen: WASMRegistration has no field for it. See
// ops.WASMKeyExtractorHandle for why nothing weaker was safe.
//
// THIS CHECK IS BEST-EFFORT, unchanged. It is evaluated against the RECEIVING
// NODE's node-wide install state, which is exactly the state allowed to lag: a
// node that has not yet applied the op cannot recognise the incoming registration
// as a contract change and will accept it. What backs it up now is the APPLY-TIME
// per-group refusal (checkWASMGroupContract), which is not best-effort: it is
// judged against the applying group's own binding, so every replica of that group
// refuses identically. Between them, a contract change is refused loudly on the
// common path and cannot take effect in any group that already has a binding.
//
// AN APPLY-TIME REFUSAL IS NOW SOUND, WHICH IT WAS NOT BEFORE, and this reversal
// is the biggest conceptual payoff of per-group binding. This comment used to say
// the check MUST NOT be moved to apply time, because a state-dependent rejection
// there is itself a divergence source: the replica that already holds the op
// rejects the entry while one that does not accepts it, and the two then run
// different modules under one name — the trap applyWASMRegistration documents
// under "WHY NOT reject a re-registration if the name already exists". That
// argument is CORRECT FOR NODE-WIDE STATE and is why this particular check stays
// where it is. It stops holding for PER-GROUP state: a refusal keyed on group g's
// own binding is a pure function of g's log prefix, and every replica of g has
// applied that same prefix, so all of them reach the same verdict on the same
// entry. Determinism, not the choice of propose-vs-apply time, was always the
// real requirement; node-wide state simply could not supply it.
//
// IDENTICAL RE-REGISTRATION IS ALLOWED, because the documented recovery path
// depends on it. broadcastWASMRegistration attempts every group even after one
// fails and returns an error inviting a retry; that retry re-sends the SAME
// registration, and it is the only way the registration reaches a group the first
// broadcast starved. Refusing it would convert a routine partial broadcast into a
// permanently unroutable op. It is allowed here for a stronger reason than
// before: it does not even reach the comparison, since an identical registration
// changes neither Kind nor the extractor.
//
// AN EPOCH-ONLY BUMP IS NOW ALLOWED, where it used to be refused. The old rule
// ("the accepted registration for a name is the exact struct accepted first") was
// a consequence of updates being unsupported at all: Epoch exists to order
// updates, so bumping it was an attempted update by definition. With updates
// supported, an Epoch bump is just an update whose module happens to be
// unchanged, and it converges like any other.
//
// CONFIG-INSTALLED MODULES ARE NOT GUARDED. A module from cfg.WASMModules is
// node-LOCAL operator state that was never proposed into anyone's log, and
// applyWASMRegistration deliberately lets a replicated registration override it
// (see wasmMeta.Source — a config module must never win, or a node that happens
// to have it configured would run different bytes from one that does not).
// Refusing here on a config install would break that override on exactly the
// subset of nodes that have the config, which is worse than the update it would
// prevent.
//
// THE FORWARDED LEG RUNS THIS CHECK TOO, and that is a deliberate reversal of
// what this comment used to say. handleRegisterWASMShard drives its group through
// callHostedShard rather than Node.Call, so it used to skip the gate entirely —
// and since it is dispatched off n.adminOps BEFORE Node.Call's __register_wasm__
// intercept, that made it a complete bypass reachable by any admin-authenticated
// external client, not only by peers. Driving ONE group's log to a CONTRACT-
// changing registration is the maximally divergent shape available: one group
// serving the op read-only, or routing it to a different group, while every other
// group does not. The cost of re-evaluating is a peer that has already applied the
// original refusing a leg its (staler) sender accepted, which turns a would-be
// silent contract change into a loud partial failure the client is told about. A
// well-behaved retry is unaffected: it re-sends the identical struct, whose
// contract compares equal on every node that holds it and which passes trivially
// on every node that does not.
//
// STATE THE REST OF THAT COST, so a future reader does not rediscover it as a
// surprise and mistake it for a regression. The forwarded leg used to be the last
// REPAIR path for a raced first-time registration: when A and B are registered
// concurrently under one name, the (Epoch, fingerprint) maximum converges only
// over the set of registrations each node actually RECEIVES, and a leg forwarded
// to a peer was one of the ways the losing node still got to see the winner. With
// the gate on that leg, once peer P holds B it refuses a CONTRACT-DIFFERING A leg
// for every group P leads, so such an A never reaches P and convergence is over a
// strictly smaller set. This is narrower than it used to be — a bytes-differing A
// now passes the gate and does reach P — but it is not gone.
func (n *Node) checkWASMUpdateGate(r ops.WASMRegistration) error {
	n.wasmApplyMu.Lock()
	cur, have := n.wasmState.installed[r.Name]
	n.wasmApplyMu.Unlock()
	if !have || !cur.replicated {
		return nil
	}
	if r.Kind == cur.reg.Kind {
		return nil // a module update: supported, per-group bound at apply time.
	}
	return fmt.Errorf("%w: op %q is registered on node %s as kind %d (epoch %d), and this registration declares kind %d; only the module may be updated in place — register the new contract under a NEW op name instead",
		ErrWASMUpdateUnsupported, r.Name, n.cfg.NodeID,
		uint8(cur.reg.Kind), cur.reg.Epoch, uint8(r.Kind))
}

// PER-GROUP VERSION BINDING — BUILT. The design recorded here is implemented;
// this is the map of where each piece lives, and of the one piece deliberately
// left out.
//
//   - wasm.Runtime keys modules by CONTENT (wasm.ModuleID: bytes, export name,
//     fuel budget) rather than by op name, so several versions of one op coexist
//     and an apply can name the one it needs. Module bytes are likewise content
//     addressed on disk (<dataDir>/wasm/blobs/<sha256>.wasm), so a version is
//     addressable and self-verifying on every node that has it, independent of any
//     op name.
//   - The __register_wasm__ hook maintains a group → op → version table from
//     tx.ShardIndex() (installedWASM.groups, folded by bindWith): the version that
//     group's log has committed as of that entry, as a maximum over the group's
//     ORDERED log prefix composed with the contract freeze. That composition is
//     order-DEPENDENT (first registration pins the contract), so the property is
//     PREFIX-DETERMINISM — same ordered prefix, same binding, on every replica of
//     the group — not permutation-invariance. See installedWASM.groups. It is
//     persisted in the sidecar
//     (wasmMeta.Bindings) and carried in that group's snapshot (wasmSnapshotBlob's
//     bindings section), so both catch-up routes reconstruct it. It is NOT
//     re-derivable: in durable mode fsm.Apply skips every entry at or below the
//     persisted applied index.
//   - The registry handler resolves the module version from tx.ShardIndex() at
//     apply time instead of from node-wide state
//     (wasm.Runtime.resolveModuleForInvoke). That is the step that closes the hole:
//     the version used to execute an entry in group g is a function of g's ORDERED
//     LOG PREFIX, so every replica of g — all of which applied that same sequence
//     — derives the same version for the same entry regardless of which other
//     groups it hosts or how far along them it is.
//   - Kind is FROZEN at first registration — an update may change the module, and
//     nothing else. It is read on the PROPOSE side to decide whether the
//     invocation is replicated at all; resolving it per group would require
//     knowing the group index first, which is what the key extractor COMPUTES.
//     The freeze is enforced at propose time (above) and, exactly, at apply time
//     against the group's own binding (checkWASMGroupContract).
//     The key extractor needs no freeze at all, and the freeze machinery does not
//     see it: ops.WASMRegistration HAS NO FIELD for it. A registration cannot
//     express a routing rule, so two registrations of one name cannot disagree
//     about one — the bad state is unrepresentable rather than refused, which is
//     why checkWASMUpdateGate and checkWASMGroupContract compare Kind alone.
//   - NOT BUILT — RETIREMENT of a superseded version. Every version any group has
//     ever bound stays instantiated in the runtime and stored on disk, because any
//     group still below the point where it was superseded may replay an entry that
//     needs it. Retiring one safely requires knowing that EVERY group this node
//     hosts has committed past it, which nothing tracks today. The cost is bounded
//     by the number of distinct registrations a deployment ever issues for a name,
//     not by traffic.
//   - NOT BUILT — Kind resolved per group at the point shard/fsm.go decides
//     whether a committed entry may run (errPBApplyReadOnly). The freeze makes a
//     group's Kind constant over its whole log prefix, so per-group and node-wide
//     Kind can only differ when two FIRST-TIME registrations of one name declare
//     different Kinds and land in different groups — a race no group has a prior
//     binding to refuse against. errPBApplyReadOnly stays classFatal for that
//     residue, which fails closed rather than diverging.
//   - CLOSED, AND NOT BY THIS MACHINERY — the KeyExtractorHandle half of the same
//     first-registration race. It used to be the one residue with NO apply-time
//     error at all: the extractor COMPUTES the group index, so two nodes that
//     ended on different extractors routed the op's invocations to DIFFERENT shard
//     groups, every apply succeeded on whichever replica set received them, and
//     the writes simply landed in different groups. Nothing surfaced. The
//     forwarded-leg gate above makes the differing-set precondition reachable
//     rather than theoretical (see "STATE THE REST OF THAT COST"), so it was a
//     live hole rather than a paper one.
//
//     It is closed by ops.WASMKeyExtractorHandle, which collapses the legal set to
//     ONE value on every path a registration can arrive by — propose, apply,
//     config, sidecar reload and Direct mode. "Two registrations of one name
//     declaring different extractors" is now unrepresentable rather than merely
//     refused, which is the only shape of fix available: there was no error to
//     classify, so no backstop could have been built here.

// wasmGateSnapshot is an immutable name → group → committed-version view,
// published by copy-on-write so Call reads it with one atomic load and no lock.
// The map and every map inside it are treated as read-only once published.
//
// It is wasm.GroupBindings, not a private type, because THE SAME VALUE serves
// two consumers that must never disagree: this package's propose-time route gate
// reads its KEY SET ("is group g's log known to carry a registration for X?"),
// and wasm.Runtime.resolveModuleForInvoke reads its VALUES at apply time ("which
// version does group g execute X with?"). Publishing one structure to both is
// what makes it impossible for the gate to call a group proven while the resolver
// finds no version for it.
type wasmGateSnapshot = wasm.GroupBindings

// wasmBindingSnapshot renders wasmState as the immutable value both consumers
// read. It is a free function so the exact structure a Node publishes can be
// built (and asserted on) without a Node.
func wasmBindingSnapshot(st *wasmState) wasmGateSnapshot {
	snap := make(wasmGateSnapshot, len(st.installed))
	for name, in := range st.installed {
		if !in.replicated {
			continue
		}
		groups := make(map[int]wasm.ModuleID, len(in.groups))
		for g, b := range in.groups {
			groups[g] = b.id
		}
		snap[name] = groups
	}
	return snap
}

// publishWASMGateLocked rebuilds and publishes the route-gate / per-group-binding
// snapshot from wasmState, to BOTH consumers: this node's Call path and the WASM
// runtime's apply-time resolver. Callers hold wasmApplyMu (every wasmState mutator
// already does).
//
// Publishing to both from one place is deliberate. The runtime's table is the
// authority for what EXECUTES; letting it be updated anywhere else would make it
// possible to install a registration without binding it, which is a classFatal
// halt (ops.ErrWASMNoGroupBinding) on the next committed invocation.
//
// Only REPLICATED installs appear. A config module is node-local operator state
// that was never proposed into anyone's log, so there is no group evidence to
// have and no group prefix to bind to: gating it would break operator-configured
// modules outright, and binding it would refuse every invocation of one. The
// resolver treats an op absent from the table as node-wide by contract — see its
// case (2).
func (n *Node) publishWASMGateLocked() {
	snap := wasmBindingSnapshot(n.wasmState)
	n.wasmGate.Store(&snap)
	// nil before finishWASMSetup has run: the shard FSM apply loops are already
	// alive at that point and can restore a snapshot into wasmState, and
	// finishWASMSetup republishes once the runtime exists.
	if n.wasmRT != nil {
		n.wasmRT.PublishGroupBindings(snap)
	}
}

// checkWASMRouteGate enforces the ROUTE GATE for one Call: it reports an error
// when name is a replicated WASM op, idx is a group this node HOSTS, and this
// node has no evidence that idx's log already carries the op's registration.
//
// THE INVARIANT. An invocation entry for op X may be proposed into shard group
// g's Raft log only if g's log ALREADY contains a registration for X at a lower
// index.
//
// WHY IT IS SUFFICIENT. An entry enters g's log only via a node that leads g,
// and leading g implies hosting g. If every node that hosts g refuses to propose
// an X-invocation until it knows g's log carries REG(X), then every INV(X) in
// g's log sits above a REG(X) in the SAME log. Raft replicas apply one log in
// index order, so every replica of g applies REG(X) — which installs X in its
// node-wide ops registry — before it ever applies INV(X). No replica of g can
// meet an invocation for an op it cannot look up. shard.ErrOpNotRegistered at
// apply time therefore means genuine registry divergence, which is exactly what
// the classFatal classification should catch, and nothing else.
//
// WHAT THE GATE DOES NOT ESTABLISH BY ITSELF (its exact scope — do not over-read
// the invariant above). It is keyed on the op NAME, and NAME-lookup is what
// avoids the halt. It is NOT what guarantees that two replicas of g execute the
// same invocation with the same MODULE; nothing checkable at PROPOSE time could
// be, because nothing at propose time can see a peer's version. That guarantee
// comes from the APPLY side: the same evidence this gate reads now carries the
// version g's log committed, and wasm.Runtime.resolveModuleForInvoke executes
// from it (see installedWASM.groups). The two are one structure precisely so the
// gate cannot call a group proven while the resolver has no version for it.
//
// THE TWO ARE COMPLEMENTARY AND BOTH ARE REQUIRED. The gate is what makes the
// resolver's lookup total: it puts REG(X) below every INV(X) in g's log, so a
// replica of g reaching an invocation has necessarily applied (or restored) the
// binding. That is why a lookup MISS is treated as a divergence signal
// (ops.ErrWASMNoGroupBinding, classFatal) rather than as a reason to fall back to
// a node-wide version.
//
// WHY THE GATE IS HERE AND NOT ON THE REGISTRY. The obvious-looking alternative
// — withhold the op from cfg.Ops until the node has applied the registration on
// every group it hosts — is UNSAFE, and the difference matters enough to record.
// The ops registry serves two masters: Node.Call reads it to route a proposal,
// and every group's FSM reads it to apply a committed entry. Delaying the second
// is a halt generator. Concretely, with NumShards=64: node M hosts only group 5,
// so it "activates" the instant group 5's registration applies, and routes an
// invocation into group 5's log. Node N hosts all 64 groups and is caught up on
// group 5 but still behind on group 63, so it has NOT activated. N's group-5 FSM
// reaches that invocation, looks the op up, misses, and halts under classFatal —
// with the registration sitting applied in its own state. Registry presence must
// track "can this node APPLY the op", which is as soon as anything installs it;
// only the right to PROPOSE is per-group, so only that is gated.
//
// WHAT IT COSTS. Nothing on the apply path, and one atomic load per Call on the
// propose path (nil until the first dynamic registration exists).
//
// PARTIAL BROADCAST (the case with no race in it at all).
// broadcastWASMRegistration attempts every group even after one fails, and the
// groups that accepted KEEP the registration, so an op can be live and routable
// while some group never received it. Under this gate every node hosting the
// starved group declines to propose into it, so no invocation ever enters its
// log; a node that does not host it forwards to one that does, which declines
// the same way, or fails Lookup outright and returns ErrUnknownOp. The client
// gets an error. Before the gate, that invocation entered the starved group's
// log and halted EVERY replica of it — a shard-wide outage from an ordinary
// partial registration.
//
// NON-HOSTED GROUPS ARE NOT GATED HERE. A node that does not host g cannot
// propose into g's log; it forwards, and the owner it forwards to applies its own
// gate. Gating a non-hosted group locally would be wrong as well as pointless:
// this node applies none of g's log, so it has no evidence about g and would
// refuse traffic for a group that is perfectly healthy.
//
// A NODE HOSTING ZERO GROUPS is therefore never gated at all. Every Call it
// receives is forwarded to an owner, which enforces the invariant for its own
// groups. That is safe by the same argument (it leads nothing, so it proposes
// nothing) and it is the behavior we want: such a node stays a working proxy.
//
// PLACEMENT CHANGES need no special handling, which is why there is none. The
// getShard call below is made at CALL time, so the instant AddShardOwner gives
// this node group g, group g starts being gated — and is refused until proven,
// because a group this node has just started hosting is a group whose log it has
// applied nothing of. It becomes proven the moment the joining store catches up,
// by either route: the InstallSnapshot that AddShardOwner-joined replicas rely on
// carries the per-group flag (see wasmSnapshotBlob), and a log-replay catch-up
// applies the registration entry itself. Note what is NOT done: the op is never
// REMOVED from the registry on a placement change. Removing it would halt this
// node on the next invocation replicated into any group it already hosts — the
// same trap as gating the registry, arrived at from the other side.
//
// The evidence itself never has to be retracted: it is a statement about a Raft
// log ("group g's log contains a registration for X"), and a Raft log only ever
// grows. See installedWASM.groups.
//
// READ-ONLY OPS ARE NOT GATED. The invariant is about what enters a Raft log,
// and shard.Store.Call serves an ops.OpReadOnly op from local state without
// proposing anything (see its kind switch). Nothing is replicated, so no replica
// can meet an entry it cannot look up, so there is nothing to protect — and
// gating them would deny reads that are perfectly safe to serve.
//
// The fast path is a single atomic load returning nil: the pointer stays nil
// until the first dynamic registration is installed, so a cluster that never
// uses dynamic WASM pays nothing measurable.
func (n *Node) checkWASMRouteGate(name string, kind ops.OpKind, idx int) error {
	if kind == ops.OpReadOnly {
		return nil
	}
	snap := n.wasmGate.Load()
	if snap == nil {
		return nil
	}
	groups, gated := (*snap)[name]
	if !gated {
		return nil
	}
	if _, proven := groups[idx]; proven {
		return nil
	}
	if n.getShard(idx) == nil {
		// Not hosted here: this node cannot propose into that group's log, so it
		// has nothing to gate. Forwarding is correct — the owner holds the
		// evidence and applies the same check.
		return nil
	}
	n.wasmGateRefusals.Add(1)
	return fmt.Errorf("%w: op %q, shard group %d on node %s (the registration has not reached this group's log; retry, or re-run RegisterWASM — it is idempotent)",
		ErrWASMOpNotInThisGroup, name, idx, n.cfg.NodeID)
}

// wasmGateStats renders the gate for Node.Stats.
//
// It reads the PUBLISHED snapshot rather than wasmState, so it takes no lock and
// cannot contend with the per-group FSM apply loops that hold wasmApplyMu. The
// snapshot is the same value checkWASMRouteGate consults, so what this reports
// is exactly what is being enforced.
func (n *Node) wasmGateStats() WASMGateStats {
	out := WASMGateStats{Refusals: n.wasmGateRefusals.Load()}
	snap := n.wasmGate.Load()
	if snap == nil {
		return out
	}
	out.ProvenGroups = make(map[string][]int, len(*snap))
	for name, groups := range *snap {
		out.ProvenGroups[name] = sortedGroups(groups)
	}
	return out
}
