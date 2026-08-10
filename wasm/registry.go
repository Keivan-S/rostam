// SPDX-License-Identifier: Apache-2.0

package wasm

import (
	"fmt"

	"github.com/rostamlabs/rostam/ops"
)

// ValidateModuleKind runs the OpReadOnly / writes-state guard against a module
// that has been COMPILED but not yet installed in any Runtime or Registry.
//
// It exists so a caller can reject a bad registration with NO side effects at
// all: no blob written to the data dir, no Cranelift compile, no engine store.
// The same guard inside RegisterModule / RegisterOrReplaceModule can only fire
// after rt.AddModule has already instantiated the module and after the caller
// has already persisted it. Validating here, before any of that, is what makes
// an invalid registration a clean no-op.
//
// It is no longer also protecting against a HALF-SWAPPED runtime: the runtime is
// content addressed now, so a late rejection can only waste work, never leave the
// handler running new bytes under the old registry Kind and key extractor. That
// hazard is gone (see Runtime and RegisterOrReplaceModule), but the wasted work
// is real and the early rejection is still the right order.
//
// The kind byte is wire-controlled, so a declared-OpReadOnly module that imports
// a state-mutating host function must be refused rather than trusted: an
// OpReadOnly op bypasses Raft (served on one replica), so a write from it would
// never be logged and would silently diverge replicas.
func ValidateModuleKind(opName string, kind ops.OpKind, m *Module) error {
	if kind != ops.OpReadOnly || m == nil {
		return nil
	}
	if m.WritesState() {
		return fmt.Errorf("wasm: module %q: OpReadOnly module imports a state-mutating host function (cache_put/cache_del/cache_expire); read-only ops must not mutate state", opName)
	}
	return nil
}

// resolveModuleForInvoke answers THE question the per-group-binding design
// exists to change: which module version must execute this invocation of opName?
//
// ############### THIS METHOD IS THE PER-GROUP-BINDING SEAM ###############
//
// THE INVARIANT IT ESTABLISHES. The version used to execute a committed entry in
// shard group g is a pure function of g's ORDERED LOG PREFIX. Every replica of g
// derives the same version for the same entry regardless of which OTHER groups it
// hosts or how far along them it is.
//
// ORDERED is load-bearing, not decoration. The fold that fills this table
// (cluster.installedWASM.groups) is a maximum COMPOSED WITH a contract freeze,
// and the freeze makes it order-DEPENDENT — the first registration of a name in
// g's log pins g's contract, so the same SET of registrations delivered in two
// orders can bind two different versions. PREFIX-DETERMINISM is what safety needs
// and what holds, because every replica of g applies one identical sequence;
// permutation-invariance is neither claimed nor true. See
// cluster.installedWASM.groups for the lemma and the counterexample.
//
// That is what a node-wide answer could not give:
// a registration is replicated into every group's log and those logs commit at
// INDEPENDENT times, so a node that has applied an update switches versions for
// every group at once while a peer that has not switched for none — and both then
// apply the same committed invocation with different bytes, silently, with no
// error to classify and no halt. See the DYNAMIC-HANDLER CAVEAT in
// shard/apply_class.go for why no error classification could have caught that.
//
// EVERY WASM INVOCATION IN THE PROCESS GOES THROUGH HERE — moduleHandler below
// builds the only handler closure any WASM op is ever registered with, on both
// registration paths, so there is no admin, config or embedded route to
// Runtime.Invoke that bypasses this.
//
// IT IS A METHOD ON *Runtime because the table has to live somewhere and the
// Runtime is the one object every RegisterModule call site already passes. Making
// it a method left every one of those call sites untouched.
//
// THE FOUR ANSWERS, in the order they are decided:
//
//  1. NO TABLE HAS EVER BEEN PUBLISHED (groupBindings() == nil) → the node-wide
//     `bound`. This is the single-node Direct backend (rostam.directStore), which
//     shares this type but has no Raft groups, no replicated registrations and
//     therefore no bindings AT ALL. It is also the state of a cluster node before
//     its first replicated registration.
//
//     THE NIL-TABLE CHECK IS WHAT PROTECTS DIRECT MODE, and it is worth being
//     precise about why, because the obvious reasoning is wrong. rostam.NewDirect
//     never calls TxContext.SetShardIndex, and ShardIndex reports the sentinel
//     only for a NIL receiver — so a Direct TxContext reports the zero value 0,
//     which is a perfectly VALID group index. Case (3) would therefore not catch
//     it. Nothing in Direct mode ever calls PublishGroupBindings either, so the
//     table stays nil forever and this case fires first regardless of what the
//     index says. That is the whole guarantee: were this case removed, or were a
//     table ever published in Direct mode, every Direct WASM call would resolve
//     against a phantom "group 0", miss, and become a hard ErrWASMNoGroupBinding
//     failure. Keep the two facts together — the check is here, and Direct mode
//     publishes nothing.
//
//  2. THE OP IS NOT IN THE TABLE → the node-wide `bound`. The table carries only
//     REPLICATED registrations, so this is an operator-configured module
//     (cfg.WASMModules). Config modules were never proposed into anyone's log, so
//     there is no group prefix to bind them to and no cross-replica agreement to
//     preserve — the node-wide contract is the whole contract for them.
//
//  3. THE CALL HAS NO GROUP PROVENANCE (ops.NoShardIndex), or the op is
//     ops.OpReadOnly → the node-wide `bound`. NoShardIndex means no dispatcher is
//     behind the TxContext (a handler invoked directly in a test). OpReadOnly is
//     the load-bearing one: shard.Store.Call serves a read-only op from local
//     state WITHOUT proposing anything, so there is no entry in any log and
//     nothing to bind — and the route gate deliberately does not gate read-only
//     ops for the same reason. Resolving them per group would newly FAIL every
//     read served by a group the registration never reached, which is a live,
//     ungated, perfectly safe path today. What it costs is stated plainly: two
//     replicas may answer one read-only invocation with different module versions
//     during a broadcast window. That cannot diverge stored state (a read-only
//     module is rejected at registration if it imports any state-mutating host
//     function — see ValidateModuleKind) and read-only ops already carry no
//     cross-replica agreement guarantee at all.
//
//  4. OTHERWISE the binding at (opName, tx.ShardIndex()) — and a MISS IS AN
//     ERROR, never a fallback. This is a replicated read-write op being applied
//     from a group whose table has no version for it. The route gate
//     (cluster.checkWASMRouteGate) guarantees a registration for opName sits BELOW
//     any invocation of it in the same group's log, and both catch-up routes
//     reconstruct the binding (log replay applies the registration entry; the
//     group's snapshot carries the binding). So a miss means this replica's state
//     disagrees with the proposer's — the proposer had an open gate, hence a
//     binding, hence peers will EXECUTE this entry while this node cannot. Falling
//     back to the node-wide version would execute it with a version this group's
//     log never named, which is precisely the silent divergence above; returning a
//     value-less error lets the FSM fail closed. ops.ErrWASMNoGroupBinding is
//     classified classFatal in shard/apply_class.go for exactly the reason
//     ErrOpNotRegistered is.
//
// FIFTH ANSWER, ADDED BY THIN MARKERS AND APPLYING TO ALL FOUR ABOVE: the
// version this resolves to may not be INSTANTIATED here. A marker names its
// module by content address and does not carry it, so applying a registration no
// longer implies holding the module — a node that was unreachable during the
// registration's push, or that restored a snapshot naming a version it never
// fetched, has a perfectly correct binding to a version it does not have.
//
// That is ops.ErrWASMModuleNotResident, and it is neither of the two things it
// resembles. It is NOT ErrWASMNoGroupBinding: the binding is present and correct,
// this replica agrees with its peers about the log, and there is nothing to fail
// closed about — halting would turn a group-local, self-healing condition into a
// process-global crash loop, since the entry replays into the same condition on
// restart. It is NOT a business error either: peers WILL execute the entry, so
// advancing past it is the silent divergence this whole mechanism exists to
// prevent. shard.classifyApplyErr gives it its own class (classRetry): mutate
// nothing, advance nothing, halt nothing, re-run when the blob lands.
//
// THE KIND GUARD RUNS HERE TOO, AND IT HAS TO. wasm.ValidateModuleKind — the
// refusal of an OpReadOnly declaration over a module that imports a
// state-mutating host function — used to run at registration apply, where the
// entry carried the bytes. It cannot run there any more on a node that has not
// fetched them, and the hazard it closes is the sharpest one in the package: an
// OpReadOnly op is served from local state WITHOUT being proposed, so a write
// from one would never be logged and would diverge that replica silently. A blob
// carries no op name, so the push's compile-verify cannot check it either — it
// does not know what Kind the bytes are about to be declared as. This is the
// first and only point where the marker's Kind and the module's imports are both
// in hand, so this is where the pairing is judged. It costs one map read that the
// residency check was already making.
//
// It stays deterministic across replicas: Kind comes from the marker and the
// imports come from a blob that is self-verifying against the marker's
// fingerprint, so every replica reaches the same verdict on the same entry. What
// differs across replicas is only WHEN — a node that had to fetch reaches it
// later, having blocked until it could.
func (r *Runtime) resolveModuleForInvoke(opName string, kind ops.OpKind, bound ModuleID, tx *ops.TxContext) (ModuleID, error) {
	id, group, err := r.bindModuleForInvoke(opName, kind, bound, tx)
	if err != nil {
		return ModuleID{}, err
	}
	writes, resident := r.residentModuleState(id)
	if !resident {
		return ModuleID{}, &ops.WASMNotResidentError{Op: opName, Group: group} // (5) not fetched yet
	}
	if kind == ops.OpReadOnly && writes {
		return ModuleID{}, fmt.Errorf("wasm: module %q: OpReadOnly module imports a state-mutating host function (cache_put/cache_del/cache_expire); read-only ops must not mutate state", opName)
	}
	return id, nil
}

// bindModuleForInvoke is the binding half of the resolution — cases (1)..(4)
// above — split out so the residency and kind guards apply to every one of them
// without being restated four times. It also returns the group the answer is
// attributed to, which is what the not-resident error has to carry so the fetcher
// knows which binding to look the fingerprint up in; ops.NoShardIndex for the
// three node-wide fallbacks.
func (r *Runtime) bindModuleForInvoke(opName string, kind ops.OpKind, bound ModuleID, tx *ops.TxContext) (ModuleID, int, error) {
	table := r.groupBindings()
	if table == nil {
		return bound, ops.NoShardIndex, nil // (1) no per-group binding exists in this process
	}
	groups, replicated := table[opName]
	if !replicated {
		return bound, ops.NoShardIndex, nil // (2) config-installed / non-replicated op
	}
	idx := tx.ShardIndex()
	if idx == ops.NoShardIndex || kind == ops.OpReadOnly {
		return bound, ops.NoShardIndex, nil // (3) no group provenance, or nothing was ever logged
	}
	id, ok := groups[idx]
	if !ok {
		return ModuleID{}, idx, fmt.Errorf("%w: op %q has no registration in shard group %d's log on this node, but a committed entry in that log invokes it",
			ops.ErrWASMNoGroupBinding, opName, idx) // (4) divergence
	}
	return id, idx, nil
}

// ValidateRuntimeKind runs the OpReadOnly / writes-state guard against a module
// that is ALREADY instantiated on rt, named by its ModuleID.
//
// It exists because the per-group binding table made "this version is already in
// the runtime" a common case that must still be kind-checked. The same bytes can
// legitimately be re-registered under a different Kind — a first-time
// registration in a group that missed the original one may declare OpReadOnly
// where the installed one declared OpReadWrite — and skipping the compile because
// rt.HasModule already holds the slot must not also skip the guard that decides
// whether the pairing is legal at all. ValidateModuleKind cannot serve here: it
// takes a freshly compiled *Module, which is the thing being skipped.
//
// An id the runtime does not hold reports no error: there is nothing to judge,
// and the caller is about to AddModule it (which runs the guard on the compiled
// artifact).
func ValidateRuntimeKind(rt *Runtime, opName string, kind ops.OpKind, id ModuleID) error {
	if kind != ops.OpReadOnly {
		return nil
	}
	if writes, ok := rt.moduleWritesState(id); ok && writes {
		return fmt.Errorf("wasm: module %q: OpReadOnly module imports a state-mutating host function (cache_put/cache_del/cache_expire); read-only ops must not mutate state", opName)
	}
	return nil
}

// RegisterModule wires a WASM op into reg. It builds a Handler closure that
// resolves the module version (resolveModuleForInvoke) and delegates to
// rt.Invoke, and registers it via reg.Register (when ke is nil) or
// reg.RegisterRoutableCrossShard (when ke is non-nil).
//
// id is the ModuleID rt.AddModule returned for this op's module. It is what
// binds the NAME in the registry to the CONTENT in the runtime, and it is
// captured by the closure rather than looked up per call, so the registry entry
// is the single place the two are tied together: reg.Replace swaps the Kind, the
// key extractor and the module version as ONE operation. That is why there is no
// separate name → module table to keep in step, and no window in which the
// handler runs new bytes under an old Kind.
//
// A WASM op's host functions (cache_get/put/del/expire) can address ANY key, so
// even a routable WASM op may touch keys outside its routing shard. It is
// therefore registered CROSS-SHARD: the routing key still selects the cluster
// shard, but a single-node serializer (the Direct backend) guards it with a
// global barrier rather than a per-shard lock, so the handler stays atomic
// against every other read-write op.
func RegisterModule(reg *ops.Registry, rt *Runtime, opName string, id ModuleID, kind ops.OpKind, ke ops.KeyExtractor) error {
	// ############ THE OpReadOnly/writes-state GUARD IS NOT RUN HERE ############
	//
	// It used to be, gated on `rt.moduleWritesState(id)` returning ok && writes —
	// i.e. it fired ONLY WHEN THE MODULE HAPPENED TO BE RESIDENT, and returned nil
	// when it was not. Under inline registration bytes that distinction did not
	// exist: every node that reached this call had just compiled and added the
	// module, so residency was universal. Thin markers made it a real fork, and
	// the resulting behaviour was exactly the residency-dependent apply-time
	// refusal that cluster.materializeWASMBlob argues must not exist:
	//
	//	node A holds the blob, refuses the registration, and does not register
	//	the op; node B has not fetched it yet, so the guard silently passes, B
	//	registers the op, opens its route gate and proposes an invocation into a
	//	group — which A then applies, cannot look up, and halts on under
	//	classFatal shard.ErrOpNotRegistered.
	//
	// It was worse on the reload path: a sidecar for such an op with its blob
	// PRESENT would fail this call with a non-ErrDuplicateOp error and take node
	// construction down, which is the precise failure the reload rewrite exists to
	// prevent — and it would do so on the nodes that had the bytes and not on the
	// nodes that did not.
	//
	// The guard did not go away; it MOVED to resolveModuleForInvoke, which asks it
	// once per invocation on EVERY node and therefore reaches one verdict
	// everywhere, whenever the bytes arrive. That is strictly stronger than this
	// one was: it cannot be skipped by non-residency, and it covers a module that
	// becomes resident AFTER the op was registered.
	//
	// The two callers that legitimately want an EARLY refusal — operator config
	// (cluster.loadOneModule) and Direct mode (rostam.directStore.RegisterWASM) —
	// call ValidateModuleKind explicitly on the compiled module they are holding.
	// Both are node-local paths with the bytes in hand and no replica to disagree
	// with, so an early refusal there is safe and is the better error.
	fn := moduleHandler(rt, opName, kind, id)
	if ke == nil {
		if err := reg.Register(opName, kind, fn); err != nil {
			return fmt.Errorf("wasm: RegisterModule %q: %w", opName, err)
		}
		return nil
	}
	if err := reg.RegisterRoutableCrossShard(opName, kind, fn, ke); err != nil {
		return fmt.Errorf("wasm: RegisterModule %q: %w", opName, err)
	}
	return nil
}

// moduleHandler builds the ops.Handler for a WASM op: resolve the version for
// the shard group this entry came from, then invoke it. Shared by both
// registration paths so there is exactly one closure shape in the codebase and
// exactly one call to the per-group-binding seam.
//
// id is the NODE-WIDE binding — the version this registry entry was installed
// with. It is what resolveModuleForInvoke falls back to for the calls that have
// no group to bind against (see its four cases); for a replicated read-write op
// applied from a group's log it is not consulted at all.
//
// kind is captured for the same resolver: a read-only op is served without
// proposing anything, so no group's log carries the entry and there is no binding
// to resolve.
func moduleHandler(rt *Runtime, opName string, kind ops.OpKind, id ModuleID) func(*ops.TxContext, []byte) ([]byte, error) {
	return func(tx *ops.TxContext, args []byte) ([]byte, error) {
		mod, err := rt.resolveModuleForInvoke(opName, kind, id, tx)
		if err != nil {
			return nil, err
		}
		return rt.Invoke(mod, tx, args)
	}
}

// RegisterOrReplaceModule is RegisterModule with replace-on-conflict semantics:
// it overwrites any existing registry entry for opName instead of failing with
// ops.ErrDuplicateOp.
//
// It is the install step for a DYNAMIC registration that won the
// (Epoch, fingerprint) comparison (see ops.WASMRegistration.Epoch). Plain
// RegisterModule cannot express that: the winning registration may carry a
// different Kind, and that is stored in the ops.Registry
// entry rather than resolved from the runtime at call time, so a duplicate-op
// no-op would leave the LOSING routing in place. Two replicas that saw the two
// registrations in opposite orders would then disagree about which shard group
// the op's invocations belong to.
//
// The same read-only/writes-state guard as RegisterModule applies.
//
// THE REPLACE IS NOW THE ONLY SWAP IN THE INSTALL, which is what retired the
// runtime rollback that used to guard this call. The runtime slot is content
// addressed (see Runtime), so rt.AddModule has ADDED the new version alongside
// the old rather than displacing it; the registry entry is what still names the
// old one. reg.Replace therefore moves Kind, key extractor and module version
// together in one step, and a failure leaves the previous entry — and hence the
// previous module — fully intact. There is no half-swapped state to repair.
func RegisterOrReplaceModule(reg *ops.Registry, rt *Runtime, opName string, id ModuleID, kind ops.OpKind, ke ops.KeyExtractor) error {
	// The OpReadOnly/writes-state guard is not run here either, and this is the
	// call site where that matters most: it is the REPLICATED install path
	// (cluster.applyWASMRegistration), so a refusal here is a refusal that would
	// differ between a node holding the blob and a node that has not fetched it
	// yet. See RegisterModule for the full argument and for where the guard lives
	// now (resolveModuleForInvoke).
	fn := moduleHandler(rt, opName, kind, id)
	// crossShard mirrors RegisterModule: a routable WASM op is cross-shard
	// because its host functions can address any key; a shardless one is not.
	if err := reg.Replace(opName, kind, fn, ke, ke != nil); err != nil {
		return fmt.Errorf("wasm: RegisterOrReplaceModule %q: %w", opName, err)
	}
	return nil
}

// MustRegisterModule is a test helper that panics if RegisterModule returns an
// error.
func MustRegisterModule(reg *ops.Registry, rt *Runtime, opName string, id ModuleID, kind ops.OpKind, ke ops.KeyExtractor) {
	if err := RegisterModule(reg, rt, opName, id, kind, ke); err != nil {
		panic(err)
	}
}
