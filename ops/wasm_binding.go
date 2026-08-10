// SPDX-License-Identifier: Apache-2.0

package ops

import (
	"errors"
	"fmt"
)

// ErrWASMNoGroupBinding is returned by the WASM op handler when a committed
// entry in shard group g invokes a REPLICATED read-write op for which g's
// per-group version binding table holds nothing.
//
// IT IS A DIVERGENCE SIGNAL, not a missing-registration one, and it is
// classified classFatal by shard.classifyApplyErr for the same reason
// shard.ErrOpNotRegistered is. The propose-time route gate
// (cluster.checkWASMRouteGate) refuses to put an invocation of op X into group
// g's log until it knows g's log already carries a registration for X, so every
// INV(X) in g's log sits above a REG(X) in the SAME log — and both catch-up
// routes reconstruct the binding from that (log replay applies the registration;
// g's snapshot carries the binding explicitly). A replica of g that reaches
// INV(X) with no binding therefore holds state that disagrees with what g's log
// says: its peers have the binding and will EXECUTE the entry. Advancing the
// applied index over it would be the silent, permanent divergence the whole
// apply-classification mechanism exists to prevent.
//
// It lives in ops rather than in wasm because shard has to recognise it and
// shard does not import wasm.
var ErrWASMNoGroupBinding = errors.New("ops: wasm op has no version bound to this shard group")

// ErrWASMModuleNotResident is returned by the WASM op handler when a committed
// entry names a module version this node CAN identify but does not yet HOLD:
// the binding (or the node-wide install) resolves to a wasm.ModuleID that is not
// instantiated on this node's runtime, because the marker that established it
// carried only the module's fingerprint and the bytes have not arrived here yet.
//
// ############ IT IS NEITHER A DIVERGENCE SIGNAL NOR A BUSINESS ERROR #########
//
// It is the ONLY apply outcome in the system that means "nothing is wrong, this
// node simply cannot answer yet". Contrast the two neighbours it must never be
// confused with:
//
//   - ErrWASMNoGroupBinding means this replica's view of the group's LOG
//     disagrees with its peers'. There is nothing to wait for: the peers will
//     execute the entry and this node cannot, so it must fail closed (classFatal).
//   - a handler's own business error is a pure function of the entry and fails
//     identically everywhere, so the index advances (classAdvance).
//
// This one is a pure function of LOCAL BYTE RESIDENCY, which is not a function of
// the log at all and which self-heals the moment the blob arrives. Advancing over
// it would skip an entry every peer executes (divergence); halting on it would
// convert a group-local, self-healing condition into a process-global crash loop,
// since the entry replays into the same condition on restart. So it is neither:
// shard.classifyApplyErr maps it to classRetry — do not advance, do not halt,
// wait and re-run the same entry.
//
// IT CARRIES THE COORDINATES THE FETCHER NEEDS. shard cannot import cluster, so
// the FSM hands this error back out through a hook and the cluster side resolves
// (Op, Group) to the fingerprint it must go and get. Op and Group are exactly
// what cluster.wasmState is keyed on; Group is NoShardIndex for the resolutions
// that have no group provenance (a read-only op, a config module, Direct mode).
//
// It lives in ops rather than in wasm for the same reason ErrWASMNoGroupBinding
// does: shard has to recognise it and shard does not import wasm.
var ErrWASMModuleNotResident = errors.New("ops: wasm module bytes are not resident on this node yet")

// WASMNotResidentError is the typed carrier for ErrWASMModuleNotResident. It
// exists so the retry hook can recover the (op, group) pair with errors.As
// instead of parsing a message: the pair is what cluster turns into a blob
// fingerprint and a fetch, and a string-parsed one would break silently the first
// time this message is reworded.
type WASMNotResidentError struct {
	// Op is the op name whose module version is missing.
	Op string
	// Group is the shard group whose binding named it, or NoShardIndex when the
	// resolution had no group provenance and used the node-wide install.
	Group int
}

func (e *WASMNotResidentError) Error() string {
	if e.Group == NoShardIndex {
		return fmt.Sprintf("%s: op %q (node-wide install)", ErrWASMModuleNotResident.Error(), e.Op)
	}
	return fmt.Sprintf("%s: op %q in shard group %d", ErrWASMModuleNotResident.Error(), e.Op, e.Group)
}

// Unwrap makes errors.Is(err, ErrWASMModuleNotResident) true through any number
// of %w wraps the apply path adds on the way out.
func (e *WASMNotResidentError) Unwrap() error { return ErrWASMModuleNotResident }
