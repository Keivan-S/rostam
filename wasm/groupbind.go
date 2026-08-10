// SPDX-License-Identifier: Apache-2.0

package wasm

import "sync/atomic"

// GroupBindings is the PER-GROUP VERSION BINDING TABLE: op name → shard group
// index → the ModuleID that group's Raft log has committed for that op.
//
// It is the whole point of per-group binding. The version used to execute a committed
// entry in shard group g must be a pure function of g's LOG PREFIX, because that
// is the only quantity every replica of g agrees on: two replicas of g may host
// entirely different sets of other groups and sit at entirely different points
// in them, so any node-wide answer ("the version I last installed under this
// name") is a value they are allowed to disagree about while both being
// perfectly correct locally. Keying the answer on (op, group) removes the
// disagreement by construction — see resolveModuleForInvoke.
//
// IT IS PUBLISHED BY COPY-ON-WRITE and treated as IMMUTABLE once published: the
// map, the inner maps, and every entry are read without a lock by
// resolveModuleForInvoke on each shard group's FSM apply goroutine. The producer
// (cluster.Node.publishWASMGateLocked) rebuilds the whole value under its own
// mutex and stores the new pointer; it never mutates a published one.
//
// THE SAME VALUE IS THE ROUTE GATE'S EVIDENCE. cluster's propose-time route gate
// asks "does group g's log carry a registration for op X?", which is exactly
// "does this table have a binding at (X, g)?". That used to be a separate
// map[int]struct{}; the version is attached to it rather than adding a
// second structure, so the gate and the resolver can never disagree about which
// groups are proven.
//
// A NIL TABLE MEANS "no per-group binding exists in this process at all" and is
// the normal state for the single-node Direct backend, which has no Raft groups,
// no registration entries, and therefore nothing to bind. See
// resolveModuleForInvoke for how the fallback is chosen.
type GroupBindings map[string]map[int]ModuleID

// groupBindingState is the published-table slot. It is a separate embeddable
// type because Runtime is declared TWICE — once in the cgo wasmtime backend and
// once in the non-cgo stub — and a field has to exist in both for the shared,
// build-tag-free methods in registry.go to reach it. Embedding it is two lines
// in each declaration instead of a duplicated field plus duplicated accessors.
type groupBindingState struct {
	published atomic.Pointer[GroupBindings]
}

// PublishGroupBindings installs b as the runtime's per-group version table,
// replacing any previous one. b must not be mutated after this call.
//
// Passing nil clears the table, which returns every resolution to the node-wide
// binding captured in the ops.Registry entry.
func (s *groupBindingState) PublishGroupBindings(b GroupBindings) {
	if b == nil {
		s.published.Store(nil)
		return
	}
	s.published.Store(&b)
}

// groupBindings returns the currently published table, or nil when none has ever
// been published.
func (s *groupBindingState) groupBindings() GroupBindings {
	p := s.published.Load()
	if p == nil {
		return nil
	}
	return *p
}
