// SPDX-License-Identifier: Apache-2.0

// Package ops provides a registry for user-supplied stored procedures
// (ops) that run inside a shard's FSM Apply path. Read-only ops bypass
// the Raft log; write ops are encoded into log entries and applied on
// every replica deterministically.
package ops

import (
	"errors"
	"fmt"
	"sync"
)

// OpKind tags ops as read-only (bypass Raft) or read-write (go through Raft).
type OpKind uint8

const (
	// OpReadOnly ops execute directly against the cache; they must not mutate state.
	OpReadOnly OpKind = 0
	// OpReadWrite ops are serialised as Raft log entries and applied by the FSM.
	OpReadWrite OpKind = 1
)

// Handler is the function signature for a registered op.
type Handler func(tx *TxContext, args []byte) ([]byte, error)

// KeyExtractor extracts the routing key from an op's args. Returns
// (keyBytes, true) when a key is present; (nil, false) when the args
// shape is invalid. Shardless ops (e.g., __ping__) register with a nil
// KeyExtractor instead.
//
// Returned key may alias into args; the caller does not retain it past
// the routing decision.
type KeyExtractor func(args []byte) ([]byte, bool)

// KeyExtractorInto is the allocation-free form of KeyExtractor: it returns the
// same routing key, but built either as a window INTO args or by appending to the
// caller-owned scratch, instead of allocating a fresh []byte per call. A nil
// return means "no key" (the (nil, false) of KeyExtractor).
//
// It exists only because a routing key is the shortest-lived value in the system:
// the caller hashes it to a shard index and drops it. The returned key therefore
// aliases args OR scratch, and the SCRATCH MUST NOT OUTLIVE THE ROUTING DECISION —
// the next op's extraction overwrites it. No implementation may store either
// slice, and no caller may keep the key past shardOf.
//
// A router does NOT reach one of these through a stored function value: routing
// selects the extractor by RouteLayout and calls RouteKeyInto, a DIRECT call. That
// is deliberate — an indirect call forces the compiler to assume the callee leaks
// its arguments, which sends the caller's stack scratch to the heap and gives back
// most of the allocation the whole path exists to remove.
type KeyExtractorInto func(args, scratch []byte) []byte

// RouteLayout names the args wire layout a built-in routable op's key lives in.
// It is the allocation-free routing path's stand-in for a stored KeyExtractorInto
// (see that type for why a function value is the wrong shape here): the registry
// records which layout an op uses, and RouteKeyInto turns (layout, args) into the
// key with a direct call.
//
// RouteLayoutNone means "no allocation-free extractor" — the op is either shardless
// or dynamically registered (a WASM op, whose layout this package cannot know), and
// routing falls back to its KeyExtractor.
type RouteLayout uint8

const (
	// RouteLayoutNone: route via KeyExtractor (or not at all, if that is nil too).
	RouteLayoutNone RouteLayout = iota
	// RouteLayoutColAt1: [colLen:u8][col]... — the collection name at offset 0.
	RouteLayoutColAt1
	// RouteLayoutColAt2: [flags:u8][colLen:u8][col]... — the name at offset 1.
	RouteLayoutColAt2
)

// ErrDuplicateOp is returned when registering a name that already exists.
var ErrDuplicateOp = errors.New("ops: op name already registered")

// maxOpNameLen mirrors the wire-format limit (uint8-length prefix).
const maxOpNameLen = 255

// entry holds one registered handler.
type entry struct {
	kind OpKind
	fn   Handler
	ke   KeyExtractor // nil = shardless
	// layout is the allocation-free routing twin of ke, set ONLY for built-in
	// routable ops (registerRoutableInto, called from this package). RouteKeyInto
	// on this layout extracts the exact same key as ke — same offsets, same
	// canonicalization — and RouteLayoutNone simply means "route through ke", so a
	// dynamically registered (WASM) op keeps the allocating path.
	layout RouteLayout
	// crossShard marks a read-write op whose handler may touch keys beyond its
	// routing key's shard (e.g. a WASM op, whose host functions can read/write
	// arbitrary keys). The routing key (ke) still picks the destination shard in
	// the cluster, but a single-node serializer (the Direct backend) must NOT
	// rely on a per-shard lock for such an op — it must take a global barrier so
	// the multi-key handler stays atomic against every other read-write op.
	// Ops that touch only their routing key (KV builtins, vector ops) leave this
	// false and benefit from per-shard parallelism.
	crossShard bool
}

// Registry is a thread-safe map of op name to handler.
type Registry struct {
	mu sync.RWMutex
	m  map[string]entry
}

// NewRegistry constructs an empty registry.
func NewRegistry() *Registry {
	return &Registry{m: make(map[string]entry, 16)}
}

// Register adds a shardless op (no routing key). Suitable for cluster-
// level ops like __ping__. For ops that should route to a specific
// shard, use RegisterRoutable.
func (r *Registry) Register(name string, kind OpKind, fn Handler) error {
	return r.registerEntry(name, kind, fn, nil, RouteLayoutNone, false)
}

// RegisterRoutable adds an op with an explicit KeyExtractor. The
// cluster layer uses the extractor to pick the destination shard.
// Returns an error if ke is nil — for shardless ops, use Register.
//
// The handler is expected to touch ONLY its routing key's shard (true for the
// KV builtins and vector ops). A single-node serializer may then guard it with
// just that shard's lock. For a routable op whose handler may touch arbitrary
// keys across shards (e.g. a WASM op), use RegisterRoutableCrossShard.
func (r *Registry) RegisterRoutable(name string, kind OpKind, fn Handler, ke KeyExtractor) error {
	if ke == nil {
		return errors.New("ops: RegisterRoutable requires non-nil KeyExtractor; use Register for shardless ops")
	}
	return r.registerEntry(name, kind, fn, ke, RouteLayoutNone, false)
}

// RegisterRoutableCrossShard is like RegisterRoutable but marks the op as
// cross-shard: its routing key still selects the destination shard in the
// cluster, but the handler may read/write keys on OTHER shards too. A single-
// node serializer (the Direct backend) must take a global barrier for such an
// op rather than a per-shard lock, so the multi-key handler stays atomic
// against every other read-write op. WASM ops register through here.
func (r *Registry) RegisterRoutableCrossShard(name string, kind OpKind, fn Handler, ke KeyExtractor) error {
	if ke == nil {
		return errors.New("ops: RegisterRoutableCrossShard requires non-nil KeyExtractor; use Register for shardless ops")
	}
	return r.registerEntry(name, kind, fn, ke, RouteLayoutNone, true)
}

// Replace installs name unconditionally, overwriting any existing entry instead
// of returning ErrDuplicateOp. It exists for exactly one caller: replacing a
// DYNAMICALLY registered WASM op when a strictly newer registration wins the
// (Epoch, fingerprint) comparison (see WASMRegistration.Epoch).
//
// Why plain Register is not enough there: a replacement may change Kind or the
// KeyExtractor, and those live in the registry entry, not in the wasm.Runtime.
// Leaving the loser's entry in place would make two replicas that received the
// same two registrations in different orders route or classify the SAME op
// differently — the divergence the epoch rule exists to prevent.
//
// Do NOT use this to redefine a built-in op: the duplicate check on Register is
// what keeps two subsystems from silently claiming the same name.
func (r *Registry) Replace(name string, kind OpKind, fn Handler, ke KeyExtractor, crossShard bool) error {
	if err := validateEntry(name, kind, fn); err != nil {
		return err
	}
	r.mu.Lock()
	defer r.mu.Unlock()
	r.m[name] = entry{kind: kind, fn: fn, ke: ke, crossShard: crossShard}
	return nil
}

// CrossShard reports whether the named op was registered as cross-shard (its
// handler may touch keys beyond its routing shard). False for unknown ops.
func (r *Registry) CrossShard(name string) bool {
	r.mu.RLock()
	defer r.mu.RUnlock()
	return r.m[name].crossShard
}

// registerRoutableInto is RegisterRoutable plus the op's RouteLayout, which unlocks
// the allocation-free routing path. It is package-private on purpose: the layout
// and ke MUST agree on every args shape (they are two spellings of one wire
// layout), which is only verifiable for the built-in ops whose layouts live in this
// package. Everything registered from outside — WASM ops included — goes through
// RegisterRoutable and routes via ke.
func (r *Registry) registerRoutableInto(name string, kind OpKind, fn Handler, ke KeyExtractor, layout RouteLayout) error {
	if ke == nil {
		return errors.New("ops: registerRoutableInto requires non-nil KeyExtractor")
	}
	if layout == RouteLayoutNone {
		return errors.New("ops: registerRoutableInto requires a RouteLayout; use RegisterRoutable when there is none")
	}
	return r.registerEntry(name, kind, fn, ke, layout, false)
}

func (r *Registry) registerEntry(name string, kind OpKind, fn Handler, ke KeyExtractor, layout RouteLayout, crossShard bool) error {
	if err := validateEntry(name, kind, fn); err != nil {
		return err
	}
	r.mu.Lock()
	defer r.mu.Unlock()
	if _, exists := r.m[name]; exists {
		return ErrDuplicateOp
	}
	r.m[name] = entry{kind: kind, fn: fn, ke: ke, layout: layout, crossShard: crossShard}
	return nil
}

// validateEntry holds the name/handler/kind rules shared by registerEntry and
// Replace, so an unconditional install cannot bypass them.
func validateEntry(name string, kind OpKind, fn Handler) error {
	if name == "" {
		return errors.New("ops: empty op name")
	}
	if len(name) > maxOpNameLen {
		return fmt.Errorf("ops: op name length %d exceeds max %d", len(name), maxOpNameLen)
	}
	// Protocol-v2 wire compatibility: an op name of length 2 is ambiguous with
	// the v2 frame's [version=0x02][...] prefix. Reject by convention; existing
	// ops (get/put/del/expire/incr/__ping__/vector_*) all clear this bar.
	if len(name) == 2 {
		return fmt.Errorf("ops: op name %q has length 2 which collides with protocol v2 version byte; use 3+ chars", name)
	}
	if fn == nil {
		return errors.New("ops: nil handler")
	}
	if kind != OpReadOnly && kind != OpReadWrite {
		return fmt.Errorf("ops: invalid OpKind %d", kind)
	}
	return nil
}

// Lookup returns the handler, its kind, its KeyExtractor (nil for
// shardless ops), and whether the op was found. Callers that also need the
// cross-shard flag (to pick a single-node lock strategy) should use LookupEntry,
// which reads it from the SAME entry in one registry acquisition.
func (r *Registry) Lookup(name string) (Handler, OpKind, KeyExtractor, bool) {
	fn, kind, ke, _, ok := r.LookupEntry(name)
	return fn, kind, ke, ok
}

// LookupRouting returns everything the cluster router needs for one op — its kind,
// its KeyExtractor, and its RouteLayout (RouteLayoutNone unless the op is a
// built-in) — from ONE registry entry under ONE RLock. A router should prefer
// RouteKeyInto(layout, ...) and fall back to ke when the layout is None; both yield
// the same key.
func (r *Registry) LookupRouting(name string) (kind OpKind, ke KeyExtractor, layout RouteLayout, ok bool) {
	r.mu.RLock()
	defer r.mu.RUnlock()
	e, ok := r.m[name]
	if !ok {
		return 0, nil, RouteLayoutNone, false
	}
	return e.kind, e.ke, e.layout, true
}

// LookupEntry is Lookup plus the crossShard flag (whether the op's handler may
// touch keys beyond its routing shard — see RegisterRoutableCrossShard), read
// from the SAME single registry entry under ONE RLock. A routing caller (e.g.
// the Direct backend's per-shard write serializer) uses it to decide its lock
// strategy without a redundant second CrossShard lock + map lookup.
func (r *Registry) LookupEntry(name string) (fn Handler, kind OpKind, ke KeyExtractor, crossShard, ok bool) {
	r.mu.RLock()
	defer r.mu.RUnlock()
	e, ok := r.m[name]
	if !ok {
		return nil, 0, nil, false, false
	}
	return e.fn, e.kind, e.ke, e.crossShard, true
}
