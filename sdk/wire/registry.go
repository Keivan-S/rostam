// SPDX-License-Identifier: Apache-2.0

// Package wire is the client-safe wire-codec leaf split out of ops: every
// Encode*/Decode* function, routing key extractor, and the ROUTING-ONLY
// registry the Go client needs, with NO dependency on the server-side engine
// packages (cache, objstore, vector/analysis). ops re-exports the shared value
// types (OpKind, KeyExtractor, RouteLayout, ...) as aliases so existing
// ops.* call sites across the server are unaffected; the server-side Registry
// (which additionally binds a Handler — a func(tx *TxContext, ...) — per op)
// stays in ops because TxContext wraps *cache.Cache and cannot move here
// without dragging cache along.
package wire

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
var ErrDuplicateOp = errors.New("wire: op name already registered")

// maxOpNameLen mirrors the wire-format limit (uint8-length prefix).
const maxOpNameLen = 255

// routeEntry holds one registered op's ROUTING metadata only.
type routeEntry struct {
	kind       OpKind
	ke         KeyExtractor
	layout     RouteLayout
	crossShard bool
}

// Registry is a thread-safe, ROUTING-ONLY registry: for each op name it records
// (kind, KeyExtractor, RouteLayout, crossShard) but carries no handler function.
//
// This is the client-safe counterpart of ops.Registry, which additionally binds a
// server-side Handler (func(tx *TxContext, args []byte) ([]byte, error)) per op.
// The Go client never invokes an op's handler locally — it only ever needs an
// op's KeyExtractor to pick a destination shard — so rather than tolerate a nil
// Handler field (which would require exposing ops.Handler/ops.TxContext here, and
// TxContext wraps *cache.Cache), this type drops the handler concept entirely.
// ops.RegisterBuiltins and wire.RegisterRoutableBuiltins both walk the single
// canonical BuiltinOps table (builtin_routing.go) so the two registries can never
// disagree about an op's name, kind, KeyExtractor, or RouteLayout.
type Registry struct {
	mu sync.RWMutex
	m  map[string]routeEntry
}

// NewRegistry constructs an empty routing registry.
func NewRegistry() *Registry {
	return &Registry{m: make(map[string]routeEntry, 16)}
}

// Register adds a shardless op (no routing key).
func (r *Registry) Register(name string, kind OpKind) error {
	return r.register(name, kind, nil, RouteLayoutNone, false)
}

// RegisterRoutable adds an op with an explicit KeyExtractor. Returns an error
// if ke is nil — for shardless ops, use Register.
func (r *Registry) RegisterRoutable(name string, kind OpKind, ke KeyExtractor) error {
	if ke == nil {
		return errors.New("wire: RegisterRoutable requires non-nil KeyExtractor; use Register for shardless ops")
	}
	return r.register(name, kind, ke, RouteLayoutNone, false)
}

// RegisterRoutableCrossShard is like RegisterRoutable but marks the op as
// cross-shard (see ops.Registry.RegisterRoutableCrossShard for the full
// rationale — the routing metadata here mirrors it exactly).
func (r *Registry) RegisterRoutableCrossShard(name string, kind OpKind, ke KeyExtractor) error {
	if ke == nil {
		return errors.New("wire: RegisterRoutableCrossShard requires non-nil KeyExtractor; use Register for shardless ops")
	}
	return r.register(name, kind, ke, RouteLayoutNone, true)
}

// RegisterRoutableInto is RegisterRoutable plus the op's RouteLayout, unlocking
// the allocation-free routing path (RouteKeyInto).
func (r *Registry) RegisterRoutableInto(name string, kind OpKind, ke KeyExtractor, layout RouteLayout) error {
	if ke == nil {
		return errors.New("wire: RegisterRoutableInto requires non-nil KeyExtractor")
	}
	if layout == RouteLayoutNone {
		return errors.New("wire: RegisterRoutableInto requires a RouteLayout; use RegisterRoutable when there is none")
	}
	return r.register(name, kind, ke, layout, false)
}

func (r *Registry) register(name string, kind OpKind, ke KeyExtractor, layout RouteLayout, crossShard bool) error {
	if err := validateName(name); err != nil {
		return err
	}
	if kind != OpReadOnly && kind != OpReadWrite {
		return fmt.Errorf("wire: invalid OpKind %d", kind)
	}
	r.mu.Lock()
	defer r.mu.Unlock()
	if _, exists := r.m[name]; exists {
		return ErrDuplicateOp
	}
	r.m[name] = routeEntry{kind: kind, ke: ke, layout: layout, crossShard: crossShard}
	return nil
}

func validateName(name string) error {
	if name == "" {
		return errors.New("wire: empty op name")
	}
	if len(name) > maxOpNameLen {
		return fmt.Errorf("wire: op name length %d exceeds max %d", len(name), maxOpNameLen)
	}
	// Protocol-v2 wire compatibility: an op name of length 2 is ambiguous with
	// the v2 frame's [version=0x02][...] prefix. Mirrors ops.validateEntry.
	if len(name) == 2 {
		return fmt.Errorf("wire: op name %q has length 2 which collides with protocol v2 version byte; use 3+ chars", name)
	}
	return nil
}

// CrossShard reports whether the named op was registered as cross-shard. False
// for unknown ops.
func (r *Registry) CrossShard(name string) bool {
	r.mu.RLock()
	defer r.mu.RUnlock()
	return r.m[name].crossShard
}

// Lookup returns an op's kind and KeyExtractor (nil for shardless ops), and
// whether the op was found. Mirrors ops.Registry.Lookup's shape minus the
// handler, since routing never needs one.
func (r *Registry) Lookup(name string) (kind OpKind, ke KeyExtractor, ok bool) {
	r.mu.RLock()
	defer r.mu.RUnlock()
	e, ok := r.m[name]
	if !ok {
		return 0, nil, false
	}
	return e.kind, e.ke, true
}

// LookupRouting returns everything a router needs for one op — its kind, its
// KeyExtractor, and its RouteLayout — from ONE registry entry under ONE RLock.
func (r *Registry) LookupRouting(name string) (kind OpKind, ke KeyExtractor, layout RouteLayout, ok bool) {
	r.mu.RLock()
	defer r.mu.RUnlock()
	e, ok := r.m[name]
	if !ok {
		return 0, nil, RouteLayoutNone, false
	}
	return e.kind, e.ke, e.layout, true
}

// LookupEntry is Lookup plus the crossShard flag, read from the SAME entry
// under ONE RLock.
func (r *Registry) LookupEntry(name string) (kind OpKind, ke KeyExtractor, layout RouteLayout, crossShard, ok bool) {
	r.mu.RLock()
	defer r.mu.RUnlock()
	e, ok := r.m[name]
	if !ok {
		return 0, nil, RouteLayoutNone, false, false
	}
	return e.kind, e.ke, e.layout, e.crossShard, true
}
