//go:build !cgo

// SPDX-License-Identifier: Apache-2.0

// Package wasm's non-cgo build. The wasmtime-go backend (Cranelift JIT)
// requires cgo, so under CGO_ENABLED=0 the WASM stored-procedure sandbox
// is unavailable. This file provides the SAME exported surface as the cgo
// backend (wasmtime_backend.go) — identical types, constructors, and method
// signatures — so packages that import wasm (the embedded/direct engine and
// cluster node) still compile and run for KV and vector workloads. Any
// operation that would actually compile or execute a module returns
// ErrNoCGO; NewRuntime itself succeeds so a node that never registers a WASM
// module starts normally.
package wasm

import (
	"errors"

	"github.com/rostamlabs/rostam/ops"
)

// ErrNoCGO is returned by every operation that would compile or execute a
// WASM module in a build without cgo. Stored procedures need the wasmtime
// backend, which is only linked when CGO_ENABLED=1.
var ErrNoCGO = errors.New("wasm: stored procedures require a cgo build")

// ErrBannedImport mirrors the cgo backend's sentinel so the exported surface
// is identical across build configs. The non-cgo path never reaches the
// determinism gate (Compile fails first with ErrNoCGO), so it is unused here.
var ErrBannedImport = errors.New("wasm: module imports a banned function")

// Module is the non-cgo counterpart of the cgo backend's Module. It carries
// no compiled artifact because modules cannot be compiled without cgo; it
// exists only so the exported type name resolves for importers.
type Module struct {
	source []byte
}

// Bytes returns the original WASM source bytes.
func (m *Module) Bytes() []byte { return m.source }

// WritesState mirrors the cgo backend's accessor. A non-cgo build never
// produces a Module (Compile always fails), so the value is unreachable.
func (m *Module) WritesState() bool { return false }

// Close is a no-op; the non-cgo Module holds no runtime resources.
func (m *Module) Close() error { return nil }

// Compile always fails in a non-cgo build: validating and compiling a module
// requires the wasmtime engine, which is not linked without cgo.
func Compile(_ []byte) (*Module, error) {
	return nil, ErrNoCGO
}

// Runtime is the non-cgo counterpart of the cgo backend's Runtime. It holds
// no engine and executes nothing; it exists so NewRuntime can hand back a
// valid value and node startup succeeds even though WASM ops are unavailable.
type Runtime struct {
	// groupBindingState carries the per-group version table so the shared,
	// build-tag-free resolver in registry.go compiles under CGO_ENABLED=0 too. It
	// is never populated here: nothing can be instantiated, so nothing can be
	// invoked.
	groupBindingState
}

// NewRuntime returns an inert Runtime. It intentionally succeeds so a node or
// direct engine that never registers a WASM module starts normally under
// CGO_ENABLED=0; the unavailability surfaces later, at Compile/AddModule time.
func NewRuntime() (*Runtime, error) {
	return &Runtime{}, nil
}

// AddModule always fails in a non-cgo build: there is no engine to instantiate
// the module against.
func (r *Runtime) AddModule(_ *Module, _ string, _ uint64) (ModuleID, error) {
	return ModuleID{}, ErrNoCGO
}

// HasModule always reports false in a non-cgo build: AddModule can never
// succeed, so no module is ever instantiated. Mirrors the cgo backend's
// signature so callers compile under CGO_ENABLED=0.
func (r *Runtime) HasModule(_ ModuleID) bool {
	return false
}

// HoldModuleTableForTest mirrors the cgo backend's test hook so cluster's
// no-apply-lock test compiles under CGO_ENABLED=0. There is no module table and
// no lock here — nothing can be instantiated — so it holds nothing and the test
// it serves degenerates to its wasmApplyMu half, which is correct: without cgo
// there is no AddModule to deadlock against.
func (r *Runtime) HoldModuleTableForTest() (release func()) {
	return func() {}
}

// Invoke always fails in a non-cgo build: no module can be registered, so this
// is unreachable in practice, but it returns a clear error defensively.
func (r *Runtime) Invoke(_ ModuleID, _ *ops.TxContext, _ []byte) ([]byte, error) {
	return nil, ErrNoCGO
}

// Close is a no-op; the inert Runtime owns no engine, goroutine, or store.
func (r *Runtime) Close() error { return nil }

// moduleWritesState reports whether an instantiated module mutates state. No
// module can be instantiated without cgo, so it always reports "not found".
// RegisterModule (in registry.go, shared across build configs) calls it.
func (r *Runtime) moduleWritesState(_ ModuleID) (writes, ok bool) {
	return false, false
}

// residentModuleState reports every module as RESIDENT in a non-cgo build, which
// is the opposite of what moduleWritesState above reports and is deliberate.
//
// resolveModuleForInvoke turns "not resident" into ops.ErrWASMModuleNotResident,
// which shard.classifyApplyErr maps to classRetry: do not advance, wait for the
// bytes to arrive, re-run. That is exactly right for a node that is MISSING a
// blob and can go and get it. It is exactly wrong for a build that has no engine
// at all: no fetch could ever help, and a WAIT WOULD BE FOREVER — a non-cgo node
// would silently wedge every group that ever applies a WASM invocation instead of
// failing. Reporting resident sends the call on to Invoke, which returns ErrNoCGO,
// which is the honest answer this whole build gives to every WASM operation.
//
// It is unreachable in practice (nothing can be registered without an AddModule
// that works), and it is the safe direction if that ever stops being true.
func (r *Runtime) residentModuleState(_ ModuleID) (writes, resident bool) {
	return false, true
}
