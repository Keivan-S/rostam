//go:build cgo

// SPDX-License-Identifier: Apache-2.0

package wasm

import (
	"strings"
	"testing"

	wasmtime "github.com/bytecodealliance/wasmtime-go/v45"

	"github.com/rostamlabs/rostam/ops"
)

// watImporting returns a minimal module that imports the named rostam host
// function (with the correct signature) and exports a no-op "apply". The module
// only needs to IMPORT the function to be able to call it, which is exactly what
// the OpReadOnly-no-write gate keys on.
func watImporting(host string) string {
	var imp string
	switch host {
	case "cache_get":
		imp = `(import "rostam" "cache_get" (func (param i32 i32) (result i64)))`
	case "cache_put":
		imp = `(import "rostam" "cache_put" (func (param i32 i32 i32 i32 i64) (result i32)))`
	case "cache_del":
		imp = `(import "rostam" "cache_del" (func (param i32 i32) (result i32)))`
	case "cache_expire":
		imp = `(import "rostam" "cache_expire" (func (param i32 i32 i64) (result i32)))`
	default:
		panic("unknown host func: " + host)
	}
	return `(module
  ` + imp + `
  (memory (export "memory") 1)
  (func (export "apply") (param i32 i32) (result i32) (i32.const 0)))`
}

func compileWAT(t *testing.T, wat string) *Module {
	t.Helper()
	b, err := wasmtime.Wat2Wasm(wat)
	if err != nil {
		t.Fatalf("Wat2Wasm: %v", err)
	}
	m, err := Compile(b)
	if err != nil {
		t.Fatalf("Compile: %v", err)
	}
	return m
}

// A WASM op declared OpReadOnly bypasses Raft (served directly on one replica),
// so a write from it would silently diverge replicas. A read-only module that
// imports any state-mutating host function must therefore never EXECUTE.
//
// ############ THE GUARD MOVED FROM REGISTRATION TO INVOCATION ############
//
// This test used to require RegisterModule to REFUSE, and to leave the op out of
// the registry. Thin registration markers made that refusal unsafe, and the
// reason is worth stating in full because it is the opposite of the obvious
// intuition — "refuse earlier" is normally strictly better.
//
// A marker names its module by content address and does not carry it, so whether
// a node can evaluate this guard at registration time depends on whether that
// node has FETCHED the bytes yet. That is not a function of the log, and the old
// implementation silently encoded it: the check was gated on
// rt.moduleWritesState(id) returning ok && writes, so it FIRED on a node holding
// the module and PASSED on a node that was not. Concretely:
//
//	node A holds the blob, refuses the registration, and does not register the
//	op. Node B has not fetched it, so the guard passes, B registers the op,
//	opens its route gate, and proposes an invocation into a shard group — which
//	A then applies, cannot look the op up, and HALTS on under classFatal
//	shard.ErrOpNotRegistered.
//
// One node's residency would halt a different node. So the verdict moved to
// resolveModuleForInvoke, which asks it once per invocation on EVERY node and
// therefore reaches the same answer everywhere, whenever the bytes arrive.
//
// THE SAFETY PROPERTY IS STRENGTHENED, NOT WEAKENED, and that is the point of
// the move rather than a consolation:
//
//   - the old check could be SKIPPED ENTIRELY by non-residency. This one cannot
//     be skipped at all — a module that is not resident does not execute either;
//   - the old check ran once, at install. This one covers a module that becomes
//     resident LATER, which is now a routine occurrence;
//   - what the op must never do is WRITE, and no write can happen without an
//     invocation. Refusing the invocation is refusing the hazard itself, where
//     refusing the registration was refusing a precondition of it.
//
// The two paths that legitimately want the early refusal — operator config
// (cluster.loadOneModule) and Direct mode (rostam.directStore.RegisterWASM) —
// call ValidateModuleKind explicitly on the compiled module they are holding.
// Both are node-local, always have the bytes, and have no replica to disagree
// with.
func TestReadOnlyWithWriteImportIsRefusedAtInvocation(t *testing.T) {
	for _, host := range []string{"cache_put", "cache_del", "cache_expire"} {
		t.Run(host, func(t *testing.T) {
			rt, err := NewRuntime()
			if err != nil {
				t.Fatalf("NewRuntime: %v", err)
			}
			defer func() { _ = rt.Close() }()

			m := compileWAT(t, watImporting(host))
			const op = "ro_writer"
			id, err := rt.AddModule(m, "apply", 0)
			if err != nil {
				t.Fatalf("AddModule: %v", err)
			}
			reg := ops.NewRegistry()
			if err := RegisterModule(reg, rt, op, id, ops.OpReadOnly, nil); err != nil {
				t.Fatalf("RegisterModule must NOT refuse: the verdict needs the module, so a refusal here would depend on byte residency and could halt a peer; got %v", err)
			}
			// The op IS in the registry — withholding it is what would produce the
			// classFatal ErrOpNotRegistered halt on a replica that applies an
			// invocation of it.
			fn, kind, _, ok := reg.Lookup(op)
			if !ok || kind != ops.OpReadOnly {
				t.Fatalf("op not registered as OpReadOnly (ok=%v kind=%v)", ok, kind)
			}
			// And it REFUSES TO RUN, which is the property that actually matters:
			// no invocation, no write, no divergence.
			_, err = fn(ops.NewTxContext(nil), nil)
			if err == nil {
				t.Fatalf("an OpReadOnly module importing %s executed; a write from it would never be logged and would silently diverge this replica", host)
			}
			if !strings.Contains(err.Error(), "read-only") {
				t.Fatalf("error %q should explain the read-only/write-import conflict", err)
			}
		})
	}
}

// A read-only module that only reads (cache_get) is fine.
func TestRegisterModuleAllowsReadOnlyReadImport(t *testing.T) {
	rt, err := NewRuntime()
	if err != nil {
		t.Fatalf("NewRuntime: %v", err)
	}
	defer func() { _ = rt.Close() }()

	m := compileWAT(t, watImporting("cache_get"))
	const op = "ro_reader"
	id, err := rt.AddModule(m, "apply", 0)
	if err != nil {
		t.Fatalf("AddModule: %v", err)
	}
	reg := ops.NewRegistry()
	if err := RegisterModule(reg, rt, op, id, ops.OpReadOnly, nil); err != nil {
		t.Fatalf("RegisterModule rejected a read-only read-only-import module: %v", err)
	}
	fn, kind, _, ok := reg.Lookup(op)
	if !ok || kind != ops.OpReadOnly {
		t.Fatalf("op not registered as OpReadOnly (ok=%v kind=%v)", ok, kind)
	}
	// The resolver's kind guard must not fire on a read-only module that only
	// READS: it now runs on every invocation, so a false positive here would
	// break every legitimate read-only UDF rather than merely its registration.
	if _, err := fn(ops.NewTxContext(nil), nil); err != nil && strings.Contains(err.Error(), "read-only") {
		t.Fatalf("the invocation-time kind guard rejected a read-only module that only reads: %v", err)
	}
}

// A read-WRITE module importing a write host function is exactly what the write
// path is for — it must still register.
func TestRegisterModuleAllowsReadWriteWithWriteImport(t *testing.T) {
	rt, err := NewRuntime()
	if err != nil {
		t.Fatalf("NewRuntime: %v", err)
	}
	defer func() { _ = rt.Close() }()

	m := compileWAT(t, watImporting("cache_put"))
	const op = "rw_writer"
	id, err := rt.AddModule(m, "apply", 0)
	if err != nil {
		t.Fatalf("AddModule: %v", err)
	}
	reg := ops.NewRegistry()
	if err := RegisterModule(reg, rt, op, id, ops.OpReadWrite, nil); err != nil {
		t.Fatalf("RegisterModule rejected a legitimate OpReadWrite writer: %v", err)
	}
	if _, kind, _, ok := reg.Lookup(op); !ok || kind != ops.OpReadWrite {
		t.Fatalf("op not registered as OpReadWrite (ok=%v kind=%v)", ok, kind)
	}
}
