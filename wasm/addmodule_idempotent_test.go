//go:build cgo

// SPDX-License-Identifier: Apache-2.0

package wasm

import (
	"os"
	"testing"
)

// TestAddModuleIdenticalIsNoOp pins AddModule's idempotency. A WASM
// registration is replicated to EVERY shard Raft group, so on a node hosting k
// groups the FSM hook calls AddModule k times with the same module — plus once
// more per client retry. Before the content check each of those spent a
// Cranelift compile and a fresh wasmtime.Store and then dropped the previous
// pair on the floor (reclaimed only by wasmtime-go's finalizers, i.e. whenever
// the GC next ran), while an in-flight Invoke could still be running against
// the displaced slot.
//
// Same module ⇒ the SAME ModuleID and the SAME runtimeModule pointer (nothing
// was rebuilt).
func TestAddModuleIdenticalIsNoOp(t *testing.T) {
	wasmBytes, err := os.ReadFile("testdata/incr.wasm")
	if err != nil {
		t.Fatal(err)
	}
	rt, err := NewRuntime()
	if err != nil {
		t.Fatalf("NewRuntime: %v", err)
	}
	defer rt.Close() //nolint:errcheck // test cleanup

	add := func(src []byte, exportName string, maxFuel uint64) (ModuleID, *runtimeModule) {
		t.Helper()
		m, err := Compile(src)
		if err != nil {
			t.Fatalf("Compile: %v", err)
		}
		id, err := rt.AddModule(m, exportName, maxFuel)
		if err != nil {
			t.Fatalf("AddModule: %v", err)
		}
		rt.mu.RLock()
		defer rt.mu.RUnlock()
		return id, rt.modules[id]
	}

	firstID, first := add(wasmBytes, "apply", 0)
	for i := 0; i < 4; i++ {
		gotID, got := add(wasmBytes, "apply", 0)
		if gotID != firstID {
			t.Fatalf("repeat %d produced a different ModuleID (%s -> %s)", i+1, firstID, gotID)
		}
		if got != first {
			t.Fatalf("repeat %d re-instantiated the module (slot %p -> %p); the identical-add fast path did not fire", i+1, first, got)
		}
	}

	if !rt.HasModule(firstID) {
		t.Error("HasModule = false for the module that is instantiated")
	}
	// Export-name and fuel-budget defaults are normalized before hashing, so the
	// explicit and implicit spellings must name the same slot.
	if ModuleIDFor(wasmBytes, "", 0) != firstID {
		t.Error(`an empty export name did not normalize to "apply"`)
	}
	if rt.HasModule(ModuleIDFor(wasmBytes, "apply", 7)) {
		t.Error("HasModule = true for a DIFFERENT fuel budget; the ModuleID must cover maxFuel")
	}
	if rt.HasModule(ModuleIDFor(wasmBytes, "other_export", 0)) {
		t.Error("HasModule = true for a DIFFERENT export name; the ModuleID must cover the export")
	}

	// A genuinely different module gets its OWN slot — and, critically, does not
	// take the first one's. That is the property the whole content-addressed
	// redesign rests on: installing a second version never displaces the first,
	// so there is nothing to roll back when a registration fails afterwards.
	otherBytes, err := os.ReadFile("testdata/put.wasm")
	if err != nil {
		t.Fatal(err)
	}
	secondID, second := add(otherBytes, "apply", 0)
	if secondID == firstID {
		t.Fatal("two different modules hashed to the same ModuleID")
	}
	if second == first {
		t.Fatal("a different module reused the first module's slot")
	}
	if !rt.HasModule(firstID) {
		t.Error("installing a second module DISPLACED the first; versions must coexist")
	}
	rt.mu.RLock()
	stillFirst := rt.modules[firstID]
	rt.mu.RUnlock()
	if stillFirst != first {
		t.Errorf("the first module's slot was rebuilt (%p -> %p)", first, stillFirst)
	}
}

// TestTwoOpsShareOneModuleSlot pins the dedup half of content addressing: the
// slot is named by what it IS, so two ops registered from identical bytes,
// export and fuel share one instantiation rather than paying a second Cranelift
// compile and a second wasmtime.Store for the same code.
func TestTwoOpsShareOneModuleSlot(t *testing.T) {
	src, err := os.ReadFile("testdata/incr.wasm")
	if err != nil {
		t.Fatal(err)
	}
	rt, err := NewRuntime()
	if err != nil {
		t.Fatalf("NewRuntime: %v", err)
	}
	defer rt.Close() //nolint:errcheck // test cleanup

	add := func() ModuleID {
		t.Helper()
		m, err := Compile(src)
		if err != nil {
			t.Fatalf("Compile: %v", err)
		}
		id, err := rt.AddModule(m, "apply", 0)
		if err != nil {
			t.Fatalf("AddModule: %v", err)
		}
		return id
	}

	// Two independent registrations — as two different op names would produce.
	a, b := add(), add()
	if a != b {
		t.Fatalf("identical bytes produced two ModuleIDs (%s, %s)", a, b)
	}
	rt.mu.RLock()
	n := len(rt.modules)
	rt.mu.RUnlock()
	if n != 1 {
		t.Errorf("runtime holds %d slots for one module's worth of content, want 1", n)
	}
}
