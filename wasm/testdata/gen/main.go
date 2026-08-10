// SPDX-License-Identifier: Apache-2.0
//go:build ignore

// gen produces the binary WASM testdata files from hand-encoded bytes.
// Run with: go run wasm/testdata/gen/main.go
//
// incr.wasm: imports rostam.{cache_get, cache_put, set_result}, exports memory
// and an "apply" function (param i32 i32)(result i32) that is a minimal stub.
//
// clock.wasm: imports wasi_snapshot_preview1.clock_time_get — used to test
// the determinism gate rejects banned imports.
//
// del.wasm / put.wasm: call exactly one state-mutating host function with the
// invoke args as the key and RETURN ITS STATUS VERBATIM. They exist to test the
// host-error boundary: when the host call fails the guest sees only -1, so these
// modules reproduce the exact shape a real update function has when a host
// mutation is refused (see hostStateHolder.hostErr).
package main

import (
	"os"
	"path/filepath"
	"runtime"
)

// encodeULEB128 encodes n as an unsigned LEB128.
func encodeULEB128(n uint32) []byte {
	var out []byte
	for {
		b := byte(n & 0x7F)
		n >>= 7
		if n != 0 {
			b |= 0x80
		}
		out = append(out, b)
		if n == 0 {
			break
		}
	}
	return out
}

func section(id byte, payload []byte) []byte {
	out := []byte{id}
	out = append(out, encodeULEB128(uint32(len(payload)))...)
	out = append(out, payload...)
	return out
}

func vecLen(n int) []byte { return encodeULEB128(uint32(n)) }

// encodeString encodes a WASM name (u32 length prefix + utf-8 bytes).
func encodeString(s string) []byte {
	out := vecLen(len(s))
	out = append(out, []byte(s)...)
	return out
}

// buildIncrWASM builds a minimal WASM module:
//
//	(import "rostam" "cache_get"  (func (param i32 i32)(result i64)))  ; func 0
//	(import "rostam" "cache_put"  (func (param i32 i32 i32 i32 i64)(result i32))) ; func 1
//	(import "rostam" "set_result" (func (param i32 i32)))               ; func 2
//	(memory 1)
//	(func (export "apply") (param i32 i32)(result i32)                  ; func 3
//	  i32.const 0
//	  i32.const 0
//	  call 2          ;; set_result(0,0)
//	  i32.const 0)    ;; return 0
//	(export "memory" (memory 0))
func buildIncrWASM() []byte {
	// --- Type section (section id 1) ---
	// type 0: (param i32 i32)(result i64)           — cache_get
	// type 1: (param i32 i32 i32 i32 i64)(result i32) — cache_put
	// type 2: (param i32 i32)                        — set_result (no result)
	// type 3: (param i32 i32)(result i32)             — apply

	const (
		i32 = 0x7F
		i64 = 0x7E
	)
	funcType := byte(0x60) // func type marker

	type0 := []byte{funcType, 2, i32, i32, 1, i64}                   // cache_get
	type1 := []byte{funcType, 4, i32, i32, i32, i32, 1, i64, 1, i32} // cache_put: but wait i64 param needs 5 params
	_ = type1
	// cache_put: (param i32 i32 i32 i32 i64)(result i32) — 5 params
	type1Fixed := []byte{funcType, 5, i32, i32, i32, i32, i64, 1, i32}
	type2 := []byte{funcType, 2, i32, i32, 0}      // set_result: no results
	type3 := []byte{funcType, 2, i32, i32, 1, i32} // apply

	typePayload := append(vecLen(4))
	typePayload = append(typePayload, type0...)
	typePayload = append(typePayload, type1Fixed...)
	typePayload = append(typePayload, type2...)
	typePayload = append(typePayload, type3...)

	// --- Import section (section id 2) ---
	// 3 imports: cache_get(type 0), cache_put(type 1), set_result(type 2)
	mkFuncImport := func(mod, name string, typeIdx uint32) []byte {
		var b []byte
		b = append(b, encodeString(mod)...)
		b = append(b, encodeString(name)...)
		b = append(b, 0x00) // import kind: function
		b = append(b, encodeULEB128(typeIdx)...)
		return b
	}
	importPayload := vecLen(3)
	importPayload = append(importPayload, mkFuncImport("rostam", "cache_get", 0)...)
	importPayload = append(importPayload, mkFuncImport("rostam", "cache_put", 1)...)
	importPayload = append(importPayload, mkFuncImport("rostam", "set_result", 2)...)

	// --- Function section (section id 3) ---
	// 1 function: type 3 (apply)
	funcPayload := append(vecLen(1), encodeULEB128(3)...)

	// --- Memory section (section id 5) ---
	// 1 memory: min 1 page, no max
	memPayload := []byte{0x01, 0x00, 0x01} // count=1, limit type=0 (min only), min=1

	// --- Export section (section id 7) ---
	mkExport := func(name string, kind byte, idx uint32) []byte {
		var b []byte
		b = append(b, encodeString(name)...)
		b = append(b, kind)
		b = append(b, encodeULEB128(idx)...)
		return b
	}
	exportPayload := vecLen(2)
	exportPayload = append(exportPayload, mkExport("memory", 0x02, 0)...) // memory export
	exportPayload = append(exportPayload, mkExport("apply", 0x00, 3)...)  // func export (func index 3 = 3 imports + 0)

	// --- Code section (section id 10) ---
	// Body of "apply":
	//   locals: 0
	//   i32.const 0
	//   i32.const 0
	//   call 2        (set_result)
	//   i32.const 0
	//   end
	applyBody := []byte{
		0x00,       // 0 local decls
		0x41, 0x00, // i32.const 0
		0x41, 0x00, // i32.const 0
		0x10, 0x02, // call 2 (set_result)
		0x41, 0x00, // i32.const 0
		0x0B, // end
	}
	// Encode function body with its size prefix
	funcBody := append(encodeULEB128(uint32(len(applyBody))), applyBody...)
	codePayload := append(vecLen(1), funcBody...)

	// Assemble sections
	var module []byte
	module = append(module, 0x00, 0x61, 0x73, 0x6D) // magic
	module = append(module, 0x01, 0x00, 0x00, 0x00) // version
	module = append(module, section(1, typePayload)...)
	module = append(module, section(2, importPayload)...)
	module = append(module, section(3, funcPayload)...)
	module = append(module, section(5, memPayload)...)
	module = append(module, section(7, exportPayload)...)
	module = append(module, section(10, codePayload)...)
	return module
}

// buildClockWASM builds a minimal WASM module that imports
// wasi_snapshot_preview1.clock_time_get — used to test that
// the determinism gate rejects it.
//
//	(import "wasi_snapshot_preview1" "clock_time_get" (func (param i32 i64 i32)(result i32)))
//	(func (export "apply") (param i32 i32)(result i32) i32.const 0)
func buildClockWASM() []byte {
	const (
		i32 = 0x7F
		i64 = 0x7E
	)
	funcType := byte(0x60)

	// type 0: (param i32 i64 i32)(result i32) — clock_time_get
	type0 := []byte{funcType, 3, i32, i64, i32, 1, i32}
	// type 1: (param i32 i32)(result i32) — apply
	type1 := []byte{funcType, 2, i32, i32, 1, i32}

	typePayload := vecLen(2)
	typePayload = append(typePayload, type0...)
	typePayload = append(typePayload, type1...)

	encodeString := func(s string) []byte {
		out := vecLen(len(s))
		out = append(out, []byte(s)...)
		return out
	}

	importPayload := vecLen(1)
	importPayload = append(importPayload, encodeString("wasi_snapshot_preview1")...)
	importPayload = append(importPayload, encodeString("clock_time_get")...)
	importPayload = append(importPayload, 0x00)                // func import
	importPayload = append(importPayload, encodeULEB128(0)...) // type 0

	funcPayload := append(vecLen(1), encodeULEB128(1)...) // 1 func, type 1

	// apply body: i32.const 0; end
	applyBody := []byte{0x00, 0x41, 0x00, 0x0B}
	funcBody := append(encodeULEB128(uint32(len(applyBody))), applyBody...)
	codePayload := append(vecLen(1), funcBody...)

	mkExport := func(name string, kind byte, idx uint32) []byte {
		var b []byte
		b = append(b, encodeString(name)...)
		b = append(b, kind)
		b = append(b, encodeULEB128(idx)...)
		return b
	}
	exportPayload := vecLen(1)
	exportPayload = append(exportPayload, mkExport("apply", 0x00, 1)...) // func 0=import, 1=apply

	var module []byte
	module = append(module, 0x00, 0x61, 0x73, 0x6D)
	module = append(module, 0x01, 0x00, 0x00, 0x00)
	module = append(module, section(1, typePayload)...)
	module = append(module, section(2, importPayload)...)
	module = append(module, section(3, funcPayload)...)
	module = append(module, section(7, exportPayload)...)
	module = append(module, section(10, codePayload)...)
	return module
}

// buildDelWASM builds a module whose apply forwards its args MINUS THE 2-BYTE
// "std" KEY-LENGTH PREFIX to cache_del as the key, and returns the host's status
// unchanged:
//
//	(import "rostam" "cache_del" (func (param i32 i32)(result i32)))  ; func 0
//	(memory (export "memory") 1)
//	(func (export "apply") (param i32 i32)(result i32)                ; func 1
//	  local.get 0 i32.const 2 i32.add
//	  local.get 1 i32.const 2 i32.sub
//	  call 0)   ;; returns 1 / 0 / -1 exactly as the host produced it
//
// THE +2 IS NOT DECORATION. ops.WASMKeyExtractorHandle pinned every WASM op to
// the "std" extractor, so a routable op's args are [keyLen u16][key][payload] and
// the module receives that WHOLE frame — the extractor selects the ROUTING key,
// it does not rewrite the args. A module that passed its raw args to cache_del
// would therefore address the key "\x00\x05hello" rather than "hello", which
// hashes to a different shard than the group the entry is executing in. Skipping
// the prefix is what keeps the key this module touches equal to the key its
// invocation routed on.
//
// It reads no length: these probes are always invoked with an EMPTY payload, so
// args[2:] IS the key. A module carrying a real payload would have to decode the
// big-endian u16 (two byte loads and a shift — WASM loads are little-endian).
func buildDelWASM() []byte {
	const i32 = 0x7F
	funcType := byte(0x60)

	// One type serves both cache_del and apply: (param i32 i32)(result i32).
	typePayload := vecLen(1)
	typePayload = append(typePayload, funcType, 2, i32, i32, 1, i32)

	importPayload := vecLen(1)
	importPayload = append(importPayload, encodeString("rostam")...)
	importPayload = append(importPayload, encodeString("cache_del")...)
	importPayload = append(importPayload, 0x00)                // func import
	importPayload = append(importPayload, encodeULEB128(0)...) // type 0

	funcPayload := append(vecLen(1), encodeULEB128(0)...) // 1 func, type 0
	memPayload := []byte{0x01, 0x00, 0x01}                // count=1, min=1, no max

	exportPayload := vecLen(2)
	exportPayload = append(exportPayload, encodeString("memory")...)
	exportPayload = append(exportPayload, 0x02, 0x00) // memory 0
	exportPayload = append(exportPayload, encodeString("apply")...)
	exportPayload = append(exportPayload, 0x00, 0x01) // func 1 (0 = the import)

	applyBody := []byte{
		0x00,       // 0 local decls
		0x20, 0x00, // local.get 0 (argsPtr)
		0x41, 0x02, // i32.const 2
		0x6A,       // i32.add          → keyPtr = argsPtr + 2
		0x20, 0x01, // local.get 1 (argsLen)
		0x41, 0x02, // i32.const 2
		0x6B,       // i32.sub          → keyLen = argsLen - 2
		0x10, 0x00, // call 0 (cache_del)
		0x0B, // end — the call's i32 result is the return value
	}
	funcBody := append(encodeULEB128(uint32(len(applyBody))), applyBody...)
	codePayload := append(vecLen(1), funcBody...)

	var module []byte
	module = append(module, 0x00, 0x61, 0x73, 0x6D)
	module = append(module, 0x01, 0x00, 0x00, 0x00)
	module = append(module, section(1, typePayload)...)
	module = append(module, section(2, importPayload)...)
	module = append(module, section(3, funcPayload)...)
	module = append(module, section(5, memPayload)...)
	module = append(module, section(7, exportPayload)...)
	module = append(module, section(10, codePayload)...)
	return module
}

// buildPutWASM is buildDelWASM's counterpart for cache_put: apply writes its args
// MINUS THE 2-BYTE "std" KEY-LENGTH PREFIX as BOTH key and value with no TTL, and
// returns the host's status unchanged. See buildDelWASM for why the prefix is
// skipped.
//
//	(import "rostam" "cache_put" (func (param i32 i32 i32 i32 i64)(result i32))) ; func 0
//	(memory (export "memory") 1)
//	(func (export "apply") (param i32 i32)(result i32)                           ; func 1
//	  local.get 0 i32.const 2 i32.add   local.get 1 i32.const 2 i32.sub
//	  local.get 0 i32.const 2 i32.add   local.get 1 i32.const 2 i32.sub
//	  i64.const 0
//	  call 0)
func buildPutWASM() []byte {
	const (
		i32 = 0x7F
		i64 = 0x7E
	)
	funcType := byte(0x60)

	// type 0: cache_put; type 1: apply.
	typePayload := vecLen(2)
	typePayload = append(typePayload, funcType, 5, i32, i32, i32, i32, i64, 1, i32)
	typePayload = append(typePayload, funcType, 2, i32, i32, 1, i32)

	importPayload := vecLen(1)
	importPayload = append(importPayload, encodeString("rostam")...)
	importPayload = append(importPayload, encodeString("cache_put")...)
	importPayload = append(importPayload, 0x00)
	importPayload = append(importPayload, encodeULEB128(0)...)

	funcPayload := append(vecLen(1), encodeULEB128(1)...) // 1 func, type 1
	memPayload := []byte{0x01, 0x00, 0x01}

	exportPayload := vecLen(2)
	exportPayload = append(exportPayload, encodeString("memory")...)
	exportPayload = append(exportPayload, 0x02, 0x00)
	exportPayload = append(exportPayload, encodeString("apply")...)
	exportPayload = append(exportPayload, 0x00, 0x01)

	applyBody := []byte{
		0x00,       // 0 local decls
		0x20, 0x00, // local.get 0 (argsPtr)
		0x41, 0x02, // i32.const 2
		0x6A,       // i32.add          → keyPtr = argsPtr + 2
		0x20, 0x01, // local.get 1 (argsLen)
		0x41, 0x02, // i32.const 2
		0x6B,       // i32.sub          → keyLen = argsLen - 2
		0x20, 0x00, // local.get 0
		0x41, 0x02, // i32.const 2
		0x6A,       // i32.add          → valPtr — the same bytes as the key
		0x20, 0x01, // local.get 1
		0x41, 0x02, // i32.const 2
		0x6B,       // i32.sub          → valLen
		0x42, 0x00, // i64.const 0 (no TTL)
		0x10, 0x00, // call 0 (cache_put)
		0x0B, // end
	}
	funcBody := append(encodeULEB128(uint32(len(applyBody))), applyBody...)
	codePayload := append(vecLen(1), funcBody...)

	var module []byte
	module = append(module, 0x00, 0x61, 0x73, 0x6D)
	module = append(module, 0x01, 0x00, 0x00, 0x00)
	module = append(module, section(1, typePayload)...)
	module = append(module, section(2, importPayload)...)
	module = append(module, section(3, funcPayload)...)
	module = append(module, section(5, memPayload)...)
	module = append(module, section(7, exportPayload)...)
	module = append(module, section(10, codePayload)...)
	return module
}

// buildReadOnlyWASM builds a module that imports ONLY non-mutating host
// functions, so it passes the OpReadOnly/writes-state guard:
//
//	(import "rostam" "cache_get"  (func (param i32 i32)(result i64)))  ; func 0
//	(import "rostam" "set_result" (func (param i32 i32)))              ; func 1
//	(memory 1)
//	(func (export "apply") (param i32 i32)(result i32)                 ; func 2
//	  i32.const 0 i32.const 0 call 1   ;; set_result(0,0)
//	  i32.const 0)
//	(export "memory" (memory 0))
//
// It is the only testdata module that can legitimately be registered as
// ops.OpReadOnly, which is what makes it the right probe for anything that has
// to distinguish "refused by the kind guard" from "accepted silently" — a
// torn sidecar decodes to Kind 0 (OpReadOnly), and with a state-writing module
// the guard masks the outcome under test.
func buildReadOnlyWASM() []byte {
	const (
		i32 = 0x7F
		i64 = 0x7E
	)
	funcType := byte(0x60)

	type0 := []byte{funcType, 2, i32, i32, 1, i64} // cache_get
	type1 := []byte{funcType, 2, i32, i32, 0}      // set_result (no results)
	type2 := []byte{funcType, 2, i32, i32, 1, i32} // apply

	typePayload := vecLen(3)
	typePayload = append(typePayload, type0...)
	typePayload = append(typePayload, type1...)
	typePayload = append(typePayload, type2...)

	mkFuncImport := func(mod, name string, typeIdx uint32) []byte {
		var b []byte
		b = append(b, encodeString(mod)...)
		b = append(b, encodeString(name)...)
		b = append(b, 0x00) // import kind: function
		b = append(b, encodeULEB128(typeIdx)...)
		return b
	}
	importPayload := vecLen(2)
	importPayload = append(importPayload, mkFuncImport("rostam", "cache_get", 0)...)
	importPayload = append(importPayload, mkFuncImport("rostam", "set_result", 1)...)

	funcPayload := append(vecLen(1), encodeULEB128(2)...) // 1 func, type 2
	memPayload := []byte{0x01, 0x00, 0x01}                // 1 memory, min 1 page

	mkExport := func(name string, kind byte, idx uint32) []byte {
		var b []byte
		b = append(b, encodeString(name)...)
		b = append(b, kind)
		b = append(b, encodeULEB128(idx)...)
		return b
	}
	exportPayload := vecLen(2)
	exportPayload = append(exportPayload, mkExport("memory", 0x02, 0)...)
	exportPayload = append(exportPayload, mkExport("apply", 0x00, 2)...) // 2 imports + 0

	applyBody := []byte{
		0x00,       // 0 local decls
		0x41, 0x00, // i32.const 0
		0x41, 0x00, // i32.const 0
		0x10, 0x01, // call 1 (set_result)
		0x41, 0x00, // i32.const 0
		0x0B, // end
	}
	funcBody := append(encodeULEB128(uint32(len(applyBody))), applyBody...)
	codePayload := append(vecLen(1), funcBody...)

	var module []byte
	module = append(module, 0x00, 0x61, 0x73, 0x6D)
	module = append(module, 0x01, 0x00, 0x00, 0x00)
	module = append(module, section(1, typePayload)...)
	module = append(module, section(2, importPayload)...)
	module = append(module, section(3, funcPayload)...)
	module = append(module, section(5, memPayload)...)
	module = append(module, section(7, exportPayload)...)
	module = append(module, section(10, codePayload)...)
	return module
}

func main() {
	_, file, _, _ := runtime.Caller(0)
	dir := filepath.Dir(file)

	write := func(name string, data []byte) {
		path := filepath.Join(dir, "..", name)
		if err := os.WriteFile(path, data, 0o644); err != nil {
			panic(err)
		}
	}

	write("incr.wasm", buildIncrWASM())
	write("readonly.wasm", buildReadOnlyWASM())
	write("clock.wasm", buildClockWASM())
	write("del.wasm", buildDelWASM())
	write("put.wasm", buildPutWASM())
}
