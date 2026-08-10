(module
  ;; The only testdata module that imports NO state-mutating host function, so
  ;; it is the only one that can legitimately be registered as ops.OpReadOnly.
  ;; Probes that need to tell "refused by the read-only/writes-state guard" apart
  ;; from "accepted silently" must use this one — a torn sidecar decodes to
  ;; Kind 0 (OpReadOnly), and with a writing module the guard masks the outcome.
  (import "rostam" "cache_get"
    (func $cache_get (param i32 i32) (result i64)))
  (import "rostam" "set_result"
    (func $set_result (param i32 i32)))
  (memory (export "memory") 1)
  (func (export "apply") (param i32 i32) (result i32)
    (call $set_result (i32.const 0) (i32.const 0))
    (i32.const 0))
)
