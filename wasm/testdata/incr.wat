(module
  (import "rostam" "cache_get"
    (func $cache_get (param i32 i32) (result i64)))
  (import "rostam" "cache_put"
    (func $cache_put (param i32 i32 i32 i32 i64) (result i32)))
  (import "rostam" "set_result"
    (func $set_result (param i32 i32)))
  (memory (export "memory") 1)
  (data (i32.const 0) "\00\00\00\00\00\00\00\00") ;; scratch 8 bytes
  (func (export "apply") (param i32 i32) (result i32)
    ;; minimal stub: ignore args, set_result to empty, return 0
    (call $set_result (i32.const 0) (i32.const 0))
    (i32.const 0))
)
