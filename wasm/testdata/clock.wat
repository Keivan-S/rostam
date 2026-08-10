(module
  (import "wasi_snapshot_preview1" "clock_time_get"
    (func $clock (param i32 i64 i32) (result i32)))
  (func (export "apply") (param i32 i32) (result i32)
    (i32.const 0))
)
