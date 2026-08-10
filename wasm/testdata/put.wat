(module
  (import "rostam" "cache_put"
    (func $cache_put (param i32 i32 i32 i32 i64) (result i32)))
  (memory (export "memory") 1)
  ;; apply writes the invoke args MINUS THE 2-BYTE "std" KEY-LENGTH PREFIX as
  ;; both key and value with no TTL, and returns cache_put's status VERBATIM
  ;; (0 ok / -1 the host refused).
  ;;
  ;; The +2 is load-bearing. Every WASM op is pinned to the "std" key extractor
  ;; (ops.WASMKeyExtractorHandle), so args are [keyLen u16][key][payload] and the
  ;; module receives that WHOLE frame — the extractor picks the ROUTING key, it
  ;; does not rewrite the args. Passing the raw args here would address the key
  ;; "\00\05hello" instead of "hello", which hashes to a different shard than the
  ;; group this entry is executing in.
  ;;
  ;; It reads no length: this probe is always invoked with an EMPTY payload, so
  ;; args[2:] IS the key.
  (func (export "apply") (param i32 i32) (result i32)
    (call $cache_put
      (i32.add (local.get 0) (i32.const 2)) (i32.sub (local.get 1) (i32.const 2))
      (i32.add (local.get 0) (i32.const 2)) (i32.sub (local.get 1) (i32.const 2))
      (i64.const 0)))
)
