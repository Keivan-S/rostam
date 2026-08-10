(module
  (import "rostam" "cache_del"
    (func $cache_del (param i32 i32) (result i32)))
  (memory (export "memory") 1)
  ;; apply forwards the invoke args MINUS THE 2-BYTE "std" KEY-LENGTH PREFIX as
  ;; the key and returns cache_del's status VERBATIM (1 existed / 0 absent / -1
  ;; the host refused). -1 is all a guest ever learns about a host failure, which
  ;; is why the host has to carry the real error out itself.
  ;;
  ;; See put.wat for why the prefix is skipped: every WASM op is pinned to the
  ;; "std" key extractor, so args are [keyLen u16][key][payload] and the module
  ;; receives the whole frame. This probe is always invoked with an EMPTY
  ;; payload, so args[2:] IS the key and no length has to be decoded.
  (func (export "apply") (param i32 i32) (result i32)
    (call $cache_del
      (i32.add (local.get 0) (i32.const 2))
      (i32.sub (local.get 1) (i32.const 2))))
)
