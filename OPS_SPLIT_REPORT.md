# ops → ops/wire protocol-module split

## Leaf package

`ops/wire` (import path `github.com/rostamlabs/rostam/ops/wire`). Imports only
`github.com/rostamlabs/rostam/vector`, `github.com/rostamlabs/rostam/grpcapi/pb`,
and stdlib/`google.golang.org/protobuf`. No `cache`, no `objstore`, no
`vector/analysis`.

### What moved (wholesale, unchanged)

`alias.go`, `consistency.go`, `keys.go`, `partition_catalog.go`,
`scroll_cursor.go`, `text.go`, `vector.go` (minus the two handler-only sync.Pool
vars, see below), `vector_routing.go`, `wasm_binding.go`, `wasm_codec.go`. These
never imported `cache`; they're pure wire codec / routing-key-extraction code.

### What was split (codec vs. handler halves)

`builtin.go`, `batch.go`, `mv_batch.go`, `multivector.go`, `named.go`, `query.go`,
`topology.go` each mixed pure Encode/Decode/routing code with server-side
`handleXxx(tx *TxContext, ...)` functions (TxContext wraps `*cache.Cache`). For
each: the pure half moved into `ops/wire/<name>.go`; the handler half stayed in
`ops/<name>.go`, now calling `wire.EncodeFoo`/`wire.DecodeFoo`. New files:
`ops/wire/builtin.go` (KV codec + admin op-name consts), `ops/wire/builtin_routing.go`
(the `BuiltinOps` routing table), `ops/wire/registry.go` (routing-only Registry,
see below).

`wasm_register.go` was NOT split — it's pure server-registration glue
(`RegisterWASMRegisterOp`, `ValidateWASMOpName`) with no client-facing codec, so
it stays entirely in `ops`.

## RegisterRoutableBuiltins / handler-binding split

**Obstacle hit and resolved differently than the plan assumed:** `ops.Registry`'s
`entry.fn` is typed `Handler = func(tx *TxContext, args []byte) ([]byte, error)`,
and `TxContext` wraps `*cache.Cache`. Making the SAME `Registry` type "allow a
nil handler for routing-only use" would require `Handler`/`TxContext` to live in
the leaf too — impossible without dragging `cache` into `ops/wire`. So instead of
one Registry type with an optional handler, there are now **two Registry types**:

- `ops.Registry` (unchanged, in `ops`) — full routing + `Handler`, used by
  `shard`, `cluster`, `wasm`, `authz`, `direct.go`, `embedded.go`, etc. exactly as
  before.
- `wire.Registry` (new, in `ops/wire`) — routing-only: `(kind, KeyExtractor,
  RouteLayout, crossShard)` per op, **no handler field at all** (not "nil
  allowed" — structurally absent). This is what `client.Config.Ops` is typed as
  now, and all it ever needed (the client only ever reads an op's KeyExtractor to
  pick a shard; it never calls a handler).

Both registries are built from **one canonical table**, `wire.BuiltinOps`
(`ops/wire/builtin_routing.go`) — a `[]wire.BuiltinOp{Name, Kind, KE, Layout,
CrossShard}` ported verbatim from the old `ops.RegisterBuiltins` op list:
- `wire.RegisterRoutableBuiltins(reg *wire.Registry)` walks `BuiltinOps` and
  registers routing-only entries — this is what `client.NewRouted` and the
  golden-oracle-adjacent test registries call.
- `ops.RegisterBuiltins(r *ops.Registry)` walks the SAME `BuiltinOps` table,
  looks each name up in a local `builtinHandlers map[string]Handler`, and installs
  `(kind, handler, ke, layout, crossShard)` via the existing unexported
  `registerEntry`. A count check (`len(builtinHandlers) != len(wire.BuiltinOps)`)
  fails loud if the two ever name a different op set.

So the two registries can never disagree about an op's name/kind/routing key —
the parity guarantee the original design wanted — without literally sharing a
struct field.

## Aliases vs. retargeting

Aliases, per the instructions ("prefer aliases to minimize blast radius"), but at
much larger scope than the client's own ~40-symbol usage: dozens of other
packages (`cluster`, `httpapi`, `wasm`, `authz`, `shard`, the root `rostam`
package, and ops's own `_test.go` files) reference `ops.EncodeFoo`/`ops.Topology`/
`ops.WASMRegistration`/etc. directly. `ops/wire_aliases.go` re-exports essentially
every moved symbol (types via `type X = wire.X`, funcs/vars via `var X =
wire.X`, consts via `const X = wire.X`) so none of those call sites needed to
change. `Topology`/`TopologyMember`/`Encode|DecodeTopology` are aliased next to
`RegisterTopology` in `ops/topology.go` instead (same idea, kept together because
`RegisterTopology` needed the wire import anyway); `OpKind`/`KeyExtractor`/
`KeyExtractorInto`/`RouteLayout` (+ enum consts) are aliased in `ops/registry.go`
since `ops.Registry`'s `entry` struct uses them directly; `NoShardIndex` is
aliased in `runtime.go`.

A handful of unexported wire-package helpers that ops's own `_test.go` files
referenced bare (e.g. `stdKeyExtractor`, `decodeKeyArgs`, `collectionNameOffset`,
`querySpecFromProto`, per-file trailer/flag constants) were exported (capitalized)
in `ops/wire` and aliased the same way, rather than rewriting those tests to
import `wire` directly — smaller diff, same effect.

**Client and golden oracle import the leaf directly**, not through the aliases:
`client/*.go` imports `ops/wire` (no `ops` import anywhere in the package
anymore); `clients/python/tests/_oracle/main.go` imports `ops/wire` too.

### `ops.Registry.ExportRouting` — the routing-registry adapter

Some callers build one full `*ops.Registry` (with real handlers, via
`ops.RegisterBuiltins`) and used to hand the SAME instance to both a server-side
dispatcher AND a `client.Config.Ops`/`rostam.ClientConfig.Ops` for client-side
routing (`examples/semantic-search/main.go`, the root `rostam.NewClient`, a couple
of `cluster`/`inttest` tests). Since `client.Config.Ops` is now `*wire.Registry`,
these needed a bridge: `(*ops.Registry).ExportRouting(dst *wire.Registry) error`
(new, in `ops/registry.go`) walks the registry's entries and installs each one's
`(kind, ke, layout, crossShard)` into `dst` — no handler crosses over. Root
`rostam.NewClient` calls it internally so `rostam.ClientConfig.Ops` keeps its
existing `*ops.Registry` field type unchanged (zero edits needed to the ~20
call sites and test files across the repo that build one) — only `client.go`'s
`NewClient` body changed. Two test files (`cluster/multinode_test.go`,
`inttest/tls_rbac_integration_test.go`) that construct a bare `client.Config`
directly (not through the root package) were updated to build a small
`wire.Registry` alongside their existing `ops.Registry` (one via
`ExportRouting`, one via a fresh `wire.RegisterRoutableBuiltins`).

## Client dependency counts

| | before | after |
|---|---|---|
| `cache` present in `./client` deps | yes (1) | **no (0)** |
| `ops` present in `./client` deps | yes (1) | **no (0)** |
| `objstore` present | yes (1) | yes (1) — see below |
| `vector/analysis` present | yes (1) | yes (1) — see below |
| total `go list -deps ./client/` count | 236 | 232 |

**The `cache|objstore|vector/analysis` count did NOT reach 0 — it went from 3 to
2, not 3 to 0.** This is a real, verified fact, not an oversight in the alias
work above:

`client/*.go` (`vector_search.go`, `vector_write.go`, `vector_read.go`,
`vector_lifecycle.go`, `vector_recommend.go`) directly imports
`github.com/rostamlabs/rostam/vector` for plain data types
(`vector.QuerySpec`, `vector.Filter`, `vector.CASCond`, `vector.Metadata`,
`vector.Result`, `vector.SparseVector`, `vector.HybridOpts`, ...) — this is
independent of `ops` and untouched by this refactor. The `vector` package
itself (the engine: HNSW, BM25, persistence) imports `objstore` directly
(`vector/coldtier.go`, for cold-tier storage) and `vector/analysis` directly
(`vector/hnsw.go`, `vector/bm25_index.go`, `vector/persist.go`, ... for
tokenization). Because Go has no way to import "just the types" from a
package, `client`'s pre-existing, unavoidable import of `vector` drags
`objstore` and `vector/analysis` in regardless of what happens to `ops`.

Fixing this would require an analogous leaf split of the `vector` package
itself (e.g. a `vector/types` package holding `QuerySpec`/`Filter`/`Metadata`/
etc., with `vector` re-exporting them as aliases the way `ops` does here) —
out of scope for "extract a client-safe wire-codec leaf out of the `ops`
package." The `ops`→`ops/wire` split is complete and correct on its own terms:
it eliminates `cache` and `ops` from the client's dependency graph, which was
the part actually caused by `ops`.

`go list -deps ./client/ | grep -c "rostam/ops$"` → **0**.

## Build / test / golden results

- `go build ./...` — clean (whole module).
- `go vet ./...` — clean (whole module).
- `gofmt -l` — clean on every file touched.
- `go test ./ops/... ./client/... ./grpcapi/... -count=1 -timeout 15m` — all pass
  (`ops/wire` has no test files of its own; existing `ops` tests cover the codec
  through the aliases).
- Golden oracle: `go run ./clients/python/tests/_oracle` output is
  **byte-identical** to the checked-in `clients/python/tests/_oracle/golden.txt`
  (diffed directly); `python3 -m pytest tests/test_vecwire_golden.py -q` → 6
  passed, 55 subtests passed.
- Cross-stack sanity beyond the required scope: `cluster`'s smart-routing tests
  (`TestSmartClientRoutesToShardLeader`, `TestSmartClientConvergesAfterLeaderKill`,
  `TestSmartClientCompatWithOpsNil`) and `inttest`'s TLS/mTLS suite (including
  `TestPlaintextUnchangedWhenTLSNil`, which exercises the new
  `ops.Registry.ExportRouting` bridge) all pass.
- No import cycle: `go list -deps ./ops/wire/` contains no `rostamlabs/rostam/ops`
  entry; `ops` imports `ops/wire`, never the reverse.

## Concerns / follow-ups

- The `objstore`/`vector/analysis` residue in client deps (see above) is the one
  open item against the letter of the HARD metric; it's a pre-existing, separate
  coupling through the `vector` package, not something this refactor introduced
  or can close without a second, larger leaf split.
