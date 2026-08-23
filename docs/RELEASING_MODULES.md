# Releasing the `sdk` and `client` modules

The repo is a Go multi-module workspace: the root module
`github.com/rostamlabs/rostam` (engine + server), the shared
`github.com/rostamlabs/rostam/sdk` module (`sdk/vtypes`, `sdk/wire`, `sdk/pb` —
engine-free), and `github.com/rostamlabs/rostam/client`. Local development uses
the committed `go.work` and `replace` directives, so in-repo builds always use
the working-tree sources. External consumers ignore `replace`; they resolve
tagged versions. Tag in this order so a `go get` never sees a broken graph:

1. **Merge** the change to `main`.
2. **Tag `sdk`:** `git tag sdk/vX.Y.Z && git push origin sdk/vX.Y.Z`.
3. **Point `client` (and root) at the tag:** in `client/go.mod` and root
   `go.mod`, set `require github.com/rostamlabs/rostam/sdk vX.Y.Z`. Keep the
   local `replace ... => ./sdk` (`../sdk` for client) for in-repo builds —
   external consumers ignore it. Commit + merge.
4. **Tag `client`:** `git tag client/vX.Y.Z && git push origin client/vX.Y.Z`.

Only after steps 2 **and** 4 does
`go get github.com/rostamlabs/rostam/client@vX.Y.Z` resolve for an external
user — pulling only `sdk` + light deps (protobuf, xxhash, puddle, x/sync), never
the engine module. **Never tag `client` before `sdk`:** `client` requires `sdk`,
so a `client` tag whose `sdk` version is untagged is unresolvable.

To verify a client tag is engine-free before announcing it:

```
cd client && GOWORK=off go list -m all | grep -qx 'github.com/rostamlabs/rostam' \
  && echo "FAIL: engine in graph" || echo "PASS: engine-free"
```

(CI's `test-client-module` job runs this on every PR.)

## Regenerating protobuf

`sdk/pb` is generated from `sdk/pb/*.proto` (`go_package = .../sdk/pb`). If you
hand-edit an import path across the tree, do NOT let a blanket rewrite touch the
serialized descriptor in `*.pb.go` — the `go_package` string is length-prefixed,
so shortening it corrupts the descriptor (a `slice bounds` panic at init).
Regenerate with the proto toolchain instead.
