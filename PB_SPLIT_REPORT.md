# gRPC service split out of grpcapi/pb — report

## Approach

**Regeneration (preferred path), not the manual fallback.** The toolchain versions
already installed matched the committed generated-file headers exactly
(`protoc v3.21.12`, `protoc-gen-go v1.36.11`, `protoc-gen-go-grpc v1.6.0`), so a
full proto split + regen was safe and reproducible.

1. Split `grpcapi/pb/rostam.proto`: removed the `service VectorService { ... }`
   block (176 lines), keeping all 121 messages and `option go_package =
   ".../grpcapi/pb;pb"` untouched.
2. Created `grpcapi/pb/service.proto`: `package rostam.v1`, `import
   "grpcapi/pb/rostam.proto"`, the same `VectorService` block, with
   `option go_package = "github.com/rostamlabs/rostam/grpcapi/grpcsvc;grpcsvc"`.
3. Regenerated with:
   ```
   protoc -I . \
     --go_out=. --go_opt=paths=import \
     --go-grpc_out=. --go-grpc_opt=paths=import \
     grpcapi/pb/rostam.proto grpcapi/pb/service.proto
   ```
4. Verified `rostam.pb.go` diff against the pre-split committed file was
   **exactly** the service's raw descriptor bytes/reflection metadata
   disappearing (`NumServices: 1` → `0`, the service method table dropping out
   of the raw descriptor string) — zero changes to any message type, field, or
   the JSON/wire contract. Added back the `// SPDX-License-Identifier:
   Apache-2.0` header line (protoc doesn't preserve it; it's added
   post-generation in this repo, same as before).
5. Moved the new service output into `grpcapi/grpcsvc/{service.pb.go,
   service_grpc.pb.go}`, deleted `grpcapi/pb/rostam_grpc.pb.go`.

## Files moved / changed

- `grpcapi/pb/rostam.proto` — service block removed.
- `grpcapi/pb/service.proto` — **new**: the service definition, imports
  `rostam.proto`, separate `go_package`.
- `grpcapi/pb/rostam.pb.go` — regenerated (messages only now; no grpc import).
- `grpcapi/pb/rostam_grpc.pb.go` — **deleted**.
- `grpcapi/grpcsvc/service.pb.go` — **new** (generated; file-descriptor/registry
  glue for service.proto, no messages of its own).
- `grpcapi/grpcsvc/service_grpc.pb.go` — **new** (generated; the actual
  `VectorServiceClient`/`VectorServiceServer`/`UnimplementedVectorServiceServer`/
  `RegisterVectorServiceServer` surface, imports `grpc` + `pb` for message
  types).

### Non-generated reference updates (pb.\* service symbols → grpcsvc.\*; message types stayed pb.\*)

- `grpcapi/server.go` — added `grpcapi/grpcsvc` import;
  `pb.UnimplementedVectorServiceServer` → `grpcsvc.UnimplementedVectorServiceServer`;
  `pb.RegisterVectorServiceServer` → `grpcsvc.RegisterVectorServiceServer` in
  `Server.Register`.
- `grpc_test.go` (root package) — `dialGRPC` now returns
  `grpcsvc.VectorServiceClient` via `grpcsvc.NewVectorServiceClient`.
- `inttest/helpers_test.go` — same `dialGRPC` update; dropped the now-unused
  `grpcapi/pb` import.
- `inttest/tls_rbac_integration_test.go` — `dialGRPCTLS` updated the same way
  (kept the `pb` import — still used for message types elsewhere in the file).
- `grpcapi/wire_bench_test.go` — `newWireBench`/`seedCollection` signatures and
  `RegisterVectorServiceServer`/`NewVectorServiceClient` calls updated.
- `grpcapi/wire_hybrid_bench_test.go` — `seedHybridCollection` signature updated.
- `grpcapi/wire_mv_bench_test.go` — `seedMVCollection`/`seedNamedCollection`
  signatures updated.

No production entry point beyond `grpcapi/server.go` referenced the gRPC
service symbols (checked `cmd/`, `server.go` at repo root calls
`grpcapi.NewServer(...).Register(gs)`, whose signature is unchanged).

## Metric

`go list -deps ./client/ | grep -c "google.golang.org/grpc"`:

- **BEFORE: 63**
- **AFTER: 0**

## Verification

- `go build ./...` — clean, whole module.
- `go vet ./...` — clean, whole module.
- `gofmt -l grpcapi/ client/` — no output (clean).
- `go test ./grpcapi/... ./ops/... ./client/... -count=1 -timeout 15m` — all
  pass:
  ```
  ok  	github.com/rostamlabs/rostam/grpcapi	0.182s
  ?   	github.com/rostamlabs/rostam/grpcapi/grpcsvc	[no test files]
  ?   	github.com/rostamlabs/rostam/grpcapi/pb	[no test files]
  ok  	github.com/rostamlabs/rostam/ops	0.546s
  ok  	github.com/rostamlabs/rostam/client	3.692s
  ```
- `go vet . ./inttest/...` and a `-run NoSuchTestXYZ` compile-only pass — both
  clean (root package's `grpc_test.go` and `inttest/`'s gRPC TLS/RBAC tests
  build against the new `grpcsvc` package).
- Python golden oracle: `cd clients/python && python3 -m pytest
  tests/test_vecwire_golden.py -q` — `6 passed, 55 subtests passed` (its
  `_oracle/main.go` imports `ops`, which imports `pb` messages only — confirmed
  unaffected, still builds and passes).

No wire behavior, message definitions, or JSON contract changed — purely a
package/file reorganization of the generated code plus reference updates.
