# Contributing to Rostam

Thanks for your interest in Rostam.

## License of contributions

Rostam is released under the **Apache License 2.0** (see `LICENSE`). Inbound
matches outbound: by submitting a contribution you license it under Apache-2.0,
which is what Section 5 of the licence already says by default. There is no CLA
and no copyright assignment — you keep the copyright on what you write.

The control plane / managed-cloud components are **not** part of this repository
and are licensed separately.

## Building

Rostam is a single, dependency-light Go module. Keep it that way — new direct
dependencies in `go.mod` are reviewed carefully.

The default full-module build requires **cgo** because the WASM
stored-procedure backend pulls in `wasmtime-go`:

```sh
# default build (cgo required by the wasmtime-go WASM backend)
go build ./...

# the core storage and vector packages are pure Go and build without cgo
CGO_ENABLED=0 go build ./vector/... ./objstore ./cache ./ops

# the optional GPU exact-KNN index requires cgo + CUDA toolchain
CGO_ENABLED=1 go build -tags cuda ./...
```

## Tests

Run the suite through the `Makefile` targets so the flags (`-count=1`, timeouts,
parallelism) live in one place and can't drift:

```sh
make test         # default run
make test-serial  # one test binary at a time (-p 1) when chasing the CPU-oversubscription flake
make race         # race detector; run before submitting
```

The full suite is sensitive to CPU oversubscription: the in-process RF=3 cluster
tests each spin many raft groups, so running several test binaries concurrently on
a busy machine can starve them and cause rare load-only flakes. `make test-serial`
runs one binary at a time (`-p 1`) to avoid that; a lighter `-p 2` reduces — but
does not eliminate — the concurrency.

## Style

- Run `gofmt -w` (or `goimports`) on changed files; CI enforces formatting.
- Match the conventions of the surrounding code.
- Every source file carries an SPDX header: `// SPDX-License-Identifier: Apache-2.0`.

## Security

Do not file public issues for security problems — see `SECURITY.md`.
