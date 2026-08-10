# Development

## Requirements

- **Go 1.26+** (from `go.mod`)
- Linux/amd64 gets mmap persistence and the AVX2 kernels; other platforms use
  portable fallbacks
- **cgo** for the full module (the WASM backend links `wasmtime-go`); the
  storage and vector packages are pure Go

## Building

```sh
# default build (cgo required by the wasmtime-go WASM backend)
go build ./...

# the core storage and vector packages are pure Go and build without cgo
CGO_ENABLED=0 go build ./vector/... ./objstore ./cache ./ops

# the optional GPU exact-KNN index requires cgo + a CUDA toolchain
CGO_ENABLED=1 go build -tags cuda ./...
```

Without cgo, WASM stored procedures are stubbed: registration/invocation
returns `wasm.ErrNoCGO`, everything else works.

## Testing

Run the suite through the Makefile targets so the flags (`-count=1`, timeouts,
parallelism) live in one place:

```sh
make test         # go test -count=1 -timeout 20m ./...
make test-serial  # -p 1: one test binary at a time
make race         # race detector (run before submitting)
make bench        # micro-benchmarks
make test-python  # Python client suite (pytest, runs against an in-process fake)
make lint         # golangci-lint
make all          # lint + test + race + bench
```

The full suite is sensitive to **CPU oversubscription**: the in-process RF=3
cluster tests each spin many Raft groups, so several test binaries running
concurrently on a busy machine can starve them into rare load-only flakes.
`make test-serial` (`-p 1`) avoids that; CI uses `-p 2` as a middle ground.

## CI

GitHub Actions runs these lanes on every push/PR:

| Lane | What it checks |
|---|---|
| `test-unit` | cgo build, non-cgo build of the pure-Go packages, `go vet`, unit/light packages |
| `test-root` / `test-inttest` / `test-cluster` / `test-shard` | the heavy integration suites, each isolated in its own job so they never oversubscribe one runner |
| `cgo-disabled-build` | the whole module keeps building with `CGO_ENABLED=0` (the wasmtime backend swaps to its `!cgo` stub) |
| `cross-compile` | build-only matrix across linux/386, linux/arm, linux/arm64, darwin, windows, freebsd — catches 32-bit and cross-platform breakage the linux/amd64 lanes can't |
| `test-386` | the wire-decoding packages **run** as a linux/386 binary, so `int(uint32)` length conversions execute with a 32-bit `int` — where an over-MaxInt32 length widens negative and slips past `len(args) < off+n` checks. The cross-compile lane only builds; this one executes |
| `race` | race detector over the concurrency-critical packages, with `-short`: the heavy differential suites in `vector/` size their corpora off it by design (see `filter_bitset_test.go`), and without it the package alone exceeds the lane's timeout |
| `python-client` | the Python client's pytest suite |

Separate workflows lint (`golangci-lint`) and build the docs site
(`mkdocs build --strict`).

## Style

- `gofmt`/`goimports` on changed files; CI enforces formatting.
- Match the conventions of the surrounding code.
- Every source file carries an SPDX header: `// SPDX-License-Identifier: Apache-2.0`.
- Keep the module dependency-light — new direct dependencies in `go.mod` are
  reviewed carefully.

## Docs site

The `docs/` tree is plain Markdown organized for a static docs site, with a
ready `mkdocs.yml` at the repo root:

```sh
pip install mkdocs-material
mkdocs serve     # live-preview at http://localhost:8000
mkdocs build     # static site in site/
```

## Licensing

Rostam is **Apache-2.0** — genuinely open source, OSI-approved. Use it, embed it,
modify it, run it in production, offer it as a service. It carries an explicit
patent grant, and asks for the conditions below whenever it is redistributed.

Three things the licence needs from a release:

- **Ship `LICENSE` and `NOTICE` together.** Apache-2.0 §4 requires redistributors
  to carry both; dropping `NOTICE` from a tarball or an image is the one easy way
  to make a distribution non-compliant.
- **Every source file keeps its SPDX header** —
  `// SPDX-License-Identifier: Apache-2.0`. Licence scanners read these, not
  prose, and a file without one reads as unlicensed.
- **The trademark is separate from the licence.** Apache-2.0 grants no rights in
  "Rostam" or the logo (§6). The licence lets anyone ship the code; the trademark
  is what governs what they may call it.

There is no Change Date, no version parameter, and no per-release licence step —
Rostam was previously under BUSL 1.1, which had all three. If you find
instructions about bumping a Change Date, they are stale.

## Contributing & security

- Contributions are licensed under Apache-2.0, inbound = outbound, no CLA — see
  [`CONTRIBUTING.md`](https://github.com/rostamlabs/rostam/blob/main/CONTRIBUTING.md).
- Security issues: **do not** open public issues — email
  security@rostamlabs.com
  ([`SECURITY.md`](https://github.com/rostamlabs/rostam/blob/main/SECURITY.md)).
