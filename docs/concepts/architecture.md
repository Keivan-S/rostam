# Architecture

Rostam is one Go module (`github.com/rostamlabs/rostam`) containing two engines
that share a facade but have no hard dependency on each other.

```
        vector/   ← standalone vector engine (usable entirely on its own)
        HNSW · IVF · Vamana · GPU · quantization (SQ8/BQ1/PQ/PRQ, mmap) ·
        hybrid sparse+dense · BM25 · payload index + planner ·
        MMR/recommend/discover · tenants/auth/quotas · snapshot/cold tier

        rostam.Store  (Direct │ Embedded │ Client)      ← unified facade
              │
   ┌──────────┼───────────────┬──────────────────┐
 cache/     shard/ + raft/    server/ + client/   wasm/
 slab pool  per-shard         TCP protocol +      wasmtime UDF
 TTL, mmap  Raft + ops        smart routing       runtime
 warm-start registry
```

## Package tour

**Vector side**

- **`vector/`** — the vector search engine. Its only in-repo dependencies are the
  stdlib-only `objstore/` package (cold-tier offload) and its own
  `vector/analysis` subpackage (full-text tokenization) — no cgo, no server code.
  You can vendor those three packages together as a pure vector library.
- **`objstore/`** — a minimal object-store abstraction (`Put`/`Get`/`List`/`Delete`)
  with an in-memory implementation and a stdlib-only S3-compatible client
  (SigV4; works with AWS, MinIO, R2).

**Key-value side**

- **`cache/`** — the storage core: sharded slab pool, TTL ring buffer, mmap
  persistence with warm restart.
- **`ops/`** — the stored-procedure registry: named ops with read-only /
  read-write kinds, shard routing via key extractors, and wire codecs for every
  built-in operation.
- **`shard/`, `raft/`** — per-shard Raft groups; each shard is an independent
  consensus group, so throughput scales with shard count.
- **`server/`, `client/`** — the compact TCP wire protocol and the
  topology-aware smart client (leader routing, connection pooling).
- **`wasm/`** — `wasmtime`-backed UDF runtime: determinism-gated imports,
  per-invocation fuel caps. Requires cgo; a pure-Go stub keeps non-cgo builds
  working (WASM ops then return `wasm.ErrNoCGO`).

**Front doors**

- **`httpapi/`** — REST/JSON on `/v1/...`.
- **`grpcapi/`** — the `VectorService` gRPC service.
- **`cmd/rostam-server`** — the server binary wiring all three transports, auth,
  TLS, clustering, and backups together.
- **`clients/python`** — a dependency-free Python REST client with optional
  LangChain / LlamaIndex / Haystack adapters.

## One dispatcher, three transports

HTTP, gRPC, and TCP all decode into the same named-op dispatch layer: every
operation — `put`, `vector_search`, `create_collection` — is a registered op in
the `ops.Registry`. That gives all transports identical semantics, a single
authorization chokepoint ([Security](../server/security.md)), and lets you invoke
custom ops by name from any client.

Read-only ops execute locally against the shard they route to; read-write ops
serialize through the shard's Raft log (on `Embedded`) so every replica applies
the same deterministic mutation. On `Direct` there is no Raft — writes apply
under the shard lock directly.

## How the two engines relate

The KV facade embeds a `vector.CollectionStore`, so the `Store` interface carries
both `Put`/`Get`/`Call` *and* the full vector API (`CreateCollection`,
`VectorSearch`, `VectorHybridSearch`, …). Vector operations are ops like any
other, which is what makes them replicable through Raft and callable over every
transport.

If you only need one engine, take only that engine: `vector.NewCollection` never
touches Raft, the cache, or the network; `cache.New` never touches the vector
index.

## Design principles

- **Zero-copy, allocation-free hot paths.** `Get` aliases the backing arena;
  search reuses scratch buffers; the distance kernels are hand-vectorized AVX2
  with scalar fallbacks.
- **Exactness over silent degradation.** Selective metadata filters take an
  exact filter-first path rather than letting graph recall collapse
  ([Filtering](../vector/filtering.md)); quantized search rescores candidates
  with exact float32 distances.
- **Determinism where it matters.** Read-write ops must be deterministic —
  that's what makes per-shard Raft replication and the WASM sandbox safe.
- **Dependency-light.** A handful of direct dependencies, each scoped to one
  subsystem — `xxhash` for key hashing, `raft` for replication, `grpc` for the
  gRPC front door, `wasmtime` for the WASM sandbox. The vector engine itself
  depends on none of them.
