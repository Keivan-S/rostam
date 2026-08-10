# Rostam

**A high-performance vector database and sub-microsecond key-value store in a
single Go engine — run it as a standalone server, replicate it across a Raft
cluster, or embed it directly in your binary with no server at all.**

At matched recall it serves **~2× the queries of Milvus and pgvector** and
**~4× Qdrant**, with the fastest load in the set (1M × 768d in 282 s) and the
highest recall measured — under
[VectorDBBench](https://github.com/zilliztech/VectorDBBench), a neutral
third-party harness.
[See the full comparison](https://github.com/rostamlabs/rostam-bench/tree/main/vectordbbench#results).

=== "Standalone server"

    REST, gRPC and a binary TCP protocol. Talk to it from Python or any
    language. → [Running the server](server/running.md)

=== "Replicated cluster"

    Per-shard Raft, online resharding, backups to S3, RBAC/JWT/mTLS.
    → [Clustering](server/clustering.md)

=== "Embedded library"

    Import it into a Go binary. No server, no cgo required.
    → [Deployment modes](concepts/deployment-modes.md)

Rostam is one Go module that ships two engines:

- **Vector search engine** ([`vector/`](vector/collections-and-indexes.md)) — HNSW,
  IVF, and Vamana indexes with quantization (int8, binary, PQ), hybrid dense+sparse
  search, BM25 full text, and metadata filtering with an exact filter-first query
  planner. Depends on nothing else in the repo — vendor it as a pure vector library.
- **Key-value store** ([`kv/`](kv/overview.md)) — a sharded in-memory store with
  zero-copy reads, TTL, mmap persistence, optional per-shard Raft replication, and
  server-side stored procedures (native Go or sandboxed WASM).

Both engines are built for the latency-sensitive end of the spectrum: reads are
zero-copy and lock-light, hot paths are allocation-free, and the distance kernels
are hand-vectorized (AVX2 with scalar fallback).

!!! note "Status"
    Rostam is in beta: actively developed and tested (race-clean, benchmarked).
    APIs may still change ahead of a 1.0 release.

## Choose your entry point

| I want to… | Start here |
|---|---|
| Add vector search to a Go program (no server) | [Quickstart → Embedded vector search](quickstart.md#embedded-vector-search-go-library) |
| Use a fast in-process KV cache / store in Go | [Quickstart → Embedded key-value store](quickstart.md#embedded-key-value-store-go-library) |
| Run Rostam as a server and talk REST/gRPC | [Quickstart → Run the server](quickstart.md#run-the-server) |
| Use Rostam from Python | [Python client](api/python.md) |
| Run a replicated multi-node cluster | [Clustering](server/clustering.md) |
| Understand how it all fits together | [Architecture](concepts/architecture.md) |

## Feature highlights

**Vector side**

- HNSW index with AVX2 dot/L2 kernels and an allocation-free search path; IVF,
  Vamana (DiskANN-style), and an optional CUDA exact-KNN index
- Quantization: scalar int8 (4× smaller), binary (32× smaller), PQ/PRQ — all with
  an exact float32 rescore stage; codes can live in RAM or a memory-mapped file
- Hybrid search: dense + sparse lanes fused via RRF, weighted blend, or DBSF;
  BM25 full-text search with pluggable analyzers
- Metadata filtering with 22 operators (equality, ranges, datetime, geo, regex)
  and a payload index + query planner: selective filters take an exact
  filter-first path instead of degrading graph recall
- Query API: MMR diversified retrieval, recommendation (± examples), discovery
  (context pairs), group-by, scroll with order-by, multi-stage fusion/rerank
- Three collection families: dense single-vector, multi-vector late interaction
  (ColBERT-style MaxSim), and named-vector multi-space (Qdrant-style)
- Multi-tenancy, API-key/RBAC/JWT/mTLS auth, per-collection quotas and rate
  limits, TTL, snapshot/restore, S3 cold tier, Prometheus metrics

**Key-value side**

- Sharded in-memory store with a slab-pool allocator: ~29 ns `Get`,
  allocation-free `GetInto`
- Atomic server-side ops: built-in `incr` and `expire`, plus your own native Go
  ops or sandboxed, fuel-capped WASM procedures shipped over the wire
- mmap-backed persistence with warm restart; per-shard Raft replication with
  online resharding; smart TCP client with leader routing

## Documentation map

- **[Quickstart](quickstart.md)** — install, first search, first KV call, run the server
- **Concepts** — [Architecture](concepts/architecture.md) ·
  [Deployment modes](concepts/deployment-modes.md) ·
  [Collections, tenants & aliases](concepts/collections.md)
- **Vector engine** — [Collections & indexes](vector/collections-and-indexes.md) ·
  [Search](vector/search.md) · [Filtering](vector/filtering.md) ·
  [Hybrid & full-text](vector/hybrid-search.md) ·
  [Quantization](vector/quantization.md) · [Persistence](vector/persistence.md)
- **Key-value store** — [Overview](kv/overview.md) · [Custom ops](kv/custom-ops.md) ·
  [WASM procedures](kv/wasm.md) · [Cache tuning](kv/cache.md)
- **Server & operations** — [Running the server](server/running.md) ·
  [Clustering](server/clustering.md) · [Security](server/security.md) ·
  [Backups & cold tier](server/backups.md) · [Monitoring](server/monitoring.md)
- **API reference** — [HTTP](api/http.md) · [gRPC](api/grpc.md) ·
  [Go client](api/go-client.md) · [Python client](api/python.md)
- **[Performance](performance.md)** · **[Development](development.md)**

## License

Rostam is open source under the
[Apache License 2.0](https://github.com/rostamlabs/rostam/blob/main/LICENSE) —
use, embed, modify, run in production, or offer as a service, with an explicit
patent grant. Redistribution carries the usual Apache-2.0 conditions: a copy of
the licence, the `NOTICE` file, the existing notices, and a mark on files you
changed.

"Rostam" and the Rostam logo are trademarks of RostamLabs; the licence grants no
trademark rights.
