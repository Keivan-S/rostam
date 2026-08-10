# Collections, tenants & aliases

## Three collection families

Rostam has three kinds of vector collections, each with its own API family and
HTTP prefix:

| Family | HTTP prefix | Model | Use for |
|---|---|---|---|
| **Dense** (single-vector) | `/v1/collections/{name}` | one vector per point + payload + optional sparse lane + optional full-text | classic semantic search, RAG, hybrid retrieval |
| **Multi-vector** (late interaction) | `/v1/multivector/{name}` | many token vectors per document, MaxSim scoring | ColBERT-style retrieval |
| **Named-vector** (multi-space) | `/v1/named/{name}` | several independent dense/sparse spaces per point, shared payload | multi-modal points (e.g. image + text vectors), Qdrant-style APIs |

Semantics worth knowing:

- **Dense `Insert` is create-only** — inserting a live id fails with
  `ErrDuplicateID`. Use `Upsert` to insert-or-replace, or `InsertIfAbsent` for an
  atomic create-if-missing.
- **Multi-vector `Add` is insert-or-replace** (it replaces a live id), with
  `AddIfAbsent` for the atomic create-only variant.
- **Named-vector `Insert` is an upsert** of per-space vectors plus the shared
  payload.
- Dense collections support WAL, snapshots, and mmap-backed quantized storage
  ([Persistence](../vector/persistence.md)). Multi-vector collections support
  the same two single-node durability modes — WAL (heap + checkpoint) or
  mmap-backed `Persistent` (which requires quantization); the two are mutually
  exclusive. Named-vector collections support the WAL mode. In a cluster, Raft
  is the durability authority and the per-collection WAL is forced off.

## Points, payloads, and content

Every point carries:

- an **id** (`uint64`),
- its vector(s),
- an optional **payload** (metadata): `map[string]Value` where `Value` is a
  tagged union — string, int, float, bool, string/int/float arrays, or geo
  point. Payloads are filterable ([Filtering](../vector/filtering.md)) and
  independently mutable (merge, overwrite, delete-keys, clear — without
  re-sending the vector).
- optionally **content** — the raw document text, stored under the reserved
  `$content` payload key by `Upsert`/`search_docs`-style APIs so search can
  return the text alongside the hit. `$content` is excluded from the payload
  index.
- optional **TTL** — per-point expiry, plus per-payload-key TTLs
  (`key_ttl_ms`) for fields that should expire independently.

Point writes support **optimistic concurrency**: every point has a version, and
CAS variants (`expected_version` over HTTP, `CASCond` in Go) fail with a version
conflict instead of overwriting concurrent updates.

## Multi-tenancy

Collection names may be namespaced as `<tenant>/<collection>`; a bare name lands
in the default tenant. API keys can be bound to a tenant, and the server's
`-tenant-isolation` flag makes the key's tenant an authoritative boundary —
see [Security](../server/security.md).

## Quotas, rate limits, TTL

Per-collection guards, all enforced by the engine:

| Config | Effect on violation |
|---|---|
| `MaxVectors` | inserts fail with `ErrCollectionFull` |
| `MaxBytes` (estimated memory) | inserts fail with `ErrCollectionFull` |
| `MaxInsertsPerSecond` (token bucket) | inserts fail with `ErrCollectionRateLimited` |

Expired points are removed lazily on read plus by a background sweeper
(`SweepInterval`, default 60 s). Rejections and expirations are visible in
collection stats ([Monitoring](../server/monitoring.md)).

## Aliases

Aliases decouple the name your application queries from the physical collection —
the standard building block for blue/green reindexing:

```
POST   /v1/aliases            {"alias":"prod-docs","collection":"docs-v2"}   # create/repoint
DELETE /v1/aliases/{alias}                                                   # remove
GET    /v1/aliases?collection=docs-v2                                        # list
POST   /v1/aliases/batch      {"actions":[{"delete":{"alias":"prod-docs"}},
                                          {"create":{"alias":"prod-docs","collection":"docs-v3"}}]}
```

`aliases/batch` is atomic: the batch is validated up front and any invalid create
rejects the entire batch, so a repoint is all-or-nothing. Aliases resolve on
every data-plane op, so `search` against `prod-docs` transparently hits the
current target.

Names (collections and aliases) are capped at **247 bytes** — the wire codecs
encode name length in a single byte, and the cap leaves headroom for the tenant
prefix.

## Partitioning & resharding

A collection can be split into `Partitions` cross-shard partitions at creation
time, spreading it across the cluster. Two ways to change partitioning later:

- **Resplit** (`POST .../resplit`) — offline: caller must quiesce writes.
- **Reshard** (`POST .../reshard`) — online: dual-write + background copy with a
  resumable, atomic cutover; abortable before cutover
  (`POST .../reshard/abort`). See [Clustering](../server/clustering.md).
