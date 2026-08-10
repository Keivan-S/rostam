# HTTP API reference

All endpoints live under `/v1`. Requests and responses are JSON.

- **Auth**: `Authorization: Bearer <token>` ([Security](../server/security.md)).
  `/health` is auth-exempt.
- **Names**: collection and alias names are capped at 247 bytes.
- **Metadata values** use the tagged encoding described in
  [Filtering → Payload values](../vector/filtering.md#payload-values);
  filters use lowercase operator names
  ([Filtering → Operators](../vector/filtering.md#operators)).
- **Read options** on search/get endpoints: `read_consistency` (0 any-replica ·
  1 leader · 2 linearizable · 3 bounded-staleness), `max_staleness` (Raft-entry
  bound for level 3), `on_partition_unavailable` (0 partial · 1 fail).
- **Write options** on point writes: `write_consistency_factor` (replica acks),
  `wait`, `expected_version` (CAS).

## Errors

Errors return `{"error": "..."}` with a conventional status code (400 invalid
request, 401/403 auth, 404 not found, 412 feature not configured, 5xx server).
During a Raft leader election, writes can return a retryable 503.

## Health & metrics

| Method & path | Description |
|---|---|
| `GET /v1/health` | liveness — the process is serving (auth-exempt) |
| `GET /v1/ready` | readiness — every hosted shard can serve (auth-exempt) |
| `GET /metrics`, `GET /v1/metrics` | Prometheus text (scope-gated) |

## Collections (dense)

| Method & path | Body | Description |
|---|---|---|
| `POST /v1/collections` | `{"name", "config"}` | create |
| `DELETE /v1/collections/{name}` | — | drop collection + data |

`config` mirrors `vector.Config`:

```json
{
  "dim": 768,
  "metric": "cosine",            // "cosine" (default) | "l2" | "dot"
  "m": 16, "ef_construction": 200, "ef_search": 64, "seed": 0,
  "quant": "sq8",                // "" | "sq8" | "bq1" | "pq" | "sq" | "prq"
  "rescore_factor": 3, "sq_bits": 0, "prq_layers": 0, "quant_pq_m": 0, "pq_nbits": 0,
  "persistent": false,
  "partitions": 1,
  "index_type": "hnsw",          // "" | "hnsw" | "ivf" | "vamana"
  "ivf_nlist": 0, "ivf_nprobe": 0, "ivf_pq": false, "ivf_pq_m": 0, "ivf_rerank": false,
  "vamana_r": 0, "vamana_l": 0, "vamana_alpha": 0,
  "filter_first_relative_bp": 0,
  "full_text": {"analyzer": "english", "k1": 1.2, "b": 0.75},
  "extend_candidates": false, "level0_full_degree": false, "quantized_build": false,
  "opq": false, "soar": false, "anisotropic_eta": 0
}
```

Only `dim` is required; zero values take engine defaults.

## Points (dense)

| Method & path | Description |
|---|---|
| `POST /v1/collections/{n}/points` | insert (or upsert with `"upsert": true`) |
| `POST /v1/collections/{n}/points/batch` | bulk insert/upsert `{"upsert", "points": [...]}`; per-point `{"id","version","status"}` results |
| `POST /v1/collections/{n}/points/bulk` | stage vectors (and optional `metadata`) for a concurrent build |
| `POST /v1/collections/{n}/points/bulk/build` | build the staged index `{"workers": 0}` (0 = all cores) |
| `GET /v1/collections/{n}/points/{id}` | fetch; query params `with_vector`, `with_payload`, read options |
| `POST /v1/collections/{n}/points/batch-get` | `{"ids", "with_vector", "with_payload"}` → `{"points", "missing"}` |
| `DELETE /v1/collections/{n}/points/{id}` | delete; query params `write_consistency_factor`, `wait`, `expected_version` |
| `POST /v1/collections/{n}/points/delete` | delete by filter `{"filter"}` → count |
| `POST /v1/collections/{n}/points/scroll` | `{"filter", "limit", "cursor"}` → page + `next_cursor` |

Point write body:

```json
{
  "id": 1,
  "vector": [0.1, 0.2],
  "content": "raw document text",         // stored under $content
  "metadata": {"tenant": {"kind":"string","str":"acme"}},
  "sparse": {"indices":[3,17], "values":[0.4,0.9]},
  "ttl_ms": 0,
  "key_ttl_ms": {"session": 60000},       // per-payload-key TTL
  "upsert": true,
  "expected_version": 7,                   // CAS (optional)
  "write_consistency_factor": 2, "wait": true
}
```

### Binary bulk ingest

`/points/bulk` and `/points/batch` also accept a dense binary body, selected by
`Content-Type: application/octet-stream`. It carries the same request as the JSON
body — the server decodes it into the identical call — but ships the vectors as
raw `f32` instead of base-10 text, which is what dominates a large initial load.
Any other content type takes the JSON path unchanged.

```
magic  "RVB1"                                   4 bytes
flags  u32     bit0 payloads present, bit1 upsert
count  u32     number of points
dim    u32     vector dimension (uniform across the body)
rows   count × [ id u64 ][ dim × f32 ]
pays   count × [ len u32 ][ len bytes of JSON ]   // only when bit0
```

Everything is **big-endian**, matching the internal op wire, so a row needs no
conversion server-side. A payload is a JSON object in the tagged metadata form
(`{"tenant":{"kind":"string","str":"acme"}}`); `len = 0` means the point has none.

Both routes carry payloads: `/points/batch` indexes each point inline, while
`/points/bulk` stages them for the multi-core build. `upsert` applies to
`/points/batch` only — `/points/bulk` rejects a body that sets it rather than
silently ignoring it, as does the JSON staging body. Responses are the same as
the JSON path (`{"staged":n}` / `{"count":n}`).

**Limits.** One request carries at most **256 MiB** and **262,144 points**, and
`dim` must equal the collection's configured dimension. Split a larger load
across requests — the reference Python client does this for you.

`count` and `dim` are validated against those limits before any row is read, and
every section is read in bounded windows that grow only as bytes arrive: a body
that over-declares its size is refused having consumed only what it actually
sent. A dimension that does not match the collection is rejected by the shard
that owns the collection (400), which also covers the JSON staging body — a
staged vector whose length is not the collection's `dim` is refused at stage
time, not at build time. Either way nothing is ingested.

### Payload mutations

All accept `write_consistency_factor`, `wait`, `expected_version` query params:

| Method & path | Description |
|---|---|
| `POST .../points/{id}/payload` | merge patch (values + optional `key_ttl_ms`) |
| `POST .../points/{id}/payload/overwrite` (also `PUT .../payload`) | replace entire payload |
| `POST .../points/{id}/payload/delete` | remove keys `{"keys": [...]}` |
| `POST .../points/{id}/payload/clear` | empty the payload |

## Search (dense)

All search bodies accept `filter` and the read options.

| Method & path | Body highlights | Returns |
|---|---|---|
| `POST .../points/search` | `{"query", "k"}` | `{"results":[{"id","score",...}], "degraded", "missing"}` |
| `POST .../points/search/docs` | same | hits + `content` + payload |
| `POST .../points/search/groups` | `{"query","k","group_by","group_size","fetch_k"}` | top-k per group |
| `POST .../points/search/hybrid` | `{"dense","sparse","k","method","alpha","rrf_k","dense_k","sparse_k"}` | fused dense+sparse |
| `POST .../points/search/text` | `{"text","k"}` | BM25 full-text (requires `full_text`) |
| `POST .../points/search/hybrid-text` | `{"dense","text","k","method",...}` | fused dense+BM25 |
| `POST /v1/collections/{n}/query` | `{"root","prefetch":[...],"mode":"fusion"\|"rerank","method","alpha","rrf_k","k"}` | unified multi-lane Query API |

`method` is `"rrf"` (default), `"weighted"` (with `alpha`), or `"dbsf"`.

## Partitioning (dense)

| Method & path | Description |
|---|---|
| `POST /v1/collections/{n}/resplit` | offline repartition `{"new_partitions"}` — quiesce writes first |
| `POST /v1/collections/{n}/resplit/cleanup` | drop orphaned partitions |
| `POST /v1/collections/{n}/reshard` | **online** repartition (dual-write, atomic cutover) |
| `POST /v1/collections/{n}/reshard/abort` | abort an in-flight reshard (pre-cutover) |

## Multi-vector collections

Prefix `/v1/multivector/{name}` — late-interaction MaxSim collections
([Hybrid → Late-interaction](../vector/hybrid-search.md#late-interaction-multi-vector-search)):

| Method & path | Description |
|---|---|
| `POST /v1/multivector/{n}` / `DELETE /v1/multivector/{n}` | create / drop |
| `POST .../docs` | add doc `{"id","tokens":[[...],...],"metadata"}` (insert-or-replace) |
| `DELETE .../docs/{id}` | delete doc |
| `GET .../points/{id}` · `POST .../points/batch-get` | fetch tokens + payload |
| `POST .../points/{id}/payload[/overwrite|/delete|/clear]` | payload mutations |
| `POST .../search` | MaxSim top-k |
| `POST .../hybrid-search` | MaxSim + sparse fusion |
| `POST .../query` | multi-lane Query API |
| `POST .../scroll` | paginated listing |
| `POST .../resplit[...]` · `POST .../reshard[...]` | partitioning (as dense) |

## Named-vector collections

Prefix `/v1/named/{name}` — multi-space collections
([Hybrid → Named-vector](../vector/hybrid-search.md#named-vector-multi-space-search)):

| Method & path | Description |
|---|---|
| `POST /v1/named/{n}` / `DELETE /v1/named/{n}` | create `{"named_vectors": {...}}` / drop |
| `GET .../config` | configured spaces |
| `POST .../points` | upsert `{"id","vectors":{space:[...]},"metadata","ttl_ms"}` |
| `GET .../points/{id}` · `DELETE .../points/{id}` · `POST .../points/delete` · `POST .../points/batch-get` | point ops |
| `POST .../points/{id}/payload[...]` | payload mutations |
| `POST .../search` | kNN in one space `{"vector_name","query","k","filter"}` |
| `POST .../sparse-search` | sparse top-k in a sparse space |
| `POST .../hybrid-search` | cross-space dense+sparse fusion |
| `POST .../query` | multi-space Query API |
| `POST .../search/docs` · `POST .../scroll` | docs variant, listing |

## Aliases

| Method & path | Description |
|---|---|
| `POST /v1/aliases` | `{"alias","collection"}` create/repoint |
| `DELETE /v1/aliases/{alias}` | delete (idempotent) |
| `GET /v1/aliases[?collection=x]` | list |
| `POST /v1/aliases/batch` | atomic `{"actions":[{"create":{...}},{"delete":{...}}]}` |

## Admin

| Method & path | Description |
|---|---|
| `POST /v1/admin/keys` | add API key `{"token","tenant","scopes","cert_cn"}` (admin scope) |
| `DELETE /v1/admin/keys` | revoke `{"token"}` — token in body, never in the path |
| `GET /v1/admin/keys` | list keys (redacted fingerprints) |
| `POST /v1/admin/backup` | trigger a backup now (412 if no object store configured) |
| `GET /v1/admin/backups` | list backups |
| `POST /v1/collections/{n}/evict` | evict to cold tier |
| `POST /v1/collections/{n}/restore` | restore from object store |
