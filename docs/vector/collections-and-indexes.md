# Collections & indexes

A collection is created with `vector.NewCollection(name, cfg)` (library),
`POST /v1/collections` (server), or `create_collection` (Python). The same set
of parameters — index type, graph parameters, quantization, quotas, durability,
filtering behaviour — is available on each; the Go `Config` struct is the
canonical list and the JSON/Python names mirror its fields.

=== "Go"

    ```go
    col, err := vector.NewCollection("docs", vector.Config{
    	Dim:    768,
    	Metric: vector.Cosine,
    })
    ```

=== "Python"

    ```python
    c.create_collection("docs", dim=768, metric="cosine")
    ```

=== "curl"

    ```sh
    curl -s localhost:8080/v1/collections \
      -d '{"name":"docs","config":{"dim":768,"metric":"cosine"}}'
    ```

!!! note "The Go snippets on this page are fragments"
    They show the call shape, not a runnable program — imports and the
    `if err != nil` check are omitted for brevity. See the
    [quickstart](../quickstart.md#your-first-search) for a complete one.

## Core parameters

| Field | Default | Meaning |
|---|---|---|
| `Dim` | — (required) | vector dimensionality |
| `Metric` | — | `Cosine`, `L2`, or `DotProduct` |
| `M` | 16 † | HNSW graph degree |
| `EfConstruction` | 200 † | build-time beam width |
| `EfSearch` | 64 † | query-time beam width (recall/latency dial) |
| `MaxEfSearch` | 1024 | cap on the automatic ef widening filtered search uses; unfiltered search ignores it |
| `Seed` | 0 | RNG seed for deterministic builds |

† **Defaulted to 16 / 200 / 64 when omitted**, on every entry point — the HTTP
API, the Python client and the Go library alike. Setting them explicitly is
optional; the examples below do so to keep the recall/latency dials visible:

=== "Go"

    ```go
    col, err := vector.NewCollection("docs", vector.Config{
    	Dim:    768,
    	Metric: vector.Cosine,

    	M:              16,   // optional — shown to make the dial visible
    	EfConstruction: 200,
    	EfSearch:       64,
    })
    ```

=== "Python"

    ```python
    # The server fills in M / ef_construction / ef_search when omitted.
    c.create_collection("docs", dim=768, metric="cosine")

    # The same collection with the knobs set explicitly. These are
    # alternatives — creating "docs" twice raises "collection already exists".
    c.create_collection("docs-tuned", dim=768, metric="cosine",
                        m=16, ef_construction=200, ef_search=64)
    ```

Higher `M`/`EfConstruction` buys recall at build cost; `EfSearch` is the runtime
knob you tune first.

!!! warning "`EfSearch` below `k` has no effect"
    The effective beam width is `max(EfSearch, k)` — search silently raises ef
    to `k`. So `ef_search=10` with `k=100` queries runs at ef=100, and a
    benchmark that sweeps ef downward while holding `k` fixed will measure the
    same number at every ef ≤ k and conclude the knob does nothing. To explore
    low-ef behaviour, lower `k` as well. Filtered search additionally floors ef
    at `2·k` and keeps doubling it (up to `MaxEfSearch`) until it collects `k`
    matches; quantized indexes floor it at `RescoreFactor·k` to feed the
    [rescore stage](quantization.md).

## Index types

Set the index type at creation; the search API is the same for all of them.

=== "Go"

    ```go
    col, err := vector.NewCollection("docs", vector.Config{
    	Dim: 768, Metric: vector.Cosine,
    	IndexType: vector.IndexVamana, VamanaR: 64, VamanaL: 100,
    })
    ```

=== "Python"

    ```python
    c.create_collection("docs", dim=768, metric="cosine",
                        index_type="vamana", vamana_r=64, vamana_l=100)
    ```

=== "curl"

    ```sh
    curl -s localhost:8080/v1/collections -d '{"name":"docs","config":{
      "dim":768,"metric":"cosine",
      "index_type":"vamana","vamana_r":64,"vamana_l":100}}'
    ```

`index_type` is `"hnsw"` (default), `"ivf"`, `"vamana"`, or `"gpu"`. The
per-index tuning fields below are set the same way — as `Config` fields (Go),
JSON config keys (HTTP), or, where the client exposes them, keyword arguments
(Python; see the note below for the one exception).

!!! note "IVF tuning is HTTP/Go only from the client"
    The Python client selects `index_type="ivf"` but does not expose
    `ivf_nlist` / `ivf_nprobe`; set those over HTTP (`ivf_nlist`, `ivf_nprobe`
    in the config) or from the Go library. HNSW, Vamana and GPU are fully
    configurable from Python.

### HNSW (default)

The default graph index: allocation-free search path, AVX2 distance kernels
(scalar fallback), incremental inserts, tombstoned deletes. Tuning extras:
`ExtendCandidates` / `ExtendCandidatesMax` (wider neighbor selection),
`Level0FullDegree` (denser base layer), `QuantizedBuild` (navigate on quantized
codes during bulk build to cut memory bandwidth; the exact rescore stage
recovers ranking).

The two build-time graph-quality knobs have been measured — QPS at **matched
recall** against the baseline curve, on SIFT (200k × 128d, L2, k=10, `M` 16,
`EfConstruction` 200), across two independent runs
(`vector/recall_levers_test.go`):

- `Level0FullDegree`: **+8.5% and +12.2%**, every measured point positive in
  both runs, for ~1.6× build time. Level 0 already reserves `2·M` edge slots
  per node, so it costs no extra memory. This is the lever that pays.
- `ExtendCandidates`: +0.6% then −4.2% — the sign flips between runs, i.e. no
  measurable effect — for ~2.5× build time. Not worth its build cost on this
  data.
- Both together measured worse than `Level0FullDegree` alone.

So when tuning for recall, reach for `Level0FullDegree` first.

!!! note "These are SIFT numbers"
    128-dimensional L2 at 200k points; graph-quality effects are
    geometry-dependent, so don't change a production default without checking
    your own corpus. The harness is committed — rerun it against your data
    layout with `ROSTAM_RECALL_LEVERS=1 go test ./vector -run TestRecallLevers
    -v -timeout 30m` (needs the SIFT corpus on disk; the test header explains
    setup and how to read the output).

### IVF

Inverted-file index: k-means coarse quantizer + inverted lists, trained when the
collection reaches `IVFTrainThreshold` (or at bulk build).

| Field | Default | Meaning |
|---|---|---|
| `IVFNlist` | ≈ 4·√N | number of cells |
| `IVFNprobe` | 8 | cells probed per query (recall/latency dial) |
| `IVFPQ`, `IVFPQM` | off | product-quantize residuals inside cells |
| `IVFRerank` | off | keep full vectors for an exact rescore pass |
| `IVFDriftFactor`, `IVFDriftGrowthFactor`, `IVFDriftRetrain` | — | online re-training as the data distribution drifts |

### Vamana (DiskANN-style)

Single-layer graph with RobustPrune construction:

| Field | Default |
|---|---|
| `VamanaR` (out-degree) | 64 |
| `VamanaL` (build/search beam) | 100 |
| `VamanaAlpha` (prune α) | 1.2 |

### GPU (CUDA exact KNN)

`IndexType: IndexGPU` with a `-tags cuda` build gives exact brute-force KNN on
the GPU for `Search`/`SearchFiltered` (k ≤ 256; larger k or heavy tombstone
counts fall back to CPU brute force). Advanced queries (MMR, groups, hybrid,
recommend, discover) fall back to HNSW. Without the CUDA build the config fails
with `ErrGPUNotCompiled`.

## Bulk loading

For large initial loads, stage vectors and build the graph on all cores instead
of inserting one by one:

=== "Go"

    ```go
    col.StageBulk(ids, vecs)                 // stage raw data
    err := col.BuildStaged(runtime.NumCPU()) // multi-core graph construction
    ```

    Use `StageBulkPayloads(ids, vecs, metas)` when the points carry metadata, so a
    filtered workload gets the same multi-core build rather than one indexed
    insert per point. Content and sparse vectors have no bulk form.

=== "Python"

    ```python
    c.bulk_stage("docs", ids, vectors)   # stage over the binary wire
    c.bulk_build("docs")                 # multi-core build (workers=0 = all cores)
    ```

    Pass `metadatas=[...]` to `bulk_stage` for a filterable load.

=== "curl"

    ```sh
    # stage, then build
    curl -s localhost:8080/v1/collections/docs/points/bulk -d '{"ids":[...],"vectors":[[...]]}'
    curl -s localhost:8080/v1/collections/docs/points/bulk/build -d '{"workers":0}'
    ```

## Writes, reads, and versions

=== "Go"

    ```go
    // Create-only — ErrDuplicateID if the id is live:
    err := col.Insert(id, vec, ttl, meta, sparse)

    // Insert-or-replace, with stored content for RAG:
    err = col.Upsert(id, vec, "document text", ttl, meta, sparse)

    // Atomic create-if-missing:
    inserted, err := col.InsertIfAbsent(id, vec, ttl, meta, sparse)

    // Optimistic concurrency — fails with ErrVersionConflict on version mismatch:
    version, err := col.InsertCAS(id, vec, ttl, meta, sparse, vector.CASCond{Expected: v, Has: true})

    // Reads (zero-copy; GetInto reuses a caller buffer):
    vec, meta, ttl, sparse, version, ok := col.Get(id)
    deleted := col.Delete(id)
    count, err := col.DeleteByFilter(filter)
    ```

=== "Python"

    ```python
    # Create-only (RostamError if the id is live) vs insert-or-replace:
    c.insert("docs", id, vec, content="document text", metadata=meta)
    c.upsert("docs", id, vec, content="document text", metadata=meta)

    # Reads and deletes:
    points = c.get_batch("docs", [id])           # -> [Point(id, vector, content, metadata)]
    c.delete("docs", id)
    c.delete_by_filter("docs", f.eq("tenant", "acme"))
    ```

    `InsertIfAbsent` and the `InsertCAS` optimistic-concurrency path are
    Go-library only; the client exposes `insert` (create-only) and `upsert`.

Per-point TTL is set at write time; per-key payload TTLs go through
`InsertKeyTTL`/`UpsertKeyTTL` (`key_ttl_ms` over HTTP).

## Quotas & rate limits

`MaxVectors`, `MaxBytes` (→ `ErrCollectionFull`) and `MaxInsertsPerSecond`
(token bucket → `ErrCollectionRateLimited`) are per-collection guards. See
[Collections, tenants & aliases](../concepts/collections.md#quotas-rate-limits-ttl).

## Stats

`col.Stats()` returns live counters: `Size`, `Tombstoned`, `SearchOps`,
`InsertOps`, `AvgSearchDepth`, `Expired`, `QuotaRejects`, `FilterRejects`,
`SparseVectors`, and search/insert latency histograms. The server exports these
per collection at `/metrics` ([Monitoring](../server/monitoring.md)).

## Errors

Config validation returns typed errors (`ErrInvalidDim`, `ErrInvalidMetric`,
`ErrInvalidQuant`, …). The runtime errors you should handle:

| Error | When |
|---|---|
| `ErrDuplicateID` | `Insert` on a live id |
| `ErrDimMismatch` | query/vector length ≠ `Dim` |
| `ErrCollectionFull` | `MaxVectors`/`MaxBytes` quota hit |
| `ErrCollectionRateLimited` | insert token bucket empty |
| `ErrVersionConflict` | CAS `expected_version` mismatch |
| `ErrFullTextDisabled` | `SearchText` without `FullText` enabled |
