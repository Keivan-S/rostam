# Collections & indexes

A collection is created with `vector.NewCollection(name, cfg)` (library) or
`POST /v1/collections` (server). The `Config` struct controls everything: index
type, graph parameters, quantization, quotas, durability, and filtering
behavior.

```go
col, err := vector.NewCollection("docs", vector.Config{
	Dim:    768,
	Metric: vector.Cosine,
})
```

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

† **Defaulted over HTTP and from the Python client, but required in the Go
library.** `vector.NewCollection` validates these and rejects zero with
`vector: invalid M (must be > 0 and <= 128)`; only the Vamana path fills them
in. Set them explicitly in Go:

=== "Go"

    ```go
    col, err := vector.NewCollection("docs", vector.Config{
    	Dim:    768,
    	Metric: vector.Cosine,

    	M:              16,   // required here — not defaulted
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

Set `Config.IndexType`; the search API is the same for all of them.

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

```go
col.StageBulk(ids, vecs)              // stage raw data
err := col.BuildStaged(runtime.NumCPU()) // multi-core graph construction
```

Use `StageBulkPayloads(ids, vecs, metas)` instead when the points carry metadata,
so a filtered workload gets the same multi-core build rather than falling back to
one indexed insert per point. Content and sparse vectors have no bulk form.

Over HTTP: `POST .../points/bulk` to stage, then `POST .../points/bulk/build`
with `{"workers": 0}` (0 = all cores).

## Writes, reads, and versions

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
