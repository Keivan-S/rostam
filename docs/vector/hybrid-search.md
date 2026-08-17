# Hybrid & full-text search

Dense embeddings capture semantics; sparse/lexical signals capture exact terms.
Rostam runs both lanes and fuses the rankings.

## Sparse vectors

A sparse vector is `{Indices []uint32, Values []float32}` with strictly
ascending indices — typically SPLADE/BM25-style term weights. Attach one to a
point at write time (the `sparse` parameter on insert/upsert), then:

=== "Go"

    ```go
    hits, err := col.HybridSearch(denseQuery, sparseQuery, 10, vector.HybridOpts{
    	Method: vector.FusionRRF, // RRF | Weighted | DBSF
    	Filter: filter,
    })
    ```

=== "Python"

    ```python
    # Attach the sparse lane at write time...
    c.upsert("docs", 1, dense_vec, content="how do i rotate api keys",
             sparse={"indices": [3, 17], "values": [0.8, 0.4]})

    # ...then query both lanes and fuse.
    hits = c.hybrid_search("docs", dense_query, k=10,
                           sparse={"indices": [3, 17], "values": [0.8, 0.4]},
                           method="rrf")   # "rrf" | "weighted" | "dbsf"
    ```

Malformed sparse data fails fast: `ErrSparseMismatch` (length mismatch),
`ErrSparseUnsorted` (indices not strictly ascending).

## Fusion methods

| Method | How it combines lanes | When to use |
|---|---|---|
| `rrf` (default) | reciprocal-rank fusion, `1/(RRFK + rank)` per lane (RRFK default 60) | scale-free, robust default — no score calibration needed |
| `weighted` | min-max normalize each lane's scores, blend with `alpha` (dense weight, 0–1) | when you've tuned the dense/sparse balance |
| `dbsf` | distribution-based score fusion (3-sigma normalization) | lanes with very different score distributions |

Each lane over-fetches before fusion: `DenseK`/`SparseK` default to
`max(k, 50)`. Fused results carry `Score` (fusion score) alongside `Distance`.

## BM25 full-text search

Dense collections can carry a full BM25 text index. Enable it at creation:

```go
col, err := vector.NewCollection("docs", vector.Config{
	Dim: 768, Metric: vector.Cosine,
	FullText: vector.FullTextConfig{ /* Analyzer: "english", K1: 1.2, B: 0.75 */ },
})
```

(HTTP: `"full_text": {"analyzer":"english","k1":1.2,"b":0.75}` — or `true` from
the Python client for defaults.)

The server tokenizes stored content and queries — no sparse encoder needed on
the client:

=== "Go"

    ```go
    docs, err := col.SearchText("how do i rotate api keys", 10, filter) // BM25 top-k
    ```

=== "Python"

    ```python
    # Pure BM25 — no embedding needed on the client at all.
    docs = c.search_text("docs", "how do i rotate api keys", k=10)

    # BM25 fused with a dense lane.
    docs = c.hybrid_text("docs", dense_query, "how do i rotate api keys",
                         k=10, method="rrf")
    ```

    Enable the index at creation with `full_text=True`; calling text search
    without it raises the client's error for `ErrFullTextDisabled`.

Through the store facade / HTTP, `search/text` runs pure BM25 and
`search/hybrid-text` (Go: `VectorHybridText`) fuses BM25 with a dense lane
(`rrf`, `weighted`, or `dbsf`). Calling text search on a collection created
without `FullText` returns `ErrFullTextDisabled`.

**Partitioned collections:** BM25 IDF statistics are per-partition by default.
The `global_idf` option (Python/HTTP) computes corpus-wide IDF across
partitions at some extra fan-out cost — use it when partitions have skewed
vocabularies.

## Late-interaction (multi-vector) search

For ColBERT-style models, use a multi-vector collection: each document stores
its token vectors, and queries score with MaxSim:

```go
// store facade
err := store.VectorMVCreateCollection(ctx, "docs-colbert", cfg)
err = store.VectorMVAdd(ctx, "docs-colbert", docID, tokenVecs, meta)
hits, meta, err := store.VectorMVSearch(ctx, "docs-colbert", queryTokens, 10, opts)
```

MV collections also support hybrid MaxSim + sparse fusion (`VectorMVHybridSearch`)
and the unified Query API. HTTP surface: `/v1/multivector/{name}/...`
([HTTP reference](../api/http.md#multi-vector-collections)).

## Named-vector (multi-space) search

Named-vector collections hold several independent vector spaces per point
(dense and sparse), sharing one payload:

```go
err := store.VectorNamedCreateCollection(ctx, "products", map[string]rostam.NamedVectorParams{
	"image": {...}, "text": {...}, "terms": {...}, // dense + sparse spaces
}, 0)
hits, err := store.VectorNamedSearch(ctx, "products", "image", queryVec, 10, filter)
hits, err = store.VectorNamedHybridSearch(ctx, "products", "text", denseQ, "terms", sparseQ, 10, opts)
```

HTTP surface: `/v1/named/{name}/...` with `search`, `sparse-search`,
`hybrid-search`, and the multi-space `query` endpoint.
