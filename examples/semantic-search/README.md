# semantic-search — a real Rostam integration

A minimal but realistic example: a Go app that connects to a running **rostam-server**
over TCP (`rostam.NewClient`), turns text into vectors with a **hosted embedding API**
(OpenAI), upserts a small document set, and runs a **semantic search**. This is the
end-to-end shape a real project uses — swap the corpus for your data and the embedder
for whatever model you standardize on.

## Run

**1. Start a server** (separate terminal, from the repo root):

```bash
go run ./cmd/rostam-server -tcp 127.0.0.1:9400 -data ./.rostam-data
```

`-data` makes it persistent; drop it for an in-memory server. For a real cluster, run
three nodes with `-cluster -peers ... -replication-factor 3`.

**2. Run the example:**

```bash
export OPENAI_API_KEY=sk-...
go run ./examples/semantic-search
go run ./examples/semantic-search "how do I cancel my subscription?"
```

It runs the query two ways so you can compare — **dense** (pure embedding similarity)
and **hybrid** (dense fused with a lexical/keyword lane via RRF):

```
upserted 6 documents into "docs"

query: "how do I cancel my subscription?"

── dense (semantic) ──
1. score=0.6123 [billing]
   To cancel your subscription, open Settings → Billing and click Cancel plan.
2. score=0.4480 [billing]
   We offer a 30-day money-back guarantee on all annual plans.
...

── hybrid (dense + lexical, RRF) ──
1. score=0.0328
   To cancel your subscription, open Settings → Billing and click Cancel plan.
...
```

Hybrid shines when the query shares **exact terms** with a document (e.g. an error
code, a product name, an API field) that pure semantic similarity might rank lower.

## Env

| var | required | meaning |
|-----|----------|---------|
| `OPENAI_API_KEY` | yes | embeddings API key |
| `ROSTAM_ADDR` | no | server TCP address (default `127.0.0.1:9400`) |

## Key points

- **Dim must match the model.** `text-embedding-3-small` emits 1536-dim vectors, so the
  collection is created with `Dim: 1536`. Change the model → change `embedDim` *and* the
  collection `Dim` (re-create the collection).
- **Metric.** OpenAI embeddings are normalized, so `vector.Cosine` is correct.
- **Swapping providers.** Only the `embed()` function and the two dim constants are
  OpenAI-specific. Point it at Cohere/Voyage/a local model by changing the endpoint +
  response shape.
- **Content + metadata round-trip.** `VectorUpsert` stores the original text as the
  document content and a `category` payload; `VectorSearchDocs` returns both, so you get
  human-readable hits, not just ids.

## Where to go next

- **Filtering:** pass `rostam.VectorSearchOpts{Filter: ...}` to restrict by metadata
  (e.g. `category == "billing"`).
- **Hybrid (already shown):** each point is upserted with a lexical sparse vector
  (`VectorInsertOpts.Sparse` — not the embedded `WriteOpts.Sparse`, which is MV-add-only
  and silently ignored on dense upserts), and `VectorHybridSearch` fuses the dense +
  sparse lanes via RRF.
  `sparseOf()` is a dependency-free stand-in for a learned sparse model (SPLADE) — swap
  it without touching the query path. Try `FusionWeighted`/`FusionDBSF` and the
  `Alpha`/`DenseK`/`SparseK` knobs in `VectorHybridOpts`.
- **Scale out:** the exact same client code works against a multi-node cluster — only the
  server deployment changes (start each node with `-cluster -peers ... -replication-factor 3`).
- **Validate for production:** durability across restarts, leader failover, backups
  (`-backup-dir`), TLS/mTLS + RBAC (`-keys-file`) — see the server flags
  (`go run ./cmd/rostam-server -h`).
