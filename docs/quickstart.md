# Quickstart

Rostam can run three ways: as a **single-node server** speaking HTTP/gRPC/TCP,
as a **replicated Raft cluster**, or as a **Go library** embedded in your binary
(no server at all). This page gets you from zero to a working search in each
mode — the server and Python paths first, the Go embedding paths after.

## Requirements

- **Go 1.26+** for library use and building the server.
- The default full-module build requires **cgo** (the WASM stored-procedure
  backend uses `wasmtime-go`). The vector engine and cache packages are pure Go —
  see [Development](development.md#building) for `CGO_ENABLED=0` builds.
- mmap persistence and the AVX2 kernels are Linux/amd64; everything has a
  portable fallback.

## Run the server

`rostam-server` exposes the same engine over three transports: REST (`-http`),
gRPC (`-grpc`), and a compact binary TCP protocol (`-tcp`). From a clone of the
repo:

```sh
git clone https://github.com/rostamlabs/rostam
cd rostam

# Single node: REST on loopback :8080, persisted to ./data
go run ./cmd/rostam-server -http 127.0.0.1:8080 -data ./data
```

!!! note "Non-loopback binds require authentication"

    The server refuses to start with no authentication on a network-reachable
    address (e.g. `-http :8080`). Bind loopback for local development, set
    `ROSTAM_API_KEY`, or pass `-insecure` to run open deliberately — see
    [Security](server/security.md).

Create a collection, insert a point, and search — with plain curl:

```sh
# Create a 4-dimensional cosine collection
curl -s localhost:8080/v1/collections \
  -d '{"name":"docs","config":{"dim":4,"metric":"cosine"}}'

# Upsert a point (metadata values use a tagged encoding — see docs/vector/filtering.md)
curl -s localhost:8080/v1/collections/docs/points \
  -d '{"id":1,"vector":[0.1,0.2,0.3,0.4],"content":"hello rostam",
       "metadata":{"tenant":{"kind":"string","str":"acme"}},"upsert":true}'

# Search
curl -s localhost:8080/v1/collections/docs/points/search \
  -d '{"query":[0.1,0.2,0.3,0.4],"k":3}'
```

The full endpoint inventory is in the [HTTP API reference](api/http.md). To add
authentication and TLS, see [Security](server/security.md); for multi-node
clusters, see [Clustering](server/clustering.md).

## Use it from Python

The Python client is a dependency-free REST wrapper (source lives under
`clients/python`):

```sh
pip install rostam-client
```

```python
from rostam import RostamClient

c = RostamClient("http://localhost:8080")
c.create_collection("docs", dim=4)
c.upsert("docs", 1, [0.1, 0.2, 0.3, 0.4], content="hello rostam",
         metadata={"tenant": "acme"})
hits = c.search_docs("docs", [0.1, 0.2, 0.3, 0.4], k=3)
```

For large initial loads, `c.bulk_stage(...)` + `c.bulk_build(...)` ship vectors
over a binary wire and build the index on all cores. Full method reference:
[Python client](api/python.md).

## Embedded vector search (Go library)

```sh
go get github.com/rostamlabs/rostam
```

The vector engine is a standalone package — no server, no cgo, no other Rostam
dependencies:

```go
package main

import (
	"fmt"

	"github.com/rostamlabs/rostam/vector"
)

func main() {
	col, err := vector.NewCollection("docs", vector.Config{
		Dim:    768,
		Metric: vector.Cosine,
		Quant:  vector.QuantSQ8, // int8 codes: 4× smaller, ~98% recall retained
	})
	if err != nil {
		panic(err)
	}
	defer col.Close()

	embedding := make([]float32, 768) // your embedding model's output
	query := make([]float32, 768)

	// Insert is create-only (ErrDuplicateID on a live id); use Upsert to replace.
	_ = col.Insert(1, embedding, 0, vector.Metadata{
		"tenant": vector.NewString("acme"),
	}, nil)

	// Exact, fast filtered search — the payload index narrows to tenant=acme.
	hits, _ := col.SearchFiltered(query, 10, vector.Filter{
		Op: vector.FilterEq, Field: "tenant", Value: vector.NewString("acme"),
	})
	fmt.Println(hits)

	// Diversified retrieval for RAG:
	diverse, _ := col.SearchMMR(query, 10, vector.MMROpts{Lambda: 0.5})
	_ = diverse
}
```

Where to go next: [Search APIs](vector/search.md), [Filtering](vector/filtering.md),
[Quantization](vector/quantization.md).

## Embedded key-value store (Go library)

The `rostam.Store` facade gives you KV *and* vector operations behind one
interface. `NewDirect` is the single-node, no-Raft backend — the fastest path:

```go
package main

import (
	"context"
	"log"
	"time"

	"github.com/rostamlabs/rostam"
	"github.com/rostamlabs/rostam/ops"
)

func main() {
	ctx := context.Background()

	reg := ops.NewRegistry()
	if err := ops.RegisterBuiltins(reg); err != nil { // get/put/del/incr/expire + vector ops
		log.Fatal(err)
	}
	store, err := rostam.NewDirect(rostam.DirectConfig{
		Ops: reg, // required
		// DataDir: "./data", // enable mmap persistence + warm restart
	})
	if err != nil {
		log.Fatal(err)
	}
	defer store.Close()

	_ = store.Put(ctx, []byte("user:42"), []byte(`{"coins":100}`), 5*time.Minute)
	v, _ := store.Get(ctx, []byte("user:42"))
	_ = v

	// Atomic server-side read-modify-write, serialized per shard:
	_, _ = store.Call(ctx, "incr", ops.EncodeIncrArgs([]byte("views:42"), 1))
	_, _ = store.Call(ctx, "expire", ops.EncodeExpireArgs([]byte("user:42"), time.Hour))
}
```

Swap the constructor to change the backend — the `Store` interface is identical:

| Constructor | Backend | When |
|---|---|---|
| `rostam.NewDirect` | in-process, no Raft | single node, library use, fastest |
| `rostam.NewEmbedded` | in-process + per-shard Raft | replicated / multi-node durability |
| `rostam.NewClient` | TCP client to a remote cluster | talk to a running server |

Details: [KV overview](kv/overview.md), [Deployment modes](concepts/deployment-modes.md).

## Worked examples

- [`examples/semantic-search`](https://github.com/rostamlabs/rostam/tree/main/examples/semantic-search)
  — end-to-end RAG-style pipeline: OpenAI embeddings → upsert → dense vs hybrid
  search over the TCP client.
- [`examples/filtered-recall-cliff`](https://github.com/rostamlabs/rostam/tree/main/examples/filtered-recall-cliff)
  — a runnable demonstration of why the filter-first query planner exists
  (post-filtering recall collapse vs exact filter-first).
