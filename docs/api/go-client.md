# Go client (TCP smart client)

`rostam.NewClient` returns the same `rostam.Store` interface as the embedded
backends, speaking the compact binary TCP protocol to a running server or
cluster (`rostam-server -tcp ...`).

```go
import (
	"os"

	"github.com/rostamlabs/rostam"
)

store, err := rostam.NewClient(rostam.ClientConfig{
	Servers:   []string{"10.0.0.1:7000", "10.0.0.2:7000"}, // bootstrap list
	AuthToken: os.Getenv("ROSTAM_TOKEN"),
})
if err != nil { ... }
defer store.Close()

// Same interface as NewDirect/NewEmbedded:
_ = store.Put(ctx, []byte("k"), []byte("v"), 0)
hits, meta, err := store.VectorSearchExt(ctx, "docs", query, 10, opts)
```

## Configuration

| Field | Default | Meaning |
|---|---|---|
| `Servers` | — (required) | initial `host:port` bootstrap list; topology is discovered from any live entry |
| `AuthToken` | "" | bearer token sent on every RPC (protocol-v2 frame; 255-byte limit — registry tokens, not JWTs) |
| `TLSConfig` | nil (plaintext) | build with `tlsutil.ClientTLS(caFile, certFile, keyFile, serverName)` for TLS/mTLS |
| `Ops` | nil | registry mirror used to route **custom ops** by key; without it, custom-op calls fall back to round-robin routing |
| `MaxConnsPerServer` | 8 | connection pool per server |
| `MaxNotLeaderHops` | 5 | retry budget when topology is stale |
| `TopologyRefreshInterval` | 5 s | how often cluster membership/leadership is re-polled |

## What "smart" means

- **Topology awareness** — the client polls cluster membership and per-shard
  leadership, so requests go to the right node the first time.
- **Leader routing** — writes route to the owning shard's Raft leader; reads
  can be served by any replica (subject to the per-request read-consistency
  level).
- **Bounded retries** — a write that lands on a stale leader is retried toward
  the new one up to `MaxNotLeaderHops` times before surfacing
  `rostam.ErrNotLeader`.
- **Connection pooling** — up to `MaxConnsPerServer` multiplexed connections
  per node.

## Calling custom ops remotely

Registered [custom ops](../kv/custom-ops.md) and
[WASM procedures](../kv/wasm.md) are invoked by name exactly as in-process:

```go
res, err := store.Call(ctx, "incr", ops.EncodeIncrArgs([]byte("views:42"), 1))
```

Mirror your op registry into `ClientConfig.Ops` so the client can extract the
routing key and send the call directly to the owning shard's leader.

`RegisterWASM` also works over the TCP client — the registration is forwarded
to the leader, and the returned push report names any cluster members that did
not yet receive the module bytes:

```go
pushReport, err := store.RegisterWASM(ctx, reg, moduleBytes)
```

## Errors

| Error | Meaning |
|---|---|
| `rostam.ErrNotFound` | key/point absent or expired |
| `rostam.ErrNotLeader` | could not reach the shard leader within the retry budget (election in progress, stale topology) |
| `vector.ErrDuplicateID` | dense `VectorInsert` on a live id |

The TCP token limit (255 bytes) means JWT auth is HTTP/gRPC-only; use registry
tokens for TCP clients.
