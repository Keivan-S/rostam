# Deployment modes

Rostam runs the same engine behind three interchangeable backends. You pick one
by choosing a constructor; the `rostam.Store` interface is identical across all
three, so code written against it is portable between modes.

| Constructor | Process model | Replication | Typical use |
|---|---|---|---|
| `rostam.NewDirect` | in-process | none | library embedding, single node, tests, fastest |
| `rostam.NewEmbedded` | in-process | per-shard Raft | replicated node — single or multi-node cluster |
| `rostam.NewClient` | remote over TCP | (server-side) | application talking to a running cluster |

## Direct — in-process, no consensus

```go
store, err := rostam.NewDirect(rostam.DirectConfig{
	Ops:     reg,        // required: ops.Registry with RegisterBuiltins
	DataDir: "./data",   // optional: mmap persistence + warm restart
})
```

`DirectConfig` fields:

| Field | Meaning |
|---|---|
| `Ops` | **Required.** The op registry; call `ops.RegisterBuiltins` and add your custom ops. |
| `DataDir` | Root directory for the cache mmap files. Empty = pure heap mode, no persistence. |
| `Cache` | Cache-layer tuning (shard count, page size, durability — see [Cache tuning](../kv/cache.md)). |
| `Authenticator` | Optional RBAC gate; `nil` = open mode. |

Writes skip consensus entirely, which is why Direct is the fastest backend
(~29 ns Get, ~240 ns Put in-process). The trade-off: no replication, and
single-key `Put`/`Del` through the facade are simple cache writes.

## Embedded — in-process with per-shard Raft

```go
store, err := rostam.NewEmbedded(rostam.EmbeddedConfig{
	NodeID:    "n1",
	DataDir:   "./n1",
	Ops:       reg,
	Bootstrap: true, // first start of a fresh cluster only
})
```

Key `EmbeddedConfig` fields:

| Field | Meaning |
|---|---|
| `NodeID` | Unique node identifier in the cluster. |
| `DataDir` | Base directory for Raft logs, snapshots, and mmap files. |
| `Ops` | **Required.** Register the same ops on every node — read-write ops replicate as Raft entries and each node's FSM looks them up by name. |
| `NumShards` | Independent Raft shards (default 64). Throughput scales with shards, but each shard runs its own goroutine set — size it near the node's core count rather than far above it. |
| `Peers` / `RaftAddr` | Static membership and this node's Raft transport address. `RaftAddr` is required when there is more than one peer. |
| `ReplicationFactor` | Replicas per shard. `0` or ≥ number of peers = full replication; smaller values partition shards across the cluster. |
| `Bootstrap` | Bootstrap a fresh cluster — set on first start only. |
| `PersistentVectors` | mmap-back vector collections off-heap (Raft remains the durability authority). |
| `InternalToken`, `InterNodeTLS`, `NodeCNAllowlist` | Inter-node auth and TLS — see [Security](../server/security.md). |
| `NoSync` | Disable fsync on Raft log writes. **Testing only.** |

Reads stay local; writes go through the shard's Raft log. See
[Clustering](../server/clustering.md) for multi-node setup and online resharding.

## Client — remote smart client

```go
store, err := rostam.NewClient(rostam.ClientConfig{
	Servers:   []string{"10.0.0.1:7000", "10.0.0.2:7000"},
	AuthToken: os.Getenv("ROSTAM_TOKEN"),
	// TLSConfig: tlsutil.ClientTLS("ca.pem", "client.pem", "client-key.pem", ""),
})
```

The client keeps a live view of cluster topology (refresh every 5 s by default),
routes writes to each shard's leader, retries bounded times on stale-leader
errors (`MaxNotLeaderHops`, default 5), and pools connections per server
(`MaxConnsPerServer`, default 8). Details: [Go client](../api/go-client.md).

## Read consistency

Reads in a replicated deployment accept a consistency level (`ReadOpts` on the
Go facade, `read_consistency` over HTTP/gRPC):

| Level | Name | Semantics |
|---|---|---|
| 0 | AnyReplica | Load-balanced replica read. Default, fastest. |
| 1 | LeaderOnly | Served by the shard's current Raft leader (best-effort — no barrier). |
| 2 | Linearizable | ReadIndex barrier: leader verification + commit-index catch-up. Read-your-writes. |
| 3 | BoundedStaleness | Any replica whose applied index lags the leader by at most `max_staleness` Raft entries. |

Cross-shard fan-out reads also accept `on_partition_unavailable`: `0` (Partial,
default) returns results from reachable partitions and flags degradation; `1`
(Fail) errors the whole request.

## Write consistency

Writes on partitioned collections accept `write_consistency_factor` and `wait`
(HTTP: on point writes; Go: `WriteOpts`): how many replicas must acknowledge
before the call returns. The default is Raft majority on the owning shard.
