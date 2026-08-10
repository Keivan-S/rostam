# gRPC API reference

Enable with `rostam-server -grpc 127.0.0.1:9090` (a non-loopback bind
additionally needs auth or `-insecure` — see [Running](../server/running.md)). The service is `VectorService`; all
RPCs are unary. Message semantics mirror the [HTTP API](http.md) one-to-one —
this page is the RPC inventory.

**Auth**: bearer token in the `authorization` request metadata. With mTLS
(`-tls-ca` + client certs), the verified certificate CN can authenticate via a
`cert_cn`-bound key ([Security](../server/security.md)).

## Health & collections

| RPC | Semantics |
|---|---|
| `Health` | liveness |
| `CreateCollection` / `DropCollection` | dense collection lifecycle |

## Dense points & payload

| RPC | Semantics |
|---|---|
| `Upsert` | insert / upsert a point |
| `Delete` | delete by id |
| `Get` / `GetBatch` | fetch by id(s) with projections |
| `SetPayload` / `OverwritePayload` / `DeletePayloadKeys` / `ClearPayload` | payload mutations |
| `DeleteByFilter` | bulk delete by filter |
| `Scroll` | paginated filtered listing (cursor-based) |

## Dense search

| RPC | Semantics |
|---|---|
| `Search` | kNN (+ filter, read options) |
| `SearchDocs` | kNN + content + metadata |
| `SearchGroups` | top-k per group |
| `HybridSearch` | dense + sparse fusion |
| `TextSearch` | BM25 full-text |
| `HybridTextSearch` | dense + BM25 fusion |
| `VectorQuery` | unified multi-lane Query API |

## Partitioning

| RPC | Semantics |
|---|---|
| `Resplit` / `ResplitCleanup` | offline repartition |
| `Reshard` / `ReshardAbort` | online repartition |

## Multi-vector collections

`MVCreateCollection`, `MVDropCollection`, `MVAdd`, `MVSearch`,
`MVHybridSearch`, `MVVectorQuery`, `MVScroll`, `MVDelete`, `MVGet`,
`MVGetBatch`, `MVSetPayload`, `MVOverwritePayload`, `MVDeletePayloadKeys`,
`MVClearPayload`, `MVResplit`, `MVResplitCleanup`, `MVReshard`,
`MVReshardAbort` — the MaxSim family
([late interaction](../vector/hybrid-search.md#late-interaction-multi-vector-search)).

## Named-vector collections

`NamedCreate`, `NamedDrop`, `NamedGetConfig`, `NamedUpsert`, `NamedDelete`,
`NamedGet`, `NamedGetBatch`, `NamedSearch`, `NamedSparseSearch`,
`NamedHybridSearch`, `NamedVectorQuery`, `NamedSearchDocs`, `NamedScroll`,
`NamedSetPayload`, `NamedOverwritePayload`, `NamedDeletePayloadKeys`,
`NamedClearPayload` — the multi-space family
([named vectors](../vector/hybrid-search.md#named-vector-multi-space-search)).

## Aliases & admin

| RPC | Semantics |
|---|---|
| `CreateAlias` / `DeleteAlias` / `ListAliases` | alias lifecycle |
| `AliasBatch` | atomic multi-action repoint |
| `KeysAdd` / `KeysRevoke` / `KeysList` | runtime API-key administration (admin scope) |

## Choosing gRPC vs HTTP vs TCP

gRPC gives you generated typed clients and protobuf efficiency; HTTP is the
easiest to integrate and what the Python client speaks; the binary TCP protocol
is the lowest-latency option and what the Go smart client uses
([Go client](go-client.md)).
