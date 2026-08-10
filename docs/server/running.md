# Running the server

`rostam-server` exposes the engine over three transports — REST, gRPC, and a
compact binary TCP protocol — all dispatching into the same store, so semantics
are identical regardless of how you connect.

```sh
# from a repo clone
go run ./cmd/rostam-server -http 127.0.0.1:8080 -data ./data

# or build a binary
go build -o rostam-server ./cmd/rostam-server
./rostam-server -http 127.0.0.1:8080 -grpc 127.0.0.1:9090 -tcp 127.0.0.1:7000 -data ./data
```

!!! warning "A bare `:8080` will not start without authentication"

    With no authenticator configured, every request is served unauthenticated —
    so the server **refuses to bind a reachable address** rather than silently
    exposing an open datastore. A bare `:8080`, `0.0.0.0`, `::` or any
    non-loopback host counts as reachable. To listen beyond loopback, give it
    auth (`-api-key` or `-keys-file`) or pass `-insecure` to run open
    deliberately. Loopback binds are unaffected, which is why the examples above
    spell out `127.0.0.1`.

## Transports & storage

| Flag | Default | Meaning |
|---|---|---|
| `-http` | `:8080` | REST/JSON listen address; `""` disables |
| `-grpc` | disabled | gRPC listen address |
| `-tcp` | disabled | binary TCP protocol listen address (what `rostam.NewClient` speaks) |
| `-data` | in-memory | persistence directory; empty = nothing survives restart |
| `-shards` | auto | cache shards (single node) or Raft shards (cluster) |

Any subset of transports can be enabled. `GET /v1/health` (auth-exempt) is the
liveness probe and `GET /v1/ready` the readiness probe — use the latter for
load-balancer membership, since health stays green on a node that has lost
quorum. `GET /metrics` serves Prometheus ([Monitoring](monitoring.md)).

## Flag reference by area

**Authentication** — `-api-key` (single superuser key; prefer the
`ROSTAM_API_KEY` env var so the key doesn't show in `/proc`), `-keys-file`
(RBAC key registry), `-internal-token` (inter-node credential; prefer
`ROSTAM_INTERNAL_TOKEN`), `-audit-log`, `-tenant-isolation`,
`-jwt-public-key` / `-jwt-issuer` / `-jwt-audience`. With **no** authentication
configured, the server refuses to start on a non-loopback bind unless you pass
`-insecure` — an open datastore on the network must be a deliberate choice.
→ [Security](security.md)

**TLS / mTLS** — `-tls-cert`, `-tls-key`, `-tls-ca`,
`-tls-require-client-cert`, `-tls-node-cert`, `-tls-node-key`,
`-node-cn-allowlist`. One certificate config covers all three transports.
→ [Security](security.md#tls)

**Storage** — `-data`, `-shards`, `-config` (carries the cache `max_memory`
stanza), `-disable-cold-compaction`. The last one is an escape hatch, not a
tuning knob: a persistent shard rewrites its pages file live-only at open, and
that rewrite is the only thing that reclaims the bytes left behind by overwritten
and expired keys — without it a shard under TTL churn eventually refuses writes.
Turn it off only to work around a problem with the rewrite itself.

**Clustering** — `-cluster`, `-node-id`, `-raft-addr`, `-bootstrap`, `-peers`,
`-replication-factor`, `-persistent-vectors`, `-reconfigure`; durability
posture: `-nosync`, `-volatile-log`; replication engine: `-replication-mode`
(`raft` | experimental `pb`, with `-min-isr`, `-pb-addr`, `-pb-commit-primary`,
`-pb-auto-failover`); Raft transport: `-raft-transport`
(`mux` | experimental `fabric`).
→ [Clustering](clustering.md)

**Backups & cold tier** — `-backup-dir`, `-backup-interval`,
`-backup-prefix`, `-backup-retention`, `-backup-bucket`, `-backup-endpoint`,
`-backup-region`, `-backup-tenant`, `-cold-tier-after`, `-s3-path-style`;
`-restore` + `-allow-missing-shards` for one-shot cluster disaster recovery.
→ [Backups & cold tier](backups.md)

**Logging** — `-log-format` (`text` | `json`), `-log-level`
(`debug`–`error`), `-access-log` (one structured line per request on every
transport, principal redacted).

## Recipes

**Dev server, everything open, in-memory:**

```sh
rostam-server -http 127.0.0.1:8080
```

A loopback bind may run open; a network-reachable bind (e.g. `-http :8080`)
with no authentication is refused at startup unless you pass `-insecure`.

**Single node with persistence and a single API key:**

```sh
ROSTAM_API_KEY=$(openssl rand -hex 32) rostam-server \
  -http :8080 -tcp :7000 -data /var/lib/rostam
```

**TLS on all transports:**

```sh
ROSTAM_API_KEY=$(openssl rand -hex 32) rostam-server \
  -http :8443 -grpc :9443 -tcp :7443 -data /var/lib/rostam \
  -tls-cert server.pem -tls-key server-key.pem
```

(TLS does not substitute for authentication — a non-loopback bind still needs
an API key, a keys file, or an explicit `-insecure`.)

**Three-node replicated cluster:** see [Clustering](clustering.md#bootstrapping-a-cluster).

## Choosing a transport

| Transport | Wire cost | Use when |
|---|---|---|
| HTTP | JSON, easiest | curl, Python client, browsers, most integrations |
| gRPC | protobuf | typed clients, streaming-friendly infrastructure |
| TCP | compact binary, lowest latency | the Go smart client (`rostam.NewClient`), latency-sensitive services |

### The TCP event-loop transport (`-epoll`)

`-tcp` can be served two ways: a goroutine per connection, or an epoll event
loop (`-epoll`, with `-epoll-loops` setting the loop count; `0` = `GOMAXPROCS`).

| Mode | `-epoll` default | Why |
|---|---|---|
| Single node | on | It beats goroutine-per-connection under core pressure — up to ~1.4x at low concurrency on an 8-core co-located box — and is within noise on dedicated cores. |
| `-cluster` | off unless `-epoll` is passed explicitly | The event loop executes dispatch **inline**, so a replicated write blocks the whole loop for a full replication round trip. Write throughput is then capped at ~(loops / RTT) regardless of connection count — measured 2.1x slower than the goroutine server on a real-network 3-node PB RF=2 cluster (61k vs 127k ops/s at 128 connections). |

The event-loop transport is **plaintext only**: with TLS configured the TCP
transport silently falls back to the goroutine server.
