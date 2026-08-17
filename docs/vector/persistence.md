# Persistence, snapshots & cold tier

Dense collections have four durability tools, composable per collection:
snapshots, persistent quantized storage, a WAL, and object-store offload.
Multi-vector collections take a WAL **or** mmap `Persistent` (mutually
exclusive; `Persistent` requires a quantization mode); named-vector collections
take a WAL. Both snapshot like dense collections.

!!! info "Go library and server configuration only"

    Nothing on this page is reachable from the Python client: it has no
    snapshot, flush, or cold-tier methods.

    Two different things are described below. **Durable storage** is chosen at
    creation — `-data` on `rostam-server`, `DataDir` on the store, or the
    persistence fields on `vector.Config` — and then keeps itself up to date
    with no further calls. **Snapshots, flush and the cold tier are explicit
    operations**: `Snapshot`, `Restore`, `Flush`, `EvictCollection` and
    `SweepCold` are calls you make from Go or drive from the server.


## Snapshots

Point-in-time serialization of a collection to any `io.Writer`:

```go
err := col.Snapshot(w)     // full collection -> writer
err = col.Restore(r)       // rebuild from a snapshot stream
```

`ErrSnapshotFormat` signals a corrupted/incompatible stream. The server layer
uses the same mechanism for [backups](../server/backups.md) (filesystem or S3)
and for Raft snapshot transfer in clusters.

## Persistent collections

```go
col, err := vector.NewCollection("docs", vector.Config{
	Dim: 768, Metric: vector.Cosine,
	Quant:      vector.QuantSQ8,  // required for Persistent
	Persistent: true,
	WAL:        true,             // append-only log of mutations
	// WALNoSync: true,           // trade fsync latency for crash-window
	// GraphMmapPath: "...",      // mmap the graph structure too
})
```

- `Persistent` stores the quantized codes durably (hence the quantization
  requirement); `SavePersist(metaPath)` checkpoints collection metadata.
- `WAL` appends mutations so restarts replay to the last write; `Flush()`
  checkpoints WAL + snapshot state.
- `MmapPath` / `GraphMmapPath` memory-map vector payloads and graph structure so
  the resident set is codes + hot pages rather than the whole index.

!!! note "Virtual size runs ahead of resident size"

    A collection's per-slot arrays (vectors, codes, graph) grow in place on a
    reserved address range rather than by copy, so `VIRT`/`VSZ` can read tens of
    gigabytes while actual memory use is unchanged. Alert on `RSS`, not `VIRT` —
    see [Growing a collection doesn't stall
    queries](../performance.md#growing-a-collection-doesnt-stall-queries).

In a **cluster**, the Raft log is the durability authority: the
`-persistent-vectors` server flag (or `EmbeddedConfig.PersistentVectors`)
mmap-backs vector data off-heap on every node to relieve GC/heap pressure,
while recovery correctness comes from Raft replay and snapshots.

## Cold tier — offload idle collections to object storage

`CollectionStore` can evict whole idle collections to an object store, leaving a
lightweight stub that restores lazily on next access:

```go
// Evict one collection now:
err := colStore.EvictCollection(ctx, name, objStore, tenant, time.Now())

// Sweep everything idle for > idle duration:
evicted, err := colStore.SweepCold(time.Now(), time.Hour, objStore, tenant)
```

`objstore.ObjectStore` implementations in-repo: an in-memory store (tests), a
filesystem store (`backup.NewFSObjectStore`), and a stdlib-only S3-compatible
client (AWS, MinIO, Cloudflare R2 — SigV4, path-style or virtual-host).

Snapshot layout in the object store:

```
<tenant>/<escaped-collection-name>/<RFC3339-timestamp>.snap
<tenant>/<escaped-collection-name>/<RFC3339-timestamp>.cfg.json
```

On the server this is driven by the `-cold-tier-after` flag plus the
`-backup-*` S3 flags, and manually via `POST /v1/collections/{name}/evict` /
`.../restore` — see [Backups & cold tier](../server/backups.md).
