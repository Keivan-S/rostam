# WASM stored procedures

WASM ops let a **client ship server-side logic to a running cluster over the
wire** — no recompile, no redeploy. The module is sandboxed, determinism-gated,
and fuel-capped; on `Embedded` it replicates through Raft so every node runs the
same procedure.

## Registering a module

```go
wasmBytes, _ := os.ReadFile("counter.wasm")

pushReport, err := store.RegisterWASM(ctx, rostam.WASMRegistration{
	Name:               "counter_add",     // op name (≤255 bytes, never exactly 2)
	Kind:               ops.OpReadWrite,   // OpReadOnly | OpReadWrite
	ExportName:         "run",             // exported function implementing the op
	MaxFuel:            0,                 // instruction budget; 0 = default
}, wasmBytes)                              // the module itself — a separate argument
if pushReport != "" {
	// Some member does not hold the module's bytes. Not fatal today; see below.
	log.Printf("wasm register: %s", pushReport)
}

// then call it like any op:
res, err := store.Call(ctx, "counter_add", args)
```

On `Embedded`, registration itself is a Raft-replicated op. It is broadcast into
**every shard group's log**, because a WASM op is routable and its invocations are
logged in `shardOf(key)`'s group, not shard 0's. Nodes joining later get it from
the snapshot or replay it from the log.

!!! warning "Your module receives the whole `[keyLen u16][key][payload]` frame"

    There is **no key-extractor setting**. Every WASM op is routed by the one
    extractor, which reads `[keyLen u16][key]` from the start of the args — so an
    op's args are `[keyLen u16][key][payload]` and there is no way to ask for
    anything else. That is what makes it impossible for two registrations of one
    op name to route to different shard groups; see *Racing two first
    registrations* below.

    The extractor selects the **routing key**; it does **not** rewrite the args
    your module is handed. A module that uses its raw args as a cache key will
    therefore address `"\x00\x05hello"` rather than `"hello"` — a key that hashes
    to a **different shard** than the group the invocation is executing in. That is
    legal (WASM ops may touch any key) but is almost never what you meant. **Skip
    the two-byte prefix.**

    An op with no natural key pins itself to one group by always passing the same
    key. There is no shardless WASM op.

!!! info "The module is a separate argument because the log does not carry it"

    What is replicated is a **thin marker**: the op's name, its contract, its
    epoch, and the module's 32-byte content address. The module itself travels
    once, out of band, to each member.

    That matters because the marker goes into *every* shard group's log. With the
    module inline, one registration cost `NumShards ×` its size in Raft log on
    every node — 64 × up to 4 MiB at the default shard count — and the same again
    in each group's snapshot once it compacted. A marker is a few dozen bytes
    whatever the module weighs.

!!! note "A successful `RegisterWASM` does not mean the op is invocable everywhere yet"

    A return means every shard group **committed** the registration. Each node
    then applies it group by group, and it will not route an invocation into a
    shard group until it has applied the registration from *that group's* log —
    which is what guarantees no replica ever meets an invocation for an op it
    cannot look up.

    Until a node gets there, an invocation may come back with `op not registered
    in this shard group yet` (HTTP 503) or, on a node that has applied nothing
    yet, `op not registered` (HTTP 404). Both are transient and safe to retry;
    so is re-running `RegisterWASM`, which is idempotent.

    An **error** return does not mean nothing happened: every group is attempted
    even after one fails, and the groups that accepted keep the registration. The
    op then works for keys routing to those groups and errors for the rest, until
    a retry lands it everywhere.

## Registration is two phases: a push, then the marker

A registration is not only a marker broadcast into every group's log. **Before**
anything enters any log, the node that received the call:

1. stores the module's bytes in its own content-addressed blob store, holding
   itself to the same acceptance rule it imposes on everyone else;
2. delivers those bytes to **every member of the cluster it can reach**, over an
   internal admin op `__wasm_blob_put__`;
3. broadcasts the marker only if step 2 produced no refusal **and a majority of
   cluster members hold the bytes**.

Any node can also serve bytes it holds to any node that asks, over
`__wasm_blob_get__`. Both ops are node-local, admin-gated, and never
re-broadcast; neither is something an application calls.

!!! success "The ack is a **compile verdict**, not a delivery receipt"

    Each member compiles the module with **its own** wasmtime before it acks. So a
    fleet whose nodes disagree about a module — a wasmtime version that rejects an
    instruction another accepts, a build without cgo, a determinism-gate refusal
    on a banned import — says so **here**, to the client that is registering, as a
    `wasm registration refused` (HTTP 400) naming the member and the reason.

    Without it that disagreement surfaces for the first time when the offending
    node *applies* the committed registration — at which point every group's log
    already carries the entry. Registration-time refusal is the difference between
    a 400 an operator can act on and a group that will not move.

!!! danger "A majority must hold the module before anything is proposed"

    The marker names its module and does not carry it, so **a node that has
    applied the marker has not necessarily got the bytes**. If the module reached
    only the registering node and that node then died, no node in the cluster
    could serve it: every group that reached an invocation would wait forever with
    no source, and the registration would be permanent and impossible to execute.

    So the push has a floor. A registration that cannot deliver the module to a
    **majority of cluster members** is refused with `wasm registration refused`
    (HTTP 400), naming how many members it reached. Above the floor, any reachable
    majority necessarily contains a holder — the same assumption the cluster
    already makes to commit anything at all — so a later fetch can always succeed
    while the cluster is available.

    The floor is a statement about the membership **at registration time**. Growing
    the cluster afterwards can erode it (push to 2 of 3, grow to 5, and the
    majority {4th, 5th, one old node} may hold nothing); a node in that position
    reports the block loudly and `__wasm_blob_put__` resolves it by hand.

!!! note "A member that cannot answer is **skipped and named**, never assumed healthy"

    A member that renders **no verdict** does not block the registration — refusing
    would mean any node being restarted could stop the whole cluster from
    registering a module. That covers a member that is down, unreachable, slow past
    the 10s per-member budget, **or running a build that does not know the push op
    at all** (which is what every node looks like during a rolling upgrade onto
    this feature — version skew, not a refusal).

    Every such member is listed in `pushReport`, by node id, with the reason. An
    empty report means every member acked.

    State the residue plainly: a skipped member's wasmtime has **not** agreed the
    module compiles, so for that member the disagreement is still an apply-time
    discovery. The push makes registration-time refusal *available*; it cannot
    make it universal, because a node that cannot be asked cannot answer.

    Server-side, the standing state is on `Stats().WASMBlobPush` (`Acks` /
    `Skips`) — a `Skips` counter that climbs across registrations means some member
    is missing every module's bytes.

## When a node does not have the module: the block

A node that applied a marker without the bytes — it was unreachable during the
push, or it restarted mid-fetch — fetches them in the background from a peer. If
a committed invocation of that version reaches its shard group first, **that
group blocks**: the entry mutates nothing, advances nothing, halts nothing, and
re-runs until the bytes arrive.

The block is deliberately not a halt. A halt is process-global for a group-local
condition, failover cannot help (every replica of the group meets the same entry
in the same log), and it would crash-loop — the entry replays into the same
missing module on restart. A retry records nothing, so there is no divergence for
a halt to prevent.

!!! warning "A blocked group cannot compact its Raft log — alert on **duration**"

    hashicorp/raft runs `Snapshot` on the same goroutine as `Apply`, so a group
    that is waiting cannot snapshot, and therefore cannot compact. **Disk grows for
    as long as the block lasts.**

    `Stats().WASMBlock` reports it:

    | field | meaning |
    |---|---|
    | `Blocked[]` | every parked `(Group, Op)` pair, with `Fingerprint`, `Since`, `Attempts`, `LastErr` |
    | `LongestBlock` | age of the oldest current block — **this is the one to alert on** |
    | `Total` | blocks entered since process start |

    Alert on `LongestBlock`, not on `len(Blocked)`. Sixty-four groups blocked for
    20ms are the system working normally; one group blocked for an hour is a
    disk-consumption incident. Logs escalate to WARN at 5s and ERROR every 30s,
    naming the group, the op, the fingerprint and every peer the fetch has tried.

!!! success "Unblock a node by hand with `__wasm_blob_put__` — no restart"

    `Fingerprint` in the table above is exactly the argument. Against the blocked
    node:

    ```
    __wasm_blob_put__  <fingerprint-hex><module-bytes>
    ```

    The put verifies the hash and compiles the module before it acks, so a wrong
    file is refused rather than accepted. On success the blocked group applies on
    its very next retry — no restart, no failover, no data movement.

    This is the remedy to reach for. The older advice for WASM disk problems is
    "wipe this node's data dir and rejoin", which is slow, does not recover
    config-installed modules, and is unqualified **data loss** if the node is the
    last healthy replica of any group it hosts.

## Updating a module

!!! success "The module may change in place; the op's **contract** may not"

    A second registration under a live name may change the **module** — its
    bytes, its export symbol, its fuel cap — and should carry a higher `Epoch`.
    It may **not** change the op's `Kind`; that is refused at propose time with
    `changing a live WASM op's kind or key extractor is unsupported` (HTTP 400),
    and needs a new op name. (The key extractor cannot change either, for the
    stronger reason that there is no setting for it.)

    ```go
    // Ship new bytes under the same name, at a higher Epoch:
    _, err = store.RegisterWASM(ctx, rostam.WASMRegistration{Name: "counter_add", Epoch: 2, ...}, v2)

    // But a different Kind needs a new name:
    _, err = store.RegisterWASM(ctx, rostam.WASMRegistration{Name: "counter_add_ro", Kind: rostam.OpReadOnly, ...}, v1)
    ```

    **Why the module can change.** The version that executes a committed entry is
    resolved from the **shard group whose log carried it** — each group binds the
    version its own log has committed. Every replica of a group has applied the
    same log prefix, so every replica of that group resolves the same version for
    the same entry, however far along its *other* groups it happens to be. There
    is no window in which two replicas of one group run different bytes.

    **Why the contract cannot.** `Kind` and the key extractor are read *before any
    group is known*: `Kind` decides whether the invocation is replicated at all,
    and the key extractor is what **computes** the group index. So they cannot be
    resolved per group without knowing the group first, and the group cannot be
    known without resolving them. `Kind` is therefore frozen at first
    registration, and the key extractor is the same for every op.

!!! note "An update rolls out per shard group, not atomically"

    Each group switches to the new module when its own log commits the
    registration, so for a short window different groups run different versions.
    Every replica of a given group agrees throughout — that is the property that
    matters for correctness — but there is no single instant at which the whole
    cluster changes over. If an update must be observed atomically by callers,
    register the new module under a new name and switch callers in one step.

    A **partial** broadcast leaves that window open indefinitely for the groups it
    starved. `RegisterWASM` is idempotent; re-run it.

!!! warning "Racing two *first* registrations of one name is still unsafe — but it now fails loudly"

    Per-group binding fixes updates. It does not fix a **race between two first
    registrations** of one name with different content, because no group has a
    prior binding to reconcile against.

    The convergence rule (a total order on `(Epoch, fingerprint)`) picks the same
    winner on every node that **received both** registrations. It does **not**
    guarantee every node receives both: once a node holds one of them it refuses a
    contract-differing leg of the other for every group it leads, so that
    registration never enters those groups' logs at all. The node-wide answer is a
    maximum over a set that genuinely differs between nodes.

    **Different `Kind` — fails closed.** A node whose registry recorded the op
    read-only refuses to apply a committed invocation its peer executes, and that
    refusal *halts* the node rather than skipping the entry. Loud, bounded, and
    recoverable.

    **Different key extractor — cannot happen.** This used to be the one case that
    diverged *silently*: the key extractor **computes** the shard group, so two
    nodes ending on different extractors routed invocations to **different
    groups**, different replica sets applied them, the module's writes landed in
    different groups, every apply succeeded, and nothing anywhere reported an
    error. There was no error to classify, so no backstop could be built for it.
    It is closed at the source instead: a registration has no field to name an
    extractor with, so two registrations of one name using different ones is not a
    state the system can be put in.

    Separately, two groups can settle on different **versions** — each binds the
    maximum of what *its own* log carried — and the group that bound the loser
    refuses the winner and keeps its own contract.

    Register a name from one place, once.

### How a module is stored

Module bytes are **content addressed**: they are written to
`<data-dir>/wasm/blobs/<sha256-of-the-bytes>.wasm`, and a blob is valid only if
its contents hash to its own filename, so every load re-verifies what it just
read. Two ops that happen to share a module share one blob.

Everything else about an op lives in a `<data-dir>/wasm/<name>.json` sidecar:
Kind, export name, key-extractor handle, fuel cap, epoch, the fingerprint of the
blob, and the **per-group version bindings** — for each shard group whose log is
known to carry the registration, the module version that group's log has
committed. The sidecar is written last and is the commit point — a crash
mid-install leaves the previous sidecar and an unreferenced blob, never a module
and its metadata describing different registrations.

The bindings are persisted because they are **not re-derivable**: in durable mode
the FSM skips every entry at or below the recorded applied index, so the
registration entries that established them never replay. A group's bound version
may differ from the node-wide install, so each binding names its own blob.

### Retiring a superseded version

By default, superseded versions are **not** garbage collected. Any shard group
still below the point where a version was superseded may replay an entry that
needs it, so every version any group has bound stays on disk and instantiated.
The cost is bounded by how many distinct registrations a deployment issues for a
name, not by traffic.

`-wasm-blob-retention` (default `0`, meaning **off**) turns that into a bounded
cost: a blob that nothing on the node references — a superseded version, or a
`__wasm_blob_put__` orphan — has its **file** deleted once it has been
unreferenced for that long. The instantiated module is left in the runtime, so an
invocation executing under a retired version, or a group whose binding still names
it, is unaffected.

!!! danger "Setting it is an assertion about your operations, not a tuning knob"

    No **local** rule can be safe against a lagging replica elsewhere. Whether
    some other replica of some group will still apply an entry under a superseded
    version is not a function of anything this node holds, and deciding it
    cluster-wide is cluster-wide GC, which is deliberately not built. Worse, the
    obvious local proxy is actively backwards: the nodes that have moved past a
    version — and would therefore drop it first — are exactly the nodes that
    *have* it, while the lagging replica that still needs it never fetched it in
    the first place.

    So the value you set is your claim that no replica of any group this node
    hosts will fall further behind the supersession of a module version than that.
    Err high — hours or days, not minutes.

    If the claim is wrong, the lagging replica **parks** on the missing version.
    It is named with its exact fingerprint in `Stats().WASMBlock`, logged with
    escalating severity, and unblocked by one `__wasm_blob_put__` against that
    node — no restart, no failover, no data movement. That the failure is loud and
    one call from recovery is the only reason the trade is offered at all.

    Two things are never retired whatever their age: a version any hosted group's
    binding still names, and a fingerprint this node is currently blocked on or
    fetching. A node hosting **no** shard group retires nothing at all — it
    applies no markers, so every blob it holds is push residue that exists to
    serve the durability floor.

`Stats().WASMBlobRetire` reports `Retention` (0 = off), `Sweeps`, `Retired` and
`Pending`. A rise in `Retired` followed by a `WASMBlock` entry naming a
fingerprint is what a window set too short looks like.

!!! warning "On-disk format break"

    A data directory written before per-group version binding is **not readable**
    by this build: its sidecars are format version 1 and are refused at startup.
    The app is unreleased, so no migration is offered — wipe the node's data
    directory and rejoin (confirm another healthy replica of every group it hosts
    first), or delete `<data-dir>/wasm` if the modules were config-only and will
    come back from `WASMModules`.

### Op names are filenames

The sidecar is named after the op, so an op name must be a single path component.
A name containing `/`, `\`, `..`, a NUL byte, or equal to `.` or `..` is refused
(HTTP 400) at propose time *and* at apply time on every replica. Use plain
identifiers: `counter_add`, `counter_add_v2`.

    Re-running `RegisterWASM` with the **same** registration is always fine — that
    is the documented retry, and it is how a partial broadcast is repaired.

## The sandbox

- **Determinism gate.** At compile time the module's imports are walked against
  a whitelist: only Rostam host functions are allowed. WASI and any other
  import are rejected (`wasm.ErrBannedImport`) — no clock, no RNG, no
  filesystem, which is what makes Raft replay safe.
- **Fuel caps.** Every invocation gets an instruction budget (`MaxFuel`); an
  op that exhausts it traps instead of hanging a shard.
- **Kind enforcement.** Modules importing state-mutating host functions are
  detected at registration, so an `OpReadOnly` module can't sneak writes in.

## Host functions

Modules import from the `"rostam"` module:

| Import | Semantics |
|---|---|
| `cache_get(key_ptr, key_len) → len` | read a key from the shard |
| `cache_put(key_ptr, key_len, val_ptr, val_len, ttl_ms)` | write |
| `cache_del(key_ptr, key_len) → existed` | delete |
| `cache_expire(key_ptr, key_len, ttl_ms)` | update TTL |
| `set_result(ptr, len)` | set the op's result bytes |

Anything that can compile to freestanding WASM works — Rust, Zig, TinyGo, C.

## Build requirements

The runtime is `wasmtime` (Cranelift JIT) via cgo. Non-cgo builds still link and
start — the stub backend returns `wasm.ErrNoCGO` when a WASM op is registered or
invoked, so the rest of the engine is unaffected. See
[Development → Building](../development.md#building).
