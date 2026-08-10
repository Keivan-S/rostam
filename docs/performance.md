# Performance

Every number on this page is **measured**, not projected, and — apart from the
one line that says otherwise — every one comes from the same machine:

> 12-core AMD EPYC Genoa (KVM guest), 22 GB RAM, Ubuntu 24.04.4, kernel
> 6.8.0-90, Go 1.26.5. AVX2, AVX-512 and AVX-512-VNNI available.

That includes the six-engine VectorDBBench comparison summarised below, which
has always run on this hardware. Stating one configuration is the point: figures
gathered on different machines cannot be compared to each other, and a page that
mixes them silently invites exactly that mistake.

**Same machine, not the same sitting.** The engine-to-engine comparison was
captured in its own continuous session, which is what makes those *ratios*
meaningful; everything else here was measured later on the same box. Where a
cross-session claim would be load-bearing it is called out — see the
re-verification note under the head-to-head.

What this configuration does *not* tell you is how Rostam behaves on yours. Core
count matters a great deal here — the key-value figures are parallel benchmarks
whose per-op cost depends directly on `GOMAXPROCS`, so they are quoted with the
thread count attached. Re-run `make bench` on your own hardware before sizing
anything.

Competitive, reproducible head-to-head comparisons against other engines live
in the separate [`rostam-bench`](https://github.com/rostamlabs/rostam-bench)
repository, so this module's `go.mod` stays dependency-light.

## Vector search (this module's numbers)

| Aspect | Result |
|---|---|
| SQ8 quantization (4× smaller) | recall@10 ≈ **0.98** of exact float32 |
| Binary quantization (32× smaller) | recall@10 ≈ **0.96** of exact (clustered corpus, rescored) |
| Selective metadata filter | filter-first path is **exact** — no graph-recall cliff |

The recall figures are properties of the algorithm and do not vary with
hardware. The distance kernels do, so they are given as measured:

| Kernel (median ns/op) | scalar | AVX2 | AVX-512 | speedup |
|---|---|---|---|---|
| float32 dot, 768d | 451.6 | 35.8 | 34.8 | **12.6×** |
| float32 dot, 1536d | 933.1 | 64.4 | 67.3 | **14.5×** |
| float32 L2², 768d | 498.5 | 35.3 | 35.4 | **14.1×** |
| int8 (SQ8) dot, 768d | 229.7 | 69.2 | — | **3.3×** |
| int8 (SQ8) dot, 1536d | 452.6 | 154.4 | — | **2.9×** |
| int8 dot, 128d, VNNI vs AVX2 | — | 7.76 | 5.87 (VNNI) | **1.3×** |

Two things worth reading off that table rather than skipping:

- **AVX-512 is not a win over AVX2 for float32 here, and sometimes loses.** Zen 4
  implements AVX-512 on 256-bit hardware, so the wider instructions buy little.
  The AVX-512 assembly is correct — its differential tests against the scalar
  reference pass — it simply is not faster on this CPU. On an Intel server part
  with native 512-bit datapaths the picture would differ.
- **VNNI is the one that pays**, and only for int8: 5.87 ns against AVX2's
  7.76 ns. That is the kernel quantized search actually leans on.

The filter-first claim is runnable:
[`examples/filtered-recall-cliff`](https://github.com/rostamlabs/rostam/tree/main/examples/filtered-recall-cliff)
compares post-filtering, filter-aware traversal, and filter-first as filter
selectivity tightens.

## Growing a collection doesn't stall queries

A vector collection stores its vectors, its graph edges, and its quantization
codes in flat arrays indexed by slot. Those arrays have to grow as the
collection does, and a flat array historically grew by allocating a bigger one
and copying — with queries locked out for the duration. At 500k × 768d that copy
was over a second.

Those three arrays now grow **in place**: each reserves a large range of virtual
address space up front and makes more of it usable as needed, so nothing is
copied and nothing moves. Worst-case query latency at a growth boundary, 500k ×
768d:

| Storage | Worst query, before | Worst query, after | after: p50 / p99.9 |
|---|---|---|---|
| Heap (`QuantInRAM`) | 1.985 s | **3.58 ms** | 886 µs / 1.465 ms |
| Memory-mapped (`QuantMmap`) | 741 ms | **4.04 ms** | 420 µs / 1.003 ms |

The p50 column is there because the maximum alone would let you assume the
typical query got slower to buy that worst case down. It didn't — median latency
is unchanged, and the tail is what collapsed. Reproduce with
`ROSTAM_GROW_LATENCY=1 go test ./vector/ -run TestGrowStallLatency -v` (needs
several GB of RAM).

Two things worth knowing when you operate this:

- **Virtual size grows, resident size does not.** A large collection reserves far
  more address space than it uses, so `VIRT`/`VSZ` in `top` or `ps` can read tens
  of gigabytes while actual memory use is unchanged. Reserved-but-unused address
  space costs no memory, no swap, and nothing against overcommit. **Measure
  `RSS`, not `VIRT`** — an alert on virtual size will fire spuriously.
- **Setting `MaxVectors` sizes the reservation.** With a declared cap the
  reservation is sized from it; without one, a generic growth factor is used.
  Either way the cap is a hint for sizing, never a limit on growth.

This is automatic and has no configuration knob. It engages only once an array
passes ~32 MiB, so small collections are unaffected. It requires 64-bit Linux;
on other platforms collections fall back to copy-on-grow and behave exactly as
before.

## Key-value (this module's numbers)

In-process, `Direct` backend. These are `b.RunParallel` benchmarks at
**GOMAXPROCS=12**, so per-op cost scales with core count — on a wider machine
the same code reports a lower ns/op, and on a narrower one, higher. The thread
count is part of the number.

| Op | Result | Notes |
|---|---|---|
| Get (hit) | **~29 ns/op**, 1 alloc | RLock + index lookup into a sharded slab store. The allocation is the returned value — see the note below, it depends on the eviction policy |
| Put | **~240 ns/op** | |
| Incr (atomic RMW) | via the op registry | `store.Call("incr", …)`, serialized per shard |

Backends trade latency for durability/replication:

| Backend | Get | Put | When |
|---|---|---|---|
| `Direct` (no Raft) | ~29 ns | ~240 ns | single node, library |
| `Embedded` (Raft, no-sync) | ~222 ns | ~12.7 µs | replicated / multi-node |

The ~8× Get gap and the ~50× Put gap between the two backends are the cost of
consensus, not of storage: the `Direct` path is the same code with the log
removed.

**Whether `Get` allocates depends on the eviction policy**, and the difference is
a semantic one, not just a number:

- **`PolicyRingbufEvict`** (the default, and what the figures above use) returns a
  freshly allocated copy you own and may retain and mutate freely — **one
  allocation per hit**. Eviction can overwrite live page bytes, so handing out a
  pointer into the store would be unsafe.
- **`PolicyRejectWrites`** never overwrites live bytes, so `Get` returns a slice
  **aliasing** the page store — **zero allocations**, but you must not retain it
  across later writes to that shard.
- **`GetInto`** allocates nothing on either policy: it copies into a buffer you
  supply. Allocation-free, not copy-free.

Earlier revisions of this page and the README quoted the zero-allocation figure
without naming the policy, which read as a property of `Get` when it is a
property of one configuration.

Networked figures (TCP over loopback, ~1.7 µs Get / ~1.8 µs Put against a
`Direct` server) are **not from this machine** — its NIC exposes a single
combined queue (`ethtool -l` → `Combined: 1`), which caps any fast engine well
below its real ceiling and would measure the network adapter rather than the
storage path. They are quoted from a multi-queue host and should be treated as
the one figure on this page that is not like-for-like with the rest.

## Search on a real corpus (SIFT-1M)

The kernel numbers above are microbenchmarks. This is the whole index doing the
whole job: SIFT-1M (1M × 128d, L2), the standard ANN corpus, with its published
ground truth. `ef_search` swept, k=10, single-threaded serial latency alongside
saturated throughput:

| ef_search | recall@10 | p50 | p99 | saturated QPS |
|---|---|---|---|---|
| 16 | 0.823 | 59 µs | 98 µs | 178,834 |
| 64 | **0.968** | 173 µs | 232 µs | 60,413 |
| 128 | **0.991** | 323 µs | 432 µs | 30,883 |

Recall and latency move together, which is the whole point of the knob: 0.97
recall costs ~173 µs at the median, and buying the last 2.3 points of recall
roughly doubles it. Pick the row, not the engine.

Reproduce with the dataset in `/tmp/rostam-sift1m/sift/`:

```sh
ROSTAM_SIFT1M=1 go test ./vector/ -run 'TestSIFT1M$|TestSIFTLatencyQPS' -v -timeout 30m
```

## Reproducing

```sh
make bench   # vector kernels, cache, end-to-end — this repo's own numbers
```

`make bench` currently interleaves hashicorp/raft's DEBUG output with the
benchmark results on the `Embedded` cases, which makes them awkward to read.
Filter with `grep -E '^Benchmark.*ns/op'`.

## Vector head-to-head (VectorDBBench)

Six engines, one machine, one continuous session: VectorDBBench 1.0.22 running
Cohere-1M (768d, cosine) across 27 cases on a 12-core AMD EPYC Genoa with
22 GB RAM, one engine at a time over loopback. HNSW parameters are pinned
identically on every engine — m=16, ef_construction=200, k=100 — and
ef_search is swept. Versions: Rostam v0.1.0, Qdrant v1.18.3, Milvus v3.0.0,
PostgreSQL 17.10 + pgvector 0.8.6, Weaviate 1.31.0, Redis 7.4.7 (redis-stack).

**Re-verified.** Rostam's arm of this comparison was re-run independently on the
same hardware at a later commit, sweeping ef_search over 64/128/256/512 to
recall 0.889/0.915/0.962/0.985. Interpolated to the matched-recall levels below
it landed within ±3% of the figures in the table — 0.95 within +2.5%, 0.97
within −2.8%, in opposite directions. Two sessions months apart agreeing to that
margin is the strongest statement available about whether these numbers are
stable; the competitor arms were not re-run, so the *ratios* rest on the original
same-session measurement, which is the only way they are meaningful anyway.

![QPS versus recall for six engines on Cohere-1M: each engine's swept ef_search points form a throughput/recall curve, and Rostam's curve sits above the other five across the measured recall range](assets/bench/pareto-light.svg#only-light)
![QPS versus recall for six engines on Cohere-1M: each engine's swept ef_search points form a throughput/recall curve, and Rostam's curve sits above the other five across the measured recall range](assets/bench/pareto-dark.svg#only-dark)

### QPS at matched recall

Each engine's throughput/recall curve is interpolated to a common recall
level. Comparing at matched *recall* rather than matched *ef* is essential:
Qdrant runs 5 segments (measured) and searches each with the full ef, so at a
nominal ef=100 it performs ~5 graph searches where a single-index engine
performs one. Equal ef is not equal work. A blank cell means that recall level
is outside the engine's measured range.

| Matched recall | Rostam | vs Milvus | vs pgvector | vs Weaviate | vs Qdrant | vs Redis |
|---|---|---|---|---|---|---|
| 0.95 | 3,161 QPS | 1.84x | 1.83x | 3.25x | — | 6.54x |
| 0.97 | 2,642 QPS | 2.02x | 2.18x | 3.03x | 4.16x | 7.53x |
| 0.98 | 2,088 QPS | 1.90x | 2.28x | 2.67x | 3.98x | 7.90x |
| 0.99 | 1,471 QPS | 1.78x | — | — | 3.55x | — |

![Rostam's throughput multiple over Milvus, pgvector, Weaviate, Qdrant and Redis at matched recall levels from 0.95 to 0.99](assets/bench/matched-recall-light.svg#only-light)
![Rostam's throughput multiple over Milvus, pgvector, Weaviate, Qdrant and Redis at matched recall levels from 0.95 to 0.99](assets/bench/matched-recall-dark.svg#only-dark)

Highest recall each engine actually reached: Rostam 0.9978 @ 714 QPS; Qdrant
0.9968 @ 277; Milvus 0.9906 @ 805; pgvector 0.9897 @ 603; Weaviate 0.9829 @
757; Redis 0.9828 @ 240.

The flip side, and it matters: at matched **ef**, Rostam's recall is *lower*
than Milvus's, pgvector's and Qdrant's. Its ef is simply cheaper per unit —
Rostam at ef=600 beats Milvus at ef=300 on both recall and throughput — which
is exactly why matched-ef comparisons mislead in either direction.

### Load

Ingesting and indexing the same 1M vectors (ef=300 row), wall-clock and
engine-only CPU-seconds:

| Engine | Wall-clock | CPU-seconds |
|---|---|---|
| Rostam | 282.2 s | 1,934 |
| pgvector | 386.1 s | 2,637 |
| Milvus | 476.3 s | 2,696 |
| Qdrant | 592.2 s | 2,386 |
| Redis | 1,354.9 s | 1,356 |
| Weaviate | 1,432.7 s | 4,937 |

![Load wall-clock time and engine-only CPU-seconds for the six engines: Rostam loads in 282 seconds using the fewest multi-threaded CPU-seconds, while Redis and Weaviate take over 1,350 seconds](assets/bench/load-cpu-light.svg#only-light)
![Load wall-clock time and engine-only CPU-seconds for the six engines: Rostam loads in 282 seconds using the fewest multi-threaded CPU-seconds, while Redis and Weaviate take over 1,350 seconds](assets/bench/load-cpu-dark.svg#only-dark)

### Read the caveats with the numbers

They are what make the numbers honest:

- **Every figure is same-session.** With engine code unchanged, Milvus's own
  max QPS has drifted 42% between sessions on that box. Cross-session
  comparison is unsupported.
- **The QPS figures are floors, not ceilings.** The benchmark client shares
  the same 12 cores with the engine, and the system runs oversubscribed at
  high concurrency — which penalises the fastest engine hardest.
- **Comparators were tuned up, not down.** pgvector was given
  `maintenance_work_mem=6GB`, 11 parallel maintenance workers and a raised
  `/dev/shm` (its defaults build HNSW single-threaded in a 64MB buffer, and
  Docker's default `/dev/shm` makes a parallel build fail outright). The stock
  Weaviate VDBBench adapter was patched because it tears down a
  `BatchExecutor` shared across concurrent insert workers.

Full per-case results, configs, and methodology:
[`rostam-bench/vectordbbench`](https://github.com/rostamlabs/rostam-bench/tree/main/vectordbbench#results).

## Comparisons (in `rostam-bench`)

| Suite | Compares Rostam against | Harness |
|---|---|---|
| **Vector** | Qdrant, Milvus, pgvector, Weaviate, Redis | [VectorDBBench](https://github.com/zilliztech/VectorDBBench) plugin + Cohere-1M (summarised above) |
| **Networked KV** | Redis, Aerospike (throughput + latency over the wire) | custom Go load-gen |
| **In-memory cache** | freecache, Ristretto, BigCache, fastcache, Otter | identical-workload Go benchmark |

All comparison code, configs, and methodology notes are in
[`rostam-bench`](https://github.com/rostamlabs/rostam-bench).
