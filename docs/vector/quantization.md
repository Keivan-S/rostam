# Quantization

Quantization compresses vectors so more of the index fits in RAM (or so the
resident set shrinks to just the codes). Rostam always pairs lossy codes with an
**exact float32 rescore stage**, so the quality cost shows up as slightly wider
candidate generation, not silently wrong top-k ranking.

## Modes

Set `Config.Quant`:

| Mode | Compression | How it scores | Notes |
|---|---|---|---|
| `QuantNone` | 1× | exact float32 | default |
| `QuantSQ8` | 4× | scalar int8 codes, AVX2 int8 kernels | Cosine metric; measured recall@10 ≈ 0.98 of exact |
| `QuantBQ1` | 32× | binary codes, Hamming distance | measured recall@10 ≈ 0.96 of exact on a clustered corpus (rescored) |
| `QuantSQ` | 4× (8-bit) | trained scalar quantizer | metric-agnostic; `SQBits` |
| `QuantPQ` | dim/`QuantPQM` bytes/vec | product quantization with LUTs | `PQNBits`: 8 (default) or 4 (LUT16, smaller/lossier candidates) |
| `QuantPRQ` | layered | product-residual quantization | `PRQLayers` (default 2); code = layers × `QuantPQM` bytes |

Advanced PQ options: `OPQ`/`OPQIters` (learned rotation before PQ),
`AnisotropicEta` (score-aware loss), `SOAR`/`SOARLambda` (spilled orthogonal
assignments), `PQDropVecs` (drop raw vectors after encoding, codes-only
memory).

## Rescoring

`RescoreFactor` (default 3) controls the exact stage: the index over-fetches
`RescoreFactor × k` candidates using codes, then re-scores them with exact
float32 distances before returning top-k. Raise it if quantized recall matters
more than latency.

## Where the bytes live

`QuantStorage` picks residency:

- `QuantInRAM` (default) — codes and raw vectors both on heap.
- `QuantMmap` — codes stay in RAM; raw float32 vectors live in a memory-mapped
  file (`MmapPath`), so only the codes are resident and the OS pages vectors in
  for rescoring. This is the "10× less RAM, same top-k" configuration.

`QuantizedBuild` additionally navigates and selects neighbors on codes during
bulk builds — a memory-bandwidth optimization for large builds; the rescore
stage recovers ranking quality.

Under either storage mode, the codes array (and the raw vectors, and the graph)
grows in place on a reserved address range once it passes ~32 MiB, so a growing
collection does not stall queries. That inflates reported *virtual* size without
changing resident size — see
[Growing a collection doesn't stall queries](../performance.md#growing-a-collection-doesnt-stall-queries).

## Choosing a mode

- **Default choice**: `QuantSQ8` — 4× smaller, ~2 % recall cost that mostly
  disappears after rescore.
- **RAM-bound at large scale**: `QuantBQ1` (32×) or `QuantPQ` with
  `QuantStorage: QuantMmap`.
- **Non-cosine metrics**: `QuantSQ` (trained scalar) — SQ8 is Cosine-scoped.
- **Maximum compression with tolerable quality**: `QuantPRQ` layered codes,
  tuned via `PRQLayers` and `QuantPQM`.

Persistent collections (`Persistent: true`) require a quantization mode — the
durable format stores codes; see [Persistence](persistence.md).
