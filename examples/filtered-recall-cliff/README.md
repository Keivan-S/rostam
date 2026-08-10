# Filtered-search: recall cliff vs. latency cliff

What a *selective metadata filter* costs a vector search — and how Rostam's
**filter-first** planner sidesteps it.

```
go run ./examples/filtered-recall-cliff
```

Three strategies on identical data, as the filter tightens:

- **naive post-filter** — retrieve the nearest by distance, then filter in app
  code. The nearest results rarely satisfy a selective filter, so **recall
  falls off a cliff**.
- **filter-aware graph** (Rostam `SearchFiltered`, graph path) — the traversal
  respects the filter, so recall holds, but it explores ever more of the graph:
  **latency explodes**.
- **filter-first** (Rostam `SearchFiltered`, planner path) — rank the matches
  pulled straight from the payload index: **exact recall and low latency**.

## Sample output (20k × 32-dim, one run; illustrative, not a fixed benchmark)

```
[1] RECALL@10
selectivity    matches  | naive post-filt  filter-aware grf  filter-first
1/2  (50.0%)   10000     | 0.960            0.976             1.000
1/10 (10.0%)   2000      | 0.865            0.999             1.000
1/50 ( 2.0%)   400       | 0.151            1.000             1.000
1/200 (0.5%)   100       | 0.033            1.000             1.000
1/1000(0.1%)   20        | 0.009            1.000             1.000

[2] LATENCY
selectivity    matches  | filter-aware grf  filter-first   speedup
1/2  (50.0%)   10000     | 316µs             2.013ms        0×
1/50 ( 2.0%)   400       | 2.862ms           40µs           72×
1/200 (0.5%)   100       | 7.93ms            10µs           828×
1/1000(0.1%)   20        | 11.974ms          2µs            6776×
```

Naive post-filtering loses recall; the filter-aware graph keeps recall but pays
up to ~38× latency to do so; filter-first gives both — and is thousands of ×
faster on the most selective filters.

## Honest caveats

- **Synthetic data**: random Gaussian vectors, not real embeddings. The *shape*
  (recall cliff for post-filtering; latency cliff for filter-aware graph;
  filter-first winning both) is the point — exact magnitudes vary with data,
  dimensionality, and parameters.
- **In-process, single box**: these are library-level latencies, not a
  networked-service benchmark.
- **"naive post-filter"** models retrieve-then-filter (filtering in application
  code, or an index whose traversal ignores the filter) — a common real
  baseline, but not what every engine does.
- Filter-first engages for equality filters that narrow to
  `Config.FilterFirstThreshold` candidates (default 10k); above that, Rostam
  falls back to the filter-aware graph path, which is why it's a planner.
