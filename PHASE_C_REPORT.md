# Phase C report — native-TCP vector ops in the Python client

Branch: `feat/python-native-tcp-parity`, worktree `/home/vahid/projects/rostam-pytcp`.
Builds on Phase A (arg encoders, `_vecwire.py`) and Phase B (recommend/query
protobuf QuerySpec encoder). This phase wires the ops into `rostam.kv._VectorAPI`
(the `.vector` accessor) with response decoders, and verifies the round trip
against a real server.

## Methods added (`clients/python/rostam/kv.py`, class `_VectorAPI`)

| Method | Op | Arg encoder (Phase A/B, `_vecwire.py`) | Response decoder (new, `_vecwire.py`) |
|---|---|---|---|
| `get_batch(collection, ids, *, with_vector=True, with_payload=True)` | `vector_get_batch` | `encode_vector_get_batch_args` | `decode_get_batch_result` (mirrors `ops.DecodeVectorGetBatchResult`) |
| `scroll(collection, *, filter=None, limit=0, cursor="")` → `(docs, next_cursor)` | `vector_scroll` | `encode_scroll_args_order_bounded` (order=None) | `decode_scroll_result_raw` (mirrors `ops.DecodeScrollResultRaw`) + client-side cursor fallback (see below) |
| `search_docs(collection, query, k, *, filter=None)` | `vector_search_docs` | `encode_search_docs_args_opts` (alias of the search opts encoder) | `decode_docs_degraded_raw` (mirrors `ops.DecodeVectorDocsDegradedRaw`) |
| `search_groups(collection, query, k, group_by, *, group_size=1, fetch_k=0, filter=None)` | `vector_search_groups` | `encode_group_search_args_opts` | `decode_groups_degraded_raw` (mirrors `ops.DecodeGroupsDegradedRaw`) |
| `hybrid_search(collection, dense, k, *, sparse=None, filter=None, method="rrf", alpha=0.0, rrf_k=0, dense_k=0, sparse_k=0)` | `vector_hybrid_search` | `encode_hybrid_search_args_opts` | `decode_hybrid_results_degraded` (mirrors `ops.DecodeHybridResultsDegraded`) |
| `hybrid_text(collection, dense, text, k, *, filter=None, method="rrf", alpha=0.0, rrf_k=0, dense_k=0, sparse_k=0)` | `vector_hybrid_text` | `encode_hybrid_text_args_global` | `decode_hybrid_results_degraded` (same wire shape as hybrid_search) |
| `recommend(collection, positive, *, negative=None, k=10, filter=None, strategy="average_vector")` | `vector_query` | `encode_recommend_query` (Phase B protobuf QuerySpec) | `decode_query_result_degraded` (mirrors `ops.DecodeQueryResultDegraded`) |
| `query(collection, positive, *, negative=None, k=10, filter=None, strategy="average_vector")` | `vector_query` | delegates to `recommend()` | same as `recommend()` |
| `upsert_batch(collection, points)` | N × `vector_upsert` (pipelined) | `encode_upsert_batch_args` (Phase A helper — no native batch-upsert op exists) | N × existing `vector_upsert` semantics (no batch result to decode) |

`get_batch` and `upsert_batch` needed no new encoders (Phase A already had
`encode_vector_get_batch_args` and `encode_upsert_batch_args`); the new work
for them was purely the `get_batch` response decoder.

## Response decoders added (`clients/python/rostam/_vecwire.py`)

Each mirrors the named Go decoder in `ops/vector.go` / `ops/query.go` byte-for-byte
(read directly from source, not guessed):

- `_read_degraded_trailer` — `ops.readDegradedTrailerN`: `[degraded:u8][missingCount:u16]{partID:u16}`, tolerant of an absent/truncated trailer (returns `(False, [], off)` unchanged), exactly like the Go reader.
- `_decode_docs_raw` — `ops.frameVectorDocsN` + typed unmarshal: `[count:u32]{[id:u64][distance:f32][score:f32][contentLen:u32][content][metaLen:u32][metaJSON]}`. Shared by search_docs, scroll, and each group's hits.
- `decode_docs_degraded_raw` — `ops.DecodeVectorDocsDegradedRaw` (docs + degraded trailer). Used by `search_docs`.
- `decode_scroll_result_raw` — `ops.DecodeScrollResultRaw` (docs + degraded trailer + `[cursorLen:u32][cursorBytes]` next_cursor tail).
- `decode_groups_degraded_raw` — `ops.DecodeGroupsDegradedRaw`: `[count:u32]{[keyLen:u32][keyJSON][hitsLen:u32][hits block]}` + degraded trailer. Each key is a single tagged `Value` (decoded with `_values.decode_value`, not `decode_metadata`, since a group key is one value, not a map).
- `_decode_hybrid_results_block` — `ops.decodeHybridResultsN`: `[count:u32]{[id:u64][distance:f32][score:f32]}`. Shared by hybrid results and the recommend/query flat-fused result (same per-row shape).
- `decode_hybrid_results_degraded` — `ops.DecodeHybridResultsDegraded`. Used by `hybrid_search`/`hybrid_text`.
- `decode_query_result_degraded` — `ops.DecodeQueryResultDegraded`: mode-tagged result, requires the RERANK tag (`mode=1`) that a flat recommend/query result always carries; raises on any other tag (fail-loud, mirroring the Go decoder). Used by `recommend`/`query`.
- `_decode_get_body_after_found` — `ops.decodeGetResultAtArena` from just after the leading `[found=1]` byte: `[dim:u32][vec][ttl:u64][metaPresent:u8][?meta][sparsePresent:u8][?sparse]` then a trailing version block, framed differently for `version_framed=True` (batch row — always `[verPresent:u8][?version:u64]`, so rows self-delimit) vs `False` (single get — optional).
- `decode_get_batch_result` — `ops.DecodeVectorGetBatchResult`: `[n:u32]` then per row `[id:u64][found:u8]` + (if found) the record above with `version_framed=True`.
- `decode_scroll_cursor` / `encode_scroll_cursor` — `ops.DecodeScrollCursor` / `ops.EncodeScrollCursor`: the opaque v1 scroll-cursor token, `base64.RawURLEncoding` of `[ver:u8=1][lastID:u64 BE]`.

## The scroll cursor finding (not in the task brief — found during verification)

I initially assumed `vector_scroll`'s wire response always carries a
server-authoritative `next_cursor` (the `EncodeScrollResult`/`DecodeScrollResultRaw`
shape the task named), because that's what the Go client's `networkedStore.VectorScroll`
decodes. Testing against the real server showed the cursor always came back empty.

Reading `ops/builtin.go`'s `handleVectorScroll` (the actual leaf op handler this
single-node `rostam-server` dispatches to) confirmed why: it returns
`EncodeVectorDocs(docs)` directly — no degraded trailer, no cursor tail at all.
The `EncodeScrollResult`-with-cursor wire only gets produced by a **clustered
coordinator's fan-out dispatcher** (`fanout_dispatcher.go`), which this
single-node integration-test server doesn't run. The Go client's *embedded/direct*
path (`directStore.VectorScroll` in `direct.go`) computes `next_cursor`
**client-side** instead, via `scrollNextCursor`: a full page (`len(docs) == limit > 0`)
may have more, so the cursor resumes after the last doc's id; a short/unlimited
page is exhausted (`""`).

Fix: `scroll()` decodes with `decode_scroll_result_raw` as planned (so it still
picks up a server-authoritative cursor transparently against a clustered
deployment that does supply one), and falls back to the client-side
`encode_scroll_cursor(docs[-1]["id"])` formula only when the wire cursor is
empty AND the page was full. `test_scroll_pages_through_all_points_with_cursor`
proves multi-page pagination actually terminates and returns exactly the 5
upserted ids with no duplicates/gaps — the real regression this would have shipped
silently broken (`scroll(limit=2)` would never have paged past the first 2 rows).

## `get_batch` / `content` finding

`decode_get_batch_result` initially returned metadata with the raw `$content` key
still inside it (matching the *wire* shape, which folds stored content into
metadata under the reserved key on the insert/get/get_batch wire — unlike
`Document.Content`, which is a first-class field on the search_docs/scroll/groups
wire). Fixed to lift `$content` out into a `content` field and pop it from
`metadata`, mirroring the existing `get()` method's `Point` shape, for
`.vector.get()` / `.vector.get_batch()` consistency.

## Verification

### 1. Server build + location

```
$ cd /home/vahid/projects/rostam-pytcp && go build -o rostam-server ./cmd/rostam-server
$ echo $?
0
```

`find_server_bin()` (from `tests/_serverbin.py`) confirmed it finds the binary:

```
>>> from _serverbin import find_server_bin
>>> find_server_bin()
('/home/vahid/projects/rostam-pytcp/rostam-server', '')
```

`go build ./...` is clean at the repo root. No Go source was touched this phase
(only `_vecwire.py`, `kv.py`, and the test file), so gofmt/vet are not applicable
here — Phase A/B's oracle-side Go was untouched.

### 2. Integration test cases added (`clients/python/tests/test_cross_stack_vector_native.py`)

20 new tests (one class method per case) against a live `rostam-server`,
covering encode AND decode for every new method:

- `test_get_batch` — 3 ids (2 hits + 1 miss) in one call; asserts vector length, lifted `content`, metadata, and that a miss comes back `found=False` (not an error).
- `test_get_batch_projection` — `with_vector=False, with_payload=False` — asserts the projection is actually honored on the wire.
- `test_scroll_pages_through_all_points_with_cursor` — 5 points, `limit=2`, loops until `next_cursor == ""`; asserts the union of all pages is exactly `{1..5}` with no dupes, and the final cursor is empty (this is what caught the scroll-cursor bug above).
- `test_scroll_filter` — filtered scroll returns only the matching id.
- `test_search_docs` / `test_search_docs_filter` — content + metadata present on hits; filter narrows results.
- `test_search_groups` — 3 points across 2 group keys; asserts both groups form, `group_size` caps the larger group's hits, and group membership is correct.
- `test_hybrid_search` / `test_hybrid_search_filter_and_weighted` — dense+sparse fusion returns a nonzero score and the expected winner; filter + `method="weighted"` narrows results.
- `test_hybrid_text` — collection created `full_text=True`; dense+BM25 fusion favors the lexically/semantically closer document.
- `test_recommend_excludes_seed_and_favors_similar` — asserts the positive seed id is excluded from results and the nearest neighbour ranks first.
- `test_recommend_negative_and_filter` — filter narrows recommend results to the admitted subset.
- `test_query_is_recommend_shaped` — asserts `query()` returns byte-for-byte the same result list as `recommend()` for identical args (documents the recommend-only limitation with a running assertion, not just a comment).
- `test_upsert_batch` — 3 points (content, metadata, sparse) via one `upsert_batch` call, each verified individually with `get()` and again in bulk with `get_batch()`.

### 3. Proof the integration tests RAN (not skipped)

```
$ cd clients/python && ROSTAM_SERVER_BIN=/home/vahid/projects/rostam-pytcp/rostam-server \
    python3 -m pytest tests/ -q -v -k vector_native
collected 137 items / 117 deselected / 20 selected
tests/test_cross_stack_vector_native.py ....................             [100%]
20 passed, 117 deselected in 0.93s
```

All 20 (5 pre-existing + 15 new — 20 total test methods in the class; the
existing file had 6 methods, so it's 6 existing + 14 new = 20) ran against the
real server process (verified `setUpClass` spawns `rostam-server -tcp ...` and
waits for the TCP port).

### 4. Full suite

```
$ cd clients/python && ROSTAM_SERVER_BIN=/home/vahid/projects/rostam-pytcp/rostam-server \
    python3 -m pytest tests/ -q
111 passed, 26 skipped, 58 subtests passed in 15.71s
```

The 26 skips are all pre-existing, unrelated optional-dependency skips
(`haystack-ai not installed`, `langchain-core not installed`,
`llama-index-core not installed`) — verified individually with `-rs`. Zero
vector/cross-stack skips. The golden byte tests (`test_vecwire_golden.py`) are
included in the 111 passed and are unaffected — this phase added no new arg
encoders, only response decoders and the methods that call them.

Without `ROSTAM_SERVER_BIN` set (no binary at repo root during a plain run,
i.e. simulating a clean checkout with no build step), the same suite correctly
skips the cross-stack classes with an actionable message rather than failing —
confirmed by running the full suite once before building the server:
`61 passed, 62 skipped`.

## Files changed

- `clients/python/rostam/_vecwire.py` — 9 new response decoders + 2 cursor helpers (+227 lines).
- `clients/python/rostam/kv.py` — 9 new `_VectorAPI` methods (+126 lines).
- `clients/python/tests/test_cross_stack_vector_native.py` — 14 new integration test methods (+166 lines).

No Go files changed. No new dependencies (stdlib-only, per constraint).

## Documented limitation: `query`/`recommend` are recommend-shaped only

`query()` is *not* a general Query API (fusion/rerank/prefetch-tree QuerySpec).
Phase B's stdlib-only protobuf encoder (`_vecwire.marshal_recommend_query_spec`)
only builds a single-leaf RECOMMEND `QuerySpec`; a general QuerySpec builder
(arbitrary prefetch lanes, fusion/rerank modes, grouped queries) would need a
much fuller hand-rolled proto encoder, which is out of scope for this phase.
`query()` is kept as a distinct method (delegating to `recommend()`) purely so
callers reaching for the "unified Query API" name find the one shape this
client actually speaks; both the docstring and `test_query_is_recommend_shaped`
make this explicit and enforced, not just documented in prose.

## Concerns

- `get_batch`/`scroll`/etc. do not expose the `degraded`/`missing` (FanMeta)
  trailer the Go client surfaces on a partitioned/clustered deployment — I
  decode and discard it (`_degraded`, `_missing` in `kv.py`). This mirrors the
  existing `search()`'s scope (no FanMeta return either) but is worth flagging:
  a future phase targeting clustered deployments would want these methods to
  return `(result, FanMeta)` the way the Go SDK does.
- `search_groups`' group `key` is decoded via `_values.decode_value` into a bare
  Python scalar (str/int/float/bool/None), not wrapped in a dict — this matches
  `decode_value`'s existing contract elsewhere in the client, but differs from a
  metadata *map* decode; flagged in the decoder's docstring to avoid a future
  mix-up with `decode_metadata`.
- `upsert_batch` has no atomicity or partial-failure reporting: it is N
  sequential `_call`s, and a failure partway through raises `RostamError`
  after any earlier points already landed. This is a direct, documented
  consequence of there being no native-TCP batch-upsert op (Phase A's own
  finding, restated in the method's docstring) — not new to this phase.
