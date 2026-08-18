# Phase A report: native-TCP wire encoders for 7 vector ops

Scope: `clients/python/rostam/_vecwire.py` encoders only, byte-matched against
Go's `ops.Encode*` via the cross-stack golden oracle. No client methods
(`r.vector.*`) were touched — that's Phase B+.

## Method

For every op I read the actual Go byte-writing code in `ops/vector.go` /
`ops/text.go` / `ops/consistency.go` (never inferred from doc comments alone),
replicated field order/width/endianness in Python, added oracle cases in
`clients/python/tests/_oracle/main.go` calling the *same* `ops.Encode*`
functions the server uses, regenerated `golden.txt`, and wrote Python test
cases with identical arguments to assert byte equality.

## Per-op layouts and implementation

### 1. get_batch — `ops.EncodeVectorGetBatchArgs(collection, ids []uint64, flags uint8)`

Wire: `[colLen:u8][col][flags:u8][n:u32][id:u64 × n]`. Straightforward, no
JSON, no opts trailer (get_batch carries no read-consistency knob in the Go
API today).

Python: `encode_vector_get_batch_args(collection, ids, flags=0)`.

Oracle cases: `get_batch/empty` (0 ids), `get_batch/plain` (3 ids, flags=0),
`get_batch/withvec_payload` (2 ids incl. a >32-bit id, `flags=GetFlagsBoth=3`).

### 2. search_docs (and search) — `ops.EncodeVectorSearchArgsOpts(collection, k, query, filter, rc, opa, bound)`

Wire: base `EncodeVectorSearchArgsExt` (`[flags:u8][colLen:u8][col][k:u32]
[dim:u32][query f32×dim][?filterLen:u32][?filterJSON]`, flags bit0=filter)
plus, only when `rc!=0 || opa!=0`: OR bit1 (`vecFlagSearchOpts`) into flags[0]
and append `[rc:u8][opa:u8]` + an 8-byte BE staleness bound **only when
rc==3 (BoundedStaleness)**. rc∈{0,1,2} never carries the bound.

search_docs decodes with the *identical* function server-side — the op string
differs at call time, not the wire shape — so one Python encoder serves both:
`encode_search_args_opts(...)`, with `encode_search_docs_args_opts` as an
alias (same convention the file already uses for
`encode_exists_args = encode_delete_args`).

Oracle cases: `opts_plain` (no opts, must equal legacy `search/plain`),
`opts_filter_leader` (filter + rc=LeaderOnly + opa=1), `opts_bounded`
(rc=BoundedStaleness, bound=12345, no filter).

### 3. search_groups — `ops.EncodeGroupSearchArgsOpts(collection, k, query, opts, rc, opa, bound)`

Wire (base `EncodeGroupSearchArgs`): `[colLen:u8][col][k:u32][groupSize:u32]
[fetchK:u32][groupByLen:u16][groupBy][dim:u32][query][filterLen:u32]
[filterJSON]`. **Important divergence from search/hybrid**: the filter block
here has NO flag bit — Go unconditionally writes `[filterLen:u32][filterJSON]`
with length 0 when there's no filter. Group args carry no flags byte at all,
so the opts trailer is self-delimiting: `[optsPresent:u8=1][rc][opa]`
(+bound iff rc==3), appended only when rc!=0 or opa!=0.

Python: `encode_group_search_args_opts(collection, k, query, opts, *,
read_consistency=0, on_partition_unavailable=0, bound=0)` where `opts` is a
dict `{group_by, group_size, fetch_k, filter}`.

Oracle cases: `group/plain`, `group/filter_opts` (filter + LeaderOnly),
`group/bounded` (BoundedStaleness + bound=777).

### 4. hybrid_search — `ops.EncodeHybridSearchArgsOpts(collection, dense, k, sparse, opts, rc, opa, bound)`

Wire (base `EncodeHybridSearchArgs`): `[flags:u8][colLen:u8][col][k:u32]
[method:u8][alpha:f64][rrfK:u32][denseK:u32][sparseK:u32][dim:u32]
[dense f32×dim][?sparse: nnz:u32{idx:u32,val:f32}][?filterLen:u32][filterJSON]`.
flags bit0=filter, bit1=sparse present, bit2=opts trailer present (only when
rc!=0||opa!=0), then `[rc][opa]`+bound-iff-3.

Python: `encode_hybrid_search_args_opts(collection, dense, k, sparse, opts, *,
read_consistency=0, on_partition_unavailable=0, bound=0)`, `sparse` a
`{indices, values}` dict, `opts` a `{filter, method, alpha, rrf_k, dense_k,
sparse_k}` dict. `method` maps `"rrf"→0, "weighted"→1, "dbsf"→2`
(`vector.FusionMethod`).

Oracle cases: `hybrid/plain`, `hybrid/sparse` (weighted fusion + sparse
lane), `hybrid/filter_opts` (filter + LeaderOnly), `hybrid/bounded` (dbsf +
sparse + BoundedStaleness).

### 5. hybrid_text — `ops.EncodeHybridTextArgsGlobal(collection, dense, text, k, opts, rc, opa, bound, globalIDF, g)`

Wire: `[flags:u8][colLen:u8][col][k:u32][method:u8][alpha:f64][rrfK:u32]
[denseK:u32][sparseK:u32][dim:u32][dense][queryLen:u32][queryUTF8]
[?filterLen:u32][filterJSON][?rc][?opa][?bound][?globalStatsBlock]`.
flags: bit0=filter, bit1=opts-trailer-present, bit2=REQUEST globalIDF
(independent of whether `g` is supplied — it's a client *request* flag),
bit3=PHASE-1 global-stats block present (only set when `g != nil`, forcing
`flags[0]` to be patched a second time after the block is appended, exactly
mirroring `ops.appendGlobalStatsBlock` + the final `buf[0] = flags` in Go).

Per the task, `g` is always `None` in Phase A (it's a coordinator-only
phase-1 optimization), but I implemented the block anyway since the spec was
fully readable from `ops/text.go` (`appendGlobalStatsBlock` /
`globalStatsBlockLen`) — cheap to get right, no guessing involved — and left
it available via the `g` kwarg for when it's needed.

Python: `encode_hybrid_text_args_global(collection, dense, query, k, opts, *,
read_consistency=0, on_partition_unavailable=0, bound=0, global_idf=False,
g=None)`.

Oracle cases: `hybrid_text/plain`, `hybrid_text/filter_opts_globalidf`
(filter + weighted fusion + LeaderOnly + `global_idf=True`, `g=nil`),
`hybrid_text/bounded` (empty query string + BoundedStaleness).

### 6. scroll — `ops.EncodeScrollArgsOrderBounded(collection, filter, limit, rc, opa, afterID, hasAfter, order, bound)`

This one has a real trap I caught only by reading both
`EncodeScrollArgsCursorBounded` (the `order==nil` path) and
`EncodeScrollArgsOrderBounded` (`order!=nil`) side by side:

- **`order == nil`**: delegates to `EncodeScrollArgsCursorBounded`, which
  returns the bare base block **unchanged** when `rc==0 && opa==0 &&
  !hasAfter` — no opts byte, no cursor byte, nothing. When any of those is
  set, it appends `[1][rc][opa]`+bound, and **only when `hasAfter`** appends
  `[1][afterID:u64]` — if `!hasAfter` there is **no `cursorPresent=0` byte at
  all**, the trailer just ends after the opts block.
- **`order != nil`**: the trailer is **forced present unconditionally**
  (regardless of rc/opa/hasAfter) and **always** writes an explicit
  `cursorPresent` byte (`1`+afterID, or `0`) before the order block, because
  the order block needs an unambiguous, self-delimiting start offset.

I replicated this asymmetry exactly in `encode_scroll_args_order_bounded`
(see the docstring in `_vecwire.py`) rather than unifying the two branches,
since unifying them would silently change the wire for the `order==nil,
!hasAfter` no-opts case.

The order block itself (`ops.appendScrollOrderBlock`) is:
```
[orderPresent:u8=1][keyLen:u32][key]
[flags:u8]  bit0=desc bit1=isDatetime bit2=Kind==string bit3=multiKey(len(Tail)>0)
[startPresent:u8][start:f64 iff present]
[resumePresent:u8][resumeKey:f64 iff present]
iff Kind==string: [resumeStrPresent:u8][strLen:u32][str] iff present
iff multiKey:
  [numTail:u8]
  per tail key: [keyLen:u32][key][flags:u8 bit0 desc bit1 datetime bit2 string]
  [resumeTuplePresent:u8]
    iff present, per key (primary + tail, in order): [kind:u8][value]
      kind==string(2): [strLen:u32][str]   else: [float64 BE]
```
Note `Kind` only ever gates the wire shape when it's `OrderString` — a
`OrderDatetime` key shares the exact `float64` resume path as `OrderNumeric`
and is distinguished purely by the separate `IsDatetime` flag bit, not by
`Kind`.

Python: `encode_scroll_args_order_bounded(collection, limit, *, filter=None,
read_consistency=0, on_partition_unavailable=0, after_id=None, order=None,
bound=0)`, `order` a dict shaped like the block above (`kind` ∈
`"numeric"/"datetime"/"string"`).

Oracle cases: `scroll/plain`, `scroll/filter`, `scroll/cursor` (`hasAfter`,
no opts — exercises the "cursor present but rc/opa absent" sub-case),
`scroll/bounded_opts` (BoundedStaleness, no cursor — exercises the "no
cursorPresent byte at all" sub-case), `scroll/order_numeric` (desc + start),
`scroll/order_datetime_resume` (IsDatetime + resume + cursor),
`scroll/order_string_resume` (string kind + resume-str tail),
`scroll/order_multikey` (2 tail keys, mixed kinds incl. a datetime tail key
whose resume value still rides the float64 path in the tuple, LeaderOnly +
cursor + full resume tuple — the densest case, exercises every branch at
once).

### 7. upsert_batch — no dedicated wire op; findings

**There is no native-TCP batch-upsert `ops.Encode*` function.**
`EncodeVectorUpsertArgs` is strictly single-point. I grepped for `Encode.*Batch`
across `ops/*.go` and the only batch-shaped *vector* wire op is
`EncodeVectorGetBatchArgs` (read-only). The only "bulk" ingestion path is
`/points/bulk` and `/points/bulk/build`, which are **HTTP-only**: an "RVB1"
binary framing defined in `httpapi/binary_bulk.go`, staged for the
multi-core HNSW build — a completely different protocol from the
`ops.Encode*` native-TCP family and not reachable from a TCP client op at
all.

So per the task's own fallback instruction, `upsert_batch` over native TCP is
just **N pipelined `vector_upsert` ops** — no batch framing exists to
replicate. I added `encode_upsert_batch_args(collection, points) -> List[bytes]`
that returns the list of per-point `encode_upsert_args()` outputs for the
caller to pipeline over one connection, with a docstring explaining why. This
isn't golden-tested (there's no Go reference to diff against); instead
`VecwireUpsertBatchTest.test_batch_equals_sequential_singles` asserts it's
byte-for-byte identical to calling `encode_upsert_args` N times directly.

## Verification

- `go build ./...` — clean.
- `go vet ./clients/python/tests/_oracle/` — clean.
- Regenerated golden: `go run ./clients/python/tests/_oracle > clients/python/tests/_oracle/golden.txt` (50 lines: 25 pre-existing + 25 new).
- `python3 -m pytest tests/test_vecwire_golden.py -q` (run from `clients/python/`, since the package isn't installed in this env): **6 passed, 46 subtests passed** — every new oracle case matches byte-for-byte.
- Full client suite for a regression check: `python3 -m pytest tests/ -q` from `clients/python/`: **61 passed, 62 skipped** (skips are the optional langchain/llamaindex/haystack adapter tests, expected — those deps aren't installed here), 46 subtests passed.

## Files changed

- `clients/python/rostam/_vecwire.py` — 7 new encoders + `_bound_tail` helper +
  flag/order-kind/fusion-method constant tables + `encode_upsert_batch_args`.
- `clients/python/tests/_oracle/main.go` — 25 new `emit(...)` oracle cases.
- `clients/python/tests/_oracle/golden.txt` — regenerated (50 lines total).
- `clients/python/tests/test_vecwire_golden.py` — 25 new golden cases in
  `_DATAPLANE` + a new `VecwireUpsertBatchTest` class.

## Concerns / follow-ups for Phase B

- `_TENANT_FILTER`'s dict key order (`op, field, value`) must keep matching
  Go's struct field declaration order for any *new* byte-exact filter test
  case to pass — this is inherent to comparing JSON byte-for-byte and already
  true of the pre-existing filter-adjacent code; flagging so Phase B doesn't
  assume filter JSON is order-independent everywhere (it's only
  order-independent where the existing code deliberately falls back to a
  round-trip/decode comparison instead of a byte comparison, as
  `test_metadata_prefix_matches_and_json_equivalent` already does for
  metadata).
- `hybrid_text`'s `g` (`BM25GlobalStats`) support is implemented and
  oracle-untested beyond `g=None` — if Phase B ever needs the two-phase
  global-DF path, it should add a byte-exact oracle case for `g != nil`
  before relying on it.
- No client methods (`r.vector.get_batch`, `.scroll`, `.search_docs`,
  `.search_groups`, `.hybrid_search`, `.hybrid_text`, `.upsert_batch`) exist
  yet — these are pure wire encoders. Phase B/C will need result *decoders*
  too (`EncodeVectorGetBatchResult`, `EncodeGroups`, `EncodeHybridResults*`,
  scroll's page/cursor response, etc.) which this phase did not touch.
