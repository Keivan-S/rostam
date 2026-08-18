# Phase B — native-TCP `recommend`/`query` protobuf encoder (Python)

Hand-rolled the `pb.QuerySpec` protobuf wire format in stdlib Python for the
`vector_query` op, byte-matched against Go's `proto.Marshal` via the golden oracle.
Encoder + golden verification only; client methods are NOT wired (Phase C).

## Proto shape (recommend QuerySpec)

Source: `ops/query.go` (`querySpecToProto`, `queryLeafToProto` LeafRecommend arm),
`grpcapi/pb/rostam.pb.go`, and the client spec shape in
`rostam-ntvc/client/vector_recommend.go` (`Recommend`).

The Go client builds, for a recommend request:

```
QuerySpec{ Mode: ModeFusion, K: req.K,
           Prefetch: [ LeafSource( QueryLeaf{Kind:LeafRecommend, Positive, Negative,
                                             Strategy, K, Filter} ) ] }
```

`querySpecToProto` maps that to the proto tree:

```
QuerySpec
  (mode      = QUERY_MODE_FUSION = 0  → OMITTED, proto3 default)
  prefetch[0] = QueryLeaf { recommend = RecommendLeaf {
                  positive, negative, k, filter_json, strategy } }
  fusion_method = "rrf"   (queryFusionToString(FusionRRF); non-empty → ALWAYS emitted)
  k             = req.K
  (root omitted — hasLeafPayload(zero root) == false for FUSION)
```

### Exact field numbers / wire types used

| Message | Field | # | wire | notes |
|---|---|---|---|---|
| QuerySpec | prefetch | 2 | 2 (LEN) | repeated QueryLeaf (messages, not packed) |
| QuerySpec | mode | 3 | 0 (varint enum) | FUSION=0 ⇒ omitted; RERANK=1 would emit |
| QuerySpec | fusion_method | 4 | 2 (LEN str) | "rrf" always present |
| QuerySpec | k | 7 | 0 (varint) | |
| QueryLeaf | recommend | 6 | 2 (LEN msg) | set oneof arm — always emitted |
| RecommendLeaf | positive | 1 | 2 (LEN) | `varint,rep,packed` |
| RecommendLeaf | negative | 2 | 2 (LEN) | `varint,rep,packed`, omit-empty |
| RecommendLeaf | k | 3 | 0 (varint) | |
| RecommendLeaf | filter_json | 4 | 2 (LEN bytes) | Go json.Marshal(vector.Filter) |
| RecommendLeaf | strategy | 5 | 0 (varint enum) | AVERAGE=0 omit; BEST=1 emit |

Enum confirmations: `QueryMode_QUERY_MODE_FUSION = 0`, `_RERANK = 1`;
`RecommendStrategy_RECOMMEND_AVERAGE_VECTOR = 0`, `_BEST_SCORE = 1`.
Filter JSON byte-shape: `{"op":"eq","field":"tenant","value":{"kind":"string","str":"acme"}}`
(struct-field order op/field/value, ValueKind marshals as a string).

Op frame (`EncodeQueryArgs`): `[colLen:u8][col][specLen:u32][specBytes][optsTrailer]`.
optsTrailer = `appendReadOptsTrailerBounded` (ops/consistency.go): omitted when
rc==0 && opa==0; else `[marker][rc][opa]`, with marker = `readOptsTrailerMarker(1)`,
plus `readOptsStalenessBit(2)` (⇒ marker byte 3) and an 8-byte BE bound only for
BOUNDED_STALENESS (rc==3).

Decoded example — `queryspec/recommend_pos` (positive=[1,2,3], k=10):
`12 09 [32 07 [0a 03 010203][18 0a]] [22 03 727266][38 0a]` =
prefetch{ recommend{ positive=1,2,3; k=10 } } fusion_method="rrf" k=10.

## Determinism check (STEP 3, done FIRST)

Ran the Go oracle 5× and diffed all runs — byte-identical every time. `proto.Marshal`
for these QuerySpec messages is stable (no map fields; protobuf-go emits in
field-number order). NOT BLOCKED — hand-rolled byte-matching is valid.

## Oracle cases added (`clients/python/tests/_oracle/main.go`)

Added `mustSpec` + `recommendSpec` helpers (builds the spec the way the Go client's
`Recommend` does) and, for each spec, emits BOTH `queryspec/*` (raw
`MarshalEngineQuerySpec` bytes) and `query/*` (`EncodeQueryArgs` frame):

- `recommend_pos` — positive=[1,2,3], k=10
- `recommend_pos_neg` — positive=[1,2], negative=[9], k=5
- `recommend_filter` — positive=[1], tenant filter, k=5
- `recommend_best_score` — positive=[1,2], strategy=BEST_SCORE, k=5
- `query/recommend_bounded` — op frame with BOUNDED_STALENESS trailer (bound=555)

9 new golden lines; golden.txt regenerated (existing 50 lines byte-unchanged).

## Golden test output

`cd clients/python && python3 -m pytest tests/test_vecwire_golden.py -q`
→ `6 passed, 55 subtests passed`. Full python suite: `61 passed, 62 skipped`
(skips are pre-existing cross-stack tests needing a live server). gofmt/go vet clean.

## Files changed

- `clients/python/rostam/_vecwire.py` — protobuf primitives (`_pb_uvarint`,
  `_pb_key`, `_pb_len_delim`, `_pb_varint_field`, `_pb_string_field`,
  `_pb_bytes_field`, `_pb_packed_varints`), `_marshal_recommend_leaf`,
  `marshal_recommend_query_spec`, `encode_query_args`, `encode_recommend_query`.
- `clients/python/tests/_oracle/main.go` — recommend oracle cases + helpers.
- `clients/python/tests/_oracle/golden.txt` — 9 new fixtures.
- `clients/python/tests/test_vecwire_golden.py` — 9 new byte-equality cases.

## Concerns

- Only the recommend shape of QuerySpec is encoded (matches Phase B scope: the
  client's `Recommend`). A general `query`/dense/discover/nested-spec encoder is
  not built — recommend is the sole ModeFusion-leaf case the task covered. A raw
  `Query(spec)` from Python would need dense/sparse/named/mv/discover leaf arms too.
- filter_json byte-matching depends on Go's json.Marshal(vector.Filter) key order;
  the golden pins it. If the Filter/Value struct field order changes, regenerate.
- Client method wiring is intentionally deferred to Phase C.
