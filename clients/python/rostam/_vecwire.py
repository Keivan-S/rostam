"""Binary encoders for the vector ops carried on the native TCP protocol.

These mirror Rostam's Go `ops.Encode*Args` byte layouts exactly. The fixed
binary parts (config trailer, ids, dims, vectors, sparse) are asserted
byte-for-byte against the Go reference in tests/test_vecwire_golden.py; the
embedded JSON parts (metadata, filter, content) are validated by round-tripping
through a real server, since the server unmarshals them and any equivalent JSON
decodes the same.
"""

from __future__ import annotations

import json
import struct
from typing import Any, Dict, List, Optional, Sequence

from ._values import decode_metadata, encode_metadata

Vector = Sequence[float]

# --- enum maps (must match vector/index.go, vector/quant.go) -----------------
_METRIC = {"cosine": 0, "l2": 1, "dot": 2, "dotproduct": 2, "ip": 2}
_QUANT = {"": 0, "none": 0, "sq8": 1, "bq1": 2, "pq": 3, "sq": 4, "prq": 5}
_INDEX = {"": 0, "hnsw": 0, "ivf": 1, "vamana": 2, "gpu": 3}

# insert/search flag bits (vector.go)
_F_TTL = 1 << 0
_F_META = 1 << 1
_F_SPARSE = 1 << 2
_F_FILTER = 1 << 0  # search flags use bit0 for filter
_F_SEARCH_OPTS = 1 << 1  # search flags: consistency opts trailer present (vecFlagSearchOpts)

# hybrid_search flag bits (vector.go: hybridFlagFilter/hybridFlagSparse/hybridFlagOpts)
_HYBRID_F_FILTER = 1 << 0
_HYBRID_F_SPARSE = 1 << 1
_HYBRID_F_OPTS = 1 << 2

# hybrid_text flag bits (text.go: textFlagFilter/textFlagOpts/textFlagGlobalIDF/textFlagGlobalStats)
_TEXT_F_FILTER = 1 << 0
_TEXT_F_OPTS = 1 << 1
_TEXT_F_GLOBAL_IDF = 1 << 2
_TEXT_F_GLOBAL_STATS = 1 << 3

# fusion method (vector/fusion.go: FusionMethod)
_FUSION_METHOD = {"rrf": 0, "weighted": 1, "dbsf": 2}

# order_by kind (vector/order.go: OrderKind) — only "string" changes the wire shape;
# numeric/datetime share the float64 path and are distinguished by the is_datetime bit.
_ORDER_KIND = {"numeric": 0, "datetime": 1, "string": 2}

# read-consistency levels (ops/consistency.go)
CONSISTENCY_ANY_REPLICA = 0
CONSISTENCY_LEADER_ONLY = 1
CONSISTENCY_LINEARIZABLE = 2
CONSISTENCY_BOUNDED_STALENESS = 3


def _col(collection: str) -> bytes:
    c = collection.encode("utf-8")
    if len(c) > 0xFF:
        raise ValueError("collection name too long")
    return bytes([len(c)]) + c


def _f32be(vec: Vector) -> bytes:
    out = bytearray(struct.pack(">I", len(vec)))
    for f in vec:
        out += struct.pack(">f", f)
    return bytes(out)


def _meta_json(metadata: Optional[Dict[str, Any]]) -> bytes:
    # encode_metadata builds the tagged form; the Go Value marshals its struct
    # fields in declaration order (kind first), so the inner dicts must NOT be
    # key-sorted. The server unmarshals this, so exact byte order is not required
    # for correctness — this only aims to look like the Go output.
    tagged = encode_metadata(metadata or {})
    return json.dumps(tagged, separators=(",", ":")).encode("utf-8")


def encode_search_args(collection: str, k: int, query: Vector,
                       filter: Optional[Dict[str, Any]] = None) -> bytes:
    flags = _F_FILTER if filter else 0
    out = bytearray([flags])
    out += _col(collection)
    out += struct.pack(">I", k)
    out += _f32be(query)
    if filter:
        fj = json.dumps(filter, separators=(",", ":")).encode("utf-8")
        out += struct.pack(">I", len(fj)) + fj
    return bytes(out)


def _bound_tail(read_consistency: int, bound: int) -> bytes:
    """Mirrors ops.appendBoundTail: the 8-byte BE staleness bound rides ONLY when
    read_consistency == CONSISTENCY_BOUNDED_STALENESS (3); every other level is
    byte-identical to the pre-bounded-staleness wire (no tail bytes at all)."""
    if read_consistency != CONSISTENCY_BOUNDED_STALENESS:
        return b""
    return struct.pack(">Q", bound)


def encode_search_args_opts(collection: str, k: int, query: Vector,
                            filter: Optional[Dict[str, Any]] = None, *,
                            read_consistency: int = 0, on_partition_unavailable: int = 0,
                            bound: int = 0) -> bytes:
    """Mirrors ops.EncodeVectorSearchArgsOpts. Shared by vector_search AND
    vector_search_docs — the two ops carry an IDENTICAL wire layout and differ only
    in the op string used at call time (search_docs additionally returns each hit's
    stored content, which is a server-side concern, not a wire-shape one)."""
    base = encode_search_args(collection, k, query, filter)
    if read_consistency == 0 and on_partition_unavailable == 0:
        return base  # byte-identical to the legacy/no-opts form
    out = bytearray(base)
    out[0] |= _F_SEARCH_OPTS
    out += bytes([read_consistency, on_partition_unavailable])
    out += _bound_tail(read_consistency, bound)
    return bytes(out)


# search_docs shares vector_search's wire encoder exactly (ops.EncodeVectorSearchArgsOpts
# backs both vector_search and vector_search_docs); only the op name differs at call time.
encode_search_docs_args_opts = encode_search_args_opts


def encode_vector_get_batch_args(collection: str, ids: Sequence[int], flags: int = 0) -> bytes:
    """Mirrors ops.EncodeVectorGetBatchArgs. Wire:
    [colLen:u8][col][flags:u8][n:u32][id:u64 x n]."""
    out = bytearray(_col(collection))
    out += bytes([flags & 0xFF])
    out += struct.pack(">I", len(ids))
    for i in ids:
        out += struct.pack(">Q", i)
    return bytes(out)


def _scroll_base(collection: str, limit: int, filter: Optional[Dict[str, Any]]) -> bytes:
    fj = b""
    if filter:
        fj = json.dumps(filter, separators=(",", ":")).encode("utf-8")
    out = bytearray(_col(collection))
    out += struct.pack(">I", limit)
    out += struct.pack(">I", len(fj)) + fj
    return bytes(out)


def _encode_scroll_order_block(order: Dict[str, Any]) -> bytes:
    """Mirrors ops.appendScrollOrderBlock. `order` shape:
    {key, desc=False, is_datetime=False, kind="numeric"|"datetime"|"string",
     has_start=False, start_from=0.0, has_resume=False, resume_key=0.0,
     has_resume_str=False, resume_str="",
     tail=[{key, desc=False, is_datetime=False, kind="numeric"}, ...],
     has_resume_keys=False, resume_keys=[{kind, num=0.0, str=""}, ...]}
    kind only changes wire shape for "string" (bit2 + the resume-str tail); the
    string-resume tail is written ONLY when kind=="string" (additive, matches Go).
    """
    key = order["key"].encode("utf-8")
    desc = bool(order.get("desc", False))
    is_datetime = bool(order.get("is_datetime", False))
    kind = order.get("kind", "numeric")
    tail = order.get("tail") or []

    flags = 0
    if desc:
        flags |= 1 << 0
    if is_datetime:
        flags |= 1 << 1
    if kind == "string":
        flags |= 1 << 2
    if tail:
        flags |= 1 << 3  # scrollOrderFlagMultiKey

    out = bytearray([1])  # orderPresent=1
    out += struct.pack(">I", len(key)) + key
    out += bytes([flags])

    if order.get("has_start"):
        out += bytes([1]) + struct.pack(">d", float(order["start_from"]))
    else:
        out += bytes([0])

    if order.get("has_resume"):
        out += bytes([1]) + struct.pack(">d", float(order["resume_key"]))
    else:
        out += bytes([0])

    if kind == "string":
        if order.get("has_resume_str"):
            rs = order["resume_str"].encode("utf-8")
            out += bytes([1]) + struct.pack(">I", len(rs)) + rs
        else:
            out += bytes([0])

    if tail:
        out += bytes([len(tail)])
        for tk in tail:
            tkey = tk["key"].encode("utf-8")
            out += struct.pack(">I", len(tkey)) + tkey
            tf = 0
            if tk.get("desc"):
                tf |= 1 << 0
            if tk.get("is_datetime"):
                tf |= 1 << 1
            if tk.get("kind") == "string":
                tf |= 1 << 2
            out += bytes([tf])
        if order.get("has_resume_keys"):
            out += bytes([1])
            for rv in order.get("resume_keys", []):
                rk = _ORDER_KIND[rv["kind"]]
                out += bytes([rk])
                if rv["kind"] == "string":
                    s = rv["str"].encode("utf-8")
                    out += struct.pack(">I", len(s)) + s
                else:
                    out += struct.pack(">d", float(rv["num"]))
        else:
            out += bytes([0])

    return bytes(out)


def encode_scroll_args_order_bounded(collection: str, limit: int, *,
                                     filter: Optional[Dict[str, Any]] = None,
                                     read_consistency: int = 0, on_partition_unavailable: int = 0,
                                     after_id: Optional[int] = None,
                                     order: Optional[Dict[str, Any]] = None,
                                     bound: int = 0) -> bytes:
    """Mirrors ops.EncodeScrollArgsOrderBounded. NOTE the asymmetry this replicates
    byte-for-byte: when order is None the opts+cursor trailer (and the cursor's
    cursorPresent=0 byte) is omitted ENTIRELY unless opts or a cursor are actually
    in use (EncodeScrollArgsCursorBounded); when order is given, the trailer AND an
    explicit cursorPresent byte (0 or 1) are always forced present, so the order
    block has an unambiguous, self-delimiting start position."""
    base = _scroll_base(collection, limit, filter)
    has_after = after_id is not None

    if order is None:
        if read_consistency == 0 and on_partition_unavailable == 0 and not has_after:
            return base  # byte-identical to the legacy form
        out = bytearray(base)
        out += bytes([1, read_consistency, on_partition_unavailable])
        out += _bound_tail(read_consistency, bound)
        if has_after:
            out += bytes([1]) + struct.pack(">Q", after_id)
        # else: no cursorPresent byte at all (matches EncodeScrollArgsCursorBounded)
        return bytes(out)

    out = bytearray(base)
    out += bytes([1, read_consistency, on_partition_unavailable])
    out += _bound_tail(read_consistency, bound)
    if has_after:
        out += bytes([1]) + struct.pack(">Q", after_id)
    else:
        out += bytes([0])  # cursorPresent=0, forced present
    out += _encode_scroll_order_block(order)
    return bytes(out)


def _encode_sparse(sparse: Optional[Dict[str, Sequence]]) -> bytes:
    """Mirrors ops.writeSparse: [nnz:u32]{[dim:u32][value:f32]}."""
    idx = list(sparse["indices"]) if sparse else []
    val = list(sparse["values"]) if sparse else []
    if len(idx) != len(val):
        raise ValueError("sparse indices and values must have the same length")
    out = bytearray(struct.pack(">I", len(idx)))
    for i, v in zip(idx, val, strict=True):
        out += struct.pack(">I", i) + struct.pack(">f", v)
    return bytes(out)


def encode_group_search_args_opts(collection: str, k: int, query: Vector,
                                  opts: Optional[Dict[str, Any]] = None, *,
                                  read_consistency: int = 0, on_partition_unavailable: int = 0,
                                  bound: int = 0) -> bytes:
    """Mirrors ops.EncodeGroupSearchArgsOpts. `opts`: {group_by, group_size=0,
    fetch_k=0, filter=None}. Unlike search/hybrid, the filter block here is
    UNCONDITIONAL (no flag bit) — Go always writes [filterLen:u32][filterJSON],
    with filterJSON empty (len 0) when there is no filter."""
    opts = opts or {}
    group_by = str(opts.get("group_by", "")).encode("utf-8")
    if len(group_by) > 0xFFFF:
        raise ValueError("group_by too long")
    filt = opts.get("filter")
    fj = json.dumps(filt, separators=(",", ":")).encode("utf-8") if filt else b""

    out = bytearray(_col(collection))
    out += struct.pack(">I", k)
    out += struct.pack(">I", int(opts.get("group_size", 0)))
    out += struct.pack(">I", int(opts.get("fetch_k", 0)))
    out += struct.pack(">H", len(group_by)) + group_by
    out += _f32be(query)
    out += struct.pack(">I", len(fj)) + fj

    if read_consistency == 0 and on_partition_unavailable == 0:
        return bytes(out)
    out += bytes([1, read_consistency, on_partition_unavailable])
    out += _bound_tail(read_consistency, bound)
    return bytes(out)


def encode_hybrid_search_args_opts(collection: str, dense: Vector, k: int,
                                   sparse: Optional[Dict[str, Sequence]] = None,
                                   opts: Optional[Dict[str, Any]] = None, *,
                                   read_consistency: int = 0, on_partition_unavailable: int = 0,
                                   bound: int = 0) -> bytes:
    """Mirrors ops.EncodeHybridSearchArgsOpts. `opts`: {filter=None, method="rrf",
    alpha=0.0, rrf_k=0, dense_k=0, sparse_k=0}."""
    opts = opts or {}
    filt = opts.get("filter")
    flags = 0
    fj = b""
    if filt:
        flags |= _HYBRID_F_FILTER
        fj = json.dumps(filt, separators=(",", ":")).encode("utf-8")
    has_sparse = bool(sparse and sparse.get("indices"))
    if has_sparse:
        flags |= _HYBRID_F_SPARSE

    out = bytearray([flags])
    out += _col(collection)
    out += struct.pack(">I", k)
    out += bytes([_FUSION_METHOD[opts.get("method", "rrf")]])
    out += struct.pack(">d", float(opts.get("alpha", 0.0)))
    out += struct.pack(">I", int(opts.get("rrf_k", 0)))
    out += struct.pack(">I", int(opts.get("dense_k", 0)))
    out += struct.pack(">I", int(opts.get("sparse_k", 0)))
    out += _f32be(dense)
    if has_sparse:
        out += _encode_sparse(sparse)
    if flags & _HYBRID_F_FILTER:
        out += struct.pack(">I", len(fj)) + fj

    if read_consistency == 0 and on_partition_unavailable == 0:
        return bytes(out)
    out[0] |= _HYBRID_F_OPTS
    out += bytes([read_consistency, on_partition_unavailable])
    out += _bound_tail(read_consistency, bound)
    return bytes(out)


def encode_hybrid_text_args_global(collection: str, dense: Vector, query: str, k: int,
                                   opts: Optional[Dict[str, Any]] = None, *,
                                   read_consistency: int = 0, on_partition_unavailable: int = 0,
                                   bound: int = 0, global_idf: bool = False,
                                   g: Optional[Dict[str, Any]] = None) -> bytes:
    """Mirrors ops.EncodeHybridTextArgsGlobal. `opts`: {filter=None, method="rrf",
    alpha=0.0, rrf_k=0, dense_k=0, sparse_k=0}. `g` (normally None — the
    coordinator-only phase-1 global-DF stats block): {n=0, avgdl=0.0, df={term:freq}}."""
    opts = opts or {}
    filt = opts.get("filter")
    flags = 0
    fj = b""
    if filt:
        flags |= _TEXT_F_FILTER
        fj = json.dumps(filt, separators=(",", ":")).encode("utf-8")
    has_opts = read_consistency != 0 or on_partition_unavailable != 0
    if has_opts:
        flags |= _TEXT_F_OPTS
    if global_idf:
        flags |= _TEXT_F_GLOBAL_IDF

    out = bytearray([flags])
    out += _col(collection)
    out += struct.pack(">I", k)
    out += bytes([_FUSION_METHOD[opts.get("method", "rrf")]])
    out += struct.pack(">d", float(opts.get("alpha", 0.0)))
    out += struct.pack(">I", int(opts.get("rrf_k", 0)))
    out += struct.pack(">I", int(opts.get("dense_k", 0)))
    out += struct.pack(">I", int(opts.get("sparse_k", 0)))
    out += _f32be(dense)
    qb = query.encode("utf-8")
    out += struct.pack(">I", len(qb)) + qb
    if flags & _TEXT_F_FILTER:
        out += struct.pack(">I", len(fj)) + fj
    if has_opts:
        out += bytes([read_consistency, on_partition_unavailable])
        out += _bound_tail(read_consistency, bound)

    if g is not None:
        flags |= _TEXT_F_GLOBAL_STATS
        out[0] = flags
        out += struct.pack(">q", int(g.get("n", 0)))
        out += struct.pack(">f", float(g.get("avgdl", 0.0)))
        df = g.get("df", {}) or {}
        out += struct.pack(">I", len(df))
        for term in sorted(df.keys()):
            out += struct.pack(">I", term) + struct.pack(">I", df[term])

    return bytes(out)


def _encode_insert_like(op_flag_prefix: bool, collection: str, id: int, vec: Vector, *,
                        content: Optional[str], ttl_ms: int,
                        metadata: Optional[Dict[str, Any]],
                        sparse: Optional[Dict[str, Sequence]]) -> bytes:
    """Shared body for insert/upsert. content!=None ⇒ upsert layout (JSON carries
    $content merged with metadata); content is None ⇒ insert layout."""
    flags = 0
    if ttl_ms > 0:
        flags |= _F_TTL
    # upsert folds content into the metadata JSON under $content
    if metadata and "$content" in metadata:
        raise ValueError("metadata key '$content' is reserved")
    meta_obj = dict(encode_metadata(metadata or {})) if metadata else {}
    if content is not None:
        meta_obj["$content"] = {"kind": "string", "str": content}
    has_meta = bool(meta_obj)
    if has_meta:
        flags |= _F_META
    if sparse:
        flags |= _F_SPARSE

    out = bytearray([flags])
    out += _col(collection)
    out += struct.pack(">Q", id)
    out += _f32be(vec)
    if flags & _F_TTL:
        out += struct.pack(">Q", ttl_ms)
    if has_meta:
        mj = json.dumps(meta_obj, separators=(",", ":")).encode("utf-8")
        out += struct.pack(">I", len(mj)) + mj
    if sparse:
        idx = list(sparse["indices"])
        val = list(sparse["values"])
        if len(idx) != len(val):
            raise ValueError("sparse indices and values must have the same length")
        out += struct.pack(">I", len(idx))
        for i, v in zip(idx, val, strict=True):
            out += struct.pack(">I", i) + struct.pack(">f", v)
    return bytes(out)


def encode_insert_args(collection: str, id: int, vec: Vector, *, ttl_ms: int = 0,
                       metadata: Optional[Dict[str, Any]] = None,
                       sparse: Optional[Dict[str, Sequence]] = None) -> bytes:
    return _encode_insert_like(False, collection, id, vec, content=None,
                               ttl_ms=ttl_ms, metadata=metadata, sparse=sparse)


def encode_upsert_args(collection: str, id: int, vec: Vector, *, content: str = "",
                       ttl_ms: int = 0, metadata: Optional[Dict[str, Any]] = None,
                       sparse: Optional[Dict[str, Sequence]] = None) -> bytes:
    return _encode_insert_like(True, collection, id, vec, content=content,
                               ttl_ms=ttl_ms, metadata=metadata, sparse=sparse)


def encode_upsert_batch_args(collection: str, points: Sequence[Dict[str, Any]]) -> List[bytes]:
    """There is NO single native-TCP batch-upsert wire op. ops.EncodeVectorUpsertArgs
    is single-point only; the batch/bulk framing that does exist (RVB1, in
    httpapi/binary_bulk.go for POST /points/bulk and /points/bulk/build) is an
    HTTP-only staging protocol for the multi-core index build, not part of the
    ops.Encode* native-TCP family and not something a TCP client op can drive.
    So a native-TCP upsert_batch is just N pipelined vector_upsert ops: this
    returns the list of per-point encode_upsert_args() outputs for the caller to
    send back-to-back (pipelined) over the same connection.
    Each point dict: {id, vector, content="", ttl_ms=0, metadata=None, sparse=None}.
    """
    out = []
    for p in points:
        out.append(encode_upsert_args(
            collection, p["id"], p["vector"],
            content=p.get("content", ""),
            ttl_ms=p.get("ttl_ms", 0),
            metadata=p.get("metadata"),
            sparse=p.get("sparse"),
        ))
    return out


def encode_delete_args(collection: str, id: int) -> bytes:
    return _col(collection) + struct.pack(">Q", id)


encode_exists_args = encode_delete_args  # same layout


def encode_get_args(collection: str, id: int, flags: int = 0) -> bytes:
    return _col(collection) + struct.pack(">Q", id) + bytes([flags & 0xFF])


# ---- create_collection: the config trailer ---------------------------------
#
# Mirrors ops.EncodeCreateCollectionArgs. The trailer blocks are appended only
# when non-default, but each late block FORCES every earlier optional block
# present (with zero values) so the decoder's greedy length guards have fixed
# anchors. The forcing chain is replicated exactly; verified byte-for-byte
# against the Go encoder in the golden test.
def encode_create_collection_args(name: str, cfg: Dict[str, Any]) -> bytes:
    g = cfg.get

    dim = int(g("dim"))
    metric = _METRIC[str(g("metric", "cosine")).lower()]
    m = int(g("m", 0))
    efc = int(g("ef_construction", 0))
    efs = int(g("ef_search", 0))
    seed = int(g("seed", 0))
    quant = _QUANT[str(g("quant", "")).lower()]
    persistent = 1 if g("persistent") else 0
    rescore = int(g("rescore_factor", 0))
    extend = 1 if g("extend_candidates") else 0
    extend_max = int(g("extend_candidates_max", 0))
    l0full = 1 if g("level0_full_degree") else 0
    qbuild = 1 if g("quantized_build") else 0
    partitions = int(g("partitions", 0))
    index_type = _INDEX[str(g("index_type", "")).lower()]

    ivf_nlist = int(g("ivf_nlist", 0))
    ivf_nprobe = int(g("ivf_nprobe", 0))
    ivf_pq = bool(g("ivf_pq"))
    ivf_pq_m = int(g("ivf_pq_m", 0))
    ivf_rerank = bool(g("ivf_rerank"))
    quant_pq_m = int(g("quant_pq_m", 0))
    opq = bool(g("opq"))
    pq_drop_vecs = bool(g("pq_drop_vecs"))
    ivf_train_threshold = int(g("ivf_train_threshold", 0))
    drift_retrain = bool(g("ivf_drift_retrain"))
    drift_growth = float(g("ivf_drift_growth_factor", 0.0))
    drift_factor = float(g("ivf_drift_factor", 0.0))
    rel_bp = int(g("filter_first_relative_bp", 0))
    opq_iters = int(g("opq_iters", 0))
    full_text = g("full_text")  # None, True/analyzer dict, or falsy
    sq_bits = int(g("sq_bits", 0))
    prq_layers = int(g("prq_layers", 0))
    vamana_r = int(g("vamana_r", 0))
    vamana_l = int(g("vamana_l", 0))
    vamana_alpha = float(g("vamana_alpha", 0.0))
    anisotropic_eta = float(g("anisotropic_eta", 0.0))
    soar = bool(g("soar"))
    soar_lambda = float(g("soar_lambda", 0.0))
    pq_nbits = int(g("pq_nbits", 0)) == 4

    # Forcing chain (bottom-up), matching the Go booleans exactly.
    ft_present = full_text is not None and full_text is not False
    _soar_lambda = soar_lambda != 0 or pq_nbits
    _soar = soar or _soar_lambda
    _aniso = anisotropic_eta != 0 or _soar
    _vamana = vamana_r != 0 or vamana_l != 0 or vamana_alpha != 0 or _aniso
    _prq = prq_layers != 0 or _vamana
    _sq = sq_bits != 0 or _prq
    _ft_slot = ft_present or _sq
    _opq_iters = opq_iters != 0 or _ft_slot
    _rel_bp = rel_bp != 0 or _opq_iters
    _drift = drift_retrain or drift_growth != 0 or drift_factor != 0 or _rel_bp
    _threshold = ivf_train_threshold != 0 or _drift
    _ivfpq = ivf_pq or ivf_rerank or _threshold
    _ivf = index_type != 0 or ivf_nlist != 0 or ivf_nprobe != 0 or _ivfpq
    _quantpqm = quant_pq_m != 0 or _threshold

    nm = name.encode("utf-8")
    out = bytearray([len(nm)]) + nm
    out += struct.pack(">I", dim)
    out += bytes([metric])
    out += struct.pack(">I", m) + struct.pack(">I", efc) + struct.pack(">I", efs)
    out += struct.pack(">q", seed)
    out += bytes([quant, persistent])
    out += struct.pack(">I", rescore)
    out += bytes([extend])
    out += struct.pack(">I", extend_max)
    out += bytes([l0full, qbuild])
    out += struct.pack(">I", partitions)

    if _ivf:
        out += bytes([index_type]) + struct.pack(">I", ivf_nlist) + struct.pack(">I", ivf_nprobe)
    if _ivfpq:
        out += bytes([1 if ivf_pq else 0]) + struct.pack(">I", ivf_pq_m) + bytes([1 if ivf_rerank else 0])
    if _quantpqm:
        out += struct.pack(">I", quant_pq_m)
    if opq or pq_drop_vecs or _threshold:
        out += bytes([1 if opq else 0])
    if pq_drop_vecs or _threshold:
        out += bytes([1 if pq_drop_vecs else 0])
    if _threshold:
        out += struct.pack(">I", ivf_train_threshold)
    if _drift:
        out += bytes([1 if drift_retrain else 0]) + struct.pack(">d", drift_growth) + struct.pack(">d", drift_factor)
    if _rel_bp:
        out += struct.pack(">I", rel_bp)
    if _opq_iters:
        out += struct.pack(">I", opq_iters)
    if _ft_slot:
        # presence byte, then analyzer/k1/b only when a real FullText config
        if ft_present:
            an = (full_text.get("analyzer", "") if isinstance(full_text, dict) else "").encode("utf-8")
            k1 = float(full_text.get("k1", 0.0)) if isinstance(full_text, dict) else 0.0
            b = float(full_text.get("b", 0.0)) if isinstance(full_text, dict) else 0.0
            out += bytes([1, len(an)]) + an + struct.pack(">f", k1) + struct.pack(">f", b)
        else:
            out += bytes([0])
    if _sq:
        out += struct.pack(">I", sq_bits)
    if _prq:
        out += struct.pack(">I", prq_layers)
    # VamanaL / VamanaAlpha force VamanaR, so when any is set all three words are
    # written; each of Aniso / SOAR / SOARLambda / PQNBits is a SEPARATE word
    # appended only under its own flag (f32, not f64).
    if _vamana:
        out += struct.pack(">I", vamana_r) + struct.pack(">I", vamana_l) + struct.pack(">f", vamana_alpha)
    if _aniso:
        out += struct.pack(">f", anisotropic_eta)
    if _soar:
        out += bytes([1 if soar else 0])
    if _soar_lambda:
        out += struct.pack(">f", soar_lambda)
    if pq_nbits:
        out += struct.pack(">I", int(g("pq_nbits", 0)))
    return bytes(out)


# ---- result decoders --------------------------------------------------------
def decode_search_results(body: bytes) -> List[Dict[str, Any]]:
    (count,) = struct.unpack(">I", body[:4])
    out = []
    off = 4
    for _ in range(count):
        rid = struct.unpack(">Q", body[off:off + 8])[0]
        dist = struct.unpack(">f", body[off + 8:off + 12])[0]
        out.append({"id": rid, "distance": dist})
        off += 12
    return out


def decode_exists_result(body: bytes) -> bool:
    return bool(body and body[0])


def decode_get_result(body: bytes) -> Optional[Dict[str, Any]]:
    """Decode a vector_get result. Returns None if the point is absent, else a
    dict with id-independent fields: vector, metadata (tagged→plain), ttl_ms,
    sparse. Fields the request did not ask for come back empty."""
    if not body or body[0] == 0:
        return None
    off = 1
    (dim,) = struct.unpack(">I", body[off:off + 4]); off += 4
    vec = None
    if dim:
        vec = [struct.unpack(">f", body[off + 4 * i:off + 4 * i + 4])[0] for i in range(dim)]
        off += 4 * dim
    (ttl_ms,) = struct.unpack(">Q", body[off:off + 8]); off += 8
    meta = None
    if body[off]:
        off += 1
        (mlen,) = struct.unpack(">I", body[off:off + 4]); off += 4
        meta = decode_metadata(json.loads(body[off:off + mlen])); off += mlen
    else:
        off += 1
    sparse = None
    if body[off]:
        off += 1
        (nnz,) = struct.unpack(">I", body[off:off + 4]); off += 4
        idx, val = [], []
        for _ in range(nnz):
            idx.append(struct.unpack(">I", body[off:off + 4])[0])
            val.append(struct.unpack(">f", body[off + 4:off + 8])[0])
            off += 8
        sparse = {"indices": idx, "values": val}
    return {"vector": vec, "metadata": meta or {}, "ttl_ms": ttl_ms, "sparse": sparse}
