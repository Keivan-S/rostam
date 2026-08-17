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
    meta_obj = dict(encode_metadata(metadata or {})) if metadata else {}
    if content is not None:
        meta_obj = {"$content": {"kind": "string", "str": content}, **meta_obj}
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
        out += struct.pack(">I", len(idx))
        for i, v in zip(idx, val):
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
