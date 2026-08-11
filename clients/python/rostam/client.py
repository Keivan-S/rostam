"""A dependency-free Python client for Rostam's REST API.

Uses only the standard library (``http.client``), so it installs and runs
anywhere with no transitive dependencies. Metadata and filter values are
converted to and from Rostam's tagged wire form automatically — callers work in
native Python.

    from rostam import RostamClient
    from rostam import filters as f

    c = RostamClient("http://localhost:8080", api_key="secret")
    c.create_collection("docs", dim=384, metric="cosine")
    c.upsert("docs", 1, vector, content="hello", metadata={"doc_id": 7})
    hits = c.search_docs("docs", query, k=5, filter=f.eq("doc_id", 7))
"""

from __future__ import annotations

import array
import http.client
import json
import struct
import sys
import threading
import urllib.parse
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Union

from ._values import decode_metadata, decode_value, encode_metadata

Vector = Sequence[float]

# Reserved payload key holding a record's document content (server-side rag.go
# contentField). get_batch lifts it into Point.content.
_RESERVED_CONTENT = "$content"


def _seg(s: Union[str, int]) -> str:
    """Percent-encode a single URL path segment (encodes /, ?, #, space, etc.)."""
    return urllib.parse.quote(str(s), safe="")


class RostamError(Exception):
    """An error returned by the Rostam server or transport.

    ``status`` is the HTTP status code (0 for a transport-level failure).
    """

    def __init__(self, message: str, status: int = 0):
        super().__init__(message)
        self.status = status
        self.message = message


@dataclass
class SearchResult:
    id: int
    distance: float
    score: float = 0.0


@dataclass
class Document:
    id: int
    distance: float
    content: str = ""
    score: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class Point:
    """A point fetched by id (get_batch): vector + content + user metadata.

    Content is lifted out of the reserved payload field ($content) into
    .content; .metadata holds only user keys (content stripped)."""
    id: int
    vector: List[float] = field(default_factory=list)
    content: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class Group:
    key: Any
    hits: List[Document] = field(default_factory=list)


@dataclass
class MultiResult:
    id: int
    score: float
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ScrollPage:
    """One page of a scroll() listing over a collection's id-ascending order.

    Iterates and ``len()``s like the underlying document list, so list-style
    callers keep working, while exposing ``next_cursor`` for pagination. When
    ``next_cursor`` is non-empty, pass it back as ``scroll(cursor=...)`` to
    fetch the next page; an empty ``next_cursor`` means the listing is
    exhausted.
    """
    documents: List[Document] = field(default_factory=list)
    next_cursor: str = ""

    def __iter__(self):
        return iter(self.documents)

    def __len__(self) -> int:
        return len(self.documents)

    def __getitem__(self, i):
        return self.documents[i]

    def __bool__(self) -> bool:
        return bool(self.documents)


def _sparse(sparse: Optional[Dict[str, Sequence]]) -> Dict[str, Any]:
    if not sparse:
        return {}
    return {"indices": list(sparse["indices"]), "values": list(sparse["values"])}


# ---- binary bulk framing ("RVB1") ----
#
# The JSON body is the ingest bottleneck for a large initial load: the server has
# to parse `dim` base-10 float literals per point, which dominates the actual
# index build. This dense framing ships the same points as raw f32 instead. It is
# stdlib-only (struct + array) and the server selects it purely by Content-Type,
# so nothing about the JSON API changes.
#
#   magic  b"RVB1"
#   flags  u32   bit0 payloads present, bit1 upsert
#   count  u32
#   dim    u32
#   rows   count x [ id u64 ][ dim x f32 ]
#   pays   count x [ len u32 ][ len bytes of JSON ]   (only when bit0)
#
# All big-endian, matching Rostam's op wire — a staged row is byte-identical to
# the server's internal staging row, so the server reads the body straight into
# the op with no per-float conversion.
_BULK_MAGIC = b"RVB1"
_BULK_FLAG_PAYLOADS = 1 << 0
_BULK_FLAG_UPSERT = 1 << 1


def _encode_bulk_body(
    ids: Sequence[int],
    vectors: Sequence[Vector],
    *,
    flags: int = 0,
    payloads: Optional[Sequence[Optional[Dict[str, Any]]]] = None,
) -> bytearray:
    """Pack (ids, vectors[, payloads]) into an RVB1 binary bulk body.

    Returns the bytearray itself rather than ``bytes(out)``: the copy would
    momentarily double the body, which is the opposite of the point in a function
    whose whole job is to make a large load cheap. http.client accepts any
    bytes-like object as a request body.
    """
    n = len(ids)
    if len(vectors) != n:
        raise ValueError(f"ids/vectors length mismatch: {n} vs {len(vectors)}")
    dim = len(vectors[0]) if n else 0
    if payloads is not None:
        if len(payloads) != n:
            raise ValueError(f"ids/payloads length mismatch: {n} vs {len(payloads)}")
        flags |= _BULK_FLAG_PAYLOADS
    out = bytearray(struct.pack(">4sIII", _BULK_MAGIC, flags, n, dim))
    swap = sys.byteorder == "little"
    for i in range(n):
        row = array.array("f", vectors[i])
        if len(row) != dim:
            raise ValueError(f"vector {i} has dim {len(row)}, expected {dim}")
        out += struct.pack(">Q", ids[i])
        if swap:
            row.byteswap()
        out += row.tobytes()
    if payloads is not None:
        for meta in payloads:
            if not meta:
                out += b"\x00\x00\x00\x00"
                continue
            blob = json.dumps(encode_metadata(meta)).encode("utf-8")
            out += struct.pack(">I", len(blob))
            out += blob
    return out


_RVQ1_MAGIC = b"RVQ1"
_RVQ1_FLAG_FILTER = 1 << 0
# The server refuses a declared dim above this, so a longer vector goes as JSON
# rather than as a request the server is certain to reject.
_RVQ1_MAX_DIM = 1 << 16


def _encode_rvq1(query: Vector, k: int, filter: Optional[Dict[str, Any]]) -> bytearray:
    """Encode a search request in the binary query framing.

    Same shape as _encode_bulk on the ingest side, and big-endian for the same
    reason: it lands in the server byte-identical to the op wire, so neither end
    swaps per float. ``array.byteswap`` does the whole vector in one C call —
    which is what makes this 0.011 ms where json.dumps of the same 768 floats is
    0.258 ms.

    read_consistency, on_partition_unavailable and max_staleness are written as
    their defaults: this client has never exposed them, and the framing carries
    them so that adding them later needs no second wire format.
    """
    vec = array.array("f", query)
    blob = b""
    flags = 0
    if filter:
        flags |= _RVQ1_FLAG_FILTER
        blob = json.dumps(filter).encode("utf-8")
    out = bytearray(struct.pack(">4sIIIBBHQ", _RVQ1_MAGIC, flags, k, len(vec), 0, 0, 0, 0))
    if sys.byteorder == "little":
        vec.byteswap()
    out += vec.tobytes()
    if blob:
        out += struct.pack(">I", len(blob))
        out += blob
    return out


# The server caps a single binary bulk body at 256 MiB and a single request at
# 262,144 points. bulk_stage/batch_upsert therefore SPLIT a large load into
# requests instead of sending one giant body — otherwise the advertised "load a
# million vectors" use case would just 413 (at dim=768 one request tops out
# around 87k points). The target below leaves generous headroom under both caps.
_BULK_TARGET_BYTES = 64 << 20  # 64 MiB per request
_BULK_MAX_POINTS = 1 << 17     # 131,072 points per request (server ceiling is 2x)


def _points_per_request(dim: int, payload_bytes: int = 0) -> int:
    """How many points fit in one request at this dim, under the server's caps.

    payload_bytes is the per-point payload size to budget for. Ignoring it was a
    bug: batch_upsert with metadata sends 4 + len(json) extra bytes per point, so
    a chunk sized purely on the vector row overshot the server's 256 MiB cap and
    413'd (131,072 points x ~2 KB of metadata is 268 MB of payload alone, before
    a single vector).
    """
    row = 8 + max(dim, 1) * 4 + 4 + max(payload_bytes, 0)
    return max(1, min(_BULK_MAX_POINTS, _BULK_TARGET_BYTES // row))


def _payload_bytes(payloads: Optional[Sequence[Optional[Dict[str, Any]]]]) -> int:
    """Estimate the per-point encoded payload size from a sample of the batch.

    Sampling rather than encoding everything: this runs to decide a chunk size,
    and encoding the whole batch twice would cost more than it saves. The sample
    is scaled up so a batch with uneven payloads still lands under the cap.
    """
    if not payloads:
        return 0
    sample = [p for p in payloads[:64] if p]
    if not sample:
        return 0
    total = sum(len(json.dumps(encode_metadata(p)).encode("utf-8")) for p in sample)
    return (total // len(sample)) * 2  # 2x headroom for uneven payloads


def _chunks(n: int, vectors: Sequence[Vector], payloads=None):
    """Yield (lo, hi) request-sized spans over n points."""
    if n == 0:
        yield 0, 0
        return
    step = _points_per_request(len(vectors[0]), _payload_bytes(payloads))
    for lo in range(0, n, step):
        yield lo, min(lo + step, n)


class _ConnectionPool:
    """Keep-alive connections to one host, safe to share between threads.

    The client used to call ``urllib.request.urlopen`` per request, which opens
    and tears down a TCP connection every time. Reusing one costs a lock and a
    list; measured against a local server it was 1.49x the throughput on
    repeated searches, and the gap widens with network latency because a fresh
    connection pays a round-trip before the request is even sent.

    ``urlopen`` was accidentally thread-safe — nothing was shared. A kept-alive
    connection is not: two threads writing requests into one socket interleave
    and desynchronize the response stream. So connections live here, one thread
    holds one connection for the length of a request, and a connection is only
    returned to the pool once its response has been fully read.
    """

    def __init__(self, base_url: str, maxsize: int = 8):
        parts = urllib.parse.urlsplit(base_url)
        if parts.scheme not in ("http", "https"):
            raise ValueError(f"base_url must be http:// or https://, got {base_url!r}")
        self._https = parts.scheme == "https"
        self._host = parts.hostname or "localhost"
        self._port = parts.port or (443 if self._https else 80)
        self._maxsize = maxsize
        self._idle: List[Any] = []
        self._lock = threading.Lock()

    def _new(self, timeout: float):
        cls = http.client.HTTPSConnection if self._https else http.client.HTTPConnection
        return cls(self._host, self._port, timeout=timeout)

    def acquire(self, timeout: float):
        """Return (connection, reused), with `timeout` applied to it.

        The timeout has to be re-applied on every acquire, not just at connect.
        http.client fixes the socket timeout when the connection is made, so a
        pooled connection carries whatever timeout its first caller happened to
        want — and bulk_build asks for 24 hours. Inheriting an earlier 30-second
        connection would abort a long build at 30 seconds, which urllib never
        did because it applied the timeout per request.
        """
        with self._lock:
            if self._idle:
                conn = self._idle.pop()
                conn.timeout = timeout
                if conn.sock is not None:
                    conn.sock.settimeout(timeout)
                return conn, True
        return self._new(timeout), False

    def release(self, conn) -> None:
        with self._lock:
            if len(self._idle) < self._maxsize:
                self._idle.append(conn)
                return
        conn.close()

    def discard(self, conn) -> None:
        try:
            conn.close()
        except Exception:
            pass

    def close(self) -> None:
        with self._lock:
            idle, self._idle = self._idle, []
        for c in idle:
            self.discard(c)


class RostamClient:
    """REST client for a Rostam HTTP server (see ``rostam.NewHTTPServer``)."""

    def __init__(
        self,
        base_url: str,
        api_key: Optional[str] = None,
        timeout: float = 30.0,
        *,
        binary_search: bool = True,
        pool_maxsize: int = 8,
    ):
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self.timeout = timeout
        # Whether to send search queries in the binary framing. Turning it off
        # forces the JSON body; see _search_body for what the framing buys and
        # for how an older server is detected and fallen back to automatically.
        self.binary_search = binary_search
        self._binary_search_supported = True
        self._pool = _ConnectionPool(self.base_url, maxsize=pool_maxsize)
        self._path_prefix = urllib.parse.urlsplit(self.base_url).path.rstrip("/")

    def close(self) -> None:
        """Close pooled connections. The client stays usable; it reconnects."""
        self._pool.close()

    def __enter__(self) -> "RostamClient":
        return self

    def __exit__(self, *exc: Any) -> None:
        self.close()

    # ---- transport ----

    def _request(
        self, method: str, path: str, body: Optional[dict] = None, *,
        idempotent: Optional[bool] = None,
    ) -> Any:
        data = None if body is None else json.dumps(body).encode("utf-8")
        if idempotent is None:
            idempotent = method in ("GET", "HEAD")
        return self._send(method, path, data, "application/json", idempotent=idempotent)

    def _send(
        self,
        method: str,
        path: str,
        data: Optional[Union[bytes, bytearray]],
        content_type: str,
        timeout: Optional[float] = None,
        *,
        idempotent: bool = False,
    ) -> Any:
        headers = {"Content-Type": content_type}
        if self.api_key:
            headers["Authorization"] = "Bearer " + self.api_key
        deadline = timeout or self.timeout
        url = self._path_prefix + path

        # One retry, and only for a request that is safe to send twice on a
        # connection taken from the pool.
        #
        # A server is free to close an idle keep-alive connection at any moment,
        # so a pooled connection can be dead before we use it. But the failure
        # does not reliably arrive while writing: RemoteDisconnected and
        # BadStatusLine are raised by getresponse(), by which point the request
        # bytes are on the wire and the server may already have executed them.
        # Retrying then would replay the write — a second /points insert that
        # fails on a duplicate id, or a double batch. So the caller says whether
        # the request can be repeated; reads say yes, writes say nothing and get
        # an error they can act on.
        attempts = 2 if idempotent else 1
        while True:
            attempts -= 1
            conn, reused = self._pool.acquire(deadline)
            try:
                conn.request(method, url, body=data, headers=headers)
                resp = conn.getresponse()
                raw = resp.read()
                status = resp.status
            except (http.client.RemoteDisconnected, http.client.BadStatusLine,
                    ConnectionResetError, BrokenPipeError) as e:
                self._pool.discard(conn)
                if reused and attempts > 0:
                    continue
                raise RostamError(f"transport error: {e}", status=0) from None
            except (OSError, http.client.HTTPException) as e:
                self._pool.discard(conn)
                raise RostamError(f"transport error: {e}", status=0) from None

            # Only a connection the server intends to keep goes back in the pool.
            # A response that closes it (HTTP/1.0, or an explicit Connection:
            # close) leaves a dead socket that the next caller would spend a
            # failed write and a retry to discover — turning pooling into a
            # per-request penalty against exactly the servers that opted out.
            if resp.will_close:
                self._pool.discard(conn)
            else:
                self._pool.release(conn)
            if status >= 400:
                msg = f"HTTP {status}"
                try:
                    msg = json.loads(raw).get("error", msg)
                except Exception:
                    pass
                raise RostamError(msg, status=status)
            if not raw:
                return None
            return json.loads(raw)

    # ---- search encoding ----

    def _search(
        self, path: str, query: Vector, k: int, filter: Optional[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """POST a search, in the binary framing when the server understands it.

        The JSON body spends its time turning float32s into decimal for the
        server to parse straight back. At dim=768, k=10, that encode is 0.258 ms
        of a 0.845 ms request — 31% — against 0.011 ms to write the same vector
        as bytes, and the server's matching decode disappears with it.
        """
        # A k the framing cannot express (negative, or past u32) goes down the
        # JSON path, where the server answers 400 exactly as it always has.
        # Encoding it here would raise struct.error instead — a different
        # exception type for the same misuse, decided by which encoding the
        # client happened to pick.
        encodable = 0 <= k <= 0xFFFFFFFF and len(query) <= _RVQ1_MAX_DIM
        if self.binary_search and self._binary_search_supported and encodable:
            try:
                res = self._send(
                    "POST", path, _encode_rvq1(query, k, filter),
                    "application/octet-stream", idempotent=True,
                )
                return res or {}
            except RostamError as e:
                # A server without RVQ1 support routes the body to its JSON
                # decoder, which chokes on byte one and says so. That specific
                # message is the signal to stop offering binary for the life of
                # this client and use JSON — anything else is a real error about
                # this request (a bad k, a bad filter) and must surface as one.
                if not (e.status == 400 and "invalid JSON body" in str(e)):
                    raise
                self._binary_search_supported = False

        body: Dict[str, Any] = {"query": list(query), "k": k}
        if filter:
            body["filter"] = filter
        return self._request("POST", path, body, idempotent=True) or {}

    # ---- collections ----

    def health(self) -> bool:
        """Return True if the server is reachable and healthy."""
        return (self._request("GET", "/v1/health") or {}).get("status") == "ok"

    def create_collection(
        self,
        name: str,
        dim: int,
        *,
        metric: str = "cosine",
        m: int = 0,
        ef_construction: int = 0,
        ef_search: int = 0,
        seed: int = 0,
        quant: str = "",
        sq_bits: int = 0,
        prq_layers: int = 0,
        index_type: str = "",
        vamana_r: int = 0,
        vamana_l: int = 0,
        vamana_alpha: float = 0.0,
        anisotropic_eta: float = 0.0,
        soar: bool = False,
        soar_lambda: float = 0.0,
        pq_nbits: int = 0,
        persistent: bool = False,
        rescore_factor: int = 0,
        full_text: Any = None,
    ) -> None:
        """Create a collection. metric: "cosine"|"l2"|"dot"; quant:
        ""|"sq8"|"bq1"|"pq"|"sq"|"prq".

        quant="sq" is the trained metric-agnostic scalar quantizer; sq_bits picks
        its bit-depth (4, 6, or 8; 0 = server default 8). quant="prq" is
        product-residual quantization; prq_layers is the residual layer count
        (0 = server default 2). Both numeric knobs are sent only when non-zero,
        so a non-SQ/PRQ create stays byte-compatible with the prior request.

        index_type selects the backing index: ""/"hnsw" (default), "ivf", or
        "vamana" (the DiskANN single-layer graph). For "vamana", vamana_r is the
        max out-degree (0 = server default 64), vamana_l the build/search beam
        width (0 = default 100), vamana_alpha the pass-2 RobustPrune α (0 = default
        1.2). index_type and the vamana knobs are sent only when set, so a non-
        Vamana create stays byte-compatible with the prior request.

        anisotropic_eta is the ScaNN score-aware PQ weight (η ≥ 1; 0/1 = isotropic,
        byte-compatible default). soar opts an "ivf" index into ScaNN-style multi-
        assignment (higher recall at fixed nprobe); soar_lambda tunes the SOAR
        orthogonality weight λ (0 = server default 1.5). pq_nbits is the per-subspace
        PQ code width (0/8 = 8-bit default, 4 = 4-bit LUT16 fast-scan). Each is sent
        only when set, so a non-ScaNN create stays byte-compatible with the prior
        request.

        full_text enables the server-side BM25 full-text lane (so search_text /
        hybrid_text work): pass True for the default English analyzer, or a dict
        like {"analyzer": "english", "k1": 1.2, "b": 0.75} to tune the BM25 knobs.
        None (default) leaves full-text disabled."""
        cfg = {
            "dim": dim,
            "metric": metric,
            "m": m,
            "ef_construction": ef_construction,
            "ef_search": ef_search,
            "seed": seed,
            "quant": quant,
            "persistent": persistent,
            "rescore_factor": rescore_factor,
        }
        if sq_bits:
            cfg["sq_bits"] = sq_bits
        if prq_layers:
            cfg["prq_layers"] = prq_layers
        if index_type:
            cfg["index_type"] = index_type
        if vamana_r:
            cfg["vamana_r"] = vamana_r
        if vamana_l:
            cfg["vamana_l"] = vamana_l
        if vamana_alpha:
            cfg["vamana_alpha"] = vamana_alpha
        if anisotropic_eta:
            cfg["anisotropic_eta"] = anisotropic_eta
        if soar:
            cfg["soar"] = soar
        if soar_lambda:
            cfg["soar_lambda"] = soar_lambda
        if pq_nbits:
            cfg["pq_nbits"] = pq_nbits
        if full_text is True:
            cfg["full_text"] = {}
        elif isinstance(full_text, dict):
            cfg["full_text"] = full_text
        self._request("POST", "/v1/collections", {"name": name, "config": cfg})

    def drop_collection(self, name: str) -> None:
        """Delete a collection and its data."""
        self._request("DELETE", "/v1/collections/" + _seg(name))

    # ---- points ----

    def upsert(
        self,
        collection: str,
        id: int,
        vector: Vector,
        *,
        content: str = "",
        metadata: Optional[Dict[str, Any]] = None,
        ttl_ms: int = 0,
        sparse: Optional[Dict[str, Sequence]] = None,
    ) -> None:
        """Insert or replace a point (the RAG write path; stores content)."""
        self._put_point(collection, id, vector, content, metadata, ttl_ms, sparse, upsert=True)

    def insert(
        self,
        collection: str,
        id: int,
        vector: Vector,
        *,
        content: str = "",
        metadata: Optional[Dict[str, Any]] = None,
        ttl_ms: int = 0,
        sparse: Optional[Dict[str, Sequence]] = None,
    ) -> None:
        """Insert a point, rejecting a duplicate id (use upsert to replace)."""
        self._put_point(collection, id, vector, content, metadata, ttl_ms, sparse, upsert=False)

    def _put_point(self, collection, id, vector, content, metadata, ttl_ms, sparse, upsert):
        body = {
            "id": id,
            "vector": list(vector),
            "content": content,
            "ttl_ms": ttl_ms,
            "metadata": encode_metadata(metadata),
            "upsert": upsert,
        }
        sp = _sparse(sparse)
        if sp:
            body["sparse"] = sp
        self._request("POST", f"/v1/collections/{_seg(collection)}/points", body)

    # ---- bulk load (binary wire) ----

    def bulk_stage(
        self,
        collection: str,
        ids: Sequence[int],
        vectors: Sequence[Vector],
        *,
        metadatas: Optional[Sequence[Optional[Dict[str, Any]]]] = None,
        timeout: Optional[float] = None,
    ) -> int:
        """Stage points for a concurrent bulk build, over the binary wire.

        The initial-load fast path: staging is cheap and parallel, and the
        multi-core index build happens once in :meth:`bulk_build`. The collection
        must be empty.

        ``metadatas`` is optional and, when given, must have one entry per id
        (``None`` for a point with no payload). The payloads are applied by the
        build itself, so a load whose points need metadata to filter on gets the
        multi-core build too — measured ~6x faster to searchable than indexing the
        same corpus inline via :meth:`batch_upsert`. Prefer this method for an
        initial load even when the points carry payloads; :meth:`batch_upsert` is
        for writes into a collection that is already built, or for points that
        need content, sparse vectors, TTLs or a CAS precondition, none of which
        the staging wire carries.

        The load is SPLIT across requests to stay under the server's per-request
        caps (256 MiB, 262,144 points), so passing a million vectors in one call
        works. Returns the number of points staged.
        """
        path = f"/v1/collections/{_seg(collection)}/points/bulk"
        staged = 0
        for lo, hi in _chunks(len(ids), vectors, metadatas):
            res = self._send(
                "POST",
                path,
                _encode_bulk_body(
                    ids[lo:hi], vectors[lo:hi],
                    payloads=None if metadatas is None else metadatas[lo:hi],
                ),
                "application/octet-stream",
                timeout=timeout,
            )
            staged += int((res or {}).get("staged", 0))
        return staged

    def bulk_build(self, collection: str, *, workers: int = 0, timeout: float = 24 * 3600) -> None:
        """Build everything staged by :meth:`bulk_stage` into the index in one pass.

        Blocks until the build finishes (minutes on a large corpus), so the
        default timeout is deliberately generous. workers=0 uses every core.
        """
        self._send(
            "POST",
            f"/v1/collections/{_seg(collection)}/points/bulk/build",
            json.dumps({"workers": workers}).encode("utf-8"),
            "application/json",
            timeout=timeout,
        )

    def batch_upsert(
        self,
        collection: str,
        ids: Sequence[int],
        vectors: Sequence[Vector],
        *,
        metadatas: Optional[Sequence[Optional[Dict[str, Any]]]] = None,
        upsert: bool = True,
        timeout: Optional[float] = None,
    ) -> int:
        """Insert/upsert many points in one request over the binary wire.

        Each point is indexed INLINE (no separate build step), which is what makes
        this the path for writing into a collection that is already built. For an
        INITIAL load prefer :meth:`bulk_stage`, which now carries metadata too and
        gets the multi-core build — measured ~6x faster to searchable on a
        payload-bearing 1M x 768d corpus. Split across requests under the server's
        per-request caps, like :meth:`bulk_stage`. Returns the number of points
        written.
        """
        flags = _BULK_FLAG_UPSERT if upsert else 0
        path = f"/v1/collections/{_seg(collection)}/points/batch"
        written = 0
        for lo, hi in _chunks(len(ids), vectors, metadatas):
            res = self._send(
                "POST",
                path,
                _encode_bulk_body(
                    ids[lo:hi], vectors[lo:hi], flags=flags,
                    payloads=None if metadatas is None else metadatas[lo:hi],
                ),
                "application/octet-stream",
                timeout=timeout,
            )
            written += int((res or {}).get("count", 0))
        return written

    def delete(self, collection: str, id: int) -> bool:
        """Delete a point by id; returns whether it existed."""
        res = self._request("DELETE", f"/v1/collections/{_seg(collection)}/points/{_seg(id)}")
        return bool((res or {}).get("deleted"))

    def delete_by_filter(self, collection: str, filter: Dict[str, Any]) -> int:
        """Delete every point matching filter; returns the count removed."""
        res = self._request("POST", f"/v1/collections/{_seg(collection)}/points/delete", {"filter": filter})
        return int((res or {}).get("deleted", 0))

    def scroll(
        self,
        collection: str,
        *,
        filter: Optional[Dict[str, Any]] = None,
        limit: int = 0,
        cursor: str = "",
    ) -> ScrollPage:
        """List live documents (content + metadata) matching filter (None = all),
        in deterministic id-ASCENDING order, up to limit (0 = no cap).

        Returns a ScrollPage: iterate/len it like a list of documents, and read
        its ``next_cursor`` to paginate. A non-empty ``next_cursor`` is the
        resume token for the following page (ids strictly greater than the last
        returned) — pass it back as ``cursor=`` to fetch it; an empty
        ``next_cursor`` means the listing is exhausted. Leaving ``cursor`` empty
        fetches the first page.
        """
        body: Dict[str, Any] = {"limit": limit}
        if filter:
            body["filter"] = filter
        if cursor:
            body["cursor"] = cursor
        res = self._request("POST", f"/v1/collections/{_seg(collection)}/points/scroll", body) or {}
        docs = [_to_document(d) for d in (res.get("documents") or [])]
        return ScrollPage(documents=docs, next_cursor=res.get("next_cursor") or "")

    def get_batch(
        self,
        collection: str,
        ids: Sequence[int],
        *,
        with_vector: bool = True,
        with_payload: bool = True,
    ) -> List["Point"]:
        """Fetch points by id in one request. Returns one Point per PRESENT id
        (absent ids are omitted; never raises on partial miss). Content is lifted
        from the reserved payload field into Point.content and removed from
        Point.metadata."""
        body = {
            "ids": [int(i) for i in ids],
            "with_vector": with_vector,
            "with_payload": with_payload,
        }
        res = self._request("POST", f"/v1/collections/{_seg(collection)}/points/batch-get", body)
        out: List[Point] = []
        for row in (res.get("points") or []):
            meta = decode_metadata(row.get("payload")) if with_payload else {}
            content = ""
            if isinstance(meta, dict) and _RESERVED_CONTENT in meta:
                cv = meta.pop(_RESERVED_CONTENT)
                content = cv if isinstance(cv, str) else ""
            out.append(Point(
                id=row["id"],
                vector=list(row.get("vector") or []),
                content=content,
                metadata=meta,
            ))
        return out

    # ---- search ----

    def search(
        self, collection: str, query: Vector, k: int, *, filter: Optional[Dict[str, Any]] = None
    ) -> List[SearchResult]:
        """k-nearest-neighbor search, returning ids + distances."""
        res = self._search(f"/v1/collections/{_seg(collection)}/points/search", query, k, filter)
        return [SearchResult(id=r["id"], distance=r.get("distance", 0.0), score=r.get("score", 0.0))
                for r in (res.get("results") or [])]

    def search_docs(
        self, collection: str, query: Vector, k: int, *, filter: Optional[Dict[str, Any]] = None
    ) -> List[Document]:
        """kNN search returning each hit enriched with content + metadata."""
        res = self._search(f"/v1/collections/{_seg(collection)}/points/search/docs", query, k, filter)
        return [_to_document(d) for d in (res.get("documents") or [])]

    def search_groups(
        self,
        collection: str,
        query: Vector,
        k: int,
        group_by: str,
        *,
        group_size: int = 1,
        fetch_k: int = 0,
        filter: Optional[Dict[str, Any]] = None,
    ) -> List[Group]:
        """Group-by-document search: top-k distinct documents, best chunk(s) each."""
        body = {"query": list(query), "k": k, "group_by": group_by, "group_size": group_size, "fetch_k": fetch_k}
        if filter:
            body["filter"] = filter
        res = self._request("POST", f"/v1/collections/{_seg(collection)}/points/search/groups", body)
        groups = []
        for g in (res.get("groups") or []):
            key = decode_value(g["key"]) if isinstance(g.get("key"), dict) else g.get("key")
            groups.append(Group(key=key, hits=[_to_document(d) for d in (g.get("hits") or [])]))
        return groups

    def hybrid_search(
        self,
        collection: str,
        dense: Vector,
        k: int,
        *,
        sparse: Optional[Dict[str, Sequence]] = None,
        filter: Optional[Dict[str, Any]] = None,
        method: str = "rrf",
        alpha: float = 0.0,
        rrf_k: int = 0,
        dense_k: int = 0,
        sparse_k: int = 0,
    ) -> List[SearchResult]:
        """Fused dense + sparse search. method: "rrf"|"weighted"."""
        body: Dict[str, Any] = {
            "dense": list(dense), "k": k, "method": method, "alpha": alpha,
            "rrf_k": rrf_k, "dense_k": dense_k, "sparse_k": sparse_k,
        }
        sp = _sparse(sparse)
        if sp:
            body["sparse"] = sp
        if filter:
            body["filter"] = filter
        res = self._request("POST", f"/v1/collections/{_seg(collection)}/points/search/hybrid", body)
        return [SearchResult(id=r["id"], distance=r.get("distance", 0.0), score=r.get("score", 0.0))
                for r in (res.get("results") or [])]

    def search_text(
        self, collection: str, text: str, k: int, *,
        filter: Optional[Dict[str, Any]] = None, global_idf: bool = False,
    ) -> List[Document]:
        """BM25 full-text search. The RAW query text is sent to the server, which
        tokenizes + scores it (the SDK ships no tokens). Returns each hit enriched
        with content + metadata. Requires a collection created with full_text=True.

        global_idf=True opts into the BM25 global-DF (dfs_query_then_fetch) two-phase
        search across partitions (default False ⇒ the per-shard-local-IDF fast path;
        single-partition collections ignore it)."""
        body: Dict[str, Any] = {"text": text, "k": k}
        if filter:
            body["filter"] = filter
        if global_idf:
            body["global_idf"] = True
        res = self._request("POST", f"/v1/collections/{_seg(collection)}/points/search/text", body)
        return [_to_document(d) for d in (res.get("documents") or [])]

    def hybrid_text(
        self,
        collection: str,
        vector: Vector,
        text: str,
        k: int,
        *,
        filter: Optional[Dict[str, Any]] = None,
        method: str = "rrf",
        alpha: float = 0.0,
        rrf_k: int = 0,
        dense_k: int = 0,
        sparse_k: int = 0,
        global_idf: bool = False,
    ) -> List[SearchResult]:
        """Fused dense + BM25 full-text search. The dense query vector plus the RAW
        query text are sent; the server analyzes the text into the BM25 lane and
        fuses it with the dense lane. method: "rrf"|"weighted"|"dbsf". Requires a
        collection created with full_text=True.

        global_idf=True opts into the BM25 global-DF (dfs_query_then_fetch) two-phase
        text lane across partitions (default False ⇒ the per-shard-local-IDF fast
        path; affects only the BM25 text lane)."""
        body: Dict[str, Any] = {
            "vector": list(vector), "text": text, "k": k, "method": method, "alpha": alpha,
            "rrf_k": rrf_k, "dense_k": dense_k, "sparse_k": sparse_k,
        }
        if filter:
            body["filter"] = filter
        if global_idf:
            body["global_idf"] = True
        res = self._request("POST", f"/v1/collections/{_seg(collection)}/points/search/hybrid-text", body)
        return [SearchResult(id=r["id"], distance=r.get("distance", 0.0), score=r.get("score", 0.0))
                for r in (res.get("results") or [])]


    # ---- late interaction (multi-vector / ColBERT MaxSim) ----

    def mv_create_collection(
        self,
        name: str,
        dim: int,
        *,
        m: int = 0,
        ef_construction: int = 0,
        ef_search: int = 0,
        seed: int = 0,
        quant: str = "",
        rescore_factor: int = 0,
        persistent: bool = False,
    ) -> None:
        """Create a late-interaction collection (token vectors + MaxSim).

        quant ("sq8"|"bq1") quantizes the first-stage graph; with persistent=True
        the float32 token vectors move off-heap into an mmap file and the
        collection is durable across restart.
        """
        body = {
            "dim": dim, "m": m, "ef_construction": ef_construction, "ef_search": ef_search,
            "seed": seed, "quant": quant, "rescore_factor": rescore_factor, "persistent": persistent,
        }
        self._request("POST", "/v1/multivector/" + _seg(name), body)

    def mv_drop_collection(self, name: str) -> None:
        """Delete a late-interaction collection."""
        self._request("DELETE", "/v1/multivector/" + _seg(name))

    def mv_add(
        self,
        name: str,
        doc_id: int,
        tokens: Sequence[Vector],
        *,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Insert or replace a document represented by its token vectors."""
        body = {"id": doc_id, "tokens": [list(t) for t in tokens], "metadata": encode_metadata(metadata)}
        self._request("POST", f"/v1/multivector/{_seg(name)}/docs", body)

    def mv_search(
        self,
        name: str,
        query: Sequence[Vector],
        k: int,
        *,
        candidates_per_token: int = 0,
    ) -> List[MultiResult]:
        """MaxSim late-interaction search: top-k documents for the multi-vector query."""
        body = {"query": [list(t) for t in query], "k": k, "candidates_per_token": candidates_per_token}
        res = self._request("POST", f"/v1/multivector/{_seg(name)}/search", body)
        return [MultiResult(id=r["id"], score=r.get("score", 0.0), metadata=decode_metadata(r.get("metadata")))
                for r in (res.get("results") or [])]

    def mv_delete(self, name: str, doc_id: int) -> bool:
        """Delete a document from a late-interaction collection."""
        res = self._request("DELETE", f"/v1/multivector/{_seg(name)}/docs/{_seg(doc_id)}")
        return bool((res or {}).get("deleted"))


def _to_document(d: Dict[str, Any]) -> Document:
    return Document(
        id=d["id"],
        distance=d.get("distance", 0.0),
        score=d.get("score", 0.0),
        content=d.get("content", ""),
        metadata=decode_metadata(d.get("metadata")),
    )
