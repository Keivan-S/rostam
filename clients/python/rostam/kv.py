"""A native-protocol client for Rostam's key-value store.

The KV store is not on the REST API — it lives only on the binary TCP protocol,
because it is built for sub-microsecond operations that an HTTP round trip would
defeat. So this client speaks that protocol directly, over a plain socket, using
only the standard library.

    from rostam import RostamKV

    kv = RostamKV("127.0.0.1", 7000)          # the server's -tcp port
    kv.put("user:42", b'{"coins":100}')
    kv.get("user:42")                          # -> b'{"coins":100}'
    kv.incr("views:42", 1)                     # -> 1
    kv.delete("user:42")                       # -> True

Keys and values may be ``str`` (encoded UTF-8) or ``bytes``; every method
returns ``bytes`` (or ``None`` for a miss), never decoding for you — a value
store holds opaque bytes.

Wire, for reference (all big-endian):

    frame     [len u32][body]
    body v1   [opNameLen u8][opName][argsLen u32][args]
    body v2   [0x02][tokenLen u8][token][opNameLen u8][opName][argsLen u32][args]
    response  [bodyLen u32][status u8][payloadLen u32][payload]

v2 is used when an auth token is set, v1 otherwise — mirroring the Go client.
"""

from __future__ import annotations

import socket
import struct
import threading
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

from . import _vecwire
from .client import RostamError

Key = Union[str, bytes]

_STATUS_OK = 0
_STATUS_NOT_FOUND = 1
_STATUS_NOT_LEADER = 2
_STATUS_ERROR = 3
_STATUS_UNAUTHORIZED = 4

_PROTOCOL_V2 = 0x02
# The server rejects a frame whose length prefix exceeds this; used only to
# fail fast with a clear message rather than send a doomed frame.
_MAX_FRAME = 64 * 1024 * 1024


def _as_bytes(x: Key) -> bytes:
    return x.encode("utf-8") if isinstance(x, str) else bytes(x)


def _enc_key(key: bytes) -> bytes:
    if len(key) > 0xFFFF:
        raise ValueError(f"key length {len(key)} exceeds 65535")
    return struct.pack(">H", len(key)) + key


class _SocketPool:
    """A tiny pool of connected sockets, safe to share across threads.

    Mirrors the HTTP client's pool: hand a live socket to a caller, take it back
    when they are done. A socket that errored mid-call is discarded rather than
    returned, so a broken connection never poisons the next request.
    """

    def __init__(self, host: str, port: int, timeout: float, maxsize: int):
        self._host = host
        self._port = port
        self._timeout = timeout
        self._maxsize = maxsize
        self._free: List[socket.socket] = []
        self._lock = threading.Lock()
        self._closed = False

    def _connect(self) -> socket.socket:
        s = socket.create_connection((self._host, self._port), timeout=self._timeout)
        # Nagle off: these are small, latency-sensitive request/response frames.
        s.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
        return s

    def acquire(self) -> socket.socket:
        with self._lock:
            if self._closed:
                raise RostamError("client is closed")
            s = self._free.pop() if self._free else None
        if s is None:
            s = self._connect()
        else:
            s.settimeout(self._timeout)
        return s

    def release(self, s: socket.socket) -> None:
        with self._lock:
            if self._closed or len(self._free) >= self._maxsize:
                drop = True
            else:
                self._free.append(s)
                drop = False
        if drop:
            _silent_close(s)

    def discard(self, s: socket.socket) -> None:
        _silent_close(s)

    def close(self) -> None:
        with self._lock:
            self._closed = True
            socks, self._free = self._free, []
        for s in socks:
            _silent_close(s)


def _silent_close(s: socket.socket) -> None:
    try:
        s.close()
    except OSError:
        pass


def _recv_exactly(s: socket.socket, n: int) -> bytes:
    """Read exactly n bytes or raise — a short read means the peer went away."""
    chunks = []
    got = 0
    while got < n:
        b = s.recv(n - got)
        if not b:
            raise RostamError("connection closed by server mid-response")
        chunks.append(b)
        got += len(b)
    return b"".join(chunks)


class Rostam:
    """Native-protocol client for Rostam over the binary TCP protocol.

    Key-value operations are methods on this object (get/put/delete/incr/expire);
    vector-database operations live under ``.vector`` (create_collection, upsert,
    search, get, delete, exists), sharing one connection pool and auth token.

    Talks to the server's binary TCP port (``-tcp``), which the container serves
    by default on ``7000``. Set ``auth_token`` when the server requires one; it
    is sent on every request via the protocol-v2 frame.
    """

    def __init__(
        self,
        host: str = "127.0.0.1",
        port: int = 7000,
        *,
        auth_token: Optional[str] = None,
        timeout: float = 30.0,
        pool_maxsize: int = 8,
    ):
        self._token = auth_token or ""
        if len(self._token.encode("utf-8")) > 255:
            raise ValueError("auth_token is longer than 255 bytes")
        self._pool = _SocketPool(host, port, timeout, pool_maxsize)
        #: Vector-database operations over the same connection. See _VectorAPI.
        self.vector = _VectorAPI(self)

    # ---- framing -----------------------------------------------------------

    def _encode_body(self, op: str, args: bytes) -> bytes:
        op_b = op.encode("ascii")
        if len(op_b) > 0xFF:
            raise ValueError("op name too long")
        v1 = bytes([len(op_b)]) + op_b + struct.pack(">I", len(args)) + args
        if not self._token:
            return v1
        tok = self._token.encode("utf-8")
        return bytes([_PROTOCOL_V2, len(tok)]) + tok + v1

    def _call(self, op: str, args: bytes) -> bytes:
        """Send one op, return its payload. Raises RostamError on a non-OK status.

        A miss (StatusNotFound) is NOT an error here — it returns ``None`` — so
        the read ops can distinguish "absent" from "empty value" without an
        exception. Every other non-OK status raises.
        """
        body = self._encode_body(op, args)
        if 4 + len(body) > _MAX_FRAME:
            raise RostamError("request frame exceeds the server's frame limit")
        frame = struct.pack(">I", len(body)) + body

        # Acquire may connect, and a failed connect raises OSError — convert it to
        # the client's RostamError contract like every other transport failure.
        try:
            s = self._pool.acquire()
        except OSError as e:
            raise RostamError(f"connect failed: {e}") from e

        # Send, read, AND fully parse inside the try: a socket is returned to the
        # pool only after a well-formed response, so a truncated or malformed
        # frame discards the connection instead of poisoning the next caller.
        try:
            s.sendall(frame)
            body_len = struct.unpack(">I", _recv_exactly(s, 4))[0]
            # A response body is at least [status u8][payloadLen u32] = 5 bytes and
            # never larger than a frame. Reject a bogus length before reading it, so
            # a broken or hostile peer cannot make _recv_exactly buffer unboundedly.
            if body_len < 5 or body_len > _MAX_FRAME:
                raise RostamError(f"invalid response frame length {body_len}")
            resp = _recv_exactly(s, body_len)
            status = resp[0]
            payload_len = struct.unpack(">I", resp[1:5])[0]
            if 5 + payload_len != body_len:
                raise RostamError("response payload length does not match frame")
            payload = resp[5:5 + payload_len]
        except (OSError, RostamError, struct.error) as e:
            self._pool.discard(s)
            if isinstance(e, RostamError):
                raise
            raise RostamError(f"transport error: {e}") from e
        # Well-formed response: the connection is healthy even if the op failed
        # at the application level (StatusError etc.), so keep it pooled.
        self._pool.release(s)

        if status == _STATUS_OK:
            return payload
        if status == _STATUS_NOT_FOUND:
            return None  # type: ignore[return-value]
        raise RostamError(_status_message(status, payload), status=status)

    # ---- key-value ops -----------------------------------------------------

    def get(self, key: Key) -> Optional[bytes]:
        """Return the value bytes, or ``None`` if the key is absent."""
        return self._call("get", _enc_key(_as_bytes(key)))

    def put(self, key: Key, value: Key, *, ttl_ms: int = 0) -> None:
        """Store ``value`` under ``key``. ``ttl_ms`` > 0 sets an expiry."""
        k = _as_bytes(key)
        v = _as_bytes(value)
        args = _enc_key(k) + struct.pack(">I", len(v)) + v + struct.pack(">Q", ttl_ms)
        self._call("put", args)

    def delete(self, key: Key) -> bool:
        """Delete ``key``. Returns whether it existed."""
        payload = self._call("del", _enc_key(_as_bytes(key)))
        return bool(payload and payload[0])

    def incr(self, key: Key, delta: int = 1) -> int:
        """Atomically add ``delta`` (may be negative) and return the new value.

        A missing key is treated as 0, so the first ``incr`` returns ``delta``.
        """
        args = _enc_key(_as_bytes(key)) + struct.pack(">q", delta)
        payload = self._call("incr", args)
        return struct.unpack(">q", payload)[0]

    def expire(self, key: Key, ttl_ms: int) -> None:
        """Set a TTL (in milliseconds) on an existing key."""
        args = _enc_key(_as_bytes(key)) + struct.pack(">Q", ttl_ms)
        self._call("expire", args)

    def ping(self) -> bool:
        """Round-trip a heartbeat; True if the server answered."""
        self._call("__ping__", b"")
        return True

    def close(self) -> None:
        self._pool.close()

    def __enter__(self) -> "Rostam":
        return self

    def __exit__(self, *exc) -> None:
        self.close()


def _status_message(status: int, payload: bytes) -> str:
    detail = payload.decode("utf-8", "replace") if payload else ""
    name = {
        _STATUS_NOT_LEADER: "not leader",
        _STATUS_ERROR: "server error",
        _STATUS_UNAUTHORIZED: "unauthorized (auth token missing or invalid)",
    }.get(status, f"status {status}")
    return f"{name}: {detail}" if detail else name


class _VectorAPI:
    """Vector-database operations, reached as ``client.vector.*``.

    Shares the parent client's socket pool and auth. The binary arg layouts are
    in rostam._vecwire and are differential-tested byte-for-byte against the Go
    encoders; the JSON-carrying parts (metadata, filter, content) round-trip
    through a real server.
    """

    def __init__(self, client: "Rostam"):
        self._c = client

    def create_collection(self, name: str, dim: int, *, metric: str = "cosine", **cfg: Any) -> None:
        """Create a vector collection. Keyword config mirrors the HTTP client:
        m, ef_construction, ef_search, seed, quant, persistent, index_type,
        ivf_nlist, ivf_nprobe, vamana_r/l/alpha, full_text, ..."""
        conf = dict(cfg); conf["dim"] = dim; conf["metric"] = metric
        self._c._call("vector_create_collection", _vecwire.encode_create_collection_args(name, conf))

    def upsert(self, collection: str, id: int, vector: Sequence[float], *, content: str = "",
               metadata: Optional[Dict[str, Any]] = None, ttl_ms: int = 0,
               sparse: Optional[Dict[str, Sequence]] = None) -> None:
        """Insert or replace a point, optionally with stored content for RAG."""
        self._c._call("vector_upsert", _vecwire.encode_upsert_args(
            collection, int(id), vector, content=content, ttl_ms=ttl_ms,
            metadata=metadata, sparse=sparse))

    def insert(self, collection: str, id: int, vector: Sequence[float], *,
               metadata: Optional[Dict[str, Any]] = None, ttl_ms: int = 0,
               sparse: Optional[Dict[str, Sequence]] = None) -> None:
        """Create-only insert (errors if the id is live)."""
        self._c._call("vector_insert", _vecwire.encode_insert_args(
            collection, int(id), vector, ttl_ms=ttl_ms, metadata=metadata, sparse=sparse))

    def search(self, collection: str, query: Sequence[float], k: int, *,
               filter: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
        """k-nearest-neighbour search. Returns [{id, distance}, ...]."""
        payload = self._c._call("vector_search", _vecwire.encode_search_args(collection, k, query, filter))
        return _vecwire.decode_search_results(payload or b"\x00\x00\x00\x00")

    def get(self, collection: str, id: int, *, with_vector: bool = True,
            with_payload: bool = True) -> Optional[Dict[str, Any]]:
        """Fetch a point by id, or None if absent. Returns {vector, metadata,
        ttl_ms, sparse} — fields not requested come back empty."""
        flags = (0x01 if with_vector else 0) | (0x02 if with_payload else 0)
        payload = self._c._call("vector_get", _vecwire.encode_get_args(collection, int(id), flags))
        if payload is None:
            return None
        got = _vecwire.decode_get_result(payload)
        if got is None:
            # A miss comes back as StatusOK with a found=0 body (not StatusNotFound),
            # so the payload is non-None but decodes to None. Absent is absent.
            return None
        # Lift stored content out of the reserved $content key, mirroring the
        # HTTP client's Point shape.
        meta = got.get("metadata") or {}
        got["content"] = meta.pop("$content", "")
        return got

    def delete(self, collection: str, id: int) -> bool:
        """Delete a point. Returns whether it existed."""
        payload = self._c._call("vector_delete", _vecwire.encode_delete_args(collection, int(id)))
        return bool(payload and payload[0])

    def exists(self, collection: str, id: int) -> bool:
        payload = self._c._call("vector_exists", _vecwire.encode_exists_args(collection, int(id)))
        return _vecwire.decode_exists_result(payload or b"\x00")

    # ---- Phase C: batch / scroll / RAG-shaped search / hybrid / recommend ---
    #
    # Below, `_degraded`/`_missing` from the decoders are intentionally not
    # surfaced to the caller yet (same as search()'s single-node-only reads) —
    # a clustered-deployment follow-up.

    def get_batch(self, collection: str, ids: Sequence[int], *, with_vector: bool = True,
                  with_payload: bool = True) -> List[Dict[str, Any]]:
        """Fetch multiple points by id in one round trip. Returns a list of
        {id, found, vector, metadata, ttl_ms, sparse, version} rows in the order
        `ids` was given; a not-found id comes back with found=False."""
        flags = (0x01 if with_vector else 0) | (0x02 if with_payload else 0)
        payload = self._c._call("vector_get_batch",
                                _vecwire.encode_vector_get_batch_args(collection, list(ids), flags))
        rows = _vecwire.decode_get_batch_result(payload or b"\x00\x00\x00\x00")
        # Lift stored content out of the reserved $content key, mirroring get()'s
        # Point shape (a not-found row's metadata is already {}, so pop is a no-op).
        for row in rows:
            meta = row.get("metadata") or {}
            row["content"] = meta.pop("$content", "")
        return rows

    def scroll(self, collection: str, *, filter: Optional[Dict[str, Any]] = None,
              limit: int = 0, cursor: str = "") -> Tuple[List[Dict[str, Any]], str]:
        """Page through a collection's points in id order. Returns
        (docs, next_cursor) — pass next_cursor back in as `cursor` to fetch the
        next page; an empty next_cursor means the scroll is exhausted."""
        after_id, _has_after = _vecwire.decode_scroll_cursor(cursor)
        args = _vecwire.encode_scroll_args_order_bounded(collection, limit, filter=filter, after_id=after_id)
        payload = self._c._call("vector_scroll", args)
        docs, _degraded, _missing, next_cursor = _vecwire.decode_scroll_result_raw(payload or b"\x00\x00\x00\x00")
        if not next_cursor and limit > 0 and len(docs) == limit:
            # This op's leaf handler (handleVectorScroll) returns a plain doc block
            # with no wire cursor on an unpartitioned/single-node server — only a
            # clustered coordinator's fan-out dispatcher supplies one. Derive it
            # client-side in that case, mirroring the Go SDK's scrollNextCursor: a
            # FULL page may have more, so resume after the last doc's id.
            next_cursor = _vecwire.encode_scroll_cursor(docs[-1]["id"])
        return docs, next_cursor

    def search_docs(self, collection: str, query: Sequence[float], k: int, *,
                    filter: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
        """k-nearest-neighbour search returning documents (content + metadata)
        instead of bare ids/distances — the RAG-shaped counterpart of search()."""
        args = _vecwire.encode_search_docs_args_opts(collection, k, query, filter)
        payload = self._c._call("vector_search_docs", args)
        docs, _degraded, _missing = _vecwire.decode_docs_degraded_raw(payload or b"\x00\x00\x00\x00")
        return docs

    def search_groups(self, collection: str, query: Sequence[float], k: int, group_by: str, *,
                      group_size: int = 1, fetch_k: int = 0,
                      filter: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
        """k-nearest-neighbour search grouped by a payload field. Returns a list
        of {key, hits} groups, where each hit has the same document shape
        search_docs() returns."""
        opts = {"group_by": group_by, "group_size": group_size, "fetch_k": fetch_k, "filter": filter}
        args = _vecwire.encode_group_search_args_opts(collection, k, query, opts)
        payload = self._c._call("vector_search_groups", args)
        groups, _degraded, _missing = _vecwire.decode_groups_degraded_raw(payload or b"\x00\x00\x00\x00")
        return groups

    def hybrid_search(self, collection: str, dense: Sequence[float], k: int, *,
                      sparse: Optional[Dict[str, Sequence]] = None,
                      filter: Optional[Dict[str, Any]] = None, method: str = "rrf",
                      alpha: float = 0.0, rrf_k: int = 0, dense_k: int = 0,
                      sparse_k: int = 0) -> List[Dict[str, Any]]:
        """Fuse a dense-KNN lane with an optional sparse lane. Returns
        [{id, distance, score}, ...] fused by `method` ("rrf"/"weighted"/"dbsf")."""
        opts = {"filter": filter, "method": method, "alpha": alpha, "rrf_k": rrf_k,
                "dense_k": dense_k, "sparse_k": sparse_k}
        args = _vecwire.encode_hybrid_search_args_opts(collection, dense, k, sparse, opts)
        payload = self._c._call("vector_hybrid_search", args)
        results, _degraded, _missing = _vecwire.decode_hybrid_results_degraded(payload or b"\x00\x00\x00\x00")
        return results

    def hybrid_text(self, collection: str, dense: Sequence[float], text: str, k: int, *,
                    filter: Optional[Dict[str, Any]] = None, method: str = "rrf",
                    alpha: float = 0.0, rrf_k: int = 0, dense_k: int = 0,
                    sparse_k: int = 0) -> List[Dict[str, Any]]:
        """Fuse a dense-KNN lane with a server-side BM25 full-text lane (the
        collection must have been created with full_text=... for a full-text
        analyzer to exist). Returns fused [{id, distance, score}, ...]."""
        opts = {"filter": filter, "method": method, "alpha": alpha, "rrf_k": rrf_k,
                "dense_k": dense_k, "sparse_k": sparse_k}
        args = _vecwire.encode_hybrid_text_args_global(collection, dense, text, k, opts)
        payload = self._c._call("vector_hybrid_text", args)
        results, _degraded, _missing = _vecwire.decode_hybrid_results_degraded(payload or b"\x00\x00\x00\x00")
        return results

    def recommend(self, collection: str, positive: Sequence[int], *,
                  negative: Optional[Sequence[int]] = None, k: int = 10,
                  filter: Optional[Dict[str, Any]] = None,
                  strategy: str = "average_vector") -> List[Dict[str, Any]]:
        """Recommend points similar to the `positive` example ids and dissimilar
        to the `negative` ones. `strategy`: "average_vector" (default, average
        the example vectors then kNN) or "best_score" (score by best per-example
        similarity). Rides the vector_query op with a RECOMMEND-shaped QuerySpec
        (see `query` for why a general QuerySpec is out of scope for this client)."""
        strat = _vecwire.RECOMMEND_STRATEGY[strategy]
        args = _vecwire.encode_recommend_query(collection, positive=positive, negative=negative,
                                               k=k, filter=filter, strategy=strat)
        payload = self._c._call("vector_query", args)
        results, _degraded, _missing = _vecwire.decode_query_result_degraded(payload or b"\x01\x00\x00\x00\x00")
        return results

    def query(self, collection: str, positive: Sequence[int], *,
             negative: Optional[Sequence[int]] = None, k: int = 10,
             filter: Optional[Dict[str, Any]] = None,
             strategy: str = "average_vector") -> List[Dict[str, Any]]:
        """The unified Query API — RECOMMEND-shaped ONLY. Phase B's stdlib-only
        protobuf QuerySpec encoder (_vecwire.marshal_recommend_query_spec) builds
        a single-leaf RECOMMEND spec; it does not build the general
        fusion/rerank/prefetch-tree QuerySpec the Go SDK's Query API otherwise
        supports (that would need a fuller hand-rolled proto encoder, which is
        out of scope here). So `query` and `recommend` make the identical call —
        `query` exists only so callers reaching for the "unified Query API" name
        find the one shape this client speaks."""
        return self.recommend(collection, positive, negative=negative, k=k,
                              filter=filter, strategy=strategy)

    def upsert_batch(self, collection: str, points: Sequence[Dict[str, Any]]) -> None:
        """N sequential vector_upsert ops over one connection, each awaited
        before the next is sent — there is no native-TCP batch-upsert wire op
        (see _vecwire.encode_upsert_batch_args), and this does not pipeline
        (matching the Go client, which also loops since there's no native batch
        op); real pipelining (write all frames, then read all responses) is a
        possible future optimization. Each point dict: {id, vector, content="",
        ttl_ms=0, metadata=None, sparse=None}."""
        for args in _vecwire.encode_upsert_batch_args(collection, points):
            self._c._call("vector_upsert", args)


# Backwards-compatible alias: the client began life as a KV-only RostamKV.
RostamKV = Rostam
