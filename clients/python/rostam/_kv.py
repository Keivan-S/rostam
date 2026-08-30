"""Key-value operations, reached as ``client.kv.*``.

The KV store is not on the REST API — it lives only on the binary TCP
protocol, because it is built for sub-microsecond operations that an HTTP
round trip would defeat. So ``_KV`` speaks that protocol directly, via the
``TcpTransport`` it is handed: it reuses that transport's connection pool,
wire framing (``_call``), and auth token rather than opening a second pool,
so KV and vector ops interleave freely on the same pooled sockets.

Wire, for reference (all big-endian) — unchanged from the pre-unification
``kv.py``:

    frame     [len u32][body]
    body v1   [opNameLen u8][opName][argsLen u32][args]
    body v2   [0x02][tokenLen u8][token][opNameLen u8][opName][argsLen u32][args]
    response  [bodyLen u32][status u8][payloadLen u32][payload]

On the HTTP backend, ``client.kv`` is instead a ``_KVUnavailable`` sentinel:
the KV store has no REST surface, so any attribute access raises
``TransportError`` rather than silently no-op-ing.
"""

from __future__ import annotations

import struct
from typing import Optional, Union

from ._types import TransportError

Key = Union[str, bytes]


def _as_bytes(x: Key) -> bytes:
    return x.encode("utf-8") if isinstance(x, str) else bytes(x)


def _enc_key(key: bytes) -> bytes:
    if len(key) > 0xFFFF:
        raise ValueError(f"key length {len(key)} exceeds 65535")
    return struct.pack(">H", len(key)) + key


def _dec_found_value(payload: Optional[bytes]) -> Optional[bytes]:
    """Decode a ``[found u8](+[valLen u32][val])`` result (getdel / getset).

    ``found=0`` (or an empty payload) is ``None``; ``found=1`` returns the value
    bytes (a non-``None`` empty ``bytes`` when the stored value was empty).
    """
    if not payload or payload[0] == 0:
        return None
    (vlen,) = struct.unpack_from(">I", payload, 1)
    return bytes(payload[5 : 5 + vlen])


class _KV:
    """Key-value operations over Rostam's native binary TCP protocol.

    Holds the parent ``TcpTransport`` and calls its ``_call`` directly, so KV
    ops share that transport's connection pool and auth token with the flat
    vector API (``r.search``, ``r.upsert``, ...) rather than owning a second
    pool of their own.
    """

    def __init__(self, transport):
        self._t = transport

    def get(self, key: Key) -> Optional[bytes]:
        """Return the value bytes, or ``None`` if the key is absent."""
        return self._t._call("get", _enc_key(_as_bytes(key)), idempotent=True)

    def put(self, key: Key, value: Key, *, ttl_ms: int = 0) -> None:
        """Store ``value`` under ``key``. ``ttl_ms`` > 0 sets an expiry."""
        k = _as_bytes(key)
        v = _as_bytes(value)
        args = _enc_key(k) + struct.pack(">I", len(v)) + v + struct.pack(">Q", ttl_ms)
        self._t._call("put", args)

    def delete(self, key: Key) -> bool:
        """Delete ``key``. Returns whether it existed."""
        payload = self._t._call("del", _enc_key(_as_bytes(key)))
        return bool(payload and payload[0])

    def incr(self, key: Key, delta: int = 1) -> int:
        """Atomically add ``delta`` (may be negative) and return the new value.

        A missing key is treated as 0, so the first ``incr`` returns ``delta``.
        """
        args = _enc_key(_as_bytes(key)) + struct.pack(">q", delta)
        payload = self._t._call("incr", args)
        return struct.unpack(">q", payload)[0]

    def set_nx(self, key: Key, value: Key, *, ttl_ms: int = 0) -> bool:
        """Set only if the key is absent. Returns True if stored, False if it already exists.

        Atomic on the server (the check and the store run under one shard write
        lock). ``ttl_ms`` > 0 sets an expiry on the stored value.
        """
        k, v = _as_bytes(key), _as_bytes(value)
        args = _enc_key(k) + struct.pack(">I", len(v)) + v + struct.pack(">Q", ttl_ms)
        payload = self._t._call("set_nx", args)
        return bool(payload and payload[0])

    def cas(self, key: Key, value: Key, expected: Optional[Key], *, ttl_ms: int = 0) -> bool:
        """Compare-and-swap: set only if current == expected (``expected=None`` ⇒ only if absent).

        Returns True if the value was stored, False on a mismatch. ``ttl_ms`` > 0
        sets an expiry on the stored value.
        """
        k, v = _as_bytes(key), _as_bytes(value)
        has = expected is not None
        e = _as_bytes(expected) if has else b""
        args = (
            _enc_key(k)
            + struct.pack(">I", len(v))
            + v
            + struct.pack(">B", 1 if has else 0)
            + struct.pack(">I", len(e))
            + e
            + struct.pack(">Q", ttl_ms)
        )
        payload = self._t._call("cas", args)
        return bool(payload and payload[0])

    def compare_and_delete(self, key: Key, expected: Key) -> bool:
        """Delete only if current == expected (safe unlock). Returns True if deleted."""
        k, e = _as_bytes(key), _as_bytes(expected)
        args = _enc_key(k) + struct.pack(">I", len(e)) + e
        payload = self._t._call("cad", args)
        return bool(payload and payload[0])

    def exists(self, key: Key) -> bool:
        """Return whether ``key`` is currently present (an expired key is absent)."""
        payload = self._t._call("exists", _enc_key(_as_bytes(key)), idempotent=True)
        return bool(payload and payload[0])

    def getdel(self, key: Key) -> Optional[bytes]:
        """Atomically return ``key``'s value and delete it; ``None`` if absent."""
        payload = self._t._call("getdel", _enc_key(_as_bytes(key)))
        return _dec_found_value(payload)

    def getset(self, key: Key, value: Key, *, ttl_ms: int = 0) -> Optional[bytes]:
        """Atomically set ``key`` to ``value`` and return the OLD value (``None`` if
        the key had no prior value). ``ttl_ms`` > 0 sets an expiry on the new value.
        """
        k, v = _as_bytes(key), _as_bytes(value)
        args = _enc_key(k) + struct.pack(">I", len(v)) + v + struct.pack(">Q", ttl_ms)
        payload = self._t._call("getset", args)
        return _dec_found_value(payload)

    def persist(self, key: Key) -> bool:
        """Remove ``key``'s TTL so it never expires. Returns True if a TTL was
        removed, False if the key is absent or already had no expiry.
        """
        payload = self._t._call("persist", _enc_key(_as_bytes(key)))
        return bool(payload and payload[0])

    def ttl(self, key: Key) -> int:
        """Return ``key``'s remaining time-to-live in milliseconds (Redis
        convention): ``-2`` if absent, ``-1`` if present but with no expiry, else
        the remaining ms (>= 0).
        """
        payload = self._t._call("ttl", _enc_key(_as_bytes(key)), idempotent=True)
        return struct.unpack(">q", payload)[0]

    def incr_ex(self, key: Key, delta: int = 1, *, ttl_ms: int = 0) -> int:
        """Atomically add ``delta`` and return the new value, setting ``ttl_ms`` as
        the expiry only when the key is newly created — on an existing key the
        increment leaves the deadline untouched. The fixed-window rate-limit
        primitive: the first hit creates the counter with the window TTL, later
        hits increment it without extending the window.
        """
        args = _enc_key(_as_bytes(key)) + struct.pack(">q", delta) + struct.pack(">Q", ttl_ms)
        payload = self._t._call("incr_ex", args)
        return struct.unpack(">q", payload)[0]

    def compare_and_expire(self, key: Key, expected: Key, *, ttl_ms: int) -> bool:
        """Refresh ``key``'s TTL only if its current value equals ``expected`` — the
        lock-renewal primitive (extend a lease only while you still hold the token).
        Returns True if the TTL was refreshed, False on a mismatch or absent key.
        """
        k, e = _as_bytes(key), _as_bytes(expected)
        args = _enc_key(k) + struct.pack(">I", len(e)) + e + struct.pack(">Q", ttl_ms)
        payload = self._t._call("caex", args)
        return bool(payload and payload[0])

    def mget(self, keys) -> list:
        """Fetch many keys in one round trip, returning a list of
        ``Optional[bytes]`` aligned to ``keys`` (``None`` for each absent key).

        The Python KV client talks to a single node and does no shard routing —
        every op goes to the connected server — so ``mget`` issues one batched
        call routed by its first key, matching how ``get`` / ``put`` already
        behave against that node (correct on a single-node store; against a
        cluster it reflects only what the connected shard owns, exactly like the
        per-key ops).
        """
        ks = [_as_bytes(k) for k in keys]
        if len(ks) > 0xFFFF:
            raise ValueError(f"mget key count {len(ks)} exceeds 65535")
        buf = bytearray(struct.pack(">H", len(ks)))
        for k in ks:
            buf += _enc_key(k)
        payload = self._t._call("mget", bytes(buf), idempotent=True)
        out: list = []
        off = 2
        count = struct.unpack_from(">H", payload, 0)[0] if payload else 0
        for _ in range(count):
            found = payload[off]
            off += 1
            if not found:
                out.append(None)
                continue
            (vlen,) = struct.unpack_from(">I", payload, off)
            off += 4
            out.append(bytes(payload[off : off + vlen]))
            off += vlen
        return out

    def expire(self, key: Key, ttl_ms: int) -> None:
        """Set a TTL (in milliseconds) on an existing key."""
        args = _enc_key(_as_bytes(key)) + struct.pack(">Q", ttl_ms)
        self._t._call("expire", args)

    def ping(self) -> bool:
        """Round-trip a heartbeat; True if the server answered."""
        self._t._call("__ping__", b"", idempotent=True)
        return True


class _KVUnavailable:
    """Installed as ``client.kv`` on the HTTP backend.

    The KV store has no REST surface, so any attribute access (``.get``,
    ``.put``, or anything else) raises ``TransportError`` — never on
    construction, only when a caller actually reaches for ``r.kv.*``.
    """

    def __getattr__(self, name: str):
        raise TransportError(
            "key-value operations require the TCP transport; connect with tcp://host:7000"
        )
