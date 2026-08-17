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
from typing import List, Optional, Union

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


class RostamKV:
    """Native-protocol client for Rostam's key-value operations.

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

    def __enter__(self) -> "RostamKV":
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
