"""Server-free framing tests for the new KV roadmap ops (exists / getdel /
getset / persist / ttl / incr_ex / caex / mget).

These do not need the Go server binary: a fake socket captures the request frame
and replies with a canned response, so they pin BOTH halves of each op's wire
contract in pure Python — the exact op name + args bytes the client sends, and
that it decodes a known response correctly. A slip in either (a wrong struct
format, a mislaid length prefix) shows up here without a cross-stack run.

The request framing (``[opLen u8][op][argsLen u32][args]``, no auth token) and
the response framing (``[bodyLen u32][status u8][payloadLen u32][payload]``) are
shared with r.kv's other ops via TcpTransport, so this exercises the same path
test_kv_framing.py does, just asserting the roadmap ops' payloads.
"""

from __future__ import annotations

import socket
import struct
import threading
import unittest

from rostam.rostam import Rostam

_STATUS_OK = 0


def _ok(payload: bytes) -> bytes:
    body = bytes([_STATUS_OK]) + struct.pack(">I", len(payload)) + payload
    return struct.pack(">I", len(body)) + body


def _recv_all(conn, n):
    buf = b""
    while len(buf) < n:
        chunk = conn.recv(n - len(buf))
        if not chunk:
            return None
        buf += chunk
    return buf


class _CapturingServer:
    """Captures each request's (op, args) and replies with a scripted response."""

    def __init__(self, responses):
        self._responses = list(responses)
        self.requests = []
        self._sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self._sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self._sock.bind(("127.0.0.1", 0))
        self._sock.listen(1)
        self.port = self._sock.getsockname()[1]
        self._t = threading.Thread(target=self._serve, daemon=True)
        self._t.start()

    def _serve(self):
        try:
            conn, _ = self._sock.accept()
        except OSError:
            return
        i = 0
        try:
            with conn:
                while True:
                    hdr = _recv_all(conn, 4)
                    if hdr is None:
                        return
                    n = struct.unpack(">I", hdr)[0]
                    body = _recv_all(conn, n)
                    if body is None:
                        return
                    op_len = body[0]
                    op = body[1 : 1 + op_len].decode("ascii")
                    args_len = struct.unpack(">I", body[1 + op_len : 5 + op_len])[0]
                    args = body[5 + op_len : 5 + op_len + args_len]
                    self.requests.append((op, args))
                    if i < len(self._responses):
                        conn.sendall(self._responses[i])
                        i += 1
                    else:
                        return
        except OSError:
            return

    def close(self):
        try:
            self._sock.close()
        except OSError:
            pass


class KVRoadmapFramingTest(unittest.TestCase):
    def _client(self, responses):
        srv = _CapturingServer(responses)
        self.addCleanup(srv.close)
        r = Rostam(f"tcp://127.0.0.1:{srv.port}", timeout=2.0)
        self.addCleanup(r.close)
        return r, srv

    def test_exists_frames_key_and_decodes_bool(self):
        r, srv = self._client([_ok(b"\x01")])
        self.assertTrue(r.kv.exists("k"))
        op, args = srv.requests[0]
        self.assertEqual(op, "exists")
        self.assertEqual(args, struct.pack(">H", 1) + b"k")

    def test_getdel_decodes_found_value(self):
        r, srv = self._client([_ok(b"\x01" + struct.pack(">I", 3) + b"abc")])
        self.assertEqual(r.kv.getdel("k"), b"abc")
        self.assertEqual(srv.requests[0][0], "getdel")

    def test_getdel_absent_is_none(self):
        r, _ = self._client([_ok(b"\x00")])
        self.assertIsNone(r.kv.getdel("k"))

    def test_getset_frames_put_layout_and_returns_old(self):
        r, srv = self._client([_ok(b"\x01" + struct.pack(">I", 3) + b"old")])
        self.assertEqual(r.kv.getset("k", "new", ttl_ms=1000), b"old")
        op, args = srv.requests[0]
        self.assertEqual(op, "getset")
        expected = (
            struct.pack(">H", 1) + b"k"
            + struct.pack(">I", 3) + b"new"
            + struct.pack(">Q", 1000)
        )
        self.assertEqual(args, expected)

    def test_ttl_decodes_signed_i64(self):
        r, srv = self._client([_ok(struct.pack(">q", -2))])
        self.assertEqual(r.kv.ttl("k"), -2)
        self.assertEqual(srv.requests[0][0], "ttl")

    def test_persist_decodes_bool(self):
        r, _ = self._client([_ok(b"\x00")])
        self.assertFalse(r.kv.persist("k"))

    def test_incr_ex_frames_delta_and_ttl(self):
        r, srv = self._client([_ok(struct.pack(">q", 7))])
        self.assertEqual(r.kv.incr_ex("k", 7, ttl_ms=5000), 7)
        op, args = srv.requests[0]
        self.assertEqual(op, "incr_ex")
        expected = struct.pack(">H", 1) + b"k" + struct.pack(">q", 7) + struct.pack(">Q", 5000)
        self.assertEqual(args, expected)

    def test_caex_frames_expected_and_ttl(self):
        r, srv = self._client([_ok(b"\x01")])
        self.assertTrue(r.kv.compare_and_expire("k", "tok", ttl_ms=60000))
        op, args = srv.requests[0]
        self.assertEqual(op, "caex")
        expected = (
            struct.pack(">H", 1) + b"k"
            + struct.pack(">I", 3) + b"tok"
            + struct.pack(">Q", 60000)
        )
        self.assertEqual(args, expected)

    def test_mget_frames_count_and_keys_and_decodes(self):
        # Response: count=3, [1][len=2]"va", [0], [1][len=0]
        payload = (
            struct.pack(">H", 3)
            + b"\x01" + struct.pack(">I", 2) + b"va"
            + b"\x00"
            + b"\x01" + struct.pack(">I", 0)
        )
        r, srv = self._client([_ok(payload)])
        vals = r.kv.mget(["a", "b", "c"])
        self.assertEqual(vals, [b"va", None, b""])
        op, args = srv.requests[0]
        self.assertEqual(op, "mget")
        expected = (
            struct.pack(">H", 3)
            + struct.pack(">H", 1) + b"a"
            + struct.pack(">H", 1) + b"b"
            + struct.pack(">H", 1) + b"c"
        )
        self.assertEqual(args, expected)


if __name__ == "__main__":
    unittest.main()
