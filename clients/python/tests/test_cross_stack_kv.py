"""Python<->Go cross-stack test for the native KV client.

The KV store is only on the binary TCP protocol, so there is no HTTP path and no
fake worth trusting: the whole point is that the Python framing, the protocol-v2
auth prefix, and each op's byte layout agree with the Go server exactly. A slip
in any of them produces a wrong value or a spurious error, not a clean failure.
So this launches the real server with `-tcp` and drives every op against it.

Skipped when no server binary is found (same rule as the other cross-stack
modules): $ROSTAM_SERVER_BIN, or a `rostam-server*` built at the repo root.
"""

from __future__ import annotations

import socket
import subprocess
import tempfile
import time
import unittest

from _serverbin import find_server_bin
from rostam import RostamKV, RostamError


def _free_port():
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.bind(("127.0.0.1", 0))
    port = s.getsockname()[1]
    s.close()
    return port


_BIN, _WHY = find_server_bin()


def _wait_tcp(host, port, deadline):
    while time.time() < deadline:
        try:
            socket.create_connection((host, port), timeout=0.5).close()
            return True
        except OSError:
            time.sleep(0.1)
    return False


@unittest.skipUnless(_BIN, _WHY)
class CrossStackKVTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.http = _free_port()
        cls.tcp = _free_port()
        cls.datadir = tempfile.mkdtemp(prefix="rostam-kv-")
        cls.proc = subprocess.Popen(
            [_BIN, "-http", f"127.0.0.1:{cls.http}", "-tcp", f"127.0.0.1:{cls.tcp}",
             "-data", cls.datadir],
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
        )
        if not _wait_tcp("127.0.0.1", cls.tcp, time.time() + 20):
            cls.proc.kill()
            raise RuntimeError("rostam-server -tcp did not come up in time")
        cls.kv = RostamKV("127.0.0.1", cls.tcp)

    @classmethod
    def tearDownClass(cls):
        cls.kv.close()
        cls.proc.terminate()
        try:
            cls.proc.wait(timeout=5)
        except Exception:
            cls.proc.kill()

    def test_ping(self):
        self.assertTrue(self.kv.ping())

    def test_put_get_bytes_and_str(self):
        self.kv.put("k:bytes", b'{"coins":100}')
        self.assertEqual(self.kv.get("k:bytes"), b'{"coins":100}')
        self.kv.put("k:str", "hello")            # str encodes UTF-8
        self.assertEqual(self.kv.get("k:str"), b"hello")

    def test_miss_returns_none_not_empty(self):
        # Absent must be None, and distinct from a stored empty value.
        self.assertIsNone(self.kv.get("k:absent"))
        self.kv.put("k:empty", b"")
        self.assertEqual(self.kv.get("k:empty"), b"")

    def test_incr_from_missing_and_negative(self):
        self.assertEqual(self.kv.incr("k:ctr", 1), 1)   # missing = 0, so +1
        self.assertEqual(self.kv.incr("k:ctr", 5), 6)
        self.assertEqual(self.kv.incr("k:ctr", -2), 4)  # signed delta

    def test_delete_reports_existence(self):
        self.kv.put("k:del", "x")
        self.assertTrue(self.kv.delete("k:del"))        # existed
        self.assertFalse(self.kv.delete("k:del"))       # already gone
        self.assertIsNone(self.kv.get("k:del"))

    def test_expire_on_a_live_key(self):
        self.kv.put("k:ttl", "x")
        self.kv.expire("k:ttl", 60_000)                 # 60s — still present now
        self.assertEqual(self.kv.get("k:ttl"), b"x")

    def test_binary_safe_keys_and_values(self):
        key = bytes(range(256))
        val = bytes([0, 1, 2, 255, 254])
        self.kv.put(key, val)
        self.assertEqual(self.kv.get(key), val)


@unittest.skipUnless(_BIN, _WHY)
class CrossStackKVAuthTest(unittest.TestCase):
    """The protocol-v2 auth prefix, end to end: a token guards every op."""

    @classmethod
    def setUpClass(cls):
        cls.http = _free_port()
        cls.tcp = _free_port()
        cls.datadir = tempfile.mkdtemp(prefix="rostam-kvauth-")
        # A token makes the authenticator active even on loopback.
        cls.proc = subprocess.Popen(
            [_BIN, "-http", f"127.0.0.1:{cls.http}", "-tcp", f"127.0.0.1:{cls.tcp}",
             "-api-key", "s3cret", "-data", cls.datadir],
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
        )
        if not _wait_tcp("127.0.0.1", cls.tcp, time.time() + 20):
            cls.proc.kill()
            raise RuntimeError("authed rostam-server did not come up in time")

    @classmethod
    def tearDownClass(cls):
        cls.proc.terminate()
        try:
            cls.proc.wait(timeout=5)
        except Exception:
            cls.proc.kill()

    def test_correct_token_works(self):
        kv = RostamKV("127.0.0.1", self.tcp, auth_token="s3cret")
        try:
            kv.put("k", "v")
            self.assertEqual(kv.get("k"), b"v")
        finally:
            kv.close()

    def test_missing_token_is_unauthorized(self):
        kv = RostamKV("127.0.0.1", self.tcp)
        try:
            with self.assertRaises(RostamError) as cm:
                kv.get("k")
            self.assertEqual(cm.exception.status, 4)  # StatusUnauthorized
        finally:
            kv.close()

    def test_wrong_token_is_unauthorized(self):
        kv = RostamKV("127.0.0.1", self.tcp, auth_token="nope")
        try:
            with self.assertRaises(RostamError):
                kv.get("k")
        finally:
            kv.close()


if __name__ == "__main__":
    unittest.main()
