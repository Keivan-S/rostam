"""Python<->Go cross-stack test for vector ops over the native TCP protocol.

The byte layouts are pinned by test_vecwire_golden; this proves the round trip:
a real server accepts each request and returns what we expect. It covers the
JSON-carrying parts (metadata, filter, content) that the golden test leaves to a
live server on purpose, and confirms create_collection actually builds a usable
collection for HNSW, Vamana and IVF.

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
from rostam import Rostam, filters as f

_BIN, _WHY = find_server_bin()


def _free_port():
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.bind(("127.0.0.1", 0))
    p = s.getsockname()[1]
    s.close()
    return p


def _wait_tcp(host, port, deadline):
    while time.time() < deadline:
        try:
            socket.create_connection((host, port), timeout=0.5).close()
            return True
        except OSError:
            time.sleep(0.1)
    return False


@unittest.skipUnless(_BIN, _WHY)
class CrossStackVectorNativeTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.http = _free_port()
        cls.tcp = _free_port()
        cls.dir = tempfile.mkdtemp(prefix="rostam-vnative-")
        cls.proc = subprocess.Popen(
            [_BIN, "-http", f"127.0.0.1:{cls.http}", "-tcp", f"127.0.0.1:{cls.tcp}", "-data", cls.dir],
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
        )
        if not _wait_tcp("127.0.0.1", cls.tcp, time.time() + 20):
            cls.proc.kill()
            raise RuntimeError("rostam-server -tcp did not come up")
        cls.r = Rostam("127.0.0.1", cls.tcp)

    @classmethod
    def tearDownClass(cls):
        cls.r.close()
        cls.proc.terminate()
        try:
            cls.proc.wait(timeout=5)
        except Exception:
            cls.proc.kill()

    def test_create_upsert_search_get_delete(self):
        r = self.r
        r.vector.create_collection("c", dim=4, metric="cosine")
        r.vector.upsert("c", 1, [0.9, 0.1, 0, 0], content="alpha", metadata={"tenant": "acme"})
        r.vector.upsert("c", 2, [0, 0, 0.9, 0.1], content="beta", metadata={"tenant": "beta"})

        hits = r.vector.search("c", [0.9, 0.1, 0, 0], k=2)
        self.assertEqual(hits[0]["id"], 1)  # nearest to cluster A is id 1

        got = r.vector.get("c", 1)
        self.assertIsNotNone(got)
        self.assertEqual(got["metadata"], {"tenant": "acme"})   # $content lifted out
        self.assertEqual(got["content"], "alpha")
        self.assertEqual(len(got["vector"]), 4)

        self.assertTrue(r.vector.exists("c", 1))
        self.assertFalse(r.vector.exists("c", 99))
        self.assertIsNone(r.vector.get("c", 99))                # miss -> None

        self.assertTrue(r.vector.delete("c", 1))
        self.assertFalse(r.vector.delete("c", 1))               # already gone
        self.assertIsNone(r.vector.get("c", 1))

    def test_metadata_filter_round_trips(self):
        r = self.r
        r.vector.create_collection("f", dim=4, metric="cosine")
        r.vector.upsert("f", 1, [0.9, 0.1, 0, 0], metadata={"tenant": "acme"})
        r.vector.upsert("f", 2, [0.8, 0.2, 0, 0], metadata={"tenant": "beta"})
        hits = r.vector.search("f", [0.9, 0.1, 0, 0], k=5, filter=f.eq("tenant", "beta"))
        self.assertEqual([h["id"] for h in hits], [2])

    def test_insert_is_create_only(self):
        r = self.r
        r.vector.create_collection("i", dim=4, metric="cosine")
        r.vector.insert("i", 1, [0.1, 0.2, 0.3, 0.4])
        self.assertTrue(r.vector.exists("i", 1))
        from rostam import RostamError
        with self.assertRaises(RostamError):
            r.vector.insert("i", 1, [0.5, 0.5, 0.5, 0.5])   # id is live

    def test_sparse_round_trips(self):
        r = self.r
        r.vector.create_collection("sp", dim=4, metric="cosine")
        r.vector.upsert("sp", 1, [0.1, 0.2, 0.3, 0.4],
                        sparse={"indices": [3, 17], "values": [0.8, 0.4]})
        got = r.vector.get("sp", 1, with_payload=True)
        self.assertIsNotNone(got)
        self.assertEqual(got["sparse"]["indices"], [3, 17])

    def test_vamana_and_ivf_collections_build(self):
        r = self.r
        r.vector.create_collection("vam", dim=4, metric="l2",
                                   index_type="vamana", vamana_r=32, vamana_l=64)
        # IVF validates the graph params and defaults none of them (unlike
        # HNSW/Vamana), so m / ef_construction / ef_search are set explicitly —
        # the rule the docs call out for IVF tuning.
        r.vector.create_collection("ivf", dim=4, metric="cosine",
                                   m=16, ef_construction=200, ef_search=64,
                                   index_type="ivf", ivf_nlist=16, ivf_nprobe=4)
        for col in ("vam", "ivf"):
            r.vector.upsert(col, 1, [0.1, 0.2, 0.3, 0.4])
            self.assertTrue(r.vector.exists(col, 1), col)

    def test_kv_and_vector_share_one_client(self):
        r = self.r
        r.put("kv:key", b"val")
        self.assertEqual(r.get("kv:key"), b"val")
        self.assertEqual(r.incr("kv:ctr", 5), 5)
        # a vector op right after a KV op on the same pooled connection
        r.vector.create_collection("mix", dim=4, metric="cosine")
        r.vector.upsert("mix", 1, [0.1, 0.2, 0.3, 0.4])
        self.assertTrue(r.vector.exists("mix", 1))
        self.assertEqual(r.get("kv:key"), b"val")


if __name__ == "__main__":
    unittest.main()
