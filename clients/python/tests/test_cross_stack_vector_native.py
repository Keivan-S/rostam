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

    # ---- Phase C: batch / scroll / RAG-shaped search / hybrid / recommend ----

    def test_get_batch(self):
        r = self.r
        r.vector.create_collection("gb", dim=4, metric="cosine")
        r.vector.upsert("gb", 1, [0.1, 0.2, 0.3, 0.4], content="one", metadata={"tenant": "a"})
        r.vector.upsert("gb", 2, [0.5, 0.6, 0.7, 0.8], metadata={"tenant": "b"})
        rows = r.vector.get_batch("gb", [1, 2, 99])
        self.assertEqual(len(rows), 3)
        self.assertEqual(rows[0]["id"], 1)
        self.assertTrue(rows[0]["found"])
        self.assertEqual(len(rows[0]["vector"]), 4)
        self.assertEqual(rows[0]["content"], "one")
        self.assertEqual(rows[0]["metadata"], {"tenant": "a"})
        self.assertEqual(rows[1]["id"], 2)
        self.assertTrue(rows[1]["found"])
        self.assertEqual(rows[2]["id"], 99)
        self.assertFalse(rows[2]["found"])              # absent id: found=False, not an error

    def test_get_batch_projection(self):
        r = self.r
        r.vector.create_collection("gbp", dim=4, metric="cosine")
        r.vector.upsert("gbp", 1, [0.1, 0.2, 0.3, 0.4], metadata={"a": 1})
        rows = r.vector.get_batch("gbp", [1], with_vector=False, with_payload=False)
        self.assertTrue(rows[0]["found"])
        self.assertIsNone(rows[0]["vector"])
        self.assertEqual(rows[0]["metadata"], {})

    def test_scroll_pages_through_all_points_with_cursor(self):
        r = self.r
        r.vector.create_collection("sc", dim=4, metric="cosine")
        for i in range(1, 6):
            r.vector.upsert("sc", i, [float(i)] * 4, content=f"doc{i}")
        seen = []
        cursor = ""
        for _ in range(10):                              # bounded loop guards against an infinite scroll
            docs, cursor = r.vector.scroll("sc", limit=2, cursor=cursor)
            seen.extend(d["id"] for d in docs)
            if not cursor:
                break
        self.assertEqual(sorted(seen), [1, 2, 3, 4, 5])
        self.assertEqual(cursor, "")                      # exhausted

    def test_scroll_filter(self):
        r = self.r
        r.vector.create_collection("scf", dim=4, metric="cosine")
        r.vector.upsert("scf", 1, [0.1, 0.2, 0.3, 0.4], metadata={"tenant": "acme"})
        r.vector.upsert("scf", 2, [0.5, 0.6, 0.7, 0.8], metadata={"tenant": "beta"})
        docs, cursor = r.vector.scroll("scf", filter=f.eq("tenant", "beta"), limit=10)
        self.assertEqual([d["id"] for d in docs], [2])
        self.assertEqual(cursor, "")

    def test_search_docs(self):
        r = self.r
        r.vector.create_collection("sd", dim=4, metric="cosine")
        r.vector.upsert("sd", 1, [0.9, 0.1, 0, 0], content="alpha", metadata={"tenant": "acme"})
        r.vector.upsert("sd", 2, [0, 0, 0.9, 0.1], content="beta", metadata={"tenant": "beta"})
        docs = r.vector.search_docs("sd", [0.9, 0.1, 0, 0], k=2)
        self.assertEqual(docs[0]["id"], 1)
        self.assertEqual(docs[0]["content"], "alpha")
        self.assertEqual(docs[0]["metadata"], {"tenant": "acme"})

    def test_search_docs_filter(self):
        r = self.r
        r.vector.create_collection("sdf", dim=4, metric="cosine")
        r.vector.upsert("sdf", 1, [0.9, 0.1, 0, 0], content="alpha", metadata={"tenant": "acme"})
        r.vector.upsert("sdf", 2, [0.8, 0.2, 0, 0], content="beta", metadata={"tenant": "beta"})
        docs = r.vector.search_docs("sdf", [0.9, 0.1, 0, 0], k=5, filter=f.eq("tenant", "beta"))
        self.assertEqual([d["id"] for d in docs], [2])

    def test_search_groups(self):
        r = self.r
        r.vector.create_collection("sg", dim=4, metric="cosine")
        r.vector.upsert("sg", 1, [0.9, 0.1, 0, 0], content="a1", metadata={"cat": "x"})
        r.vector.upsert("sg", 2, [0.8, 0.2, 0, 0], content="a2", metadata={"cat": "x"})
        r.vector.upsert("sg", 3, [0, 0, 0.9, 0.1], content="b1", metadata={"cat": "y"})
        groups = r.vector.search_groups("sg", [0.9, 0.1, 0, 0], k=5, group_by="cat", group_size=2)
        self.assertGreaterEqual(len(groups), 2)           # groups formed for both cat values
        by_key = {g["key"]: g for g in groups}
        self.assertIn("x", by_key)
        self.assertIn("y", by_key)
        self.assertEqual(len(by_key["x"]["hits"]), 2)      # group_size=2 caps the "x" group at 2 hits
        self.assertEqual({h["id"] for h in by_key["x"]["hits"]}, {1, 2})
        self.assertEqual(by_key["y"]["hits"][0]["id"], 3)

    def test_hybrid_search(self):
        r = self.r
        r.vector.create_collection("hs", dim=4, metric="cosine")
        r.vector.upsert("hs", 1, [0.9, 0.1, 0, 0], sparse={"indices": [1, 5], "values": [0.5, 0.3]})
        r.vector.upsert("hs", 2, [0, 0, 0.9, 0.1], sparse={"indices": [2, 5], "values": [0.9, 0.1]})
        hits = r.vector.hybrid_search("hs", [0.9, 0.1, 0, 0], k=2,
                                      sparse={"indices": [1, 5], "values": [0.5, 0.3]})
        self.assertEqual(len(hits), 2)
        self.assertEqual(hits[0]["id"], 1)                 # dense+sparse both favor id 1
        self.assertGreater(hits[0]["score"], 0)

    def test_hybrid_search_filter_and_weighted(self):
        r = self.r
        r.vector.create_collection("hsf", dim=4, metric="cosine")
        r.vector.upsert("hsf", 1, [0.9, 0.1, 0, 0], metadata={"tenant": "acme"})
        r.vector.upsert("hsf", 2, [0.8, 0.2, 0, 0], metadata={"tenant": "beta"})
        hits = r.vector.hybrid_search("hsf", [0.9, 0.1, 0, 0], k=5,
                                      filter=f.eq("tenant", "beta"), method="weighted", alpha=0.5)
        self.assertEqual([h["id"] for h in hits], [2])

    def test_hybrid_text(self):
        r = self.r
        r.vector.create_collection("ht", dim=4, metric="cosine", full_text=True)
        r.vector.upsert("ht", 1, [0.9, 0.1, 0, 0], content="the quick brown fox jumps")
        r.vector.upsert("ht", 2, [0, 0, 0.9, 0.1], content="a lazy dog sleeps all day")
        hits = r.vector.hybrid_text("ht", [0.9, 0.1, 0, 0], "quick fox", k=2)
        self.assertEqual(len(hits), 2)
        self.assertEqual(hits[0]["id"], 1)                 # both dense and text lanes favor id 1
        self.assertGreater(hits[0]["score"], 0)

    def test_recommend_excludes_seed_and_favors_similar(self):
        r = self.r
        r.vector.create_collection("rc", dim=4, metric="cosine")
        r.vector.upsert("rc", 1, [1, 0, 0, 0])
        r.vector.upsert("rc", 2, [0.9, 0.1, 0, 0])   # close to the seed
        r.vector.upsert("rc", 3, [0, 0, 1, 0])       # far from the seed
        recs = r.vector.recommend("rc", [1], k=5)
        rec_ids = [x["id"] for x in recs]
        self.assertNotIn(1, rec_ids)                       # the positive seed itself is excluded
        self.assertEqual(rec_ids[0], 2)                    # nearest neighbour of the seed ranks first

    def test_recommend_negative_and_filter(self):
        r = self.r
        r.vector.create_collection("rcn", dim=4, metric="cosine")
        r.vector.upsert("rcn", 1, [1, 0, 0, 0])
        r.vector.upsert("rcn", 2, [0.9, 0.1, 0, 0], metadata={"tenant": "acme"})
        r.vector.upsert("rcn", 3, [0, 0, 1, 0], metadata={"tenant": "beta"})
        recs = r.vector.recommend("rcn", [1], k=5, filter=f.eq("tenant", "beta"))
        self.assertEqual([x["id"] for x in recs], [3])      # filter admits only the beta point

    def test_query_is_recommend_shaped(self):
        # query() is documented as recommend-shaped only (Phase B's QuerySpec encoder
        # builds a single-leaf RECOMMEND spec, not a general fusion/rerank tree) — it
        # must return exactly what recommend() returns for the same arguments.
        r = self.r
        r.vector.create_collection("qy", dim=4, metric="cosine")
        r.vector.upsert("qy", 1, [1, 0, 0, 0])
        r.vector.upsert("qy", 2, [0.9, 0.1, 0, 0])
        r.vector.upsert("qy", 3, [0, 0, 1, 0])
        via_query = r.vector.query("qy", [1], k=5)
        via_recommend = r.vector.recommend("qy", [1], k=5)
        self.assertEqual(via_query, via_recommend)
        self.assertNotIn(1, [x["id"] for x in via_query])

    def test_upsert_batch(self):
        r = self.r
        r.vector.create_collection("ub", dim=4, metric="cosine")
        r.vector.upsert_batch("ub", [
            {"id": 1, "vector": [1, 0, 0, 0], "content": "p1"},
            {"id": 2, "vector": [0, 1, 0, 0], "metadata": {"k": "v"}},
            {"id": 3, "vector": [0, 0, 1, 0], "sparse": {"indices": [2], "values": [0.5]}},
        ])
        got1 = r.vector.get("ub", 1)
        got2 = r.vector.get("ub", 2)
        got3 = r.vector.get("ub", 3)
        self.assertEqual(got1["content"], "p1")
        self.assertEqual(got2["metadata"], {"k": "v"})
        self.assertEqual(got3["sparse"]["indices"], [2])
        rows = r.vector.get_batch("ub", [1, 2, 3])
        self.assertEqual({row["id"] for row in rows if row["found"]}, {1, 2, 3})


if __name__ == "__main__":
    unittest.main()
