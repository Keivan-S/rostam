"""Mem0 adapter tests (skipped without mem0ai), against the stateful fake store."""

import unittest

from rostam import Rostam

try:
    import mem0  # noqa: F401

    HAVE_MEM0 = True
except Exception:  # pragma: no cover
    HAVE_MEM0 = False

from _fakestore import FakeRostam


@unittest.skipUnless(HAVE_MEM0, "mem0ai not installed")
class Mem0AdapterTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.fake = FakeRostam()

    @classmethod
    def tearDownClass(cls):
        cls.fake.close()

    def setUp(self):
        from rostam.mem0 import RostamVectorStore

        self.fake.docs.clear()  # isolate each test (the fake is shared per class)
        self.store = RostamVectorStore(
            collection_name="mem0", embedding_model_dims=3, url=self.fake.url, metric="l2",
        )

    def _insert(self):
        self.store.insert(
            vectors=[[0.0, 0.0, 0.0], [5.0, 5.0, 5.0]],
            payloads=[
                {"data": "alpha", "user_id": "bob"},
                {"data": "beta", "user_id": "alice"},
            ],
            ids=["m1", "m2"],
        )

    def test_insert_and_search(self):
        self._insert()
        hits = self.store.search("ignored", vectors=[0.1, 0.1, 0.1], top_k=2)
        self.assertEqual(hits[0].id, "m1")  # closest to the query
        self.assertEqual(hits[0].payload.get("data"), "alpha")
        self.assertNotIn("_mem0_id", hits[0].payload)
        self.assertTrue(all(h.score is not None and h.score > 0 for h in hits))

    def test_search_accepts_nested_vector(self):
        self._insert()
        hits = self.store.search("ignored", vectors=[[0.1, 0.1, 0.1]], top_k=1)
        self.assertEqual(hits[0].id, "m1")

    def test_get(self):
        self._insert()
        out = self.store.get("m1")
        self.assertIsNotNone(out)
        self.assertEqual(out.id, "m1")
        self.assertEqual(out.payload.get("data"), "alpha")
        self.assertNotIn("_mem0_id", out.payload)
        self.assertIsNone(out.score)

    def test_get_missing_returns_none(self):
        self.assertIsNone(self.store.get("does-not-exist"))

    def test_update(self):
        self._insert()
        self.store.update("m1", vector=[9.0, 9.0, 9.0], payload={"data": "alpha-updated", "user_id": "bob"})
        out = self.store.get("m1")
        self.assertEqual(out.payload.get("data"), "alpha-updated")

    def test_update_partial_preserves_existing(self):
        self._insert()
        # Payload-only update: vector should be preserved from the existing point.
        self.store.update("m1", payload={"data": "alpha2", "user_id": "bob"})
        out = self.store.get("m1")
        self.assertEqual(out.payload.get("data"), "alpha2")

    def test_delete(self):
        self._insert()
        self.store.delete("m1")
        self.assertIsNone(self.store.get("m1"))

    def test_list_with_filter(self):
        self._insert()
        results, _ = self.store.list(filters={"user_id": "bob"})
        self.assertEqual([r.id for r in results], ["m1"])
        self.assertIsNone(results[0].score)

    def test_list_all(self):
        self._insert()
        results, _ = self.store.list()
        self.assertEqual(sorted(r.id for r in results), ["m1", "m2"])


if __name__ == "__main__":
    unittest.main()
