"""CrewAI adapter tests (skipped without crewai), against the stateful fake
store."""

import unittest

from rostam import Rostam, RostamError

try:
    import crewai  # noqa: F401

    HAVE_CREWAI = True
except Exception:  # pragma: no cover
    HAVE_CREWAI = False

from _fakestore import FakeRostam


def _embed(text: str):
    # A trivial local "embedder": no network/model needed for the test.
    return [float(len(text)), 1.0, 0.0]


@unittest.skipUnless(HAVE_CREWAI, "crewai not installed")
class CrewAIStorageTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.fake = FakeRostam()

    @classmethod
    def tearDownClass(cls):
        cls.fake.close()

    def setUp(self):
        from rostam.crewai import RostamStorage

        self.fake.docs.clear()  # isolate each test (the fake is shared per class)
        self.client = Rostam(self.fake.url)
        self.client.create_collection("mem", dim=3, metric="l2")
        self.storage = RostamStorage(self.client, "mem", embedder=_embed, metric="l2")

    def test_save_search_above_threshold(self):
        self.storage.save("alpha fact", {"topic": "cats"})
        self.storage.save("a much longer beta fact", {"topic": "dogs"})

        hits = self.storage.search("alpha fact", limit=2, score_threshold=0.0)
        self.assertGreaterEqual(len(hits), 1)
        top = hits[0]
        self.assertEqual(top["context"], "alpha fact")
        self.assertEqual(top["metadata"].get("topic"), "cats")
        self.assertIn("score", top)
        self.assertGreater(top["score"], 0.0)

    def test_search_threshold_filters_far_hits(self):
        self.storage.save("alpha fact", {"topic": "cats"})
        self.storage.save("a much longer beta fact indeed", {"topic": "dogs"})

        hits = self.storage.search("alpha fact", limit=5, score_threshold=0.99)
        # Only the exact-length match (distance 0, score 1.0) clears a 0.99 bar.
        self.assertEqual([h["context"] for h in hits], ["alpha fact"])

    def test_reset_clears_collection(self):
        self.storage.save("alpha fact", {"topic": "cats"})
        self.storage.save("beta fact", {"topic": "dogs"})
        self.assertEqual(len(self.client.scroll("mem")), 2)

        self.storage.reset()
        self.assertEqual(len(self.client.scroll("mem")), 0)

        # Storage stays usable after reset.
        self.storage.save("gamma fact", {"topic": "birds"})
        self.assertEqual(len(self.client.scroll("mem")), 1)

    def test_distinct_ids_for_repeated_saves(self):
        self.storage.save("same text", {})
        self.storage.save("same text", {})
        self.assertEqual(len(self.client.scroll("mem")), 2)

    def test_search_before_any_save_on_auto_create_store(self):
        from rostam.crewai import RostamStorage

        # A fresh collection name that no save() has touched yet: search()
        # must create the collection itself instead of failing against one
        # that was never created.
        storage = RostamStorage(self.client, "mem_fresh", embedder=_embed, metric="l2")
        self.assertEqual(storage.search("anything", limit=3, score_threshold=0.0), [])

    def test_search_with_filter(self):
        self.storage.save("alpha fact", {"topic": "cats"})
        self.storage.save("beta fact", {"topic": "dogs"})

        hits = self.storage.search("alpha fact", limit=5, filter={"topic": "cats"}, score_threshold=0.0)
        self.assertEqual([h["context"] for h in hits], ["alpha fact"])

    def test_reset_propagates_scroll_error(self):
        def boom(*a, **kw):
            raise RostamError("internal error: boom")

        self.client.scroll = boom
        with self.assertRaises(RostamError):
            self.storage.reset()

    def test_reset_propagates_delete_error(self):
        self.storage.save("alpha fact", {"topic": "cats"})
        self.storage.save("beta fact", {"topic": "dogs"})

        calls = []
        orig_delete = self.client.delete

        def flaky_delete(collection, pid):
            calls.append(pid)
            if len(calls) == 2:
                raise RostamError("internal error: delete failed")
            return orig_delete(collection, pid)

        self.client.delete = flaky_delete
        try:
            with self.assertRaises(RostamError):
                self.storage.reset()
        finally:
            self.client.delete = orig_delete
        self.assertEqual(len(calls), 2)


if __name__ == "__main__":
    unittest.main()
