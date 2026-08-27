"""Semantic Router adapter tests (skipped without semantic-router), against the
stateful fake store."""

import unittest

from rostam import Rostam

try:
    from semantic_router.index.base import IndexConfig  # noqa: F401

    HAVE_SR = True
except Exception:  # pragma: no cover
    HAVE_SR = False

from _fakestore import FakeRostam


@unittest.skipUnless(HAVE_SR, "semantic-router not installed")
class SemanticRouterIndexTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.fake = FakeRostam()

    @classmethod
    def tearDownClass(cls):
        cls.fake.close()

    def setUp(self):
        from rostam.semantic_router import RostamIndex

        self.fake.docs.clear()  # isolate each test (the fake is shared per class)
        Rostam(self.fake.url).create_collection("sr", dim=3, metric="l2")
        self.client = Rostam(self.fake.url)
        self.index = RostamIndex(client=self.client, collection="sr", metric="l2")

    def test_add_and_query(self):
        self.index.add(
            embeddings=[[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]],
            routes=["greeting", "farewell"],
            utterances=["hello", "goodbye"],
            metadata_list=[{"lang": "en"}, {"lang": "en"}],
        )
        scores, route_names = self.index.query([1.0, 0.0, 0.0], top_k=2)
        self.assertEqual(route_names[0], "greeting")  # closest match
        self.assertTrue(len(scores) == len(route_names) == 2)
        self.assertTrue(scores[0] > 0)

    def test_query_route_filter(self):
        self.index.add(
            embeddings=[[1.0, 0.0, 0.0], [0.9, 0.1, 0.0]],
            routes=["greeting", "farewell"],
            utterances=["hello", "bye"],
        )
        scores, route_names = self.index.query([1.0, 0.0, 0.0], top_k=5, route_filter=["farewell"])
        self.assertEqual(route_names, ["farewell"])

    def test_delete_route(self):
        self.index.add(
            embeddings=[[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]],
            routes=["greeting", "farewell"],
            utterances=["hello", "goodbye"],
        )
        self.index.delete("greeting")
        scores, route_names = self.index.query([1.0, 0.0, 0.0], top_k=5)
        self.assertNotIn("greeting", route_names)
        self.assertIn("farewell", route_names)

    def test_describe_and_is_ready(self):
        self.assertFalse(self.index.is_ready())
        self.index.add(
            embeddings=[[1.0, 0.0, 0.0]],
            routes=["greeting"],
            utterances=["hello"],
        )
        self.assertTrue(self.index.is_ready())
        cfg = self.index.describe()
        self.assertEqual(cfg.type, "rostam")
        self.assertEqual(cfg.dimensions, 3)
        self.assertEqual(cfg.vectors, 1)

    def test_delete_all(self):
        self.index.add(
            embeddings=[[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]],
            routes=["greeting", "farewell"],
            utterances=["hello", "goodbye"],
        )
        self.index.delete_all()
        self.assertEqual(self.index.describe().vectors, 0)


if __name__ == "__main__":
    unittest.main()
