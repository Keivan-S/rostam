"""DSPy adapter tests (skipped without dspy installed), against the stateful
fake store."""

import unittest

from rostam import Rostam

try:
    import dspy

    HAVE_DSPY = True
except Exception:  # pragma: no cover
    HAVE_DSPY = False

from _fakestore import FakeRostam


def _embed(text: str):
    """Deterministic 3-dim embedding: text-length feature, no model needed."""
    return [float(len(text)), 1.0, 0.0]


@unittest.skipUnless(HAVE_DSPY, "dspy not installed")
class DSPyAdapterTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.fake = FakeRostam()

    @classmethod
    def tearDownClass(cls):
        cls.fake.close()

    def setUp(self):
        from rostam.dspy import RostamRetriever

        self.fake.docs.clear()  # isolate each test (the fake is shared per class)
        self.client = Rostam(self.fake.url)
        self.client.create_collection("dspy", dim=3, metric="l2")
        self.retriever = RostamRetriever(self.client, "dspy", embedder=_embed, k=3)

    def test_index_and_call_returns_prediction(self):
        self.retriever.index(["alpha", "betabeta"])
        out = self.retriever("alpha")
        self.assertIsInstance(out, dspy.Prediction)
        # Closest to the query first ("alpha" is an exact length/vector match).
        self.assertEqual(out.passages[0], "alpha")
        self.assertEqual(set(out.passages), {"alpha", "betabeta"})

    def test_forward_respects_k(self):
        self.retriever.index(["alpha", "betabeta"])
        out = self.retriever.forward("alpha", k=1)
        self.assertEqual(out.passages, ["alpha"])

    def test_index_with_explicit_ids_and_metadata(self):
        ids = self.retriever.index(["cat food"], ids=["doc-1"], metadatas=[{"topic": "cats"}])
        self.assertEqual(ids, ["doc-1"])
        out = self.retriever("cat food")
        self.assertEqual(out.passages[0], "cat food")

    def test_index_generates_ids_when_omitted(self):
        ids = self.retriever.index(["one", "two"])
        self.assertEqual(len(ids), 2)
        self.assertEqual(len(set(ids)), 2)  # distinct, deterministic ids

    def test_index_empty_is_noop(self):
        self.assertEqual(self.retriever.index([]), [])

    def test_index_auto_creates_collection(self):
        from rostam.dspy import RostamRetriever

        calls = []
        orig = self.client.create_collection

        def spy(name, dim, **kw):
            calls.append((name, dim, kw))
            return orig(name, dim, **kw)

        self.client.create_collection = spy
        r = RostamRetriever(self.client, "dspy_auto", embedder=_embed, k=2, metric="l2")
        r.index(["hello world"])
        self.assertEqual(len(calls), 1)
        self.assertEqual(calls[0][0], "dspy_auto")
        self.assertEqual(calls[0][1], 3)  # dim inferred from the embedder

        r.index(["second doc"])
        self.assertEqual(len(calls), 1, "create_collection called more than once")

        out = r("hello world")
        self.assertEqual(out.passages[0], "hello world")


if __name__ == "__main__":
    unittest.main()
