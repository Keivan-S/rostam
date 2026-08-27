"""DSPy retriever adapter backed by a Rostam collection.

Requires ``dspy`` (``pip install rostam-client[dspy]``). Targets DSPy's current
retriever pattern (DSPy >=2.5, verified against 3.1.3 docs): a plain
``dspy.Module`` whose ``forward`` returns ``dspy.Prediction(passages=...)``,
callable directly like ``dspy.retrievers.Embeddings`` — the pattern the current
DSPy RAG tutorial recommends — rather than the legacy ``dspy.Retrieve`` class,
which draws from a single globally configured retrieval model via
``dspy.settings.configure(rm=...)`` and doesn't compose with a per-collection
client.

DSPy retrievers receive a query STRING; Rostam search needs a query VECTOR, so
``RostamRetriever`` takes an embedder callable (``str -> list[float]``) at
construction and uses it to embed the incoming query.

    from rostam import Rostam
    from rostam.dspy import RostamRetriever

    client = Rostam("http://localhost:8080")
    retrieve = RostamRetriever(client, "docs", embedder=embed_fn, k=5)
    retrieve.index(["doc one text", "doc two text"])

    class RAG(dspy.Module):
        def __init__(self):
            self.retrieve = retrieve
            self.answer = dspy.ChainOfThought("question, context -> answer")

        def forward(self, question):
            passages = self.retrieve(question).passages
            return self.answer(question=question, context=passages)
"""

from __future__ import annotations

import hashlib
from typing import Any, Callable, Dict, List, Optional, Sequence

import dspy

from ._ids import to_uint64
from ._types import RostamError
from .rostam import Rostam

Embedder = Callable[[str], List[float]]


def _scalar(meta: Dict[str, Any]) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    for k, v in (meta or {}).items():
        if isinstance(v, (str, int, float, bool)):
            out[k] = v
        elif isinstance(v, (list, tuple)) and v and all(
            isinstance(x, (str, int, float)) and not isinstance(x, bool) for x in v
        ):
            out[k] = list(v)
    return out


class RostamRetriever(dspy.Module):
    """A DSPy retriever module over a single Rostam collection.

    Call it directly (``retriever(query)``) — ``dspy.Module.__call__``
    dispatches to :meth:`forward` — to get back a ``dspy.Prediction`` whose
    ``passages`` is a list of the top-k matching document contents.
    """

    def __init__(
        self,
        client: Rostam,
        collection: str,
        *,
        embedder: Embedder,
        k: int = 10,
        auto_create: bool = True,
        metric: str = "cosine",
        filter: Optional[Dict[str, Any]] = None,
    ):
        super().__init__()
        self._client = client
        self._collection = collection
        self._embedder = embedder
        self._k = k
        self._auto_create = auto_create
        self._metric = metric
        self._filter = filter
        self._created = False

    def forward(
        self, query: str, k: Optional[int] = None, filter: Optional[Dict[str, Any]] = None
    ) -> "dspy.Prediction":
        vector = self._embedder(query)
        hits = self._client.search_docs(
            self._collection,
            vector,
            k if k is not None else self._k,
            filter=filter if filter is not None else self._filter,
        )
        return dspy.Prediction(passages=[h.content for h in hits])

    def index(
        self,
        texts: Sequence[str],
        *,
        embeddings: Optional[Sequence[List[float]]] = None,
        ids: Optional[Sequence[str]] = None,
        metadatas: Optional[Sequence[Dict[str, Any]]] = None,
    ) -> List[str]:
        """Load documents into the backing collection.

        Embeds via the configured embedder when ``embeddings`` aren't supplied.
        Returns the ids used (generated deterministically from content when
        ``ids`` is omitted)."""
        texts = list(texts)
        if not texts:
            return []
        if embeddings is not None and len(embeddings) != len(texts):
            raise ValueError(
                f"embeddings length ({len(embeddings)}) must match texts length ({len(texts)})"
            )
        if ids is not None and len(ids) != len(texts):
            raise ValueError(f"ids length ({len(ids)}) must match texts length ({len(texts)})")
        if metadatas is not None and len(metadatas) != len(texts):
            raise ValueError(
                f"metadatas length ({len(metadatas)}) must match texts length ({len(texts)})"
            )
        vectors = list(embeddings) if embeddings is not None else [self._embedder(t) for t in texts]
        if vectors:
            self._ensure_collection(len(vectors[0]))
        if ids is None:
            ids = [hashlib.blake2b(t.encode("utf-8"), digest_size=16).hexdigest() for t in texts]
        else:
            ids = list(ids)
        metadatas = list(metadatas) if metadatas is not None else [{} for _ in texts]
        for text, vec, meta, ext_id in zip(texts, vectors, metadatas, ids):
            self._client.upsert(
                self._collection, to_uint64(ext_id), vec, content=text, metadata=_scalar(meta or {})
            )
        return ids

    def _ensure_collection(self, dim: int) -> None:
        """Create the collection on first index() call (idempotent). No-op if
        auto_create is off or we already created it this session."""
        if not self._auto_create or self._created:
            return
        try:
            self._client.create_collection(self._collection, dim, metric=self._metric)
        except RostamError as e:
            # Already-exists is fine; anything else propagates. Match
            # "already exist" (not the looser "exist"), so an unrelated
            # "does not exist" error can't be swallowed here and mark the
            # collection created when it wasn't (the later upsert would then
            # fail with a confusing error).
            if "already exist" not in (e.message or "").lower():
                raise
        self._created = True
