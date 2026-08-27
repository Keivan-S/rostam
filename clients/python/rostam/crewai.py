"""CrewAI memory-storage adapter backed by a Rostam collection.

Requires ``crewai`` (``pip install rostam-client[crewai]``). Provides
``RostamStorage``, a duck-typed implementation of CrewAI's classic memory
``Storage`` contract (``crewai.memory.storage.interface.Storage`` /
``crewai.memory.storage.rag_storage.RAGStorage``): ``save(value, metadata)``,
``search(query, limit, score_threshold) -> list[dict]``, ``reset()``. It plugs
into ``ShortTermMemory``, ``LongTermMemory``, ``EntityMemory`` (each accepts a
``storage=`` override) and into ``ExternalMemory(storage=...)`` the same way::

    from rostam import Rostam
    from rostam.crewai import RostamStorage
    from crewai.memory import ShortTermMemory
    from crewai.memory.external.external_memory import ExternalMemory

    client = Rostam("http://localhost:8080")
    storage = RostamStorage(client, "crew_memory", embedder=my_embed_fn)
    short_term = ShortTermMemory(storage=storage)
    external = ExternalMemory(storage=storage)

Research note (context7, crewAIInc/crewai, checked against docs.crewai.com):
CrewAI's storage layer -- both the classic ``RAGStorage``/``KnowledgeStorage``
hierarchy and the newer ``Memory``/``StorageBackend`` system that some current
docs describe -- always hands the storage backend RAW TEXT, never a
precomputed embedding. Embedding is a concern the storage/memory layer owns
for itself (``KnowledgeStorage`` holds its own ``embedder`` config and embeds
internally via its RAG client). ``RostamStorage`` follows that same shape: it
takes an ``embedder`` callable (``str -> list[float]``) at construction, like
the other Rostam adapters accept an embeddings object, and calls it inside
``save()``/``search()`` to produce the vector Rostam needs.

Uncertainty: context7's index for this library mixes two documented
generations of the CrewAI memory API -- the classic ``Storage``/``RAGStorage``
class hierarchy named in this module's target, and a newer ``Memory`` /
``StorageBackend`` protocol (default backend "lancedb") that some current
docs describe with a different call shape (``remember``/``recall`` returning
``MemoryMatch`` objects). No concrete method signature for the newer
``StorageBackend`` protocol surfaced in the indexed docs, only the classic
dict-returning ``save``/``search``/``reset`` shape did. ``RostamStorage``
targets the classic, long-stable duck-typed contract (the one third-party
storage backends have implemented for years), since that is what installs
with released ``crewai`` versions and is what this module's target classes
name.
"""

from __future__ import annotations

import threading
import time
from typing import Any, Callable, Dict, List, Optional, Sequence

from . import filters as f
from ._ids import to_uint64
from ._types import RostamError
from .rostam import Rostam

Embedder = Callable[[str], Sequence[float]]


def _scalar(meta: Dict[str, Any]) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    for k, v in (meta or {}).items():
        if isinstance(v, (str, int, float, bool)):
            out[k] = v
        elif isinstance(v, (list, tuple)) and v and all(isinstance(x, (str, int, float)) and not isinstance(x, bool) for x in v):
            out[k] = list(v)
    return out


def _translate_filter(filt: Dict[str, Any]):
    """Translate CrewAI's flat ``{field: value}`` filter dict into a Rostam
    filter predicate."""
    clauses = [f.eq(k, v) for k, v in filt.items()]
    if not clauses:
        return None
    return clauses[0] if len(clauses) == 1 else f.and_(*clauses)


class RostamStorage:
    """A CrewAI memory ``Storage`` backend over a single Rostam collection."""

    def __init__(
        self,
        client: Rostam,
        collection: str,
        *,
        embedder: Embedder,
        auto_create: bool = True,
        metric: str = "cosine",
    ):
        self._client = client
        self._collection = collection
        self._embedder = embedder
        self._auto_create = auto_create
        self._metric = metric
        self._created = False
        self._counter = 0
        self._lock = threading.Lock()

    def _ensure_collection(self, dim: int) -> None:
        if not self._auto_create or self._created:
            return
        try:
            self._client.create_collection(self._collection, dim, metric=self._metric)
        except RostamError as e:
            # Already-exists is fine; anything else propagates.
            if "exist" not in (e.message or "").lower():
                raise
        self._created = True

    def _next_id(self, value: str) -> int:
        with self._lock:
            self._counter += 1
            n = self._counter
        return to_uint64(f"{value}|{n}|{time.time_ns()}")

    def save(self, value: str, metadata: Optional[Dict[str, Any]] = None) -> None:
        vector = list(self._embedder(value))
        self._ensure_collection(len(vector))
        pid = self._next_id(value)
        self._client.upsert(self._collection, pid, vector, content=value, metadata=_scalar(metadata or {}))

    def search(
        self,
        query: str,
        limit: int = 3,
        filter: Optional[Dict[str, Any]] = None,
        score_threshold: float = 0.35,
    ) -> List[Dict[str, Any]]:
        vector = list(self._embedder(query))
        # Ensure the collection exists before the first search on an
        # auto_create store — otherwise a search preceding any save() fails
        # against a collection that was never created.
        self._ensure_collection(len(vector))
        flt = _translate_filter(filter) if filter else None
        hits = self._client.search_docs(self._collection, vector, limit, filter=flt)
        out: List[Dict[str, Any]] = []
        for h in hits:
            score = 1.0 / (1.0 + max(h.distance, 0.0))
            if score < score_threshold:
                continue
            out.append({"id": h.id, "metadata": h.metadata, "context": h.content, "score": score})
        return out

    def reset(self) -> None:
        # Clear every point in the collection. Uses scroll + per-id delete (the
        # same shape as the Semantic Router adapter's delete_all) rather than a
        # "match all" delete_by_filter — delete_by_filter always sends the
        # filter, so None would post filter:null, whose server-side semantics
        # aren't specified. The collection itself is kept, so later saves need
        # no recreate.
        #
        # Only the initial scroll's "collection never created" error is
        # suppressed; any other scroll failure, and every delete() failure
        # (left outside this try block), propagates instead of being reported
        # as a successful reset.
        try:
            docs = self._client.scroll(self._collection)
        except RostamError as e:
            msg = (e.message or "").lower()
            if "no collection" in msg or "unknown collection" in msg or "not found" in msg or "not exist" in msg:
                return
            raise
        for d in docs:
            self._client.delete(self._collection, d.id)
