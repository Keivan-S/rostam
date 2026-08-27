"""Semantic Router ``BaseIndex`` implementation backed by a Rostam collection.

Requires ``semantic-router`` (``pip install rostam-client[semantic-router]``).
Targets the current (v1.x) ``BaseIndex`` interface: ``add`` stores each route's
utterances as embedded points, ``query`` performs a kNN search and returns the
matching route names, and ``delete``/``delete_index``/``delete_all`` remove
routes or the whole collection.

    from semantic_router.routers import SemanticRouter
    from rostam import Rostam
    from rostam.semantic_router import RostamIndex

    index = RostamIndex(client=Rostam("http://localhost:8080"), collection="routes")
    router = SemanticRouter(encoder=encoder, routes=routes, index=index)

Each utterance is stored as a Rostam point with ``content`` set to the
utterance text and ``metadata`` carrying ``{"route": <route name>, **extra}``.
Point ids are derived from ``f"{route}:{utterance}"`` via
``rostam._ids.to_uint64``, so re-adding the same (route, utterance) pair is
idempotent. ``function_schemas`` is accepted for interface compatibility but
not persisted (Rostam metadata values are scalars/scalar-lists; a function
schema is an arbitrary nested object).
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from semantic_router.index.base import BaseIndex, IndexConfig
from semantic_router.schema import Utterance
from pydantic import Field, PrivateAttr

from . import filters as f
from ._ids import to_uint64
from ._types import RostamError
from .rostam import Rostam


class RostamIndex(BaseIndex):
    """A Semantic Router index over a single Rostam collection."""

    type: str = "rostam"
    client: Any = Field(default=None, exclude=True)
    collection: str = "semantic-router"
    auto_create: bool = True
    metric: str = "cosine"
    # Tracks whether this index has created its backing collection, kept
    # separate from ``dimensions`` because ``BaseRouter.__init__`` can set
    # ``dimensions`` on the index before ``add()`` ever runs -- gating
    # creation on ``dimensions is not None`` would then skip create_collection
    # forever and every upsert would fail against a collection that never
    # exists.
    _created: bool = PrivateAttr(default=False)

    def _ensure_collection(self, dim: int) -> None:
        """Create the collection on the first ``add`` (once). No-op on later
        calls."""
        if self._created:
            return
        if self.auto_create:
            try:
                self.client.create_collection(self.collection, dim, metric=self.metric)
            except RostamError as e:
                if "exist" not in (e.message or "").lower():
                    raise
        self.dimensions = dim
        self._created = True

    def add(
        self,
        embeddings: List[List[float]],
        routes: List[str],
        utterances: List[Any],
        function_schemas: Optional[List[Dict[str, Any]]] = None,
        metadata_list: List[Dict[str, Any]] = [],
        **kwargs: Any,
    ) -> None:
        if not embeddings:
            return
        self._ensure_collection(len(embeddings[0]))
        metas = metadata_list or [{} for _ in utterances]
        for emb, route, utt, meta in zip(embeddings, routes, utterances, metas):
            point_meta = dict(meta or {})
            point_meta["route"] = route
            pid = to_uint64(f"{route}:{utt}")
            self.client.upsert(self.collection, pid, emb, content=str(utt), metadata=point_meta)

    def query(
        self,
        vector: np.ndarray,
        top_k: int = 5,
        route_filter: Optional[List[str]] = None,
        sparse_vector: Optional[Any] = None,
    ) -> Tuple[np.ndarray, List[str]]:
        """Note: this adapter assumes the collection's configured metric is
        ``cosine`` (this class's default). Rostam returns
        ``distance = 1 - cosine_similarity`` for that metric, and Semantic
        Router's route thresholds are compared against cosine similarity, so
        the score returned here is ``1.0 - distance``. Using a non-cosine
        ``metric`` will make these scores meaningless against Semantic
        Router's threshold semantics."""
        flt = f.in_("route", route_filter) if route_filter else None
        hits = self.client.search_docs(self.collection, [float(x) for x in vector], top_k, filter=flt)
        scores = np.array([1.0 - h.distance for h in hits])
        route_names = [h.metadata.get("route", "") for h in hits]
        return scores, route_names

    def delete(self, route_name: str) -> None:
        self.client.delete_by_filter(self.collection, f.eq("route", route_name))

    def delete_all(self) -> None:
        for d in self.client.scroll(self.collection):
            self.client.delete(self.collection, d.id)

    def delete_index(self) -> None:
        try:
            self.client.drop_collection(self.collection)
        except RostamError:
            pass
        self.dimensions = None

    def describe(self) -> IndexConfig:
        try:
            vectors = len(self.client.scroll(self.collection))
        except RostamError:
            vectors = 0
        return IndexConfig(type=self.type, dimensions=self.dimensions or 0, vectors=vectors)

    def is_ready(self) -> bool:
        return self.dimensions is not None

    def get_utterances(self, include_metadata: bool = False) -> List[Utterance]:
        """Reconstruct the persisted (route, utterance) pairs from the
        collection.

        The inherited ``BaseIndex.get_utterances`` returns ``[]`` because it
        reads ``self.index``, which this adapter never sets -- leaving
        ``RouterConfig.from_index``/``auto_sync``/route-diff operations to
        treat a populated collection as empty. Scroll the collection instead
        and rebuild an ``Utterance`` per point from its ``content`` (the
        stored utterance text) and ``metadata["route"]``."""
        try:
            docs = self.client.scroll(self.collection)
        except RostamError:
            return []
        out: List[Utterance] = []
        for d in docs:
            meta = dict(d.metadata)
            route = meta.pop("route", "")
            kwargs: Dict[str, Any] = {"route": route, "utterance": d.content}
            if include_metadata:
                kwargs["metadata"] = meta
            out.append(Utterance(**kwargs))
        return out
