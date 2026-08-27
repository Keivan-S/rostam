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
from pydantic import Field

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

    def _ensure_collection(self, dim: int) -> None:
        """Record the embedding dim and (optionally) create the collection on
        the first ``add``. No-op on later calls."""
        if self.dimensions is not None:
            return
        if self.auto_create:
            try:
                self.client.create_collection(self.collection, dim, metric=self.metric)
            except RostamError as e:
                if "exist" not in (e.message or "").lower():
                    raise
        self.dimensions = dim

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
        flt = f.in_("route", route_filter) if route_filter else None
        hits = self.client.search_docs(self.collection, [float(x) for x in vector], top_k, filter=flt)
        scores = np.array([1.0 / (1.0 + max(h.distance, 0.0)) for h in hits])
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
