"""Mem0 vector-store provider backed by a Rostam collection.

Requires ``mem0ai`` (``pip install rostam-client[mem0]``). Implements Mem0's
``VectorStoreBase`` (``mem0.vector_stores.base``, targeting mem0ai's current
main-branch interface, mem0ai>=0.1) so a Mem0 ``Memory`` can use Rostam as its
vector store::

    from rostam.mem0 import RostamVectorStore

    store = RostamVectorStore(
        collection_name="mem0", embedding_model_dims=1536, url="http://localhost:8080",
    )

Mem0 memory ids are opaque strings (uuid4); Rostam point ids are uint64, so
string ids are mapped via ``rostam._ids.to_uint64`` and the original string is
preserved in a reserved metadata key ``_mem0_id`` (stripped from returned
payloads, like ``_hs_id``/``_to_id`` in the Haystack/LangChain adapters).

Mem0's ``filters`` are a flat ``{field: value}`` map, with a list value meaning
membership (``field in [...]``) — see ``VectorStoreBase._apply_filters`` in the
reference implementations (e.g. ``mem0/vector_stores/faiss.py``). There is no
nested operator/logical tree like Haystack's.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence, Tuple

from mem0.vector_stores.base import VectorStoreBase
from pydantic import BaseModel

from . import filters as f
from ._ids import to_uint64
from ._types import RostamError
from .rostam import Rostam

# Reserved metadata key preserving the original Mem0 (string) memory id, since
# Rostam point ids are uint64.
_MEM0_ID = "_mem0_id"


class OutputData(BaseModel):
    """Mirrors the ``OutputData`` model each mem0 vector-store adapter defines
    locally (e.g. ``mem0/vector_stores/faiss.py``, ``.../pinecone.py``) — mem0
    has no shared/importable one, every provider ships its own copy."""

    id: Optional[str]  # memory id
    score: Optional[float]  # similarity, higher = better
    payload: Optional[Dict]  # metadata


def _scalar(meta: Dict[str, Any]) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    for k, v in (meta or {}).items():
        if isinstance(v, (str, int, float, bool)):
            out[k] = v
        elif isinstance(v, (list, tuple)) and v and all(isinstance(x, (str, int, float)) and not isinstance(x, bool) for x in v):
            out[k] = list(v)
    return out


def _translate(filters: Optional[Dict[str, Any]]):
    """Translate a Mem0 flat filter dict into a Rostam filter. A list value
    means membership; anything else is equality."""
    if not filters:
        return None
    clauses = [
        f.in_(k, list(v)) if isinstance(v, (list, tuple)) else f.eq(k, v)
        for k, v in filters.items()
    ]
    return clauses[0] if len(clauses) == 1 else f.and_(*clauses)


def _query_vector(vectors: Sequence[Any]) -> List[float]:
    """Mem0 calls ``search(query, vectors=embedding, ...)`` with a single query
    embedding — usually a flat float list, occasionally wrapped as [[...]]."""
    if vectors and isinstance(vectors[0], (list, tuple)):
        return list(vectors[0])
    return list(vectors)


class RostamVectorStore(VectorStoreBase):
    """A Mem0 ``VectorStoreBase`` over a single Rostam collection."""

    def __init__(
        self,
        collection_name: str,
        embedding_model_dims: int,
        url: str = "http://localhost:8080",
        api_key: Optional[str] = None,
        metric: str = "cosine",
    ):
        self.collection_name = collection_name
        self._dims = embedding_model_dims
        self._metric = metric
        self._client = Rostam(url, api_key=api_key)
        self.create_col(collection_name, embedding_model_dims, metric)

    # ---- collections ----

    def create_col(self, name: str, vector_size: int, distance: str = "cosine") -> None:
        """Create the collection if it doesn't already exist (idempotent)."""
        try:
            self._client.create_collection(name, vector_size, metric=distance)
        except RostamError as e:
            # Already-exists is fine; anything else propagates.
            if "exist" not in (e.message or "").lower():
                raise

    def delete_col(self) -> None:
        self._client.drop_collection(self.collection_name)

    def col_info(self) -> Dict[str, Any]:
        return {"name": self.collection_name, "dimension": self._dims, "distance": self._metric}

    def list_cols(self) -> List[str]:
        raise NotImplementedError(
            "Rostam's REST client has no list-collections endpoint; "
            "RostamVectorStore tracks only the single collection it was constructed with."
        )

    def reset(self) -> None:
        self.delete_col()
        self.create_col(self.collection_name, self._dims, self._metric)

    # ---- writes ----

    def insert(
        self,
        vectors: List[list],
        payloads: Optional[List[Dict[str, Any]]] = None,
        ids: Optional[List[str]] = None,
    ) -> None:
        if ids is None:
            raise ValueError("RostamVectorStore.insert requires ids")
        payloads = payloads or [{} for _ in vectors]
        for vec, payload, ext_id in zip(vectors, payloads, ids):
            meta = _scalar(payload or {})
            meta[_MEM0_ID] = ext_id
            self._client.upsert(self.collection_name, to_uint64(ext_id), vec, metadata=meta)

    def update(
        self,
        vector_id: str,
        vector: Optional[list] = None,
        payload: Optional[Dict[str, Any]] = None,
    ) -> None:
        if vector is None or payload is None:
            pts = self._client.get_batch(self.collection_name, [to_uint64(vector_id)])
            if not pts:
                raise ValueError(f"vector {vector_id!r} not found")
            existing = pts[0]
            if vector is None:
                vector = existing.vector
            if payload is None:
                payload = dict(existing.metadata)
                payload.pop(_MEM0_ID, None)
        meta = _scalar(payload)
        meta[_MEM0_ID] = vector_id
        self._client.upsert(self.collection_name, to_uint64(vector_id), vector, metadata=meta)

    def delete(self, vector_id: str) -> None:
        self._client.delete(self.collection_name, to_uint64(vector_id))

    # ---- reads ----

    def get(self, vector_id: str) -> Optional[OutputData]:
        pts = self._client.get_batch(self.collection_name, [to_uint64(vector_id)], with_vector=False)
        if not pts:
            return None
        meta = dict(pts[0].metadata)
        meta.pop(_MEM0_ID, None)
        return OutputData(id=vector_id, score=None, payload=meta)

    def search(
        self,
        query: str,
        vectors: List[list],
        top_k: int = 5,
        filters: Optional[Dict[str, Any]] = None,
    ) -> List[OutputData]:
        vec = _query_vector(vectors)
        hits = self._client.search_docs(self.collection_name, vec, top_k, filter=_translate(filters))
        return [self._to_output(h, scored=True) for h in hits]

    def list(
        self, filters: Optional[Dict[str, Any]] = None, top_k: Optional[int] = None
    ) -> Tuple[List[OutputData], Optional[list]]:
        page = self._client.scroll(self.collection_name, filter=_translate(filters), limit=top_k or 0)
        return [self._to_output(d) for d in page], None

    def _to_output(self, d, scored: bool = False) -> OutputData:
        meta = dict(d.metadata)
        ext_id = meta.pop(_MEM0_ID, str(d.id))
        score = (1.0 / (1.0 + max(d.distance, 0.0))) if scored else None
        return OutputData(id=ext_id, score=score, payload=meta)
