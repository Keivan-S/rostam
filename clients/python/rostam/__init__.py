"""Rostam Python client — a dependency-free REST SDK for the Rostam vector store.

The core client (``RostamClient``) uses only the standard library. The optional
LangChain adapter lives in ``rostam.langchain`` and requires the ``langchain``
extra.
"""

from . import filters
from .client import Document, Group, MultiResult, Point, RostamClient, RostamError, ScrollPage, SearchResult
from .embeddings import Embedder, FunctionEmbedder, OpenAIEmbedder, TextStore

__all__ = [
    "RostamClient",
    "RostamError",
    "SearchResult",
    "Document",
    "Point",
    "Group",
    "MultiResult",
    "ScrollPage",
    "filters",
    "Embedder",
    "FunctionEmbedder",
    "OpenAIEmbedder",
    "TextStore",
]

__version__ = "0.1.0"
