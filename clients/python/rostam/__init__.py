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

# Keep in step with [project] version in pyproject.toml. test_version.py fails
# when they drift: this constant sat at 0.1.0 through the 0.1.1 release, so
# anything asking rostam.__version__ was told the wrong release, and nothing
# noticed. It is a literal rather than an importlib.metadata lookup so that
# importing the package stays free of a dist-info read.
__version__ = "0.1.2"
