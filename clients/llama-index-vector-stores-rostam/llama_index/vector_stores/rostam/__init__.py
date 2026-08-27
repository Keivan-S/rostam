"""LlamaIndex vector store integration for the Rostam vector database.

Re-exports ``RostamVectorStore`` from the ``rostam-client`` package under the
conventional ``llama_index.vector_stores.rostam`` namespace, so it is
discoverable in the LlamaIndex integrations ecosystem. The implementation lives
in ``rostam-client`` (``rostam.llamaindex``).
"""

from rostam.llamaindex import RostamVectorStore

__all__ = ["RostamVectorStore"]
