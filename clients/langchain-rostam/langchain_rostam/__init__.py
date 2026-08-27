"""LangChain integration for the Rostam vector database.

Re-exports ``RostamVectorStore`` from the ``rostam-client`` package under the
conventional ``langchain_rostam`` partner-package name, so it is discoverable in
the LangChain integrations ecosystem. The implementation lives in
``rostam-client`` (``rostam.langchain``).
"""

from rostam.langchain import RostamVectorStore

__all__ = ["RostamVectorStore"]
