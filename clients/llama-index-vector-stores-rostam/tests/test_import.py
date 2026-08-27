def test_reexport():
    from llama_index.vector_stores.rostam import RostamVectorStore
    from llama_index.core.vector_stores.types import BasePydanticVectorStore

    assert issubclass(RostamVectorStore, BasePydanticVectorStore)
