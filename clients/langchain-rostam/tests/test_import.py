def test_reexport():
    from langchain_rostam import RostamVectorStore
    from langchain_core.vectorstores import VectorStore

    assert issubclass(RostamVectorStore, VectorStore)
