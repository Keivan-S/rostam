# langchain-rostam

LangChain [`VectorStore`](https://python.langchain.com/docs/integrations/vectorstores/)
integration for [Rostam](https://rostamlabs.com) — a high-performance vector
database and sub-microsecond key-value store in a single Go engine.

```bash
pip install langchain-rostam
```

```python
from langchain_rostam import RostamVectorStore

store = RostamVectorStore(
    url="http://localhost:8080",
    collection="docs",
    embedding=my_embeddings,   # any LangChain Embeddings
)
store.add_texts(["Rostam is a vector DB and KV store in one Go engine."])
docs = store.similarity_search("what is rostam?", k=5)
```

Run a Rostam server:

```bash
docker run -p 8080:8080 -e ROSTAM_API_KEY=secret ghcr.io/rostamlabs/rostam:latest
```

`RostamVectorStore` implements the standard LangChain `VectorStore` interface
(add/search/MMR/delete + metadata filtering). The implementation ships in the
[`rostam-client`](https://pypi.org/project/rostam-client/) package; this package
re-exports it under the conventional `langchain_rostam` name.

See the [Rostam docs](https://docs.rostamlabs.com/) for indexes, hybrid search,
and filtering. Apache-2.0.
