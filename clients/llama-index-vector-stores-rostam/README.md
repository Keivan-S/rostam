# llama-index-vector-stores-rostam

LlamaIndex vector store integration for [Rostam](https://rostamlabs.com) — a
high-performance vector database and sub-microsecond key-value store in a single
Go engine.

```bash
pip install llama-index-vector-stores-rostam
```

```python
from llama_index.core import VectorStoreIndex, StorageContext
from llama_index.vector_stores.rostam import RostamVectorStore

vector_store = RostamVectorStore(url="http://localhost:8080", collection="docs")
storage_context = StorageContext.from_defaults(vector_store=vector_store)
index = VectorStoreIndex.from_documents(documents, storage_context=storage_context)
```

Run a Rostam server:

```bash
docker run -p 8080:8080 -e ROSTAM_API_KEY=secret ghcr.io/rostamlabs/rostam:latest
```

`RostamVectorStore` implements LlamaIndex's `BasePydanticVectorStore`. The
implementation ships in the [`rostam-client`](https://pypi.org/project/rostam-client/)
package; this package re-exports it under the conventional
`llama_index.vector_stores.rostam` namespace.

See the [Rostam docs](https://docs.rostamlabs.com/) for indexes, hybrid search,
and filtering. Apache-2.0.
