# Framework partner packages

Discovery-oriented packages so Rostam appears in the LangChain and LlamaIndex
integration registries. Each is a thin re-export of the adapter that already
ships in [`rostam-client`](./python) — the implementation lives there; these
packages only expose it under the name each framework's ecosystem expects.

| Package | Exposes | Wraps |
|---|---|---|
| [`langchain-rostam`](./langchain-rostam) | `langchain_rostam.RostamVectorStore` | `rostam.langchain.RostamVectorStore` |
| [`llama-index-vector-stores-rostam`](./llama-index-vector-stores-rostam) | `llama_index.vector_stores.rostam.RostamVectorStore` | `rostam.llamaindex.RostamVectorStore` |

Haystack needs no package — it's a catalog entry pointing at `rostam-client`
(submitted at deepset-ai/haystack-integrations#583).

## Publishing (per package)

```sh
cd clients/langchain-rostam            # or llama-index-vector-stores-rostam
python -m build                        # -> dist/*.whl and *.tar.gz
python -m twine upload dist/*          # needs your PyPI token
```

Both pin `rostam-client>=0.2` (the 0.2.0 release already contains the adapter
modules). Bump the partner package's own version on each release; it does not
have to track `rostam-client`.

## Getting listed in the registries (after publishing)

**LlamaIndex** — the package follows the `llama-index-vector-stores-*`
convention, so once it's on PyPI, open a PR to
[`run-llama/llama_index`](https://github.com/run-llama/llama_index) adding it
under `llama-index-integrations/vector_stores/` (or referencing the published
package per their current contribution guide). Include a short usage example and
note that the store implements `BasePydanticVectorStore`.

**LangChain** — publish `langchain-rostam`, then open a docs PR to
[`langchain-ai/langchain`](https://github.com/langchain-ai/langchain) adding a
notebook at `docs/docs/integrations/vectorstores/rostam.ipynb` that demonstrates
`RostamVectorStore` (add texts, similarity search, MMR, filtering). Note:
LangChain has become selective about new community vector stores — expect
review, and lead with the working notebook.

## Verifying before you publish

```sh
pip install ./clients/langchain-rostam ./clients/llama-index-vector-stores-rostam
pytest clients/langchain-rostam/tests clients/llama-index-vector-stores-rostam/tests --import-mode=importlib
```

Both re-export tests assert the class is importable under the framework name and
subclasses the right base type.
