# RAG CLI

`rostam-server rag` is a batteries-included retrieval-augmented-generation
CLI built on the vector engine: point it at some files, ask questions, get
cited answers. There's no separate ingestion pipeline or vector-store
setup to wire up — `rag ingest` chunks and stores your documents, `rag ask`
retrieves the relevant chunks and asks an LLM to answer from them (citing
which chunk it used), and `rag query` does retrieval alone, with no LLM
involved. It works out of the box on plain keyword (BM25) search with zero
configuration, and upgrades to dense vector retrieval the moment you point
it at an embedding endpoint.

## Quickstart

```sh
rostam-server rag ingest ./docs
rostam-server rag ask "How does the LLM proxy decide what's cacheable?"
```

The first command chunks and indexes every recognized file under `./docs`
into a local corpus (default directory: `./.rostam-rag`). The second
retrieves the most relevant chunks and asks an LLM to answer, citing the
chunks it relied on as `[1]`, `[2]`, etc., followed by a `Sources:` list of
`source#chunk-index` for each.

To inspect retrieval without invoking an LLM at all:

```sh
rostam-server rag query "cache scoping rules"
```

This prints each hit as `[n] source#index (score)` with a one-line content
excerpt — useful for checking what would be handed to the LLM before you
spend a generation call on it.

## Flags and environment

| Flag | Env | Default | Purpose |
|---|---|---|---|
| `-data` | — | `./.rostam-rag` | Embedded data directory. Ignored when `-endpoint` is set. |
| `-endpoint` | — | (unset) | Talk to a running `rostam-server` (`host:port`) instead of the local `-data` dir. `-endpoint` takes precedence over `-data`. |
| `-corpus` | — | `default` | Corpus (collection) name — lets you keep multiple document sets side by side in one data directory. |
| `-k` | — | `5` | Number of chunks to retrieve. |
| `-chunk-size` | — | `512` | Chunk size in words (`0` uses the `rag` package default of 512). |
| `-chunk-overlap` | — | `64` | Chunk overlap in words (`0` uses the `rag` package default of 64). |
| `-embed-url` | `ROSTAM_EMBED_URL` | (unset) | Embedding endpoint URL. Needs `-embed-model` and `-embed-dim` too — with any of the three missing, retrieval falls back to BM25. |
| `-embed-model` | `ROSTAM_EMBED_MODEL` | (unset) | Embedding model id. |
| `-embed-dim` | `ROSTAM_EMBED_DIM` | (unset) | Embedding vector dimension. |
| `-llm-url` | `ROSTAM_LLM_URL` | (unset) | LLM chat-completions endpoint URL (`ask` only). |
| `-llm-model` | `ROSTAM_LLM_MODEL` | (unset) | LLM model id (`ask` only). |
| — | `ROSTAM_EMBED_KEY` | (unset) | Bearer key for the embedding endpoint. Env-only, deliberately: there is no `-embed-key` flag, so the key never lands in `/proc` or shell history. |
| — | `ROSTAM_LLM_KEY` | (unset) | Bearer key for the LLM endpoint. Env-only, same reasoning as `ROSTAM_EMBED_KEY`. |

Every flag above has an env-variable counterpart except the two secrets and
`-data`/`-endpoint`/`-corpus`/`-k`/`-chunk-size`/`-chunk-overlap`, which are
flag-only. Where both a flag and an env variable are set, the flag wins.

## Embedded vs. remote (`-endpoint`)

By default `rag` embeds the vector engine directly and owns the `-data`
directory — no server process required. Point it at an already-running
`rostam-server` instead with `-endpoint`:

```sh
rostam-server rag ingest -endpoint 127.0.0.1:7000 ./docs
rostam-server rag ask -endpoint 127.0.0.1:7000 "..."
```

This is the way to share one corpus across multiple `rag` invocations (or
processes) without each one locking its own embedded data directory.

## Retrieval: BM25 vs. dense

Retrieval defaults to BM25 full-text search — no embedder configuration
needed, works immediately after `rag ingest`. Set `-embed-url`,
`-embed-model`, and `-embed-dim` (or the equivalent `ROSTAM_EMBED_*` env
vars, plus `ROSTAM_EMBED_KEY` if the endpoint needs one) and retrieval
switches to dense kNN search over the embedded vectors instead.

Note that ingestion and retrieval must agree: chunks embedded at ingest
time are only useful for dense search if the same embedder configuration
is present at query/ask time too.

A corpus's vector dimensionality is fixed the moment it's first created.
Re-ingesting the same paths with the *same* embedder configuration (or no
embedder at all) is idempotent and safe — see
[Re-ingesting](#re-ingesting). But switching a corpus between BM25 and a
dense embedder, or changing `-embed-dim`, changes the dimension a fresh
ingest would need, and an existing corpus can't be resized in place:
`rag ingest` refuses with an error telling you to pick a new `-corpus` (or
wipe the `-data` dir) rather than silently leaving stale or mismatched
vectors behind.

Combined dense+BM25 fusion (hybrid search) is a planned upgrade, not yet
wired into the CLI — today it's one or the other, decided entirely by
whether an embedder is configured.

## Offline use with Ollama

`rag ask` needs an LLM; point `-llm-url` at a local server such as
[Ollama](https://ollama.com) to keep everything on-box:

```sh
rostam-server rag ask \
  -llm-url http://localhost:11434/v1 \
  -llm-model llama3.1 \
  "How does the LLM proxy decide what's cacheable?"
```

`ROSTAM_LLM_KEY` can stay unset for local endpoints that don't require
authentication. The same pattern works for `-embed-url` with a local
embedding model, keeping ingestion and retrieval offline too.

`rag query` never needs an LLM — it only requires `-llm-url`/`-llm-model`
(or the env equivalents) when you run `rag ask`.

## Supported file types

`rag ingest` walks each given path (file or directory, recursively) and
indexes files with these extensions, skipping everything else (and any
file that isn't valid UTF-8):

```
.txt .md .markdown
.go .py .js .ts .rs .java .c .h .cpp
.json .yaml .yml .toml
```

`rag ingest` reports how many files/chunks were indexed and lists any
skipped paths.

## Re-ingesting

Re-ingesting a path is idempotent: `rag ingest` deletes a source file's
previous chunks before writing the fresh ones, so running it again after
editing a file (or on an unchanged one) never leaves stale or duplicate
chunks behind.
