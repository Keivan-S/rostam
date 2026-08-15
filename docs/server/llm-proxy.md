# LLM caching proxy

`rostam-server llm-proxy` runs Rostam as an OpenAI-compatible caching reverse
proxy: point your existing OpenAI SDK or `curl` calls at it instead of
`api.openai.com`, and every chat-completions request that repeats a prior
prompt is answered from a local semantic cache instead of hitting the
upstream API — zero generation cost, no round trip. Everything else (every
other `/v1/*` route, and any chat request the cache can't safely answer) is
forwarded to the real upstream verbatim.

With no embedding endpoint configured it still does useful work out of the
box: exact byte-identical prompts are cached and served locally, no API key
or embedding model required. Point it at an embedder and it upgrades to
semantic matching — near-duplicate prompts (different whitespace, minor
rewording) hit the cache too.

## Quickstart

Change one line in the OpenAI Python SDK:

```python
from openai import OpenAI

client = OpenAI(base_url="http://localhost:8484/v1", api_key="sk-...")
resp = client.chat.completions.create(
    model="gpt-4o-mini",
    messages=[{"role": "user", "content": "What is the capital of France?"}],
)
```

Or with `curl`:

```sh
rostam-server llm-proxy &

curl http://localhost:8484/v1/chat/completions \
  -H "Authorization: Bearer $OPENAI_API_KEY" \
  -H "Content-Type: application/json" \
  -d '{"model":"gpt-4o-mini","messages":[{"role":"user","content":"What is the capital of France?"}]}'
```

The response carries an `x-rostam-cache` header (`hit`, `miss`, `uncacheable`,
or `bypass`) so you can see, at a glance, which requests are actually costing
you generation tokens.

## Exact vs. semantic mode

The mode is decided at startup from whether an embedder is configured — see
[Embedder configuration](#embedder-configuration-environment-only) below:

- **Exact mode** (default, no `ROSTAM_EMBED_*` env set): prompts are matched
  with a deterministic stub embedder at a `0.999` cosine-similarity floor, so
  only a byte-identical prompt (within the same scope) is ever treated as a
  hit.
- **Semantic mode** (`ROSTAM_EMBED_ENDPOINT` set): prompts are matched with a
  real hosted embedder at a `0.97` floor by default, so a paraphrased or
  lightly-edited prompt can still hit the cache.

`-threshold` overrides either mode's default floor.

### What forms the cache key

A cache entry is scoped to more than just the prompt text. Two requests only
match if **both** their prompt and their scope agree:

- **Prompt**: every non-system message's role and content, concatenated in
  order. Only the plain-string form of `content` is used — a multimodal
  message (the array-of-parts form: images, etc.) can't be reduced to text,
  so it's never cached.
- **Scope**: `model`, the concatenated `system` message(s), the requested
  `tools`' names, the effective `temperature`, `max_tokens`, a hash of the
  caller's credential headers (tenancy — see below), the embedder identity (so
  switching embedders never mixes cached answers from a different embedding
  space), and a hash of **every remaining field in the request body**.

That last part matters: `response_format`, `seed`, `top_p`, `stop`,
`frequency_penalty` and anything else a provider adds all partition the cache,
so a request asking for JSON is never answered with prose cached from the
identical messages. Only `messages` (that's the prompt) and `stream` /
`stream_options` are excluded — the streaming and non-streaming forms of one
call deliberately share an answer. The hash is computed over the parsed body,
so key order and whitespace don't matter.

Two prompts that are textually identical but differ in model, system prompt,
tools, any sampling parameter, or caller never share a cache entry.

## Flags

| Flag | Default | Meaning |
|---|---|---|
| `-listen` | `127.0.0.1:8484` | HTTP listen address for the proxy. |
| `-upstream` | `https://api.openai.com` | Upstream OpenAI-compatible API base URL. |
| `-collection` | `llm-cache` | Cache collection name (created if absent). |
| `-threshold` | `0` | Cosine-similarity hit floor. `0` = mode default: `0.999` in exact mode, `0.97` in semantic mode. |
| `-max-temp` | `1.0` | Do not cache chat requests with a temperature above this. |
| `-ttl` | `168h` | Per-entry cache expiry. |
| `-data` | `auto` | Embedded data directory. `auto` resolves to `~/.rostam/llmcache` (created if missing); `""` runs heap/ephemeral mode; any other value is used as given. Mutually exclusive with `-connect`. |
| `-connect` | disabled | Connect to a remote `rostam-server` instead of embedding the engine (`host:port`). Mutually exclusive with `-data`. |
| `-auth-token` | — | Bearer token for `-connect`. Prefer `ROSTAM_AUTH_TOKEN` — a flag-passed secret is visible to other local users via `/proc` and shell history. |
| `-tls-ca` | — | CA bundle PEM to verify the remote server's certificate (`-connect`). |
| `-tls-cert` / `-tls-key` | — | Client certificate/key PEM for mTLS (`-connect`; both required together). |
| `-tls-server-name` | — | Expected server certificate name (SNI + verification) for `-connect`. |
| `-insecure` | `false` | Acknowledge running with a non-loopback `-listen`. The proxy has no auth layer of its own (it relays whatever `Authorization` header the client sends straight to upstream), so binding it to a reachable address without this flag is refused at startup. Dev/trusted-network use only. |

`-data` and `-connect` are mutually exclusive: pass one or the other, never
both.

## Embedder configuration (environment only)

Like the [MCP server](mcp.md#embedder-configuration-environment-only), the
embedder is configured entirely through environment variables, using the
same `ROSTAM_EMBED_*` variables:

| Variable | Required | Meaning |
|---|---|---|
| `ROSTAM_EMBED_ENDPOINT` | trigger | OpenAI-compatible `/embeddings` URL. Unset (with the other three also unset) means exact-match mode. |
| `ROSTAM_EMBED_MODEL` | if endpoint set | Model id sent to the endpoint. |
| `ROSTAM_EMBED_DIM` | if endpoint set | Output embedding dimension, as an integer. |
| `ROSTAM_EMBED_API_KEY` | optional | Bearer token for the endpoint (local endpoints like Ollama typically don't need one). |

All four unset is the zero-config default: exact-match caching, no embedding
API key required. Setting `ROSTAM_EMBED_ENDPOINT` without
`ROSTAM_EMBED_MODEL` or `ROSTAM_EMBED_DIM` fails at startup naming the exact
missing variable.

Example: a local Ollama embedding model for semantic matching:

```sh
ROSTAM_EMBED_ENDPOINT=http://localhost:11434/v1/embeddings \
ROSTAM_EMBED_MODEL=nomic-embed-text \
ROSTAM_EMBED_DIM=768 \
rostam-server llm-proxy
```

## What's never cached

The following requests are always forwarded to upstream and never answered
from (or written to) the cache:

- A response with `tool_calls` present.
- `n` greater than 1 (multiple choices can't be represented by one cached
  answer).
- `stream_options` (e.g. `{"include_usage": true}`) — it asks for a stream
  shape a cache replay cannot reproduce.
- Non-string message content — the array-of-parts (multimodal) form.
- A response whose `finish_reason` isn't `"stop"` (truncated, length-limited,
  content-filtered, etc.).
- A streamed response that aborts mid-stream (never reaches a clean
  `[DONE]`).

## Temperature and cacheability

OpenAI treats an omitted `temperature` field as `1.0`, and the proxy follows
the same rule for cache scoping. Combined with the default `-max-temp 1.0`,
that means a typical request with no explicit `temperature` **is cached by
default** — high-temperature, "creative" requests are exactly the ones most
API traffic sends without setting the field at all.

If you only want to cache fully-deterministic requests, tighten the ceiling:

```sh
rostam-server llm-proxy -max-temp 0
```

With `-max-temp 0`, only requests that explicitly set `"temperature": 0`
are cached; anything with a higher (or omitted, i.e. `1.0`) temperature
always passes through.

## Tenancy

Cache entries are scoped by caller: the proxy hashes the request's credential
headers — `Authorization`, `api-key` (Azure OpenAI) and `x-api-key` — never
storing any of them raw, and includes that hash in the scope key, so one
client's cached answers are never served to another client using a different
key. All three are covered because an OpenAI-compatible surface is not only
OpenAI: a proxy that looked at `Authorization` alone would put every Azure
caller in the same tenant. Requests carrying none of the three share one "no
auth" tenant scope.

## `/stats`

`GET /stats` returns the proxy's running counters as JSON:

```json
{
  "hits": 42,
  "misses": 130,
  "stored": 118,
  "uncacheable": 9,
  "tokens_saved": 15872,
  "mode": "exact"
}
```

`tokens_saved` sums the completion-token count of every cache hit — the
generation cost you didn't pay. `mode` is `"exact"` or `"semantic"`, fixed
for the life of the process.

## Remote mode

By default `rostam-server llm-proxy` embeds the engine and owns a local data
directory. Point it at an already-running server instead with `-connect`,
the same way as the [MCP server](mcp.md#remote-mode):

```sh
rostam-server llm-proxy -connect 127.0.0.1:7000 -auth-token "$ROSTAM_AUTH_TOKEN"
```

This is also how multiple proxy processes share one cache: a `-data`
directory has a single writer (see below), so concurrent proxies sharing a
cache must all `-connect` to one `rostam-server` rather than each embedding
their own.

## Honest limits

- **Cache entries do not survive a proxy restart in v1.** The semantic-cache
  collection backing the proxy is created as an in-memory (non-persistent)
  vector collection regardless of `-data` — a real `-data` directory still
  claims the directory and persists other engine state, but the cache itself
  starts empty on every restart. Don't rely on it as a durable answer store.
- **One `-data` directory, one writer.** Embedded mode locks the data
  directory (the same mechanism the MCP server uses); a second `llm-proxy`
  process pointed at the same directory is refused at startup. Run one
  process per `-data` directory, or share a cache across processes with
  `-connect` instead.
- **OpenAI-compatible surface only.** The proxy understands the
  `/v1/chat/completions` request/response shape (including its streaming SSE
  variant) well enough to cache it; every other route is opaque passthrough.
  It does not cache `/v1/embeddings`, `/v1/completions`, or any
  provider-specific endpoint.
- **No single-flight.** Concurrent identical requests that all arrive before
  the first one's answer is stored each reach the upstream once — the cache
  deduplicates across time, not within a burst. A cold start that fans out N
  copies of the same prompt pays for N generations.
- **Streaming replay doesn't preserve upstream cadence.** A cache hit on a
  `stream: true` request is replayed as a synthetic SSE stream with a fixed
  three-chunk shape (role, content, finish) — fewer, larger chunks than the
  original upstream stream produced. A client reading only the assembled
  content sees the same answer; a client sensitive to chunk-by-chunk timing
  or granularity will notice the difference.
