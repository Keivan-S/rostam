# MCP server

`rostam-server mcp` runs Rostam as a [Model Context Protocol](https://modelcontextprotocol.io/)
server over stdio, giving an MCP client (Claude Code, Claude Desktop, Cursor,
…) persistent agent memory and generic vector-DB tools. There is no daemon to
run and nothing to sign up for: the process embeds the engine directly
(`rostam.NewDirect`) and persists to a local directory, so `claude mcp add`
and the agent has durable memory a few seconds later.

Memory works with zero configuration. With no embedder set, `remember` and
`recall` run on Rostam's built-in BM25 full-text search — no embedding API
key, no external service. Pointing `ROSTAM_EMBED_ENDPOINT` at any
OpenAI-compatible `/embeddings` endpoint (OpenAI, Azure, Ollama, LM Studio,
TEI, LiteLLM) upgrades recall to hybrid dense+BM25 fusion with the same
tools and the same call shapes — nothing about how the agent uses memory
changes, only how well it ranks.

## Quickstart

**Claude Code:**

```sh
claude mcp add rostam -- rostam-server mcp
```

**Claude Desktop** (`claude_desktop_config.json`):

```json
{
  "mcpServers": {
    "rostam": {
      "command": "rostam-server",
      "args": ["mcp"]
    }
  }
}
```

**Cursor** (`.cursor/mcp.json`):

```json
{
  "mcpServers": {
    "rostam": {
      "command": "rostam-server",
      "args": ["mcp"]
    }
  }
}
```

All three launch the same binary the same way: no flags are required. Memory
persists to `~/.rostam/memory` by default, so it survives across sessions:
close the client, reopen it tomorrow, and `recall` still finds what you told
it. Only `-data ""` opts out, which runs entirely in memory.

A data directory has one writer. Sessions that come and go **sequentially**
all share `~/.rostam/memory` and see each other's memories, but two clients
running at the **same time** cannot both embed the engine over it — the second
one is refused at startup with an error saying so. To give concurrent clients
one shared memory, run a single `rostam-server` and point each client at it
with [`-connect`](#remote-mode).

### What "persists" means here

An embedded session stores memories in mmap-backed files under `-data` and
writes the index sidecar that makes them readable again when the process shuts
down cleanly — which is what happens when an MCP client closes the connection
or exits, and also on `SIGINT`/`SIGTERM`: both are handled, so the server
finishes the tool call in flight, flushes, releases the data-dir lock, and
exits 0. Reopening the same directory then restores everything.

The flush point is that clean shutdown, so a session killed outright (`SIGKILL`,
a machine losing power) loses the memories added since the last one. If that
matters, run a real `rostam-server` and use `-connect`: a server's own
durability (Raft log and snapshots) does not depend on how the MCP process
ends.

## Flags

| Flag | Default | Meaning |
|---|---|---|
| `-data` | `auto` | Embedded data directory. `auto` resolves to `~/.rostam/memory` (created if missing); `""` runs heap/ephemeral mode (nothing persists past the process); any other value is used as given. Mutually exclusive with `-connect`. |
| `-connect` | disabled | Remote mode: `host:port` of a running `rostam-server` to connect to over the binary TCP protocol, instead of embedding the engine. Mutually exclusive with `-data`. |
| `-auth-token` | — | Bearer token for `-connect`. Prefer `ROSTAM_AUTH_TOKEN` — a flag-passed secret is visible to other local users via `/proc` and shell history. |
| `-tls-ca` | — | CA bundle PEM to verify the remote server's certificate (`-connect`). |
| `-tls-cert` / `-tls-key` | — | Client certificate/key PEM for mTLS (`-connect`; both required together). |
| `-tls-server-name` | — | Expected server certificate name (SNI + verification) for `-connect`. |
| `-enable-destructive` | `false` | Register the `delete` and `delete_by_filter` tools for arbitrary collections. Without it, those two tools are absent from `tools/list` entirely — not present-but-refusing. |

`-data` and `-connect` are mutually exclusive: pass one or the other, never
both.

## Embedder configuration (environment only)

Unlike the flags above, the embedder is configured entirely through
environment variables — that matches how MCP clients pass configuration to a
server (the `env` block in the JSON snippets above), rather than through
command-line arguments baked into `args`.

| Variable | Required | Meaning |
|---|---|---|
| `ROSTAM_EMBED_ENDPOINT` | trigger | OpenAI-compatible `/embeddings` URL. Unset (with the other three also unset) means BM25-only mode. |
| `ROSTAM_EMBED_MODEL` | if endpoint set | Model id sent to the endpoint. |
| `ROSTAM_EMBED_DIM` | if endpoint set | Output embedding dimension, as an integer. |
| `ROSTAM_EMBED_API_KEY` | optional | Bearer token for the endpoint (local endpoints like Ollama typically don't need one). |

All four unset is the zero-config default: `remember`/`recall` and the
generic `search` tool run on BM25 full-text alone. Setting
`ROSTAM_EMBED_ENDPOINT` without `ROSTAM_EMBED_MODEL` or `ROSTAM_EMBED_DIM`
fails at startup with a message naming the exact missing variable — never a
silent fall-back to BM25-only.

Example: a local Ollama embedding model, wired through the Claude Desktop
config from above:

```json
{
  "mcpServers": {
    "rostam": {
      "command": "rostam-server",
      "args": ["mcp"],
      "env": {
        "ROSTAM_EMBED_ENDPOINT": "http://localhost:11434/v1/embeddings",
        "ROSTAM_EMBED_MODEL": "nomic-embed-text",
        "ROSTAM_EMBED_DIM": "768"
      }
    }
  }
}
```

!!! note "Changing embedders fails loudly, not silently"

    The embedder identity (model, dimension, BM25-only vs. hybrid) a data
    directory's memory was first created with is recorded and checked on
    every subsequent start. A mismatch — a different model, a different
    `ROSTAM_EMBED_DIM`, or adding/removing the embedder entirely — refuses to
    start rather than silently mixing embedding spaces or crashing deep
    inside the vector index on a dimension mismatch. The error names all
    three ways out: unset the embedder configuration to go back to the
    original mode, point `-data` at a different directory, or wipe the
    existing data directory to start over. There is no automatic re-embed
    migration in this release.

## Filters

Tools that accept a `filter` argument take
[`vector.Filter`](../vector/filtering.md)'s JSON form directly. Leaf values are a
**tagged union**, not a bare scalar — this is the one detail every filter
example in this page depends on:

```json
{"op": "eq", "field": "lang", "value": {"kind": "string", "str": "en"}}
```

`kind` is one of `string` (`str`), `int` (`int`), `float` (`flt`), `bool`
(`bool`), `strings` (`strs`), `ints` (`ints`), `floats` (`flts`), or `geo`
(`lat`/`lon`) — the field holding the value is named after its kind.
Composite filters nest with `and`/`or`/`not`:

```json
{
  "op": "and",
  "and": [
    {"op": "eq", "field": "lang", "value": {"kind": "string", "str": "en"}},
    {"op": "gte", "field": "year", "value": {"kind": "int", "int": 2020}}
  ]
}
```

A bare `{"lang": "en"}` is **not** a valid filter — it has to go through the
tagged form above.

## Tool reference

Every tool returns `content: [{"type": "text", "text": "<json>"}]` on
success; a tool-level failure sets `isError: true` on the same shape rather
than a protocol-level error, so a bad argument never tears down the session.

### Memory tools (always registered)

| Tool | Args | Behavior |
|---|---|---|
| `remember` | `content` (required), `namespace` (default `"default"`), `metadata` (flat JSON object) | Embeds (or stubs) `content` and upserts it into the `mcp_memory` collection. Re-remembering identical content in the same namespace upserts the same point — dedupe by `(namespace, content)` is free. Returns `{id, namespace}`. |
| `recall` | `query` (required), `namespace` (default `"default"`), `k` (default 5), `filter` (optional, ANDed with the namespace) | BM25-only or hybrid dense+BM25, per the embedder mode. Returns `{hits: [{id, content, score, metadata}]}`. |
| `forget` | `ids` (required, array of ids) | Deletes memories by id; a per-id failure doesn't abort the rest of the batch. Returns `{"deleted":[...],"missing":[...],"errors":[...]}` — `errors` is present only when at least one id failed to delete, so a fully successful call returns just `{deleted, missing}`. Same shape as `delete` below. |
| `list_memories` | `namespace` (default `"default"`), `limit` (default 50, max 500), `cursor` (from a previous call's `next_cursor`) | Pages through a namespace's memories in id order. Returns `{memories: [...], next_cursor}`. |
| `list_namespaces` | — | Returns `{namespaces: [...]}`, sorted. Derived from the memories themselves by scanning the collection, so it always agrees with what `recall`/`list_memories` can find — a namespace exists exactly as long as at least one memory carries it, and forgetting the last one makes it disappear with no separate bookkeeping step. |

Memory hits (`remember`/`recall`/`list_memories`) never carry a `distance`
field — it has no meaning for BM25-only recall or for a plain listing.

### Generic DB tools (always registered)

| Tool | Args | Behavior |
|---|---|---|
| `create_collection` | `name` (required), `dim` (required), `metric` (`cosine`\|`l2`\|`dot`, default `cosine`), `full_text` (default `true`), `persistent` (default `true`) | Creates a vector collection. `persistent` collections are stored on disk and survive a restart; `persistent: false` makes an in-memory collection that is gone when the process exits. A persistent collection is SQ8-quantized (disk-backed vector storage requires a quantizer): candidates are re-ranked exactly against the full-precision vectors, so scores are unaffected and only recall moves, marginally. |
| `upsert` | `collection` (required), `id` (required), `vector` (optional array), `content` (optional), `metadata` (optional) | Inserts or updates a point. Provide `vector` explicitly, or omit it and provide `content` with an embedder configured to auto-embed. Returns `{id}`. |
| `search` | `collection` (required), `mode` (`text`\|`dense`\|`hybrid`; default `text` with no embedder, `hybrid` with one), `query_text`, `vector`, `k` (default 10), `filter` | Text search, dense nearest-neighbor, or dense+BM25 hybrid fusion. In `dense`/`hybrid` mode, an omitted `vector` is derived by embedding `query_text` (requires an embedder). Returns `{hits: [{id, content, score, distance, metadata}]}`. |
| `get` | `collection` (required), `ids` (required), `with_vector` (default `false`) | Fetches points by id. Returns `{points: [{id, content, metadata, vector?}], missing: [...]}`. |

!!! warning "Point ids above 2^53 from a JavaScript client"

    Rostam's point ids are full-width `uint64`, but a JSON number in a
    JavaScript MCP client is an IEEE-754 double — exact only up to
    2^53−1 (9007199254740991). An id above that is rounded the moment the
    client parses it, so echoing an id from a `search` result back into
    `get`, `upsert`, or `delete` would silently name a *different* point.

    For those collections, pass the id as a **decimal string** instead:
    `{"collection": "docs", "ids": ["18446744073709551000"]}`. Every id
    argument accepts either form; results always come back as numbers.

    Memory ids (`remember`/`recall`/`list_memories` → `forget`) are generated
    inside the safe range on purpose, so this never applies to them.

**Every** generic tool — `create_collection`, `upsert`, `search`, `get`,
`delete`, and `delete_by_filter` — refuses the `mcp_memory` collection. The
memory tools above are its only interface.

For the writes, the reason is corruption: `mcp_memory`'s schema, reserved
metadata fields, and embedder-identity bootstrap belong to
`remember`/`recall`/`forget`. For the reads, the reason is namespace
isolation: `recall` and `list_memories` always scope their query to one
namespace and strip the reserved fields out of what they return, while
`search` and `get` do neither — so allowing them here would let any client
read every namespace's memories at once, and see the internal field naming
which namespace each one came from.

### Destructive tools (only with `-enable-destructive`)

| Tool | Args | Behavior |
|---|---|---|
| `delete` | `collection` (required), `ids` (required) | Deletes points by id; a per-id failure doesn't abort the rest of the batch. Returns `{"deleted":[...],"missing":[...],"errors":[...]}` — `errors` is present only when at least one id failed to delete. |
| `delete_by_filter` | `collection` (required), `filter` (required) | Deletes every point matching `filter`. A match-all (empty/zero) filter is refused even with the gate open — `-enable-destructive` authorizes targeted deletes, not a blanket wipe. |

Without `-enable-destructive`, `delete` and `delete_by_filter` are absent from
`tools/list` entirely. Memory's `forget` tool is always available regardless
of this flag — it's scoped to `mcp_memory` only.

## Remote mode

By default `rostam-server mcp` embeds the engine and owns a local data
directory. Point it at an already-running server instead with `-connect`:

```sh
rostam-server mcp -connect 127.0.0.1:7000 -auth-token "$ROSTAM_AUTH_TOKEN"
```

`-connect` speaks the binary TCP protocol via `rostam.NewClient` — the same
client the Go smart client uses — and every tool works identically against
either backend; only the backend selection differs. Authentication and TLS
follow the same conventions as the rest of the client tooling:

- **Auth**: `-auth-token`, or `ROSTAM_AUTH_TOKEN` (preferred — a flag-passed
  secret is visible via `/proc` and shell history).
- **TLS**: set any of `-tls-ca`, `-tls-cert`, `-tls-key`, `-tls-server-name`
  to enable it; plaintext stays the default when none are set. `-tls-cert`
  and `-tls-key` together enable mTLS.

See [Security](security.md) for how the target server's own auth and TLS are
configured.
