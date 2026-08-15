// SPDX-License-Identifier: Apache-2.0

// Package llmproxy implements an OpenAI-compatible chat-completions reverse
// proxy that caches answers in semcache.
//
// POST /v1/chat/completions is the cache path. A request whose identity
// (prompt plus scope: model, system prompt, tools, sampling parameters,
// caller) matches a stored answer is served locally — no upstream call, no
// generation cost — and everything else is forwarded upstream and stored on
// the way back. Streaming is cached too: a stream:true miss is relayed to the
// client live while the answer is assembled from the SSE chunks, and a
// subsequent hit is replayed as a synthetic SSE stream, in either direction
// between the streaming and non-streaming forms of the same call.
//
// Requests the cache cannot stand in for — n > 1, stream_options, multimodal
// (array-form) message content, a temperature above the configured ceiling —
// are forwarded as uncacheable passthrough, as is every route other than
// /v1/chat/completions. Responses that would be unsafe to replay (tool calls,
// a finish_reason other than "stop", a stream that never reaches [DONE]) are
// relayed to the client but never stored.
//
// Every chat response carries an x-rostam-cache header: hit, miss,
// uncacheable, or bypass (a cache lookup that errored and fell through to
// upstream). GET /stats reports the running counters.
//
// Like the rest of the repository's protocol code, the OpenAI wire format is
// hand-rolled against stdlib net/http and encoding/json — no SDK, no new
// dependency.
package llmproxy
