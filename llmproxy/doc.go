// SPDX-License-Identifier: Apache-2.0

// Package llmproxy implements an OpenAI-compatible chat-completions reverse
// proxy that caches answers in semcache. Requests that hit the cache are
// answered locally (no upstream call, no cost); everything else — streaming,
// uncacheable request shapes, cache misses — is forwarded to the configured
// upstream and, on a miss, the answer is stored for next time. Like the rest
// of the repository's protocol code, the OpenAI wire format is hand-rolled
// against stdlib net/http and encoding/json — no SDK, no new dependency.
package llmproxy
