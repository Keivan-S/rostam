// SPDX-License-Identifier: Apache-2.0

// Package semcache is a semantic cache for LLM responses backed by a Rostam
// vector collection: it embeds an incoming prompt, finds the nearest prior
// prompt within a similarity threshold, and serves the stored answer — turning
// a near-duplicate request into zero generation tokens. It wraps the public
// rostam.Store interface and works against an embedded or remote engine.
package semcache
