// SPDX-License-Identifier: Apache-2.0

package semcache

import (
	"sort"
	"strconv"
	"strings"

	"github.com/cespare/xxhash/v2"
)

// Scope is the set of request attributes that MUST match for a cached answer to
// be valid for a new prompt. Two prompts that look similar but run under a
// different system prompt, tool set, temperature, or token budget must NOT
// collide — they belong to different scopes and never see each other's cache.
type Scope struct {
	Model       string   // generation model id (e.g. "claude-opus-4-8")
	System      string   // system prompt (verbatim)
	Tools       []string // tool names available to the call (order-independent)
	Temperature float64  // sampling temperature
	MaxTokens   int      // max output tokens
	Tenant      string   // caller identity hash (prevents one client's cached answers from being served to another; never a raw credential)
	// Extra is an opaque discriminator: callers fold in any request surface
	// that must partition the cache but has no named field here. The fields
	// above cover what a semantic cache always cares about, and no fixed list
	// can keep up with a provider's sampling and formatting knobs
	// (response_format, seed, top_p, stop, frequency_penalty, …) — a request
	// asking for JSON must not be answered with prose cached from the same
	// messages. Same contract as Tenant: a hash or short tag, never a raw
	// credential, since the value rides in cache metadata.
	Extra string
}

// key returns a stable hex digest of the scope plus the embedding model. Tool
// order is normalized so two calls with the same tools in different order share
// a scope. embedModel is included so a change of embedding model partitions the
// cache (you can never compare vectors across embedding models).
func (s Scope) key(embedModel string) string {
	tools := append([]string(nil), s.Tools...)
	sort.Strings(tools)

	var b strings.Builder
	b.WriteString("emb=")
	b.WriteString(embedModel)
	b.WriteString("\x00model=")
	b.WriteString(s.Model)
	b.WriteString("\x00sys=")
	b.WriteString(s.System)
	b.WriteString("\x00tools=")
	b.WriteString(strings.Join(tools, ","))
	b.WriteString("\x00temp=")
	b.WriteString(strconv.FormatFloat(s.Temperature, 'g', 6, 64))
	b.WriteString("\x00maxtok=")
	b.WriteString(strconv.Itoa(s.MaxTokens))
	b.WriteString("\x00ten=")
	b.WriteString(s.Tenant)
	b.WriteString("\x00extra=")
	b.WriteString(s.Extra)

	return strconv.FormatUint(xxhash.Sum64String(b.String()), 16)
}
