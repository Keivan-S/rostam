// SPDX-License-Identifier: Apache-2.0

package semcache

import "testing"

func TestScopeKeyStableAndSensitive(t *testing.T) {
	base := Scope{Model: "claude-x", System: "be brief", Tools: []string{"a", "b"}, Temperature: 0, MaxTokens: 1024}

	// Stable across calls and tool ordering.
	if base.key("emb-1") != base.key("emb-1") { //nolint:staticcheck // SA4000: intentional self-comparison verifies key() is deterministic across calls
		t.Fatal("key not stable across calls")
	}
	reordered := base
	reordered.Tools = []string{"b", "a"}
	if base.key("emb-1") != reordered.key("emb-1") {
		t.Fatal("key should be tool-order-independent")
	}

	// Sensitive to each scoping dimension.
	for name, mut := range map[string]func(Scope) Scope{
		"embed-model": func(s Scope) Scope { return s }, // varied via arg below
		"model":       func(s Scope) Scope { s.Model = "claude-y"; return s },
		"system":      func(s Scope) Scope { s.System = "be verbose"; return s },
		"tools":       func(s Scope) Scope { s.Tools = []string{"a"}; return s },
		"temperature": func(s Scope) Scope { s.Temperature = 0.7; return s },
		"max-tokens":  func(s Scope) Scope { s.MaxTokens = 2048; return s },
	} {
		if name == "embed-model" {
			if base.key("emb-1") == base.key("emb-2") {
				t.Errorf("key insensitive to embed model")
			}
			continue
		}
		if base.key("emb-1") == mut(base).key("emb-1") {
			t.Errorf("key insensitive to %s", name)
		}
	}
}
