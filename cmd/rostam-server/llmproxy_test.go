// SPDX-License-Identifier: Apache-2.0

package main

import (
	"context"
	"strings"
	"testing"
	"time"

	"github.com/rostamlabs/rostam/semcache"
)

// baseLlmproxyFlags returns a minimal valid llmproxyFlags (heap store, default
// upstream, loopback listen) that individual test cases mutate one field at a
// time — keeping each table entry focused on the one thing it's testing.
func baseLlmproxyFlags() llmproxyFlags {
	return llmproxyFlags{
		storeFlags: storeFlags{data: ""}, // heap mode: no real filesystem touched
		listen:     "127.0.0.1:8484",
		upstream:   "https://api.openai.com",
		collection: "llm-cache",
		maxTemp:    1.0,
		ttl:        168 * time.Hour,
	}
}

func TestLlmproxySetup_BadUpstreamURL(t *testing.T) {
	for _, u := range []string{
		"not-a-url",         // no scheme: not absolute
		"ftp://example.com", // wrong scheme
		"://bad",            // fails url.Parse outright
		"",                  // empty
	} {
		t.Run(u, func(t *testing.T) {
			fl := baseLlmproxyFlags()
			fl.upstream = u
			_, err := llmproxySetup(fl, noEnv)
			if err == nil {
				t.Fatalf("upstream %q: expected an error", u)
			}
			if !strings.Contains(err.Error(), "-upstream") {
				t.Errorf("upstream %q: error should mention -upstream, got: %v", u, err)
			}
		})
	}
}

func TestLlmproxySetup_DataAndConnectConflict(t *testing.T) {
	fl := baseLlmproxyFlags()
	fl.data = "/some/dir"
	fl.connect = "127.0.0.1:7000"
	_, err := llmproxySetup(fl, noEnv)
	if err == nil {
		t.Fatal("expected an error when both -data and -connect are set")
	}
	if !strings.Contains(err.Error(), "-data") || !strings.Contains(err.Error(), "-connect") {
		t.Errorf("error should name both -data and -connect, got: %v", err)
	}
}

func TestLlmproxySetup_IncompleteEmbedderEnv(t *testing.T) {
	fl := baseLlmproxyFlags()
	_, err := llmproxySetup(fl, envMap(map[string]string{
		"ROSTAM_EMBED_ENDPOINT": "http://localhost:11434/v1/embeddings",
		// ROSTAM_EMBED_MODEL and ROSTAM_EMBED_DIM missing
	}))
	if err == nil {
		t.Fatal("expected an error for an incomplete embedder env")
	}
	if !strings.Contains(err.Error(), "ROSTAM_EMBED_MODEL") {
		t.Errorf("error should name the missing ROSTAM_EMBED_MODEL, got: %v", err)
	}
}

func TestLlmproxySetup_ListenLoopbackGate(t *testing.T) {
	tests := []struct {
		name     string
		listen   string
		insecure bool
		wantErr  bool
	}{
		{"non-loopback without insecure => error", "0.0.0.0:8484", false, true},
		{"loopback IP => ok", "127.0.0.1:8484", false, false},
		{"literal localhost => ok", "localhost:8484", false, false},
		{"non-loopback with insecure => ok", "0.0.0.0:8484", true, false},
	}
	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			fl := baseLlmproxyFlags()
			fl.listen = tc.listen
			fl.insecure = tc.insecure
			rt, err := llmproxySetup(fl, noEnv)
			if tc.wantErr {
				if err == nil {
					t.Fatal("expected an error")
				}
				return
			}
			if err != nil {
				t.Fatalf("unexpected error: %v", err)
			}
			defer rt.store.Close()
		})
	}
}

// TestLlmproxyModeThreshold exercises the mode/threshold defaulting rules in
// isolation (no store needed): no embedder => exact mode, 0.999 default
// floor; an embedder => semantic mode, semcache.DefaultThreshold (0.97)
// default floor; an explicit non-zero -threshold always wins.
func TestLlmproxyModeThreshold(t *testing.T) {
	tests := []struct {
		name          string
		embedder      semcache.Embedder
		flagThreshold float64
		wantMode      string
		wantThreshold float64
	}{
		{"no embedder, no flag => exact, 0.999", nil, 0, "exact", exactThreshold},
		{"embedder set, no flag => semantic, 0.97", semcache.NewStubEmbedder("hosted", 8), 0, "semantic", semcache.DefaultThreshold},
		{"no embedder, explicit flag wins", nil, 0.5, "exact", 0.5},
		{"embedder set, explicit flag wins", semcache.NewStubEmbedder("hosted", 8), 0.5, "semantic", 0.5},
	}
	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			mode, threshold, resolved := llmproxyModeThreshold(tc.embedder, tc.flagThreshold)
			if mode != tc.wantMode {
				t.Errorf("mode = %q, want %q", mode, tc.wantMode)
			}
			if threshold != tc.wantThreshold {
				t.Errorf("threshold = %v, want %v", threshold, tc.wantThreshold)
			}
			if resolved == nil {
				t.Fatal("resolved embedder must never be nil")
			}
		})
	}
}

func TestLlmproxySetup_ModeFromEnv(t *testing.T) {
	tests := []struct {
		name     string
		env      map[string]string
		wantMode string
	}{
		{"no env => exact", nil, "exact"},
		{
			"full embedder env => semantic",
			map[string]string{
				"ROSTAM_EMBED_ENDPOINT": "http://localhost:11434/v1/embeddings",
				"ROSTAM_EMBED_MODEL":    "nomic-embed-text",
				"ROSTAM_EMBED_DIM":      "768",
			},
			"semantic",
		},
	}
	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			fl := baseLlmproxyFlags()
			rt, err := llmproxySetup(fl, envMap(tc.env))
			if err != nil {
				t.Fatalf("llmproxySetup: unexpected error: %v", err)
			}
			defer rt.store.Close()
			if rt.mode != tc.wantMode {
				t.Errorf("mode = %q, want %q", rt.mode, tc.wantMode)
			}
			if rt.handler == nil {
				t.Error("handler must not be nil")
			}
		})
	}
}

// TestLlmproxySetup_HeapModeCacheRoundTrip proves the built runtime is
// actually usable: a Store/Lookup round trip through rt.cache is the only
// reliable signal the cache (and the store underneath it) is wired up
// correctly, not just that the constructors returned non-nil values.
func TestLlmproxySetup_HeapModeCacheRoundTrip(t *testing.T) {
	fl := baseLlmproxyFlags()
	rt, err := llmproxySetup(fl, noEnv)
	if err != nil {
		t.Fatalf("llmproxySetup: %v", err)
	}
	defer rt.store.Close()

	ctx := context.Background()
	scope := semcache.Scope{Model: "gpt-4o-mini", Temperature: 0}
	prompt := "what is the capital of France?"

	if _, found, err := rt.cache.Lookup(ctx, prompt, scope); err != nil {
		t.Fatalf("Lookup (miss): %v", err)
	} else if found {
		t.Fatal("expected a clean miss before any Store")
	}

	if err := rt.cache.Store(ctx, prompt, scope, "Paris", 3); err != nil {
		t.Fatalf("Store: %v", err)
	}

	hit, found, err := rt.cache.Lookup(ctx, prompt, scope)
	if err != nil {
		t.Fatalf("Lookup (hit): %v", err)
	}
	if !found {
		t.Fatal("expected a hit after Store")
	}
	if hit.Answer != "Paris" {
		t.Errorf("Answer = %q, want %q", hit.Answer, "Paris")
	}
	if hit.OutTokens != 3 {
		t.Errorf("OutTokens = %d, want 3", hit.OutTokens)
	}
}
