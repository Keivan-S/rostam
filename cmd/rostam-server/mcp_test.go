// SPDX-License-Identifier: Apache-2.0

package main

import (
	"context"
	"strings"
	"testing"

	"github.com/rostamlabs/rostam/semcache"
)

// noEnv is a lookupEnv that reports every variable unset — the "operator set
// nothing" baseline most cases build on.
func noEnv(string) (string, bool) { return "", false }

// envMap returns a lookupEnv backed by a fixed map, so a test never touches
// the real process environment.
func envMap(m map[string]string) func(string) (string, bool) {
	return func(k string) (string, bool) {
		v, ok := m[k]
		return v, ok
	}
}

func TestMcpSetup_DataAndConnectConflict(t *testing.T) {
	fl := mcpFlags{data: "/some/dir", connect: "127.0.0.1:7000"}
	_, err := mcpSetup(fl, noEnv)
	if err == nil {
		t.Fatal("expected an error when both -data and -connect are set")
	}
	if !strings.Contains(err.Error(), "-data") || !strings.Contains(err.Error(), "-connect") {
		t.Errorf("error should name both -data and -connect, got: %v", err)
	}
}

func TestMcpSetup_Embedder(t *testing.T) {
	tests := []struct {
		name      string
		env       map[string]string
		wantErr   string // substring; "" = no error expected
		wantNil   bool   // expect a nil embedder
		wantEndpt string
	}{
		{
			name:    "no env => nil embedder",
			env:     nil,
			wantNil: true,
		},
		{
			name: "endpoint without dim => error naming ROSTAM_EMBED_DIM",
			env: map[string]string{
				"ROSTAM_EMBED_ENDPOINT": "http://localhost:11434/v1/embeddings",
				"ROSTAM_EMBED_MODEL":    "nomic-embed-text",
			},
			wantErr: "ROSTAM_EMBED_DIM",
		},
		{
			name: "endpoint without model => error naming ROSTAM_EMBED_MODEL",
			env: map[string]string{
				"ROSTAM_EMBED_ENDPOINT": "http://localhost:11434/v1/embeddings",
				"ROSTAM_EMBED_DIM":      "768",
			},
			wantErr: "ROSTAM_EMBED_MODEL",
		},
		{
			name: "endpoint with unparsable dim => error naming ROSTAM_EMBED_DIM",
			env: map[string]string{
				"ROSTAM_EMBED_ENDPOINT": "http://localhost:11434/v1/embeddings",
				"ROSTAM_EMBED_MODEL":    "nomic-embed-text",
				"ROSTAM_EMBED_DIM":      "not-a-number",
			},
			wantErr: "ROSTAM_EMBED_DIM",
		},
		{
			name: "endpoint+model+dim => non-nil embedder with overridden Endpoint",
			env: map[string]string{
				"ROSTAM_EMBED_ENDPOINT": "http://localhost:11434/v1/embeddings",
				"ROSTAM_EMBED_MODEL":    "nomic-embed-text",
				"ROSTAM_EMBED_API_KEY":  "sk-test",
				"ROSTAM_EMBED_DIM":      "768",
			},
			wantEndpt: "http://localhost:11434/v1/embeddings",
		},
	}

	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			fl := mcpFlags{data: ""} // heap mode so the store side never fails
			rt, err := mcpSetup(fl, envMap(tc.env))
			if tc.wantErr != "" {
				if err == nil {
					t.Fatalf("expected an error containing %q, got nil", tc.wantErr)
				}
				if !strings.Contains(err.Error(), tc.wantErr) {
					t.Fatalf("error %q does not mention %q", err.Error(), tc.wantErr)
				}
				return
			}
			if err != nil {
				t.Fatalf("mcpSetup: unexpected error: %v", err)
			}
			defer rt.store.Close()
			if tc.wantNil && rt.embedder != nil {
				t.Fatalf("expected a nil embedder, got %T", rt.embedder)
			}
			if tc.wantEndpt != "" {
				if rt.embedder == nil {
					t.Fatal("expected a non-nil embedder")
				}
				oe, ok := rt.embedder.(*semcache.OpenAIEmbedder)
				if !ok {
					t.Fatalf("expected *semcache.OpenAIEmbedder, got %T", rt.embedder)
				}
				if oe.Endpoint != tc.wantEndpt {
					t.Errorf("Endpoint = %q, want %q", oe.Endpoint, tc.wantEndpt)
				}
			}
		})
	}
}

// TestMcpSetup_HeapMode proves the resolved-heap ("" data) path returns a
// working store: a Put/Get round trip is the only reliable signal the store
// is actually usable, not just a non-nil interface value.
func TestMcpSetup_HeapMode(t *testing.T) {
	rt, err := mcpSetup(mcpFlags{data: ""}, noEnv)
	if err != nil {
		t.Fatalf("mcpSetup: %v", err)
	}
	defer rt.store.Close()

	ctx := context.Background()
	key, val := []byte("mcp-setup-test-key"), []byte("mcp-setup-test-value")
	if err := rt.store.Put(ctx, key, val, 0); err != nil {
		t.Fatalf("Put: %v", err)
	}
	got, err := rt.store.Get(ctx, key)
	if err != nil {
		t.Fatalf("Get: %v", err)
	}
	if string(got) != string(val) {
		t.Fatalf("Get returned %q, want %q", got, val)
	}
}
