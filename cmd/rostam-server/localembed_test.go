// SPDX-License-Identifier: Apache-2.0

package main

import (
	"strings"
	"testing"
)

func TestEmbedderFromEnv_LocalMutualExclusion(t *testing.T) {
	_, err := embedderFromEnv(envMap(map[string]string{
		"ROSTAM_EMBED_LOCAL":    "minilm-l6-v2",
		"ROSTAM_EMBED_ENDPOINT": "https://x/embeddings",
	}))
	if err == nil || !strings.Contains(err.Error(), "mutually exclusive") {
		t.Fatalf("want mutual-exclusion error, got %v", err)
	}
}

// TestEmbedderFromEnv_LocalCompiledOut lives in localembed_stub_test.go
// (//go:build !localembed): under -tags localembed, ROSTAM_EMBED_LOCAL
// dispatches to the real embedder, not the compiled-out stub.
