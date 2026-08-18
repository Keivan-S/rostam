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

func TestEmbedderFromEnv_LocalCompiledOut(t *testing.T) {
	// Default build (no -tags localembed): selecting a local model must fail loud.
	_, err := embedderFromEnv(envMap(map[string]string{"ROSTAM_EMBED_LOCAL": "minilm-l6-v2"}))
	if err == nil || !strings.Contains(err.Error(), "localembed") {
		t.Fatalf("want compiled-out error mentioning localembed, got %v", err)
	}
}
