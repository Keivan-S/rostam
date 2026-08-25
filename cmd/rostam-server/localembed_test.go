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

// A bare (non-catalog, non-"org/model") name is neither a catalog entry nor a
// valid Hugging Face id, so newLocalEmbedder must reject it before touching the
// network — no build tag involved, since local embedding is always compiled in.
func TestEmbedderFromEnv_LocalUnknownBareName(t *testing.T) {
	_, err := embedderFromEnv(envMap(map[string]string{"ROSTAM_EMBED_LOCAL": "not-a-real-model"}))
	if err == nil || !strings.Contains(err.Error(), "unknown local embedding model") {
		t.Fatalf("want unknown-model error, got %v", err)
	}
}
