// SPDX-License-Identifier: Apache-2.0
//go:build localembed

package main

import (
	"strings"
	"testing"
)

// TestEmbedderFromEnv_LocalDispatchesToRealEmbedder is the -tags localembed
// counterpart of TestEmbedderFromEnv_LocalCompiledOut (localembed_stub_test.go,
// !localembed): here ROSTAM_EMBED_LOCAL reaches the real newLocalEmbedder, so
// an unknown model name fails with a catalog error, not a compiled-out one.
func TestEmbedderFromEnv_LocalDispatchesToRealEmbedder(t *testing.T) {
	_, err := embedderFromEnv(envMap(map[string]string{"ROSTAM_EMBED_LOCAL": "not-a-real-model"}))
	if err == nil || !strings.Contains(err.Error(), "unknown local embedding model") {
		t.Fatalf("want unknown-model error from the real embedder path, got %v", err)
	}
}
