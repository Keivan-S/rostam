// SPDX-License-Identifier: Apache-2.0
//go:build !localembed

package main

import (
	"strings"
	"testing"
)

func TestEmbedderFromEnv_LocalCompiledOut(t *testing.T) {
	// Default build (no -tags localembed): selecting a local model must fail loud.
	_, err := embedderFromEnv(envMap(map[string]string{"ROSTAM_EMBED_LOCAL": "minilm-l6-v2"}))
	if err == nil || !strings.Contains(err.Error(), "localembed") {
		t.Fatalf("want compiled-out error mentioning localembed, got %v", err)
	}
}
