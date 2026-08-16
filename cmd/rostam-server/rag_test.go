// SPDX-License-Identifier: Apache-2.0

package main

import (
	"os"
	"path/filepath"
	"testing"
)

// runRagCmdE is a test seam that returns an error instead of calling os.Exit.
// runRagCmd is implemented as a thin wrapper: it calls runRagCmdE and, on
// error, calls fatal.
func TestRagIngestThenQuery(t *testing.T) {
	data := t.TempDir()
	src := t.TempDir()
	if err := os.WriteFile(filepath.Join(src, "a.md"), []byte("epoll loop count comes from gomaxprocs"), 0o600); err != nil {
		t.Fatal(err)
	}
	if err := runRagCmdE([]string{"ingest", "--data", data, "--corpus", "docs", src}); err != nil {
		t.Fatalf("ingest: %v", err)
	}
	if err := runRagCmdE([]string{"query", "--data", data, "--corpus", "docs", "epoll loop count"}); err != nil {
		t.Fatalf("query: %v", err)
	}
}

func TestRagAskWithoutLLMErrors(t *testing.T) {
	data := t.TempDir()
	err := runRagCmdE([]string{"ask", "--data", data, "--corpus", "docs", "anything"})
	if err == nil {
		t.Fatal("ask without an LLM configured should error")
	}
}

func TestRagAlphaOutOfRangeErrors(t *testing.T) {
	data := t.TempDir()
	err := runRagCmdE([]string{"query", "--data", data, "--corpus", "docs", "--alpha", "2", "hello"})
	if err == nil {
		t.Fatal("alpha > 1 should error")
	}
	// NaN passes every float comparison as false, so it slips past a naive
	// `> 1` guard and poisons the weighted fusion — it must be rejected.
	if err := runRagCmdE([]string{"query", "--data", data, "--corpus", "docs", "--alpha", "NaN", "hello"}); err == nil {
		t.Fatal("alpha NaN should error")
	}
}

func TestRagQueryWithNoHybridFlagRuns(t *testing.T) {
	data := t.TempDir()
	src := t.TempDir()
	if err := os.WriteFile(filepath.Join(src, "a.md"), []byte("epoll loop count"), 0o600); err != nil {
		t.Fatal(err)
	}
	if err := runRagCmdE([]string{"ingest", "--data", data, "--corpus", "docs", src}); err != nil {
		t.Fatal(err)
	}
	if err := runRagCmdE([]string{"query", "--data", data, "--corpus", "docs", "--no-hybrid", "epoll"}); err != nil {
		t.Fatalf("query --no-hybrid: %v", err)
	}
}
