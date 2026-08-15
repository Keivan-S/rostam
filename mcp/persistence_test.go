// SPDX-License-Identifier: Apache-2.0

package mcp

import (
	"testing"

	"github.com/rostamlabs/rostam"
	"github.com/rostamlabs/rostam/ops"
)

// newDirStore opens an embedded store on an existing data directory. Unlike
// newHeapStore it does NOT register a t.Cleanup close: these tests close the
// store and reopen the same directory, so ownership has to be explicit.
//
// Callers must take the directory from t.TempDir() BEFORE the store exists.
// t.Cleanup runs LIFO, so a temp dir registered first is removed last — the
// reverse order would delete the directory out from under a still-open store.
func newDirStore(t *testing.T, dir string) rostam.Store {
	t.Helper()
	reg := ops.NewRegistry()
	if err := ops.RegisterBuiltins(reg); err != nil {
		t.Fatalf("ops: %v", err)
	}
	st, err := rostam.NewDirect(rostam.DirectConfig{DataDir: dir, Ops: reg})
	if err != nil {
		t.Fatalf("NewDirect(%q): %v", dir, err)
	}
	return st
}

// TestMemorySurvivesReopen is the durability contract the whole memory
// subsystem exists for: a fact remembered in one process must still be
// recallable after that process exits and a new one opens the same -data
// directory. It fails if mcp_memory is created as a heap collection, where
// only the collection's config JSON reaches disk and its contents do not.
func TestMemorySurvivesReopen(t *testing.T) {
	dir := t.TempDir()

	// Session one: remember two facts in two namespaces, then shut down as a
	// clean process exit would (Serve returns, then the store closes).
	st := newDirStore(t, dir)
	c, stop := startServerStop(t, Config{Store: st})
	c.initialize()
	var got struct {
		ID uint64 `json:"id"`
	}
	c.callTool("remember", map[string]any{"content": "the deploy key lives in vault under ops/deploy"}, &got, false)
	c.callTool("remember", map[string]any{"content": "project atlas ships in march", "namespace": "projA"}, nil, false)
	stop()
	if err := st.Close(); err != nil {
		t.Fatalf("close session one: %v", err)
	}

	// Session two: a brand-new store and server over the same directory.
	st2 := newDirStore(t, dir)
	defer func() { _ = st2.Close() }()
	c2, stop2 := startServerStop(t, Config{Store: st2})
	defer stop2()
	c2.initialize()

	var rec struct {
		Hits []struct {
			ID      uint64 `json:"id"`
			Content string `json:"content"`
		} `json:"hits"`
	}
	c2.callTool("recall", map[string]any{"query": "deploy key vault", "k": 5}, &rec, false)
	if len(rec.Hits) == 0 {
		t.Fatalf("recall after reopen found nothing; the memory did not survive the restart")
	}
	if rec.Hits[0].ID != got.ID {
		t.Fatalf("recall after reopen: id = %d, want %d", rec.Hits[0].ID, got.ID)
	}

	// list_namespaces is derived from the surviving memories themselves, so
	// this is a second, independent read of the same durability claim: both
	// namespaces are only reported if both memories actually came back.
	var ns struct {
		Namespaces []string `json:"namespaces"`
	}
	c2.callTool("list_namespaces", map[string]any{}, &ns, false)
	if len(ns.Namespaces) != 2 || ns.Namespaces[0] != "default" || ns.Namespaces[1] != "projA" {
		t.Fatalf("namespaces after reopen = %v, want [default projA]", ns.Namespaces)
	}
	var lst struct {
		Memories []struct {
			Content string `json:"content"`
		} `json:"memories"`
	}
	c2.callTool("list_memories", map[string]any{"namespace": "projA"}, &lst, false)
	if len(lst.Memories) != 1 || lst.Memories[0].Content != "project atlas ships in march" {
		t.Fatalf("list_memories(projA) after reopen = %+v, want the one remembered fact", lst.Memories)
	}
}

// TestCreateCollectionPersistsAcrossReopen covers the same contract for the
// generic DB tools: a collection created with persistent=true (the default)
// must still hold its points after a reopen.
func TestCreateCollectionPersistsAcrossReopen(t *testing.T) {
	dir := t.TempDir()

	st := newDirStore(t, dir)
	c, stop := startServerStop(t, Config{Store: st})
	c.initialize()
	c.callTool("create_collection", map[string]any{"name": "docs", "dim": 4}, nil, false)
	c.callTool("upsert", map[string]any{
		"collection": "docs",
		"id":         7,
		"vector":     []float32{1, 0, 0, 0},
		"content":    "the runbook for on-call rotations",
	}, nil, false)
	stop()
	if err := st.Close(); err != nil {
		t.Fatalf("close session one: %v", err)
	}

	st2 := newDirStore(t, dir)
	defer func() { _ = st2.Close() }()
	c2, stop2 := startServerStop(t, Config{Store: st2})
	defer stop2()
	c2.initialize()

	var got struct {
		Points []struct {
			ID      uint64 `json:"id"`
			Content string `json:"content"`
		} `json:"points"`
		Missing []uint64 `json:"missing"`
	}
	c2.callTool("get", map[string]any{"collection": "docs", "ids": []uint64{7}}, &got, false)
	if len(got.Points) != 1 || got.Points[0].ID != 7 {
		t.Fatalf("get after reopen: points = %+v, missing = %v", got.Points, got.Missing)
	}
	if got.Points[0].Content != "the runbook for on-call rotations" {
		t.Fatalf("content after reopen = %q", got.Points[0].Content)
	}
}

// TestCreateCollectionNonPersistent checks the opt-out: persistent=false makes
// a heap collection whose contents are deliberately not expected to survive.
// It asserts only that the flag is accepted and the collection works in-process
// — what happens to it on reopen is the engine's business, not this tool's.
func TestCreateCollectionNonPersistent(t *testing.T) {
	dir := t.TempDir()
	st := newDirStore(t, dir)
	defer func() { _ = st.Close() }()
	c, stop := startServerStop(t, Config{Store: st})
	defer stop()
	c.initialize()

	c.callTool("create_collection", map[string]any{"name": "scratch", "dim": 4, "persistent": false}, nil, false)
	c.callTool("upsert", map[string]any{
		"collection": "scratch",
		"id":         1,
		"vector":     []float32{0, 1, 0, 0},
		"content":    "throwaway",
	}, nil, false)
	var got struct {
		Points []struct {
			ID uint64 `json:"id"`
		} `json:"points"`
	}
	c.callTool("get", map[string]any{"collection": "scratch", "ids": []uint64{1}}, &got, false)
	if len(got.Points) != 1 {
		t.Fatalf("heap collection lost its point in-process: %+v", got.Points)
	}
}
