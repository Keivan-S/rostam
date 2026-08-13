// SPDX-License-Identifier: Apache-2.0

package rostam

import (
	"context"
	"os"
	"path/filepath"
	"testing"

	"github.com/rostamlabs/rostam/ops"
	"github.com/rostamlabs/rostam/vector"
)

// An unset DataDir is documented as heap mode ("Empty = pure heap mode, no
// persistence"). It used to mean something else entirely: every store path was
// relative, so the store wrote ./vectors/<tenant>/ into the process's working
// directory and read it back next time. Three consequences, all of them
// surprising to a caller who configured nothing:
//
//   - files appeared next to whatever binary happened to run,
//   - a fresh store inherited the previous run's collections, so a create
//     returned "collection already exists" for a name never used before,
//   - two stores in one process shared a namespace.
//
// It also made the test suite self-poisoning: the first run left ok_unset.json
// behind and every later run in that checkout failed. CI never saw it, because
// a fresh runner is always the first run.
func heapStore(t *testing.T) Store {
	t.Helper()
	reg := ops.NewRegistry()
	if err := ops.RegisterBuiltins(reg); err != nil {
		t.Fatalf("RegisterBuiltins: %v", err)
	}
	s, err := NewDirect(DirectConfig{Ops: reg}) // no DataDir: heap mode
	if err != nil {
		t.Fatalf("NewDirect: %v", err)
	}
	t.Cleanup(func() { _ = s.Close() })
	return s
}

func heapConfig() VectorConfig {
	return VectorConfig{Dim: 4, Metric: vector.L2, M: 8, EfConstruction: 50, EfSearch: 32, Seed: 1}
}

func TestHeapModeWritesNothingToTheWorkingDirectory(t *testing.T) {
	dir := t.TempDir()
	t.Chdir(dir)

	s := heapStore(t)
	if err := s.CreateCollection(context.Background(), "docs", heapConfig()); err != nil {
		t.Fatalf("CreateCollection: %v", err)
	}

	entries, err := os.ReadDir(".")
	if err != nil {
		t.Fatal(err)
	}
	if len(entries) != 0 {
		names := make([]string, 0, len(entries))
		for _, e := range entries {
			names = append(names, e.Name())
		}
		t.Errorf("heap mode wrote into the working directory: %v", names)
	}
}

func TestHeapModeStoresDoNotShareCollections(t *testing.T) {
	t.Chdir(t.TempDir())
	ctx := context.Background()

	first := heapStore(t)
	if err := first.CreateCollection(ctx, "docs", heapConfig()); err != nil {
		t.Fatalf("first store: %v", err)
	}
	// A second, independent store must not see the first one's collections —
	// which is also what a process restarting in the same directory looks like.
	second := heapStore(t)
	if err := second.CreateCollection(ctx, "docs", heapConfig()); err != nil {
		t.Errorf("second store could not create \"docs\": %v", err)
	}
}

// Creating a collection twice in ONE store is still an error — the fix must not
// have turned a real conflict into a silent overwrite.
func TestHeapModeStillRejectsADuplicateInTheSameStore(t *testing.T) {
	t.Chdir(t.TempDir())
	ctx := context.Background()
	s := heapStore(t)
	if err := s.CreateCollection(ctx, "docs", heapConfig()); err != nil {
		t.Fatalf("first create: %v", err)
	}
	if err := s.CreateCollection(ctx, "docs", heapConfig()); err == nil {
		t.Error("creating the same collection twice in one store was accepted; it must be refused")
	}
}

// The scratch directory a heap-mode store owns must not outlive it.
//
// This points TMPDIR at a private directory rather than counting rostam-heap-*
// entries in the machine-wide temp dir. Counting was racy: any other process
// creating or removing a matching directory between the snapshots could mask a
// real leak or invent one — the exact species of flaky test this whole change
// exists to stamp out.
func TestHeapModeScratchDirIsRemovedOnClose(t *testing.T) {
	tmpRoot := t.TempDir()
	t.Setenv("TMPDIR", tmpRoot) // os.MkdirTemp("") resolves through this
	t.Chdir(t.TempDir())

	reg := ops.NewRegistry()
	if err := ops.RegisterBuiltins(reg); err != nil {
		t.Fatal(err)
	}
	s, err := NewDirect(DirectConfig{Ops: reg})
	if err != nil {
		t.Fatalf("NewDirect: %v", err)
	}
	if err := s.CreateCollection(context.Background(), "docs", heapConfig()); err != nil {
		t.Fatalf("CreateCollection: %v", err)
	}

	// Exactly one scratch dir, and we know which one.
	created, err := os.ReadDir(tmpRoot)
	if err != nil {
		t.Fatal(err)
	}
	if len(created) != 1 {
		t.Fatalf("expected one scratch dir under %s, got %d", tmpRoot, len(created))
	}
	scratch := filepath.Join(tmpRoot, created[0].Name())
	if _, err := os.Stat(scratch); err != nil {
		t.Fatalf("scratch dir %s not usable: %v", scratch, err)
	}

	if err := s.Close(); err != nil {
		t.Fatalf("Close: %v", err)
	}
	if _, err := os.Stat(scratch); !os.IsNotExist(err) {
		t.Errorf("scratch dir %s survived Close (stat err: %v)", scratch, err)
	}
}
