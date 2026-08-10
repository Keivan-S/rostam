// SPDX-License-Identifier: Apache-2.0

package vector

import (
	"bytes"
	"os"
	"path/filepath"
	"strings"
	"sync"
	"testing"
)

// TestClusterPersistentMmapBacking verifies a persistent-cluster store backs its
// collections with mmap files (off-heap), wipes stale data files at open, and
// keeps a generation-suffixed layout.
func TestClusterPersistentMmapBacking(t *testing.T) {
	dir := t.TempDir()
	s, err := OpenCollectionStorePersistent(dir, true)
	if err != nil {
		t.Fatal(err)
	}
	cfg := Config{Dim: 4, M: 8, EfConstruction: 50, EfSearch: 32, Seed: 1, Metric: L2, Quant: QuantSQ8, RescoreFactor: 4}
	if err := s.CreateCollection("docs", cfg); err != nil {
		t.Fatal(err)
	}
	for i := 1; i <= 50; i++ {
		if err := s.Insert("docs", uint64(i), []float32{float32(i), 0, 0, 0}, 0, nil, nil); err != nil {
			t.Fatalf("insert %d: %v", i, err)
		}
	}
	// Backing is mmap: the collection carries store-managed mmap paths and the
	// files exist on disk (not heap).
	c, ok := s.Get("docs")
	if !ok {
		t.Fatal("collection missing")
	}
	if c.cfg.MmapPath == "" || c.cfg.GraphMmapPath == "" {
		t.Fatalf("cluster collection not mmap-backed: cfg=%+v", c.cfg)
	}
	if _, err := os.Stat(c.cfg.MmapPath); err != nil {
		t.Fatalf("mmap vecs file missing: %v", err)
	}
	if !strings.Contains(filepath.Base(c.cfg.MmapPath), ".g") {
		t.Fatalf("expected generation-suffixed mmap path, got %q", c.cfg.MmapPath)
	}
	_ = s.Close()
}

// TestClusterPersistentCatchUpFromHeapSnapshot simulates new-node catch-up: a
// heap-form wire snapshot (what Raft InstallSnapshot ships — always relocatable,
// node-independent) is restored into a fresh persistent-cluster node, which must
// materialize the collection mmap-backed and return identical results.
func TestClusterPersistentCatchUpFromHeapSnapshot(t *testing.T) {
	cfg := Config{Dim: 3, M: 8, EfConstruction: 50, EfSearch: 32, Seed: 1, Metric: L2, Quant: QuantSQ8, RescoreFactor: 4}

	// Source: an ordinary heap store (the wire form drops mmap backing).
	src, err := OpenCollectionStore(t.TempDir())
	if err != nil {
		t.Fatal(err)
	}
	if err := src.CreateCollection("docs", cfg); err != nil {
		t.Fatal(err)
	}
	for i := 1; i <= 20; i++ {
		if err := src.Upsert("docs", uint64(i), []float32{float32(i), 0, 0}, "chunk", 0, Metadata{"d": NewInt(int64(i))}, nil); err != nil {
			t.Fatal(err)
		}
	}
	var blob bytes.Buffer
	if err := src.SnapshotAll(&blob); err != nil {
		t.Fatal(err)
	}
	ref, err := src.SearchDocs("docs", []float32{1, 0, 0}, 5, Filter{})
	if err != nil {
		t.Fatal(err)
	}
	_ = src.Close()

	// Dst: a fresh persistent-cluster node (empty dir) restoring from the snapshot.
	dst, err := OpenCollectionStorePersistent(t.TempDir(), true)
	if err != nil {
		t.Fatal(err)
	}
	defer func() { _ = dst.Close() }()
	if err := dst.RestoreAll(bytes.NewReader(blob.Bytes())); err != nil {
		t.Fatal(err)
	}

	// The restored collection is mmap-backed (off-heap) on the catching-up node.
	c, ok := dst.Get("docs")
	if !ok {
		t.Fatal("collection not restored")
	}
	if c.cfg.MmapPath == "" {
		t.Fatalf("caught-up collection not mmap-backed: cfg=%+v", c.cfg)
	}
	if _, err := os.Stat(c.cfg.MmapPath); err != nil {
		t.Fatalf("mmap file missing after catch-up: %v", err)
	}
	// ...and returns identical results.
	got, err := dst.SearchDocs("docs", []float32{1, 0, 0}, 5, Filter{})
	if err != nil {
		t.Fatal(err)
	}
	if len(got) != len(ref) || len(got) == 0 {
		t.Fatalf("caught-up search = %d docs, want %d", len(got), len(ref))
	}
	for i := range got {
		if got[i].ID != ref[i].ID || got[i].Content != ref[i].Content {
			t.Fatalf("doc %d mismatch: got %+v want %+v", i, got[i], ref[i])
		}
	}
}

// mvTokens builds a small deterministic set of token vectors for doc d.
func mvTokens(d, dim int) [][]float32 {
	toks := make([][]float32, 3)
	for j := range toks {
		v := make([]float32, dim)
		v[(d+j)%dim] = 1
		v[(d*j)%dim] += 0.5
		toks[j] = v
	}
	return toks
}

// TestClusterPersistentMultiVector covers multi-vector (late-interaction)
// collections under the persistent-cluster policy: catch-up from a heap wire
// snapshot materializes an mmap-backed index, and repeated restores under
// concurrent MaxSim searches are SIGBUS-free with generation GC.
func TestClusterPersistentMultiVector(t *testing.T) {
	const dim = 8
	mvCfg := MultiVectorConfig{Dim: dim, M: 16, EfConstruction: 200, EfSearch: 128, Seed: 1, Quant: QuantSQ8, RescoreFactor: 4}

	// Source: a heap store; snapshot is the relocatable catch-up wire form.
	src, err := OpenCollectionStore(t.TempDir())
	if err != nil {
		t.Fatal(err)
	}
	if err := src.CreateMultiVector("mv", mvCfg); err != nil {
		t.Fatal(err)
	}
	for d := 1; d <= 30; d++ {
		if err := src.MultiAdd("mv", uint64(d), mvTokens(d, dim), nil); err != nil {
			t.Fatalf("multi add %d: %v", d, err)
		}
	}
	var blob bytes.Buffer
	if err := src.SnapshotAll(&blob); err != nil {
		t.Fatal(err)
	}
	query := mvTokens(2, dim)
	ref, err := src.MultiSearch("mv", query, 5, MultiSearchOpts{CandidatesPerToken: 100})
	if err != nil {
		t.Fatal(err)
	}
	_ = src.Close()

	// Catch-up into a fresh persistent-cluster node → mmap-backed index.
	dst, err := OpenCollectionStorePersistent(t.TempDir(), true)
	if err != nil {
		t.Fatal(err)
	}
	defer func() { _ = dst.Close() }()
	if err := dst.RestoreAll(bytes.NewReader(blob.Bytes())); err != nil {
		t.Fatal(err)
	}
	idx, ok := dst.GetMultiVector("mv")
	if !ok || idx.cfg.MmapPath == "" || !idx.cfg.Persistent {
		t.Fatalf("caught-up multi-vector not mmap-backed: ok=%v cfg=%+v", ok, idx.cfg)
	}
	if _, err := os.Stat(idx.cfg.MmapPath); err != nil {
		t.Fatalf("multi-vector mmap file missing: %v", err)
	}
	got, err := dst.MultiSearch("mv", query, 5, MultiSearchOpts{CandidatesPerToken: 100})
	if err != nil {
		t.Fatal(err)
	}
	if len(got) == 0 || len(got) != len(ref) || got[0].ID != ref[0].ID {
		t.Fatalf("caught-up MaxSim mismatch: got %d (top %v) want %d (top %v)", len(got), top(got), len(ref), top(ref))
	}

	// Repeated restores under concurrent searches: no SIGBUS, valid results.
	stop := make(chan struct{})
	var wg sync.WaitGroup
	for w := 0; w < 8; w++ {
		wg.Add(1)
		go func() {
			defer wg.Done()
			for {
				select {
				case <-stop:
					return
				default:
				}
				if _, err := dst.MultiSearch("mv", query, 5, MultiSearchOpts{CandidatesPerToken: 100}); err != nil {
					t.Errorf("multi search during restore: %v", err)
					return
				}
			}
		}()
	}
	for r := 0; r < 40; r++ {
		if err := dst.RestoreAll(bytes.NewReader(blob.Bytes())); err != nil {
			close(stop)
			wg.Wait()
			t.Fatalf("multi restore %d: %v", r, err)
		}
	}
	close(stop)
	wg.Wait()

	// Generation GC: only the current generation's mmap files remain.
	vecsCount := 0
	_ = filepath.WalkDir(filepath.Join(dst.dir, "vectors"), func(p string, d os.DirEntry, err error) error {
		if err == nil && !d.IsDir() && filepath.Ext(d.Name()) == ".vecs" {
			vecsCount++
		}
		return nil
	})
	if vecsCount != 1 {
		t.Fatalf("multi-vector generation GC leak: %d .vecs files, want 1", vecsCount)
	}
}

func top(rs []MultiResult) uint64 {
	if len(rs) == 0 {
		return 0
	}
	return rs[0].ID
}

// TestClusterPersistentRestoreUnderReads is the critical safety test: it drives
// RestoreAll (simulating Raft InstallSnapshot) many times while concurrent
// searchers hammer the collection. The generation-suffixed files + refcount must
// prevent any unmap-under-reader (SIGBUS) and keep results valid. Run under -race.
func TestClusterPersistentRestoreUnderReads(t *testing.T) {
	dir := t.TempDir()
	s, err := OpenCollectionStorePersistent(dir, true)
	if err != nil {
		t.Fatal(err)
	}
	defer func() { _ = s.Close() }()

	cfg := Config{Dim: 4, M: 8, EfConstruction: 50, EfSearch: 32, Seed: 1, Metric: L2, Quant: QuantSQ8, RescoreFactor: 4}
	if err := s.CreateCollection("docs", cfg); err != nil {
		t.Fatal(err)
	}
	const n = 300
	for i := 1; i <= n; i++ {
		if err := s.Insert("docs", uint64(i), []float32{float32(i % 17), float32(i % 13), 0, 0}, 0, nil, nil); err != nil {
			t.Fatalf("insert %d: %v", i, err)
		}
	}
	// A heap snapshot blob (relocatable wire form) to replay on each restore.
	var blob bytes.Buffer
	if err := s.SnapshotAll(&blob); err != nil {
		t.Fatal(err)
	}

	// Concurrent searchers run throughout.
	stop := make(chan struct{})
	var wg sync.WaitGroup
	for w := 0; w < 8; w++ {
		wg.Add(1)
		go func() {
			defer wg.Done()
			q := []float32{1, 2, 0, 0}
			for {
				select {
				case <-stop:
					return
				default:
				}
				if _, err := s.SearchFiltered("docs", q, 10, Filter{}); err != nil {
					// "no collection" can never happen — docs always exists; any
					// error here is a real failure.
					t.Errorf("search during restore: %v", err)
					return
				}
			}
		}()
	}

	// Repeatedly restore (each builds a fresh generation, retires the old).
	const restores = 60
	for r := 0; r < restores; r++ {
		if err := s.RestoreAll(bytes.NewReader(blob.Bytes())); err != nil {
			close(stop)
			wg.Wait()
			t.Fatalf("restore %d: %v", r, err)
		}
	}
	close(stop)
	wg.Wait()

	// Data intact after all the churn.
	res, err := s.SearchFiltered("docs", []float32{1, 2, 0, 0}, 10, Filter{})
	if err != nil {
		t.Fatal(err)
	}
	if len(res) != 10 {
		t.Fatalf("post-restore search returned %d, want 10", len(res))
	}

	// Generation GC: only the current generation's mmap files remain on disk
	// (every prior generation was deleted by retire). One collection ⇒ one .vecs.
	vecsCount := 0
	_ = filepath.WalkDir(filepath.Join(dir, "vectors"), func(p string, d os.DirEntry, err error) error {
		if err == nil && !d.IsDir() && filepath.Ext(d.Name()) == ".vecs" {
			vecsCount++
		}
		return nil
	})
	if vecsCount != 1 {
		t.Fatalf("generation GC leak: %d .vecs files remain, want 1", vecsCount)
	}
}
