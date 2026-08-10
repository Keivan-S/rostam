// SPDX-License-Identifier: Apache-2.0

package vector

import (
	"bytes"
	"math/rand"
	"os"
	"path/filepath"
	"testing"
)

// TestMVConfigMarkerIVFRoundtrip proves the on-disk .mvcfg marker persists the
// IndexType + IVF knobs, so a reopened MV IVF collection reconstructs its inner
// index as IVF (not HNSW). An all-HNSW config (the zero IVF fields) must produce
// a marker JSON with no IVF keys — byte-identical to the pre-IVF marker.
func TestMVConfigMarkerIVFRoundtrip(t *testing.T) {
	dir := t.TempDir()
	path := filepath.Join(dir, "mv.mvcfg")
	cfg := MultiVectorConfig{
		Dim: 32, M: 16, EfConstruction: 200, EfSearch: 64, Seed: 9,
		IndexType: IndexIVF, IVFNlist: 64, IVFNprobe: 12,
		IVFPQ: true, IVFPQM: 8, IVFRerank: true, OPQ: true, IVFTrainThreshold: 1000,
	}
	if err := writeMVConfig(path, cfg); err != nil {
		t.Fatal(err)
	}
	got, err := readMVConfig(path)
	if err != nil {
		t.Fatal(err)
	}
	if got.IndexType != IndexIVF || got.IVFNlist != 64 || got.IVFNprobe != 12 ||
		!got.IVFPQ || got.IVFPQM != 8 || !got.IVFRerank || !got.OPQ || got.IVFTrainThreshold != 1000 {
		t.Fatalf("IVF marker not persisted: %+v", got)
	}

	// An all-HNSW config's marker must carry no IVF keys (back-compat: old markers
	// decode to IndexHNSW/zero, and a new HNSW marker matches the old shape).
	hnswPath := filepath.Join(dir, "hnsw.mvcfg")
	if err := writeMVConfig(hnswPath, MultiVectorConfig{Dim: 32, M: 16, EfConstruction: 200, EfSearch: 64}); err != nil {
		t.Fatal(err)
	}
	b, err := os.ReadFile(hnswPath)
	if err != nil {
		t.Fatal(err)
	}
	for _, key := range []string{"index_type", "ivf_nlist", "ivf_nprobe", "ivf_pq", "ivf_rerank", "opq", "ivf_train_threshold"} {
		if bytes.Contains(b, []byte(key)) {
			t.Fatalf("HNSW marker unexpectedly contains IVF key %q: %s", key, b)
		}
	}
}

// TestCollectionStoreMultiVectorPersistentReopen exercises the store path: a
// Persistent multi-vector collection survives Flush + store close + reopen, with
// its documents and MaxSim ranking intact (instant-restart from the mmap files +
// sidecars).
func TestCollectionStoreMultiVectorPersistentReopen(t *testing.T) {
	const dim = 16
	dir := t.TempDir()
	cs, err := OpenCollectionStore(dir)
	if err != nil {
		t.Fatal(err)
	}
	if err := cs.CreateMultiVector("docs", MultiVectorConfig{Dim: dim, Quant: QuantSQ8, RescoreFactor: 3, Persistent: true}); err != nil {
		t.Fatalf("create: %v", err)
	}

	rng := rand.New(rand.NewSource(9))
	docs := map[uint64][][]float32{}
	for id := uint64(1); id <= 15; id++ {
		toks := randTokens(rng, 4, dim)
		docs[id] = toks
		if err := cs.MultiAdd("docs", id, toks, Metadata{"id": NewInt(int64(id))}); err != nil {
			t.Fatalf("add %d: %v", id, err)
		}
	}
	query := randTokens(rng, 4, dim)
	before, err := cs.MultiSearch("docs", query, 5, MultiSearchOpts{CandidatesPerToken: 200})
	if err != nil {
		t.Fatal(err)
	}
	if len(before) == 0 {
		t.Fatal("no results before reopen")
	}

	if err := cs.FlushMultiVector("docs"); err != nil {
		t.Fatalf("flush: %v", err)
	}
	if err := cs.Close(); err != nil {
		t.Fatal(err)
	}

	// Reopen the store from the same dir — the multi-vector collection must come
	// back from its config marker + mmap files + sidecars.
	cs2, err := OpenCollectionStore(dir)
	if err != nil {
		t.Fatalf("reopen: %v", err)
	}
	defer cs2.Close()
	if _, ok := cs2.GetMultiVector("docs"); !ok {
		t.Fatal("multi-vector collection missing after reopen")
	}
	after, err := cs2.MultiSearch("docs", query, 5, MultiSearchOpts{CandidatesPerToken: 200})
	if err != nil {
		t.Fatal(err)
	}
	if len(after) != len(before) {
		t.Fatalf("after reopen %d results, want %d", len(after), len(before))
	}
	for i := range before {
		if after[i].ID != before[i].ID {
			t.Errorf("rank %d: id %d after reopen, want %d", i, after[i].ID, before[i].ID)
		}
		if after[i].Metadata["id"].Int != int64(after[i].ID) {
			t.Errorf("rank %d: metadata lost: %+v", i, after[i].Metadata)
		}
	}

	// Drop removes it and its files.
	if err := cs2.DropMultiVector("docs"); err != nil {
		t.Fatalf("drop: %v", err)
	}
	if _, ok := cs2.GetMultiVector("docs"); ok {
		t.Error("collection still present after drop")
	}
}
