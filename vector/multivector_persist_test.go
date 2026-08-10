// SPDX-License-Identifier: Apache-2.0

package vector

import (
	"math/rand"
	"path/filepath"
	"testing"
)

// TestMultiVectorQuantizedMatchesExactWinner checks that an SQ8-quantized
// first stage still surfaces the correct MaxSim winner (the rerank is exact on
// the retained float32 vectors).
func TestMultiVectorQuantizedMatchesExactWinner(t *testing.T) {
	const dim = 32
	rng := rand.New(rand.NewSource(11))
	m, err := NewMultiVectorIndex(MultiVectorConfig{Dim: dim, M: 16, EfConstruction: 200, EfSearch: 128, Seed: 1, Quant: QuantSQ8, RescoreFactor: 3})
	if err != nil {
		t.Fatal(err)
	}
	defer m.Close()

	docs := map[uint64][][]float32{}
	for id := uint64(1); id <= 30; id++ {
		toks := randTokens(rng, 3+rng.Intn(4), dim)
		docs[id] = toks
		if err := m.Add(id, toks, nil); err != nil {
			t.Fatal(err)
		}
	}
	query := randTokens(rng, 4, dim)

	// Exact brute-force best document.
	var bestID uint64
	var bestScore float32
	for id, toks := range docs {
		if s := bruteMaxSim(query, toks); bestID == 0 || s > bestScore {
			bestID, bestScore = id, s
		}
	}

	got, err := m.Search(query, 3, MultiSearchOpts{CandidatesPerToken: 200})
	if err != nil {
		t.Fatal(err)
	}
	if len(got) == 0 || got[0].ID != bestID {
		t.Fatalf("quantized winner = %+v, want id %d", got, bestID)
	}
}

// TestMultiVectorPersistentRoundtrip checks that a Persistent (mmap-backed,
// off-heap, quantized) index survives Flush + reopen: documents, scores, and
// metadata all come back.
func TestMultiVectorPersistentRoundtrip(t *testing.T) {
	const dim = 16
	dir := t.TempDir()
	cfg := MultiVectorConfig{
		Dim: dim, M: 16, EfConstruction: 100, EfSearch: 64, Seed: 1,
		Quant: QuantSQ8, RescoreFactor: 3, Persistent: true,
		MmapPath:      filepath.Join(dir, "docs.mv.vecs"),
		GraphMmapPath: filepath.Join(dir, "docs.mv.graph"),
	}
	m, err := NewMultiVectorIndex(cfg)
	if err != nil {
		t.Fatal(err)
	}

	rng := rand.New(rand.NewSource(5))
	docs := map[uint64][][]float32{}
	for id := uint64(1); id <= 20; id++ {
		toks := randTokens(rng, 4, dim)
		docs[id] = toks
		if err := m.Add(id, toks, Metadata{"id": NewInt(int64(id))}); err != nil {
			t.Fatal(err)
		}
	}
	query := randTokens(rng, 4, dim)
	before, err := m.Search(query, 5, MultiSearchOpts{CandidatesPerToken: 200})
	if err != nil {
		t.Fatal(err)
	}
	wantDocs := m.NumDocs()

	if err := m.Flush(); err != nil {
		t.Fatalf("flush: %v", err)
	}
	if err := m.Close(); err != nil {
		t.Fatal(err)
	}

	// Reopen by mapping the files (instant restart) + reading the maps sidecar.
	m2, err := openPersistentMultiVector(cfg)
	if err != nil {
		t.Fatalf("reopen: %v", err)
	}
	defer m2.Close()

	if m2.NumDocs() != wantDocs {
		t.Errorf("reopened NumDocs = %d, want %d", m2.NumDocs(), wantDocs)
	}
	after, err := m2.Search(query, 5, MultiSearchOpts{CandidatesPerToken: 200})
	if err != nil {
		t.Fatal(err)
	}
	if len(after) != len(before) {
		t.Fatalf("after reopen got %d results, want %d", len(after), len(before))
	}
	for i := range before {
		if after[i].ID != before[i].ID {
			t.Errorf("rank %d: id %d after reopen, want %d", i, after[i].ID, before[i].ID)
		}
		if after[i].Metadata["id"].Int != int64(after[i].ID) {
			t.Errorf("rank %d: metadata not restored: %+v", i, after[i].Metadata)
		}
	}
}

func TestMultiVectorFlushRequiresPersistent(t *testing.T) {
	m, _ := NewMultiVectorIndex(MultiVectorConfig{Dim: 4, Seed: 1})
	defer m.Close()
	if err := m.Flush(); err == nil {
		t.Error("Flush on in-memory index should error")
	}
}
