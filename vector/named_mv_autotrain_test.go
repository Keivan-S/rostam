// SPDX-License-Identifier: Apache-2.0

package vector

import (
	"math/rand"
	"testing"
)

// TestNamedIVFPQAutoTrainsEndToEnd: a NAMED collection with an IVF-PQ dense space,
// populated purely via the incremental named Insert path (named NEVER calls the
// inner BuildConcurrent), now ACTUALLY trains + compresses once the per-space live
// count crosses the threshold. This is the whole point of auto-train for named: it
// makes IVF-PQ non-inert.
func TestNamedIVFPQAutoTrainsEndToEnd(t *testing.T) {
	const (
		dim       = 32
		threshold = 600
		n         = 900
	)
	cfg := map[string]NamedVectorParams{
		"title": {
			Dim: dim, Metric: L2, IndexType: IndexIVF,
			IVFNprobe: 16, IVFPQ: true, IVFPQM: 8, IVFRerank: true,
			IVFTrainThreshold: threshold,
		},
	}
	nc, err := NewNamedCollection("default/named-autotrain", cfg)
	if err != nil {
		t.Fatalf("new: %v", err)
	}
	defer nc.Close()

	rng := rand.New(rand.NewSource(17))
	for i := 1; i <= n; i++ {
		v := randVec(rng, dim)
		if err := nc.Insert(uint64(i), map[string][]float32{"title": v}, nil, 0); err != nil {
			t.Fatalf("insert %d: %v", i, err)
		}
	}

	ivfIdx, ok := nc.indexes["title"].(*ivf)
	if !ok {
		t.Fatalf("title space is %T, want *ivf", nc.indexes["title"])
	}
	if !ivfIdx.trained {
		t.Fatal("named IVF-PQ space did NOT auto-train after crossing the threshold (still inert)")
	}
	if !ivfIdx.pqActive() {
		t.Fatal("named IVF-PQ codebooks not built after auto-train (compression inactive)")
	}
	if ivfIdx.arena.Code(0) == nil {
		t.Fatal("no residual code written for the named IVF-PQ space (compression not engaged)")
	}

	// Search through the named path still returns results.
	q := randVec(rng, dim)
	res, err := nc.SearchNamed("title", q, 5, Filter{})
	if err != nil {
		t.Fatalf("search: %v", err)
	}
	if len(res) == 0 {
		t.Fatal("named IVF-PQ search returned no results after auto-train")
	}
}

// TestMVIVFPQAutoTrainsEndToEnd: an MV collection with an IVF-PQ INNER token index,
// populated via the incremental MV Add path (MV NEVER calls inner BuildConcurrent),
// now trains + compresses the inner index once the live TOKEN count crosses the
// threshold. MV's token index is its dominant memory cost, so this is the footprint
// win.
func TestMVIVFPQAutoTrainsEndToEnd(t *testing.T) {
	const (
		dim          = 32
		threshold    = 600
		tokensPerDoc = 8
		docs         = 120 // 120 * 8 = 960 tokens > threshold
	)
	cfg := MultiVectorConfig{
		Dim: dim, Seed: 3,
		IndexType: IndexIVF, IVFNprobe: 16,
		IVFPQ: true, IVFPQM: 8, IVFRerank: true,
		IVFTrainThreshold: threshold,
	}
	m, err := NewMultiVectorIndex(cfg)
	if err != nil {
		t.Fatalf("new: %v", err)
	}
	defer m.Close()

	rng := rand.New(rand.NewSource(29))
	corpus := make(map[uint64][][]float32, docs)
	for id := uint64(1); id <= docs; id++ {
		toks := randTokens(rng, tokensPerDoc, dim)
		corpus[id] = toks
		if err := m.Add(id, toks, Metadata{"id": NewInt(int64(id))}); err != nil {
			t.Fatalf("add %d: %v", id, err)
		}
	}

	inner, ok := m.idx.(*ivf)
	if !ok {
		t.Fatalf("inner index is %T, want *ivf", m.idx)
	}
	if !inner.trained {
		t.Fatal("MV inner IVF-PQ did NOT auto-train after crossing the token threshold (still inert)")
	}
	if !inner.pqActive() {
		t.Fatal("MV inner IVF-PQ codebooks not built after auto-train (compression inactive)")
	}
	if inner.arena.Code(0) == nil {
		t.Fatal("no residual code written for the MV inner IVF-PQ index")
	}

	// MaxSim search still works after the inner index trained + compressed.
	query := randTokens(rng, 5, dim)
	got, err := m.Search(query, 5, MultiSearchOpts{CandidatesPerToken: 200})
	if err != nil {
		t.Fatalf("search: %v", err)
	}
	if len(got) == 0 {
		t.Fatal("MV IVF-PQ MaxSim search returned no results after auto-train")
	}
}
