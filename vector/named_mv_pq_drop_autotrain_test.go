// SPDX-License-Identifier: Apache-2.0

package vector

import (
	"bytes"
	"math/rand"
	"testing"
)

// pqDropClusteredCorpus builds n clustered Dim-d vectors plus a query set. Clustered data
// gives PQ/ADC real structure to exploit (ADC-only recall is meaningless on
// isotropic noise). Mirrors buildPQDropCorpus but returns the corpus shape the
// named/MV helpers want.
func pqDropClusteredCorpus(n, dim, nClusters, nQueries int, seed int64) (vecs, queries [][]float32) {
	rng := rand.New(rand.NewSource(seed))
	centers := make([][]float32, nClusters)
	for c := range centers {
		centers[c] = make([]float32, dim)
		for d := 0; d < dim; d++ {
			centers[c][d] = float32(rng.NormFloat64()) * 5
		}
	}
	vecs = make([][]float32, n)
	for i := range vecs {
		c := centers[rng.Intn(nClusters)]
		v := make([]float32, dim)
		for d := 0; d < dim; d++ {
			v[d] = c[d] + float32(rng.NormFloat64())
		}
		vecs[i] = v
	}
	queries = make([][]float32, nQueries)
	for i := range queries {
		c := centers[rng.Intn(nClusters)]
		q := make([]float32, dim)
		for d := 0; d < dim; d++ {
			q[d] = c[d] + float32(rng.NormFloat64())
		}
		queries[i] = q
	}
	return vecs, queries
}

// TestNamedPQDropVecsAutoTrainsEndToEnd: a NAMED collection with a QuantPQ dense
// space + PQDropVecs + a low IVFTrainThreshold, populated purely via the
// incremental named Insert path (named NEVER calls the inner BuildConcurrent), now
// ACTUALLY trains its codebooks, DROPS the resident floats, and serves ADC-only
// search once the per-space live count crosses the threshold. A post-drop Insert
// is rejected ErrPQDropVecsReadOnly. This proves PQDropVecs is HONORED for named
// (not stored-but-ignored).
func TestNamedPQDropVecsAutoTrainsEndToEnd(t *testing.T) {
	const (
		dim       = 64
		threshold = 800
		n         = 800 // the insert that crosses the threshold trains + drops
		nClusters = 30
	)
	cfg := map[string]NamedVectorParams{
		"title": {
			Dim: dim, Metric: L2,
			Quant: QuantPQ, RescoreFactor: 3,
			IVFTrainThreshold: threshold, PQDropVecs: true,
		},
	}
	nc, err := NewNamedCollection("default/named-pqdrop", cfg)
	if err != nil {
		t.Fatalf("new: %v", err)
	}
	defer nc.Close()

	vecs, queries := pqDropClusteredCorpus(n, dim, nClusters, 50, 50)
	for i := 0; i < n; i++ {
		if err := nc.Insert(uint64(i+1), map[string][]float32{"title": vecs[i]}, nil, 0); err != nil {
			t.Fatalf("insert %d: %v", i+1, err)
		}
	}

	h, ok := nc.indexes["title"].(*hnsw)
	if !ok {
		t.Fatalf("title space is %T, want *hnsw", nc.indexes["title"])
	}
	if h.pqUntrained() {
		t.Fatal("named PQ-HNSW space did NOT auto-train after crossing the threshold (still inert)")
	}
	if !h.vecsDropped() {
		t.Fatal("named PQDropVecs space did NOT drop the resident floats after auto-train (stored-but-ignored)")
	}
	if h.arena.vecs != nil {
		t.Fatal("named arena.vecs must be nil after the auto-train float drop")
	}
	// Codes survive the drop.
	nonZero := false
	for slot := 0; slot < n && !nonZero; slot++ {
		for _, b := range h.arena.Code(uint32(slot)) {
			if b != 0 {
				nonZero = true
				break
			}
		}
	}
	if !nonZero {
		t.Fatal("all PQ codes zero after the named auto-train drop (codes lost)")
	}

	// ADC-only search through the named path still returns sane results.
	var matches int
	const k = 10
	for _, q := range queries {
		truth := bruteTopK(vecs, q, k) // bruteTopK keys by uint64(i+1) (1-based)
		res, err := nc.SearchNamed("title", q, k, Filter{})
		if err != nil {
			t.Fatalf("search: %v", err)
		}
		for _, r := range res {
			if truth[r.ID] { // named ids are 1-based, matching bruteTopK keys
				matches++
			}
		}
	}
	rec := float64(matches) / float64(len(queries)*k)
	t.Logf("named post-drop ADC-only recall@%d=%.3f", k, rec)
	if rec < 0.30 {
		t.Fatalf("named post-drop ADC recall@%d=%.3f below floor 0.30", k, rec)
	}

	// Post-drop Insert is rejected (read-mostly).
	if err := nc.Insert(uint64(n+1), map[string][]float32{"title": vecs[0]}, nil, 0); err != ErrPQDropVecsReadOnly {
		t.Fatalf("post-drop named Insert should return ErrPQDropVecsReadOnly, got %v", err)
	}
}

// TestNamedPQDropVecsSnapshotRestore: an incrementally-trained-then-dropped named
// PQ-HNSW space round-trips through Snapshot/Restore. After restore the trained
// codebooks + per-slot codes + dropped state persist (ADC-capable, NOT exact
// float), so search still works.
func TestNamedPQDropVecsSnapshotRestore(t *testing.T) {
	const (
		dim       = 64
		threshold = 600
		n         = 600
		nClusters = 25
	)
	cfg := map[string]NamedVectorParams{
		"title": {
			Dim: dim, Metric: L2, Quant: QuantPQ, RescoreFactor: 3,
			IVFTrainThreshold: threshold, PQDropVecs: true,
		},
	}
	nc, err := NewNamedCollection("default/named-pqdrop-persist", cfg)
	if err != nil {
		t.Fatalf("new: %v", err)
	}
	vecs, queries := pqDropClusteredCorpus(n, dim, nClusters, 5, 51)
	for i := 0; i < n; i++ {
		if err := nc.Insert(uint64(i+1), map[string][]float32{"title": vecs[i]}, nil, 0); err != nil {
			t.Fatalf("insert %d: %v", i+1, err)
		}
	}
	h := nc.indexes["title"].(*hnsw)
	if h.pqUntrained() || !h.vecsDropped() {
		t.Fatal("precondition: named space must be trained + dropped before snapshot")
	}
	q := queries[0]
	before, err := nc.SearchNamed("title", q, 10, Filter{})
	if err != nil {
		t.Fatal(err)
	}

	var buf bytes.Buffer
	if err := nc.Snapshot(&buf); err != nil {
		t.Fatalf("snapshot: %v", err)
	}
	nc.Close()

	restored, err := NewNamedCollection("default/named-pqdrop-persist", cfg)
	if err != nil {
		t.Fatalf("new restored: %v", err)
	}
	defer restored.Close()
	if err := restored.Restore(bytes.NewReader(buf.Bytes())); err != nil {
		t.Fatalf("restore: %v", err)
	}

	rh := restored.indexes["title"].(*hnsw)
	if rh.pqUntrained() {
		t.Fatal("restored named space lost its trained codebooks")
	}
	if !rh.vecsDropped() {
		t.Fatal("restored named space lost its dropped state (would be exact float, not ADC)")
	}
	if rh.arena.vecs != nil {
		t.Fatal("restored named arena.vecs must be nil (dropped state)")
	}
	after, err := restored.SearchNamed("title", q, 10, Filter{})
	if err != nil {
		t.Fatal(err)
	}
	if len(after) == 0 {
		t.Fatal("restored named PQDropVecs search returned no results")
	}
	if len(after) != len(before) {
		t.Fatalf("restored result count %d, want %d", len(after), len(before))
	}
	for i := range before {
		if after[i].ID != before[i].ID {
			t.Errorf("rank %d: restored id %d, want %d (ADC ordering must persist)", i, after[i].ID, before[i].ID)
		}
	}
}

// TestNamedPQDropVecsValidationFailLoud proves a PQDropVecs space WITHOUT QuantPQ
// fails loud at create (ErrInvalidPQDropVecs via toConfig().Validate()), so the
// flag is never silently ignored.
func TestNamedPQDropVecsValidationFailLoud(t *testing.T) {
	cfg := map[string]NamedVectorParams{
		"bad": {Dim: 32, Metric: L2, PQDropVecs: true}, // Quant == QuantNone
	}
	_, err := NewNamedCollection("default/named-bad", cfg)
	if err != ErrInvalidPQDropVecs {
		t.Fatalf("PQDropVecs without QuantPQ should fail ErrInvalidPQDropVecs, got %v", err)
	}
}

// TestMVPQDropVecsAutoTrainsEndToEnd: an MV collection with a QuantPQ INNER token
// index + PQDropVecs + a low IVFTrainThreshold, populated via the incremental MV
// Add path, now trains the inner codebooks, DROPS the resident token floats, and
// serves MaxSim search (reconstructing approximate floats from the codes) once the
// live TOKEN count crosses the threshold. A post-drop Add is rejected. Proves
// PQDropVecs is HONORED for the MV inner index.
func TestMVPQDropVecsAutoTrainsEndToEnd(t *testing.T) {
	const (
		dim          = 64
		threshold    = 800
		tokensPerDoc = 8
		docs         = 100 // 100 * 8 = 800 tokens == threshold (the crossing add drops)
		nClusters    = 30
	)
	cfg := MultiVectorConfig{
		Dim: dim, Seed: 7,
		Quant: QuantPQ, RescoreFactor: 3,
		IVFTrainThreshold: threshold, PQDropVecs: true,
	}
	mv, err := NewMultiVectorIndex(cfg)
	if err != nil {
		t.Fatalf("new: %v", err)
	}
	defer mv.Close()

	// Clustered token vectors so the per-doc tokens cluster (real PQ structure).
	pool, queries := pqDropClusteredCorpus(docs*tokensPerDoc, dim, nClusters, 5, 70)
	corpus := make(map[uint64][][]float32, docs)
	for id := uint64(1); id <= docs; id++ {
		toks := make([][]float32, tokensPerDoc)
		for j := 0; j < tokensPerDoc; j++ {
			toks[j] = pool[(int(id)-1)*tokensPerDoc+j]
		}
		corpus[id] = toks
		if err := mv.Add(id, toks, Metadata{"id": NewInt(int64(id))}); err != nil {
			t.Fatalf("add %d: %v", id, err)
		}
	}

	inner, ok := mv.idx.(*hnsw)
	if !ok {
		t.Fatalf("inner index is %T, want *hnsw", mv.idx)
	}
	if inner.pqUntrained() {
		t.Fatal("MV inner PQ-HNSW did NOT auto-train after crossing the token threshold (still inert)")
	}
	if !inner.vecsDropped() {
		t.Fatal("MV inner PQDropVecs did NOT drop the resident token floats after auto-train (stored-but-ignored)")
	}
	if inner.arena.vecs != nil {
		t.Fatal("MV inner arena.vecs must be nil after the auto-train float drop")
	}

	// MaxSim search still works (reconstructs approximate floats from the codes).
	got, err := mv.Search(queries, 5, MultiSearchOpts{CandidatesPerToken: 200})
	if err != nil {
		t.Fatalf("search: %v", err)
	}
	if len(got) == 0 {
		t.Fatal("MV PQDropVecs MaxSim search returned no results after auto-train")
	}

	// Post-drop Add is rejected (the inner index is read-only).
	if err := mv.Add(uint64(docs+1), corpus[1], nil); err != ErrPQDropVecsReadOnly {
		t.Fatalf("post-drop MV Add should return ErrPQDropVecsReadOnly, got %v", err)
	}
}

// TestMVPQDropVecsSnapshotRestore: an incrementally-trained-then-dropped MV inner
// PQ-HNSW index round-trips through snapshot/restore — trained codebooks + codes +
// dropped state persist (ADC-capable, not exact float), so MaxSim still works.
func TestMVPQDropVecsSnapshotRestore(t *testing.T) {
	const (
		dim          = 64
		threshold    = 600
		tokensPerDoc = 6
		docs         = 100 // 600 tokens == threshold
		nClusters    = 25
	)
	cfg := MultiVectorConfig{
		Dim: dim, Seed: 11,
		Quant: QuantPQ, RescoreFactor: 3,
		IVFTrainThreshold: threshold, PQDropVecs: true,
	}
	mv, err := NewMultiVectorIndex(cfg)
	if err != nil {
		t.Fatal(err)
	}
	defer mv.Close()

	pool, queries := pqDropClusteredCorpus(docs*tokensPerDoc, dim, nClusters, 5, 71)
	for id := uint64(1); id <= docs; id++ {
		toks := make([][]float32, tokensPerDoc)
		for j := 0; j < tokensPerDoc; j++ {
			toks[j] = pool[(int(id)-1)*tokensPerDoc+j]
		}
		if err := mv.Add(id, toks, Metadata{"id": NewInt(int64(id))}); err != nil {
			t.Fatalf("add %d: %v", id, err)
		}
	}
	inner := mv.idx.(*hnsw)
	if inner.pqUntrained() || !inner.vecsDropped() {
		t.Fatal("precondition: MV inner must be trained + dropped before snapshot")
	}
	query := queries
	before, err := mv.Search(query, 5, MultiSearchOpts{CandidatesPerToken: 200})
	if err != nil {
		t.Fatal(err)
	}

	var buf bytes.Buffer
	if err := mv.snapshot(&buf); err != nil {
		t.Fatalf("snapshot: %v", err)
	}

	mv2, err := NewMultiVectorIndex(cfg)
	if err != nil {
		t.Fatal(err)
	}
	defer mv2.Close()
	if err := mv2.restore(&buf); err != nil {
		t.Fatalf("restore: %v", err)
	}
	rinner := mv2.idx.(*hnsw)
	if rinner.pqUntrained() {
		t.Fatal("restored MV inner lost its trained codebooks")
	}
	if !rinner.vecsDropped() {
		t.Fatal("restored MV inner lost its dropped state (would be exact float, not ADC)")
	}
	if rinner.arena.vecs != nil {
		t.Fatal("restored MV inner arena.vecs must be nil (dropped state)")
	}
	if mv2.NumDocs() != mv.NumDocs() {
		t.Fatalf("restored NumDocs = %d, want %d", mv2.NumDocs(), mv.NumDocs())
	}
	after, err := mv2.Search(query, 5, MultiSearchOpts{CandidatesPerToken: 200})
	if err != nil {
		t.Fatal(err)
	}
	if len(after) == 0 {
		t.Fatal("restored MV PQDropVecs MaxSim search returned no results")
	}
	if len(after) != len(before) {
		t.Fatalf("restored result count %d, want %d", len(after), len(before))
	}
}

// TestMVPQDropVecsValidationFailLoud proves a PQDropVecs MV config WITHOUT QuantPQ
// fails loud at create (ErrInvalidPQDropVecs via innerConfig().Validate()).
func TestMVPQDropVecsValidationFailLoud(t *testing.T) {
	cfg := MultiVectorConfig{Dim: 32, Seed: 1, PQDropVecs: true} // Quant == QuantNone
	if _, err := NewMultiVectorIndex(cfg); err != ErrInvalidPQDropVecs {
		t.Fatalf("PQDropVecs without QuantPQ should fail ErrInvalidPQDropVecs, got %v", err)
	}
}
