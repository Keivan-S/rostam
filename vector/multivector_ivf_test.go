// SPDX-License-Identifier: Apache-2.0

package vector

import (
	"bytes"
	"errors"
	"math/rand"
	"sort"
	"testing"
)

// mvIVFConfig is a heap-backed MV config with an IVF-Flat inner token index.
// Untrained (the MV inner index is built incrementally, never BuildConcurrent) so
// the inner IVF search is exact brute force — recall is identical to HNSW inner.
func mvIVFConfig(dim int) MultiVectorConfig {
	return MultiVectorConfig{
		Dim: dim, Seed: 1,
		IndexType: IndexIVF, IVFNlist: 8, IVFNprobe: 8,
	}
}

// refMaxSimRanking returns the exact MaxSim top-k ranking (ids) over docs.
func refMaxSimRanking(query [][]float32, docs map[uint64][][]float32, k int) []uint64 {
	type sd struct {
		id    uint64
		score float32
	}
	ref := make([]sd, 0, len(docs))
	for id, toks := range docs {
		ref = append(ref, sd{id, bruteMaxSim(query, toks)})
	}
	sort.Slice(ref, func(i, j int) bool {
		if ref[i].score != ref[j].score {
			return ref[i].score > ref[j].score
		}
		return ref[i].id < ref[j].id
	})
	out := make([]uint64, 0, k)
	for i := 0; i < k && i < len(ref); i++ {
		out = append(out, ref[i].id)
	}
	return out
}

// TestMVIVFBasic exercises add/search/get/delete/scroll on an MV index whose inner
// token index is IVF-Flat. The inner IVF is untrained ⇒ exact brute force, so the
// MaxSim ranking must match the exact reference (same as the HNSW inner default).
func TestMVIVFBasic(t *testing.T) {
	const dim = 16
	rng := rand.New(rand.NewSource(11))
	m, err := NewMultiVectorIndex(mvIVFConfig(dim))
	if err != nil {
		t.Fatalf("new: %v", err)
	}
	defer m.Close()

	docs := map[uint64][][]float32{}
	for id := uint64(1); id <= 40; id++ {
		toks := randTokens(rng, 3+rng.Intn(5), dim)
		docs[id] = toks
		if err := m.Add(id, toks, Metadata{"id": NewInt(int64(id))}); err != nil {
			t.Fatalf("add %d: %v", id, err)
		}
	}
	if m.NumDocs() != 40 {
		t.Fatalf("NumDocs = %d, want 40", m.NumDocs())
	}

	query := randTokens(rng, 5, dim)
	ref := refMaxSimRanking(query, docs, 5)
	got, err := m.Search(query, 5, MultiSearchOpts{CandidatesPerToken: 200})
	if err != nil {
		t.Fatal(err)
	}
	if len(got) != 5 {
		t.Fatalf("got %d results, want 5", len(got))
	}
	for i, r := range got {
		if r.ID != ref[i] {
			t.Errorf("rank %d: got id %d, want %d", i, r.ID, ref[i])
		}
		if r.Metadata["id"].Int != int64(r.ID) {
			t.Errorf("rank %d: metadata id %v, want %d", i, r.Metadata["id"], r.ID)
		}
	}

	// Get returns the exact stored (normalized) token vectors for an IVF-Flat inner.
	gotToks, payload, ver, ok := m.Get(7)
	if !ok {
		t.Fatal("Get(7) not found")
	}
	if ver == 0 {
		t.Error("Get(7) version = 0")
	}
	wantToks := normCopy(docs[7])
	if len(gotToks) != len(wantToks) {
		t.Fatalf("Get(7) tokens = %d, want %d", len(gotToks), len(wantToks))
	}
	for i := range wantToks {
		for d := range wantToks[i] {
			if diff := gotToks[i][d] - wantToks[i][d]; diff > 1e-4 || diff < -1e-4 {
				t.Errorf("Get(7) token[%d][%d] = %f, want %f (IVF-Flat must be exact)", i, d, gotToks[i][d], wantToks[i][d])
			}
		}
	}
	if payload["id"].Int != 7 {
		t.Errorf("Get(7) payload id = %v, want 7", payload["id"])
	}

	// Delete then confirm absence.
	if !m.Delete(7) {
		t.Error("Delete(7) returned false")
	}
	if m.Exists(7) {
		t.Error("doc 7 still exists after delete")
	}
	if _, _, _, ok := m.Get(7); ok {
		t.Error("Get(7) found after delete")
	}
	if m.NumDocs() != 39 {
		t.Errorf("NumDocs = %d after delete, want 39", m.NumDocs())
	}

	// Scroll returns the live docs.
	page, _, _, err := m.ScrollDocsPage(Filter{}, 0, false, 1000)
	if err != nil {
		t.Fatalf("scroll: %v", err)
	}
	if len(page) != 39 {
		t.Errorf("scroll returned %d docs, want 39", len(page))
	}
}

// TestMVIVFMaxSimExactVsHNSW proves the MaxSim ranking + scores from an IVF-Flat
// inner index are IDENTICAL to the HNSW inner default on the SAME data (the floats
// are present in both ⇒ exact dot products). This is the "same-results-ish" proof
// that the vecsForIDs refactor preserves MaxSim semantics for floats-present inners.
func TestMVIVFMaxSimExactVsHNSW(t *testing.T) {
	const dim = 24
	rng := rand.New(rand.NewSource(99))
	docs := map[uint64][][]float32{}
	for id := uint64(1); id <= 50; id++ {
		docs[id] = randTokens(rng, 4+rng.Intn(4), dim)
	}
	query := randTokens(rng, 6, dim)

	build := func(cfg MultiVectorConfig) []MultiResult {
		m, err := NewMultiVectorIndex(cfg)
		if err != nil {
			t.Fatalf("new: %v", err)
		}
		defer m.Close()
		// Add in deterministic id order so both indexes see identical input.
		for id := uint64(1); id <= 50; id++ {
			if err := m.Add(id, docs[id], nil); err != nil {
				t.Fatalf("add: %v", err)
			}
		}
		res, err := m.Search(query, 10, MultiSearchOpts{CandidatesPerToken: 400})
		if err != nil {
			t.Fatalf("search: %v", err)
		}
		return res
	}

	hnsw := build(MultiVectorConfig{Dim: dim, Seed: 1}) // default inner
	ivf := build(MultiVectorConfig{Dim: dim, Seed: 1, IndexType: IndexIVF, IVFNlist: 8, IVFNprobe: 8})

	if len(hnsw) != len(ivf) {
		t.Fatalf("result count: hnsw %d vs ivf %d", len(hnsw), len(ivf))
	}
	for i := range hnsw {
		if hnsw[i].ID != ivf[i].ID {
			t.Errorf("rank %d: hnsw id %d vs ivf id %d", i, hnsw[i].ID, ivf[i].ID)
		}
		if diff := hnsw[i].Score - ivf[i].Score; diff > 1e-4 || diff < -1e-4 {
			t.Errorf("rank %d id %d: hnsw score %.6f vs ivf score %.6f (MaxSim must be exact for IVF-Flat)",
				i, hnsw[i].ID, hnsw[i].Score, ivf[i].Score)
		}
	}
}

// TestMVHNSWDefaultUnchanged is the byte/behaviour guard: a zero-IndexType MV index
// (the default) reproduces the exact MaxSim reference, unchanged by the refactor.
func TestMVHNSWDefaultUnchanged(t *testing.T) {
	const dim = 16
	rng := rand.New(rand.NewSource(7))
	m, err := NewMultiVectorIndex(MultiVectorConfig{Dim: dim, M: 16, EfConstruction: 200, EfSearch: 128, Seed: 1})
	if err != nil {
		t.Fatal(err)
	}
	defer m.Close()
	docs := map[uint64][][]float32{}
	for id := uint64(1); id <= 40; id++ {
		toks := randTokens(rng, 3+rng.Intn(5), dim)
		docs[id] = toks
		if err := m.Add(id, toks, Metadata{"id": NewInt(int64(id))}); err != nil {
			t.Fatal(err)
		}
	}
	query := randTokens(rng, 5, dim)
	ref := refMaxSimRanking(query, docs, 5)
	got, err := m.Search(query, 5, MultiSearchOpts{CandidatesPerToken: 200})
	if err != nil {
		t.Fatal(err)
	}
	for i, r := range got {
		if r.ID != ref[i] {
			t.Errorf("rank %d: got id %d, want %d", i, r.ID, ref[i])
		}
	}
}

// TestMVIVFSnapshotRestore round-trips an IVF-inner MV index through the generic
// snapshot/restore codec (the cluster-snapshot path): the inner IVF state rides the
// m.idx.Snapshot blob, the maps sidecar carries doc/token bookkeeping, and search is
// identical after restore.
func TestMVIVFSnapshotRestore(t *testing.T) {
	const dim = 16
	rng := rand.New(rand.NewSource(5))
	cfg := mvIVFConfig(dim)
	m, err := NewMultiVectorIndex(cfg)
	if err != nil {
		t.Fatal(err)
	}
	defer m.Close()
	for id := uint64(1); id <= 25; id++ {
		if err := m.Add(id, randTokens(rng, 4, dim), Metadata{"id": NewInt(int64(id))}); err != nil {
			t.Fatal(err)
		}
	}
	query := randTokens(rng, 4, dim)
	before, err := m.Search(query, 5, MultiSearchOpts{CandidatesPerToken: 200})
	if err != nil {
		t.Fatal(err)
	}

	var buf bytes.Buffer
	if err := m.snapshot(&buf); err != nil {
		t.Fatalf("snapshot: %v", err)
	}

	// Restore into a fresh MV index built from the SAME create config (the
	// IndexType-persisted-in-create-cfg invariant; here we feed the
	// matching cfg directly, like named).
	m2, err := NewMultiVectorIndex(cfg)
	if err != nil {
		t.Fatal(err)
	}
	defer m2.Close()
	if err := m2.restore(&buf); err != nil {
		t.Fatalf("restore: %v", err)
	}
	if m2.NumDocs() != m.NumDocs() {
		t.Fatalf("restored NumDocs = %d, want %d", m2.NumDocs(), m.NumDocs())
	}
	after, err := m2.Search(query, 5, MultiSearchOpts{CandidatesPerToken: 200})
	if err != nil {
		t.Fatal(err)
	}
	if len(after) != len(before) {
		t.Fatalf("restored got %d results, want %d", len(after), len(before))
	}
	for i := range before {
		if after[i].ID != before[i].ID {
			t.Errorf("rank %d: restored id %d, want %d", i, after[i].ID, before[i].ID)
		}
		if diff := after[i].Score - before[i].Score; diff > 1e-4 || diff < -1e-4 {
			t.Errorf("rank %d id %d: restored score %.6f, want %.6f", i, after[i].ID, after[i].Score, before[i].Score)
		}
	}
}

// TestMVIVFPersistentRejected confirms the single-node mmap instant-restart path is
// fail-loud for an IVF inner index (it has no mmap sidecar; SavePersist unsupported).
// The coexist path for MV-Persistent maps + IVF inner is the cluster snapshot/restore
// path (covered by TestMVIVFSnapshotRestore), never this mmap reopen.
func TestMVIVFPersistentRejected(t *testing.T) {
	cfg := mvIVFConfig(16)
	if _, err := openPersistentMultiVector(cfg); !errors.Is(err, ErrInvalidIVFPersistent) {
		t.Fatalf("openPersistentMultiVector with IVF inner err = %v, want ErrInvalidIVFPersistent", err)
	}

	// Flush on an IVF-inner index also fails loud (heap-backed ⇒ not Persistent here;
	// the inner SavePersist would return ErrPersistUnsupported on a Persistent one).
	m, err := NewMultiVectorIndex(cfg)
	if err != nil {
		t.Fatal(err)
	}
	defer m.Close()
	if err := m.Flush(); err == nil {
		t.Error("Flush on heap-backed IVF-inner index should error")
	}
}

// TestMVIVFBadConfig asserts a malformed IVF inner config fails loud at construction
// (the inner Config.Validate, reached via newIndex).
func TestMVIVFBadConfig(t *testing.T) {
	cases := []struct {
		name string
		cfg  MultiVectorConfig
		want error
	}{
		{"negative nprobe", MultiVectorConfig{Dim: 16, IndexType: IndexIVF, IVFNprobe: -1}, ErrInvalidIVFParams},
		{"pqm not divide dim", MultiVectorConfig{Dim: 18, IndexType: IndexIVF, IVFPQ: true, IVFPQM: 4}, ErrInvalidIVFPQM},
		{"pq without ivf", MultiVectorConfig{Dim: 16, IVFPQ: true}, ErrInvalidIVFPQ},
		{"bad index type", MultiVectorConfig{Dim: 16, IndexType: IndexType(99)}, ErrInvalidIndexType},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			if _, err := NewMultiVectorIndex(tc.cfg); !errors.Is(err, tc.want) {
				t.Fatalf("err = %v, want %v", err, tc.want)
			}
		})
	}
}

// TestMVIVFPQApproxRecall builds an IVF-PQ inner MV index. Because the MV inner
// index is built INCREMENTALLY (never bulk-trained), the IVF-PQ codec stays
// UNTRAINED and the inner index brute-forces exact floats — so MaxSim is still
// EXACT here (IVF-PQ inner compression engages only after a bulk train, a documented
// follow-up). The test asserts correctness (exact ranking) and documents the policy.
func TestMVIVFPQApproxRecall(t *testing.T) {
	const dim = 16
	rng := rand.New(rand.NewSource(3))
	cfg := MultiVectorConfig{Dim: dim, Seed: 1, IndexType: IndexIVF, IVFNlist: 8, IVFNprobe: 8, IVFPQ: true, IVFPQM: 4}
	m, err := NewMultiVectorIndex(cfg)
	if err != nil {
		t.Fatalf("new IVF-PQ inner: %v", err)
	}
	defer m.Close()
	docs := map[uint64][][]float32{}
	for id := uint64(1); id <= 30; id++ {
		toks := randTokens(rng, 4, dim)
		docs[id] = toks
		if err := m.Add(id, toks, nil); err != nil {
			t.Fatal(err)
		}
	}
	query := randTokens(rng, 4, dim)
	ref := refMaxSimRanking(query, docs, 5)
	got, err := m.Search(query, 5, MultiSearchOpts{CandidatesPerToken: 300})
	if err != nil {
		t.Fatal(err)
	}
	// Untrained IVF-PQ inner ⇒ exact brute force ⇒ exact MaxSim ranking.
	hits := 0
	for i := range got {
		if i < len(ref) && got[i].ID == ref[i] {
			hits++
		}
	}
	if hits < len(ref) {
		t.Errorf("IVF-PQ inner (untrained, exact) recall@5 = %d/%d, want exact %d", hits, len(ref), len(ref))
	}
}
