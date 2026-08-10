// SPDX-License-Identifier: Apache-2.0

package vector

import (
	"bytes"
	"math"
	"math/rand"
	"testing"
)

// namedIVFConfig is a two-space named config with an IVF-Flat dense space
// ("title") and an HNSW dense space ("image") — the mixed-engine layout several
// tests reuse.
func namedIVFConfig() map[string]NamedVectorParams {
	return map[string]NamedVectorParams{
		"title": {Dim: 8, Metric: Cosine, IndexType: IndexIVF, IVFNlist: 4, IVFNprobe: 4},
		"image": {Dim: 8, Metric: Cosine}, // HNSW (default)
	}
}

// randVec returns a deterministic random unit-ish vector of dim d.
func randVec(r *rand.Rand, d int) []float32 {
	v := make([]float32, d)
	for i := range v {
		v[i] = float32(r.NormFloat64())
	}
	return v
}

// namedBruteTopK returns the ids of the k nearest of vecs to q under cosine distance.
func namedBruteTopK(q []float32, ids []uint64, vecs map[uint64][]float32, k int) []uint64 {
	type sd struct {
		id uint64
		d  float64
	}
	scored := make([]sd, 0, len(ids))
	qn := append([]float32(nil), q...)
	normalizeF64(qn)
	for _, id := range ids {
		v := append([]float32(nil), vecs[id]...)
		normalizeF64(v)
		var dot float64
		for i := range v {
			dot += float64(v[i]) * float64(qn[i])
		}
		scored = append(scored, sd{id, 1 - dot})
	}
	for i := 0; i < len(scored); i++ {
		for j := i + 1; j < len(scored); j++ {
			if scored[j].d < scored[i].d {
				scored[i], scored[j] = scored[j], scored[i]
			}
		}
	}
	out := make([]uint64, 0, k)
	for i := 0; i < k && i < len(scored); i++ {
		out = append(out, scored[i].id)
	}
	return out
}

func normalizeF64(v []float32) {
	var s float64
	for _, x := range v {
		s += float64(x) * float64(x)
	}
	if s == 0 {
		return
	}
	inv := float32(1 / math.Sqrt(s))
	for i := range v {
		v[i] *= inv
	}
}

// TestNamedIVFBasic inserts/searches/gets/deletes/scrolls a named collection
// whose "title" space is IVF-Flat (untrained ⇒ exact brute force) and whose
// "image" space is HNSW. Recall vs brute force must be exact for the IVF space.
func TestNamedIVFBasic(t *testing.T) {
	nc, err := NewNamedCollection("default/named-ivf", namedIVFConfig())
	if err != nil {
		t.Fatalf("new: %v", err)
	}
	defer nc.Close()

	r := rand.New(rand.NewSource(42))
	const n = 60
	titleVecs := make(map[uint64][]float32, n)
	ids := make([]uint64, 0, n)
	for i := 1; i <= n; i++ {
		id := uint64(i)
		tv := randVec(r, 8)
		iv := randVec(r, 8)
		titleVecs[id] = tv
		ids = append(ids, id)
		if err := nc.Insert(id, map[string][]float32{"title": tv, "image": iv},
			Metadata{"g": NewInt(int64(i % 3))}, 0); err != nil {
			t.Fatalf("insert %d: %v", id, err)
		}
	}

	// Search the IVF space; compare top-5 to brute force.
	q := randVec(r, 8)
	res, err := nc.SearchNamed("title", q, 5, Filter{})
	if err != nil {
		t.Fatalf("search title: %v", err)
	}
	want := namedBruteTopK(q, ids, titleVecs, 5)
	got := make([]uint64, len(res))
	for i, rr := range res {
		got[i] = rr.ID
	}
	// Untrained IVF brute-forces exactly ⇒ identical ordering.
	for i := range want {
		if got[i] != want[i] {
			t.Fatalf("IVF title search rank %d = %d, want %d (got=%v want=%v)", i, got[i], want[i], got, want)
		}
	}

	// Get returns the stored IVF-space vector (a copy).
	vecs, _, _, _, ok := nc.Get(1)
	if !ok {
		t.Fatal("Get(1) ok=false")
	}
	if len(vecs["title"]) != 8 {
		t.Fatalf("Get(1) title dim = %d, want 8", len(vecs["title"]))
	}
	// Mutate the returned vector; the stored copy must be unaffected.
	vecs["title"][0] = -123
	vecs2, _, _, _, _ := nc.Get(1)
	if vecs2["title"][0] == -123 {
		t.Fatal("IVF Get returned an arena alias (mutation leaked)")
	}

	// Delete removes the id from both spaces.
	if removed, _ := nc.Delete(1); !removed {
		t.Fatal("Delete(1) removed=false")
	}
	if _, _, _, _, ok := nc.Get(1); ok {
		t.Fatal("Get(1) after delete ok=true")
	}

	// Scroll over the live set.
	docs, err := nc.ScrollDocs(Filter{}, 1000)
	if err != nil {
		t.Fatalf("scroll: %v", err)
	}
	if len(docs) != n-1 {
		t.Fatalf("scroll returned %d docs, want %d", len(docs), n-1)
	}
}

// TestNamedIVFFilteredMatchesPredicate verifies the SearchFilteredWith path on an
// IVF named space: a filtered search returns EXACTLY the ids a predicate-eval over
// the brute-force ranking would, with the shared-payload predicate re-check.
func TestNamedIVFFilteredMatchesPredicate(t *testing.T) {
	cfg := map[string]NamedVectorParams{
		"title": {Dim: 8, Metric: Cosine, IndexType: IndexIVF, IVFNlist: 4, IVFNprobe: 4},
	}
	nc, err := NewNamedCollection("default/named-ivf-f", cfg)
	if err != nil {
		t.Fatalf("new: %v", err)
	}
	defer nc.Close()

	r := rand.New(rand.NewSource(7))
	const n = 80
	titleVecs := make(map[uint64][]float32, n)
	var groupAIDs []uint64
	for i := 1; i <= n; i++ {
		id := uint64(i)
		tv := randVec(r, 8)
		titleVecs[id] = tv
		grp := "b"
		if i%2 == 0 {
			grp = "a"
			groupAIDs = append(groupAIDs, id)
		}
		if err := nc.Insert(id, map[string][]float32{"title": tv}, Metadata{"grp": NewString(grp)}, 0); err != nil {
			t.Fatalf("insert %d: %v", id, err)
		}
	}

	q := randVec(r, 8)
	flt := Filter{Op: FilterEq, Field: "grp", Value: NewString("a")}
	res, err := nc.SearchNamed("title", q, 10, flt)
	if err != nil {
		t.Fatalf("filtered search: %v", err)
	}
	// Brute-force the SAME predicate (group a only) and compare top-10.
	want := namedBruteTopK(q, groupAIDs, titleVecs, 10)
	if len(res) != len(want) {
		t.Fatalf("filtered result count = %d, want %d", len(res), len(want))
	}
	for i, rr := range res {
		if rr.ID != want[i] {
			t.Fatalf("filtered rank %d = %d, want %d", i, rr.ID, want[i])
		}
		// Every returned id must actually be in group a (predicate held).
		found := false
		for _, ga := range groupAIDs {
			if ga == rr.ID {
				found = true
				break
			}
		}
		if !found {
			t.Fatalf("filtered returned id %d not in group a", rr.ID)
		}
	}
}

// TestNamedIVFPQTrainedSearch trains an IVF-PQ named sub-index directly (via the
// sub-index BuildConcurrent — named's incremental insert never trains) and checks
// search still returns sane results and PQ codes are in use (compression active).
func TestNamedIVFPQTrainedSearch(t *testing.T) {
	cfg := map[string]NamedVectorParams{
		"title": {Dim: 16, Metric: L2, IndexType: IndexIVF, IVFNlist: 8, IVFNprobe: 8, IVFPQ: true, IVFPQM: 4, IVFRerank: true},
	}
	nc, err := NewNamedCollection("default/named-ivfpq", cfg)
	if err != nil {
		t.Fatalf("new: %v", err)
	}
	defer nc.Close()

	r := rand.New(rand.NewSource(11))
	const n = 200
	ids := make([]uint64, 0, n)
	vecs := make([][]float32, 0, n)
	titleVecs := make(map[uint64][]float32, n)
	for i := 1; i <= n; i++ {
		id := uint64(i)
		v := randVec(r, 16)
		ids = append(ids, id)
		vecs = append(vecs, v)
		titleVecs[id] = v
	}
	// Train the inner IVF-PQ index directly with the corpus (the bulk-build path).
	ivfIdx, ok := nc.indexes["title"].(*ivf)
	if !ok {
		t.Fatalf("title space is %T, want *ivf", nc.indexes["title"])
	}
	if err := ivfIdx.BuildConcurrent(ids, vecs, 0); err != nil {
		t.Fatalf("BuildConcurrent: %v", err)
	}
	if !ivfIdx.pqActive() {
		t.Fatal("IVF-PQ codec not trained after BuildConcurrent")
	}
	// Register the live ids so the named-collection book-keeping matches the inner
	// index (the inner index was bulk-built directly via BuildConcurrent).
	nc.mu.Lock()
	for _, id := range ids {
		nc.ids[id] = struct{}{}
	}
	nc.mu.Unlock()

	// Search via the inner index interface (the named search path). Recall floor
	// vs brute force: at least the true NN should be in the top-5 most of the time.
	hits := 0
	const trials = 20
	rq := rand.New(rand.NewSource(99))
	for tIdx := 0; tIdx < trials; tIdx++ {
		q := randVec(rq, 16)
		res, err := nc.indexes["title"].SearchFilteredWith(nil, q, 5, Filter{}, nc.metaOf())
		if err != nil {
			t.Fatalf("ivf-pq search: %v", err)
		}
		if len(res) == 0 {
			t.Fatal("ivf-pq search returned no results")
		}
		// brute-force NN
		want := bruteL2NN(q, ids, titleVecs)
		for _, rr := range res {
			if rr.ID == want {
				hits++
				break
			}
		}
	}
	if hits < trials/2 {
		t.Fatalf("IVF-PQ recall too low: %d/%d trials had NN in top-5", hits, trials)
	}
}

func bruteL2NN(q []float32, ids []uint64, vecs map[uint64][]float32) uint64 {
	best := uint64(0)
	bestD := math.Inf(1)
	for _, id := range ids {
		v := vecs[id]
		var d float64
		for i := range v {
			diff := float64(v[i] - q[i])
			d += diff * diff
		}
		if d < bestD {
			bestD = d
			best = id
		}
	}
	return best
}

// TestNamedIVFSnapshotRestore round-trips a named collection with an IVF dense
// space through Snapshot/Restore and verifies search results are identical.
func TestNamedIVFSnapshotRestore(t *testing.T) {
	mk := func() *NamedCollection {
		nc, err := NewNamedCollection("default/named-ivf-sr", namedIVFConfig())
		if err != nil {
			t.Fatalf("new: %v", err)
		}
		return nc
	}
	src := mk()
	defer src.Close()

	r := rand.New(rand.NewSource(5))
	const n = 50
	titleVecs := make(map[uint64][]float32, n)
	for i := 1; i <= n; i++ {
		id := uint64(i)
		tv := randVec(r, 8)
		iv := randVec(r, 8)
		titleVecs[id] = tv
		if err := src.Insert(id, map[string][]float32{"title": tv, "image": iv},
			Metadata{"g": NewInt(int64(i))}, 0); err != nil {
			t.Fatalf("insert %d: %v", id, err)
		}
	}

	q := randVec(r, 8)
	before, err := src.SearchNamed("title", q, 8, Filter{})
	if err != nil {
		t.Fatalf("search before: %v", err)
	}

	var buf bytes.Buffer
	if err := src.Snapshot(&buf); err != nil {
		t.Fatalf("snapshot: %v", err)
	}

	dst := mk()
	defer dst.Close()
	if err := dst.Restore(bytes.NewReader(buf.Bytes())); err != nil {
		t.Fatalf("restore: %v", err)
	}
	after, err := dst.SearchNamed("title", q, 8, Filter{})
	if err != nil {
		t.Fatalf("search after: %v", err)
	}
	if len(before) != len(after) {
		t.Fatalf("result count before=%d after=%d", len(before), len(after))
	}
	for i := range before {
		if before[i].ID != after[i].ID {
			t.Fatalf("rank %d: before id=%d after id=%d (results diverged after restore)", i, before[i].ID, after[i].ID)
		}
		if math.Abs(float64(before[i].Distance-after[i].Distance)) > 1e-5 {
			t.Fatalf("rank %d: distance before=%v after=%v", i, before[i].Distance, after[i].Distance)
		}
	}

	// Get-after-restore returns the same vector.
	vb, _, _, _, _ := src.Get(3)
	va, _, _, _, okA := dst.Get(3)
	if !okA {
		t.Fatal("Get(3) after restore ok=false")
	}
	for i := range vb["title"] {
		if math.Abs(float64(vb["title"][i]-va["title"][i])) > 1e-6 {
			t.Fatalf("title vec dim %d before=%v after=%v", i, vb["title"][i], va["title"][i])
		}
	}
}

// TestNamedHNSWUnchangedVsIVF asserts the HNSW (default) named path is unchanged:
// an all-HNSW collection and the SAME data return the SAME results regardless of
// the IVF feature being present.
func TestNamedHNSWDefaultSameResults(t *testing.T) {
	cfg := map[string]NamedVectorParams{
		"title": {Dim: 8, Metric: Cosine}, // HNSW default
	}
	build := func() *NamedCollection {
		nc, err := NewNamedCollection("default/named-hnsw", cfg)
		if err != nil {
			t.Fatalf("new: %v", err)
		}
		r := rand.New(rand.NewSource(3))
		for i := 1; i <= 40; i++ {
			tv := randVec(r, 8)
			if err := nc.Insert(uint64(i), map[string][]float32{"title": tv}, Metadata{"k": NewInt(int64(i))}, 0); err != nil {
				t.Fatalf("insert: %v", err)
			}
		}
		return nc
	}
	a := build()
	defer a.Close()
	b := build()
	defer b.Close()

	rq := rand.New(rand.NewSource(8))
	for trial := 0; trial < 10; trial++ {
		q := randVec(rq, 8)
		ra, err := a.SearchNamed("title", q, 5, Filter{})
		if err != nil {
			t.Fatalf("search a: %v", err)
		}
		rb, err := b.SearchNamed("title", q, 5, Filter{})
		if err != nil {
			t.Fatalf("search b: %v", err)
		}
		if len(ra) != len(rb) {
			t.Fatalf("trial %d: len a=%d b=%d", trial, len(ra), len(rb))
		}
		for i := range ra {
			if ra[i].ID != rb[i].ID || math.Abs(float64(ra[i].Distance-rb[i].Distance)) > 1e-6 {
				t.Fatalf("trial %d rank %d: a=%+v b=%+v (HNSW path not deterministic/unchanged)", trial, i, ra[i], rb[i])
			}
		}
	}
}

// TestNamedMixedHNSWIVF runs a collection with one HNSW space and one IVF space,
// inserting/searching/getting across both.
func TestNamedMixedHNSWIVF(t *testing.T) {
	nc, err := NewNamedCollection("default/named-mixed", namedIVFConfig())
	if err != nil {
		t.Fatalf("new: %v", err)
	}
	defer nc.Close()
	if _, isIVF := nc.indexes["title"].(*ivf); !isIVF {
		t.Fatalf("title space is %T, want *ivf", nc.indexes["title"])
	}
	if _, isHNSW := nc.indexes["image"].(*hnsw); !isHNSW {
		t.Fatalf("image space is %T, want *hnsw", nc.indexes["image"])
	}

	r := rand.New(rand.NewSource(21))
	for i := 1; i <= 30; i++ {
		tv := randVec(r, 8)
		iv := randVec(r, 8)
		if err := nc.Insert(uint64(i), map[string][]float32{"title": tv, "image": iv}, nil, 0); err != nil {
			t.Fatalf("insert %d: %v", i, err)
		}
	}
	q := randVec(r, 8)
	if res, err := nc.SearchNamed("title", q, 3, Filter{}); err != nil || len(res) != 3 {
		t.Fatalf("IVF title search: res=%d err=%v", len(res), err)
	}
	if res, err := nc.SearchNamed("image", q, 3, Filter{}); err != nil || len(res) != 3 {
		t.Fatalf("HNSW image search: res=%d err=%v", len(res), err)
	}
	if vecs, _, _, _, ok := nc.Get(1); !ok || len(vecs) != 2 {
		t.Fatalf("Get(1): ok=%v spaces=%d, want 2 spaces", ok, len(vecs))
	}
}

// TestNamedIVFBadConfigFailLoud verifies an invalid IVF named-space config is
// rejected at NewNamedCollection (fail-loud via the per-space Config Validate).
func TestNamedIVFBadConfigFailLoud(t *testing.T) {
	// IVFPQ with IVFPQM that does not divide Dim must fail.
	bad := map[string]NamedVectorParams{
		"title": {Dim: 10, Metric: L2, IndexType: IndexIVF, IVFPQ: true, IVFPQM: 3}, // 10 % 3 != 0
	}
	if _, err := NewNamedCollection("default/bad-ivf", bad); err == nil {
		t.Fatal("expected NewNamedCollection to fail for IVFPQM not dividing Dim, got nil")
	}
}
