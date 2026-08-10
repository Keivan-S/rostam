// SPDX-License-Identifier: Apache-2.0

package vector

import (
	"bytes"
	"math"
	"math/rand"
	"sort"
	"testing"
	"time"
)

// ivfTestConfig returns an L2 config for the given dim. L2 needs no query
// normalization, so brute-force ground truth is straightforward to compute.
func ivfTestConfig(dim int) Config {
	c := DefaultConfig()
	c.Dim = dim
	c.Metric = L2
	c.Seed = 42
	return c
}

// randVecs returns n deterministic random dim-vectors.
func randVecs(rng *rand.Rand, n, dim int) [][]float32 {
	vs := make([][]float32, n)
	for i := range vs {
		v := make([]float32, dim)
		for d := range v {
			v[d] = rng.Float32()*2 - 1
		}
		vs[i] = v
	}
	return vs
}

// bruteForceNN returns the ids of the k nearest neighbors of q among (ids,vecs)
// under L2, ascending by distance — the ground truth.
func bruteForceNN(q []float32, ids []uint64, vecs [][]float32, k int) []uint64 {
	type sd struct {
		id uint64
		d  float32
	}
	all := make([]sd, len(ids))
	for i := range ids {
		all[i] = sd{ids[i], l2SquaredScalar(q, vecs[i])}
	}
	sort.Slice(all, func(a, b int) bool { return all[a].d < all[b].d })
	out := make([]uint64, 0, k)
	for i := 0; i < k && i < len(all); i++ {
		out = append(out, all[i].id)
	}
	return out
}

func idSet(rs []Result) map[uint64]bool {
	m := make(map[uint64]bool, len(rs))
	for _, r := range rs {
		m[r.ID] = true
	}
	return m
}

// TestIVFUntrainedExactNN: before training, IVF search is exact brute force.
func TestIVFUntrainedExactNN(t *testing.T) {
	dim := 16
	rng := rand.New(rand.NewSource(1))
	ix, err := newIVF(ivfTestConfig(dim))
	if err != nil {
		t.Fatal(err)
	}
	n := 200
	vecs := randVecs(rng, n, dim)
	ids := make([]uint64, n)
	for i := range ids {
		ids[i] = uint64(i + 1)
		if _, _, err := ix.Insert(ids[i], vecs[i], 0, nil, nil, nil, CASCond{}); err != nil {
			t.Fatal(err)
		}
	}
	if ix.trained {
		t.Fatal("index should be untrained after single inserts")
	}

	q := randVecs(rng, 1, dim)[0]
	k := 10
	got, err := ix.Search(q, k)
	if err != nil {
		t.Fatal(err)
	}
	want := bruteForceNN(q, ids, vecs, k)
	if len(got) != k {
		t.Fatalf("got %d results, want %d", len(got), k)
	}
	for i := range want {
		if got[i].ID != want[i] {
			t.Fatalf("untrained result %d = id %d, want exact NN %d", i, got[i].ID, want[i])
		}
	}
}

// clusteredVecs returns n vectors drawn from `nc` gaussian clusters — the
// structured data IVF is designed for, where probing a few cells captures most
// true neighbors. Returns the vectors and their cluster ids.
func clusteredVecs(rng *rand.Rand, n, dim, nc int) [][]float32 {
	centers := make([][]float32, nc)
	for c := range centers {
		cv := make([]float32, dim)
		for d := range cv {
			cv[d] = rng.Float32()*20 - 10 // well-separated centers
		}
		centers[c] = cv
	}
	vs := make([][]float32, n)
	for i := range vs {
		c := centers[rng.Intn(nc)]
		v := make([]float32, dim)
		for d := range v {
			v[d] = c[d] + float32(rng.NormFloat64()) // tight gaussian blob
		}
		vs[i] = v
	}
	return vs
}

// TestIVFTrainedRecall: after BuildConcurrent (train), recall@10 is high vs
// brute-force ground truth on clustered data at a reasonable nprobe, and
// nprobe=nlist is essentially exact.
func TestIVFTrainedRecall(t *testing.T) {
	dim := 32
	rng := rand.New(rand.NewSource(7))
	n := 2000
	vecs := clusteredVecs(rng, n, dim, 40)
	ids := make([]uint64, n)
	for i := range ids {
		ids[i] = uint64(i + 1)
	}

	ix, err := newIVF(ivfTestConfig(dim))
	if err != nil {
		t.Fatal(err)
	}
	if err := ix.BuildConcurrent(ids, vecs, 0); err != nil {
		t.Fatal(err)
	}
	if !ix.trained {
		t.Fatal("index should be trained after BuildConcurrent")
	}
	if ix.nlist < 1 || len(ix.lists) != ix.nlist {
		t.Fatalf("bad nlist/lists: nlist=%d lists=%d", ix.nlist, len(ix.lists))
	}
	// Lists should contain every live slot exactly once.
	total := 0
	for _, l := range ix.lists {
		total += len(l)
	}
	if total != n {
		t.Fatalf("lists hold %d slots, want %d", total, n)
	}

	k := 10
	queries := clusteredVecs(rng, 50, dim, 40)

	// Recall at a reasonable nprobe (a quarter of the cells — a standard IVF
	// operating point that trades a little recall for a big candidate-set cut).
	ix.nprobe = ix.nlist/4 + 1
	hits, denom := 0, 0
	for _, q := range queries {
		want := bruteForceNN(q, ids, vecs, k)
		got, err := ix.Search(q, k)
		if err != nil {
			t.Fatal(err)
		}
		gs := idSet(got)
		for _, w := range want {
			denom++
			if gs[w] {
				hits++
			}
		}
	}
	recall := float64(hits) / float64(denom)
	t.Logf("IVF recall@%d at nprobe=%d (nlist=%d): %.3f", k, ix.nprobe, ix.nlist, recall)
	if recall < 0.80 {
		t.Fatalf("recall@%d = %.3f, want >= 0.80", k, recall)
	}

	// nprobe = nlist => probe every cell => exact.
	ix.nprobe = ix.nlist
	exactHits, exactDenom := 0, 0
	for _, q := range queries {
		want := bruteForceNN(q, ids, vecs, k)
		got, err := ix.Search(q, k)
		if err != nil {
			t.Fatal(err)
		}
		gs := idSet(got)
		for _, w := range want {
			exactDenom++
			if gs[w] {
				exactHits++
			}
		}
	}
	exactRecall := float64(exactHits) / float64(exactDenom)
	t.Logf("IVF recall@%d at nprobe=nlist=%d: %.3f", k, ix.nlist, exactRecall)
	if exactRecall < 0.999 {
		t.Fatalf("nprobe=nlist recall@%d = %.3f, want ~1.0 (exact)", k, exactRecall)
	}
}

// TestIVFFilterFirst: a filtered IVF search returns ONLY matching docs and is
// exact among matches.
func TestIVFFilterFirst(t *testing.T) {
	dim := 16
	rng := rand.New(rand.NewSource(3))
	n := 1000
	vecs := randVecs(rng, n, dim)
	ids := make([]uint64, n)

	ix, err := newIVF(ivfTestConfig(dim))
	if err != nil {
		t.Fatal(err)
	}
	for i := 0; i < n; i++ {
		ids[i] = uint64(i + 1)
		bucket := "miss"
		if i%5 == 0 {
			bucket = "hit"
		}
		meta := Metadata{"bucket": NewString(bucket)}
		if _, _, err := ix.Insert(ids[i], vecs[i], 0, meta, nil, nil, CASCond{}); err != nil {
			t.Fatal(err)
		}
	}
	// Train so the filtered path runs over inverted lists, not brute force.
	// (Insert before build leaves it untrained; explicitly retrain by rebuilding
	// a trained index for this check.)
	trained, err := newIVF(ivfTestConfig(dim))
	if err != nil {
		t.Fatal(err)
	}
	mds := make([]Metadata, n)
	for i := 0; i < n; i++ {
		bucket := "miss"
		if i%5 == 0 {
			bucket = "hit"
		}
		mds[i] = Metadata{"bucket": NewString(bucket)}
	}
	if err := trained.BuildConcurrent(ids, vecs, 0); err != nil {
		t.Fatal(err)
	}
	// BuildConcurrent is vectors-only; attach metadata via SetPayload so the
	// trained index has the same filterable payload.
	for i := 0; i < n; i++ {
		if _, _, _, err := trained.SetPayload(ids[i], mds[i], nil, CASCond{}); err != nil {
			t.Fatal(err)
		}
	}
	trained.nprobe = trained.nlist // exact among the probed cells

	q := randVecs(rng, 1, dim)[0]
	filter := Filter{Op: FilterEq, Field: "bucket", Value: NewString("hit")}
	k := 8
	got, err := trained.SearchFiltered(q, k, filter)
	if err != nil {
		t.Fatal(err)
	}
	if len(got) == 0 {
		t.Fatal("filtered search returned nothing")
	}
	// Build the ground-truth hit-only NN.
	hitIDs := make([]uint64, 0)
	hitVecs := make([][]float32, 0)
	for i := 0; i < n; i++ {
		if i%5 == 0 {
			hitIDs = append(hitIDs, ids[i])
			hitVecs = append(hitVecs, vecs[i])
		}
	}
	want := bruteForceNN(q, hitIDs, hitVecs, k)
	for _, r := range got {
		if (r.ID-1)%5 != 0 {
			t.Fatalf("filtered result id %d is not a 'hit' doc", r.ID)
		}
	}
	for i := range want {
		if got[i].ID != want[i] {
			t.Fatalf("filtered result %d = %d, want exact hit-NN %d", i, got[i].ID, want[i])
		}
	}
}

// TestIVFDeleteAndTTL: deleted and TTL-expired points are excluded from results.
func TestIVFDeleteAndTTL(t *testing.T) {
	dim := 8
	rng := rand.New(rand.NewSource(5))
	n := 300
	vecs := randVecs(rng, n, dim)
	ids := make([]uint64, n)

	ix, err := newIVF(ivfTestConfig(dim))
	if err != nil {
		t.Fatal(err)
	}
	clock := int64(1_000_000)
	ix.now = func() int64 { return clock }
	for i := 0; i < n; i++ {
		ids[i] = uint64(i + 1)
	}
	if err := ix.BuildConcurrent(ids, vecs, 0); err != nil {
		t.Fatal(err)
	}
	ix.nprobe = ix.nlist

	q := vecs[0] // query equals point 1; nearest neighbor is id 1.
	got, _ := ix.Search(q, 5)
	if got[0].ID != 1 {
		t.Fatalf("expected id 1 nearest to itself, got %d", got[0].ID)
	}

	// Delete id 1; it must vanish from results.
	if ok, _ := ix.Delete(1, CASCond{}); !ok {
		t.Fatal("delete returned false")
	}
	got, _ = ix.Search(q, 5)
	if idSet(got)[1] {
		t.Fatal("deleted id 1 still in results")
	}

	// TTL: insert id 9999 with a short TTL, advance the clock, confirm it expires.
	ttlVec := vecs[1]
	if _, _, err := ix.Insert(9999, ttlVec, 50*time.Millisecond, nil, nil, nil, CASCond{}); err != nil {
		t.Fatal(err)
	}
	got, _ = ix.Search(ttlVec, 5)
	if !idSet(got)[9999] {
		t.Fatal("freshly-inserted TTL point not found before expiry")
	}
	clock += 1000 // advance well past the 50ms deadline
	got, _ = ix.Search(ttlVec, 5)
	if idSet(got)[9999] {
		t.Fatal("TTL-expired id 9999 still in results")
	}
}

// TestIVFDataPlaneParity: arena-backed ops (payload set/get, scroll) behave like
// hnsw — spot-check a payload set+get and a scroll listing.
func TestIVFDataPlaneParity(t *testing.T) {
	dim := 8
	rng := rand.New(rand.NewSource(11))
	n := 50
	vecs := randVecs(rng, n, dim)

	ix, err := newIVF(ivfTestConfig(dim))
	if err != nil {
		t.Fatal(err)
	}
	for i := 0; i < n; i++ {
		id := uint64(i + 1)
		meta := Metadata{"k": NewInt(int64(i))}
		if _, _, err := ix.Insert(id, vecs[i], 0, meta, nil, nil, CASCond{}); err != nil {
			t.Fatal(err)
		}
	}

	// Get returns the stored vector + metadata.
	v, meta, _, _, _, ok := ix.Get(7)
	if !ok {
		t.Fatal("Get(7) not ok")
	}
	if len(v) != dim {
		t.Fatalf("Get vec dim %d, want %d", len(v), dim)
	}
	if meta["k"].Int != 6 {
		t.Fatalf("Get meta k=%d, want 6", meta["k"].Int)
	}

	// SetPayload merges; Get reflects it.
	if _, _, _, err := ix.SetPayload(7, Metadata{"extra": NewString("x")}, nil, CASCond{}); err != nil {
		t.Fatal(err)
	}
	_, meta, _, _, _, _ = ix.Get(7)
	if meta["extra"].Str != "x" || meta["k"].Int != 6 {
		t.Fatalf("SetPayload merge wrong: %+v", meta)
	}

	// scrollDocs returns all live docs, id-ascending.
	docs, err := ix.scrollDocs(Filter{}, 0)
	if err != nil {
		t.Fatal(err)
	}
	if len(docs) != n {
		t.Fatalf("scroll returned %d docs, want %d", len(docs), n)
	}
	for i := 1; i < len(docs); i++ {
		if docs[i-1].ID >= docs[i].ID {
			t.Fatal("scroll not id-ascending")
		}
	}

	// scanVectors returns every live record.
	if recs := ix.scanVectors(); len(recs) != n {
		t.Fatalf("scanVectors returned %d, want %d", len(recs), n)
	}

	// Stats reflects size.
	if s := ix.Stats(); s.Size != n {
		t.Fatalf("Stats.Size = %d, want %d", s.Size, n)
	}
}

// TestIVFSnapshotRestore: snapshot->restore reconstructs centroids/lists/trained
// and search results are identical post-restore.
func TestIVFSnapshotRestore(t *testing.T) {
	dim := 16
	rng := rand.New(rand.NewSource(13))
	n := 500
	vecs := randVecs(rng, n, dim)
	ids := make([]uint64, n)
	for i := range ids {
		ids[i] = uint64(i + 1)
	}

	ix, err := newIVF(ivfTestConfig(dim))
	if err != nil {
		t.Fatal(err)
	}
	if err := ix.BuildConcurrent(ids, vecs, 0); err != nil {
		t.Fatal(err)
	}
	// Attach some payload + a TTL so the arena blocks round-trip.
	if _, _, _, err := ix.SetPayload(1, Metadata{"tag": NewString("a")}, nil, CASCond{}); err != nil {
		t.Fatal(err)
	}
	ix.nprobe = 4

	queries := randVecs(rng, 20, dim)
	before := make([][]Result, len(queries))
	for i, q := range queries {
		before[i], _ = ix.Search(q, 10)
	}

	var buf bytes.Buffer
	if err := ix.Snapshot(&buf); err != nil {
		t.Fatal(err)
	}

	restored, err := newIVF(ivfTestConfig(dim))
	if err != nil {
		t.Fatal(err)
	}
	if err := restored.Restore(bytes.NewReader(buf.Bytes())); err != nil {
		t.Fatal(err)
	}
	if !restored.trained {
		t.Fatal("restored index not trained")
	}
	if restored.nlist != ix.nlist || len(restored.centroids) != len(ix.centroids) {
		t.Fatalf("restored nlist/centroids mismatch: %d/%d vs %d/%d",
			restored.nlist, len(restored.centroids), ix.nlist, len(ix.centroids))
	}
	if restored.nprobe != ix.nprobe {
		t.Fatalf("restored nprobe %d, want %d", restored.nprobe, ix.nprobe)
	}

	// Centroids byte-identical.
	for c := range ix.centroids {
		for d := range ix.centroids[c] {
			if math.Abs(float64(ix.centroids[c][d]-restored.centroids[c][d])) > 1e-9 {
				t.Fatalf("centroid %d dim %d differs after restore", c, d)
			}
		}
	}

	// Search results identical post-restore.
	for i, q := range queries {
		got, _ := restored.Search(q, 10)
		if len(got) != len(before[i]) {
			t.Fatalf("query %d: restored %d results, before %d", i, len(got), len(before[i]))
		}
		for j := range got {
			if got[j].ID != before[i][j].ID {
				t.Fatalf("query %d pos %d: restored id %d != before %d", i, j, got[j].ID, before[i][j].ID)
			}
		}
	}

	// Payload survives.
	_, meta, _, _, _, ok := restored.Get(1)
	if !ok || meta["tag"].Str != "a" {
		t.Fatalf("restored payload wrong: ok=%v meta=%+v", ok, meta)
	}
}

// TestIVFExoticVariants: MMR / Recommend / Discover / SearchGroups all WORK on
// IVF (unaccelerated reuse over the candidate gather).
func TestIVFExoticVariants(t *testing.T) {
	dim := 12
	rng := rand.New(rand.NewSource(17))
	n := 400
	vecs := randVecs(rng, n, dim)
	ids := make([]uint64, n)
	for i := range ids {
		ids[i] = uint64(i + 1)
	}
	ix, err := newIVF(ivfTestConfig(dim))
	if err != nil {
		t.Fatal(err)
	}
	if err := ix.BuildConcurrent(ids, vecs, 0); err != nil {
		t.Fatal(err)
	}
	// Group field via payload.
	for i := 0; i < n; i++ {
		if _, _, _, err := ix.SetPayload(ids[i], Metadata{"doc": NewInt(int64(i % 20))}, nil, CASCond{}); err != nil {
			t.Fatal(err)
		}
	}
	ix.nprobe = ix.nlist

	q := randVecs(rng, 1, dim)[0]

	if res, err := ix.SearchMMR(q, 5, MMROpts{Lambda: 0.5}); err != nil || len(res) == 0 {
		t.Fatalf("SearchMMR: err=%v len=%d", err, len(res))
	}
	if res, err := ix.Recommend(5, RecommendOpts{Positive: []uint64{1, 2, 3}}); err != nil || len(res) == 0 {
		t.Fatalf("Recommend: err=%v len=%d", err, len(res))
	}
	if res, err := ix.Discover(5, DiscoverOpts{Context: []ContextPair{{Positive: 1, Negative: 2}}}); err != nil || len(res) == 0 {
		t.Fatalf("Discover: err=%v len=%d", err, len(res))
	}
	if groups, err := ix.SearchGroups(q, 5, GroupOpts{GroupBy: "doc", GroupSize: 2}); err != nil || len(groups) == 0 {
		t.Fatalf("SearchGroups: err=%v len=%d", err, len(groups))
	}
}

// TestIVFSavePersistUnsupported documents the snapshot-only v1 limitation.
// TestIVFSavePersistUnsupportedWithoutMmap: a non-mmap-backed IVF (the snapshot-only
// path — heap arena, vecs present) still rejects SavePersist with ErrPersistUnsupported.
// The instant-restart sidecar requires the vecs to live in the cfg.MmapPath mmap file
// (a Persistent IVF; see TestIVFFlatPersistRoundTrip). This keeps the snapshot-only IVF
// path behaviour-identical.
func TestIVFSavePersistUnsupportedWithoutMmap(t *testing.T) {
	ix, err := newIVF(ivfTestConfig(8))
	if err != nil {
		t.Fatal(err)
	}
	if err := ix.SavePersist("/tmp/ignored"); err != ErrPersistUnsupported {
		t.Fatalf("SavePersist err = %v, want ErrPersistUnsupported", err)
	}
}
