// SPDX-License-Identifier: Apache-2.0

package vector

import (
	"bytes"
	"errors"
	"math"
	"math/rand"
	"os"
	"path/filepath"
	"testing"
)

// ivfPersistConfig builds a Persistent IVF Config wired with a vecs mmap path under
// dir, the way the store will — the test sets cfg.MmapPath directly
// and calls SavePersist/openPersistIVF. IndexType is IVF; nlist/nprobe small for tests.
func ivfPersistConfig(dir string, dim int) Config {
	c := DefaultConfig()
	c.Dim = dim
	c.Metric = L2
	c.Seed = 42
	c.IndexType = IndexIVF
	c.IVFNlist = 16
	c.IVFNprobe = 8
	c.Persistent = true
	c.MmapPath = filepath.Join(dir, "ivf.vecs")
	return c
}

// resultsIdentical asserts two result slices have identical ids and distances (the
// instant-restart round-trip must reproduce search EXACTLY — same centroids/lists/codes).
func resultsIdentical(t *testing.T, before, after []Result, qi int) {
	t.Helper()
	if len(before) != len(after) {
		t.Fatalf("query %d: result len %d != %d", qi, len(after), len(before))
	}
	for i := range before {
		if before[i].ID != after[i].ID {
			t.Fatalf("query %d rank %d: id %d != %d", qi, i, after[i].ID, before[i].ID)
		}
		if math.Abs(float64(before[i].Distance-after[i].Distance)) > 1e-6 {
			t.Fatalf("query %d rank %d: distance %v != %v", qi, i, after[i].Distance, before[i].Distance)
		}
	}
}

// TestIVFFlatPersistRoundTrip: an IVF-Flat Persistent index — built past train, with
// payload + TTL — SavePersist → Close → openPersistIVF reproduces search/get/scroll
// IDENTICALLY, and the restored vecs come from the mmap file (zero-copy), not a re-read.
func TestIVFFlatPersistRoundTrip(t *testing.T) {
	dir := t.TempDir()
	dim := 16
	rng := rand.New(rand.NewSource(7))
	n := 600
	vecs := randVecs(rng, n, dim)
	ids := make([]uint64, n)
	for i := range ids {
		ids[i] = uint64(i + 1)
	}

	cfg := ivfPersistConfig(dir, dim)
	ix, err := newIVF(cfg)
	if err != nil {
		t.Fatal(err)
	}
	if ix.arena.mmapF == nil {
		t.Fatal("Persistent IVF arena must be mmap-backed (newIVF useMmap)")
	}
	if err := ix.BuildConcurrent(ids, vecs, 0); err != nil {
		t.Fatal(err)
	}
	if !ix.trained {
		t.Fatal("index should be trained")
	}
	if _, _, _, err := ix.SetPayload(1, Metadata{"tag": NewString("a")}, nil, CASCond{}); err != nil {
		t.Fatal(err)
	}
	ix.nprobe = 4

	queries := randVecs(rng, 20, dim)
	before := make([][]Result, len(queries))
	for i, q := range queries {
		before[i], _ = ix.Search(q, 10)
	}
	getVec1, getMeta1, _, _, _, ok1 := ix.Get(1)
	if !ok1 {
		t.Fatal("Get(1) before save not ok")
	}
	scrollBefore, err := ix.scrollDocs(Filter{}, 0)
	if err != nil {
		t.Fatal(err)
	}

	metaPath := filepath.Join(dir, "ivf.meta")
	if err := ix.SavePersist(metaPath); err != nil {
		t.Fatalf("SavePersist: %v", err)
	}
	if err := ix.Close(); err != nil {
		t.Fatal(err)
	}

	restored, err := openPersistIVF(cfg, metaPath)
	if err != nil {
		t.Fatalf("openPersistIVF: %v", err)
	}
	defer restored.Close()

	// PROOF the vecs came from the mmap file, not re-read into the heap: the restored
	// arena is mmap-backed and its float slice aliases the mapped region.
	if restored.arena.mmapF == nil {
		t.Fatal("restored arena is not mmap-backed (vecs were re-read, not mapped)")
	}
	if len(restored.arena.vecs) != n*dim {
		t.Fatalf("restored mmap vecs len %d, want %d", len(restored.arena.vecs), n*dim)
	}

	if !restored.trained || restored.nlist != ix.nlist {
		t.Fatalf("restored trained=%v nlist=%d, want true/%d", restored.trained, restored.nlist, ix.nlist)
	}
	if restored.nprobe != 4 {
		t.Fatalf("restored nprobe %d, want 4", restored.nprobe)
	}
	for i, q := range queries {
		got, _ := restored.Search(q, 10)
		resultsIdentical(t, before[i], got, i)
	}
	getVec2, getMeta2, _, _, _, ok2 := restored.Get(1)
	if !ok2 {
		t.Fatal("Get(1) after restore not ok")
	}
	for d := range getVec1 {
		if math.Abs(float64(getVec1[d]-getVec2[d])) > 1e-6 {
			t.Fatalf("Get(1) vec dim %d differs after restore", d)
		}
	}
	if getMeta1["tag"].Str != getMeta2["tag"].Str {
		t.Fatalf("Get(1) meta differs: %q vs %q", getMeta2["tag"].Str, getMeta1["tag"].Str)
	}
	scrollAfter, err := restored.scrollDocs(Filter{}, 0)
	if err != nil {
		t.Fatal(err)
	}
	if len(scrollBefore) != len(scrollAfter) {
		t.Fatalf("scroll count %d != %d", len(scrollAfter), len(scrollBefore))
	}
}

// TestIVFIncrementalPersistRoundTrip: insert past the auto-train threshold (so lists +
// slotCell grow on incremental assignToList), then a few MORE inserts after train, then
// SavePersist → reopen reproduces search identically (the post-train list growth is
// captured in the sidecar).
func TestIVFIncrementalPersistRoundTrip(t *testing.T) {
	dir := t.TempDir()
	dim := 16
	rng := rand.New(rand.NewSource(11))
	cfg := ivfPersistConfig(dir, dim)
	cfg.IVFTrainThreshold = 200 // small so incremental inserts cross train in-test
	ix, err := newIVF(cfg)
	if err != nil {
		t.Fatal(err)
	}
	n := 300
	vecs := randVecs(rng, n, dim)
	for i := 0; i < n; i++ {
		if _, _, err := ix.Insert(uint64(i+1), vecs[i], 0, nil, nil, nil, CASCond{}); err != nil {
			t.Fatal(err)
		}
	}
	if !ix.trained {
		t.Fatal("index should auto-train after crossing threshold")
	}
	// A few more inserts AFTER train → these go through incremental assignToList (list
	// + slot growth past the train sample).
	more := randVecs(rng, 50, dim)
	for i := 0; i < len(more); i++ {
		if _, _, err := ix.Insert(uint64(1000+i), more[i], 0, nil, nil, nil, CASCond{}); err != nil {
			t.Fatal(err)
		}
	}
	ix.nprobe = 6

	queries := randVecs(rng, 15, dim)
	before := make([][]Result, len(queries))
	for i, q := range queries {
		before[i], _ = ix.Search(q, 10)
	}
	listLens := make([]int, len(ix.lists))
	for i := range ix.lists {
		listLens[i] = len(ix.lists[i])
	}

	metaPath := filepath.Join(dir, "ivf.meta")
	if err := ix.SavePersist(metaPath); err != nil {
		t.Fatal(err)
	}
	if err := ix.Close(); err != nil {
		t.Fatal(err)
	}
	restored, err := openPersistIVF(cfg, metaPath)
	if err != nil {
		t.Fatal(err)
	}
	defer restored.Close()

	if len(restored.lists) != len(listLens) {
		t.Fatalf("restored lists count %d, want %d", len(restored.lists), len(listLens))
	}
	for i := range listLens {
		if len(restored.lists[i]) != listLens[i] {
			t.Fatalf("restored list %d len %d, want %d", i, len(restored.lists[i]), listLens[i])
		}
	}
	for i, q := range queries {
		got, _ := restored.Search(q, 10)
		resultsIdentical(t, before[i], got, i)
	}
}

// TestIVFPQPersistRoundTrip: IVF-PQ with vecs present (IVFRerank) — SavePersist → reopen
// reproduces search identically and the codebooks + residual codes + slotCell survive.
// The vecs stay mmap-backed (IVFRerank keeps floats for exact rescore).
func TestIVFPQPersistRoundTrip(t *testing.T) {
	dir := t.TempDir()
	dim := 16
	rng := rand.New(rand.NewSource(19))
	cfg := ivfPQConfig(dim, 16, 4, true) // IVFRerank => vecs present
	cfg.Persistent = true
	cfg.MmapPath = filepath.Join(dir, "ivf.vecs")
	ix, err := newIVF(cfg)
	if err != nil {
		t.Fatal(err)
	}
	if ix.arena.mmapF == nil {
		t.Fatal("Persistent IVF-PQ (rerank) arena must be mmap-backed")
	}
	n := 800
	vecs := randVecs(rng, n, dim)
	ids := make([]uint64, n)
	for i := range ids {
		ids[i] = uint64(i + 1)
	}
	if err := ix.BuildConcurrent(ids, vecs, 0); err != nil {
		t.Fatal(err)
	}
	if ix.pq == nil {
		t.Fatal("IVF-PQ codec must be trained")
	}
	ix.nprobe = 6

	queries := randVecs(rng, 15, dim)
	before := make([][]Result, len(queries))
	for i, q := range queries {
		before[i], _ = ix.Search(q, 10)
	}
	metaPath := filepath.Join(dir, "ivf.meta")
	if err := ix.SavePersist(metaPath); err != nil {
		t.Fatal(err)
	}
	if err := ix.Close(); err != nil {
		t.Fatal(err)
	}
	restored, err := openPersistIVF(cfg, metaPath)
	if err != nil {
		t.Fatal(err)
	}
	defer restored.Close()
	if restored.pq == nil {
		t.Fatal("restored IVF-PQ codec nil")
	}
	if restored.arena.mmapF == nil {
		t.Fatal("restored IVF-PQ (rerank) arena not mmap-backed")
	}
	if len(restored.slotCell) != len(ix.slotCell) {
		t.Fatalf("restored slotCell len %d, want %d", len(restored.slotCell), len(ix.slotCell))
	}
	for i, q := range queries {
		got, _ := restored.Search(q, 10)
		resultsIdentical(t, before[i], got, i)
	}
}

// TestIVFPQDroppedPersistRoundTrip: IVF-PQ-only with the float vectors DROPPED
// (PQDropVecs) — there is NO vecs file; the sidecar carries the residual codes
// VERBATIM and openPersistIVF restores them via the dropped path (no re-encode, no
// vecs mmap). Search reproduces identically.
func TestIVFPQDroppedPersistRoundTrip(t *testing.T) {
	dir := t.TempDir()
	dim := 16
	rng := rand.New(rand.NewSource(23))
	cfg := ivfPQConfig(dim, 16, 4, false) // PQ-only (no rerank) => floats dropped at train
	cfg.Persistent = true
	cfg.MmapPath = filepath.Join(dir, "ivf.vecs")
	ix, err := newIVF(cfg)
	if err != nil {
		t.Fatal(err)
	}
	// PQDropVecs stays heap (no mmap): the floats are dropped after train.
	if ix.arena.mmapF != nil {
		t.Fatal("PQDropVecs IVF should NOT be mmap-backed (floats are dropped)")
	}
	n := 800
	vecs := randVecs(rng, n, dim)
	ids := make([]uint64, n)
	for i := range ids {
		ids[i] = uint64(i + 1)
	}
	if err := ix.BuildConcurrent(ids, vecs, 0); err != nil {
		t.Fatal(err)
	}
	if !ix.pqDropped || !ix.arena.vecsDropped {
		t.Fatal("PQ-only build should have dropped the float vectors")
	}
	ix.nprobe = 6

	queries := randVecs(rng, 15, dim)
	before := make([][]Result, len(queries))
	for i, q := range queries {
		before[i], _ = ix.Search(q, 10)
	}
	metaPath := filepath.Join(dir, "ivf.meta")
	if err := ix.SavePersist(metaPath); err != nil {
		t.Fatal(err)
	}
	// No vecs file should have been created for the dropped case.
	if _, statErr := os.Stat(cfg.MmapPath); statErr == nil {
		t.Fatal("dropped IVF-PQ wrote a vecs file; it must not")
	}
	if err := ix.Close(); err != nil {
		t.Fatal(err)
	}
	restored, err := openPersistIVF(cfg, metaPath)
	if err != nil {
		t.Fatal(err)
	}
	defer restored.Close()
	if !restored.pqDropped || !restored.arena.vecsDropped {
		t.Fatal("restored PQ-only index should be in the dropped state")
	}
	if restored.arena.mmapF != nil {
		t.Fatal("restored dropped IVF-PQ must NOT map a vecs file")
	}
	for i, q := range queries {
		got, _ := restored.Search(q, 10)
		resultsIdentical(t, before[i], got, i)
	}
}

// TestIVFOPQPersistRoundTrip: an IVF-PQ index with OPQ enabled — the OPQ rotation R
// survives the sidecar round-trip (search identical means the codec rotates the same
// way post-restore).
func TestIVFOPQPersistRoundTrip(t *testing.T) {
	dir := t.TempDir()
	dim := 16
	rng := rand.New(rand.NewSource(29))
	cfg := ivfPQConfig(dim, 16, 4, true)
	cfg.OPQ = true
	cfg.Persistent = true
	cfg.MmapPath = filepath.Join(dir, "ivf.vecs")
	ix, err := newIVF(cfg)
	if err != nil {
		t.Fatal(err)
	}
	n := 800
	vecs := randVecs(rng, n, dim)
	ids := make([]uint64, n)
	for i := range ids {
		ids[i] = uint64(i + 1)
	}
	if err := ix.BuildConcurrent(ids, vecs, 0); err != nil {
		t.Fatal(err)
	}
	if ix.pq == nil || ix.pq.rotation == nil {
		t.Fatal("OPQ rotation R should be trained")
	}
	ix.nprobe = 6
	queries := randVecs(rng, 15, dim)
	before := make([][]Result, len(queries))
	for i, q := range queries {
		before[i], _ = ix.Search(q, 10)
	}
	metaPath := filepath.Join(dir, "ivf.meta")
	if err := ix.SavePersist(metaPath); err != nil {
		t.Fatal(err)
	}
	if err := ix.Close(); err != nil {
		t.Fatal(err)
	}
	restored, err := openPersistIVF(cfg, metaPath)
	if err != nil {
		t.Fatal(err)
	}
	defer restored.Close()
	if restored.pq == nil || restored.pq.rotation == nil {
		t.Fatal("restored OPQ rotation R nil")
	}
	if len(restored.pq.rotation) != dim*dim {
		t.Fatalf("restored R len %d, want %d", len(restored.pq.rotation), dim*dim)
	}
	for i := range ix.pq.rotation {
		if math.Abs(float64(ix.pq.rotation[i]-restored.pq.rotation[i])) > 1e-9 {
			t.Fatalf("R[%d] differs after restore", i)
		}
	}
	for i, q := range queries {
		got, _ := restored.Search(q, 10)
		resultsIdentical(t, before[i], got, i)
	}
}

// TestIVFPersistHeaderMismatch: openPersistIVF fails loud on a header dim/metric
// mismatch (never a silent wrong-restore), and on a corrupt magic.
func TestIVFPersistHeaderMismatch(t *testing.T) {
	dir := t.TempDir()
	dim := 16
	rng := rand.New(rand.NewSource(31))
	cfg := ivfPersistConfig(dir, dim)
	ix, err := newIVF(cfg)
	if err != nil {
		t.Fatal(err)
	}
	n := 300
	vecs := randVecs(rng, n, dim)
	ids := make([]uint64, n)
	for i := range ids {
		ids[i] = uint64(i + 1)
	}
	if err := ix.BuildConcurrent(ids, vecs, 0); err != nil {
		t.Fatal(err)
	}
	metaPath := filepath.Join(dir, "ivf.meta")
	if err := ix.SavePersist(metaPath); err != nil {
		t.Fatal(err)
	}
	_ = ix.Close()

	// Wrong dim → ErrPersistMismatch.
	badDim := cfg
	badDim.Dim = 32
	badDim.MmapPath = filepath.Join(dir, "other.vecs")
	if _, err := openPersistIVF(badDim, metaPath); !errors.Is(err, ErrPersistMismatch) {
		t.Fatalf("wrong-dim open err = %v, want ErrPersistMismatch", err)
	}
	// Wrong metric → ErrPersistMismatch.
	badMetric := cfg
	badMetric.Metric = Cosine
	if _, err := openPersistIVF(badMetric, metaPath); !errors.Is(err, ErrPersistMismatch) {
		t.Fatalf("wrong-metric open err = %v, want ErrPersistMismatch", err)
	}
	// Corrupt magic → ErrPersistFormat.
	corrupt := filepath.Join(dir, "corrupt.meta")
	data, _ := os.ReadFile(metaPath)
	data[0] ^= 0xFF
	if err := os.WriteFile(corrupt, data, 0o600); err != nil {
		t.Fatal(err)
	}
	if _, err := openPersistIVF(cfg, corrupt); !errors.Is(err, ErrPersistFormat) {
		t.Fatalf("corrupt-magic open err = %v, want ErrPersistFormat", err)
	}
}

// TestIVFSnapshotPathUnchangedBySidecar: the snapshot-only IVF path (Persistent=false)
// is UNCHANGED by the sidecar feature — the on-disk snapshot bytes are byte-identical
// before/after the writeArena vecsMode refactor (a heap, non-persistent IVF still
// snapshots via vecsInlineMode), and a Snapshot→Restore round-trip reproduces search.
func TestIVFSnapshotPathUnchangedBySidecar(t *testing.T) {
	dim := 16
	rng := rand.New(rand.NewSource(37))
	n := 500
	vecs := randVecs(rng, n, dim)
	ids := make([]uint64, n)
	for i := range ids {
		ids[i] = uint64(i + 1)
	}
	ix, err := newIVF(ivfTestConfig(dim)) // non-persistent, heap arena
	if err != nil {
		t.Fatal(err)
	}
	if ix.arena.mmapF != nil {
		t.Fatal("non-persistent IVF must stay heap-backed (snapshot path)")
	}
	if err := ix.BuildConcurrent(ids, vecs, 0); err != nil {
		t.Fatal(err)
	}
	ix.nprobe = 4
	queries := randVecs(rng, 10, dim)
	before := make([][]Result, len(queries))
	for i, q := range queries {
		before[i], _ = ix.Search(q, 10)
	}

	var buf bytes.Buffer
	if err := ix.Snapshot(&buf); err != nil {
		t.Fatal(err)
	}
	// The arena marker in a non-persistent snapshot must be vecsInlineMode (the float
	// block is inlined) — never the sidecar external marker. The marker sits right
	// after the 8-byte magic, the version u32, dim u32, metric byte, M u32,
	// EfConstruction u32, EfSearch u32, Seed i64, then the arena's size u32 +
	// capacity u32, then the hasVecs byte.
	snap := buf.Bytes()
	off := 8 + 4 + 4 + 1 + 4 + 4 + 4 + 8 + 4 + 4
	if snap[off] != vecsInlineMode {
		t.Fatalf("snapshot arena marker = %d, want vecsInlineMode(%d) — sidecar refactor changed the snapshot format", snap[off], vecsInlineMode)
	}

	restored, err := newIVF(ivfTestConfig(dim))
	if err != nil {
		t.Fatal(err)
	}
	if err := restored.Restore(bytes.NewReader(snap)); err != nil {
		t.Fatal(err)
	}
	for i, q := range queries {
		got, _ := restored.Search(q, 10)
		resultsIdentical(t, before[i], got, i)
	}
}
