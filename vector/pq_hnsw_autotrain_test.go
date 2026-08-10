// SPDX-License-Identifier: Apache-2.0

package vector

import (
	"bytes"
	"encoding/binary"
	"math"
	"math/rand"
	"sync"
	"testing"
)

// hnswPQAutoTrainConfig is a PQ-HNSW config with a LOW IVFTrainThreshold so the
// incremental Insert path trips the deterministic auto-train quickly in tests
// (the production default, defaultIVFTrainThreshold=2048, keeps small collections
// exact-brute-force). Mirror of ivfAutoTrainConfig for the HNSW-PQ family.
func hnswPQAutoTrainConfig(dim, threshold, m int, drop bool) Config {
	return Config{
		Dim: dim, Metric: L2, M: 16, EfConstruction: 200, EfSearch: 64, Seed: 42,
		Quant: QuantPQ, QuantPQM: m, PQDropVecs: drop, IVFTrainThreshold: threshold,
	}
}

// dumpCodebooks serializes the trained PQ codebooks to a deterministic byte
// stream ([m][centroids][dsub] float32 bit-patterns in fixed order) for a
// byte-identical cross-index comparison. Must be a trained pqQuantizer.
func dumpCodebooks(t *testing.T, h *hnsw) []byte {
	t.Helper()
	pq, ok := h.quant.(*pqQuantizer)
	if !ok || !pq.trained() {
		t.Fatalf("dumpCodebooks: index is not a trained pqQuantizer (ok=%v)", ok)
	}
	cb := pq.codebooks()
	var buf bytes.Buffer
	var u [4]byte
	for _, sub := range cb {
		for _, cen := range sub {
			for _, f := range cen {
				binary.LittleEndian.PutUint32(u[:], math.Float32bits(f))
				buf.Write(u[:])
			}
		}
	}
	return buf.Bytes()
}

// dumpCodes serializes every live slot's PQ code (in ascending slot order) for a
// byte-identical cross-index comparison of the per-slot encodings.
func dumpCodes(t *testing.T, h *hnsw, n int) []byte {
	t.Helper()
	var buf bytes.Buffer
	for slot := 0; slot < n; slot++ {
		buf.Write(h.arena.Code(uint32(slot)))
	}
	return buf.Bytes()
}

// insertAllHNSW inserts (ids,vecs) into h via the incremental Insert path.
func insertAllHNSW(t *testing.T, h *hnsw, ids []uint64, vecs [][]float32) {
	t.Helper()
	for i := range ids {
		if _, _, err := h.Insert(ids[i], vecs[i], 0, nil, nil, nil, CASCond{}); err != nil {
			t.Fatalf("Insert(%d): %v", ids[i], err)
		}
	}
}

// TestPQHNSWAutoTrainReplicaDeterminism is THE load-bearing proof: two HNSW-PQ
// indexes that apply the IDENTICAL inserts in the IDENTICAL order must, after the
// incremental auto-train trips, hold BYTE-IDENTICAL codebooks AND byte-identical
// per-slot PQ codes. This is the cross-replica consistency guarantee — a
// divergence would mean different ADC distances on different replicas (silent
// wrong results). Determinism comes from: (1) the trigger is a pure function of
// applied state (arena size) + the replicated threshold; (2) the sample is
// slot-ordered, not Go-map-ordered; (3) the k-means seed is cfg.Seed, not
// rand/wall-clock.
func TestPQHNSWAutoTrainReplicaDeterminism(t *testing.T) {
	const (
		dim       = 32
		threshold = 600
		n         = 700 // > threshold so the auto-train trips during the inserts
		m         = 8
	)
	ids, vecs := siftLikeCorpus(n, dim, 123)
	cfg := hnswPQAutoTrainConfig(dim, threshold, m, false)

	a, err := newHNSW(cfg)
	if err != nil {
		t.Fatal(err)
	}
	b, err := newHNSW(cfg)
	if err != nil {
		t.Fatal(err)
	}
	insertAllHNSW(t, a, ids, vecs)
	insertAllHNSW(t, b, ids, vecs)

	// Both must have auto-trained (live count crossed the threshold).
	if a.pqUntrained() || b.pqUntrained() {
		t.Fatalf("both indexes must be trained after incremental auto-train (a=%v b=%v)",
			a.pqUntrained(), b.pqUntrained())
	}

	cbA, cbB := dumpCodebooks(t, a), dumpCodebooks(t, b)
	if !bytes.Equal(cbA, cbB) {
		t.Fatalf("codebooks DIVERGED across replicas (len %d vs %d) — non-deterministic train", len(cbA), len(cbB))
	}
	codesA, codesB := dumpCodes(t, a, n), dumpCodes(t, b, n)
	if !bytes.Equal(codesA, codesB) {
		t.Fatalf("per-slot PQ codes DIVERGED across replicas — non-deterministic encode")
	}
	t.Logf("replica-determinism OK: %d codebook bytes + %d code bytes byte-identical", len(cbA), len(codesA))
}

// TestPQHNSWAutoTrainBelowThreshold: fewer live vectors than the threshold leaves
// the index UNTRAINED — search falls back to EXACT float (correct, no
// approximation). This is the pre-train fallback half of the policy.
func TestPQHNSWAutoTrainBelowThreshold(t *testing.T) {
	const (
		dim       = 32
		threshold = 500
		n         = 300 // < threshold
		m         = 8
	)
	ids, vecs := siftLikeCorpus(n, dim, 9)
	h, err := newHNSW(hnswPQAutoTrainConfig(dim, threshold, m, false))
	if err != nil {
		t.Fatal(err)
	}
	insertAllHNSW(t, h, ids, vecs)
	if !h.pqUntrained() {
		t.Fatal("index with live count below threshold must stay untrained (exact fallback)")
	}
	// Untrained = exact float: every vector's own NN is itself, exactly.
	for i := 0; i < 50; i++ {
		res, err := h.Search(vecs[i], 1)
		if err != nil {
			t.Fatal(err)
		}
		if len(res) == 0 || res[0].ID != ids[i] {
			t.Fatalf("pre-train exact search: query %d got %v, want self id %d", i, res, ids[i])
		}
	}
}

// TestPQHNSWAutoTrainCrossesThreshold proves the threshold crossing flips the
// index from exact-fallback to trained-ADC: below the threshold pqUntrained() is
// true; the insert that crosses it trains synchronously; afterwards search is ADC
// with sane recall.
func TestPQHNSWAutoTrainCrossesThreshold(t *testing.T) {
	const (
		dim       = 64
		threshold = 1000
		n         = 1500
		nClusters = 30
		m         = 16
		k         = 10
	)
	ids, vecs, queries := buildPQDropCorpus(n, dim, nClusters, 17)
	h, err := newHNSW(hnswPQAutoTrainConfig(dim, threshold, m, false))
	if err != nil {
		t.Fatal(err)
	}
	// Insert exactly threshold-1 points: still untrained.
	insertAllHNSW(t, h, ids[:threshold-1], vecs[:threshold-1])
	if !h.pqUntrained() {
		t.Fatalf("at %d live (< %d) the index must still be untrained", threshold-1, threshold)
	}
	// The insert that crosses the threshold trains synchronously.
	if _, _, err := h.Insert(ids[threshold-1], vecs[threshold-1], 0, nil, nil, nil, CASCond{}); err != nil {
		t.Fatal(err)
	}
	if h.pqUntrained() {
		t.Fatalf("the insert crossing the threshold (%d live) must train", threshold)
	}
	// Insert the remainder (post-train assign path) and check ADC recall is sane.
	insertAllHNSW(t, h, ids[threshold:], vecs[threshold:])
	rec := recallOf(t, h, vecs, queries, k)
	t.Logf("post-auto-train ADC recall@%d=%.3f", k, rec)
	if rec < 0.40 {
		t.Fatalf("post-auto-train ADC recall@%d=%.3f below floor 0.40", k, rec)
	}
}

// TestPQHNSWAutoTrainTrainOnce proves the auto-train fires EXACTLY ONCE: after the
// first crossing the codebooks are fixed, and a SECOND threshold crossing (many
// more inserts) does NOT retrain — the codebooks stay byte-identical.
func TestPQHNSWAutoTrainTrainOnce(t *testing.T) {
	const (
		dim       = 32
		threshold = 400
		m         = 8
	)
	ids, vecs := siftLikeCorpus(2*threshold, dim, 55)
	h, err := newHNSW(hnswPQAutoTrainConfig(dim, threshold, m, false))
	if err != nil {
		t.Fatal(err)
	}
	// Cross the threshold once.
	insertAllHNSW(t, h, ids[:threshold], vecs[:threshold])
	if h.pqUntrained() {
		t.Fatal("must be trained after the first crossing")
	}
	before := dumpCodebooks(t, h)
	// Insert a lot more (well past a hypothetical second threshold): must NOT retrain.
	insertAllHNSW(t, h, ids[threshold:], vecs[threshold:])
	after := dumpCodebooks(t, h)
	if !bytes.Equal(before, after) {
		t.Fatal("codebooks changed after a second threshold crossing — auto-train is NOT train-once")
	}
}

// TestPQHNSWAutoTrainMatchesBuildConcurrent proves the incremental auto-trained
// index has equivalent TRAINED search behaviour to a BuildConcurrent-trained
// index on the SAME data: both navigate on ADC, so their recall is within noise.
// (The graphs differ — serial Insert vs concurrent link — so result ids are not
// byte-identical, but the trained quantizer quality is equivalent.)
func TestPQHNSWAutoTrainMatchesBuildConcurrent(t *testing.T) {
	const (
		dim       = 64
		n         = 3000
		nClusters = 40
		m         = 16
		k         = 10
		threshold = 500 // < n so the incremental path trains
	)
	ids, vecs, queries := buildPQDropCorpus(n, dim, nClusters, 71)

	// Incremental auto-trained.
	inc, err := newHNSW(hnswPQAutoTrainConfig(dim, threshold, m, false))
	if err != nil {
		t.Fatal(err)
	}
	insertAllHNSW(t, inc, ids, vecs)
	if inc.pqUntrained() {
		t.Fatal("incremental index must be trained")
	}
	incRecall := recallOf(t, inc, vecs, queries, k)

	// BuildConcurrent-trained on the same data.
	bulkCfg := hnswPQAutoTrainConfig(dim, threshold, m, false)
	bulk, err := newHNSW(bulkCfg)
	if err != nil {
		t.Fatal(err)
	}
	if err := bulk.BuildConcurrent(ids, vecs, 4); err != nil {
		t.Fatal(err)
	}
	bulkRecall := recallOf(t, bulk, vecs, queries, k)

	t.Logf("recall@%d incremental-auto-train=%.3f BuildConcurrent=%.3f", k, incRecall, bulkRecall)
	if incRecall < 0.40 {
		t.Fatalf("incremental auto-train recall@%d=%.3f below floor 0.40", k, incRecall)
	}
	if math.Abs(incRecall-bulkRecall) > 0.15 {
		t.Fatalf("incremental (%.3f) and BuildConcurrent (%.3f) recall differ by > 0.15 — not equivalent trained behaviour",
			incRecall, bulkRecall)
	}
}

// TestPQHNSWAutoTrainFloatDrop proves the float-drop fold: with PQDropVecs + a low
// threshold, the incremental insert that trips the auto-train ALSO drops the
// resident floats (vecsDropped()), ADC-only search still works, and a post-drop
// Insert is rejected with ErrPQDropVecsReadOnly.
func TestPQHNSWAutoTrainFloatDrop(t *testing.T) {
	const (
		dim       = 64
		threshold = 800
		n         = 800 // == threshold: the LAST insert trips train+drop; the index
		// is read-only afterwards, so we insert exactly threshold points (the float
		// drop folds in on the crossing insert) and then prove post-drop rejection.
		nClusters = 30
		m         = 16
		k         = 10
	)
	ids, vecs, queries := buildPQDropCorpus(n, dim, nClusters, 31)
	h, err := newHNSW(hnswPQAutoTrainConfig(dim, threshold, m, true)) // PQDropVecs=true
	if err != nil {
		t.Fatal(err)
	}
	// The insert that crosses the threshold trains + drops the floats.
	insertAllHNSW(t, h, ids, vecs)
	if h.pqUntrained() {
		t.Fatal("index must be trained after incremental auto-train")
	}
	if !h.vecsDropped() {
		t.Fatal("PQDropVecs + incremental auto-train must drop the resident floats")
	}
	if h.arena.vecs != nil {
		t.Fatal("arena.vecs must be nil after the auto-train float drop")
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
		t.Fatal("all PQ codes zero after the auto-train drop (codes lost)")
	}
	// ADC-only search still works with sane recall.
	rec := recallOf(t, h, vecs, queries, k)
	t.Logf("post-drop ADC-only recall@%d=%.3f", k, rec)
	if rec < 0.40 {
		t.Fatalf("post-drop ADC recall@%d=%.3f below floor 0.40", k, rec)
	}
	// Post-drop Insert is rejected (read-mostly).
	if _, _, err := h.Insert(uint64(n+1), vecs[0], 0, nil, nil, nil, CASCond{}); err != ErrPQDropVecsReadOnly {
		t.Fatalf("post-drop Insert should return ErrPQDropVecsReadOnly, got %v", err)
	}
}

// TestPQHNSWAutoTrainFloatDropDeterminism proves the PQDropVecs path is ALSO
// replica-deterministic: two indexes applying the identical inserts both drop and
// hold byte-identical codebooks + codes (the codes are dumped before any further
// mutation; the float drop does not touch the codes).
func TestPQHNSWAutoTrainFloatDropDeterminism(t *testing.T) {
	const (
		dim       = 32
		threshold = 500
		n         = 500 // == threshold: the last insert trips train+drop (read-only after)
		m         = 8
	)
	ids, vecs := siftLikeCorpus(n, dim, 88)
	cfg := hnswPQAutoTrainConfig(dim, threshold, m, true)

	a, err := newHNSW(cfg)
	if err != nil {
		t.Fatal(err)
	}
	b, err := newHNSW(cfg)
	if err != nil {
		t.Fatal(err)
	}
	insertAllHNSW(t, a, ids, vecs)
	insertAllHNSW(t, b, ids, vecs)
	if !a.vecsDropped() || !b.vecsDropped() {
		t.Fatal("both PQDropVecs indexes must have dropped after auto-train")
	}
	if !bytes.Equal(dumpCodebooks(t, a), dumpCodebooks(t, b)) {
		t.Fatal("PQDropVecs codebooks diverged across replicas")
	}
	if !bytes.Equal(dumpCodes(t, a, n), dumpCodes(t, b, n)) {
		t.Fatal("PQDropVecs per-slot codes diverged across replicas")
	}
}

// TestPQHNSWNonPQInsertUnaffected is the regression guard: a NON-PQ HNSW index
// with IVFTrainThreshold set never auto-trains (shouldAutoTrainPQ is false for a
// non-pqQuantizer), so the incremental Insert path is byte/behaviour-identical.
func TestPQHNSWNonPQInsertUnaffected(t *testing.T) {
	const (
		dim       = 32
		threshold = 100
		n         = 500
		seed      = 3
		k         = 10
	)
	ids, vecs := siftLikeCorpus(n, dim, seed)
	_, queries := siftLikeCorpus(50, dim, 99)

	base := Config{Dim: dim, Metric: L2, M: 16, EfConstruction: 200, EfSearch: 64, Seed: seed}

	// Reference: no threshold set.
	ref, err := newHNSW(base)
	if err != nil {
		t.Fatal(err)
	}
	insertAllHNSW(t, ref, ids, vecs)

	// Same but with a low IVFTrainThreshold — must be byte-identical (no PQ, so the
	// auto-train trigger never fires).
	withThr := base
	withThr.IVFTrainThreshold = threshold
	got, err := newHNSW(withThr)
	if err != nil {
		t.Fatal(err)
	}
	insertAllHNSW(t, got, ids, vecs)

	for qi, q := range queries {
		rRef, err := ref.Search(q, k)
		if err != nil {
			t.Fatal(err)
		}
		rGot, err := got.Search(q, k)
		if err != nil {
			t.Fatal(err)
		}
		if len(rRef) != len(rGot) {
			t.Fatalf("query %d: len %d != %d", qi, len(rGot), len(rRef))
		}
		for i := range rRef {
			if rRef[i].ID != rGot[i].ID || rRef[i].Distance != rGot[i].Distance {
				t.Fatalf("query %d rank %d: non-PQ insert path changed under IVFTrainThreshold", qi, i)
			}
		}
	}
}

// TestPQHNSWAutoTrainConcurrentSearch runs concurrent searches across the
// train-crossing insert sequence under -race: the auto-train runs under the write
// lock (h.mu), so a concurrent reader either sees the fully-untrained (exact) or
// fully-trained (ADC) state — never a torn quantizer. Must be -race clean.
func TestPQHNSWAutoTrainConcurrentSearch(t *testing.T) {
	const (
		dim       = 32
		threshold = 400
		n         = 800
		m         = 8
	)
	ids, vecs := siftLikeCorpus(n, dim, 64)
	h, err := newHNSW(hnswPQAutoTrainConfig(dim, threshold, m, false))
	if err != nil {
		t.Fatal(err)
	}
	// Seed a few points so searches have something to traverse from the start.
	insertAllHNSW(t, h, ids[:10], vecs[:10])

	var wg sync.WaitGroup
	stop := make(chan struct{})
	// Concurrent readers spanning the train-crossing.
	for r := 0; r < 4; r++ {
		wg.Add(1)
		go func(seed int) {
			defer wg.Done()
			rng := rand.New(rand.NewSource(int64(seed)))
			for {
				select {
				case <-stop:
					return
				default:
				}
				q := vecs[rng.Intn(n)]
				if _, err := h.Search(q, 10); err != nil {
					t.Errorf("concurrent Search: %v", err)
					return
				}
			}
		}(r)
	}
	// Writer crosses the threshold while readers run.
	insertAllHNSW(t, h, ids[10:], vecs[10:])
	close(stop)
	wg.Wait()
	if h.pqUntrained() {
		t.Fatal("index must be trained after crossing the threshold")
	}
}
