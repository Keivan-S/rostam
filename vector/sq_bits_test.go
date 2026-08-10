// SPDX-License-Identifier: Apache-2.0

package vector

import (
	"bytes"
	"math/rand"
	"runtime"
	"sort"
	"testing"
)

// TestTrainedSQCodeLen verifies CodeLen() = ceil(dim*bits/8) for every supported
// bit-depth, including dims where dim*bits is NOT a multiple of 8 (a partial final
// byte).
func TestTrainedSQCodeLen(t *testing.T) {
	cases := []struct {
		dim, bits, want int
	}{
		{16, 8, 16}, // byte-aligned 8-bit
		{16, 4, 8},  // 2 dims/byte
		{16, 6, 12}, // 96 bits exactly
		{10, 6, 8},  // 60 bits -> 8 bytes (partial last byte)
		{1, 4, 1},   // 4 bits -> 1 byte
		{1, 6, 1},   // 6 bits -> 1 byte
		{3, 4, 2},   // 12 bits -> 2 bytes (partial)
		{7, 6, 6},   // 42 bits -> 6 bytes (partial)
		{64, 4, 32}, // 4-bit, 8x compression
		{64, 6, 48}, // 6-bit
		{64, 8, 64}, // 8-bit
	}
	for _, c := range cases {
		q := newTrainedSQ(c.dim, c.bits, L2)
		if got := q.CodeLen(); got != c.want {
			t.Errorf("CodeLen(dim=%d,bits=%d) = %d, want %d", c.dim, c.bits, got, c.want)
		}
	}
}

// TestTrainedSQPackRoundTrip is the BIT-EXACT packing proof: for SQ4/SQ6/SQ8, pack
// dim random levels into a code and unpack them back to the identical levels
// (pack -> unpack == identity). Includes dims where dim*bits is NOT a multiple of 8
// (e.g. dim=10 at 6-bit) to exercise the partial final byte, and asserts the code
// length matches CodeLen() exactly (no overrun into unused trailing bits).
func TestTrainedSQPackRoundTrip(t *testing.T) {
	rng := rand.New(rand.NewSource(7))
	for _, bits := range []int{4, 6, 8} {
		for _, dim := range []int{1, 3, 7, 8, 10, 16, 17, 64, 100} {
			q := newTrainedSQ(dim, bits, L2)
			lvlMax := uint32(int(1<<uint(bits)) - 1)
			for trial := 0; trial < 200; trial++ {
				levels := make([]uint32, dim)
				code := make([]byte, q.CodeLen())
				for i := range levels {
					lvl := uint32(rng.Intn(int(lvlMax) + 1))
					levels[i] = lvl
					if bits == 8 {
						code[i] = byte(lvl)
					} else {
						q.packLevel(code, i, lvl)
					}
				}
				for i := 0; i < dim; i++ {
					got := q.readLevel(code, i)
					if got != levels[i] {
						t.Fatalf("bits=%d dim=%d trial=%d: level[%d] round-trip got %d want %d",
							bits, dim, trial, i, got, levels[i])
					}
				}
			}
		}
	}
}

// TestTrainedSQEncodeUnpackIdentity verifies the full Encode -> readLevel ->
// re-quantize path is stable across all bit-depths: encoding a vector, reading the
// stored levels back, and re-deriving levels from the same ranges yields identical
// levels (the encoder is deterministic and the packer is lossless). Also covers a
// dim where dim*bits is not byte-aligned.
func TestTrainedSQEncodeUnpackIdentity(t *testing.T) {
	rng := rand.New(rand.NewSource(13))
	const dim = 10 // dim*6 = 60 (partial byte at 6-bit)
	sample := make([][]float32, 300)
	for i := range sample {
		v := make([]float32, dim)
		for j := range v {
			v[j] = float32(rng.NormFloat64()) * 2
		}
		sample[i] = v
	}
	for _, bits := range []int{4, 6, 8} {
		q := trainSQ(sample, dim, bits, L2)
		if !q.trained() {
			t.Fatalf("bits=%d: trainSQ produced untrained quantizer", bits)
		}
		lvlMax := q.levelMax()
		code := make([]byte, q.CodeLen())
		for _, v := range sample {
			q.Encode(code, v)
			for i := 0; i < dim; i++ {
				want := q.level(v[i], i, lvlMax)
				got := q.readLevel(code, i)
				if got != want {
					t.Fatalf("bits=%d dim %d: stored level %d != recomputed %d", bits, i, got, want)
				}
			}
		}
	}
}

// TestTrainedSQ8ByteIdentical asserts the 8-bit path is byte-identical to a direct
// one-byte-per-dim encode (the Task-1 wire format). This guards against the
// generalization accidentally routing 8-bit through the generic bit stream.
func TestTrainedSQ8ByteIdentical(t *testing.T) {
	rng := rand.New(rand.NewSource(21))
	const dim = 48
	sample := make([][]float32, 256)
	for i := range sample {
		v := make([]float32, dim)
		for j := range v {
			v[j] = float32(rng.NormFloat64())
		}
		sample[i] = v
	}
	q := trainSQ(sample, dim, 8, Cosine)
	if q.CodeLen() != dim {
		t.Fatalf("8-bit CodeLen = %d, want %d", q.CodeLen(), dim)
	}
	lvlMax := q.levelMax()
	code := make([]byte, q.CodeLen())
	for _, v := range sample {
		q.Encode(code, v)
		// Direct Task-1-style encode: one clamped byte per dimension.
		for i := 0; i < dim; i++ {
			want := byte(q.level(v[i], i, lvlMax))
			if code[i] != want {
				t.Fatalf("8-bit code[%d] = %d, want byte-identical %d", i, code[i], want)
			}
		}
	}
}

// sqRecallForBits is sqRecallForMetric specialized to a configurable SQBits: it
// builds an exact (QuantNone) and a QuantSQ(bits) index over the SAME clustered
// corpus and returns (exactRecall, sqRecall) measured against brute-force truth.
func sqRecallForBits(t *testing.T, metric Metric, bits int) (float64, float64) {
	t.Helper()
	const (
		n        = 3_000
		dim      = 64
		nq       = 50
		k        = 10
		seed     = 11
		clusters = 48
		noise    = 0.15
	)
	rng := rand.New(rand.NewSource(seed))
	centers := make([][]float32, clusters)
	for c := range centers {
		v := make([]float32, dim)
		for j := range v {
			v[j] = float32(rng.NormFloat64())
		}
		centers[c] = v
	}
	pointNear := func(c int) []float32 {
		v := make([]float32, dim)
		for j := range v {
			v[j] = centers[c][j] + float32(noise*rng.NormFloat64())
		}
		return v
	}
	corpus := make([][]float32, n)
	for i := range corpus {
		corpus[i] = pointNear(i % clusters)
	}
	queries := make([][]float32, nq)
	for i := range queries {
		queries[i] = pointNear(rng.Intn(clusters))
	}

	dist := pickDist(metric)
	prep := func(v []float32) []float32 {
		out := append([]float32(nil), v...)
		if metric == Cosine {
			normalize(out)
		}
		return out
	}
	groundTruth := func(q []float32) map[uint64]bool {
		qn := prep(q)
		type pair struct {
			id uint64
			d  float32
		}
		ds := make([]pair, n)
		for i, v := range corpus {
			ds[i] = pair{id: uint64(i + 1), d: dist(qn, prep(v))}
		}
		sort.Slice(ds, func(a, b int) bool { return ds[a].d < ds[b].d })
		out := make(map[uint64]bool, k)
		for i := 0; i < k; i++ {
			out[ds[i].id] = true
		}
		return out
	}
	build := func(mode QuantMode, sqbits int) *hnsw {
		h, err := newHNSW(Config{
			Dim: dim, M: 16, EfConstruction: 200, EfSearch: 64,
			Seed: seed, Metric: metric, Quant: mode, SQBits: sqbits, RescoreFactor: 3,
		})
		if err != nil {
			t.Fatal(err)
		}
		ids := make([]uint64, n)
		for i := range ids {
			ids[i] = uint64(i + 1)
		}
		if err := h.BuildConcurrent(ids, corpus, runtime.GOMAXPROCS(0)); err != nil {
			t.Fatal(err)
		}
		return h
	}
	recallOf := func(h *hnsw) float64 {
		var matches int
		for _, q := range queries {
			truth := groundTruth(q)
			results, err := h.Search(q, k)
			if err != nil {
				t.Fatalf("search: %v", err)
			}
			for _, r := range results {
				if truth[r.ID] {
					matches++
				}
			}
		}
		return float64(matches) / float64(nq*k)
	}
	return recallOf(build(QuantNone, 0)), recallOf(build(QuantSQ, bits))
}

// TestSQHNSWRecallByBits is the bit-depth recall headline: recall@10 is roughly
// monotonic in bit-depth (SQ8 >= SQ6 >= SQ4, with a small slack for traversal
// noise) and every depth clears its floor, for L2 and Cosine. Because HNSW
// rescores the candidate set on exact float32, even SQ4 holds a usable floor.
func TestSQHNSWRecallByBits(t *testing.T) {
	if testing.Short() {
		t.Skip("skipping recall test in -short mode")
	}
	const slack = 0.03
	for _, metric := range []struct {
		name string
		m    Metric
	}{
		{"L2", L2},
		{"Cosine", Cosine},
	} {
		t.Run(metric.name, func(t *testing.T) {
			base4, sq4 := sqRecallForBits(t, metric.m, 4)
			base6, sq6 := sqRecallForBits(t, metric.m, 6)
			base8, sq8 := sqRecallForBits(t, metric.m, 8)
			t.Logf("recall@10 %s exact(4/6/8)=%.3f/%.3f/%.3f sq4=%.3f sq6=%.3f sq8=%.3f",
				metric.name, base4, base6, base8, sq4, sq6, sq8)

			// Per-depth floors (SQ4 lower, SQ6/SQ8 near-exact thanks to rescore).
			if sq4 < 0.75 {
				t.Errorf("%s SQ4 recall@10 = %.3f, want >= 0.75", metric.name, sq4)
			}
			if sq6 < 0.88 {
				t.Errorf("%s SQ6 recall@10 = %.3f, want >= 0.88", metric.name, sq6)
			}
			if sq8 < 0.90 {
				t.Errorf("%s SQ8 recall@10 = %.3f, want >= 0.90", metric.name, sq8)
			}
			// Roughly monotonic in bits (allow slack for graph-traversal noise).
			if sq6 < sq4-slack {
				t.Errorf("%s SQ6 (%.3f) should be >= SQ4 (%.3f) - %.2f", metric.name, sq6, sq4, slack)
			}
			if sq8 < sq6-slack {
				t.Errorf("%s SQ8 (%.3f) should be >= SQ6 (%.3f) - %.2f", metric.name, sq8, sq6, slack)
			}
		})
	}
}

// TestSQHNSWSnapshotByBits builds a QuantSQ index at SQ4 and SQ6, snapshots it,
// restores into a fresh index, and asserts the restored quantizer is TRAINED, the
// re-encoded per-slot codes are BIT-IDENTICAL, and search results are identical
// post-restore — the bit-packed persistence-soundness proof at non-8 depths.
func TestSQHNSWSnapshotByBits(t *testing.T) {
	const (
		n    = 2_000
		dim  = 64
		k    = 10
		seed = 31
	)
	ids, vecs := siftLikeCorpus(n, dim, seed)
	_, queries := siftLikeCorpus(60, dim, 99)

	for _, bits := range []int{4, 6} {
		t.Run(map[int]string{4: "SQ4", 6: "SQ6"}[bits], func(t *testing.T) {
			cfg := Config{Dim: dim, Metric: L2, M: 16, EfConstruction: 200, EfSearch: 64, Seed: seed, Quant: QuantSQ, SQBits: bits}
			src, err := newHNSW(cfg)
			if err != nil {
				t.Fatal(err)
			}
			if err := src.BuildConcurrent(ids, vecs, 4); err != nil {
				t.Fatal(err)
			}
			if src.sqUntrained() {
				t.Fatal("source SQ index should be trained after BuildConcurrent")
			}
			// CodeLen must reflect the packed bit-depth.
			wantLen := (dim*bits + 7) / 8
			if got := src.quant.(*trainedSQ).CodeLen(); got != wantLen {
				t.Fatalf("CodeLen = %d, want %d for SQ%d", got, wantLen, bits)
			}

			before := make([][]uint64, len(queries))
			for i, q := range queries {
				res, serr := src.Search(q, k)
				if serr != nil {
					t.Fatal(serr)
				}
				before[i] = resultIDs(res)
			}
			srcCodes := make([][]byte, n)
			for slot := 0; slot < n; slot++ {
				srcCodes[slot] = append([]byte(nil), src.arena.Code(uint32(slot))...)
			}

			var buf bytes.Buffer
			if err := src.Snapshot(&buf); err != nil {
				t.Fatalf("Snapshot: %v", err)
			}
			dst, err := newHNSW(cfg)
			if err != nil {
				t.Fatal(err)
			}
			if err := dst.Restore(&buf); err != nil {
				t.Fatalf("Restore: %v", err)
			}
			if dst.sqUntrained() {
				t.Fatal("restored SQ index is UNTRAINED — ranges did not survive snapshot")
			}
			for slot := 0; slot < n; slot++ {
				if !bytes.Equal(srcCodes[slot], dst.arena.Code(uint32(slot))) {
					t.Fatalf("slot %d code not bit-identical after restore", slot)
				}
			}
			for i, q := range queries {
				res, serr := dst.Search(q, k)
				if serr != nil {
					t.Fatal(serr)
				}
				if !eqUint64(resultIDs(res), before[i]) {
					t.Fatalf("query %d: restored %v != original %v", i, resultIDs(res), before[i])
				}
			}
		})
	}
}
