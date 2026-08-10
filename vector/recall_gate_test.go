// SPDX-License-Identifier: Apache-2.0

package vector

import "testing"

// TestRecallGate is an automated correctness GATE for the dense ANN search path,
// independent of our hand-written expectation tests. It scores the APPROXIMATE
// HNSW index against an EXACT brute-force oracle (bruteTopK) on a deterministic
// seeded dataset and fails if mean recall@10 drops below a floor.
//
// Why this is worth having: most of our tests assert behavior we ourselves
// specified, so a wrong mental model can hide in both code and test. Here the
// oracle is a *different, exact* algorithm (full linear scan), so a real
// regression in graph build/search (bad neighbor selection, broken distance,
// wrong entry point, corrupted arena) tanks recall far below the floor and trips
// this gate — with no external dataset, fast enough for CI.
//
// For the full external-ground-truth check on real data, see TestSIFT1MBench
// (SIFT-1M with the dataset authors' exact neighbors; opt-in via ROSTAM_SIFT1M=1).
func TestRecallGate(t *testing.T) {
	const (
		n    = 8000
		dim  = 64
		k    = 10
		seed = 42
		// HNSW on this corpus reaches ~0.99 recall@10 at efSearch=128; the floor is
		// set well below the measured value so it never flakes but still trips hard
		// on a real index regression (a broken graph drops to <0.5).
		floor = 0.95
	)
	ids, vecs := siftLikeCorpus(n, dim, seed)
	_, queries := siftLikeCorpus(300, dim, 7)

	cfg := Config{Dim: dim, Metric: L2, M: 16, EfConstruction: 200, EfSearch: 128, Seed: seed}
	h, err := newHNSW(cfg)
	if err != nil {
		t.Fatal(err)
	}
	for i, v := range vecs {
		if _, _, err := h.Insert(ids[i], v, 0, nil, nil, nil, CASCond{}); err != nil {
			t.Fatalf("insert %d: %v", i, err)
		}
	}

	r := recallOf(t, h, vecs, queries, k)
	t.Logf("HNSW recall@%d vs exact brute-force = %.4f (floor %.2f)  [n=%d dim=%d M=16 efC=200 efS=128]",
		k, r, floor, n, dim)
	if r < floor {
		t.Errorf("HNSW recall@%d = %.4f fell below floor %.2f — likely an ANN index regression "+
			"(graph build/search/distance/entry-point). Investigate before merging.", k, r, floor)
	}
}

// TestRecallGateIVF is the IVF-Flat counterpart to TestRecallGate: it scores the
// IVF index — which prunes to nprobe lists then exact-rescores candidates on
// float32 — against the same exact brute-force oracle. A regression in IVF
// training (kmeans), list assignment, probe selection, or the rescore step tanks
// recall below the floor. Gaussian (unclustered) data is the hard case for IVF,
// so the floor is set conservatively below the measured value.
func TestRecallGateIVF(t *testing.T) {
	const (
		n      = 8000
		dim    = 64
		k      = 10
		seed   = 42
		nprobe = 32
		// Measured ~0.909 on this gaussian corpus (deterministic). Floor is set well
		// below that for headroom against benign kmeans/seed drift, while still
		// tripping hard on a real IVF regression (broken assignment/probe → <0.5).
		floor = 0.85
	)
	ids, vecs := siftLikeCorpus(n, dim, seed)
	_, queries := siftLikeCorpus(300, dim, 7)

	cfg := DefaultConfig()
	cfg.Dim = dim
	cfg.Metric = L2
	cfg.Seed = seed
	cfg.IndexType = IndexIVF
	cfg.IVFNprobe = nprobe
	cfg.IVFTrainThreshold = 1000 // train into clusters well below n so the IVF path engages
	ix, err := newIVF(cfg)
	if err != nil {
		t.Fatal(err)
	}
	insertAll(t, ix, ids, vecs)

	var matches int
	for _, q := range queries {
		truth := bruteTopK(vecs, q, k)
		res, err := ix.Search(q, k)
		if err != nil {
			t.Fatal(err)
		}
		for _, r := range res {
			if truth[r.ID] {
				matches++
			}
		}
	}
	r := float64(matches) / float64(len(queries)*k)
	t.Logf("IVF-Flat recall@%d vs exact brute-force = %.4f (floor %.2f)  [n=%d dim=%d nprobe=%d]",
		k, r, floor, n, dim, nprobe)
	if r < floor {
		t.Errorf("IVF-Flat recall@%d = %.4f fell below floor %.2f — likely an IVF regression "+
			"(kmeans train / list assignment / probe selection / rescore).", k, r, floor)
	}
}
