// SPDX-License-Identifier: Apache-2.0

package vector

import (
	"math/rand"
	"strconv"
	"testing"
)

// AUTO-TRAIN AT A TINY THRESHOLD.
//
// The insert that trips a quantizer's auto-train threshold links and trains
// inside the placement critical section rather than deferring its link phase.
// That branch returns before the ordinary path's setup runs, so anything the
// link phase needs must already exist when it is taken.
//
// The link stripes are exactly such a thing, and they are allocated lazily. On
// the ordinary path allocation happens on every insert; on the training branch
// it did not happen at all. Whether that mattered depended on arithmetic nobody
// was looking at: the branch fires at insert max(2, threshold), the very first
// insert returns early (it anchors the index and links nothing), so the stripes
// are only guaranteed to exist if at least one insert took the ordinary path
// first. With IVFTrainThreshold at 1 or 2 none has, and linkNode — which locks a
// stripe unconditionally, for its forward write and for every back-edge —
// indexed an empty slice.
//
// Not merely a panic. It fires holding h.mu's write lock, on a path whose unlock
// is not deferred, so a recover() anywhere upstream leaves the collection
// permanently wedged with the write lock held: every subsequent insert, delete,
// snapshot and search on it blocks forever.
//
// The threshold is an ordinary replicated Config field — JSON-serialized, and
// Validate rejects only negatives — so 1 and 2 are reachable configurations, not
// abuse. No existing test used a threshold below 600, which is why a suite that
// covers auto-training heavily was green throughout.

// autoTrainAtThreshold inserts n vectors into a fresh index built from cfg,
// which must name a quantizer with a tiny IVFTrainThreshold, and returns the
// index. A panic here is the regression: it means the training branch reached
// the link phase without the state that phase requires.
func autoTrainAtThreshold(t *testing.T, cfg Config, n int) *hnsw {
	t.Helper()
	h, err := newHNSW(cfg)
	if err != nil {
		t.Fatalf("newHNSW(threshold=%d): %v", cfg.IVFTrainThreshold, err)
	}
	rng := rand.New(rand.NewSource(int64(cfg.IVFTrainThreshold)*31 + int64(cfg.Dim)))
	v := make([]float32, cfg.Dim)
	for i := 0; i < n; i++ {
		for d := range v {
			v[d] = rng.Float32()
		}
		if _, _, err := h.Insert(uint64(i), v, 0, nil, nil, nil, CASCond{}); err != nil {
			t.Fatalf("insert %d at threshold %d: %v", i, cfg.IVFTrainThreshold, err)
		}
	}
	return h
}

// TestAutoTrainTinyThreshold covers every quantizer whose auto-train runs on the
// incremental insert path, at the two thresholds that make the training branch
// the FIRST insert to need the link stripes.
//
// It also searches afterwards: a stripe array that exists but is wrong would
// pass the insert loop and fail here.
func TestAutoTrainTinyThreshold(t *testing.T) {
	const dim = 8
	base := Config{Dim: dim, Metric: L2, M: 8, EfConstruction: 64, EfSearch: 32, Seed: 42}

	quantizers := []struct {
		name  string
		apply func(c *Config)
	}{
		{"PQ", func(c *Config) { c.Quant = QuantPQ; c.QuantPQM = 4 }},
		{"SQ", func(c *Config) { c.Quant = QuantSQ; c.SQBits = 8 }},
		{"PRQ", func(c *Config) { c.Quant = QuantPRQ; c.QuantPQM = 4; c.PRQLayers = 2 }},
	}
	for _, q := range quantizers {
		for _, threshold := range []int{1, 2, 3} {
			t.Run(q.name+"/threshold"+strconv.Itoa(threshold), func(t *testing.T) {
				cfg := base
				q.apply(&cfg)
				cfg.IVFTrainThreshold = threshold
				h := autoTrainAtThreshold(t, cfg, 40)

				// The index must be usable, not merely un-panicked.
				res, err := h.Search([]float32{0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5}, 5)
				if err != nil {
					t.Fatalf("search: %v", err)
				}
				if len(res) == 0 {
					t.Fatal("search returned nothing from a 40-point index")
				}
				// And the graph must be whole: the training insert's own link phase
				// runs on the branch under test, so a silent failure there shows up
				// as an unreachable node.
				if bad := unreachableLivePoints(h); len(bad) != 0 {
					t.Fatalf("%d point(s) unreachable after auto-train at threshold %d: %v",
						len(bad), threshold, bad[:min(len(bad), 8)])
				}
			})
		}
	}
}
