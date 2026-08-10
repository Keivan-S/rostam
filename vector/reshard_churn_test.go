// SPDX-License-Identifier: Apache-2.0

package vector

import (
	"math/rand"
	"sync"
	"sync/atomic"
	"testing"
	"time"
)

// RESHARD-SHAPED CHURN.
//
// A package-level model of what TestClusterOnlineReshardRemote does to a single
// hnsw, built to iterate in seconds instead of the integration test's ~9.
//
// The shape that matters, and the locking that goes with it:
//
//   - a churn worker UPSERTS a contiguous band in a loop. An upsert is
//     delete-then-insert, and Collection.UpsertCASKeyTTL{,At} holds opMu across
//     both halves unconditionally — even with the WAL off, which is the cluster
//     configuration. So upserts serialize against each other and against deletes.
//   - a copy worker inserts the same ids IF-ABSENT, modelling the reshard
//     backfill. Its replicated-apply entry point is InsertCASKeyTTLAt, which
//     returns BEFORE taking opMu when c.wal == nil. On a replicated collection
//     the WAL is off, so this path holds NO collection-level lock at all and
//     runs concurrently with the churn worker.
//   - a delete worker removes a band ADJACENT to the survivors. That adjacency is
//     load-bearing: a survivor's in-edges come mostly from its nearest
//     neighbours, so deleting the band next to it is what strips its in-degree.
//   - searches run throughout.
//
// The assertion is the one the integration test makes at the end, reduced to its
// mechanism: after everything quiesces, every live point must still be reachable
// from the entry point, and must still be findable by search.

// TestReshardShapedChurnKeepsGraphWhole is the local stand-in for the cluster
// reshard regression. It is deliberately harsher than production in one respect
// only — the copy worker is unserialized, which is exactly what the WAL-off
// apply path does.
func TestReshardShapedChurnKeepsGraphWhole(t *testing.T) {
	const (
		n        = 900
		dim      = 8
		delFrom  = 200
		delTo    = 240 // survivors start here; the band below is deleted
		bandFrom = 240
		bandTo   = 320
	)
	cfg := Config{Dim: dim, Metric: L2, M: 8, EfConstruction: 100, EfSearch: 64, Seed: 5}
	h, err := newHNSW(cfg)
	if err != nil {
		t.Fatal(err)
	}
	vecOf := func(id uint64, tag int) []float32 {
		v := make([]float32, dim)
		rng := rand.New(rand.NewSource(int64(id)*7 + int64(tag)))
		for d := range v {
			v[d] = rng.Float32()
		}
		return v
	}
	// The index under test is the reshard's NEW generation: it starts EMPTY and is
	// populated concurrently by the copy backfill and the dual-writes, rather than
	// seeded and then churned.
	seedFrom := 0

	// opMu models Collection.opMu, which the upsert and delete paths hold
	// unconditionally. The copy worker deliberately does NOT take it.
	var opMu sync.Mutex
	var stop atomic.Bool
	var wg sync.WaitGroup
	spawn := func(fn func()) {
		wg.Add(1)
		go func() { defer wg.Done(); fn() }()
	}

	// Backfill: the bulk of the corpus, if-absent, unserialized — the copy pass.
	// Several workers, because the reshard copy is itself parallel and the
	// unserialized apply path lets their link phases overlap.
	for w := 0; w < 4; w++ {
		spawn(func() {
			for i := seedFrom + w; i < n && !stop.Load(); i += 4 {
				if _, err := h.InsertIfAbsent(uint64(i), vecOf(uint64(i), 1), 0, nil, nil); err != nil {
					t.Errorf("backfill %d: %v", i, err)
					return
				}
			}
		})
	}
	// Churn: upsert the band in a loop (delete + insert under opMu).
	spawn(func() {
		for !stop.Load() {
			for id := uint64(bandFrom); id < bandTo && !stop.Load(); id++ {
				opMu.Lock()
				h.Delete(id, CASCond{})
				_, _, err := h.Insert(id, vecOf(id, 2), 0, nil, nil, nil, CASCond{})
				opMu.Unlock()
				if err != nil && err != ErrDuplicateID {
					t.Errorf("churn upsert %d: %v", id, err)
					return
				}
			}
		}
	})
	// Reshard copy: if-absent over the same band, NOT under opMu.
	spawn(func() {
		for !stop.Load() {
			for id := uint64(bandFrom); id < bandTo && !stop.Load(); id++ {
				if _, err := h.InsertIfAbsent(id, vecOf(id, 3), 0, nil, nil); err != nil {
					t.Errorf("copy if-absent %d: %v", id, err)
					return
				}
			}
		}
	})
	// Delete the band adjacent to the survivors, then reclaim it, repeatedly.
	spawn(func() {
		for !stop.Load() {
			for id := uint64(delFrom); id < delTo && !stop.Load(); id++ {
				opMu.Lock()
				h.Delete(id, CASCond{})
				opMu.Unlock()
			}
			h.Reclaim()
			for id := uint64(delFrom); id < delTo && !stop.Load(); id++ {
				opMu.Lock()
				_, _, err := h.Insert(id, vecOf(id, 4), 0, nil, nil, nil, CASCond{})
				opMu.Unlock()
				if err != nil && err != ErrDuplicateID {
					t.Errorf("re-insert %d: %v", id, err)
					return
				}
			}
		}
	})
	for r := 0; r < 2; r++ {
		spawn(func() {
			rng := rand.New(rand.NewSource(int64(r) + 77))
			q := make([]float32, dim)
			var dst []Result
			for !stop.Load() {
				for d := range q {
					q[d] = rng.Float32()
				}
				var serr error
				if dst, serr = h.SearchInto(dst[:0], q, 10, Filter{}); serr != nil {
					t.Errorf("search: %v", serr)
					return
				}
			}
		})
	}

	time.Sleep(2500 * time.Millisecond)
	stop.Store(true)
	wg.Wait()

	// ---- quiesced: the graph must be whole ----
	if bad := unreachableLivePoints(h); len(bad) != 0 {
		// DISCRIMINATING PROBE: for each unreachable point, is it still flagged
		// unlinked (its link phase never ran), does it have out-edges (it linked
		// but nobody points back), or does it have in-edges from nodes that are
		// themselves unreachable (a whole detached component)?
		inDeg := make([]int, len(h.nodes))
		for _, nd := range h.nodes {
			if nd == nil {
				continue
			}
			for lc := 0; lc <= nd.level; lc++ {
				for _, nb := range h.nbrsAt(nd, lc) {
					if int(nb) < len(inDeg) {
						inDeg[nb]++
					}
				}
			}
		}
		neverLinked, noIn, hasIn := 0, 0, 0
		for _, slot := range bad {
			nd := h.nodes[slot]
			switch {
			case nd.unlinked.Load():
				neverLinked++
			case inDeg[slot] == 0:
				noIn++
			default:
				hasIn++
			}
		}
		outDeg := make([]int, 0, len(bad))
		inDegs := make([]int, 0, len(bad))
		lvls := make([]int, 0, len(bad))
		for _, slot := range bad[:min(len(bad), 10)] {
			nd := h.nodes[slot]
			o := 0
			for lc := 0; lc <= nd.level; lc++ {
				o += h.nbrLen(nd, lc)
			}
			outDeg = append(outDeg, o)
			inDegs = append(inDegs, inDeg[slot])
			lvls = append(lvls, nd.level)
		}
		t.Fatalf("%d live point(s) unreachable after quiesce: %d never linked, %d linked-but-zero-in-degree, %d in a detached component\n  slots=%v\n  level=%v\n  outDeg=%v\n  inDeg=%v\n  entryPoint=%d maxLevel=%d epLevel=%d",
			len(bad), neverLinked, noIn, hasIn, bad[:min(len(bad), 10)], lvls, outDeg, inDegs,
			h.entryPoint, h.maxLevel, h.nodes[h.entryPoint].level)
	}

	// And every live point must be its own nearest neighbour, which is what the
	// integration test actually asserts.
	misses := 0
	var firstMiss uint64
	h.mu.RLock()
	live := make([]uint64, 0, n)
	for id, slot := range h.arena.idMap {
		if !h.tombstoned[slot] {
			live = append(live, id)
		}
	}
	h.mu.RUnlock()
	for _, id := range live {
		vec, _, _, _, _, ok := h.Get(id)
		if !ok {
			continue
		}
		res, err := h.Search(vec, 1)
		if err != nil {
			t.Fatal(err)
		}
		if len(res) == 0 || res[0].ID != id {
			if misses == 0 {
				firstMiss = id
			}
			misses++
		}
	}
	// A small number of self-misses is ordinary ANN behaviour at these parameters
	// (M=8, ef=64 over random 8-dim points) and reproduces identically on the
	// pre-Option-B base, so it is a tolerance rather than an assertion. The
	// REACHABILITY check above is the one that discriminates: base leaves zero
	// unreachable points, the regression left one to nearly forty.
	if maxMiss := len(live) / 200; misses > maxMiss {
		t.Fatalf("%d of %d live points are no longer their own nearest neighbour (first: id %d, tolerance %d) — search recall collapsed",
			misses, len(live), firstMiss, maxMiss)
	}
	t.Logf("%d live points, all reachable and self-findable after reshard-shaped churn", len(live))
}
