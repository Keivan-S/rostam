// SPDX-License-Identifier: Apache-2.0

package vector

import (
	"sync"
	"sync/atomic"
	"testing"
	"time"
)

// THE RESHARD BOUNDARY SHAPE.
//
// A faithful model of the ONE hnsw that TestClusterOnlineReshardRemote's final
// search hits: the new generation's partition holding id 40. Everything about it
// that matters is different from a generic churn test, and each difference is
// taken from the integration test rather than invented:
//
//   - the corpus is a LINE. vecOf(id, tag) is {id, tag, 0, 0}, so distance is
//     (Δid)² + (Δtag)² and a point's true neighbours are its id-neighbours. That
//     makes neighbourhoods one-dimensional and narrow, which is what makes an
//     in-edge loss unrecoverable — there is no second direction to reach a point
//     from.
//   - it is TINY. Partitioning splits ~350 ids eight ways, so the index under
//     test holds roughly forty points at M=8.
//   - the deleted band is CONTIGUOUS AND PERMANENT, and it sits at one END of the
//     line: ids [0,40) go and never come back, leaving id 40 as the new extreme
//     point of the line. An endpoint has every neighbour on one side.
//   - the copy is if-absent and races the deletes of THE SAME ids, and Reclaim
//     runs on the new generation, physically removing the deleted slots and with
//     them every in-edge they held.
//
// The assertion is the integration test's, reduced: after quiesce, the surviving
// endpoint must still be findable by search, and no live point may be
// unreachable from the entry point.
//
// WHAT THIS TEST DOES NOT CATCH, AND WHY IT IS STILL HERE. It was built to chase
// the hypothesis that the boundary point loses every in-edge when the band below
// it is tombstoned and then Reclaimed — the slots vanish and take their edges
// with them. That hypothesis is WRONG, and this test passing forty attempts in a
// row is part of how that was established. Instrumenting the real cluster index
// instead showed the surviving orphans carry in-degree 6 to 16 and out-degree 8
// to 14 on a forty-node graph, all with unlinked=false: they are healthily
// connected TO EACH OTHER and severed from the entry point's component as a
// group. The residual is a graph PARTITION, not edge starvation, so any fix
// aimed at in-degree repair — reclaim-time reverse-edge repair, a minimum
// in-degree top-up at link time — would not have touched it.
//
// It stays because the invariant it asserts is real and cheap, and because the
// next person to work on this should not have to rediscover that this shape is
// not the one that breaks.

// TestReshardBoundaryEndpointStaysFindable builds the new generation the way the
// reshard does — from empty, by unserialized if-absent copies racing opMu-held
// upserts and deletes, with Reclaim running — over the line corpus, and then
// asks for the point at the boundary of the deleted band.
func TestReshardBoundaryEndpointStaysFindable(t *testing.T) {
	const (
		base     = 200
		addFrom  = 1000
		addTo    = 1150
		delFrom  = 0
		delTo    = 40
		upsFrom  = 40
		upsTo    = 120
		parts    = 8
		partIdx  = 0
		attempts = 40
	)
	vecOf := func(id uint64, tag float32) []float32 {
		return []float32{float32(id), tag, 0, 0}
	}
	// One partition's worth of ids, mirroring BOTH the size and the MEMBERSHIP of
	// the index that serves the failing search. The membership is load-bearing:
	// routing hashes the id (ops.PartitionOf, splitmix64), so a partition holds a
	// scattered subset of the line and the deleted band removes a RANDOMLY
	// INTERLEAVED fifth of it. A stride-based partition would leave every survivor
	// a surviving id-neighbour and quietly hide the whole failure mode.
	partitionOf := func(id uint64, p uint64) uint64 {
		z := id + 0x9e3779b97f4a7c15
		z = (z ^ (z >> 30)) * 0xbf58476d1ce4e5b9
		z = (z ^ (z >> 27)) * 0x94d049bb133111eb
		z = z ^ (z >> 31)
		return z % p
	}
	var ids []uint64
	for id := uint64(0); id < base; id++ {
		if partitionOf(id, parts) == partIdx {
			ids = append(ids, id)
		}
	}
	for id := uint64(addFrom); id < addTo; id++ {
		if partitionOf(id, parts) == partIdx {
			ids = append(ids, id)
		}
	}

	failures := 0
	var firstDetail string
	for attempt := 0; attempt < attempts; attempt++ {
		cfg := Config{Dim: 4, Metric: L2, M: 8, EfConstruction: 50, EfSearch: 64, Seed: 1}
		h, err := newHNSW(cfg)
		if err != nil {
			t.Fatal(err)
		}

		var opMu sync.Mutex // Collection.opMu: upserts and deletes hold it, copies do not
		var wg sync.WaitGroup
		spawn := func(fn func()) {
			wg.Add(1)
			go func() { defer wg.Done(); fn() }()
		}

		// The reshard copy: if-absent, unserialized, over EVERY id including the
		// ones the deleter is removing right now.
		for w := 0; w < 3; w++ {
			spawn(func() {
				for i := w; i < len(ids); i += 3 {
					id := ids[i]
					tag := float32(1)
					if id >= upsFrom && id < upsTo {
						tag = 2
					}
					if _, err := h.InsertIfAbsent(id, vecOf(id, tag), 0, nil, nil); err != nil {
						t.Errorf("copy %d: %v", id, err)
						return
					}
				}
			})
		}
		// Dual-write: overwrite the [upsFrom,upsTo) band to tag 2.
		spawn(func() {
			for _, id := range ids {
				if id < upsFrom || id >= upsTo {
					continue
				}
				opMu.Lock()
				h.Delete(id, CASCond{})
				_, _, err := h.Insert(id, vecOf(id, 2), 0, nil, nil, nil, CASCond{})
				opMu.Unlock()
				if err != nil && err != ErrDuplicateID {
					t.Errorf("upsert %d: %v", id, err)
					return
				}
			}
		})
		// Dual-write: delete the low band PERMANENTLY, racing the copy of the same
		// ids, and reclaim so the slots (and every in-edge they hold) are gone.
		spawn(func() {
			for pass := 0; pass < 3; pass++ {
				for _, id := range ids {
					if id < delFrom || id >= delTo {
						continue
					}
					opMu.Lock()
					h.Delete(id, CASCond{})
					opMu.Unlock()
				}
				h.Reclaim()
				time.Sleep(time.Millisecond)
			}
		})
		var searches atomic.Int64
		spawn(func() {
			var dst []Result
			for i := 0; i < 4000; i++ {
				dst, _ = h.SearchInto(dst[:0], vecOf(uint64(i%base), 1), 5, Filter{})
				searches.Add(1)
			}
		})
		wg.Wait()

		// Settle: the copy may have resurrected a deleted id after the last delete
		// pass, so re-delete and reclaim once more, exactly as the reshard's final
		// convergence does.
		for _, id := range ids {
			if id >= delFrom && id < delTo {
				h.Delete(id, CASCond{})
			}
		}
		h.Reclaim()

		// ---- the integration test's question ----
		// The surviving endpoint: the smallest live id at or above delTo.
		var endpoint = ^uint64(0)
		for _, id := range ids {
			if id >= delTo && id < endpoint {
				if _, _, _, _, _, ok := h.Get(id); ok {
					endpoint = id
				}
			}
		}
		if endpoint == ^uint64(0) {
			t.Fatal("no surviving endpoint — the model deleted everything")
		}
		res, err := h.Search(vecOf(endpoint, 1), 1)
		if err != nil {
			t.Fatal(err)
		}
		bad := unreachableLivePoints(h)
		if (len(res) == 0 || res[0].ID != endpoint) || len(bad) != 0 {
			failures++
			if firstDetail == "" {
				slot, _ := h.arena.Slot(endpoint)
				in, out := 0, 0
				for _, nd := range h.nodes {
					if nd == nil {
						continue
					}
					for lc := 0; lc <= nd.level; lc++ {
						for _, nb := range h.nbrsAt(nd, lc) {
							if nb == slot {
								in++
							}
						}
					}
				}
				if nd := h.nodeAt(slot); nd != nil {
					for lc := 0; lc <= nd.level; lc++ {
						out += h.nbrLen(nd, lc)
					}
				}
				firstDetail = detail(endpoint, slot, in, out, res, bad)
			}
		}
	}
	if failures != 0 {
		t.Fatalf("%d of %d attempts lost the boundary endpoint or orphaned a live point\n  first: %s",
			failures, attempts, firstDetail)
	}
	t.Logf("%d attempts, boundary endpoint findable and no live point unreachable in every one", attempts)
}

func detail(endpoint uint64, slot uint32, in, out int, res []Result, bad []uint32) string {
	s := "endpoint id=" + itoaU(endpoint) + " slot=" + itoaU(uint64(slot)) +
		" inDeg=" + itoaU(uint64(in)) + " outDeg=" + itoaU(uint64(out)) +
		" unreachableLive=" + itoaU(uint64(len(bad)))
	if len(res) > 0 {
		s += " gotRank0=" + itoaU(res[0].ID)
	} else {
		s += " gotRank0=<empty>"
	}
	return s
}

func itoaU(n uint64) string {
	if n == 0 {
		return "0"
	}
	var b []byte
	for n > 0 {
		b = append([]byte{byte('0' + n%10)}, b...)
		n /= 10
	}
	return string(b)
}
