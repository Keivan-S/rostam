// SPDX-License-Identifier: Apache-2.0

package vector

import (
	"bytes"
	"math/rand"
	"sync"
	"sync/atomic"
	"testing"
	"time"
)

// THE PLACED-BUT-UNLINKED SNAPSHOT WINDOW.
//
// Under Option B an insert holds NO lock between its placement section (write
// lock) and its link phase (read lock). The point is already in h.nodes and in
// the arena at the start of that gap and has no edges at any level until the end
// of it. A graph serializer — Snapshot or SavePersist — needs only the read lock
// plus the link barrier, and if the barrier covered only the link phase both
// would be free in the gap. The snapshot then records the point with an empty
// adjacency list, and the RESTORED index holds a point that Get returns and
// vector search can never reach. Not a transient inconsistency: a permanent one,
// baked into a durable artifact.
//
// Production reaches this through SnapshotAll, putColdSnapshot, the non-WAL
// writeSnapshotFile, SavePersist on a Persistent-no-WAL collection, and
// Collection.Snapshot.
//
// The fix is that the barrier spans placement AND linking, taken by the insert
// body before the write lock and released after the link phase — with linkMu
// strictly OUTSIDE h.mu, since the reverse order deadlocks (see
// link_stripes.go). These tests pin both halves: that the window is closed, and
// that closing it did not introduce the cycle.

// TestSnapshotNeverSerializesAnUnlinkedPoint drives a REAL hnsw.Insert and
// interposes at the placement/link gap through linkGapHook.
//
// It deliberately does not re-implement insertBody's lock sequence by hand. An
// earlier version did, and it was worthless as a regression gate for exactly the
// reason such tests usually are: emulating the body means taking the barrier
// ITSELF, so the test kept passing with the barrier deleted from the code it was
// supposed to be guarding. The hook costs one nil check in production and buys
// an assertion that is actually about insertBody.
//
// Two assertions, both deterministic in the failing direction. The snapshot must
// not COMPLETE while the point is unlinked — it should be parked on the barrier
// the real insert is holding — and the bytes it eventually produces must restore
// to an index where that point has both out-edges and in-edges.
func TestSnapshotNeverSerializesAnUnlinkedPoint(t *testing.T) {
	const n, dim = 400, 16
	cfg := Config{Dim: dim, Metric: L2, M: 8, EfConstruction: 64, EfSearch: 32, Seed: 9}
	h, err := newHNSW(cfg)
	if err != nil {
		t.Fatal(err)
	}
	ids, vecs := siftLikeCorpus(n, dim, 4)
	insertAllHNSW(t, h, ids, vecs)

	// The point under test. A fresh id, so placement takes the append path.
	const victim = uint64(1 << 40)
	rng := rand.New(rand.NewSource(77))
	stored := make([]float32, dim)
	for d := range stored {
		stored[d] = rng.Float32()
	}

	var buf bytes.Buffer
	var snapErr error
	done := make(chan struct{})

	// The hook runs on the inserting goroutine, inside insertBody, after
	// placement and before the link phase — the whole window, entered for real.
	defer func() { linkGapHook = nil }()
	linkGapHook = func() {
		// Precondition: the point is visible to a serializer and has no edges. If
		// this stops holding, the window has moved and the assertions below are
		// measuring something else.
		h.mu.RLock()
		slot, present := h.arena.Slot(victim)
		nd := h.nodeAt(slot)
		edges := 0
		if present && nd != nil {
			for lc := 0; lc <= nd.level; lc++ {
				edges += h.nbrLen(nd, lc)
			}
		}
		h.mu.RUnlock()
		if !present || nd == nil {
			t.Error("the placed point is not visible in the gap — nothing for a snapshot to capture")
			close(done)
			return
		}
		if edges != 0 {
			t.Errorf("placed node already has %d edges before linking — the gap under test does not exist", edges)
			close(done)
			return
		}

		started := make(chan struct{})
		go func() {
			close(started)
			snapErr = h.Snapshot(&buf)
			close(done)
		}()
		<-started
		// Give the serializer a real chance to acquire. This only affects how
		// forcefully a REGRESSION is caught; it cannot cause a false failure,
		// because a correct barrier makes completion here impossible no matter how
		// long we wait.
		time.Sleep(100 * time.Millisecond)
		select {
		case <-done:
			t.Error("Snapshot completed while a point was placed but unlinked — the barrier does not span the placement/link gap")
		default:
		}
	}

	if _, _, err := h.Insert(victim, stored, 0, nil, nil, nil, CASCond{}); err != nil {
		t.Fatalf("insert: %v", err)
	}
	<-done
	if t.Failed() {
		t.FailNow()
	}
	if snapErr != nil {
		t.Fatalf("snapshot: %v", snapErr)
	}

	// ---- the durable outcome ----
	restored, err := newHNSW(cfg)
	if err != nil {
		t.Fatal(err)
	}
	if err := restored.Restore(bytes.NewReader(buf.Bytes())); err != nil {
		t.Fatalf("restore: %v", err)
	}
	slot, ok := restored.arena.Slot(victim)
	if !ok {
		t.Fatal("the point is missing from the restored index entirely")
	}
	rnd := restored.nodeAt(slot)
	if rnd == nil {
		t.Fatal("the point has no graph node in the restored index")
	}
	out := 0
	for lc := 0; lc <= rnd.level; lc++ {
		out += restored.nbrLen(rnd, lc)
	}
	in := 0
	for _, other := range restored.nodes {
		if other == nil || other.slot == slot {
			continue
		}
		for lc := 0; lc <= other.level; lc++ {
			for _, nb := range restored.nbrsAt(other, lc) {
				if nb == slot {
					in++
				}
			}
		}
	}
	if out == 0 || in == 0 {
		t.Fatalf("restored point has out-degree %d, in-degree %d — a snapshot captured it unlinked, so it is permanently unreachable by search", out, in)
	}
	t.Logf("snapshot parked on the barrier; restored point has out-degree %d, in-degree %d", out, in)
}

// TestSnapshotBarrierUnderInsertSweepReclaim is the deadlock half.
//
// Closing the window means an insert holds the barrier across a section where it
// re-acquires h.mu. Take the two locks in the wrong order and that closes a
// three-way cycle the moment a second write-lock contender appears: a Snapshot
// holding h.mu.R and waiting on linkMu.W, a TTL sweep or Reclaim queued on
// h.mu.W behind it, and a linker holding linkMu.R that Go's RWMutex will no
// longer admit as a reader because a writer is waiting.
//
// This runs exactly that mix — inserts, snapshots, sweeps, reclaims and searches
// — behind a watchdog, and checks every snapshot it captures restores with no
// unreachable point. Both non-opMu writers named in the review (sweepOnce,
// Reclaim) are present, and TTLs are set on some points so sweepOnce takes the
// write lock instead of its atomic fast path.
func TestSnapshotBarrierUnderInsertSweepReclaim(t *testing.T) {
	const dim = 12
	cfg := Config{Dim: dim, Metric: L2, M: 8, EfConstruction: 64, EfSearch: 32, Seed: 21}
	h, err := newHNSW(cfg)
	if err != nil {
		t.Fatal(err)
	}
	ids, vecs := siftLikeCorpus(200, dim, 6)
	insertAllHNSW(t, h, ids, vecs)

	var stop atomic.Bool
	var wg sync.WaitGroup
	var nextID atomic.Uint64
	nextID.Store(1 << 40)

	spawn := func(fn func()) {
		wg.Add(1)
		go func() {
			defer wg.Done()
			fn()
		}()
	}

	// Inserters. Half the points carry a TTL so the sweeper has work and actually
	// takes the write lock.
	for w := 0; w < 3; w++ {
		spawn(func() {
			rng := rand.New(rand.NewSource(int64(w) + 5))
			v := make([]float32, dim)
			for !stop.Load() {
				for d := range v {
					v[d] = rng.Float32()
				}
				ttl := time.Duration(0)
				if rng.Intn(2) == 0 {
					ttl = time.Duration(rng.Intn(20)+1) * time.Millisecond
				}
				if _, _, err := h.Insert(nextID.Add(1), v, ttl, nil, nil, nil, CASCond{}); err != nil {
					t.Errorf("insert: %v", err)
					return
				}
			}
		})
	}
	// The two non-opMu write-lock contenders.
	spawn(func() {
		for !stop.Load() {
			h.sweepOnce()
		}
	})
	spawn(func() {
		for !stop.Load() {
			h.Reclaim()
		}
	})
	// Searchers, so the striped read path is exercised alongside all of it.
	for r := 0; r < 2; r++ {
		spawn(func() {
			rng := rand.New(rand.NewSource(int64(r) + 300))
			q := make([]float32, dim)
			var dst []Result
			for !stop.Load() {
				for d := range q {
					q[d] = rng.Float32()
				}
				// A goroutine-local error: assigning the enclosing function's `err`
				// from two searchers is itself a data race, and -race reports it
				// against the test rather than the code under test.
				var serr error
				if dst, serr = h.SearchInto(dst[:0], q, 5, Filter{}); serr != nil {
					t.Errorf("search: %v", serr)
					return
				}
			}
		})
	}
	// The serializer under test.
	var snaps [][]byte
	var snapMu sync.Mutex
	spawn(func() {
		for !stop.Load() {
			var buf bytes.Buffer
			if err := h.Snapshot(&buf); err != nil {
				t.Errorf("snapshot: %v", err)
				return
			}
			snapMu.Lock()
			if len(snaps) < 12 {
				snaps = append(snaps, buf.Bytes())
			}
			snapMu.Unlock()
		}
	})

	// Watchdog. A lock-order inversion shows up as a hang, and `go test`'s package
	// timeout would report it as an unrelated 25-minute failure; this names it.
	finished := make(chan struct{})
	go func() {
		time.Sleep(2 * time.Second)
		stop.Store(true)
		wg.Wait()
		close(finished)
	}()
	select {
	case <-finished:
	case <-time.After(90 * time.Second):
		t.Fatal("insert + snapshot + sweep + reclaim did not quiesce — deadlock (check the linkMu/h.mu acquisition order)")
	}

	snapMu.Lock()
	captured := snaps
	snapMu.Unlock()
	if len(captured) == 0 {
		t.Fatal("no snapshots captured — the test exercised nothing")
	}
	for i, raw := range captured {
		restored, err := newHNSW(cfg)
		if err != nil {
			t.Fatal(err)
		}
		if err := restored.Restore(bytes.NewReader(raw)); err != nil {
			t.Fatalf("snapshot %d: restore: %v", i, err)
		}
		if bad := unreachableLivePoints(restored); len(bad) != 0 {
			t.Fatalf("snapshot %d restored with %d point(s) unreachable from the entry point (first few: %v) — a serializer captured a placed-but-unlinked node",
				i, len(bad), bad[:min(len(bad), 8)])
		}
	}
	t.Logf("%d snapshots taken under concurrent insert/sweep/reclaim/search, all restore fully reachable", len(captured))
}

// unreachableLivePoints returns the slots of nodes that a level-0 walk from the
// entry point cannot reach — the exact shape a placed-but-unlinked node takes
// after restore.
func unreachableLivePoints(h *hnsw) []uint32 {
	if h.maxLevel < 0 || len(h.nodes) == 0 {
		return nil
	}
	seen := make([]bool, len(h.nodes))
	stack := []uint32{h.entryPoint}
	seen[h.entryPoint] = true
	for len(stack) > 0 {
		s := stack[len(stack)-1]
		stack = stack[:len(stack)-1]
		nd := h.nodes[s]
		if nd == nil {
			continue
		}
		for lc := 0; lc <= nd.level; lc++ {
			for _, nb := range h.nbrsAt(nd, lc) {
				if int(nb) < len(seen) && !seen[nb] {
					seen[nb] = true
					stack = append(stack, nb)
				}
			}
		}
	}
	var bad []uint32
	for slot, nd := range h.nodes {
		if nd == nil || h.tombstoned[uint32(slot)] {
			continue
		}
		if !seen[slot] {
			bad = append(bad, uint32(slot))
		}
	}
	return bad
}
