// SPDX-License-Identifier: Apache-2.0

package vector

import (
	"errors"
	"testing"
	"time"
)

// A REJECTED UPSERT MUST NOT RESURRECT THE POINT IT WAS REPLACING.
//
// The upsert-over-dead-slot reclaim (hnsw.placeLockedAt / ivf.insertLockedAt) is
// a FREE followed by a REUSE: it un-tombstones the id's slot, hands it back to
// the arena's free list, lets arena.Delete clear its expiry — and then relies on
// the arena.Insert right after to take that same slot straight back. The freed
// slot keeps its id, its vector and its metadata the whole time, and its in-edges
// are deliberately left dangling (HNSW) / its inverted-list entry left in place
// (IVF), because the point is about to be the same point again.
//
// Everything about that is safe only if nothing can return in between. The
// admission gate — hnsw.admitVerdictOf, ivf.admits — asks exactly three
// questions: is the slot tombstoned, is it expired, (HNSW) is it inside a link
// window. The reclaim clears the first two and never sets the third, so a slot
// abandoned mid-reclaim answers "live" to all of them. A traversal that reaches
// it through those dangling edges passes admission and emits Result{ID: <the
// deleted id>} scored against the OLD vector, with the OLD payload readable
// through Search's projection. A deleted point comes back from the dead.
//
// Both quota checks USED TO SIT IN THAT GAP. They looked self-healing — the
// reclaim releases one vector and one insert's worth of bytes before they run, so
// an upsert into an exactly-full collection passes — but "self-healing" only
// covers a collection that is AT its quota. A collection that is OVER it was not
// hypothetical: BuildConcurrentMeta, the bulk-load path behind
// Collection.StageBulk/BuildStaged and the server's bulk-ingest op, consulted
// neither MaxVectors nor MaxBytes. Bulk-load past the quota and every subsequent
// upsert resurrected the id it was trying to replace.
//
// THAT ROUTE IS NOW CLOSED (see bulkQuotaErr and bulk_quota_test.go): the bulk
// builders bound the load the same way the insert path bounds one insert, and
// TestBulkLoadCannotCreateTheOverQuotaState below pins that they do. The
// over-quota STATE is still reachable by other means — most plainly an operator
// tightening MaxVectors on a collection that already holds more — so the
// mechanism tests below keep their value, and they now construct that state the
// way it actually arises rather than through a hole that no longer exists.
//
// The fix decides both quotas BEFORE the reclaim, judged against the accounting
// the reclaim is about to produce, so free-and-reuse is unconditional. These
// tests pin both halves: no ghost when the quota rejects, and — the thing a naive
// reordering breaks — an upsert into an exactly-full collection still succeeds.

// reachableGhostSlots audits the WHOLE index for the shape this bug produced: a
// slot that no longer holds a live point but that a traversal can still reach AND
// that the admission gate accepts. Any such slot is a ghost — a search stepping
// onto it emits arena.ID(slot), an id that is not in the index.
//
// This is a STRUCTURAL check, not a check on one search, and that is the point.
// The alternative hardening — teaching admitVerdictOf to reject a nil node —
// would put a random-access pointer load on the per-candidate hot loop, which is
// where an equivalent per-hop check was measured and rejected before. Auditing
// the invariant from outside costs the hot loop nothing and catches any FUTURE
// producer of the same shape, wherever it lives.
//
// Reachability is over-approximated as "appears in some non-nil node's neighbour
// list, or is the entry point": searchLayerCore expands tombstoned nodes on
// purpose, so out-edges of a dead-but-present node are real paths, and a superset
// is the safe direction for an invariant that must be empty.
func reachableGhostSlots(h *hnsw) []uint32 {
	h.mu.RLock()
	defer h.mu.RUnlock()
	now := uint64(h.now())
	seen := make(map[uint32]bool)
	var ghosts []uint32
	consider := func(slot uint32) {
		if seen[slot] {
			return
		}
		seen[slot] = true
		if h.arena.Allocated(slot) {
			return // slot holds a live point: emitting its id is correct
		}
		if h.admits(slot, nil, now) {
			ghosts = append(ghosts, slot)
		}
	}
	if h.maxLevel >= 0 {
		consider(h.entryPoint)
	}
	for _, nd := range h.nodes {
		if nd == nil {
			continue
		}
		for lc := 0; lc <= nd.level; lc++ {
			for _, m := range h.nbrsAt(nd, lc) {
				consider(m)
			}
		}
	}
	return ghosts
}

// reachableGhostSlotsIVF is reachableGhostSlots for IVF, where "reachable" means
// "still filed in an inverted list" — the probe loop scans those lists and runs
// admits() on every entry.
func reachableGhostSlotsIVF(ix *ivf) []uint32 {
	ix.mu.RLock()
	defer ix.mu.RUnlock()
	now := uint64(ix.now())
	seen := make(map[uint32]bool)
	var ghosts []uint32
	for _, list := range ix.lists {
		for _, slot := range list {
			if seen[slot] {
				continue
			}
			seen[slot] = true
			if ix.arena.Allocated(slot) {
				continue
			}
			if ix.admits(slot, nil, now) {
				ghosts = append(ghosts, slot)
			}
		}
	}
	return ghosts
}

// overQuotaHNSW produces an index holding n points whose quota `cfg` cannot
// hold, which is the state that makes the reclaim's quota check reject.
//
// It loads the points with the quota UNSET and applies cfg's quota afterwards,
// which is the sequence an operator produces by tightening MaxVectors/MaxBytes
// on a collection that already holds more than the new cap. It used to just
// bulk-load past the quota directly — the bulk builders ignored it — and that is
// no longer possible, which is itself now asserted by
// TestBulkLoadCannotCreateTheOverQuotaState.
func overQuotaHNSW(t *testing.T, cfg Config, n, dim int, seed int64) (*hnsw, []uint64, [][]float32) {
	t.Helper()
	load := cfg
	load.MaxVectors, load.MaxBytes = 0, 0
	h, err := newHNSW(load)
	if err != nil {
		t.Fatal(err)
	}
	ids, vecs := siftLikeCorpus(n, dim, seed)
	if err := h.BuildConcurrent(ids, vecs, 2); err != nil {
		t.Fatalf("bulk build: %v", err)
	}
	h.mu.Lock()
	h.cfg.MaxVectors, h.cfg.MaxBytes = cfg.MaxVectors, cfg.MaxBytes
	h.mu.Unlock()
	return h, ids, vecs
}

// hnswState is the accounting a rejected write must leave untouched.
type hnswState struct {
	arenaSize  int
	tombstoned int
	bytesUsed  int64
}

func snapshotHNSW(h *hnsw) hnswState {
	h.mu.RLock()
	defer h.mu.RUnlock()
	return hnswState{h.arena.Size(), len(h.tombstoned), h.bytesUsed}
}

// assertNoHNSWGhost fails if `victim` is reachable through ANY read path.
func assertNoHNSWGhost(t *testing.T, h *hnsw, victim uint64, near []float32) {
	t.Helper()
	if _, _, _, _, _, ok := h.Get(victim); ok {
		t.Errorf("Get(%d) returned the deleted point", victim)
	}
	// k = whole corpus, so the traversal has every reason to reach the slot.
	res, err := h.Search(near, h.arena.Capacity())
	if err != nil {
		t.Fatalf("search: %v", err)
	}
	for _, r := range res {
		if r.ID == victim {
			t.Errorf("Search returned deleted id %d (dist %v) after a rejected upsert — the point was resurrected with its stale vector", r.ID, r.Distance)
		}
	}
	// The slot is also reachable by brute-force scan paths, which run the same
	// admission gate.
	if h.Exists(victim) {
		t.Errorf("Exists(%d) is true for the deleted point", victim)
	}
	// And the structural form: no dead-but-admissible slot anywhere in the graph,
	// whether or not this particular query happened to walk into it.
	for _, slot := range reachableGhostSlots(h) {
		t.Errorf("slot %d holds no live point yet is reachable AND admissible — it would be emitted as id %d", slot, h.arena.ID(slot))
	}
}

func TestRejectedUpsertLeavesNoGhostHNSWMaxVectors(t *testing.T) {
	const n, dim = 64, 8
	cfg := Config{Dim: dim, Metric: L2, M: 8, EfConstruction: 64, EfSearch: 64, Seed: 12, MaxVectors: 32}
	h, ids, vecs := overQuotaHNSW(t, cfg, n, dim, 21)
	if int64(h.arena.Size()) <= cfg.MaxVectors {
		t.Fatalf("arena size %d is not over MaxVectors %d — this test is not exercising the rejection it claims", h.arena.Size(), cfg.MaxVectors)
	}

	victim := ids[n/2]
	if ok, err := h.Delete(victim, CASCond{}); err != nil || !ok {
		t.Fatalf("delete: ok=%v err=%v", ok, err)
	}
	before := snapshotHNSW(h)

	if _, _, err := h.Insert(victim, vecs[n/2], 0, nil, nil, nil, CASCond{}); !errors.Is(err, ErrCollectionFull) {
		t.Fatalf("upsert into an over-quota collection: got %v, want ErrCollectionFull", err)
	}
	assertNoHNSWGhost(t, h, victim, vecs[n/2])

	// A rejected write must mutate nothing: the slot is still tombstoned, still
	// counted, and the byte estimate has not been credited for a release that
	// never happened.
	if after := snapshotHNSW(h); after != before {
		t.Errorf("rejected upsert mutated the index accounting: before %+v, after %+v", before, after)
	}
	// Every other point is untouched.
	for i, id := range ids {
		if id == victim {
			continue
		}
		if _, _, _, _, _, ok := h.Get(id); !ok {
			t.Fatalf("point %d (index %d) disappeared after the rejected upsert", id, i)
		}
	}
}

func TestRejectedUpsertLeavesNoGhostHNSWMaxBytes(t *testing.T) {
	const n, dim = 64, 8
	per := estimateInsertBytes(dim, 8)
	// Budget for n-1 points; the bulk build places n, so the collection is over
	// budget by exactly one insert and the reclaim's release cannot cover it.
	cfg := Config{Dim: dim, Metric: L2, M: 8, EfConstruction: 64, EfSearch: 64, Seed: 12, MaxBytes: per * (n - 1)}
	h, ids, vecs := overQuotaHNSW(t, cfg, n, dim, 21)

	victim := ids[n/2]
	if ok, err := h.Delete(victim, CASCond{}); err != nil || !ok {
		t.Fatalf("delete: ok=%v err=%v", ok, err)
	}
	before := snapshotHNSW(h)

	if _, _, err := h.Insert(victim, vecs[n/2], 0, nil, nil, nil, CASCond{}); !errors.Is(err, ErrCollectionFull) {
		t.Fatalf("upsert over the byte budget: got %v, want ErrCollectionFull", err)
	}
	assertNoHNSWGhost(t, h, victim, vecs[n/2])
	if after := snapshotHNSW(h); after != before {
		t.Errorf("rejected upsert mutated the index accounting: before %+v, after %+v", before, after)
	}
}

// The reclaim fires for an EXPIRED slot too, and that case never involved a
// Delete at all: the point aged out, the sweep has not run, and re-inserting the
// same id is an ordinary write. A rejection there used to make an EXPIRED point
// searchable again.
func TestRejectedUpsertLeavesNoGhostHNSWExpired(t *testing.T) {
	const n, dim = 64, 8
	cfg := Config{Dim: dim, Metric: L2, M: 8, EfConstruction: 64, EfSearch: 64, Seed: 12, MaxVectors: 32}
	h, ids, vecs := overQuotaHNSW(t, cfg, n, dim, 21)

	clock := int64(1_000_000)
	h.SetNowFunc(func() int64 { return clock })

	// Give one point a TTL through the ordinary path, then walk the clock past it.
	victim := ids[n/2]
	if ok, err := h.Delete(victim, CASCond{}); err != nil || !ok {
		t.Fatalf("delete: ok=%v err=%v", ok, err)
	}
	// Re-inserting is itself over quota, so seed the TTL by lowering the bar for
	// exactly this write: raise the quota, insert with a TTL, put it back.
	h.mu.Lock()
	saved := h.cfg.MaxVectors
	h.cfg.MaxVectors = 0
	h.mu.Unlock()
	if _, _, err := h.Insert(victim, vecs[n/2], time.Second, nil, nil, nil, CASCond{}); err != nil {
		t.Fatalf("seeding the TTL point: %v", err)
	}
	h.mu.Lock()
	h.cfg.MaxVectors = saved
	h.mu.Unlock()

	clock += 5_000 // past the TTL; the sweep has not run, so the slot is expired-not-swept
	before := snapshotHNSW(h)

	if _, _, err := h.Insert(victim, vecs[n/2], 0, nil, nil, nil, CASCond{}); !errors.Is(err, ErrCollectionFull) {
		t.Fatalf("upsert over an expired slot in an over-quota collection: got %v, want ErrCollectionFull", err)
	}
	assertNoHNSWGhost(t, h, victim, vecs[n/2])
	if after := snapshotHNSW(h); after != before {
		t.Errorf("rejected upsert mutated the index accounting: before %+v, after %+v", before, after)
	}
}

// ---- IVF ----

type ivfState struct {
	arenaSize  int
	tombstoned int
	bytesUsed  int64
}

func snapshotIVF(ix *ivf) ivfState {
	ix.mu.RLock()
	defer ix.mu.RUnlock()
	return ivfState{ix.arena.Size(), len(ix.tombstoned), ix.bytesUsed}
}

func assertNoIVFGhost(t *testing.T, ix *ivf, victim uint64, near []float32) {
	t.Helper()
	if _, _, _, _, _, ok := ix.Get(victim); ok {
		t.Errorf("Get(%d) returned the deleted point", victim)
	}
	res, err := ix.Search(near, ix.arena.Capacity())
	if err != nil {
		t.Fatalf("search: %v", err)
	}
	for _, r := range res {
		if r.ID == victim {
			t.Errorf("Search returned deleted id %d (dist %v) after a rejected upsert — the point was resurrected with its stale vector", r.ID, r.Distance)
		}
	}
	if ix.Exists(victim) {
		t.Errorf("Exists(%d) is true for the deleted point", victim)
	}
	for _, slot := range reachableGhostSlotsIVF(ix) {
		t.Errorf("slot %d holds no live point yet is still filed in a list AND admissible — it would be emitted as id %d", slot, ix.arena.ID(slot))
	}
}

// overQuotaIVF is overQuotaHNSW's IVF twin, and reaches the state the same way
// and for the same reason — see there.
func overQuotaIVF(t *testing.T, cfg Config, n, dim int, seed int64) (*ivf, []uint64, [][]float32) {
	t.Helper()
	load := cfg
	load.MaxVectors, load.MaxBytes = 0, 0
	ix, err := newIVF(load)
	if err != nil {
		t.Fatal(err)
	}
	ids, vecs := siftLikeCorpus(n, dim, seed)
	if err := ix.BuildConcurrent(ids, vecs, 2); err != nil {
		t.Fatalf("bulk build: %v", err)
	}
	ix.mu.Lock()
	ix.cfg.MaxVectors, ix.cfg.MaxBytes = cfg.MaxVectors, cfg.MaxBytes
	ix.mu.Unlock()
	return ix, ids, vecs
}

func TestRejectedUpsertLeavesNoGhostIVF(t *testing.T) {
	const n, dim = 32, 8
	per := estimateInsertBytes(dim, 8)
	base := Config{Dim: dim, Metric: L2, M: 8, EfConstruction: 64, EfSearch: 64, Seed: 7, IVFNlist: 4, IVFNprobe: 4}

	for _, tc := range []struct {
		name string
		cfg  Config
	}{
		{"MaxVectors", func() Config { c := base; c.MaxVectors = n / 2; return c }()},
		{"MaxBytes", func() Config { c := base; c.MaxBytes = per * (n - 1); return c }()},
	} {
		t.Run(tc.name, func(t *testing.T) {
			ix, ids, vecs := overQuotaIVF(t, tc.cfg, n, dim, 3)
			victim := ids[n/2]
			if ok, err := ix.Delete(victim, CASCond{}); err != nil || !ok {
				t.Fatalf("delete: ok=%v err=%v", ok, err)
			}
			before := snapshotIVF(ix)

			if _, _, err := ix.Insert(victim, vecs[n/2], 0, nil, nil, nil, CASCond{}); !errors.Is(err, ErrCollectionFull) {
				t.Fatalf("upsert into an over-quota collection: got %v, want ErrCollectionFull", err)
			}
			assertNoIVFGhost(t, ix, victim, vecs[n/2])
			if after := snapshotIVF(ix); after != before {
				t.Errorf("rejected upsert mutated the index accounting: before %+v, after %+v", before, after)
			}
			for _, id := range ids {
				if id == victim {
					continue
				}
				if _, _, _, _, _, ok := ix.Get(id); !ok {
					t.Fatalf("point %d disappeared after the rejected upsert", id)
				}
			}
		})
	}
}

// ---- the behaviour a naive reordering breaks ----

// UPSERTING INTO A FULL COLLECTION IS A REPLACE AND MUST SUCCEED. The reclaim
// branch exists to make room, so moving the quota checks in front of it without
// crediting what it is about to release would start rejecting the one write a
// full collection is always allowed to take. This pins that, at the exact
// boundary, for both quotas and both index types — and pins that the quota is
// still ENFORCED for a genuinely new id.
func TestUpsertIntoExactlyFullCollectionStillSucceeds(t *testing.T) {
	const n, dim = 24, 8
	per := estimateInsertBytes(dim, 8)
	base := Config{Dim: dim, Metric: L2, M: 8, EfConstruction: 64, EfSearch: 64, Seed: 9, IVFNlist: 4, IVFNprobe: 4}

	fullByCount := func() Config { c := base; c.MaxVectors = n; return c }
	fullByBytes := func() Config { c := base; c.MaxBytes = per * n; return c }

	type idx interface {
		Insert(uint64, []float32, time.Duration, Metadata, *SparseVector, map[string]int64, CASCond) (uint64, map[string]uint64, error)
		Delete(uint64, CASCond) (bool, error)
		Search([]float32, int) ([]Result, error)
		Exists(uint64) bool
	}

	for _, engine := range []struct {
		name string
		make func(Config) (idx, error)
	}{
		{"hnsw", func(c Config) (idx, error) { return newHNSW(c) }},
		{"ivf", func(c Config) (idx, error) { return newIVF(c) }},
	} {
		for _, quota := range []struct {
			name string
			cfg  func() Config
		}{
			{"MaxVectors", fullByCount},
			{"MaxBytes", fullByBytes},
		} {
			t.Run(engine.name+"/"+quota.name, func(t *testing.T) {
				ix, err := engine.make(quota.cfg())
				if err != nil {
					t.Fatal(err)
				}
				ids, vecs := siftLikeCorpus(n, dim, 5)
				for i := range ids {
					if _, _, err := ix.Insert(ids[i], vecs[i], 0, nil, nil, nil, CASCond{}); err != nil {
						t.Fatalf("filling to capacity, insert %d: %v", i, err)
					}
				}
				// The collection is now exactly full: a NEW id must be rejected.
				if _, _, err := ix.Insert(9999, vecs[0], 0, nil, nil, nil, CASCond{}); !errors.Is(err, ErrCollectionFull) {
					t.Fatalf("new id into a full collection: got %v, want ErrCollectionFull", err)
				}
				// But replacing an existing one must still work.
				victim := ids[n/2]
				// Far from every corpus vector (which are unit-ish normals), so a k=1
				// search for it can only return the upserted point — no distance ties.
				replacement := make([]float32, dim)
				for j := range replacement {
					replacement[j] = 100
				}
				if ok, err := ix.Delete(victim, CASCond{}); err != nil || !ok {
					t.Fatalf("delete: ok=%v err=%v", ok, err)
				}
				if _, _, err := ix.Insert(victim, replacement, 0, nil, nil, nil, CASCond{}); err != nil {
					t.Fatalf("upsert into an exactly-full collection was rejected (%v) — a replace does not grow the collection and must be allowed", err)
				}
				if !ix.Exists(victim) {
					t.Fatal("the replaced point is not live after a successful upsert")
				}
				res, err := ix.Search(replacement, 1)
				if err != nil {
					t.Fatal(err)
				}
				if len(res) == 0 || res[0].ID != victim {
					t.Errorf("searching for the replacement vector did not return the upserted point: got %+v", res)
				}
			})
		}
	}
}

// The auditor is only worth something if it is clean on a HEALTHY index across
// every operation that legitimately frees a slot — Delete, Reclaim, TTL expiry
// and upsert churn. Driving all of them and asserting zero ghosts throughout is
// what makes reachableGhostSlots a regression net rather than a restatement of
// one test's setup: any future code that frees a slot without closing its
// admission gate trips this, wherever it lives.
func TestNoReachableGhostSlotsAcrossLifecycle(t *testing.T) {
	const n, dim = 300, 8
	cfg := Config{Dim: dim, Metric: L2, M: 8, EfConstruction: 64, EfSearch: 64, Seed: 3}
	h, err := newHNSW(cfg)
	if err != nil {
		t.Fatal(err)
	}
	clock := int64(1_000_000)
	h.SetNowFunc(func() int64 { return clock })
	ids, vecs := siftLikeCorpus(n, dim, 17)

	audit := func(stage string) {
		t.Helper()
		for _, slot := range reachableGhostSlots(h) {
			t.Fatalf("%s: slot %d holds no live point yet is reachable AND admissible (would be emitted as id %d)", stage, slot, h.arena.ID(slot))
		}
	}

	insertAllHNSW(t, h, ids, vecs)
	audit("after build")

	for i := 0; i < n; i += 3 { // tombstone a third of the corpus
		if _, err := h.Delete(ids[i], CASCond{}); err != nil {
			t.Fatal(err)
		}
	}
	audit("after deletes")

	if got := h.Reclaim(); got == 0 {
		t.Fatal("Reclaim freed nothing — this stage is not exercising slot reuse")
	}
	audit("after Reclaim")

	for i := 1; i < n; i += 5 { // upsert churn: delete + reinsert onto the same slot
		if _, err := h.Delete(ids[i], CASCond{}); err != nil {
			t.Fatal(err)
		}
		if _, _, err := h.Insert(ids[i], vecs[i], 0, nil, nil, nil, CASCond{}); err != nil {
			t.Fatal(err)
		}
	}
	audit("after upsert churn")

	// TTL: a batch that ages out, audited both before and after the sweep frees
	// the slots.
	for i := 2; i < 60; i += 2 {
		if _, err := h.Delete(ids[i], CASCond{}); err != nil {
			t.Fatal(err)
		}
		if _, _, err := h.Insert(ids[i], vecs[i], time.Second, nil, nil, nil, CASCond{}); err != nil {
			t.Fatal(err)
		}
	}
	clock += 5_000
	audit("after TTL expiry, before sweep")
	if h.sweepOnce() == 0 {
		t.Fatal("the TTL sweep freed nothing — the expiry stage did not fire")
	}
	audit("after TTL sweep")
}

// TestBulkLoadCannotCreateTheOverQuotaState is the assertion that this file's
// original premise is GONE. It used to run StageBulk -> BuildStaged past the
// quota and continue into the ghost checks, on the grounds that this was "the
// exact route a server takes" and therefore proof a user could reach the
// over-quota state. The bulk builders now bound the load, so the route ends in a
// refusal — and the refusal is the thing worth pinning, because it is the only
// reason the resurrection window is out of a user's direct reach.
func TestBulkLoadCannotCreateTheOverQuotaState(t *testing.T) {
	const n, dim = 64, 8
	cfg := Config{Dim: dim, Metric: L2, M: 8, EfConstruction: 64, EfSearch: 64, Seed: 12, MaxVectors: 32}
	c, err := NewCollection("ghost", cfg)
	if err != nil {
		t.Fatal(err)
	}
	ids, vecs := siftLikeCorpus(n, dim, 21)
	if err := c.StageBulk(ids, vecs); err != nil {
		t.Fatal(err)
	}
	// Whichever stage refuses is fine — what must NOT happen is both succeeding
	// and leaving a collection holding 64 points under a 32-point quota.
	err = c.BuildStaged(0)
	if err == nil {
		t.Fatalf("StageBulk+BuildStaged loaded %d points into a MaxVectors=%d collection — "+
			"the bulk path is back to ignoring the quota, which is what put the "+
			"free-then-reuse resurrection window within a user's reach", n, cfg.MaxVectors)
	}
	if !errors.Is(err, ErrCollectionFull) {
		t.Fatalf("BuildStaged past the quota returned %v, want ErrCollectionFull", err)
	}
}

// The ghost mechanism itself, end to end through the public Collection API. The
// index-level tests above prove it at the index seam; this one proves it through
// the type a user actually holds. The over-quota state is applied AFTER the load
// (see overQuotaHNSW for why that is now the honest construction).
func TestRejectedUpsertLeavesNoGhostThroughCollectionAPI(t *testing.T) {
	const n, dim = 64, 8
	cfg := Config{Dim: dim, Metric: L2, M: 8, EfConstruction: 64, EfSearch: 64, Seed: 12}
	c, err := NewCollection("ghost", cfg)
	if err != nil {
		t.Fatal(err)
	}
	ids, vecs := siftLikeCorpus(n, dim, 21)
	if err := c.StageBulk(ids, vecs); err != nil {
		t.Fatal(err)
	}
	if err := c.BuildStaged(0); err != nil {
		t.Fatal(err)
	}
	// Tighten the quota under the loaded collection, as an operator would.
	h, ok := c.idx.(*hnsw)
	if !ok {
		t.Fatalf("collection is backed by %T, not *hnsw", c.idx)
	}
	h.mu.Lock()
	h.cfg.MaxVectors = 32
	h.mu.Unlock()

	victim := ids[n/2]
	if !c.Delete(victim) {
		t.Fatal("delete failed")
	}
	if err := c.Upsert(victim, vecs[n/2], "", 0, nil, nil); !errors.Is(err, ErrCollectionFull) {
		t.Fatalf("upsert into an over-quota collection: got %v, want ErrCollectionFull", err)
	}
	if _, _, _, _, _, ok := c.Get(victim); ok {
		t.Errorf("Get(%d) returned the deleted point", victim)
	}
	res, err := c.Search(vecs[n/2], n)
	if err != nil {
		t.Fatal(err)
	}
	for _, r := range res {
		if r.ID == victim {
			t.Errorf("Search returned deleted id %d after a rejected Upsert — a deleted point is visible to users", r.ID)
		}
	}
}
