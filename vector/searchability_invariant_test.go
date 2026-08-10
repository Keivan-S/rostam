// SPDX-License-Identifier: Apache-2.0

package vector

import (
	"bytes"
	"fmt"
	"math"
	"math/rand"
	"sync"
	"sync/atomic"
	"testing"
	"time"
)

// THE SEARCHABILITY INVARIANT: EVERY LIVE POINT MUST BE FINDABLE BY SEARCHING
// FOR ITS OWN VECTOR.
//
// This file is the standing property gate for that one sentence. It exists
// because three separate bugs in two days each produced the same shape — a
// point that was stored, live, counted by the live-count metric, and returned
// by Get, but that NO search of any kind would return — and the suite as it
// stood asserted storage, Get and counts without ever asking "can I search for
// what I just stored":
//
//  1. id 0 as an empty-slot sentinel. `arena.ID(slot) == 0` stood in for "this
//     slot holds no point" at fourteen result-assembly sites, so a point
//     genuinely stored under user id 0 was dropped by HNSW, IVF, Vamana, named,
//     filtered, filter-first, sparse and full-text search alike. Fixed;
//     the focused repro is vector/id_zero_test.go.
//  2. An empty candidate set written as an empty edge set. searchLayer admits
//     only LIVE points and cannot terminate early while empty, so an empty
//     return means every node reachable at that level is tombstoned — and
//     linkNode accepted it, writing a node with no out-edges and (since
//     back-edges only go to the neighbours just chosen) no in-edges. A
//     permanent orphan that could promote ITSELF to entry point. Fixed;
//     see vector/empty_candidate_orphan_test.go.
//  3. A reused slot turned into a traversal dead end. An upsert lands on the
//     slot the deleted point held; zeroing that slot's level-0 edges at
//     placement made every surviving in-edge into it a dead end for the whole
//     link window, and under churn an id-ordered sweep of dead ends partitions
//     the graph. Fixed; see vector/reused_slot_traversable_test.go.
//
// Each of those has a focused regression test pinning its own mechanism. What
// was missing — and what this file is — is the PROPERTY those three tests are
// all instances of, swept across index families, lifecycle states, id patterns
// and configurations, so the NEXT bug of this shape is caught by construction
// rather than by a fourth incident.
//
// WHY THE SELF-QUERY IS THE RIGHT PROBE. Querying an index with a point's own
// stored vector is the easiest possible query: the target sits at distance
// exactly 0, which is the global minimum. For an exact (unquantized) index over
// distinct vectors it is therefore the unique arg-min, so RANK 0 is not an
// approximation of the invariant — it is the invariant, stated in the
// strongest form the API can express. An approximate index that cannot answer
// its own easiest query has lost the point, not merely mis-ranked it.
//
// HOW THE CAVEATS ARE HANDLED (none of them weaken the exact assertion):
//
//   - TIES. Rank 0 is only well defined when no other stored vector sits at
//     distance 0 too. Rather than assume it, newInvCorpus ASSERTS pairwise
//     distinctness at construction (see invCorpus) and fails loudly if a
//     generator ever produces a collision.
//   - DOTPRODUCT. For unnormalized vectors a self-query is genuinely NOT
//     rank-0: -dot(v,w) can beat -dot(v,v) whenever |w| > |v| and w is aligned
//     with v, so the arg-min is a longer vector, not the point itself. Every
//     corpus here is therefore generated at UNIT NORM, which makes -dot(v,w) =
//     -cos(v,w) >= -1 = -dot(v,v) with equality only for v itself. The rank-0
//     assertion is then mathematically sound for all three metrics rather than
//     true-by-luck for two of them. (This is a property of the corpus, not a
//     weakening of the check.)
//   - QUANTIZATION. PQ/SQ/PRQ score candidates on lossy codes, so the arg-min
//     of the CODED distance need not be the point itself and rank 0 is not
//     owed. Quantized configurations live in their own test
//     (TestSearchabilityInvariantQuantized) and assert TOP-K MEMBERSHIP with a
//     documented k, never rank 0. They are deliberately kept out of the exact
//     matrix so that the exact assertion never has to be loosened to
//     accommodate them.
//   - APPROXIMATE RECALL. HNSW/Vamana traversal is approximate, so in principle
//     a self-query could miss. In practice the target is the distance-0 global
//     minimum, so a miss means a lost point rather than a starved beam — and
//     ONE configuration swept here genuinely does lose points on fixed code: a
//     full upsert sweep at graph degree M <= 6 leaves points UNREACHABLE from
//     the entry point, which no width of beam can recover. That is measured,
//     bounded and written up in full on TestSearchabilityInvariantConfigs,
//     which asserts the low-degree configs on fresh insert and holds the churn
//     phase to M >= 8. It is bounded rather than absorbed on purpose: the
//     assertion is never weakened to make a config pass.
//
// RUNTIME. The default arm is sized to run on every push; the large-corpus arms
// are behind testing.Short() (skipped by -short), following the -short
// precedent in the filter-bitset gate suites.

// invSearchK is the k used for every self-query. It is deliberately larger than
// 1: asking for ten and demanding the point at index 0 distinguishes "ranked
// badly" from "absent", and the failure message can then print what DID come
// back instead of just an empty slice.
const invSearchK = 10

// invQuantTopK is the weaker bound used for quantized configurations, where the
// coded distance is lossy and the point's own vector is not owed the arg-min.
// Ten out of a 512-point corpus is ~2%: loose enough to absorb code error,
// tight enough that a point which has fallen out of the index entirely (the bug
// class this file exists for) still fails.
const invQuantTopK = 10

// ---------------------------------------------------------------------------
// Corpus
// ---------------------------------------------------------------------------

// invMix is splitmix64. It turns an id into a well-distributed stream so that
// the vector for id N is a pure function of N — stable no matter what order the
// ids are inserted in, which is what lets a shuffled-insertion arm still look
// up "the vector stored under id N".
func invMix(x uint64) uint64 {
	x += 0x9E3779B97F4A7C15
	x ^= x >> 30
	x *= 0xBF58476D1CE4E5B9
	x ^= x >> 27
	x *= 0x94D049BB133111EB
	x ^= x >> 31
	return x
}

// invUnitVec builds the unit-norm vector for (id, salt). Unit norm is
// load-bearing for the DotProduct arm — see the caveats at the top of the file.
// salt lets one id carry different vectors in different phases (the upsert arm
// needs the SAME id to hold a DIFFERENT vector after the upsert).
func invUnitVec(dim int, id, salt uint64) []float32 {
	s := invMix(id ^ invMix(salt))
	v := make([]float32, dim)
	var norm float64
	for i := range v {
		s = invMix(s)
		// 24 bits of entropy per component, centred on zero, in [-1, 1).
		f := float32(int64(s>>40)-(1<<23)) / float32(1<<23)
		v[i] = f
		norm += float64(f) * float64(f)
	}
	norm = math.Sqrt(norm)
	if norm == 0 {
		// Unreachable for dim >= 1 with 24-bit components, but a zero vector is
		// invalid under Cosine, so refuse to emit one rather than produce a
		// mysterious failure downstream.
		v[0] = 1
		return v
	}
	inv := float32(1 / norm)
	for i := range v {
		v[i] *= inv
	}
	return v
}

// invCorpus is a set of ids with the vector each one holds. It is built with a
// pairwise-distinctness check so that "rank 0" is unambiguous by construction
// rather than by assumption: two identical stored vectors would make the
// arg-min a tie, and a tie would make the invariant untestable.
type invCorpus struct {
	dim int
	ids []uint64 // in insertion order
	vec map[uint64][]float32
}

func newInvCorpus(t *testing.T, ids []uint64, dim int, salt uint64) *invCorpus {
	t.Helper()
	c := &invCorpus{dim: dim, ids: ids, vec: make(map[uint64][]float32, len(ids))}
	seen := make(map[string]uint64, len(ids))
	for _, id := range ids {
		if _, dup := c.vec[id]; dup {
			t.Fatalf("corpus id %d appears twice — the id list must be a set", id)
		}
		v := invUnitVec(dim, id, salt)
		key := invVecKey(v)
		if other, dup := seen[key]; dup {
			t.Fatalf("corpus ids %d and %d generated IDENTICAL vectors (dim=%d salt=%d) — "+
				"rank 0 would be a tie and the invariant untestable; change the generator",
				other, id, dim, salt)
		}
		seen[key] = id
		c.vec[id] = v
	}
	return c
}

// invVecKey is an exact identity key over a vector's bits, used only for the
// distinctness check.
func invVecKey(v []float32) string {
	var b []byte
	for _, f := range v {
		u := math.Float32bits(f)
		b = append(b, byte(u), byte(u>>8), byte(u>>16), byte(u>>24))
	}
	return string(b)
}

// invSeedIDs is the DEFAULT id pattern for the matrix: non-contiguous (stride
// 7), including id 0, and shuffled, so that id N does not land on slot N and
// neither an id-keyed nor a slot-keyed sentinel can pass by accident. The two
// halves of that are what separated bug 1 from a slot-0 bug.
func invSeedIDs(n int, seed int64) []uint64 {
	ids := make([]uint64, n)
	for i := range ids {
		ids[i] = uint64(i) * 7
	}
	r := rand.New(rand.NewSource(seed)) //nolint:gosec // deterministic test fixture
	r.Shuffle(len(ids), func(i, j int) { ids[i], ids[j] = ids[j], ids[i] })
	return ids
}

// ---------------------------------------------------------------------------
// Index adapters
// ---------------------------------------------------------------------------

// invIndex is the minimum surface the invariant needs from an index family:
// put a point in, take a point out, and ask for a vector back. Everything
// beyond that (bulk build, reclaim, snapshot, TTL) is an OPTIONAL interface, so
// a family that cannot do a thing SKIPS the corresponding stage visibly rather
// than silently passing it.
type invIndex interface {
	insert(id uint64, vec []float32, ttl time.Duration) error
	upsert(id uint64, vec []float32) error
	remove(id uint64) error
	// search returns result ids in rank order, best first.
	search(q []float32, k int) ([]uint64, error)
	close()
}

type invBulkBuilder interface {
	bulkBuild(ids []uint64, vecs [][]float32) error
}

// invPayloadBulkBuilder is the payload-bearing bulk load. It is a SEPARATE
// capability from invBulkBuilder because it takes a different route through the
// build — the placement pass writes the arena's metadata, the payload index and
// (where the family has one) the BM25 corpus while the graph is still unbuilt —
// and a point lost by that route is lost exactly as invisibly as one lost by any
// of the three bugs at the top of this file.
type invPayloadBulkBuilder interface {
	bulkBuildPayloads(ids []uint64, vecs [][]float32, metas []Metadata) error
}

type invReclaimer interface {
	reclaim() int
}

// invRoundTripper snapshots the index and restores it into a NEW one of the
// same configuration, returning the restored index (and closing the source).
type invRoundTripper interface {
	roundTrip(t *testing.T) invIndex
}

// invClocked is implemented by families whose TTL clock can be driven from the
// test, so expiry is deterministic rather than a sleep.
type invClocked interface {
	setNow(fn func() int64)
	expiredCount() uint64
}

// invBackfillProbe lets the delete-all-then-backfill stage assert the
// preconditions that make it a real test rather than a vacuous one. See the
// stage for why they are load-bearing.
type invBackfillProbe interface {
	backfillPrecondition() string
}

// invColl adapts *Collection — the dense HNSW, IVF and Vamana families.
type invColl struct {
	c    *Collection
	cfg  Config
	name string
}

func (a *invColl) insert(id uint64, vec []float32, ttl time.Duration) error {
	return a.c.Insert(id, vec, ttl, nil, nil)
}

func (a *invColl) upsert(id uint64, vec []float32) error {
	return a.c.Upsert(id, vec, "", 0, nil, nil)
}

func (a *invColl) remove(id uint64) error {
	if !a.c.Delete(id) {
		return fmt.Errorf("Delete(%d) reported the point absent", id)
	}
	return nil
}

func (a *invColl) search(q []float32, k int) ([]uint64, error) {
	res, err := a.c.Search(q, k)
	if err != nil {
		return nil, err
	}
	return resultIDs(res), nil
}

func (a *invColl) close()                                      { _ = a.c.Close() }
func (a *invColl) reclaim() int                                { return a.c.Reclaim() }
func (a *invColl) setNow(fn func() int64)                      { a.c.SetNowFunc(fn) }
func (a *invColl) expiredCount() uint64                        { return a.c.Stats().Expired }
func (a *invColl) bulkBuild(ids []uint64, v [][]float32) error { return a.c.BuildConcurrent(ids, v, 4) }

func (a *invColl) bulkBuildPayloads(ids []uint64, v [][]float32, m []Metadata) error {
	return a.c.BuildConcurrentMeta(ids, v, m, 4)
}

func (a *invColl) roundTrip(t *testing.T) invIndex {
	t.Helper()
	var buf bytes.Buffer
	if err := a.c.Snapshot(&buf); err != nil {
		t.Fatalf("Snapshot: %v", err)
	}
	dst, err := NewCollection(a.name+"_restored", a.cfg)
	if err != nil {
		t.Fatalf("NewCollection(restore target): %v", err)
	}
	if err := dst.Restore(bytes.NewReader(buf.Bytes())); err != nil {
		_ = dst.Close()
		t.Fatalf("Restore: %v", err)
	}
	a.close()
	return &invColl{c: dst, cfg: a.cfg, name: a.name + "_restored"}
}

// backfillPrecondition checks the two conditions that make the
// delete-all-then-backfill stage a real test on a LAYERED graph: the traversal
// must still be rooted in the dead band (so the backfilled node's link search
// starts somewhere tombstoned), and the graph must be more than one level tall
// (so a backfilled node drawing level 0 cannot promote ITSELF to entry point
// and be repaired by the inserts that follow it, which would mask the failure).
// Returns "" when the state is right, or when this index is not a layered graph
// and the conditions do not apply.
func (a *invColl) backfillPrecondition() string {
	h, ok := a.c.idx.(*hnsw)
	if !ok || h.vamana {
		return "" // not a layered HNSW: nothing to require
	}
	h.mu.RLock()
	defer h.mu.RUnlock()
	if h.maxLevel < 1 {
		return fmt.Sprintf("maxLevel %d, want >= 1 — a level-0 backfill could promote itself "+
			"to entry point and mask the orphan; raise the seed count", h.maxLevel)
	}
	if int(h.entryPoint) < len(h.tombstoned) && !h.tombstoned[h.entryPoint] {
		return fmt.Sprintf("entry point slot %d is live — the setup did not leave the traversal "+
			"rooted in the dead band", h.entryPoint)
	}
	return ""
}

// invNamed adapts *NamedCollection over a single vector space.
type invNamed struct {
	nc    *NamedCollection
	name  string
	space string
	spec  map[string]NamedVectorParams
}

func (a *invNamed) insert(id uint64, vec []float32, ttl time.Duration) error {
	return a.nc.Insert(id, map[string][]float32{a.space: vec}, nil, ttl)
}

func (a *invNamed) upsert(id uint64, vec []float32) error { return a.insert(id, vec, 0) }

func (a *invNamed) remove(id uint64) error {
	ok, err := a.nc.Delete(id)
	if err != nil {
		return err
	}
	if !ok {
		return fmt.Errorf("Delete(%d) reported the point absent", id)
	}
	return nil
}

func (a *invNamed) search(q []float32, k int) ([]uint64, error) {
	res, err := a.nc.SearchNamed(a.space, q, k, Filter{})
	if err != nil {
		return nil, err
	}
	return resultIDs(res), nil
}

func (a *invNamed) close() { _ = a.nc.Close() }

func (a *invNamed) roundTrip(t *testing.T) invIndex {
	t.Helper()
	var buf bytes.Buffer
	if err := a.nc.Snapshot(&buf); err != nil {
		t.Fatalf("named Snapshot: %v", err)
	}
	dst, err := NewNamedCollection(a.name+"_restored", a.spec)
	if err != nil {
		t.Fatalf("NewNamedCollection(restore target): %v", err)
	}
	if err := dst.Restore(bytes.NewReader(buf.Bytes())); err != nil {
		_ = dst.Close()
		t.Fatalf("named Restore: %v", err)
	}
	a.close()
	return &invNamed{nc: dst, name: a.name + "_restored", space: a.space, spec: a.spec}
}

// invMulti adapts *MultiVectorIndex. Each point is stored as a ONE-TOKEN
// document, which makes the MaxSim score of a self-query exactly the cosine of
// the token with itself (1, the maximum) and every other document's score
// strictly lower — so rank 0 carries the same meaning it does elsewhere.
type invMulti struct{ m *MultiVectorIndex }

func (a *invMulti) insert(id uint64, vec []float32, ttl time.Duration) error {
	if ttl != 0 {
		return fmt.Errorf("multivector family has no per-document TTL surface")
	}
	return a.m.Add(id, [][]float32{vec}, nil)
}

func (a *invMulti) upsert(id uint64, vec []float32) error { return a.insert(id, vec, 0) }

func (a *invMulti) remove(id uint64) error {
	if !a.m.Delete(id) {
		return fmt.Errorf("Delete(%d) reported the document absent", id)
	}
	return nil
}

func (a *invMulti) search(q []float32, k int) ([]uint64, error) {
	res, err := a.m.Search([][]float32{q}, k, MultiSearchOpts{CandidatesPerToken: 256})
	if err != nil {
		return nil, err
	}
	ids := make([]uint64, len(res))
	for i, r := range res {
		ids[i] = r.ID
	}
	return ids, nil
}

func (a *invMulti) close() { _ = a.m.Close() }

// ---------------------------------------------------------------------------
// Families
// ---------------------------------------------------------------------------

// invCap is the set of lifecycle stages a family can take part in. A family
// that lacks a capability SKIPS the stage with a visible reason rather than
// silently passing it.
type invCap uint16

const (
	capBulkBuild invCap = 1 << iota
	capReclaim
	capSnapshot
	capTTL
)

// invOpts are the creation-time knobs a stage needs to control.
type invOpts struct {
	sweepInterval time.Duration
	suppressSweep bool
}

type invFamily struct {
	name string
	dim  int
	n    int // corpus size for the matrix stages
	caps invCap
	make func(t *testing.T, o invOpts) invIndex
}

// invDenseCfg is the shared baseline: bounded well above the recall floor (see
// the APPROXIMATE RECALL caveat at the top) so that a self-query miss means a
// lost point, not a starved beam.
func invDenseCfg(dim int, metric Metric) Config {
	return Config{Dim: dim, Metric: metric, M: 16, EfConstruction: 200, EfSearch: 64, Seed: 1}
}

func invNewColl(t *testing.T, name string, cfg Config, o invOpts) invIndex {
	t.Helper()
	cfg.SweepInterval = o.sweepInterval
	cfg.SuppressSweep = o.suppressSweep
	c, err := NewCollection(name, cfg)
	if err != nil {
		t.Fatalf("NewCollection(%s): %v", name, err)
	}
	return &invColl{c: c, cfg: cfg, name: name}
}

func invFamilies() []invFamily {
	dense := func(name string, dim int, n int, tweak func(*Config)) invFamily {
		return invFamily{
			name: name, dim: dim, n: n,
			caps: capBulkBuild | capReclaim | capSnapshot | capTTL,
			make: func(t *testing.T, o invOpts) invIndex {
				cfg := invDenseCfg(dim, L2)
				if tweak != nil {
					tweak(&cfg)
				}
				return invNewColl(t, "inv_"+name, cfg, o)
			},
		}
	}
	return []invFamily{
		dense("hnsw-l2", 8, 256, nil),
		dense("hnsw-cosine", 8, 256, func(c *Config) { c.Metric = Cosine }),
		dense("hnsw-dot", 8, 256, func(c *Config) { c.Metric = DotProduct }),
		// A realistic embedding width. Smaller corpus so the matrix stays inside
		// the every-push budget; the shape coverage is identical.
		dense("hnsw-dim128", 128, 128, nil),
		// IVF. nprobe is 4 of 8 lists rather than 1: a point is assigned to its
		// NEAREST centroid and a self-query's nearest centroid is that same one,
		// so probing 1 would already suffice in theory — 4 is margin against the
		// train/retrain boundary, not a weakening (it is still a minority of the
		// lists, so the search is not secretly exhaustive).
		dense("ivf", 8, 256, func(c *Config) {
			c.IndexType = IndexIVF
			c.IVFNlist = 8
			c.IVFNprobe = 4
			c.IVFTrainThreshold = 64
		}),
		dense("vamana", 8, 256, func(c *Config) { c.IndexType = IndexVamana }),
		{
			// Named vectors: per-space sub-indexes reusing the same result
			// assembly. No Reclaim/BuildConcurrent surface on NamedCollection, and
			// its TTL expiry has no Stats().Expired hook to synchronise on, so
			// those stages are skipped explicitly.
			name: "named", dim: 8, n: 256, caps: capSnapshot,
			make: func(t *testing.T, o invOpts) invIndex {
				spec := map[string]NamedVectorParams{
					"title": {Dim: 8, Metric: L2, M: 16, EfConstruction: 200, EfSearch: 64, Seed: 1},
				}
				nc, err := NewNamedCollection("default/inv_named", spec)
				if err != nil {
					t.Fatalf("NewNamedCollection: %v", err)
				}
				return &invNamed{nc: nc, name: "default/inv_named", space: "title", spec: spec}
			},
		},
		{
			// Multi-vector. Snapshot/Restore is unexported on MultiVectorIndex
			// (it round-trips through CollectionStore), and there is no Reclaim or
			// bulk-build surface, so this family runs the in-memory lifecycle
			// stages only.
			name: "multivector", dim: 8, n: 192, caps: 0,
			make: func(t *testing.T, o invOpts) invIndex {
				m, err := NewMultiVectorIndex(MultiVectorConfig{
					Dim: 8, M: 16, EfConstruction: 200, EfSearch: 128, Seed: 1,
				})
				if err != nil {
					t.Fatalf("NewMultiVectorIndex: %v", err)
				}
				return &invMulti{m: m}
			},
		},
	}
}

// ---------------------------------------------------------------------------
// The assertions
// ---------------------------------------------------------------------------

// assertSelfSearchable IS the invariant: for every live id, a query with that
// point's own stored vector returns it at RANK 0.
//
// Failures are capped at invMaxReport so that a whole-index loss prints a
// diagnosis rather than a wall of text, but the COUNT is always exact.
const invMaxReport = 5

func assertSelfSearchable(t *testing.T, idx invIndex, c *invCorpus, live []uint64, what string) {
	t.Helper()
	var missing, misranked int
	for _, id := range live {
		got, err := idx.search(c.vec[id], invSearchK)
		if err != nil {
			t.Fatalf("%s: self-query for id %d: %v", what, id, err)
		}
		switch rank := invRankOf(got, id); {
		case rank < 0:
			missing++
			if missing <= invMaxReport {
				t.Errorf("%s: id %d is live but its OWN vector does not return it — got %v",
					what, id, got)
			}
		case rank > 0:
			misranked++
			if misranked <= invMaxReport {
				t.Errorf("%s: id %d returned at rank %d, want rank 0 (its own vector is at "+
					"distance 0, the global minimum) — got %v", what, id, rank, got)
			}
		}
	}
	if missing+misranked > invMaxReport {
		t.Errorf("%s: %d/%d live points fail the searchability invariant (%d absent, %d "+
			"misranked); only the first %d of each are printed",
			what, missing+misranked, len(live), missing, misranked, invMaxReport)
	}
}

// assertSelfInTopK is the WEAKER form, used only where the index scores on
// lossy codes and the arg-min is therefore not owed to the point itself. It is
// never used for an exact configuration.
func assertSelfInTopK(t *testing.T, idx invIndex, c *invCorpus, live []uint64, k int, what string) {
	t.Helper()
	var missing int
	for _, id := range live {
		got, err := idx.search(c.vec[id], k)
		if err != nil {
			t.Fatalf("%s: self-query for id %d: %v", what, id, err)
		}
		if invRankOf(got, id) < 0 {
			missing++
			if missing <= invMaxReport {
				t.Errorf("%s: id %d is live but absent from the top-%d of its OWN vector — got %v",
					what, id, k, got)
			}
		}
	}
	if missing > invMaxReport {
		t.Errorf("%s: %d/%d live points absent from their own top-%d; only the first %d printed",
			what, missing, len(live), k, invMaxReport)
	}
}

// assertNotSearchable is the complement, and it is what stops this file from
// being gameable. Without it, "return everything from everywhere" — deleting
// the liveness gate — would satisfy every assertion above. Dead points (deleted
// or TTL-expired) must NOT come back from their own vector.
func assertNotSearchable(t *testing.T, idx invIndex, c *invCorpus, dead []uint64, what string) {
	t.Helper()
	var leaked int
	for _, id := range dead {
		got, err := idx.search(c.vec[id], invSearchK)
		if err != nil {
			t.Fatalf("%s: self-query for dead id %d: %v", what, id, err)
		}
		if invRankOf(got, id) >= 0 {
			leaked++
			if leaked <= invMaxReport {
				t.Errorf("%s: id %d is DEAD but its own vector still returns it — got %v",
					what, id, got)
			}
		}
	}
	if leaked > invMaxReport {
		t.Errorf("%s: %d/%d dead points still searchable; only the first %d printed",
			what, leaked, len(dead), invMaxReport)
	}
}

func invRankOf(got []uint64, id uint64) int {
	for i, g := range got {
		if g == id {
			return i
		}
	}
	return -1
}

// ---------------------------------------------------------------------------
// Lifecycle stages
// ---------------------------------------------------------------------------

// invStage is one lifecycle state to assert the invariant in. run builds and
// mutates an index and returns it together with the corpus the assertions must
// query by and the ids that MUST be findable; dead is the complement that must
// NOT be findable.
type invStage struct {
	name  string
	needs invCap
	run   func(t *testing.T, f invFamily) (idx invIndex, c *invCorpus, live, dead []uint64)
}

func invSeedAll(t *testing.T, idx invIndex, c *invCorpus) {
	t.Helper()
	for _, id := range c.ids {
		if err := idx.insert(id, c.vec[id], 0); err != nil {
			t.Fatalf("insert id %d: %v", id, err)
		}
	}
}

func invStages() []invStage {
	return []invStage{
		{
			// The floor: incremental inserts, shuffled so id N is not on slot N.
			name: "fresh-insert",
			run: func(t *testing.T, f invFamily) (invIndex, *invCorpus, []uint64, []uint64) {
				idx := f.make(t, invOpts{})
				c := newInvCorpus(t, invSeedIDs(f.n, 11), f.dim, 1)
				invSeedAll(t, idx, c)
				return idx, c, c.ids, nil
			},
		},
		{
			// The bulk path builds the graph with a different linker than the
			// incremental one, so it gets its own arm.
			name:  "bulk-build",
			needs: capBulkBuild,
			run: func(t *testing.T, f invFamily) (invIndex, *invCorpus, []uint64, []uint64) {
				idx := f.make(t, invOpts{})
				c := newInvCorpus(t, invSeedIDs(f.n, 12), f.dim, 1)
				vecs := make([][]float32, len(c.ids))
				for i, id := range c.ids {
					vecs[i] = c.vec[id]
				}
				if err := idx.(invBulkBuilder).bulkBuild(c.ids, vecs); err != nil {
					t.Fatalf("bulk build: %v", err)
				}
				return idx, c, c.ids, nil
			},
		},
		{
			// The PAYLOAD-BEARING bulk path. Same concurrent linker as the arm
			// above, but the placement pass additionally fills the arena's metadata,
			// the payload index and the BM25 corpus before the graph exists — and
			// the searchability question is the same one: is a point loaded this way
			// still returned by its own vector? Payloads on every point but every
			// fifth, so the mixed case (a payload-less point inside a
			// payload-bearing load) is inside the sweep rather than beside it.
			//
			// The FILTERED half of this — a point findable by its own vector under a
			// filter that selects exactly its own payload — lives in
			// bulk_payload_equivalence_test.go, which has the filter surface this
			// harness deliberately does not.
			name:  "bulk-build-payloads",
			needs: capBulkBuild,
			run: func(t *testing.T, f invFamily) (invIndex, *invCorpus, []uint64, []uint64) {
				idx := f.make(t, invOpts{})
				b, ok := idx.(invPayloadBulkBuilder)
				if !ok {
					t.Skipf("family %q declares capBulkBuild but has no payload-bearing bulk surface", f.name)
				}
				c := newInvCorpus(t, invSeedIDs(f.n, 18), f.dim, 1)
				vecs := make([][]float32, len(c.ids))
				metas := make([]Metadata, len(c.ids))
				for i, id := range c.ids {
					vecs[i] = c.vec[id]
					if id%5 == 0 {
						continue
					}
					metas[i] = Metadata{
						"id":     NewInt(int64(id)), //nolint:gosec // small test ids
						"bucket": NewString(fmt.Sprintf("b%d", id%11)),
					}
				}
				if err := b.bulkBuildPayloads(c.ids, vecs, metas); err != nil {
					t.Fatalf("payload-bearing bulk build: %v", err)
				}
				return idx, c, c.ids, nil
			},
		},
		{
			// Tombstoned but not reclaimed: the dead slots are still graph
			// vertices, still traversed, and still carry edges. Survivors must
			// stay findable THROUGH them.
			name: "delete-subset-tombstoned",
			run: func(t *testing.T, f invFamily) (invIndex, *invCorpus, []uint64, []uint64) {
				idx := f.make(t, invOpts{})
				c := newInvCorpus(t, invSeedIDs(f.n, 13), f.dim, 1)
				invSeedAll(t, idx, c)
				live, dead := invSplitEveryNth(c.ids, 3)
				for _, id := range dead {
					if err := idx.remove(id); err != nil {
						t.Fatalf("delete id %d: %v", id, err)
					}
				}
				return idx, c, live, dead
			},
		},
		{
			// Reclaim physically removes the dead vertices, prunes the edges into
			// them and frees their slots to the LIFO free list — the step that
			// rewires the graph most aggressively.
			name:  "delete-subset-reclaimed",
			needs: capReclaim,
			run: func(t *testing.T, f invFamily) (invIndex, *invCorpus, []uint64, []uint64) {
				idx := f.make(t, invOpts{})
				c := newInvCorpus(t, invSeedIDs(f.n, 14), f.dim, 1)
				invSeedAll(t, idx, c)
				live, dead := invSplitEveryNth(c.ids, 3)
				for _, id := range dead {
					if err := idx.remove(id); err != nil {
						t.Fatalf("delete id %d: %v", id, err)
					}
				}
				if got := idx.(invReclaimer).reclaim(); got != len(dead) {
					t.Fatalf("Reclaim removed %d, want %d — the stage did not reach the state "+
						"it claims to test", got, len(dead))
				}
				return idx, c, live, dead
			},
		},
		{
			// SLOT REUSE (bug 3's territory). An upsert is delete-then-insert and
			// the insert lands back on the slot the deleted point held, so every
			// other node's in-edges into that slot survive it. Upserting the whole
			// corpus puts every slot through that window. The corpus SALT changes,
			// so a stale vector left behind would fail the assertion.
			name: "upsert-slot-reuse",
			run: func(t *testing.T, f invFamily) (invIndex, *invCorpus, []uint64, []uint64) {
				idx := f.make(t, invOpts{})
				ids := invSeedIDs(f.n, 15)
				before := newInvCorpus(t, ids, f.dim, 1)
				invSeedAll(t, idx, before)
				after := newInvCorpus(t, ids, f.dim, 2)
				for _, id := range ids {
					if err := idx.upsert(id, after.vec[id]); err != nil {
						t.Fatalf("upsert id %d: %v", id, err)
					}
				}
				return idx, after, ids, nil
			},
		},
		{
			// BUG 2's SHAPE: tombstone the ENTIRE index without reclaiming, then
			// backfill. Every candidate a backfilled node's link search can reach
			// is dead, which is the exact precondition under which linkNode used to
			// write an empty edge set and orphan the point forever.
			//
			// The seed count is load-bearing: a taller graph means a backfilled
			// node drawing level 0 cannot promote ITSELF to entry point and be
			// repaired by the inserts after it. backfillPrecondition asserts that
			// rather than trusting it.
			name: "delete-all-then-backfill",
			run: func(t *testing.T, f invFamily) (invIndex, *invCorpus, []uint64, []uint64) {
				idx := f.make(t, invOpts{})
				const seeded, backfill = 64, 6
				ids := make([]uint64, 0, seeded+backfill)
				for i := 0; i < seeded; i++ {
					ids = append(ids, uint64(i)*7)
				}
				fill := make([]uint64, 0, backfill)
				for i := 0; i < backfill; i++ {
					fill = append(fill, 100_000+uint64(i))
				}
				c := newInvCorpus(t, append(ids, fill...), f.dim, 3)

				for _, id := range ids {
					if err := idx.insert(id, c.vec[id], 0); err != nil {
						t.Fatalf("seed id %d: %v", id, err)
					}
				}
				for _, id := range ids {
					if err := idx.remove(id); err != nil {
						t.Fatalf("tombstone id %d: %v", id, err)
					}
				}
				if p, ok := idx.(invBackfillProbe); ok {
					if why := p.backfillPrecondition(); why != "" {
						t.Fatalf("delete-all-then-backfill precondition not met: %s", why)
					}
				}
				for _, id := range fill {
					if err := idx.insert(id, c.vec[id], 0); err != nil {
						t.Fatalf("backfill id %d: %v", id, err)
					}
				}
				return idx, c, fill, ids
			},
		},
		{
			// Snapshot then restore into a fresh index. The graph is rebuilt or
			// remapped on the far side, so slot assignment and edge encoding both
			// get re-derived.
			name:  "snapshot-restore",
			needs: capSnapshot,
			run: func(t *testing.T, f invFamily) (invIndex, *invCorpus, []uint64, []uint64) {
				idx := f.make(t, invOpts{})
				c := newInvCorpus(t, invSeedIDs(f.n, 16), f.dim, 1)
				invSeedAll(t, idx, c)
				return idx.(invRoundTripper).roundTrip(t), c, c.ids, nil
			},
		},
		{
			// TTL expiry with the background sweeper RUNNING, so expired points
			// are physically tombstoned out from under concurrent searches. The
			// clock is injected, so expiry is deterministic; only the sweeper's
			// own ticker is waited on.
			name:  "ttl-expiry-sweeper-on",
			needs: capTTL,
			run: func(t *testing.T, f invFamily) (invIndex, *invCorpus, []uint64, []uint64) {
				return invTTLStage(t, f, invOpts{sweepInterval: 10 * time.Millisecond}, true)
			},
		},
		{
			// TTL expiry with the sweeper SUPPRESSED (the replicated-cluster
			// policy): nothing is physically removed and expired points are
			// filtered lazily at read time instead. Survivors must still be
			// findable through a graph full of expired-but-present vertices.
			name:  "ttl-expiry-sweeper-off",
			needs: capTTL,
			run: func(t *testing.T, f invFamily) (invIndex, *invCorpus, []uint64, []uint64) {
				return invTTLStage(t, f, invOpts{suppressSweep: true}, false)
			},
		},
	}
}

// invSplitEveryNth partitions ids into keep/drop, dropping every nth.
func invSplitEveryNth(ids []uint64, n int) (keep, drop []uint64) {
	for i, id := range ids {
		if i%n == 0 {
			drop = append(drop, id)
		} else {
			keep = append(keep, id)
		}
	}
	return keep, drop
}

// invClock is a test-driven millisecond clock. It is atomic because the
// background sweeper goroutine reads it concurrently in the sweeper-on arm.
type invClock struct{ ms atomic.Int64 }

func (c *invClock) now() int64      { return c.ms.Load() }
func (c *invClock) advance(d int64) { c.ms.Add(d) }

func invTTLStage(t *testing.T, f invFamily, o invOpts, waitForSweep bool) (invIndex, *invCorpus, []uint64, []uint64) {
	t.Helper()
	idx := f.make(t, o)
	clk := &invClock{}
	clk.ms.Store(1_000_000)
	idx.(invClocked).setNow(clk.now)

	c := newInvCorpus(t, invSeedIDs(f.n, 17), f.dim, 1)
	live, expiring := invSplitEveryNth(c.ids, 3)
	exp := make(map[uint64]bool, len(expiring))
	for _, id := range expiring {
		exp[id] = true
	}
	for _, id := range c.ids {
		var ttl time.Duration
		if exp[id] {
			ttl = 50 * time.Millisecond
		}
		if err := idx.insert(id, c.vec[id], ttl); err != nil {
			t.Fatalf("insert id %d: %v", id, err)
		}
	}
	clk.advance(1000)

	if waitForSweep {
		deadline := time.Now().Add(10 * time.Second)
		for idx.(invClocked).expiredCount() < uint64(len(expiring)) {

			if time.Now().After(deadline) {
				t.Fatalf("sweeper reported %d expiries, want %d — the stage never reached the "+
					"state it claims to test", idx.(invClocked).expiredCount(), len(expiring))
			}
			time.Sleep(2 * time.Millisecond)
		}
	}
	return idx, c, live, expiring
}

// ---------------------------------------------------------------------------
// The matrix
// ---------------------------------------------------------------------------

// TestSearchabilityInvariant is the standing gate: every index family, in every
// lifecycle state, must return every live point when queried with that point's
// own vector.
func TestSearchabilityInvariant(t *testing.T) {
	for _, f := range invFamilies() {
		for _, st := range invStages() {
			t.Run(f.name+"/"+st.name, func(t *testing.T) {
				if f.caps&st.needs != st.needs {
					t.Skipf("family %q has no %s surface (see invFamilies for why)", f.name, st.name)
				}
				idx, c, live, dead := st.run(t, f)
				defer idx.close()
				if len(live) == 0 {
					t.Fatal("stage produced no live points — it would assert nothing")
				}
				assertSelfSearchable(t, idx, c, live, f.name+"/"+st.name)
				if len(dead) > 0 {
					assertNotSearchable(t, idx, c, dead, f.name+"/"+st.name)
				}
			})
		}
	}
}

// ---------------------------------------------------------------------------
// Id patterns
// ---------------------------------------------------------------------------

// TestSearchabilityInvariantIDPatterns sweeps the id space itself. Bug 1 lived
// entirely here: the sentinel keyed off the VALUE of the id, so it was
// invisible to any corpus that happened not to contain a zero. The patterns
// below pin the boundaries of the uint64 id space and, crucially, the
// relationship between an id and the SLOT it lands on — the arena assigns slots
// in insertion order, so shuffling the insertion breaks id==slot and separates
// an id-keyed bug from a slot-keyed one.
func TestSearchabilityInvariantIDPatterns(t *testing.T) {
	families := []struct {
		name  string
		tweak func(*Config)
	}{
		{"hnsw", nil},
		{"ivf", func(c *Config) {
			c.IndexType = IndexIVF
			c.IVFNlist = 8
			c.IVFNprobe = 4
			c.IVFTrainThreshold = 32
		}},
		{"vamana", func(c *Config) { c.IndexType = IndexVamana }},
	}

	const dim = 8
	max := uint64(math.MaxUint64)
	patterns := []struct {
		name string
		ids  []uint64
	}{
		// The single-point index whose only point is id 0. Pre-fix this answered
		// its own vector with an empty result set.
		{"id0-alone", []uint64{0}},
		// id 0 first, so it also lands on slot 0.
		{"id0-at-slot0", invContiguous(0, 64)},
		// id 0 LAST, so it lands on the final slot: separates "id 0 is dropped"
		// from "slot 0 is dropped".
		{"id0-at-last-slot", append(invContiguous(1, 63), 0)},
		// A non-zero id at slot 0 — the counter-experiment.
		{"nonzero-id-at-slot0", invContiguous(1, 64)},
		// The top of the id space, next to the bottom of it.
		{"boundaries", []uint64{0, 1, 2, max, max - 1, max / 2, max/2 + 1}},
		// Sparse, wildly non-contiguous ids spanning every order of magnitude.
		{"sparse", []uint64{0, 1, 7, 63, 1023, 1 << 20, 1 << 40, 1 << 62, max}},
		// A large contiguous range inserted in shuffled order, so id N is
		// systematically NOT on slot N.
		{"contiguous-shuffled", invShuffled(invContiguous(0, 512), 21)},
	}

	for _, fam := range families {
		for _, p := range patterns {
			t.Run(fam.name+"/"+p.name, func(t *testing.T) {
				cfg := invDenseCfg(dim, L2)
				if fam.tweak != nil {
					fam.tweak(&cfg)
				}
				idx := invNewColl(t, "inv_ids_"+fam.name+"_"+p.name, cfg, invOpts{})
				defer idx.close()
				c := newInvCorpus(t, p.ids, dim, 5)
				invSeedAll(t, idx, c)
				assertSelfSearchable(t, idx, c, c.ids, fam.name+"/"+p.name)
			})
		}
	}
}

func invContiguous(lo uint64, n int) []uint64 {
	out := make([]uint64, n)
	for i := range out {
		out[i] = lo + uint64(i)
	}
	return out
}

func invShuffled(ids []uint64, seed int64) []uint64 {
	out := append([]uint64(nil), ids...)
	r := rand.New(rand.NewSource(seed)) //nolint:gosec // deterministic test fixture
	r.Shuffle(len(out), func(i, j int) { out[i], out[j] = out[j], out[i] })
	return out
}

// ---------------------------------------------------------------------------
// Config sweep
// ---------------------------------------------------------------------------

// TestSearchabilityInvariantConfigs sweeps the graph and metric knobs across
// all three metrics and three widths, in two phases: FRESH INSERT and AFTER AN
// UPSERT SWEEP.
//
// THE FLOORS BELOW ARE A MEASURED FINDING, NOT A CONVENIENCE. Writing this test
// turned up configurations in which the invariant genuinely does not hold on
// FIXED code, and the brief for this file says to bound such a config and say
// so rather than quietly loosen the assertion. The finding is one coherent
// thing: AT DEGENERATE BUILD PARAMETERS THE CONSTRUCTION ITSELF LEAVES POINTS
// UNREACHABLE FROM THE ENTRY POINT, and churn widens the window.
//
// Measured, L2, ids counted by unreachableLivePoints (the same helper the
// orphan and reused-slot regressions use):
//
//	(a) BUILD. dim 128, n=96, M=4, EfConstruction=16 — one point is unreachable
//	    straight out of a fresh insert. Raising EfSearch to 64, 256 or 1024
//	    changes nothing (still 1 miss, still 1 unreachable): no width of beam
//	    can find a node with no path to it. Raising EfConstruction to 32, or M
//	    to 6, takes it to zero. So the floor belongs on the BUILD parameters.
//
//	(b) CHURN — FIXED, see TestSearchabilityInvariantUpsertSweepDegrees. This
//	    entry is kept because the numbers are the before/after of the fix.
//	    dim 8, n=192, EfConstruction=200, EfSearch=256, three seeds:
//
//	      M     fresh insert     after a full upsert sweep
//	      4     0 unreachable    2-3 unreachable (and unfindable)   -> 0
//	      5     0 unreachable    0-1 unreachable                    -> 0
//	      6     0 unreachable    0-1 unreachable                    -> 0
//	      7     0 unreachable    0                                  -> 0
//	      8+    0 unreachable    0                                  -> 0
//
//	    Again unaffected by EfSearch, and again absent on fresh insert at the
//	    same M — it was the upsert sweep that produced it, through three
//	    reused-slot defects in the link path (self-loops, one-way upper-level
//	    edges, and the forward write discarding a slot's inherited edges). The
//	    right-hand column is this same measurement on fixed code.
//
// M=4 and EfConstruction=16 are both fully legal (Validate accepts M in 1..128
// and any positive ef), so this is worth its own investigation: it is the same
// failure SHAPE as the empty-candidate orphan and the reused-slot dead end that
// this file exists to guard, surviving at degenerate parameters. It is recorded
// here rather than absorbed, because absorbing it would mean weakening the
// assertion for every configuration to accommodate the worst one.
//
// The sweep therefore keeps the smallest legal DEGREE (M=4) and the narrowest
// SEARCH width (EfSearch=16) — the parts that make it a real low-end gate — and
// floors only what was measured to break: EfConstruction at buildWidthFloor for
// every phase, and M at churnDegreeFloor for the upsert phase.
func TestSearchabilityInvariantConfigs(t *testing.T) {
	// buildWidthFloor is the smallest EfConstruction at which a fresh insert is
	// known to leave no orphan at M=4 (measured in (a): efC=16 orphans, 32 does
	// not).
	const buildWidthFloor = 32
	// churnDegreeFloor is the smallest graph degree at which a full upsert sweep
	// is known to preserve reachability. It was 8 — measured in (b), where M=7 was
	// clean across seeds and M=6 was not — and is now 4, the smallest degree the
	// graphs table below exercises, because the churn defect behind that floor is
	// fixed. The permanent M-sweep that holds it there is
	// TestSearchabilityInvariantUpsertSweepDegrees; this constant remains so that
	// the phase is skipped LOUDLY rather than silently if a degree below the
	// measured floor is ever added to the table above.
	const churnDegreeFloor = 4

	graphs := []struct {
		name        string
		m, efC, efS int
	}{
		// Smallest legal degree and narrowest search width, at the measured
		// build-width floor.
		{"small-m4-ef16", 4, buildWidthFloor, 16},
		{"low-m8-ef32", 8, 64, 32},
		{"default-m16-ef64", 16, 200, 64},
		{"large-m48-ef256", 48, 400, 256},
	}
	metrics := []struct {
		name string
		m    Metric
	}{
		{"l2", L2}, {"cosine", Cosine}, {"dot", DotProduct},
	}
	dims := []int{4, 8, 128}

	for _, g := range graphs {
		for _, mt := range metrics {
			for _, dim := range dims {
				name := fmt.Sprintf("%s/%s/dim%d", g.name, mt.name, dim)
				t.Run(name, func(t *testing.T) {
					n := 192
					if dim >= 128 {
						n = 96
					}
					cfg := Config{
						Dim: dim, Metric: mt.m, M: g.m,
						EfConstruction: g.efC, EfSearch: g.efS, Seed: 1,
					}
					idx := invNewColl(t, "inv_cfg", cfg, invOpts{})
					defer idx.close()
					c := newInvCorpus(t, invSeedIDs(n, 31), dim, 7)
					invSeedAll(t, idx, c)
					assertSelfSearchable(t, idx, c, c.ids, name)

					// The same config after a churn pass, since slot reuse is the
					// state where a config-dependent linking bug would show. Below
					// the degree floor this phase is a KNOWN failure on fixed code
					// (see the table above), so it is skipped loudly rather than
					// deleted or asserted more weakly.
					if g.m < churnDegreeFloor {
						t.Logf("skipping the upsert phase at M=%d: an upsert sweep is known to "+
							"orphan points below M=%d on fixed code (see the measured table on "+
							"TestSearchabilityInvariantConfigs). The fresh-insert phase above "+
							"still asserts this configuration in full.", g.m, churnDegreeFloor)
						return
					}
					after := newInvCorpus(t, c.ids, dim, 8)
					for _, id := range c.ids {
						if err := idx.upsert(id, after.vec[id]); err != nil {
							t.Fatalf("upsert id %d: %v", id, err)
						}
					}
					assertSelfSearchable(t, idx, after, after.ids, name+"/after-upsert")
				})
			}
		}
	}
}

// invSelfLoops returns the level-0 self edges in the graph — a node listed as
// its own neighbour. There is never a legitimate one: it is not a connection, it
// spends an edge slot, and because it sits at distance 0 it sorts first in every
// prune and can therefore never be evicted. Zero is the invariant.
func invSelfLoops(t *testing.T, idx invIndex) int {
	t.Helper()
	a, ok := idx.(*invColl)
	if !ok {
		return 0
	}
	h, ok := a.c.idx.(*hnsw)
	if !ok {
		return 0
	}
	n := 0
	for slot, nd := range h.nodes {
		if nd == nil {
			continue
		}
		for _, nb := range h.nbrsAt(nd, 0) {
			if nb == uint32(slot) {
				n++
			}
		}
	}
	return n
}

// TestSearchabilityInvariantUpsertSweepDegrees is the STANDING M-SWEEP GATE on
// slot reuse: a full upsert sweep over an existing corpus, repeated, across
// every graph degree from the smallest one the suite claims to support upward.
//
// It exists because the churn row of TestSearchabilityInvariantConfigs used to
// be a documented FAILURE — "an upsert sweep orphans points below M=8 on fixed
// code" — and that phase was skipped for M<8 rather than asserted. This test is
// what let the floor come down to M=4, so it has to be the thing that keeps it
// down. Three defects produced that failure, all of them specific to an upsert
// landing on the slot the deleted point held, and all three are pinned here:
//
//  1. SELF-LOOPS. A reused slot keeps its predecessor's level-0 edges
//     so it stays traversable during the link window — which means the link
//     traversal can walk back into the very node it is linking. It is live and
//     at distance exactly 0, so it was admitted as the best candidate, picked
//     first by the heuristic, and then handed its own back-edge. Every upserted
//     node ended with exactly two, at every M: 368-384 per 192-point sweep
//     against 0 on fresh insert. They are unprunable (distance 0 sorts first),
//     so they permanently cost 2 of the node's m0=2M level-0 slots — a quarter
//     of the budget at M=4. Fixed by linkLayer; pinned by the selfLoops == 0
//     assertion below, which is exact and needs no threshold.
//
//  2. ONE-WAY UPPER-LEVEL EDGES. An upsert redraws the node's level, so
//     in-edges into the slot at levels above the new level become permanent dead
//     ends — and, worse, the link traversal then ADMITTED those nodes at levels
//     they do not carry and wrote fresh edges to them, whose back-edge half the
//     linker skips. The population was self-sustaining: 58.3% of all upper-level
//     edges were dead after one 1000-point sweep, and the total upper-edge count
//     had collapsed from 1392 to 854. linkLayer's `nd.level >= level` gate stops
//     them being created (58.3% -> 18.8%, 854 -> 1246 edges).
//
//  3. THE FORWARD WRITE DISCARDING THE SLOT'S INHERITED EDGES. This is the one
//     that dominated the count. The level-0 forward write was the only place in
//     the index that destroys a node's out-edges en bloc; every other write
//     appends and lets the heuristic prune. A sweep therefore demolished the
//     whole level-0 edge population exactly once, leaving each point only the
//     back-edges it happened to be granted. linkNode now merges them through the
//     same append-and-prune path as a raced-in back-edge.
//
// Measured together, dim 8, L2, EfConstruction=200, three seeds x three
// successive sweeps, range of points unreachable from the entry point (an exact
// walk, so these numbers have no beam in them), before the fix -> after:
//
//	M   n=192               n=1000
//	2   33-65  -> 0-2       133-205 -> 8-13
//	3   6-17   -> 0         23-44   -> 0-2
//	4   1-5    -> 0         5-15    -> 0
//	5   0-4    -> 0         0-5     -> 0
//	6   0-1    -> 0         0-4     -> 0
//
// THE BEAM WIDTH IS PER CORPUS SIZE, AND THAT IS A MEASUREMENT, NOT A DODGE.
// The n=1000 arm searches at EfSearch=1024 rather than 256. At M=4 the graph is
// WHOLE after every sweep (assertGraphWhole, which is exact and has no beam in
// it) and yet 1-2 self-queries per 1000 still miss at EfSearch=256 — and ALL of
// them come back at EfSearch=1024, and again at 4096. That is the distinction
// this whole file is built on, run in the other direction: the original defect
// was explicitly NOT beam-recoverable (identical orphan counts at EfSearch 16,
// 64 and 256, because there was no path at any width), and this residual is
// nothing but ordinary approximation at a degenerate out-degree. The rank-0
// assertion itself is never weakened; only the width is set to one that a
// degree-4 graph over 1000 points can actually answer at.
//
// WHAT IS STILL BOUNDED, and why it is not absorbed. M<=3 is not swept: at M=2
// the fresh BUILD already orphans points (4-8 per 1000 with no churn at all), so
// it is the degenerate-build-parameter finding (a) on
// TestSearchabilityInvariantConfigs, not a churn defect. The same is true at the
// top end of the corpus size: at M=4, dim 32, n=4000 a fresh build leaves 48
// points unreachable and the sweep then leaves 26 — churn no longer AMPLIFIES
// the floor (it was 48 -> 258 before), it now sits below it. The residual that
// is genuinely churn's is a statistical tail: 1 point in 4000 at M=8 (from 27),
// a node with a full out-degree and an in-degree that later prunes took to zero.
// Closing that needs a minimum-in-degree guarantee, i.e. a maintained per-slot
// in-edge counter on the hot build path; it is not built here, and the measured
// size of what it would buy is that one number.
func TestSearchabilityInvariantUpsertSweepDegrees(t *testing.T) {
	const dim, sweeps = 8, 3
	degrees := []int{4, 5, 6, 7, 8, 16}
	sizes := []struct{ n, efS int }{{192, 256}}
	if !testing.Short() {
		// The 1000-point arm is where the defect was loudest and where the fix has
		// the most headroom to regress; -short (the -race lane) keeps the 192-point
		// arm, which reproduces every one of the three mechanisms.
		sizes = append(sizes, struct{ n, efS int }{1000, 1024})
	}
	for _, m := range degrees {
		for _, sz := range sizes {
			for _, seed := range []int64{1, 2, 3} {
				n, efS := sz.n, sz.efS
				name := fmt.Sprintf("m%d/n%d/seed%d", m, n, seed)
				t.Run(name, func(t *testing.T) {
					cfg := Config{
						Dim: dim, Metric: L2, M: m,
						EfConstruction: 200, EfSearch: efS, Seed: 1,
					}
					idx := invNewColl(t, "inv_sweep_deg", cfg, invOpts{})
					defer idx.close()
					c := newInvCorpus(t, invSeedIDs(n, seed), dim, 7)
					invSeedAll(t, idx, c)
					assertSelfSearchable(t, idx, c, c.ids, name+"/fresh")
					assertGraphWhole(t, idx, name+"/fresh")

					// Repeated sweeps, not one: every defect above compounds across
					// sweeps (the upper-level edge population degrades, the level-0
					// population is demolished once per sweep), so a single pass can
					// pass on luck where three do not.
					for pass := 1; pass <= sweeps; pass++ {
						after := newInvCorpus(t, c.ids, dim, uint64(7+pass))
						for _, id := range c.ids {
							if err := idx.upsert(id, after.vec[id]); err != nil {
								t.Fatalf("sweep %d: upsert id %d: %v", pass, id, err)
							}
						}
						what := fmt.Sprintf("%s/sweep%d", name, pass)
						assertGraphWhole(t, idx, what)
						assertSelfSearchable(t, idx, after, after.ids, what)
						if loops := invSelfLoops(t, idx); loops != 0 {
							t.Errorf("%s: %d level-0 SELF-LOOP(s) — a node listed as its own "+
								"neighbour. It is not a connection, it spends an edge slot, and at "+
								"distance 0 it sorts first in every prune so it is never evicted. "+
								"See linkLayer.", what, loops)
						}
					}
				})
			}
		}
	}
}

// ---------------------------------------------------------------------------
// Quantized configurations
// ---------------------------------------------------------------------------

// TestSearchabilityInvariantQuantized is the DELIBERATELY WEAKER arm.
//
// PQ/SQ/PRQ score candidates on lossy codes, so the arg-min of the coded
// distance is genuinely not owed to the point itself: a self-query can rank the
// point second or third without anything being wrong. Demanding rank 0 here
// would be false, and — much worse — the temptation to fix that by weakening
// the assertion for EVERY config is precisely how a test of this kind gets
// blunted. So quantized configs live here, assert TOP-K MEMBERSHIP with an
// explicit k, and never touch the exact matrix.
//
// Membership is still a real gate: it is the difference between "the codes
// re-ordered the neighbourhood" (fine) and "the point is not in the index at
// all" (the bug class this file exists for).
func TestSearchabilityInvariantQuantized(t *testing.T) {
	const n = 512
	cases := []struct {
		name string
		cfg  Config
	}{
		{"sq8-cosine", Config{Dim: 32, Metric: Cosine, M: 16, EfConstruction: 200, EfSearch: 64,
			Seed: 1, Quant: QuantSQ8, RescoreFactor: 3}},
		{"sq-trained-8bit", Config{Dim: 32, Metric: L2, M: 16, EfConstruction: 200, EfSearch: 64,
			Seed: 1, Quant: QuantSQ, SQBits: 8, RescoreFactor: 3}},
		{"pq-8bit", Config{Dim: 32, Metric: L2, M: 16, EfConstruction: 200, EfSearch: 64,
			Seed: 1, Quant: QuantPQ, QuantPQM: 8, RescoreFactor: 3}},
		{"prq-2layer", Config{Dim: 32, Metric: L2, M: 16, EfConstruction: 200, EfSearch: 64,
			Seed: 1, Quant: QuantPRQ, QuantPQM: 8, PRQLayers: 2, RescoreFactor: 3}},
	}

	for _, tc := range cases {

		t.Run(tc.name, func(t *testing.T) {
			idx := invNewColl(t, "inv_quant_"+tc.name, tc.cfg, invOpts{})
			defer idx.close()
			c := newInvCorpus(t, invSeedIDs(n, 41), tc.cfg.Dim, 9)
			// Bulk-build so the quantizer actually trains; an untrained quantizer
			// falls back to exact float and the arm would be testing nothing.
			vecs := make([][]float32, len(c.ids))
			for i, id := range c.ids {
				vecs[i] = c.vec[id]
			}
			if err := idx.(invBulkBuilder).bulkBuild(c.ids, vecs); err != nil {
				t.Fatalf("bulk build: %v", err)
			}
			assertSelfInTopK(t, idx, c, c.ids, invQuantTopK, tc.name)

			// And after a churn pass, which is where slot reuse meets code
			// re-encoding.
			after := newInvCorpus(t, c.ids, tc.cfg.Dim, 10)
			for _, id := range c.ids {
				if err := idx.upsert(id, after.vec[id]); err != nil {
					t.Fatalf("upsert id %d: %v", id, err)
				}
			}
			assertSelfInTopK(t, idx, after, after.ids, invQuantTopK, tc.name+"/after-upsert")
		})
	}
}

// ---------------------------------------------------------------------------
// Churn and concurrency
// ---------------------------------------------------------------------------

// invLineCorpus builds a LINE-SHAPED corpus: the points lie along one axis with
// a small orthogonal jitter, so the graph they induce is chain-like.
//
// The shape is the point. On a line, a contiguous band of traversal dead ends
// CUTS the graph in two — the searches that build the graph stop being able to
// cross the band, so the halves link only within themselves and points end up
// present, byte-correct, and unreachable by vector search. That is bug 3's
// failure mode, and a rotationally symmetric blob does not reproduce it: there
// is always a way around.
//
// L2 only (the vectors are deliberately not unit-norm — normalizing would
// collapse the line onto a point), which is why the churn arms fix the metric.
func invLineCorpus(t *testing.T, n, dim int, salt uint64) *invCorpus {
	t.Helper()
	ids := make([]uint64, n)
	c := &invCorpus{dim: dim, ids: ids, vec: make(map[uint64][]float32, n)}
	seen := make(map[string]uint64, n)
	for i := 0; i < n; i++ {
		id := uint64(i)
		ids[i] = id
		v := make([]float32, dim)
		v[0] = float32(i)
		s := invMix(id ^ invMix(salt))
		for j := 1; j < dim; j++ {
			s = invMix(s)
			v[j] = float32(int64(s>>40)-(1<<23)) / float32(1<<23) * 0.25
		}
		key := invVecKey(v)
		if other, dup := seen[key]; dup {
			t.Fatalf("line corpus ids %d and %d collided", other, id)
		}
		seen[key] = id
		c.vec[id] = v
	}
	return c
}

// TestSearchabilityInvariantDuringUpsertWindow asserts the invariant AT THE ONE
// INSTANT IT IS HARDEST TO HOLD, and it is the arm that covers bug 3.
//
// An upsert is delete-then-insert onto the SAME slot: placeLockedAt's reclaim
// branch frees it and the arena's LIFO free list hands it straight back. Under
// Option B the new point is not linked when placement returns, so there is a
// window in which the point is fully STORED — in the arena, live, returned by
// Get — while its own forward edges have not been written yet. Every other
// node's in-edges into that slot survived the upsert and lead there.
//
// That window is where bug 3 lived, and it is invisible to any after-the-fact
// assertion: by the time the upsert returns, the node has written its own
// correct edges and the damage has healed. TestSearchabilityInvariantUnderChurn
// below hammers the same code path with concurrency and does NOT detect the
// reverted fix (measured: 0 detections in 10 runs) precisely because it looks
// only at the end state. So this arm looks INSIDE the window, through the
// linkGapHook seam, and asks the same question the rest of the file asks:
//
//	the point is live — Get returns it — so does searching for its own vector?
//
// On fixed code the reused slot keeps the predecessor's level-0 edges and stays
// visible to traversal, so it does: the predecessor held the same id at the
// same place, so those edges point at real, near neighbours and the point is
// found at rank 0. With the fix reverted (level0Len zeroed at placement and the
// node marked unlinked) the slot is an edgeless, hidden dead end, and the point
// is stored, live, returned by Get, and returned by NO search — measured 5/5
// found on fixed code, 0/5 on reverted.
//
// The Get assertion is not decoration. It is what makes a search miss here
// MEAN something: without it, "not found mid-upsert" could just be "not stored
// yet", and the arm would prove nothing.
func TestSearchabilityInvariantDuringUpsertWindow(t *testing.T) {
	const dim, n = 8, 600
	cfg := Config{Dim: dim, Metric: L2, M: 8, EfConstruction: 64, EfSearch: 64, Seed: 12}
	idx := invNewColl(t, "inv_window", cfg, invOpts{})
	defer idx.close()
	coll := idx.(*invColl).c

	base := invLineCorpus(t, n, dim, 1)
	invSeedAll(t, idx, base)
	// The upsert corpus keeps each point's position on the line and changes only
	// its jitter, which is what an upsert of the same logical record looks like —
	// and what makes the predecessor's inherited edges the right ones.
	next := invLineCorpus(t, n, dim, 2)

	victims := []uint64{100, 200, 300, 400, 500}
	var checked int
	defer func() { linkGapHook = nil }()

	for _, victim := range victims {

		newVec := next.vec[victim]
		linkGapHook = func() {
			checked++
			// PRECONDITION: the point is stored and live at this instant. If this
			// ever stops holding, the search assertion below would be vacuous.
			if _, _, _, _, _, ok := coll.Get(victim); !ok {
				t.Errorf("id %d: Get reports it absent inside its own link window — the "+
					"search assertion below would prove nothing", victim)
				return
			}
			res, err := coll.Search(newVec, invSearchK)
			if err != nil {
				t.Errorf("id %d: search inside the link window: %v", victim, err)
				return
			}
			rank := invRankOf(resultIDs(res), victim)
			if rank == 0 {
				return
			}
			where := fmt.Sprintf("at rank %d", rank)
			if rank < 0 {
				where = "NOT AT ALL"
			}
			t.Errorf("id %d is stored and live (Get returns it) INSIDE its placement/link "+
				"window, but searching for its own vector returns it %s — got %v. The reused "+
				"slot is an edgeless dead end for the whole window, which is what partitions "+
				"the graph under churn.", victim, where, resultIDs(res))
		}
		if err := coll.Upsert(victim, newVec, "", 0, nil, nil); err != nil {
			t.Fatalf("upsert id %d: %v", victim, err)
		}
		linkGapHook = nil
	}

	if checked != len(victims) {
		t.Fatalf("the link-gap hook fired %d times for %d upserts — the upserts did not defer a "+
			"link phase, so nothing inside the window was verified", checked, len(victims))
	}

	// And the end state is still whole, so a pass here is not bought by leaving
	// the graph damaged.
	assertSelfSearchable(t, idx, next, next.ids, "upsert-window/after")
	assertGraphWhole(t, idx, "upsert-window/after")
}

// TestSearchabilityInvariantUnderChurn is the CONCURRENT END-STATE arm.
//
// It runs many upserts CONCURRENTLY, drawing ids from one shared ascending
// cursor so the open placement/link windows form a contiguous band sweeping
// along a line-shaped corpus, with searches running alongside as load and as
// the -race lane's shared-state exercise. Then it asserts the invariant over
// the whole corpus plus a direct reachability check, which localises a failure
// to "the graph is partitioned" rather than "some query missed".
//
// WHAT IT DOES NOT DO, stated so nobody mistakes it for the bug-3 gate: it does
// not detect the reverted reused-slot fix (0 detections in 10 runs). The damage
// that fix prevents is transient — each upserted node writes its own correct
// edges as its window closes — so an end-state assertion cannot see it. That is
// TestSearchabilityInvariantDuringUpsertWindow's job. This arm's value is the
// concurrent read/write exercise and the end-state guarantee that sustained
// churn leaves the graph whole.
func TestSearchabilityInvariantUnderChurn(t *testing.T) {
	const dim = 8
	n := 1200
	passes := 3
	if testing.Short() {
		n, passes = 400, 2
	}

	// M and EfConstruction are deliberately modest: a sparse graph is where a
	// band of dead ends actually severs connectivity, which is the regime the
	// measured failures came from.
	cfg := Config{Dim: dim, Metric: L2, M: 8, EfConstruction: 64, EfSearch: 64, Seed: 12}
	idx := invNewColl(t, "inv_churn", cfg, invOpts{})
	defer idx.close()

	base := invLineCorpus(t, n, dim, 1)
	invSeedAll(t, idx, base)
	assertSelfSearchable(t, idx, base, base.ids, "churn/seeded")

	cur := base
	for pass := 0; pass < passes; pass++ {
		next := invLineCorpus(t, n, dim, uint64(pass)+2)

		var cursor atomic.Int64
		var wg sync.WaitGroup
		stop := make(chan struct{})

		// Writers: eight goroutines pulling the next id from ONE ascending
		// cursor, so at any instant a contiguous band of ~8 slots is inside its
		// link window simultaneously.
		const writers = 8
		errs := make(chan error, writers)
		for w := 0; w < writers; w++ {
			wg.Add(1)
			go func() {
				defer wg.Done()
				for {
					i := cursor.Add(1) - 1
					if int(i) >= n {
						return
					}
					id := next.ids[i]
					if err := idx.upsert(id, next.vec[id]); err != nil {
						select {
						case errs <- fmt.Errorf("upsert id %d: %w", id, err):
						default:
						}
						return
					}
				}
			}()
		}

		// Readers: concurrent searches for the duration, so the -race lane sees
		// the read path against every write path, and so the graph is being
		// traversed while the band of reused slots sweeps through it.
		const readers = 4
		for r := 0; r < readers; r++ {
			wg.Add(1)
			go func(seed int64) {
				defer wg.Done()
				rng := rand.New(rand.NewSource(seed)) //nolint:gosec // test fixture
				for {
					select {
					case <-stop:
						return
					default:
					}
					id := cur.ids[rng.Intn(n)]
					if _, err := idx.search(cur.vec[id], invSearchK); err != nil {
						select {
						case errs <- fmt.Errorf("concurrent search: %w", err):
						default:
						}
						return
					}
				}
			}(int64(r) + 1)
		}

		// Wait for the writers, then release the readers.
		go func() {
			for cursor.Load() < int64(n) {
				time.Sleep(time.Millisecond)
			}
			close(stop)
		}()
		wg.Wait()
		select {
		case err := <-errs:
			t.Fatal(err)
		default:
		}

		cur = next
		assertSelfSearchable(t, idx, cur, cur.ids, fmt.Sprintf("churn/pass%d", pass))
	}

	// Reachability is the mechanism behind searchability, and asserting it
	// directly turns "a query missed" into "the graph is partitioned".
	if h, ok := idx.(*invColl).c.idx.(*hnsw); ok {
		if bad := unreachableLivePoints(h); len(bad) != 0 {
			ids := make([]uint64, 0, len(bad))
			for _, slot := range bad {
				ids = append(ids, h.arena.ID(slot))
			}
			if len(ids) > 16 {
				ids = ids[:16]
			}
			t.Errorf("%d live point(s) unreachable from the entry point after %d concurrent "+
				"upsert passes — the graph is partitioned (first ids: %v)", len(bad), passes, ids)
		}
	}
}

// TestSearchabilityInvariantConcurrentInsert covers the other concurrency
// shape: fresh inserts landing while searches run, with no deletes in play, so
// a failure isolates the placement/link window itself rather than slot reuse.
func TestSearchabilityInvariantConcurrentInsert(t *testing.T) {
	const dim = 8
	n := 1024
	if testing.Short() {
		n = 256
	}
	cfg := Config{Dim: dim, Metric: L2, M: 16, EfConstruction: 100, EfSearch: 64, Seed: 3}
	idx := invNewColl(t, "inv_concurrent_insert", cfg, invOpts{})
	defer idx.close()

	c := newInvCorpus(t, invSeedIDs(n, 51), dim, 11)

	var cursor atomic.Int64
	var wg sync.WaitGroup
	stop := make(chan struct{})
	errs := make(chan error, 16)

	const writers = 6
	for w := 0; w < writers; w++ {
		wg.Add(1)
		go func() {
			defer wg.Done()
			for {
				i := cursor.Add(1) - 1
				if int(i) >= n {
					return
				}
				id := c.ids[i]
				if err := idx.insert(id, c.vec[id], 0); err != nil {
					select {
					case errs <- fmt.Errorf("insert id %d: %w", id, err):
					default:
					}
					return
				}
			}
		}()
	}
	const readers = 4
	for r := 0; r < readers; r++ {
		wg.Add(1)
		go func(seed int64) {
			defer wg.Done()
			rng := rand.New(rand.NewSource(seed)) //nolint:gosec // test fixture
			for {
				select {
				case <-stop:
					return
				default:
				}
				id := c.ids[rng.Intn(n)]
				if _, err := idx.search(c.vec[id], invSearchK); err != nil {
					select {
					case errs <- fmt.Errorf("concurrent search: %w", err):
					default:
					}
					return
				}
			}
		}(int64(r) + 100)
	}
	go func() {
		for cursor.Load() < int64(n) {
			time.Sleep(time.Millisecond)
		}
		close(stop)
	}()
	wg.Wait()
	select {
	case err := <-errs:
		t.Fatal(err)
	default:
	}

	assertSelfSearchable(t, idx, c, c.ids, "concurrent-insert")
}

// ---------------------------------------------------------------------------
// Large corpus
// ---------------------------------------------------------------------------

// TestSearchabilityInvariantLargeCorpus runs the same property at a size that
// exercises multi-level graphs and real slot pressure. It is skipped by -short
// (the -race lane), following the precedent in the filter-bitset gate suites:
// the shape coverage above is what runs on every push, and this is the
// unabridged pass.
//
// THE BUILD HERE IS SERIAL (one worker) ON PURPOSE, and it is now the SERIAL
// arm rather than a workaround. Writing this test turned up a defect in the
// MULTI-WORKER bulk build at this scale — it left points unreachable from the
// entry point, the exact bug shape this file exists to guard — and this arm was
// pinned to one worker so it would not flicker while that stayed open. The
// defect is fixed (BuildConcurrent now arms node.unlinked over the placement/
// link window), and the multi-worker case is guarded permanently and ungated by
// TestSearchabilityInvariantConcurrentBulkBuildOrphan. Keeping this arm serial
// keeps a build path with NO concurrency in the matrix, so a future regression
// can be localized to the parallel phase rather than to graph quality.
func TestSearchabilityInvariantLargeCorpus(t *testing.T) {
	if testing.Short() {
		t.Skip("large-corpus arm: skipped in -short (the -race lane); the matrix above covers the shapes")
	}
	const n, dim = 20_000, 64
	cfg := Config{Dim: dim, Metric: L2, M: 16, EfConstruction: 200, EfSearch: 64, Seed: 1}
	idx := invNewColl(t, "inv_large", cfg, invOpts{})
	defer idx.close()

	c := newInvCorpus(t, invSeedIDs(n, 61), dim, 13)
	vecs := make([][]float32, len(c.ids))
	for i, id := range c.ids {
		vecs[i] = c.vec[id]
	}
	if err := idx.(*invColl).c.BuildConcurrent(c.ids, vecs, 1); err != nil {
		t.Fatalf("bulk build: %v", err)
	}
	assertSelfSearchable(t, idx, c, c.ids, "large/bulk")
	assertGraphWhole(t, idx, "large/bulk")

	// Delete a third without reclaiming, then assert the survivors again: the
	// graph is now full of tombstoned vertices at a scale where a traversal has
	// to route around them.
	live, dead := invSplitEveryNth(c.ids, 3)
	for _, id := range dead {
		if err := idx.remove(id); err != nil {
			t.Fatalf("delete id %d: %v", id, err)
		}
	}
	assertSelfSearchable(t, idx, c, live, "large/after-delete")
	assertNotSearchable(t, idx, c, dead, "large/after-delete")

	if got := idx.(invReclaimer).reclaim(); got != len(dead) {
		t.Fatalf("Reclaim removed %d, want %d", got, len(dead))
	}
	assertSelfSearchable(t, idx, c, live, "large/after-reclaim")
	assertGraphWhole(t, idx, "large/after-reclaim")
}

// assertGraphWhole checks the structural property behind searchability: every
// live point has a path from the entry point. It is exact — no beam width, no
// approximation — so it separates "a query missed" from "the point is gone",
// which is the distinction every bug in this file's header turned on.
func assertGraphWhole(t *testing.T, idx invIndex, what string) {
	t.Helper()
	a, ok := idx.(*invColl)
	if !ok {
		return
	}
	h, ok := a.c.idx.(*hnsw)
	if !ok {
		return
	}
	bad := unreachableLivePoints(h)
	if len(bad) == 0 {
		return
	}
	ids := make([]uint64, 0, len(bad))
	for _, slot := range bad {
		ids = append(ids, h.arena.ID(slot))
	}
	if len(ids) > 16 {
		ids = ids[:16]
	}
	t.Errorf("%s: %d live point(s) UNREACHABLE from the entry point — the graph is partitioned, "+
		"so no width of search beam can find them (first ids: %v)", what, len(bad), ids)
}

// TestSearchabilityInvariantConcurrentBulkBuildOrphan is the STANDING GATE on
// the default bulk-ingest path: BuildConcurrent with many workers, at the scale
// where the graph is genuinely multi-level. It began life as an env-gated
// reproducer for an open defect and is now ungated, because the defect is fixed
// and this is the only arm that would catch its return.
//
// THE DEFECT IT GUARDS. BuildConcurrent with MORE THAN ONE WORKER left live
// points permanently unreachable from the entry point at ~20k scale. They were
// present in the arena, byte-correct, returned by Get, counted live — and
// absent from every vector search, because no path reached them. Searching
// harder did not help and that was the whole point: at EfSearch 64, 128, 256
// and 512 the same ids stayed missing. Unreachable is unreachable.
//
// Measured before the fix, dim 64, M=16, EfConstruction=200, counted by
// unreachableLivePoints (the helper the orphan and reused-slot regressions
// already trust):
//
//	build path                    n=5,000        n=20,000
//	incremental Insert            0, 0           0, 0
//	BuildConcurrent workers=1     0, 0           0, 0
//	BuildConcurrent workers=2     0, 0           0, 0
//	BuildConcurrent workers=4     0, 0           0, 1
//	BuildConcurrent workers=8     0, 0           1, 4
//
// (two seeds per cell). The pattern was the diagnosis: it scaled with the
// WORKER COUNT and not with the corpus alone, it was absent from the serial
// build of the same corpus with the same seed, and absent from incremental
// insert entirely — so it lived in the concurrent link phase, not in graph
// quality.
//
// THE CAUSE, for whoever sees this go red again. linkNode publishes a node's
// levels top-down, so between writing level lc and writing level lc-1 the node
// is REACHABLE at lc with an empty list at lc-1. BuildConcurrent left
// node.unlinked false for the whole parallel phase, so another worker's width-1
// descent could land on such a node at an upper level, carry it down as the
// sole level-0 frontier, and expand a node with zero level-0 edges. Every
// orphan measured had a one-element candidate set whose single member had
// degree 0 at level 0 at that instant; the lone back-edge it wrote was then
// pruned away when that frontier node wrote its own 2M level-0 edges. The fix
// arms node.unlinked in BuildConcurrent's placement loop. Ablation, 5 seeds per
// cell, orphans WITHOUT the flag / WITH it:
//
//	(5k, w4)   0 / 0      (20k, w8)   7 / 0
//	(5k, w8)   0 / 0      (20k, w16) 11 / 0
//	(20k, w4)  4 / 0      (50k, w8)  24 / 0
//
// and the count of STARVED links (a candidate set of <= 2) falls from 99-443,
// rising with both n and worker count, to a flat 33-42 independent of both —
// that residual is the slot-0 bootstrap the serial build also pays.
func TestSearchabilityInvariantConcurrentBulkBuildOrphan(t *testing.T) {
	if testing.Short() {
		t.Skip("20k-point bulk build: skipped in -short (the -race lane); the shape matrix above covers -race")
	}
	const n, dim = 20_000, 64
	cfg := Config{Dim: dim, Metric: L2, M: 16, EfConstruction: 200, EfSearch: 64, Seed: 2}
	idx := invNewColl(t, "inv_bulk_orphan", cfg, invOpts{})
	defer idx.close()

	c := newInvCorpus(t, invSeedIDs(n, 61), dim, 13)
	vecs := make([][]float32, len(c.ids))
	for i, id := range c.ids {
		vecs[i] = c.vec[id]
	}
	if err := idx.(*invColl).c.BuildConcurrent(c.ids, vecs, 8); err != nil {
		t.Fatalf("bulk build: %v", err)
	}
	assertGraphWhole(t, idx, "concurrent-bulk-build")
	assertSelfSearchable(t, idx, c, c.ids, "concurrent-bulk-build")
}

// TestSearchabilityInvariantConcurrentBulkBuildOrphanWithPayloads is the arm
// above, run through the PAYLOAD-BEARING build.
//
// The two changes met here and the gap was real: the orphan fix was ablated on
// vector-only builds, and the payload path was added against a base that did not
// have the fix, so nobody had run the combination at the scale where the defect
// appears. It shares BuildConcurrentMeta's placement loop with the vectors-only
// build — applyBulkMeta runs a few lines above where `unlinked` is armed — so it
// was exposed to exactly the same defect and is closed by exactly the same two
// lines. Measured on this branch at this shape: 0 orphans with the flag armed, 4
// orphans and 10 unsearchable points with it disarmed.
//
// It asserts more than the vectors-only arm because a payload-bearing build has
// more ways to be wrong: the graph must be whole, every point must be findable by
// its own vector, every payload must have landed on ITS OWN point, and a filter
// selecting exactly one point must return that point. A payload applied to the
// WRONG SLOT passes the first two and fails the last two.
func TestSearchabilityInvariantConcurrentBulkBuildOrphanWithPayloads(t *testing.T) {
	if testing.Short() {
		t.Skip("20k-point bulk build: skipped in -short (the -race lane)")
	}
	const n, dim = 20_000, 64
	cfg := Config{Dim: dim, Metric: L2, M: 16, EfConstruction: 200, EfSearch: 64, Seed: 2}
	idx := invNewColl(t, "inv_bulk_orphan_pay", cfg, invOpts{})
	defer idx.close()

	c := newInvCorpus(t, invSeedIDs(n, 61), dim, 13)
	vecs := make([][]float32, len(c.ids))
	metas := make([]Metadata, len(c.ids))
	for i, id := range c.ids {
		vecs[i] = c.vec[id]
		// Every fifth point payload-less, so the mixed shape rides along.
		if id%5 != 0 {
			metas[i] = Metadata{"id": NewInt(int64(id))} //nolint:gosec // small test ids
		}
	}
	coll := idx.(*invColl).c
	if err := coll.BuildConcurrentMeta(c.ids, vecs, metas, 8); err != nil {
		t.Fatalf("payload-bearing bulk build: %v", err)
	}
	assertGraphWhole(t, idx, "concurrent-bulk-build-payloads")
	assertSelfSearchable(t, idx, c, c.ids, "concurrent-bulk-build-payloads")

	var badMeta, badFiltered int
	for _, id := range c.ids {
		_, m, _, _, _, ok := coll.Get(id)
		if !ok {
			badMeta++
			continue
		}
		if id%5 == 0 {
			if len(m) != 0 {
				badMeta++
			}
			continue
		}
		if len(m) != 1 || m["id"].Int != int64(id) { //nolint:gosec // small test ids
			badMeta++
			continue
		}
		res, err := coll.SearchFiltered(c.vec[id], invSearchK,
			Filter{Op: FilterEq, Field: "id", Value: NewInt(int64(id))}) //nolint:gosec
		if err != nil {
			t.Fatalf("filtered self-query for id %d: %v", id, err)
		}
		if invRankOf(resultIDs(res), id) != 0 {
			badFiltered++
		}
	}
	if badMeta != 0 || badFiltered != 0 {
		t.Errorf("concurrent-bulk-build-payloads: %d points carry the wrong payload, %d are not "+
			"returned by a filter selecting exactly their own payload (of %d)", badMeta, badFiltered, n)
	}
}
