// SPDX-License-Identifier: Apache-2.0

package vector

import (
	"bytes"
	"fmt"
	"math"
	"math/rand"
	"sort"
	"sync"
	"sync/atomic"
	"testing"
	"time"
)

// GATES FOR READER-NON-BLOCKING INSERT (Option B).
//
// Moving the link phase out of the exclusive lock changes WHEN a query may
// observe the graph, not WHAT the graph ends up being. These tests pin both
// halves of that claim: the state a concurrent reader can see must never be
// corrupt or under-connected (orphan + race gates), and the state the index
// settles on must be exactly what the fully-serial build produced (determinism
// + recall gates).

// searchLoad starts `n` goroutines hammering h with random queries until the
// returned stop function is called. It reports how many searches completed, so a
// test can assert it actually applied load rather than silently racing nothing.
func searchLoad(t *testing.T, h *hnsw, dim, n int) (stop func() int64) {
	t.Helper()
	var halt atomic.Bool
	var done atomic.Int64
	var wg sync.WaitGroup
	for i := 0; i < n; i++ {
		wg.Add(1)
		go func(seed int64) {
			defer wg.Done()
			rng := rand.New(rand.NewSource(seed))
			q := make([]float32, dim)
			var dst []Result
			for !halt.Load() {
				for d := range q {
					q[d] = rng.Float32()
				}
				var err error
				if dst, err = h.SearchInto(dst[:0], q, 10, Filter{}); err != nil {
					t.Errorf("search: %v", err)
					return
				}
				done.Add(1)
			}
		}(int64(i)*7919 + 1)
	}
	return func() int64 {
		halt.Store(true)
		wg.Wait()
		return done.Load()
	}
}

// TestOptionBAutoTrainDeterminismUnderSearchLoad is the replica-determinism gate
// for the relocked insert path. It is TestPQHNSWAutoTrainReplicaDeterminism with
// one addition that the relock makes load-bearing: each replica is driven with
// concurrent SEARCH traffic while it applies the sequence.
//
// Why that matters now. The deterministic auto-train has two moving parts — WHICH
// insert trips the threshold, and WHAT sample it trains on — and the relock moved
// the first one. The trigger used to be evaluated after the node was linked, with
// the write lock held for the whole insert; it is now evaluated inside the
// placement section, and the training itself runs after the link phase under a
// re-acquired write lock. Both the quantizer's slot-ordered sample and the seeded
// level RNG stay inside that serial placement section, so replicas must still
// agree BYTE-FOR-BYTE — and readers, which take no part in either, must be unable
// to perturb them.
//
// The inserts per replica stay serial. That is the production model: every
// mutator goes through Collection.opMu, so two inserts never overlap on one
// index. If that ever changed — an FSM that pipelined applies — the level RNG
// draw order would be the thing to re-examine first, since it is a shared
// *rand.Rand consumed once per placement; it is deterministic today precisely
// because placement is serialized.
func TestOptionBAutoTrainDeterminismUnderSearchLoad(t *testing.T) {
	const (
		dim       = 32
		threshold = 600
		n         = 700 // > threshold so the auto-train trips during the inserts
		m         = 8
	)
	ids, vecs := siftLikeCorpus(n, dim, 123)
	cfg := hnswPQAutoTrainConfig(dim, threshold, m, false)

	run := func(readers int) (*hnsw, int64) {
		h, err := newHNSW(cfg)
		if err != nil {
			t.Fatal(err)
		}
		var searches int64
		stop := func() int64 { return 0 }
		if readers > 0 {
			stop = searchLoad(t, h, dim, readers)
		}
		insertAllHNSW(t, h, ids, vecs)
		searches = stop()
		return h, searches
	}

	a, sa := run(4)
	b, sb := run(4)
	quiet, _ := run(0)

	if sa == 0 || sb == 0 {
		t.Fatalf("no concurrent search load applied (a=%d b=%d) — the gate proves nothing", sa, sb)
	}
	if a.pqUntrained() || b.pqUntrained() || quiet.pqUntrained() {
		t.Fatal("every replica must have auto-trained after crossing the threshold")
	}

	cbA, cbB, cbQ := dumpCodebooks(t, a), dumpCodebooks(t, b), dumpCodebooks(t, quiet)
	if !bytes.Equal(cbA, cbB) {
		t.Fatal("codebooks DIVERGED between two search-loaded replicas — auto-train is not deterministic under concurrent reads")
	}
	if !bytes.Equal(cbA, cbQ) {
		t.Fatal("codebooks DIVERGED between a search-loaded replica and an unloaded one — concurrent reads perturbed the training sample or its trigger")
	}
	codesA, codesB, codesQ := dumpCodes(t, a, n), dumpCodes(t, b, n), dumpCodes(t, quiet, n)
	if !bytes.Equal(codesA, codesB) || !bytes.Equal(codesA, codesQ) {
		t.Fatal("per-slot PQ codes DIVERGED — non-deterministic encode under concurrent reads")
	}
	t.Logf("determinism OK under %d+%d concurrent searches: %d codebook bytes + %d code bytes byte-identical",
		sa, sb, len(cbA), len(codesA))
}

// TestOptionBGraphIsIdenticalUnderSearchLoad is the strongest determinism
// statement available for the relock: the graph an index settles on must be
// EDGE-FOR-EDGE the graph the same insert sequence built with no readers at all.
//
// A serial insert sequence is deterministic, so any difference here would mean
// the link phase's outcome depends on reader interleaving — the exact failure the
// striped neighbor reads exist to prevent (a reader cannot mutate, but the gate
// it flips changes which code path the LINKER's own neighbor reads take).
func TestOptionBGraphIsIdenticalUnderSearchLoad(t *testing.T) {
	const n, dim = 3000, 24
	ids, vecs := siftLikeCorpus(n, dim, 55)
	cfg := Config{Dim: dim, Metric: L2, M: 16, EfConstruction: 200, EfSearch: 64, Seed: 42}

	build := func(readers int) *hnsw {
		h, err := newHNSW(cfg)
		if err != nil {
			t.Fatal(err)
		}
		stop := func() int64 { return 0 }
		if readers > 0 {
			stop = searchLoad(t, h, dim, readers)
		}
		insertAllHNSW(t, h, ids, vecs)
		if readers > 0 && stop() == 0 {
			t.Fatal("no search load applied")
		}
		return h
	}

	quiet, loaded := build(0), build(4)
	dq, dl := dumpGraph(t, quiet), dumpGraph(t, loaded)
	if !bytes.Equal(dq, dl) {
		t.Fatalf("graph DIVERGED under concurrent search load (%d vs %d bytes) — the link phase's result depends on reader interleaving", len(dq), len(dl))
	}
	t.Logf("graph identical under search load: %d bytes of edges", len(dq))
}

// dumpGraph serializes every node's level-by-level neighbor list in slot order —
// a total, order-sensitive fingerprint of the graph's edges.
func dumpGraph(t *testing.T, h *hnsw) []byte {
	t.Helper()
	var buf bytes.Buffer
	fmt.Fprintf(&buf, "ep=%d maxLevel=%d\n", h.entryPoint, h.maxLevel)
	for slot, nd := range h.nodes {
		if nd == nil {
			fmt.Fprintf(&buf, "%d nil\n", slot)
			continue
		}
		for lc := 0; lc <= nd.level; lc++ {
			fmt.Fprintf(&buf, "%d/%d %v\n", slot, lc, h.nbrsAt(nd, lc))
		}
	}
	return buf.Bytes()
}

// TestOptionBIncrementalOrphansUnderConcurrency is the orphan gate for the
// INCREMENTAL path, extending the pattern build_concurrent_orphan_test.go
// established for the bulk builder.
//
// The failure it guards is the same one the bulk builder hit: a node's forward
// write BLOWING AWAY back-edges another linker appended in the meantime, leaving
// a point with zero in-edges — retrievable by id, unreachable by search, forever.
// The incremental path never needed the back-edge merge while it held the
// exclusive lock (nothing else could be linking); running under the read lock it
// does, and the dense-arc corpus is chosen because its thin in-degree makes a
// single lost back-edge orphan a node.
//
// It drives hnsw.Insert from several goroutines directly. Production serializes
// mutators upstream, so this is deliberately HARSHER than production — the point
// is that the graph invariant holds even when the upstream serialization is
// removed, which is what makes the invariant an invariant rather than a
// coincidence of the caller. It uses the SAME 40-point arc and repeat-many-times
// shape as TestBuildConcurrentNoOrphans, because that is the shape that actually
// reproduced the bulk-builder bug: a single large build hides it, while a
// hundred small ones sample enough interleavings to hit it.
func TestOptionBIncrementalOrphansUnderConcurrency(t *testing.T) {
	const (
		n     = 40
		iters = 150
	)
	ids, vecs := denseArcCorpus(n)
	orphans, unreach, dangling := 0, 0, 0
	var searches int64
	for it := 0; it < iters; it++ {
		h, err := newHNSW(Config{Dim: 4, Metric: Cosine, M: 16, EfConstruction: 200, EfSearch: 64, Seed: int64(it + 1)})
		if err != nil {
			t.Fatal(err)
		}
		stop := searchLoad(t, h, 4, 2)
		var wg sync.WaitGroup
		var next atomic.Int64
		for w := 0; w < 4; w++ {
			wg.Add(1)
			go func() {
				defer wg.Done()
				for {
					i := int(next.Add(1)) - 1
					if i >= n {
						return
					}
					if _, _, err := h.Insert(ids[i], vecs[i], 0, nil, nil, nil, CASCond{}); err != nil {
						t.Errorf("insert %d: %v", i, err)
						return
					}
				}
			}()
		}
		wg.Wait()
		searches += stop()

		inDeg, reach := arcInDegreeAndReach(h)
		for slot, nd := range h.nodes {
			if nd == nil {
				continue
			}
			if inDeg[slot] == 0 {
				orphans++
			}
			if !reach[slot] {
				unreach++
			}
			// Dangling edges: every neighbor slot must name a live node. A
			// concurrent prune writing back a stale list would show up here.
			for lc := 0; lc <= nd.level; lc++ {
				for _, nb := range h.nbrsAt(nd, lc) {
					if int(nb) >= len(h.nodes) || h.nodes[nb] == nil {
						dangling++
					}
				}
			}
		}
	}
	if searches == 0 {
		t.Fatal("no concurrent search load applied")
	}
	if orphans != 0 || unreach != 0 || dangling != 0 {
		t.Errorf("over %d concurrent incremental builds: %d zero-in-degree nodes, %d unreachable, %d dangling edges — want 0/0/0",
			iters, orphans, unreach, dangling)
	}
	t.Logf("%d builds x %d nodes, %d concurrent searches: no orphans, no unreachable nodes, no dangling edges",
		iters, n, searches)
}

// TestOptionBRecallUnderConcurrentInserts is the recall gate: an index built by
// CONCURRENT incremental inserts must score within 0.01 recall@10 of one built by
// strictly serial inserts on the same corpus. Concurrency reorders which node
// links first, so the graphs differ; what must not differ is their quality.
func TestOptionBRecallUnderConcurrentInserts(t *testing.T) {
	const n, dim, k = 6000, 32, 10
	const epsilon = 0.01
	ids, vecs := siftLikeCorpus(n, dim, 31)
	_, queries := siftLikeCorpus(300, dim, 77)
	cfg := Config{Dim: dim, Metric: L2, M: 16, EfConstruction: 200, EfSearch: 128, Seed: 42}

	serial, err := newHNSW(cfg)
	if err != nil {
		t.Fatal(err)
	}
	insertAllHNSW(t, serial, ids, vecs)

	conc, err := newHNSW(cfg)
	if err != nil {
		t.Fatal(err)
	}
	stop := searchLoad(t, conc, dim, 2)
	var wg sync.WaitGroup
	var next atomic.Int64
	for w := 0; w < 4; w++ {
		wg.Add(1)
		go func() {
			defer wg.Done()
			for {
				i := int(next.Add(1)) - 1
				if i >= n {
					return
				}
				if _, _, err := conc.Insert(ids[i], vecs[i], 0, nil, nil, nil, CASCond{}); err != nil {
					t.Errorf("insert %d: %v", i, err)
					return
				}
			}
		}()
	}
	wg.Wait()
	if stop() == 0 {
		t.Fatal("no concurrent search load applied")
	}

	rs := recallOf(t, serial, vecs, queries, k)
	rc := recallOf(t, conc, vecs, queries, k)
	t.Logf("recall@%d serial=%.4f concurrent=%.4f delta=%+.4f (epsilon %.2f)", k, rs, rc, rc-rs, epsilon)
	if math.Abs(rc-rs) > epsilon {
		t.Errorf("concurrent-insert recall %.4f differs from serial %.4f by more than %.2f", rc, rs, epsilon)
	}
}

// ---- the measurement that justifies the change ----

// searchLatencyUnderInserts measures the distribution of single-query latency
// while `writers` goroutines insert continuously in the background. It returns
// p50/p99 in microseconds plus the number of background inserts that landed.
//
// It reports a LATENCY distribution rather than a throughput average on purpose.
// The change does not make any individual query faster — it makes a query stop
// WAITING for an insert's link phase, which is a tail effect. An average would
// blur exactly the thing being fixed.
func searchLatencyUnderInserts(h *hnsw, dim, queries, writers int, seed int64) (lat []float64, inserts int64) {
	var halt atomic.Bool
	var ins atomic.Int64
	var wg sync.WaitGroup
	for w := 0; w < writers; w++ {
		wg.Add(1)
		go func(seed int64) {
			defer wg.Done()
			rng := rand.New(rand.NewSource(seed))
			v := make([]float32, dim)
			for !halt.Load() {
				for d := range v {
					v[d] = rng.Float32()
				}
				// A duplicate id would abort this goroutine, and a silently dead
				// writer measures the IDLE case while the log still claims otherwise
				// — the one way this harness could lie. Ids come from a process-wide
				// counter so no arm can ever collide with a previous one.
				if _, _, err := h.Insert(bgInsertID.Add(1), v, 0, nil, nil, nil, CASCond{}); err != nil {
					panic(err)
				}
				ins.Add(1)
			}
		}(seed*31 + int64(w))
	}

	rng := rand.New(rand.NewSource(seed))
	q := make([]float32, dim)
	var dst []Result
	lat = make([]float64, 0, queries)
	for i := 0; i < queries; i++ {
		for d := range q {
			q[d] = rng.Float32()
		}
		start := time.Now()
		dst, _ = h.SearchInto(dst[:0], q, 10, Filter{})
		lat = append(lat, float64(time.Since(start).Nanoseconds())/1000)
	}
	halt.Store(true)
	wg.Wait()

	sort.Float64s(lat)
	return lat, ins.Load()
}

// bgInsertID hands out process-unique ids for background inserters.
var bgInsertID atomic.Uint64

// pctl returns the p-th percentile of an already-sorted sample.
func pctl(sorted []float64, p float64) float64 {
	i := int(p / 100 * float64(len(sorted)))
	if i >= len(sorted) {
		i = len(sorted) - 1
	}
	return sorted[i]
}

// TestOptionBSearchLatencyUnderInserts prints the measurement this whole change
// exists to move: query latency with a SATURATING background inserter. Run it on
// both sides of the relock and compare.
//
// Each arm gets a FRESH index of the same size, so an arm never inherits the
// points a previous arm's writer added — otherwise the later arms would be
// searching a bigger graph and the comparison would fold index growth into the
// locking effect.
//
// The only ASSERTION is that the background writer actually ran. Nothing about
// the latencies themselves is asserted, deliberately: they are wall-clock
// measurements, and the full suite runs test binaries in parallel, so a
// cross-package neighbour can inflate either arm by more than the effect being
// measured. Read the numbers from a dedicated run on an otherwise-idle machine:
//
//	go test ./vector -run TestOptionBSearchLatencyUnderInserts -v -count=1
func TestOptionBSearchLatencyUnderInserts(t *testing.T) {
	const n, dim, queries = 20000, 64, 8000
	ids, vecs := siftLikeCorpus(n, dim, 5)
	cfg := Config{Dim: dim, Metric: L2, M: 16, EfConstruction: 200, EfSearch: 128, Seed: 42}
	bgInsertID.Store(1 << 40)

	fresh := func() *hnsw {
		h, err := newHNSW(cfg)
		if err != nil {
			t.Fatal(err)
		}
		insertAllHNSW(t, h, ids, vecs)
		return h
	}
	show := func(label string, lat []float64, ins int64) {
		t.Logf("%-10s p50=%7.1f p90=%7.1f p99=%7.1f p99.9=%8.1f max=%9.1f us  (%d background inserts)",
			label, pctl(lat, 50), pctl(lat, 90), pctl(lat, 99), pctl(lat, 99.9), lat[len(lat)-1], ins)
	}

	idle, _ := searchLatencyUnderInserts(fresh(), dim, queries, 0, 1)
	show("idle", idle, 0)
	loaded, ins := searchLatencyUnderInserts(fresh(), dim, queries, 1, 2)
	show("1 writer", loaded, ins)

	if ins == 0 {
		t.Fatal("the background writer never ran — the loaded arm measured the idle case")
	}
}

// BenchmarkSearchIntoZeroLinkers measures the QUIESCENT read path — the cost the
// relock adds to an index that nobody is writing to, which is the state a
// read-mostly deployment is in essentially all the time.
//
// The gate is that it is free. The old gate for "is anything linking" was a nil
// check on a slice field; the new one is an atomic load of a counter that lives
// alone on its cache line and is written twice per insert. On amd64 that load
// compiles to an ordinary MOV, so the two should be indistinguishable — this
// benchmark is what says so rather than assuming it.
func BenchmarkSearchIntoZeroLinkers(b *testing.B) {
	const n, dim = 20000, 64
	ids, vecs := siftLikeCorpus(n, dim, 5)
	cfg := Config{Dim: dim, Metric: L2, M: 16, EfConstruction: 200, EfSearch: 128, Seed: 42}
	h, err := newHNSW(cfg)
	if err != nil {
		b.Fatal(err)
	}
	for i := range ids {
		if _, _, err := h.Insert(ids[i], vecs[i], 0, nil, nil, nil, CASCond{}); err != nil {
			b.Fatal(err)
		}
	}
	if h.linking() {
		b.Fatal("expected a quiescent index (no linkers) for this benchmark")
	}
	_, queries := siftLikeCorpus(256, dim, 99)
	var dst []Result
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		dst, _ = h.SearchInto(dst[:0], queries[i%len(queries)], 10, Filter{})
	}
}
