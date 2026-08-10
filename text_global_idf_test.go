// SPDX-License-Identifier: Apache-2.0

package rostam

import (
	"context"
	"reflect"
	"testing"

	"github.com/rostamlabs/rostam/cluster"
	"github.com/rostamlabs/rostam/ops"
)

// TestMergeFanResults proves phase-0 (stats) degraded/missing propagate into the
// combined FanResult alongside phase-1 (scoring): the union is sorted + de-duped, and
// degraded is the OR of both phases. This is the degraded-partition propagation path
// honored on BOTH phases of the two-phase fan-out.
func TestMergeFanResults(t *testing.T) {
	// Phase 0 lost partition 2; phase 1 lost partitions 0 and 2 (2 overlaps).
	phase0 := cluster.FanResult{Degraded: true, Missing: []int{2}}
	phase1 := cluster.FanResult{Degraded: true, Missing: []int{0, 2}}
	got := mergeFanResults(phase0, phase1)
	if !got.Degraded {
		t.Fatal("combined Degraded = false, want true")
	}
	if !reflect.DeepEqual(got.Missing, []int{0, 2}) {
		t.Fatalf("combined Missing = %v, want [0 2] (sorted, de-duped union)", got.Missing)
	}

	// Phase 0 clean, phase 1 lost partition 1 ⇒ combined reflects phase 1.
	got = mergeFanResults(cluster.FanResult{}, cluster.FanResult{Degraded: true, Missing: []int{1}})
	if !got.Degraded || !reflect.DeepEqual(got.Missing, []int{1}) {
		t.Fatalf("phase-1-only degraded: got %+v", got)
	}

	// Phase 0 lost a partition, phase 1 clean ⇒ phase-0 degraded still surfaces (the
	// whole point — a stats-phase failure must not be silently dropped).
	got = mergeFanResults(cluster.FanResult{Degraded: true, Missing: []int{3}}, cluster.FanResult{})
	if !got.Degraded || !reflect.DeepEqual(got.Missing, []int{3}) {
		t.Fatalf("phase-0-only degraded: got %+v", got)
	}

	// Both clean ⇒ clean.
	got = mergeFanResults(cluster.FanResult{}, cluster.FanResult{})
	if got.Degraded || len(got.Missing) != 0 {
		t.Fatalf("both clean: got %+v", got)
	}
}

// docScores extracts the per-result BM25 scores in order.
func docScores(d []VectorDocument) []float32 {
	out := make([]float32, len(d))
	for i, r := range d {
		out[i] = r.Score
	}
	return out
}

// dfSplitsAcrossPartitions reports whether term-bearing docs for the dog-query
// actually land on >1 partition under P, so a per-shard-local df differs from the
// global df (the precondition that makes GlobalIDF observably matter vs the local
// path). It checks the dog-bearing docs (1,2,3,7).
func dfSplitsAcrossPartitions(P int) bool {
	parts := map[int]bool{}
	for _, id := range []uint64{1, 2, 3, 7} {
		parts[ops.PartitionOf(id, P)] = true
	}
	return len(parts) > 1
}

// TestSearchTextGlobalIDFBitIdentical is the HEADLINE test of the global-DF
// (dfs_query_then_fetch) two-phase fan-out: over a corpus whose per-shard df
// DIFFERS from the global df (the dog-bearing docs span multiple partitions), a
// P>1 collection searched with GlobalIDF=true returns BIT-IDENTICAL ids+scores to
//   - the same corpus in a P=1 collection (the single-node oracle), AND
//   - itself with the local-IDF path, which it must NOT match (sanity that the flag
//     actually changes the scoring).
func TestSearchTextGlobalIDFBitIdentical(t *testing.T) {
	const P = 4
	if !dfSplitsAcrossPartitions(P) {
		t.Skipf("dog-bearing docs do not split across %d partitions; corpus can't exercise global-vs-local df", P)
	}
	ctx := context.Background()

	s1 := newSingleEmbedded(t)
	waitLeaderEmbedded(t, s1)
	seedFullTextCollection(t, s1, "ft1", 1)

	sP := newSingleEmbedded(t)
	waitLeaderEmbedded(t, sP)
	seedFullTextCollection(t, sP, "ftP", P)

	const q = "dog"

	// P=1 is the single-node oracle: its local corpus IS the global corpus.
	oracle, _, err := s1.VectorSearchText(ctx, "ft1", q, 10, VectorSearchOpts{})
	if err != nil {
		t.Fatalf("P1 oracle SearchText: %v", err)
	}
	if len(oracle) == 0 {
		t.Fatal("P1 oracle returned no docs")
	}

	// P>1 with GlobalIDF=true must equal the oracle EXACTLY (ids + scores).
	global, fm, err := sP.VectorSearchText(ctx, "ftP", q, 10, VectorSearchOpts{GlobalIDF: true})
	if err != nil {
		t.Fatalf("P%d GlobalIDF SearchText: %v", P, err)
	}
	if fm.Degraded {
		t.Fatalf("P%d GlobalIDF degraded unexpectedly", P)
	}
	// BIT-IDENTITY by (id → score): the global-DF P>1 search must return the SAME id
	// set as the single-node oracle, each with the SAME (bit-identical) BM25 score.
	// Equal-score ties admit any valid top-k ORDER (the design-doc invariant), so the
	// per-position id sequence may differ on ties; the load-bearing claim is that
	// every doc's GLOBAL score equals the single-node score exactly. The ordered
	// scores (which ARE position-stable, ties or not) must also match exactly.
	oScore := map[uint64]float32{}
	for _, d := range oracle {
		oScore[d.ID] = d.Score
	}
	if len(global) != len(oracle) {
		t.Fatalf("GlobalIDF P%d count=%d != oracle %d", P, len(global), len(oracle))
	}
	for _, d := range global {
		os, ok := oScore[d.ID]
		if !ok {
			t.Fatalf("GlobalIDF P%d returned id %d absent from oracle %v", P, d.ID, docIDs(oracle))
		}
		if d.Score != os {
			t.Fatalf("GlobalIDF P%d score for id %d = %v != oracle %v (NOT bit-identical)", P, d.ID, d.Score, os)
		}
	}
	// The ORDERED score sequences are identical regardless of tie order.
	gs, os := docScores(global), docScores(oracle)
	for i := range os {
		if gs[i] != os[i] {
			t.Fatalf("GlobalIDF P%d ordered score[%d]=%v != oracle %v (NOT bit-identical)", P, i, gs[i], os[i])
		}
	}

	// SANITY: GlobalIDF=false (the per-shard-local path) must DIVERGE from the oracle
	// in scores — otherwise the flag would be a no-op and the test would prove nothing.
	local, _, err := sP.VectorSearchText(ctx, "ftP", q, 10, VectorSearchOpts{})
	if err != nil {
		t.Fatalf("P%d local SearchText: %v", P, err)
	}
	ls := docScores(local)
	diverged := len(ls) != len(os)
	if !diverged {
		for i := range os {
			if ls[i] != os[i] {
				diverged = true
				break
			}
		}
	}
	if !diverged {
		t.Fatalf("local-IDF P%d scores == oracle — the corpus does not actually split df, test has no teeth", P)
	}
}

// TestHybridTextGlobalIDFBitIdentical is the hybrid-text analogue: a P>1 hybrid-text
// search with GlobalIDF=true must equal the P=1 oracle exactly (ids + fused scores),
// because the global-DF text lane reproduces the single-node BM25 scores and the
// fusion math is already exact (truncate-before-fuse).
func TestHybridTextGlobalIDFBitIdentical(t *testing.T) {
	const P = 4
	// "fox" stems to docs 1 ("fox") and 5 ("foxes"), which land on DIFFERENT
	// partitions (p1, p2) under P=4: each shard sees local df("fox")=1 while the
	// global df=2, so global-IDF observably changes the text-lane score. The two fox
	// docs have DISTINCT BM25 scores (different docLen) ⇒ no text-lane tie, so RRF
	// rank assignment is deterministic and the fused score is itself bit-identical
	// (RRF amplifies tie-order differences, so a tie-free query is the clean hybrid
	// oracle). dog (with ties) is covered bit-exactly at the text-lane level by
	// TestSearchTextGlobalIDFBitIdentical.
	if a, b := ops.PartitionOf(1, P), ops.PartitionOf(5, P); a == b {
		t.Skipf("fox docs 1,5 share partition %d under P=%d; can't exercise global-vs-local df", a, P)
	}
	ctx := context.Background()

	s1 := newSingleEmbedded(t)
	waitLeaderEmbedded(t, s1)
	seedFullTextCollection(t, s1, "ft1", 1)

	sP := newSingleEmbedded(t)
	waitLeaderEmbedded(t, sP)
	seedFullTextCollection(t, sP, "ftP", P)

	q := "fox"
	dense := denseFor(1)

	oracle, _, err := s1.VectorHybridText(ctx, "ft1", dense, q, 10, VectorHybridOpts{Method: FusionRRF})
	if err != nil {
		t.Fatalf("P1 oracle HybridText: %v", err)
	}
	global, fm, err := sP.VectorHybridText(ctx, "ftP", dense, q, 10, VectorHybridOpts{Method: FusionRRF, GlobalIDF: true})
	if err != nil {
		t.Fatalf("P%d GlobalIDF HybridText: %v", P, err)
	}
	if fm.Degraded {
		t.Fatalf("P%d GlobalIDF HybridText degraded unexpectedly", P)
	}
	if len(global) != len(oracle) {
		t.Fatalf("GlobalIDF hybrid count=%d != oracle %d", len(global), len(oracle))
	}
	// Same id set, each with the SAME fused score (ties admit any order, so compare
	// by id); the ordered fused-score sequence must also match exactly.
	oScore := map[uint64]float32{}
	for _, r := range oracle {
		oScore[r.ID] = r.Score
	}
	for _, r := range global {
		os, ok := oScore[r.ID]
		if !ok {
			t.Fatalf("GlobalIDF hybrid id %d absent from oracle %v", r.ID, resultIDs(oracle))
		}
		if r.Score != os {
			t.Fatalf("GlobalIDF hybrid score for id %d = %v != oracle %v", r.ID, r.Score, os)
		}
	}
	for i := range oracle {
		if global[i].Score != oracle[i].Score {
			t.Fatalf("GlobalIDF hybrid ordered score[%d]=%v != oracle %v", i, global[i].Score, oracle[i].Score)
		}
	}
}

// TestBM25StatsPhase0Aggregation checks the phase-0 fan-out sums n/tokenTotal/df
// across partitions correctly. It compares the P>1 summed global stats against the
// per-partition stats gathered INDIVIDUALLY (each physical partition's
// vector_bm25_stats), and asserts N equals the full live corpus (all 8 ftDocs) — so
// the cross-shard sum reconstructs the whole-corpus df/n the single-node scorer uses.
func TestBM25StatsPhase0Aggregation(t *testing.T) {
	const P = 4
	if !dfSplitsAcrossPartitions(P) {
		t.Skipf("dog-bearing docs do not split across %d partitions", P)
	}

	eP := newSingleEmbedded(t).(*embedded)
	waitLeaderEmbedded(t, eP)
	seedFullTextCollection(t, eP, "ftP", P)

	q := "dog"
	gP, frP, err := eP.bm25StatsFanOut("ftP", P, 0, q, 0, 0, 0)
	if err != nil {
		t.Fatalf("P%d bm25StatsFanOut: %v", P, err)
	}
	if frP.Degraded {
		t.Fatalf("P%d stats degraded", P)
	}

	// Independently gather each physical partition's stats and SUM them by hand; the
	// fan-out's aggregate must equal this reference (no double-count, term-id keying
	// correct, disjoint shards).
	var refN int
	var refTok uint64
	refDF := map[uint32]int{}
	touchedParts := 0
	for p := 0; p < P; p++ {
		phys := string(ops.PartitionKeyGen("ftP", 0, p))
		raw, err := eP.node.CallPhysical(phys, "vector_bm25_stats", ops.EncodeBM25StatsArgs(phys, q, 0, 0, 0), false)
		if err != nil {
			t.Fatalf("partition %d stats: %v", p, err)
		}
		n, tok, df, err := ops.DecodeBM25StatsResult(raw)
		if err != nil {
			t.Fatalf("partition %d decode: %v", p, err)
		}
		if n > 0 {
			touchedParts++
		}
		refN += n
		refTok += tok
		for term, d := range df {
			refDF[term] += d
		}
	}
	if touchedParts < 2 {
		t.Fatalf("corpus landed on %d partitions; not a genuine multi-shard sum", touchedParts)
	}
	if gP.N != refN {
		t.Fatalf("summed N=%d != hand-summed %d", gP.N, refN)
	}
	if gP.N != len(ftDocs) {
		t.Fatalf("summed N=%d != full live corpus %d", gP.N, len(ftDocs))
	}
	wantAvgdl := float32(refTok) / float32(refN)
	if gP.Avgdl != wantAvgdl {
		t.Fatalf("summed avgdl=%v != %v", gP.Avgdl, wantAvgdl)
	}
	if len(gP.DF) != len(refDF) {
		t.Fatalf("df key-set mismatch: fanout=%v ref=%v", gP.DF, refDF)
	}
	for term, d := range refDF {
		if gP.DF[term] != d {
			t.Fatalf("df[%d]=%d != hand-summed %d", term, gP.DF[term], d)
		}
	}
}

// TestSearchTextGlobalIDFNetworked proves the GlobalIDF flag survives the networked
// path: the client encodes it in the vector_search_text REQUEST wire (a flags bit),
// the server's fanout dispatcher decodes it and runs the two-phase coordinator, and
// the P>1 result is bit-identical (by id→score) to an embedded P=1 oracle. This is
// the end-to-end proof that the flag reaches the server coordinator (no proto field).
func TestSearchTextGlobalIDFNetworked(t *testing.T) {
	const P = 4
	if !dfSplitsAcrossPartitions(P) {
		t.Skipf("dog docs do not split across %d partitions", P)
	}
	ctx := context.Background()

	// Embedded P=1 oracle.
	s1 := newSingleEmbedded(t)
	waitLeaderEmbedded(t, s1)
	seedFullTextCollection(t, s1, "ft1", 1)
	oracle, _, err := s1.VectorSearchText(ctx, "ft1", "dog", 10, VectorSearchOpts{})
	if err != nil {
		t.Fatalf("oracle: %v", err)
	}

	// Networked client → server coordinator, P=4 collection.
	reg := ops.NewRegistry()
	if err := ops.RegisterBuiltins(reg); err != nil {
		t.Fatal(err)
	}
	srv, err := NewDirectServer("127.0.0.1:0", DirectConfig{
		DataDir: t.TempDir(), Ops: reg, Cache: CacheConfig{NumShardsPerNode: 4},
	})
	if err != nil {
		t.Fatal(err)
	}
	defer srv.Close()
	store, err := NewClient(ClientConfig{Servers: []string{srv.Addr()}})
	if err != nil {
		t.Fatal(err)
	}
	defer store.Close()
	seedFullTextCollection(t, store, "ftP", P)

	global, fm, err := store.VectorSearchText(ctx, "ftP", "dog", 10, VectorSearchOpts{GlobalIDF: true})
	if err != nil {
		t.Fatalf("networked GlobalIDF: %v", err)
	}
	if fm.Degraded {
		t.Fatal("networked GlobalIDF degraded unexpectedly")
	}
	if len(global) != len(oracle) {
		t.Fatalf("networked count=%d != oracle %d", len(global), len(oracle))
	}
	oScore := map[uint64]float32{}
	for _, d := range oracle {
		oScore[d.ID] = d.Score
	}
	for _, d := range global {
		os, ok := oScore[d.ID]
		if !ok {
			t.Fatalf("networked id %d absent from oracle %v", d.ID, docIDs(oracle))
		}
		if d.Score != os {
			t.Fatalf("networked score for id %d = %v != oracle %v", d.ID, d.Score, os)
		}
	}
}

// TestGlobalIDFSinglePartitionIgnored confirms P==1 ignores the flag: the result is
// identical with GlobalIDF on and off (the single shard's local corpus IS global,
// so no phase 0 runs and no trailer rides).
func TestGlobalIDFSinglePartitionIgnored(t *testing.T) {
	ctx := context.Background()
	s := newSingleEmbedded(t)
	waitLeaderEmbedded(t, s)
	seedFullTextCollection(t, s, "ft", 1)

	q := "dog"
	off, _, err := s.VectorSearchText(ctx, "ft", q, 10, VectorSearchOpts{})
	if err != nil {
		t.Fatalf("off: %v", err)
	}
	on, _, err := s.VectorSearchText(ctx, "ft", q, 10, VectorSearchOpts{GlobalIDF: true})
	if err != nil {
		t.Fatalf("on: %v", err)
	}
	if !equalIDs(docIDs(on), docIDs(off)) {
		t.Fatalf("P1 GlobalIDF changed ids: on=%v off=%v", docIDs(on), docIDs(off))
	}
	for i := range off {
		if on[i].Score != off[i].Score {
			t.Fatalf("P1 GlobalIDF changed score[%d]: on=%v off=%v", i, on[i].Score, off[i].Score)
		}
	}
}
