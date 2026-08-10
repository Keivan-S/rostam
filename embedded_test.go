// SPDX-License-Identifier: Apache-2.0

package rostam

import (
	"context"
	"errors"
	"fmt"
	"math"
	"reflect"
	"sort"
	"strings"
	"sync"
	"testing"
	"time"

	"github.com/rostamlabs/rostam/ops"
	"github.com/rostamlabs/rostam/vector"
)

// newSingleEmbedded builds a single-node embedded Store suitable for unit
// tests. It uses a temp directory, bootstraps a fresh cluster, and
// registers the built-in ops. The Store is closed via t.Cleanup.
func newSingleEmbedded(t *testing.T) Store {
	t.Helper()
	dir := t.TempDir()
	reg := ops.NewRegistry()
	if err := ops.RegisterBuiltins(reg); err != nil {
		t.Fatalf("RegisterBuiltins: %v", err)
	}

	s, err := NewEmbedded(EmbeddedConfig{
		NodeID:    "test-node",
		DataDir:   dir,
		NumShards: 1,
		Bootstrap: true,
		Ops:       reg,
	})
	if err != nil {
		t.Fatalf("NewEmbedded: %v", err)
	}
	t.Cleanup(func() {
		if err := s.Close(); err != nil {
			t.Logf("embedded Close: %v", err)
		}
	})
	return s
}

// waitLeaderEmbedded spins until the node reports a non-empty leader address,
// ensuring Raft has elected a leader before test ops run.
func waitLeaderEmbedded(t *testing.T, s Store) {
	t.Helper()
	deadline := time.Now().Add(10 * time.Second)
	for time.Now().Before(deadline) {
		if s.LeaderAddr(nil) != "" {
			return
		}
		time.Sleep(20 * time.Millisecond)
	}
	t.Fatal("waitLeaderEmbedded: timed out waiting for leader election")
}

func TestEmbeddedPutGetRoundtrip(t *testing.T) {
	s := newSingleEmbedded(t)
	waitLeaderEmbedded(t, s)

	ctx := context.Background()
	key := []byte("hello")
	val := []byte("world")

	if err := s.Put(ctx, key, val, 0); err != nil {
		t.Fatalf("Put: %v", err)
	}

	got, err := s.Get(ctx, key)
	if err != nil {
		t.Fatalf("Get: %v", err)
	}
	if string(got) != string(val) {
		t.Fatalf("Get = %q, want %q", got, val)
	}
}

func TestEmbeddedGetMissingReturnsErrNotFound(t *testing.T) {
	s := newSingleEmbedded(t)
	waitLeaderEmbedded(t, s)

	ctx := context.Background()
	_, err := s.Get(ctx, []byte("no-such-key"))
	if err != ErrNotFound {
		t.Fatalf("Get missing key: got %v, want ErrNotFound", err)
	}
}

func TestEmbeddedDelReturnsBool(t *testing.T) {
	s := newSingleEmbedded(t)
	waitLeaderEmbedded(t, s)

	ctx := context.Background()
	key := []byte("del-me")

	// Not present yet — Del should return false.
	deleted, err := s.Del(ctx, key)
	if err != nil {
		t.Fatalf("Del (absent): %v", err)
	}
	if deleted {
		t.Fatal("Del absent key: want false, got true")
	}

	// Put then Del — should return true.
	if err := s.Put(ctx, key, []byte("v"), 0); err != nil {
		t.Fatalf("Put: %v", err)
	}
	deleted, err = s.Del(ctx, key)
	if err != nil {
		t.Fatalf("Del (present): %v", err)
	}
	if !deleted {
		t.Fatal("Del present key: want true, got false")
	}
}

func TestEmbeddedCallInvokesRegisteredOp(t *testing.T) {
	s := newSingleEmbedded(t)
	waitLeaderEmbedded(t, s)

	ctx := context.Background()
	key := []byte("call-key")
	val := []byte("call-val")

	// Put via Call (uses the built-in "put" op).
	_, err := s.Call(ctx, "put", ops.EncodePutArgs(key, val, 0))
	if err != nil {
		t.Fatalf("Call put: %v", err)
	}

	// Get via Call (uses the built-in "get" op).
	got, err := s.Call(ctx, "get", ops.EncodeKeyArgs(key))
	if err != nil {
		t.Fatalf("Call get: %v", err)
	}
	if string(got) != string(val) {
		t.Fatalf("Call get = %q, want %q", got, val)
	}
}

func TestEmbeddedPartitionedInsertSearchSingleNode(t *testing.T) {
	s := newSingleEmbedded(t)
	waitLeaderEmbedded(t, s)

	ctx := context.Background()
	if err := s.CreateCollection(ctx, "docs", VectorConfig{Dim: 4, M: 8, EfConstruction: 50, EfSearch: 32, Seed: 1, Metric: vector.L2, Partitions: 4}); err != nil {
		t.Fatalf("CreateCollection: %v", err)
	}
	for id := uint64(1); id <= 200; id++ {
		v := []float32{float32(id), 0, 0, 0}
		if err := s.VectorInsert(ctx, "docs", id, v); err != nil {
			t.Fatalf("insert %d: %v", id, err)
		}
	}
	res, err := s.VectorSearch(ctx, "docs", []float32{1, 0, 0, 0}, 5)
	if err != nil {
		t.Fatalf("VectorSearch: %v", err)
	}
	if len(res) == 0 || res[0].ID != 1 {
		t.Fatalf("top result = %+v, want ID 1 first", res)
	}

	// Prove routing actually happened: the logical "docs" collection must hold
	// no data (every insert was routed to a physical partition), and each
	// physical partition docs#p must hold a strict fraction of the 200 vectors.
	// Without partition routing all 200 would land in "docs" and these checks
	// fail. Use a raw single-Call search (no fan-out) per name.
	rawSearch := func(name string, k int) []VectorResult {
		t.Helper()
		body, err := s.Call(ctx, "vector_search", ops.EncodeVectorSearchArgs(name, k, []float32{1, 0, 0, 0}))
		if err != nil {
			t.Fatalf("raw search %q: %v", name, err)
		}
		out, err := ops.DecodeVectorSearchResults(body)
		if err != nil {
			t.Fatalf("decode raw search %q: %v", name, err)
		}
		return out
	}
	if got := rawSearch("docs", 250); len(got) != 0 {
		t.Fatalf("logical collection docs held %d vectors, want 0 (data should be partitioned)", len(got))
	}
	total := 0
	for p := 0; p < 4; p++ {
		phys := string(ops.PartitionKey("docs", p))
		n := len(rawSearch(phys, 250))
		if n == 0 || n >= 200 {
			t.Fatalf("partition %s held %d vectors, want a strict fraction of 200", phys, n)
		}
		total += n
	}
	if total != 200 {
		t.Fatalf("partitions held %d vectors total, want 200", total)
	}
}

// sameFusedResults reports whether two fused hybrid result lists are equivalent:
// same length, same ID set, and per-ID Score within ~1e-5. It is tie-order
// tolerant (compares by ID, not slice position) because fan-out fuses via a
// different code path than single-partition and float32 accumulation order may
// differ slightly across the two paths.
func sameFusedResults(a, b []VectorResult) bool {
	if len(a) != len(b) {
		return false
	}
	bScore := make(map[uint64]float32, len(b))
	for _, r := range b {
		bScore[r.ID] = r.Score
	}
	for _, r := range a {
		bs, ok := bScore[r.ID]
		if !ok {
			return false
		}
		d := r.Score - bs
		if d < 0 {
			d = -d
		}
		if d > 1e-5 {
			return false
		}
	}
	return true
}

func idsOf(rs []VectorResult) []uint64 {
	out := make([]uint64, len(rs))
	for i, r := range rs {
		out[i] = r.ID
	}
	return out
}

// TestEmbeddedHybridFanOutSingleNodeExact proves exact hybrid fan-out: hybrid
// search over a P=4 partitioned collection (union per-partition lanes, truncate
// to global denseK/sparseK, fuse once) reproduces the P=1 single-partition oracle
// EXACTLY for both RRF and Weighted fusion.
func TestEmbeddedHybridFanOutSingleNodeExact(t *testing.T) {
	s := newSingleEmbedded(t)
	waitLeaderEmbedded(t, s)

	ctx := context.Background()
	if err := s.CreateCollection(ctx, "docs", VectorConfig{Dim: 4, M: 8, EfConstruction: 50, EfSearch: 32, Seed: 1, Metric: vector.L2, Partitions: 4}); err != nil {
		t.Fatalf("CreateCollection (P=4): %v", err)
	}
	if err := s.CreateCollection(ctx, "docs1", VectorConfig{Dim: 4, M: 8, EfConstruction: 50, EfSearch: 32, Seed: 1, Metric: vector.L2, Partitions: 1}); err != nil {
		t.Fatalf("CreateCollection (P=1): %v", err)
	}
	for id := uint64(1); id <= 200; id++ {
		v := []float32{float32(id), 0, 0, 0}
		sp := VectorSparse{Indices: []uint32{uint32(id % 7)}, Values: []float32{1}}
		for _, col := range []string{"docs", "docs1"} {
			if err := s.VectorInsertExt(ctx, col, id, v, VectorInsertOpts{Sparse: sp}); err != nil {
				t.Fatalf("VectorInsertExt %s/%d: %v", col, id, err)
			}
		}
	}
	// The query sits BELOW the smallest id (0.5 < 1), so dense distances are
	// strictly increasing in id with NO ties. Equal-distance dense ties would be
	// resolved by arbitrary HNSW heap order (the single-graph oracle itself breaks
	// such ties non-deterministically), so a tie-free query is what makes the
	// fan-out-vs-oracle equality EXACT rather than tie-order-dependent.
	query := []float32{0.5, 0, 0, 0}
	qs := VectorSparse{Indices: []uint32{3}, Values: []float32{1}}
	for _, method := range []FusionMethod{FusionRRF, FusionWeighted} {
		opts := VectorHybridOpts{Sparse: qs, Method: method}
		got, _, err := s.VectorHybridSearch(ctx, "docs", query, 10, opts)
		if err != nil {
			t.Fatalf("method=%v: fan-out hybrid: %v", method, err)
		}
		want, _, err := s.VectorHybridSearch(ctx, "docs1", query, 10, opts)
		if err != nil {
			t.Fatalf("method=%v: single-partition hybrid: %v", method, err)
		}
		if !sameFusedResults(got, want) {
			t.Fatalf("method=%v: fan-out %v != single-partition %v", method, idsOf(got), idsOf(want))
		}
	}
}

func TestEmbeddedIsLeaderTrueOnSingleNode(t *testing.T) {
	s := newSingleEmbedded(t)
	waitLeaderEmbedded(t, s)

	// On a single-node cluster, this node must be the leader for any key.
	if !s.IsLeader([]byte("any-key")) {
		t.Fatal("IsLeader: want true on single-node cluster, got false")
	}
}

func TestEmbeddedLeaderAddrNonEmptyOnSingleNode(t *testing.T) {
	s := newSingleEmbedded(t)
	waitLeaderEmbedded(t, s)

	addr := s.LeaderAddr([]byte("any-key"))
	if addr == "" {
		t.Fatal("LeaderAddr: want non-empty on single-node cluster, got empty")
	}
}

func TestEmbeddedCloseIsIdempotent(t *testing.T) {
	dir := t.TempDir()
	reg := ops.NewRegistry()
	if err := ops.RegisterBuiltins(reg); err != nil {
		t.Fatalf("RegisterBuiltins: %v", err)
	}
	s, err := NewEmbedded(EmbeddedConfig{
		NodeID:    "test-node-close",
		DataDir:   dir,
		NumShards: 1,
		Bootstrap: true,
		Ops:       reg,
	})
	if err != nil {
		t.Fatalf("NewEmbedded: %v", err)
	}

	if err := s.Close(); err != nil {
		t.Fatalf("Close (first): %v", err)
	}
	if err := s.Close(); err != nil {
		t.Fatalf("Close (second): %v", err)
	}
}

// TestEmbeddedUnpartitionedOpsWork guards the P<=1 invariant: on an
// unpartitioned collection the ops that gained cross-shard fan-out (search_docs,
// scroll, hybrid, delete_by_filter) must still take the single-partition path and
// succeed. (Their partitioned fan-out is covered by the dedicated *FanOut* tests;
// every vector op now works on partitioned collections, so the former loud-fail
// guard — ErrPartitionedUnsupported — no longer fires for any of them.)
func TestEmbeddedUnpartitionedOpsWork(t *testing.T) {
	s := newSingleEmbedded(t)
	waitLeaderEmbedded(t, s)

	ctx := context.Background()
	q := []float32{1, 0, 0, 0}

	if err := s.CreateCollection(ctx, "plain", VectorConfig{Dim: 4, M: 8, EfConstruction: 50, EfSearch: 32, Seed: 1, Metric: vector.L2}); err != nil {
		t.Fatalf("CreateCollection (plain): %v", err)
	}
	for id := uint64(1); id <= 10; id++ {
		if err := s.VectorUpsert(ctx, "plain", id, []float32{float32(id), 0, 0, 0}, "content", VectorInsertOpts{}); err != nil {
			t.Fatalf("upsert %d: %v", id, err)
		}
	}

	t.Run("search_docs", func(t *testing.T) {
		docs, _, err := s.VectorSearchDocs(ctx, "plain", q, 5, VectorSearchOpts{})
		if err != nil {
			t.Fatalf("VectorSearchDocs (unpartitioned): %v", err)
		}
		if len(docs) == 0 {
			t.Fatalf("VectorSearchDocs (unpartitioned) returned 0 docs, want >0")
		}
	})
	t.Run("scroll", func(t *testing.T) {
		docs, _, _, err := s.VectorScroll(ctx, "plain", VectorFilter{}, 100, VectorScrollOpts{})
		if err != nil {
			t.Fatalf("VectorScroll (unpartitioned): %v", err)
		}
		if len(docs) == 0 {
			t.Fatalf("VectorScroll (unpartitioned) returned 0 docs, want >0")
		}
	})
	t.Run("hybrid", func(t *testing.T) {
		if _, _, err := s.VectorHybridSearch(ctx, "plain", q, 5, VectorHybridOpts{}); err != nil {
			t.Fatalf("VectorHybridSearch (unpartitioned): %v", err)
		}
	})
	t.Run("delete_by_filter", func(t *testing.T) {
		f := VectorFilter{Op: vector.FilterEq, Field: "tenant", Value: vector.NewString("acme")}
		if _, err := s.VectorDeleteByFilter(ctx, "plain", f); err != nil {
			t.Fatalf("VectorDeleteByFilter (unpartitioned): %v", err)
		}
	})
}

// TestEmbeddedDeleteByFilterFanOutSingleNode verifies delete_by_filter fans out
// across all partitions (P>1), sums the per-partition deleted counts, rejects an
// empty filter (parity with single-partition vector.ErrEmptyFilter), and is
// idempotent (re-running the same filter deletes nothing more).
func TestEmbeddedDeleteByFilterFanOutSingleNode(t *testing.T) {
	e := newSingleEmbedded(t)
	waitLeaderEmbedded(t, e)
	ctx := context.Background()
	if err := e.CreateCollection(ctx, "docs", VectorConfig{Dim: 4, Metric: vector.L2, M: 8, EfConstruction: 50, EfSearch: 64, Seed: 1, Partitions: 4}); err != nil {
		t.Fatal(err)
	}
	for id := uint64(1); id <= 200; id++ {
		v := []float32{float32(id), 0, 0, 0}
		md := vector.Metadata{"even": vector.NewBool(id%2 == 0)}
		if err := e.VectorInsertExt(ctx, "docs", id, v, VectorInsertOpts{Metadata: md}); err != nil {
			t.Fatal(err)
		}
	}
	// Empty filter rejected (parity with single-partition ErrEmptyFilter).
	if _, err := e.VectorDeleteByFilter(ctx, "docs", VectorFilter{}); !errors.Is(err, vector.ErrEmptyFilter) {
		t.Fatalf("empty filter err = %v, want ErrEmptyFilter", err)
	}
	even := VectorFilter{Op: vector.FilterEq, Field: "even", Value: vector.NewBool(true)}
	n, err := e.VectorDeleteByFilter(ctx, "docs", even)
	if err != nil {
		t.Fatal(err)
	}
	if n != 100 {
		t.Fatalf("deleted %d, want 100", n)
	}
	// Survivors: 100 odd docs, none even.
	rest, _, _, err := e.VectorScroll(ctx, "docs", VectorFilter{}, 0, VectorScrollOpts{})
	if err != nil {
		t.Fatal(err)
	}
	if len(rest) != 100 {
		t.Fatalf("after delete: %d docs remain, want 100", len(rest))
	}
	for _, d := range rest {
		if d.ID%2 == 0 {
			t.Fatalf("even id %d survived delete", d.ID)
		}
	}
	// Idempotent: re-delete removes nothing more.
	n2, err := e.VectorDeleteByFilter(ctx, "docs", even)
	if err != nil {
		t.Fatal(err)
	}
	if n2 != 0 {
		t.Fatalf("idempotent re-delete removed %d, want 0", n2)
	}
}

// sameGroupsResults reports whether two group-search result lists are equivalent:
// same length, same ordered group keys, and the same ordered hit IDs per group.
// Group order and per-group hit order are deterministic for a tie-free dense
// query, so equality is exact (not order-tolerant) — the proof that group
// fan-out reproduces the single-partition oracle exactly.
func sameGroupsResults(a, b []VectorGroup) bool {
	if len(a) != len(b) {
		return false
	}
	for i := range a {
		if !a[i].Key.Equal(b[i].Key) {
			return false
		}
		if len(a[i].Hits) != len(b[i].Hits) {
			return false
		}
		for j := range a[i].Hits {
			if a[i].Hits[j].ID != b[i].Hits[j].ID {
				return false
			}
		}
	}
	return true
}

// TestEmbeddedGroupFanOutSingleNodeExact proves exact group-search fan-out:
// group search over a P=4 partitioned collection (union each partition's
// candidates, truncate to the global top-fetchK by distance, GroupDocuments
// once) reproduces the P=1 single-partition oracle EXACTLY.
func TestEmbeddedGroupFanOutSingleNodeExact(t *testing.T) {
	s := newSingleEmbedded(t)
	waitLeaderEmbedded(t, s)
	ctx := context.Background()

	if err := s.CreateCollection(ctx, "docs", VectorConfig{Dim: 4, M: 8, EfConstruction: 50, EfSearch: 32, Seed: 1, Metric: vector.L2, Partitions: 4}); err != nil {
		t.Fatalf("CreateCollection docs: %v", err)
	}
	if err := s.CreateCollection(ctx, "docs1", VectorConfig{Dim: 4, M: 8, EfConstruction: 50, EfSearch: 32, Seed: 1, Metric: vector.L2, Partitions: 1}); err != nil {
		t.Fatalf("CreateCollection docs1: %v", err)
	}
	for id := uint64(1); id <= 200; id++ {
		v := []float32{float32(id), 0, 0, 0}
		md := VectorMetadata{"doc": vector.NewInt(int64(id % 20))}
		for _, col := range []string{"docs", "docs1"} {
			if err := s.VectorInsertExt(ctx, col, id, v, VectorInsertOpts{Metadata: md}); err != nil {
				t.Fatalf("VectorInsertExt %s/%d: %v", col, id, err)
			}
		}
	}

	query := []float32{0.5, 0, 0, 0} // tie-free dense order (below smallest id)
	opts := VectorGroupOpts{GroupBy: "doc", GroupSize: 3}
	got, _, err := s.VectorSearchGroups(ctx, "docs", query, 5, opts)
	if err != nil {
		t.Fatalf("VectorSearchGroups (P=4): %v", err)
	}
	want, _, err := s.VectorSearchGroups(ctx, "docs1", query, 5, opts)
	if err != nil {
		t.Fatalf("VectorSearchGroups (P=1 oracle): %v", err)
	}
	if len(got) == 0 {
		t.Fatalf("group fan-out returned 0 groups")
	}
	if !sameGroupsResults(got, want) {
		t.Fatalf("group fan-out != single-partition oracle\n got=%+v\nwant=%+v", got, want)
	}
}

// TestEmbeddedGroupFanOutNonPositiveK asserts that groupFanOut matches the
// single-partition SearchGroups behaviour for non-positive k: no panic, no
// error, and a nil/empty result.  The collection is partitioned (P=4) so the
// fan-out path (groupFanOut) is exercised — that is exactly where a negative k
// would trigger make([]Group, 0, k) with a negative cap and panic without the
// guard.
func TestEmbeddedGroupFanOutNonPositiveK(t *testing.T) {
	s := newSingleEmbedded(t)
	waitLeaderEmbedded(t, s)
	ctx := context.Background()

	if err := s.CreateCollection(ctx, "docs", VectorConfig{Dim: 4, M: 8, EfConstruction: 50, EfSearch: 32, Seed: 1, Metric: vector.L2, Partitions: 4}); err != nil {
		t.Fatalf("CreateCollection docs: %v", err)
	}
	for id := uint64(1); id <= 20; id++ {
		v := []float32{float32(id), 0, 0, 0}
		md := VectorMetadata{"doc": vector.NewInt(int64(id % 5))}
		if err := s.VectorInsertExt(ctx, "docs", id, v, VectorInsertOpts{Metadata: md}); err != nil {
			t.Fatalf("VectorInsertExt docs/%d: %v", id, err)
		}
	}

	for _, k := range []int{0, -1} {
		got, _, err := s.VectorSearchGroups(ctx, "docs", []float32{0.5, 0, 0, 0}, k, VectorGroupOpts{GroupBy: "doc", GroupSize: 3})
		if err != nil {
			t.Fatalf("k=%d: unexpected error %v", k, err)
		}
		if len(got) != 0 {
			t.Fatalf("k=%d: got %d groups, want 0", k, len(got))
		}
	}
}

// idSet returns the set of distinct doc IDs in docs.
func idSet(docs []VectorDocument) map[uint64]bool {
	m := map[uint64]bool{}
	for _, d := range docs {
		m[d.ID] = true
	}
	return m
}

// TestEmbeddedScrollFanOutSingleNode proves scroll fan-out completeness over a
// P=4 partitioned collection on a single node:
//   - full scroll (limit=0) returns every inserted doc exactly once
//   - filtered scroll returns only the matching subset
//   - limited scroll caps the result to exactly limit docs
func TestEmbeddedScrollFanOutSingleNode(t *testing.T) {
	e := newSingleEmbedded(t)
	waitLeaderEmbedded(t, e)
	ctx := context.Background()

	if err := e.CreateCollection(ctx, "docs", VectorConfig{Dim: 4, M: 8, EfConstruction: 50, EfSearch: 32, Seed: 1, Metric: vector.L2, Partitions: 4}); err != nil {
		t.Fatalf("CreateCollection: %v", err)
	}
	for id := uint64(1); id <= 200; id++ {
		v := []float32{float32(id), 0, 0, 0}
		md := VectorMetadata{"even": vector.NewBool(id%2 == 0)}
		if err := e.VectorInsertExt(ctx, "docs", id, v, VectorInsertOpts{Metadata: md}); err != nil {
			t.Fatalf("VectorInsertExt %d: %v", id, err)
		}
	}

	// Full scroll (limit=0): every doc exactly once.
	all, _, _, err := e.VectorScroll(ctx, "docs", VectorFilter{}, 0, VectorScrollOpts{})
	if err != nil {
		t.Fatal(err)
	}
	if len(all) != 200 || len(idSet(all)) != 200 {
		t.Fatalf("full scroll: %d docs (%d distinct), want 200", len(all), len(idSet(all)))
	}

	// Filtered scroll: even-only -> 100.
	evenFilter := VectorFilter{Op: vector.FilterEq, Field: "even", Value: vector.NewBool(true)}
	even, _, _, err := e.VectorScroll(ctx, "docs", evenFilter, 0, VectorScrollOpts{})
	if err != nil {
		t.Fatalf("filtered scroll: %v", err)
	}
	if len(idSet(even)) != 100 {
		t.Fatalf("filtered scroll: %d distinct, want 100", len(idSet(even)))
	}

	// Limited scroll: exactly limit distinct docs.
	lim, _, _, err := e.VectorScroll(ctx, "docs", VectorFilter{}, 10, VectorScrollOpts{})
	if err != nil {
		t.Fatalf("limited scroll: %v", err)
	}
	if len(lim) != 10 || len(idSet(lim)) != 10 {
		t.Fatalf("limited scroll: %d (%d distinct), want 10", len(lim), len(idSet(lim)))
	}
}

// TestEmbeddedSearchDegraded proves the fan-out degraded signal surfaces through
// VectorSearchExt's FanMeta third return. It builds a P=4 partitioned collection,
// makes one partition unreachable by dropping its physical collection (so the
// coordinator's CallPhysical -> vector_search returns "unknown collection"), then:
//   - Partial mode (default): err==nil, meta.Degraded==true, meta.Missing==[]int{2},
//     and reachable partitions still return results.
//   - Fail mode: the whole query errors.
//   - A fresh, fully-healthy P=4 collection: meta.Degraded==false && meta.Missing==nil.
func TestEmbeddedSearchDegraded(t *testing.T) {
	emb := newSingleEmbedded(t)
	waitLeaderEmbedded(t, emb)
	ctx := context.Background()

	const P = 4
	cfg := VectorConfig{Dim: 4, Metric: vector.L2, M: 8, EfConstruction: 50, EfSearch: 64, Seed: 1, Partitions: P}

	mkColl := func(name string) {
		if err := emb.CreateCollection(ctx, name, cfg); err != nil {
			t.Fatalf("CreateCollection %q: %v", name, err)
		}
		// ~80 tie-free vectors: strictly increasing L2 distance from the query by id.
		for id := uint64(1); id <= 80; id++ {
			v := []float32{float32(id), 0, 0, 0}
			if err := emb.VectorInsert(ctx, name, id, v); err != nil {
				t.Fatalf("VectorInsert %q id=%d: %v", name, id, err)
			}
		}
	}

	const coll = "degraded"
	mkColl(coll)
	query := []float32{0.5, 0, 0, 0} // tie-free: nearest is id=1, then 2, 3, ...

	// Make partition 2 unreachable: drop its physical collection. The coordinator
	// still fans out to all P partitions; partition 2's vector_search then fails
	// with "unknown collection", which CallPhysical propagates.
	if _, err := emb.Call(ctx, "vector_drop_collection",
		ops.EncodeDropCollectionArgs(string(ops.PartitionKeyGen(coll, 0, 2)))); err != nil {
		t.Fatalf("drop partition 2: %v", err)
	}

	// Partial mode (default): degraded, partition 2 missing, partial results.
	res, meta, err := emb.VectorSearchExt(ctx, coll, query, 10, VectorSearchOpts{})
	if err != nil {
		t.Fatalf("Partial VectorSearchExt: unexpected err: %v", err)
	}
	if !meta.Degraded {
		t.Fatalf("Partial: meta.Degraded = false, want true")
	}
	if !reflect.DeepEqual(meta.Missing, []int{2}) {
		t.Fatalf("Partial: meta.Missing = %v, want [2]", meta.Missing)
	}
	if len(res) == 0 {
		t.Fatalf("Partial: expected partial results from reachable partitions, got none")
	}

	// Fail mode: the unreachable partition errors the whole query.
	if _, _, err := emb.VectorSearchExt(ctx, coll, query, 10,
		VectorSearchOpts{OnPartitionUnavailable: 1}); err == nil {
		t.Fatalf("Fail mode: expected error from unreachable partition, got nil")
	}

	// Fresh, fully-healthy P=4 collection: not degraded, nil Missing.
	const healthy = "healthy"
	mkColl(healthy)
	_, meta, err = emb.VectorSearchExt(ctx, healthy, query, 10, VectorSearchOpts{})
	if err != nil {
		t.Fatalf("healthy VectorSearchExt: %v", err)
	}
	if meta.Degraded || meta.Missing != nil {
		t.Fatalf("healthy: meta = %+v, want {Degraded:false Missing:nil}", meta)
	}
}

// TestEmbeddedConsistencyOptsFailMode proves that ReadConsistency /
// OnPartitionUnavailable are honored across the hybrid, groups, scroll, and MV
// fan-out READ paths, mirroring TestEmbeddedSearchDegraded (dense). For each op,
// dropping a physical partition makes Partial mode report Degraded with non-empty
// partial results while Fail mode (OnPartitionUnavailable=1) errors the whole
// query; a fully-healthy collection stays not-degraded. This is the new Fail-mode
// capability for MV, and parity for the other three.
func TestEmbeddedConsistencyOptsFailMode(t *testing.T) {
	emb := newSingleEmbedded(t)
	waitLeaderEmbedded(t, emb)
	ctx := context.Background()

	const P = 4
	cfg := VectorConfig{Dim: 4, Metric: vector.L2, M: 8, EfConstruction: 50, EfSearch: 64, Seed: 1, Partitions: P}

	// --- Dense-family collection (hybrid, groups, scroll) ---------------------
	mkDense := func(name string) {
		if err := emb.CreateCollection(ctx, name, cfg); err != nil {
			t.Fatalf("CreateCollection %q: %v", name, err)
		}
		for id := uint64(1); id <= 80; id++ {
			v := []float32{float32(id), 0, 0, 0}
			sp := VectorSparse{Indices: []uint32{uint32(id % 7)}, Values: []float32{1}}
			md := VectorMetadata{"doc": vector.NewInt(int64(id % 20))}
			if err := emb.VectorInsertExt(ctx, name, id, v,
				VectorInsertOpts{Sparse: sp, Metadata: md}); err != nil {
				t.Fatalf("VectorInsertExt %q id=%d: %v", name, id, err)
			}
		}
	}

	const dcoll = "co-dense"
	mkDense(dcoll)
	if _, err := emb.Call(ctx, "vector_drop_collection",
		ops.EncodeDropCollectionArgs(string(ops.PartitionKeyGen(dcoll, 0, 2)))); err != nil {
		t.Fatalf("drop dense partition 2: %v", err)
	}

	query := []float32{0.5, 0, 0, 0}
	qs := VectorSparse{Indices: []uint32{3}, Values: []float32{1}}

	// hybrid
	if res, meta, err := emb.VectorHybridSearch(ctx, dcoll, query, 10,
		VectorHybridOpts{Sparse: qs}); err != nil {
		t.Fatalf("hybrid Partial: unexpected err: %v", err)
	} else if !meta.Degraded {
		t.Fatalf("hybrid Partial: meta.Degraded = false, want true")
	} else if len(res) == 0 {
		t.Fatalf("hybrid Partial: expected partial results, got none")
	}
	if _, _, err := emb.VectorHybridSearch(ctx, dcoll, query, 10,
		VectorHybridOpts{Sparse: qs, OnPartitionUnavailable: 1}); err == nil {
		t.Fatalf("hybrid Fail: expected error from unreachable partition, got nil")
	}

	// groups
	gopts := VectorGroupOpts{GroupBy: "doc", GroupSize: 3}
	if res, meta, err := emb.VectorSearchGroups(ctx, dcoll, query, 5, gopts); err != nil {
		t.Fatalf("groups Partial: unexpected err: %v", err)
	} else if !meta.Degraded {
		t.Fatalf("groups Partial: meta.Degraded = false, want true")
	} else if len(res) == 0 {
		t.Fatalf("groups Partial: expected partial results, got none")
	}
	failGopts := gopts
	failGopts.OnPartitionUnavailable = 1
	if _, _, err := emb.VectorSearchGroups(ctx, dcoll, query, 5, failGopts); err == nil {
		t.Fatalf("groups Fail: expected error from unreachable partition, got nil")
	}

	// scroll
	if res, meta, _, err := emb.VectorScroll(ctx, dcoll, VectorFilter{}, 0, VectorScrollOpts{}); err != nil {
		t.Fatalf("scroll Partial: unexpected err: %v", err)
	} else if !meta.Degraded {
		t.Fatalf("scroll Partial: meta.Degraded = false, want true")
	} else if len(res) == 0 {
		t.Fatalf("scroll Partial: expected partial results, got none")
	}
	if _, _, _, err := emb.VectorScroll(ctx, dcoll, VectorFilter{}, 0,
		VectorScrollOpts{OnPartitionUnavailable: 1}); err == nil {
		t.Fatalf("scroll Fail: expected error from unreachable partition, got nil")
	}

	// Healthy dense collection: not degraded.
	const dhealthy = "co-dense-ok"
	mkDense(dhealthy)
	if _, meta, _, err := emb.VectorScroll(ctx, dhealthy, VectorFilter{}, 0, VectorScrollOpts{}); err != nil {
		t.Fatalf("healthy scroll: %v", err)
	} else if meta.Degraded || meta.Missing != nil {
		t.Fatalf("healthy scroll: meta = %+v, want not degraded", meta)
	}

	// --- MV collection (gains a Fail mode) -----------------------------------
	const mvcoll = "co-mv"
	mkMV := func(name string) {
		must(t, emb.VectorMVCreateCollection(ctx, name, MultiVectorConfig{Dim: 4, Partitions: P}))
		for id := 0; id < 80; id++ {
			must(t, emb.VectorMVAdd(ctx, name, uint64(id), [][]float32{mvTokenAt(id)}, nil))
		}
	}
	mkMV(mvcoll)
	// MV partitions are dropped via vector_mv_drop_collection + MV-delete args.
	if _, err := emb.Call(ctx, "vector_mv_drop_collection",
		ops.EncodeMVDeleteArgs(string(ops.PartitionKeyGen(mvcoll, 0, 2)), 0)); err != nil {
		t.Fatalf("drop mv partition 2: %v", err)
	}
	mvQuery := [][]float32{mvTokenAt(17)}
	if res, meta, err := emb.VectorMVSearch(ctx, mvcoll, mvQuery, 10,
		MultiSearchOpts{CandidatesPerToken: 100}); err != nil {
		t.Fatalf("mv Partial: unexpected err: %v", err)
	} else if !meta.Degraded {
		t.Fatalf("mv Partial: meta.Degraded = false, want true")
	} else if len(res) == 0 {
		t.Fatalf("mv Partial: expected partial results, got none")
	}
	if _, _, err := emb.VectorMVSearch(ctx, mvcoll, mvQuery, 10,
		MultiSearchOpts{CandidatesPerToken: 100, OnPartitionUnavailable: 1}); err == nil {
		t.Fatalf("mv Fail: expected error from unreachable partition, got nil (MV must now honor Fail mode)")
	}

	// Healthy MV collection: not degraded.
	const mvhealthy = "co-mv-ok"
	mkMV(mvhealthy)
	if _, meta, err := emb.VectorMVSearch(ctx, mvhealthy, mvQuery, 10,
		MultiSearchOpts{CandidatesPerToken: 100}); err != nil {
		t.Fatalf("healthy mv: %v", err)
	} else if meta.Degraded || meta.Missing != nil {
		t.Fatalf("healthy mv: meta = %+v, want not degraded", meta)
	}
}

// TestEmbeddedSearchDocsFanOutSingleNodeExact proves that search_docs fan-out
// over a P=4 partitioned collection reproduces the P=1 single-partition oracle
// EXACTLY — including Content and Distance — for both the unfiltered and
// filtered cases.
func TestEmbeddedSearchDocsFanOutSingleNodeExact(t *testing.T) {
	e := newSingleEmbedded(t)
	waitLeaderEmbedded(t, e)
	ctx := context.Background()
	mustCreate := func(name string, p int) {
		if err := e.CreateCollection(ctx, name, VectorConfig{Dim: 4, Metric: vector.L2, M: 8, EfConstruction: 50, EfSearch: 64, Seed: 1, Partitions: p}); err != nil {
			t.Fatal(err)
		}
	}
	mustCreate("docs", 4)
	mustCreate("docs1", 1)
	for id := uint64(1); id <= 200; id++ {
		v := []float32{float32(id), 0, 0, 0}
		content := fmt.Sprintf("doc-%d", id)
		md := vector.Metadata{"even": vector.NewBool(id%2 == 0)}
		for _, col := range []string{"docs", "docs1"} {
			if err := e.VectorUpsert(ctx, col, id, v, content, VectorInsertOpts{Metadata: md}); err != nil {
				t.Fatal(err)
			}
		}
	}
	query := []float32{0.5, 0, 0, 0} // tie-free: strictly increasing L2 distance by id
	for _, f := range []VectorFilter{{}, {Op: vector.FilterEq, Field: "even", Value: vector.NewBool(true)}} {
		got, _, err := e.VectorSearchDocs(ctx, "docs", query, 10, VectorSearchOpts{Filter: f})
		if err != nil {
			t.Fatal(err)
		}
		want, _, err := e.VectorSearchDocs(ctx, "docs1", query, 10, VectorSearchOpts{Filter: f})
		if err != nil {
			t.Fatal(err)
		}
		if !sameDocs(got, want) {
			t.Fatalf("filter=%+v: search_docs fan-out != single-partition\n got=%v\nwant=%v", f, docIDs(got), docIDs(want))
		}
		if len(got) == 0 {
			t.Fatalf("filter=%+v: empty (vacuous)", f)
		}
	}
}

func sameDocs(a, b []VectorDocument) bool {
	if len(a) != len(b) {
		return false
	}
	for i := range a {
		if a[i].ID != b[i].ID || a[i].Content != b[i].Content || a[i].Distance != b[i].Distance {
			return false
		}
	}
	return true
}

func docIDs(d []VectorDocument) []uint64 {
	out := make([]uint64, len(d))
	for i := range d {
		out[i] = d[i].ID
	}
	return out
}

// TestEmbeddedCatalogVisibleOnOtherNode proves the multi-node durable catalog:
// a partitioned collection created via node 0's coordinator becomes visible (its
// partition count P) from a node that did NOT create it, because the partition
// count is committed to the meta-Raft catalog rather than node 0's in-process map.
func TestEmbeddedCatalogVisibleOnOtherNode(t *testing.T) {
	stores := newInmemEmbeddedCluster(t, 3, 8)
	ctx := context.Background()

	retryUntil(t, "CreateCollection docs", func() error {
		return stores[0].CreateCollection(ctx, "docs", VectorConfig{
			Dim: 4, M: 8, EfConstruction: 50, EfSearch: 32, Seed: 1, Metric: vector.L2, Partitions: 4,
		})
	})

	// Node 1 (did NOT create it) must see P=4 via its own embedded catalog.
	e1, ok := stores[1].(*embedded)
	if !ok {
		t.Fatalf("store 1 is %T, want *embedded", stores[1])
	}
	deadline := time.Now().Add(5 * time.Second)
	for time.Now().Before(deadline) {
		if p, _, ok := e1.catalog.PartitionsGen("docs"); ok && p == 4 {
			return
		}
		time.Sleep(20 * time.Millisecond)
	}
	p, _, ok := e1.catalog.PartitionsGen("docs")
	t.Fatalf("node 1 catalog docs = (%d,%v), want (4,true)", p, ok)
}

func TestCreateCollectionRejectsReservedNameChars(t *testing.T) {
	e := newSingleEmbedded(t)
	waitLeaderEmbedded(t, e)
	ctx := context.Background()
	// Use a fully valid config so that only the name — not config validation —
	// can cause a rejection.
	goodCfg := VectorConfig{Dim: 4, Metric: vector.L2, M: 8, EfConstruction: 50, EfSearch: 64, Seed: 1}
	for _, bad := range []string{"foo#0", "foo@2", "a#b", "x@1#0", "t/foo#3"} {
		if err := e.CreateCollection(ctx, bad, goodCfg); err == nil {
			t.Errorf("CreateCollection(%q) should be rejected (reserved # or @)", bad)
		}
	}
	// A normal name still works.
	if err := e.CreateCollection(ctx, "docs", goodCfg); err != nil {
		t.Fatalf("normal create failed: %v", err)
	}
	// A PARTITIONED create still works (its internal physical names contain '#'/'@'
	// but go through the op directly, bypassing the user-facing guard).
	if err := e.CreateCollection(ctx, "parts", VectorConfig{Dim: 4, Metric: vector.L2, M: 8, EfConstruction: 50, EfSearch: 64, Seed: 1, Partitions: 4}); err != nil {
		t.Fatalf("partitioned create failed: %v", err)
	}
}

// TestEmbeddedSearchRoutesByGeneration proves the SEARCH fan-out path reads the
// catalog generation and fans out to the gen-1 physical partitions. Without gen
// threading, fan-out would query the gen-0 names ("docs#p") and find nothing.
func TestEmbeddedSearchRoutesByGeneration(t *testing.T) {
	e := newSingleEmbedded(t)
	waitLeaderEmbedded(t, e)
	ctx := context.Background()
	ee := e.(*embedded)
	const P = 4
	cfg := VectorConfig{Dim: 4, Metric: vector.L2, M: 8, EfConstruction: 50, EfSearch: 64, Seed: 1}
	for p := 0; p < P; p++ {
		if _, err := ee.Call(ctx, "vector_create_collection", ops.EncodeCreateCollectionArgs(string(ops.PartitionKeyGen("docs", 1, p)), cfg)); err != nil {
			t.Fatal(err)
		}
	}
	if err := ee.catalog.SetPartitionsGen("docs", P, 1); err != nil {
		t.Fatal(err)
	}
	for id := uint64(1); id <= 50; id++ {
		if err := e.VectorInsert(ctx, "docs", id, []float32{float32(id), 0, 0, 0}); err != nil {
			t.Fatal(err)
		}
	}
	res, err := e.VectorSearch(ctx, "docs", []float32{1, 0, 0, 0}, 5)
	if err != nil {
		t.Fatal(err)
	}
	if len(res) == 0 || res[0].ID != 1 {
		t.Fatalf("gen-1 fan-out search wrong: %+v", res)
	}
}

// TestEmbeddedPointOpRoutesByGeneration proves the point-op routing path reads
// the catalog generation and lands the inserted vector in the gen-1 physical
// partition, not the gen-0 one. Construct a gen-1 partitioned collection by
// hand (catalog says gen 1 + gen-1 physical partitions exist).
func TestEmbeddedPointOpRoutesByGeneration(t *testing.T) {
	e := newSingleEmbedded(t)
	waitLeaderEmbedded(t, e)
	ctx := context.Background()
	ee := e.(*embedded)
	const P = 4
	cfg := VectorConfig{Dim: 4, Metric: vector.L2, M: 8, EfConstruction: 50, EfSearch: 64, Seed: 1}
	// Create gen-1 physical partitions directly (bypass CreateCollection's gen-0 path).
	for p := 0; p < P; p++ {
		phys := string(ops.PartitionKeyGen("docs", 1, p))
		if _, err := ee.Call(ctx, "vector_create_collection", ops.EncodeCreateCollectionArgs(phys, cfg)); err != nil {
			t.Fatal(err)
		}
	}
	// Record {P:4, gen:1} in the catalog.
	if err := ee.catalog.SetPartitionsGen("docs", P, 1); err != nil {
		t.Fatal(err)
	}
	// Insert via the normal point-op path — must route to gen-1 partitions.
	id := uint64(42)
	if err := e.VectorInsert(ctx, "docs", id, []float32{42, 0, 0, 0}); err != nil {
		t.Fatal(err)
	}
	wantPhys := string(ops.PartitionKeyGen("docs", 1, ops.PartitionOf(id, P)))
	rawCount := func(name string) int {
		body, err := ee.Call(ctx, "vector_search", ops.EncodeVectorSearchArgs(name, 10, []float32{42, 0, 0, 0}))
		if err != nil {
			return 0
		}
		res, _ := ops.DecodeVectorSearchResults(body)
		return len(res)
	}
	if rawCount(wantPhys) != 1 {
		t.Fatalf("vector not in gen-1 partition %s", wantPhys)
	}
	if rawCount(string(ops.PartitionKeyGen("docs", 0, ops.PartitionOf(id, P)))) != 0 {
		t.Fatal("vector leaked into gen-0 partition")
	}
}

// must fails the test immediately if err is non-nil.
func must(t *testing.T, err error) {
	t.Helper()
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
}

// TestEmbeddedNamedSingleNode is the single-node embedded end-to-end test for the
// named-vector family: create a 2-space collection -> insert points (map of named
// vecs + shared payload) -> search each space -> delete -> scroll. It rides the
// full embedded -> op-handler -> NamedCollection path.
func TestEmbeddedNamedSingleNode(t *testing.T) {
	s := newSingleEmbedded(t)
	waitLeaderEmbedded(t, s)
	ctx := context.Background()

	cfg := map[string]NamedVectorParams{
		"title": {Dim: 4, Metric: vector.Cosine},
		"image": {Dim: 4, Metric: vector.Cosine},
	}
	must(t, s.VectorNamedCreateCollection(ctx, "named", cfg, 0))

	// point 1: title aligned with the query, image populated; tagged en.
	must(t, s.VectorNamedInsert(ctx, "named", 1,
		map[string][]float32{"title": {1, 0, 0, 0}, "image": {0, 0, 1, 0}},
		vector.Metadata{"lang": vector.NewString("en")}, 0))
	// point 2: title NOT aligned, omits image; tagged fr.
	must(t, s.VectorNamedInsert(ctx, "named", 2,
		map[string][]float32{"title": {0, 1, 0, 0}},
		vector.Metadata{"lang": vector.NewString("fr")}, 0))

	// get_config round-trips the spaces.
	gotCfg, err := s.VectorNamedGetConfig(ctx, "named")
	must(t, err)
	if !reflect.DeepEqual(gotCfg, cfg) {
		t.Fatalf("get_config = %+v, want %+v", gotCfg, cfg)
	}

	// search the title space: point 1 first.
	res, err := s.VectorNamedSearch(ctx, "named", "title", []float32{1, 0, 0, 0}, 5, vector.Filter{})
	must(t, err)
	if len(res) != 2 || res[0].ID != 1 {
		t.Fatalf("title search = %+v, want point 1 first", res)
	}

	// filtered search (lang=en) -> only point 1.
	en := vector.Filter{Op: vector.FilterEq, Field: "lang", Value: vector.NewString("en")}
	res, err = s.VectorNamedSearch(ctx, "named", "title", []float32{1, 0, 0, 0}, 5, en)
	must(t, err)
	if len(res) != 1 || res[0].ID != 1 {
		t.Fatalf("filtered search = %+v, want only point 1", res)
	}

	// search_docs carries the shared payload.
	docs, err := s.VectorNamedSearchDocs(ctx, "named", "title", []float32{1, 0, 0, 0}, 5, vector.Filter{})
	must(t, err)
	var p1Found bool
	for _, d := range docs {
		if d.ID == 1 {
			p1Found = true
			if d.Metadata["lang"].Str != "en" {
				t.Errorf("point 1 payload = %+v, want lang=en", d.Metadata)
			}
		}
	}
	if !p1Found {
		t.Errorf("point 1 missing from search_docs: %+v", docs)
	}

	// image space: only point 1 populated it.
	res, err = s.VectorNamedSearch(ctx, "named", "image", []float32{0, 0, 1, 0}, 5, vector.Filter{})
	must(t, err)
	if len(res) != 1 || res[0].ID != 1 {
		t.Fatalf("image search = %+v, want only point 1", res)
	}

	// unknown vector name fails loud.
	if _, err := s.VectorNamedSearch(ctx, "named", "nope", []float32{1, 0, 0, 0}, 5, vector.Filter{}); err == nil {
		t.Error("search unknown space: want error, got nil")
	}

	// delete point 1 from every space + shared payload.
	existed, err := s.VectorNamedDelete(ctx, "named", 1)
	must(t, err)
	if !existed {
		t.Error("delete point 1: existed = false, want true")
	}
	res, err = s.VectorNamedSearch(ctx, "named", "title", []float32{1, 0, 0, 0}, 5, vector.Filter{})
	must(t, err)
	if len(res) != 1 || res[0].ID != 2 {
		t.Fatalf("title after delete = %+v, want only point 2", res)
	}
	res, err = s.VectorNamedSearch(ctx, "named", "image", []float32{0, 0, 1, 0}, 5, vector.Filter{})
	must(t, err)
	if len(res) != 0 {
		t.Fatalf("image after delete = %+v, want empty", res)
	}

	// scroll lists the live points (only point 2) with shared payload.
	docs, _, err = s.VectorNamedScroll(ctx, "named", vector.Filter{}, 0, "")
	must(t, err)
	if len(docs) != 1 || docs[0].ID != 2 || docs[0].Metadata["lang"].Str != "fr" {
		t.Fatalf("scroll = %+v, want only point 2 (lang=fr)", docs)
	}

	must(t, s.VectorNamedDropCollection(ctx, "named"))
}

// TestEmbeddedResplitShrinkSingleNode exercises the offline generational resplit
// for the SHRINK direction (newP < oldP): P=8 → P=3 over 300 docs.
// It proves: catalog updates to {3, gen:1}, all 300 docs re-routed into the 3
// new gen-1 partitions, all 8 old gen-0 partitions dropped, and search/scroll/filter intact.
func TestEmbeddedResplitShrinkSingleNode(t *testing.T) {
	e := newSingleEmbedded(t)
	waitLeaderEmbedded(t, e)
	ctx := context.Background()
	ee := e.(*embedded)
	must(t, e.CreateCollection(ctx, "docs", VectorConfig{Dim: 4, Metric: vector.L2, M: 8, EfConstruction: 50, EfSearch: 64, Seed: 1, Partitions: 8}))
	for id := uint64(1); id <= 300; id++ {
		must(t, e.VectorUpsert(ctx, "docs", id, []float32{float32(id), 0, 0, 0}, fmt.Sprintf("doc-%d", id), VectorInsertOpts{Metadata: vector.Metadata{"even": vector.NewBool(id%2 == 0)}}))
	}
	// Shrink 8 -> 3.
	must(t, e.VectorResplit(ctx, "docs", 3))
	// Catalog now {P:3, gen:1}.
	p, gen, ok := ee.catalog.PartitionsGen("docs")
	if !ok || p != 3 || gen != 1 {
		t.Fatalf("catalog (%d,%d,%v) want (3,1,true)", p, gen, ok)
	}
	// All 300 searchable, content + filter intact.
	res, _, err := e.VectorSearchDocs(ctx, "docs", []float32{0.5, 0, 0, 0}, 5, VectorSearchOpts{})
	must(t, err)
	if len(res) == 0 || res[0].ID != 1 || res[0].Content != "doc-1" {
		t.Fatalf("post-shrink search: %+v", res)
	}
	even, _, err := e.VectorSearchDocs(ctx, "docs", []float32{0.5, 0, 0, 0}, 400, VectorSearchOpts{Filter: VectorFilter{Op: vector.FilterEq, Field: "even", Value: vector.NewBool(true)}})
	must(t, err)
	if len(even) != 150 {
		t.Fatalf("filtered after shrink: %d, want 150", len(even))
	}
	for _, d := range even {
		if d.ID%2 != 0 {
			t.Fatalf("filtered returned odd id %d", d.ID)
		}
	}
	// gen-1 partitions (0..2) hold all 300; gen-0 partitions (0..7) all dropped.
	rawCount := func(name string) int {
		body, err := ee.Call(ctx, "vector_search", ops.EncodeVectorSearchArgs(name, 400, []float32{0.5, 0, 0, 0}))
		if err != nil {
			return 0
		}
		r, _ := ops.DecodeVectorSearchResults(body)
		return len(r)
	}
	total := 0
	for pp := 0; pp < 3; pp++ {
		total += rawCount(string(ops.PartitionKeyGen("docs", 1, pp)))
	}
	if total != 300 {
		t.Fatalf("gen-1 (3 partitions) hold %d, want 300", total)
	}
	for pp := 0; pp < 8; pp++ {
		if n := rawCount(string(ops.PartitionKeyGen("docs", 0, pp))); n != 0 {
			t.Fatalf("old gen-0 partition %d still holds %d", pp, n)
		}
	}
	// Scroll completeness.
	all, _, _, err := e.VectorScroll(ctx, "docs", VectorFilter{}, 0, VectorScrollOpts{})
	must(t, err)
	if len(all) != 300 || len(idSet(all)) != 300 {
		t.Fatalf("scroll after shrink: %d docs (%d distinct), want 300/300", len(all), len(idSet(all)))
	}
}

// TestEmbeddedResplitSingleNode exercises the offline generational resplit:
// build the next generation's physical partitions, stream every vector into them
// re-hashed by PartitionOf(id, newP), flip the catalog atomically to {newP,gen+1},
// then drop the old generation. P=4 → P=8 over 300 docs.
func TestEmbeddedResplitSingleNode(t *testing.T) {
	e := newSingleEmbedded(t)
	waitLeaderEmbedded(t, e)
	ctx := context.Background()
	ee := e.(*embedded)
	must(t, e.CreateCollection(ctx, "docs", VectorConfig{Dim: 4, Metric: vector.L2, M: 8, EfConstruction: 50, EfSearch: 64, Seed: 1, Partitions: 4}))
	for id := uint64(1); id <= 300; id++ {
		must(t, e.VectorUpsert(ctx, "docs", id, []float32{float32(id), 0, 0, 0}, fmt.Sprintf("doc-%d", id), VectorInsertOpts{Metadata: vector.Metadata{"even": vector.NewBool(id%2 == 0)}}))
	}
	must(t, e.VectorResplit(ctx, "docs", 8))
	// Catalog now {P:8, gen:1}.
	p, gen, ok := ee.catalog.PartitionsGen("docs")
	if !ok || p != 8 || gen != 1 {
		t.Fatalf("catalog (%d,%d,%v) want (8,1,true)", p, gen, ok)
	}
	// All 300 searchable, content + filter intact.
	res, _, err := e.VectorSearchDocs(ctx, "docs", []float32{0.5, 0, 0, 0}, 5, VectorSearchOpts{})
	must(t, err)
	if len(res) == 0 || res[0].ID != 1 || res[0].Content != "doc-1" {
		t.Fatalf("post-resplit search: %+v", res)
	}
	even, _, err := e.VectorSearchDocs(ctx, "docs", []float32{0.5, 0, 0, 0}, 400, VectorSearchOpts{Filter: VectorFilter{Op: vector.FilterEq, Field: "even", Value: vector.NewBool(true)}})
	must(t, err)
	if len(even) != 150 {
		t.Fatalf("filtered after resplit: %d, want 150", len(even))
	}
	// gen-1 partitions hold all 300; gen-0 partitions gone.
	rawCount := func(name string) int {
		body, err := ee.Call(ctx, "vector_search", ops.EncodeVectorSearchArgs(name, 400, []float32{0.5, 0, 0, 0}))
		if err != nil {
			return 0
		}
		r, _ := ops.DecodeVectorSearchResults(body)
		return len(r)
	}
	total := 0
	for pp := 0; pp < 8; pp++ {
		total += rawCount(string(ops.PartitionKeyGen("docs", 1, pp)))
	}
	if total != 300 {
		t.Fatalf("gen-1 partitions hold %d, want 300", total)
	}
	for pp := 0; pp < 4; pp++ {
		if n := rawCount(string(ops.PartitionKeyGen("docs", 0, pp))); n != 0 {
			t.Fatalf("old gen-0 partition %d still holds %d", pp, n)
		}
	}
	// Scroll completeness.
	all, _, _, err := e.VectorScroll(ctx, "docs", VectorFilter{}, 0, VectorScrollOpts{})
	must(t, err)
	if len(idSet(all)) != 300 {
		t.Fatalf("scroll after resplit: %d distinct, want 300", len(idSet(all)))
	}
}

func TestEmbeddedResplitRetryAfterPreFlipOrphans(t *testing.T) {
	e := newSingleEmbedded(t)
	waitLeaderEmbedded(t, e)
	ctx := context.Background()
	ee := e.(*embedded)
	must(t, e.CreateCollection(ctx, "docs", VectorConfig{Dim: 4, Metric: vector.L2, M: 8, EfConstruction: 50, EfSearch: 64, Seed: 1, Partitions: 4}))
	for id := uint64(1); id <= 100; id++ {
		must(t, e.VectorUpsert(ctx, "docs", id, []float32{float32(id), 0, 0, 0}, fmt.Sprintf("doc-%d", id), VectorInsertOpts{}))
	}
	// Simulate a prior resplit-to-8 that failed BEFORE the catalog flip: gen-1 partitions
	// exist as orphans, catalog still {4, gen0}. Create 3 of the 8 (a partial create).
	cfg := VectorConfig{Dim: 4, Metric: vector.L2, M: 8, EfConstruction: 50, EfSearch: 64, Seed: 1}
	for p := 0; p < 3; p++ {
		_, err := ee.Call(ctx, "vector_create_collection", ops.EncodeCreateCollectionArgs(string(ops.PartitionKeyGen("docs", 1, p)), cfg))
		must(t, err)
	}
	// Without self-heal this fails: "create new partition 0: ... collection already exists".
	must(t, e.VectorResplit(ctx, "docs", 8))
	p, gen, ok := ee.catalog.PartitionsGen("docs")
	if !ok || p != 8 || gen != 1 {
		t.Fatalf("catalog (%d,%d,%v) want (8,1,true)", p, gen, ok)
	}
	res, _, err := e.VectorSearchDocs(ctx, "docs", []float32{0.5, 0, 0, 0}, 3, VectorSearchOpts{})
	must(t, err)
	if len(res) == 0 || res[0].ID != 1 || res[0].Content != "doc-1" {
		t.Fatalf("post-retry search: %+v", res)
	}
	all, _, _, err := e.VectorScroll(ctx, "docs", VectorFilter{}, 0, VectorScrollOpts{})
	must(t, err)
	if len(idSet(all)) != 100 {
		t.Fatalf("after retry: %d distinct, want 100", len(idSet(all)))
	}
}

func TestEmbeddedResplitCleanup(t *testing.T) {
	e := newSingleEmbedded(t)
	waitLeaderEmbedded(t, e)
	ctx := context.Background()
	ee := e.(*embedded)
	must(t, e.CreateCollection(ctx, "docs", VectorConfig{Dim: 4, Metric: vector.L2, M: 8, EfConstruction: 50, EfSearch: 64, Seed: 1, Partitions: 4}))
	for id := uint64(1); id <= 100; id++ {
		must(t, e.VectorInsert(ctx, "docs", id, []float32{float32(id), 0, 0, 0}))
	}
	must(t, e.VectorResplit(ctx, "docs", 8)) // live {8, gen1}; gen0 dropped by resplit
	cfg := VectorConfig{Dim: 4, Metric: vector.L2, M: 8, EfConstruction: 50, EfSearch: 64, Seed: 1}
	exists := func(name string) bool {
		_, err := ee.Call(ctx, "vector_get_config", ops.EncodeGetConfigArgs(name))
		return err == nil
	}
	// Simulate a POST-flip drop-old failure: re-create old gen-0 partitions as leaks.
	for p := 0; p < 4; p++ {
		_, err := ee.Call(ctx, "vector_create_collection", ops.EncodeCreateCollectionArgs(string(ops.PartitionKeyGen("docs", 0, p)), cfg))
		must(t, err)
	}
	// Simulate a FORWARD pre-flip orphan: a failed resplit-to-16 left some gen-2 partitions.
	for p := 0; p < 5; p++ {
		_, err := ee.Call(ctx, "vector_create_collection", ops.EncodeCreateCollectionArgs(string(ops.PartitionKeyGen("docs", 2, p)), cfg))
		must(t, err)
	}
	dropped, err := e.VectorResplitCleanup(ctx, "docs")
	must(t, err)
	if dropped != 9 {
		t.Fatalf("cleanup dropped %d, want 9", dropped)
	}
	for p := 0; p < 4; p++ {
		if exists(string(ops.PartitionKeyGen("docs", 0, p))) {
			t.Fatalf("gen0 partition %d not cleaned", p)
		}
	}
	for p := 0; p < 5; p++ {
		if exists(string(ops.PartitionKeyGen("docs", 2, p))) {
			t.Fatalf("gen2 partition %d not cleaned", p)
		}
	}
	for p := 0; p < 8; p++ {
		if !exists(string(ops.PartitionKeyGen("docs", 1, p))) {
			t.Fatalf("live gen1 partition %d wrongly dropped", p)
		}
	}
	res, _, err := e.VectorSearchDocs(ctx, "docs", []float32{0.5, 0, 0, 0}, 3, VectorSearchOpts{})
	must(t, err)
	if len(res) == 0 || res[0].ID != 1 {
		t.Fatalf("live data damaged by cleanup: %+v", res)
	}
	all, _, _, err := e.VectorScroll(ctx, "docs", VectorFilter{}, 0, VectorScrollOpts{})
	must(t, err)
	if len(idSet(all)) != 100 {
		t.Fatalf("after cleanup: %d distinct, want 100", len(idSet(all)))
	}
	dropped2, err := e.VectorResplitCleanup(ctx, "docs")
	must(t, err)
	if dropped2 != 0 {
		t.Fatalf("second cleanup dropped %d, want 0", dropped2)
	}
}

// TestEmbeddedMVPartitioned exercises the partition-aware embedded MV path:
// create (P=4) writes the catalog so MVSearch fans out, add routes each doc to
// its physical partition, MVSearch's MaxSim winner is deterministic (tie-free
// scaled axes), MVDelete routes by docID, and drop fans out (every physical
// MV partition gone, catalog neutralized to unpartitioned).
func TestEmbeddedMVPartitioned(t *testing.T) {
	e := newSingleEmbedded(t)
	waitLeaderEmbedded(t, e)
	ctx := context.Background()
	ee := e.(*embedded)

	const P = 4
	must(t, e.VectorMVCreateCollection(ctx, "mvp", MultiVectorConfig{Dim: 4, Partitions: P}))
	if p, _, ok := ee.catalog.PartitionsGen("mvp"); !ok || p != P {
		t.Fatalf("PartitionsGen(mvp) = (%d,%v), want (4,true)", p, ok)
	}

	// The inner index is Cosine (vectors L2-normalized), so MaxSim is a dot
	// product over unit vectors — scaling is invisible. To stay tie-free, give
	// each doc a single token at a DISTINCT angle in the (x,y) plane:
	// θ_i = i * (π/2 / 40), all in the first quadrant, strictly monotonic. The
	// query is doc D's exact direction, so doc D scores 1.0 (unique max) and
	// every other doc scores strictly < 1.0.
	const winner = 17 // PartitionOf(17,4)=3 (non-zero) -> proves routing
	tokenAt := func(i int) []float32 {
		theta := float64(i) * (math.Pi / 2 / 40)
		return []float32{float32(math.Cos(theta)), float32(math.Sin(theta)), 0, 0}
	}
	for i := 0; i < 40; i++ {
		must(t, e.VectorMVAdd(ctx, "mvp", uint64(i), [][]float32{tokenAt(i)}, nil))
	}

	// Prove the docs actually fanned out across MORE THAN ONE physical
	// partition (so the logical MVSearch fan-out proof above is not vacuous: if
	// every doc had hashed to one partition the search would still pass). Probe
	// each physical partition's index directly with a raw single-Call MVSearch
	// (no fan-out) and count its docs; the logical "mvp" must hold none, the
	// per-partition counts must sum to 40, and at least TWO partitions must be
	// non-empty. The query token is (cos,sin) in the first quadrant so every
	// stored doc scores > 0 and is returned when K and candidates are large.
	probeQuery := [][]float32{{1, 0, 0, 0}}
	rawMVCount := func(name string) int {
		t.Helper()
		body, err := ee.Call(ctx, "vector_mv_search", ops.EncodeMVSearchArgs(name, probeQuery, 100, 200))
		if err != nil {
			t.Fatalf("raw MVSearch %q: %v", name, err)
		}
		out, err := ops.DecodeMVResults(body)
		if err != nil {
			t.Fatalf("decode raw MVSearch %q: %v", name, err)
		}
		return len(out)
	}
	if n := rawMVCount("mvp"); n != 0 {
		t.Fatalf("logical collection mvp held %d docs, want 0 (data should be partitioned)", n)
	}
	total, nonEmpty := 0, 0
	for p := 0; p < P; p++ {
		phys := string(ops.PartitionKeyGen("mvp", 0, p))
		n := rawMVCount(phys)
		total += n
		if n > 0 {
			nonEmpty++
		}
	}
	if total != 40 {
		t.Fatalf("physical partitions held %d docs total, want 40", total)
	}
	if nonEmpty < 2 {
		t.Fatalf("docs landed in %d physical partitions, want >= 2 (fan-out proof)", nonEmpty)
	}

	query := [][]float32{tokenAt(winner)}
	res, _, err := e.VectorMVSearch(ctx, "mvp", query, 5, MultiSearchOpts{CandidatesPerToken: 100})
	must(t, err)
	if len(res) != 5 {
		t.Fatalf("MVSearch returned %d results, want 5", len(res))
	}
	if res[0].ID != winner {
		t.Fatalf("MVSearch winner = %d, want %d", res[0].ID, winner)
	}

	// Delete a docID that routes to a NON-zero physical partition.
	var victim uint64
	found := false
	for i := uint64(0); i < 40; i++ {
		if int(i) != winner && ops.PartitionOf(i, P) != 0 {
			victim = i
			found = true
			break
		}
	}
	if !found {
		t.Fatal("no non-winner doc routes to a non-zero partition")
	}
	ok, err := e.VectorMVDelete(ctx, "mvp", victim)
	must(t, err)
	if !ok {
		t.Fatalf("MVDelete(%d) = false, want true", victim)
	}
	res2, _, err := e.VectorMVSearch(ctx, "mvp", query, 40, MultiSearchOpts{CandidatesPerToken: 100})
	must(t, err)
	for _, r := range res2 {
		if r.ID == victim {
			t.Fatalf("deleted doc %d still present in search", victim)
		}
	}

	// Drop fans out: every physical MV partition gone, catalog neutralized.
	must(t, e.VectorMVDropCollection(ctx, "mvp"))
	for p := 0; p < P; p++ {
		phys := string(ops.PartitionKeyGen("mvp", 0, p))
		if _, err := ee.Call(ctx, "vector_mv_get_config", ops.EncodeMVGetConfigArgs(phys)); err == nil {
			t.Fatalf("physical MV partition %q still exists after drop", phys)
		}
	}
	if p, _, ok := ee.catalog.PartitionsGen("mvp"); ok && p > 1 {
		t.Fatalf("PartitionsGen(mvp) = (%d,%v) after drop, want unpartitioned", p, ok)
	}
}

// TestEmbeddedMVPartitionedMetadata proves a partitioned (P>1) MV search
// preserves per-result Metadata through the fan-out, matching the single-shard
// path. The oracle is the KNOWN inserted metadata (not another VectorMVSearch
// call, which shares the fan-out path and so could not catch a metadata drop).
// Each doc gets distinct metadata keyed by its docID; we assert each returned
// result carries exactly what was inserted for that docID. The winner routes to
// a non-zero physical partition, so fan-out is genuinely exercised.
func TestEmbeddedMVPartitionedMetadata(t *testing.T) {
	e := newSingleEmbedded(t)
	waitLeaderEmbedded(t, e)
	ctx := context.Background()
	ee := e.(*embedded)

	const P = 4
	must(t, e.VectorMVCreateCollection(ctx, "mvm", MultiVectorConfig{Dim: 4, Partitions: P}))
	if p, _, ok := ee.catalog.PartitionsGen("mvm"); !ok || p != P {
		t.Fatalf("PartitionsGen(mvm) = (%d,%v), want (4,true)", p, ok)
	}

	// Tie-free axes: each doc's single token sits at a distinct first-quadrant
	// angle, so MaxSim ordering is deterministic. (Mirrors TestEmbeddedMVPartitioned.)
	const winner = 17 // PartitionOf(17,4)=3 (non-zero) -> proves fan-out routing
	tokenAt := func(i int) []float32 {
		theta := float64(i) * (math.Pi / 2 / 40)
		return []float32{float32(math.Cos(theta)), float32(math.Sin(theta)), 0, 0}
	}
	// Insert distinct metadata per docID: an int "docid" and a string "tag".
	wantMeta := func(id int) VectorMetadata {
		return VectorMetadata{
			"docid": vector.NewInt(int64(id)),
			"tag":   vector.NewString(fmt.Sprintf("doc-%d", id)),
		}
	}
	for i := 0; i < 40; i++ {
		must(t, e.VectorMVAdd(ctx, "mvm", uint64(i), [][]float32{tokenAt(i)}, wantMeta(i)))
	}

	if ops.PartitionOf(winner, P) == 0 {
		t.Fatalf("winner %d routes to partition 0; want non-zero so fan-out is exercised", winner)
	}

	query := [][]float32{tokenAt(winner)}
	res, _, err := e.VectorMVSearch(ctx, "mvm", query, 5, MultiSearchOpts{CandidatesPerToken: 100})
	must(t, err)
	if len(res) != 5 {
		t.Fatalf("MVSearch returned %d results, want 5", len(res))
	}
	if res[0].ID != winner {
		t.Fatalf("MVSearch winner = %d, want %d", res[0].ID, winner)
	}

	// Each result must carry exactly the metadata inserted for its docID.
	for _, r := range res {
		want := wantMeta(int(r.ID))
		if len(r.Metadata) == 0 {
			t.Fatalf("result id=%d has no metadata; want %+v (metadata dropped by fan-out)", r.ID, want)
		}
		for key, wv := range want {
			gv, ok := r.Metadata[key]
			if !ok {
				t.Fatalf("result id=%d missing metadata key %q; got %+v", r.ID, key, r.Metadata)
			}
			if !gv.Equal(wv) {
				t.Fatalf("result id=%d metadata[%q] = %+v, want %+v", r.ID, key, gv, wv)
			}
		}
	}
}

// TestEmbeddedMVCrossTypeGuard proves the fail-loud cross-type guard rejects a
// name partitioned as both dense and MV — in BOTH directions, and even when the
// pre-existing other-type collection is UNpartitioned.
func TestEmbeddedMVCrossTypeGuard(t *testing.T) {
	e := newSingleEmbedded(t)
	waitLeaderEmbedded(t, e)
	ctx := context.Background()

	// Dense partitioned "dup" exists -> partitioned MV "dup" rejected.
	must(t, e.CreateCollection(ctx, "dup", VectorConfig{Dim: 4, Metric: vector.L2, M: 8, EfConstruction: 50, EfSearch: 64, Seed: 1, Partitions: 4}))
	if err := e.VectorMVCreateCollection(ctx, "dup", MultiVectorConfig{Dim: 4, Partitions: 4}); err == nil {
		t.Fatal("partitioned MV over dense-partitioned name: want error, got nil")
	}

	// UNpartitioned dense "dup2" exists -> partitioned MV "dup2" rejected.
	must(t, e.CreateCollection(ctx, "dup2", VectorConfig{Dim: 4, Metric: vector.L2, M: 8, EfConstruction: 50, EfSearch: 64, Seed: 1}))
	if err := e.VectorMVCreateCollection(ctx, "dup2", MultiVectorConfig{Dim: 4, Partitions: 4}); err == nil {
		t.Fatal("partitioned MV over unpartitioned-dense name: want error, got nil")
	}

	// MV partitioned "mdup" exists -> dense partitioned "mdup" rejected.
	must(t, e.VectorMVCreateCollection(ctx, "mdup", MultiVectorConfig{Dim: 4, Partitions: 4}))
	if err := e.CreateCollection(ctx, "mdup", VectorConfig{Dim: 4, Metric: vector.L2, M: 8, EfConstruction: 50, EfSearch: 64, Seed: 1, Partitions: 4}); err == nil {
		t.Fatal("dense-partitioned over MV-partitioned name: want error, got nil")
	}

	// UNpartitioned MV "mdup2" exists -> dense partitioned "mdup2" rejected.
	must(t, e.VectorMVCreateCollection(ctx, "mdup2", MultiVectorConfig{Dim: 4}))
	if err := e.CreateCollection(ctx, "mdup2", VectorConfig{Dim: 4, Metric: vector.L2, M: 8, EfConstruction: 50, EfSearch: 64, Seed: 1, Partitions: 4}); err == nil {
		t.Fatal("dense-partitioned over unpartitioned-MV name: want error, got nil")
	}

	// Name guard.
	if err := e.VectorMVCreateCollection(ctx, "bad#name", MultiVectorConfig{Dim: 4, Partitions: 4}); err == nil {
		t.Fatal("MV create with reserved-char name: want error, got nil")
	}
}

// TestMVGetConfigReturnsConfig proves vector_mv_get_config round-trips the full
// MV config (not just an existence probe with a nil body) — the introspection
// primitive an offline resplit uses to re-create new-generation partitions with
// the same configuration.
func TestMVGetConfigReturnsConfig(t *testing.T) {
	e := newSingleEmbedded(t)
	waitLeaderEmbedded(t, e)
	ctx := context.Background()

	want := MultiVectorConfig{Dim: 4, M: 8, EfConstruction: 100, EfSearch: 32, Quant: vector.QuantSQ8, RescoreFactor: 3}
	must(t, e.VectorMVCreateCollection(ctx, "mvc", want))

	body, err := e.Call(ctx, "vector_mv_get_config", ops.EncodeMVGetConfigArgs("mvc"))
	must(t, err)
	_, cfg, err := ops.DecodeMVCreateArgs(body)
	must(t, err)
	if cfg.Dim != want.Dim || cfg.M != want.M || cfg.EfConstruction != want.EfConstruction ||
		cfg.EfSearch != want.EfSearch || cfg.Quant != want.Quant || cfg.RescoreFactor != want.RescoreFactor {
		t.Fatalf("mv get_config round-trip = %+v, want %+v", cfg, want)
	}
}

// TestMVScanVectors proves vector_mv_scan_vectors enumerates every LIVE document
// with its EXACT token vectors and metadata, and that a deleted doc is absent —
// the read primitive an offline MV resplit uses to re-insert each doc into a
// re-hashed generation. Single-partition collection so the raw Call hits the one
// physical index directly.
func TestMVScanVectors(t *testing.T) {
	e := newSingleEmbedded(t)
	waitLeaderEmbedded(t, e)
	ctx := context.Background()

	must(t, e.VectorMVCreateCollection(ctx, "mvscan", MultiVectorConfig{Dim: 4}))

	// Three docs with DISTINCT token counts and distinct metadata. Tokens are
	// already UNIT vectors (axis-aligned) so the Cosine index's insert-time
	// normalization is a no-op and the scanned floats round-trip EXACTLY.
	axis := [][]float32{{1, 0, 0, 0}, {0, 1, 0, 0}, {0, 0, 1, 0}, {0, 0, 0, 1}}
	tokensFor := func(id int) [][]float32 {
		out := make([][]float32, id) // id tokens (id=1,2,3 → distinct counts)
		for i := range out {
			out[i] = axis[i%len(axis)]
		}
		return out
	}
	for id := 1; id <= 3; id++ {
		md := VectorMetadata{"docid": vector.NewInt(int64(id))}
		must(t, e.VectorMVAdd(ctx, "mvscan", uint64(id), tokensFor(id), md))
	}

	// Delete doc 2.
	ok, err := e.VectorMVDelete(ctx, "mvscan", 2)
	must(t, err)
	if !ok {
		t.Fatal("VectorMVDelete(2) = false, want true")
	}

	body, err := e.Call(ctx, "vector_mv_scan_vectors", ops.EncodeMVScanArgs("mvscan"))
	must(t, err)
	recs, err := ops.DecodeMVScanResult(body)
	must(t, err)
	sort.Slice(recs, func(i, j int) bool { return recs[i].ID < recs[j].ID })

	if len(recs) != 2 {
		t.Fatalf("scan returned %d live docs, want 2 (deleted doc 2 must be absent)", len(recs))
	}
	wantIDs := []uint64{1, 3}
	for i, r := range recs {
		if r.ID != wantIDs[i] {
			t.Fatalf("rec[%d].ID = %d, want %d", i, r.ID, wantIDs[i])
		}
		want := tokensFor(int(r.ID))
		if len(r.Tokens) != len(want) {
			t.Fatalf("doc %d: %d tokens, want %d", r.ID, len(r.Tokens), len(want))
		}
		for ti := range want {
			if len(r.Tokens[ti]) != len(want[ti]) {
				t.Fatalf("doc %d token %d: dim %d, want %d", r.ID, ti, len(r.Tokens[ti]), len(want[ti]))
			}
			for k := range want[ti] {
				if r.Tokens[ti][k] != want[ti][k] {
					t.Fatalf("doc %d token %d elem %d = %v, want %v", r.ID, ti, k, r.Tokens[ti][k], want[ti][k])
				}
			}
		}
		wantMD := vector.NewInt(int64(r.ID))
		gv, ok := r.Metadata["docid"]
		if !ok {
			t.Fatalf("doc %d: missing metadata key docid; got %+v", r.ID, r.Metadata)
		}
		if !gv.Equal(wantMD) {
			t.Fatalf("doc %d metadata[docid] = %+v, want %+v", r.ID, gv, wantMD)
		}
	}
}

// mvPhysCount returns the number of live docs in a physical MV partition, via a
// raw scan (no fan-out).
func mvPhysCount(t *testing.T, ee *embedded, phys string) int {
	t.Helper()
	body, err := ee.Call(context.Background(), "vector_mv_scan_vectors", ops.EncodeMVScanArgs(phys))
	if err != nil {
		return 0
	}
	recs, err := ops.DecodeMVScanResult(body)
	if err != nil {
		t.Fatalf("decode scan %q: %v", phys, err)
	}
	return len(recs)
}

// TestEmbeddedMVResplitSingleNode exercises the offline generational MV resplit
// (grow 4 -> 8): build the next generation's physical MV partitions, stream every
// doc into them re-hashed by PartitionOf(id, newP), flip the catalog atomically to
// {newP, gen+1}, then drop the old generation. Asserts metadata preservation and
// per-partition redistribution.
func TestEmbeddedMVResplitSingleNode(t *testing.T) {
	e := newSingleEmbedded(t)
	waitLeaderEmbedded(t, e)
	ctx := context.Background()
	ee := e.(*embedded)

	const (
		name = "mvr"
		N    = 40
	)
	must(t, e.VectorMVCreateCollection(ctx, name, MultiVectorConfig{Dim: 4, Partitions: 4}))
	for id := 0; id < N; id++ {
		md := VectorMetadata{"docid": vector.NewInt(int64(id))}
		must(t, e.VectorMVAdd(ctx, name, uint64(id), [][]float32{mvTokenAt(id)}, md))
	}

	must(t, e.VectorMVResplit(ctx, name, 8))

	// Catalog now {P:8, gen:1}.
	p, gen, ok := ee.catalog.PartitionsGen(name)
	if !ok || p != 8 || gen != 1 {
		t.Fatalf("catalog (%d,%d,%v) want (8,1,true)", p, gen, ok)
	}

	// Global top-k: querying a doc's own token returns it at rank 0 with metadata.
	const winner = 17 // PartitionOf(17,8)=1 (non-zero) -> proves fan-out routing
	res, _, err := e.VectorMVSearch(ctx, name, [][]float32{mvTokenAt(winner)}, 5, MultiSearchOpts{CandidatesPerToken: 100})
	must(t, err)
	if len(res) != 5 {
		t.Fatalf("MVSearch returned %d results, want 5", len(res))
	}
	if res[0].ID != winner {
		t.Fatalf("MVSearch winner = %d, want %d", res[0].ID, winner)
	}
	wantMD := vector.NewInt(int64(winner))
	gv, hasMD := res[0].Metadata["docid"]
	if !hasMD || !gv.Equal(wantMD) {
		t.Fatalf("winner metadata[docid] = %+v (present=%v), want %+v (metadata dropped across flip)", gv, hasMD, wantMD)
	}

	// Every original doc still findable at rank 0 by its own token.
	for id := 0; id < N; id++ {
		r, _, err := e.VectorMVSearch(ctx, name, [][]float32{mvTokenAt(id)}, 1, MultiSearchOpts{CandidatesPerToken: 100})
		must(t, err)
		if len(r) == 0 || r[0].ID != uint64(id) {
			t.Fatalf("doc %d not findable after resplit: %+v", id, r)
		}
	}

	// gen-1 physical partitions sum to N; gen-0 partitions are gone.
	total := 0
	for pp := 0; pp < 8; pp++ {
		total += mvPhysCount(t, ee, string(ops.PartitionKeyGen(name, 1, pp)))
	}
	if total != N {
		t.Fatalf("gen-1 partitions hold %d docs, want %d", total, N)
	}
	for pp := 0; pp < 4; pp++ {
		if _, err := ee.Call(ctx, "vector_mv_get_config", ops.EncodeMVGetConfigArgs(string(ops.PartitionKeyGen(name, 0, pp)))); err == nil {
			t.Fatalf("gen-0 physical partition %d still exists after resplit", pp)
		}
	}
}

// TestEmbeddedMVResplitShrinkSingleNode is the shrink mirror (8 -> 3).
func TestEmbeddedMVResplitShrinkSingleNode(t *testing.T) {
	e := newSingleEmbedded(t)
	waitLeaderEmbedded(t, e)
	ctx := context.Background()
	ee := e.(*embedded)

	const (
		name = "mvrs"
		N    = 40
	)
	must(t, e.VectorMVCreateCollection(ctx, name, MultiVectorConfig{Dim: 4, Partitions: 8}))
	for id := 0; id < N; id++ {
		md := VectorMetadata{"docid": vector.NewInt(int64(id))}
		must(t, e.VectorMVAdd(ctx, name, uint64(id), [][]float32{mvTokenAt(id)}, md))
	}

	must(t, e.VectorMVResplit(ctx, name, 3))

	p, gen, ok := ee.catalog.PartitionsGen(name)
	if !ok || p != 3 || gen != 1 {
		t.Fatalf("catalog (%d,%d,%v) want (3,1,true)", p, gen, ok)
	}

	const winner = 17
	res, _, err := e.VectorMVSearch(ctx, name, [][]float32{mvTokenAt(winner)}, 5, MultiSearchOpts{CandidatesPerToken: 100})
	must(t, err)
	if len(res) != 5 || res[0].ID != winner {
		t.Fatalf("post-shrink search: %+v", res)
	}
	wantMD := vector.NewInt(int64(winner))
	gv, hasMD := res[0].Metadata["docid"]
	if !hasMD || !gv.Equal(wantMD) {
		t.Fatalf("winner metadata[docid] = %+v (present=%v), want %+v", gv, hasMD, wantMD)
	}

	for id := 0; id < N; id++ {
		r, _, err := e.VectorMVSearch(ctx, name, [][]float32{mvTokenAt(id)}, 1, MultiSearchOpts{CandidatesPerToken: 100})
		must(t, err)
		if len(r) == 0 || r[0].ID != uint64(id) {
			t.Fatalf("doc %d not findable after shrink: %+v", id, r)
		}
	}

	total := 0
	for pp := 0; pp < 3; pp++ {
		total += mvPhysCount(t, ee, string(ops.PartitionKeyGen(name, 1, pp)))
	}
	if total != N {
		t.Fatalf("gen-1 partitions hold %d docs, want %d", total, N)
	}
	for pp := 0; pp < 8; pp++ {
		if _, err := ee.Call(ctx, "vector_mv_get_config", ops.EncodeMVGetConfigArgs(string(ops.PartitionKeyGen(name, 0, pp)))); err == nil {
			t.Fatalf("gen-0 physical partition %d still exists after shrink", pp)
		}
	}
}

// TestEmbeddedMVResplitRetryAfterPreFlipOrphans proves the self-heal: a prior
// resplit that failed before the catalog flip left gen-1 physical partitions
// behind; a retry drops the orphans and completes.
func TestEmbeddedMVResplitRetryAfterPreFlipOrphans(t *testing.T) {
	e := newSingleEmbedded(t)
	waitLeaderEmbedded(t, e)
	ctx := context.Background()
	ee := e.(*embedded)

	const (
		name = "mvret"
		N    = 40
	)
	must(t, e.VectorMVCreateCollection(ctx, name, MultiVectorConfig{Dim: 4, Partitions: 4}))
	for id := 0; id < N; id++ {
		md := VectorMetadata{"docid": vector.NewInt(int64(id))}
		must(t, e.VectorMVAdd(ctx, name, uint64(id), [][]float32{mvTokenAt(id)}, md))
	}

	// Simulate a prior resplit-to-8 that failed BEFORE the catalog flip: a few
	// gen-1 physical MV partitions exist as orphans, catalog still {4, gen0}.
	physCfg := MultiVectorConfig{Dim: 4}
	for p := 0; p < 3; p++ {
		_, err := ee.Call(ctx, "vector_mv_create_collection", ops.EncodeMVCreateArgs(string(ops.PartitionKeyGen(name, 1, p)), physCfg))
		must(t, err)
	}
	// Without self-heal this fails: "create new partition 0: ... already exists".
	must(t, e.VectorMVResplit(ctx, name, 8))

	p, gen, ok := ee.catalog.PartitionsGen(name)
	if !ok || p != 8 || gen != 1 {
		t.Fatalf("catalog (%d,%d,%v) want (8,1,true)", p, gen, ok)
	}
	const winner = 17
	res, _, err := e.VectorMVSearch(ctx, name, [][]float32{mvTokenAt(winner)}, 5, MultiSearchOpts{CandidatesPerToken: 100})
	must(t, err)
	if len(res) != 5 || res[0].ID != winner {
		t.Fatalf("post-retry search: %+v", res)
	}
	for id := 0; id < N; id++ {
		r, _, err := e.VectorMVSearch(ctx, name, [][]float32{mvTokenAt(id)}, 1, MultiSearchOpts{CandidatesPerToken: 100})
		must(t, err)
		if len(r) == 0 || r[0].ID != uint64(id) {
			t.Fatalf("doc %d not findable after retry: %+v", id, r)
		}
	}
	total := 0
	for pp := 0; pp < 8; pp++ {
		total += mvPhysCount(t, ee, string(ops.PartitionKeyGen(name, 1, pp)))
	}
	if total != N {
		t.Fatalf("gen-1 partitions hold %d docs, want %d", total, N)
	}
}

// TestEmbeddedMVResplitCleanup proves VectorMVResplitCleanup drops every
// non-live-generation physical partition (post-flip old-gen leaks + forward
// orphans), is idempotent, and never touches the live generation.
func TestEmbeddedMVResplitCleanup(t *testing.T) {
	e := newSingleEmbedded(t)
	waitLeaderEmbedded(t, e)
	ctx := context.Background()
	ee := e.(*embedded)

	const (
		name = "mvcln"
		N    = 40
	)
	must(t, e.VectorMVCreateCollection(ctx, name, MultiVectorConfig{Dim: 4, Partitions: 4}))
	for id := 0; id < N; id++ {
		md := VectorMetadata{"docid": vector.NewInt(int64(id))}
		must(t, e.VectorMVAdd(ctx, name, uint64(id), [][]float32{mvTokenAt(id)}, md))
	}
	must(t, e.VectorMVResplit(ctx, name, 8)) // live {8, gen1}; gen0 dropped by resplit

	physCfg := MultiVectorConfig{Dim: 4}
	exists := func(phys string) bool {
		_, err := ee.Call(ctx, "vector_mv_get_config", ops.EncodeMVGetConfigArgs(phys))
		return err == nil
	}
	// Simulate a POST-flip drop-old failure: re-create old gen-0 partitions as leaks.
	for p := 0; p < 4; p++ {
		_, err := ee.Call(ctx, "vector_mv_create_collection", ops.EncodeMVCreateArgs(string(ops.PartitionKeyGen(name, 0, p)), physCfg))
		must(t, err)
	}
	// Simulate a FORWARD pre-flip orphan: a failed resplit-to-16 left gen-2 partitions.
	for p := 0; p < 5; p++ {
		_, err := ee.Call(ctx, "vector_mv_create_collection", ops.EncodeMVCreateArgs(string(ops.PartitionKeyGen(name, 2, p)), physCfg))
		must(t, err)
	}

	dropped, err := e.VectorMVResplitCleanup(ctx, name)
	must(t, err)
	if dropped != 9 {
		t.Fatalf("cleanup dropped %d, want 9", dropped)
	}
	for p := 0; p < 4; p++ {
		if exists(string(ops.PartitionKeyGen(name, 0, p))) {
			t.Fatalf("gen0 partition %d not cleaned", p)
		}
	}
	for p := 0; p < 5; p++ {
		if exists(string(ops.PartitionKeyGen(name, 2, p))) {
			t.Fatalf("gen2 partition %d not cleaned", p)
		}
	}
	for p := 0; p < 8; p++ {
		if !exists(string(ops.PartitionKeyGen(name, 1, p))) {
			t.Fatalf("live gen1 partition %d wrongly dropped", p)
		}
	}
	// Live data intact + searchable.
	const winner = 17
	res, _, err := e.VectorMVSearch(ctx, name, [][]float32{mvTokenAt(winner)}, 5, MultiSearchOpts{CandidatesPerToken: 100})
	must(t, err)
	if len(res) == 0 || res[0].ID != winner {
		t.Fatalf("live data damaged by cleanup: %+v", res)
	}
	// Idempotent: a second cleanup drops nothing.
	dropped2, err := e.VectorMVResplitCleanup(ctx, name)
	must(t, err)
	if dropped2 != 0 {
		t.Fatalf("second cleanup dropped %d, want 0", dropped2)
	}
}

// TestEmbeddedResplitRejectsNewPRange asserts the server-side newP-range guard
// in VectorResplit rejects 0, 1, and a value above maxResplitPartitions — all on
// a genuinely-partitioned collection so it is the newP guard (not the
// not-partitioned guard) that rejects — and that the catalog is left unmutated
// (PartitionsGen unchanged) so the billions-of-sub-ops loop never runs.
func TestEmbeddedResplitRejectsNewPRange(t *testing.T) {
	e := newSingleEmbedded(t)
	waitLeaderEmbedded(t, e)
	ctx := context.Background()
	ee := e.(*embedded)

	const (
		coll = "rr"
		P    = 4
	)
	must(t, e.CreateCollection(ctx, coll, VectorConfig{Dim: 4, Metric: vector.L2, M: 8, EfConstruction: 50, EfSearch: 64, Partitions: P}))
	wantP, wantGen, ok := ee.catalog.PartitionsGen(coll)
	if !ok || wantP != P {
		t.Fatalf("setup PartitionsGen = (%d,%d,%v), want (%d,_,true)", wantP, wantGen, ok, P)
	}

	for _, newP := range []int{0, 1, maxResplitPartitions + 1} {
		if err := e.VectorResplit(ctx, coll, newP); err == nil {
			t.Errorf("VectorResplit(newP=%d): expected error, got nil", newP)
		}
		if p, gen, ok := ee.catalog.PartitionsGen(coll); !ok || p != wantP || gen != wantGen {
			t.Fatalf("VectorResplit(newP=%d) mutated catalog: (%d,%d,%v), want (%d,%d,true)",
				newP, p, gen, ok, wantP, wantGen)
		}
	}
}

// TestEmbeddedMVResplitRejectsNewPRange mirrors TestEmbeddedResplitRejectsNewPRange
// for the multi-vector path.
func TestEmbeddedMVResplitRejectsNewPRange(t *testing.T) {
	e := newSingleEmbedded(t)
	waitLeaderEmbedded(t, e)
	ctx := context.Background()
	ee := e.(*embedded)

	const (
		coll = "mvrr"
		P    = 4
	)
	must(t, e.VectorMVCreateCollection(ctx, coll, MultiVectorConfig{Dim: 4, Partitions: P}))
	wantP, wantGen, ok := ee.catalog.PartitionsGen(coll)
	if !ok || wantP != P {
		t.Fatalf("setup PartitionsGen = (%d,%d,%v), want (%d,_,true)", wantP, wantGen, ok, P)
	}

	for _, newP := range []int{0, 1, maxResplitPartitions + 1} {
		if err := e.VectorMVResplit(ctx, coll, newP); err == nil {
			t.Errorf("VectorMVResplit(newP=%d): expected error, got nil", newP)
		}
		if p, gen, ok := ee.catalog.PartitionsGen(coll); !ok || p != wantP || gen != wantGen {
			t.Fatalf("VectorMVResplit(newP=%d) mutated catalog: (%d,%d,%v), want (%d,%d,true)",
				newP, p, gen, ok, wantP, wantGen)
		}
	}
}

// ---- dual-write routing during reshard ----

// physVecExists reports whether id is live in the given physical dense partition
// by reading it back directly (single-Call, no fan-out), so dual-write landing
// can be asserted per generation.
func physVecExists(t *testing.T, ee *embedded, phys string, id uint64) bool {
	t.Helper()
	body, err := ee.Call(context.Background(), "vector_exists", ops.EncodeExistsArgs(phys, id))
	if err != nil {
		t.Fatalf("vector_exists %q id=%d: %v", phys, id, err)
	}
	ok, err := ops.DecodeExistsResult(body)
	if err != nil {
		t.Fatalf("decode exists %q: %v", phys, err)
	}
	return ok
}

// physMVExists reports whether docID is live in the given physical MV partition.
func physMVExists(t *testing.T, ee *embedded, phys string, docID uint64) bool {
	t.Helper()
	body, err := ee.Call(context.Background(), "vector_mv_exists", ops.EncodeMVExistsArgs(phys, docID))
	if err != nil {
		t.Fatalf("vector_mv_exists %q id=%d: %v", phys, docID, err)
	}
	ok, err := ops.DecodeExistsResult(body)
	if err != nil {
		t.Fatalf("decode mv exists %q: %v", phys, err)
	}
	return ok
}

// mustCall runs a raw op and discards the body, surfacing the error for must().
func mustCall(ee *embedded, op string, args []byte) error {
	_, err := ee.Call(context.Background(), op, args)
	return err
}

// setupReshardingDense builds a dense collection in Resharding{OldP,OldGen=0,
// NewP,NewGen=1}: it creates both the old-gen and new-gen physical partitions,
// records {OldP, gen 0} in the catalog (the live read source of truth), and sets
// the reshard state so writes dual-target. Returns the embedded backend.
func setupReshardingDense(t *testing.T, coll string, oldP, newP int) *embedded {
	t.Helper()
	e := newSingleEmbedded(t)
	waitLeaderEmbedded(t, e)
	ee := e.(*embedded)
	cfg := VectorConfig{Dim: 4, Metric: vector.L2, M: 8, EfConstruction: 50, EfSearch: 64, Seed: 1}
	for p := 0; p < oldP; p++ {
		must(t, mustCall(ee, "vector_create_collection", ops.EncodeCreateCollectionArgs(string(ops.PartitionKeyGen(coll, 0, p)), cfg)))
	}
	for p := 0; p < newP; p++ {
		must(t, mustCall(ee, "vector_create_collection", ops.EncodeCreateCollectionArgs(string(ops.PartitionKeyGen(coll, 1, p)), cfg)))
	}
	must(t, ee.catalog.SetPartitionsGen(coll, oldP, 0))
	must(t, ee.catalog.SetReshardState(coll, ReshardState{Status: 1, OldP: oldP, OldGen: 0, NewP: newP, NewGen: 1}))
	return ee
}

// setupReshardingMV is the MV-collection analogue of setupReshardingDense: it
// creates both the old-gen and new-gen physical MV partitions, records {OldP,
// gen 0} in the catalog, and sets the reshard state so MV writes dual-target.
func setupReshardingMV(t *testing.T, coll string, oldP, newP int) *embedded {
	t.Helper()
	e := newSingleEmbedded(t)
	waitLeaderEmbedded(t, e)
	ee := e.(*embedded)
	mvCfg := MultiVectorConfig{Dim: 4}
	for p := 0; p < oldP; p++ {
		must(t, mustCall(ee, "vector_mv_create_collection", ops.EncodeMVCreateArgs(string(ops.PartitionKeyGen(coll, 0, p)), mvCfg)))
	}
	for p := 0; p < newP; p++ {
		must(t, mustCall(ee, "vector_mv_create_collection", ops.EncodeMVCreateArgs(string(ops.PartitionKeyGen(coll, 1, p)), mvCfg)))
	}
	must(t, ee.catalog.SetPartitionsGen(coll, oldP, 0))
	must(t, ee.catalog.SetReshardState(coll, ReshardState{Status: 1, OldP: oldP, OldGen: 0, NewP: newP, NewGen: 1}))
	return ee
}

// TestEmbeddedDualWriteInsertUpsertHitsBothGens proves insert and upsert during a
// reshard land in BOTH the live (old) gen partition and the target (new) gen
// partition, while a non-target old-gen partition stays empty.
func TestEmbeddedDualWriteInsertUpsertHitsBothGens(t *testing.T) {
	const coll = "dw"
	const oldP, newP = 2, 4
	ee := setupReshardingDense(t, coll, oldP, newP)
	ctx := context.Background()

	var id uint64 = 7
	oldPhys := string(ops.PartitionKeyGen(coll, 0, ops.PartitionOf(id, oldP)))
	newPhys := string(ops.PartitionKeyGen(coll, 1, ops.PartitionOf(id, newP)))

	must(t, ee.VectorInsert(ctx, coll, id, []float32{1, 0, 0, 0}))
	if !physVecExists(t, ee, oldPhys, id) {
		t.Fatalf("insert did not land in OLD gen %q", oldPhys)
	}
	if !physVecExists(t, ee, newPhys, id) {
		t.Fatalf("insert did not land in NEW gen %q", newPhys)
	}
	otherOld := string(ops.PartitionKeyGen(coll, 0, (ops.PartitionOf(id, oldP)+1)%oldP))
	if physVecExists(t, ee, otherOld, id) {
		t.Fatalf("insert leaked into non-target old partition %q", otherOld)
	}

	var uid uint64 = 12
	uOld := string(ops.PartitionKeyGen(coll, 0, ops.PartitionOf(uid, oldP)))
	uNew := string(ops.PartitionKeyGen(coll, 1, ops.PartitionOf(uid, newP)))
	must(t, ee.VectorUpsert(ctx, coll, uid, []float32{2, 0, 0, 0}, "c", VectorInsertOpts{}))
	if !physVecExists(t, ee, uOld, uid) || !physVecExists(t, ee, uNew, uid) {
		t.Fatalf("upsert did not dual-write: old=%v new=%v", physVecExists(t, ee, uOld, uid), physVecExists(t, ee, uNew, uid))
	}
}

// TestEmbeddedDualWriteDeleteRemovesBothGens proves a delete during a reshard
// removes the id from BOTH gens.
func TestEmbeddedDualWriteDeleteRemovesBothGens(t *testing.T) {
	const coll = "dwdel"
	const oldP, newP = 2, 4
	ee := setupReshardingDense(t, coll, oldP, newP)
	ctx := context.Background()

	var id uint64 = 5
	oldPhys := string(ops.PartitionKeyGen(coll, 0, ops.PartitionOf(id, oldP)))
	newPhys := string(ops.PartitionKeyGen(coll, 1, ops.PartitionOf(id, newP)))
	must(t, ee.VectorInsert(ctx, coll, id, []float32{3, 0, 0, 0}))
	if !physVecExists(t, ee, oldPhys, id) || !physVecExists(t, ee, newPhys, id) {
		t.Fatal("precondition: insert must dual-write before delete")
	}
	existed, err := ee.VectorDelete(ctx, coll, id)
	must(t, err)
	if !existed {
		t.Fatal("delete should report existed=true (present in live gen)")
	}
	if physVecExists(t, ee, oldPhys, id) {
		t.Fatalf("delete did not remove from OLD gen %q", oldPhys)
	}
	if physVecExists(t, ee, newPhys, id) {
		t.Fatalf("delete did not remove from NEW gen %q", newPhys)
	}
}

// TestEmbeddedDualWriteReadsStayOnOldGen proves reads (search/scroll) only hit the
// LIVE (old) gen during a reshard: a vector written ONLY to the new gen (by hand)
// is invisible to search/scroll, while a vector written via dual-write is visible.
func TestEmbeddedDualWriteReadsStayOnOldGen(t *testing.T) {
	const coll = "dwread"
	const oldP, newP = 2, 4
	ee := setupReshardingDense(t, coll, oldP, newP)
	ctx := context.Background()

	var visible uint64 = 1
	must(t, ee.VectorInsert(ctx, coll, visible, []float32{1, 0, 0, 0}))

	var newOnly uint64 = 999
	newPhys := string(ops.PartitionKeyGen(coll, 1, ops.PartitionOf(newOnly, newP)))
	must(t, mustCall(ee, "vector_insert", ops.EncodeVectorInsertArgs(newPhys, newOnly, []float32{1, 0, 0, 0})))

	res, err := ee.VectorSearch(ctx, coll, []float32{1, 0, 0, 0}, 50)
	must(t, err)
	seen := map[uint64]bool{}
	for _, r := range res {
		seen[r.ID] = true
	}
	if !seen[visible] {
		t.Fatalf("search missed old-gen id %d (reads should hit live gen)", visible)
	}
	if seen[newOnly] {
		t.Fatalf("search returned new-gen-only id %d — reads must stay on OLD gen", newOnly)
	}

	docs, _, _, err := ee.VectorScroll(ctx, coll, VectorFilter{}, 100, VectorScrollOpts{})
	must(t, err)
	sseen := map[uint64]bool{}
	for _, d := range docs {
		sseen[d.ID] = true
	}
	if !sseen[visible] {
		t.Fatalf("scroll missed old-gen id %d", visible)
	}
	if sseen[newOnly] {
		t.Fatalf("scroll returned new-gen-only id %d — reads must stay on OLD gen", newOnly)
	}
}

// TestEmbeddedDualWriteMVHitsBothGens proves MV add/delete dual-write to both gens.
func TestEmbeddedDualWriteMVHitsBothGens(t *testing.T) {
	const coll = "dwmv"
	const oldP, newP = 2, 4
	ee := setupReshardingMV(t, coll, oldP, newP)
	ctx := context.Background()

	var docID uint64 = 9
	oldPhys := string(ops.PartitionKeyGen(coll, 0, ops.PartitionOf(docID, oldP)))
	newPhys := string(ops.PartitionKeyGen(coll, 1, ops.PartitionOf(docID, newP)))
	tokens := [][]float32{{1, 0, 0, 0}}
	must(t, ee.VectorMVAdd(ctx, coll, docID, tokens, nil))
	if !physMVExists(t, ee, oldPhys, docID) {
		t.Fatalf("mv add did not land in OLD gen %q", oldPhys)
	}
	if !physMVExists(t, ee, newPhys, docID) {
		t.Fatalf("mv add did not land in NEW gen %q", newPhys)
	}

	existed, err := ee.VectorMVDelete(ctx, coll, docID)
	must(t, err)
	if !existed {
		t.Fatal("mv delete should report existed=true")
	}
	if physMVExists(t, ee, oldPhys, docID) {
		t.Fatalf("mv delete did not remove from OLD gen %q", oldPhys)
	}
	if physMVExists(t, ee, newPhys, docID) {
		t.Fatalf("mv delete did not remove from NEW gen %q", newPhys)
	}
}

// TestEmbeddedStablePassthroughNoDualWrite proves the Stable (no reshard) path is
// the existing single-target behavior: a write lands in exactly ONE partition
// (the live gen), with no second leg. This is the no-regression guard.
func TestEmbeddedStablePassthroughNoDualWrite(t *testing.T) {
	const coll = "stable"
	const P = 4
	e := newSingleEmbedded(t)
	waitLeaderEmbedded(t, e)
	ctx := context.Background()
	ee := e.(*embedded)
	must(t, e.CreateCollection(ctx, coll, VectorConfig{Dim: 4, Metric: vector.L2, M: 8, EfConstruction: 50, EfSearch: 64, Seed: 1, Partitions: P}))

	if _, _, dual := ee.dualTargets(coll, 7); dual {
		t.Fatal("Stable collection must not report dual=true")
	}

	var id uint64 = 7
	must(t, e.VectorInsert(ctx, coll, id, []float32{1, 0, 0, 0}))
	livePhys := string(ops.PartitionKeyGen(coll, 0, ops.PartitionOf(id, P)))
	if !physVecExists(t, ee, livePhys, id) {
		t.Fatalf("stable insert missing from live partition %q", livePhys)
	}
	for p := 0; p < P; p++ {
		phys := string(ops.PartitionKeyGen(coll, 0, p))
		if phys == livePhys {
			continue
		}
		if physVecExists(t, ee, phys, id) {
			t.Fatalf("stable insert leaked into second partition %q (dual-write on Stable!)", phys)
		}
	}
}

// TestEmbeddedDualWriteNewGenLegErrorSurfaces proves a failure of the NEW-gen leg
// surfaces as an error. The new-gen target partition is dropped so the second
// (target) Call fails; the live leg still succeeded (old gen is the source of
// truth), and an idempotent client retry re-applies both legs.
func TestEmbeddedDualWriteNewGenLegErrorSurfaces(t *testing.T) {
	const coll = "dwerr"
	const oldP, newP = 2, 4
	ee := setupReshardingDense(t, coll, oldP, newP)
	ctx := context.Background()

	var id uint64 = 3
	oldPhys := string(ops.PartitionKeyGen(coll, 0, ops.PartitionOf(id, oldP)))
	newPhys := string(ops.PartitionKeyGen(coll, 1, ops.PartitionOf(id, newP)))
	must(t, mustCall(ee, "vector_drop_collection", ops.EncodeDropCollectionArgs(newPhys)))

	err := ee.VectorInsert(ctx, coll, id, []float32{1, 0, 0, 0})
	if err == nil {
		t.Fatal("expected new-gen leg failure to surface as an error")
	}
	if !physVecExists(t, ee, oldPhys, id) {
		t.Fatal("old-gen leg should have been applied before the failing new-gen leg")
	}
}

// TestDualTargetsBothGensThroughCutover is the unit-level proof of the
// linearizable-catalog fix: dualTargets returns BOTH the old-gen and the new-gen
// physical partitions while Status==Resharding — BEFORE the catalog flip (live ==
// old, target == new) AND AFTER it (live == new, target == old). The post-cutover
// collapse to new-gen-only (the pre-fix bug) is gone. Steady-state (Status!=1)
// returns (live,"",false), byte-identical to e.partitionOf.
func TestDualTargetsBothGensThroughCutover(t *testing.T) {
	const coll = "dtcut"
	const oldP, newP = 2, 4
	ee := setupReshardingDense(t, coll, oldP, newP) // catalog at (oldP, gen0), Status=1, Old pinned
	var id uint64 = 7
	oldPhys := string(ops.PartitionKeyGen(coll, 0, ops.PartitionOf(id, oldP)))
	newPhys := string(ops.PartitionKeyGen(coll, 1, ops.PartitionOf(id, newP)))

	// Pre-cutover: catalog gen == OldGen ⇒ live == old, target == new, dual.
	live, target, dual := ee.dualTargets(coll, id)
	if !dual || live != oldPhys || target != newPhys {
		t.Fatalf("pre-cutover dualTargets = (%q,%q,%v), want (%q,%q,true)", live, target, dual, oldPhys, newPhys)
	}

	// Phase-4 cutover flip: live read gen becomes the new gen, reshard STILL on.
	must(t, ee.catalog.SetPartitionsGen(coll, newP, 1))

	// Post-cutover: catalog gen == NewGen ⇒ live == new, target == OLD, STILL dual.
	// This is the fix — the old gen keeps receiving writes after the flip.
	live, target, dual = ee.dualTargets(coll, id)
	if !dual || live != newPhys || target != oldPhys {
		t.Fatalf("post-cutover dualTargets = (%q,%q,%v), want (%q,%q,true) — old gen must still be written", live, target, dual, newPhys, oldPhys)
	}

	// Clear the reshard: back to steady state ⇒ single live target, no dual,
	// byte-identical to partitionOf.
	must(t, ee.catalog.SetReshardState(coll, ReshardState{Status: 0}))
	live, target, dual = ee.dualTargets(coll, id)
	pLive, _ := ee.partitionOf(coll, id)
	if dual || target != "" || live != pLive {
		t.Fatalf("post-clear dualTargets = (%q,%q,%v), want (%q,\"\",false)", live, target, dual, pLive)
	}
	if live != newPhys {
		t.Fatalf("post-clear live = %q, want new gen %q", live, newPhys)
	}
}

// TestEmbeddedDualWritePostCutoverLandsInBothGens is the end-to-end proof: a write
// submitted AFTER the catalog cutover flip (but while Status==Resharding) lands in
// BOTH the old-gen and new-gen physical partitions, and a read of the OLD gen
// reflects that post-cutover write (the old gen stays fresh for a lagging
// follower). Once the reshard state is cleared, a new write goes to the new gen
// ONLY (no regression to the steady-state path).
func TestEmbeddedDualWritePostCutoverLandsInBothGens(t *testing.T) {
	const coll = "dwpc"
	const oldP, newP = 2, 4
	ee := setupReshardingDense(t, coll, oldP, newP)
	ctx := context.Background()

	// Simulate the Phase-4 cutover: flip the live catalog gen to NEW while the
	// reshard state stays Resharding (Old still pinned).
	must(t, ee.catalog.SetPartitionsGen(coll, newP, 1))

	var id uint64 = 7
	oldPhys := string(ops.PartitionKeyGen(coll, 0, ops.PartitionOf(id, oldP)))
	newPhys := string(ops.PartitionKeyGen(coll, 1, ops.PartitionOf(id, newP)))

	must(t, ee.VectorInsert(ctx, coll, id, []float32{1, 2, 3, 4}))
	if !physVecExists(t, ee, newPhys, id) {
		t.Fatalf("post-cutover insert missing from NEW (live) gen %q", newPhys)
	}
	if !physVecExists(t, ee, oldPhys, id) {
		t.Fatalf("post-cutover insert missing from OLD gen %q — old gen stopped getting writes (the bug)", oldPhys)
	}

	// A delete after cutover must also remove from BOTH gens (old stays fresh).
	existed, err := ee.VectorDelete(ctx, coll, id)
	must(t, err)
	if !existed {
		t.Fatal("post-cutover delete should report existed=true")
	}
	if physVecExists(t, ee, newPhys, id) || physVecExists(t, ee, oldPhys, id) {
		t.Fatalf("post-cutover delete left a copy: new=%v old=%v", physVecExists(t, ee, newPhys, id), physVecExists(t, ee, oldPhys, id))
	}

	// Clear the reshard ⇒ writes go to the NEW gen only now.
	must(t, ee.catalog.SetReshardState(coll, ReshardState{Status: 0}))
	var id2 uint64 = 9
	new2 := string(ops.PartitionKeyGen(coll, 1, ops.PartitionOf(id2, newP)))
	old2 := string(ops.PartitionKeyGen(coll, 0, ops.PartitionOf(id2, oldP)))
	must(t, ee.VectorInsert(ctx, coll, id2, []float32{5, 0, 0, 0}))
	if !physVecExists(t, ee, new2, id2) {
		t.Fatalf("post-clear insert missing from new gen %q", new2)
	}
	if physVecExists(t, ee, old2, id2) {
		t.Fatalf("post-clear insert leaked into old gen %q (dual-write should be off)", old2)
	}
}

// TestEmbeddedDualWriteRetiringOldGenLegGoneTolerated is the regression test for
// the post-cutover drop race: after the catalog flips to the NEW gen (live=new),
// the SECONDARY dual-write leg targets the RETIRING OLD gen. If that old-gen
// partition was concurrently DROPPED (Phase-6 cleanup) the leg returns a
// "collection not found"-class error — but the authoritative new-gen leg already
// committed and all reads route there, so the WRITE MUST SUCCEED (the bug was the
// whole write failing with `unknown collection "...#0"`).
//
// It also proves the tolerance is TIGHTLY scoped:
//   - a TRANSIENT/other error (not partition-gone) on the SAME retiring old-gen leg
//     still SURFACES (so a still-alive old gen never goes silently stale);
//   - a not-found on the NEW gen (pre-cutover, the gen being built) still SURFACES
//     (covered by TestEmbeddedDualWriteNewGenLegErrorSurfaces; re-asserted here for
//     the pre-cutover direction directly).
func TestEmbeddedDualWriteRetiringOldGenLegGoneTolerated(t *testing.T) {
	const coll = "dwretire"
	const oldP, newP = 2, 4
	ee := setupReshardingDense(t, coll, oldP, newP)
	ctx := context.Background()

	// Phase-4 cutover: catalog routes reads to the NEW gen. Reshard still on, so the
	// secondary dual-write leg now targets the OLD (retiring) gen.
	must(t, ee.catalog.SetPartitionsGen(coll, newP, 1))

	var id uint64 = 7
	oldPhys := string(ops.PartitionKeyGen(coll, 0, ops.PartitionOf(id, oldP)))
	newPhys := string(ops.PartitionKeyGen(coll, 1, ops.PartitionOf(id, newP)))

	// Sanity: post-cutover dualTargets must put live=new, target=old (retiring).
	live, target, dual := ee.dualTargets(coll, id)
	if !dual || live != newPhys || target != oldPhys {
		t.Fatalf("post-cutover dualTargets = (%q,%q,%v), want (%q,%q,true)", live, target, dual, newPhys, oldPhys)
	}
	if ops.PartitionGenOf(target) >= ops.PartitionGenOf(live) {
		t.Fatalf("target %q (gen %d) should be retiring vs live %q (gen %d)",
			target, ops.PartitionGenOf(target), live, ops.PartitionGenOf(live))
	}

	// Simulate Phase-6 dropping the OLD-gen partition out from under the concurrent
	// write (the race the remote round-trip widens).
	must(t, mustCall(ee, "vector_drop_collection", ops.EncodeDropCollectionArgs(oldPhys)))

	// THE FIX: the write SUCCEEDS even though the retiring old-gen leg is gone.
	if err := ee.VectorInsert(ctx, coll, id, []float32{1, 2, 3, 4}); err != nil {
		t.Fatalf("retiring-old-gen-leg not-found must be tolerated, got: %v", err)
	}
	// The authoritative live (new-gen) leg committed.
	if !physVecExists(t, ee, newPhys, id) {
		t.Fatalf("live (new-gen) leg %q missing the committed write", newPhys)
	}

	// (a) A TRANSIENT/other error on the retiring old-gen leg still SURFACES: drive
	// applyDualWrite directly with live=new (alive) and target=old (retiring) but an
	// op that yields a NON-not-found error (an unregistered op => "op ... not
	// registered", which isCollectionGone does NOT match). Must propagate.
	_, err := ee.applyDualWrite("__no_such_op__", newPhys, oldPhys, true, WriteOpts{},
		func(phys string) []byte { return ops.EncodeExistsArgs(phys, id) })
	if err == nil {
		t.Fatal("a transient/other error on the retiring old-gen leg must surface (not be tolerated)")
	}
	if isCollectionGone(err) {
		t.Fatalf("the simulated transient error must not look collection-gone: %v", err)
	}

	// (b) Pre-cutover direction: target = NEW gen (being built, NOT retiring). A
	// not-found there must SURFACE. Drive applyDualWrite with live=old, target=new,
	// new partition dropped. targetRetiring is false (new gen > old gen), so even a
	// collection-gone error must propagate.
	var id2 uint64 = 11
	oldPhys2 := string(ops.PartitionKeyGen(coll, 0, ops.PartitionOf(id2, oldP)))
	newPhys2 := string(ops.PartitionKeyGen(coll, 1, ops.PartitionOf(id2, newP)))
	// oldPhys for id=7 was dropped above; recreate the old partition id2 lands in
	// (idempotent — create returns benign if it already exists) so the live (old)
	// leg can commit, isolating the assertion to the NEW-gen leg's not-found.
	cfg := VectorConfig{Dim: 4, Metric: vector.L2, M: 8, EfConstruction: 50, EfSearch: 64, Seed: 1}
	_ = mustCall(ee, "vector_create_collection", ops.EncodeCreateCollectionArgs(oldPhys2, cfg))
	must(t, mustCall(ee, "vector_drop_collection", ops.EncodeDropCollectionArgs(newPhys2)))
	_, err = ee.applyDualWrite("vector_insert", oldPhys2, newPhys2, true, WriteOpts{},
		func(phys string) []byte { return ops.EncodeVectorInsertArgs(phys, id2, []float32{1, 0, 0, 0}) })
	if err == nil {
		t.Fatal("pre-cutover NEW-gen leg not-found must surface (data loss in the gen being built)")
	}
	if !isCollectionGone(err) {
		t.Fatalf("expected a collection-gone error from the dropped new-gen leg, got: %v", err)
	}
}

// TestDualTargetsPreUpgradeMissingSourceFallback proves the backward-compat path:
// a reshard whose state lacks the old-gen pin (OldP<=0, e.g. begun by pre-upgrade
// code) degrades to today's behavior — dual-write to the new gen, collapsing to a
// single write once the catalog flips (live == new). No misroute, no panic.
func TestDualTargetsPreUpgradeMissingSourceFallback(t *testing.T) {
	const coll = "dtfb"
	const oldP, newP = 2, 4
	ee := setupReshardingDense(t, coll, oldP, newP)
	// Overwrite with a NO-source reshard state (simulate a pre-upgrade entry).
	must(t, ee.catalog.SetReshardState(coll, ReshardState{Status: 1, NewP: newP, NewGen: 1}))
	var id uint64 = 7
	oldPhys := string(ops.PartitionKeyGen(coll, 0, ops.PartitionOf(id, oldP)))
	newPhys := string(ops.PartitionKeyGen(coll, 1, ops.PartitionOf(id, newP)))

	// Pre-cutover: fallback still dual-writes (target=new != live=old).
	live, target, dual := ee.dualTargets(coll, id)
	if !dual || live != oldPhys || target != newPhys {
		t.Fatalf("fallback pre-cutover = (%q,%q,%v), want (%q,%q,true)", live, target, dual, oldPhys, newPhys)
	}

	// Post-cutover: fallback COLLAPSES to new-gen-only (live==new==target) — exactly
	// the pre-fix behavior for this one degraded reshard.
	must(t, ee.catalog.SetPartitionsGen(coll, newP, 1))
	live, target, dual = ee.dualTargets(coll, id)
	if dual || target != "" || live != newPhys {
		t.Fatalf("fallback post-cutover = (%q,%q,%v), want (%q,\"\",false)", live, target, dual, newPhys)
	}
}

// ---------------------------------------------------------------------------
// Online dense reshard (VectorReshard / VectorReshardAbort) tests.
// ---------------------------------------------------------------------------

// reshardScanGen returns the set of live ids found across all P partitions of
// generation gen (raw scan of the physical partitions, bypassing routing). Fails
// the test if any id appears in more than one partition or is mis-routed for
// PartitionOf(id, P).
func reshardScanGen(t *testing.T, ee *embedded, coll string, P int, gen uint32) map[uint64][]float32 {
	t.Helper()
	out := map[uint64][]float32{}
	for p := 0; p < P; p++ {
		phys := string(ops.PartitionKeyGen(coll, gen, p))
		body, err := ee.Call(context.Background(), "vector_scan_vectors", ops.EncodeScanVectorsArgs(phys))
		if err != nil {
			t.Fatalf("scan %s: %v", phys, err)
		}
		recs, err := ops.DecodeScanVectorsResult(body)
		if err != nil {
			t.Fatalf("decode scan %s: %v", phys, err)
		}
		for _, r := range recs {
			if want := ops.PartitionOf(r.ID, P); want != p {
				t.Fatalf("id %d found in partition %d but PartitionOf says %d (gen %d, P %d)", r.ID, p, want, gen, P)
			}
			if _, dup := out[r.ID]; dup {
				t.Fatalf("id %d present in more than one gen-%d partition", r.ID, gen)
			}
			out[r.ID] = append([]float32(nil), r.Vec...)
		}
	}
	return out
}

// genPartitionExists reports whether the physical partition (coll,gen,p) exists.
func genPartitionExists(t *testing.T, ee *embedded, coll string, gen uint32, p int) bool {
	t.Helper()
	phys := string(ops.PartitionKeyGen(coll, gen, p))
	_, err := ee.Call(context.Background(), "vector_get_config", ops.EncodeGetConfigArgs(phys))
	return err == nil
}

// runOnlineReshardQuiet seeds N vectors into a P=oldP collection, reshards to
// newP with NO concurrent traffic, and asserts full correctness of the result.
func runOnlineReshardQuiet(t *testing.T, oldP, newP, N int) {
	t.Helper()
	old := reshardDrainGrace
	reshardDrainGrace = 20 * time.Millisecond
	defer func() { reshardDrainGrace = old }()

	e := newSingleEmbedded(t)
	waitLeaderEmbedded(t, e)
	ee := e.(*embedded)
	ctx := context.Background()
	must(t, e.CreateCollection(ctx, "docs", VectorConfig{Dim: 4, Metric: vector.L2, M: 8, EfConstruction: 50, EfSearch: 64, Seed: 1, Partitions: oldP}))
	for id := uint64(1); id <= uint64(N); id++ {
		must(t, e.VectorUpsert(ctx, "docs", id, []float32{float32(id), 0, 0, 0}, fmt.Sprintf("doc-%d", id), VectorInsertOpts{Metadata: vector.Metadata{"even": vector.NewBool(id%2 == 0)}}))
	}

	must(t, e.VectorReshard(ctx, "docs", newP))

	// Catalog flipped to (newP, gen 1), reshard state cleared.
	p, gen, ok := ee.catalog.PartitionsGen("docs")
	if !ok || p != newP || gen != 1 {
		t.Fatalf("catalog (%d,%d,%v) want (%d,1,true)", p, gen, ok, newP)
	}
	if st, on := ee.catalog.ReshardState("docs"); on || st.Status != 0 {
		t.Fatalf("reshard state still set after completion: %+v on=%v", st, on)
	}

	// New gen holds exactly ids 1..N, each in the correct partition, once.
	got := reshardScanGen(t, ee, "docs", newP, 1)
	if len(got) != N {
		t.Fatalf("new gen holds %d ids, want %d", len(got), N)
	}
	for id := uint64(1); id <= uint64(N); id++ {
		if _, ok := got[id]; !ok {
			t.Fatalf("id %d missing from new gen", id)
		}
	}
	// Old-gen partitions all dropped.
	for p := 0; p < oldP; p++ {
		if genPartitionExists(t, ee, "docs", 0, p) {
			t.Fatalf("old gen-0 partition %d not dropped", p)
		}
	}
	// Search + scroll return the full set.
	all, _, _, err := e.VectorScroll(ctx, "docs", VectorFilter{}, 0, VectorScrollOpts{})
	must(t, err)
	if len(idSet(all)) != N {
		t.Fatalf("scroll: %d distinct ids, want %d", len(idSet(all)), N)
	}
	res, _, err := e.VectorSearchDocs(ctx, "docs", []float32{0.5, 0, 0, 0}, 5, VectorSearchOpts{})
	must(t, err)
	if len(res) == 0 || res[0].ID != 1 || res[0].Content != "doc-1" {
		t.Fatalf("post-reshard search: %+v", res)
	}
}

func TestEmbeddedOnlineReshardGrow(t *testing.T)   { runOnlineReshardQuiet(t, 4, 8, 300) }
func TestEmbeddedOnlineReshardShrink(t *testing.T) { runOnlineReshardQuiet(t, 8, 4, 300) }

// TestEmbeddedOnlineReshardConcurrentWrites is the core correctness gate for
// Races A (value clobber) and B (delete resurrection). It seeds a base set, then
// starts a reshard while many goroutines concurrently insert NEW ids, upsert
// (change values of) existing ids, and delete existing ids for the whole reshard
// duration. After VectorReshard returns and every writer joins, the new gen must
// contain EXACTLY the expected live set (base + inserts + upserted-values −
// deletes) with no lost writes, no resurrected deletes, and no stale/clobbered
// values. Run under -race.
func TestEmbeddedOnlineReshardConcurrentWrites(t *testing.T) {
	old := reshardDrainGrace
	reshardDrainGrace = 30 * time.Millisecond
	defer func() { reshardDrainGrace = old }()

	e := newSingleEmbedded(t)
	waitLeaderEmbedded(t, e)
	ee := e.(*embedded)
	ctx := context.Background()
	const oldP, newP = 4, 8
	must(t, e.CreateCollection(ctx, "docs", VectorConfig{Dim: 4, Metric: vector.L2, M: 8, EfConstruction: 50, EfSearch: 64, Seed: 1, Partitions: oldP}))

	// ID space layout (disjoint per writer role so the expected live set is exact):
	//   base ids:        1..baseN              (some deleted, some upserted)
	//   insert ids:      10000..10000+insN-1   (added during the reshard)
	const baseN = 400
	const insN = 400
	const upsertN = 200 // first upsertN base ids get their value changed
	const deleteN = 100 // base ids [upsertN+1 .. upsertN+deleteN] get deleted

	// vecOf encodes the id and a "version" tag in the vector so we can detect a
	// clobbered/stale value: base value tag=1, upserted value tag=2.
	vecOf := func(id uint64, tag float32) []float32 { return []float32{float32(id), tag, 0, 0} }

	for id := uint64(1); id <= baseN; id++ {
		must(t, e.VectorInsertExt(ctx, "docs", id, vecOf(id, 1), VectorInsertOpts{}))
	}

	// Expected live set, computed deterministically (writers below do EXACTLY this).
	expected := map[uint64][]float32{}
	for id := uint64(1); id <= baseN; id++ {
		expected[id] = vecOf(id, 1)
	}
	for id := uint64(1); id <= upsertN; id++ {
		expected[id] = vecOf(id, 2) // upserted value wins
	}
	for id := uint64(upsertN + 1); id <= upsertN+deleteN; id++ {
		delete(expected, id) // deleted ids gone
	}
	for i := 0; i < insN; i++ {
		id := uint64(10000 + i)
		expected[id] = vecOf(id, 1)
	}

	var wg sync.WaitGroup
	reshardErr := make(chan error, 1)
	wg.Add(1)
	go func() {
		defer wg.Done()
		reshardErr <- e.VectorReshard(ctx, "docs", newP)
	}()

	// Concurrent writers. Each role drives its disjoint id range to its final state
	// repeatedly (idempotent), hammering the copy/dual-write windows throughout the
	// reshard. They keep going until the reshard goroutine signals done.
	done := make(chan struct{})
	worker := func(fn func()) {
		wg.Add(1)
		go func() {
			defer wg.Done()
			for {
				select {
				case <-done:
					return
				default:
					fn()
				}
			}
		}()
	}
	// Upserter: change base ids 1..upsertN to tag=2.
	worker(func() {
		for id := uint64(1); id <= upsertN; id++ {
			_ = e.VectorUpsert(ctx, "docs", id, vecOf(id, 2), "", VectorInsertOpts{})
		}
	})
	// Deleter: delete base ids [upsertN+1 .. upsertN+deleteN].
	worker(func() {
		for id := uint64(upsertN + 1); id <= upsertN+deleteN; id++ {
			_, _ = e.VectorDelete(ctx, "docs", id)
		}
	})
	// Inserter: add new ids 10000..10000+insN-1.
	worker(func() {
		for i := 0; i < insN; i++ {
			_ = e.VectorInsertExt(ctx, "docs", uint64(10000+i), vecOf(uint64(10000+i), 1), VectorInsertOpts{})
		}
	})

	err := <-reshardErr
	close(done)
	wg.Wait()
	must(t, err)

	// Catalog flipped + state cleared.
	p, gen, ok := ee.catalog.PartitionsGen("docs")
	if !ok || p != newP || gen != 1 {
		t.Fatalf("catalog (%d,%d,%v) want (%d,1,true)", p, gen, ok, newP)
	}
	if st, on := ee.catalog.ReshardState("docs"); on || st.Status != 0 {
		t.Fatalf("reshard state still set: %+v on=%v", st, on)
	}

	// The new gen must contain EXACTLY the expected live set, with exact values.
	got := reshardScanGen(t, ee, "docs", newP, 1)
	if len(got) != len(expected) {
		t.Fatalf("new gen has %d ids, want %d", len(got), len(expected))
	}
	for id, wantVec := range expected {
		gotVec, ok := got[id]
		if !ok {
			t.Fatalf("id %d missing from new gen (lost write or wrongly deleted)", id)
		}
		if !reflect.DeepEqual(gotVec, wantVec) {
			t.Fatalf("id %d value = %v, want %v (clobbered/stale)", id, gotVec, wantVec)
		}
	}
	// No resurrected deletes: deleted ids must be absent.
	for id := uint64(upsertN + 1); id <= upsertN+deleteN; id++ {
		if _, ok := got[id]; ok {
			t.Fatalf("deleted id %d resurrected in new gen", id)
		}
	}
}

// TestEmbeddedOnlineReshardResume simulates a coordinator death mid-reshard
// (after Phase 1 turned on dual-write and partial new-gen partitions were created,
// before cutover) and asserts that re-invoking VectorReshard with the same target
// resumes and converges to a correct result.
func TestEmbeddedOnlineReshardResume(t *testing.T) {
	old := reshardDrainGrace
	reshardDrainGrace = 20 * time.Millisecond
	defer func() { reshardDrainGrace = old }()

	e := newSingleEmbedded(t)
	waitLeaderEmbedded(t, e)
	ee := e.(*embedded)
	ctx := context.Background()
	const oldP, newP, N = 4, 8, 200
	must(t, e.CreateCollection(ctx, "docs", VectorConfig{Dim: 4, Metric: vector.L2, M: 8, EfConstruction: 50, EfSearch: 64, Seed: 1, Partitions: oldP}))
	for id := uint64(1); id <= N; id++ {
		must(t, e.VectorUpsert(ctx, "docs", id, []float32{float32(id), 0, 0, 0}, fmt.Sprintf("doc-%d", id), VectorInsertOpts{}))
	}

	// Simulate a coordinator that died before cutover: dual-write is on
	// (ReshardState set), and only SOME new-gen partitions exist (a partial Phase 0),
	// with SOME records already copied into them. The catalog still reports the old
	// gen (no flip happened).
	cfgBody, err := ee.Call(ctx, "vector_get_config", ops.EncodeGetConfigArgs(string(ops.PartitionKeyGen("docs", 0, 0))))
	must(t, err)
	cfg, err := ops.DecodeGetConfigResult(cfgBody)
	must(t, err)
	cfg.Partitions = 0
	// Create only half of the new-gen partitions (partial Phase 0).
	for p := 0; p < newP/2; p++ {
		_, err := ee.Call(ctx, "vector_create_collection", ops.EncodeCreateCollectionArgs(string(ops.PartitionKeyGen("docs", 1, p)), cfg))
		must(t, err)
	}
	// Turn on the reshard state (Phase 1 result) toward (newP, gen1).
	must(t, ee.catalog.SetReshardState("docs", ReshardState{Status: 1, OldP: oldP, OldGen: 0, NewP: newP, NewGen: 1}))
	// Copy a handful of records into one existing new-gen partition (partial Phase 3)
	// so resume must tolerate already-present records (if-absent no-op).
	for id := uint64(1); id <= 20; id++ {
		newPhys := string(ops.PartitionKeyGen("docs", 1, ops.PartitionOf(id, newP)))
		if ops.PartitionOf(id, newP) < newP/2 { // only into partitions that exist
			_, err := ee.Call(ctx, "vector_insert_if_absent", ops.EncodeVectorInsertArgsExt(newPhys, id, []float32{float32(id), 0, 0, 0}, 0, nil, vector.SparseVector{}))
			must(t, err)
		}
	}

	// Resume: same target. Should self-heal the missing partitions and converge.
	must(t, e.VectorReshard(ctx, "docs", newP))

	p, gen, ok := ee.catalog.PartitionsGen("docs")
	if !ok || p != newP || gen != 1 {
		t.Fatalf("catalog (%d,%d,%v) want (%d,1,true)", p, gen, ok, newP)
	}
	if st, on := ee.catalog.ReshardState("docs"); on || st.Status != 0 {
		t.Fatalf("reshard state still set after resume: %+v on=%v", st, on)
	}
	got := reshardScanGen(t, ee, "docs", newP, 1)
	if len(got) != N {
		t.Fatalf("after resume new gen holds %d ids, want %d", len(got), N)
	}
	for id := uint64(1); id <= N; id++ {
		if _, ok := got[id]; !ok {
			t.Fatalf("id %d missing after resume", id)
		}
	}
	for p := 0; p < oldP; p++ {
		if genPartitionExists(t, ee, "docs", 0, p) {
			t.Fatalf("old gen-0 partition %d not dropped after resume", p)
		}
	}
}

// TestEmbeddedOnlineReshardAbort asserts a pre-cutover abort restores the old gen
// (data intact, new gen dropped, state Stable) and that an abort AFTER cutover is
// rejected.
func TestEmbeddedOnlineReshardAbort(t *testing.T) {
	e := newSingleEmbedded(t)
	waitLeaderEmbedded(t, e)
	ee := e.(*embedded)
	ctx := context.Background()
	const oldP, newP, N = 4, 8, 150
	must(t, e.CreateCollection(ctx, "docs", VectorConfig{Dim: 4, Metric: vector.L2, M: 8, EfConstruction: 50, EfSearch: 64, Seed: 1, Partitions: oldP}))
	for id := uint64(1); id <= N; id++ {
		must(t, e.VectorUpsert(ctx, "docs", id, []float32{float32(id), 0, 0, 0}, fmt.Sprintf("doc-%d", id), VectorInsertOpts{}))
	}

	// Drive a reshard into the pre-cutover state by hand: Phase 0 + Phase 1 (create
	// new-gen partitions, turn on dual-write) WITHOUT cutover.
	cfgBody, err := ee.Call(ctx, "vector_get_config", ops.EncodeGetConfigArgs(string(ops.PartitionKeyGen("docs", 0, 0))))
	must(t, err)
	cfg, err := ops.DecodeGetConfigResult(cfgBody)
	must(t, err)
	cfg.Partitions = 0
	for p := 0; p < newP; p++ {
		_, err := ee.Call(ctx, "vector_create_collection", ops.EncodeCreateCollectionArgs(string(ops.PartitionKeyGen("docs", 1, p)), cfg))
		must(t, err)
	}
	must(t, ee.catalog.SetReshardState("docs", ReshardState{Status: 1, OldP: oldP, OldGen: 0, NewP: newP, NewGen: 1}))

	// Abort (pre-cutover).
	must(t, e.VectorReshardAbort(ctx, "docs"))

	// Back on the old gen, state Stable, all data intact, new gen dropped.
	p, gen, ok := ee.catalog.PartitionsGen("docs")
	if !ok || p != oldP || gen != 0 {
		t.Fatalf("after abort catalog (%d,%d,%v) want (%d,0,true)", p, gen, ok, oldP)
	}
	if st, on := ee.catalog.ReshardState("docs"); on || st.Status != 0 {
		t.Fatalf("reshard state still set after abort: %+v on=%v", st, on)
	}
	all, _, _, err := e.VectorScroll(ctx, "docs", VectorFilter{}, 0, VectorScrollOpts{})
	must(t, err)
	if len(idSet(all)) != N {
		t.Fatalf("after abort scroll: %d distinct, want %d", len(idSet(all)), N)
	}
	for p := 0; p < newP; p++ {
		if genPartitionExists(t, ee, "docs", 1, p) {
			t.Fatalf("new gen-1 partition %d not dropped after abort", p)
		}
	}

	// Abort with no reshard in progress is an error.
	if err := e.VectorReshardAbort(ctx, "docs"); err == nil {
		t.Fatal("abort with no reshard in progress should error")
	}

	// Now simulate AFTER cutover: status Resharding but live gen == new gen. Abort
	// must be rejected (committed).
	must(t, ee.catalog.SetPartitionsGen("docs", newP, 1))
	must(t, ee.catalog.SetReshardState("docs", ReshardState{Status: 1, OldP: oldP, OldGen: 0, NewP: newP, NewGen: 1}))
	if err := e.VectorReshardAbort(ctx, "docs"); err == nil {
		t.Fatal("abort AFTER cutover should error (reshard committed)")
	}
}

// mvDoc is a complete MV test record: its token matrix (unit vectors, so they
// survive index normalization byte-for-byte) and its metadata. The expected-set
// oracles below compare against these exact values to prove the copy threads the
// FULL token matrix + metadata through (a prior MV feature lost metadata).
type mvDoc struct {
	tokens [][]float32
	tag    int64 // metadata "tag" — bumped on overwrite so a clobber is detectable
}

// mvTokensFor builds a deterministic K-token matrix for id, each token a distinct
// unit vector so normalization is a no-op (exact float preservation is assertable)
// and MaxSim is tie-free. K varies with id so multi-token preservation is tested.
func mvTokensFor(id uint64) [][]float32 {
	k := int(id%3) + 1
	out := make([][]float32, k)
	for j := 0; j < k; j++ {
		out[j] = mvTokenAt(int(id)*4 + j)
	}
	return out
}

// reshardScanGenMV scans every physical partition of (coll,gen,P) and returns a
// map docID -> mvDoc (tokens + tag metadata). It fails the test if any docID is
// mis-routed for PartitionOf(id,P) or appears in more than one partition — the
// same invariants reshardScanGen checks for dense.
func reshardScanGenMV(t *testing.T, ee *embedded, coll string, P int, gen uint32) map[uint64]mvDoc {
	t.Helper()
	out := map[uint64]mvDoc{}
	for p := 0; p < P; p++ {
		phys := string(ops.PartitionKeyGen(coll, gen, p))
		body, err := ee.Call(context.Background(), "vector_mv_scan_vectors", ops.EncodeMVScanArgs(phys))
		if err != nil {
			t.Fatalf("mv scan %s: %v", phys, err)
		}
		recs, err := ops.DecodeMVScanResult(body)
		if err != nil {
			t.Fatalf("decode mv scan %s: %v", phys, err)
		}
		for _, r := range recs {
			if want := ops.PartitionOf(r.ID, P); want != p {
				t.Fatalf("doc %d found in partition %d but PartitionOf says %d (gen %d, P %d)", r.ID, p, want, gen, P)
			}
			if _, dup := out[r.ID]; dup {
				t.Fatalf("doc %d present in more than one gen-%d partition", r.ID, gen)
			}
			tag := int64(-1)
			if r.Metadata != nil {
				if v, ok := r.Metadata["tag"]; ok && v.Kind == vector.ValueInt {
					tag = v.Int
				}
			}
			toks := make([][]float32, len(r.Tokens))
			for i := range r.Tokens {
				toks[i] = append([]float32(nil), r.Tokens[i]...)
			}
			out[r.ID] = mvDoc{tokens: toks, tag: tag}
		}
	}
	return out
}

// mvTokensEqual compares two token matrices for exact float equality.
func mvTokensEqual(a, b [][]float32) bool {
	if len(a) != len(b) {
		return false
	}
	for i := range a {
		if len(a[i]) != len(b[i]) {
			return false
		}
		for j := range a[i] {
			if a[i][j] != b[i][j] {
				return false
			}
		}
	}
	return true
}

// runMVOnlineReshardQuiet seeds N MV docs (token matrix + metadata) into a P=oldP
// collection, reshards LIVE to newP with NO concurrent traffic, then asserts full
// correctness: every doc present once in the correct new-gen partition with its
// token matrix AND metadata preserved, old gen dropped, catalog at (newP,1), Stable.
func runMVOnlineReshardQuiet(t *testing.T, oldP, newP, N int) {
	t.Helper()
	old := reshardDrainGrace
	reshardDrainGrace = 20 * time.Millisecond
	defer func() { reshardDrainGrace = old }()

	e := newSingleEmbedded(t)
	waitLeaderEmbedded(t, e)
	ee := e.(*embedded)
	ctx := context.Background()
	must(t, e.VectorMVCreateCollection(ctx, "mv", MultiVectorConfig{Dim: 4, Partitions: oldP}))

	want := map[uint64]mvDoc{}
	for id := uint64(1); id <= uint64(N); id++ {
		toks := mvTokensFor(id)
		md := VectorMetadata{"tag": vector.NewInt(1), "name": vector.NewString(fmt.Sprintf("doc-%d", id))}
		must(t, e.VectorMVAdd(ctx, "mv", id, toks, md))
		want[id] = mvDoc{tokens: toks, tag: 1}
	}

	must(t, e.VectorMVReshard(ctx, "mv", newP))

	// Catalog flipped to (newP, gen 1), reshard state cleared (Stable).
	p, gen, ok := ee.catalog.PartitionsGen("mv")
	if !ok || p != newP || gen != 1 {
		t.Fatalf("catalog (%d,%d,%v) want (%d,1,true)", p, gen, ok, newP)
	}
	if st, on := ee.catalog.ReshardState("mv"); on || st.Status != 0 {
		t.Fatalf("reshard state still set after completion: %+v on=%v", st, on)
	}

	// New gen holds exactly ids 1..N, correct partition, once, tokens+metadata intact.
	got := reshardScanGenMV(t, ee, "mv", newP, 1)
	if len(got) != N {
		t.Fatalf("new gen holds %d docs, want %d", len(got), N)
	}
	for id, w := range want {
		g, ok := got[id]
		if !ok {
			t.Fatalf("doc %d missing from new gen", id)
		}
		if !mvTokensEqual(g.tokens, w.tokens) {
			t.Fatalf("doc %d tokens not preserved: got %v want %v", id, g.tokens, w.tokens)
		}
		if g.tag != w.tag {
			t.Fatalf("doc %d metadata tag = %d, want %d (metadata dropped across reshard)", id, g.tag, w.tag)
		}
	}
	// Old-gen partitions all dropped.
	for p := 0; p < oldP; p++ {
		if _, err := ee.Call(ctx, "vector_mv_get_config", ops.EncodeMVGetConfigArgs(string(ops.PartitionKeyGen("mv", 0, p)))); err == nil {
			t.Fatalf("old gen-0 partition %d not dropped", p)
		}
	}
	// MV search through the post-reshard fan-out returns the winner at rank 0 with
	// metadata intact (proves the new-gen partitions are searchable + metadata
	// survived the flip). The full-set token/metadata preservation is asserted above
	// via the raw scan oracle (reshardScanGenMV), which does not depend on MaxSim
	// angle monotonicity. The winner uses a small id so its token angle stays in the
	// first quadrant (tie-free).
	const winner = 7
	res, _, err := e.VectorMVSearch(ctx, "mv", [][]float32{mvTokensFor(winner)[0]}, 5, MultiSearchOpts{CandidatesPerToken: 100})
	must(t, err)
	if len(res) == 0 || res[0].ID != winner {
		t.Fatalf("winner %d not at rank 0 after reshard: %+v", winner, res)
	}
	gv, hasMD := res[0].Metadata["tag"]
	if !hasMD || gv.Kind != vector.ValueInt || gv.Int != 1 {
		t.Fatalf("winner %d metadata tag = %+v (present=%v) via search, want int 1 (metadata dropped across reshard)", winner, gv, hasMD)
	}
}

func TestEmbeddedMVOnlineReshardGrow(t *testing.T)   { runMVOnlineReshardQuiet(t, 4, 8, 300) }
func TestEmbeddedMVOnlineReshardShrink(t *testing.T) { runMVOnlineReshardQuiet(t, 8, 4, 300) }

// TestEmbeddedMVOnlineReshardConcurrentWrites is the MV money test for Races A
// (value clobber) and B (delete resurrection). It seeds base MV docs, starts a
// reshard, then concurrently mv-adds NEW docs, mv-adds (re-adds = overwrites)
// EXISTING docs with a changed token + metadata tag, and mv-deletes existing docs
// for the whole reshard duration. After join, the new gen must contain EXACTLY the
// expected live set with exact per-doc token matrix + metadata tag — no lost
// writes, no resurrected deletes, no clobbered values. Run under -race.
func TestEmbeddedMVOnlineReshardConcurrentWrites(t *testing.T) {
	old := reshardDrainGrace
	reshardDrainGrace = 30 * time.Millisecond
	defer func() { reshardDrainGrace = old }()

	e := newSingleEmbedded(t)
	waitLeaderEmbedded(t, e)
	ee := e.(*embedded)
	ctx := context.Background()
	const oldP, newP = 4, 8
	must(t, e.VectorMVCreateCollection(ctx, "mv", MultiVectorConfig{Dim: 4, Partitions: oldP}))

	// Disjoint id ranges per writer role so the expected live set is exact:
	//   base ids:    1..baseN          (some deleted, some overwritten)
	//   insert ids:  10000..10000+insN-1
	const baseN = 400
	const insN = 400
	const upsertN = 200 // first upsertN base ids get re-added (overwritten) tag=2
	const deleteN = 100 // base ids [upsertN+1 .. upsertN+deleteN] get deleted

	// tokensTagged returns the doc's base token matrix, but the FIRST token is
	// chosen by tag so an overwrite (tag 1 -> 2) produces a detectably different
	// token matrix; copy-clobber would leave the stale tag-1 tokens.
	tokensTagged := func(id uint64, tag int) [][]float32 {
		toks := mvTokensFor(id)
		toks[0] = mvTokenAt(int(id)*4 + 1000*tag)
		return toks
	}
	mdTagged := func(tag int) VectorMetadata { return VectorMetadata{"tag": vector.NewInt(int64(tag))} }

	for id := uint64(1); id <= baseN; id++ {
		must(t, e.VectorMVAdd(ctx, "mv", id, tokensTagged(id, 1), mdTagged(1)))
	}

	// Expected live set, computed deterministically (writers below do EXACTLY this).
	expected := map[uint64]mvDoc{}
	for id := uint64(1); id <= baseN; id++ {
		expected[id] = mvDoc{tokens: tokensTagged(id, 1), tag: 1}
	}
	for id := uint64(1); id <= upsertN; id++ {
		expected[id] = mvDoc{tokens: tokensTagged(id, 2), tag: 2} // overwrite wins
	}
	for id := uint64(upsertN + 1); id <= upsertN+deleteN; id++ {
		delete(expected, id) // deleted ids gone
	}
	for i := 0; i < insN; i++ {
		id := uint64(10000 + i)
		expected[id] = mvDoc{tokens: tokensTagged(id, 1), tag: 1}
	}

	var wg sync.WaitGroup
	reshardErr := make(chan error, 1)
	wg.Add(1)
	go func() {
		defer wg.Done()
		reshardErr <- e.VectorMVReshard(ctx, "mv", newP)
	}()

	done := make(chan struct{})
	worker := func(fn func()) {
		wg.Add(1)
		go func() {
			defer wg.Done()
			for {
				select {
				case <-done:
					return
				default:
					fn()
				}
			}
		}()
	}
	// Overwriter: re-add base ids 1..upsertN with tag=2 tokens+metadata.
	worker(func() {
		for id := uint64(1); id <= upsertN; id++ {
			_ = e.VectorMVAdd(ctx, "mv", id, tokensTagged(id, 2), mdTagged(2))
		}
	})
	// Deleter: delete base ids [upsertN+1 .. upsertN+deleteN].
	worker(func() {
		for id := uint64(upsertN + 1); id <= upsertN+deleteN; id++ {
			_, _ = e.VectorMVDelete(ctx, "mv", id)
		}
	})
	// Inserter: add new ids 10000..10000+insN-1.
	worker(func() {
		for i := 0; i < insN; i++ {
			id := uint64(10000 + i)
			_ = e.VectorMVAdd(ctx, "mv", id, tokensTagged(id, 1), mdTagged(1))
		}
	})

	err := <-reshardErr
	close(done)
	wg.Wait()
	must(t, err)

	p, gen, ok := ee.catalog.PartitionsGen("mv")
	if !ok || p != newP || gen != 1 {
		t.Fatalf("catalog (%d,%d,%v) want (%d,1,true)", p, gen, ok, newP)
	}
	if st, on := ee.catalog.ReshardState("mv"); on || st.Status != 0 {
		t.Fatalf("reshard state still set: %+v on=%v", st, on)
	}

	got := reshardScanGenMV(t, ee, "mv", newP, 1)
	if len(got) != len(expected) {
		t.Fatalf("new gen has %d docs, want %d", len(got), len(expected))
	}
	for id, w := range expected {
		g, ok := got[id]
		if !ok {
			t.Fatalf("doc %d missing from new gen (lost write or wrongly deleted)", id)
		}
		if g.tag != w.tag {
			t.Fatalf("doc %d metadata tag = %d, want %d (clobbered/stale)", id, g.tag, w.tag)
		}
		if !mvTokensEqual(g.tokens, w.tokens) {
			t.Fatalf("doc %d tokens = %v, want %v (clobbered/stale)", id, g.tokens, w.tokens)
		}
	}
	for id := uint64(upsertN + 1); id <= upsertN+deleteN; id++ {
		if _, ok := got[id]; ok {
			t.Fatalf("deleted doc %d resurrected in new gen", id)
		}
	}
}

// TestEmbeddedMVOnlineReshardResume simulates an MV coordinator death mid-reshard
// (dual-write on, partial Phase 0 + Phase 3) and asserts re-invoking VectorMVReshard
// with the same target resumes and converges.
func TestEmbeddedMVOnlineReshardResume(t *testing.T) {
	old := reshardDrainGrace
	reshardDrainGrace = 20 * time.Millisecond
	defer func() { reshardDrainGrace = old }()

	e := newSingleEmbedded(t)
	waitLeaderEmbedded(t, e)
	ee := e.(*embedded)
	ctx := context.Background()
	const oldP, newP, N = 4, 8, 200
	must(t, e.VectorMVCreateCollection(ctx, "mv", MultiVectorConfig{Dim: 4, Partitions: oldP}))
	for id := uint64(1); id <= N; id++ {
		must(t, e.VectorMVAdd(ctx, "mv", id, mvTokensFor(id), VectorMetadata{"tag": vector.NewInt(1)}))
	}

	// Simulate pre-cutover crash: read config from an old-gen partition, create only
	// half the new-gen partitions (partial Phase 0), turn on dual-write (Phase 1
	// result), copy a handful of docs into existing new-gen partitions (partial Phase 3).
	cfgBody, err := ee.Call(ctx, "vector_mv_get_config", ops.EncodeMVGetConfigArgs(string(ops.PartitionKeyGen("mv", 0, 0))))
	must(t, err)
	_, cfg, err := ops.DecodeMVCreateArgs(cfgBody)
	must(t, err)
	cfg.Partitions = 0
	for p := 0; p < newP/2; p++ {
		_, err := ee.Call(ctx, "vector_mv_create_collection", ops.EncodeMVCreateArgs(string(ops.PartitionKeyGen("mv", 1, p)), cfg))
		must(t, err)
	}
	must(t, ee.catalog.SetReshardState("mv", ReshardState{Status: 1, OldP: oldP, OldGen: 0, NewP: newP, NewGen: 1}))
	for id := uint64(1); id <= 20; id++ {
		if ops.PartitionOf(id, newP) < newP/2 {
			newPhys := string(ops.PartitionKeyGen("mv", 1, ops.PartitionOf(id, newP)))
			_, err := ee.Call(ctx, "vector_mv_add_if_absent", ops.EncodeMVAddArgs(newPhys, id, mvTokensFor(id), vector.Metadata{"tag": vector.NewInt(1)}))
			must(t, err)
		}
	}

	must(t, e.VectorMVReshard(ctx, "mv", newP))

	p, gen, ok := ee.catalog.PartitionsGen("mv")
	if !ok || p != newP || gen != 1 {
		t.Fatalf("catalog (%d,%d,%v) want (%d,1,true)", p, gen, ok, newP)
	}
	if st, on := ee.catalog.ReshardState("mv"); on || st.Status != 0 {
		t.Fatalf("reshard state still set after resume: %+v on=%v", st, on)
	}
	got := reshardScanGenMV(t, ee, "mv", newP, 1)
	if len(got) != N {
		t.Fatalf("after resume new gen holds %d docs, want %d", len(got), N)
	}
	for id := uint64(1); id <= N; id++ {
		g, ok := got[id]
		if !ok {
			t.Fatalf("doc %d missing after resume", id)
		}
		if g.tag != 1 {
			t.Fatalf("doc %d metadata tag = %d after resume, want 1", id, g.tag)
		}
	}
	for p := 0; p < oldP; p++ {
		if _, err := ee.Call(ctx, "vector_mv_get_config", ops.EncodeMVGetConfigArgs(string(ops.PartitionKeyGen("mv", 0, p)))); err == nil {
			t.Fatalf("old gen-0 partition %d not dropped after resume", p)
		}
	}
}

// TestEmbeddedMVOnlineReshardAbort asserts a pre-cutover MV abort restores the old
// gen (data intact, new gen dropped, Stable) and that an abort AFTER cutover errors.
func TestEmbeddedMVOnlineReshardAbort(t *testing.T) {
	e := newSingleEmbedded(t)
	waitLeaderEmbedded(t, e)
	ee := e.(*embedded)
	ctx := context.Background()
	const oldP, newP, N = 4, 8, 150
	must(t, e.VectorMVCreateCollection(ctx, "mv", MultiVectorConfig{Dim: 4, Partitions: oldP}))
	for id := uint64(1); id <= N; id++ {
		must(t, e.VectorMVAdd(ctx, "mv", id, mvTokensFor(id), VectorMetadata{"tag": vector.NewInt(1)}))
	}

	// Drive into pre-cutover state by hand: Phase 0 + Phase 1 without cutover.
	cfgBody, err := ee.Call(ctx, "vector_mv_get_config", ops.EncodeMVGetConfigArgs(string(ops.PartitionKeyGen("mv", 0, 0))))
	must(t, err)
	_, cfg, err := ops.DecodeMVCreateArgs(cfgBody)
	must(t, err)
	cfg.Partitions = 0
	for p := 0; p < newP; p++ {
		_, err := ee.Call(ctx, "vector_mv_create_collection", ops.EncodeMVCreateArgs(string(ops.PartitionKeyGen("mv", 1, p)), cfg))
		must(t, err)
	}
	must(t, ee.catalog.SetReshardState("mv", ReshardState{Status: 1, OldP: oldP, OldGen: 0, NewP: newP, NewGen: 1}))

	must(t, e.VectorMVReshardAbort(ctx, "mv"))

	// Back on the old gen, Stable, all data intact, new gen dropped.
	p, gen, ok := ee.catalog.PartitionsGen("mv")
	if !ok || p != oldP || gen != 0 {
		t.Fatalf("after abort catalog (%d,%d,%v) want (%d,0,true)", p, gen, ok, oldP)
	}
	if st, on := ee.catalog.ReshardState("mv"); on || st.Status != 0 {
		t.Fatalf("reshard state still set after abort: %+v on=%v", st, on)
	}
	total := 0
	for p := 0; p < oldP; p++ {
		total += mvPhysCount(t, ee, string(ops.PartitionKeyGen("mv", 0, p)))
	}
	if total != N {
		t.Fatalf("after abort old gen holds %d docs, want %d", total, N)
	}
	for p := 0; p < newP; p++ {
		if _, err := ee.Call(ctx, "vector_mv_get_config", ops.EncodeMVGetConfigArgs(string(ops.PartitionKeyGen("mv", 1, p)))); err == nil {
			t.Fatalf("new gen-1 partition %d not dropped after abort", p)
		}
	}

	// Abort with no reshard in progress is an error.
	if err := e.VectorMVReshardAbort(ctx, "mv"); err == nil {
		t.Fatal("mv abort with no reshard in progress should error")
	}

	// Simulate AFTER cutover: status Resharding but live gen == new gen. Abort errors.
	must(t, ee.catalog.SetPartitionsGen("mv", newP, 1))
	must(t, ee.catalog.SetReshardState("mv", ReshardState{Status: 1, OldP: oldP, OldGen: 0, NewP: newP, NewGen: 1}))
	if err := e.VectorMVReshardAbort(ctx, "mv"); err == nil {
		t.Fatal("mv abort AFTER cutover should error (reshard committed)")
	}
}

// TestEmbeddedOnlineReshardFinalizeAfterPostCutoverCrash reproduces the
// post-cutover/pre-Stable-clear crash window: the Phase-4 catalog flip is durable
// (live PartitionsGen == {newP,newGen}) but the Phase-5 Stable clear never ran, so
// the collection is stuck Resharding toward its own live gen, with old-gen
// partitions still present as orphans. Re-invoking VectorReshard with the
// already-completed target must FINALIZE: clear the status (-> Stable) and sweep
// the orphan old-gen partitions, returning nil and leaving live data intact at the
// new gen. A subsequent reshard to a different P must then work (un-stuck).
func TestEmbeddedOnlineReshardFinalizeAfterPostCutoverCrash(t *testing.T) {
	old := reshardDrainGrace
	reshardDrainGrace = 20 * time.Millisecond
	defer func() { reshardDrainGrace = old }()

	e := newSingleEmbedded(t)
	waitLeaderEmbedded(t, e)
	ee := e.(*embedded)
	ctx := context.Background()
	const oldP, newP, N = 4, 8, 200
	must(t, e.CreateCollection(ctx, "docs", VectorConfig{Dim: 4, Metric: vector.L2, M: 8, EfConstruction: 50, EfSearch: 64, Seed: 1, Partitions: oldP}))
	for id := uint64(1); id <= N; id++ {
		must(t, e.VectorUpsert(ctx, "docs", id, []float32{float32(id), 0, 0, 0}, fmt.Sprintf("doc-%d", id), VectorInsertOpts{}))
	}

	// Build the new-gen partitions and copy every record into them (a completed
	// Phase 0 + Phase 3), then simulate a crash AFTER the Phase-4 flip but BEFORE
	// the Phase-5 Stable clear: live PartitionsGen flipped to (newP,1), status still
	// Resharding toward (newP,1), and the old-gen partitions still present as orphans.
	cfgBody, err := ee.Call(ctx, "vector_get_config", ops.EncodeGetConfigArgs(string(ops.PartitionKeyGen("docs", 0, 0))))
	must(t, err)
	cfg, err := ops.DecodeGetConfigResult(cfgBody)
	must(t, err)
	cfg.Partitions = 0
	for p := 0; p < newP; p++ {
		_, err := ee.Call(ctx, "vector_create_collection", ops.EncodeCreateCollectionArgs(string(ops.PartitionKeyGen("docs", 1, p)), cfg))
		must(t, err)
	}
	for id := uint64(1); id <= N; id++ {
		newPhys := string(ops.PartitionKeyGen("docs", 1, ops.PartitionOf(id, newP)))
		_, err := ee.Call(ctx, "vector_insert_if_absent", ops.EncodeVectorInsertArgsExt(newPhys, id, []float32{float32(id), 0, 0, 0}, 0, nil, vector.SparseVector{}))
		must(t, err)
	}
	must(t, ee.catalog.SetPartitionsGen("docs", newP, 1)) // Phase-4 flip durable
	must(t, ee.catalog.SetReshardState("docs", ReshardState{Status: 1, OldP: oldP, OldGen: 0, NewP: newP, NewGen: 1}))

	// Re-invoke with the already-completed target. Must finalize (not error).
	must(t, e.VectorReshard(ctx, "docs", newP))

	// Status cleared to Stable; catalog still at the new gen.
	p, gen, ok := ee.catalog.PartitionsGen("docs")
	if !ok || p != newP || gen != 1 {
		t.Fatalf("after finalize catalog (%d,%d,%v) want (%d,1,true)", p, gen, ok, newP)
	}
	if st, on := ee.catalog.ReshardState("docs"); on || st.Status != 0 {
		t.Fatalf("reshard state still set after finalize: %+v on=%v", st, on)
	}
	// Orphan old-gen partitions swept.
	for p := 0; p < oldP; p++ {
		if genPartitionExists(t, ee, "docs", 0, p) {
			t.Fatalf("orphan old gen-0 partition %d not dropped by finalize", p)
		}
	}
	// Live data intact and correct at the new gen.
	got := reshardScanGen(t, ee, "docs", newP, 1)
	if len(got) != N {
		t.Fatalf("after finalize new gen holds %d ids, want %d", len(got), N)
	}
	for id := uint64(1); id <= N; id++ {
		if _, ok := got[id]; !ok {
			t.Fatalf("id %d missing after finalize", id)
		}
	}
	all, _, _, err := e.VectorScroll(ctx, "docs", VectorFilter{}, 0, VectorScrollOpts{})
	must(t, err)
	if len(idSet(all)) != N {
		t.Fatalf("after finalize scroll: %d distinct, want %d", len(idSet(all)), N)
	}

	// The collection is un-stuck: a fresh reshard to a DIFFERENT P now works.
	must(t, e.VectorReshard(ctx, "docs", 4))
	p, gen, ok = ee.catalog.PartitionsGen("docs")
	if !ok || p != 4 || gen != 2 {
		t.Fatalf("after subsequent reshard catalog (%d,%d,%v) want (4,2,true)", p, gen, ok)
	}
	if st, on := ee.catalog.ReshardState("docs"); on || st.Status != 0 {
		t.Fatalf("reshard state set after subsequent reshard: %+v on=%v", st, on)
	}
	got2 := reshardScanGen(t, ee, "docs", 4, 2)
	if len(got2) != N {
		t.Fatalf("after subsequent reshard gen2 holds %d ids, want %d", len(got2), N)
	}

	// Abort on the stuck post-cutover state still refuses (committed), with the
	// recovery-pointing message.
	must(t, ee.catalog.SetPartitionsGen("docs", newP, 5))
	for p := 0; p < newP; p++ {
		_, _ = ee.Call(ctx, "vector_create_collection", ops.EncodeCreateCollectionArgs(string(ops.PartitionKeyGen("docs", 5, p)), cfg))
	}
	must(t, ee.catalog.SetReshardState("docs", ReshardState{Status: 1, OldP: 4, OldGen: 2, NewP: newP, NewGen: 5}))
	err = e.VectorReshardAbort(ctx, "docs")
	if err == nil {
		t.Fatal("abort on post-cutover state should error (committed)")
	}
	if !strings.Contains(err.Error(), "re-invoke VectorReshard to finalize") {
		t.Fatalf("abort error should point to recovery path, got: %v", err)
	}
}

// TestEmbeddedMVOnlineReshardFinalizeAfterPostCutoverCrash is the MV mirror of
// TestEmbeddedOnlineReshardFinalizeAfterPostCutoverCrash.
func TestEmbeddedMVOnlineReshardFinalizeAfterPostCutoverCrash(t *testing.T) {
	old := reshardDrainGrace
	reshardDrainGrace = 20 * time.Millisecond
	defer func() { reshardDrainGrace = old }()

	e := newSingleEmbedded(t)
	waitLeaderEmbedded(t, e)
	ee := e.(*embedded)
	ctx := context.Background()
	const oldP, newP, N = 4, 8, 150
	must(t, e.VectorMVCreateCollection(ctx, "mv", MultiVectorConfig{Dim: 4, Partitions: oldP}))
	for id := uint64(1); id <= N; id++ {
		must(t, e.VectorMVAdd(ctx, "mv", id, mvTokensFor(id), VectorMetadata{"tag": vector.NewInt(1)}))
	}

	// Completed Phase 0 + Phase 3, then crash after the Phase-4 flip but before the
	// Phase-5 Stable clear: live flipped to (newP,1), status still Resharding toward
	// (newP,1), old-gen MV partitions still present as orphans.
	cfgBody, err := ee.Call(ctx, "vector_mv_get_config", ops.EncodeMVGetConfigArgs(string(ops.PartitionKeyGen("mv", 0, 0))))
	must(t, err)
	_, cfg, err := ops.DecodeMVCreateArgs(cfgBody)
	must(t, err)
	cfg.Partitions = 0
	for p := 0; p < newP; p++ {
		_, err := ee.Call(ctx, "vector_mv_create_collection", ops.EncodeMVCreateArgs(string(ops.PartitionKeyGen("mv", 1, p)), cfg))
		must(t, err)
	}
	for id := uint64(1); id <= N; id++ {
		newPhys := string(ops.PartitionKeyGen("mv", 1, ops.PartitionOf(id, newP)))
		_, err := ee.Call(ctx, "vector_mv_add_if_absent", ops.EncodeMVAddArgs(newPhys, id, mvTokensFor(id), VectorMetadata{"tag": vector.NewInt(1)}))
		must(t, err)
	}
	must(t, ee.catalog.SetPartitionsGen("mv", newP, 1)) // Phase-4 flip durable
	must(t, ee.catalog.SetReshardState("mv", ReshardState{Status: 1, OldP: oldP, OldGen: 0, NewP: newP, NewGen: 1}))

	// Re-invoke with the already-completed target. Must finalize (not error).
	must(t, e.VectorMVReshard(ctx, "mv", newP))

	p, gen, ok := ee.catalog.PartitionsGen("mv")
	if !ok || p != newP || gen != 1 {
		t.Fatalf("after finalize catalog (%d,%d,%v) want (%d,1,true)", p, gen, ok, newP)
	}
	if st, on := ee.catalog.ReshardState("mv"); on || st.Status != 0 {
		t.Fatalf("reshard state still set after finalize: %+v on=%v", st, on)
	}
	for p := 0; p < oldP; p++ {
		if _, err := ee.Call(ctx, "vector_mv_get_config", ops.EncodeMVGetConfigArgs(string(ops.PartitionKeyGen("mv", 0, p)))); err == nil {
			t.Fatalf("orphan old gen-0 MV partition %d not dropped by finalize", p)
		}
	}
	// Live MV data intact, token matrix + metadata preserved.
	got := reshardScanGenMV(t, ee, "mv", newP, 1)
	if len(got) != N {
		t.Fatalf("after finalize new gen holds %d docs, want %d", len(got), N)
	}
	for id := uint64(1); id <= N; id++ {
		d, ok := got[id]
		if !ok {
			t.Fatalf("doc %d missing after finalize", id)
		}
		if !mvTokensEqual(d.tokens, mvTokensFor(id)) {
			t.Fatalf("doc %d token matrix not preserved by finalize", id)
		}
		if d.tag != 1 {
			t.Fatalf("doc %d metadata tag=%d, want 1", id, d.tag)
		}
	}

	// Un-stuck: a fresh reshard to a DIFFERENT P now works.
	must(t, e.VectorMVReshard(ctx, "mv", 4))
	p, gen, ok = ee.catalog.PartitionsGen("mv")
	if !ok || p != 4 || gen != 2 {
		t.Fatalf("after subsequent mv reshard catalog (%d,%d,%v) want (4,2,true)", p, gen, ok)
	}
	got2 := reshardScanGenMV(t, ee, "mv", 4, 2)
	if len(got2) != N {
		t.Fatalf("after subsequent mv reshard gen2 holds %d docs, want %d", len(got2), N)
	}

	// Abort on stuck post-cutover state still refuses, with recovery-pointing message.
	must(t, ee.catalog.SetPartitionsGen("mv", newP, 5))
	for p := 0; p < newP; p++ {
		_, _ = ee.Call(ctx, "vector_mv_create_collection", ops.EncodeMVCreateArgs(string(ops.PartitionKeyGen("mv", 5, p)), cfg))
	}
	must(t, ee.catalog.SetReshardState("mv", ReshardState{Status: 1, OldP: 4, OldGen: 2, NewP: newP, NewGen: 5}))
	err = e.VectorMVReshardAbort(ctx, "mv")
	if err == nil {
		t.Fatal("mv abort on post-cutover state should error (committed)")
	}
	if !strings.Contains(err.Error(), "re-invoke VectorMVReshard to finalize") {
		t.Fatalf("mv abort error should point to recovery path, got: %v", err)
	}
}

// ----------------------------------------------------------------------------
// Phase-4.5 all-nodes-applied cutover gate wiring
// ----------------------------------------------------------------------------

// TestEmbeddedOnlineReshardGateSequenceDense asserts the cutover→gate→clear→drop
// ordering of the dense online reshard: when awaitCutoverGate is reached, the
// catalog has ALREADY flipped to the new gen, the reshard state is STILL Resharding
// (dual-write to both gens is on), and EVERY old-gen partition is STILL present
// (not yet dropped). The gate must be invoked with the correct (collection,newGen).
// After the reshard returns, the old gen is gone. This proves the old gen is never
// retired before the gate, the invariant the lagging-follower linearizability rests
// on. (Single-node: the gate itself is a no-op, but the SEQUENCING around it is what
// we assert here — the gate is wired in the right place relative to clear+drop.)
func TestEmbeddedOnlineReshardGateSequenceDense(t *testing.T) {
	oldGrace := reshardDrainGrace
	reshardDrainGrace = 20 * time.Millisecond
	defer func() { reshardDrainGrace = oldGrace }()

	e := newSingleEmbedded(t)
	waitLeaderEmbedded(t, e)
	ee := e.(*embedded)
	ctx := context.Background()
	const oldP, newP, N = 4, 8, 120
	must(t, e.CreateCollection(ctx, "docs", VectorConfig{Dim: 4, Metric: vector.L2, M: 8, EfConstruction: 50, EfSearch: 64, Seed: 1, Partitions: oldP}))
	for id := uint64(1); id <= N; id++ {
		must(t, e.VectorUpsert(ctx, "docs", id, []float32{float32(id), 0, 0, 0}, fmt.Sprintf("doc-%d", id), VectorInsertOpts{}))
	}

	var (
		gateCalled    bool
		gateColl      string
		gateGen       uint32
		flippedAtGate bool
		statusAtGate  uint8
		oldGenAtGate  int // count of old-gen partitions present at the gate
	)
	reshardGateHook = func(collection string, newGen uint32) {
		gateCalled = true
		gateColl, gateGen = collection, newGen
		// The catalog flip precedes the gate.
		if p, gen, ok := ee.catalog.PartitionsGen("docs"); ok && p == newP && gen == newGen {
			flippedAtGate = true
		}
		// Dual-write must still be on at the gate (Status == Resharding).
		if st, on := ee.catalog.ReshardState("docs"); on {
			statusAtGate = st.Status
		}
		// The old gen must still be fully present at the gate (not dropped yet).
		for p := 0; p < oldP; p++ {
			if genPartitionExists(t, ee, "docs", 0, p) {
				oldGenAtGate++
			}
		}
	}
	defer func() { reshardGateHook = nil }()

	must(t, e.VectorReshard(ctx, "docs", newP))

	if !gateCalled {
		t.Fatal("awaitCutoverGate was never reached")
	}
	if gateColl != "docs" || gateGen != 1 {
		t.Fatalf("gate called with (%q,%d), want (\"docs\",1)", gateColl, gateGen)
	}
	if !flippedAtGate {
		t.Fatal("at the gate the catalog had NOT flipped to the new gen (gate ran before cutover?)")
	}
	if statusAtGate != 1 {
		t.Fatalf("at the gate ReshardState.Status = %d, want 1 (dual-write must stay on THROUGH the gate)", statusAtGate)
	}
	if oldGenAtGate != oldP {
		t.Fatalf("at the gate %d/%d old-gen partitions present, want all %d (old gen must not be dropped before the gate)", oldGenAtGate, oldP, oldP)
	}

	// After the reshard: state cleared, old gen gone, new gen holds the full set.
	if st, on := ee.catalog.ReshardState("docs"); on || st.Status != 0 {
		t.Fatalf("reshard state still set after completion: %+v on=%v", st, on)
	}
	for p := 0; p < oldP; p++ {
		if genPartitionExists(t, ee, "docs", 0, p) {
			t.Fatalf("old gen-0 partition %d not dropped after the gate", p)
		}
	}
	got := reshardScanGen(t, ee, "docs", newP, 1)
	if len(got) != N {
		t.Fatalf("new gen holds %d ids, want %d", len(got), N)
	}
}

// TestEmbeddedMVOnlineReshardGateSequence is the MV mirror of
// TestEmbeddedOnlineReshardGateSequenceDense: it asserts the SAME
// cutover→gate→clear→drop ordering for the multi-vector online reshard path.
func TestEmbeddedMVOnlineReshardGateSequence(t *testing.T) {
	oldGrace := reshardDrainGrace
	reshardDrainGrace = 20 * time.Millisecond
	defer func() { reshardDrainGrace = oldGrace }()

	e := newSingleEmbedded(t)
	waitLeaderEmbedded(t, e)
	ee := e.(*embedded)
	ctx := context.Background()
	const oldP, newP, N = 4, 8, 120
	must(t, e.VectorMVCreateCollection(ctx, "mv", MultiVectorConfig{Dim: 4, Partitions: oldP}))
	for id := uint64(1); id <= N; id++ {
		md := VectorMetadata{"tag": vector.NewInt(1)}
		must(t, e.VectorMVAdd(ctx, "mv", id, mvTokensFor(id), md))
	}

	mvOldGenExists := func(p int) bool {
		_, err := ee.Call(ctx, "vector_mv_get_config", ops.EncodeMVGetConfigArgs(string(ops.PartitionKeyGen("mv", 0, p))))
		return err == nil
	}

	var (
		gateCalled    bool
		gateColl      string
		gateGen       uint32
		flippedAtGate bool
		statusAtGate  uint8
		oldGenAtGate  int
	)
	reshardGateHook = func(collection string, newGen uint32) {
		gateCalled = true
		gateColl, gateGen = collection, newGen
		if p, gen, ok := ee.catalog.PartitionsGen("mv"); ok && p == newP && gen == newGen {
			flippedAtGate = true
		}
		if st, on := ee.catalog.ReshardState("mv"); on {
			statusAtGate = st.Status
		}
		for p := 0; p < oldP; p++ {
			if mvOldGenExists(p) {
				oldGenAtGate++
			}
		}
	}
	defer func() { reshardGateHook = nil }()

	must(t, e.VectorMVReshard(ctx, "mv", newP))

	if !gateCalled {
		t.Fatal("mv awaitCutoverGate was never reached")
	}
	if gateColl != "mv" || gateGen != 1 {
		t.Fatalf("mv gate called with (%q,%d), want (\"mv\",1)", gateColl, gateGen)
	}
	if !flippedAtGate {
		t.Fatal("mv: at the gate the catalog had NOT flipped to the new gen")
	}
	if statusAtGate != 1 {
		t.Fatalf("mv: at the gate ReshardState.Status = %d, want 1 (dual-write must stay on)", statusAtGate)
	}
	if oldGenAtGate != oldP {
		t.Fatalf("mv: at the gate %d/%d old-gen partitions present, want all %d", oldGenAtGate, oldP, oldP)
	}

	if st, on := ee.catalog.ReshardState("mv"); on || st.Status != 0 {
		t.Fatalf("mv reshard state still set after completion: %+v on=%v", st, on)
	}
	for p := 0; p < oldP; p++ {
		if mvOldGenExists(p) {
			t.Fatalf("mv old gen-0 partition %d not dropped after the gate", p)
		}
	}
	if got := reshardScanGenMV(t, ee, "mv", newP, 1); len(got) != N {
		t.Fatalf("mv new gen holds %d docs, want %d", len(got), N)
	}
}

// TestClusterReshardGateMultiNode drives a full dense online reshard on a REAL
// 3-node cluster (so the Phase-4.5 gate actually polls remote peers via
// __catalog_gen__, not the single-node no-op). It proves no regression: the reshard
// completes, the gate genuinely confirmed every node routed to the new gen before
// the old gen was dropped, and reads from a NON-coordinator node return the full,
// correct set on the new generation.
func TestClusterReshardGateMultiNode(t *testing.T) {
	oldGrace := reshardDrainGrace
	reshardDrainGrace = 30 * time.Millisecond
	defer func() { reshardDrainGrace = oldGrace }()

	stores := newInmemEmbeddedCluster(t, 3, 8)
	ctx := context.Background()
	const oldP, newP, N = 4, 8, 300

	createCollectionTolerant(t, ctx, stores[0], "docs", VectorConfig{
		Dim: 4, Metric: vector.L2, M: 8, EfConstruction: 50, EfSearch: 64, Seed: 1, Partitions: oldP,
	})
	for id := uint64(1); id <= N; id++ {
		idc := id
		retryUntil(t, "upsert", func() error {
			return stores[0].VectorUpsert(ctx, "docs", idc, []float32{float32(idc), 0, 0, 0}, fmt.Sprintf("doc-%d", idc), VectorInsertOpts{})
		})
	}

	// At the gate (on the coordinator), assert all peers already report the new gen
	// (the gate is what GUARANTEES this) and the old gen is still present.
	ee0 := stores[0].(*embedded)
	var gateSawAllConfirmed bool
	reshardGateHook = func(collection string, newGen uint32) {
		// The gate body itself does the waiting; by the time it returns the reshard
		// proceeds. Here we just record that we reached the gate on the real cluster
		// and that the coordinator's own catalog already shows the flip.
		if p, gen, ok := ee0.catalog.PartitionsGen("docs"); ok && p == newP && gen == newGen {
			gateSawAllConfirmed = true
		}
	}
	defer func() { reshardGateHook = nil }()

	retryUntil(t, "reshard", func() error { return stores[0].VectorReshard(ctx, "docs", newP) })

	if !gateSawAllConfirmed {
		t.Fatal("gate was not reached with the catalog flipped on the 3-node cluster")
	}

	// New gen converged on the other nodes; reads from node 2 see the full set.
	ee2 := stores[2].(*embedded)
	// widened 5s->20s for CPU-contended CI; finite so a real hang still fails.
	waitEmbeddedCatalogGen(t, ee2, "docs", newP, 1, cpuScaled(20*time.Second))
	all, _, _, err := stores[2].VectorScroll(ctx, "docs", VectorFilter{}, 0, VectorScrollOpts{})
	must(t, err)
	if len(idSet(all)) != N {
		t.Fatalf("scroll from node 2 after reshard: %d distinct ids, want %d", len(idSet(all)), N)
	}
	// Old gen dropped on the coordinator (the gate passed, so Phase 6 ran).
	for p := 0; p < oldP; p++ {
		if genPartitionExists(t, ee0, "docs", 0, p) {
			t.Fatalf("old gen-0 partition %d still present after a completed cluster reshard", p)
		}
	}
}

// TestClusterReshardGateUnreachableNodeProceeds proves the gate is BEST-EFFORT and
// never hangs/fails the reshard: with one node downed mid-cluster, the Phase-4.5
// gate times out (that node never confirms the new gen), the coordinator LOGS the
// residual and PROCEEDS — the reshard completes (state cleared, old gen dropped on
// the coordinator) within a bounded time rather than hanging forever. This is the
// "a permanently-unreachable node must not block reshard" guarantee.
func TestClusterReshardGateUnreachableNodeProceeds(t *testing.T) {
	oldGrace := reshardDrainGrace
	oldTimeout := reshardCutoverGateTimeout
	reshardDrainGrace = 30 * time.Millisecond
	reshardCutoverGateTimeout = 600 * time.Millisecond // short so the test is fast
	defer func() {
		reshardDrainGrace = oldGrace
		reshardCutoverGateTimeout = oldTimeout
	}()

	// RF=3 so EVERY physical-partition shard is replicated on ALL three nodes: when
	// node 2 goes down, nodes 0+1 still hold a quorum (2/3) and a reachable replica
	// of every shard, so the reshard's copy / get-config never hit a lost partition.
	// The ONLY effect of downing node 2 is that its __catalog_gen__ poll (and its
	// meta-apply of the cutover) is unreachable — exactly the gate-timeout condition.
	stores, servers := newInmemEmbeddedClusterServers(t, 3, 8, 3)
	ctx := context.Background()
	const oldP, newP, N = 4, 8, 150

	createCollectionTolerant(t, ctx, stores[0], "docs", VectorConfig{
		Dim: 4, Metric: vector.L2, M: 8, EfConstruction: 50, EfSearch: 64, Seed: 1, Partitions: oldP,
	})
	for id := uint64(1); id <= N; id++ {
		idc := id
		retryUntil(t, "upsert", func() error {
			return stores[0].VectorUpsert(ctx, "docs", idc, []float32{float32(idc), 0, 0, 0}, fmt.Sprintf("doc-%d", idc), VectorInsertOpts{})
		})
	}

	// Down node 2 so its __catalog_gen__ poll fails (unreachable). The gate must
	// time out naming it, then proceed. Node 0 (coordinator) + node 1 keep meta
	// quorum so the cutover flip still commits, and RF=3 keeps every shard reachable.
	servers[2].Close()

	ee0 := stores[0].(*embedded)
	start := time.Now()
	// The reshard must COMPLETE (not error) despite the unreachable node.
	retryUntil(t, "reshard", func() error { return stores[0].VectorReshard(ctx, "docs", newP) })
	elapsed := time.Since(start)

	// It must not have hung indefinitely on the unreachable node — the gate is
	// best-effort and times out (the ASSERTED behavior, governed by the untouched
	// 600ms reshardCutoverGateTimeout above), then the reshard proceeds. This is a
	// FINITE anti-hang progress bound on the WHOLE reshard (copy + drain + cutover +
	// gate), not the asserted gate-timeout value: the base 40s bound is now cpuScaled
	// (3x on a 2-core runner = 120s) for CPU-contended CI, where the copy/dual-write/
	// cutover work around the gate legitimately takes longer under oversubscription.
	// Kept finite so a gate that genuinely hangs forever (reshard never returns) still
	// fails loud here rather than masking a real hang.
	if elapsed > cpuScaled(40*time.Second) {
		t.Fatalf("reshard with an unreachable node took %s — gate appears to hang rather than time out", elapsed)
	}

	// Reshard completed on the coordinator: catalog flipped, state cleared, old gen
	// dropped (Phase 6 ran past the gate timeout).
	if p, gen, ok := ee0.catalog.PartitionsGen("docs"); !ok || p != newP || gen != 1 {
		t.Fatalf("coordinator catalog (%d,%d,%v) want (%d,1,true)", p, gen, ok, newP)
	}
	if st, on := ee0.catalog.ReshardState("docs"); on || st.Status != 0 {
		t.Fatalf("reshard state still set after a timed-out gate: %+v on=%v", st, on)
	}
	for p := 0; p < oldP; p++ {
		if genPartitionExists(t, ee0, "docs", 0, p) {
			t.Fatalf("old gen-0 partition %d not dropped after the gate timed out + proceeded", p)
		}
	}
	got := reshardScanGen(t, ee0, "docs", newP, 1)
	if len(got) != N {
		t.Fatalf("new gen holds %d ids, want %d", len(got), N)
	}
}
