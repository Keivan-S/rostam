// SPDX-License-Identifier: Apache-2.0

package rostam

import (
	"context"
	"testing"

	"github.com/rostamlabs/rostam/ops"
	"github.com/rostamlabs/rostam/vector"
)

// openEmbeddedAt opens a single-node embedded Store on a CALLER-CONTROLLED data
// dir (so the same dir can be reopened to prove durability across a restart). The
// caller is responsible for Close (these tests close + reopen explicitly).
func openEmbeddedAt(t *testing.T, dir string) Store {
	t.Helper()
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
		t.Fatalf("NewEmbedded(%s): %v", dir, err)
	}
	return s
}

// rawPartitionSearch issues a single-Call (no fan-out) search against a physical
// name, used to prove a vector physically lives in a partition (i.e. that the
// catalog partition count was honored at route time).
func rawPartitionSearch(t *testing.T, s Store, name string, k int) []VectorResult {
	t.Helper()
	body, err := s.Call(context.Background(), "vector_search", ops.EncodeVectorSearchArgs(name, k, []float32{1, 0, 0, 0}))
	if err != nil {
		t.Fatalf("raw search %q: %v", name, err)
	}
	out, err := ops.DecodeVectorSearchResults(body)
	if err != nil {
		t.Fatalf("decode raw search %q: %v", name, err)
	}
	return out
}

// TestSingleNodeCatalogDurability proves the single-node durable CATALOG: a
// partitioned collection's partition COUNT and the ALIAS map both survive a
// close/reopen on the SAME DataDir (backed by the shard Raft KV under __vcat__/).
//
// The proof of partition-count durability is that a data op via the alias, run on
// the REOPENED node, still routes to the right partitioned physical collections —
// i.e. the restored catalog (not a default P=1) drives routing. Vector-index
// persistence is a separate concern from this catalog change; the routing proof
// uses fresh post-reopen inserts so it depends only on the restored partition
// count, not on old index data.
func TestSingleNodeCatalogDurability(t *testing.T) {
	dir := t.TempDir()
	ctx := context.Background()

	// --- session 1: create partitioned collection + alias ---
	s := openEmbeddedAt(t, dir)
	waitLeaderEmbedded(t, s)

	if err := s.CreateCollection(ctx, "docs", VectorConfig{Dim: 4, M: 8, EfConstruction: 50, EfSearch: 32, Seed: 1, Metric: vector.L2, Partitions: 4}); err != nil {
		t.Fatalf("CreateCollection docs P=4: %v", err)
	}
	if err := s.CreateAlias(ctx, "prod", "docs"); err != nil {
		t.Fatalf("CreateAlias prod->docs: %v", err)
	}
	if err := s.Close(); err != nil {
		t.Fatalf("Close session 1: %v", err)
	}

	// --- session 2: reopen SAME dir, nothing recreated ---
	s2 := openEmbeddedAt(t, dir)
	t.Cleanup(func() { _ = s2.Close() })
	waitLeaderEmbedded(t, s2)
	emb := s2.(*embedded)

	// Aliases survived (user-facing Store API strips the default/ namespace).
	list, err := s2.ListAliases(ctx, "")
	if err != nil {
		t.Fatalf("ListAliases after reopen: %v", err)
	}
	if list["prod"] != "docs" {
		t.Fatalf("after reopen list[prod]=%q, want docs (list=%v)", list["prod"], list)
	}
	// Catalog-level ResolveAlias returns the canonical target.
	if canon, ok := emb.catalog.ResolveAlias("prod"); !ok || canon != ops.CanonicalName("docs") {
		t.Fatalf("after reopen ResolveAlias(prod)=(%q,%v), want (%q,true)", canon, ok, ops.CanonicalName("docs"))
	}

	// Partition count survived: PartitionsGen reports P=4.
	if p, _, ok := emb.catalog.PartitionsGen("docs"); !ok || p != 4 {
		t.Fatalf("after reopen PartitionsGen(docs)=(%d,_,%v), want (4,_,true)", p, ok)
	}

	// A data op via the ALIAS, on the reopened node, routes to the right
	// partitioned collection — driven entirely by the RESTORED partition count.
	for id := uint64(1); id <= 200; id++ {
		if err := s2.VectorInsert(ctx, "prod", id, []float32{float32(id), 0, 0, 0}); err != nil {
			t.Fatalf("post-reopen insert via alias %d: %v", id, err)
		}
	}
	// The logical "docs" holds nothing; the physical partitions hold all 200.
	if got := rawPartitionSearch(t, s2, "docs", 250); len(got) != 0 {
		t.Fatalf("logical docs held %d vectors, want 0 (data routed to partitions via restored catalog)", len(got))
	}
	total := 0
	for p := 0; p < 4; p++ {
		phys := string(ops.PartitionKey("docs", p))
		n := len(rawPartitionSearch(t, s2, phys, 250))
		if n == 0 || n >= 200 {
			t.Fatalf("partition %s held %d vectors, want a strict fraction of 200 (restored P=4 routing)", phys, n)
		}
		total += n
	}
	if total != 200 {
		t.Fatalf("partitions held %d total, want 200", total)
	}

	// The alias resolves on the live search path too.
	res, err := s2.VectorSearch(ctx, "prod", []float32{1, 0, 0, 0}, 5)
	if err != nil {
		t.Fatalf("VectorSearch via alias after reopen: %v", err)
	}
	if len(res) == 0 || res[0].ID != 1 {
		t.Fatalf("search via alias after reopen top=%+v, want ID 1", res)
	}
}

// TestSingleNodeAliasBatchAtomicDurability proves an alias_batch atomic swap
// survives a restart ALL-OR-NOTHING: after a {delete prod, create prod->docs2}
// swap the reopened node sees prod->docs2 and never a half-applied state.
func TestSingleNodeAliasBatchAtomicDurability(t *testing.T) {
	dir := t.TempDir()
	ctx := context.Background()

	s := openEmbeddedAt(t, dir)
	waitLeaderEmbedded(t, s)
	for _, c := range []string{"docs", "docs2"} {
		if err := s.CreateCollection(ctx, c, denseCfg(1)); err != nil {
			t.Fatalf("CreateCollection %s: %v", c, err)
		}
	}
	if err := s.CreateAlias(ctx, "prod", "docs"); err != nil {
		t.Fatalf("CreateAlias prod->docs: %v", err)
	}
	// Atomic swap in one batch.
	if err := s.AliasBatch(ctx, []AliasAction{
		{Alias: "prod", Delete: true},
		{Alias: "prod", Canonical: "docs2"},
	}); err != nil {
		t.Fatalf("AliasBatch swap: %v", err)
	}
	if err := s.Close(); err != nil {
		t.Fatalf("Close: %v", err)
	}

	s2 := openEmbeddedAt(t, dir)
	t.Cleanup(func() { _ = s2.Close() })
	waitLeaderEmbedded(t, s2)

	list, err := s2.ListAliases(ctx, "")
	if err != nil {
		t.Fatalf("ListAliases after reopen: %v", err)
	}
	if len(list) != 1 || list["prod"] != "docs2" {
		t.Fatalf("after reopen alias map=%v, want exactly {prod:docs2} (atomic swap must be all-or-nothing)", list)
	}
}

// TestSingleNodeNoAliasNoRegression proves the no-alias / default-partition path
// has no spurious persisted state across a restart: a plain unpartitioned
// collection has no catalog entry on reopen, and the alias map is empty.
func TestSingleNodeNoAliasNoRegression(t *testing.T) {
	dir := t.TempDir()
	ctx := context.Background()

	s := openEmbeddedAt(t, dir)
	waitLeaderEmbedded(t, s)
	if err := s.CreateCollection(ctx, "docs", denseCfg(1)); err != nil {
		t.Fatalf("CreateCollection docs P=1: %v", err)
	}
	if err := s.Close(); err != nil {
		t.Fatalf("Close: %v", err)
	}

	s2 := openEmbeddedAt(t, dir)
	t.Cleanup(func() { _ = s2.Close() })
	waitLeaderEmbedded(t, s2)
	emb := s2.(*embedded)

	// No spurious alias state.
	list, err := s2.ListAliases(ctx, "")
	if err != nil {
		t.Fatalf("ListAliases: %v", err)
	}
	if len(list) != 0 {
		t.Fatalf("after reopen no-alias map=%v, want empty", list)
	}
	if _, ok := emb.catalog.ResolveAlias("docs"); ok {
		t.Fatalf("ResolveAlias(docs) reported alias, want none")
	}
	// An unpartitioned collection has no catalog entry (single-partition default):
	// no spurious catalog state from the no-partition path.
	if p, _, ok := emb.catalog.PartitionsGen("docs"); ok || p != 1 {
		t.Fatalf("PartitionsGen(docs)=(%d,_,%v), want (1,_,false) default", p, ok)
	}
	// The unpartitioned collection still works on reopen (fresh insert + search).
	if err := s2.VectorInsert(ctx, "docs", 1, []float32{1, 0, 0, 0}); err != nil {
		t.Fatalf("post-reopen insert into unpartitioned docs: %v", err)
	}
	res, err := s2.VectorSearch(ctx, "docs", []float32{1, 0, 0, 0}, 1)
	if err != nil {
		t.Fatalf("post-reopen search: %v", err)
	}
	if len(res) != 1 || res[0].ID != 1 {
		t.Fatalf("post-reopen search=%+v, want ID 1", res)
	}
}

// TestSingleNodeAliasWriteThroughConsistency proves the in-memory read cache and
// the durable KV never diverge: a fresh catalog reader over the SAME node, loaded
// from the durable __vcat__/aliases key, agrees with the live write-through cache
// after a sequence of create/delete/upsert operations.
func TestSingleNodeAliasWriteThroughConsistency(t *testing.T) {
	dir := t.TempDir()
	ctx := context.Background()
	s := openEmbeddedAt(t, dir)
	t.Cleanup(func() { _ = s.Close() })
	waitLeaderEmbedded(t, s)
	emb := s.(*embedded)

	for _, c := range []string{"a", "b", "c"} {
		if err := s.CreateCollection(ctx, c, denseCfg(1)); err != nil {
			t.Fatalf("CreateCollection %s: %v", c, err)
		}
	}
	if err := s.CreateAlias(ctx, "x", "a"); err != nil {
		t.Fatalf("CreateAlias x->a: %v", err)
	}
	if err := s.CreateAlias(ctx, "y", "b"); err != nil {
		t.Fatalf("CreateAlias y->b: %v", err)
	}
	if err := s.DeleteAlias(ctx, "x"); err != nil {
		t.Fatalf("DeleteAlias x: %v", err)
	}
	if err := s.CreateAlias(ctx, "x", "c"); err != nil {
		t.Fatalf("CreateAlias x->c: %v", err)
	}

	// Fresh catalog reader over the same durable node — must equal the live cache.
	fresh := &singleNodeCatalog{cat: ops.NewCatalog(newKVCatalogStore(emb.node)), node: emb.node}
	freshList := fresh.ListAliases()
	live := emb.catalog.ListAliases()
	if len(freshList) != len(live) {
		t.Fatalf("durable reload size %d != live cache size %d (cache/KV diverged)", len(freshList), len(live))
	}
	for k, v := range live {
		if freshList[k] != v {
			t.Fatalf("durable reload[%q]=%q != live[%q]=%q (cache/KV diverged)", k, freshList[k], k, v)
		}
	}
	wantX, wantC := ops.CanonicalName("x"), ops.CanonicalName("c")
	wantY, wantB := ops.CanonicalName("y"), ops.CanonicalName("b")
	if live[wantX] != wantC || live[wantY] != wantB {
		t.Fatalf("live cache=%v, want {%s:%s,%s:%s}", live, wantX, wantC, wantY, wantB)
	}
}
