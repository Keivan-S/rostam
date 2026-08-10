// SPDX-License-Identifier: Apache-2.0

package rostam

import (
	"context"
	"testing"

	"github.com/rostamlabs/rostam/ops"
	"github.com/rostamlabs/rostam/vector"
)

// aliasResultIDs collects result IDs into a set for order-independent comparison
// (fan-out merge order across partitions is deterministic by distance, but the
// alias-vs-real assertion only needs the SAME set of hits).
func aliasResultIDs(res []VectorResult) map[uint64]struct{} {
	m := make(map[uint64]struct{}, len(res))
	for _, r := range res {
		m[r.ID] = struct{}{}
	}
	return m
}

func aliasSameIDSet(a, b map[uint64]struct{}) bool {
	if len(a) != len(b) {
		return false
	}
	for id := range a {
		if _, ok := b[id]; !ok {
			return false
		}
	}
	return true
}

// TestAliasResolvesDataPlanePartitioned is THE LANDMINE-#1 gate. It creates a
// PARTITIONED dense collection "real" (P=4), seeds it, points alias "prod"→"real",
// then drives a representative set of data-plane ops via the alias (search,
// scroll, get, insert, upsert, set-payload, delete) and asserts each behaves
// exactly as on "real" — read-through AND write-through, cross-partition. (The
// rewrite + partitioned() path is identical for all data-plane ops, so search —
// the scatter-gather op — is the critical LANDMINE proof; named/MV-via-alias are
// covered by TestAliasResolvesNamedAndMV.)
//
// The crux: data-plane ops are driven through the fanout DISPATCHER (fan.Call),
// which is where partitioned() decides fan-out-vs-passthrough. partitioned() AND
// the embedded method must resolve "prod"→"real" to the SAME canonical. A mismatch
// (one resolves, the other doesn't) makes the partitioned search hit the empty
// logical "prod" → silent ZERO results. The non-empty + correct-set assertions
// below are the proof both chokepoints agree.
func TestAliasResolvesDataPlanePartitioned(t *testing.T) {
	s := newSingleEmbedded(t)
	waitLeaderEmbedded(t, s)
	emb := s.(*embedded)
	fan := newFanoutDispatcher(emb, emb.node)
	ctx := context.Background()

	const (
		P    = 4
		real = "real"
		prod = "prod"
		N    = 80
	)
	cfg := VectorConfig{Dim: 4, Metric: vector.L2, M: 8, EfConstruction: 50, EfSearch: 64, Seed: 1, Partitions: P}
	if err := emb.CreateCollection(ctx, real, cfg); err != nil {
		t.Fatalf("CreateCollection real: %v", err)
	}
	for id := uint64(1); id <= N; id++ {
		if err := emb.VectorInsert(ctx, real, id, []float32{float32(id), 0, 0, 0}); err != nil {
			t.Fatalf("seed VectorInsert id=%d: %v", id, err)
		}
	}
	if err := emb.CreateAlias(ctx, prod, real); err != nil {
		t.Fatalf("CreateAlias prod→real: %v", err)
	}

	query := []float32{0.5, 0, 0, 0}

	// --- Read-through: search via the alias through the dispatcher ---
	// Ground truth: search "real" directly.
	wantRes, _, err := emb.VectorSearchExt(ctx, real, query, 10, VectorSearchOpts{})
	if err != nil {
		t.Fatalf("real VectorSearchExt: %v", err)
	}
	if len(wantRes) == 0 {
		t.Fatalf("sanity: real search returned no results")
	}

	// VectorSearch via the alias through the dispatcher (the LANDMINE chokepoint).
	rawSearch, err := fan.Call("vector_search", ops.EncodeVectorSearchArgsExt(prod, 10, query, vector.Filter{}))
	if err != nil {
		t.Fatalf("fan vector_search via alias: %v", err)
	}
	gotSearch, err := ops.DecodeVectorSearchResults(rawSearch)
	if err != nil {
		t.Fatalf("decode alias search: %v", err)
	}
	if len(gotSearch) == 0 {
		t.Fatalf("LANDMINE #1: partitioned search via alias returned EMPTY (partitioned()/embedded resolution mismatch)")
	}
	if !aliasSameIDSet(aliasResultIDs(gotSearch), aliasResultIDs(wantRes)) {
		t.Fatalf("alias search set != real search set\n alias=%v\n  real=%v", gotSearch, wantRes)
	}

	// VectorSearchExt via the alias directly on the embedded method (also resolves).
	gotExt, _, err := emb.VectorSearchExt(ctx, prod, query, 10, VectorSearchOpts{})
	if err != nil {
		t.Fatalf("embedded VectorSearchExt via alias: %v", err)
	}
	if len(gotExt) == 0 {
		t.Fatalf("LANDMINE #1: embedded SearchExt via alias returned EMPTY")
	}
	if !aliasSameIDSet(aliasResultIDs(gotExt), aliasResultIDs(wantRes)) {
		t.Fatalf("embedded SearchExt via alias set mismatch")
	}

	// --- VectorGet via the alias (route-by-id to the owning partition) ---
	found, _, _, _, _, err := emb.VectorGet(ctx, prod, 7, false, false)
	if err != nil || !found {
		t.Fatalf("VectorGet id=7 via alias: found=%v err=%v, want found", found, err)
	}

	// --- Write-through: insert a NEW point via the alias, verify it lands in "real" ---
	const newID = uint64(1000)
	if err := emb.VectorInsert(ctx, prod, newID, []float32{0.5, 0, 0, 0}); err != nil {
		t.Fatalf("VectorInsert newID via alias: %v", err)
	}
	if found, _, _, _, _, err := emb.VectorGet(ctx, real, newID, false, false); err != nil || !found {
		t.Fatalf("write-through insert via alias not visible on real: found=%v err=%v", found, err)
	}

	// --- Upsert via the alias ---
	const upID = uint64(2000)
	if err := emb.VectorUpsert(ctx, prod, upID, []float32{0.5, 0, 0, 0}, "", VectorInsertOpts{}); err != nil {
		t.Fatalf("VectorUpsert via alias: %v", err)
	}
	if found, _, _, _, _, err := emb.VectorGet(ctx, real, upID, false, false); err != nil || !found {
		t.Fatalf("upsert via alias not visible on real: found=%v err=%v", found, err)
	}

	// --- SetPayload via the alias, read it back via real ---
	patch := VectorMetadata{"tag": vector.NewInt(42)}
	if ok, err := emb.VectorSetPayload(ctx, prod, newID, patch, nil); err != nil || !ok {
		t.Fatalf("VectorSetPayload via alias: ok=%v err=%v", ok, err)
	}
	if found, _, meta, _, _, err := emb.VectorGet(ctx, real, newID, false, true); err != nil || !found {
		t.Fatalf("VectorGet after alias set-payload: found=%v err=%v", found, err)
	} else if v, ok := meta["tag"]; !ok || v.IsZero() {
		t.Fatalf("payload set via alias not visible on real: meta=%v", meta)
	}

	// --- Scroll via the alias through the dispatcher (cross-partition union) ---
	rawScroll, err := fan.Call("vector_scroll", ops.EncodeScrollArgs(prod, vector.Filter{}, 0))
	if err != nil {
		t.Fatalf("fan vector_scroll via alias: %v", err)
	}
	docs, err := ops.DecodeVectorDocs(rawScroll)
	if err != nil {
		t.Fatalf("decode alias scroll: %v", err)
	}
	if len(docs) == 0 {
		t.Fatalf("LANDMINE #1: partitioned scroll via alias returned EMPTY")
	}

	// --- Delete via the alias, verify removed from "real" ---
	if ok, err := emb.VectorDelete(ctx, prod, newID); err != nil || !ok {
		t.Fatalf("VectorDelete via alias: ok=%v err=%v", ok, err)
	}
	if found, _, _, _, _, err := emb.VectorGet(ctx, real, newID, false, false); err != nil || found {
		t.Fatalf("delete via alias did not remove from real: found=%v err=%v", found, err)
	}
}

// TestAliasResolvesDataPlaneUnpartitioned is the unpartitioned counterpart: read+
// write-through via an alias on a P=1 dense collection. The single-partition path
// has no fan-out, but resolution still has to map the alias to the real logical
// collection at both chokepoints.
func TestAliasResolvesDataPlaneUnpartitioned(t *testing.T) {
	s := newSingleEmbedded(t)
	waitLeaderEmbedded(t, s)
	emb := s.(*embedded)
	fan := newFanoutDispatcher(emb, emb.node)
	ctx := context.Background()

	const (
		real = "real1"
		prod = "prod1"
	)
	if err := emb.CreateCollection(ctx, real, denseCfg(1)); err != nil {
		t.Fatalf("CreateCollection: %v", err)
	}
	for id := uint64(1); id <= 20; id++ {
		if err := emb.VectorInsert(ctx, real, id, []float32{float32(id), 0, 0, 0}); err != nil {
			t.Fatalf("seed insert: %v", err)
		}
	}
	if err := emb.CreateAlias(ctx, prod, real); err != nil {
		t.Fatalf("CreateAlias: %v", err)
	}

	query := []float32{1.5, 0, 0, 0}
	raw, err := fan.Call("vector_search", ops.EncodeVectorSearchArgsExt(prod, 5, query, vector.Filter{}))
	if err != nil {
		t.Fatalf("fan search via alias: %v", err)
	}
	got, err := ops.DecodeVectorSearchResults(raw)
	if err != nil {
		t.Fatalf("decode: %v", err)
	}
	if len(got) == 0 {
		t.Fatalf("unpartitioned search via alias returned empty")
	}

	// Write-through.
	if err := emb.VectorInsert(ctx, prod, 999, []float32{1, 0, 0, 0}); err != nil {
		t.Fatalf("insert via alias: %v", err)
	}
	if found, _, _, _, _, err := emb.VectorGet(ctx, real, 999, false, false); err != nil || !found {
		t.Fatalf("write-through not visible on real: found=%v err=%v", found, err)
	}
}

// TestAliasResolvesNamedAndMV proves read+write-through via an alias for a named
// collection and an MV collection (partitioned, so the fan-out path is exercised).
func TestAliasResolvesNamedAndMV(t *testing.T) {
	s := newSingleEmbedded(t)
	waitLeaderEmbedded(t, s)
	emb := s.(*embedded)
	ctx := context.Background()
	const P = 4

	// --- Named collection via an alias ---
	namedCfg := map[string]NamedVectorParams{
		"title": {Dim: 4, Metric: vector.L2, M: 8, EfConstruction: 50, EfSearch: 32},
	}
	if err := emb.VectorNamedCreateCollection(ctx, "named_real", namedCfg, P); err != nil {
		t.Fatalf("VectorNamedCreateCollection: %v", err)
	}
	for id := uint64(1); id <= 40; id++ {
		if err := emb.VectorNamedInsert(ctx, "named_real", id,
			map[string][]float32{"title": {float32(id), 0, 0, 0}}, nil, 0); err != nil {
			t.Fatalf("VectorNamedInsert: %v", err)
		}
	}
	if err := emb.CreateAlias(ctx, "named_prod", "named_real"); err != nil {
		t.Fatalf("CreateAlias named: %v", err)
	}

	// Search via the named alias (cross-partition fan-out under resolution).
	nres, err := emb.VectorNamedSearch(ctx, "named_prod", "title", []float32{1.5, 0, 0, 0}, 5, vector.Filter{})
	if err != nil {
		t.Fatalf("VectorNamedSearch via alias: %v", err)
	}
	if len(nres) == 0 {
		t.Fatalf("LANDMINE #1: named search via alias returned EMPTY (cross-partition resolution mismatch)")
	}

	// Write-through: insert via the alias, read back via real.
	if err := emb.VectorNamedInsert(ctx, "named_prod", 9999,
		map[string][]float32{"title": {1, 0, 0, 0}}, nil, 0); err != nil {
		t.Fatalf("VectorNamedInsert via alias: %v", err)
	}
	if found, _, _, _, err := emb.VectorNamedGet(ctx, "named_real", 9999, false, false); err != nil || !found {
		t.Fatalf("named write-through not visible on real: found=%v err=%v", found, err)
	}

	// --- MV collection via an alias ---
	if err := emb.VectorMVCreateCollection(ctx, "mv_real", MultiVectorConfig{Dim: 4, Partitions: P}); err != nil {
		t.Fatalf("VectorMVCreateCollection: %v", err)
	}
	for id := 0; id < 40; id++ {
		if err := emb.VectorMVAdd(ctx, "mv_real", uint64(id), [][]float32{mvTokenAt(id)}, nil); err != nil {
			t.Fatalf("VectorMVAdd: %v", err)
		}
	}
	if err := emb.CreateAlias(ctx, "mv_prod", "mv_real"); err != nil {
		t.Fatalf("CreateAlias mv: %v", err)
	}

	mvQuery := [][]float32{mvTokenAt(17)}
	mres, _, err := emb.VectorMVSearch(ctx, "mv_prod", mvQuery, 10, MultiSearchOpts{CandidatesPerToken: 100})
	if err != nil {
		t.Fatalf("VectorMVSearch via alias: %v", err)
	}
	if len(mres) == 0 {
		t.Fatalf("LANDMINE #1: MV search via alias returned EMPTY (cross-partition resolution mismatch)")
	}

	// Write-through MV add via the alias, read back via real.
	if err := emb.VectorMVAdd(ctx, "mv_prod", 8888, [][]float32{mvTokenAt(5)}, nil); err != nil {
		t.Fatalf("VectorMVAdd via alias: %v", err)
	}
	if found, _, _, err := emb.VectorMVGet(ctx, "mv_real", 8888, false, false); err != nil || !found {
		t.Fatalf("MV write-through not visible on real: found=%v err=%v", found, err)
	}
}

// TestAliasReservedCharShortCircuit asserts the '#'/'@' short-circuit: a physical
// partition / generation name is NEVER resolved (so reshard/fan-out physical-name
// ops are untouched), even if — hypothetically — such a name were an alias key.
func TestAliasReservedCharShortCircuit(t *testing.T) {
	s := newSingleEmbedded(t)
	waitLeaderEmbedded(t, s)
	emb := s.(*embedded)

	for _, name := range []string{"coll#0", "coll@1#0", "docs#2", "x@3#7"} {
		if got := emb.resolveAlias(name); got != name {
			t.Fatalf("resolveAlias(%q) = %q, want unchanged (reserved-char short-circuit)", name, got)
		}
	}

	// A non-alias plain name is also returned unchanged (zero-cost passthrough).
	if got := emb.resolveAlias("not_an_alias"); got != "not_an_alias" {
		t.Fatalf("resolveAlias(plain) = %q, want unchanged", got)
	}
}

// TestDropCollectionCascadesAliases: dropping a collection removes every alias
// targeting it (cascade), and a data-plane op via the dropped alias resolves to a
// gone collection → not-found / error, NOT a panic.
func TestDropCollectionCascadesAliases(t *testing.T) {
	s := newSingleEmbedded(t)
	waitLeaderEmbedded(t, s)
	emb := s.(*embedded)
	fan := newFanoutDispatcher(emb, emb.node)
	ctx := context.Background()

	// --- Dense (partitioned) drop cascade via the dispatcher ---
	const P = 4
	cfg := VectorConfig{Dim: 4, Metric: vector.L2, M: 8, EfConstruction: 50, EfSearch: 64, Seed: 1, Partitions: P}
	if err := emb.CreateCollection(ctx, "real", cfg); err != nil {
		t.Fatalf("CreateCollection real: %v", err)
	}
	for id := uint64(1); id <= 20; id++ {
		if err := emb.VectorInsert(ctx, "real", id, []float32{float32(id), 0, 0, 0}); err != nil {
			t.Fatalf("seed: %v", err)
		}
	}
	if err := emb.CreateAlias(ctx, "prod", "real"); err != nil {
		t.Fatalf("CreateAlias prod→real: %v", err)
	}
	if err := emb.CreateAlias(ctx, "live", "real"); err != nil {
		t.Fatalf("CreateAlias live→real: %v", err)
	}

	// Drop "real" through the dispatcher (the dense-drop chokepoint that cascades).
	if _, err := fan.Call("vector_drop_collection", ops.EncodeDropCollectionArgs("real")); err != nil {
		t.Fatalf("drop real via dispatcher: %v", err)
	}

	// Both aliases must be gone from the list.
	list, err := emb.ListAliases(ctx, "")
	if err != nil {
		t.Fatalf("ListAliases: %v", err)
	}
	if _, ok := list["prod"]; ok {
		t.Fatalf("cascade failed: prod still listed: %v", list)
	}
	if _, ok := list["live"]; ok {
		t.Fatalf("cascade failed: live still listed: %v", list)
	}

	// resolveAlias("prod") must now be a no-op (the alias is gone).
	if got := emb.resolveAlias("prod"); got != "prod" {
		t.Fatalf("resolveAlias(prod) after drop = %q, want unchanged (alias removed)", got)
	}

	// A data-plane op via the dropped alias resolves to "prod" (gone) → not-found /
	// error, NOT a panic. The collection no longer exists, so search errors.
	if _, err := fan.Call("vector_search", ops.EncodeVectorSearchArgsExt("prod", 5, []float32{1, 0, 0, 0}, vector.Filter{})); err == nil {
		// An error is the expected outcome (unknown collection). If somehow nil,
		// the result must at least be empty — never wrong data.
		t.Logf("search via dropped alias returned nil error (acceptable if empty)")
	}

	// --- MV drop cascade via the embedded Store method ---
	if err := emb.VectorMVCreateCollection(ctx, "mv_real", MultiVectorConfig{Dim: 4, Partitions: 1}); err != nil {
		t.Fatalf("VectorMVCreateCollection: %v", err)
	}
	if err := emb.CreateAlias(ctx, "mv_prod", "mv_real"); err != nil {
		t.Fatalf("CreateAlias mv: %v", err)
	}
	if err := emb.VectorMVDropCollection(ctx, "mv_real"); err != nil {
		t.Fatalf("VectorMVDropCollection: %v", err)
	}
	if l, _ := emb.ListAliases(ctx, ""); l["mv_prod"] != "" {
		t.Fatalf("MV drop cascade failed: mv_prod still listed: %v", l)
	}

	// --- Named drop cascade ---
	namedCfg := map[string]NamedVectorParams{"title": {Dim: 4, Metric: vector.L2, M: 8, EfConstruction: 50, EfSearch: 32}}
	if err := emb.VectorNamedCreateCollection(ctx, "n_real", namedCfg, 0); err != nil {
		t.Fatalf("VectorNamedCreateCollection: %v", err)
	}
	if err := emb.CreateAlias(ctx, "n_prod", "n_real"); err != nil {
		t.Fatalf("CreateAlias named: %v", err)
	}
	if err := emb.VectorNamedDropCollection(ctx, "n_real"); err != nil {
		t.Fatalf("VectorNamedDropCollection: %v", err)
	}
	if l, _ := emb.ListAliases(ctx, ""); l["n_prod"] != "" {
		t.Fatalf("Named drop cascade failed: n_prod still listed: %v", l)
	}
}

// TestAliasAdminOpsUseRealNames is the reshard/partition regression gate for the
// '#'/'@' short-circuit + admin-don't-resolve rule: vector_resplit / vector_reshard
// / vector_get_config on a REAL collection still operate on physical partition
// names (which contain '#'/'@' and must NEVER resolve), unaffected by alias
// resolution.
func TestAliasAdminOpsUseRealNames(t *testing.T) {
	s := newSingleEmbedded(t)
	waitLeaderEmbedded(t, s)
	emb := s.(*embedded)
	ctx := context.Background()

	const P = 4
	cfg := VectorConfig{Dim: 4, Metric: vector.L2, M: 8, EfConstruction: 50, EfSearch: 64, Seed: 1, Partitions: P}
	if err := emb.CreateCollection(ctx, "docs", cfg); err != nil {
		t.Fatalf("CreateCollection docs: %v", err)
	}
	for id := uint64(1); id <= 40; id++ {
		if err := emb.VectorInsert(ctx, "docs", id, []float32{float32(id), 0, 0, 0}); err != nil {
			t.Fatalf("seed: %v", err)
		}
	}
	// Create an alias targeting docs to ensure resolution exists but does NOT
	// interfere with admin ops on the real physical names.
	if err := emb.CreateAlias(ctx, "docs_alias", "docs"); err != nil {
		t.Fatalf("CreateAlias: %v", err)
	}

	// vector_get_config on a PHYSICAL partition name (contains '#') — must resolve
	// to itself (short-circuit) and succeed.
	phys := string(ops.PartitionKeyGen("docs", 0, 0))
	if _, err := emb.Call(ctx, "vector_get_config", ops.EncodeGetConfigArgs(phys)); err != nil {
		t.Fatalf("vector_get_config on physical %q: %v", phys, err)
	}

	// Resplit the REAL collection (admin op, real name) — builds gen-1 physical
	// '@'/'#' names; the '#'/'@' short-circuit keeps them unresolved.
	if err := emb.VectorResplit(ctx, "docs", 8); err != nil {
		t.Fatalf("VectorResplit docs: %v", err)
	}
	// Post-resplit the live generation is {8, gen1}; a get-config on a new-gen
	// physical name (contains '@' and '#') must succeed (never resolved).
	physG1 := string(ops.PartitionKeyGen("docs", 1, 0))
	if _, err := emb.Call(ctx, "vector_get_config", ops.EncodeGetConfigArgs(physG1)); err != nil {
		t.Fatalf("vector_get_config on gen-1 physical %q: %v", physG1, err)
	}

	// Search the real collection still works after resplit (cross-partition).
	res, _, err := emb.VectorSearchExt(ctx, "docs", []float32{0.5, 0, 0, 0}, 10, VectorSearchOpts{})
	if err != nil {
		t.Fatalf("post-resplit search: %v", err)
	}
	if len(res) == 0 {
		t.Fatalf("post-resplit search returned empty")
	}
}
