// SPDX-License-Identifier: Apache-2.0

package rostam

import (
	"context"
	"fmt"
	"sort"
	"strings"
	"time"

	"github.com/rostamlabs/rostam/cluster"
	"github.com/rostamlabs/rostam/ops"
	"github.com/rostamlabs/rostam/vector"
)

// Cross-partition fan-out for the named-vector (Qdrant-style multi-vector-space)
// collection family. Named collections can be partitioned at the collection
// level (one P, shared by all named spaces). The named-vector NAME rides in the
// op args (not the routing key): a point's id deterministically maps to ONE
// physical partition (PartitionOf(id,P)) regardless of which named spaces it
// populates, exactly like a dense point's id. So routing is by collection + id;
// the named map of vecs + payload + filter travel inside the encoded args to
// the destination partition.
//
// These coordinators mirror the dense/MV fan-out precisely:
//   - create re-expands physical partitions (mirror CreateCollection / VectorMVCreateCollection)
//   - insert routes by PartitionOf(id,P) (mirror VectorInsert)
//   - search / search_docs SCATTER to all partitions, union + global top-k (mirror denseFanOut / docsFanOut)
//   - delete fans to all partitions (mirror dense delete-by-id below)
//   - scroll fans to all partitions, unions (mirror scrollFanOut)
//   - drop drops every physical partition (mirror fanoutDrop)
//
// Each named space is its own sub-index inside one NamedCollection, but a search
// queries exactly ONE named space; the per-partition op carries the vector name,
// so each partition searches the same named space and the union/top-k by
// ascending distance is exact (IDs are disjoint across partitions — a point
// lives in exactly one).

// VectorNamedCreateCollection registers a named-vector collection. A
// single-partition create (partitions<=1) goes straight to the in-process
// handler, byte-for-byte the pre-fan-out path. A partitioned create
// (partitions>1) re-expands into P physical single-partition named collections
// (PartitionKeyGen(name,0,p)) — each carrying the SAME NamedVectors config — and
// records P in the catalog, mirroring the dense/MV create re-expansion exactly
// (incl. the logical-with-Partitions-stripped marker so a forwarded logical
// create is never double-expanded on a remote node).
func (e *embedded) VectorNamedCreateCollection(ctx context.Context, name string, cfg map[string]NamedVectorParams, partitions int) error {
	if strings.ContainsAny(name, "#@") {
		return fmt.Errorf("vector: collection name %q must not contain reserved characters '#' or '@'", name)
	}
	// Alias-shadow guard: a new named collection must not take the name of an
	// existing alias (see CreateCollection).
	if _, ok := e.catalog.ResolveAlias(name); ok {
		return fmt.Errorf("vector: collection name %q is already an alias: %w", name, ErrAliasShadowsCollection)
	}
	// Cross-type / re-partition guard (fail-loud, all Partitions values): the
	// catalog is shared with the dense + MV families — a name already partitioned
	// as ANY type must never be re-partitioned, or routing would corrupt.
	if _, _, ok := e.catalog.PartitionsGen(name); ok {
		return fmt.Errorf("vector: collection %q is already partitioned", name)
	}
	// Partitioned named: reject if a dense or MV collection of the same name
	// already exists (a nil error from the probe means it found one).
	if partitions > 1 {
		if _, err := e.Call(ctx, "vector_get_config", ops.EncodeGetConfigArgs(name)); err == nil {
			return fmt.Errorf("vector: a dense collection named %q already exists", name)
		}
		if _, err := e.Call(ctx, "vector_mv_get_config", ops.EncodeMVGetConfigArgs(name)); err == nil {
			return fmt.Errorf("vector: a multi-vector collection named %q already exists", name)
		}
	}

	// Single-partition path: byte-for-byte unchanged (no catalog write, no extra
	// collections). partitions<=1 is encoded as 0 so the handler creates one
	// plain named collection.
	if partitions <= 1 {
		_, err := e.Call(ctx, "vector_named_create_collection", ops.EncodeNamedCreateArgs(name, cfg, 0))
		return err
	}

	// Partitioned path: create the logical named collection (so the logical name
	// still resolves) with partitions stripped to 0 — the catalog
	// (SetPartitionsGen below) is the sole source of truth for P, and a stripped
	// logical create can't be re-expanded by a remote node's fan-out dispatcher
	// (which only expands partitions>1), avoiding a race with this coordinator's
	// physical-create loop. Then one physical single-partition named collection
	// per partition, all with the same NamedVectors config.
	P := partitions
	if _, err := e.Call(ctx, "vector_named_create_collection", ops.EncodeNamedCreateArgs(name, cfg, 0)); err != nil {
		return err
	}
	for p := 0; p < P; p++ {
		phys := string(ops.PartitionKeyGen(name, 0, p))
		if _, err := e.Call(ctx, "vector_named_create_collection", ops.EncodeNamedCreateArgs(phys, cfg, 0)); err != nil {
			return fmt.Errorf("create partition %d/%d for %q: %w", p, P, name, err)
		}
	}
	return e.catalog.SetPartitionsGen(name, P, 0)
}

// VectorNamedInsert routes the upsert to the ONE physical partition that owns id
// (PartitionOf(id,P)) when the collection is partitioned, else to the logical
// name unchanged. The named map of vecs + shared payload + ttl all travel inside
// the encoded args to that partition, so nothing is dropped. Mirrors
// VectorInsert (a point's id maps to exactly one partition regardless of which
// named spaces it populates).
func (e *embedded) VectorNamedInsert(ctx context.Context, name string, id uint64, vectors map[string][]float32, payload VectorMetadata, ttl time.Duration, opts ...WriteOpts) error {
	return e.VectorNamedInsertSparse(ctx, name, id, vectors, nil, payload, ttl, opts...)
}

// VectorNamedInsertSparse is VectorNamedInsert carrying per-space SPARSE values
// (sparseVectors[space] is the *vector.SparseVector for a sparse space) routed to
// the ONE owning physical partition alongside the dense vectors. The sparse values
// ride the additive sparse sub-block in the encoded args
// (EncodeNamedInsertArgsSparseCASKeyTTL), byte-identical to the dense path when
// sparseVectors is empty. The coordinator-internal sparse insert entry point used
// by the fan-out dispatcher's named-insert path; the public VectorStore
// VectorNamedInsert delegates here with nil sparseVectors.
func (e *embedded) VectorNamedInsertSparse(_ context.Context, name string, id uint64, vectors map[string][]float32, sparseVectors map[string]*vector.SparseVector, payload VectorMetadata, ttl time.Duration, opts ...WriteOpts) error {
	name = e.resolveAlias(name)
	if phys, ok := e.partitionOf(name, id); ok {
		name = phys
	}
	wo := firstWriteOpts(opts)
	exp, hasExp := wo.expectedVersion()
	// CAS + per-key TTL + sparse coexist: the sparse sub-block rides AFTER the base
	// block, then the keyTTL block, then the CAS block. Empty sparse + keyTTL + no
	// CAS = byte-identical to EncodeNamedInsertArgs.
	_, err := e.Call(context.Background(), "vector_named_insert", ops.EncodeNamedInsertArgsSparseCASKeyTTL(name, id, vectors, sparseVectors, payload, ttl, exp, hasExp, wo.KeyTTLMs))
	if err != nil {
		return err
	}
	// Named insert is single-target (not dual-written during reshard — like
	// VectorMVAddIfAbsent it routes only to the live gen's partition), so the
	// barrier targets that one physical name. Default opts → no-op.
	if wo.wcActive() {
		return e.barrierPhys(name, wo)
	}
	return nil
}

// VectorNamedDelete fans the delete-by-id to ALL physical partitions when the
// collection is partitioned (the point lives in exactly one, but delete-by-id
// scatters like dense delete-by-filter — every partition no-ops except the
// owner). The returned existed-flag is the OR across partitions. Unpartitioned
// goes straight to the single handler.
func (e *embedded) VectorNamedDelete(_ context.Context, name string, id uint64, opts ...WriteOpts) (bool, error) {
	name = e.resolveAlias(name)
	wo := firstWriteOpts(opts)
	exp, hasExp := wo.expectedVersion()
	if P, gen, ok := e.catalog.PartitionsGen(name); ok && P > 1 {
		// A CAS delete must check the version at the ONE owning partition — a fan-out
		// scatter would falsely conflict on the non-owner partitions (they see
		// version 0). Route-by-id to the owner with the CAS guard (mirror dense
		// VectorDelete's route-by-id). A non-CAS delete keeps the historical fan-out.
		if hasExp {
			owning := string(ops.PartitionKeyGen(name, gen, ops.PartitionOf(id, P)))
			body, err := e.node.CallPhysical(owning, "vector_named_delete", ops.EncodeNamedDeleteArgsCAS(owning, id, exp, hasExp), false)
			if err != nil {
				return false, err
			}
			if wo.wcActive() {
				if err := e.barrierPhys(owning, wo); err != nil {
					return false, err
				}
			}
			return len(body) > 0 && body[0] == 1, nil
		}
		return e.namedDeleteFanOut(name, P, gen, id, wo)
	}
	body, err := e.Call(context.Background(), "vector_named_delete", ops.EncodeNamedDeleteArgsCAS(name, id, exp, hasExp))
	if err != nil {
		return false, err
	}
	// Unpartitioned: the single logical name IS the physical target. Default
	// opts → no-op.
	if wo.wcActive() {
		if err := e.barrierPhys(name, wo); err != nil {
			return false, err
		}
	}
	return len(body) > 0 && body[0] == 1, nil
}

// namedDeleteFanOut scatters vector_named_delete to every physical partition and
// ORs the per-partition existed-flags (the id lives in exactly one partition, so
// at most one reports true). Fail-loud on any unreachable partition
// (OnUnavailable=Fail); the op is idempotent so an interrupted fan-out retries
// cleanly. Writes route to each partition's Raft leader via CallPhysical.
func (e *embedded) namedDeleteFanOut(name string, P int, gen uint32, id uint64, opts WriteOpts) (bool, error) {
	a := cluster.FanArgs{
		Collection:    name,
		P:             P,
		Generation:    gen,
		K:             -1,
		Op:            "vector_named_delete",
		Consistency:   cluster.LeaderOnly,
		OnUnavailable: cluster.Fail,
		Encode: func(physCol string) []byte {
			return ops.EncodeNamedDeleteArgs(physCol, id)
		},
	}
	decode := func(raw []byte) ([]int, error) {
		if len(raw) > 0 && raw[0] == 1 {
			return []int{1}, nil
		}
		return []int{0}, nil
	}
	merge := func(parts [][]int, _ int) []int {
		sum := 0
		for _, p := range parts {
			for _, c := range p {
				sum += c
			}
		}
		return []int{sum}
	}
	counts, _, err := cluster.FanOut(a, e.node.CallPhysical, decode, merge)
	if err != nil {
		return false, err
	}
	// Write-consistency: a named delete-by-id scatters to every partition, but the
	// point lives in (and is genuinely removed from) exactly ONE partition —
	// PartitionOf(id, P). The barrier targets that owning partition's shard (the
	// others no-op). Default opts → no-op (the common case skips this entirely).
	if opts.wcActive() {
		owning := string(ops.PartitionKeyGen(name, gen, ops.PartitionOf(id, P)))
		if err := e.barrierPhys(owning, opts); err != nil {
			return false, err
		}
	}
	for _, c := range counts {
		if c > 0 {
			return true, nil
		}
	}
	return false, nil
}

// VectorNamedSearch fans a single-named-space KNN search across all P physical
// partitions and merges by ascending distance into the global top-k. The named
// space NAME + query + filter ride in each scattered op so every partition
// searches the same space; IDs are disjoint across partitions so the global
// top-k is exact. Mirrors denseFanOut. Back-compat convenience: defaults to
// AnyReplica / Partial (zero NamedSearchOpts).
func (e *embedded) VectorNamedSearch(ctx context.Context, name, vectorName string, query []float32, k int, filter VectorFilter) ([]VectorResult, error) {
	return e.VectorNamedSearchExt(ctx, name, vectorName, query, k, NamedSearchOpts{Filter: filter})
}

// VectorNamedSearchExt is VectorNamedSearch with read-consistency opts. It
// mirrors VectorMVSearch: a Linearizable read runs the meta readIndex barrier
// (resolveCollectionForRead) on the coordinator BEFORE resolving the catalog,
// routes a single-partition read to the leader via callReadLeader, and re-encodes
// rc into EVERY per-partition arg so each shard's data barrier fires.
func (e *embedded) VectorNamedSearchExt(_ context.Context, name, vectorName string, query []float32, k int, opts NamedSearchOpts) ([]VectorResult, error) {
	name, err := e.resolveCollectionForRead(name, opts.ReadConsistency, time.Now().Add(metaReadIndexReadTimeout))
	if err != nil {
		return nil, err
	}
	if P, gen, ok := e.catalog.PartitionsGen(name); ok && P > 1 {
		return e.namedSearchFanOut(name, vectorName, P, gen, query, k, opts.Filter, opts.ReadConsistency, opts.OnPartitionUnavailable, opts.MaxStaleness)
	}
	body, err := e.callReadLeader("vector_named_search",
		ops.EncodeNamedSearchArgsOpts(name, vectorName, query, k, opts.Filter, opts.ReadConsistency, opts.OnPartitionUnavailable, opts.MaxStaleness),
		opts.ReadConsistency)
	if err != nil {
		return nil, err
	}
	return ops.DecodeVectorSearchResults(body)
}

// namedSearchFanOut scatters vector_named_search to all partitions and merges by
// ascending distance into the global top-k — the named-vector mirror of
// denseFanOut. The vector NAME rides in every per-partition op, and the rc/opa
// trailer carries Linearizable to each shard so the readIndex barrier runs on
// every partition's leader (mirror mvFanOut).
func (e *embedded) namedSearchFanOut(name, vectorName string, P int, gen uint32, query []float32, k int, filter VectorFilter, rc, opa uint8, bound uint64) ([]VectorResult, error) {
	a := cluster.FanArgs{
		Collection:    name,
		P:             P,
		Generation:    gen,
		K:             k,
		Op:            "vector_named_search",
		Consistency:   cluster.Consistency(rc),
		OnUnavailable: cluster.OnUnavailable(opa),
		Encode: func(physCol string) []byte {
			// rc rides EVERY per-partition arg so each shard's data barrier fires
			// (anti-silent-drop). Mirrors mvFanOut's EncodeMVSearchArgsOptsFilter.
			return ops.EncodeNamedSearchArgsOpts(physCol, vectorName, query, k, filter, rc, opa, bound)
		},
	}
	decode := func(raw []byte) ([]cluster.Scored, error) {
		results, err := ops.DecodeVectorSearchResults(raw)
		if err != nil {
			return nil, err
		}
		out := make([]cluster.Scored, len(results))
		for i, r := range results {
			out[i] = cluster.Scored{ID: r.ID, Dist: r.Distance, Score: r.Score}
		}
		return out, nil
	}
	scored, _, err := e.scatterMerge(a, decode, func(x, y cluster.Scored) bool { return x.Dist < y.Dist })
	if err != nil {
		return nil, err
	}
	out := make([]VectorResult, len(scored))
	for i, s := range scored {
		out[i] = VectorResult{ID: s.ID, Distance: s.Dist, Score: s.Score}
	}
	return out, nil
}

// VectorNamedSparseSearch fans a sparse-dot-product top-k search against a SPARSE
// named space across all P physical partitions and merges by DESCENDING score into
// the global top-k. Back-compat convenience: defaults to AnyReplica / Partial.
func (e *embedded) VectorNamedSparseSearch(ctx context.Context, name, space string, query VectorSparse, k int, filter VectorFilter) ([]VectorResult, error) {
	return e.VectorNamedSparseSearchExt(ctx, name, space, query, k, NamedSearchOpts{Filter: filter})
}

// VectorNamedSparseSearchExt is VectorNamedSparseSearch with read-consistency opts.
// Same barrier/leader/fan-out wiring as VectorNamedSearchExt — a Linearizable read
// runs the meta readIndex barrier on the coordinator, routes a single-partition read
// to the leader via callReadLeader, and re-encodes rc into EVERY per-partition arg.
func (e *embedded) VectorNamedSparseSearchExt(_ context.Context, name, space string, query VectorSparse, k int, opts NamedSearchOpts) ([]VectorResult, error) {
	name, err := e.resolveCollectionForRead(name, opts.ReadConsistency, time.Now().Add(metaReadIndexReadTimeout))
	if err != nil {
		return nil, err
	}
	if P, gen, ok := e.catalog.PartitionsGen(name); ok && P > 1 {
		return e.namedSparseSearchFanOut(name, space, P, gen, query, k, opts.Filter, opts.ReadConsistency, opts.OnPartitionUnavailable, opts.MaxStaleness)
	}
	body, err := e.callReadLeader("vector_named_sparse_search",
		ops.EncodeNamedSparseSearchArgsOpts(name, space, query, k, opts.Filter, opts.ReadConsistency, opts.OnPartitionUnavailable, opts.MaxStaleness),
		opts.ReadConsistency)
	if err != nil {
		return nil, err
	}
	// VectorResult is an alias of vector.Result, so DecodeHybridResults's slice IS a
	// []VectorResult (the score-carrying decode the sparse lane needs).
	return ops.DecodeHybridResults(body)
}

// namedSparseSearchFanOut scatters vector_named_sparse_search to all partitions and
// merges by DESCENDING score into the global top-k. Unlike the dense named search
// (which merges by ascending distance), the sparse lane ranks by sparse dot product,
// so the merge comparator is score-descending. The sparse query + rc/opa trailer
// ride every per-partition op; IDs are disjoint across partitions so the global
// top-k is exact.
func (e *embedded) namedSparseSearchFanOut(name, space string, P int, gen uint32, query VectorSparse, k int, filter VectorFilter, rc, opa uint8, bound uint64) ([]VectorResult, error) {
	a := cluster.FanArgs{
		Collection:    name,
		P:             P,
		Generation:    gen,
		K:             k,
		Op:            "vector_named_sparse_search",
		Consistency:   cluster.Consistency(rc),
		OnUnavailable: cluster.OnUnavailable(opa),
		Encode: func(physCol string) []byte {
			return ops.EncodeNamedSparseSearchArgsOpts(physCol, space, query, k, filter, rc, opa, bound)
		},
	}
	decode := func(raw []byte) ([]cluster.Scored, error) {
		// The per-partition sparse handler encodes with EncodeHybridResults (carries
		// SCORE); decode it score-aware so the merge below ranks by the real dot product.
		results, err := ops.DecodeHybridResults(raw)
		if err != nil {
			return nil, err
		}
		out := make([]cluster.Scored, len(results))
		for i, r := range results {
			out[i] = cluster.Scored{ID: r.ID, Dist: r.Distance, Score: r.Score}
		}
		return out, nil
	}
	// Sparse ranks by score DESCENDING (ties by lower id), matching the engine's
	// searchTopK ordering, NOT by ascending distance.
	scored, _, err := e.scatterMerge(a, decode, func(x, y cluster.Scored) bool {
		if x.Score != y.Score {
			return x.Score > y.Score
		}
		return x.ID < y.ID
	})
	if err != nil {
		return nil, err
	}
	out := make([]VectorResult, len(scored))
	for i, s := range scored {
		out[i] = VectorResult{ID: s.ID, Distance: s.Dist, Score: s.Score}
	}
	return out, nil
}

// VectorNamedHybridSearch fuses a DENSE named space and a SPARSE named space into
// the top-k (cross-space hybrid). On a single partition it runs the fused
// vector_named_hybrid_search op; on P>1 it fans the UNFUSED-lanes op
// (vector_named_hybrid_lanes) to every partition and fuses ONCE globally
// (partition-exact, mirroring the dense hybridFanOut). A Linearizable read runs the
// meta readIndex barrier on the coordinator, routes a single-partition read to the
// leader via callReadLeader, and re-encodes rc into EVERY per-partition arg so each
// shard's data barrier fires.
func (e *embedded) VectorNamedHybridSearch(ctx context.Context, name, denseSpace string, denseQ []float32, sparseSpace string, sparseQ VectorSparse, k int, opts NamedHybridOpts) ([]VectorResult, error) {
	res, _, err := e.VectorNamedHybridSearchExt(ctx, name, denseSpace, denseQ, sparseSpace, sparseQ, k, opts)
	return res, err
}

// VectorNamedHybridSearchExt is VectorNamedHybridSearch with the partition-
// degradation signal exposed as FanMeta, mirroring VectorHybridSearch. Under the
// default OnPartitionUnavailable=Partial, a partitioned (P>1) search over a cluster
// with an unreachable partition fuses the reachable partitions' lanes and returns
// FanMeta{Degraded:true, Missing:...} rather than silently truncating the top-k. A
// single-partition search never fans out, so it always reports a zero FanMeta.
func (e *embedded) VectorNamedHybridSearchExt(_ context.Context, name, denseSpace string, denseQ []float32, sparseSpace string, sparseQ VectorSparse, k int, opts NamedHybridOpts) ([]VectorResult, FanMeta, error) {
	name, err := e.resolveCollectionForRead(name, opts.ReadConsistency, time.Now().Add(metaReadIndexReadTimeout))
	if err != nil {
		return nil, FanMeta{}, err
	}
	hopts := toNamedHybridVectorOpts(opts)
	if P, gen, ok := e.catalog.PartitionsGen(name); ok && P > 1 {
		res, fr, err := e.namedHybridFanOut(name, denseSpace, denseQ, sparseSpace, sparseQ, P, gen, k, hopts, opts.ReadConsistency, opts.OnPartitionUnavailable, opts.MaxStaleness)
		return res, FanMeta{Degraded: fr.Degraded, Missing: fr.Missing}, err
	}
	body, err := e.callReadLeader("vector_named_hybrid_search",
		ops.EncodeNamedHybridArgs(name, denseSpace, denseQ, sparseSpace, sparseQ, k, hopts, opts.ReadConsistency, opts.OnPartitionUnavailable, opts.MaxStaleness),
		opts.ReadConsistency)
	if err != nil {
		return nil, FanMeta{}, err
	}
	// The fused result carries Score (EncodeHybridResults); DecodeHybridResults's
	// slice IS a []VectorResult (the alias).
	res, err := ops.DecodeHybridResults(body)
	return res, FanMeta{}, err
}

// namedHybridFanOut runs vector_named_hybrid_lanes on every physical partition,
// unions the per-partition dense + sparse lanes, truncates each unioned lane to the
// GLOBAL denseK/sparseK, then fuses ONCE — reproducing the single-partition named
// hybrid EXACTLY. This is the named cross-space mirror of hybridFanOut.
//
// EXACTNESS INVARIANT: the union is truncated to denseK/sparseK BEFORE fusing (RRF
// rank + Weighted min-max normalization are computed over exactly the
// denseK/sparseK lane lists, matching the engine's NamedHybrid). The denseK/sparseK
// defaults (max(k,50)) match NamedHybrid / namedHybridLanesLocked. Single-lane
// degradation (empty dense or empty sparse query) is collapsed here exactly as
// NamedHybrid does, so a degraded fan-out matches the degraded single-partition path.
func (e *embedded) namedHybridFanOut(name, denseSpace string, denseQ []float32, sparseSpace string, sparseQ VectorSparse, P int, gen uint32, k int, hopts vector.HybridOpts, rc, opa uint8, bound uint64) ([]VectorResult, cluster.FanResult, error) {
	denseK := hopts.DenseK
	if denseK <= 0 {
		if denseK = k; denseK < 50 {
			denseK = 50
		}
	}
	sparseK := hopts.SparseK
	if sparseK <= 0 {
		if sparseK = k; sparseK < 50 {
			sparseK = 50
		}
	}
	a := cluster.FanArgs{
		Collection:    name,
		P:             P,
		Generation:    gen,
		K:             k,
		Op:            "vector_named_hybrid_lanes",
		Consistency:   cluster.Consistency(rc),
		OnUnavailable: cluster.OnUnavailable(opa),
		Encode: func(physCol string) []byte {
			// rc rides EVERY per-partition arg so each shard's data barrier fires.
			return ops.EncodeNamedHybridArgs(physCol, denseSpace, denseQ, sparseSpace, sparseQ, k, hopts, rc, opa, bound)
		},
	}
	decode := func(raw []byte) ([]hybridLanes, error) {
		d, s, derr := ops.DecodeHybridLanesResult(raw)
		if derr != nil {
			return nil, derr
		}
		return []hybridLanes{{dense: d, sparse: s}}, nil
	}
	merge := func(parts [][]hybridLanes, _ int) []hybridLanes {
		var all []hybridLanes
		for _, p := range parts {
			all = append(all, p...)
		}
		return all
	}
	parts, fr, err := cluster.FanOut(a, e.node.CallPhysical, decode, merge)
	if err != nil {
		return nil, fr, err
	}

	var allDense, allSparse []vector.Result
	for _, p := range parts {
		allDense = append(allDense, p.dense...)
		allSparse = append(allSparse, p.sparse...)
	}
	// EXACTNESS INVARIANT: truncate to the global denseK/sparseK BEFORE fusing.
	// Dense lane: ascending Distance, secondary ascending ID; sparse lane:
	// descending Score, secondary ascending ID (the only cross-partition-stable
	// tiebreaks — equal scores/distances admit any valid top-k order). Mirrors
	// hybridFanOut exactly.
	sort.SliceStable(allDense, func(i, j int) bool {
		if allDense[i].Distance != allDense[j].Distance {
			return allDense[i].Distance < allDense[j].Distance
		}
		return allDense[i].ID < allDense[j].ID
	})
	if len(allDense) > denseK {
		allDense = allDense[:denseK]
	}
	sort.SliceStable(allSparse, func(i, j int) bool {
		if allSparse[i].Score != allSparse[j].Score {
			return allSparse[i].Score > allSparse[j].Score
		}
		return allSparse[i].ID < allSparse[j].ID
	})
	if len(allSparse) > sparseK {
		allSparse = allSparse[:sparseK]
	}

	// Mirror NamedHybrid's single-lane degradation, then fuse both lanes.
	var fused []vector.Result
	switch {
	case sparseQ.IsZero():
		fused = allDense
		if len(fused) > k {
			fused = fused[:k]
		}
	case len(denseQ) == 0:
		fused = allSparse
		if len(fused) > k {
			fused = fused[:k]
		}
	default:
		fused = vector.Fuse(allDense, allSparse, hopts.Method, hopts.Alpha, hopts.RRFK, k)
	}
	out := make([]VectorResult, len(fused))
	for i, r := range fused {
		out[i] = VectorResult(r)
	}
	return out, fr, nil
}

// VectorNamedQuery runs the unified Query API (vector_named_query) against a NAMED
// collection — a root + N prefetch leaves where EVERY leaf targets a named SPACE
// (dense or sparse), combined by FUSION (RRF/Weighted/DBSF over N>2 lanes) or
// RERANK (the root re-scores the union of the prefetch candidates). This is the
// named-family mirror of (*embedded).VectorQuery: specBytes is the marshaled
// pb.QuerySpec carried on the wire to each shard; spec is the decoded engine spec
// the coordinator needs for the fusion/rerank merge. For a partitioned collection
// (P>1) it fans vector_named_query to every partition and merges per mode
// (namedQueryFanOut); for a single shard it runs the op on the read leader and
// routes the one shard's mode-tagged QueryResult through the SAME fusion/rerank
// merge (FUSION fills Lanes, never a flat Fused — reading qr.Fused directly would
// drop every FUSION result). A Linearizable read arms the meta + per-shard
// barriers (rc rides every per-partition arg).
func (e *embedded) VectorNamedQuery(_ context.Context, name string, specBytes []byte, spec vector.QuerySpec, opts ReadOpts) ([]VectorResult, FanMeta, error) {
	name, err := e.resolveCollectionForRead(name, opts.ReadConsistency, time.Now().Add(metaReadIndexReadTimeout))
	if err != nil {
		return nil, FanMeta{}, err
	}
	if P, gen, ok := e.catalog.PartitionsGen(name); ok && P > 1 {
		// RECOMMEND/DISCOVER named cluster pre-passes (mirror VectorQuery's dense
		// pre-passes): a recommend/discover leaf's example/target/context ids may live
		// on OTHER partitions, so resolve them cluster-wide IN THE LEAF'S SPACE +
		// derive (AVERAGE) / embed (BEST_SCORE/discover) ONCE on the coordinator BEFORE
		// fanning out (the partition handlers reject an un-rewritten recommend/discover
		// leaf, or would re-resolve absent ids against their local index). Then fan out
		// the rewritten (partition-invariant) spec via the EXISTING namedQueryFanOut and
		// prune the recommend example ids from the merged top-k. A spec without
		// recommend/discover leaves is fanned out verbatim (rewritten=false),
		// byte-identical to before. Recommend and discover compose: each pre-pass only
		// touches its own leaf kind. P==1 routes through the single-node
		// (*NamedCollection).Query pre-pass below (exactly one derive/embed per P).
		rewSpec, rewBytes, exclude, rewritten, rerr := e.resolveNamedRecommendForFanOut(name, spec)
		if rerr != nil {
			return nil, FanMeta{}, rerr
		}
		if rewritten {
			spec, specBytes = rewSpec, rewBytes
		}
		discSpec, discBytes, discRewritten, derr := e.resolveNamedDiscoverForFanOut(name, spec)
		if derr != nil {
			return nil, FanMeta{}, derr
		}
		if discRewritten {
			spec, specBytes = discSpec, discBytes
		}
		res, fr, err := e.namedQueryFanOut(name, P, gen, specBytes, spec, opts.ReadConsistency, opts.OnPartitionUnavailable, opts.MaxStaleness)
		if err != nil {
			return res, FanMeta{Degraded: fr.Degraded, Missing: fr.Missing}, err
		}
		if rewritten && len(exclude) > 0 {
			// Prune the example ids AFTER the global fuse/rerank merge + re-truncate to the
			// requested k (the over-fetch made room). Mirrors VectorQuery's dense path.
			res = vector.ExcludeExamplesFromResults(res, func(r VectorResult) uint64 { return r.ID }, exclude, queryK(rewSpec)-len(exclude))
		}
		return res, FanMeta{Degraded: fr.Degraded, Missing: fr.Missing}, err
	}
	// RECOMMEND/DISCOVER pre-pass for a single-shard NESTED-FUSION spec: mirror the
	// P>1 coordinator path (resolveNamedRecommend/DiscoverForFanOut) so P==1 and P>1
	// produce byte-identical results. The partition's NamedQueryTreeLanes would
	// otherwise resolve a recommend/discover leaf LOCALLY and keep its example ids IN
	// the DBSF/Weighted lane normalization with no post-fold prune; rewriting at the
	// coordinator here (so the partition receives an already-resolved spec) and pruning
	// the merged result POST-fold matches the P>1 flow. A flat / non-recommend spec
	// short-circuits (rewritten=false), byte-identical to before.
	var p1Exclude map[uint64]struct{}
	var p1RewSpec vector.QuerySpec
	var p1Rewritten bool
	if vector.SpecHasNestedFusion(spec) {
		rewSpec, rewBytes, excl, rewritten, rerr := e.resolveNamedRecommendForFanOut(name, spec)
		if rerr != nil {
			return nil, FanMeta{}, rerr
		}
		if rewritten {
			spec, specBytes = rewSpec, rewBytes
			p1RewSpec, p1Exclude, p1Rewritten = rewSpec, excl, true
		}
		discSpec, discBytes, discRewritten, derr := e.resolveNamedDiscoverForFanOut(name, spec)
		if derr != nil {
			return nil, FanMeta{}, derr
		}
		if discRewritten {
			spec, specBytes = discSpec, discBytes
		}
	}
	body, err := e.callReadLeader("vector_named_query",
		ops.EncodeQueryArgs(name, specBytes, opts.ReadConsistency, opts.OnPartitionUnavailable, opts.MaxStaleness),
		opts.ReadConsistency)
	if err != nil {
		return nil, FanMeta{}, err
	}
	qr, derr := ops.DecodeQueryResult(body)
	if derr != nil {
		return nil, FanMeta{}, derr
	}
	// Single shard: route the mode-tagged result through the SAME coordinator merge
	// the P>1 path uses so one partition fuses/reranks identically to the union (and
	// FUSION actually fuses — the wire carries UNFUSED lanes, not a flat Fused).
	single := []vector.QueryResult{qr}
	switch spec.Mode {
	case vector.ModeRerank:
		return rerankMergeFanOut(single, spec.Root, queryK(spec)), FanMeta{}, nil
	default: // ModeFusion
		// A nested MULTI-lane FUSION spec ships UNFUSED tree-lanes (the named handler
		// emits them via SpecHasNestedFusion); the single shard runs the SAME recursive
		// tree fold the P>1 coordinator uses over its one lane list ⇒ P==1 is exactly the
		// single-node engine fold at every FUSION node. Mirrors the dense VectorQuery.
		if vector.SpecHasNestedFusion(spec) {
			res, terr := treeFusionMergeFanOut(single, spec, queryK(spec))
			if terr != nil {
				return nil, FanMeta{}, terr
			}
			if p1Rewritten && len(p1Exclude) > 0 {
				// Post-fold recommend exclusion (the over-fetch widened spec.K so k results
				// survive). Mirrors the dense VectorQuery single-shard nested path.
				res = vector.ExcludeExamplesFromResults(res, func(r VectorResult) uint64 { return r.ID }, p1Exclude, queryK(p1RewSpec)-len(p1Exclude))
			}
			return res, FanMeta{}, nil
		}
		return fusionMergeFanOut(single, spec, queryK(spec)), FanMeta{}, nil
	}
}

// namedQueryFanOut runs vector_named_query on every physical partition and
// reproduces the single-partition (*NamedCollection).Query result EXACTLY for both
// root modes — the named-collection generalization of queryFanOut (v1 dense). It
// REUSES the v1 fusionMergeFanOut / rerankMergeFanOut merges VERBATIM: named point
// ids are partition-disjoint (PartitionOf(id,P) maps a point to ONE partition
// regardless of which named spaces it populates — see the partition-disjoint id
// note at the top of this file), so FUSION (union lane[i] + truncate-per-lane +
// fold once) and RERANK (merge per-partition top-Ks) are partition-exact exactly
// as in v1.
//
// rc/opa/bound ride EVERY per-partition arg (via EncodeQueryArgs' opts trailer) so
// a Linearizable query arms each partition leader's readIndex barrier; they are
// never silently dropped on the fan-out path. Mirrors namedHybridFanOut's FanMeta /
// degradation handling.
func (e *embedded) namedQueryFanOut(name string, P int, gen uint32, specBytes []byte, spec vector.QuerySpec, rc, opa uint8, bound uint64) ([]VectorResult, cluster.FanResult, error) {
	k := queryK(spec)

	a := cluster.FanArgs{
		Collection:    name,
		P:             P,
		Generation:    gen,
		K:             k,
		Op:            "vector_named_query",
		Consistency:   cluster.Consistency(rc),
		OnUnavailable: cluster.OnUnavailable(opa),
		Encode: func(physCol string) []byte {
			// specBytes is reused verbatim for every partition (the named spaces +
			// per-leaf filters travel inside it); only the collection name in the
			// header changes. The opts trailer carries rc/opa/bound to each shard.
			return ops.EncodeQueryArgs(physCol, specBytes, rc, opa, bound)
		},
	}
	decode := func(raw []byte) ([]vector.QueryResult, error) {
		qr, err := ops.DecodeQueryResult(raw)
		if err != nil {
			return nil, err
		}
		return []vector.QueryResult{qr}, nil
	}
	merge := func(parts [][]vector.QueryResult, _ int) []vector.QueryResult {
		var all []vector.QueryResult
		for _, p := range parts {
			all = append(all, p...)
		}
		return all
	}
	parts, fr, err := cluster.FanOut(a, e.node.CallPhysical, decode, merge)
	if err != nil {
		return nil, fr, err
	}

	switch spec.Mode {
	case vector.ModeRerank:
		return rerankMergeFanOut(parts, spec.Root, k), fr, nil
	default: // ModeFusion
		// A spec with a nested MULTI-lane FUSION node shipped per-partition UNFUSED
		// tree-lanes (the named handler emits them via SpecHasNestedFusion); re-walk the
		// spec tree and fold each FUSION node over the global union (P>1==P1 EXACT,
		// multi-space dense/sparse lanes via the orientation-aware fold). A flat /
		// nested-single-lane / nested-RERANK FUSION spec takes the unchanged top-level
		// fold. Mirrors the dense queryFanOut.
		if vector.SpecHasNestedFusion(spec) {
			res, terr := treeFusionMergeFanOut(parts, spec, k)
			return res, fr, terr
		}
		return fusionMergeFanOut(parts, spec, k), fr, nil
	}
}

// VectorMVHybridSearch fuses an MV collection's MaxSim (late-interaction dense) lane
// and its per-doc sparse lane into the top-k (cross-modality hybrid). On a single
// partition it runs the fused vector_mv_hybrid_search op; on P>1 it fans the
// UNFUSED-lanes op (vector_mv_hybrid_lanes) to every partition and fuses ONCE
// globally (partition-exact, mirroring VectorNamedHybridSearch). A Linearizable read
// runs the meta readIndex barrier on the coordinator, routes a single-partition read
// to the leader via callReadLeader, and re-encodes rc into EVERY per-partition arg so
// each shard's data barrier fires.
func (e *embedded) VectorMVHybridSearch(ctx context.Context, name string, query [][]float32, sparseQ VectorSparse, k int, opts MVHybridOpts) ([]VectorResult, error) {
	res, _, err := e.VectorMVHybridSearchExt(ctx, name, query, sparseQ, k, opts)
	return res, err
}

// VectorMVHybridSearchExt is VectorMVHybridSearch with the partition-degradation
// signal exposed as FanMeta, mirroring VectorHybridSearch / VectorNamedHybridSearchExt.
// Under the default OnPartitionUnavailable=Partial, a partitioned (P>1) search over a
// cluster with an unreachable partition fuses the reachable partitions' lanes and
// returns FanMeta{Degraded:true, Missing:...} rather than silently truncating the
// top-k. A single-partition search never fans out, so it always reports a zero FanMeta.
func (e *embedded) VectorMVHybridSearchExt(_ context.Context, name string, query [][]float32, sparseQ VectorSparse, k int, opts MVHybridOpts) ([]VectorResult, FanMeta, error) {
	name, err := e.resolveCollectionForRead(name, opts.ReadConsistency, time.Now().Add(metaReadIndexReadTimeout))
	if err != nil {
		return nil, FanMeta{}, err
	}
	hopts := toMVHybridVectorOpts(opts)
	if P, gen, ok := e.catalog.PartitionsGen(name); ok && P > 1 {
		res, fr, err := e.mvHybridFanOut(name, query, sparseQ, P, gen, k, hopts, opts.ReadConsistency, opts.OnPartitionUnavailable, opts.MaxStaleness)
		return res, FanMeta{Degraded: fr.Degraded, Missing: fr.Missing}, err
	}
	body, err := e.callReadLeader("vector_mv_hybrid_search",
		ops.EncodeMVHybridArgs(name, query, sparseQ, k, hopts, opts.ReadConsistency, opts.OnPartitionUnavailable, opts.MaxStaleness),
		opts.ReadConsistency)
	if err != nil {
		return nil, FanMeta{}, err
	}
	// The fused result carries Score (EncodeHybridResults); DecodeHybridResults's
	// slice IS a []VectorResult (the alias).
	res, err := ops.DecodeHybridResults(body)
	return res, FanMeta{}, err
}

// mvHybridFanOut runs vector_mv_hybrid_lanes on every physical partition, unions the
// per-partition MaxSim + sparse lanes, truncates each unioned lane to the GLOBAL
// denseK/sparseK, then fuses ONCE — reproducing the single-partition MV hybrid
// EXACTLY. The MV cross-modality mirror of namedHybridFanOut.
//
// LANE-ORIENTATION DIFFERENCE vs namedHybridFanOut: the MV MaxSim lane is a
// DESCENDING relevance SCORE (higher = better), NOT an ascending distance. So the
// MaxSim ("dense") union is sorted DESCENDING by Score (secondary ascending ID)
// before truncation — the same order as the sparse lane — and the final fusion uses
// FuseScoreLanes (both lanes score-desc), matching the engine's MVHybrid.
//
// EXACTNESS INVARIANT: each unioned lane is truncated to denseK/sparseK BEFORE
// fusing (RRF rank + Weighted min-max normalization are computed over exactly the
// denseK/sparseK lane lists, matching MVHybrid). The denseK/sparseK defaults
// (max(k,50)) match MVHybrid / mvHybridLanesLocked. Single-lane degradation (empty
// query tokens or empty sparse query) is collapsed here exactly as MVHybrid does, so
// a degraded fan-out matches the degraded single-partition path.
func (e *embedded) mvHybridFanOut(name string, query [][]float32, sparseQ VectorSparse, P int, gen uint32, k int, hopts vector.HybridOpts, rc, opa uint8, bound uint64) ([]VectorResult, cluster.FanResult, error) {
	denseK := hopts.DenseK
	if denseK <= 0 {
		if denseK = k; denseK < 50 {
			denseK = 50
		}
	}
	sparseK := hopts.SparseK
	if sparseK <= 0 {
		if sparseK = k; sparseK < 50 {
			sparseK = 50
		}
	}
	a := cluster.FanArgs{
		Collection:    name,
		P:             P,
		Generation:    gen,
		K:             k,
		Op:            "vector_mv_hybrid_lanes",
		Consistency:   cluster.Consistency(rc),
		OnUnavailable: cluster.OnUnavailable(opa),
		Encode: func(physCol string) []byte {
			// rc rides EVERY per-partition arg so each shard's data barrier fires.
			return ops.EncodeMVHybridArgs(physCol, query, sparseQ, k, hopts, rc, opa, bound)
		},
	}
	decode := func(raw []byte) ([]hybridLanes, error) {
		d, s, derr := ops.DecodeHybridLanesResult(raw)
		if derr != nil {
			return nil, derr
		}
		return []hybridLanes{{dense: d, sparse: s}}, nil
	}
	merge := func(parts [][]hybridLanes, _ int) []hybridLanes {
		var all []hybridLanes
		for _, p := range parts {
			all = append(all, p...)
		}
		return all
	}
	parts, fr, err := cluster.FanOut(a, e.node.CallPhysical, decode, merge)
	if err != nil {
		return nil, fr, err
	}

	var allDense, allSparse []vector.Result
	for _, p := range parts {
		allDense = append(allDense, p.dense...)
		allSparse = append(allSparse, p.sparse...)
	}
	// EXACTNESS INVARIANT: truncate to the global denseK/sparseK BEFORE fusing. BOTH
	// lanes are descending Score (the MaxSim lane is a desc SCORE, not a distance),
	// secondary ascending ID (the only cross-partition-stable tiebreaks).
	sort.SliceStable(allDense, func(i, j int) bool {
		if allDense[i].Score != allDense[j].Score {
			return allDense[i].Score > allDense[j].Score
		}
		return allDense[i].ID < allDense[j].ID
	})
	if len(allDense) > denseK {
		allDense = allDense[:denseK]
	}
	sort.SliceStable(allSparse, func(i, j int) bool {
		if allSparse[i].Score != allSparse[j].Score {
			return allSparse[i].Score > allSparse[j].Score
		}
		return allSparse[i].ID < allSparse[j].ID
	})
	if len(allSparse) > sparseK {
		allSparse = allSparse[:sparseK]
	}

	// Mirror MVHybrid's single-lane degradation, then fuse both lanes (score-desc).
	var fused []vector.Result
	switch {
	case sparseQ.IsZero():
		fused = allDense
		if len(fused) > k {
			fused = fused[:k]
		}
	case len(query) == 0:
		fused = allSparse
		if len(fused) > k {
			fused = fused[:k]
		}
	default:
		fused = vector.FuseScoreLanes(allDense, allSparse, hopts.Method, hopts.Alpha, hopts.RRFK, k)
	}
	out := make([]VectorResult, len(fused))
	for i, r := range fused {
		out[i] = VectorResult(r)
	}
	return out, fr, nil
}

// VectorNamedSearchDocs is VectorNamedSearch returning each hit enriched with the
// shared per-point payload. Fans out per named space, unions by ascending
// distance — mirrors docsFanOut. Back-compat convenience: defaults to
// AnyReplica / Partial.
func (e *embedded) VectorNamedSearchDocs(ctx context.Context, name, vectorName string, query []float32, k int, filter VectorFilter) ([]VectorDocument, error) {
	return e.VectorNamedSearchDocsExt(ctx, name, vectorName, query, k, NamedSearchOpts{Filter: filter})
}

// VectorNamedSearchDocsExt is VectorNamedSearchDocs with read-consistency opts.
// Same barrier/leader/fan-out wiring as VectorNamedSearchExt.
func (e *embedded) VectorNamedSearchDocsExt(_ context.Context, name, vectorName string, query []float32, k int, opts NamedSearchOpts) ([]VectorDocument, error) {
	name, err := e.resolveCollectionForRead(name, opts.ReadConsistency, time.Now().Add(metaReadIndexReadTimeout))
	if err != nil {
		return nil, err
	}
	if P, gen, ok := e.catalog.PartitionsGen(name); ok && P > 1 {
		return e.namedDocsFanOut(name, vectorName, P, gen, query, k, opts.Filter, opts.ReadConsistency, opts.OnPartitionUnavailable, opts.MaxStaleness)
	}
	body, err := e.callReadLeader("vector_named_search_docs",
		ops.EncodeNamedSearchArgsOpts(name, vectorName, query, k, opts.Filter, opts.ReadConsistency, opts.OnPartitionUnavailable, opts.MaxStaleness),
		opts.ReadConsistency)
	if err != nil {
		return nil, err
	}
	return ops.DecodeVectorDocs(body)
}

// namedDocsFanOut scatters vector_named_search_docs to all partitions and merges
// the enriched documents by ascending distance into the global top-k — the
// named-vector mirror of docsFanOut. The shared payload round-trips inside each
// Document. The rc/opa trailer rides every per-partition arg (mirror mvFanOut).
func (e *embedded) namedDocsFanOut(name, vectorName string, P int, gen uint32, query []float32, k int, filter VectorFilter, rc, opa uint8, bound uint64) ([]VectorDocument, error) {
	a := cluster.FanArgs{
		Collection:    name,
		P:             P,
		Generation:    gen,
		K:             k,
		Op:            "vector_named_search_docs",
		Consistency:   cluster.Consistency(rc),
		OnUnavailable: cluster.OnUnavailable(opa),
		Encode: func(physCol string) []byte {
			return ops.EncodeNamedSearchArgsOpts(physCol, vectorName, query, k, filter, rc, opa, bound)
		},
	}
	decode := func(raw []byte) ([]vector.Document, error) {
		return ops.DecodeVectorDocs(raw)
	}
	merge := func(parts [][]vector.Document, k int) []vector.Document {
		var all []vector.Document
		for _, p := range parts {
			all = append(all, p...)
		}
		sort.SliceStable(all, func(i, j int) bool {
			if all[i].Distance != all[j].Distance {
				return all[i].Distance < all[j].Distance
			}
			return all[i].ID < all[j].ID
		})
		if k >= 0 && len(all) > k {
			all = all[:k]
		}
		return all
	}
	docs, _, err := cluster.FanOut(a, e.node.CallPhysical, decode, merge)
	return docs, err
}

// VectorNamedScroll fans vector_named_scroll to all P physical partitions, unions
// the per-partition documents (IDs disjoint — no dedup), and caps to limit
// (ascending ID for determinism). The shared payload round-trips per Document.
// Mirrors scrollFanOut.
func (e *embedded) VectorNamedScroll(ctx context.Context, name string, filter VectorFilter, limit int, cursor string) ([]VectorDocument, string, error) {
	return e.VectorNamedScrollExt(ctx, name, filter, limit, cursor, NamedScrollOpts{})
}

// VectorNamedScrollExt is VectorNamedScroll with read-consistency opts. A
// Linearizable scroll runs the meta readIndex barrier on the coordinator, routes
// a single-partition scroll to the leader via callReadLeader, and re-encodes rc
// into every per-partition arg (mirror VectorNamedSearchExt).
func (e *embedded) VectorNamedScrollExt(_ context.Context, name string, filter VectorFilter, limit int, cursor string, opts NamedScrollOpts) ([]VectorDocument, string, error) {
	// Decode the input cursor TYPED so the order_by (v2) and id-scroll (v1) paths both
	// validate the cursor version against opts.OrderBy before dispatch (mirror VectorScroll).
	dec, err := ops.DecodeScrollCursorTyped(cursor)
	if err != nil {
		return nil, "", err
	}
	var order *ops.ScrollOrder
	var afterID uint64
	var hasAfter bool
	if opts.OrderBy != nil {
		ob := opts.OrderBy
		var verr error
		order, afterID, hasAfter, verr = buildScrollOrder(ob, dec)
		if verr != nil {
			return nil, "", verr
		}
	} else {
		// No order_by: only a v1 (id-only) cursor is valid; a v2 cursor here means a
		// client dropped order_by mid-pagination — reject loud (symmetric guard).
		if dec.Present && dec.Version != 1 {
			return nil, "", ops.ErrCursorOrderMismatch
		}
		afterID, hasAfter = dec.LastID, dec.Present
	}
	name, err = e.resolveCollectionForRead(name, opts.ReadConsistency, time.Now().Add(metaReadIndexReadTimeout))
	if err != nil {
		return nil, "", err
	}
	if P, gen, ok := e.catalog.PartitionsGen(name); ok && P > 1 {
		return e.namedScrollFanOut(name, P, gen, filter, limit, afterID, hasAfter, opts.ReadConsistency, opts.OnPartitionUnavailable, order, opts.MaxStaleness)
	}
	body, err := e.callReadLeader("vector_named_scroll",
		ops.EncodeNamedScrollArgsOrderBounded(name, filter, limit, afterID, hasAfter, opts.ReadConsistency, opts.OnPartitionUnavailable, order, opts.MaxStaleness),
		opts.ReadConsistency)
	if err != nil {
		return nil, "", err
	}
	docs, err := ops.DecodeVectorDocs(body)
	if err != nil {
		return nil, "", err
	}
	return docs, scrollNextCursorOrder(docs, limit, opts.OrderBy), nil
}

// namedScrollFanOut scatters vector_named_scroll to all partitions, unions,
// GLOBAL-sorts by ascending id, truncates to limit, and derives next_cursor —
// the named-vector mirror of scrollFanOut. Each partition receives the SAME
// global (filter, limit, afterID); the merge is gap-free + dup-free (ids disjoint
// across partitions, the global first `limit` ids > afterID are within the union
// of the per-partition first-`limit`). See scrollFanOut for the correctness note.
func (e *embedded) namedScrollFanOut(name string, P int, gen uint32, filter VectorFilter, limit int, afterID uint64, hasAfter bool, rc, opa uint8, order *ops.ScrollOrder, bound uint64) ([]VectorDocument, string, error) {
	a := cluster.FanArgs{
		Collection:    name,
		P:             P,
		Generation:    gen,
		K:             limit,
		Op:            "vector_named_scroll",
		Consistency:   cluster.Consistency(rc),
		OnUnavailable: cluster.OnUnavailable(opa),
		Encode: func(physCol string) []byte {
			// rc + the SAME global cursor/order ride every per-partition arg so each
			// shard's data barrier fires and every shard pages from the same bound.
			return ops.EncodeNamedScrollArgsOrderBounded(physCol, filter, limit, afterID, hasAfter, rc, opa, order, bound)
		},
	}
	decode := func(raw []byte) ([]vector.Document, error) {
		return ops.DecodeVectorDocs(raw)
	}
	merge := func(parts [][]vector.Document, _ int) []vector.Document {
		var all []vector.Document
		for _, p := range parts {
			all = append(all, p...)
		}
		return all
	}
	all, _, err := cluster.FanOut(a, e.node.CallPhysical, decode, merge)
	if err != nil {
		return nil, "", err
	}
	if order != nil {
		// GLOBAL (value, id) merge: sort the union by the order's total order (NOT id),
		// then truncate. Disjoint per-partition ids ⇒ no equal-comparing dups; missing-field
		// points were already EXCLUDED per partition. See scrollFanOut for the re-derivation.
		ob := scrollOrderByFromOps(order)
		all = sortDocsByOrder(all, ob)
		if limit > 0 && len(all) > limit {
			all = all[:limit]
		}
		return all, scrollNextCursorOrder(all, limit, ob), nil
	}
	sort.SliceStable(all, func(i, j int) bool { return all[i].ID < all[j].ID })
	if limit > 0 && len(all) > limit {
		all = all[:limit]
	}
	return all, scrollNextCursor(all, limit), nil
}

// VectorNamedDropCollection drops a partitioned named collection's every physical
// partition (+ the logical marker) and neutralizes the catalog, mirroring
// fanoutDrop; an unpartitioned drop goes straight to the single handler.
// VectorNamedDropCollection is an ADMIN op: it drops the REAL collection (name is
// NOT alias-resolved) and then CASCADE-removes every alias that pointed at it
// (best-effort). See VectorMVDropCollection.
func (e *embedded) VectorNamedDropCollection(ctx context.Context, name string) error {
	if P, gen, ok := e.catalog.PartitionsGen(name); ok && P > 1 {
		if err := e.namedDropCollectionFanout(ctx, name, P, gen); err != nil {
			return err
		}
		e.cleanupAliasesFor(ctx, name)
		return nil
	}
	if _, err := e.Call(ctx, "vector_named_drop_collection", ops.EncodeNamedNameArgs(name)); err != nil {
		return err
	}
	e.cleanupAliasesFor(ctx, name)
	return nil
}

// namedDropCollectionFanout drops a partitioned named collection. See fanoutDrop;
// this passes the named drop op + name-only args encoder. The named drop handler
// is idempotent (swallows ErrNoNamed), so a retry after a partial drop is safe.
func (e *embedded) namedDropCollectionFanout(ctx context.Context, name string, P int, gen uint32) error {
	return e.fanoutDrop(ctx, name, P, gen, "vector_named_drop_collection", func(phys string) []byte {
		return ops.EncodeNamedNameArgs(phys)
	})
}

// VectorNamedGetConfig is single-partition / logical: the named config is
// identical on every physical partition (they share the NamedVectors map), so a
// get-config reads the logical collection directly — no fan-out (mirror MV
// get_config passthrough). For a partitioned collection the logical marker
// carries the same config, so this is correct.
func (e *embedded) VectorNamedGetConfig(ctx context.Context, name string) (map[string]NamedVectorParams, error) {
	return e.VectorNamedGetConfigExt(ctx, name, ReadOpts{})
}

// VectorNamedGetConfigExt is VectorNamedGetConfig with read-consistency. A
// Linearizable read first arms the meta-catalog read barrier
// (resolveCollectionForRead) so the alias/catalog view is fresh, then routes the
// catalog read to the owning shard leader (+ shard readIndex barrier) so a
// just-created / just-reconfigured collection's config is visible. rc==0 is the
// legacy plain-Call path. The named config is identical on every physical
// partition (shared NamedVectors map), so this reads the logical collection
// directly — no fan-out.
func (e *embedded) VectorNamedGetConfigExt(_ context.Context, name string, opts ReadOpts) (map[string]NamedVectorParams, error) {
	// rc==0 keeps the EXACT legacy path: the logical name passed raw to e.Call (no
	// alias resolution, no barrier) — byte/behaviour-identical to the old method.
	// Only a Linearizable read runs the meta-catalog barrier + leader routing.
	if opts.ReadConsistency >= ops.ConsistencyLeaderOnly {
		resolved, err := e.resolveCollectionForRead(name, opts.ReadConsistency, time.Now().Add(metaReadIndexReadTimeout))
		if err != nil {
			return nil, err
		}
		name = resolved
	}
	body, err := e.callReadLeader("vector_named_get_config",
		ops.EncodeNamedNameArgsOpts(name, opts.ReadConsistency, opts.OnPartitionUnavailable, opts.MaxStaleness),
		opts.ReadConsistency)
	if err != nil {
		return nil, err
	}
	return ops.DecodeNamedConfigResult(body)
}
