// SPDX-License-Identifier: Apache-2.0

package rostam

import (
	"context"
	"time"

	"github.com/rostamlabs/rostam/cluster"
	"github.com/rostamlabs/rostam/ops"
	"github.com/rostamlabs/rostam/vector"
)

// VectorMVQuery runs the unified Query API (vector_mv_query) against a MULTI-VECTOR
// collection — a root + N prefetch leaves where every leaf is an MV node (a MaxSim
// late-interaction lane and/or the doc-level sparse field), combined by FUSION
// (RRF/Weighted/DBSF) or RERANK (the root re-scores the union of the prefetch
// candidates). This is the MV-family mirror of (*embedded).VectorNamedQuery:
// specBytes is the marshaled pb.QuerySpec carried on the wire to each shard; spec is
// the decoded engine spec the coordinator needs for the fusion/rerank merge. For a
// partitioned collection (P>1) it fans vector_mv_query to every partition and merges
// per mode (mvQueryFanOut); for a single shard it runs the op on the read leader and
// routes the one shard's mode-tagged QueryResult through the SAME fusion/rerank merge
// (FUSION fills Lanes, never a flat Fused — reading qr.Fused directly would drop every
// FUSION result). A Linearizable read arms the meta + per-shard barriers (rc rides
// every per-partition arg).
//
// MV-SPECIFIC ORIENTATION: both MV lanes (MaxSim relevance + doc-sparse dot product)
// are SCORE-descending (unlike dense lane0, which is distance-ascending). spec's
// per-leaf ScoreDesc flags (all true for MV) drive the shared orientation-aware
// fusionMergeFanOut / rerankMergeFanOut — the SAME merges v1/v2 use — to truncate
// each unioned lane score-descending and fold via FuseScoreLanes. No new
// fuse logic here.
func (e *embedded) VectorMVQuery(_ context.Context, name string, specBytes []byte, spec vector.QuerySpec, opts ReadOpts) ([]VectorResult, FanMeta, error) {
	name, err := e.resolveCollectionForRead(name, opts.ReadConsistency, time.Now().Add(metaReadIndexReadTimeout))
	if err != nil {
		return nil, FanMeta{}, err
	}
	if P, gen, ok := e.catalog.PartitionsGen(name); ok && P > 1 {
		res, fr, err := e.mvQueryFanOut(name, P, gen, specBytes, spec, opts.ReadConsistency, opts.OnPartitionUnavailable, opts.MaxStaleness)
		return res, FanMeta{Degraded: fr.Degraded, Missing: fr.Missing}, err
	}
	body, err := e.callReadLeader("vector_mv_query",
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
		// A nested MULTI-lane FUSION spec ships UNFUSED tree-lanes (the MV handler emits
		// them via SpecHasNestedFusion); the single shard runs the SAME recursive tree
		// fold the P>1 coordinator uses over its one lane list ⇒ P==1 is exactly the
		// single-node engine fold at every FUSION node. MV has no recommend/discover (the
		// engine fails them loud), so there is no single-shard rewrite/prune. Mirrors the
		// dense VectorQuery.
		if vector.SpecHasNestedFusion(spec) {
			res, terr := treeFusionMergeFanOut(single, spec, queryK(spec))
			if terr != nil {
				return nil, FanMeta{}, terr
			}
			return res, FanMeta{}, nil
		}
		return fusionMergeFanOut(single, spec, queryK(spec)), FanMeta{}, nil
	}
}

// mvQueryFanOut runs vector_mv_query on every physical partition and reproduces the
// single-partition (*MultiVectorIndex).Query result EXACTLY for both root modes — the
// MV-collection generalization of queryFanOut (v1 dense) / namedQueryFanOut (v2
// named). It REUSES the v1 fusionMergeFanOut / rerankMergeFanOut merges VERBATIM: MV
// point ids are partition-disjoint (PartitionOf(id,P) maps a doc to ONE partition),
// so FUSION (union lane[i] + truncate-per-lane + fold once) and RERANK (merge
// per-partition top-Ks) are partition-exact exactly as in v1/v2.
//
// MV ORIENTATION: every MV lane is score-descending (ScoreDesc=true on each leaf), so
// fusionMergeFanOut truncates each unioned lane score-descending and foldUnionedLanes
// folds from lane0 via FuseScoreLanes (matching mvHybridFanOut + the single-node
// MVHybrid). The merges read the orientation off spec's per-leaf ScoreDesc flags
// — no MV-specific fold lives here.
//
// rc/opa/bound ride EVERY per-partition arg (via EncodeQueryArgs' opts trailer) so a
// Linearizable query arms each partition leader's readIndex barrier; they are never
// silently dropped on the fan-out path. Mirrors namedQueryFanOut / mvHybridFanOut's
// FanMeta / degradation handling.
func (e *embedded) mvQueryFanOut(name string, P int, gen uint32, specBytes []byte, spec vector.QuerySpec, rc, opa uint8, bound uint64) ([]VectorResult, cluster.FanResult, error) {
	k := queryK(spec)

	a := cluster.FanArgs{
		Collection:    name,
		P:             P,
		Generation:    gen,
		K:             k,
		Op:            "vector_mv_query",
		Consistency:   cluster.Consistency(rc),
		OnUnavailable: cluster.OnUnavailable(opa),
		Encode: func(physCol string) []byte {
			// specBytes is reused verbatim for every partition (the MaxSim token
			// matrices + doc-sparse queries + per-leaf filters travel inside it); only
			// the collection name in the header changes. The opts trailer carries
			// rc/opa/bound to each shard.
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
		// tree-lanes (the MV handler emits them via SpecHasNestedFusion); re-walk the
		// spec tree and fold each FUSION node over the global union (P>1==P1 EXACT,
		// score-desc MV lanes via the orientation-aware fold). A flat / nested-single-lane
		// / nested-RERANK FUSION spec takes the unchanged top-level fold. Mirrors the
		// dense queryFanOut / namedQueryFanOut.
		if vector.SpecHasNestedFusion(spec) {
			res, terr := treeFusionMergeFanOut(parts, spec, k)
			return res, fr, terr
		}
		return fusionMergeFanOut(parts, spec, k), fr, nil
	}
}
