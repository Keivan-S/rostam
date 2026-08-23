// Command _oracle dumps golden hex for Rostam's op arg encoders, so the Python
// native client can be differential-tested byte-for-byte against the Go
// reference. It is a test oracle, not shipped: `go run` it from the repo root.
//
//	go run ./clients/python/tests/_oracle > clients/python/tests/_oracle/golden.txt
//
// Each line is  name<TAB>hex.  The Python test recomputes each and asserts equal.
package main

import (
	"encoding/hex"
	"fmt"
	"time"

	"github.com/rostamlabs/rostam/sdk/wire"
	"github.com/rostamlabs/rostam/vector"
)

func emit(name string, b []byte) { fmt.Printf("%s\t%s\n", name, hex.EncodeToString(b)) }

// mustSpec marshals an engine vector.QuerySpec into the pb.QuerySpec bytes the
// vector_query op carries (proto.Marshal). Panics on an invalid spec — the oracle
// only feeds it well-formed specs.
func mustSpec(spec vector.QuerySpec) []byte {
	b, err := wire.MarshalEngineQuerySpec(spec)
	if err != nil {
		panic(err)
	}
	return b
}

// recommendSpec builds the QuerySpec exactly the way the Go client's
// (*Collection).Recommend does (rostam-ntvc client/vector_recommend.go): a single
// LeafRecommend prefetch lane under ModeFusion, no Root, spec.K == leaf.K.
func recommendSpec(positive, negative []uint64, k int, filter vector.Filter, strategy vector.RecommendStrategy) vector.QuerySpec {
	leaf := vector.QueryLeaf{
		Kind:      vector.LeafRecommend,
		Positive:  positive,
		Negative:  negative,
		Strategy:  strategy,
		ScoreDesc: strategy == vector.RecommendBestScore,
		K:         k,
		Filter:    filter,
	}
	return vector.QuerySpec{
		Mode:     vector.ModeFusion,
		K:        k,
		Prefetch: []vector.QuerySource{vector.LeafSource(leaf)},
	}
}

func main() {
	// ---- create_collection: a matrix that exercises the config trailer -------
	base := vector.Config{Dim: 8, Metric: vector.Cosine, M: 16, EfConstruction: 200, EfSearch: 64}
	emit("create/basic", wire.EncodeCreateCollectionArgs("docs", base))

	c := base
	c.Metric = vector.L2
	c.Seed = 42
	emit("create/l2_seed", wire.EncodeCreateCollectionArgs("docs", c))

	c = base
	c.Quant = vector.QuantSQ8
	emit("create/sq8", wire.EncodeCreateCollectionArgs("c-sq8", c))

	c = base
	c.Quant = vector.QuantBQ1
	emit("create/bq1", wire.EncodeCreateCollectionArgs("c", c))

	c = base
	c.Persistent = true
	c.RescoreFactor = 3
	emit("create/persistent_rescore", wire.EncodeCreateCollectionArgs("p", c))

	c = base
	c.Partitions = 4
	emit("create/partitions", wire.EncodeCreateCollectionArgs("part", c))

	c = base
	c.ExtendCandidates = true
	c.ExtendCandidatesMax = 200
	c.Level0FullDegree = true
	c.QuantizedBuild = true
	emit("create/hnsw_levers", wire.EncodeCreateCollectionArgs("lev", c))

	c = base
	c.IndexType = vector.IndexIVF
	c.IVFNlist = 64
	c.IVFNprobe = 8
	emit("create/ivf", wire.EncodeCreateCollectionArgs("ivf", c))

	c = base
	c.IndexType = vector.IndexVamana
	c.VamanaR = 64
	c.VamanaL = 100
	c.VamanaAlpha = 1.2
	emit("create/vamana", wire.EncodeCreateCollectionArgs("vam", c))

	c = base
	c.FullText = &vector.FullTextConfig{Analyzer: "english", K1: 1.2, B: 0.75}
	emit("create/fulltext", wire.EncodeCreateCollectionArgs("ft", c))

	c = base
	c.Dim = 128
	c.M = 32
	c.EfConstruction = 400
	c.EfSearch = 128
	emit("create/tuned", wire.EncodeCreateCollectionArgs("longname-collection", c))

	c = base
	c.AnisotropicEta = 0.2
	emit("create/aniso", wire.EncodeCreateCollectionArgs("an", c))

	c = base
	c.SOAR = true
	c.SOARLambda = 1.5
	emit("create/soar", wire.EncodeCreateCollectionArgs("so", c))

	c = base
	c.Quant = vector.QuantSQ
	c.SQBits = 6
	emit("create/sqbits", wire.EncodeCreateCollectionArgs("sq", c))

	c = base
	c.IVFTrainThreshold = 10000
	c.IndexType = vector.IndexIVF
	emit("create/ivf_threshold", wire.EncodeCreateCollectionArgs("ivt", c))

	// ---- data-plane ops ------------------------------------------------------
	vec := []float32{0.1, 0.2, 0.3, 0.4}
	emit("insert/plain", wire.EncodeVectorInsertArgs("docs", 1, vec))
	emit("insert/ttl_meta", wire.EncodeVectorInsertArgsExt("docs", 7, vec, 5*time.Minute,
		vector.Metadata{"tenant": vector.NewString("acme"), "year": vector.NewInt(2020)},
		vector.SparseVector{}))
	emit("insert/sparse", wire.EncodeVectorInsertArgsExt("docs", 9, vec, 0, nil,
		vector.SparseVector{Indices: []uint32{3, 17}, Values: []float32{0.8, 0.4}}))

	emit("upsert/plain", wire.EncodeVectorUpsertArgs("docs", 1, vec, "hello", 0, nil, vector.SparseVector{}))
	emit("upsert/content_meta", wire.EncodeVectorUpsertArgs("docs", 2, vec, "world", 0,
		vector.Metadata{"tenant": vector.NewString("acme")}, vector.SparseVector{}))

	emit("search/plain", wire.EncodeVectorSearchArgs("docs", 10, vec))
	emit("search/filter", wire.EncodeVectorSearchArgsExt("docs", 5, vec,
		vector.Filter{Op: vector.FilterEq, Field: "tenant", Value: vector.NewString("acme")}))

	emit("delete", wire.EncodeVectorDeleteArgs("docs", 42))
	emit("exists", wire.EncodeExistsArgs("docs", 42))
	emit("drop_collection/plain", wire.EncodeDropCollectionArgs("docs"))
	emit("drop_collection/longname", wire.EncodeDropCollectionArgs("longname-collection"))
	emit("get/plain", wire.EncodeVectorGetArgs("docs", 1, 0))
	emit("get/withvec_payload", wire.EncodeVectorGetArgs("docs", 1, 0x03))

	// ---- Phase A: get_batch / scroll / search_docs / search_groups /
	//      hybrid_search / hybrid_text ------------------------------------------
	tenantFilter := vector.Filter{Op: vector.FilterEq, Field: "tenant", Value: vector.NewString("acme")}

	// get_batch
	emit("get_batch/empty", wire.EncodeVectorGetBatchArgs("docs", nil, 0))
	emit("get_batch/plain", wire.EncodeVectorGetBatchArgs("docs", []uint64{1, 2, 3}, 0))
	emit("get_batch/withvec_payload", wire.EncodeVectorGetBatchArgs("docs", []uint64{7, 9999999999}, wire.GetFlagsBoth))

	// search / search_docs (same encoder — EncodeVectorSearchArgsOpts)
	emit("search/opts_plain", wire.EncodeVectorSearchArgsOpts("docs", 10, vec, vector.Filter{}, 0, 0, 0))
	emit("search/opts_filter_leader", wire.EncodeVectorSearchArgsOpts("docs", 5, vec, tenantFilter, wire.ConsistencyLeaderOnly, 1, 0))
	emit("search/opts_bounded", wire.EncodeVectorSearchArgsOpts("docs", 5, vec, vector.Filter{}, wire.ConsistencyBoundedStaleness, 0, 12345))

	// search_groups
	emit("group/plain", wire.EncodeGroupSearchArgsOpts("docs", 5, vec, vector.GroupOpts{GroupBy: "doc_id", GroupSize: 2, FetchK: 50}, 0, 0, 0))
	emit("group/filter_opts", wire.EncodeGroupSearchArgsOpts("docs", 5, vec, vector.GroupOpts{GroupBy: "doc_id", GroupSize: 3, FetchK: 100, Filter: tenantFilter}, wire.ConsistencyLeaderOnly, 1, 0))
	emit("group/bounded", wire.EncodeGroupSearchArgsOpts("docs", 5, vec, vector.GroupOpts{GroupBy: "cat"}, wire.ConsistencyBoundedStaleness, 0, 777))

	// hybrid_search
	emit("hybrid/plain", wire.EncodeHybridSearchArgsOpts("docs", vec, 10, vector.SparseVector{}, vector.HybridOpts{}, 0, 0, 0))
	emit("hybrid/sparse", wire.EncodeHybridSearchArgsOpts("docs", vec, 10,
		vector.SparseVector{Indices: []uint32{3, 17}, Values: []float32{0.8, 0.4}},
		vector.HybridOpts{Method: vector.FusionWeighted, Alpha: 0.6, RRFK: 60, DenseK: 50, SparseK: 50}, 0, 0, 0))
	emit("hybrid/filter_opts", wire.EncodeHybridSearchArgsOpts("docs", vec, 10, vector.SparseVector{},
		vector.HybridOpts{Filter: tenantFilter}, wire.ConsistencyLeaderOnly, 1, 0))
	emit("hybrid/bounded", wire.EncodeHybridSearchArgsOpts("docs", vec, 10,
		vector.SparseVector{Indices: []uint32{1}, Values: []float32{1.0}},
		vector.HybridOpts{Method: vector.FusionDBSF}, wire.ConsistencyBoundedStaleness, 0, 999))

	// hybrid_text (globalIDF true/false; g always nil per Phase A scope)
	emit("hybrid_text/plain", wire.EncodeHybridTextArgsGlobal("docs", vec, "hello world", 10, vector.HybridOpts{}, 0, 0, 0, false, nil))
	emit("hybrid_text/filter_opts_globalidf", wire.EncodeHybridTextArgsGlobal("docs", vec, "quick fox", 5,
		vector.HybridOpts{Filter: tenantFilter, Method: vector.FusionWeighted, Alpha: 0.7}, wire.ConsistencyLeaderOnly, 1, 0, true, nil))
	emit("hybrid_text/bounded", wire.EncodeHybridTextArgsGlobal("docs", vec, "", 5, vector.HybridOpts{}, wire.ConsistencyBoundedStaleness, 0, 555, false, nil))

	// scroll
	emit("scroll/plain", wire.EncodeScrollArgsOrderBounded("docs", vector.Filter{}, 50, 0, 0, 0, false, nil, 0))
	emit("scroll/filter", wire.EncodeScrollArgsOrderBounded("docs", tenantFilter, 20, 0, 0, 0, false, nil, 0))
	emit("scroll/cursor", wire.EncodeScrollArgsOrderBounded("docs", vector.Filter{}, 20, 0, 0, 42, true, nil, 0))
	emit("scroll/bounded_opts", wire.EncodeScrollArgsOrderBounded("docs", vector.Filter{}, 20, wire.ConsistencyBoundedStaleness, 1, 0, false, nil, 4242))
	emit("scroll/order_numeric", wire.EncodeScrollArgsOrderBounded("docs", vector.Filter{}, 20, 0, 0, 0, false,
		&wire.ScrollOrder{Key: "score", Desc: true, HasStart: true, StartFrom: 1.5}, 0))
	emit("scroll/order_datetime_resume", wire.EncodeScrollArgsOrderBounded("docs", vector.Filter{}, 20, 0, 0, 99, true,
		&wire.ScrollOrder{Key: "created_at", IsDatetime: true, Kind: vector.OrderDatetime, HasResume: true, ResumeKey: 12345.0}, 0))
	emit("scroll/order_string_resume", wire.EncodeScrollArgsOrderBounded("docs", vector.Filter{}, 20, 0, 0, 0, false,
		&wire.ScrollOrder{Key: "title", Kind: vector.OrderString, HasResumeStr: true, ResumeStr: "foo"}, 0))
	emit("scroll/order_multikey", wire.EncodeScrollArgsOrderBounded("docs", vector.Filter{}, 20, wire.ConsistencyLeaderOnly, 1, 5, true,
		&wire.ScrollOrder{
			Key:  "score",
			Desc: true,
			Tail: []wire.ScrollOrderKey{
				{Key: "title", Kind: vector.OrderString},
				{Key: "created_at", IsDatetime: true, Kind: vector.OrderDatetime, Desc: true},
			},
			HasResumeKeys: true,
			ResumeKeys: []wire.ScrollOrderVal{
				{Kind: vector.OrderNumeric, Num: 3.5},
				{Kind: vector.OrderString, Str: "bar"},
				{Kind: vector.OrderDatetime, Num: 999.0},
			},
		}, 0))

	// ---- Phase B: vector_query recommend specs -------------------------------
	// The QuerySpec is built exactly as the Go client's Recommend does; each case
	// emits BOTH the raw marshaled pb.QuerySpec blob (queryspec/*) and the full
	// EncodeQueryArgs op frame (query/*). Golden byte-matches the hand-rolled
	// Python protobuf against Go's proto.Marshal.
	recPos := recommendSpec([]uint64{1, 2, 3}, nil, 10, vector.Filter{}, vector.RecommendAverageVector)
	emit("queryspec/recommend_pos", mustSpec(recPos))
	emit("query/recommend_pos", wire.EncodeQueryArgs("docs", mustSpec(recPos), 0, 0, 0))

	recPosNeg := recommendSpec([]uint64{1, 2}, []uint64{9}, 5, vector.Filter{}, vector.RecommendAverageVector)
	emit("queryspec/recommend_pos_neg", mustSpec(recPosNeg))
	emit("query/recommend_pos_neg", wire.EncodeQueryArgs("docs", mustSpec(recPosNeg), 0, 0, 0))

	recFilter := recommendSpec([]uint64{1}, nil, 5, tenantFilter, vector.RecommendAverageVector)
	emit("queryspec/recommend_filter", mustSpec(recFilter))
	emit("query/recommend_filter", wire.EncodeQueryArgs("docs", mustSpec(recFilter), 0, 0, 0))

	recBest := recommendSpec([]uint64{1, 2}, nil, 5, vector.Filter{}, vector.RecommendBestScore)
	emit("queryspec/recommend_best_score", mustSpec(recBest))
	emit("query/recommend_best_score", wire.EncodeQueryArgs("docs", mustSpec(recBest), 0, 0, 0))

	// op-frame with the bounded-staleness read-opts trailer (exercises the
	// [marker|stalenessbit][rc][opa][bound:u64] tail EncodeQueryArgs appends).
	emit("query/recommend_bounded", wire.EncodeQueryArgs("docs", mustSpec(recPos), wire.ConsistencyBoundedStaleness, 0, 555))
}
