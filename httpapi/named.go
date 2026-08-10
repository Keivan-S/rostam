// SPDX-License-Identifier: Apache-2.0

package httpapi

import (
	"net/http"
	"strconv"
	"strings"
	"time"

	"github.com/rostamlabs/rostam/ops"
	"github.com/rostamlabs/rostam/vector"
)

// REST handlers for named-vector (Qdrant-style per-point multi-vector-space)
// collections, under /v1/named. A collection holds a MAP of named dense vector
// spaces, each its own HNSW index, sharing ONE per-point payload + id
// namespace. Insert provides a map of named vectors; search selects which named
// space to query. Filters are validate-at-edge compiled (a bad filter is a 400
// BEFORE dispatch, like the dense/rich/geo path — a malformed filter must never
// reach the engine, notably never trigger an over-broad delete/scroll). This
// mirrors the multivector (/v1/multivector) transport surface: a separate route
// group dispatching the vector_named_* op family, leaving the dense + MV
// transports byte-for-byte untouched.

// namedCreateReq is the create-collection body: a map of named space -> per-
// space index params (dim/metric/m/ef_*/quant/...). At least one space is
// required; engine-side validation rejects an empty map and bad params.
type namedCreateReq struct {
	NamedVectors map[string]vector.NamedVectorParams `json:"named_vectors"`
	// Partitions is the collection-level partition count (omit or 0/1 =
	// single-partition; >1 splits the collection across shards via cross-shard
	// fan-out, like the dense/MV partitions field).
	Partitions int `json:"partitions,omitempty"`
}

func (a *api) namedCreate(w http.ResponseWriter, r *http.Request) {
	name := r.PathValue("name")
	if strings.ContainsAny(name, "#@") {
		writeError(w, http.StatusBadRequest,
			"vector: collection name "+strconv.Quote(name)+" must not contain reserved characters '#' or '@'")
		return
	}
	var req namedCreateReq
	if !decodeBody(w, r, &req) {
		return
	}
	if req.Partitions < 0 {
		writeError(w, http.StatusBadRequest, "partitions must be >= 0")
		return
	}
	if _, ok := a.call(w, r, "vector_named_create_collection", ops.EncodeNamedCreateArgs(name, req.NamedVectors, req.Partitions)); !ok {
		return
	}
	writeJSON(w, http.StatusCreated, map[string]string{"name": name})
}

func (a *api) namedDrop(w http.ResponseWriter, r *http.Request) {
	name := r.PathValue("name")
	if _, ok := a.call(w, r, "vector_named_drop_collection", ops.EncodeNamedNameArgs(name)); !ok {
		return
	}
	writeJSON(w, http.StatusOK, map[string]string{"dropped": name})
}

// namedPointReq is the upsert body: the per-point map of named-space -> vector,
// the shared per-point payload, and an optional TTL. A provided name must be a
// configured space and each vector's length must equal that space's dim (the
// engine fails loud → 400); a point MAY omit configured spaces.
type namedPointReq struct {
	ID      uint64               `json:"id"`
	Vectors map[string][]float32 `json:"vectors"`
	// SparseVectors carries per-space SPARSE values: space -> {indices, values}.
	// A space entry is dense XOR sparse — a dense value rides "vectors", a sparse
	// value rides here; the engine validates the modality against the configured
	// space (400 on a mismatch). Absent/empty = a dense-only point (byte-identical
	// wire).
	SparseVectors map[string]sparseVecReq `json:"sparse_vectors"`
	Metadata      vector.Metadata         `json:"metadata"`
	TTLMs         int64                   `json:"ttl_ms"`
	// ExpectedVersion is the optimistic-CAS precondition: the per-point version the
	// caller expects the point to currently have (0 = expect-absent / insert-if-absent).
	// nil (the default) = an unconditional upsert. A mismatch returns 409 Conflict.
	ExpectedVersion *uint64 `json:"expected_version"`
	// KeyTTLMs is an OPTIONAL per-key payload TTL map (payload key -> RELATIVE ms).
	// At insert/upsert the engine computes the absolute deadline now+ttl per key
	// (mirroring set_payload's key_ttl_ms) and lazily drops the key once it passes,
	// while the point lives on. Absent/empty = no per-key TTL (byte-identical wire).
	KeyTTLMs map[string]int64 `json:"key_ttl_ms"`
	writeConsistency
}

// sparseVecReq is one space's sparse value in a named upsert / the sparse query in
// a named sparse search: parallel indices (strictly ascending) + values arrays.
type sparseVecReq struct {
	Indices []uint32  `json:"indices"`
	Values  []float32 `json:"values"`
}

// toSparseMap converts the per-space JSON sparse-vector map to the engine's
// map[string]*vector.SparseVector (nil when empty, the dense-only path).
func toSparseMap(in map[string]sparseVecReq) map[string]*vector.SparseVector {
	if len(in) == 0 {
		return nil
	}
	out := make(map[string]*vector.SparseVector, len(in))
	for name, sv := range in {
		out[name] = &vector.SparseVector{Indices: sv.Indices, Values: sv.Values}
	}
	return out
}

func (a *api) namedUpsert(w http.ResponseWriter, r *http.Request) {
	name := r.PathValue("name")
	var req namedPointReq
	if !decodeBody(w, r, &req) {
		return
	}
	if !req.validate(w) {
		return
	}
	ttl := time.Duration(req.TTLMs) * time.Millisecond
	exp, hasExp := uint64(0), false
	if req.ExpectedVersion != nil {
		exp, hasExp = *req.ExpectedVersion, true
	}
	args := ops.EncodeNamedInsertArgsSparseCASKeyTTL(name, req.ID, req.Vectors, toSparseMap(req.SparseVectors), req.Metadata, ttl, exp, hasExp, req.KeyTTLMs)
	if _, ok := a.callWrite(w, r, "vector_named_insert", args, req.WriteConsistencyFactor, req.wait()); !ok {
		return
	}
	writeJSON(w, http.StatusOK, map[string]uint64{"id": req.ID})
}

// namedSearchReq selects which named space to query (vector_name) plus the
// query/k and an optional shared-payload filter.
type namedSearchReq struct {
	VectorName             string        `json:"vector_name"`
	Query                  []float32     `json:"query"`
	K                      int           `json:"k"`
	Filter                 vector.Filter `json:"filter"`
	ReadConsistency        uint8         `json:"read_consistency"`
	OnPartitionUnavailable uint8         `json:"on_partition_unavailable"`
	MaxStaleness           uint64        `json:"max_staleness"` // bound for rc==3 (bounded-staleness)
}

func (a *api) namedSearch(w http.ResponseWriter, r *http.Request) {
	name := r.PathValue("name")
	var req namedSearchReq
	if !decodeBody(w, r, &req) {
		return
	}
	if !validConsistency(w, req.ReadConsistency, req.OnPartitionUnavailable) {
		return
	}
	if !validFilter(w, req.Filter) {
		return
	}
	args := ops.EncodeNamedSearchArgsOpts(name, req.VectorName, req.Query, req.K, req.Filter, req.ReadConsistency, req.OnPartitionUnavailable, req.MaxStaleness)
	body, ok := a.call(w, r, "vector_named_search", args)
	if !ok {
		return
	}
	results, err := ops.DecodeVectorSearchResults(body)
	if err != nil {
		writeInternalError(w, r.URL.Path, err)
		return
	}
	writeJSON(w, http.StatusOK, map[string]any{"results": results})
}

// namedSparseSearchReq selects which SPARSE named space to query (vector_name)
// plus the sparse query {indices, values}, k, and an optional shared-payload filter.
type namedSparseSearchReq struct {
	VectorName             string        `json:"vector_name"`
	Query                  sparseVecReq  `json:"query"`
	K                      int           `json:"k"`
	Filter                 vector.Filter `json:"filter"`
	ReadConsistency        uint8         `json:"read_consistency"`
	OnPartitionUnavailable uint8         `json:"on_partition_unavailable"`
	MaxStaleness           uint64        `json:"max_staleness"` // bound for rc==3 (bounded-staleness)
}

// namedSparseSearch runs a sparse-dot-product top-k against a SPARSE named space.
// A dense space (or an unknown space) is a 400/relevant error from the engine.
// Results are ranked DESCENDING by score (the sparse dot product).
func (a *api) namedSparseSearch(w http.ResponseWriter, r *http.Request) {
	name := r.PathValue("name")
	var req namedSparseSearchReq
	if !decodeBody(w, r, &req) {
		return
	}
	if !validConsistency(w, req.ReadConsistency, req.OnPartitionUnavailable) {
		return
	}
	if !validFilter(w, req.Filter) {
		return
	}
	query := vector.SparseVector{Indices: req.Query.Indices, Values: req.Query.Values}
	args := ops.EncodeNamedSparseSearchArgsOpts(name, req.VectorName, query, req.K, req.Filter, req.ReadConsistency, req.OnPartitionUnavailable, req.MaxStaleness)
	body, ok := a.call(w, r, "vector_named_sparse_search", args)
	if !ok {
		return
	}
	// The sparse handler returns score-carrying hybrid results (id+distance+score).
	results, err := ops.DecodeHybridResults(body)
	if err != nil {
		writeInternalError(w, r.URL.Path, err)
		return
	}
	writeJSON(w, http.StatusOK, map[string]any{"results": results})
}

// namedHybridReq fuses a DENSE named space (dense_space + dense) and a SPARSE named
// space (sparse_space + sparse {indices,values}) of a named collection. method ""/
// "rrf"|"weighted"|"dbsf"; alpha is the dense weight for weighted/dbsf fusion. filter applies to
// BOTH lanes. An empty dense query degrades to the sparse lane only; an absent/empty
// sparse degrades to the dense lane only.
type namedHybridReq struct {
	DenseSpace             string               `json:"dense_space"`
	Dense                  []float32            `json:"dense"`
	SparseSpace            string               `json:"sparse_space"`
	Sparse                 *vector.SparseVector `json:"sparse"`
	K                      int                  `json:"k"`
	Method                 string               `json:"method"` // "" / "rrf" | "weighted" | "dbsf"
	Alpha                  float64              `json:"alpha"`
	RRFK                   int                  `json:"rrf_k"`
	DenseK                 int                  `json:"dense_k"`
	SparseK                int                  `json:"sparse_k"`
	Filter                 vector.Filter        `json:"filter"`
	ReadConsistency        uint8                `json:"read_consistency"`
	OnPartitionUnavailable uint8                `json:"on_partition_unavailable"`
	MaxStaleness           uint64               `json:"max_staleness"` // bound for rc==3 (bounded-staleness)
}

// namedHybrid fuses a dense + a sparse named space into the top-k (cross-space
// hybrid). An unknown space / modality mismatch surfaces from the engine as 400.
// Results are ranked by the fused score.
func (a *api) namedHybrid(w http.ResponseWriter, r *http.Request) {
	name := r.PathValue("name")
	var req namedHybridReq
	if !decodeBody(w, r, &req) {
		return
	}
	if !validConsistency(w, req.ReadConsistency, req.OnPartitionUnavailable) {
		return
	}
	if !validFilter(w, req.Filter) {
		return
	}
	var method vector.FusionMethod
	switch req.Method {
	case "", "rrf":
		method = vector.FusionRRF
	case "weighted":
		method = vector.FusionWeighted
	case "dbsf":
		method = vector.FusionDBSF
	default:
		writeError(w, http.StatusBadRequest, "unknown fusion method "+strconv.Quote(req.Method))
		return
	}
	var sparse vector.SparseVector
	if req.Sparse != nil {
		sparse = *req.Sparse
	}
	opts := vector.HybridOpts{
		Filter: req.Filter, Method: method, Alpha: req.Alpha,
		RRFK: req.RRFK, DenseK: req.DenseK, SparseK: req.SparseK,
	}
	args := ops.EncodeNamedHybridArgs(name, req.DenseSpace, req.Dense, req.SparseSpace, sparse, req.K, opts, req.ReadConsistency, req.OnPartitionUnavailable, req.MaxStaleness)
	body, ok := a.call(w, r, "vector_named_hybrid_search", args)
	if !ok {
		return
	}
	// The fused handler returns score-carrying hybrid results (id+distance+score).
	results, err := ops.DecodeHybridResults(body)
	if err != nil {
		writeInternalError(w, r.URL.Path, err)
		return
	}
	writeJSON(w, http.StatusOK, map[string]any{"results": results})
}

// namedQueryLeafReq is one JSON query node in a /named/{name}/query body: a named
// vector space plus exactly one of a dense vector or a sparse vector. space is
// REQUIRED (a named leaf must target a configured space — a missing space is a 400
// at the edge, never silently routed). k is the leaf's own top-k; lane_k is the
// per-lane candidate pool (0 = engine default); filter is the optional per-leaf
// metadata predicate (same JSON contract as the named hybrid filter).
type namedQueryLeafReq struct {
	Space     string               `json:"space"`
	Dense     []float32            `json:"dense"`
	Sparse    *vector.SparseVector `json:"sparse"`
	Recommend *recommendLeafReq    `json:"recommend"`
	Discover  *discoverLeafReq     `json:"discover"`
	K         int                  `json:"k"`
	LaneK     int                  `json:"lane_k"`
	Filter    vector.Filter        `json:"filter"`
}

// toLeaf converts the JSON named leaf into an engine vector.QueryLeaf carrying the
// named Space. A leaf with a recommend payload is LeafRecommend (AVERAGE → the
// coordinator derives it to a dense leaf IN-SPACE; BEST_SCORE → a custom per-space
// scorer); a leaf with a discover payload is LeafDiscover (a per-space context
// scorer); a leaf with a sparse payload is LeafSparse; otherwise LeafDense (an empty
// dense leaf is the "no root" marker for FUSION). Every kind PRESERVES the named
// Space so the per-space pre-pass/exec runs against the right space. The Space-bearing
// leaf round-trips through the shared QuerySpec proto codec (MarshalEngineQuerySpec).
func (l namedQueryLeafReq) toLeaf() vector.QueryLeaf {
	if l.Recommend != nil {
		// BEST_SCORE is a custom per-candidate scorer (score-descending, like Discover);
		// AVERAGE_VECTOR (the default) the coordinator derives to a dense leaf in-space.
		strategy := parseRecommendStrategy(l.Recommend.Strategy)
		return vector.QueryLeaf{
			Kind:      vector.LeafRecommend,
			Space:     l.Space,
			Positive:  l.Recommend.Positive,
			Negative:  l.Recommend.Negative,
			Strategy:  strategy,
			ScoreDesc: strategy == vector.RecommendBestScore,
			K:         l.K, LaneK: l.LaneK, Filter: l.Filter,
		}
	}
	if l.Discover != nil {
		leaf := vector.QueryLeaf{
			Kind:      vector.LeafDiscover,
			Space:     l.Space,
			ScoreDesc: true, // discover ranks score-descending (like MV MaxSim)
			K:         l.K, LaneK: l.LaneK, Filter: l.Filter,
		}
		if l.Discover.TargetID != nil {
			leaf.DiscoverTargetID = []uint64{*l.Discover.TargetID}
		} else if len(l.Discover.Target) > 0 {
			leaf.DiscoverTarget = l.Discover.Target
		}
		// Each pair is wholly the vector form or wholly the id form — validDiscover
		// rejected every other shape at the edge, so there is no half-specified
		// pair left to guess at here.
		for _, p := range l.Discover.Context {
			if p.isVecForm() {
				leaf.DiscoverContext = append(leaf.DiscoverContext, vector.DiscoverPair{Pos: p.PositiveVec, Neg: p.NegativeVec})
			} else {
				leaf.DiscoverContextIDs = append(leaf.DiscoverContextIDs, vector.ContextPair{Positive: *p.Positive, Negative: *p.Negative})
			}
		}
		return leaf
	}
	if l.Sparse != nil && !l.Sparse.IsZero() {
		return vector.QueryLeaf{
			Kind:   vector.LeafSparse,
			Space:  l.Space,
			Sparse: *l.Sparse,
			K:      l.K, LaneK: l.LaneK, Filter: l.Filter,
		}
	}
	return vector.QueryLeaf{
		Kind:  vector.LeafDense,
		Space: l.Space,
		Dense: l.Dense,
		K:     l.K, LaneK: l.LaneK, Filter: l.Filter,
	}
}

// hasPayload reports whether a named leaf carries an actual query payload — used to
// decide whether a FUSION spec's empty root needs a space (it does not). A
// recommend/discover leaf carries a payload (its example/context ids).
func (l namedQueryLeafReq) hasPayload() bool {
	if l.Recommend != nil || l.Discover != nil {
		return true
	}
	if l.Sparse != nil && !l.Sparse.IsZero() {
		return true
	}
	return len(l.Dense) > 0
}

// namedQueryReq is the named-collection Query API HTTP body: a root leaf (RERANK)
// + N prefetch leaves — each targeting a named SPACE — a combine mode
// ("fusion"|"rerank"), the fusion config, the final top-k, and the standard
// read-consistency knobs. Mirrors queryReq (the dense Query API) plus a per-leaf
// space.
type namedQueryReq struct {
	Root     *namedQueryLeafReq  `json:"root"`
	Prefetch []namedQueryLeafReq `json:"prefetch"`
	Mode     string              `json:"mode"`   // "fusion" (default) | "rerank"
	Method   string              `json:"method"` // "" / "rrf" | "weighted" | "dbsf"
	Alpha    float64             `json:"alpha"`
	RRFK     int                 `json:"rrf_k"`
	K        int                 `json:"k"`

	ReadConsistency        uint8  `json:"read_consistency"`
	OnPartitionUnavailable uint8  `json:"on_partition_unavailable"`
	MaxStaleness           uint64 `json:"max_staleness"` // bound for rc==3 (bounded-staleness)
}

// namedQuery runs the unified Query API over a NAMED collection: each leaf targets
// a named vector space, so the N prefetch lanes fuse across >2 named spaces
// (FUSION) or the root re-scores the prefetch candidate union in its space
// (RERANK). Mirrors the dense /query handler plus the per-leaf space requirement.
// An unknown mode/method, a missing space, or a bad filter is a 400 at the edge;
// degraded/missing flow through DecodeQueryResultDegraded.
func (a *api) namedQuery(w http.ResponseWriter, r *http.Request) {
	name := r.PathValue("name")
	var req namedQueryReq
	if !decodeBody(w, r, &req) {
		return
	}
	if !validConsistency(w, req.ReadConsistency, req.OnPartitionUnavailable) {
		return
	}
	var mode vector.QueryMode
	switch req.Mode {
	case "", "fusion":
		mode = vector.ModeFusion
	case "rerank":
		mode = vector.ModeRerank
	default:
		writeError(w, http.StatusBadRequest, "unknown query mode "+strconv.Quote(req.Mode))
		return
	}
	var method vector.FusionMethod
	switch req.Method {
	case "", "rrf":
		method = vector.FusionRRF
	case "weighted":
		method = vector.FusionWeighted
	case "dbsf":
		method = vector.FusionDBSF
	default:
		writeError(w, http.StatusBadRequest, "unknown fusion method "+strconv.Quote(req.Method))
		return
	}
	spec := vector.QuerySpec{
		Mode: mode, Method: method, Alpha: req.Alpha, RRFK: req.RRFK, K: req.K,
	}
	// Every prefetch leaf MUST carry a space; the root must too when it carries a
	// payload (a FUSION spec's empty root is exempt). RERANK always consults the
	// root, so a RERANK root must carry a space.
	for i := range req.Prefetch {
		if req.Prefetch[i].Space == "" {
			writeError(w, http.StatusBadRequest, "named query: every leaf must target a space")
			return
		}
		if !validFilter(w, req.Prefetch[i].Filter) || !validDiscover(w, req.Prefetch[i].Discover) {
			return
		}
		spec.Prefetch = append(spec.Prefetch, vector.LeafSource(req.Prefetch[i].toLeaf()))
	}
	if req.Root != nil {
		if (req.Root.hasPayload() || mode == vector.ModeRerank) && req.Root.Space == "" {
			writeError(w, http.StatusBadRequest, "named query: root leaf must target a space")
			return
		}
		if !validFilter(w, req.Root.Filter) || !validDiscover(w, req.Root.Discover) {
			return
		}
		spec.Root = req.Root.toLeaf()
	} else if mode == vector.ModeRerank {
		writeError(w, http.StatusBadRequest, "named query: rerank mode requires a root leaf")
		return
	}
	specBytes, err := ops.MarshalEngineQuerySpec(spec)
	if err != nil {
		writeError(w, http.StatusBadRequest, err.Error())
		return
	}
	body, ok := a.call(w, r, "vector_named_query", ops.EncodeQueryArgs(name, specBytes, req.ReadConsistency, req.OnPartitionUnavailable, req.MaxStaleness))
	if !ok {
		return
	}
	results, degraded, missing, err := ops.DecodeQueryResultDegraded(body)
	if err != nil {
		writeInternalError(w, r.URL.Path, err)
		return
	}
	writeJSON(w, http.StatusOK, map[string]any{"results": results, "degraded": degraded, "missing": missingJSON(missing)})
}

func (a *api) namedSearchDocs(w http.ResponseWriter, r *http.Request) {
	name := r.PathValue("name")
	var req namedSearchReq
	if !decodeBody(w, r, &req) {
		return
	}
	if !validConsistency(w, req.ReadConsistency, req.OnPartitionUnavailable) {
		return
	}
	if !validFilter(w, req.Filter) {
		return
	}
	args := ops.EncodeNamedSearchArgsOpts(name, req.VectorName, req.Query, req.K, req.Filter, req.ReadConsistency, req.OnPartitionUnavailable, req.MaxStaleness)
	body, ok := a.call(w, r, "vector_named_search_docs", args)
	if !ok {
		return
	}
	docs, err := ops.DecodeVectorDocsRaw(body)
	if err != nil {
		writeInternalError(w, r.URL.Path, err)
		return
	}
	writeJSON(w, http.StatusOK, map[string]any{"documents": docs})
}

func (a *api) namedDelete(w http.ResponseWriter, r *http.Request) {
	name := r.PathValue("name")
	id, err := strconv.ParseUint(r.PathValue("id"), 10, 64)
	if err != nil {
		writeError(w, http.StatusBadRequest, "invalid id: "+err.Error())
		return
	}
	// Path-id delete carries no JSON body, so the WC knobs ride the query string.
	wc, ok := queryWC(w, r)
	if !ok {
		return
	}
	a.namedDeletePoint(w, r, name, id, wc)
}

// namedDeleteReq is the POST delete-point body (the id-carrying alternative to
// DELETE /points/{id}). Named delete is by point id (remove from every named
// sub-index + the shared payload), NOT delete-by-filter.
type namedDeleteReq struct {
	ID uint64 `json:"id"`
	writeConsistency
}

// namedDeleteByID is the POST /points/delete body-route: the WC knobs ride the
// JSON body here (namedDeleteReq.writeConsistency), whereas the path-id route
// namedDelete carries them in the query string (it has no body).
func (a *api) namedDeleteByID(w http.ResponseWriter, r *http.Request) {
	name := r.PathValue("name")
	var req namedDeleteReq
	if !decodeBody(w, r, &req) {
		return
	}
	if !req.validate(w) {
		return
	}
	a.namedDeletePoint(w, r, name, req.ID, req.writeConsistency)
}

func (a *api) namedDeletePoint(w http.ResponseWriter, r *http.Request, name string, id uint64, wc writeConsistency) {
	// Optional ?expected_version=N optimistic-CAS precondition (a mismatch → 409).
	exp, hasExp, ok := queryExpectedVersion(w, r)
	if !ok {
		return
	}
	body, ok := a.callWrite(w, r, "vector_named_delete", ops.EncodeNamedDeleteArgsCAS(name, id, exp, hasExp), wc.WriteConsistencyFactor, wc.wait())
	if !ok {
		return
	}
	writeJSON(w, http.StatusOK, map[string]bool{"deleted": len(body) > 0 && body[0] == 1})
}

type namedScrollReq struct {
	Filter                 vector.Filter `json:"filter"`
	Limit                  int           `json:"limit"`
	Cursor                 string        `json:"cursor,omitempty"`
	ReadConsistency        uint8         `json:"read_consistency"`
	OnPartitionUnavailable uint8         `json:"on_partition_unavailable"`
	MaxStaleness           uint64        `json:"max_staleness"` // bound for rc==3 (bounded-staleness)
	// OrderBy paginates by an arbitrary numeric/datetime payload field (Qdrant
	// order_by). Absent ⇒ id-ascending scroll. Reuses the dense scroll's orderByReq.
	OrderBy *orderByReq `json:"order_by,omitempty"`
}

func (a *api) namedScroll(w http.ResponseWriter, r *http.Request) {
	name := r.PathValue("name")
	var req namedScrollReq
	if !decodeBody(w, r, &req) {
		return
	}
	if !validConsistency(w, req.ReadConsistency, req.OnPartitionUnavailable) {
		return
	}
	// Validate-at-edge: a malformed filter must NEVER reach the engine (a bad
	// scroll filter would otherwise traverse with a broken predicate). Compile
	// first, 400 on error, only then dispatch.
	if !validFilter(w, req.Filter) {
		return
	}
	// Validate-at-edge: a bad order_by (empty key / malformed start_from) is a 400.
	order, err := req.OrderBy.toOrderBy()
	if err != nil {
		writeError(w, http.StatusBadRequest, err.Error())
		return
	}
	// A malformed cursor — or a cursor whose version disagrees with order_by — is a
	// client error: reject with 400 BEFORE dispatch.
	afterID, hasAfter, scrollOrder, err := scrollCursorAndOrder(req.Cursor, order)
	if err != nil {
		writeError(w, http.StatusBadRequest, err.Error())
		return
	}
	args := ops.EncodeNamedScrollArgsOrderBounded(name, req.Filter, req.Limit, afterID, hasAfter, req.ReadConsistency, req.OnPartitionUnavailable, scrollOrder, req.MaxStaleness)
	body, ok := a.call(w, r, "vector_named_scroll", args)
	if !ok {
		return
	}
	docs, _, _, nextCursor, err := ops.DecodeScrollResultRaw(body)
	if err != nil {
		writeInternalError(w, r.URL.Path, err)
		return
	}
	writeJSON(w, http.StatusOK, map[string]any{"documents": docs, "next_cursor": nextCursor})
}

// namedGet retrieves a named point by id → {found, vectors, payload, ttl_ms}.
// vectors is the per-point map of named-space -> vector (omitted spaces absent).
// with_vector/with_payload query params (default both true) gate the projections;
// a not-found point (found=0 flag) is HTTP 404. Mirrors the dense getPoint.
func (a *api) namedGet(w http.ResponseWriter, r *http.Request) {
	name := r.PathValue("name")
	id, ok := pathID(w, r)
	if !ok {
		return
	}
	rc, opa, bound, ok := parseReadConsistencyBounded(w, r)
	if !ok {
		return
	}
	body, ok := a.call(w, r, "vector_named_get", ops.EncodeVectorGetArgsOpts(name, id, parseGetFlags(r), rc, opa, bound))
	if !ok {
		return
	}
	found, vectors, payload, ttl, version, err := ops.DecodeNamedGetResultV(body)
	if err != nil {
		writeInternalError(w, r.URL.Path, err)
		return
	}
	if !found {
		writeError(w, http.StatusNotFound, "point not found")
		return
	}
	writeJSON(w, http.StatusOK, map[string]any{
		"found": found, "vectors": vectors, "payload": payload, "ttl_ms": ttl.Milliseconds(), "version": version,
	})
}

// namedBatchGetPoint is one present point in the named batch-get response: its id
// + the same projected fields a single named get carries (the per-space vectors
// map / payload / ttl_ms). The named clone of batchGetPoint.
type namedBatchGetPoint struct {
	ID      uint64               `json:"id"`
	Vectors map[string][]float32 `json:"vectors"`
	Payload vector.Metadata      `json:"payload"`
	TTLMs   int64                `json:"ttl_ms"`
	Version uint64               `json:"version"`
}

// namedGetBatch retrieves MANY named points by id in ONE request (POST since the
// id list can be large). Body {ids:[...], with_vector, with_payload} → {points:
// [{id,vectors,payload,ttl_ms}], missing:[...]}. The coordinator scatters the ids
// to their owning partitions and merges (via the vector_named_get_batch op). A
// partial miss is NORMAL — absent ids come back in "missing", NOT a 404 (unlike
// single named get). Empty ids → empty points + missing. A malformed body is 400.
// The named clone of getPointsBatch.
func (a *api) namedGetBatch(w http.ResponseWriter, r *http.Request) {
	name := r.PathValue("name")
	var req batchGetReq
	if !decodeBodyBulk(w, r, &req) {
		return
	}
	var flags uint8
	if req.WithVector == nil || *req.WithVector {
		flags |= ops.GetFlagWithVector
	}
	if req.WithPayload == nil || *req.WithPayload {
		flags |= ops.GetFlagWithPayload
	}
	body, ok := a.call(w, r, "vector_named_get_batch", ops.EncodeVectorGetBatchArgs(name, req.IDs, flags))
	if !ok {
		return
	}
	rows, err := ops.DecodeNamedGetBatchResult(body)
	if err != nil {
		writeInternalError(w, r.URL.Path, err)
		return
	}
	points := make([]namedBatchGetPoint, 0, len(rows))
	missing := make([]uint64, 0)
	for i := range rows {
		row := &rows[i]
		if !row.Found {
			missing = append(missing, row.ID)
			continue
		}
		points = append(points, namedBatchGetPoint{
			ID:      row.ID,
			Vectors: row.Vectors,
			Payload: row.Meta,
			TTLMs:   int64(row.TTLMs), //nolint:gosec // TTL ms >= 0
			Version: row.Version,
		})
	}
	writeJSON(w, http.StatusOK, map[string]any{"points": points, "missing": missing})
}

func (a *api) namedSetPayload(w http.ResponseWriter, r *http.Request) {
	name := r.PathValue("name")
	id, ok := pathID(w, r)
	if !ok {
		return
	}
	// An optional key_ttl_ms object in the body sets per-key payload TTLs (relative
	// ms); the rest of the body is the payload (see decodePayloadBodyWithTTL).
	meta, keyTTLMs, ok := decodePayloadBodyWithTTL(w, r)
	if !ok {
		return
	}
	exp, hasExp, ok := queryExpectedVersion(w, r)
	if !ok {
		return
	}
	a.applyPayload(w, r, "vector_named_set_payload", ops.EncodeSetPayloadArgsCAS(name, id, meta, keyTTLMs, exp, hasExp))
}

func (a *api) namedOverwritePayload(w http.ResponseWriter, r *http.Request) {
	name := r.PathValue("name")
	id, ok := pathID(w, r)
	if !ok {
		return
	}
	meta, keyTTLMs, ok := decodePayloadBodyWithTTL(w, r)
	if !ok {
		return
	}
	exp, hasExp, ok := queryExpectedVersion(w, r)
	if !ok {
		return
	}
	a.applyPayload(w, r, "vector_named_overwrite_payload", ops.EncodeSetPayloadArgsCAS(name, id, meta, keyTTLMs, exp, hasExp))
}

func (a *api) namedDeletePayloadKeys(w http.ResponseWriter, r *http.Request) {
	name := r.PathValue("name")
	id, ok := pathID(w, r)
	if !ok {
		return
	}
	var req deletePayloadKeysReq
	if !decodeBody(w, r, &req) {
		return
	}
	exp, hasExp, ok := queryExpectedVersion(w, r)
	if !ok {
		return
	}
	a.applyPayload(w, r, "vector_named_delete_payload_keys", ops.EncodeDeletePayloadKeysArgsCAS(name, id, req.Keys, exp, hasExp))
}

func (a *api) namedClearPayload(w http.ResponseWriter, r *http.Request) {
	name := r.PathValue("name")
	id, ok := pathID(w, r)
	if !ok {
		return
	}
	exp, hasExp, ok := queryExpectedVersion(w, r)
	if !ok {
		return
	}
	a.applyPayload(w, r, "vector_named_clear_payload", ops.EncodeClearPayloadArgsCAS(name, id, exp, hasExp))
}

func (a *api) namedGetConfig(w http.ResponseWriter, r *http.Request) {
	name := r.PathValue("name")
	rc, opa, bound, ok := parseReadConsistencyBounded(w, r)
	if !ok {
		return
	}
	body, ok := a.call(w, r, "vector_named_get_config", ops.EncodeNamedNameArgsOpts(name, rc, opa, bound))
	if !ok {
		return
	}
	cfg, err := ops.DecodeNamedConfigResult(body)
	if err != nil {
		writeInternalError(w, r.URL.Path, err)
		return
	}
	writeJSON(w, http.StatusOK, map[string]any{"name": name, "named_vectors": cfg})
}
