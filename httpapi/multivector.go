// SPDX-License-Identifier: Apache-2.0

package httpapi

import (
	"net/http"
	"strconv"
	"strings"

	"github.com/rostamlabs/rostam/ops"
	"github.com/rostamlabs/rostam/vector"
)

// REST handlers for late-interaction (multi-vector / ColBERT MaxSim) collections,
// under /v1/multivector. A document is a list of token vectors; relevance is
// MaxSim. These collections are in-memory only.

type mvCreateReq struct {
	Dim            int    `json:"dim"`
	M              int    `json:"m"`
	EfConstruction int    `json:"ef_construction"`
	EfSearch       int    `json:"ef_search"`
	Seed           int64  `json:"seed"`
	Quant          string `json:"quant"`          // "" / "none" | "sq8" | "bq1"
	RescoreFactor  int    `json:"rescore_factor"` // quantized first-stage over-fetch
	Persistent     bool   `json:"persistent"`     // mmap-backed, off-heap, durable
	Partitions     int    `json:"partitions"`
	// IndexType selects the inner token index: "" / "hnsw" (default) or "ivf".
	// The IVF knobs mirror the dense collectionConfig IVF fields and parameterize
	// the inner IVF-Flat / IVF-PQ index (compresses the dominant MV token memory).
	// All ignored unless index_type == "ivf". Absent/zero => byte-identical wire.
	IndexType string `json:"index_type"`
	IVFNlist  int    `json:"ivf_nlist"`
	IVFNprobe int    `json:"ivf_nprobe"`
	IVFPQ     bool   `json:"ivf_pq"`
	IVFPQM    int    `json:"ivf_pq_m"`
	IVFRerank bool   `json:"ivf_rerank"`
	OPQ       bool   `json:"opq"`
	// OPQIters drives full-OPQ iterative Procrustes refinement on the inner token
	// index (0 = 1 = v1 behavior, byte-identical; > 1 = that many refine
	// iterations). Ignored unless opq is set. Must be in [0, 20].
	OPQIters int `json:"opq_iters"`
	// IVFTrainThreshold is the live token count at which the incrementally-built
	// quantized inner index (IVF or HNSW-PQ) deterministically auto-trains (0 =
	// engine default).
	IVFTrainThreshold int `json:"ivf_train_threshold"`
	// IVFDriftRetrain / IVFDriftGrowthFactor / IVFDriftFactor mirror the dense
	// drift-retrain knobs: the inner IVF token index opts into deterministic
	// auto-retrain-on-drift. Factors must be > 1.0 (0 = default). Ignored unless
	// index_type == "ivf".
	IVFDriftRetrain      bool    `json:"ivf_drift_retrain"`
	IVFDriftGrowthFactor float64 `json:"ivf_drift_growth_factor"`
	IVFDriftFactor       float64 `json:"ivf_drift_factor"`
	// PQDropVecs (HNSW-PQ only, quant == "pq") drops the resident float token
	// vectors once the incrementally-built inner HNSW-PQ index auto-trains
	// (maximum compression, ADC-only first stage). Requires quant == "pq".
	PQDropVecs bool `json:"pq_drop_vecs"`
	// FilterFirstRelativeBP is the opt-in relative selectivity gate (basis points of
	// the live document count, 0..10000; 0 = off = byte-identical).
	FilterFirstRelativeBP int `json:"filter_first_relative_bp"`
}

func (a *api) mvCreate(w http.ResponseWriter, r *http.Request) {
	name := r.PathValue("name")
	if strings.ContainsAny(name, "#@") {
		writeError(w, http.StatusBadRequest,
			"vector: collection name "+strconv.Quote(name)+" must not contain reserved characters '#' or '@'")
		return
	}
	var req mvCreateReq
	if !decodeBody(w, r, &req) {
		return
	}
	quant, ok := parseQuant(req.Quant)
	if !ok {
		writeError(w, http.StatusBadRequest, "unknown quant "+strconv.Quote(req.Quant))
		return
	}
	if req.Partitions < 0 {
		writeError(w, http.StatusBadRequest, "partitions must be non-negative")
		return
	}
	indexType, ok := parseIndexType(req.IndexType)
	if !ok {
		writeError(w, http.StatusBadRequest, "unknown index_type "+strconv.Quote(req.IndexType))
		return
	}
	if req.IVFNlist < 0 || req.IVFNprobe < 0 {
		writeError(w, http.StatusBadRequest, "ivf_nlist and ivf_nprobe must be non-negative")
		return
	}
	if req.IVFDriftGrowthFactor != 0 && req.IVFDriftGrowthFactor <= 1.0 {
		writeError(w, http.StatusBadRequest, "ivf_drift_growth_factor must be > 1.0")
		return
	}
	if req.IVFDriftFactor != 0 && req.IVFDriftFactor <= 1.0 {
		writeError(w, http.StatusBadRequest, "ivf_drift_factor must be > 1.0")
		return
	}
	if req.FilterFirstRelativeBP < 0 || req.FilterFirstRelativeBP > 10000 {
		writeError(w, http.StatusBadRequest, "filter_first_relative_bp must be in [0, 10000]")
		return
	}
	cfg := vector.MultiVectorConfig{
		Dim: req.Dim, M: req.M, EfConstruction: req.EfConstruction, EfSearch: req.EfSearch,
		Seed: req.Seed, Quant: quant, RescoreFactor: req.RescoreFactor, Persistent: req.Persistent,
		Partitions:            req.Partitions,
		IndexType:             indexType,
		IVFNlist:              req.IVFNlist,
		IVFNprobe:             req.IVFNprobe,
		IVFPQ:                 req.IVFPQ,
		IVFPQM:                req.IVFPQM,
		IVFRerank:             req.IVFRerank,
		OPQ:                   req.OPQ,
		OPQIters:              req.OPQIters,
		IVFTrainThreshold:     req.IVFTrainThreshold,
		PQDropVecs:            req.PQDropVecs,
		IVFDriftRetrain:       req.IVFDriftRetrain,
		IVFDriftGrowthFactor:  req.IVFDriftGrowthFactor,
		IVFDriftFactor:        req.IVFDriftFactor,
		FilterFirstRelativeBP: req.FilterFirstRelativeBP,
	}
	if _, ok := a.call(w, r, "vector_mv_create_collection", ops.EncodeMVCreateArgs(name, cfg)); !ok {
		return
	}
	writeJSON(w, http.StatusCreated, map[string]string{"name": name})
}

func (a *api) mvDrop(w http.ResponseWriter, r *http.Request) {
	name := r.PathValue("name")
	if _, ok := a.call(w, r, "vector_mv_drop_collection", ops.EncodeMVDeleteArgs(name, 0)); !ok {
		return
	}
	writeJSON(w, http.StatusOK, map[string]string{"dropped": name})
}

type mvAddReq struct {
	ID       uint64          `json:"id"`
	Tokens   [][]float32     `json:"tokens"`
	Metadata vector.Metadata `json:"metadata"`
	// ExpectedVersion is the optimistic-CAS precondition (0 = expect-absent /
	// add-if-absent). nil = an unconditional add. A mismatch returns 409 Conflict.
	ExpectedVersion *uint64 `json:"expected_version"`
	// KeyTTLMs is an OPTIONAL per-key payload TTL map (payload key -> RELATIVE ms).
	// At add the engine computes the absolute deadline now+ttl per key (mirroring
	// set_payload's key_ttl_ms) and lazily drops the key once it passes, while the
	// document lives on. Absent/empty = no per-key TTL (byte-identical wire).
	KeyTTLMs map[string]int64 `json:"key_ttl_ms"`
	// Sparse is an OPTIONAL doc-level sparse vector carried alongside the dense token
	// matrix (the MV analogue of a named point's sparse space; consumed by MV hybrid
	// search in a later feature). nil/zero ⇒ dense-only (byte-identical wire — no add
	// trailer, no persist block). Mirrors the dense/named "sparse" field shape.
	Sparse *vector.SparseVector `json:"sparse"`
	writeConsistency
}

func (a *api) mvAdd(w http.ResponseWriter, r *http.Request) {
	name := r.PathValue("name")
	var req mvAddReq
	if !decodeBodyBulk(w, r, &req) {
		return
	}
	if !req.validate(w) {
		return
	}
	exp, hasExp := uint64(0), false
	if req.ExpectedVersion != nil {
		exp, hasExp = *req.ExpectedVersion, true
	}
	if _, ok := a.callWrite(w, r, "vector_mv_add", ops.EncodeMVAddArgsCASKeyTTLSparse(name, req.ID, req.Tokens, req.Metadata, exp, hasExp, req.KeyTTLMs, req.Sparse), req.WriteConsistencyFactor, req.wait()); !ok {
		return
	}
	writeJSON(w, http.StatusOK, map[string]uint64{"id": req.ID})
}

func (a *api) mvDelete(w http.ResponseWriter, r *http.Request) {
	name := r.PathValue("name")
	id, err := strconv.ParseUint(r.PathValue("id"), 10, 64)
	if err != nil {
		writeError(w, http.StatusBadRequest, "invalid id: "+err.Error())
		return
	}
	wc, ok := queryWC(w, r)
	if !ok {
		return
	}
	exp, hasExp, ok := queryExpectedVersion(w, r)
	if !ok {
		return
	}
	body, ok := a.callWrite(w, r, "vector_mv_delete", ops.EncodeMVDeleteArgsCAS(name, id, exp, hasExp), wc.WriteConsistencyFactor, wc.wait())
	if !ok {
		return
	}
	writeJSON(w, http.StatusOK, map[string]bool{"deleted": len(body) > 0 && body[0] == 1})
}

// mvGet retrieves a multi-vector document by id → {found, tokens, payload}.
// tokens is the document's token matrix (MV documents have no TTL).
// with_vector/with_payload query params (default both true) gate the projections;
// a not-found document (found=0 flag) is HTTP 404. Mirrors the dense getPoint.
func (a *api) mvGet(w http.ResponseWriter, r *http.Request) {
	name := r.PathValue("name")
	id, ok := pathID(w, r)
	if !ok {
		return
	}
	rc, opa, bound, ok := parseReadConsistencyBounded(w, r)
	if !ok {
		return
	}
	body, ok := a.call(w, r, "vector_mv_get", ops.EncodeVectorGetArgsOpts(name, id, parseGetFlags(r), rc, opa, bound))
	if !ok {
		return
	}
	found, tokens, payload, version, err := ops.DecodeMVGetResultV(body)
	if err != nil {
		writeInternalError(w, r.URL.Path, err)
		return
	}
	if !found {
		writeError(w, http.StatusNotFound, "document not found")
		return
	}
	writeJSON(w, http.StatusOK, map[string]any{
		"found": found, "tokens": tokens, "payload": payload, "version": version,
	})
}

// mvBatchGetPoint is one present point in the MV batch-get response: its id + the
// same projected fields a single MV get carries (the token matrix / payload). MV
// has NO ttl. The MV clone of namedBatchGetPoint.
type mvBatchGetPoint struct {
	ID      uint64          `json:"id"`
	Tokens  [][]float32     `json:"tokens"`
	Payload vector.Metadata `json:"payload"`
	Version uint64          `json:"version"`
}

// mvGetBatch retrieves MANY MV documents by id in ONE request (POST since the id
// list can be large). Body {ids:[...], with_vector, with_payload} → {points:
// [{id,tokens,payload}], missing:[...]}. The coordinator scatters the ids to
// their owning partitions and merges (via the vector_mv_get_batch op). A partial
// miss is NORMAL — absent ids come back in "missing", NOT a 404 (unlike single MV
// get). Empty ids → empty points + missing. A malformed body is 400. MV has NO
// ttl. The MV clone of namedGetBatch.
func (a *api) mvGetBatch(w http.ResponseWriter, r *http.Request) {
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
	body, ok := a.call(w, r, "vector_mv_get_batch", ops.EncodeVectorGetBatchArgs(name, req.IDs, flags))
	if !ok {
		return
	}
	rows, err := ops.DecodeMVGetBatchResult(body)
	if err != nil {
		writeInternalError(w, r.URL.Path, err)
		return
	}
	points := make([]mvBatchGetPoint, 0, len(rows))
	missing := make([]uint64, 0)
	for i := range rows {
		row := &rows[i]
		if !row.Found {
			missing = append(missing, row.ID)
			continue
		}
		points = append(points, mvBatchGetPoint{
			ID:      row.ID,
			Tokens:  row.Tokens,
			Payload: row.Meta,
			Version: row.Version,
		})
	}
	writeJSON(w, http.StatusOK, map[string]any{"points": points, "missing": missing})
}

func (a *api) mvSetPayload(w http.ResponseWriter, r *http.Request) {
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
	a.applyPayload(w, r, "vector_mv_set_payload", ops.EncodeSetPayloadArgsCAS(name, id, meta, keyTTLMs, exp, hasExp))
}

func (a *api) mvOverwritePayload(w http.ResponseWriter, r *http.Request) {
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
	a.applyPayload(w, r, "vector_mv_overwrite_payload", ops.EncodeSetPayloadArgsCAS(name, id, meta, keyTTLMs, exp, hasExp))
}

func (a *api) mvDeletePayloadKeys(w http.ResponseWriter, r *http.Request) {
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
	a.applyPayload(w, r, "vector_mv_delete_payload_keys", ops.EncodeDeletePayloadKeysArgsCAS(name, id, req.Keys, exp, hasExp))
}

func (a *api) mvClearPayload(w http.ResponseWriter, r *http.Request) {
	name := r.PathValue("name")
	id, ok := pathID(w, r)
	if !ok {
		return
	}
	exp, hasExp, ok := queryExpectedVersion(w, r)
	if !ok {
		return
	}
	a.applyPayload(w, r, "vector_mv_clear_payload", ops.EncodeClearPayloadArgsCAS(name, id, exp, hasExp))
}

type mvSearchReq struct {
	Query                  [][]float32   `json:"query"`
	K                      int           `json:"k"`
	CandidatesPerToken     int           `json:"candidates_per_token"`
	Filter                 vector.Filter `json:"filter"`
	ReadConsistency        uint8         `json:"read_consistency"`
	OnPartitionUnavailable uint8         `json:"on_partition_unavailable"`
	MaxStaleness           uint64        `json:"max_staleness"` // bound for rc==3 (bounded-staleness)
}

func (a *api) mvSearch(w http.ResponseWriter, r *http.Request) {
	name := r.PathValue("name")
	var req mvSearchReq
	if !decodeBody(w, r, &req) {
		return
	}
	if !validConsistency(w, req.ReadConsistency, req.OnPartitionUnavailable) {
		return
	}
	if !validFilter(w, req.Filter) {
		return
	}
	args := ops.EncodeMVSearchArgsOptsFilter(name, req.Query, req.K, req.CandidatesPerToken, req.ReadConsistency, req.OnPartitionUnavailable, req.Filter, req.MaxStaleness)
	body, ok := a.call(w, r, "vector_mv_search", args)
	if !ok {
		return
	}
	results, degraded, missing, err := ops.DecodeMVResultsDegraded(body)
	if err != nil {
		writeInternalError(w, r.URL.Path, err)
		return
	}
	writeJSON(w, http.StatusOK, map[string]any{"results": results, "degraded": degraded, "missing": missingJSON(missing)})
}

// mvHybridReq fuses an MV collection's MaxSim (late-interaction dense) lane (query
// token matrix) and its per-doc sparse lane (sparse {indices,values}) into the
// top-k. method ""/"rrf"|"weighted"|"dbsf"; alpha is the MaxSim-lane weight for weighted/dbsf
// fusion; filter applies to BOTH lanes. An empty query degrades to the sparse lane
// only; an absent/empty sparse degrades to the MaxSim lane only.
type mvHybridReq struct {
	Query                  [][]float32          `json:"query"`
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

// mvHybrid fuses an MV collection's MaxSim lane + its doc-level sparse lane into the
// top-k (cross-modality hybrid). Results are ranked by the fused score (id + distance
// + score). Mirrors namedHybrid.
func (a *api) mvHybrid(w http.ResponseWriter, r *http.Request) {
	name := r.PathValue("name")
	var req mvHybridReq
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
	args := ops.EncodeMVHybridArgs(name, req.Query, sparse, req.K, opts, req.ReadConsistency, req.OnPartitionUnavailable, req.MaxStaleness)
	body, ok := a.call(w, r, "vector_mv_hybrid_search", args)
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

// mvQueryLeafReq is one JSON query node in a /multivector/{name}/query body: an MV
// MaxSim leaf (maxsim is the token query matrix, [][]float32 — same shape the MV
// hybrid query field carries) OR the doc-level sparse field (sparse). Exactly one
// payload; an empty leaf is the "no root" marker for FUSION. k is the leaf's own
// top-k; lane_k is the per-lane candidate pool (0 = engine default); filter is the
// optional per-leaf metadata predicate (same JSON contract as the MV hybrid filter).
type mvQueryLeafReq struct {
	MaxSim [][]float32          `json:"maxsim"`
	Sparse *vector.SparseVector `json:"sparse"`
	K      int                  `json:"k"`
	LaneK  int                  `json:"lane_k"`
	Filter vector.Filter        `json:"filter"`
}

// toLeaf converts the JSON MV leaf into an engine vector.QueryLeaf. A leaf with a
// sparse payload is LeafSparse (the MV doc sparse field, no space); otherwise it is
// a LeafMVMaxSim carrying the token matrix. BOTH MV lanes are score-descending, so
// ScoreDesc is set on both (the orientation-aware fuse reads lane0's orientation).
func (l mvQueryLeafReq) toLeaf() vector.QueryLeaf {
	if l.Sparse != nil && !l.Sparse.IsZero() {
		return vector.QueryLeaf{
			Kind:   vector.LeafSparse,
			Sparse: *l.Sparse,
			K:      l.K, LaneK: l.LaneK, Filter: l.Filter,
			ScoreDesc: true,
		}
	}
	return vector.QueryLeaf{
		Kind:   vector.LeafMVMaxSim,
		Tokens: l.MaxSim,
		K:      l.K, LaneK: l.LaneK, Filter: l.Filter,
		ScoreDesc: true,
	}
}

// mvQueryReq is the MV-collection Query API HTTP body: a root leaf (RERANK) + N
// prefetch leaves — each an MV MaxSim leaf (token matrix) and/or the doc sparse
// field — a combine mode ("fusion"|"rerank"), the fusion config, the final top-k,
// and the standard read-consistency knobs. Mirrors namedQueryReq, swapping the
// per-leaf named space for the MV MaxSim token matrix / doc sparse field.
type mvQueryReq struct {
	Root     *mvQueryLeafReq  `json:"root"`
	Prefetch []mvQueryLeafReq `json:"prefetch"`
	Mode     string           `json:"mode"`   // "fusion" (default) | "rerank"
	Method   string           `json:"method"` // "" / "rrf" | "weighted" | "dbsf"
	Alpha    float64          `json:"alpha"`
	RRFK     int              `json:"rrf_k"`
	K        int              `json:"k"`

	ReadConsistency        uint8  `json:"read_consistency"`
	OnPartitionUnavailable uint8  `json:"on_partition_unavailable"`
	MaxStaleness           uint64 `json:"max_staleness"` // bound for rc==3 (bounded-staleness)
}

// mvQuery runs the unified Query API over a MULTI-VECTOR (late-interaction)
// collection: each leaf is an MV MaxSim leaf (token matrix) and/or the doc sparse
// field, so the N prefetch lanes fuse the MaxSim + sparse lanes (FUSION) or the root
// re-scores the prefetch candidate union (RERANK). Mirrors the named /query handler,
// swapping the per-leaf space for the MV token matrix. An unknown mode/method, a bad
// filter, or a malformed payload is a 400 at the edge (ValidateAndMarshalQuerySpec on
// the engine side fails loud on an empty MaxSim leaf); degraded/missing flow through
// DecodeQueryResultDegraded.
func (a *api) mvQuery(w http.ResponseWriter, r *http.Request) {
	name := r.PathValue("name")
	var req mvQueryReq
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
	for i := range req.Prefetch {
		if !validFilter(w, req.Prefetch[i].Filter) {
			return
		}
		spec.Prefetch = append(spec.Prefetch, vector.LeafSource(req.Prefetch[i].toLeaf()))
	}
	if req.Root != nil {
		if !validFilter(w, req.Root.Filter) {
			return
		}
		spec.Root = req.Root.toLeaf()
	} else if mode == vector.ModeRerank {
		writeError(w, http.StatusBadRequest, "mv query: rerank mode requires a root leaf")
		return
	}
	specBytes, err := ops.MarshalEngineQuerySpec(spec)
	if err != nil {
		writeError(w, http.StatusBadRequest, err.Error())
		return
	}
	body, ok := a.call(w, r, "vector_mv_query", ops.EncodeQueryArgs(name, specBytes, req.ReadConsistency, req.OnPartitionUnavailable, req.MaxStaleness))
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

type mvScrollReq struct {
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

// mvScroll lists live multi-vector documents (id + payload, no token matrix) with
// cursor pagination → {documents, degraded, missing, next_cursor}. Mirrors the
// dense/named scroll routes: validConsistency + bad-cursor fail-loud (400) BEFORE
// dispatch.
func (a *api) mvScroll(w http.ResponseWriter, r *http.Request) {
	name := r.PathValue("name")
	var req mvScrollReq
	if !decodeBody(w, r, &req) {
		return
	}
	if !validConsistency(w, req.ReadConsistency, req.OnPartitionUnavailable) {
		return
	}
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
	args := ops.EncodeMVScrollArgsOrderBounded(name, req.Filter, req.Limit, req.ReadConsistency, req.OnPartitionUnavailable, afterID, hasAfter, scrollOrder, req.MaxStaleness)
	body, ok := a.call(w, r, "vector_mv_scroll", args)
	if !ok {
		return
	}
	docs, degraded, missing, nextCursor, err := ops.DecodeScrollResultRaw(body)
	if err != nil {
		writeInternalError(w, r.URL.Path, err)
		return
	}
	writeJSON(w, http.StatusOK, map[string]any{"documents": docs, "degraded": degraded, "missing": missingJSON(missing), "next_cursor": nextCursor})
}

// mvResplit re-partitions a multi-vector (late-interaction) collection into
// new_partitions shards. Like the dense path, resplit is SYNCHRONOUS and
// OFFLINE: it rebuilds partition state in place, so writes must be quiesced for
// its duration and the caller/proxy must allow a long request timeout. Reuses
// resplitReq (same package). Orphaned old partitions remain until cleanup.
func (a *api) mvResplit(w http.ResponseWriter, r *http.Request) {
	name := r.PathValue("name")
	var req resplitReq
	if !decodeBody(w, r, &req) {
		return
	}
	if req.NewPartitions < 0 {
		writeError(w, http.StatusBadRequest, "new_partitions must be non-negative")
		return
	}
	if _, ok := a.call(w, r, "vector_mv_resplit", ops.EncodeResplitArgs(name, req.NewPartitions)); !ok {
		return
	}
	writeJSON(w, http.StatusOK, map[string]any{"name": name, "new_partitions": req.NewPartitions})
}

// mvResplitCleanup drops the orphaned old partitions left behind by a prior
// multi-vector resplit, returning how many were dropped. Like resplit it is
// synchronous and offline (quiesce writes; allow a long request timeout).
func (a *api) mvResplitCleanup(w http.ResponseWriter, r *http.Request) {
	name := r.PathValue("name")
	body, ok := a.call(w, r, "vector_mv_resplit_cleanup", ops.EncodeResplitCleanupArgs(name))
	if !ok {
		return
	}
	dropped, err := ops.DecodeResplitCleanupResult(body)
	if err != nil {
		writeInternalError(w, r.URL.Path, err)
		return
	}
	writeJSON(w, http.StatusOK, map[string]any{"name": name, "dropped": dropped})
}

// mvReshard re-partitions a multi-vector collection ONLINE into new_partitions
// shards. Like the dense path, reads AND writes stay live (dual-write) for the
// duration; the request is synchronous (blocks until cutover) so the caller/
// proxy must allow a long request timeout. Reuses resplitReq (same package).
func (a *api) mvReshard(w http.ResponseWriter, r *http.Request) {
	name := r.PathValue("name")
	var req resplitReq
	if !decodeBody(w, r, &req) {
		return
	}
	if req.NewPartitions < 0 {
		writeError(w, http.StatusBadRequest, "new_partitions must be non-negative")
		return
	}
	if _, ok := a.call(w, r, "vector_mv_reshard", ops.EncodeReshardArgs(name, req.NewPartitions)); !ok {
		return
	}
	writeJSON(w, http.StatusOK, map[string]any{"name": name, "new_partitions": req.NewPartitions})
}

// mvReshardAbort aborts an in-flight multi-vector reshard; see reshardAbort.
// Pre-cutover only.
func (a *api) mvReshardAbort(w http.ResponseWriter, r *http.Request) {
	name := r.PathValue("name")
	if _, ok := a.call(w, r, "vector_mv_reshard_abort", ops.EncodeReshardAbortArgs(name)); !ok {
		return
	}
	writeJSON(w, http.StatusOK, map[string]any{"name": name})
}
