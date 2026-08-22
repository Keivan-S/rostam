// SPDX-License-Identifier: Apache-2.0

package wire

import (
	"encoding/binary"
	"encoding/json"
	"fmt"

	"google.golang.org/protobuf/proto"

	"github.com/rostamlabs/rostam/grpcapi/pb"
	"github.com/rostamlabs/rostam/vector"
)

// parseFilterJSON decodes a per-leaf metadata filter from its JSON string. An
// empty string yields the zero vector.Filter (no filter); a malformed JSON is a
// fail-loud error (never silently dropped to no-filter — the same convention the
// hybrid/named decoders use).
func parseFilterJSON(s string) (vector.Filter, error) {
	if s == "" {
		return vector.Filter{}, nil
	}
	var f vector.Filter
	if err := json.Unmarshal([]byte(s), &f); err != nil {
		return vector.Filter{}, fmt.Errorf("ops: decode filter: %w", err)
	}
	return f, nil
}

// Result mode tags for the mode-tagged vector_query result payload.
const (
	QueryResultModeFusion  uint8 = 0 // [mode][nLanes:u32]{EncodeHybridResults(lane)}
	QueryResultModeRerank  uint8 = 1 // [mode]EncodeHybridResults(scored)
	queryResultModeGrouped uint8 = 2 // [mode]EncodeGroups(groups) — a GROUPED query result
	// QueryResultModeTreeLanes carries the per-partition UNFUSED tree-lanes (the flat
	// pre-order spec-tree lane list) for a spec with a nested MULTI-lane FUSION node
	// (vector.SpecHasNestedFusion). Wire shape is IDENTICAL to the FUSION mode body
	// ([nLanes:u32]{EncodeHybridResults(lane)}) — only the tag differs — but the lane
	// list is the EXPANDED tree traversal (collectTreeLanesAt), not the top-level lanes,
	// so the coordinator re-walks the spec tree and folds each FUSION node over the
	// global union. A spec WITHOUT a nested multi-lane FUSION never uses this tag, so the
	// flat FUSION/RERANK/nested-RERANK-only wire is BYTE-IDENTICAL to before.
	QueryResultModeTreeLanes uint8 = 4 // [mode][nLanes:u32]{EncodeHybridResults(lane)}
)

// EncodeQueryArgs serializes vector_query args: the collection name, the
// marshaled QuerySpec bytes, and the optional read-opts trailer. When rc==0 &&
// opa==0 the trailer is omitted (byte-clean), matching the hybrid opts
// convention. bound rides only when rc==ConsistencyBoundedStaleness.
func EncodeQueryArgs(collection string, specBytes []byte, readConsistency, onPartitionUnavailable uint8, bound uint64) []byte {
	n := 1 + len(collection) + 4 + len(specBytes)
	buf := make([]byte, n)
	buf[0] = byte(len(collection))
	off := 1 + copy(buf[1:], collection)
	binary.BigEndian.PutUint32(buf[off:], uint32(len(specBytes)))
	off += 4
	copy(buf[off:], specBytes)
	return AppendReadOptsTrailerBounded(buf, readConsistency, onPartitionUnavailable, bound)
}

// DecodeQueryArgs reads vector_query args produced by EncodeQueryArgs. specBytes
// aliases args (the caller proto-unmarshals it immediately). It is fail-loud on
// truncation (a present opts marker with a short rc/opa/bound block is corruption,
// never a silent drop of an explicit Linearizable/bounded request).
func DecodeQueryArgs(args []byte) (collection string, specBytes []byte, readConsistency, onPartitionUnavailable uint8, bound uint64, err error) {
	if len(args) < 1 {
		return "", nil, 0, 0, 0, ErrVectorArgsTruncated
	}
	colLen := int(args[0])
	if len(args) < 1+colLen+4 {
		return "", nil, 0, 0, 0, ErrVectorArgsTruncated
	}
	collection = string(args[1 : 1+colLen])
	off := 1 + colLen
	specLen := int(binary.BigEndian.Uint32(args[off:]))
	off += 4
	if len(args) < off+specLen {
		return "", nil, 0, 0, 0, ErrVectorArgsTruncated
	}
	specBytes = args[off : off+specLen]
	off += specLen
	readConsistency, onPartitionUnavailable, bound, err = DecodeReadOptsTrailerBounded(args, off)
	if err != nil {
		return "", nil, 0, 0, 0, err
	}
	return collection, specBytes, readConsistency, onPartitionUnavailable, bound, nil
}

// DecodeQuerySpecArgs is the full vector_query arg decoder for the coordinator
// (fan-out dispatcher): it decodes the op args, proto-unmarshals the spec blob,
// and converts it into the engine's proto-free vector.QuerySpec — everything the
// cross-partition coordinator needs (the spec for the fusion/rerank merge, the
// raw specBytes to re-fan to each partition, and the rc/opa/bound opts). It is
// fail-loud on a truncated arg frame, a malformed spec blob, or an invalid spec
// (unknown mode/method/leaf, bad filter JSON) — never a silent default. The
// handler-side path (handleVectorQuery) uses the same conversion; this is the
// coordinator-side public entry so the rostam package never imports pb.
func DecodeQuerySpecArgs(args []byte) (collection string, specBytes []byte, spec vector.QuerySpec, readConsistency, onPartitionUnavailable uint8, bound uint64, err error) {
	collection, specBytes, readConsistency, onPartitionUnavailable, bound, err = DecodeQueryArgs(args)
	if err != nil {
		return "", nil, vector.QuerySpec{}, 0, 0, 0, err
	}
	var pbSpec pb.QuerySpec
	if uerr := proto.Unmarshal(specBytes, &pbSpec); uerr != nil {
		return "", nil, vector.QuerySpec{}, 0, 0, 0, fmt.Errorf("ops: decode query spec: %w", uerr)
	}
	spec, err = QuerySpecFromProto(&pbSpec, 0)
	if err != nil {
		return "", nil, vector.QuerySpec{}, 0, 0, 0, err
	}
	return collection, specBytes, spec, readConsistency, onPartitionUnavailable, bound, nil
}

// EncodeQueryResultFused encodes a FLAT final top-k (the coordinator's fused /
// reranked result) as a mode-tagged vector_query result the client/direct
// VectorQuery decodes via DecodeQueryResult. The coordinator has already fused
// or reranked into a single list, so it is encoded with the RERANK tag (a flat
// scored list in Fused) — both DecodeQueryResult modes the data-plane reader
// consumes from qr.Fused, so a coordinator-fused result and a single-shard
// RERANK result decode identically into Fused. Used by the fan-out dispatcher to
// re-encode the partitioned vector_query result for the networked client.
func EncodeQueryResultFused(results []vector.Result) []byte {
	return EncodeQueryResult(vector.QueryResult{Mode: vector.ModeRerank, Fused: results})
}

// EncodeQueryResultFusedDegraded is EncodeQueryResultFused plus the optional
// degraded/missing partition trailer (same wire format as the hybrid/search
// trailers). The coordinator (fan-out dispatcher) uses it so a partitioned
// vector_query carries its FanMeta back to the networked client; the RERANK tag
// ends with a single EncodeHybridResults block, so the trailer appends cleanly
// after it and DecodeQueryResultDegraded reads it back. When degraded is false
// and missing is empty the output is byte-identical to EncodeQueryResultFused.
func EncodeQueryResultFusedDegraded(results []vector.Result, degraded bool, missing []uint16) []byte {
	return appendDegradedTrailer(EncodeQueryResultFused(results), degraded, missing)
}

// DecodeQueryResultDegraded decodes a flat (RERANK-tagged) fused vector_query
// result produced by EncodeQueryResultFused / EncodeQueryResultFusedDegraded,
// plus the optional degraded/missing trailer. It is the coordinator-result
// reader for the dedicated VectorQuery RPC (so degraded/missing flow into the
// SearchResponse, mirroring DecodeHybridResultsDegraded). Backward-compatible
// with a trailer-less fused body. A FUSION-tagged body (unfused lanes) is NOT a
// valid coordinator result here — the coordinator always re-encodes a flat
// RERANK-tagged result — so it is fail-loud.
func DecodeQueryResultDegraded(body []byte) (results []vector.Result, degraded bool, missing []uint16, err error) {
	if len(body) < 1 {
		return nil, false, nil, ErrVectorArgsTruncated
	}
	if body[0] != QueryResultModeRerank {
		return nil, false, nil, fmt.Errorf("ops: query result mode %d is not a flat fused result", body[0])
	}
	results, n, err := decodeHybridResultsN(body[1:])
	if err != nil {
		return nil, false, nil, err
	}
	degraded, missing, err = readDegradedTrailer(body, 1+n)
	return results, degraded, missing, err
}

// ValidateAndMarshalQuerySpec validates a proto pb.QuerySpec at the transport
// edge (gRPC/HTTP) and returns its marshaled bytes for the vector_query op's
// spec blob. It runs the SAME fail-loud conversion the op handler uses
// (QuerySpecFromProto: unknown fusion method, unknown mode, empty leaf oneof,
// bad per-leaf filter JSON), so the edge returns an InvalidArgument/400 instead
// of the op handler later failing with an Internal error. The returned bytes are
// exactly what EncodeQueryArgs carries; the handler re-unmarshals them.
func ValidateAndMarshalQuerySpec(p *pb.QuerySpec) ([]byte, error) {
	if _, err := QuerySpecFromProto(p, 0); err != nil {
		return nil, err
	}
	return proto.Marshal(p)
}

// MarshalEngineQuerySpec converts a proto-free engine vector.QuerySpec into the
// marshaled pb.QuerySpec bytes the vector_query op carries. It is the inverse of
// QuerySpecFromProto and lets a pb-free caller (the HTTP front end, which builds
// the engine struct from JSON) produce the op's spec blob without importing pb.
// Per-leaf filters are re-encoded to their JSON string (the leaf proto's
// filter_json), matching the contract the handler decodes back. Fail-loud on an
// unknown mode/method/leaf kind or a filter that cannot marshal to JSON.
func MarshalEngineQuerySpec(spec vector.QuerySpec) ([]byte, error) {
	p, err := QuerySpecToProto(spec)
	if err != nil {
		return nil, err
	}
	return proto.Marshal(p)
}

// QuerySpecToProto is the engine→proto direction (the inverse of
// QuerySpecFromProto). The ops layer owns BOTH directions so the engine and the
// HTTP front end stay pb-free.
func QuerySpecToProto(spec vector.QuerySpec) (*pb.QuerySpec, error) {
	method, err := queryFusionToString(spec.Method)
	if err != nil {
		return nil, err
	}
	var mode pb.QueryMode
	switch spec.Mode {
	case vector.ModeFusion:
		mode = pb.QueryMode_QUERY_MODE_FUSION
	case vector.ModeRerank:
		mode = pb.QueryMode_QUERY_MODE_RERANK
	default:
		return nil, fmt.Errorf("ops: unknown query mode %d", spec.Mode)
	}
	p := &pb.QuerySpec{
		Mode:         mode,
		FusionMethod: method,
		Alpha:        spec.Alpha,
		RrfK:         int32(spec.RRFK),       //nolint:gosec // bounded RRF constant
		K:            int32(spec.K),          //nolint:gosec // bounded top-k
		GroupBy:      spec.GroupBy,           // empty ⇒ flat query (byte-identical wire)
		GroupSize:    uint32(spec.GroupSize), //nolint:gosec // bounded group size
	}
	// Encode the root only when it carries a payload (RERANK needs it; FUSION
	// leaves it empty). An empty-kind dense root with no vector is treated as
	// "no root" so a FUSION spec round-trips without a spurious empty leaf.
	if hasLeafPayload(spec.Root) {
		root, rerr := queryLeafToProto(spec.Root)
		if rerr != nil {
			return nil, rerr
		}
		p.Root = root
	}
	// BACK-COMPAT WIRE: a flat leaf-only spec encodes its prefetch into the v1
	// `prefetch` field (repeated QueryLeaf) byte-identically; a spec with ANY nested
	// QuerySpec source instead encodes the WHOLE prefetch list into the additive
	// `prefetch_sources` field (repeated QuerySource). This keeps the v1 wire (and
	// every existing query) byte-for-byte unchanged while carrying nested trees.
	nested := false
	for i := range spec.Prefetch {
		if spec.Prefetch[i].Spec != nil {
			nested = true
			break
		}
	}
	if nested {
		for i := range spec.Prefetch {
			src, serr := querySourceToProto(spec.Prefetch[i])
			if serr != nil {
				return nil, serr
			}
			p.PrefetchSources = append(p.PrefetchSources, src)
		}
		return p, nil
	}
	for i := range spec.Prefetch {
		// Leaf-only: a well-formed leaf source carries Leaf != nil. Defensive nil guard
		// (a malformed empty source) fails loud rather than panicking.
		if spec.Prefetch[i].Leaf == nil {
			return nil, fmt.Errorf("ops: query prefetch source is neither a leaf nor a spec")
		}
		leaf, lerr := queryLeafToProto(*spec.Prefetch[i].Leaf)
		if lerr != nil {
			return nil, lerr
		}
		p.Prefetch = append(p.Prefetch, leaf)
	}
	return p, nil
}

// querySourceToProto converts one engine QuerySource (a leaf OR a nested spec)
// into a proto QuerySource oneof. A leaf source → the leaf arm; a nested spec
// source → the spec arm (recursively encoded). Fail-loud on a malformed source
// (neither arm set).
func querySourceToProto(s vector.QuerySource) (*pb.QuerySource, error) {
	switch {
	case s.Leaf != nil:
		leaf, err := queryLeafToProto(*s.Leaf)
		if err != nil {
			return nil, err
		}
		return &pb.QuerySource{Source: &pb.QuerySource_Leaf{Leaf: leaf}}, nil
	case s.Spec != nil:
		sub, err := QuerySpecToProto(*s.Spec)
		if err != nil {
			return nil, err
		}
		return &pb.QuerySource{Source: &pb.QuerySource_Spec{Spec: sub}}, nil
	default:
		return nil, fmt.Errorf("ops: query prefetch source is neither a leaf nor a spec")
	}
}

// hasLeafPayload reports whether a leaf carries an actual query payload (a dense
// vector or a sparse vector) — used to decide whether to encode the root.
func hasLeafPayload(l vector.QueryLeaf) bool {
	switch l.Kind {
	case vector.LeafSparse:
		return len(l.Sparse.Indices) > 0 || len(l.Sparse.Values) > 0
	case vector.LeafMVMaxSim:
		return len(l.Tokens) > 0
	case vector.LeafRecommend:
		// A recommend root carries its query as the positive example ids (the dense
		// vector is derived later), so it has a payload when it names any positives — or,
		// for a coordinator-resolved BEST_SCORE leaf, when it carries embedded positives.
		return len(l.Positive) > 0 || len(l.RecPosVecs) > 0
	case vector.LeafDiscover:
		// A discover root carries its query as the context pairs (resolved vectors or
		// unresolved ids), so it has a payload when it names any context pair.
		return len(l.DiscoverContext) > 0 || len(l.DiscoverContextIDs) > 0
	default: // LeafDense
		return len(l.Dense) > 0
	}
}

// queryLeafToProto converts one engine QueryLeaf into a proto QueryLeaf oneof. A
// Space-bearing leaf (the named family) is encoded as a NamedDense / NamedSparse
// arm carrying the space; a Space-less leaf is encoded as the dense-family Dense /
// Sparse arm — so a leaf round-trips into the same family it came from.
func queryLeafToProto(l vector.QueryLeaf) (*pb.QueryLeaf, error) {
	filterJSON, err := marshalFilterJSON(l.Filter)
	if err != nil {
		return nil, err
	}
	// A Space-bearing dense/sparse leaf is the named family's dense/sparse lane —
	// encode it as the NamedDense / NamedSparse arm carrying the space. A Space-bearing
	// RECOMMEND / DISCOVER leaf (the named recommend/discover family) is NOT a named
	// dense/sparse arm: it rides the SHARED RecommendLeaf / DiscoverLeaf message (whose
	// `space` field round-trips the named target space — see those arms below), so it
	// falls through to the unified switch.
	if l.Space != "" {
		switch l.Kind {
		case vector.LeafDense:
			return &pb.QueryLeaf{Leaf: &pb.QueryLeaf_NamedDense{NamedDense: &pb.NamedDenseLeaf{
				Space:      l.Space,
				Dense:      l.Dense,
				K:          int32(l.K), //nolint:gosec // bounded top-k
				FilterJson: filterJSON,
				LaneK:      int32(l.LaneK), //nolint:gosec // bounded lane pool
			}}}, nil
		case vector.LeafSparse:
			return &pb.QueryLeaf{Leaf: &pb.QueryLeaf_NamedSparse{NamedSparse: &pb.NamedSparseLeaf{
				Space:      l.Space,
				Indices:    l.Sparse.Indices,
				Values:     l.Sparse.Values,
				K:          int32(l.K), //nolint:gosec // bounded top-k
				FilterJson: filterJSON,
				LaneK:      int32(l.LaneK), //nolint:gosec // bounded lane pool
			}}}, nil
		case vector.LeafRecommend, vector.LeafDiscover:
			// Fall through to the shared recommend/discover arms (they encode Space).
		default:
			return nil, fmt.Errorf("ops: query leaf has an unknown kind %d", l.Kind)
		}
	}
	switch l.Kind {
	case vector.LeafDense:
		return &pb.QueryLeaf{Leaf: &pb.QueryLeaf_Dense{Dense: &pb.DenseLeaf{
			Dense:      l.Dense,
			K:          int32(l.K), //nolint:gosec // bounded top-k
			FilterJson: filterJSON,
			LaneK:      int32(l.LaneK), //nolint:gosec // bounded lane pool
		}}}, nil
	case vector.LeafSparse:
		return &pb.QueryLeaf{Leaf: &pb.QueryLeaf_Sparse{Sparse: &pb.SparseLeaf{
			Indices:    l.Sparse.Indices,
			Values:     l.Sparse.Values,
			K:          int32(l.K), //nolint:gosec // bounded top-k
			FilterJson: filterJSON,
			LaneK:      int32(l.LaneK), //nolint:gosec // bounded lane pool
		}}}, nil
	case vector.LeafMVMaxSim:
		// The MaxSim token query matrix is encoded as repeated TokenVector (one per
		// query token), the same [][]float32 wire MVSearchRequest carries.
		query := make([]*pb.TokenVector, len(l.Tokens))
		for i, tok := range l.Tokens {
			query[i] = &pb.TokenVector{Values: tok}
		}
		return &pb.QueryLeaf{Leaf: &pb.QueryLeaf_MvMaxsim{MvMaxsim: &pb.MVMaxSimLeaf{
			Query:      query,
			K:          int32(l.K), //nolint:gosec // bounded top-k
			FilterJson: filterJSON,
			LaneK:      int32(l.LaneK), //nolint:gosec // bounded lane pool
		}}}, nil
	case vector.LeafRecommend:
		// The recommend leaf carries the example POINT-IDS (positive/negative) + the
		// strategy. For AVERAGE_VECTOR the derive to a dense vector happens in the engine
		// coordinator pre-pass; for BEST_SCORE the coordinator resolves the ids and embeds
		// the example VECTORS (RecPosVecs/RecNegVecs → best_pos/best_neg), so the codec
		// transports both forms (a leaf round-trips whether it arrived as ids or already
		// embedded).
		var bestPos, bestNeg []*pb.TokenVector
		if n := len(l.RecPosVecs); n > 0 {
			bestPos = make([]*pb.TokenVector, n)
			for i, v := range l.RecPosVecs {
				bestPos[i] = &pb.TokenVector{Values: v}
			}
		}
		if n := len(l.RecNegVecs); n > 0 {
			bestNeg = make([]*pb.TokenVector, n)
			for i, v := range l.RecNegVecs {
				bestNeg[i] = &pb.TokenVector{Values: v}
			}
		}
		return &pb.QueryLeaf{Leaf: &pb.QueryLeaf_Recommend{Recommend: &pb.RecommendLeaf{
			Positive:   l.Positive,
			Negative:   l.Negative,
			K:          int32(l.K), //nolint:gosec // bounded top-k
			FilterJson: filterJSON,
			Strategy:   pb.RecommendStrategy(l.Strategy), //nolint:gosec // bounded enum
			BestPos:    bestPos,
			BestNeg:    bestNeg,
			// Space carries the named-family target space (empty for a dense recommend),
			// so a named recommend leaf round-trips into the named family.
			Space: l.Space,
		}}}, nil
	case vector.LeafDiscover:
		// The discover leaf carries BOTH the RESOLVED target/context VECTORS (what the
		// execLeaf consumes once resolved) AND the UNRESOLVED target/context IDS (the
		// input the coordinator resolves into the vector fields). The codec transports
		// both so a leaf round-trips whether it arrived pre-resolved or as ids.
		context := make([]*pb.ContextPair, len(l.DiscoverContext))
		for i, p := range l.DiscoverContext {
			context[i] = &pb.ContextPair{Positive: p.Pos, Negative: p.Neg}
		}
		contextIDs := make([]*pb.ContextPairIDs, len(l.DiscoverContextIDs))
		for i, cp := range l.DiscoverContextIDs {
			contextIDs[i] = &pb.ContextPairIDs{Positive: cp.Positive, Negative: cp.Negative}
		}
		return &pb.QueryLeaf{Leaf: &pb.QueryLeaf_Discover{Discover: &pb.DiscoverLeaf{
			Target:     l.DiscoverTarget,
			Context:    context,
			K:          int32(l.K), //nolint:gosec // bounded top-k
			FilterJson: filterJSON,
			TargetId:   l.DiscoverTargetID,
			ContextIds: contextIDs,
			// Space carries the named-family target space (empty for a dense discover),
			// so a named discover leaf round-trips into the named family.
			Space: l.Space,
		}}}, nil
	default:
		return nil, fmt.Errorf("ops: query leaf has an unknown kind %d", l.Kind)
	}
}

// marshalFilterJSON encodes a vector.Filter to its JSON string for a leaf's
// filter_json field. A zero filter yields "" (no filter), so the proto carries no
// filter_json and the handler decodes it back to the zero filter.
func marshalFilterJSON(f vector.Filter) (string, error) {
	if f.IsZero() {
		return "", nil
	}
	b, err := json.Marshal(f)
	if err != nil {
		return "", fmt.Errorf("ops: encode leaf filter: %w", err)
	}
	return string(b), nil
}

// queryFusionToString is the inverse of parseQueryFusion (engine method →
// fusion-method string). Fail-loud on an unknown method.
func queryFusionToString(m vector.FusionMethod) (string, error) {
	switch m {
	case vector.FusionRRF:
		return "rrf", nil
	case vector.FusionWeighted:
		return "weighted", nil
	case vector.FusionDBSF:
		return "dbsf", nil
	default:
		return "", fmt.Errorf("ops: unknown fusion method %d", m)
	}
}

// MaxQueryDepth bounds the nested-prefetch tree (anti-DoS): the root spec is depth
// 0, each nested QuerySpec source increments the depth, and a depth exceeding this
// bound is rejected fail-loud at DECODE (ErrQuerySpecTooDeep) BEFORE any engine
// work — a maliciously deep spec cannot exhaust the stack/heap. A flat 1-level
// (leaf-source) spec is depth 0 and never approaches the bound.
const MaxQueryDepth = 4

// QuerySpecFromProto converts a proto pb.QuerySpec into the engine's proto-free
// vector.QuerySpec (the engine never imports pb). It is the single proto↔struct
// conversion point (the OPS layer owns it; the vector engine takes Go structs).
// It is fail-loud: an unknown fusion method, an empty/unknown leaf oneof, a bad
// per-leaf filter JSON, or a nesting depth over MaxQueryDepth returns an error
// rather than silently defaulting/truncating.
//
// depth is the current nesting level (0 for the top-level spec); each nested
// QuerySpec source recurses at depth+1 and a depth over MaxQueryDepth is rejected.
//
// BACK-COMPAT: the prefetch decode prefers the additive `prefetch_sources` field
// (repeated QuerySource — present only for nested trees); when it is empty it falls
// back to the flat v1 `prefetch` field (repeated QueryLeaf), lifting each leaf into
// a QuerySource{leaf} — so a v1 client (and every existing leaf-only query) decodes
// byte/behaviour-identically to today.
func QuerySpecFromProto(p *pb.QuerySpec, depth int) (vector.QuerySpec, error) {
	if p == nil {
		return vector.QuerySpec{}, fmt.Errorf("ops: nil query spec")
	}
	if depth > MaxQueryDepth {
		return vector.QuerySpec{}, vector.ErrQuerySpecTooDeep
	}
	// Breadth bound (the companion to the depth bound): reject a node carrying more than
	// vector.MaxPrefetchSources prefetch sources fail-loud at DECODE, at EVERY nesting
	// level (the recursion below re-decodes each nested spec through this function). The
	// count is the additive prefetch_sources when present, else the flat v1 prefetch —
	// the same precedence the decode below uses. Structural ⇒ identical regardless of P.
	if n := len(p.GetPrefetchSources()); n > vector.MaxPrefetchSources {
		return vector.QuerySpec{}, vector.ErrTooManyPrefetchSources
	} else if n == 0 && len(p.GetPrefetch()) > vector.MaxPrefetchSources {
		return vector.QuerySpec{}, vector.ErrTooManyPrefetchSources
	}
	method, err := parseQueryFusion(p.GetFusionMethod())
	if err != nil {
		return vector.QuerySpec{}, err
	}
	var mode vector.QueryMode
	switch p.GetMode() {
	case pb.QueryMode_QUERY_MODE_FUSION:
		mode = vector.ModeFusion
	case pb.QueryMode_QUERY_MODE_RERANK:
		mode = vector.ModeRerank
	default:
		return vector.QuerySpec{}, fmt.Errorf("ops: unknown query mode %d", p.GetMode())
	}
	spec := vector.QuerySpec{
		Mode:      mode,
		Method:    method,
		Alpha:     p.GetAlpha(),
		RRFK:      int(p.GetRrfK()),
		K:         int(p.GetK()),
		GroupBy:   p.GetGroupBy(), // empty ⇒ flat query (byte-identical decode)
		GroupSize: int(p.GetGroupSize()),
	}
	// The root leaf is required for RERANK and harmless (empty) for FUSION; decode
	// it when present so the engine validates it for the active mode.
	if p.GetRoot() != nil {
		root, rerr := queryLeafFromProto(p.GetRoot())
		if rerr != nil {
			return vector.QuerySpec{}, rerr
		}
		spec.Root = root
	}
	// Prefer the nested `prefetch_sources` (recursion-capable) when present; else the
	// flat v1 `prefetch` (each leaf lifted into a leaf source — byte-identical path).
	if srcs := p.GetPrefetchSources(); len(srcs) > 0 {
		for _, ps := range srcs {
			src, serr := querySourceFromProto(ps, depth)
			if serr != nil {
				return vector.QuerySpec{}, serr
			}
			spec.Prefetch = append(spec.Prefetch, src)
		}
		return spec, nil
	}
	for _, pl := range p.GetPrefetch() {
		leaf, lerr := queryLeafFromProto(pl)
		if lerr != nil {
			return vector.QuerySpec{}, lerr
		}
		spec.Prefetch = append(spec.Prefetch, vector.QuerySource{Leaf: &leaf})
	}
	return spec, nil
}

// querySourceFromProto converts one proto QuerySource (a leaf OR a nested spec)
// into a vector.QuerySource. A leaf arm → QuerySource{Leaf}; a spec arm →
// QuerySource{Spec} decoded RECURSIVELY at depth+1 (the depth bound is enforced in
// QuerySpecFromProto). Fail-loud on an empty oneof.
func querySourceFromProto(p *pb.QuerySource, depth int) (vector.QuerySource, error) {
	if p == nil {
		return vector.QuerySource{}, fmt.Errorf("ops: nil query source")
	}
	switch src := p.GetSource().(type) {
	case *pb.QuerySource_Leaf:
		leaf, err := queryLeafFromProto(src.Leaf)
		if err != nil {
			return vector.QuerySource{}, err
		}
		return vector.QuerySource{Leaf: &leaf}, nil
	case *pb.QuerySource_Spec:
		sub, err := QuerySpecFromProto(src.Spec, depth+1)
		if err != nil {
			return vector.QuerySource{}, err
		}
		return vector.QuerySource{Spec: &sub}, nil
	default:
		return vector.QuerySource{}, fmt.Errorf("ops: query source has neither a leaf nor a spec")
	}
}

// queryLeafFromProto converts one proto QueryLeaf oneof into a vector.QueryLeaf.
// Fail-loud on an empty oneof or a bad filter JSON.
func queryLeafFromProto(p *pb.QueryLeaf) (vector.QueryLeaf, error) {
	if p == nil {
		return vector.QueryLeaf{}, fmt.Errorf("ops: nil query leaf")
	}
	switch leaf := p.GetLeaf().(type) {
	case *pb.QueryLeaf_Dense:
		d := leaf.Dense
		filter, err := parseFilterJSON(d.GetFilterJson())
		if err != nil {
			return vector.QueryLeaf{}, err
		}
		return vector.QueryLeaf{
			Kind:   vector.LeafDense,
			Dense:  d.GetDense(),
			K:      int(d.GetK()),
			Filter: filter,
			LaneK:  int(d.GetLaneK()),
			// Dense (hnsw) lane is distance-ascending.
			ScoreDesc: false,
		}, nil
	case *pb.QueryLeaf_Sparse:
		s := leaf.Sparse
		filter, err := parseFilterJSON(s.GetFilterJson())
		if err != nil {
			return vector.QueryLeaf{}, err
		}
		return vector.QueryLeaf{
			Kind:   vector.LeafSparse,
			Sparse: vector.SparseVector{Indices: s.GetIndices(), Values: s.GetValues()},
			K:      int(s.GetK()),
			Filter: filter,
			LaneK:  int(s.GetLaneK()),
			// Sparse (inverted-index dot-product) lane is score-descending.
			ScoreDesc: true,
		}, nil
	case *pb.QueryLeaf_NamedDense:
		d := leaf.NamedDense
		filter, err := parseFilterJSON(d.GetFilterJson())
		if err != nil {
			return vector.QueryLeaf{}, err
		}
		return vector.QueryLeaf{
			Kind:   vector.LeafDense,
			Space:  d.GetSpace(),
			Dense:  d.GetDense(),
			K:      int(d.GetK()),
			Filter: filter,
			LaneK:  int(d.GetLaneK()),
			// Named-dense (per-space hnsw) lane is distance-ascending.
			ScoreDesc: false,
		}, nil
	case *pb.QueryLeaf_NamedSparse:
		s := leaf.NamedSparse
		filter, err := parseFilterJSON(s.GetFilterJson())
		if err != nil {
			return vector.QueryLeaf{}, err
		}
		return vector.QueryLeaf{
			Kind:   vector.LeafSparse,
			Space:  s.GetSpace(),
			Sparse: vector.SparseVector{Indices: s.GetIndices(), Values: s.GetValues()},
			K:      int(s.GetK()),
			Filter: filter,
			LaneK:  int(s.GetLaneK()),
			// Named-sparse lane is score-descending.
			ScoreDesc: true,
		}, nil
	case *pb.QueryLeaf_MvMaxsim:
		mv := leaf.MvMaxsim
		filter, err := parseFilterJSON(mv.GetFilterJson())
		if err != nil {
			return vector.QueryLeaf{}, err
		}
		// The MaxSim token query matrix rides as repeated TokenVector (one per
		// query token), the same [][]float32 wire MVSearchRequest carries.
		tokens := make([][]float32, len(mv.GetQuery()))
		for i, tv := range mv.GetQuery() {
			tokens[i] = tv.GetValues()
		}
		return vector.QueryLeaf{
			Kind:   vector.LeafMVMaxSim,
			Tokens: tokens,
			K:      int(mv.GetK()),
			Filter: filter,
			LaneK:  int(mv.GetLaneK()),
			// MaxSim (late-interaction) lane is a descending relevance score.
			ScoreDesc: true,
		}, nil
	case *pb.QueryLeaf_Recommend:
		r := leaf.Recommend
		filter, err := parseFilterJSON(r.GetFilterJson())
		if err != nil {
			return vector.QueryLeaf{}, err
		}
		// Carry the example ids + the strategy. For AVERAGE_VECTOR the coordinator
		// pre-pass derives the dense vector and rewrites this leaf to a dense leaf
		// (distance-ascending, so the recommend leaf is distance-ascending too). For
		// BEST_SCORE the leaf STAYS a LeafRecommend the best-score execLeaf runs (a custom
		// per-candidate scorer, so SCORE-descending like Discover); it carries BOTH the
		// example ids and any already-resolved best_pos/best_neg VECTORS (the coordinator
		// embeds them and clears the ids).
		strategy := vector.RecommendStrategy(r.GetStrategy())
		var recPos, recNeg [][]float32
		if n := len(r.GetBestPos()); n > 0 {
			recPos = make([][]float32, n)
			for i, tv := range r.GetBestPos() {
				recPos[i] = tv.GetValues()
			}
		}
		if n := len(r.GetBestNeg()); n > 0 {
			recNeg = make([][]float32, n)
			for i, tv := range r.GetBestNeg() {
				recNeg[i] = tv.GetValues()
			}
		}
		return vector.QueryLeaf{
			Kind:       vector.LeafRecommend,
			Space:      r.GetSpace(),
			Positive:   r.GetPositive(),
			Negative:   r.GetNegative(),
			K:          int(r.GetK()),
			Filter:     filter,
			Strategy:   strategy,
			RecPosVecs: recPos,
			RecNegVecs: recNeg,
			// AVERAGE_VECTOR is distance-ascending (rewritten to dense); BEST_SCORE is a
			// custom per-candidate scorer, score-descending.
			ScoreDesc: strategy == vector.RecommendBestScore,
		}, nil
	case *pb.QueryLeaf_Discover:
		d := leaf.Discover
		filter, err := parseFilterJSON(d.GetFilterJson())
		if err != nil {
			return vector.QueryLeaf{}, err
		}
		// Carry BOTH the resolved target/context VECTORS and the unresolved
		// target/context IDS: the coordinator resolve pre-pass fills the vector fields
		// from the ids; the execLeaf (DiscoverVecs) consumes the vectors. The discover
		// lane is a custom per-candidate score, so it is SCORE-descending (like MaxSim).
		var context []vector.DiscoverPair
		if n := len(d.GetContext()); n > 0 {
			context = make([]vector.DiscoverPair, n)
			for i, cp := range d.GetContext() {
				context[i] = vector.DiscoverPair{Pos: cp.GetPositive(), Neg: cp.GetNegative()}
			}
		}
		var contextIDs []vector.ContextPair
		if n := len(d.GetContextIds()); n > 0 {
			contextIDs = make([]vector.ContextPair, n)
			for i, cp := range d.GetContextIds() {
				contextIDs[i] = vector.ContextPair{Positive: cp.GetPositive(), Negative: cp.GetNegative()}
			}
		}
		return vector.QueryLeaf{
			Kind:               vector.LeafDiscover,
			Space:              d.GetSpace(),
			DiscoverTarget:     d.GetTarget(),
			DiscoverContext:    context,
			DiscoverTargetID:   d.GetTargetId(),
			DiscoverContextIDs: contextIDs,
			K:                  int(d.GetK()),
			Filter:             filter,
			ScoreDesc:          true,
		}, nil
	default:
		return vector.QueryLeaf{}, fmt.Errorf("ops: query leaf has no dense/sparse/mv payload")
	}
}

// parseQueryFusion maps a fusion-method string to a vector.FusionMethod (empty
// or "rrf" → RRF, "weighted" → Weighted, "dbsf" → DBSF). Unknown is fail-loud so
// a typo never silently degrades to RRF.
func parseQueryFusion(s string) (vector.FusionMethod, error) {
	switch s {
	case "", "rrf":
		return vector.FusionRRF, nil
	case "weighted":
		return vector.FusionWeighted, nil
	case "dbsf":
		return vector.FusionDBSF, nil
	default:
		return 0, fmt.Errorf("ops: unknown fusion method %q", s)
	}
}

// EncodeQueryResult serializes a mode-tagged vector_query engine result:
//
//	FUSION: [mode=0:u8][nLanes:u32]{EncodeHybridResults(lane)}…
//	RERANK: [mode=1:u8]EncodeHybridResults(scored)
//
// FUSION carries the UNFUSED prefetch lanes (the cross-partition coordinator
// unions them and fuses globally; the single-node caller may fuse via
// vector.Fuse). RERANK carries the reranked top-k.
func EncodeQueryResult(qr vector.QueryResult) []byte {
	// A GROUPED query result (Groups populated) is mode-tagged separately and carries
	// the EncodeGroups blob (REUSED from the standalone groups codec). A flat query
	// (Groups nil) encodes EXACTLY as before — byte-identical (the grouped branch is
	// guarded on Groups != nil so the no-group path is unchanged).
	if qr.Groups != nil {
		out := []byte{queryResultModeGrouped}
		return append(out, EncodeGroups(qr.Groups)...)
	}
	switch qr.Mode {
	case vector.ModeRerank:
		out := []byte{QueryResultModeRerank}
		return append(out, EncodeHybridResults(qr.Fused)...)
	default: // ModeFusion
		out := make([]byte, 5)
		out[0] = QueryResultModeFusion
		binary.BigEndian.PutUint32(out[1:], uint32(len(qr.Lanes)))
		for _, lane := range qr.Lanes {
			out = append(out, EncodeHybridResults(lane)...)
		}
		return out
	}
}

// DecodeQueryResult reads a mode-tagged vector_query result produced by
// EncodeQueryResult. For FUSION, Lanes holds the N unfused lanes and Fused is nil
// (the caller/coordinator fuses). For RERANK, Fused holds the reranked top-k and
// Lanes is nil.
func DecodeQueryResult(body []byte) (vector.QueryResult, error) {
	if len(body) < 1 {
		return vector.QueryResult{}, ErrVectorArgsTruncated
	}
	mode := body[0]
	switch mode {
	case queryResultModeGrouped:
		groups, err := DecodeGroups(body[1:])
		if err != nil {
			return vector.QueryResult{}, err
		}
		return vector.QueryResult{Mode: vector.ModeRerank, Groups: groups}, nil
	case QueryResultModeRerank:
		fused, err := DecodeHybridResults(body[1:])
		if err != nil {
			return vector.QueryResult{}, err
		}
		return vector.QueryResult{Mode: vector.ModeRerank, Fused: fused}, nil
	case QueryResultModeFusion:
		if len(body) < 5 {
			return vector.QueryResult{}, ErrVectorArgsTruncated
		}
		nLanes := int(binary.BigEndian.Uint32(body[1:]))
		off := 5
		// A lane costs >= 4 bytes (its own [count:u32] header).
		if !CountFitsIn(nLanes, len(body)-off, 4) {
			return vector.QueryResult{}, ErrVectorArgsTruncated
		}
		lanes := make([][]vector.Result, 0, nLanes)
		for i := 0; i < nLanes; i++ {
			lane, n, err := decodeHybridResultsN(body[off:])
			if err != nil {
				return vector.QueryResult{}, err
			}
			lanes = append(lanes, lane)
			off += n
		}
		return vector.QueryResult{Mode: vector.ModeFusion, Lanes: lanes}, nil
	case QueryResultModeTreeLanes:
		// The per-partition UNFUSED tree-lanes (the flat pre-order spec-tree lane list).
		// Decoded into Lanes (same field as FUSION) — the coordinator re-walks the spec
		// tree (vector.SpecHasNestedFusion) and folds each FUSION node over the global
		// union; the single-shard path runs the SAME re-walk over its one lane list.
		lanes, err := DecodeQueryTreeLanes(body)
		if err != nil {
			return vector.QueryResult{}, err
		}
		return vector.QueryResult{Mode: vector.ModeFusion, Lanes: lanes}, nil
	default:
		return vector.QueryResult{}, fmt.Errorf("ops: unknown query result mode %d", mode)
	}
}

// EncodeQueryTreeLanes serializes the per-partition UNFUSED tree-lanes (the flat
// pre-order spec-tree lane list collectTreeLanesAt produces) as a QueryResultModeTreeLanes
// payload: [mode=4:u8][nLanes:u32]{EncodeHybridResults(lane)}. The body shape mirrors
// the FUSION mode exactly; only the tag distinguishes the EXPANDED tree traversal from
// the top-level lanes, so the coordinator picks the tree-lanes re-walk (vs the flat
// fusion fold) by the spec shape. Used ONLY when vector.SpecHasNestedFusion(spec).
func EncodeQueryTreeLanes(lanes [][]vector.Result) []byte {
	out := make([]byte, 5)
	out[0] = QueryResultModeTreeLanes
	binary.BigEndian.PutUint32(out[1:], uint32(len(lanes))) //nolint:gosec // bounded by spec breadth×depth
	for _, lane := range lanes {
		out = append(out, EncodeHybridResults(lane)...)
	}
	return out
}

// DecodeQueryTreeLanes reads a QueryResultModeTreeLanes payload produced by
// EncodeQueryTreeLanes into the flat pre-order lane list. Fail-loud on a wrong tag or a
// truncated lane block (a corrupt tree-lanes result is never silently treated as flat).
func DecodeQueryTreeLanes(body []byte) ([][]vector.Result, error) {
	if len(body) < 5 || body[0] != QueryResultModeTreeLanes {
		return nil, fmt.Errorf("ops: not a tree-lanes query result")
	}
	nLanes := int(binary.BigEndian.Uint32(body[1:]))
	off := 5
	// A lane costs >= 4 bytes (its own [count:u32] header).
	if !CountFitsIn(nLanes, len(body)-off, 4) {
		return nil, ErrVectorArgsTruncated
	}
	lanes := make([][]vector.Result, 0, nLanes)
	for i := 0; i < nLanes; i++ {
		lane, n, err := decodeHybridResultsN(body[off:])
		if err != nil {
			return nil, err
		}
		lanes = append(lanes, lane)
		off += n
	}
	return lanes, nil
}

// decodeQueryResultN reads a FLAT mode-tagged query result (FUSION lanes / RERANK
// fused — the only shapes a grouped fan-out partition prefixes) and returns the bytes
// consumed, so a trailing block (the grouped fan-out id→key map) can be parsed after
// it. A grouped-tagged body is NOT valid here (the partition flat result is always
// ungrouped) — fail-loud.
func decodeQueryResultN(body []byte) (vector.QueryResult, int, error) {
	if len(body) < 1 {
		return vector.QueryResult{}, 0, ErrVectorArgsTruncated
	}
	mode := body[0]
	switch mode {
	case QueryResultModeRerank:
		fused, n, err := decodeHybridResultsN(body[1:])
		if err != nil {
			return vector.QueryResult{}, 0, err
		}
		return vector.QueryResult{Mode: vector.ModeRerank, Fused: fused}, 1 + n, nil
	case QueryResultModeFusion:
		if len(body) < 5 {
			return vector.QueryResult{}, 0, ErrVectorArgsTruncated
		}
		nLanes := int(binary.BigEndian.Uint32(body[1:]))
		off := 5
		// A lane costs >= 4 bytes (its own [count:u32] header).
		if !CountFitsIn(nLanes, len(body)-off, 4) {
			return vector.QueryResult{}, 0, ErrVectorArgsTruncated
		}
		lanes := make([][]vector.Result, 0, nLanes)
		for i := 0; i < nLanes; i++ {
			lane, n, err := decodeHybridResultsN(body[off:])
			if err != nil {
				return vector.QueryResult{}, 0, err
			}
			lanes = append(lanes, lane)
			off += n
		}
		return vector.QueryResult{Mode: vector.ModeFusion, Lanes: lanes}, off, nil
	default:
		return vector.QueryResult{}, 0, fmt.Errorf("ops: unexpected flat query result mode %d", mode)
	}
}

// groupedFanOutMarker tags a GROUPED fan-out result: the FLAT mode-tagged query
// result (FUSION lanes / RERANK fused — UNGROUPED) followed by a self-delimiting
// id→group-key map. The coordinator decodes the flat result with its NORMAL reader,
// runs its NORMAL fuse/rerank merge to get the global ordered pool, then maps each
// pooled id→its key and folds via GroupDocuments ONCE. The flat-result prefix is the
// byte-identical EncodeQueryResult output so the no-group path is never touched.
const groupedFanOutMarker uint8 = 3

// EncodeQueryResultGroupedFanOut encodes the PER-PARTITION grouped fan-out result:
// the byte-identical flat EncodeQueryResult(qr) (UNGROUPED — FUSION lanes / RERANK
// fused over the wide pool) preceded by groupedFanOutMarker, followed by the per-id
// group-key map [count:u32]{[id:u64][keyLen:u32][keyJSON]} (each key JSON-marshaled
// like EncodeGroups' Key). The coordinator merges the flat result with its UNCHANGED
// merge, then maps the ordered ids→keys and groups ONCE.
func EncodeQueryResultGroupedFanOut(qr vector.QueryResult, keys map[uint64]vector.Value) []byte {
	flat := EncodeQueryResult(vector.QueryResult{Mode: qr.Mode, Fused: qr.Fused, Lanes: qr.Lanes})
	out := make([]byte, 0, 1+len(flat)+4+len(keys)*16)
	out = append(out, groupedFanOutMarker)
	out = append(out, flat...)
	var cnt [4]byte
	binary.BigEndian.PutUint32(cnt[:], uint32(len(keys))) //nolint:gosec // bounded by pool
	out = append(out, cnt[:]...)
	for id, v := range keys {
		var idb [8]byte
		binary.BigEndian.PutUint64(idb[:], id)
		out = append(out, idb[:]...)
		kb, _ := json.Marshal(v)
		var kl [4]byte
		binary.BigEndian.PutUint32(kl[:], uint32(len(kb))) //nolint:gosec // bounded JSON scalar
		out = append(out, kl[:]...)
		out = append(out, kb...)
	}
	return out
}

// DecodeQueryResultGroupedFanOut reads a block produced by
// EncodeQueryResultGroupedFanOut: the flat (UNGROUPED) QueryResult plus the per-id
// group-key map. Fail-loud on a missing marker or a truncated map (a corrupt grouped
// result is never silently treated as ungrouped).
func DecodeQueryResultGroupedFanOut(body []byte) (vector.QueryResult, map[uint64]vector.Value, error) {
	if len(body) < 1 || body[0] != groupedFanOutMarker {
		return vector.QueryResult{}, nil, fmt.Errorf("ops: not a grouped fan-out result")
	}
	qr, n, err := decodeQueryResultN(body[1:])
	if err != nil {
		return vector.QueryResult{}, nil, err
	}
	off := 1 + n
	if len(body) < off+4 {
		return vector.QueryResult{}, nil, ErrVectorArgsTruncated
	}
	count := int(binary.BigEndian.Uint32(body[off:]))
	off += 4
	// An entry costs >= 12 bytes ([id:u64] + the value's own 4-byte header).
	if !CountFitsIn(count, len(body)-off, 12) {
		return vector.QueryResult{}, nil, ErrVectorArgsTruncated
	}
	keys := make(map[uint64]vector.Value, count)
	for i := 0; i < count; i++ {
		if len(body) < off+12 {
			return vector.QueryResult{}, nil, ErrVectorArgsTruncated
		}
		id := binary.BigEndian.Uint64(body[off:])
		off += 8
		kl := int(binary.BigEndian.Uint32(body[off:]))
		off += 4
		if len(body) < off+kl {
			return vector.QueryResult{}, nil, ErrVectorArgsTruncated
		}
		var v vector.Value
		if uerr := json.Unmarshal(body[off:off+kl], &v); uerr != nil {
			return vector.QueryResult{}, nil, fmt.Errorf("ops: decode group key: %w", uerr)
		}
		off += kl
		keys[id] = v
	}
	return qr, keys, nil
}
