// SPDX-License-Identifier: Apache-2.0

package mcp

import (
	"context"
	"encoding/json"
	"fmt"

	"github.com/rostamlabs/rostam"
	"github.com/rostamlabs/rostam/vector"
)

// rejectMemoryCollection guards the write tools (create_collection, upsert)
// against touching mcp_memory directly: that collection's schema, reserved
// metadata fields, and embedder-identity bootstrap are all owned by the
// remember/recall/forget tools in memory.go, and a raw write here could
// corrupt them. Reads (search, get) have no such hazard and are allowed
// through unchanged.
func rejectMemoryCollection(collection string) error {
	if collection == memCollection {
		return fmt.Errorf("mcp: %q is reserved for the memory tools; use remember/recall/forget instead", memCollection)
	}
	return nil
}

// registerDBTools registers the generic vector-DB tools: create_collection,
// upsert, search, and get. Unlike the memory tools, these operate on
// whatever collection the caller names.
func (s *Server) registerDBTools() {
	s.register(toolDef{
		Name:        "create_collection",
		Description: "Create a new vector collection.",
		InputSchema: map[string]any{
			"type": "object",
			"properties": map[string]any{
				"name":      map[string]any{"type": "string", "description": "collection name"},
				"dim":       map[string]any{"type": "integer", "description": "vector dimensionality"},
				"metric":    map[string]any{"type": "string", "description": `distance metric: "cosine", "l2", or "dot" (default "cosine")`},
				"full_text": map[string]any{"type": "boolean", "description": "enable BM25 full-text indexing (default true)"},
			},
			"required": []any{"name", "dim"},
		},
		Handler: s.handleCreateCollection,
	})
	s.register(toolDef{
		Name:        "upsert",
		Description: "Insert or update a point in a collection.",
		InputSchema: map[string]any{
			"type": "object",
			"properties": map[string]any{
				"collection": map[string]any{"type": "string"},
				"id":         map[string]any{"type": "integer", "description": "point id"},
				"vector":     map[string]any{"type": "array", "items": map[string]any{"type": "number"}, "description": "explicit embedding; omit to auto-embed content when an embedder is configured"},
				"content":    map[string]any{"type": "string", "description": "text content (BM25-indexed, and auto-embedded when vector is omitted)"},
				"metadata":   map[string]any{"type": "object", "description": "optional metadata"},
			},
			"required": []any{"collection", "id"},
		},
		Handler: s.handleUpsert,
	})
	s.register(toolDef{
		Name:        "search",
		Description: "Search a collection by text, dense vector, or a hybrid fusion of both.",
		InputSchema: map[string]any{
			"type": "object",
			"properties": map[string]any{
				"collection": map[string]any{"type": "string"},
				"mode":       map[string]any{"type": "string", "description": `"text", "dense", or "hybrid" (default: "text" with no embedder configured, "hybrid" with one)`},
				"query_text": map[string]any{"type": "string", "description": "query text; required for text mode, and used to embed the dense side of dense/hybrid mode when vector is omitted"},
				"vector":     map[string]any{"type": "array", "items": map[string]any{"type": "number"}, "description": "explicit query vector for dense/hybrid mode"},
				"k":          map[string]any{"type": "integer", "description": "max hits to return (default 10)"},
				"filter":     map[string]any{"type": "object", "description": `optional metadata filter using the tagged value form, e.g. {"op":"eq","field":"lang","value":{"kind":"string","str":"en"}}`},
			},
			"required": []any{"collection"},
		},
		Handler: s.handleSearch,
	})
	s.register(toolDef{
		Name:        "get",
		Description: "Fetch points from a collection by id.",
		InputSchema: map[string]any{
			"type": "object",
			"properties": map[string]any{
				"collection":  map[string]any{"type": "string"},
				"ids":         map[string]any{"type": "array", "items": map[string]any{"type": "integer"}},
				"with_vector": map[string]any{"type": "boolean", "description": "include each point's raw vector (default false)"},
			},
			"required": []any{"collection", "ids"},
		},
		Handler: s.handleGet,
	})
}

// registerDestructiveTools registers delete and delete_by_filter. Called
// only when s.destructive is set: unlike the write tools in
// registerDBTools, these are absent from tools/list entirely on a
// non-destructive server rather than merely refusing at call time.
func (s *Server) registerDestructiveTools() {
	s.register(toolDef{
		Name:        "delete",
		Description: `Delete points from a collection by id. Returns {"deleted":[...],"missing":[...],"errors":[...]}; a per-id delete failure does not abort the rest of the batch.`,
		InputSchema: map[string]any{
			"type": "object",
			"properties": map[string]any{
				"collection": map[string]any{"type": "string"},
				"ids":        map[string]any{"type": "array", "items": map[string]any{"type": "integer"}},
			},
			"required": []any{"collection", "ids"},
		},
		Handler: s.handleDelete,
	})
	s.register(toolDef{
		Name:        "delete_by_filter",
		Description: "Delete all points in a collection matching a metadata filter.",
		InputSchema: map[string]any{
			"type": "object",
			"properties": map[string]any{
				"collection": map[string]any{"type": "string"},
				"filter":     map[string]any{"type": "object", "description": `metadata filter using the tagged value form, e.g. {"op":"eq","field":"lang","value":{"kind":"string","str":"en"}}; a match-all (empty) filter is refused`},
			},
			"required": []any{"collection", "filter"},
		},
		Handler: s.handleDeleteByFilter,
	})
}

// deleteArgs is the delete tool's decoded input.
type deleteArgs struct {
	Collection string   `json:"collection"`
	IDs        []uint64 `json:"ids"`
}

func (s *Server) handleDelete(ctx context.Context, raw json.RawMessage) (any, error) {
	var args deleteArgs
	if err := json.Unmarshal(raw, &args); err != nil {
		return nil, fmt.Errorf("mcp: bad delete args: %w", err)
	}
	if args.Collection == "" {
		return nil, fmt.Errorf("mcp: delete: collection is required")
	}
	if err := rejectMemoryCollection(args.Collection); err != nil {
		return nil, err
	}
	if len(args.IDs) == 0 {
		return nil, fmt.Errorf("mcp: delete: ids is required and must be non-empty")
	}

	// Every id's delete is attempted even after an earlier one fails (mirrors
	// forget's approach in memory.go): a batch with a mix of good and bad ids
	// must still report which ones actually deleted rather than aborting and
	// leaving the caller unable to tell "nothing happened" from "some of
	// this went through". Unlike forget, the failures are surfaced in the
	// result's errors field rather than as a top-level tool error, so the
	// partial deleted/missing outcome isn't discarded.
	deleted := make([]uint64, 0, len(args.IDs))
	missing := make([]uint64, 0)
	var errs []string
	for _, id := range args.IDs {
		ok, err := s.store.VectorDelete(ctx, args.Collection, id)
		if err != nil {
			errs = append(errs, fmt.Sprintf("id %d: %s", id, err))
			continue
		}
		if ok {
			deleted = append(deleted, id)
		} else {
			missing = append(missing, id)
		}
	}
	return deleteResult{Deleted: deleted, Missing: missing, Errors: errs}, nil
}

// deleteResult is the delete tool's response shape. Errors is omitted on
// full success; when present, each entry names the id it came from.
type deleteResult struct {
	Deleted []uint64 `json:"deleted"`
	Missing []uint64 `json:"missing"`
	Errors  []string `json:"errors,omitempty"`
}

// deleteByFilterArgs is the delete_by_filter tool's decoded input.
type deleteByFilterArgs struct {
	Collection string          `json:"collection"`
	Filter     json.RawMessage `json:"filter"`
}

func (s *Server) handleDeleteByFilter(ctx context.Context, raw json.RawMessage) (any, error) {
	var args deleteByFilterArgs
	if err := json.Unmarshal(raw, &args); err != nil {
		return nil, fmt.Errorf("mcp: bad delete_by_filter args: %w", err)
	}
	if args.Collection == "" {
		return nil, fmt.Errorf("mcp: delete_by_filter: collection is required")
	}
	if err := rejectMemoryCollection(args.Collection); err != nil {
		return nil, err
	}

	f, err := parseFilter(args.Filter)
	if err != nil {
		return nil, err
	}
	// The gate being open (Destructive=true) authorizes targeted deletes,
	// not a blanket wipe: an empty/zero filter matches every point in the
	// collection, so refuse it outright rather than trusting the caller to
	// mean it.
	if f.IsZero() {
		return nil, fmt.Errorf("mcp: delete_by_filter: refusing match-all delete_by_filter; provide a filter")
	}

	n, err := s.store.VectorDeleteByFilter(ctx, args.Collection, f)
	if err != nil {
		return nil, fmt.Errorf("mcp: delete_by_filter: %w", err)
	}
	return map[string]any{"deleted_count": n}, nil
}

// createCollectionArgs is the create_collection tool's decoded input.
type createCollectionArgs struct {
	Name     string `json:"name"`
	Dim      int    `json:"dim"`
	Metric   string `json:"metric"`
	FullText *bool  `json:"full_text"`
}

func (s *Server) handleCreateCollection(ctx context.Context, raw json.RawMessage) (any, error) {
	var args createCollectionArgs
	if err := json.Unmarshal(raw, &args); err != nil {
		return nil, fmt.Errorf("mcp: bad create_collection args: %w", err)
	}
	if args.Name == "" {
		return nil, fmt.Errorf("mcp: create_collection: name is required")
	}
	if err := rejectMemoryCollection(args.Name); err != nil {
		return nil, err
	}
	if args.Dim <= 0 {
		return nil, fmt.Errorf("mcp: create_collection: dim is required and must be positive")
	}

	// vector.DefaultConfig fills in the HNSW build knobs (M/EfConstruction/
	// EfSearch) that Config.Validate requires but a caller has no opinion on
	// here; only Dim/Metric/FullText come from the tool args.
	cfg := vector.DefaultConfig()
	cfg.Dim = args.Dim
	switch args.Metric {
	case "", "cosine":
		cfg.Metric = vector.Cosine
	case "l2":
		cfg.Metric = vector.L2
	case "dot":
		cfg.Metric = vector.DotProduct
	default:
		return nil, fmt.Errorf("mcp: create_collection: unknown metric %q (want cosine, l2, or dot)", args.Metric)
	}
	fullText := true
	if args.FullText != nil {
		fullText = *args.FullText
	}
	if fullText {
		cfg.FullText = &vector.FullTextConfig{}
	}

	if err := s.store.CreateCollection(ctx, args.Name, cfg); err != nil {
		return nil, fmt.Errorf("mcp: create_collection: %w", err)
	}
	return map[string]any{"created": args.Name}, nil
}

// upsertArgs is the upsert tool's decoded input.
type upsertArgs struct {
	Collection string                     `json:"collection"`
	ID         uint64                     `json:"id"`
	Vector     []float32                  `json:"vector"`
	Content    string                     `json:"content"`
	Metadata   map[string]json.RawMessage `json:"metadata"`
}

func (s *Server) handleUpsert(ctx context.Context, raw json.RawMessage) (any, error) {
	var args upsertArgs
	if err := json.Unmarshal(raw, &args); err != nil {
		return nil, fmt.Errorf("mcp: bad upsert args: %w", err)
	}
	if args.Collection == "" {
		return nil, fmt.Errorf("mcp: upsert: collection is required")
	}
	if err := rejectMemoryCollection(args.Collection); err != nil {
		return nil, err
	}

	vec := args.Vector
	if vec == nil {
		if args.Content == "" || !s.hybrid {
			return nil, fmt.Errorf("mcp: upsert: provide a vector, or set an embedder and provide content")
		}
		vecs, err := s.emb.Embed(ctx, []string{args.Content})
		if err != nil {
			return nil, fmt.Errorf("mcp: upsert: embed content: %w", err)
		}
		vec = vecs[0]
	}

	md, err := jsonToMetadata(args.Metadata)
	if err != nil {
		return nil, err
	}

	if err := s.store.VectorUpsert(ctx, args.Collection, args.ID, vec, args.Content, rostam.VectorInsertOpts{Metadata: md}); err != nil {
		return nil, fmt.Errorf("mcp: upsert: %w", err)
	}
	return map[string]any{"id": args.ID}, nil
}

// searchArgs is the search tool's decoded input.
type searchArgs struct {
	Collection string          `json:"collection"`
	Mode       string          `json:"mode"`
	QueryText  string          `json:"query_text"`
	Vector     []float32       `json:"vector"`
	K          int             `json:"k"`
	Filter     json.RawMessage `json:"filter"`
}

func (s *Server) handleSearch(ctx context.Context, raw json.RawMessage) (any, error) {
	var args searchArgs
	if err := json.Unmarshal(raw, &args); err != nil {
		return nil, fmt.Errorf("mcp: bad search args: %w", err)
	}
	if args.Collection == "" {
		return nil, fmt.Errorf("mcp: search: collection is required")
	}
	k := args.K
	if k == 0 {
		k = 10
	}
	f, err := parseFilter(args.Filter)
	if err != nil {
		return nil, err
	}

	mode := args.Mode
	if mode == "" {
		if s.hybrid {
			mode = "hybrid"
		} else {
			mode = "text"
		}
	}

	switch mode {
	case "text":
		if args.QueryText == "" {
			return nil, fmt.Errorf("mcp: search: query_text is required for text mode")
		}
		docs, _, err := s.store.VectorSearchText(ctx, args.Collection, args.QueryText, k, rostam.VectorSearchOpts{Filter: f})
		if err != nil {
			return nil, fmt.Errorf("mcp: search: %w", err)
		}
		return map[string]any{"hits": docsToHits(docs)}, nil

	case "dense":
		vec, err := s.searchVector(ctx, args.Vector, args.QueryText)
		if err != nil {
			return nil, err
		}
		docs, _, err := s.store.VectorSearchDocs(ctx, args.Collection, vec, k, rostam.VectorSearchOpts{Filter: f})
		if err != nil {
			return nil, fmt.Errorf("mcp: search: %w", err)
		}
		return map[string]any{"hits": docsToHits(docs)}, nil

	case "hybrid":
		vec, err := s.searchVector(ctx, args.Vector, args.QueryText)
		if err != nil {
			return nil, err
		}
		hits, err := s.hybridDocs(ctx, args.Collection, vec, args.QueryText, k, f)
		if err != nil {
			return nil, fmt.Errorf("mcp: search: %w", err)
		}
		return map[string]any{"hits": hits}, nil

	default:
		return nil, fmt.Errorf("mcp: search: unknown mode %q (want text, dense, or hybrid)", mode)
	}
}

// searchVector resolves the dense query vector for dense/hybrid search: an
// explicit vector wins; otherwise queryText is embedded, which requires a
// real embedder to be configured.
func (s *Server) searchVector(ctx context.Context, explicit []float32, queryText string) ([]float32, error) {
	if explicit != nil {
		return explicit, nil
	}
	if !s.hybrid || queryText == "" {
		return nil, fmt.Errorf("mcp: search: provide a vector, or set an embedder and provide query_text")
	}
	vecs, err := s.emb.Embed(ctx, []string{queryText})
	if err != nil {
		return nil, fmt.Errorf("mcp: search: embed query_text: %w", err)
	}
	return vecs[0], nil
}

// searchHit is one hit from the generic search tool: id, content, relevance
// score, distance, and metadata. Distance is meaningful for dense/hybrid
// search (nearest-neighbor distance) but has no meaning for a BM25-only
// recall or a scroll listing — that's why the memory tools use the narrower
// memoryHit shape (memory.go) instead of this type.
type searchHit struct {
	ID       uint64         `json:"id"`
	Content  string         `json:"content"`
	Score    float32        `json:"score"`
	Distance float32        `json:"distance"`
	Metadata map[string]any `json:"metadata,omitempty"`
}

// docsToHits converts VectorDocument results (text/dense search, which
// already carry content and metadata) into the tool's hit shape.
func docsToHits(docs []rostam.VectorDocument) []searchHit {
	hits := make([]searchHit, len(docs))
	for i, d := range docs {
		hits[i] = searchHit{ID: d.ID, Content: d.Content, Score: d.Score, Distance: d.Distance, Metadata: metadataToJSON(d.Metadata)}
	}
	return hits
}

// hybridDocs runs a dense+BM25 fusion search against collection and joins
// the results with a batch fetch: VectorHybridText returns only ids/scores
// (no content/metadata), so the hits are assembled against a batch fetch by
// id. GetBatch's return order is not guaranteed to match the fusion ranking,
// hence the id->point map. Shared by the generic search tool's hybrid mode
// and memory's recallHybrid, which narrows the result down to memoryHit's
// shape (dropping Distance, stripping reserved metadata) itself.
func (s *Server) hybridDocs(ctx context.Context, collection string, dense []float32, query string, k int, f rostam.VectorFilter) ([]searchHit, error) {
	results, _, err := s.store.VectorHybridText(ctx, collection, dense, query, k, rostam.VectorHybridOpts{Filter: f})
	if err != nil {
		return nil, fmt.Errorf("hybrid search: %w", err)
	}
	if len(results) == 0 {
		return []searchHit{}, nil
	}

	ids := make([]uint64, len(results))
	for i, r := range results {
		ids[i] = r.ID
	}
	points, _, err := s.store.VectorGetBatch(ctx, collection, ids, false, true)
	if err != nil {
		return nil, fmt.Errorf("hybrid search: fetching hit content: %w", err)
	}
	byID := make(map[uint64]rostam.BatchGetPoint, len(points))
	for _, p := range points {
		byID[p.ID] = p
	}

	hits := make([]searchHit, 0, len(results))
	for _, r := range results {
		p, ok := byID[r.ID]
		if !ok {
			continue // point vanished between the fusion search and the batch fetch
		}
		var content string
		if cv, ok := p.Meta["$content"]; ok && cv.Kind == vector.ValueString {
			content = cv.Str
		}
		hits = append(hits, searchHit{ID: r.ID, Content: content, Score: r.Score, Distance: r.Distance, Metadata: metadataToJSON(p.Meta)})
	}
	return hits, nil
}

// getArgs is the get tool's decoded input.
type getArgs struct {
	Collection string   `json:"collection"`
	IDs        []uint64 `json:"ids"`
	WithVector bool     `json:"with_vector"`
}

// getPoint is one point returned by the get tool.
type getPoint struct {
	ID       uint64         `json:"id"`
	Vector   []float32      `json:"vector,omitempty"`
	Content  string         `json:"content"`
	Metadata map[string]any `json:"metadata,omitempty"`
}

func (s *Server) handleGet(ctx context.Context, raw json.RawMessage) (any, error) {
	var args getArgs
	if err := json.Unmarshal(raw, &args); err != nil {
		return nil, fmt.Errorf("mcp: bad get args: %w", err)
	}
	if args.Collection == "" {
		return nil, fmt.Errorf("mcp: get: collection is required")
	}
	if len(args.IDs) == 0 {
		return nil, fmt.Errorf("mcp: get: ids is required and must be non-empty")
	}

	points, missing, err := s.store.VectorGetBatch(ctx, args.Collection, args.IDs, args.WithVector, true)
	if err != nil {
		return nil, fmt.Errorf("mcp: get: %w", err)
	}

	out := make([]getPoint, len(points))
	for i, p := range points {
		var content string
		if cv, ok := p.Meta["$content"]; ok && cv.Kind == vector.ValueString {
			content = cv.Str
		}
		gp := getPoint{ID: p.ID, Content: content, Metadata: metadataToJSON(p.Meta)}
		if args.WithVector {
			gp.Vector = p.Vec
		}
		out[i] = gp
	}
	if missing == nil {
		missing = []uint64{}
	}
	return map[string]any{"points": out, "missing": missing}, nil
}
