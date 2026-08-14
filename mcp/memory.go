// SPDX-License-Identifier: Apache-2.0

package mcp

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"sort"
	"strings"
	"time"

	"github.com/cespare/xxhash/v2"

	"github.com/rostamlabs/rostam"
	"github.com/rostamlabs/rostam/vector"
)

// Reserved names for the memory subsystem's collection, KV bookkeeping keys,
// and metadata fields. Callers of remember/recall never see these directly.
const (
	memCollection = "mcp_memory"       // the vector collection backing remember/recall
	nsField       = "__ns"             // metadata field: the memory's namespace
	createdField  = "__created_unix"   // metadata field: unix-seconds creation time
	defaultNS     = "default"          // namespace used when the caller omits one
	kvEmbedder    = "__mcp/embedder"   // KV key: the embedIdentity this store was bootstrapped with
	kvNamespaces  = "__mcp/namespaces" // KV key: JSON array of known namespaces
)

// embedIdentity fingerprints the embedder a data dir's mcp_memory collection
// was created with. It is stored once at bootstrap and checked on every
// subsequent NewServer so a config change (a different model, dimension, or
// BM25-only <-> hybrid switch) fails loudly instead of silently corrupting
// search results or crashing on a dimension mismatch deep in the vector index.
type embedIdentity struct {
	Model  string `json:"model"`
	Dim    int    `json:"dim"`
	Hybrid bool   `json:"hybrid"`
}

// checkEmbedderIdentity compares the configured embedder against the one
// recorded in KV, if any. No record yet means the memory collection has
// never been bootstrapped, so there is nothing to conflict with.
func (s *Server) checkEmbedderIdentity(ctx context.Context) error {
	raw, err := s.store.Get(ctx, []byte(kvEmbedder))
	if err != nil {
		if errors.Is(err, rostam.ErrNotFound) {
			return nil
		}
		return fmt.Errorf("mcp: reading stored embedder identity: %w", err)
	}
	var stored embedIdentity
	if err := json.Unmarshal(raw, &stored); err != nil {
		return fmt.Errorf("mcp: decoding stored embedder identity: %w", err)
	}
	want := embedIdentity{Model: s.emb.Model(), Dim: s.emb.Dim(), Hybrid: s.hybrid}
	if stored != want {
		return fmt.Errorf("mcp: embedder mismatch: this data dir's memory was bootstrapped with model=%q dim=%d hybrid=%v, but this run is configured with model=%q dim=%d hybrid=%v; unset the embedder configuration to go back to the original mode, point -data at a different directory, or wipe the existing data dir to start over",
			stored.Model, stored.Dim, stored.Hybrid, want.Model, want.Dim, want.Hybrid)
	}
	return nil
}

// ensureMemory lazily bootstraps the mcp_memory collection on first use:
// creates it, records the embedder identity that owns it, and seeds an empty
// namespace list. Guarded by s.memOnce so concurrent tool calls only
// bootstrap once; the result (including any error) is cached in s.memErr for
// every caller.
func (s *Server) ensureMemory(ctx context.Context) error {
	s.memOnce.Do(func() {
		_, err := s.store.Get(ctx, []byte(kvEmbedder))
		if err == nil {
			// Already bootstrapped; checkEmbedderIdentity already verified this
			// run's embedder matches, so there is nothing left to do.
			return
		}
		if !errors.Is(err, rostam.ErrNotFound) {
			s.memErr = fmt.Errorf("mcp: reading stored embedder identity: %w", err)
			return
		}
		// vector.DefaultConfig fills in the HNSW build knobs (M/EfConstruction/
		// EfSearch) that Config.Validate requires but the memory collection has
		// no opinion on; only Dim/Metric/FullText are memory-specific.
		cfg := vector.DefaultConfig()
		cfg.Dim, cfg.Metric, cfg.FullText = s.emb.Dim(), vector.Cosine, &vector.FullTextConfig{}
		if err := s.store.CreateCollection(ctx, memCollection, cfg); err != nil {
			s.memErr = fmt.Errorf("mcp: creating memory collection: %w", err)
			return
		}
		id := embedIdentity{Model: s.emb.Model(), Dim: s.emb.Dim(), Hybrid: s.hybrid}
		b, err := json.Marshal(id)
		if err != nil {
			s.memErr = fmt.Errorf("mcp: encoding embedder identity: %w", err)
			return
		}
		if err := s.store.Put(ctx, []byte(kvEmbedder), b, 0); err != nil {
			s.memErr = fmt.Errorf("mcp: storing embedder identity: %w", err)
			return
		}
		if err := s.store.Put(ctx, []byte(kvNamespaces), []byte("[]"), 0); err != nil {
			s.memErr = fmt.Errorf("mcp: storing namespace list: %w", err)
			return
		}
	})
	return s.memErr
}

// memoryID derives a memory's point id from its namespace and content, so
// remembering the same fact twice (in the same namespace) upserts the same
// point instead of accumulating duplicates.
func memoryID(ns, content string) uint64 {
	return xxhash.Sum64String(ns + "\x00" + content)
}

// namespaces returns the known namespace list. An absent KV key (memory
// never bootstrapped) is reported as an empty list, not an error.
func (s *Server) namespaces(ctx context.Context) ([]string, error) {
	raw, err := s.store.Get(ctx, []byte(kvNamespaces))
	if err != nil {
		if errors.Is(err, rostam.ErrNotFound) {
			return nil, nil
		}
		return nil, fmt.Errorf("mcp: reading namespace list: %w", err)
	}
	var ns []string
	if err := json.Unmarshal(raw, &ns); err != nil {
		return nil, fmt.Errorf("mcp: decoding namespace list: %w", err)
	}
	return ns, nil
}

// addNamespace records ns in the namespace list if it is not already there,
// keeping the list sorted.
func (s *Server) addNamespace(ctx context.Context, ns string) error {
	cur, err := s.namespaces(ctx)
	if err != nil {
		return err
	}
	for _, n := range cur {
		if n == ns {
			return nil
		}
	}
	cur = append(cur, ns)
	sort.Strings(cur)
	b, err := json.Marshal(cur)
	if err != nil {
		return fmt.Errorf("mcp: encoding namespace list: %w", err)
	}
	return s.store.Put(ctx, []byte(kvNamespaces), b, 0)
}

// memoryHit is one recalled (or listed, Task 5) memory: its id, content,
// relevance score, and user metadata with the reserved fields stripped.
type memoryHit struct {
	ID       uint64         `json:"id"`
	Content  string         `json:"content"`
	Score    float32        `json:"score"`
	Metadata map[string]any `json:"metadata,omitempty"`
}

// registerMemoryTools registers remember and recall. Task 5 appends
// forget/list_memories/list_namespaces to the same registrar.
func (s *Server) registerMemoryTools() {
	s.register(toolDef{
		Name:        "remember",
		Description: "Store a fact in persistent memory for later recall.",
		InputSchema: map[string]any{
			"type": "object",
			"properties": map[string]any{
				"content":   map[string]any{"type": "string", "description": "the fact to store"},
				"namespace": map[string]any{"type": "string", "description": `isolation namespace (default "default")`},
				"metadata":  map[string]any{"type": "object", "description": "optional extra metadata to attach"},
			},
			"required": []any{"content"},
		},
		Handler: s.handleRemember,
	})
	s.register(toolDef{
		Name:        "recall",
		Description: "Search stored memories by relevance to a query, scoped to a namespace.",
		InputSchema: map[string]any{
			"type": "object",
			"properties": map[string]any{
				"query":     map[string]any{"type": "string", "description": "search text"},
				"namespace": map[string]any{"type": "string", "description": `restrict the search to this namespace (default "default")`},
				"k":         map[string]any{"type": "integer", "description": "max hits to return (default 5)"},
				"filter":    map[string]any{"type": "object", "description": "optional metadata filter, ANDed with the namespace"},
			},
			"required": []any{"query"},
		},
		Handler: s.handleRecall,
	})
}

// rememberArgs is the remember tool's decoded input.
type rememberArgs struct {
	Content   string                     `json:"content"`
	Namespace string                     `json:"namespace"`
	Metadata  map[string]json.RawMessage `json:"metadata"`
}

func (s *Server) handleRemember(ctx context.Context, raw json.RawMessage) (any, error) {
	var args rememberArgs
	if err := json.Unmarshal(raw, &args); err != nil {
		return nil, fmt.Errorf("mcp: bad remember args: %w", err)
	}
	if strings.TrimSpace(args.Content) == "" {
		return nil, fmt.Errorf("mcp: remember: content is required")
	}
	ns := args.Namespace
	if ns == "" {
		ns = defaultNS
	}
	for _, reserved := range [...]string{nsField, createdField, "$content"} {
		if _, ok := args.Metadata[reserved]; ok {
			return nil, fmt.Errorf("mcp: remember: metadata key %q is reserved", reserved)
		}
	}

	if err := s.ensureMemory(ctx); err != nil {
		return nil, err
	}

	md, err := jsonToMetadata(args.Metadata)
	if err != nil {
		return nil, err
	}
	if md == nil {
		md = make(rostam.VectorMetadata, 2)
	}
	md[nsField] = vector.NewString(ns)
	md[createdField] = vector.NewInt(time.Now().Unix())

	vecs, err := s.emb.Embed(ctx, []string{args.Content})
	if err != nil {
		return nil, fmt.Errorf("mcp: embed content: %w", err)
	}

	id := memoryID(ns, args.Content)
	if err := s.store.VectorUpsert(ctx, memCollection, id, vecs[0], args.Content, rostam.VectorInsertOpts{Metadata: md}); err != nil {
		return nil, fmt.Errorf("mcp: remember: %w", err)
	}
	if err := s.addNamespace(ctx, ns); err != nil {
		return nil, err
	}
	return map[string]any{"id": id, "namespace": ns}, nil
}

// recallArgs is the recall tool's decoded input.
type recallArgs struct {
	Query     string          `json:"query"`
	Namespace string          `json:"namespace"`
	K         int             `json:"k"`
	Filter    json.RawMessage `json:"filter"`
}

func (s *Server) handleRecall(ctx context.Context, raw json.RawMessage) (any, error) {
	var args recallArgs
	if err := json.Unmarshal(raw, &args); err != nil {
		return nil, fmt.Errorf("mcp: bad recall args: %w", err)
	}
	if strings.TrimSpace(args.Query) == "" {
		return nil, fmt.Errorf("mcp: recall: query is required")
	}
	ns := args.Namespace
	if ns == "" {
		ns = defaultNS
	}
	k := args.K
	if k == 0 {
		k = 5
	}

	if err := s.ensureMemory(ctx); err != nil {
		return nil, err
	}

	userFilter, err := parseFilter(args.Filter)
	if err != nil {
		return nil, err
	}
	nsFilter := rostam.VectorFilter{Op: vector.FilterEq, Field: nsField, Value: vector.NewString(ns)}
	f := nsFilter
	if !userFilter.IsZero() {
		f = rostam.VectorFilter{Op: vector.FilterAnd, And: []rostam.VectorFilter{nsFilter, userFilter}}
	}

	if !s.hybrid {
		return s.recallBM25(ctx, args.Query, k, f)
	}
	return s.recallHybrid(ctx, args.Query, k, f)
}

// recallBM25 runs the text-only search path (no embedder configured): a
// VectorDocument already carries content and metadata, so no batch fetch is
// needed.
func (s *Server) recallBM25(ctx context.Context, query string, k int, f rostam.VectorFilter) (any, error) {
	docs, _, err := s.store.VectorSearchText(ctx, memCollection, query, k, rostam.VectorSearchOpts{Filter: f})
	if err != nil {
		return nil, fmt.Errorf("mcp: recall: %w", err)
	}
	hits := make([]memoryHit, len(docs))
	for i, d := range docs {
		hits[i] = memoryHit{ID: d.ID, Content: d.Content, Score: d.Score, Metadata: metadataToJSON(stripReservedMetadata(d.Metadata))}
	}
	return map[string]any{"hits": hits}, nil
}

// recallHybrid runs the dense+BM25 fusion path. VectorHybridText returns only
// ids/scores (no content/metadata), so the hits are joined against a batch
// fetch by id — GetBatch's return order is not guaranteed to match the
// fusion ranking, hence the id->point map.
func (s *Server) recallHybrid(ctx context.Context, query string, k int, f rostam.VectorFilter) (any, error) {
	dense, err := s.emb.Embed(ctx, []string{query})
	if err != nil {
		return nil, fmt.Errorf("mcp: embed query: %w", err)
	}
	results, _, err := s.store.VectorHybridText(ctx, memCollection, dense[0], query, k, rostam.VectorHybridOpts{Filter: f})
	if err != nil {
		return nil, fmt.Errorf("mcp: recall: %w", err)
	}
	if len(results) == 0 {
		return map[string]any{"hits": []memoryHit{}}, nil
	}

	ids := make([]uint64, len(results))
	for i, r := range results {
		ids[i] = r.ID
	}
	points, _, err := s.store.VectorGetBatch(ctx, memCollection, ids, false, true)
	if err != nil {
		return nil, fmt.Errorf("mcp: recall: fetching hit content: %w", err)
	}
	byID := make(map[uint64]rostam.BatchGetPoint, len(points))
	for _, p := range points {
		byID[p.ID] = p
	}

	hits := make([]memoryHit, 0, len(results))
	for _, r := range results {
		p, ok := byID[r.ID]
		if !ok {
			continue // point vanished between the fusion search and the batch fetch
		}
		var content string
		if cv, ok := p.Meta["$content"]; ok && cv.Kind == vector.ValueString {
			content = cv.Str
		}
		hits = append(hits, memoryHit{ID: r.ID, Content: content, Score: r.Score, Metadata: metadataToJSON(stripReservedMetadata(p.Meta))})
	}
	return map[string]any{"hits": hits}, nil
}

// stripReservedMetadata removes the memory subsystem's own bookkeeping
// fields (namespace, creation time) before metadata is handed back to a
// tool caller. $content is stripped separately by metadataToJSON.
func stripReservedMetadata(md rostam.VectorMetadata) rostam.VectorMetadata {
	if len(md) == 0 {
		return md
	}
	out := make(rostam.VectorMetadata, len(md))
	for k, v := range md {
		if k == nsField || k == createdField {
			continue
		}
		out[k] = v
	}
	return out
}
