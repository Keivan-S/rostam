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
	memCollection = "mcp_memory"     // the vector collection backing remember/recall
	nsField       = "__ns"           // metadata field: the memory's namespace
	createdField  = "__created_unix" // metadata field: unix-seconds creation time
	defaultNS     = "default"        // namespace used when the caller omits one
	kvEmbedder    = "__mcp/embedder" // KV key: the embedIdentity this store was bootstrapped with
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

// identityReadAttempts / identityReadBackoff bound the wait for a concurrent
// bootstrapper to finish writing the embedder identity key. The gap between
// its CreateCollection and its Put is one round trip, so a couple of retries
// is generous; the point of the bound is that a creator which died in that gap
// must surface as an error rather than an indefinite hang.
const (
	identityReadAttempts = 3
	identityReadBackoff  = 50 * time.Millisecond
)

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
	return s.matchStoredIdentity(raw)
}

// matchStoredIdentity decodes a recorded identity and checks it against this
// run's embedder. Shared by the startup check and the concurrent-bootstrap
// path, which must apply exactly the same rule to the identity it reads back.
func (s *Server) matchStoredIdentity(raw []byte) error {
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
// creates it and records the embedder identity that owns it.
//
// Only SUCCESS is latched. sync.Once used to guard this and cached whatever
// the first attempt produced, error included — so one canceled context, one
// not-leader reply, or one transport blip on the very first memory tool call
// left every later remember/recall/forget in the process failing with that
// stale error forever, long after the store recovered. A failed attempt now
// leaves the latch closed and the next caller tries again.
func (s *Server) ensureMemory(ctx context.Context) error {
	s.memMu.Lock()
	defer s.memMu.Unlock()
	if s.memReady {
		return nil
	}
	if err := s.bootstrapMemory(ctx); err != nil {
		return err
	}
	s.memReady = true
	return nil
}

// bootstrapMemory is one bootstrap attempt. The caller holds s.memMu.
func (s *Server) bootstrapMemory(ctx context.Context) error {
	_, err := s.store.Get(ctx, []byte(kvEmbedder))
	if err == nil {
		// Already bootstrapped; checkEmbedderIdentity already verified this
		// run's embedder matches, so there is nothing left to do.
		return nil
	}
	if !errors.Is(err, rostam.ErrNotFound) {
		return fmt.Errorf("mcp: reading stored embedder identity: %w", err)
	}
	// vector.DefaultConfig fills in the HNSW build knobs (M/EfConstruction/
	// EfSearch) that Config.Validate requires but the memory collection has
	// no opinion on; only Dim/Metric/FullText are memory-specific.
	cfg := vector.DefaultConfig()
	cfg.Dim, cfg.Metric, cfg.FullText = s.emb.Dim(), vector.Cosine, &vector.FullTextConfig{}
	applyPersistence(&cfg)
	switch err := s.store.CreateCollection(ctx, memCollection, cfg); {
	case err == nil:
	case isCollectionExists(err):
		// A second session bootstrapped between our existence check and this
		// call — routine in remote mode, where several `mcp -connect` processes
		// share one store. The collection this server needs now exists, so the
		// race is won by joining it, not by failing.
		return s.awaitStoredIdentity(ctx)
	default:
		return fmt.Errorf("mcp: creating memory collection: %w", err)
	}
	id := embedIdentity{Model: s.emb.Model(), Dim: s.emb.Dim(), Hybrid: s.hybrid}
	b, err := json.Marshal(id)
	if err != nil {
		return fmt.Errorf("mcp: encoding embedder identity: %w", err)
	}
	if err := s.store.Put(ctx, []byte(kvEmbedder), b, 0); err != nil {
		return fmt.Errorf("mcp: storing embedder identity: %w", err)
	}
	return nil
}

// isCollectionExists reports whether err is "that collection is already
// there". Embedded mode returns vector.ErrCollectionExists directly; over
// -connect the same failure arrives as a reconstructed error whose chain does
// not survive the wire, so the message is matched as well.
func isCollectionExists(err error) bool {
	return errors.Is(err, vector.ErrCollectionExists) ||
		strings.Contains(err.Error(), "collection already exists")
}

// awaitStoredIdentity reads back the embedder identity a concurrent creator
// recorded and verifies it matches this run's configuration.
//
// The other session writes the identity key one round trip AFTER creating the
// collection, so reading it immediately can legitimately miss it; a few
// bounded retries cover that window. Exhausting them means the other creator
// died mid-bootstrap, which this run reports rather than proceeding against a
// collection whose embedder it was never able to check.
func (s *Server) awaitStoredIdentity(ctx context.Context) error {
	for attempt := range identityReadAttempts {
		if attempt > 0 {
			select {
			case <-ctx.Done():
				return ctx.Err()
			case <-time.After(identityReadBackoff):
			}
		}
		raw, err := s.store.Get(ctx, []byte(kvEmbedder))
		switch {
		case err == nil:
			return s.matchStoredIdentity(raw)
		case errors.Is(err, rostam.ErrNotFound):
			continue
		default:
			return fmt.Errorf("mcp: reading stored embedder identity: %w", err)
		}
	}
	return fmt.Errorf("mcp: the %q collection already exists but carries no embedder identity after %d reads; a concurrent bootstrap looks to have failed part-way, leaving a collection whose embedder cannot be verified",
		memCollection, identityReadAttempts)
}

// firstEmbedding pulls the single vector an Embed of one text should have
// produced.
//
// Every embed call in this package passes exactly one string, so it is
// tempting to index [0] straight away — but the embedder is not necessarily
// ours. semcache's OpenAI-compatible embedder is an HTTP client pointed at
// whatever ROSTAM_EMBED_ENDPOINT names, and a response with a short or empty
// data array decodes into a short slice with a nil error. Indexing that panics
// the dispatch goroutine and takes the session down, so a result that is not
// exactly one usable vector is reported as the bad response it is.
func firstEmbedding(vecs [][]float32) ([]float32, error) {
	if len(vecs) == 0 {
		return nil, errors.New("mcp: the embedder returned no vectors for one input")
	}
	if len(vecs[0]) == 0 {
		return nil, errors.New("mcp: the embedder returned an empty vector")
	}
	return vecs[0], nil
}

// memoryID derives a memory's point id from its namespace and content, so
// remembering the same fact twice (in the same namespace) upserts the same
// point instead of accumulating duplicates.
func memoryID(ns, content string) uint64 {
	return xxhash.Sum64String(ns + "\x00" + content)
}

// memoryHit is one recalled (or listed, Task 5) memory: its id, content,
// relevance score, and user metadata with the reserved fields stripped. No
// Distance field: it's meaningless for BM25-only recall and for a scroll
// listing (no query), and this shape is locked by Task 4's tests. The
// generic search tool (Task 6) has its own searchHit type in db.go for
// exactly this reason — see hybridDocs and recallHybrid.
type memoryHit struct {
	ID       uint64         `json:"id"`
	Content  string         `json:"content"`
	Score    float32        `json:"score"`
	Metadata map[string]any `json:"metadata,omitempty"`
}

// registerMemoryTools registers remember, recall, forget, list_memories, and
// list_namespaces.
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
	s.register(toolDef{
		Name:        "forget",
		Description: `Delete stored memories by id. Returns {"deleted":[...],"missing":[...],"errors":[...]}; a per-id delete failure does not abort the rest of the batch.`,
		InputSchema: map[string]any{
			"type": "object",
			"properties": map[string]any{
				"ids": map[string]any{
					"type":        "array",
					"items":       map[string]any{"type": "integer"},
					"description": "ids of the memories to delete",
				},
			},
			"required": []any{"ids"},
		},
		Handler: s.handleForget,
	})
	s.register(toolDef{
		Name:        "list_memories",
		Description: "List stored memories in a namespace, paginated.",
		InputSchema: map[string]any{
			"type": "object",
			"properties": map[string]any{
				"namespace": map[string]any{"type": "string", "description": `namespace to list (default "default")`},
				"limit":     map[string]any{"type": "integer", "description": "max memories per page (default 50, max 500)"},
				"cursor":    map[string]any{"type": "string", "description": "resume-after cursor from a previous call's next_cursor"},
			},
		},
		Handler: s.handleListMemories,
	})
	s.register(toolDef{
		Name:        "list_namespaces",
		Description: "List all known memory namespaces.",
		InputSchema: map[string]any{
			"type":       "object",
			"properties": map[string]any{},
		},
		Handler: s.handleListNamespaces,
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
	vec, err := firstEmbedding(vecs)
	if err != nil {
		return nil, fmt.Errorf("mcp: embed content: %w", err)
	}

	id := memoryID(ns, args.Content)
	if err := s.store.VectorUpsert(ctx, memCollection, id, vec, args.Content, rostam.VectorInsertOpts{Metadata: md}); err != nil {
		return nil, fmt.Errorf("mcp: remember: %w", err)
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
	// <= 0, not == 0: a client can send "k": -1, and a negative k reaching the
	// search call is the same nonsense as a zero one. Matches list_memories.
	k := args.K
	if k <= 0 {
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

// recallHybrid runs the dense+BM25 fusion path via the shared hybridDocs
// helper (db.go), then narrows each hit down to memoryHit's locked shape:
// dropping Distance (meaningless for a memory recall) and stripping the
// memory subsystem's reserved metadata fields. hybridDocs itself is
// collection-agnostic and leaves metadata/distance untouched, since a
// generic search caller (Task 6) wants both.
func (s *Server) recallHybrid(ctx context.Context, query string, k int, f rostam.VectorFilter) (any, error) {
	dense, err := s.emb.Embed(ctx, []string{query})
	if err != nil {
		return nil, fmt.Errorf("mcp: embed query: %w", err)
	}
	vec, err := firstEmbedding(dense)
	if err != nil {
		return nil, fmt.Errorf("mcp: embed query: %w", err)
	}
	docs, err := s.hybridDocs(ctx, memCollection, vec, query, k, f)
	if err != nil {
		return nil, fmt.Errorf("mcp: recall: %w", err)
	}
	hits := make([]memoryHit, len(docs))
	for i, d := range docs {
		delete(d.Metadata, nsField)
		delete(d.Metadata, createdField)
		hits[i] = memoryHit{ID: d.ID, Content: d.Content, Score: d.Score, Metadata: d.Metadata}
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

// forgetArgs is the forget tool's decoded input.
type forgetArgs struct {
	IDs []uint64 `json:"ids"`
}

// handleForget deletes memories by id. Ids not found in the collection are
// reported back as "missing" rather than erroring, so a caller can forget a
// batch without first checking which ids still exist; per-id failures land in
// "errors" for the same reason (see forgetResult).
//
// There is no namespace bookkeeping to do afterwards: a namespace is defined
// by the memories carrying it (see handleListNamespaces), so deleting the last
// one is all that "removing a namespace" means.
func (s *Server) handleForget(ctx context.Context, raw json.RawMessage) (any, error) {
	var args forgetArgs
	if err := json.Unmarshal(raw, &args); err != nil {
		return nil, fmt.Errorf("mcp: bad forget args: %w", err)
	}
	if len(args.IDs) == 0 {
		return nil, fmt.Errorf("mcp: forget: ids is required and must be non-empty")
	}

	if err := s.ensureMemory(ctx); err != nil {
		return nil, err
	}

	points, missing, err := s.store.VectorGetBatch(ctx, memCollection, args.IDs, false, true)
	if err != nil {
		return nil, fmt.Errorf("mcp: forget: %w", err)
	}

	// Every point's delete is attempted, even after an earlier one fails: a
	// batch with a mix of good and bad ids must still delete whatever it
	// legitimately can, instead of aborting the whole call and leaving the
	// caller unable to tell "nothing happened" from "some of this went
	// through".
	deleted := make([]uint64, 0, len(points))
	var delErrs []string
	for _, p := range points {
		if _, err := s.store.VectorDelete(ctx, memCollection, p.ID); err != nil {
			delErrs = append(delErrs, fmt.Sprintf("id %d: %s", p.ID, err))
			continue
		}
		deleted = append(deleted, p.ID)
	}

	if missing == nil {
		missing = []uint64{}
	}
	return forgetResult{Deleted: deleted, Missing: missing, Errors: delErrs}, nil
}

// forgetResult is the forget tool's response shape, deliberately identical to
// delete's (db.go): the two tools do the same thing to different collections,
// so a caller should not have to learn two contracts. Errors is omitted on full
// success, which keeps the {deleted, missing} shape callers already depend on;
// when present, a partial batch still reports what it managed to delete rather
// than discarding that outcome behind a tool-level error.
type forgetResult struct {
	Deleted []uint64 `json:"deleted"`
	Missing []uint64 `json:"missing"`
	Errors  []string `json:"errors,omitempty"`
}

// listMemoriesArgs is the list_memories tool's decoded input.
type listMemoriesArgs struct {
	Namespace string `json:"namespace"`
	Limit     int    `json:"limit"`
	Cursor    string `json:"cursor"`
}

// handleListMemories pages through a namespace's memories in id order.
// Score is left at its zero value on every hit: scroll has no query to rank
// results against.
func (s *Server) handleListMemories(ctx context.Context, raw json.RawMessage) (any, error) {
	var args listMemoriesArgs
	if err := json.Unmarshal(raw, &args); err != nil {
		return nil, fmt.Errorf("mcp: bad list_memories args: %w", err)
	}
	ns := args.Namespace
	if ns == "" {
		ns = defaultNS
	}
	limit := args.Limit
	switch {
	case limit <= 0:
		limit = 50
	case limit > 500:
		limit = 500
	}

	if err := s.ensureMemory(ctx); err != nil {
		return nil, err
	}

	nsFilter := rostam.VectorFilter{Op: vector.FilterEq, Field: nsField, Value: vector.NewString(ns)}
	docs, _, cursor, err := s.store.VectorScroll(ctx, memCollection, nsFilter, limit, rostam.VectorScrollOpts{Cursor: args.Cursor})
	if err != nil {
		return nil, fmt.Errorf("mcp: list_memories: %w", err)
	}
	memories := make([]memoryHit, len(docs))
	for i, d := range docs {
		memories[i] = memoryHit{ID: d.ID, Content: d.Content, Metadata: metadataToJSON(stripReservedMetadata(d.Metadata))}
	}
	return map[string]any{"memories": memories, "next_cursor": cursor}, nil
}

// nsScanPage is how many memories one list_namespaces page pulls. Only each
// doc's __ns field is kept, so the page size trades round trips against
// per-page memory, and memory sets are small.
const nsScanPage = 500

// handleListNamespaces reports every namespace currently holding at least one
// memory, by scrolling the collection and collecting the distinct __ns values.
//
// This used to read a KV registry that remember appended to and forget pruned.
// That registry was a read-modify-write on one key with no CAS, correct only
// under the assumption of a single session — which does not hold in remote
// mode, where several `mcp -connect` processes share one store and each has
// its own Server. Two of them interleaving could drop a namespace from the
// list permanently while its memories sat there, and no later operation would
// ever notice.
//
// A scan costs more than the registry's single Get, but it cannot be wrong:
// the memories ARE the namespace list. It also deletes the whole class of
// bookkeeping bugs the registry needed (seeding it at bootstrap, appending on
// remember, pruning on forget, and keeping all three consistent when a batch
// half-failed).
func (s *Server) handleListNamespaces(ctx context.Context, _ json.RawMessage) (any, error) {
	if err := s.ensureMemory(ctx); err != nil {
		return nil, err
	}
	seen := make(map[string]struct{})
	var cursor string
	for {
		// A zero filter is match-all: every memory, every namespace.
		docs, _, next, err := s.store.VectorScroll(ctx, memCollection, rostam.VectorFilter{}, nsScanPage, rostam.VectorScrollOpts{Cursor: cursor})
		if err != nil {
			return nil, fmt.Errorf("mcp: list_namespaces: %w", err)
		}
		for _, d := range docs {
			if v, ok := d.Metadata[nsField]; ok && v.Kind == vector.ValueString {
				seen[v.Str] = struct{}{}
			}
		}
		// A cursor that does not advance would loop forever; treat it as
		// exhausted rather than trusting the backend to always terminate.
		if next == "" || next == cursor {
			break
		}
		cursor = next
	}
	out := make([]string, 0, len(seen))
	for n := range seen {
		out = append(out, n)
	}
	sort.Strings(out)
	return map[string]any{"namespaces": out}, nil
}
