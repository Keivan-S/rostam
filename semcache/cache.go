// SPDX-License-Identifier: Apache-2.0

package semcache

import (
	"context"
	"errors"
	"fmt"
	"strings"
	"time"

	"github.com/cespare/xxhash/v2"

	"github.com/rostamlabs/rostam"
	"github.com/rostamlabs/rostam/vector"
)

// DefaultThreshold is the cosine-similarity floor for a cache hit when Config
// leaves Threshold unset. Conservative on purpose: a false hit returns a wrong
// answer, which is worse than a miss.
const DefaultThreshold = 0.97

// scopeField is the metadata field the scope key is stored under and filtered
// on at lookup; tokensField stores the cached answer's output-token count for
// savings accounting.
const (
	scopeField  = "__scope"
	tokensField = "__out_tokens"
)

// Config configures a Cache.
type Config struct {
	Store      rostam.Store  // required: the backing engine (embedded or client)
	Embedder   Embedder      // required: text -> vectors
	Collection string        // required: collection name (created if absent)
	Threshold  float64       // cosine-similarity hit floor; 0 -> DefaultThreshold
	TTL        time.Duration // per-entry expiry; 0 -> no expiry
	MaxTemp    float64       // do not cache requests with Temperature above this; default 0
}

// Cache is a semantic cache over one Rostam collection. Safe for concurrent use
// (the underlying Store is).
type Cache struct {
	store      rostam.Store
	embedder   Embedder
	collection string
	threshold  float64
	ttl        time.Duration
	maxTemp    float64
}

// New validates cfg and ensures the backing Cosine collection exists. It is
// idempotent: an already-existing collection is accepted.
func New(ctx context.Context, cfg Config) (*Cache, error) {
	if cfg.Store == nil {
		return nil, errors.New("semcache: store is required")
	}
	if cfg.Embedder == nil {
		return nil, errors.New("semcache: embedder is required")
	}
	if cfg.Collection == "" {
		return nil, errors.New("semcache: Config.Collection is required")
	}
	if cfg.Embedder.Dim() <= 0 {
		return nil, fmt.Errorf("semcache: embedder Dim must be positive, got %d", cfg.Embedder.Dim())
	}
	threshold := cfg.Threshold
	if threshold == 0 {
		threshold = DefaultThreshold
	}

	c := &Cache{
		store:      cfg.Store,
		embedder:   cfg.Embedder,
		collection: cfg.Collection,
		threshold:  threshold,
		ttl:        cfg.TTL,
		maxTemp:    cfg.MaxTemp,
	}

	vc := rostam.VectorConfig{
		Dim:            cfg.Embedder.Dim(),
		Metric:         vector.Cosine,
		M:              16,
		EfConstruction: 200,
		EfSearch:       64,
	}
	if err := cfg.Store.CreateCollection(ctx, cfg.Collection, vc); err != nil &&
		!strings.Contains(strings.ToLower(err.Error()), "exist") {
		return nil, fmt.Errorf("semcache: create collection: %w", err)
	}
	return c, nil
}

// entryID is the stable record id for a (scope, prompt) pair. Re-storing the
// same prompt under the same scope overwrites in place (idempotent); different
// prompts get different ids. 64-bit collision risk is negligible.
func (c *Cache) entryID(scopeKey, prompt string) uint64 {
	return xxhash.Sum64String(scopeKey + "\x00" + prompt)
}

// Hit is a successful cache lookup.
type Hit struct {
	Answer    string  // the cached completion to return
	Score     float64 // cosine similarity of the matched prior prompt, approximately in [0,1] (float rounding may place it marginally above 1.0)
	OutTokens int     // output tokens of the cached completion (savings accounting)
}

// Cacheable reports whether a request at the given sampling temperature may be
// cached. High-temperature (intentionally non-deterministic) calls are never
// cached. The proxy also declines to cache responses that invoked tools.
func (c *Cache) Cacheable(temperature float64) bool {
	return temperature <= c.maxTemp
}

// Lookup embeds prompt and returns the cached answer for the nearest prior
// prompt within the same scope, if its cosine similarity meets the threshold.
// ok=false is a clean miss (no error). A hit means zero generation tokens.
func (c *Cache) Lookup(ctx context.Context, prompt string, scope Scope) (Hit, bool, error) {
	vecs, err := c.embedder.Embed(ctx, []string{prompt})
	if err != nil {
		return Hit{}, false, fmt.Errorf("semcache: embed prompt: %w", err)
	}
	if len(vecs) != 1 || len(vecs[0]) != c.embedder.Dim() {
		return Hit{}, false, fmt.Errorf("semcache: embedder returned %d vecs / unexpected dim", len(vecs))
	}
	sk := scope.key(c.embedder.Model())
	filter := rostam.VectorFilter{Op: vector.FilterEq, Field: scopeField, Value: vector.NewString(sk)}

	docs, _, err := c.store.VectorSearchDocs(ctx, c.collection, vecs[0], 1,
		rostam.VectorSearchOpts{Filter: filter})
	if err != nil {
		return Hit{}, false, fmt.Errorf("semcache: search: %w", err)
	}
	if len(docs) == 0 {
		return Hit{}, false, nil
	}
	// Cosine distance over normalized vectors is 1 - similarity (see
	// vector/distance.go); convert back to a similarity in [0,1].
	sim := 1.0 - float64(docs[0].Distance)
	if sim < c.threshold {
		return Hit{}, false, nil
	}
	hit := Hit{Answer: docs[0].Content, Score: sim}
	if tv, ok := docs[0].Metadata[tokensField]; ok {
		hit.OutTokens = int(tv.Int)
	}
	return hit, true, nil
}

// Store records answer (with its output-token count, for savings accounting) as
// the cache entry for prompt under scope. The prompt is embedded and stored as
// the record's vector; the scope key rides in metadata so Lookup can filter on
// it. Honors the configured TTL.
func (c *Cache) Store(ctx context.Context, prompt string, scope Scope, answer string, outTokens int) error {
	vecs, err := c.embedder.Embed(ctx, []string{prompt})
	if err != nil {
		return fmt.Errorf("semcache: embed prompt: %w", err)
	}
	if len(vecs) != 1 || len(vecs[0]) != c.embedder.Dim() {
		return fmt.Errorf("semcache: embedder returned %d vecs / unexpected dim", len(vecs))
	}
	sk := scope.key(c.embedder.Model())
	meta := rostam.VectorMetadata{
		scopeField:  vector.NewString(sk),
		tokensField: vector.NewInt(int64(outTokens)),
	}
	return c.store.VectorUpsert(ctx, c.collection, c.entryID(sk, prompt), vecs[0], answer,
		rostam.VectorInsertOpts{TTL: c.ttl, Metadata: meta})
}
