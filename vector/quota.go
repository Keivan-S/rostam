// SPDX-License-Identifier: Apache-2.0

package vector

import (
	"sync"
	"time"
)

// tokenBucket is a simple lazy-refill rate limiter. Take() returns true when
// a token is available; false when the bucket is empty. Refill rate is `rate`
// tokens per second up to capacity `burst`. The implementation refills only
// on Take() — there is no background goroutine.
//
// Zero-cost when not configured: callers wrap Take() in a nil-check, so
// hnsw paths without rate limits pay nothing.
type tokenBucket struct {
	mu         sync.Mutex
	tokens     float64
	rate       float64 // tokens per second
	burst      float64 // bucket capacity
	lastRefill time.Time
	now        func() time.Time // injectable clock for tests
}

// newTokenBucket builds a bucket that allows `rate` inserts per second with a
// burst capacity equal to `rate`. Returns nil when rate <= 0, which callers
// treat as "no rate limit".
func newTokenBucket(rate int) *tokenBucket {
	if rate <= 0 {
		return nil
	}
	return &tokenBucket{
		tokens:     float64(rate),
		rate:       float64(rate),
		burst:      float64(rate),
		lastRefill: time.Now(),
		now:        time.Now,
	}
}

// Take attempts to consume one token. Returns true on success. A nil receiver
// always returns true so callers don't need their own nil-check.
func (b *tokenBucket) Take() bool {
	if b == nil {
		return true
	}
	b.mu.Lock()
	defer b.mu.Unlock()

	now := b.now()
	elapsed := now.Sub(b.lastRefill).Seconds()
	if elapsed > 0 {
		b.tokens += elapsed * b.rate
		if b.tokens > b.burst {
			b.tokens = b.burst
		}
		b.lastRefill = now
	}
	if b.tokens >= 1 {
		b.tokens--
		return true
	}
	return false
}

// estimateInsertBytes returns a conservative byte estimate for one HNSW
// insert. Captures vector storage (4 bytes × dim) plus a rough graph
// per-node overhead: each level holds up to M neighbor slots (level 0 holds
// up to 2*M); a node is at level 0 with high probability and falls off
// geometrically. We use 2*M+M = 3*M slots × 4 bytes/slot = 12*M as the
// upper-bound proxy, plus ~32 bytes for the node struct itself.
func estimateInsertBytes(dim, m int) int64 {
	return int64(4*dim + 12*m + 32)
}

// bulkQuotaErr reports the error a bulk load of n points must return to keep the
// collection inside cfg's quota, or nil when the load fits.
//
// THE GAP THIS CLOSES. insertLocked enforces MaxVectors/MaxBytes on every single
// insert, but the bulk builders — hnsw.BuildConcurrentMeta, buildVamana and
// ivf.BuildConcurrentMeta — consulted neither, so the one path that writes N
// points at once was the one path that could not say no. A bulk load could leave
// a collection permanently OVER its budget, which is the state that makes the
// free-then-reuse resurrection window reachable, and it did so silently: the
// caller got a success and the quota counters never moved.
//
// The arithmetic is simpler than insertLocked's because a bulk load requires an
// EMPTY index (every caller checks ErrBuildNonEmpty before reaching here): there
// is no slot to reclaim, bytesUsed is 0, and the payload index holds no column
// sidecar. The collection after the load is exactly n points and n inserts'
// worth of bytes.
//
// It is EQUIVALENT to running insertLocked's checks n times, not a new policy.
// Serially, the k-th insert into an empty index is rejected when k >= MaxVectors,
// so all n land iff n <= MaxVectors; and when (k+1)*insertBytes > MaxBytes, so
// all n land iff n*insertBytes <= MaxBytes. Both bounds below are those, stated
// once.
func bulkQuotaErr(cfg Config, n int) error {
	if cfg.MaxVectors > 0 && int64(n) > cfg.MaxVectors {
		return ErrCollectionFull
	}
	if cfg.MaxBytes > 0 && int64(n)*estimateInsertBytes(cfg.Dim, cfg.M) > cfg.MaxBytes {
		return ErrCollectionFull
	}
	return nil
}
