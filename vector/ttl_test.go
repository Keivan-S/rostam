// SPDX-License-Identifier: Apache-2.0

package vector

import (
	"testing"
	"time"
)

func TestSweepOnceTombstonesExpired(t *testing.T) {
	h, err := newHNSW(Config{Dim: 4, Metric: L2, M: 16, EfConstruction: 200, EfSearch: 64, Seed: 1})
	if err != nil {
		t.Fatalf("newHNSW: %v", err)
	}
	var fakeNow int64 = 1_000_000
	h.now = func() int64 { return fakeNow }

	for i, v := range [][]float32{
		{1, 0, 0, 0},
		{0, 1, 0, 0},
		{0, 0, 1, 0},
	} {
		ttl := time.Duration(0)
		if i == 0 {
			ttl = 50 * time.Millisecond
		}
		if _, _, err := h.Insert(uint64(i+1), v, ttl, nil, nil, nil, CASCond{}); err != nil {
			t.Fatalf("Insert %d: %v", i+1, err)
		}
	}

	// Before expiry: sweep finds nothing.
	if n := h.sweepOnce(); n != 0 {
		t.Fatalf("sweepOnce pre-expiry = %d, want 0", n)
	}

	fakeNow += 100

	// After expiry: id=1 swept.
	if n := h.sweepOnce(); n != 1 {
		t.Fatalf("sweepOnce post-expiry = %d, want 1", n)
	}
	if stats := h.Stats(); stats.Expired != 1 {
		t.Fatalf("Stats.Expired = %d, want 1", stats.Expired)
	}
	if stats := h.Stats(); stats.Tombstoned != 1 {
		t.Fatalf("Stats.Tombstoned = %d, want 1", stats.Tombstoned)
	}

	// Re-sweeping is idempotent.
	if n := h.sweepOnce(); n != 0 {
		t.Fatalf("sweepOnce idempotent = %d, want 0", n)
	}
}

func TestCollectionTTLRoundtrip(t *testing.T) {
	c, err := NewCollection("test", Config{
		Dim: 4, Metric: L2, M: 16, EfConstruction: 200, EfSearch: 64, Seed: 1,
		SweepInterval: 20 * time.Millisecond,
	})
	if err != nil {
		t.Fatalf("NewCollection: %v", err)
	}
	defer c.Stop()

	var fakeNow int64 = 1_000_000
	c.idx.(*hnsw).now = func() int64 { return fakeNow }

	if err := c.Insert(1, []float32{1, 0, 0, 0}, 50*time.Millisecond, nil, nil); err != nil {
		t.Fatalf("Insert ttl: %v", err)
	}
	if err := c.Insert(2, []float32{0, 1, 0, 0}, 0, nil, nil); err != nil {
		t.Fatalf("Insert no-ttl: %v", err)
	}

	results, err := c.Search([]float32{1, 0, 0, 0}, 10)
	if err != nil {
		t.Fatalf("Search before expiry: %v", err)
	}
	if len(results) != 2 {
		t.Fatalf("before expiry got %d results, want 2", len(results))
	}

	fakeNow += 100

	// Wait for the sweeper to catch up. The ticker fires every 20 ms; we poll
	// for Stats.Expired to flip before a generous real-time deadline.
	deadline := time.Now().Add(2 * time.Second)
	for time.Now().Before(deadline) {
		if c.Stats().Expired >= 1 {
			break
		}
		time.Sleep(5 * time.Millisecond)
	}

	if got := c.Stats().Expired; got != 1 {
		t.Fatalf("Expired after sweep = %d, want 1", got)
	}

	results, err = c.Search([]float32{1, 0, 0, 0}, 10)
	if err != nil {
		t.Fatalf("Search after expiry: %v", err)
	}
	if len(results) != 1 || results[0].ID != 2 {
		t.Fatalf("after expiry got %v, want [id=2]", results)
	}
}
