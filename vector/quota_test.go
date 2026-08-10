// SPDX-License-Identifier: Apache-2.0

package vector

import (
	"testing"
	"time"
)

func TestTokenBucketBasic(t *testing.T) {
	b := newTokenBucket(3)
	if b == nil {
		t.Fatal("newTokenBucket(3) returned nil")
	}
	for i := 0; i < 3; i++ {
		if !b.Take() {
			t.Fatalf("Take() %d/3 = false, want true (bucket should be full)", i+1)
		}
	}
	if b.Take() {
		t.Fatal("4th Take() = true, want false (bucket should be empty)")
	}
}

func TestTokenBucketRefill(t *testing.T) {
	b := newTokenBucket(2)
	now := time.Unix(0, 0)
	b.now = func() time.Time { return now }
	b.lastRefill = now

	for i := 0; i < 2; i++ {
		if !b.Take() {
			t.Fatalf("initial Take %d/2 = false", i+1)
		}
	}
	if b.Take() {
		t.Fatal("3rd Take = true, want false")
	}

	// Half a second → one token refilled (rate=2/s).
	now = now.Add(500 * time.Millisecond)
	if !b.Take() {
		t.Fatal("Take after refill = false")
	}
	if b.Take() {
		t.Fatal("Take after using refill = true")
	}

	// Full second → bucket back to burst capacity (2 tokens).
	now = now.Add(1 * time.Second)
	for i := 0; i < 2; i++ {
		if !b.Take() {
			t.Fatalf("Take after full refill %d/2 = false", i+1)
		}
	}
	if b.Take() {
		t.Fatal("3rd Take after full refill = true")
	}
}

func TestTokenBucketNilIsUnlimited(t *testing.T) {
	var b *tokenBucket
	for i := 0; i < 1000; i++ {
		if !b.Take() {
			t.Fatal("nil bucket Take() = false, want unlimited")
		}
	}
}

func TestEstimateInsertBytes(t *testing.T) {
	// dim=128, M=16 → 4*128 + 12*16 + 32 = 512 + 192 + 32 = 736.
	if got, want := estimateInsertBytes(128, 16), int64(736); got != want {
		t.Fatalf("estimateInsertBytes(128, 16) = %d, want %d", got, want)
	}
}

func TestMaxVectorsQuota(t *testing.T) {
	h, err := newHNSW(Config{
		Dim: 4, Metric: L2, M: 16, EfConstruction: 200, EfSearch: 64, Seed: 1,
		MaxVectors: 3,
	})
	if err != nil {
		t.Fatalf("newHNSW: %v", err)
	}
	for i := 1; i <= 3; i++ {
		if _, _, err := h.Insert(uint64(i), []float32{float32(i), 0, 0, 0}, 0, nil, nil, nil, CASCond{}); err != nil {
			t.Fatalf("Insert %d: %v", i, err)
		}
	}
	if _, _, err := h.Insert(4, []float32{4, 0, 0, 0}, 0, nil, nil, nil, CASCond{}); err != ErrCollectionFull {
		t.Fatalf("Insert 4 err = %v, want ErrCollectionFull", err)
	}
	if stats := h.Stats(); stats.QuotaRejects != 1 {
		t.Fatalf("QuotaRejects = %d, want 1", stats.QuotaRejects)
	}
}

func TestMaxBytesQuota(t *testing.T) {
	// estimateInsertBytes(4, 16) = 16 + 192 + 32 = 240. Cap at 500 → 2 inserts fit.
	h, err := newHNSW(Config{
		Dim: 4, Metric: L2, M: 16, EfConstruction: 200, EfSearch: 64, Seed: 1,
		MaxBytes: 500,
	})
	if err != nil {
		t.Fatalf("newHNSW: %v", err)
	}
	if _, _, err := h.Insert(1, []float32{1, 0, 0, 0}, 0, nil, nil, nil, CASCond{}); err != nil {
		t.Fatalf("Insert 1: %v", err)
	}
	if _, _, err := h.Insert(2, []float32{0, 1, 0, 0}, 0, nil, nil, nil, CASCond{}); err != nil {
		t.Fatalf("Insert 2: %v", err)
	}
	if _, _, err := h.Insert(3, []float32{0, 0, 1, 0}, 0, nil, nil, nil, CASCond{}); err != ErrCollectionFull {
		t.Fatalf("Insert 3 err = %v, want ErrCollectionFull", err)
	}
}

func TestMaxInsertsPerSecondQuota(t *testing.T) {
	h, err := newHNSW(Config{
		Dim: 4, Metric: L2, M: 16, EfConstruction: 200, EfSearch: 64, Seed: 1,
		MaxInsertsPerSecond: 5,
	})
	if err != nil {
		t.Fatalf("newHNSW: %v", err)
	}
	// Freeze the bucket clock so we control refill explicitly.
	now := time.Unix(0, 0)
	h.bucket.now = func() time.Time { return now }
	h.bucket.lastRefill = now

	for i := 1; i <= 5; i++ {
		if _, _, err := h.Insert(uint64(i), []float32{float32(i), 0, 0, 0}, 0, nil, nil, nil, CASCond{}); err != nil {
			t.Fatalf("Insert %d: %v", i, err)
		}
	}
	if _, _, err := h.Insert(6, []float32{6, 0, 0, 0}, 0, nil, nil, nil, CASCond{}); err != ErrCollectionRateLimited {
		t.Fatalf("Insert 6 err = %v, want ErrCollectionRateLimited", err)
	}
	if stats := h.Stats(); stats.QuotaRejects != 1 {
		t.Fatalf("QuotaRejects = %d, want 1", stats.QuotaRejects)
	}

	// Refill over a full second; bucket returns to capacity.
	now = now.Add(1 * time.Second)
	for i := 7; i <= 11; i++ {
		if _, _, err := h.Insert(uint64(i), []float32{float32(i), 0, 0, 0}, 0, nil, nil, nil, CASCond{}); err != nil {
			t.Fatalf("Insert %d after refill: %v", i, err)
		}
	}
	if _, _, err := h.Insert(12, []float32{12, 0, 0, 0}, 0, nil, nil, nil, CASCond{}); err != ErrCollectionRateLimited {
		t.Fatalf("Insert 12 err = %v, want ErrCollectionRateLimited", err)
	}
}

func TestReclaimReleasesBytes(t *testing.T) {
	h, err := newHNSW(Config{
		Dim: 4, Metric: L2, M: 16, EfConstruction: 200, EfSearch: 64, Seed: 1,
		MaxBytes: 1_000_000,
	})
	if err != nil {
		t.Fatalf("newHNSW: %v", err)
	}
	for i := 1; i <= 5; i++ {
		if _, _, err := h.Insert(uint64(i), []float32{float32(i), 0, 0, 0}, 0, nil, nil, nil, CASCond{}); err != nil {
			t.Fatalf("Insert %d: %v", i, err)
		}
	}
	insertBytes := estimateInsertBytes(4, 16)
	if got, want := h.bytesUsed, insertBytes*5; got != want {
		t.Fatalf("bytesUsed after 5 inserts = %d, want %d", got, want)
	}
	for i := uint64(1); i <= 3; i++ {
		if ok, _ := h.Delete(i, CASCond{}); !ok {
			t.Fatalf("Delete %d: not present", i)
		}
	}
	if got, want := h.Reclaim(), 3; got != want {
		t.Fatalf("Reclaim = %d, want %d", got, want)
	}
	if got, want := h.bytesUsed, insertBytes*2; got != want {
		t.Fatalf("bytesUsed after reclaim = %d, want %d", got, want)
	}
}
