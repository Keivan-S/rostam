// SPDX-License-Identifier: Apache-2.0

package vector

import (
	"testing"
	"time"
)

func TestLatencyHistogramBucketBoundaries(t *testing.T) {
	var h latencyHistogram

	cases := []struct {
		dur time.Duration
		bkt int
	}{
		{0, 0},
		{500 * time.Nanosecond, 0}, // floor to 0 µs → bucket 0 (<1µs)
		{1 * time.Microsecond, 1},  // exactly the boundary belongs to next bucket
		{9 * time.Microsecond, 1},
		{50 * time.Microsecond, 2},
		{500 * time.Microsecond, 3},
		{5 * time.Millisecond, 4},
		{50 * time.Millisecond, 5},
		{500 * time.Millisecond, 6},
		{5 * time.Second, 7},
		{50 * time.Second, 8},
		{500 * time.Second, 9}, // catch-all
	}
	for _, c := range cases {
		h.observe(c.dur)
	}

	for i, c := range cases {
		if h.buckets[c.bkt].Load() == 0 {
			t.Errorf("case %d: dur=%v expected bucket %d to be incremented", i, c.dur, c.bkt)
		}
	}
	if got, want := h.count.Load(), uint64(len(cases)); got != want {
		t.Errorf("count = %d, want %d", got, want)
	}
}

func TestLatencyHistogramSnapshot(t *testing.T) {
	var h latencyHistogram
	for i := 0; i < 5; i++ {
		h.observe(50 * time.Microsecond)
	}
	for i := 0; i < 3; i++ {
		h.observe(50 * time.Millisecond)
	}
	snap := h.snapshot()
	if snap.Buckets[2] != 5 {
		t.Errorf("snapshot Buckets[2] = %d, want 5", snap.Buckets[2])
	}
	if snap.Buckets[5] != 3 {
		t.Errorf("snapshot Buckets[5] = %d, want 3", snap.Buckets[5])
	}
	if snap.Count != 8 {
		t.Errorf("snapshot Count = %d, want 8", snap.Count)
	}
	// Sum: 5*50 + 3*50_000 = 250 + 150000 = 150250 µs
	if snap.Sum != 150_250 {
		t.Errorf("snapshot Sum = %d, want 150250", snap.Sum)
	}
}

func TestStatsIncludesHistograms(t *testing.T) {
	h, err := newHNSW(Config{Dim: 4, Metric: L2, M: 16, EfConstruction: 200, EfSearch: 64, Seed: 1})
	if err != nil {
		t.Fatalf("newHNSW: %v", err)
	}
	for i := 1; i <= 10; i++ {
		if _, _, err := h.Insert(uint64(i), []float32{float32(i), 0, 0, 0}, 0, nil, nil, nil, CASCond{}); err != nil {
			t.Fatalf("Insert %d: %v", i, err)
		}
	}
	for i := 0; i < 5; i++ {
		if _, err := h.Search([]float32{1, 0, 0, 0}, 3); err != nil {
			t.Fatalf("Search %d: %v", i, err)
		}
	}
	stats := h.Stats()
	if stats.InsertLatency.Count != 10 {
		t.Errorf("InsertLatency.Count = %d, want 10", stats.InsertLatency.Count)
	}
	if stats.SearchLatency.Count != 5 {
		t.Errorf("SearchLatency.Count = %d, want 5", stats.SearchLatency.Count)
	}
	// All buckets summed must equal Count.
	var sum uint64
	for _, b := range stats.InsertLatency.Buckets {
		sum += b
	}
	if sum != stats.InsertLatency.Count {
		t.Errorf("InsertLatency buckets sum to %d, Count is %d", sum, stats.InsertLatency.Count)
	}
}

func TestLatencyHistogramNilObserve(t *testing.T) {
	var h *latencyHistogram
	// Must not panic.
	h.observe(time.Millisecond)
	if got := h.snapshot(); got.Count != 0 {
		t.Errorf("nil snapshot Count = %d, want 0", got.Count)
	}
}
