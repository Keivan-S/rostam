// SPDX-License-Identifier: Apache-2.0

package vector

import (
	"sync/atomic"
	"time"
)

// LatencyHistogram is a fixed log-scale latency distribution. The 10 buckets
// cover 1 µs, 10 µs, 100 µs, 1 ms, 10 ms, 100 ms, 1 s, 10 s, 100 s, and ∞.
// Bucket index i counts observations with duration < latencyBucketBounds[i],
// with the last bucket capturing everything above 100 s.
//
// LatencyHistogram is a value type — copying it (e.g., when returned from
// Stats()) is a shallow snapshot. Observations through the live index update
// the source counters via atomic ops; the snapshot is read-once and stale by
// the time the caller inspects it (acceptable for metering).
type LatencyHistogram struct {
	Buckets [latencyBucketCount]uint64
	Count   uint64
	Sum     uint64 // microseconds total, for mean = Sum/Count
}

const latencyBucketCount = 10

// latencyBucketBounds gives the upper edge (exclusive) of each bucket, in
// microseconds. The last bound (math.MaxUint64) is the catch-all.
var latencyBucketBounds = [latencyBucketCount]uint64{
	1, 10, 100, 1_000, 10_000, 100_000, 1_000_000, 10_000_000, 100_000_000, ^uint64(0),
}

// latencyHistogram is the live, atomic counterpart of LatencyHistogram.
// Observations are recorded via atomic.AddUint64 on each of the 12 counters.
type latencyHistogram struct {
	buckets [latencyBucketCount]atomic.Uint64
	count   atomic.Uint64
	sum     atomic.Uint64
}

// observe records one latency sample. d is rounded to microseconds and
// dropped into the lowest bucket whose bound exceeds it.
func (h *latencyHistogram) observe(d time.Duration) {
	if h == nil {
		return
	}
	micros := uint64(d / time.Microsecond)
	idx := latencyBucketCount - 1
	for i, bound := range latencyBucketBounds {
		if micros < bound {
			idx = i
			break
		}
	}
	h.buckets[idx].Add(1)
	h.count.Add(1)
	h.sum.Add(micros)
}

// snapshot copies the live counters into a value-type LatencyHistogram.
func (h *latencyHistogram) snapshot() LatencyHistogram {
	var out LatencyHistogram
	if h == nil {
		return out
	}
	for i := range h.buckets {
		out.Buckets[i] = h.buckets[i].Load()
	}
	out.Count = h.count.Load()
	out.Sum = h.sum.Load()
	return out
}
