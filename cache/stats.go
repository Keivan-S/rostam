// SPDX-License-Identifier: Apache-2.0

package cache

// Stats is a snapshot of cache counters.
// All fields are accumulated since cache creation. Use rate calculations
// by sampling and diffing.
type Stats struct {
	Gets             uint64
	Hits             uint64
	Misses           uint64
	Puts             uint64
	Dels             uint64
	Expirations      uint64 // expired by sweeper or lazy-on-read
	Evictions        uint64 // displaced by ringbuf eviction
	Rejects          uint64 // refused due to PolicyRejectWrites
	PagesAllocated   uint64
	BytesAllocated   uint64
	BytesUsed        uint64
	CorruptionErrors uint64 // CRC mismatches on read

	// Cold compaction at shard open (mmap only; cache/compact.go). These are the
	// operator's view of whether restarts are actually reclaiming the ghost page
	// bytes a persistent shard cannot reclaim while running:
	//   - Compactions: pages files rewritten live-only and swapped in;
	//   - CompactionsAborted: rewrites decided against or abandoned (no space to
	//     stage, pack overflow, failed rename) — the original file was kept;
	//   - CompactionBytesReclaimed: page bytes dropped by those rewrites;
	//   - CompactionDurationMs: total time spent in them (they run at open, so
	//     this is startup latency).
	Compactions              uint64
	CompactionsAborted       uint64
	CompactionBytesReclaimed uint64
	CompactionDurationMs     uint64
}

// HitRate returns Hits / Gets, or 0 when Gets == 0.
func (s Stats) HitRate() float64 {
	if s.Gets == 0 {
		return 0
	}
	return float64(s.Hits) / float64(s.Gets)
}
