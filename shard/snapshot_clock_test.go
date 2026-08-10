// SPDX-License-Identifier: Apache-2.0

package shard

import (
	"bytes"
	"io"
	"testing"

	"github.com/rostamlabs/rostam/cache"
)

// TestSnapshotCarriesLogicalClock pins the v4 trailer field that keeps a
// snapshot-restored node's LOGICAL CLOCK from being 0.
//
// Why it matters: the clock only advances on the STAMPED apply path, and a
// snapshot installs through PutAbs, which does not advance it. A node whose
// committed state arrives entirely by snapshot would therefore hold 0 — and if
// it were then elected leader it would stamp new writes at bare wall time,
// potentially BELOW a peer's persisted clock. Both deterministic TTL reclaimers
// (the B3b logical sweeper and cold compaction at shard open) have already
// dropped entries on the strength of "every future committed stamp dominates my
// persisted clock", so a leader stamping below it would retroactively invalidate
// that. Carrying the clock in the snapshot and folding it in on restore is what
// makes the assumption hold.
func TestSnapshotCarriesLogicalClock(t *testing.T) {
	src, err := cache.New(cache.DefaultConfig())
	if err != nil {
		t.Fatalf("cache.New: %v", err)
	}
	defer func() { _ = src.Close() }()

	const stamp = uint64(1_700_000_000_000)
	if err := src.PutAt([]byte("k1"), []byte("v1"), 0, stamp); err != nil {
		t.Fatalf("PutAt: %v", err)
	}
	if got := src.LastAppliedStampMs(); got != stamp {
		t.Fatalf("source clock = %d, want %d", got, stamp)
	}

	data, err := serializeSnapshot(src, nil, 7, nil)
	if err != nil {
		t.Fatalf("serializeSnapshot: %v", err)
	}

	dst, err := cache.New(cache.DefaultConfig())
	if err != nil {
		t.Fatalf("cache.New: %v", err)
	}
	defer func() { _ = dst.Close() }()
	if got := dst.LastAppliedStampMs(); got != 0 {
		t.Fatalf("fresh cache clock = %d, want 0", got)
	}
	if _, err := restoreSnapshot(dst, nil, nil, io.NopCloser(bytes.NewReader(data))); err != nil {
		t.Fatalf("restoreSnapshot: %v", err)
	}
	if got := dst.LastAppliedStampMs(); got != stamp {
		t.Fatalf("restored clock = %d, want %d — a snapshot-restored node must not sit at 0 "+
			"while its peers hold the leader's clock", got, stamp)
	}
	if v, err := dst.Get([]byte("k1")); err != nil || !bytes.Equal(v, []byte("v1")) {
		t.Fatalf("restored entry: %q, %v", v, err)
	}

	// The fold is a MAX: a snapshot from a LAGGING peer must never rewind a clock
	// that has already advanced further.
	dst.AdvanceAppliedStamp(stamp + 5_000)
	if _, err := restoreSnapshot(dst, nil, nil, io.NopCloser(bytes.NewReader(data))); err != nil {
		t.Fatalf("restoreSnapshot (second): %v", err)
	}
	if got := dst.LastAppliedStampMs(); got != stamp+5_000 {
		t.Fatalf("clock rewound to %d by an older snapshot, want %d", got, stamp+5_000)
	}
}
