// SPDX-License-Identifier: Apache-2.0

package cache

import (
	"encoding/binary"
	"fmt"
	"path/filepath"
	"testing"
)

// ============================================================================
// The DURABLE PB FRONTIER in the cache header.
//
// The one rule every test here is about: the persisted watermark may UNDER-report
// freely (it only costs catch-up work) and must NEVER over-report (the node would
// claim a prefix it does not hold, and pbisr's log matching compares an incoming
// frame against this very number, so it would certify a divergent append).
// ============================================================================

// TestPBFrontierHeaderRoundTrip: the field survives a write/read cycle and does
// not disturb ANY byte the pre-existing readers depend on — the core header
// (0..27 + its CRC) and the v3 logical clock (32..43) must be untouched.
func TestPBFrontierHeaderRoundTrip(t *testing.T) {
	region := make([]byte, headerSize)
	writeHeader(region, 4096, 8, 99)
	setAppliedStamp(region, 12345)

	// A freshly written header states genesis with a VALID checksum, not an
	// unreadable zero blob that merely decodes to the same answer.
	if seq, epoch := readPBFrontier(region); seq != 0 || epoch != 0 {
		t.Fatalf("fresh header PB frontier = (%d,%d), want (0,0)", seq, epoch)
	}
	crcBefore := binary.LittleEndian.Uint32(region[28:32])

	setPBFrontier(region, 7, 3)
	seq, epoch := readPBFrontier(region)
	if seq != 7 || epoch != 3 {
		t.Fatalf("PB frontier = (%d,%d), want (7,3)", seq, epoch)
	}
	// Core header untouched: same CRC, same applied index, still validates.
	if got := binary.LittleEndian.Uint32(region[28:32]); got != crcBefore {
		t.Fatalf("header CRC changed by setPBFrontier: %x -> %x", crcBefore, got)
	}
	idx, fresh, err := validateHeader(region, 4096, 8)
	if err != nil || fresh || idx != 99 {
		t.Fatalf("validateHeader after setPBFrontier: idx=%d fresh=%v err=%v", idx, fresh, err)
	}
	if got := readAppliedStamp(region); got != 12345 {
		t.Fatalf("logical clock clobbered by setPBFrontier: %d", got)
	}

	// ...and the reverse: stamping the logical clock must not disturb the frontier.
	setAppliedStamp(region, 999)
	if seq, epoch := readPBFrontier(region); seq != 7 || epoch != 3 {
		t.Fatalf("PB frontier clobbered by setAppliedStamp: (%d,%d)", seq, epoch)
	}
}

// TestPBFrontierUnreadableRestoresGenesis covers the three ways the field can
// fail to decode. All three must collapse to (0,0) — the maximal UNDER-report —
// rather than to whatever bytes happen to be there.
func TestPBFrontierUnreadableRestoresGenesis(t *testing.T) {
	t.Run("pre-field header (reserved zeros)", func(t *testing.T) {
		region := make([]byte, headerSize)
		writeHeader(region, 4096, 8, 5)
		// Simulate a pages file written before this field existed: the bytes are
		// reserved zeros, INCLUDING the CRC slot.
		for i := hdrPBSeqOff; i < headerSize; i++ {
			region[i] = 0
		}
		if seq, epoch := readPBFrontier(region); seq != 0 || epoch != 0 {
			t.Fatalf("pre-field header = (%d,%d), want (0,0)", seq, epoch)
		}
	})

	t.Run("torn write: value stored, CRC not yet", func(t *testing.T) {
		region := make([]byte, headerSize)
		writeHeader(region, 4096, 8, 5)
		setPBFrontier(region, 42, 2)
		// A crash between the 16-byte store and its checksum.
		binary.LittleEndian.PutUint64(region[hdrPBSeqOff:hdrPBSeqOff+8], 43)
		if seq, epoch := readPBFrontier(region); seq != 0 || epoch != 0 {
			t.Fatalf("torn frontier = (%d,%d), want (0,0) — a torn value must NEVER be trusted", seq, epoch)
		}
	})

	t.Run("torn write: epoch half only", func(t *testing.T) {
		region := make([]byte, headerSize)
		writeHeader(region, 4096, 8, 5)
		setPBFrontier(region, 42, 2)
		binary.LittleEndian.PutUint64(region[hdrPBEpochOff:hdrPBEpochOff+8], 9)
		if seq, epoch := readPBFrontier(region); seq != 0 || epoch != 0 {
			t.Fatalf("torn epoch = (%d,%d), want (0,0) — the pair is one identity, not two numbers", seq, epoch)
		}
	})
}

// TestPBFrontierSurvivesReopen is the storage-level restart round trip: what
// SetPBFrontier persisted is what a reopened cache reports.
func TestPBFrontierSurvivesReopen(t *testing.T) {
	dir := t.TempDir()
	cfg := DefaultConfig()
	cfg.NumShards = 2
	cfg.DataDir = filepath.Join(dir, "cache")

	c, err := New(cfg)
	if err != nil {
		t.Fatalf("New: %v", err)
	}
	if seq, epoch := c.PBFrontier(); seq != 0 || epoch != 0 {
		t.Fatalf("fresh cache PBFrontier = (%d,%d), want (0,0)", seq, epoch)
	}
	if err := c.Put([]byte("k"), []byte("v"), 0); err != nil {
		t.Fatalf("Put: %v", err)
	}
	c.SetPBFrontier(17, 4)
	if err := c.Close(); err != nil {
		t.Fatalf("Close: %v", err)
	}

	c2, err := New(cfg)
	if err != nil {
		t.Fatalf("reopen: %v", err)
	}
	defer func() { _ = c2.Close() }()
	seq, epoch := c2.PBFrontier()
	if seq != 17 || epoch != 4 {
		t.Fatalf("reopened PBFrontier = (%d,%d), want (17,4)", seq, epoch)
	}
	if v, err := c2.Get([]byte("k")); err != nil || string(v) != "v" {
		t.Fatalf("warm restart lost the data the frontier describes: %q %v", v, err)
	}
}

// TestPBFrontierReportsMinAcrossShards. SetPBFrontier walks the shards one at a
// time, so a crash mid-loop leaves some shards stamped and some not. Cache-level
// reporting must take the MINIMUM — the strongest claim true of EVERY shard —
// and must take the epoch from the shard holding that minimum, since a seq from
// one shard paired with an epoch from another names a write that never existed.
func TestPBFrontierReportsMinAcrossShards(t *testing.T) {
	cfg := DefaultConfig()
	cfg.NumShards = 4
	cfg.DataDir = filepath.Join(t.TempDir(), "cache")
	c, err := New(cfg)
	if err != nil {
		t.Fatalf("New: %v", err)
	}
	defer func() { _ = c.Close() }()

	c.SetPBFrontier(50, 2)
	// Simulate a crash that interrupted a later SetPBFrontier(60,3) partway: two
	// shards took the new pair, two kept the old one.
	c.shards[0].pbFrontierSeq.Store(60)
	c.shards[0].pbFrontierEpoch.Store(3)
	c.shards[1].pbFrontierSeq.Store(60)
	c.shards[1].pbFrontierEpoch.Store(3)

	seq, epoch := c.PBFrontier()
	if seq != 50 || epoch != 2 {
		t.Fatalf("PBFrontier = (%d,%d), want (50,2) — the min pair, not a mix", seq, epoch)
	}
}

// TestPBFrontierHeapModeIsGenesis: a heap-mode cache has no header to stamp, so
// it reports genesis and SetPBFrontier is a no-op. This is what keeps the PB seam
// unconditional at the call site.
func TestPBFrontierHeapModeIsGenesis(t *testing.T) {
	cfg := DefaultConfig()
	cfg.NumShards = 2 // no DataDir ⇒ heap
	c, err := New(cfg)
	if err != nil {
		t.Fatalf("New: %v", err)
	}
	defer func() { _ = c.Close() }()

	c.SetPBFrontier(9, 1)
	if seq, epoch := c.PBFrontier(); seq != 0 || epoch != 0 {
		t.Fatalf("heap-mode PBFrontier = (%d,%d), want (0,0)", seq, epoch)
	}
}

// TestPBFrontierSurvivesColdCompaction. Cold compaction at open REWRITES the
// pages file through writeHeader, which zero-fills the reserved area. It must
// carry the frontier across: compaction preserves the live entry set exactly, so
// the frontier that described the original still describes the compacted file.
// Dropping it would be safe but would silently reset every durable PB node to
// genesis on whichever restart happened to compact.
func TestPBFrontierSurvivesColdCompaction(t *testing.T) {
	dir := t.TempDir()
	cfg := compactTestCfg(dir, true, 5_000_000)

	c := openCompactCache(t, cfg)
	s := c.shards[0]
	// Overwrite a small key set until the shard is full: the superseded copies are
	// the ghost bytes that make compaction at the next open worth running.
	fillWithOverwrites(t, 8, func(i int) []byte { return []byte(fmt.Sprintf("live%04d", i)) },
		func(k, v []byte) error { return c.Put(k, v, 0) })
	if occ := s.occupancyRatio(); occ < mmapCompactMinOccupancy {
		t.Fatalf("setup: occupancy %.3f < %.2f — compaction would not run, so this test would prove nothing",
			occ, mmapCompactMinOccupancy)
	}
	usedBefore := s.bytesUsed()
	c.SetPBFrontier(200, 5)
	if err := c.Close(); err != nil {
		t.Fatalf("Close: %v", err)
	}

	c2 := openCompactCache(t, cfg) // rebuild + compactAtOpen
	defer func() { _ = c2.Close() }()
	if used := c2.shards[0].bytesUsed(); used >= usedBefore {
		t.Fatalf("setup: bytesUsed %d -> %d — compaction did not actually run", usedBefore, used)
	}
	if seq, epoch := c2.PBFrontier(); seq != 200 || epoch != 5 {
		t.Fatalf("post-compaction PBFrontier = (%d,%d), want (200,5)", seq, epoch)
	}
}
