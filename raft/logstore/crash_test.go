// SPDX-License-Identifier: Apache-2.0

package logstore

import (
	"encoding/binary"
	"fmt"
	"hash/crc32"
	"os"
	"path/filepath"
	"testing"

	hraft "github.com/hashicorp/raft"
)

// completeRecordsPrefix returns the index of the last log entry whose record is
// FULLY and validly present in b, mirroring exactly what WAL.scanSegment accepts.
// It is the oracle for "a crash truncated the file to len(b) bytes".
func completeRecordsPrefix(t *testing.T, b []byte) uint64 {
	t.Helper()
	var off int
	var last uint64
	for off < len(b) {
		if off+frameHdr > len(b) {
			break // short frame header => torn
		}
		recLen := binary.LittleEndian.Uint32(b[off : off+recLenSize])
		if recLen < crcSize || recLen > maxRecBytes {
			break
		}
		total := recLenSize + int(recLen)
		if off+total > len(b) {
			break // body missing => torn
		}
		payload := b[off+frameHdr : off+total]
		if binary.LittleEndian.Uint32(b[off+recLenSize:off+frameHdr]) != crc32.ChecksumIEEE(payload) {
			break // bad crc => torn
		}
		last = binary.LittleEndian.Uint64(payload) // index is first payload field
		off += total
	}
	return last
}

func writeSegment(t *testing.T, dir string, base uint64, raw []byte) {
	t.Helper()
	name := fmt.Sprintf("%020d%s", base, segExt)
	if err := os.WriteFile(filepath.Join(dir, name), raw, 0o600); err != nil {
		t.Fatal(err)
	}
}

// TestCrash_TruncationSweep is the core crash-injection test. It builds a
// reference log, captures the raw segment bytes, then for EVERY truncation
// length L (every possible "crash after writing L bytes") recovers a fresh WAL
// and asserts:
//   - OpenWAL never errors or panics,
//   - the recovered log is exactly the complete-record prefix that fits in L,
//   - every recovered entry decodes to its original value,
//   - the WAL is usable afterwards (a fresh append lands at the right index).
func TestCrash_TruncationSweep(t *testing.T) {
	const N = 40
	refDir := t.TempDir()
	w, err := OpenWAL(refDir, true)
	if err != nil {
		t.Fatal(err)
	}
	for i := uint64(1); i <= N; i++ {
		if err := w.StoreLog(mkLog(i, fmt.Sprintf("payload-%03d", i))); err != nil {
			t.Fatal(err)
		}
	}
	_ = w.Close()

	segs, _ := filepath.Glob(filepath.Join(refDir, "*"+segExt))
	if len(segs) != 1 {
		t.Fatalf("expected 1 segment, got %d", len(segs))
	}
	raw, err := os.ReadFile(segs[0])
	if err != nil {
		t.Fatal(err)
	}

	for L := 0; L <= len(raw); L++ {
		wantLast := completeRecordsPrefix(t, raw[:L])
		dir := t.TempDir()
		writeSegment(t, dir, 1, raw[:L])

		w2, err := OpenWAL(dir, true)
		if err != nil {
			t.Fatalf("L=%d: OpenWAL: %v", L, err)
		}
		gotLast, _ := w2.LastIndex()
		if gotLast != wantLast {
			t.Fatalf("L=%d: recovered LastIndex=%d want %d", L, gotLast, wantLast)
		}
		var out hraft.Log
		for i := uint64(1); i <= gotLast; i++ {
			if err := w2.GetLog(i, &out); err != nil {
				t.Fatalf("L=%d: GetLog(%d): %v", L, i, err)
			}
			if want := fmt.Sprintf("payload-%03d", i); string(out.Data) != want {
				t.Fatalf("L=%d: GetLog(%d) data=%q want %q", L, i, out.Data, want)
			}
		}
		// The recovered WAL must be usable: appending at last+1 must succeed and
		// read back — proves recovery left a consistent write cursor.
		next := gotLast + 1
		if err := w2.StoreLog(mkLog(next, "after-recovery")); err != nil {
			t.Fatalf("L=%d: append after recovery: %v", L, err)
		}
		if err := w2.GetLog(next, &out); err != nil || string(out.Data) != "after-recovery" {
			t.Fatalf("L=%d: read appended entry: err=%v data=%q", L, err, out.Data)
		}
		_ = w2.Close()
	}
}

// TestCrash_MidFileCorruption flips a byte inside a middle record. Recovery must
// treat it as a torn tail from that point: earlier records survive, the corrupt
// record and everything after it are discarded (a corrupt record means the rest
// of the file is untrustworthy).
func TestCrash_MidFileCorruption(t *testing.T) {
	refDir := t.TempDir()
	w, err := OpenWAL(refDir, true)
	if err != nil {
		t.Fatal(err)
	}
	for i := uint64(1); i <= 30; i++ {
		_ = w.StoreLog(mkLog(i, fmt.Sprintf("v%02d", i)))
	}
	_ = w.Close()
	segs, _ := filepath.Glob(filepath.Join(refDir, "*"+segExt))
	raw, _ := os.ReadFile(segs[0])

	// Find the offset of record #15 and corrupt a payload byte inside it.
	off := 0
	var recStart int
	for idx := uint64(1); ; idx++ {
		recLen := int(binary.LittleEndian.Uint32(raw[off : off+recLenSize]))
		if idx == 15 {
			recStart = off
			break
		}
		off += recLenSize + recLen
	}
	corrupt := append([]byte(nil), raw...)
	corrupt[recStart+frameHdr+2] ^= 0xff // flip a payload byte => CRC fails

	dir := t.TempDir()
	writeSegment(t, dir, 1, corrupt)
	w2, err := OpenWAL(dir, true)
	if err != nil {
		t.Fatalf("OpenWAL with mid-file corruption: %v", err)
	}
	defer w2.Close()
	if last, _ := w2.LastIndex(); last != 14 {
		t.Fatalf("LastIndex=%d want 14 (records 15+ discarded after corruption)", last)
	}
	var out hraft.Log
	if err := w2.GetLog(14, &out); err != nil || string(out.Data) != "v14" {
		t.Fatalf("GetLog(14): err=%v data=%q", err, out.Data)
	}
	if err := w2.GetLog(15, &out); err != hraft.ErrLogNotFound {
		t.Fatalf("GetLog(15)=%v want ErrLogNotFound", err)
	}
}

// TestCrash_BatchSpansRotation stores a batch large enough to cross a segment
// rotation, then reopens. Every entry must survive — the fix fsyncs BOTH the old
// and new segment (and the directory), not just the active one.
func TestCrash_BatchSpansRotation(t *testing.T) {
	dir := t.TempDir()
	w, err := OpenWAL(dir, true)
	if err != nil {
		t.Fatal(err)
	}
	w.maxSeg = 256 // tiny, so an 8-entry batch rotates mid-batch
	batch := make([]*hraft.Log, 8)
	for i := range batch {
		batch[i] = &hraft.Log{Index: uint64(i + 1), Term: 1, Type: hraft.LogCommand, Data: []byte(fmt.Sprintf("entry-%d-padding-padding", i+1))}
	}
	if err := w.StoreLogs(batch); err != nil {
		t.Fatal(err)
	}
	segs, _ := filepath.Glob(filepath.Join(dir, "*"+segExt))
	if len(segs) < 2 {
		t.Fatalf("batch did not span a rotation: %d segment(s)", len(segs))
	}
	_ = w.Close()

	w2, err := OpenWAL(dir, true)
	if err != nil {
		t.Fatal(err)
	}
	defer w2.Close()
	if last, _ := w2.LastIndex(); last != 8 {
		t.Fatalf("recovered LastIndex=%d want 8 (rotation-spanning batch must survive whole)", last)
	}
	var out hraft.Log
	for i := uint64(1); i <= 8; i++ {
		if err := w2.GetLog(i, &out); err != nil || out.Index != i {
			t.Fatalf("GetLog(%d) across rotation: err=%v idx=%d", i, err, out.Index)
		}
	}
}

// TestCrash_TailTruncationResurrection is the HIGH3 scenario: a tail-truncation
// removed a later segment and re-appended a conflicting entry at a HIGHER term,
// but a crash left the removed segment on disk. Recovery must NOT resurrect the
// stale entries — the term-monotonicity check rejects them (their term predates
// the conflict).
func TestCrash_TailTruncationResurrection(t *testing.T) {
	dir := t.TempDir()
	w, err := OpenWAL(dir, true)
	if err != nil {
		t.Fatal(err)
	}
	w.maxSeg = 200 // force a second segment for indices 11+

	// Term-1 entries 1..20 across two segments.
	for i := uint64(1); i <= 20; i++ {
		_ = w.StoreLog(&hraft.Log{Index: i, Term: 1, Type: hraft.LogCommand, Data: []byte(fmt.Sprintf("old-%02d", i))})
	}
	// Capture the second segment's file (holds old 11..20) to "resurrect" later.
	segs, _ := filepath.Glob(filepath.Join(dir, "*"+segExt))
	if len(segs) < 2 {
		t.Fatalf("need >=2 segments, got %d", len(segs))
	}
	staleSeg := segs[len(segs)-1]
	staleBytes, _ := os.ReadFile(staleSeg)

	// Conflict at 11: truncate the tail, re-append 11.. at term 2 (higher).
	_ = w.DeleteRange(11, 20)
	for i := uint64(11); i <= 15; i++ {
		_ = w.StoreLog(&hraft.Log{Index: i, Term: 2, Type: hraft.LogCommand, Data: []byte(fmt.Sprintf("new-%02d", i))})
	}
	_ = w.Close()

	// Simulate a crash where the removed stale segment was NOT durably deleted:
	// put its old bytes back on disk.
	if err := os.WriteFile(staleSeg, staleBytes, 0o600); err != nil {
		t.Fatal(err)
	}

	w2, err := OpenWAL(dir, true)
	if err != nil {
		t.Fatalf("recover with resurrected stale segment: %v", err)
	}
	defer w2.Close()
	var out hraft.Log
	// 11 must be the NEW term-2 value, never the resurrected old one.
	if err := w2.GetLog(11, &out); err != nil || string(out.Data) != "new-11" {
		t.Fatalf("GetLog(11)=%q (err %v) want new-11 — stale entry resurrected!", out.Data, err)
	}
	// 16..20 (old, term 1) must be gone: they were truncated and their stale
	// segment records must be rejected by the term check.
	if err := w2.GetLog(16, &out); err != hraft.ErrLogNotFound {
		t.Fatalf("GetLog(16)=%v want ErrLogNotFound — old truncated entry resurrected!", err)
	}
}

// TestCrash_FullResetClearsFloor verifies a full-log delete (resetLocked path)
// removes the floor file and every segment, and reopening yields an empty log —
// no stale floor or segment lingers to be rebuilt.
func TestCrash_FullResetClearsFloor(t *testing.T) {
	dir := t.TempDir()
	w, err := OpenWAL(dir, true)
	if err != nil {
		t.Fatal(err)
	}
	for i := uint64(1); i <= 30; i++ {
		_ = w.StoreLog(mkLog(i, fmt.Sprintf("v%d", i)))
	}
	_ = w.DeleteRange(1, 10) // create a floor file
	if _, err := os.Stat(filepath.Join(dir, "first")); err != nil {
		t.Fatalf("expected floor file after compaction: %v", err)
	}
	_ = w.DeleteRange(11, 30) // full delete of the remaining log => resetLocked
	if _, err := os.Stat(filepath.Join(dir, "first")); !os.IsNotExist(err) {
		t.Fatal("floor file must be removed on full reset")
	}
	segs, _ := filepath.Glob(filepath.Join(dir, "*"+segExt))
	if len(segs) != 0 {
		t.Fatalf("segments must be removed on full reset, found %d", len(segs))
	}
	_ = w.Close()

	w2, err := OpenWAL(dir, true)
	if err != nil {
		t.Fatal(err)
	}
	defer w2.Close()
	if fi, _ := w2.FirstIndex(); fi != 0 {
		t.Fatalf("reopened FirstIndex=%d want 0 (empty)", fi)
	}
	if li, _ := w2.LastIndex(); li != 0 {
		t.Fatalf("reopened LastIndex=%d want 0 (empty)", li)
	}
}

// TestCrash_TornStableFileRejected corrupts the stable (vote/term) file and
// verifies OpenWAL refuses to start rather than loading a partial vote — the CRC
// added for the CRITICAL finding.
func TestCrash_TornStableFileRejected(t *testing.T) {
	dir := t.TempDir()
	w, err := OpenWAL(dir, true)
	if err != nil {
		t.Fatal(err)
	}
	_ = w.SetUint64([]byte("currentTerm"), 9)
	_ = w.Set([]byte("votedFor"), []byte("nodeA"))
	_ = w.Close()

	// Flip a byte in the stable file.
	sp := filepath.Join(dir, "stable")
	b, _ := os.ReadFile(sp)
	if len(b) == 0 {
		t.Fatal("empty stable file")
	}
	b[len(b)/2] ^= 0xff
	_ = os.WriteFile(sp, b, 0o600)

	if _, err := OpenWAL(dir, true); err == nil {
		t.Fatal("OpenWAL accepted a corrupt stable file; must reject to avoid a lost/partial vote")
	}
}

// TestCrash_StaleTmpFiles leaves the atomic-write temp files behind (as a crash
// mid-rename would) and checks recovery ignores them — they must not be parsed
// as segments and must not corrupt the floor/stable state.
func TestCrash_StaleTmpFiles(t *testing.T) {
	dir := t.TempDir()
	w, err := OpenWAL(dir, true)
	if err != nil {
		t.Fatal(err)
	}
	for i := uint64(1); i <= 10; i++ {
		_ = w.StoreLog(mkLog(i, fmt.Sprintf("v%d", i)))
	}
	_ = w.SetUint64([]byte("term"), 3)
	_ = w.Close()

	// Simulate crashes mid-rename: orphaned .tmp files.
	_ = os.WriteFile(filepath.Join(dir, "first.tmp"), []byte("garbage"), 0o600)
	_ = os.WriteFile(filepath.Join(dir, "stable.tmp"), []byte("garbage"), 0o600)
	_ = os.WriteFile(filepath.Join(dir, "notasegment.txt"), []byte("x"), 0o600)

	w2, err := OpenWAL(dir, true)
	if err != nil {
		t.Fatalf("OpenWAL with stale tmp files: %v", err)
	}
	defer w2.Close()
	if last, _ := w2.LastIndex(); last != 10 {
		t.Fatalf("LastIndex=%d want 10", last)
	}
	if term, _ := w2.GetUint64([]byte("term")); term != 3 {
		t.Fatalf("term=%d want 3", term)
	}
}

// TestCrash_DuringFrontTruncation simulates a crash mid-compaction: the floor
// file was written (recording the new first index) but the now-dead segment
// records were never physically removed. Recovery must honour the floor and
// skip the dead prefix rather than resurrect it.
func TestCrash_DuringFrontTruncation(t *testing.T) {
	dir := t.TempDir()
	w, err := OpenWAL(dir, true)
	if err != nil {
		t.Fatal(err)
	}
	for i := uint64(1); i <= 50; i++ {
		_ = w.StoreLog(mkLog(i, fmt.Sprintf("v%02d", i)))
	}
	_ = w.Close()

	// Hand-write a floor of 21 (as if compaction advanced first to 21 then
	// crashed before rewriting/removing the segment holding 1..50).
	var b [8]byte
	binary.LittleEndian.PutUint64(b[:], 21)
	if err := os.WriteFile(filepath.Join(dir, "first"), b[:], 0o600); err != nil {
		t.Fatal(err)
	}

	w2, err := OpenWAL(dir, true)
	if err != nil {
		t.Fatalf("OpenWAL after crash-during-compaction: %v", err)
	}
	defer w2.Close()
	if first, _ := w2.FirstIndex(); first != 21 {
		t.Fatalf("FirstIndex=%d want 21 (dead prefix must stay compacted)", first)
	}
	if last, _ := w2.LastIndex(); last != 50 {
		t.Fatalf("LastIndex=%d want 50", last)
	}
	var out hraft.Log
	if err := w2.GetLog(20, &out); err != hraft.ErrLogNotFound {
		t.Fatalf("GetLog(20)=%v want ErrLogNotFound (below floor)", err)
	}
	if err := w2.GetLog(21, &out); err != nil || string(out.Data) != "v21" {
		t.Fatalf("GetLog(21): err=%v data=%q", err, out.Data)
	}
}
