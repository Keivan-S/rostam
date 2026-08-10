// SPDX-License-Identifier: Apache-2.0

package cache

import (
	"bytes"
	"encoding/binary"
	"testing"
)

func newTestPage(t *testing.T, size int) *page {
	t.Helper()
	p := newHeapPage(size)
	if len(p.entries()) != size {
		t.Fatalf("page entries len = %d, want %d", len(p.entries()), size)
	}
	return p
}

func TestPageWriteThenRead(t *testing.T) {
	p := newTestPage(t, 1<<20)
	key := []byte("k1")
	val := []byte("v1")

	off, _, err := p.Write(key, val, 0, 0)
	if err != nil {
		t.Fatalf("Write: %v", err)
	}
	gotKey, gotVal, exp, err := p.Read(off)
	if err != nil {
		t.Fatalf("Read: %v", err)
	}
	if !bytes.Equal(gotKey, key) || !bytes.Equal(gotVal, val) {
		t.Fatalf("roundtrip mismatch: %q=%q expiry=%d", gotKey, gotVal, exp)
	}
}

func TestPageWriteReturnsErrPageFullWhenNoRoom(t *testing.T) {
	// page just big enough for one small entry.
	p := newTestPage(t, entryHeaderSize+8)
	if _, _, err := p.Write([]byte("k"), []byte("v"), 0, 0); err != nil {
		t.Fatalf("first write should succeed: %v", err)
	}
	if _, _, err := p.Write([]byte("k2"), []byte("vv"), 0, 0); err != errPageFull {
		t.Fatalf("second write should return errPageFull, got %v", err)
	}
}

func TestPageEvictReclaimsSpaceFromHead(t *testing.T) {
	// Build a page where two entries fit, evict the first, and write a third.
	keyA, valA := []byte("a"), []byte("AAAA")
	keyB, valB := []byte("b"), []byte("BBBB")
	// Size for exactly 2 entries initially, then both evicted to make room
	size := 2 * (entryHeaderSize + len(keyA) + len(valA))
	p := newTestPage(t, size)

	offA, szA, err := p.Write(keyA, valA, 0, 0)
	if err != nil {
		t.Fatalf("write A: %v", err)
	}
	if _, _, err := p.Write(keyB, valB, 0, 0); err != nil {
		t.Fatalf("write B: %v", err)
	}
	// page is full; until eviction, no writes should succeed.
	if _, _, err := p.Write([]byte("c"), []byte("CCCC"), 0, 0); err != errPageFull {
		t.Fatalf("write C without evict: got %v, want errPageFull", err)
	}
	// Evict the front entry (entry A).
	evictedKey, evictedSize, err := p.EvictFront()
	if err != nil {
		t.Fatalf("EvictFront: %v", err)
	}
	if !bytes.Equal(evictedKey, keyA) {
		t.Errorf("evicted key = %q, want %q", evictedKey, keyA)
	}
	if evictedSize != szA || evictedSize == 0 {
		t.Errorf("evicted size = %d, want %d", evictedSize, szA)
	}
	if offA != 0 {
		t.Errorf("first entry expected at offset 0, got %d", offA)
	}
	// page still full (B occupies the tail). Evict B.
	if _, _, err := p.EvictFront(); err != nil {
		t.Fatalf("EvictFront (B): %v", err)
	}
	// Now page is empty and reset; write C should succeed.
	if _, _, err := p.Write([]byte("c"), []byte("CCCC"), 0, 0); err != nil {
		t.Fatalf("write C after evict: %v", err)
	}
}

func TestEvictFrontRejectsTornEntry(t *testing.T) {
	// A crash-torn or stale entry header can carry a garbage valLen. EvictFront
	// must validate the whole entry lies within [head, tail) before advancing
	// head — otherwise it walks head past tail, breaking the 0 <= head <= tail
	// invariant and wedging the page.
	p := newTestPage(t, 1<<20)
	if _, _, err := p.Write([]byte("k"), []byte("vvvv"), 0, 0); err != nil {
		t.Fatalf("Write: %v", err)
	}
	headBefore, tailBefore := p.head(), p.tail()

	// Corrupt valLen in place to a value that runs past tail.
	binary.LittleEndian.PutUint32(p.entries()[2:6], 0xFFFFFFF0)

	_, _, err := p.EvictFront()
	if err != errEntryTruncated {
		t.Fatalf("EvictFront on torn entry = %v, want errEntryTruncated", err)
	}
	if p.head() != headBefore {
		t.Errorf("head advanced to %d on torn entry (was %d); invariant broken", p.head(), headBefore)
	}
	if p.head() > p.tail() {
		t.Errorf("head=%d > tail=%d after torn EvictFront", p.head(), p.tail())
	}
	_ = tailBefore
}

func TestPageMmapWriteRead(t *testing.T) {
	region := make([]byte, 4096)
	p := newMmapPage(region)
	off, _, err := p.Write([]byte("k"), []byte("v"), 0, 0)
	if err != nil {
		t.Fatal(err)
	}
	key, val, _, err := p.Read(off)
	if err != nil {
		t.Fatal(err)
	}
	if string(key) != "k" || string(val) != "v" {
		t.Errorf("got %q=%q, want k=v", key, val)
	}
}

func TestPageMmapPersistedAcrossNewMmapPage(t *testing.T) {
	region := make([]byte, 4096)
	p1 := newMmapPage(region)
	off, sz, _ := p1.Write([]byte("k"), []byte("v"), 0, 0)

	// Simulate process restart: fresh page object on the same region.
	p2 := newMmapPage(region)
	key, val, _, err := p2.Read(off)
	if err != nil {
		t.Fatal(err)
	}
	if string(key) != "k" || string(val) != "v" {
		t.Errorf("got %q=%q after re-wrap, want k=v", key, val)
	}
	// head and tail should also survive.
	if p2.head() != 0 || p2.tail() != int(sz) {
		t.Errorf("head=%d tail=%d, want 0 and %d", p2.head(), p2.tail(), sz)
	}
}

func TestPageMmapHeadTailInData(t *testing.T) {
	region := make([]byte, 4096)
	p := newMmapPage(region)
	if _, _, err := p.Write([]byte("k"), []byte("v"), 0, 0); err != nil {
		t.Fatal(err)
	}
	// The first 8 bytes of region should contain head/tail.
	if region[0] != 0 {
		t.Errorf("byte 0 (head LSB) = %d, want 0", region[0])
	}
	if region[4] == 0 && region[5] == 0 && region[6] == 0 && region[7] == 0 {
		t.Error("tail bytes are all zero — Write didn't update tail in mmap region")
	}
}

func TestPageHeapBackingUnchanged(t *testing.T) {
	// Heap-mode behavior must match heap-only mode exactly.
	p := newHeapPage(4096)
	off, _, _ := p.Write([]byte("k"), []byte("v"), 0, 0)
	key, val, _, _ := p.Read(off)
	if string(key) != "k" || string(val) != "v" {
		t.Errorf("heap mode roundtrip broken: %q=%q", key, val)
	}
}
