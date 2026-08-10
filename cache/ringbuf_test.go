// SPDX-License-Identifier: Apache-2.0

package cache

import (
	"bytes"
	"testing"
)

func TestEntryEncodeDecode(t *testing.T) {
	buf := make([]byte, 256)
	key := []byte("user:42")
	val := []byte(`{"coins":100}`)
	expiry := uint64(1717000000000) // arbitrary ms
	meta := makeMeta(987654321, false)

	n, err := encodeEntry(buf, key, val, expiry, meta)
	if err != nil {
		t.Fatalf("encodeEntry: %v", err)
	}
	if n != entryHeaderSize+len(key)+len(val) {
		t.Fatalf("encoded size = %d, want %d", n, entryHeaderSize+len(key)+len(val))
	}

	dk, dv, dexp, dmeta, derr := decodeEntry(buf[:n])
	if derr != nil {
		t.Fatalf("decodeEntry: %v", derr)
	}
	if !bytes.Equal(dk, key) {
		t.Errorf("key roundtrip: got %q want %q", dk, key)
	}
	if !bytes.Equal(dv, val) {
		t.Errorf("val roundtrip: got %q want %q", dv, val)
	}
	if dexp != expiry {
		t.Errorf("expiry roundtrip: got %d want %d", dexp, expiry)
	}
	if dmeta != meta {
		t.Errorf("meta roundtrip: got %#x want %#x", dmeta, meta)
	}
	// entryMetaAt is the recovery-path accessor; it must agree with the full decode
	// so the hot decode never has to widen its signature to carry meta.
	if got := entryMetaAt(buf[:n]); got != meta {
		t.Errorf("entryMetaAt = %#x, want %#x", got, meta)
	}
}

// TestEntryMetaPacking pins the meta word's field split: 56 bits of sequence, one
// tombstone flag above them, and no interference in either direction.
func TestEntryMetaPacking(t *testing.T) {
	const maxSeq = entrySeqMask // 2^56 - 1
	for _, tc := range []struct {
		seq  uint64
		tomb bool
	}{
		{0, false}, {0, true}, {1, false}, {1, true},
		{maxSeq, false}, {maxSeq, true}, {1 << 55, true},
	} {
		m := makeMeta(tc.seq, tc.tomb)
		if got := metaSeq(m); got != tc.seq {
			t.Errorf("makeMeta(%d,%v): seq = %d, want %d", tc.seq, tc.tomb, got, tc.seq)
		}
		if got := metaIsTombstone(m); got != tc.tomb {
			t.Errorf("makeMeta(%d,%v): tombstone = %v, want %v", tc.seq, tc.tomb, got, tc.tomb)
		}
	}
	// A tombstone must not be forgeable by a large sequence alone.
	if metaIsTombstone(makeMeta(entrySeqMask, false)) {
		t.Fatal("the maximum sequence set the tombstone flag — the mask is wrong")
	}
}

// TestEntryGoldenBytes pins the exact on-disk serialization of a known entry.
// The ring-buffer entry codec is a persisted format: any drift (endianness,
// field order/width, CRC) silently breaks reopening a DataDir written by an
// older build. This golden vector fails the suite the moment the codec changes
// so such a change must be paired with a cacheVersion bump (see cache/file.go).
//
// It also pins the ORDERING decision the v4 format rests on: meta sits between the
// expiry and the CRC, i.e. BEFORE the key, so the key still starts at the fixed
// entryHeaderSize offset and decodeEntryFast needs no extra arithmetic.
func TestEntryGoldenBytes(t *testing.T) {
	key := []byte("user:1")
	val := []byte("hi")
	expiry := uint64(0x0102030405060708)
	meta := makeMeta(0x0102030405, true) // both the flag and the sequence are non-trivial

	// Layout (little-endian): keyLen(2) valLen(4) expiry(8) meta(8) crc(4) key value.
	want := []byte{
		0x06, 0x00, // keyLen = 6
		0x02, 0x00, 0x00, 0x00, // valLen = 2
		0x08, 0x07, 0x06, 0x05, 0x04, 0x03, 0x02, 0x01, // expiry
		0x05, 0x04, 0x03, 0x02, 0x01, 0x00, 0x00, 0x01, // meta = tombstone | seq 0x0102030405
		0xc6, 0x9d, 0x2a, 0x1f, // crc32(IEEE) over the 22 header bytes + key + value
		0x75, 0x73, 0x65, 0x72, 0x3a, 0x31, // "user:1"
		0x68, 0x69, // "hi"
	}

	buf := make([]byte, len(want))
	n, err := encodeEntry(buf, key, val, expiry, meta)
	if err != nil {
		t.Fatalf("encodeEntry: %v", err)
	}
	if n != len(want) {
		t.Fatalf("encoded length = %d, want %d", n, len(want))
	}
	if !bytes.Equal(buf, want) {
		t.Fatalf("golden mismatch:\n got %x\nwant %x", buf, want)
	}
}

// TestEntryCRCCoversMeta: the meta word is inside the checksummed range, so a
// flipped sequence or tombstone bit is a detected corruption, not a silently
// accepted change of recency.
func TestEntryCRCCoversMeta(t *testing.T) {
	buf := make([]byte, 64)
	n, err := encodeEntry(buf, []byte("k"), []byte("v"), 0, makeMeta(7, false))
	if err != nil {
		t.Fatal(err)
	}
	buf[entryMetaOff] ^= 0x01
	if _, _, _, _, err := decodeEntry(buf[:n]); err != errCRCMismatch {
		t.Fatalf("decodeEntry after flipping a meta bit = %v, want errCRCMismatch", err)
	}
}

func TestEntryDecodeCorruption(t *testing.T) {
	buf := make([]byte, 64)
	n, _ := encodeEntry(buf, []byte("k"), []byte("v"), 0, 0)
	// flip a payload byte after the header
	buf[entryHeaderSize] ^= 0xFF
	if _, _, _, _, err := decodeEntry(buf[:n]); err == nil {
		t.Fatal("expected CRC error after payload mutation, got nil")
	}
}

func TestEntryEncodeBufferTooSmall(t *testing.T) {
	buf := make([]byte, 4)
	_, err := encodeEntry(buf, []byte("longer-than-buffer"), []byte("v"), 0, 0)
	if err == nil {
		t.Fatal("expected errBufferTooSmall")
	}
}

func TestEntryKeyTooLong(t *testing.T) {
	buf := make([]byte, 1<<16)
	key := make([]byte, 1<<16) // exceeds uint16 max
	_, err := encodeEntry(buf, key, []byte("v"), 0, 0)
	if err == nil {
		t.Fatal("expected key-too-long error")
	}
}
