// SPDX-License-Identifier: Apache-2.0

package cache

import (
	"encoding/binary"
	"errors"
	"hash/crc32"
)

// Entry wire layout (within a page), format version 4:
//
//	[keyLen:2][valueLen:4][expiryMs:8][meta:8][crc32:4][key][value]
//	└──────────────────entryHeaderSize (26)──────────────────┘
//
// Little-endian throughout, matching the memory layout on amd64/arm64 and
// consistent with the page head/tail offsets in page.go, which are also
// little-endian. CRC covers keyLen, valueLen, expiry, meta, key, and value
// (everything except itself) — one contiguous fixed range [0,entryCRCOff) plus
// the key/value bytes, exactly as before.
//
// WHY meta SITS BEFORE THE KEY. It is the only placement that leaves the hot
// lock-free read path untouched. decodeEntryFast reads src[0:2], src[2:6],
// src[6:14] and then slices the key/value at entryHeaderSize; putting meta at
// [14:22] changes exactly one compile-time CONSTANT in that function and nothing
// else — same loads, same order, no branch, and the read NEVER touches the meta
// word. Appending meta after the value instead would have forced the reader to
// compute a trailing offset from valLen before it could be skipped, and putting it
// between the key and value would have moved the key's start off a constant.
// decodeEntryFast's signature is deliberately left unchanged for the same reason:
// no read-path caller may acquire a way to ask for the sequence number.
//
// meta = flags<<56 | seq:
//
//	bits 0..55  seq — the shard's per-entry MONOTONIC write sequence. It is the
//	            persisted write-recency signal that makes warm restart correct:
//	            page order is NOT write order (findOrMakePageLocked revisits lower
//	            pages through firstPageWithRoomLocked), so rebuildIndexFromPages
//	            resolves a key to the copy with the HIGHEST seq rather than to the
//	            last one the page walk happens to reach (#12A).
//	bit 56      entryFlagTombstone — this entry RECORDS A DELETE. Del on a
//	            persistent shard appends one so the removal is part of the page
//	            bytes and survives the rebuild (#12B); the rebuild strips the slot
//	            afterwards and compaction drops the tombstone and every older copy
//	            of its key in the same pass.
//	bits 57..63 reserved (must be zero).
//
// 56 bits of sequence is 7.2e16 writes per shard — at ten million writes a second
// that is 228 years, so wrap is not a condition the code needs to handle.
const (
	entryHeaderSize = 2 + 4 + 8 + 8 + 4
	// entryMetaOff / entryCRCOff locate the meta word and the CRC slot. The CRC's
	// fixed-range coverage is [0, entryCRCOff).
	entryMetaOff = 14
	entryCRCOff  = 22
	maxKeyLen    = 1<<16 - 1 // uint16 max
	// maxValueLen is typed int64 (not the untyped-int default) because it is the
	// uint32 max, 4294967295, which overflows the 32-bit `int` of a 386/arm/
	// windows-386 build. int64 holds it on every platform; the len(value)
	// comparison below widens to int64 to match, which costs nothing on 64-bit
	// (where int already covers the full range) and is simply always-false on
	// 32-bit (where a slice can never reach 2^32-1 elements) rather than a
	// truncated, lower limit.
	maxValueLen int64 = 1<<32 - 1 // uint32 max (4 GiB - 1)

	// entryFlagTombstone marks an entry as a delete record. entrySeqMask isolates
	// the sequence number from the flag bits.
	entryFlagTombstone uint64 = 1 << 56
	entrySeqMask       uint64 = entryFlagTombstone - 1
)

var (
	// errBufferTooSmall indicates the destination slice is too small for the entry.
	errBufferTooSmall = errors.New("ringbuf: buffer too small")
	// errKeyTooLong indicates the key exceeds maxKeyLen.
	errKeyTooLong = errors.New("ringbuf: key too long")
	// errValueTooLong indicates the value exceeds maxValueLen.
	errValueTooLong = errors.New("ringbuf: value too long")
	// errEntryTruncated indicates the source slice is shorter than the entry header
	// claims, or the header itself is missing.
	errEntryTruncated = errors.New("ringbuf: entry truncated")
	// errCRCMismatch indicates the stored CRC does not match the computed CRC.
	errCRCMismatch = errors.New("ringbuf: CRC mismatch")
)

// makeMeta packs a write sequence and the tombstone flag into an entry's meta word.
//
// The mask is a WRAP, not a truncation. seq comes from shard.writeSeq, a full
// uint64 counter, so at 2^56 the stored sequence returns to 0 and NEWER entries
// start carrying LOWER sequences than older ones — which the rebuild reads as
// "the older copy is more recent" and silently resolves the key to stale data
// (#12A, reintroduced). It is not a saturating clamp and there is no wrap
// handling anywhere; the 56-bit budget is what makes that unnecessary, at ten
// million writes a second per shard it is 228 years.
func makeMeta(seq uint64, tombstone bool) uint64 {
	m := seq & entrySeqMask
	if tombstone {
		m |= entryFlagTombstone
	}
	return m
}

// metaSeq returns the write sequence carried by a meta word.
func metaSeq(meta uint64) uint64 { return meta & entrySeqMask }

// metaIsTombstone reports whether a meta word marks a delete record.
func metaIsTombstone(meta uint64) bool { return meta&entryFlagTombstone != 0 }

// entryMetaAt reads the meta word of the entry framed at src[0:]. Callers that
// already decoded the entry (so len(src) >= entryHeaderSize is established) use
// this instead of a wider decode signature, which keeps the meta word off the hot
// read path entirely. Returns 0 for a slice too short to hold a header.
func entryMetaAt(src []byte) uint64 {
	if len(src) < entryHeaderSize {
		return 0
	}
	return binary.LittleEndian.Uint64(src[entryMetaOff:entryCRCOff])
}

// crcTable is shared; crc32.IEEETable is a stable global.
var crcTable = crc32.IEEETable

// encodeEntry writes an entry into dst starting at index 0.
// Returns the number of bytes written. Computes the CRC so a future
// rebuildIndexFromPages can validate the entry; for heap-mode shards
// (no rebuild path) use [encodeEntryNoCRC] to skip the per-Put CRC
// cost.
func encodeEntry(dst []byte, key, value []byte, expiryMs, meta uint64) (int, error) {
	n, err := encodeEntryHeader(dst, key, value, expiryMs, meta)
	if err != nil {
		return 0, err
	}
	// CRC covers everything except the CRC slot itself. encodeEntryHeader
	// returned without error, which guarantees len(dst) >= total == n and
	// total >= entryHeaderSize (26), so all three fixed offsets below
	// (entryCRCOff, entryHeaderSize, n) are provably within dst.
	crc := crc32.Checksum(dst[0:entryCRCOff], crcTable)                  //nolint:gosec // len(dst) >= entryHeaderSize after encodeEntryHeader; bounds are invariant
	crc = crc32.Update(crc, crcTable, dst[entryHeaderSize:n])            //nolint:gosec // entryHeaderSize <= n <= len(dst) after encodeEntryHeader; bounds are invariant
	binary.LittleEndian.PutUint32(dst[entryCRCOff:entryHeaderSize], crc) //nolint:gosec // len(dst) >= entryHeaderSize after encodeEntryHeader; bounds are invariant
	return n, nil
}

// encodeEntryNoCRC is encodeEntry minus the CRC computation. Safe for
// heap-mode shards: the index is fully in memory, page bytes can't be
// corrupted by anything external (no mmap, no disk), and the only path
// that consumed the CRC field — rebuildIndexFromPages — never runs for
// heap-backed pages. The CRC slot is left untouched (whatever bytes
// the previous occupant left there); decodeEntryFast doesn't read it.
func encodeEntryNoCRC(dst []byte, key, value []byte, expiryMs, meta uint64) (int, error) {
	return encodeEntryHeader(dst, key, value, expiryMs, meta)
}

// encodeEntryHeader writes the keyLen / valLen / expiry / meta / key / value
// fields, returning the total bytes written. The CRC slot at
// dst[entryCRCOff:entryHeaderSize] is left for the caller to fill (or skip).
func encodeEntryHeader(dst []byte, key, value []byte, expiryMs, meta uint64) (int, error) {
	if len(key) > maxKeyLen {
		return 0, errKeyTooLong
	}
	if int64(len(value)) > maxValueLen {
		return 0, errValueTooLong
	}
	total := entryHeaderSize + len(key) + len(value)
	if len(dst) < total {
		return 0, errBufferTooSmall
	}
	binary.LittleEndian.PutUint16(dst[0:2], uint16(len(key)))   //nolint:gosec // len(key) <= maxKeyLen
	binary.LittleEndian.PutUint32(dst[2:6], uint32(len(value))) //nolint:gosec // len(value) <= maxValueLen
	binary.LittleEndian.PutUint64(dst[6:14], expiryMs)
	binary.LittleEndian.PutUint64(dst[entryMetaOff:entryCRCOff], meta)
	copy(dst[entryHeaderSize:entryHeaderSize+len(key)], key)
	copy(dst[entryHeaderSize+len(key):total], value)
	return total, nil
}

// decodeEntry reads an entry from src and returns its key, value, expiry and
// meta (key/value reference into src — zero-copy). Verifies the stored CRC; use
// decodeEntryFast on hot paths where the slabRef has already vouched for the
// entry (see [decodeEntryFast]).
func decodeEntry(src []byte) (key, value []byte, expiryMs, meta uint64, err error) {
	if len(src) < entryHeaderSize {
		return nil, nil, 0, 0, errEntryTruncated
	}
	keyLen := int(binary.LittleEndian.Uint16(src[0:2]))
	valLen := int(binary.LittleEndian.Uint32(src[2:6]))
	// On a 32-bit platform int is 32 bits, so a corrupted/torn valLen near the
	// uint32 max widens NEGATIVE instead of huge. Reject that before it reaches
	// the arithmetic below: a negative valLen would shrink total, potentially
	// passing the length check with a bogus (or negative, panicking) slice
	// bound. On 64-bit this is always false — valLen tops out at maxValueLen,
	// which fits comfortably positive — so the check costs nothing there.
	if valLen < 0 {
		return nil, nil, 0, 0, errEntryTruncated
	}
	expiryMs = binary.LittleEndian.Uint64(src[6:14])
	meta = binary.LittleEndian.Uint64(src[entryMetaOff:entryCRCOff])
	storedCRC := binary.LittleEndian.Uint32(src[entryCRCOff:entryHeaderSize])

	total := entryHeaderSize + keyLen + valLen
	if len(src) < total {
		return nil, nil, 0, 0, errEntryTruncated
	}

	crc := crc32.Checksum(src[0:entryCRCOff], crcTable)
	crc = crc32.Update(crc, crcTable, src[entryHeaderSize:total])
	if crc != storedCRC {
		return nil, nil, 0, 0, errCRCMismatch
	}

	key = src[entryHeaderSize : entryHeaderSize+keyLen]
	value = src[entryHeaderSize+keyLen : total]
	return key, value, expiryMs, meta, nil
}

// decodeEntryFast reads an entry from src without verifying the stored CRC.
// Use on the hot Get/Del/sweep paths: those reach an entry via the shard's
// in-memory index, which itself was populated only after a successful
// CRC-verified decode (at startup in rebuildIndexFromPages, or at write
// time after encodeEntry laid down a fresh CRC). The CRC slot is still
// written on encode so a future cold rebuild revalidates the page.
//
// Its signature deliberately does NOT expose the meta word. Nothing on a read
// path may consult the write sequence — a read resolves an entry through the
// index, which already encodes recency — so the hot decode stays exactly the
// three header loads plus two slices it has always been. Recovery-time callers
// that DO need the meta take it from [entryMetaAt] separately.
func decodeEntryFast(src []byte) (key, value []byte, expiryMs uint64, err error) {
	if len(src) < entryHeaderSize {
		return nil, nil, 0, errEntryTruncated
	}
	keyLen := int(binary.LittleEndian.Uint16(src[0:2]))
	valLen := int(binary.LittleEndian.Uint32(src[2:6]))
	// See the identical guard in decodeEntry: on a 32-bit platform a valLen near
	// the uint32 max widens NEGATIVE through int(), which would undershoot total
	// below and pass the length check with a bogus slice bound. No-op on 64-bit.
	if valLen < 0 {
		return nil, nil, 0, errEntryTruncated
	}
	expiryMs = binary.LittleEndian.Uint64(src[6:14])

	total := entryHeaderSize + keyLen + valLen
	if len(src) < total {
		return nil, nil, 0, errEntryTruncated
	}

	key = src[entryHeaderSize : entryHeaderSize+keyLen]
	value = src[entryHeaderSize+keyLen : total]
	return key, value, expiryMs, nil
}

// entrySize returns the byte size an entry will occupy on disk.
func entrySize(keyLen, valueLen int) int {
	return entryHeaderSize + keyLen + valueLen
}
