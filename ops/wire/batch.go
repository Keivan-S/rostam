// SPDX-License-Identifier: Apache-2.0

package wire

import (
	"encoding/binary"
	"time"
)

// MaxPutBatchSize is the canonical per-batch entry cap: it bounds how long a
// single put_batch holds the shard write-lock during its apply and caps the size
// of the one Raft log/fsync payload the whole batch forms. Callers chunk larger
// batches to this size (cluster.Node.PutBatch, the Store/client PutBatch paths).
const MaxPutBatchSize = 4096

// PutEntry is one key/value/ttl mutation in a put_batch.
type PutEntry struct {
	Key []byte
	Val []byte
	TTL time.Duration
}

// minPutEntrySize is the wire size of the smallest possible entry (empty key +
// empty val): {keyLen u16=2}{valLen u32=4}{ttlMs u64=8}. Used to reject a bogus
// count before it drives a huge pre-allocation.
const minPutEntrySize = 2 + 4 + 8

// EncodePutBatchArgs encodes "{count u32}" followed by `count` entries, each in
// the exact single-put layout {keyLen u16}{key}{valLen u32}{val}{ttlMs u64}.
//
// A put_batch is applied as ONE Raft log entry — one fsync, one replicate-to-
// majority round-trip, one FSM apply — for all N puts, which is the throughput
// win for bulk inserts. Because a batch is a single log entry routed to ONE
// shard, every key in it MUST hash to the same shard; the cluster fan-out groups
// entries by shard before encoding (cluster.Node.PutBatch). A raw call mixing
// keys from different shards would store them all on the first key's shard, where
// the others become unreadable (reads route by key hash to a different shard).
func EncodePutBatchArgs(entries []PutEntry) []byte {
	total := 4
	for _, e := range entries {
		total += 2 + len(e.Key) + 4 + len(e.Val) + 8
	}
	buf := make([]byte, 4, total)
	binary.BigEndian.PutUint32(buf[0:4], uint32(len(entries))) //nolint:gosec // len(entries) fits u32 for any realistic batch (callers cap via cluster.MaxPutBatchSize)
	for _, e := range entries {
		buf = append(buf, EncodePutArgs(e.Key, e.Val, e.TTL)...)
	}
	return buf
}

// DecodePutBatchArgs reads args produced by EncodePutBatchArgs.
func DecodePutBatchArgs(args []byte) ([]PutEntry, error) {
	if len(args) < 4 {
		return nil, ErrShortArgs
	}
	n := int(binary.BigEndian.Uint32(args[0:4]))
	// Bound n by the smallest possible entry so an attacker-controlled count
	// cannot drive a huge pre-allocation before per-entry validation.
	//
	// Via CountFitsIn rather than the bare `n > remaining/min` comparison this
	// used to make, because that comparison MISSES a negative n. On a 32-bit
	// build `int` is 32 bits, so a declared count above MaxInt32 widens NEGATIVE
	// through the conversion above; a negative n passes `n > remaining/min`
	// happily and reaches make([]PutEntry, 0, n), which panics with "makeslice:
	// cap out of range" — from network-supplied args, on a decode that runs
	// before any authorization. CountFitsIn rejects n < 0 explicitly and is the
	// guard the rest of the codebase already uses for exactly this.
	if !CountFitsIn(n, len(args)-4, minPutEntrySize) {
		return nil, ErrShortArgs
	}
	off := 4
	entries := make([]PutEntry, 0, n)
	for range n {
		key, val, ttl, consumed, err := decodeOnePut(args[off:])
		if err != nil {
			return nil, err
		}
		entries = append(entries, PutEntry{Key: key, Val: val, TTL: ttl})
		off += consumed
	}
	return entries, nil
}

// decodeOnePut decodes a single {keyLen u16}{key}{valLen u32}{val}{ttlMs u64}
// entry from the front of buf and reports how many bytes it consumed, so batched
// entries can be walked in sequence.
func decodeOnePut(buf []byte) (key, val []byte, ttl time.Duration, consumed int, err error) {
	if len(buf) < 2 {
		return nil, nil, 0, 0, ErrShortArgs
	}
	klen := int(binary.BigEndian.Uint16(buf[0:2]))
	off := 2
	if len(buf) < off+klen+4 {
		return nil, nil, 0, 0, ErrShortArgs
	}
	key = buf[off : off+klen]
	off += klen
	vlen := int(binary.BigEndian.Uint32(buf[off : off+4]))
	off += 4
	// vlen < 0 is unreachable on 64-bit (a uint32 always fits) and REACHABLE on
	// 32-bit, where a declared length above MaxInt32 widens negative. Without
	// this, `len(buf) < off+vlen+8` is satisfied by the negative sum and the
	// slice expression below runs with a high low-bound and a lower high-bound,
	// panicking on out-of-range bounds instead of returning ErrShortArgs.
	if vlen < 0 {
		return nil, nil, 0, 0, ErrShortArgs
	}
	if len(buf) < off+vlen+8 {
		return nil, nil, 0, 0, ErrShortArgs
	}
	val = buf[off : off+vlen]
	off += vlen
	ttl = time.Duration(binary.BigEndian.Uint64(buf[off:off+8])) * time.Millisecond
	off += 8
	return key, val, ttl, off, nil
}

// putBatchKeyExtractor returns the FIRST entry's key as the batch's routing key.
// The caller guarantees every key in the batch hashes to the same shard.
func putBatchKeyExtractor(args []byte) ([]byte, bool) {
	if len(args) < 4 {
		return nil, false
	}
	return StdKeyExtractor(args[4:]) // first entry starts at offset 4
}

// EncodePutBatchResult / DecodePutBatchResult carry the applied count back to the
// caller (a u32), so a client can confirm every entry was applied.
func EncodePutBatchResult(n int) []byte {
	b := make([]byte, 4)
	binary.BigEndian.PutUint32(b, uint32(n)) //nolint:gosec // n is a decoded/applied entry count, fits u32
	return b
}

func DecodePutBatchResult(b []byte) (int, error) {
	if len(b) < 4 {
		return 0, ErrShortArgs
	}
	return int(binary.BigEndian.Uint32(b[0:4])), nil
}
