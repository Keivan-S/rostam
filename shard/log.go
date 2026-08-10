// SPDX-License-Identifier: Apache-2.0

package shard

import (
	"encoding/binary"
	"errors"
	"fmt"
	"sync"
)

// Log entry wire formats. TWO layouts coexist on the wire, discriminated by the
// FIRST byte, and both DecodeLogEntry variants read either transparently:
//
//   LEGACY (first byte != 0x00):
//     [opNameLen u8 (≥1)][opName ASCII][argsLen u32 big-endian][args bytes]
//
//   STAMPED / extended (first byte == 0x00):
//     [0x00 marker][ver u8][stampMs u64 big-endian][opNameLen u8][opName][argsLen u32][args]
//
// The discriminator is unambiguous: a legacy entry's first byte is opNameLen,
// which is ALWAYS ≥ 1 for a registered op (op names are non-empty; the Registry
// enforces it), so 0x00 can never begin a legacy entry and cleanly marks the new
// format. A legacy entry decodes with stampMs = 0, which the apply path reads as
// "no leader stamp → fall back to the wall clock" — preserving byte-identical
// behavior for old on-disk logs and for the stamping-disabled rollout phase.
//
// The stamp is the leader/primary's wall clock (UnixMilli) at propose time, baked
// INTO the replicated bytes so every replica evaluates a committed write's expiry
// against the SAME clock — the core of the #4 Phase B / B1 determinism fix. See
// EncodeLogEntryStamped and DecodeLogEntry.

const maxOpNameLen = 255

// logStampMarker is the first byte of a stamped/extended entry. It can never be a
// legacy opNameLen (which is ≥1), so it unambiguously selects the extended layout.
const logStampMarker = 0x00

// logStampVersion is the current extended-format version. Bumped only on a wire
// change; DecodeLogEntry rejects an unknown version rather than mis-parsing it.
const logStampVersion = 1

// stampedHeaderLen is the fixed prefix before opName in a stamped entry:
// marker(1) + ver(1) + stampMs(8) + opNameLen(1).
const stampedHeaderLen = 1 + 1 + 8 + 1

// ErrLogEntryTruncated indicates the entry buffer is shorter than the header claims.
var ErrLogEntryTruncated = errors.New("shard: log entry truncated")

// ErrLogEntryVersion indicates a stamped (0x00-marked) entry carries a version
// this binary does not understand. It is returned rather than silently
// mis-parsed, so a mixed-version rollout that enabled stamping before every node
// could decode it fails LOUD (the op lookup never runs) and Phase A's
// fail-closed halt engages — never a silent divergence. See the two-phase rollout
// note on EncodeLogEntryStamped.
var ErrLogEntryVersion = errors.New("shard: unsupported log entry version")

// EncodeLogEntry produces the byte slice stored in a Raft log entry's Data field.
// Panics if opName exceeds 255 bytes — opName names are static and should be
// validated by callers (the Registry already enforces this).
func EncodeLogEntry(opName string, args []byte) []byte {
	if len(opName) > maxOpNameLen {
		panic(fmt.Sprintf("shard: opName length %d exceeds %d", len(opName), maxOpNameLen))
	}
	buf := make([]byte, 1+len(opName)+4+len(args))
	buf[0] = byte(len(opName)) //nolint:gosec // opName length validated above and bounded to 255
	copy(buf[1:1+len(opName)], opName)
	binary.BigEndian.PutUint32(buf[1+len(opName):1+len(opName)+4], uint32(len(args))) //nolint:gosec // args length bounded by upstream Raft log entry size limit
	copy(buf[1+len(opName)+4:], args)
	return buf
}

// logEntryPool recycles PB-mode log-entry buffers: EncodeLogEntryPooled draws
// from it and the pbisr engine's catch-up-ring eviction (WithDataRelease ->
// ReleaseLogEntry) returns them, so the per-write entry allocation — the
// largest remaining write-path allocation after the 2026-07-22 cuts — is
// amortized to the steady-state ring population. Raft mode also encodes via
// the pooled Get (harmless: nothing ever Puts on that path, so it degrades to
// plain allocation).
var logEntryPool = sync.Pool{
	New: func() any {
		b := make([]byte, 0, 512)
		return &b
	},
}

// logEntryMaxPooledCap bounds retained buffer capacity, mirroring pbisr's
// payload pool: an occasional huge entry must not pin memory in the pool.
const logEntryMaxPooledCap = 64 << 10

// EncodeLogEntryPooled is EncodeLogEntry into a pool-backed buffer. Ownership:
// in PB mode the buffer is retained by the engine's catch-up ring and comes
// back via ReleaseLogEntry on eviction; callers must NOT release it
// themselves after handing it to a replicator.
func EncodeLogEntryPooled(opName string, args []byte) []byte {
	if len(opName) > maxOpNameLen {
		panic(fmt.Sprintf("shard: opName length %d exceeds %d", len(opName), maxOpNameLen))
	}
	bp := logEntryPool.Get().(*[]byte)
	buf := (*bp)[:0]
	buf = append(buf, byte(len(opName)))
	buf = append(buf, opName...)
	var n [4]byte
	binary.BigEndian.PutUint32(n[:], uint32(len(args))) //nolint:gosec // args bounded upstream
	buf = append(buf, n[:]...)
	buf = append(buf, args...)
	return buf
}

// ReleaseLogEntry returns a pooled log-entry buffer (see EncodeLogEntryPooled).
func ReleaseLogEntry(b []byte) {
	if cap(b) > logEntryMaxPooledCap {
		return
	}
	b = b[:0]
	logEntryPool.Put(&b)
}

// EncodeLogEntryStamped produces the STAMPED/extended entry (see the format note
// at the top of this file): the leader/primary bakes stampMs — its wall clock at
// propose time — into the replicated bytes so every replica evaluates the write's
// expiry against the same clock (#4 Phase B / B1). It is emitted ONLY when apply
// stamping is enabled (shard.Config.EnableApplyStamp); the disabled path keeps
// calling the legacy EncodeLogEntry* so old and new binaries produce identical
// bytes during the first rollout phase.
//
// ROLLOUT ORDERING (two-phase, load-bearing): deploy a binary that can DECODE the
// stamped format to EVERY node with stamping DISABLED first; only THEN enable
// stamping on leaders. A node that receives a 0x00 entry it cannot parse fails
// its op lookup and Phase A halts it loud — safe, but an availability hit — so the
// decode-everywhere-first ordering avoids it entirely.
func EncodeLogEntryStamped(opName string, args []byte, stampMs uint64) []byte {
	if len(opName) > maxOpNameLen {
		panic(fmt.Sprintf("shard: opName length %d exceeds %d", len(opName), maxOpNameLen))
	}
	buf := make([]byte, stampedHeaderLen+len(opName)+4+len(args))
	appendStampedHeader(buf, opName, args, stampMs)
	return buf
}

// EncodeLogEntryStampedPooled is EncodeLogEntryStamped into a pool-backed buffer,
// matching EncodeLogEntryPooled's ownership contract (in PB mode the engine's
// catch-up ring retains the buffer and returns it via ReleaseLogEntry; callers
// must NOT release it themselves after handing it to a replicator).
func EncodeLogEntryStampedPooled(opName string, args []byte, stampMs uint64) []byte {
	if len(opName) > maxOpNameLen {
		panic(fmt.Sprintf("shard: opName length %d exceeds %d", len(opName), maxOpNameLen))
	}
	bp := logEntryPool.Get().(*[]byte)
	need := stampedHeaderLen + len(opName) + 4 + len(args)
	buf := (*bp)[:0]
	if cap(buf) < need {
		buf = make([]byte, need)
	} else {
		buf = buf[:need]
	}
	appendStampedHeader(buf, opName, args, stampMs)
	return buf
}

// appendStampedHeader writes the extended layout into buf, which MUST already be
// sized to stampedHeaderLen+len(opName)+4+len(args). Centralizes the byte layout
// so the pooled and non-pooled encoders cannot drift.
func appendStampedHeader(buf []byte, opName string, args []byte, stampMs uint64) {
	buf[0] = logStampMarker
	buf[1] = logStampVersion
	binary.BigEndian.PutUint64(buf[2:10], stampMs)
	buf[10] = byte(len(opName)) //nolint:gosec // opName length validated ≤ 255 by the caller
	copy(buf[stampedHeaderLen:stampedHeaderLen+len(opName)], opName)
	off := stampedHeaderLen + len(opName)
	binary.BigEndian.PutUint32(buf[off:off+4], uint32(len(args))) //nolint:gosec // args length bounded by the Raft/PB entry size limit
	copy(buf[off+4:], args)
}

// DecodeLogEntry reads either entry layout (see the format note at the top of
// this file). It returns the decoded op name, the args slice (aliasing buf),
// stampMs (the leader-stamped clock; 0 for a legacy entry), and stamped: true iff
// this was the extended (0x00-marked) format.
//
// stamped is the AUTHORITATIVE "use the stamped apply clock" signal — NOT
// stampMs != 0. A stamped entry whose stamp legitimately decodes to 0 must still
// take the stamped apply path (where every replica deterministically uses 0),
// never the per-node wall-clock fallback; keying off stampMs != 0 would silently
// route that entry back to wall clocks and diverge replicas. See TxContext.
//
// An extended entry with an unknown version returns ErrLogEntryVersion so a
// premature-stamping / post-decoder-version-bump rollout fails closed
// (classFatal) rather than mis-parsing or silently skipping.
func DecodeLogEntry(buf []byte) (opName string, args []byte, stampMs uint64, stamped bool, err error) {
	if len(buf) < 1 {
		return "", nil, 0, false, ErrLogEntryTruncated
	}
	if buf[0] == logStampMarker {
		return decodeStampedLogEntry(buf)
	}
	// Legacy layout: first byte is opNameLen (≥1), stampMs is 0, stamped is false.
	nameLen := int(buf[0])
	if len(buf) < 1+nameLen+4 {
		return "", nil, 0, false, ErrLogEntryTruncated
	}
	opName = string(buf[1 : 1+nameLen])
	argsLen := int(binary.BigEndian.Uint32(buf[1+nameLen : 1+nameLen+4]))
	if len(buf) < 1+nameLen+4+argsLen {
		return "", nil, 0, false, ErrLogEntryTruncated
	}
	args = buf[1+nameLen+4 : 1+nameLen+4+argsLen]
	return opName, args, 0, false, nil
}

// decodeStampedLogEntry decodes the extended (0x00-marked) layout. buf[0] is the
// marker already; it validates the version, then reads stampMs, opName, and args.
// It always returns stamped=true on success (the format IS the stamped signal,
// independent of the stamp value).
func decodeStampedLogEntry(buf []byte) (opName string, args []byte, stampMs uint64, stamped bool, err error) {
	if len(buf) < stampedHeaderLen {
		return "", nil, 0, false, ErrLogEntryTruncated
	}
	if buf[1] != logStampVersion {
		return "", nil, 0, false, fmt.Errorf("%w: %d", ErrLogEntryVersion, buf[1])
	}
	stampMs = binary.BigEndian.Uint64(buf[2:10])
	nameLen := int(buf[10])
	if len(buf) < stampedHeaderLen+nameLen+4 {
		return "", nil, 0, false, ErrLogEntryTruncated
	}
	opName = string(buf[stampedHeaderLen : stampedHeaderLen+nameLen])
	off := stampedHeaderLen + nameLen
	argsLen := int(binary.BigEndian.Uint32(buf[off : off+4]))
	if len(buf) < off+4+argsLen {
		return "", nil, 0, false, ErrLogEntryTruncated
	}
	args = buf[off+4 : off+4+argsLen]
	return opName, args, stampMs, true, nil
}
