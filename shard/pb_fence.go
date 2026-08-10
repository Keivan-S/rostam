// SPDX-License-Identifier: Apache-2.0

package shard

import (
	"encoding/binary"
	"fmt"
	"hash/crc32"
	"os"
	"path/filepath"
)

// ============================================================================
// The PB RESTORE POISON FENCE.
//
// restoreSnapshot is delete-then-put across a wipe and a replay. It is NOT
// atomic, and no single durable store spans it. So a mid-install abort — a crash,
// an epoch change, an I/O failure — leaves a HALF-WIPED shard whose persisted PB
// frontier describes neither the old state nor the new one. That state must be
// UNUSABLE, not merely stale: a half-wiped node that still answers the catch-up
// handshake with a plausible watermark can be re-admitted to the ISR, or worse
// PROMOTED, and either silently destroys committed data.
//
// The fence is a durable marker raised BEFORE the wipe and lowered only AFTER
// both the install and the durable frontier write have succeeded. Every
// intermediate state is therefore a state in which the marker is on disk, and a
// node that boots with it on disk refuses to serve, reports CatchupInfo{OK:false}
// (so cluster's failover gate treats it as unverifiable and never promotes it),
// and can only be re-snapshotted.
//
// WHERE IT LIVES, AND WHY NOT THE CACHE HEADER. The durable frontier consumed the last of
// the 64-byte cache header (bytes 44..63, zero reserved bytes remain), and
// bumping cacheVersion would rotate every existing pages file aside —
// destroying exactly the data the frontier describes. But the header would have
// been the WRONG home even with room: a PB shard's FSM is the cache AND the
// vector CollectionStore, which is a separate store with its own files. A marker
// inside the cache header could not certify that the vector half of an install
// completed. A sidecar file in the shard's DataDir covers the whole FSM, which is
// what the fence actually has to protect, and it needs no format change at all.
//
// HEAP MODE / NO DATADIR. With no DataDir there is nothing durable to protect: a
// crash loses the entire FSM and the node restarts empty at frontier (0,0), which
// is honest. The fence is then a no-op and the engine's IN-MEMORY poisoned latch
// (set across every install regardless) carries the whole property — which is the
// right split, because an in-memory half-wipe is just as unusable as a durable
// one, it simply does not survive a restart to be discovered later.
// ============================================================================

// pbFenceFile is the sidecar marker's name inside the shard DataDir.
const pbFenceFile = "pb-restore.fence"

// pbFenceMagic / pbFenceVersion identify the marker. The layout is
// magic(8) version(4) seq(8) epoch(8) crc32(4) = 32 bytes, CRC over bytes 0..23.
// A file that fails ANY check is treated as a RAISED fence, not an absent one:
// the file's existence already proves an install began, and an unreadable body
// only means we cannot say which one.
const (
	pbFenceMagic   uint64 = 0x45434e4546425250 // "PRBFENCE" little-endian
	pbFenceVersion uint32 = 1
	pbFenceSize           = 8 + 4 + 8 + 8 + 4
)

// pbRestoreFence is the durable poison fence for one shard's DataDir. A zero dir
// makes every operation a no-op (see the heap-mode note above).
type pbRestoreFence struct{ dir string }

func newPBRestoreFence(dataDir string) *pbRestoreFence { return &pbRestoreFence{dir: dataDir} }

func (f *pbRestoreFence) path() string { return filepath.Join(f.dir, pbFenceFile) }

// raise durably records "a restore to (seq, epoch) is in progress". It writes a
// temp file, fsyncs it, renames it into place, and fsyncs the DIRECTORY — the
// last step is what makes the rename itself durable, without which a crash could
// lose the marker entirely and un-poison a half-wiped shard.
func (f *pbRestoreFence) raise(seq, epoch uint64) error {
	if f.dir == "" {
		return nil
	}
	var buf [pbFenceSize]byte
	binary.LittleEndian.PutUint64(buf[0:8], pbFenceMagic)
	binary.LittleEndian.PutUint32(buf[8:12], pbFenceVersion)
	binary.LittleEndian.PutUint64(buf[12:20], seq)
	binary.LittleEndian.PutUint64(buf[20:28], epoch)
	binary.LittleEndian.PutUint32(buf[28:32], crc32.ChecksumIEEE(buf[0:28]))

	tmp := f.path() + ".tmp"
	fh, err := os.OpenFile(tmp, os.O_CREATE|os.O_TRUNC|os.O_WRONLY, 0o600)
	if err != nil {
		return fmt.Errorf("shard: pb fence create: %w", err)
	}
	if _, err := fh.Write(buf[:]); err != nil {
		_ = fh.Close()
		return fmt.Errorf("shard: pb fence write: %w", err)
	}
	if err := fh.Sync(); err != nil {
		_ = fh.Close()
		return fmt.Errorf("shard: pb fence sync: %w", err)
	}
	if err := fh.Close(); err != nil {
		return fmt.Errorf("shard: pb fence close: %w", err)
	}
	if err := os.Rename(tmp, f.path()); err != nil {
		return fmt.Errorf("shard: pb fence rename: %w", err)
	}
	return f.syncDir()
}

// lower removes the marker durably. It is called ONLY after the install and the
// durable frontier write have both succeeded (or on the one abort path where the
// FSM is provably untouched).
func (f *pbRestoreFence) lower() error {
	if f.dir == "" {
		return nil
	}
	if err := os.Remove(f.path()); err != nil && !os.IsNotExist(err) {
		return fmt.Errorf("shard: pb fence remove: %w", err)
	}
	return f.syncDir()
}

// syncDir fsyncs the shard directory so a create or unlink of the marker is
// itself durable. A directory that cannot be opened for sync is not fatal on
// every filesystem, but here it would silently weaken the fence, so it is
// reported.
func (f *pbRestoreFence) syncDir() error {
	d, err := os.Open(f.dir)
	if err != nil {
		return fmt.Errorf("shard: pb fence dir open: %w", err)
	}
	defer func() { _ = d.Close() }()
	if err := d.Sync(); err != nil {
		return fmt.Errorf("shard: pb fence dir sync: %w", err)
	}
	return nil
}

// raised reports whether the fence is up, and the (seq, epoch) it was raised for
// when that is readable. A present-but-unreadable marker still reports raised
// with a zero identity: existence alone proves an install began.
func (f *pbRestoreFence) raised() (seq, epoch uint64, up bool) {
	if f.dir == "" {
		return 0, 0, false
	}
	b, err := os.ReadFile(f.path())
	if err != nil {
		return 0, 0, false // absent (or unreadable dir): no install was in progress
	}
	if len(b) != pbFenceSize ||
		binary.LittleEndian.Uint64(b[0:8]) != pbFenceMagic ||
		binary.LittleEndian.Uint32(b[8:12]) != pbFenceVersion ||
		binary.LittleEndian.Uint32(b[28:32]) != crc32.ChecksumIEEE(b[0:28]) {
		return 0, 0, true
	}
	return binary.LittleEndian.Uint64(b[12:20]), binary.LittleEndian.Uint64(b[20:28]), true
}
