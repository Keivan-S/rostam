// SPDX-License-Identifier: Apache-2.0
//go:build linux

package vector

import (
	"errors"
	"fmt"
	"os"

	"golang.org/x/sys/unix"
)

// openVecMmap opens path (creating if missing), sizes it to size bytes, and
// maps it MAP_SHARED read/write. Used as the float32 vector backing store when
// QuantStorage is QuantMmap.
func openVecMmap(path string, size int64) (*os.File, []byte, error) {
	f, err := os.OpenFile(path, os.O_RDWR|os.O_CREATE, 0o600) //nolint:gosec // G304: path is caller-supplied and intentional
	if err != nil {
		return nil, nil, fmt.Errorf("vector: open %s: %w", path, err)
	}
	if err := f.Truncate(size); err != nil {
		_ = f.Close()
		return nil, nil, fmt.Errorf("vector: truncate %s to %d: %w", path, size, err)
	}
	fd := int(f.Fd()) //nolint:gosec // uintptr->int: fd values are small and positive on Linux
	region, err := unix.Mmap(fd, 0, int(size), unix.PROT_READ|unix.PROT_WRITE, unix.MAP_SHARED)
	if err != nil {
		_ = f.Close()
		return nil, nil, fmt.Errorf("vector: mmap %s: %w", path, err)
	}
	return f, region, nil
}

// growVecMmap unmaps old, grows the file to newSize bytes, and remaps it. The
// caller must rebuild any slice headers that aliased the old region (the mapping
// address changes). The unmap comes first, so once it has run the old view is
// gone and a later failure would otherwise leave nothing mapped — hence the
// restore step below and the TRI-STATE return:
//
//   - (region != nil, err == nil): grew successfully; region covers newSize bytes.
//   - (region != nil, err != nil): the grow failed, but a mapping of the OLD data
//     was restored (grow only ever appends, so bytes [0,oldSize) are intact).
//     len(region) == oldSize; the caller MUST rebind its header to region at the
//     old logical length so the valid mapping is neither leaked nor left dangling.
//     Whether the failure is then RECOVERABLE is the caller's call, and the two
//     callers differ: arena.growVecs grows pre-commit, so it treats this as
//     non-terminal (only the one op fails); hnsw.growLevel0 grows post-commit with
//     no clean rollback, so it rebinds AND poisons. See each call site.
//   - (region == nil, err != nil): the grow AND the restore both failed; no valid
//     mapping remains, so the backing is unusable and the caller must go terminal.
func growVecMmap(f *os.File, old []byte, newSize int64) ([]byte, error) {
	oldSize := int64(len(old))
	if oldSize > 0 {
		if err := unix.Munmap(old); err != nil {
			return nil, fmt.Errorf("vector: munmap (grow): %w", err)
		}
	}
	fd := int(f.Fd()) //nolint:gosec // uintptr->int: fd values are small and positive on Linux

	// gErr is the original grow failure (nil until something fails). truncated
	// records whether the file was already extended to newSize, so the restore can
	// shrink it back before remapping the old extent.
	var gErr error
	truncated := false
	if fp := growVecMmapFailpoint; fp != nil {
		gErr = fp() // test-only: simulate a Truncate/mmap failure past the unmap.
	}
	if gErr == nil {
		if err := f.Truncate(newSize); err != nil {
			gErr = fmt.Errorf("vector: truncate (grow) to %d: %w", newSize, err)
		} else {
			truncated = true
			region, err := unix.Mmap(fd, 0, int(newSize), unix.PROT_READ|unix.PROT_WRITE, unix.MAP_SHARED)
			if err != nil {
				gErr = fmt.Errorf("vector: mmap (grow) to %d: %w", newSize, err)
			} else {
				return region, nil
			}
		}
	}

	// The grow failed after the old view was unmapped. If there was nothing mapped
	// to begin with (first grow from empty), there is nothing to restore.
	if oldSize == 0 {
		return nil, gErr
	}
	// Reclaim the just-added tail if the truncate had succeeded (best-effort — a
	// failure to shrink does not stop us mapping the old extent), then map the old
	// data back so the index stays valid.
	if truncated {
		_ = f.Truncate(oldSize)
	}
	restored, rerr := unix.Mmap(fd, 0, int(oldSize), unix.PROT_READ|unix.PROT_WRITE, unix.MAP_SHARED)
	if rerr != nil {
		// Both the grow and the restore failed. Join them so errors.Is can reach EITHER
		// cause (the restore failure is primary, the original grow failure secondary).
		return nil, fmt.Errorf("vector: mmap (grow-restore) to %d: %w", oldSize, errors.Join(rerr, gErr))
	}
	return restored, gErr
}

// syncVecMmap flushes dirty pages of region to its backing file, so a
// subsequently-written sidecar can safely reference the mmap'd contents.
//
// The file is accepted but unused here: MS_SYNC already returns only once the
// pages are on the disk. Windows needs the handle for the second half of that
// same guarantee (see mmap_windows.go), and one signature across the platforms
// keeps the difference inside these files instead of at every call site.
func syncVecMmap(_ *os.File, region []byte) error {
	if len(region) == 0 {
		return nil
	}
	if err := unix.Msync(region, unix.MS_SYNC); err != nil {
		return fmt.Errorf("vector: msync: %w", err)
	}
	return nil
}

// closeVecMmap syncs, unmaps, and closes the backing file.
func closeVecMmap(f *os.File, region []byte) error {
	if len(region) > 0 {
		_ = unix.Msync(region, unix.MS_SYNC)
		if err := unix.Munmap(region); err != nil {
			return fmt.Errorf("vector: munmap: %w", err)
		}
	}
	if f != nil {
		if err := f.Close(); err != nil {
			return fmt.Errorf("vector: close: %w", err)
		}
	}
	return nil
}

// unmapVecMmap releases a mapping made by openVecMmap/growVecMmap WITHOUT
// closing the file — used when a slab migrates from the legacy whole-file
// mapping onto a reservation, where the file stays but its old mapping must go.
func unmapVecMmap(region []byte) error {
	if len(region) == 0 {
		return nil
	}
	_ = unix.Msync(region, unix.MS_SYNC)
	if err := unix.Munmap(region); err != nil {
		return fmt.Errorf("vector: munmap (migrate): %w", err)
	}
	return nil
}
