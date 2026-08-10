// SPDX-License-Identifier: Apache-2.0
//go:build linux

package vector

import (
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

// growVecMmap unmaps old, grows the file to newSize bytes, and remaps it.
// Returns the new region. The caller must rebuild any slice headers that
// aliased the old region (the mapping address changes).
func growVecMmap(f *os.File, old []byte, newSize int64) ([]byte, error) {
	if len(old) > 0 {
		if err := unix.Munmap(old); err != nil {
			return nil, fmt.Errorf("vector: munmap (grow): %w", err)
		}
	}
	if err := f.Truncate(newSize); err != nil {
		return nil, fmt.Errorf("vector: truncate (grow) to %d: %w", newSize, err)
	}
	fd := int(f.Fd()) //nolint:gosec // uintptr->int: fd values are small and positive on Linux
	region, err := unix.Mmap(fd, 0, int(newSize), unix.PROT_READ|unix.PROT_WRITE, unix.MAP_SHARED)
	if err != nil {
		return nil, fmt.Errorf("vector: mmap (grow) to %d: %w", newSize, err)
	}
	return region, nil
}

// syncVecMmap flushes dirty pages of region to its backing file, so a
// subsequently-written sidecar can safely reference the mmap'd contents.
func syncVecMmap(region []byte) error {
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
