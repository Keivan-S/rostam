// SPDX-License-Identifier: Apache-2.0
//go:build windows

package vector

import (
	"fmt"
	"os"
	"unsafe"

	"golang.org/x/sys/windows"
)

// openVecMmap opens path (creating if missing), sizes it to size bytes, and
// maps it read/write and shared. Used as the float32 vector backing store when
// QuantStorage is QuantMmap.
func openVecMmap(path string, size int64) (*os.File, []byte, error) {
	if size <= 0 {
		return nil, nil, fmt.Errorf("vector: mmap %s: size must be > 0, got %d", path, size)
	}
	f, err := os.OpenFile(path, os.O_RDWR|os.O_CREATE, 0o600) //nolint:gosec // G304: path is caller-supplied and intentional
	if err != nil {
		return nil, nil, fmt.Errorf("vector: open %s: %w", path, err)
	}
	if err := f.Truncate(size); err != nil {
		_ = f.Close()
		return nil, nil, fmt.Errorf("vector: truncate %s to %d: %w", path, size, err)
	}
	region, err := mapVecView(f, size)
	if err != nil {
		_ = f.Close()
		return nil, nil, fmt.Errorf("vector: mmap %s: %w", path, err)
	}
	return f, region, nil
}

// growVecMmap unmaps old, grows the file to newSize bytes, and remaps it.
// Returns the new region. The caller must rebuild any slice headers that
// aliased the old region (the mapping address changes).
//
// The unmap has to come first for a second reason here that it does not have on
// Linux: Windows refuses to change the length of a file that still has a view
// mapped (ERROR_USER_MAPPED_FILE), so the Truncate below would fail outright
// rather than merely racing readers.
func growVecMmap(f *os.File, old []byte, newSize int64) ([]byte, error) {
	if len(old) > 0 {
		if err := windows.UnmapViewOfFile(vecRegionAddr(old)); err != nil {
			return nil, fmt.Errorf("vector: UnmapViewOfFile (grow): %w", err)
		}
	}
	if err := f.Truncate(newSize); err != nil {
		return nil, fmt.Errorf("vector: truncate (grow) to %d: %w", newSize, err)
	}
	region, err := mapVecView(f, newSize)
	if err != nil {
		return nil, fmt.Errorf("vector: mmap (grow) to %d: %w", newSize, err)
	}
	return region, nil
}

// syncVecMmap flushes dirty pages of region to its backing file, so a
// subsequently-written sidecar can safely reference the mmap'd contents.
//
// FlushViewOfFile alone only pushes the pages into the filesystem; the sidecar
// is a durability record, so the flush has to reach the disk, which on this
// platform is the FlushFileBuffers that follows.
func syncVecMmap(f *os.File, region []byte) error {
	if len(region) == 0 {
		return nil
	}
	if err := windows.FlushViewOfFile(vecRegionAddr(region), uintptr(len(region))); err != nil {
		return fmt.Errorf("vector: FlushViewOfFile: %w", err)
	}
	if f == nil {
		return nil
	}
	if err := windows.FlushFileBuffers(windows.Handle(f.Fd())); err != nil {
		return fmt.Errorf("vector: FlushFileBuffers: %w", err)
	}
	return nil
}

// closeVecMmap syncs, unmaps, and closes the backing file.
func closeVecMmap(f *os.File, region []byte) error {
	if len(region) > 0 {
		_ = syncVecMmap(f, region)
		if err := windows.UnmapViewOfFile(vecRegionAddr(region)); err != nil {
			return fmt.Errorf("vector: UnmapViewOfFile: %w", err)
		}
	}
	if f != nil {
		if err := f.Close(); err != nil {
			return fmt.Errorf("vector: close: %w", err)
		}
	}
	return nil
}

// unmapVecMmap releases a mapped region whose backing file stays open — the
// migrate path, where a slab moves off the mapping and the file outlives it.
//
// The flush is best-effort and view-only: without the file handle it cannot
// reach FlushFileBuffers, which mirrors the best-effort MS_SYNC the Linux side
// does at the same point.
func unmapVecMmap(region []byte) error {
	if len(region) == 0 {
		return nil
	}
	_ = windows.FlushViewOfFile(vecRegionAddr(region), uintptr(len(region)))
	if err := windows.UnmapViewOfFile(vecRegionAddr(region)); err != nil {
		return fmt.Errorf("vector: UnmapViewOfFile (migrate): %w", err)
	}
	return nil
}

// mapVecView maps [0, size) of f read/write and shared.
//
// The section handle is closed before returning: the view holds its own
// reference and stays valid without it, which is what keeps the (file, region)
// pair this package passes around sufficient to release everything later.
func mapVecView(f *os.File, size int64) ([]byte, error) {
	h, err := windows.CreateFileMapping(
		windows.Handle(f.Fd()),
		nil,
		windows.PAGE_READWRITE,
		uint32(uint64(size)>>32),        //nolint:gosec // deliberate 64->2x32 split of the section size
		uint32(uint64(size)&0xFFFFFFFF), //nolint:gosec // deliberate 64->2x32 split of the section size
		nil,
	)
	if err != nil {
		return nil, fmt.Errorf("CreateFileMapping: %w", err)
	}
	defer func() { _ = windows.CloseHandle(h) }()
	addr, err := windows.MapViewOfFile(h, windows.FILE_MAP_READ|windows.FILE_MAP_WRITE, 0, 0, uintptr(size))
	if err != nil {
		return nil, fmt.Errorf("MapViewOfFile: %w", err)
	}
	// unsafeptr: addr is an OS mapping address, not a Go pointer — there is no
	// object for the collector to move out from under it, and the view outlives
	// this conversion by construction (it is released only by the Unmap in
	// closeVecMmap/unmapVecMmap). go vet's unsafeptr check has no way to express that, and
	// this is the one conversion every Windows file mapping has to make.
	//nolint:govet,gosec // unsafeptr: see above; the view is exactly size bytes long
	return unsafe.Slice((*byte)(unsafe.Pointer(addr)), size), nil
}

// vecRegionAddr returns the base address of a mapped region.
func vecRegionAddr(region []byte) uintptr {
	return uintptr(unsafe.Pointer(&region[0]))
}
