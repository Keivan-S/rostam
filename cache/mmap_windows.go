// SPDX-License-Identifier: Apache-2.0
//go:build windows

package cache

import (
	"errors"
	"fmt"
	"log/slog"
	"os"
	"unsafe"

	"golang.org/x/sys/windows"
)

// mmapSupported reports that a shard on this platform can be backed by a file
// mapping, so Config.Validate accepts a non-empty DataDir. See mmap_linux.go
// and mmap_other.go for the other two answers.
const mmapSupported = true

// mmapFile opens path (creating if missing), sizes it to size bytes, and
// returns the mapped region. If mlockRequested is true, attempts to lock the
// region into the working set; failure logs and continues without the lock.
func mmapFile(path string, size int64, mlockRequested bool) (*os.File, []byte, error) {
	return mmapFileAlloc(path, size, 0, mlockRequested)
}

// mmapFileAlloc is mmapFile plus a RESERVATION of the first allocBytes of the
// file — which on Windows costs nothing to honour, because it has already
// happened by the time this function looks at allocBytes.
//
// WHY THE PARAMETER IS IGNORED HERE. On Linux the file is sized with a sparse
// truncate, so blocks are allocated lazily inside a page fault and a full disk
// turns into SIGBUS, which Go cannot recover from; fallocate exists to pull
// that failure forward into an ordinary ENOSPC. NTFS does not create the file
// sparse: os.File.Truncate lands on SetEndOfFile, which commits the clusters
// there and then, so a full disk fails the Truncate above with an ordinary
// error and the mapped writes that follow cannot run out of space. The caller's
// invariant ("the bytes I am about to store are already reserved") therefore
// holds for the WHOLE file, not just the first allocBytes, without a separate
// call. The cost of the stronger guarantee is that the file occupies its full
// length on disk immediately instead of growing as it fills.
func mmapFileAlloc(path string, size, _ int64, mlockRequested bool) (*os.File, []byte, error) {
	if size <= 0 {
		return nil, nil, fmt.Errorf("cache: mmap %s: size must be > 0, got %d", path, size)
	}
	f, err := os.OpenFile(path, os.O_RDWR|os.O_CREATE, 0o600) //nolint:gosec // G304: path is caller-supplied and intentional
	if err != nil {
		return nil, nil, fmt.Errorf("cache: open %s: %w", path, err)
	}
	if err := f.Truncate(size); err != nil {
		_ = f.Close()
		return nil, nil, fmt.Errorf("cache: truncate %s to %d: %w", path, size, err)
	}
	region, err := mapWholeFile(f, size)
	if err != nil {
		_ = f.Close()
		return nil, nil, fmt.Errorf("cache: mmap %s: %w", path, err)
	}
	if mlockRequested {
		if lkErr := windows.VirtualLock(regionAddr(region), uintptr(size)); lkErr != nil {
			slog.Warn("VirtualLock failed (continuing without lock; consider raising the process working-set limit)", "component", "cache", "path", path, "err", lkErr)
		}
	}
	return f, region, nil
}

// mapWholeFile maps [0, size) of f read/write and shared, and returns it as a
// byte slice aliasing the view.
//
// The section handle is closed before returning ON PURPOSE: a mapped view keeps
// its own reference to the section object, so the view stays valid after the
// handle goes away. That is what lets this platform carry the same (file,
// region) pair as Linux — release needs no third value to be threaded through
// every caller and stored on every shard.
func mapWholeFile(f *os.File, size int64) ([]byte, error) {
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
	// munmapAndClose). go vet's unsafeptr check has no way to express that, and
	// this is the one conversion every Windows file mapping has to make.
	//nolint:govet,gosec // unsafeptr: see above; the view is exactly size bytes long
	return unsafe.Slice((*byte)(unsafe.Pointer(addr)), size), nil
}

// regionAddr returns the base address of a mapped region. Callers pass either
// the whole region or a prefix of it (the header slice); both share a base, so
// this is the address the Win32 view calls want in either case.
func regionAddr(region []byte) uintptr {
	return uintptr(unsafe.Pointer(&region[0]))
}

// msync flushes the region to disk synchronously.
//
// It takes the file because ONE Win32 call is not enough. FlushViewOfFile only
// hands the dirty pages to the filesystem; it returns before they reach the
// disk, so on its own it is closer to Linux's MS_ASYNC than to the MS_SYNC this
// function's callers assume. FlushFileBuffers supplies the second half. Both
// are needed for -durable to mean on this platform what it means on Linux: the
// msyncLoop and the applied-index watermark log a DURABILITY WARNING when this
// fails precisely because a silent partial flush is the failure they exist to
// catch.
func msync(f *os.File, region []byte) error {
	if len(region) == 0 {
		return nil
	}
	if err := windows.FlushViewOfFile(regionAddr(region), uintptr(len(region))); err != nil {
		return fmt.Errorf("cache: FlushViewOfFile: %w", err)
	}
	if f == nil {
		return nil
	}
	if err := windows.FlushFileBuffers(windows.Handle(f.Fd())); err != nil {
		return fmt.Errorf("cache: FlushFileBuffers: %w", err)
	}
	return nil
}

// munmapAndClose releases the mapped view and closes the file.
func munmapAndClose(f *os.File, region []byte) error {
	var errs []error
	if len(region) > 0 {
		if err := windows.UnmapViewOfFile(regionAddr(region)); err != nil {
			errs = append(errs, fmt.Errorf("cache: UnmapViewOfFile: %w", err))
		}
	}
	if f != nil {
		if err := f.Close(); err != nil {
			errs = append(errs, fmt.Errorf("cache: close: %w", err))
		}
	}
	return errors.Join(errs...)
}
