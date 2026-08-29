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
// file (allocBytes <= 0 reserves nothing, i.e. plain mmapFile), mirroring the
// Linux fallocate contract.
//
// WHY A SEPARATE RESERVATION IS STILL NEEDED HERE. On Linux the file is sized
// with a sparse truncate, so blocks are allocated lazily inside a page fault and
// a full disk turns into SIGBUS, which Go cannot recover from; fallocate exists
// to pull that failure forward into an ordinary ENOSPC. For an ORDINARY NTFS
// file this platform does not have that problem — os.File.Truncate lands on
// SetEndOfFile, which commits the clusters there and then, so a full disk fails
// the Truncate above with an ordinary error. But a SPARSE or COMPRESSED file
// (including any file created inside a compressed directory) is the exception:
// SetEndOfFile can leave AllocationSize < EndOfFile, i.e. the tail unbacked, so
// a later mapped write into that tail can still hit the delayed out-of-space
// failure this API exists to prevent. reserveAlloc forces the physical
// allocation and then VERIFIES it, and ABORTS (returns an error) if the
// filesystem refused to back the requested bytes, so cold compaction never
// publishes a shard whose tail is only virtually there.
//
// The reservation target is at least allocBytes AND never below size: Windows'
// FILE_ALLOCATION_INFO controls the whole file's allocation and would shrink a
// correctly-sized file if set below its length, whereas Linux fallocate only
// ever adds blocks. Clamping up to size keeps the "never deallocate what
// Truncate just sized" invariant while still honouring "reserve >= allocBytes".
// For an ordinary file the clusters are already committed, so the call is a
// no-op cost.
func mmapFileAlloc(path string, size, allocBytes int64, mlockRequested bool) (*os.File, []byte, error) {
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
	if allocBytes > 0 {
		needed := size
		if allocBytes > needed {
			needed = allocBytes
		}
		if aerr := reserveAlloc(f, needed); aerr != nil {
			_ = f.Close()
			return nil, nil, fmt.Errorf("cache: preallocate %d bytes of %s: %w", needed, path, aerr)
		}
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

// fileAllocationInfo is the input for SetFileInformationByHandle's
// FileAllocationInfo class (Win32 FILE_ALLOCATION_INFO). Setting it forces the
// filesystem to reserve AllocationSize bytes of physical clusters for the file.
// x/sys/windows exports the call and the class constant but not this struct, so
// it is declared locally, matching how the standard library issues the sibling
// FileEndOfFileInfo call.
type fileAllocationInfo struct {
	AllocationSize int64
}

// fileStandardInfo is the output of GetFileInformationByHandleEx's
// FileStandardInfo class (Win32 FILE_STANDARD_INFO). AllocationSize is the
// number of bytes of physical clusters actually reserved for the file, which is
// what the reservation must be verified against.
type fileStandardInfo struct {
	AllocationSize int64
	EndOfFile      int64
	NumberOfLinks  uint32
	DeletePending  byte
	Directory      byte
}

// reserveAlloc forces at least needed bytes of physical allocation for f and
// verifies it, returning an error when the filesystem could not (or would not)
// back the request — the sparse/compressed case where SetEndOfFile alone leaves
// the tail unallocated. It is the Windows analogue of Linux fallocate: pull a
// would-be out-of-space failure forward into an ordinary error at staging time.
func reserveAlloc(f *os.File, needed int64) error {
	h := windows.Handle(f.Fd())
	alloc := fileAllocationInfo{AllocationSize: needed}
	if err := windows.SetFileInformationByHandle(
		h,
		windows.FileAllocationInfo,
		(*byte)(unsafe.Pointer(&alloc)),
		uint32(unsafe.Sizeof(alloc)),
	); err != nil {
		return fmt.Errorf("SetFileInformationByHandle(FileAllocationInfo, %d): %w", needed, err)
	}
	// Verify: a sparse/compressed file can accept the call yet still report a
	// short AllocationSize, which is exactly the delayed-ENOSPC hazard. Confirm
	// the clusters are really there before letting the caller map and write.
	var std fileStandardInfo
	if err := windows.GetFileInformationByHandleEx(
		h,
		windows.FileStandardInfo,
		(*byte)(unsafe.Pointer(&std)),
		uint32(unsafe.Sizeof(std)),
	); err != nil {
		return fmt.Errorf("GetFileInformationByHandleEx(FileStandardInfo): %w", err)
	}
	if std.AllocationSize < needed {
		return fmt.Errorf("reservation short: allocated %d of %d bytes (sparse or compressed backing refused the reservation)", std.AllocationSize, needed)
	}
	return nil
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
