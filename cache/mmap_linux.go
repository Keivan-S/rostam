// SPDX-License-Identifier: Apache-2.0
//go:build linux

package cache

import (
	"errors"
	"fmt"
	"log/slog"
	"os"

	"golang.org/x/sys/unix"
)

// mmapFile opens path (creating if missing), truncates to size bytes,
// and returns the mmap'd region. If mlockRequested is true, attempts
// mlock; failure logs and continues without lock.
func mmapFile(path string, size int64, mlockRequested bool) (*os.File, []byte, error) {
	return mmapFileAlloc(path, size, 0, mlockRequested)
}

// mmapFileAlloc is mmapFile plus a RESERVATION of the first allocBytes of the
// file (allocBytes <= 0 reserves nothing, i.e. plain mmapFile).
//
// WHY. Truncate creates a SPARSE file: it sets the size but allocates no blocks,
// so the blocks are allocated lazily as the mapping is written through. On a
// full filesystem that lazy allocation fails inside a page fault, and the kernel
// reports it by delivering SIGBUS — which Go cannot recover from, so the process
// DIES. Any caller that is about to STORE into a freshly truncated mapping must
// therefore reserve the bytes it intends to touch up front, where ENOSPC comes
// back as an ordinary error it can handle. Cold compaction is exactly that
// caller: without this, a restart meant to rescue a near-full shard on a
// near-full disk turns into a startup crash loop (see compactAtOpen).
//
// fallocate is not universally implemented; EOPNOTSUPP/ENOSYS mean "this
// filesystem cannot reserve", which is the pre-existing behavior and not a
// reason to fail the caller, so they are ignored. ENOSPC (and any other real
// error) is returned.
func mmapFileAlloc(path string, size, allocBytes int64, mlockRequested bool) (*os.File, []byte, error) {
	f, err := os.OpenFile(path, os.O_RDWR|os.O_CREATE, 0o600) //nolint:gosec // G304: path is caller-supplied and intentional
	if err != nil {
		return nil, nil, fmt.Errorf("cache: open %s: %w", path, err)
	}
	if err := f.Truncate(size); err != nil {
		_ = f.Close()
		return nil, nil, fmt.Errorf("cache: truncate %s to %d: %w", path, size, err)
	}
	fd := int(f.Fd()) //nolint:gosec // uintptr->int: fd values are small and positive on Linux
	if allocBytes > 0 {
		if aerr := unix.Fallocate(fd, 0, 0, allocBytes); aerr != nil &&
			!errors.Is(aerr, unix.EOPNOTSUPP) && !errors.Is(aerr, unix.ENOSYS) {
			_ = f.Close()
			return nil, nil, fmt.Errorf("cache: preallocate %d bytes of %s: %w", allocBytes, path, aerr)
		}
	}
	region, err := unix.Mmap(fd, 0, int(size), unix.PROT_READ|unix.PROT_WRITE, unix.MAP_SHARED)
	if err != nil {
		_ = f.Close()
		return nil, nil, fmt.Errorf("cache: mmap %s: %w", path, err)
	}
	if mlockRequested {
		if mlErr := unix.Mlock(region); mlErr != nil {
			slog.Warn("mlock failed (continuing without mlock; consider raising RLIMIT_MEMLOCK)", "component", "cache", "path", path, "err", mlErr)
		}
	}
	return f, region, nil
}

// msync flushes the region to disk synchronously.
func msync(region []byte) error {
	if len(region) == 0 {
		return nil
	}
	if err := unix.Msync(region, unix.MS_SYNC); err != nil {
		return fmt.Errorf("cache: msync: %w", err)
	}
	return nil
}

// munmapAndClose releases the mmap region and closes the file.
func munmapAndClose(f *os.File, region []byte) error {
	var errs []error
	if len(region) > 0 {
		if err := unix.Munmap(region); err != nil {
			errs = append(errs, fmt.Errorf("cache: munmap: %w", err))
		}
	}
	if f != nil {
		if err := f.Close(); err != nil {
			errs = append(errs, fmt.Errorf("cache: close: %w", err))
		}
	}
	return errors.Join(errs...)
}
