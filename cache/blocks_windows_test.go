// SPDX-License-Identifier: Apache-2.0
//go:build windows

package cache

import (
	"os"
	"testing"
	"unsafe"

	"golang.org/x/sys/windows"
)

// allocatedBlocks reports the 512-byte blocks actually allocated to path, so
// the compaction tests can assert the same thing they assert on Linux: that the
// published pages file is fully allocated and carries no hole a later write
// could fault into.
//
// NTFS does not hand out sparse files unless asked (FSCTL_SET_SPARSE), so this
// is expected to equal the logical size rounded up to the cluster — that the
// assertion is easy to satisfy here is the point, not a reason to skip it.
func allocatedBlocks(t *testing.T, path string) int64 {
	t.Helper()
	f, err := os.Open(path) //nolint:gosec // G304: test-supplied path
	if err != nil {
		t.Fatalf("open %s: %v", path, err)
	}
	defer func() { _ = f.Close() }()
	var info fileStandardInfo
	if err := windows.GetFileInformationByHandleEx(
		windows.Handle(f.Fd()),
		windows.FileStandardInfo,
		(*byte)(unsafe.Pointer(&info)),
		uint32(unsafe.Sizeof(info)),
	); err != nil {
		t.Fatalf("GetFileInformationByHandleEx %s: %v", path, err)
	}
	return info.AllocationSize / 512
}
