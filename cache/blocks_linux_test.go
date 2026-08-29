// SPDX-License-Identifier: Apache-2.0
//go:build linux

package cache

import (
	"syscall"
	"testing"
)

// allocatedBlocks reports the 512-byte blocks actually allocated to path. The
// pages file is always truncated to its full logical size, so "the file shrank"
// means fewer allocated blocks (the tail of a compacted file is a hole), not a
// smaller st_size.
func allocatedBlocks(t *testing.T, path string) int64 {
	t.Helper()
	var st syscall.Stat_t
	if err := syscall.Stat(path, &st); err != nil {
		t.Fatalf("stat %s: %v", path, err)
	}
	return st.Blocks
}
