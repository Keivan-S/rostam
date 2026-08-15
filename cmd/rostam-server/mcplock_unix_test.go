// SPDX-License-Identifier: Apache-2.0
//go:build unix

package main

import (
	"errors"
	"testing"

	"golang.org/x/sys/unix"
)

// TestFlockErrClassification pins which flock(2) failures may carry
// errDataDirBusy. Only nonblocking contention means "another process holds
// it"; ENOSYS/ENOTSUP (a filesystem without flock support), EBADF and friends
// are ordinary I/O failures, and runMcpCmd turns the sentinel into an
// actionable "use -connect" message that would be a wild goose chase for any
// of them.
//
// Tested at this level because the other errnos cannot be provoked through a
// real flock call on a temp dir — which is exactly why the misclassification
// survived: the only case reachable in a test was the one that was correct.
func TestFlockErrClassification(t *testing.T) {
	const path = "/some/data/dir/.mcp.lock"
	for _, tc := range []struct {
		name     string
		err      error
		wantBusy bool
	}{
		{"contention", unix.EWOULDBLOCK, true},
		{"contention EAGAIN", unix.EAGAIN, true},
		{"no flock support", unix.ENOSYS, false},
		{"not supported", unix.ENOTSUP, false},
		{"bad descriptor", unix.EBADF, false},
		{"interrupted", unix.EINTR, false},
		{"io error", unix.EIO, false},
	} {
		t.Run(tc.name, func(t *testing.T) {
			got := flockErr(path, tc.err)
			if busy := errors.Is(got, errDataDirBusy); busy != tc.wantBusy {
				t.Fatalf("flockErr(%v): errDataDirBusy = %v, want %v (%v)", tc.err, busy, tc.wantBusy, got)
			}
			// The underlying errno stays reachable either way: the operator
			// needs to see what actually failed.
			if !errors.Is(got, tc.err) {
				t.Fatalf("flockErr(%v) dropped the cause: %v", tc.err, got)
			}
		})
	}
}
