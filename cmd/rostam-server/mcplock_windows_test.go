// SPDX-License-Identifier: Apache-2.0
//go:build windows

package main

import (
	"errors"
	"testing"

	"golang.org/x/sys/windows"
)

// TestLockFileErrClassification is the Windows counterpart of
// TestFlockErrClassification: only genuine contention may carry
// errDataDirBusy, since runMcpCmd turns that sentinel into "another
// rostam-server mcp process is using this data directory".
func TestLockFileErrClassification(t *testing.T) {
	const path = `C:\data\.mcp.lock`
	for _, tc := range []struct {
		name     string
		err      error
		wantBusy bool
	}{
		{"lock violation", windows.ERROR_LOCK_VIOLATION, true},
		{"sharing violation", windows.ERROR_SHARING_VIOLATION, true},
		{"access denied", windows.ERROR_ACCESS_DENIED, false},
		{"not supported", windows.ERROR_NOT_SUPPORTED, false},
		{"invalid handle", windows.ERROR_INVALID_HANDLE, false},
	} {
		t.Run(tc.name, func(t *testing.T) {
			got := lockFileErr(path, tc.err)
			if busy := errors.Is(got, errDataDirBusy); busy != tc.wantBusy {
				t.Fatalf("lockFileErr(%v): errDataDirBusy = %v, want %v (%v)", tc.err, busy, tc.wantBusy, got)
			}
			if !errors.Is(got, tc.err) {
				t.Fatalf("lockFileErr(%v) dropped the cause: %v", tc.err, got)
			}
		})
	}
}
