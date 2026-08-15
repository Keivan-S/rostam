// SPDX-License-Identifier: Apache-2.0
//go:build !unix && !windows

package main

import (
	"fmt"
	"runtime"
)

// lockDataDir refuses persistent embedded mode on a platform with no file lock
// to enforce it (js/wasm, plan9).
//
// This used to return a silent no-op, on the reasoning that refusing to start
// was worse than the race. It is not: the race corrupts the store. Two
// processes on one -data directory map the same cache files and both believe
// they own them, and the user-facing docs promise the second one is refused —
// so a no-op does not avoid the failure, it just moves it somewhere the user
// cannot see it coming and cannot attribute it afterwards.
//
// The two modes that need no lock stay available here, and the error names
// both: heap mode (-data "") has nothing on disk to share, and -connect makes
// concurrency the remote server's problem rather than ours.
func lockDataDir(dir string) (func() error, error) {
	return nil, fmt.Errorf("persistent embedded mode is not available on %s: this platform has no file lock, "+
		"so the single-writer contract on data directory %s cannot be enforced and a second process would corrupt it; "+
		`run with -data "" for heap mode (no persistence), or -connect to a rostam-server`, runtime.GOOS, dir)
}
