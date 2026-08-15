// SPDX-License-Identifier: Apache-2.0
//go:build !unix

package main

// lockDataDir is a no-op off unix: flock(2) has no portable equivalent here, and
// refusing to start would be worse than the single-writer race it guards
// against — the concurrent-clients case is documented (use -connect) and the
// unix build is where rostam-server actually runs.
func lockDataDir(_ string) (func() error, error) {
	return func() error { return nil }, nil
}
