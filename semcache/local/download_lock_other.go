// SPDX-License-Identifier: Apache-2.0
//go:build localembed && !linux && !darwin

package local

// lockDir is a best-effort no-op on platforms with no flock(2) equivalent
// wired up here. Cross-process download coordination is unix-only; other
// platforms assume a single first-run process downloads the model (no
// concurrent Ensure callers racing on the same cache directory).
func lockDir(dir string) (func() error, error) {
	return func() error { return nil }, nil
}
