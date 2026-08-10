// SPDX-License-Identifier: Apache-2.0

//go:build !linux

package rostam

// systemMemoryBytes returns 0 on platforms with no probe wired up, so
// defaultCacheBudget falls back to a fixed conservative budget. The engine's
// production target is Linux; this keeps non-Linux builds (and the
// cgo-disabled cross-compile guard) compiling and safely bounded.
func systemMemoryBytes() int64 { return 0 }
