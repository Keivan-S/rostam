// SPDX-License-Identifier: Apache-2.0
//go:build !amd64 && !arm64

package vector

import "unsafe"

// prefetchRange is a no-op on platforms with no assembly prefetch (amd64 issues
// PREFETCHT0, arm64 issues PRFM PLDL1KEEP). Callers must treat it as a pure
// hint — search results never depend on it.
func prefetchRange(unsafe.Pointer, int) {}
