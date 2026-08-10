// SPDX-License-Identifier: Apache-2.0
//go:build !amd64

package vector

import "unsafe"

// prefetch is a no-op on non-amd64 platforms (the amd64 build issues a
// PREFETCHT0). arm64 has a native range prefetch (see prefetchRange in
// distance_arm64.s); this single-line form has no arm64 caller.
func prefetch(unsafe.Pointer) {}
