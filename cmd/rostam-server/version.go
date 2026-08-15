// SPDX-License-Identifier: Apache-2.0

package main

import (
	"fmt"
	"runtime"

	"github.com/rostamlabs/rostam/internal/buildinfo"
)

// versionString is the one line `-version` prints. It carries the toolchain and
// target too, because "which Go built it" and "which platform is this" are the
// next two questions after "which version", and a downloaded binary cannot be
// asked any other way.
//
// The version itself comes from internal/buildinfo rather than a var in this
// package: the MCP server has to report the same number to its client, and two
// copies of "what version am I" is how one of them ends up stale.
func versionString() string {
	return fmt.Sprintf("rostam-server %s (%s %s/%s)",
		buildinfo.Version(), runtime.Version(), runtime.GOOS, runtime.GOARCH)
}
