// SPDX-License-Identifier: Apache-2.0

package main

import (
	"fmt"
	"runtime"
	"runtime/debug"
)

// version is stamped at release time with -ldflags "-X main.version=v1.2.3".
// It stays empty for every other build, where buildVersion derives an identity
// from what the toolchain recorded instead — so `go install ...@v0.1.0` and a
// local `go build` both report something truthful without a release pipeline.
var version = ""

// buildVersion reports the most specific identity available, in descending
// order of confidence:
//
//	v0.1.0                       a release binary (ldflags), or
//	                             `go install ...@v0.1.0`
//	v0.1.1-0.2026...-abc123+dirty a local build: Go derives a pseudo-version from
//	                             the last tag, the commit time and the revision,
//	                             and marks a modified tree
//	devel-abc123def456           a build with VCS data but no module version
//	(unknown)                    no build info at all (stripped/synthetic build)
//
// The dirty marker matters for bug reports: it is the difference between a
// revision someone else can check out and one that exists only on one machine.
func buildVersion() string {
	if version != "" {
		return version
	}
	info, ok := debug.ReadBuildInfo()
	if !ok {
		return "(unknown)"
	}
	// `go install module@version` records the module version. A build from a
	// local checkout records "(devel)" here and puts the detail in Settings.
	if v := info.Main.Version; v != "" && v != "(devel)" {
		return v
	}
	var rev, suffix string
	for _, s := range info.Settings {
		switch s.Key {
		case "vcs.revision":
			rev = s.Value
		case "vcs.modified":
			if s.Value == "true" {
				suffix = "-dirty"
			}
		}
	}
	if rev == "" {
		return "(devel)"
	}
	if len(rev) > 12 {
		rev = rev[:12]
	}
	return "devel-" + rev + suffix
}

// versionString is the one line `-version` prints. It carries the toolchain and
// target too, because "which Go built it" and "which platform is this" are the
// next two questions after "which version", and a downloaded binary cannot be
// asked any other way.
func versionString() string {
	return fmt.Sprintf("rostam-server %s (%s %s/%s)",
		buildVersion(), runtime.Version(), runtime.GOOS, runtime.GOARCH)
}
