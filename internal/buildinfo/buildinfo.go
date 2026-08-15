// SPDX-License-Identifier: Apache-2.0

// Package buildinfo reports the binary's own version.
//
// This lives outside `main` because more than one place has to answer "which
// version is this?" and they must not answer differently. The MCP server
// reports a version to its client in `initialize`, and it used to carry a
// literal that was written once and never updated — so a v0.2.0 binary
// introduced itself to Claude Code as 0.1.0, and every bug report filed
// through an MCP client named the wrong release.
package buildinfo

import "runtime/debug"

// version is stamped at release time with
// -ldflags "-X github.com/rostamlabs/rostam/internal/buildinfo.version=v1.2.3".
// It stays empty for every other build, where Version derives an identity from
// what the toolchain recorded instead — so `go install ...@v0.1.0` and a local
// `go build` both report something truthful without a release pipeline.
//
// Note that an -X path that does not match a real package-level string is
// silently ignored by the linker: a typo here does not fail the build, it just
// leaves the fallback in place. TestLdflagsTargetIsReachable covers that.
var version = ""

// Version reports the most specific identity available, in descending order of
// confidence:
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
func Version() string {
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
