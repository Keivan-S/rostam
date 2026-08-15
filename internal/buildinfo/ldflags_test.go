// SPDX-License-Identifier: Apache-2.0

package buildinfo

import (
	"os"
	"regexp"
	"strings"
	"testing"
)

// The linker silently ignores an -X flag whose target does not name a real
// package-level string: a typo, a moved package or a renamed variable does not
// fail the release build, it just quietly leaves the fallback in place and
// ships binaries that report a derived version instead of the release. Nothing
// else in CI would notice, because everything still compiles and runs.
//
// So this asserts the release pipeline's -X target still resolves to this
// package's variable, by reading the target out of the release config rather
// than restating it.
func TestGoreleaserLdflagTargetIsThisVariable(t *testing.T) {
	const wantPkg = "github.com/rostamlabs/rostam/internal/buildinfo"
	const wantVar = "version"

	cfg, err := os.ReadFile("../../.goreleaser.yaml")
	if err != nil {
		t.Fatalf("read release config: %v", err)
	}

	// -X <import/path>.<varname>=<value>
	re := regexp.MustCompile(`-X\s+([^\s=]+)=`)
	matches := re.FindAllStringSubmatch(string(cfg), -1)
	if len(matches) == 0 {
		t.Fatal("no -X ldflag found in .goreleaser.yaml; the release no longer stamps a version")
	}

	var targets []string
	for _, m := range matches {
		targets = append(targets, m[1])
		dot := strings.LastIndex(m[1], ".")
		if dot < 0 {
			continue
		}
		if m[1][:dot] == wantPkg && m[1][dot+1:] == wantVar {
			return // found it
		}
	}
	t.Fatalf("release config stamps %v, but the version variable is %s.%s — "+
		"a mismatched -X target is ignored by the linker, so releases would "+
		"report a derived version instead of the tag", targets, wantPkg, wantVar)
}

// The variable the -X target names must actually be declared here. The test
// above proves the config and this package agree on a name; this proves the
// name exists.
//
// The case it catches is a *consistent* rename -- declaration and uses renamed
// together, as any refactoring tool would do. That compiles cleanly and passes
// every other test, and the only symptom is that released binaries quietly stop
// reporting their tag. (Renaming just the declaration is caught by the compiler
// and needs no test.)
//
// Matching the declaration rather than a full literal keeps this from failing
// on a harmless change of initializer form.
func TestVersionVariableIsDeclared(t *testing.T) {
	src, err := os.ReadFile("buildinfo.go")
	if err != nil {
		t.Fatalf("read source: %v", err)
	}
	if !regexp.MustCompile(`(?m)^var\s+version\b`).Match(src) {
		t.Fatal("buildinfo.go no longer declares a package-level 'version' var; " +
			"the release ldflag targets that exact symbol, and the linker " +
			"ignores an -X flag it cannot resolve")
	}
}

// With no ldflag set (the normal `go test` case) Version must still answer
// something usable rather than an empty string — the MCP server puts this
// straight into its initialize response.
func TestVersionIsNeverEmpty(t *testing.T) {
	if got := Version(); got == "" {
		t.Fatal("Version() returned an empty string")
	}
}
