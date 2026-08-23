// SPDX-License-Identifier: Apache-2.0
package wire

import (
	"bytes"
	"os/exec"
	"testing"
)

// TestWireIsEngineFree locks the ops/wire half of Phase 1: the wire codec must
// not transitively import the engine vector package (or objstore / vector/
// analysis). client depends on ops/wire, so this is partly covered by
// TestClientIsEngineFree — but a future refactor that stops client importing
// ops/wire would remove that cover, and this gate keeps the boundary honest on
// its own.
func TestWireIsEngineFree(t *testing.T) {
	out, err := exec.Command("go", "list", "-deps", "github.com/rostamlabs/rostam/sdk/wire").Output()
	if err != nil {
		t.Fatalf("go list -deps ops/wire: %v", err)
	}
	for _, banned := range []string{
		"github.com/rostamlabs/rostam/vector",
		"github.com/rostamlabs/rostam/objstore",
		"github.com/rostamlabs/rostam/vector/analysis",
	} {
		if bytes.Contains(out, []byte("\n"+banned+"\n")) ||
			bytes.HasPrefix(out, []byte(banned+"\n")) {
			t.Errorf("ops/wire transitively imports engine package %q — the vtypes leaf must cover its needs", banned)
		}
	}
}
