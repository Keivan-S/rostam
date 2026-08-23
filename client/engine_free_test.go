// SPDX-License-Identifier: Apache-2.0
package client

import (
	"bytes"
	"os/exec"
	"testing"
)

// TestClientIsEngineFree locks Phase 1: the client must not transitively import
// the engine vector package (or objstore / vector/analysis). A future import that
// re-couples the client to the engine fails here.
func TestClientIsEngineFree(t *testing.T) {
	out, err := exec.Command("go", "list", "-deps", "github.com/rostamlabs/rostam/client").Output()
	if err != nil {
		t.Fatalf("go list -deps client: %v", err)
	}
	for _, banned := range []string{
		"github.com/rostamlabs/rostam/vector",
		"github.com/rostamlabs/rostam/objstore",
		"github.com/rostamlabs/rostam/vector/analysis",
	} {
		if bytes.Contains(out, []byte("\n"+banned+"\n")) ||
			bytes.HasPrefix(out, []byte(banned+"\n")) {
			t.Errorf("client transitively imports engine package %q — the vtypes leaf must cover its needs", banned)
		}
	}
}
