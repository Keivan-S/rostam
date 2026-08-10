// SPDX-License-Identifier: Apache-2.0

package httpapi

import (
	"net/http/httptest"
	"testing"
)

// TestValidConsistencyBoundedStaleness proves the HTTP edge accepts rc==3
// (bounded-staleness) and rejects rc==4 with a 400 — the additive, fail-loud range
// check mirroring the gRPC server.
func TestValidConsistencyBoundedStaleness(t *testing.T) {
	for _, rc := range []uint8{0, 1, 2, 3} {
		w := httptest.NewRecorder()
		if !validConsistency(w, rc, 0) {
			t.Fatalf("validConsistency(rc=%d) = false (status %d), want true (in range)", rc, w.Code)
		}
	}
	w := httptest.NewRecorder()
	if validConsistency(w, 4, 0) {
		t.Fatal("validConsistency(rc=4) = true, want false (400)")
	}
	if w.Code != 400 {
		t.Fatalf("validConsistency(rc=4) status = %d, want 400", w.Code)
	}
}
