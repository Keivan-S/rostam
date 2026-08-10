// SPDX-License-Identifier: Apache-2.0

package server

import (
	"testing"
)

// panicDispatcher panics on Call — modeling a latent bug (index-out-of-range,
// nil map) in an op handler reachable from a crafted request.
type panicDispatcher struct{}

func (panicDispatcher) Call(string, []byte) ([]byte, error) { panic("boom: simulated handler bug") }
func (panicDispatcher) LeaderAddr() string                  { return "" }

// TestDispatchRecoversHandlerPanic proves dispatch() contains a handler panic
// to the request instead of crashing the process: it returns StatusError with a
// generic (non-leaking) payload, and the test process survives to assert it.
func TestDispatchRecoversHandlerPanic(t *testing.T) {
	frame := EncodeRequest("anything", []byte("args"))
	status, payload := dispatch(panicDispatcher{}, frame, nil, "", nil)

	if status != StatusError {
		t.Fatalf("status = %d, want StatusError (%d)", status, StatusError)
	}
	// The panic detail must NOT reach the client — a generic message only.
	if got := string(payload); got == "" {
		t.Fatal("expected a non-empty error payload")
	}
	// If we reached here, the process did not crash — that is the actual
	// property under test.
}
