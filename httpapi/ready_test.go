// SPDX-License-Identifier: Apache-2.0

package httpapi

import (
	"errors"
	"net/http"
	"testing"

	"github.com/rostamlabs/rostam/authz"
	"github.com/rostamlabs/rostam/ops"
)

// readyDispatcher answers __ready__ with a configurable outcome and records
// whether auth was required (it must NOT be — readiness is an infra probe).
type readyDispatcher struct {
	readyErr error
	gotOp    string
}

func (d *readyDispatcher) Call(name string, _ []byte) ([]byte, error) {
	d.gotOp = name
	if name == ops.ReadyOp {
		return nil, d.readyErr
	}
	return nil, nil
}
func (d *readyDispatcher) LeaderAddr() string { return "" }

func TestReadyEndpoint(t *testing.T) {
	t.Run("ready → 200", func(t *testing.T) {
		d := &readyDispatcher{readyErr: nil}
		// Auth is ENABLED (deny-all) to prove readiness bypasses it like health.
		h := Handler(d, Options{Authenticator: func(authz.AuthRequest) bool { panic("auth must not be consulted for readiness") }})
		rec := do(t, h, "GET", "/v1/ready", "", nil)
		if rec.Code != http.StatusOK {
			t.Fatalf("code = %d, want 200 (%s)", rec.Code, rec.Body)
		}
		if d.gotOp != ops.ReadyOp {
			t.Fatalf("dispatched %q, want %q", d.gotOp, ops.ReadyOp)
		}
	})

	t.Run("not ready → 503", func(t *testing.T) {
		d := &readyDispatcher{readyErr: errors.New("hosted shards without a leader: [3]")}
		h := Handler(d, Options{})
		rec := do(t, h, "GET", "/v1/ready", "", nil)
		if rec.Code != http.StatusServiceUnavailable {
			t.Fatalf("code = %d, want 503 (%s)", rec.Code, rec.Body)
		}
	})
}
