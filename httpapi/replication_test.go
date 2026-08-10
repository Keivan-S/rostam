// SPDX-License-Identifier: Apache-2.0

package httpapi

import (
	"net/http"
	"testing"

	"github.com/rostamlabs/rostam/authz"
	"github.com/rostamlabs/rostam/ops"
)

// replDispatcher answers __repl_metrics__ with a canned JSON body and records
// the op it was asked for. It must NOT be auth-gated (an ops probe carries no
// token), mirroring readyDispatcher.
type replDispatcher struct {
	body  []byte
	gotOp string
}

func (d *replDispatcher) Call(name string, _ []byte) ([]byte, error) {
	d.gotOp = name
	if name == ops.ReplMetricsOp {
		return d.body, nil
	}
	return nil, nil
}
func (d *replDispatcher) LeaderAddr() string { return "" }

func TestReplicationEndpoint(t *testing.T) {
	want := `{"node":"n1","shards":[{"shard":0,"mode":"pb","is_primary":true,"primary":"n1","epoch":1,"isr_size":1,"min_isr":2,"under_replicated":true,"last_seq":5,"committed":3,"backups":[{"node":"n2","acked_seq":3,"lag":2}]}]}`
	d := &replDispatcher{body: []byte(want)}
	// Auth is ENABLED (deny-all) to prove the endpoint bypasses it like ready.
	h := Handler(d, Options{Authenticator: func(authz.AuthRequest) bool { panic("auth must not be consulted for replication metrics") }})
	rec := do(t, h, "GET", "/v1/replication", "", nil)
	if rec.Code != http.StatusOK {
		t.Fatalf("code = %d, want 200 (%s)", rec.Code, rec.Body)
	}
	if d.gotOp != ops.ReplMetricsOp {
		t.Fatalf("dispatched %q, want %q", d.gotOp, ops.ReplMetricsOp)
	}
	if got := rec.Body.String(); got != want {
		t.Fatalf("body = %s, want %s", got, want)
	}
	if ct := rec.Header().Get("Content-Type"); ct != "application/json" {
		t.Fatalf("content-type = %q, want application/json", ct)
	}
}
