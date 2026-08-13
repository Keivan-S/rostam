// SPDX-License-Identifier: Apache-2.0

package rostam

import (
	"errors"
	"net"
	"syscall"
	"testing"

	"github.com/rostamlabs/rostam/ops"
)

// newTestOps builds the registry NewServer requires for the single-node backend.
func newTestOps(t *testing.T) *ops.Registry {
	t.Helper()
	reg := ops.NewRegistry()
	if err := ops.RegisterBuiltins(reg); err != nil {
		t.Fatalf("RegisterBuiltins: %v", err)
	}
	return reg
}

// freeTCPPort returns a loopback address nothing is listening on at the moment
// it returns. The epoll transport cannot report a kernel-chosen port (gnet
// exposes no accessor for the bound address), so these tests must name a real
// port rather than binding :0 and reading it back.
//
// That leaves an unavoidable gap between releasing the reservation here and the
// server binding it, in which another process can take the port. The gap cannot
// be closed without a seam that hands a pre-bound listener to the server, so
// serveOnFreePort below retries instead — see the note there.
func freeTCPPort(t *testing.T) string {
	t.Helper()
	ln, err := net.Listen("tcp", "127.0.0.1:0")
	if err != nil {
		t.Fatalf("reserve port: %v", err)
	}
	addr := ln.Addr().String()
	if err := ln.Close(); err != nil {
		t.Fatalf("release port: %v", err)
	}
	return addr
}

// serveOnFreePort starts a server on a free loopback port, retrying only when
// the port was taken between reservation and bind.
//
// Retrying ONLY that error matters: a blanket retry would paper over a genuine
// bind failure and turn a real defect into a slow test. Anything that is not an
// address-in-use error fails immediately.
func serveOnFreePort(t *testing.T, epoll bool) (*Server, string) {
	t.Helper()
	const attempts = 5
	for i := range attempts {
		addr := freeTCPPort(t)
		srv, err := NewServer(ServerConfig{
			TCPAddr:      addr,
			EpollTCP:     epoll,
			DirectConfig: DirectConfig{Ops: newTestOps(t)},
		})
		if err == nil {
			return srv, addr
		}
		if !errors.Is(err, syscall.EADDRINUSE) {
			t.Fatalf("NewServer(%s): %v", addr, err)
		}
		t.Logf("attempt %d: %s was taken between reservation and bind, retrying", i+1, addr)
	}
	t.Fatalf("no free port survived %d attempts", attempts)
	return nil, ""
}

// TCPAddr has to answer for whichever TCP transport is running. Epoll is the
// DEFAULT for single-node and lives in a different field, so a version of this
// accessor that only knew about the goroutine server reported "TCP is disabled"
// for the configuration most servers actually run — and the startup log, which
// iterates these accessors, silently omitted the listener while it was up and
// accepting connections.
func TestTCPAddrReportsBothTransports(t *testing.T) {
	for _, tc := range []struct {
		name  string
		epoll bool
	}{
		{"epoll (the single-node default)", true},
		{"goroutine-per-connection", false},
	} {
		t.Run(tc.name, func(t *testing.T) {
			srv, addr := serveOnFreePort(t, tc.epoll)
			defer func() {
				if cerr := srv.Close(); cerr != nil {
					t.Errorf("Close: %v", cerr)
				}
			}()

			if got := srv.TCPAddr(); got != addr {
				t.Errorf("TCPAddr() = %q, want %q", got, addr)
			}
			// The address must be dialable, so the accessor cannot be reporting a
			// listener that does not exist.
			c, derr := net.Dial("tcp", srv.TCPAddr())
			if derr != nil {
				t.Fatalf("dial %s: %v", srv.TCPAddr(), derr)
			}
			if cerr := c.Close(); cerr != nil {
				t.Errorf("close conn: %v", cerr)
			}
		})
	}
}

// With no TCP address configured there is no transport to report, on either path.
func TestTCPAddrEmptyWhenDisabled(t *testing.T) {
	srv, err := NewServer(ServerConfig{
		HTTPAddr:     "127.0.0.1:0",
		DirectConfig: DirectConfig{Ops: newTestOps(t)},
	})
	if err != nil {
		t.Fatalf("NewServer: %v", err)
	}
	defer func() { _ = srv.Close() }()
	if got := srv.TCPAddr(); got != "" {
		t.Errorf("TCPAddr() = %q, want \"\" when -tcp is unset", got)
	}
}

// A server closed promptly after construction must not take the process with it.
//
// The HTTP serving goroutine used to read srv.httpSrv when it was first
// scheduled, while Close sets that field to nil — so closing before the
// goroutine ran dereferenced nil, and a panic in a goroutine cannot be recovered
// by the caller. This loops to widen a window that is otherwise scheduler-
// dependent; it panics rather than fails when the bug is present.
func TestCloseImmediatelyAfterStartDoesNotPanic(t *testing.T) {
	for i := 0; i < 50; i++ {
		srv, err := NewServer(ServerConfig{
			HTTPAddr:     "127.0.0.1:0",
			DirectConfig: DirectConfig{Ops: newTestOps(t)},
		})
		if err != nil {
			t.Fatalf("iteration %d: NewServer: %v", i, err)
		}
		if err := srv.Close(); err != nil {
			t.Fatalf("iteration %d: Close: %v", i, err)
		}
	}
}
