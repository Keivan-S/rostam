// SPDX-License-Identifier: Apache-2.0

package rostam

import (
	"bytes"
	"context"
	"errors"
	"testing"

	"github.com/rostamlabs/rostam/ops"
)

func newSingleDirect(t *testing.T) Store {
	t.Helper()
	reg := ops.NewRegistry()
	if err := ops.RegisterBuiltins(reg); err != nil {
		t.Fatalf("RegisterBuiltins: %v", err)
	}
	s, err := NewDirect(DirectConfig{
		Ops:   reg,
		Cache: CacheConfig{NumShardsPerNode: 1},
	})
	if err != nil {
		t.Fatalf("NewDirect: %v", err)
	}
	t.Cleanup(func() {
		if err := s.Close(); err != nil {
			t.Logf("direct Close: %v", err)
		}
	})
	return s
}

func TestDirectPutGetRoundtrip(t *testing.T) {
	s := newSingleDirect(t)
	ctx := context.Background()
	if err := s.Put(ctx, []byte("k"), []byte("v"), 0); err != nil {
		t.Fatalf("Put: %v", err)
	}
	got, err := s.Get(ctx, []byte("k"))
	if err != nil {
		t.Fatalf("Get: %v", err)
	}
	if !bytes.Equal(got, []byte("v")) {
		t.Fatalf("Get = %q, want v", got)
	}
}

func TestDirectIsLeaderAlwaysTrue(t *testing.T) {
	s := newSingleDirect(t)
	if !s.IsLeader([]byte("anything")) {
		t.Fatal("Direct.IsLeader: got false, want true")
	}
}

func TestDirectLeaderAddrAlwaysEmpty(t *testing.T) {
	s := newSingleDirect(t)
	if s.LeaderAddr([]byte("anything")) != "" {
		t.Fatal("Direct.LeaderAddr: non-empty (Direct has no network surface)")
	}
}

func TestDirectMissingOpReturnsError(t *testing.T) {
	s := newSingleDirect(t)
	_, err := s.Call(context.Background(), "no_such_op", nil)
	if err == nil {
		t.Fatal("Call unknown op: expected error, got nil")
	}
}

func TestDirectNilOpsRejected(t *testing.T) {
	_, err := NewDirect(DirectConfig{})
	if err == nil {
		t.Fatal("NewDirect with nil Ops: expected error, got nil")
	}
}

func TestDirectGetMissingReturnsErrNotFound(t *testing.T) {
	s := newSingleDirect(t)
	_, err := s.Get(context.Background(), []byte("missing"))
	if !errors.Is(err, ErrNotFound) {
		t.Fatalf("Get missing: err = %v, want ErrNotFound", err)
	}
}

func TestDirectServerRoundtrip(t *testing.T) {
	reg := ops.NewRegistry()
	if err := ops.RegisterBuiltins(reg); err != nil {
		t.Fatal(err)
	}
	srv, err := NewDirectServer("127.0.0.1:0", DirectConfig{Ops: reg})
	if err != nil {
		t.Fatalf("NewDirectServer: %v", err)
	}
	defer func() { _ = srv.Close() }()
	if srv.Addr() == "" {
		t.Fatal("Addr() returned empty")
	}

	cli, err := NewClient(ClientConfig{Servers: []string{srv.Addr()}, Ops: reg})
	if err != nil {
		t.Fatalf("NewClient: %v", err)
	}
	defer func() { _ = cli.Close() }()

	ctx := context.Background()
	if err := cli.Put(ctx, []byte("k"), []byte("v"), 0); err != nil {
		t.Fatalf("Put: %v", err)
	}
	got, err := cli.Get(ctx, []byte("k"))
	if err != nil {
		t.Fatalf("Get: %v", err)
	}
	if !bytes.Equal(got, []byte("v")) {
		t.Fatalf("Get = %q, want v", got)
	}
}

func TestDirectServerCloseIsIdempotent(t *testing.T) {
	reg := ops.NewRegistry()
	_ = ops.RegisterBuiltins(reg)
	srv, err := NewDirectServer("127.0.0.1:0", DirectConfig{Ops: reg})
	if err != nil {
		t.Fatal(err)
	}
	if err := srv.Close(); err != nil {
		t.Fatalf("first Close: %v", err)
	}
	if err := srv.Close(); err != nil {
		t.Fatalf("second Close: %v", err)
	}
}

func TestDirectServerRejectsNilOps(t *testing.T) {
	_, err := NewDirectServer("127.0.0.1:0", DirectConfig{})
	if err == nil {
		t.Fatal("NewDirectServer with nil Ops: expected error, got nil")
	}
}
