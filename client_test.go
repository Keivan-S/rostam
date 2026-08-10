// SPDX-License-Identifier: Apache-2.0

package rostam

import (
	"bytes"
	"context"
	"errors"
	"net"
	"testing"
	"time"

	"github.com/rostamlabs/rostam/ops"
	"github.com/rostamlabs/rostam/server"
)

// startEmbeddedServer brings up an embedded node + TCP server. Returns
// the server's bind address and a cleanup function. Used by the
// client-mode tests so they exercise the real wire.
func startEmbeddedServer(t *testing.T) (addr string, cleanup func(), reg *ops.Registry) {
	t.Helper()
	reg = ops.NewRegistry()
	if err := ops.RegisterBuiltins(reg); err != nil {
		t.Fatal(err)
	}

	// Pre-bind to get a real port number.
	ln, err := net.Listen("tcp", "127.0.0.1:0")
	if err != nil {
		t.Fatal(err)
	}
	addr = ln.Addr().String()
	_ = ln.Close()

	embeddedStore, err := NewEmbedded(EmbeddedConfig{
		NodeID:    "n1",
		DataDir:   t.TempDir(),
		NumShards: 1,
		Bootstrap: true,
		Ops:       reg,
	})
	if err != nil {
		t.Fatal(err)
	}
	waitLeaderEmbedded(t, embeddedStore)

	dispatcher := embeddedStore.(*embedded).node
	srv, err := server.New(server.Config{Addr: addr, Dispatcher: dispatcher})
	if err != nil {
		_ = embeddedStore.Close()
		t.Fatal(err)
	}
	go func() { _ = srv.Serve() }()

	cleanup = func() {
		_ = srv.Close()
		_ = embeddedStore.Close()
	}
	return addr, cleanup, reg
}

func newSingleClient(t *testing.T) Store {
	t.Helper()
	addr, cleanup, reg := startEmbeddedServer(t)
	t.Cleanup(cleanup)
	store, err := NewClient(ClientConfig{
		Servers: []string{addr},
		Ops:     reg,
	})
	if err != nil {
		t.Fatalf("NewClient: %v", err)
	}
	t.Cleanup(func() { _ = store.Close() })
	return store
}

func TestClientPutGetRoundtrip(t *testing.T) {
	s := newSingleClient(t)
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

func TestClientGetMissingReturnsErrNotFound(t *testing.T) {
	s := newSingleClient(t)
	_, err := s.Get(context.Background(), []byte("missing"))
	if !errors.Is(err, ErrNotFound) {
		t.Fatalf("Get missing: err = %v, want ErrNotFound", err)
	}
}

func TestClientDelReturnsBool(t *testing.T) {
	s := newSingleClient(t)
	ctx := context.Background()
	_ = s.Put(ctx, []byte("k"), []byte("v"), 0)
	existed, err := s.Del(ctx, []byte("k"))
	if err != nil {
		t.Fatalf("Del existing: %v", err)
	}
	if !existed {
		t.Fatal("Del existing: existed=false, want true")
	}
	existed, err = s.Del(ctx, []byte("k"))
	if err != nil {
		t.Fatalf("Del absent: %v", err)
	}
	if existed {
		t.Fatal("Del absent: existed=true, want false")
	}
}

// Suppress "unused" if time is not directly used in the test bodies.
var _ = time.Now

func TestClientCallInvokesRegisteredOp(t *testing.T) {
	s := newSingleClient(t)
	ctx := context.Background()
	if _, err := s.Call(ctx, "put", ops.EncodePutArgs([]byte("k"), []byte("v"), 0)); err != nil {
		t.Fatalf("Call put: %v", err)
	}
	got, err := s.Call(ctx, "get", ops.EncodeKeyArgs([]byte("k")))
	if err != nil {
		t.Fatalf("Call get: %v", err)
	}
	if !bytes.Equal(got, []byte("v")) {
		t.Fatalf("Call get = %q, want v", got)
	}
}

func TestClientIsLeaderEventuallyTrue(t *testing.T) {
	s := newSingleClient(t)
	if err := s.Put(context.Background(), []byte("probe"), []byte("v"), 0); err != nil {
		t.Fatalf("warmup Put: %v", err)
	}
	deadline := time.Now().Add(2 * time.Second)
	for time.Now().Before(deadline) {
		if s.IsLeader([]byte("probe")) {
			return
		}
		time.Sleep(20 * time.Millisecond)
	}
	t.Fatal("IsLeader never returned true within 2s")
}

func TestClientLeaderAddrEventuallyNonEmpty(t *testing.T) {
	s := newSingleClient(t)
	if err := s.Put(context.Background(), []byte("probe"), []byte("v"), 0); err != nil {
		t.Fatalf("warmup Put: %v", err)
	}
	deadline := time.Now().Add(2 * time.Second)
	for time.Now().Before(deadline) {
		if s.LeaderAddr([]byte("probe")) != "" {
			return
		}
		time.Sleep(20 * time.Millisecond)
	}
	t.Fatal("LeaderAddr never became non-empty within 2s")
}

// TestNetworkedCreateCollectionRejectsReservedName asserts the client-side
// guard rejects '#'/'@' names before any network round-trip — the store has a
// nil client, so reaching n.c.Call would panic.
func TestNetworkedCreateCollectionRejectsReservedName(t *testing.T) {
	n := &networkedStore{} // nil client: the guard must short-circuit before Call
	for _, name := range []string{"bad#name", "bad@name"} {
		if err := n.CreateCollection(context.Background(), name, VectorConfig{Dim: 8}); err == nil {
			t.Errorf("%q: expected error, got nil", name)
		}
	}
}

// TestNetworkedMVCreateCollectionRejectsReservedName asserts the client-side
// guard rejects '#'/'@' names before any network round-trip — the store has a
// nil client, so reaching n.c.Call would panic.
func TestNetworkedMVCreateCollectionRejectsReservedName(t *testing.T) {
	n := &networkedStore{} // nil client: the guard must short-circuit before Call
	for _, name := range []string{"bad#name", "bad@name"} {
		if err := n.VectorMVCreateCollection(context.Background(), name, MultiVectorConfig{Dim: 8}); err == nil {
			t.Errorf("%q: expected error, got nil", name)
		}
	}
}

// TestNetworkedResplitRejectsLowNewP asserts the client-side guard rejects
// newP <= 1 before any network round-trip — the store has a nil client, so
// reaching n.c.Call would panic. Covers both the dense and MV variants.
func TestNetworkedResplitRejectsLowNewP(t *testing.T) {
	n := &networkedStore{} // nil client: the guard must short-circuit before Call
	for _, newP := range []int{0, 1, -1} {
		if err := n.VectorResplit(context.Background(), "c", newP); err == nil {
			t.Errorf("VectorResplit(newP=%d): expected error, got nil", newP)
		}
		if err := n.VectorMVResplit(context.Background(), "c", newP); err == nil {
			t.Errorf("VectorMVResplit(newP=%d): expected error, got nil", newP)
		}
	}
}

func TestClientCloseIsIdempotent(t *testing.T) {
	addr, cleanup, reg := startEmbeddedServer(t)
	defer cleanup()
	s, err := NewClient(ClientConfig{Servers: []string{addr}, Ops: reg})
	if err != nil {
		t.Fatal(err)
	}
	if err := s.Close(); err != nil {
		t.Fatalf("first Close: %v", err)
	}
	if err := s.Close(); err != nil {
		t.Fatalf("second Close: %v", err)
	}
}
