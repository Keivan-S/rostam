// SPDX-License-Identifier: Apache-2.0

package client

import (
	"bufio"
	"bytes"
	"context"
	"encoding/binary"
	"errors"
	"io"
	"net"
	"strings"
	"sync"
	"sync/atomic"
	"testing"
	"time"

	"github.com/rostamlabs/rostam/cache"
	"github.com/rostamlabs/rostam/ops"
	"github.com/rostamlabs/rostam/ops/wire"
	"github.com/rostamlabs/rostam/server"
	"github.com/rostamlabs/rostam/shard"
)

func startTestStack(t *testing.T) (string, func()) {
	t.Helper()
	// The embedded shard store dispatches ops locally, so it needs the
	// full handler-carrying ops.Registry (not the client's routing-only
	// wire.Registry, which the client-construction tests below use instead).
	reg := ops.NewRegistry()
	if err := ops.RegisterBuiltins(reg); err != nil {
		t.Fatal(err)
	}
	cc := cache.DefaultConfig()
	cc.NumShards = 1
	store, err := shard.New(shard.Config{
		NodeID: "node1", DataDir: t.TempDir(),
		Cache: cc, Ops: reg,
		Bootstrap:       true,
		RaftHeartbeatMs: 50, RaftElectionMs: 100, NoSync: true,
	})
	if err != nil {
		t.Fatal(err)
	}
	deadline := time.Now().Add(3 * time.Second)
	for time.Now().Before(deadline) {
		if store.IsLeader() {
			break
		}
		time.Sleep(20 * time.Millisecond)
	}
	if !store.IsLeader() {
		t.Fatal("store never became leader")
	}

	srv, err := server.New(server.Config{Addr: "127.0.0.1:0", Dispatcher: store})
	if err != nil {
		t.Fatal(err)
	}
	go func() { _ = srv.Serve() }()

	return srv.Addr().String(), func() {
		_ = srv.Close()
		_ = store.Close()
	}
}

func TestClientPutGet(t *testing.T) {
	addr, stop := startTestStack(t)
	defer stop()
	c, err := New(Config{Servers: []string{addr}})
	if err != nil {
		t.Fatal(err)
	}
	defer func() { _ = c.Close() }()

	ctx := context.Background()
	if _, err := c.Call(ctx, "put", wire.EncodePutArgs([]byte("k"), []byte("v"), 0)); err != nil {
		t.Fatalf("Call put: %v", err)
	}
	res, err := c.Call(ctx, "get", wire.EncodeKeyArgs([]byte("k")))
	if err != nil {
		t.Fatalf("Call get: %v", err)
	}
	if !bytes.Equal(res, []byte("v")) {
		t.Fatalf("get result = %q, want v", res)
	}
}

func TestClientGetMissingReturnsErrNotFound(t *testing.T) {
	addr, stop := startTestStack(t)
	defer stop()
	c, _ := New(Config{Servers: []string{addr}})
	defer func() { _ = c.Close() }()
	_, err := c.Call(context.Background(), "get", wire.EncodeKeyArgs([]byte("absent")))
	if !errors.Is(err, ErrNotFound) {
		t.Fatalf("err = %v, want ErrNotFound", err)
	}
}

func TestClientUnknownOpReturnsRemoteError(t *testing.T) {
	addr, stop := startTestStack(t)
	defer stop()
	c, _ := New(Config{Servers: []string{addr}})
	defer func() { _ = c.Close() }()
	_, err := c.Call(context.Background(), "no_such_op", nil)
	var rErr *RemoteError
	if !errors.As(err, &rErr) {
		t.Fatalf("err type = %T (%v), want *RemoteError", err, err)
	}
	if rErr.Op != "no_such_op" || rErr.Msg == "" {
		t.Fatalf("RemoteError = %+v", rErr)
	}
}

func TestClientIncrAccumulates(t *testing.T) {
	addr, stop := startTestStack(t)
	defer stop()
	c, _ := New(Config{Servers: []string{addr}})
	defer func() { _ = c.Close() }()
	ctx := context.Background()
	for range 3 {
		_, err := c.Call(ctx, "incr", wire.EncodeIncrArgs([]byte("counter"), 2))
		if err != nil {
			t.Fatal(err)
		}
	}
	res, _ := c.Call(ctx, "incr", wire.EncodeIncrArgs([]byte("counter"), -1))
	v, _ := wire.DecodeIncrResult(res)
	if v != 5 {
		t.Fatalf("incr accumulated = %d, want 5", v)
	}
}

func TestClientConcurrentCalls(t *testing.T) {
	addr, stop := startTestStack(t)
	defer stop()
	c, _ := New(Config{Servers: []string{addr}, MaxConnsPerServer: 4})
	defer func() { _ = c.Close() }()

	var wg sync.WaitGroup
	const goroutines = 16
	const iters = 50
	for g := range goroutines {
		wg.Add(1)
		go func(id int) {
			defer wg.Done()
			ctx := context.Background()
			for i := range iters {
				k := []byte{byte(id), byte(i)} //nolint:gosec // id and i are bounded by goroutines/iters constants (16, 50)
				if _, err := c.Call(ctx, "put", wire.EncodePutArgs(k, []byte{1}, 0)); err != nil {
					t.Errorf("put id=%d i=%d: %v", id, i, err)
					return
				}
			}
		}(g)
	}
	wg.Wait()
}

func TestClientCloseIdempotent(t *testing.T) {
	addr, stop := startTestStack(t)
	defer stop()
	c, _ := New(Config{Servers: []string{addr}})
	if err := c.Close(); err != nil {
		t.Fatalf("Close 1: %v", err)
	}
	if err := c.Close(); err != nil {
		t.Fatalf("Close 2: %v", err)
	}
}

func acceptAndRespond(ln net.Listener, respond func([]byte) (uint8, []byte)) {
	for {
		conn, err := ln.Accept()
		if err != nil {
			return
		}
		go func(c net.Conn) {
			defer func() { _ = c.Close() }()
			r := bufio.NewReader(c)
			w := bufio.NewWriter(c)
			for {
				var hdr [4]byte
				if _, err := io.ReadFull(r, hdr[:]); err != nil {
					return
				}
				n := binary.BigEndian.Uint32(hdr[:])
				body := make([]byte, n)
				if _, err := io.ReadFull(r, body); err != nil {
					return
				}
				status, payload := respond(body)
				resp := make([]byte, 1+4+len(payload))
				resp[0] = status
				binary.BigEndian.PutUint32(resp[1:5], uint32(len(payload))) //nolint:gosec // test-only
				copy(resp[5:], payload)
				var respHdr [4]byte
				binary.BigEndian.PutUint32(respHdr[:], uint32(len(resp))) //nolint:gosec // test-only
				if _, werr := w.Write(respHdr[:]); werr != nil {
					return
				}
				if _, werr := w.Write(resp); werr != nil {
					return
				}
				if ferr := w.Flush(); ferr != nil {
					return
				}
			}
		}(conn)
	}
}

func encodeLeaderAddrFrame(addr string) []byte {
	out := make([]byte, 2+len(addr))
	binary.BigEndian.PutUint16(out[0:2], uint16(len(addr))) //nolint:gosec // test-only
	copy(out[2:], addr)
	return out
}

func TestClientRetriesOnNotLeader(t *testing.T) {
	// Start the leader first so its address is known.
	leaderAddr, stopLeader := startFakeServer(t, func(_ []byte) (uint8, []byte) {
		return StatusOK, []byte("ok")
	})
	defer stopLeader()
	// Start the not-leader, hinting at the leader.
	notLeaderAddr, stopNotLeader := startFakeServer(t, func(_ []byte) (uint8, []byte) {
		return StatusNotLeader, encodeLeaderAddrFrame(leaderAddr)
	})
	defer stopNotLeader()

	c, _ := New(Config{Servers: []string{notLeaderAddr, leaderAddr}})
	defer func() { _ = c.Close() }()

	res, err := c.Call(context.Background(), "anything", nil)
	if err != nil {
		t.Fatalf("Call: %v", err)
	}
	if !bytes.Equal(res, []byte("ok")) {
		t.Fatalf("result = %q, want ok", res)
	}
}

// startFakeServer starts a TCP server that calls handler for each request.
// handler receives the raw request body and returns (status, payload).
// Returns the server address and a stop function.
func startFakeServer(t *testing.T, handler func(body []byte) (status uint8, payload []byte)) (addr string, stop func()) {
	t.Helper()
	ln, err := net.Listen("tcp", "127.0.0.1:0")
	if err != nil {
		t.Fatal(err)
	}
	go acceptAndRespond(ln, func(body []byte) (uint8, []byte) {
		return handler(body)
	})
	return ln.Addr().String(), func() { _ = ln.Close() }
}

// TestClientFollowsNotLeaderHint: server A returns StatusNotLeader hinting B;
// server B returns StatusOK. Client is configured with only A. Expect success.
func TestClientFollowsNotLeaderHint(t *testing.T) {
	// Start B first so its address is known before A's handler closes over it.
	addrB, stopB := startFakeServer(t, func(_ []byte) (uint8, []byte) {
		return StatusOK, []byte("ok")
	})
	defer stopB()
	// Now start A, hinting B.
	addrA, stopA := startFakeServer(t, func(_ []byte) (uint8, []byte) {
		return StatusNotLeader, encodeLeaderAddrFrame(addrB)
	})
	defer stopA()

	// Client only knows about A; B is discovered via the NotLeader hint.
	c, err := New(Config{Servers: []string{addrA}, MaxNotLeaderHops: 3})
	if err != nil {
		t.Fatal(err)
	}
	defer func() { _ = c.Close() }()

	res, err := c.Call(context.Background(), "ping", nil)
	if err != nil {
		t.Fatalf("Call: %v", err)
	}
	if !bytes.Equal(res, []byte("ok")) {
		t.Fatalf("result = %q, want ok", res)
	}
}

// TestClientBoundedByMaxNotLeaderHops: A hints B, B hints A, creating a cycle.
// MaxNotLeaderHops=2 means 1 initial + 2 retries = 3 total attempts.
// The loop exhausts and returns an error containing the hop count.
func TestClientBoundedByMaxNotLeaderHops(t *testing.T) {
	var callA, callB atomic.Int32

	// Start B first so its address is known before A's handler closes over it.
	var addrA string
	addrB, stopB := startFakeServer(t, func(_ []byte) (uint8, []byte) {
		callB.Add(1)
		return StatusNotLeader, encodeLeaderAddrFrame(addrA)
	})
	defer stopB()
	addrA, stopA := startFakeServer(t, func(_ []byte) (uint8, []byte) {
		callA.Add(1)
		return StatusNotLeader, encodeLeaderAddrFrame(addrB)
	})
	defer stopA()

	c, err := New(Config{Servers: []string{addrA}, MaxNotLeaderHops: 2})
	if err != nil {
		t.Fatal(err)
	}
	defer func() { _ = c.Close() }()

	_, err = c.Call(context.Background(), "ping", nil)
	if err == nil {
		t.Fatal("expected error after exhausting hops, got nil")
	}
	// MaxNotLeaderHops=2: loop runs hop=0,1,2 → 3 total calls (initial + 2 retries).
	total := int(callA.Load() + callB.Load())
	if total != 3 {
		t.Fatalf("total server calls = %d, want 3 (initial + 2 retries)", total)
	}
	if !strings.Contains(err.Error(), "2") {
		t.Fatalf("error %q should mention hop count 2", err.Error())
	}
}

// TestClientSurfacesEmptyLeaderHintImmediately: server returns StatusNotLeader
// with an empty addr. Call must return an error immediately, not retry.
func TestClientSurfacesEmptyLeaderHintImmediately(t *testing.T) {
	var callCount atomic.Int32
	addr, stop := startFakeServer(t, func(_ []byte) (uint8, []byte) {
		callCount.Add(1)
		return StatusNotLeader, encodeLeaderAddrFrame("") // empty hint
	})
	defer stop()

	c, err := New(Config{Servers: []string{addr}, MaxNotLeaderHops: 3})
	if err != nil {
		t.Fatal(err)
	}
	defer func() { _ = c.Close() }()

	_, err = c.Call(context.Background(), "ping", nil)
	if err == nil {
		t.Fatal("expected error for empty leader hint, got nil")
	}
	if n := callCount.Load(); n != 1 {
		t.Fatalf("expected exactly 1 call (no retry), got %d", n)
	}
}

func TestClientFallsBackWhenOpsNotConfigured(t *testing.T) {
	// Same shape as TestClientFollowsNotLeaderHint but Ops=nil.
	// Verifies smart-routing branch is dormant.
	addrB, stopB := startFakeServer(t, func(_ []byte) (uint8, []byte) {
		return StatusOK, []byte("ok")
	})
	defer stopB()
	addrA, stopA := startFakeServer(t, func(_ []byte) (uint8, []byte) {
		return StatusNotLeader, encodeLeaderAddrFrame(addrB)
	})
	defer stopA()
	c, err := New(Config{Servers: []string{addrA}, MaxNotLeaderHops: 3})
	if err != nil {
		t.Fatal(err)
	}
	defer func() { _ = c.Close() }()
	res, err := c.Call(context.Background(), "get", wire.EncodeKeyArgs([]byte("k")))
	if err != nil {
		t.Fatalf("Call: %v", err)
	}
	if string(res) != "ok" {
		t.Errorf("res = %q, want ok", res)
	}
}

func TestClientPickInitialTargetUsesTopology(t *testing.T) {
	// Manually populate the topology cache; verify pickInitialTarget
	// hashes to the right shard and returns its leader.
	reg := wire.NewRegistry()
	if err := wire.RegisterRoutableBuiltins(reg); err != nil {
		t.Fatal(err)
	}
	c, err := New(Config{
		Servers:                 []string{"127.0.0.1:1"}, // never dialed in this test
		Ops:                     reg,
		TopologyRefreshInterval: 10 * time.Second,
	})
	if err != nil {
		t.Fatal(err)
	}
	defer func() { _ = c.Close() }()

	c.topology.set(wire.Topology{
		NumShards: 4,
		Members: []wire.TopologyMember{
			{NodeID: "n1", ServerAddr: "10.0.0.1:7001"},
			{NodeID: "n2", ServerAddr: "10.0.0.2:7001"},
		},
		Leaders: []string{"10.0.0.1:7001", "10.0.0.2:7001", "10.0.0.1:7001", "10.0.0.2:7001"},
	})

	target := c.pickInitialTarget("get", wire.EncodeKeyArgs([]byte("k")))
	if target != "10.0.0.1:7001" && target != "10.0.0.2:7001" {
		t.Errorf("target = %q, want a topology leader", target)
	}
}

func TestClientPickInitialTargetFallsBackForShardlessOp(t *testing.T) {
	// __ping__ has nil KeyExtractor — pickInitialTarget must fall back
	// to firstServer regardless of topology.
	reg := wire.NewRegistry()
	if err := wire.RegisterRoutableBuiltins(reg); err != nil {
		t.Fatal(err)
	}
	c, err := New(Config{
		Servers:                 []string{"127.0.0.1:9999"},
		Ops:                     reg,
		TopologyRefreshInterval: 10 * time.Second,
	})
	if err != nil {
		t.Fatal(err)
	}
	defer func() { _ = c.Close() }()
	c.topology.set(wire.Topology{
		NumShards: 2,
		Leaders:   []string{"a:1", "b:1"},
	})
	target := c.pickInitialTarget("__ping__", nil)
	if target != "127.0.0.1:9999" {
		t.Errorf("target = %q, want firstServer fallback", target)
	}
}

func TestClientRefreshLoopFires(t *testing.T) {
	var calls atomic.Int32

	// The fake server always returns a valid topology; every request the
	// client makes to it (bootstrap + periodic ticks) increments calls.
	var addr string
	addr, stop := startFakeServer(t, func(_ []byte) (uint8, []byte) {
		calls.Add(1)
		top, _ := wire.EncodeTopology(wire.Topology{
			NumShards: 1,
			Leaders:   []string{addr},
		})
		return StatusOK, top
	})
	defer stop()

	reg := wire.NewRegistry()
	if err := wire.RegisterRoutableBuiltins(reg); err != nil {
		t.Fatal(err)
	}
	c, err := New(Config{
		Servers:                 []string{addr},
		Ops:                     reg,
		TopologyRefreshInterval: 1 * time.Second,
	})
	if err != nil {
		t.Fatal(err)
	}
	defer func() { _ = c.Close() }()

	// Wait up to 2.5s for at least 2 calls: 1 initial bootstrap + 1 tick.
	deadline := time.Now().Add(2500 * time.Millisecond)
	for time.Now().Before(deadline) {
		if calls.Load() >= 2 {
			return
		}
		time.Sleep(50 * time.Millisecond)
	}
	t.Errorf("refresh fired %d times, want >= 2", calls.Load())
}

// TestClientRespectsCanceledContext: a pre-canceled context must cause Call to
// return immediately with a context error, before any server dial is attempted
// on the second hop.
func TestClientRespectsCanceledContext(t *testing.T) {
	addr, stop := startFakeServer(t, func(_ []byte) (uint8, []byte) {
		// Points at an unreachable address; the second hop should never reach here.
		return StatusNotLeader, encodeLeaderAddrFrame("127.0.0.1:1")
	})
	defer stop()

	c, err := New(Config{Servers: []string{addr}, MaxNotLeaderHops: 5})
	if err != nil {
		t.Fatal(err)
	}
	defer func() { _ = c.Close() }()

	ctx, cancel := context.WithCancel(context.Background())
	cancel() // pre-canceled before the call

	_, err = c.Call(ctx, "get", []byte("k"))
	if err == nil {
		t.Fatal("expected canceled context error, got nil")
	}
	if !errors.Is(err, context.Canceled) {
		t.Fatalf("err = %v, want context.Canceled", err)
	}
}
