// SPDX-License-Identifier: Apache-2.0

package client

import (
	"bufio"
	"context"
	"encoding/binary"
	"io"
	"net"
	"sync"
	"testing"
	"time"
)

// startEchoServer accepts TCP connections and responds to each request
// with StatusOK + the same args (so __ping__ etc. round-trip cleanly).
func startEchoServer(t *testing.T) (string, func()) {
	t.Helper()
	ln, err := net.Listen("tcp", "127.0.0.1:0")
	if err != nil {
		t.Fatal(err)
	}
	stop := make(chan struct{})
	var wg sync.WaitGroup
	wg.Add(1)
	go func() {
		defer wg.Done()
		for {
			conn, err := ln.Accept()
			if err != nil {
				return
			}
			go handleEchoConn(conn, stop)
		}
	}()
	return ln.Addr().String(), func() {
		close(stop)
		_ = ln.Close()
		wg.Wait()
	}
}

func handleEchoConn(c net.Conn, stop chan struct{}) {
	defer func() { _ = c.Close() }()
	r := bufio.NewReader(c)
	w := bufio.NewWriter(c)
	for {
		select {
		case <-stop:
			return
		default:
		}
		var hdr [4]byte
		if _, err := io.ReadFull(r, hdr[:]); err != nil {
			return
		}
		n := binary.BigEndian.Uint32(hdr[:])
		body := make([]byte, n)
		if _, err := io.ReadFull(r, body); err != nil {
			return
		}
		// Parse the request to extract args (we echo them back as the result).
		// Layout: [opNameLen u8][opName][argsLen u32][args].
		opNameLen := int(body[0])
		argsLenOff := 1 + opNameLen
		argsLen := int(binary.BigEndian.Uint32(body[argsLenOff : argsLenOff+4]))
		args := body[argsLenOff+4 : argsLenOff+4+argsLen]
		// Build response: [status][payloadLen][payload].
		resp := make([]byte, 1+4+len(args))
		resp[0] = StatusOK
		binary.BigEndian.PutUint32(resp[1:5], uint32(len(args))) //nolint:gosec // test-only
		copy(resp[5:], args)
		var respHdr [4]byte
		binary.BigEndian.PutUint32(respHdr[:], uint32(len(resp))) //nolint:gosec // test-only
		if _, err := w.Write(respHdr[:]); err != nil {
			return
		}
		if _, err := w.Write(resp); err != nil {
			return
		}
		if err := w.Flush(); err != nil {
			return
		}
	}
}

func TestPoolAcquireRelease(t *testing.T) {
	addr, stop := startEchoServer(t)
	defer stop()
	cfg := Config{Servers: []string{addr}}
	cfg.applyDefaults()
	p, err := newPerServerPool(addr, &cfg)
	if err != nil {
		t.Fatal(err)
	}
	defer p.close()

	res, err := p.acquire(context.Background())
	if err != nil {
		t.Fatalf("acquire: %v", err)
	}
	p.release(res)

	stat := p.stat()
	if stat.NewConnsCount != 1 {
		t.Fatalf("NewConnsCount = %d, want 1", stat.NewConnsCount)
	}
}

func TestPoolMaxConnsEnforced(t *testing.T) {
	addr, stop := startEchoServer(t)
	defer stop()
	cfg := Config{Servers: []string{addr}, MaxConnsPerServer: 2}
	cfg.applyDefaults()
	p, err := newPerServerPool(addr, &cfg)
	if err != nil {
		t.Fatal(err)
	}
	defer p.close()

	r1, _ := p.acquire(context.Background())
	r2, _ := p.acquire(context.Background())
	defer p.release(r1)
	defer p.release(r2)

	// Third acquire should block until ctx times out.
	ctx, cancel := context.WithTimeout(context.Background(), 100*time.Millisecond)
	defer cancel()
	if _, err := p.acquire(ctx); err == nil {
		t.Fatal("3rd acquire returned without ctx timeout; want error")
	}
}

func TestPoolMinConnsWarmup(t *testing.T) {
	addr, stop := startEchoServer(t)
	defer stop()
	cfg := Config{Servers: []string{addr}, MinConnsPerServer: 2}
	cfg.applyDefaults()
	p, err := newPerServerPool(addr, &cfg)
	if err != nil {
		t.Fatal(err)
	}
	defer p.close()

	deadline := time.Now().Add(2 * time.Second)
	for time.Now().Before(deadline) {
		if p.pool.Stat().TotalResources() >= 2 {
			return
		}
		time.Sleep(20 * time.Millisecond)
	}
	t.Fatalf("warmup never reached 2 conns: stat=%+v", p.stat())
}

func TestPoolPoisonedConnDestroyed(t *testing.T) {
	addr, stop := startEchoServer(t)
	defer stop()
	cfg := Config{Servers: []string{addr}}
	cfg.applyDefaults()
	p, err := newPerServerPool(addr, &cfg)
	if err != nil {
		t.Fatal(err)
	}
	defer p.close()

	res, _ := p.acquire(context.Background())
	res.Value().poisoned = true
	p.release(res)

	// Re-acquire should produce a fresh conn (new connect count).
	stat1 := p.stat()
	res2, _ := p.acquire(context.Background())
	defer p.release(res2)
	stat2 := p.stat()
	if stat2.NewConnsCount <= stat1.NewConnsCount {
		t.Fatalf("NewConnsCount did not grow: %d -> %d", stat1.NewConnsCount, stat2.NewConnsCount)
	}
}

func TestPoolHookBeforeAcquireVetoes(t *testing.T) {
	addr, stop := startEchoServer(t)
	defer stop()
	veto := false
	cfg := Config{
		Servers: []string{addr},
		BeforeAcquire: func(_ context.Context, _ *Conn) bool {
			return !veto
		},
	}
	cfg.applyDefaults()
	p, err := newPerServerPool(addr, &cfg)
	if err != nil {
		t.Fatal(err)
	}
	defer p.close()

	// Normal acquire succeeds.
	r, err := p.acquire(context.Background())
	if err != nil {
		t.Fatalf("first acquire: %v", err)
	}
	p.release(r)
	before := p.stat().NewConnsCount

	// Toggle veto: each acquire forces a destroy + new dial, then succeeds
	// (after a finite number of loops because of MaxConns and timing).
	veto = true
	ctx, cancel := context.WithTimeout(context.Background(), 200*time.Millisecond)
	defer cancel()
	_, err = p.acquire(ctx)
	if err == nil {
		t.Fatal("acquire with always-vetoing BeforeAcquire did not error")
	}
	after := p.stat().NewConnsCount
	if after <= before {
		t.Fatalf("NewConnsCount did not grow: %d -> %d", before, after)
	}
}
