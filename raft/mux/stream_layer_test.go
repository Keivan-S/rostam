// SPDX-License-Identifier: Apache-2.0

package mux

import (
	"io"
	"net"
	"sync"
	"testing"
	"time"

	"github.com/hashicorp/raft"
)

// TestStreamLayerLoopbackTwoGroups: two mux instances exchange traffic
// on two distinct group IDs to verify no cross-group leakage.
func TestStreamLayerLoopbackTwoGroups(t *testing.T) {
	a, err := New("127.0.0.1:0", []uint32{1, 2}, nil, nil, nil)
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() { _ = a.Close() })

	b, err := New("127.0.0.1:0", []uint32{1, 2}, nil, nil, nil)
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() { _ = b.Close() })

	aGroup1 := a.For(1)
	aGroup2 := a.For(2)
	bGroup1 := b.For(1)
	bGroup2 := b.For(2)

	var wg sync.WaitGroup

	expect := func(t *testing.T, want string, ln raft.StreamLayer) {
		t.Helper()
		wg.Add(1)
		go func() {
			defer wg.Done()
			conn, err := ln.Accept()
			if err != nil {
				t.Errorf("accept: %v", err)
				return
			}
			defer conn.Close() //nolint:errcheck
			buf := make([]byte, len(want))
			if _, err := io.ReadFull(conn, buf); err != nil {
				t.Errorf("read: %v", err)
				return
			}
			if string(buf) != want {
				t.Errorf("got %q, want %q", buf, want)
			}
		}()
	}

	send := func(t *testing.T, msg string, sl raft.StreamLayer, target string) {
		t.Helper()
		wg.Add(1)
		go func() {
			defer wg.Done()
			c, err := sl.Dial(raft.ServerAddress(target), 2*time.Second)
			if err != nil {
				t.Errorf("dial: %v", err)
				return
			}
			defer c.Close() //nolint:errcheck
			if _, err := c.Write([]byte(msg)); err != nil {
				t.Errorf("write: %v", err)
			}
		}()
	}

	expect(t, "hello-1", aGroup1)
	expect(t, "hello-2", aGroup2)
	send(t, "hello-1", bGroup1, a.Addr().String())
	send(t, "hello-2", bGroup2, a.Addr().String())

	doneCh := make(chan struct{})
	go func() {
		wg.Wait()
		close(doneCh)
	}()
	select {
	case <-doneCh:
	case <-time.After(5 * time.Second):
		t.Fatal("timed out waiting for round trips")
	}
}

func TestStreamLayerRejectsUnknownGroup(t *testing.T) {
	a, err := New("127.0.0.1:0", []uint32{1}, nil, nil, nil)
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() { _ = a.Close() })

	raw, err := net.Dial("tcp", a.Addr().String())
	if err != nil {
		t.Fatal(err)
	}
	defer raw.Close() //nolint:errcheck
	if err := writeGroupID(raw, 0xBEEF); err != nil {
		t.Fatal(err)
	}
	_ = raw.SetReadDeadline(time.Now().Add(2 * time.Second))
	buf := make([]byte, 1)
	_, err = raw.Read(buf)
	if err == nil {
		t.Error("expected error reading from rejected conn")
	}
}

func TestStreamLayerCloseIdempotent(t *testing.T) {
	s, err := New("127.0.0.1:0", []uint32{1}, nil, nil, nil)
	if err != nil {
		t.Fatal(err)
	}
	if err := s.Close(); err != nil {
		t.Errorf("first Close: %v", err)
	}
	if err := s.Close(); err != nil {
		t.Errorf("second Close: %v", err)
	}
}

func TestForPanicsOnUnknownGroup(t *testing.T) {
	s, err := New("127.0.0.1:0", []uint32{1}, nil, nil, nil)
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() { _ = s.Close() })
	defer func() {
		if r := recover(); r == nil {
			t.Error("expected panic for unknown group")
		}
	}()
	s.For(999)
}
