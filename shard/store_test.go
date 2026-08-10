// SPDX-License-Identifier: Apache-2.0

package shard

import (
	"bytes"
	"errors"
	"path/filepath"
	"runtime"
	"testing"
	"time"

	"github.com/rostamlabs/rostam/ops"
)

func newSingleNodeStore(t *testing.T) *Store {
	t.Helper()
	reg := ops.NewRegistry()
	if err := ops.RegisterBuiltins(reg); err != nil {
		t.Fatal(err)
	}
	cfg := DefaultConfig(t.TempDir(), "node1", reg)
	cfg.Bootstrap = true
	cfg.RaftHeartbeatMs = 50
	cfg.RaftElectionMs = 100
	cfg.NoSync = true
	s, err := New(cfg)
	if err != nil {
		t.Fatalf("shard.New: %v", err)
	}
	t.Cleanup(func() { _ = s.Close() })
	waitLeader(t, s)
	return s
}

func waitLeader(t *testing.T, s *Store) {
	t.Helper()
	deadline := time.Now().Add(3 * time.Second)
	for time.Now().Before(deadline) {
		if s.IsLeader() {
			return
		}
		time.Sleep(20 * time.Millisecond)
	}
	t.Fatalf("node never became leader")
}

func TestStorePutGet(t *testing.T) {
	s := newSingleNodeStore(t)
	if err := s.Put([]byte("k"), []byte("v"), 0); err != nil {
		t.Fatalf("Put: %v", err)
	}
	got, err := s.Get([]byte("k"))
	if err != nil {
		t.Fatalf("Get: %v", err)
	}
	if !bytes.Equal(got, []byte("v")) {
		t.Fatalf("Get = %q, want v", got)
	}
}

func TestStoreDel(t *testing.T) {
	s := newSingleNodeStore(t)
	_ = s.Put([]byte("k"), []byte("v"), 0)
	if del, err := s.Del([]byte("k")); err != nil || !del {
		t.Fatalf("Del = (%v, %v), want (true, nil)", del, err)
	}
	if _, err := s.Get([]byte("k")); err == nil {
		t.Fatal("post-Del Get must error")
	}
}

func TestStoreExpire(t *testing.T) {
	s := newSingleNodeStore(t)
	_ = s.Put([]byte("k"), []byte("v"), 0)
	if err := s.Expire([]byte("k"), 10*time.Millisecond); err != nil {
		t.Fatalf("Expire: %v", err)
	}
	time.Sleep(40 * time.Millisecond)
	if _, err := s.Get([]byte("k")); err == nil {
		t.Fatal("post-Expire Get must error")
	}
}

func TestStoreCallReadOnly(t *testing.T) {
	s := newSingleNodeStore(t)
	_ = s.Put([]byte("k"), []byte("v"), 0)
	res, err := s.Call("get", ops.EncodeKeyArgs([]byte("k")))
	if err != nil {
		t.Fatalf("Call get: %v", err)
	}
	if !bytes.Equal(res, []byte("v")) {
		t.Fatalf("Call get = %q, want v", res)
	}
}

func TestStoreCallReadWrite(t *testing.T) {
	s := newSingleNodeStore(t)
	res, err := s.Call("incr", ops.EncodeIncrArgs([]byte("counter"), 7))
	if err != nil {
		t.Fatalf("Call incr: %v", err)
	}
	v, err := ops.DecodeIncrResult(res)
	if err != nil {
		t.Fatal(err)
	}
	if v != 7 {
		t.Fatalf("incr = %d, want 7", v)
	}
}

func TestStoreCallUnknownOp(t *testing.T) {
	s := newSingleNodeStore(t)
	_, err := s.Call("doesnotexist", nil)
	if !errors.Is(err, ErrOpNotRegistered) {
		t.Fatalf("Call unknown: err = %v, want ErrOpNotRegistered", err)
	}
}

func TestStoreLeaderAddrWhenLeader(t *testing.T) {
	s := newSingleNodeStore(t)
	if addr := s.LeaderAddr(); addr == "" {
		t.Fatal("LeaderAddr empty even though we're leader")
	}
}

func TestStoreWarmRestartPreservesData(t *testing.T) {
	if runtime.GOOS != "linux" {
		t.Skip("mmap only on linux")
	}
	dir := t.TempDir()
	reg := ops.NewRegistry()
	if err := ops.RegisterBuiltins(reg); err != nil {
		t.Fatal(err)
	}
	cfg := DefaultConfig(dir, "node1", reg)
	cfg.Bootstrap = true
	cfg.RaftHeartbeatMs = 50
	cfg.RaftElectionMs = 100
	cfg.NoSync = true
	cfg.Cache.DataDir = filepath.Join(dir, "cache")
	cfg.Cache.Durable = true

	s1, err := New(cfg)
	if err != nil {
		t.Fatalf("first New: %v", err)
	}
	waitLeader(t, s1)
	if err := s1.Put([]byte("k"), []byte("v"), 0); err != nil {
		t.Fatalf("Put: %v", err)
	}
	appliedBefore := s1.AppliedIndex()
	if appliedBefore == 0 {
		t.Fatalf("AppliedIndex should be > 0 after a Put, got 0")
	}
	if err := s1.Close(); err != nil {
		t.Fatalf("first Close: %v", err)
	}

	// Reopen with the same config and verify the data is still there.
	cfg.Bootstrap = false
	s2, err := New(cfg)
	if err != nil {
		t.Fatalf("second New: %v", err)
	}
	defer func() { _ = s2.Close() }()
	waitLeader(t, s2)
	got, err := s2.Get([]byte("k"))
	if err != nil {
		t.Fatalf("post-restart Get: %v", err)
	}
	if !bytes.Equal(got, []byte("v")) {
		t.Fatalf("post-restart Get = %q, want v", got)
	}
	if got := s2.AppliedIndex(); got < appliedBefore {
		t.Errorf("AppliedIndex regressed: before=%d after=%d", appliedBefore, got)
	}
}
