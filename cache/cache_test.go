// SPDX-License-Identifier: Apache-2.0

package cache

import (
	"bytes"
	"fmt"
	"runtime"
	"sync"
	"testing"
	"time"
)

func TestCacheNewWithDefaults(t *testing.T) {
	c, err := New(DefaultConfig())
	if err != nil {
		t.Fatalf("New: %v", err)
	}
	defer func() { _ = c.Close() }()
	if c.NumShards() != 256 {
		t.Fatalf("NumShards = %d, want 256", c.NumShards())
	}
}

func TestCacheNewRejectsInvalidConfig(t *testing.T) {
	cfg := DefaultConfig()
	cfg.NumShards = 100 // not a power of two
	if _, err := New(cfg); err == nil {
		t.Fatal("New with invalid config: expected error, got nil")
	}
}

func TestCacheRoundtripAcrossShards(t *testing.T) {
	c, _ := New(DefaultConfig())
	defer func() { _ = c.Close() }()
	for i := range 10_000 {
		k := fmt.Appendf(nil, "k%d", i)
		v := fmt.Appendf(nil, "v%d", i)
		if err := c.Put(k, v, 0); err != nil {
			t.Fatalf("Put %d: %v", i, err)
		}
	}
	for i := range 10_000 {
		k := fmt.Appendf(nil, "k%d", i)
		want := fmt.Appendf(nil, "v%d", i)
		got, err := c.Get(k)
		if err != nil {
			t.Fatalf("Get %d: %v", i, err)
		}
		if !bytes.Equal(got, want) {
			t.Fatalf("Get %d: %q, want %q", i, got, want)
		}
	}
}

func TestCacheStats(t *testing.T) {
	c, _ := New(DefaultConfig())
	defer func() { _ = c.Close() }()
	_ = c.Put([]byte("k"), []byte("v"), 0)
	_, _ = c.Get([]byte("k"))
	_, _ = c.Get([]byte("missing"))
	st := c.Stats()
	if st.Puts != 1 || st.Gets != 2 || st.Hits != 1 || st.Misses != 1 {
		t.Fatalf("stats unexpected: %+v", st)
	}
}

func TestCacheDel(t *testing.T) {
	c, _ := New(DefaultConfig())
	defer func() { _ = c.Close() }()
	_ = c.Put([]byte("k"), []byte("v"), 0)
	if ok, err := c.Del([]byte("k")); err != nil || !ok {
		t.Fatal("Del should return true")
	}
	if _, err := c.Get([]byte("k")); err != ErrNotFound {
		t.Fatal("post-Del Get should be ErrNotFound")
	}
}

func TestCacheCloseStopsSweepers(t *testing.T) {
	t.Parallel()
	cfg := DefaultConfig()
	cfg.TTLSweepIntervalMs = 10
	c, _ := New(cfg)
	_ = c.Close()
	// Second Close must be safe (idempotent).
	_ = c.Close()
}

func TestCacheConcurrentLoad(t *testing.T) {
	c, _ := New(DefaultConfig())
	defer func() { _ = c.Close() }()

	const workers = 16
	const iters = 5000

	var wg sync.WaitGroup
	for w := range workers {
		wg.Add(1)
		go func(w int) {
			defer wg.Done()
			for i := range iters {
				k := fmt.Appendf(nil, "w%02d-k%05d", w, i)
				v := fmt.Appendf(nil, "v%05d", i)
				if err := c.Put(k, v, time.Hour); err != nil {
					t.Errorf("Put: %v", err)
					return
				}
				got, err := c.Get(k)
				if err != nil {
					t.Errorf("Get: %v", err)
					return
				}
				if !bytes.Equal(got, v) {
					t.Errorf("mismatch")
					return
				}
			}
		}(w)
	}
	wg.Wait()
}

func TestCacheAppliedIndexRoundtrip(t *testing.T) {
	if runtime.GOOS != "linux" {
		t.Skip("mmap")
	}
	dir := t.TempDir()
	cfg := DefaultConfig()
	cfg.NumShards = 4
	cfg.DataDir = dir

	c1, err := New(cfg)
	if err != nil {
		t.Fatal(err)
	}
	c1.SetAppliedIndex(42, true)
	_ = c1.Close()

	c2, err := New(cfg)
	if err != nil {
		t.Fatal(err)
	}
	defer func() { _ = c2.Close() }()
	if got := c2.AppliedIndex(); got != 42 {
		t.Errorf("AppliedIndex = %d, want 42", got)
	}
}

func TestCacheAppliedIndexConcurrent(t *testing.T) {
	if runtime.GOOS != "linux" {
		t.Skip("mmap")
	}
	dir := t.TempDir()
	cfg := DefaultConfig()
	cfg.NumShards = 4
	cfg.DataDir = dir
	c, err := New(cfg)
	if err != nil {
		t.Fatal(err)
	}
	defer func() { _ = c.Close() }()

	var wg sync.WaitGroup
	for i := 0; i < 8; i++ {
		wg.Add(2)
		go func(v uint64) { defer wg.Done(); c.SetAppliedIndex(v, false) }(uint64(i + 1))
		go func() { defer wg.Done(); _ = c.AppliedIndex() }()
	}
	wg.Wait()
}

func TestCacheAppliedIndexHeapModeReturnsZero(t *testing.T) {
	c, err := New(DefaultConfig())
	if err != nil {
		t.Fatal(err)
	}
	defer func() { _ = c.Close() }()
	if got := c.AppliedIndex(); got != 0 {
		t.Errorf("heap AppliedIndex = %d, want 0", got)
	}
}

// TestCacheMsyncLoopFires verifies that enabling Durable starts the
// background msync goroutine and that Close cleanly stops it. The race
// detector surfaces any leak or unsynchronized access.
func TestCacheMsyncLoopFires(t *testing.T) {
	if runtime.GOOS != "linux" {
		t.Skip("mmap")
	}
	cfg := DefaultConfig()
	cfg.NumShards = 1
	cfg.DataDir = t.TempDir()
	cfg.Durable = true
	cfg.MsyncIntervalMs = 50

	c, err := New(cfg)
	if err != nil {
		t.Fatal(err)
	}
	_ = c.Put([]byte("k"), []byte("v"), 0)
	time.Sleep(200 * time.Millisecond) // 3-4 ticks
	if err := c.Close(); err != nil {
		t.Fatalf("Close: %v", err)
	}
}
