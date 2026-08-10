// SPDX-License-Identifier: Apache-2.0

package cache

import (
	"fmt"
	"math/rand"
	"sync"
	"sync/atomic"
	"testing"
	"time"
)

// TestRetireHeavyRace is the permanent regression test for the lock-free
// heap-mode PolicyRingbufEvict read path (v2). It forces very frequent heap-mode
// ringbuf retires: values are large enough that each 1 MiB page holds only a
// handful of entries, so pages drain (and retire) constantly while many
// lock-free readers hammer stale hot keys.
//
// It is DELIBERATELY finer-grained than TestConcurrentHotKeyGetPutDelEvict,
// whose 512 B values pack ~2000 entries per page and retire far too rarely to
// expose the retire/read race. If retirePageLocked's page swap ever raced a
// lock-free reader (the exact defect a naive in-place-Reset v2 shipped), `go
// test -race` flags the byte-level overwrite fast, and any torn value fails the
// per-read verify below. It MUST pass `go test -race -count=3`.
func TestRetireHeavyRace(t *testing.T) {
	const (
		hotKeys = 32
		valLen  = 180_000 // ~5 entries per 1 MiB page -> drains every ~5 puts
	)
	cfg := DefaultConfig()
	cfg.NumShards = 1
	cfg.PageSize = 1 << 20
	cfg.MaxMemoryPerShard = 2 << 20 // 2 pages
	cfg.AtCapPolicy = PolicyRingbufEvict
	cfg.TTLSweepIntervalMs = 0
	c, err := New(cfg)
	if err != nil {
		t.Fatal(err)
	}
	defer func() { _ = c.Close() }()

	keyFor := func(i int) []byte { return fmt.Appendf(nil, "hot-%04d", i) }
	valFor := func(i int) []byte {
		v := make([]byte, valLen)
		for j := range v {
			v[j] = byte(i)
		}
		return v
	}
	verify := func(i int, v []byte) bool {
		if len(v) != valLen {
			return false
		}
		want := byte(i)
		for _, b := range v {
			if b != want {
				return false
			}
		}
		return true
	}
	for i := 0; i < hotKeys; i++ {
		_ = c.Put(keyFor(i), valFor(i), 0)
	}

	var (
		stop atomic.Bool
		bad  atomic.Int64
		wg   sync.WaitGroup
	)
	for r := 0; r < 24; r++ {
		wg.Add(1)
		go func(seed int64) {
			defer wg.Done()
			rng := rand.New(rand.NewSource(seed)) //nolint:gosec // deterministic per-reader RNG; not security-sensitive
			var buf []byte
			for !stop.Load() {
				i := rng.Intn(hotKeys)
				if seed%2 == 0 {
					out, gerr := c.GetInto(buf[:0], keyFor(i))
					if gerr == nil && !verify(i, out) {
						bad.Add(1)
					}
				} else {
					v, gerr := c.Get(keyFor(i))
					if gerr == nil && !verify(i, v) {
						bad.Add(1)
					}
				}
			}
		}(int64(r) + 1)
	}
	wg.Add(1)
	go func() {
		defer wg.Done()
		rng := rand.New(rand.NewSource(0xBEEF)) //nolint:gosec // deterministic writer RNG; not security-sensitive
		for n := 0; n < 4_000_000 && !stop.Load(); n++ {
			i := rng.Intn(hotKeys)
			_ = c.Put(keyFor(i), valFor(i), 0)
		}
		stop.Store(true)
	}()
	done := make(chan struct{})
	go func() { wg.Wait(); close(done) }()
	select {
	case <-done:
	case <-time.After(60 * time.Second):
		stop.Store(true)
		<-done
	}
	if n := bad.Load(); n != 0 {
		t.Fatalf("%d malformed/torn reads", n)
	}
}
