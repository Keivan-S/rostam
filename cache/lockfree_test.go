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

// TestConcurrentHotKeyGetPutDelEvict hammers the read path with many reader
// goroutines while a single writer does Put/Del over a small hot-key set on a
// capacity-constrained shard, under both at-cap policies. Under
// PolicyRingbufEvict eviction (in-place page overwrite) fires continuously;
// under PolicyRejectWrites the shard fills and the read path is fully lock-free.
// It asserts that every value a reader receives is well-formed for the key it
// asked for: never a torn value (mixed bytes) and never another key's value. Run
// under -race to also catch data races.
//
// The value for hot key i is 512 bytes all equal to byte(i). That encoding makes
// both failure modes observable: a torn read shows non-uniform bytes, and a
// cross-key mismatch shows a uniform run of the wrong index.
func TestConcurrentHotKeyGetPutDelEvict(t *testing.T) {
	for _, tc := range []struct {
		name   string
		policy AtCapPolicy
	}{
		{"ringbuf", PolicyRingbufEvict},
		{"reject", PolicyRejectWrites},
	} {
		t.Run(tc.name, func(t *testing.T) {
			runHotKeyStress(t, tc.policy)
		})
	}
}

func runHotKeyStress(t *testing.T, policy AtCapPolicy) {
	t.Helper()
	const (
		hotKeys  = 64 // < 256 so byte(i) uniquely tags each key's value
		valLen   = 512
		writeOps = 150_000
	)

	cfg := DefaultConfig()
	cfg.NumShards = 1
	cfg.PageSize = 1 << 20          // 1 MiB (minimum)
	cfg.MaxMemoryPerShard = 2 << 20 // 2 pages → repeated hot-key Puts fill (and, under ringbuf, evict)
	cfg.AtCapPolicy = policy
	cfg.TTLSweepIntervalMs = 5 // exercise the background sweeper concurrently
	c, err := New(cfg)
	if err != nil {
		t.Fatalf("New: %v", err)
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
	// verify reports whether v is a well-formed value for key index i: exactly
	// valLen bytes, every one equal to byte(i).
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

	for i := range hotKeys {
		if err := c.Put(keyFor(i), valFor(i), 0); err != nil {
			t.Fatalf("seed Put %d: %v", i, err)
		}
	}

	var (
		stop     atomic.Bool
		bad      atomic.Int64
		wg       sync.WaitGroup
		readers  = 16
		firstBad atomic.Value // string
	)
	record := func(desc string) {
		bad.Add(1)
		firstBad.CompareAndSwap(nil, desc)
	}

	for r := range readers {
		wg.Add(1)
		useInto := r%2 == 0
		go func(seed int64) {
			defer wg.Done()
			rng := rand.New(rand.NewSource(seed)) //nolint:gosec // test RNG
			var buf []byte
			for !stop.Load() {
				i := rng.Intn(hotKeys)
				k := keyFor(i)
				if useInto {
					out, gerr := c.GetInto(buf[:0], k)
					if gerr == nil && !verify(i, out) {
						record(fmt.Sprintf("GetInto key %d: malformed value %v", i, truncBytes(out)))
					}
				} else {
					v, gerr := c.Get(k)
					if gerr == nil && !verify(i, v) {
						record(fmt.Sprintf("Get key %d: malformed value %v", i, truncBytes(v)))
					}
				}
			}
		}(int64(r) + 1)
	}

	wg.Add(1)
	go func() {
		defer wg.Done()
		rng := rand.New(rand.NewSource(0xC0FFEE)) //nolint:gosec // test RNG
		for n := range writeOps {
			i := rng.Intn(hotKeys)
			if n%16 == 0 {
				_, _ = c.Del(keyFor(i))
			} else if err := c.Put(keyFor(i), valFor(i), 0); err != nil && err != ErrFull {
				// ErrFull is expected once a reject-writes shard fills.
				record(fmt.Sprintf("Put key %d: %v", i, err))
			}
		}
		stop.Store(true)
	}()

	done := make(chan struct{})
	go func() { wg.Wait(); close(done) }()
	select {
	case <-done:
	case <-time.After(120 * time.Second):
		stop.Store(true)
		<-done
		t.Fatal("stress test did not finish within 120s")
	}

	if n := bad.Load(); n != 0 {
		t.Fatalf("%d malformed/torn reads; first: %v", n, firstBad.Load())
	}
}

// truncBytes returns a short prefix of b for error messages.
func truncBytes(b []byte) []byte {
	if len(b) > 8 {
		return b[:8]
	}
	return b
}

// BenchmarkCacheHotKey_GetHit measures Get on a skewed (Zipfian) hot-key
// workload swept across shard counts, for both at-cap policies. The skew
// concentrates readers on a few keys — and thus a few shards — which is the read
// contention the lock-free path removes. A uniform-random key distribution would
// spread load evenly and hide the win, so the access pattern here is
// deliberately Zipfian. The reject-writes rows exercise the fully lock-free read
// path; the ringbuf rows take the shard read lock (see doc.go).
func BenchmarkCacheHotKey_GetHit(b *testing.B) {
	const keyspace = 4096
	for _, pol := range []struct {
		name   string
		policy AtCapPolicy
	}{
		{"ringbuf", PolicyRingbufEvict},
		{"reject", PolicyRejectWrites},
	} {
		for _, shards := range []int{1, 8, 64, 256} {
			b.Run(fmt.Sprintf("%s/shards=%d", pol.name, shards), func(b *testing.B) {
				cfg := DefaultConfig()
				cfg.NumShards = shards
				cfg.AtCapPolicy = pol.policy
				c, err := New(cfg)
				if err != nil {
					b.Fatal(err)
				}
				defer func() { _ = c.Close() }()
				keys := buildKeys(keyspace)
				val := benchValue()
				for _, k := range keys {
					_ = c.Put(k, val, 0)
				}
				b.ResetTimer()
				b.RunParallel(func(pb *testing.PB) {
					r := rand.New(rand.NewSource(time.Now().UnixNano())) //nolint:gosec // benchmark RNG
					z := rand.NewZipf(r, 1.2, 1, keyspace-1)
					for pb.Next() {
						k := keys[z.Uint64()]
						_, _ = c.Get(k)
					}
				})
			})
		}
	}
}
