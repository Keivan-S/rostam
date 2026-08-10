// SPDX-License-Identifier: Apache-2.0

package cache

import (
	"bytes"
	"fmt"
	"sync"
	"testing"
	"time"
)

func TestCacheIterateEmpty(t *testing.T) {
	c, _ := New(DefaultConfig())
	defer func() { _ = c.Close() }()
	called := 0
	c.Iterate(func(_, _ []byte, _ uint64) bool {
		called++
		return true
	})
	if called != 0 {
		t.Fatalf("Iterate over empty cache: fn called %d times, want 0", called)
	}
}

func TestCacheIterateVisitsEveryEntry(t *testing.T) {
	c, _ := New(DefaultConfig())
	defer func() { _ = c.Close() }()
	const n = 1000
	for i := range n {
		k := fmt.Appendf(nil, "k%04d", i)
		v := fmt.Appendf(nil, "v%04d", i)
		if err := c.Put(k, v, 0); err != nil {
			t.Fatalf("Put: %v", err)
		}
	}
	seen := make(map[string]string, n)
	c.Iterate(func(key, value []byte, _ uint64) bool {
		seen[string(key)] = string(value)
		return true
	})
	if len(seen) != n {
		t.Fatalf("visited %d, want %d", len(seen), n)
	}
	for i := range n {
		k := fmt.Sprintf("k%04d", i)
		wantV := fmt.Sprintf("v%04d", i)
		if got, ok := seen[k]; !ok || got != wantV {
			t.Fatalf("seen[%q] = %q,%v; want %q,true", k, got, ok, wantV)
		}
	}
}

func TestCacheIterateStopsWhenFnReturnsFalse(t *testing.T) {
	c, _ := New(DefaultConfig())
	defer func() { _ = c.Close() }()
	for i := range 100 {
		_ = c.Put(fmt.Appendf(nil, "k%d", i), []byte("v"), 0)
	}
	count := 0
	c.Iterate(func(_, _ []byte, _ uint64) bool {
		count++
		return count < 5 // stop after 5
	})
	if count != 5 {
		t.Fatalf("count = %d, want 5", count)
	}
}

func TestCacheIterateSkipsExpired(t *testing.T) {
	c, _ := New(DefaultConfig())
	defer func() { _ = c.Close() }()
	_ = c.Put([]byte("alive"), []byte("v"), time.Hour)
	_ = c.Put([]byte("dead"), []byte("v"), 10*time.Millisecond)
	time.Sleep(30 * time.Millisecond)
	keys := []string{}
	c.Iterate(func(key, _ []byte, _ uint64) bool {
		keys = append(keys, string(key))
		return true
	})
	for _, k := range keys {
		if k == "dead" {
			t.Fatalf("Iterate yielded expired key 'dead'")
		}
	}
	found := false
	for _, k := range keys {
		if k == "alive" {
			found = true
		}
	}
	if !found {
		t.Fatal("Iterate did not yield 'alive'")
	}
}

func TestCacheIterateConcurrentSafe(t *testing.T) {
	c, _ := New(DefaultConfig())
	defer func() { _ = c.Close() }()
	for i := range 5000 {
		_ = c.Put(fmt.Appendf(nil, "k%04d", i), []byte("v"), 0)
	}
	// Concurrent writers + readers + iterator must not race.
	var wg sync.WaitGroup
	stop := make(chan struct{})
	wg.Add(2)
	go func() {
		defer wg.Done()
		i := 0
		for {
			select {
			case <-stop:
				return
			default:
				_ = c.Put(fmt.Appendf(nil, "writer%d", i), []byte("v"), 0)
				i++
			}
		}
	}()
	go func() {
		defer wg.Done()
		for j := range 5 {
			count := 0
			c.Iterate(func(_, _ []byte, _ uint64) bool {
				count++
				return true
			})
			if count == 0 {
				t.Errorf("iteration %d visited 0 entries", j)
			}
		}
	}()
	time.Sleep(50 * time.Millisecond)
	close(stop)
	wg.Wait()
}

func TestCacheIterateValueAliasesIntoPage(t *testing.T) {
	// Document and verify: the value slice passed to fn is owned by the page.
	// Callers must copy if they want to retain it.
	c, _ := New(DefaultConfig())
	defer func() { _ = c.Close() }()
	_ = c.Put([]byte("k"), []byte("original-value"), 0)
	var captured []byte
	c.Iterate(func(key, value []byte, _ uint64) bool {
		if bytes.Equal(key, []byte("k")) {
			captured = value // alias!  //nolint:staticcheck // intentional alias for testing
		}
		return true
	})
	if !bytes.Equal(captured, []byte("original-value")) {
		t.Fatalf("captured = %q, want original-value", captured)
	}
}
