// SPDX-License-Identifier: Apache-2.0

package ops

import (
	"bytes"
	"testing"
	"time"

	"github.com/rostamlabs/rostam/cache"
)

func newTestTxContext(t *testing.T) *TxContext {
	t.Helper()
	cfg := cache.DefaultConfig()
	cfg.NumShards = 1
	c, err := cache.New(cfg)
	if err != nil {
		t.Fatalf("cache.New: %v", err)
	}
	t.Cleanup(func() { _ = c.Close() })
	return NewTxContext(c)
}

func TestTxContextPutGet(t *testing.T) {
	tx := newTestTxContext(t)
	if err := tx.Put([]byte("k"), []byte("v"), 0); err != nil {
		t.Fatalf("Put: %v", err)
	}
	got, err := tx.Get([]byte("k"))
	if err != nil {
		t.Fatalf("Get: %v", err)
	}
	if !bytes.Equal(got, []byte("v")) {
		t.Fatalf("Get = %q, want v", got)
	}
}

func TestTxContextGetMissing(t *testing.T) {
	tx := newTestTxContext(t)
	_, err := tx.Get([]byte("missing"))
	if err != cache.ErrNotFound {
		t.Fatalf("Get missing: err = %v, want ErrNotFound", err)
	}
}

func TestTxContextDel(t *testing.T) {
	tx := newTestTxContext(t)
	_ = tx.Put([]byte("k"), []byte("v"), 0)
	if ok, err := tx.Del([]byte("k")); err != nil || !ok {
		t.Fatal("Del: expected true on existing key")
	}
	if ok, err := tx.Del([]byte("k")); err != nil || ok {
		t.Fatal("Del: expected false on missing key")
	}
}

func TestTxContextExpire(t *testing.T) {
	tx := newTestTxContext(t)
	_ = tx.Put([]byte("k"), []byte("v"), 0)
	if err := tx.Expire([]byte("k"), 10*time.Millisecond); err != nil {
		t.Fatalf("Expire: %v", err)
	}
	time.Sleep(30 * time.Millisecond)
	if _, err := tx.Get([]byte("k")); err != cache.ErrNotFound {
		t.Fatalf("after Expire+sleep: err = %v, want ErrNotFound", err)
	}
}

func TestTxContextExpireMissing(t *testing.T) {
	tx := newTestTxContext(t)
	if err := tx.Expire([]byte("missing"), time.Second); err != cache.ErrNotFound {
		t.Fatalf("Expire missing: err = %v, want ErrNotFound", err)
	}
}
