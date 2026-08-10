// SPDX-License-Identifier: Apache-2.0

package rostam

import (
	"bytes"
	"context"
	"errors"
	"testing"
	"time"

	"github.com/rostamlabs/rostam/ops"
)

// storeFactory creates a fresh Store and registers cleanup with t.
type storeFactory func(t *testing.T) Store

// runConformanceSuite asserts the contract every Store must satisfy.
// Both backends run through this — divergence fails loudly.
func runConformanceSuite(t *testing.T, makeStore storeFactory) {
	t.Helper()
	t.Run("PutGet", func(t *testing.T) {
		s := makeStore(t)
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
	})

	t.Run("GetMissingIsErrNotFound", func(t *testing.T) {
		s := makeStore(t)
		_, err := s.Get(context.Background(), []byte("missing"))
		if !errors.Is(err, ErrNotFound) {
			t.Fatalf("err = %v, want ErrNotFound", err)
		}
	})

	t.Run("DelReturnsBool", func(t *testing.T) {
		s := makeStore(t)
		ctx := context.Background()
		_ = s.Put(ctx, []byte("k"), []byte("v"), 0)
		existed, err := s.Del(ctx, []byte("k"))
		if err != nil {
			t.Fatalf("Del existing: %v", err)
		}
		if !existed {
			t.Fatal("Del existing: existed=false, want true")
		}
		existed, _ = s.Del(ctx, []byte("k"))
		if existed {
			t.Fatal("Del absent: existed=true, want false")
		}
	})

	t.Run("CallRoundtripsThroughBuiltins", func(t *testing.T) {
		s := makeStore(t)
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
	})

	t.Run("TTLExpires", func(t *testing.T) {
		s := makeStore(t)
		ctx := context.Background()
		if err := s.Put(ctx, []byte("k"), []byte("v"), 30*time.Millisecond); err != nil {
			t.Fatalf("Put: %v", err)
		}
		time.Sleep(80 * time.Millisecond)
		_, err := s.Get(ctx, []byte("k"))
		if !errors.Is(err, ErrNotFound) {
			t.Fatalf("post-TTL Get: err = %v, want ErrNotFound", err)
		}
	})
}

func TestEmbeddedSatisfiesStoreContract(t *testing.T) {
	runConformanceSuite(t, func(t *testing.T) Store {
		s := newSingleEmbedded(t)
		waitLeaderEmbedded(t, s)
		return s
	})
}

func TestClientSatisfiesStoreContract(t *testing.T) {
	runConformanceSuite(t, func(t *testing.T) Store {
		s := newSingleClient(t)
		// Client's first Call will block until the server's leader is up;
		// no explicit wait needed for the conformance assertions.
		return s
	})
}

func TestDirectSatisfiesStoreContract(t *testing.T) {
	runConformanceSuite(t, func(t *testing.T) Store { return newSingleDirect(t) })
}
