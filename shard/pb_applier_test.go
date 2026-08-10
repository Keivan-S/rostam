// SPDX-License-Identifier: Apache-2.0

package shard

import (
	"testing"
	"time"

	"github.com/rostamlabs/rostam/ops"
)

func TestShardApplierOpErrorStillApplies(t *testing.T) {
	f, _ := newTestFSM(t) // newTestFSM returns (*fsm, *cache.Cache); the cache handle is unused here
	a := newShardApplier(f)

	// A well-formed put: applies, encodes a nil-error result, returns nil infra error.
	putEntry := EncodeLogEntry("put", ops.EncodePutArgs([]byte("k"), []byte("v"), 0))
	b, err := a.Apply(putEntry)
	if err != nil {
		t.Fatalf("put Apply infra error: %v", err)
	}
	if resp := decodePBResult(b); resp.Err != nil {
		t.Fatalf("put decoded op error: %v", resp.Err)
	}

	// An unregistered op is an INFRA failure: Apply must return a non-nil error
	// so the engine aborts before burning a seq.
	badEntry := EncodeLogEntry("no_such_op", nil)
	if _, err := a.Apply(badEntry); err == nil {
		t.Fatal("unregistered op: expected infra error from Apply, got nil")
	}
}

// TestShardApplierBusinessErrorStillApplies covers the op-level business-error
// path: "expire" on a missing key returns cache.ErrNotFound from the handler
// (via TxContext.Expire -> Get), which is not one of the infra sentinels
// (ErrOpNotRegistered / errPBApplyDecode / errPBApplyReadOnly). Apply must
// therefore return a NIL error (so the engine still assigns a seq and
// replicates), with the business error carried inside the encoded result
// bytes, recoverable via decodePBResult.
func TestShardApplierBusinessErrorStillApplies(t *testing.T) {
	f, _ := newTestFSM(t)
	a := newShardApplier(f)

	expireEntry := EncodeLogEntry("expire", ops.EncodeExpireArgs([]byte("missing-key"), time.Second))
	b, err := a.Apply(expireEntry)
	if err != nil {
		t.Fatalf("expire on missing key: Apply returned infra error, want nil: %v", err)
	}
	resp := decodePBResult(b)
	if resp.Err == nil {
		t.Fatal("expire on missing key: decoded op error is nil, want non-nil business error")
	}
}

// TestDecodePBResultGuards asserts decodePBResult fails safe (non-nil .Err,
// no panic) on malformed frames: one shorter than the 5-byte header, and one
// whose header claims more payload than is actually present.
func TestDecodePBResultGuards(t *testing.T) {
	if resp := decodePBResult([]byte{0, 0}); resp.Err == nil {
		t.Fatal("short frame (<5 bytes): decodePBResult .Err is nil, want non-nil")
	}

	// Header: hasErr=0, len=10, but only 2 payload bytes follow (truncated).
	truncated := []byte{0, 10, 0, 0, 0, 0xAA, 0xBB}
	if resp := decodePBResult(truncated); resp.Err == nil {
		t.Fatal("truncated frame: decodePBResult .Err is nil, want non-nil")
	}
}
