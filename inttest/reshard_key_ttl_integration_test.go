// SPDX-License-Identifier: Apache-2.0

package inttest

import (
	"context"
	"testing"
	"time"

	"github.com/rostamlabs/rostam"
	"github.com/rostamlabs/rostam/vector"
)

// readKeyPresent reports whether payload key k is present for id (a per-key TTL
// drops the key lazily on read once its absolute deadline passes, while the point
// itself lives on).
func readKeyPresent(t *testing.T, s rostam.Store, coll string, id uint64, k string) (pointLive bool, keyPresent bool) {
	t.Helper()
	found, _, meta, _, _, err := s.VectorGet(context.Background(), coll, id, false, true)
	if err != nil {
		t.Fatalf("VectorGet %s id=%d: %v", coll, id, err)
	}
	if !found {
		return false, false
	}
	_, ok := meta[k]
	return true, ok
}

// TestOnlineReshardPreservesKeyTTL is the per-key-TTL preservation gate for the
// ONLINE reshard path (VectorReshard → reshardCopyPass → vector_insert_if_absent
// with the ABSOLUTE keyExpires trailer). Before the fix the copy dropped the
// per-key deadlines, so a resharded key became PERMANENT. Here a point carries a
// per-key TTL set BOTH at insert time (key_ttl_ms) AND via set_payload; after an
// online reshard the keys must still expire at their ORIGINAL absolute deadlines
// (sleeping past the deadline drops them), proving the deadlines were carried
// VERBATIM rather than lost. revert-fails-it: with the copy dropping keyExpires
// the keys would be permanent and this assertion fails.
func TestOnlineReshardPreservesKeyTTL(t *testing.T) {
	defer rostam.SetReshardDrainGrace(20 * time.Millisecond)()

	s := newSingleEmbedded(t)
	waitLeaderEmbedded(t, s)
	ctx := context.Background()

	const ttlMs = 1500
	must(t, s.CreateCollection(ctx, "kt", rostam.VectorConfig{Dim: 4, Metric: vector.L2, M: 8, EfConstruction: 50, EfSearch: 64, Seed: 1, Partitions: 2}))

	// id 1: insert-time per-key TTL on 'tIns' + a permanent key 'perm'.
	must(t, s.VectorInsert(ctx, "kt", 1, []float32{1, 0, 0, 0}))
	applied, err := s.VectorSetPayload(ctx, "kt", 1, vector.Metadata{"perm": vector.NewString("keep"), "tIns": vector.NewString("x")},
		map[string]int64{"tIns": ttlMs})
	must(t, err)
	if !applied {
		t.Fatal("set_payload (insert-time-style) not applied")
	}
	// id 1 also: a second per-key TTL set via set_payload on 'tSet'.
	applied, err = s.VectorSetPayload(ctx, "kt", 1, vector.Metadata{"tSet": vector.NewString("y")}, map[string]int64{"tSet": ttlMs})
	must(t, err)
	if !applied {
		t.Fatal("set_payload 'tSet' not applied")
	}
	deadline := time.Now().Add(ttlMs * time.Millisecond)

	// Sanity: both TTL keys present before the deadline.
	if live, ok := readKeyPresent(t, s, "kt", 1, "tIns"); !live || !ok {
		t.Fatalf("pre-reshard: tIns missing (live=%v ok=%v)", live, ok)
	}
	if _, ok := readKeyPresent(t, s, "kt", 1, "tSet"); !ok {
		t.Fatal("pre-reshard: tSet missing")
	}

	// ONLINE reshard 2 -> 4 (happens well before the deadline).
	must(t, s.VectorReshard(ctx, "kt", 4))

	// Right after reshard (still before the deadline): the keys must survive the copy.
	if _, ok := readKeyPresent(t, s, "kt", 1, "tIns"); !ok {
		t.Fatal("post-reshard pre-deadline: tIns dropped by the copy (per-key TTL not carried)")
	}

	// Sleep past the ORIGINAL absolute deadline: both per-key TTLs must now be gone
	// (their carried deadlines fired); the point + permanent key survive. If the copy
	// had dropped the deadlines (pre-fix), the keys would be permanent and still present.
	time.Sleep(time.Until(deadline) + 750*time.Millisecond)
	live, okIns := readKeyPresent(t, s, "kt", 1, "tIns")
	if !live {
		t.Fatal("point 1 vanished after key expiry (only the keys should expire)")
	}
	if okIns {
		t.Fatal("time-stable violated: 'tIns' still present after its original deadline (reshard reset/dropped the per-key TTL)")
	}
	if _, okSet := readKeyPresent(t, s, "kt", 1, "tSet"); okSet {
		t.Fatal("time-stable violated: 'tSet' still present after its original deadline")
	}
	if _, okPerm := readKeyPresent(t, s, "kt", 1, "perm"); !okPerm {
		t.Fatal("permanent key 'perm' lost after reshard")
	}
}

// TestOfflineResplitPreservesKeyTTL mirrors TestOnlineReshardPreservesKeyTTL for
// the OFFLINE path (VectorResplit → vector_insert with the ABSOLUTE keyExpires
// trailer → RestoreInsert verbatim).
func TestOfflineResplitPreservesKeyTTL(t *testing.T) {
	s := newSingleEmbedded(t)
	waitLeaderEmbedded(t, s)
	ctx := context.Background()

	const ttlMs = 1500
	must(t, s.CreateCollection(ctx, "ktr", rostam.VectorConfig{Dim: 4, Metric: vector.L2, M: 8, EfConstruction: 50, EfSearch: 64, Seed: 1, Partitions: 2}))

	must(t, s.VectorInsert(ctx, "ktr", 1, []float32{1, 0, 0, 0}))
	applied, err := s.VectorSetPayload(ctx, "ktr", 1, vector.Metadata{"perm": vector.NewString("keep"), "tSet": vector.NewString("y")},
		map[string]int64{"tSet": ttlMs})
	must(t, err)
	if !applied {
		t.Fatal("set_payload not applied")
	}
	deadline := time.Now().Add(ttlMs * time.Millisecond)

	if _, ok := readKeyPresent(t, s, "ktr", 1, "tSet"); !ok {
		t.Fatal("pre-resplit: tSet missing")
	}

	// OFFLINE resplit 2 -> 4.
	must(t, s.VectorResplit(ctx, "ktr", 4))

	if _, ok := readKeyPresent(t, s, "ktr", 1, "tSet"); !ok {
		t.Fatal("post-resplit pre-deadline: tSet dropped by the copy (per-key TTL not carried)")
	}

	time.Sleep(time.Until(deadline) + 750*time.Millisecond)
	live, okSet := readKeyPresent(t, s, "ktr", 1, "tSet")
	if !live {
		t.Fatal("point 1 vanished after key expiry")
	}
	if okSet {
		t.Fatal("time-stable violated: 'tSet' still present after its original deadline (resplit reset/dropped the per-key TTL)")
	}
	if _, okPerm := readKeyPresent(t, s, "ktr", 1, "perm"); !okPerm {
		t.Fatal("permanent key 'perm' lost after resplit")
	}
}
