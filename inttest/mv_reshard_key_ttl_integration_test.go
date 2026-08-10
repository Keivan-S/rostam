// SPDX-License-Identifier: Apache-2.0

package inttest

import (
	"context"
	"testing"
	"time"

	"github.com/rostamlabs/rostam"
	"github.com/rostamlabs/rostam/vector"
)

// readMVKeyPresent reports whether payload key k is present for docID (a per-key
// TTL drops the key lazily on read once its absolute deadline passes, while the
// document itself lives on). The MV analog of readKeyPresent.
func readMVKeyPresent(t *testing.T, s rostam.Store, coll string, id uint64, k string) (docLive bool, keyPresent bool) {
	t.Helper()
	found, _, payload, err := s.VectorMVGet(context.Background(), coll, id, false, true)
	if err != nil {
		t.Fatalf("VectorMVGet %s id=%d: %v", coll, id, err)
	}
	if !found {
		return false, false
	}
	_, ok := payload[k]
	return true, ok
}

// TestMVOnlineReshardPreservesKeyTTL is the per-key-TTL preservation gate for the
// ONLINE MV reshard path (VectorMVReshard → mvReshardCopyPass →
// vector_mv_add_if_absent with the ABSOLUTE keyExpires trailer). Before the fix the
// copy dropped the per-key deadlines, so a resharded key became PERMANENT. Here a
// document carries a per-key TTL set BOTH at add time (WriteOpts.KeyTTLMs) AND via
// set_payload; after an online MV reshard the keys must still expire at their
// ORIGINAL absolute deadlines, proving the deadlines were carried VERBATIM rather
// than lost. revert-fails-it: with the copy dropping keyExpires the keys would be
// permanent and this assertion fails.
func TestMVOnlineReshardPreservesKeyTTL(t *testing.T) {
	defer rostam.SetReshardDrainGrace(20 * time.Millisecond)()

	s := newSingleEmbedded(t)
	waitLeaderEmbedded(t, s)
	ctx := context.Background()

	const ttlMs = 1500
	must(t, s.VectorMVCreateCollection(ctx, "mvkt", rostam.MultiVectorConfig{Dim: 4, M: 8, EfConstruction: 50, EfSearch: 64, Seed: 1, Partitions: 2}))

	// doc 1: add-time per-key TTL on 'tIns' + a permanent key 'perm'.
	must(t, s.VectorMVAdd(ctx, "mvkt", 1, [][]float32{{1, 0, 0, 0}},
		vector.Metadata{"perm": vector.NewString("keep"), "tIns": vector.NewString("x")},
		rostam.WriteOpts{KeyTTLMs: map[string]int64{"tIns": ttlMs}}))
	// doc 1 also: a second per-key TTL set via set_payload on 'tSet'.
	applied, err := s.VectorMVSetPayload(ctx, "mvkt", 1, vector.Metadata{"tSet": vector.NewString("y")}, map[string]int64{"tSet": ttlMs})
	must(t, err)
	if !applied {
		t.Fatal("set_payload 'tSet' not applied")
	}
	deadline := time.Now().Add(ttlMs * time.Millisecond)

	// Sanity: both TTL keys present before the deadline.
	if live, ok := readMVKeyPresent(t, s, "mvkt", 1, "tIns"); !live || !ok {
		t.Fatalf("pre-reshard: tIns missing (live=%v ok=%v)", live, ok)
	}
	if _, ok := readMVKeyPresent(t, s, "mvkt", 1, "tSet"); !ok {
		t.Fatal("pre-reshard: tSet missing")
	}

	// ONLINE MV reshard 2 -> 4 (happens well before the deadline).
	must(t, s.VectorMVReshard(ctx, "mvkt", 4))

	// Right after reshard (still before the deadline): the keys must survive the copy.
	if _, ok := readMVKeyPresent(t, s, "mvkt", 1, "tIns"); !ok {
		t.Fatal("post-reshard pre-deadline: tIns dropped by the copy (per-key TTL not carried)")
	}

	// Sleep past the ORIGINAL absolute deadline: both per-key TTLs must now be gone
	// (their carried deadlines fired); the doc + permanent key survive. If the copy
	// had dropped the deadlines (pre-fix), the keys would be permanent and present.
	time.Sleep(time.Until(deadline) + 750*time.Millisecond)
	live, okIns := readMVKeyPresent(t, s, "mvkt", 1, "tIns")
	if !live {
		t.Fatal("doc 1 vanished after key expiry (only the keys should expire)")
	}
	if okIns {
		t.Fatal("time-stable violated: 'tIns' still present after its original deadline (reshard reset/dropped the per-key TTL)")
	}
	if _, okSet := readMVKeyPresent(t, s, "mvkt", 1, "tSet"); okSet {
		t.Fatal("time-stable violated: 'tSet' still present after its original deadline")
	}
	if _, okPerm := readMVKeyPresent(t, s, "mvkt", 1, "perm"); !okPerm {
		t.Fatal("permanent key 'perm' lost after reshard")
	}
}

// TestMVOfflineResplitPreservesKeyTTL mirrors TestMVOnlineReshardPreservesKeyTTL
// for the OFFLINE MV path (VectorMVResplit → vector_mv_add_versioned with the
// ABSOLUTE keyExpires trailer → MultiRestoreAdd → restoreAdd verbatim).
func TestMVOfflineResplitPreservesKeyTTL(t *testing.T) {
	s := newSingleEmbedded(t)
	waitLeaderEmbedded(t, s)
	ctx := context.Background()

	const ttlMs = 1500
	must(t, s.VectorMVCreateCollection(ctx, "mvktr", rostam.MultiVectorConfig{Dim: 4, M: 8, EfConstruction: 50, EfSearch: 64, Seed: 1, Partitions: 2}))

	must(t, s.VectorMVAdd(ctx, "mvktr", 1, [][]float32{{1, 0, 0, 0}}, vector.Metadata{"perm": vector.NewString("keep")}, rostam.WriteOpts{}))
	applied, err := s.VectorMVSetPayload(ctx, "mvktr", 1, vector.Metadata{"tSet": vector.NewString("y")}, map[string]int64{"tSet": ttlMs})
	must(t, err)
	if !applied {
		t.Fatal("set_payload not applied")
	}
	deadline := time.Now().Add(ttlMs * time.Millisecond)

	if _, ok := readMVKeyPresent(t, s, "mvktr", 1, "tSet"); !ok {
		t.Fatal("pre-resplit: tSet missing")
	}

	// OFFLINE MV resplit 2 -> 4.
	must(t, s.VectorMVResplit(ctx, "mvktr", 4))

	if _, ok := readMVKeyPresent(t, s, "mvktr", 1, "tSet"); !ok {
		t.Fatal("post-resplit pre-deadline: tSet dropped by the copy (per-key TTL not carried)")
	}

	time.Sleep(time.Until(deadline) + 750*time.Millisecond)
	live, okSet := readMVKeyPresent(t, s, "mvktr", 1, "tSet")
	if !live {
		t.Fatal("doc 1 vanished after key expiry")
	}
	if okSet {
		t.Fatal("time-stable violated: 'tSet' still present after its original deadline (resplit reset/dropped the per-key TTL)")
	}
	if _, okPerm := readMVKeyPresent(t, s, "mvktr", 1, "perm"); !okPerm {
		t.Fatal("permanent key 'perm' lost after resplit")
	}
}
