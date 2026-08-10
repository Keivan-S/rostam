// SPDX-License-Identifier: Apache-2.0

package shard

import (
	"bytes"
	"context"
	"fmt"
	"io"
	"sync"
	"sync/atomic"
	"testing"
	"time"

	"github.com/rostamlabs/rostam/cache"
	"github.com/rostamlabs/rostam/ops"
	"github.com/rostamlabs/rostam/vector"
)

// restoreInto restores a BackupSnapshot blob into a fresh cache + vector store and
// returns them plus the carried applied index. The caller owns closing both.
func restoreInto(t *testing.T, data []byte) (*cache.Cache, *vector.CollectionStore, uint64) {
	t.Helper()
	c2, err := cache.New(cache.DefaultConfig())
	if err != nil {
		t.Fatalf("cache.New: %v", err)
	}
	v2, err := vector.OpenCollectionStore(t.TempDir())
	if err != nil {
		t.Fatalf("OpenCollectionStore: %v", err)
	}
	idx, err := restoreSnapshot(c2, v2, nil, io.NopCloser(bytes.NewReader(data)))
	if err != nil {
		t.Fatalf("restoreSnapshot: %v", err)
	}
	return c2, v2, idx
}

// seedVectors adds a small dense collection to a Store's vector store so the
// backup round-trip covers both the KV cache and the vector subsystem.
func seedVectors(t *testing.T, vs *vector.CollectionStore) {
	t.Helper()
	if err := vs.CreateCollection("docs", vector.Config{Dim: 3, Metric: vector.L2, M: 8, EfConstruction: 50, EfSearch: 32, Seed: 1}); err != nil {
		t.Fatalf("CreateCollection: %v", err)
	}
	for i := 1; i <= 5; i++ {
		if err := vs.Upsert("docs", uint64(i), []float32{float32(i), 0, 0}, "chunk", 0, vector.Metadata{"d": vector.NewInt(int64(i))}, nil); err != nil {
			t.Fatalf("Upsert: %v", err)
		}
	}
}

// TestBackupSnapshotRoundTripRaft proves a Raft-mode BackupSnapshot round-trips
// through restoreSnapshot to a byte-identical logical state (cache key/value/exp
// set + vector collections), and that the applied index is carried in the v3
// trailer.
func TestBackupSnapshotRoundTripRaft(t *testing.T) {
	s := newSingleNodeStore(t)
	seedVectors(t, s.vectors)
	const n = 20
	for i := 0; i < n; i++ {
		if err := s.Put([]byte(fmt.Sprintf("k%03d", i)), []byte(fmt.Sprintf("v%03d", i)), 0); err != nil {
			t.Fatalf("Put %d: %v", i, err)
		}
	}

	data, appliedIndex, err := s.BackupSnapshot(context.Background())
	if err != nil {
		t.Fatalf("BackupSnapshot: %v", err)
	}
	if appliedIndex == 0 {
		t.Fatal("Raft BackupSnapshot returned appliedIndex 0 (want the raft log index)")
	}

	c2, v2, carriedIdx := restoreInto(t, data)
	defer c2.Close()
	defer v2.Close()
	if carriedIdx == 0 {
		t.Fatal("restored trailer appliedIndex is 0 — v3 index not carried")
	}
	for i := 0; i < n; i++ {
		got, err := c2.Get([]byte(fmt.Sprintf("k%03d", i)))
		if err != nil || string(got) != fmt.Sprintf("v%03d", i) {
			t.Fatalf("restored k%03d = %q err=%v", i, got, err)
		}
	}
	docs, err := v2.SearchDocs("docs", []float32{1, 0, 0}, 5, vector.Filter{})
	if err != nil || len(docs) != 5 {
		t.Fatalf("restored docs = %d err=%v, want 5", len(docs), err)
	}
}

// newPBStore builds a single-node primary-backup Store (ISR={self}, so every
// write commits immediately through the sweep). It is the primary for shard 0.
func newPBStore(t *testing.T) *Store {
	t.Helper()
	reg := ops.NewRegistry()
	if err := ops.RegisterBuiltins(reg); err != nil {
		t.Fatal(err)
	}
	cc := cache.DefaultConfig()
	cc.NumShards = 1
	cfg := Config{
		NodeID:          "n1",
		DataDir:         t.TempDir(),
		Cache:           cc,
		Ops:             reg,
		ReplicationMode: ReplicationModePB,
		PBControl:       fakeControl{node: "n1"},
	}
	s, err := New(cfg)
	if err != nil {
		t.Fatalf("New(pb): %v", err)
	}
	t.Cleanup(func() { _ = s.Close() })
	if !s.IsLeader() {
		t.Fatal("single-node PB primary must be leader")
	}
	return s
}

// TestBackupSnapshotRoundTripPB proves a PB-mode BackupSnapshot (taken under the
// engine quiesce) round-trips to a byte-identical logical state, and that the
// applied index it carries equals the primary's assigned frontier (lastSeq).
func TestBackupSnapshotRoundTripPB(t *testing.T) {
	s := newPBStore(t)
	seedVectors(t, s.vectors)
	const n = 20
	for i := 0; i < n; i++ {
		if err := s.Put([]byte(fmt.Sprintf("k%03d", i)), []byte(fmt.Sprintf("v%03d", i)), 0); err != nil {
			t.Fatalf("Put %d: %v", i, err)
		}
	}

	data, appliedIndex, err := s.BackupSnapshot(context.Background())
	if err != nil {
		t.Fatalf("BackupSnapshot: %v", err)
	}
	// The PB primary applies each write locally before assigning its seq, so the
	// frontier reflected in the snapshot is exactly the highest seq (== n writes).
	if appliedIndex != n {
		t.Fatalf("PB BackupSnapshot appliedIndex = %d, want %d (lastSeq)", appliedIndex, n)
	}

	c2, v2, carriedIdx := restoreInto(t, data)
	defer c2.Close()
	defer v2.Close()
	if carriedIdx != appliedIndex {
		t.Fatalf("restored trailer appliedIndex = %d, want %d", carriedIdx, appliedIndex)
	}
	for i := 0; i < n; i++ {
		got, err := c2.Get([]byte(fmt.Sprintf("k%03d", i)))
		if err != nil || string(got) != fmt.Sprintf("v%03d", i) {
			t.Fatalf("restored k%03d = %q err=%v", i, got, err)
		}
	}
	docs, err := v2.SearchDocs("docs", []float32{1, 0, 0}, 5, vector.Filter{})
	if err != nil || len(docs) != 5 {
		t.Fatalf("restored docs = %d err=%v, want 5", len(docs), err)
	}
}

// TestBackupSnapshotPBNoTornUnderConcurrentApplies is the correctness crux: it
// hammers a PB primary with concurrent applies (many writers) WHILE repeatedly
// taking BackupSnapshots, and asserts every snapshot is a CONSISTENT point-in-time
// image — it decodes, its CRC verifies (restoreSnapshot fails otherwise), and every
// key present carries its EXACT expected value (a serialization that raced a
// mid-apply cache mutation would corrupt the entry stream / CRC / a value). This is
// what the engine quiesce (RunExclusive holding writeMu+e.mu, excluding BOTH the
// primary Propose and backup Receive apply sites) guarantees.
//
// Run under -race: a torn read would also surface as a data race on the cache /
// vector store between an in-flight apply and the serialization walk.
func TestBackupSnapshotPBNoTornUnderConcurrentApplies(t *testing.T) {
	s := newPBStore(t)

	const writers = 8
	const perWriter = 400
	// key(w,i) -> value(w,i): each writer owns a disjoint key space, and value is a
	// deterministic function of the key, so ANY key observed in a snapshot must map
	// to its one correct value. A torn cache walk cannot fabricate a valid pairing.
	keyOf := func(w, i int) []byte { return []byte(fmt.Sprintf("w%02d-k%05d", w, i)) }
	valOf := func(w, i int) []byte { return []byte(fmt.Sprintf("w%02d-v%05d", w, i)) }

	var wg sync.WaitGroup
	var stop atomic.Bool
	var snaps atomic.Int64

	// Snapshotter: take snapshots as fast as possible during the write storm and
	// validate each one.
	wg.Add(1)
	go func() {
		defer wg.Done()
		for !stop.Load() {
			data, _, err := s.BackupSnapshot(context.Background())
			if err != nil {
				t.Errorf("BackupSnapshot during storm: %v", err)
				return
			}
			c2, err := cache.New(cache.DefaultConfig())
			if err != nil {
				t.Errorf("cache.New: %v", err)
				return
			}
			v2, err := vector.OpenCollectionStore(t.TempDir())
			if err != nil {
				_ = c2.Close()
				t.Errorf("OpenCollectionStore: %v", err)
				return
			}
			if _, err := restoreSnapshot(c2, v2, nil, io.NopCloser(bytes.NewReader(data))); err != nil {
				_ = c2.Close()
				_ = v2.Close()
				t.Errorf("restoreSnapshot (torn/corrupt snapshot): %v", err)
				return
			}
			// Every key present must carry its exact expected value.
			bad := ""
			c2.Iterate(func(k, v []byte, _ uint64) bool {
				var w, i int
				if _, err := fmt.Sscanf(string(k), "w%02d-k%05d", &w, &i); err != nil {
					bad = fmt.Sprintf("unparseable key %q", k)
					return false
				}
				if want := valOf(w, i); string(v) != string(want) {
					bad = fmt.Sprintf("key %q = %q, want %q (torn snapshot)", k, v, want)
					return false
				}
				return true
			})
			_ = c2.Close()
			_ = v2.Close()
			if bad != "" {
				t.Error(bad)
				return
			}
			snaps.Add(1)
		}
	}()

	// Writers.
	for w := 0; w < writers; w++ {
		wg.Add(1)
		go func(w int) {
			defer wg.Done()
			for i := 0; i < perWriter; i++ {
				if err := s.Put(keyOf(w, i), valOf(w, i), 0); err != nil {
					t.Errorf("writer %d put %d: %v", w, i, err)
					return
				}
			}
		}(w)
	}

	// Give the write storm a bounded window that overlaps many snapshots, then
	// stop the snapshotter and let every goroutine unwind.
	deadline := time.Now().Add(10 * time.Second)
	for time.Now().Before(deadline) {
		if snaps.Load() > 5 {
			time.Sleep(300 * time.Millisecond)
			break
		}
		time.Sleep(20 * time.Millisecond)
	}
	stop.Store(true)
	wg.Wait()

	if t.Failed() {
		return
	}
	if snaps.Load() == 0 {
		t.Fatal("took zero valid snapshots during the storm")
	}

	// Final snapshot: after the storm, every written key must be present with its
	// correct value.
	data, _, err := s.BackupSnapshot(context.Background())
	if err != nil {
		t.Fatalf("final BackupSnapshot: %v", err)
	}
	c2, v2, _ := restoreInto(t, data)
	defer c2.Close()
	defer v2.Close()
	for w := 0; w < writers; w++ {
		for i := 0; i < perWriter; i++ {
			got, err := c2.Get(keyOf(w, i))
			if err != nil || string(got) != string(valOf(w, i)) {
				t.Fatalf("final restore w%d k%d = %q err=%v", w, i, got, err)
			}
		}
	}
}
