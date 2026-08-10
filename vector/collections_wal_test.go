// SPDX-License-Identifier: Apache-2.0

package vector

import (
	"errors"
	"os"
	"path/filepath"
	"testing"
)

func walCfg() Config {
	// WAL collections are heap-backed (snapshot + replay durability); WAL is
	// mutually exclusive with Persistent.
	return Config{
		Dim: 16, Metric: Cosine, M: 16, EfConstruction: 100, EfSearch: 64, Seed: 1,
		Quant: QuantSQ8, RescoreFactor: 3, WAL: true,
	}
}

// TestWALRecoversUnflushedInserts is the crash-durability headline: inserts into
// a WAL collection that is NEVER flushed are recovered on reopen by replaying
// the log onto a fresh (empty) index — proving the WAL, not the checkpoint, is
// what makes them durable.
func TestWALRecoversUnflushedInserts(t *testing.T) {
	const dim, k = 16, 10
	dir := t.TempDir()

	cs, err := OpenCollectionStore(dir)
	if err != nil {
		t.Fatal(err)
	}
	if err := cs.CreateCollection("docs", walCfg()); err != nil {
		t.Fatal(err)
	}
	_, vecs := siftLikeCorpus(300, dim, 4)
	for _, v := range vecs {
		normalize(v)
	}
	for i, v := range vecs {
		if err := cs.Insert("docs", uint64(i+1), v, 0, nil, nil); err != nil {
			t.Fatal(err)
		}
	}
	q := vecs[0]
	before, err := cs.SearchFiltered("docs", q, k, Filter{})
	if err != nil {
		t.Fatal(err)
	}
	// Deliberately DO NOT Flush — simulate a crash with only the WAL on disk.
	_ = cs.Close()

	cs2, err := OpenCollectionStore(dir)
	if err != nil {
		t.Fatalf("reopen: %v", err)
	}
	defer func() { _ = cs2.Close() }()
	c2, ok := cs2.Get("docs")
	if !ok {
		t.Fatal("collection missing after reopen")
	}
	if got := c2.Stats().Size; got != len(vecs) {
		t.Errorf("recovered size = %d, want %d (WAL replay missed inserts)", got, len(vecs))
	}
	after, err := cs2.SearchFiltered("docs", q, k, Filter{})
	if err != nil {
		t.Fatal(err)
	}
	if !eqUint64(resultIDs(before), resultIDs(after)) {
		t.Errorf("post-recovery results %v != %v", resultIDs(after), resultIDs(before))
	}
}

// TestWALCheckpointThenTail covers the common case: a checkpoint (Flush) plus a
// WAL tail of post-checkpoint ops. Recovery = openPersist(checkpoint) + replay.
func TestWALCheckpointThenTail(t *testing.T) {
	const dim = 16
	dir := t.TempDir()
	cs, err := OpenCollectionStore(dir)
	if err != nil {
		t.Fatal(err)
	}
	if err := cs.CreateCollection("docs", walCfg()); err != nil {
		t.Fatal(err)
	}
	_, vecs := siftLikeCorpus(200, dim, 7)
	for _, v := range vecs {
		normalize(v)
	}
	// Batch A, then checkpoint (WAL rotates empty).
	for i := 0; i < 120; i++ {
		if err := cs.Insert("docs", uint64(i+1), vecs[i], 0, nil, nil); err != nil {
			t.Fatal(err)
		}
	}
	if err := cs.Flush("docs"); err != nil {
		t.Fatal(err)
	}
	// Batch B goes only to the WAL tail (no further flush).
	for i := 120; i < 200; i++ {
		if err := cs.Insert("docs", uint64(i+1), vecs[i], 0, nil, nil); err != nil {
			t.Fatal(err)
		}
	}
	// One id deleted post-checkpoint — must stay deleted after recovery.
	cs.mustDelete(t, "docs", 5)
	_ = cs.Close()

	cs2, err := OpenCollectionStore(dir)
	if err != nil {
		t.Fatal(err)
	}
	defer func() { _ = cs2.Close() }()
	c2, _ := cs2.Get("docs")
	if got := c2.Stats().Size; got != 199 { // 200 inserted, 1 deleted
		t.Errorf("recovered size = %d, want 199", got)
	}
	// The deleted id must not resurface.
	res, _ := cs2.SearchFiltered("docs", vecs[4], 200, Filter{})
	for _, r := range res {
		if r.ID == 5 {
			t.Error("deleted id 5 resurfaced after recovery")
		}
	}
}

// TestWALTornTailTolerated checks a truncated/garbage tail record (a crash
// mid-append) is ignored on replay, with all prior intact records recovered.
func TestWALTornTailTolerated(t *testing.T) {
	const dim = 16
	dir := t.TempDir()
	cs, err := OpenCollectionStore(dir)
	if err != nil {
		t.Fatal(err)
	}
	if err := cs.CreateCollection("docs", walCfg()); err != nil {
		t.Fatal(err)
	}
	_, vecs := siftLikeCorpus(50, dim, 2)
	for i, v := range vecs {
		normalize(v)
		if err := cs.Insert("docs", uint64(i+1), v, 0, nil, nil); err != nil {
			t.Fatal(err)
		}
	}
	_ = cs.Close()

	// Corrupt the tail: append a partial record header (a crash mid-append).
	walPath := filepath.Join(dir, "vectors", "default", "docs.wal")
	f, err := os.OpenFile(walPath, os.O_WRONLY|os.O_APPEND, 0o600)
	if err != nil {
		t.Fatal(err)
	}
	_, _ = f.Write([]byte{0, 0, 0, 9, 1, 2}) // claims a 9-byte payload, only 2 bytes follow
	_ = f.Close()

	cs2, err := OpenCollectionStore(dir)
	if err != nil {
		t.Fatalf("reopen with torn tail: %v", err)
	}
	defer func() { _ = cs2.Close() }()
	c2, _ := cs2.Get("docs")
	if got := c2.Stats().Size; got != len(vecs) {
		t.Errorf("recovered size = %d, want %d (torn tail should be ignored, prior records kept)", got, len(vecs))
	}
}

// TestWALValidation rejects WAL combined with Persistent (mutually exclusive).
func TestWALValidation(t *testing.T) {
	cs, err := OpenCollectionStore(t.TempDir())
	if err != nil {
		t.Fatal(err)
	}
	defer func() { _ = cs.Close() }()
	err = cs.CreateCollection("bad", Config{
		Dim: 8, Metric: Cosine, M: 8, EfConstruction: 50, EfSearch: 50,
		Quant: QuantSQ8, RescoreFactor: 2, WAL: true, Persistent: true,
	})
	if !errors.Is(err, ErrInvalidWAL) {
		t.Errorf("WAL+Persistent = %v, want ErrInvalidWAL", err)
	}
}

// mustDelete deletes id from the named collection via the store, failing if the
// collection is absent.
func (s *CollectionStore) mustDelete(t *testing.T, name string, id uint64) {
	t.Helper()
	c, ok := s.Get(name)
	if !ok {
		t.Fatalf("no collection %q", name)
	}
	c.Delete(id)
}
