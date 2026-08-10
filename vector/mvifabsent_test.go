// SPDX-License-Identifier: Apache-2.0

package vector

import (
	"sync"
	"testing"
)

func newMVIfAbsent(t *testing.T) *MultiVectorIndex {
	t.Helper()
	m, err := NewMultiVectorIndex(MultiVectorConfig{Dim: 4, M: 16, EfConstruction: 200, EfSearch: 64, Seed: 1})
	if err != nil {
		t.Fatalf("NewMultiVectorIndex: %v", err)
	}
	return m
}

func mvTok(vals ...float32) [][]float32 {
	// one 4-dim token built from the first 4 supplied values (zero-padded)
	tok := make([]float32, 4)
	copy(tok, vals)
	return [][]float32{tok}
}

func TestMVAddIfAbsentInsertsWhenAbsent(t *testing.T) {
	m := newMVIfAbsent(t)
	inserted, err := m.AddIfAbsent(1, mvTok(1, 0, 0, 0), nil)
	if err != nil {
		t.Fatalf("AddIfAbsent: %v", err)
	}
	if !inserted {
		t.Fatalf("AddIfAbsent on absent doc returned inserted=false, want true")
	}
	if !m.Exists(1) {
		t.Fatalf("Exists(1) = false after add, want true")
	}
}

func TestMVAddIfAbsentNoOpWhenLive(t *testing.T) {
	m := newMVIfAbsent(t)
	if err := m.Add(1, mvTok(1, 0, 0, 0), Metadata{"k": NewString("v1")}); err != nil {
		t.Fatalf("Add: %v", err)
	}
	before := m.NumVectors()
	inserted, err := m.AddIfAbsent(1, mvTok(0, 1, 0, 0), Metadata{"k": NewString("v2")})
	if err != nil {
		t.Fatalf("AddIfAbsent: %v", err)
	}
	if inserted {
		t.Fatalf("AddIfAbsent on live doc returned inserted=true, want false")
	}
	// No-op: token count and metadata must be unchanged (not clobbered to v2).
	if m.NumVectors() != before {
		t.Fatalf("AddIfAbsent no-op changed token count: got %d, want %d", m.NumVectors(), before)
	}
	m.mu.RLock()
	meta := m.docMeta[1]
	m.mu.RUnlock()
	if got := meta["k"].Str; got != "v1" {
		t.Fatalf("AddIfAbsent no-op clobbered metadata: got %q, want v1", got)
	}
}

func TestMVAddIfAbsentAfterDeleteInserts(t *testing.T) {
	m := newMVIfAbsent(t)
	if err := m.Add(1, mvTok(1, 0, 0, 0), nil); err != nil {
		t.Fatalf("Add: %v", err)
	}
	if !m.Delete(1) {
		t.Fatalf("Delete(1) = false, want true")
	}
	if m.Exists(1) {
		t.Fatalf("Exists(1) = true after delete, want false")
	}
	inserted, err := m.AddIfAbsent(1, mvTok(0, 1, 0, 0), nil)
	if err != nil {
		t.Fatalf("AddIfAbsent: %v", err)
	}
	if !inserted {
		t.Fatalf("AddIfAbsent after delete returned inserted=false, want true")
	}
}

func TestMVExistsLiveDeletedAbsent(t *testing.T) {
	m := newMVIfAbsent(t)
	if m.Exists(99) {
		t.Fatalf("Exists(absent) = true, want false")
	}
	if err := m.Add(1, mvTok(1, 0, 0, 0), nil); err != nil {
		t.Fatalf("Add: %v", err)
	}
	if !m.Exists(1) {
		t.Fatalf("Exists(live) = false, want true")
	}
	m.Delete(1)
	if m.Exists(1) {
		t.Fatalf("Exists(deleted) = true, want false")
	}
}

func TestMVAddIfAbsentEmptyDocument(t *testing.T) {
	m := newMVIfAbsent(t)
	if _, err := m.AddIfAbsent(1, nil, nil); err != ErrEmptyDocument {
		t.Fatalf("AddIfAbsent(empty) err = %v, want ErrEmptyDocument", err)
	}
}

// TestMVAddIfAbsentAtomicVsAdd mirrors Race A for the MV path: a copy's
// add-if-absent (v1) racing the dual-write replace Add (v2) must always leave v2
// when the ops are serialized as they are on one partition's Raft log.
func TestMVAddIfAbsentAtomicVsAdd(t *testing.T) {
	const iters = 400
	for it := 0; it < iters; it++ {
		m := newMVIfAbsent(t)
		const id = 7
		var opMu sync.Mutex
		var wg sync.WaitGroup
		wg.Add(2)
		go func() {
			defer wg.Done()
			opMu.Lock()
			_ = m.Add(id, mvTok(0, 1, 0, 0), Metadata{"k": NewString("v2")}) // replace = upsert leg
			opMu.Unlock()
		}()
		go func() {
			defer wg.Done()
			opMu.Lock()
			_, _ = m.AddIfAbsent(id, mvTok(1, 0, 0, 0), Metadata{"k": NewString("v1")})
			opMu.Unlock()
		}()
		wg.Wait()
		m.mu.RLock()
		meta := m.docMeta[id]
		m.mu.RUnlock()
		if got := meta["k"].Str; got != "v2" {
			t.Fatalf("iter %d: final metadata = %q, want v2 (Race A: stale add-if-absent clobbered the add)", it, got)
		}
	}
}
