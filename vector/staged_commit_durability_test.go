// SPDX-License-Identifier: Apache-2.0

package vector

import (
	"testing"
)

// Durability pinning for the staged delete/payload lanes.
//
// The group-commit effectiveness tests measure "few fsyncs for many ops", which
// is the PERFORMANCE claim — and on its own it is a dangerously one-sided
// metric: deleting the commitWaitStaged call outright would drive the fsync
// count toward zero, i.e. it scores BETTER on effectiveness while introducing
// the exact bug the staged split must not have (acking a write whose bytes were
// never fsynced). These tests pin the CORRECTNESS half.
//
// The invariant: when a WAL-mode mutator returns to its caller, the bytes it
// staged are durable. Because these ops run SERIALLY here, no other writer's
// sync can mask a missing wait — after each ack, the highest sequence covered
// by a completed Sync (syncedSeq) must have caught up to every sequence written
// so far (writeSeq). A staged write that was never waited on leaves
// syncedSeq < writeSeq and fails immediately.

// assertDurableAtAck asserts that every record written to w so far is covered
// by a completed fsync — the condition that must hold at the moment a mutator
// returns. Serial callers only: with concurrent writers in flight, writeSeq can
// legitimately run ahead of an unrelated in-progress op.
func assertDurableAtAck(t *testing.T, w *wal, stage string) {
	t.Helper()
	if w.noSync {
		t.Fatalf("%s: wal is in noSync mode — this test cannot pin durability", stage)
	}
	w.syncMu.Lock()
	synced, written := w.syncedSeq, w.writeSeq
	w.syncMu.Unlock()
	if synced < written {
		t.Fatalf("%s: op acked with syncedSeq=%d < writeSeq=%d — %d staged record(s) were acked but never fsynced",
			stage, synced, written, written-synced)
	}
}

// TestDenseStagedOpsDurableAtAck covers the three dense chokepoints:
// DeleteCAS, DeleteCASAt, and payloadOpCAS (via all four payload mutators).
func TestDenseStagedOpsDurableAtAck(t *testing.T) {
	cs := newCollectionStore(t)
	if err := cs.CreateCollection("docs", walCfg()); err != nil {
		t.Fatal(err)
	}
	c, ok := cs.Get("docs")
	if !ok {
		t.Fatal("collection missing")
	}
	vec := make([]float32, 16)
	for i := 1; i <= 8; i++ {
		vec[i%16] = float32(i)
		if err := c.Insert(uint64(i), vec, 0, Metadata{"seed": NewInt(int64(i))}, nil); err != nil {
			t.Fatalf("seed Insert(%d): %v", i, err)
		}
	}
	assertDurableAtAck(t, c.wal, "seed inserts")

	// payloadOpCAS, once per mutator that funnels through it.
	if err := c.SetPayload(1, Metadata{"a": NewInt(1)}, nil); err != nil {
		t.Fatalf("SetPayload: %v", err)
	}
	assertDurableAtAck(t, c.wal, "SetPayload")

	if err := c.OverwritePayload(2, Metadata{"b": NewInt(2)}, nil); err != nil {
		t.Fatalf("OverwritePayload: %v", err)
	}
	assertDurableAtAck(t, c.wal, "OverwritePayload")

	if err := c.DeletePayloadKeys(3, []string{"seed"}); err != nil {
		t.Fatalf("DeletePayloadKeys: %v", err)
	}
	assertDurableAtAck(t, c.wal, "DeletePayloadKeys")

	if err := c.ClearPayload(4); err != nil {
		t.Fatalf("ClearPayload: %v", err)
	}
	assertDurableAtAck(t, c.wal, "ClearPayload")

	// DeleteCAS.
	if !c.Delete(5) {
		t.Fatal("Delete(5) reported not-removed")
	}
	assertDurableAtAck(t, c.wal, "Delete")

	// DeleteCASAt — the stamped-apply chokepoint, which had no coverage at all.
	ok2, err := c.DeleteCASAt(6, CASCond{}, c.idx.(*hnsw).now())
	if err != nil || !ok2 {
		t.Fatalf("DeleteCASAt: ok=%v err=%v", ok2, err)
	}
	assertDurableAtAck(t, c.wal, "DeleteCASAt")

	// DeleteByFilter fans out over DeleteCAS; exercise the batch shape too.
	if _, err := c.DeleteByFilter(Filter{Op: FilterEq, Field: "seed", Value: NewInt(7)}); err != nil {
		t.Fatalf("DeleteByFilter: %v", err)
	}
	assertDurableAtAck(t, c.wal, "DeleteByFilter")
}

// TestNamedStagedOpsDurableAtAck covers the two named chokepoints:
// NamedCollection.DeleteCAS and NamedCollection.logPayloadOp — neither of which
// had any test before.
func TestNamedStagedOpsDurableAtAck(t *testing.T) {
	cs := newCollectionStore(t)
	if err := cs.CreateCollection("named", namedWALConfig()); err != nil {
		t.Fatal(err)
	}
	nc, ok := cs.GetNamed("named")
	if !ok {
		t.Fatal("named collection missing")
	}
	for i := 1; i <= 6; i++ {
		vectors := map[string][]float32{
			"title": {float32(i), 0, 0, 0},
			"image": {0, float32(i), 0},
		}
		if _, err := nc.InsertCASKeyTTL(uint64(i), vectors, nil, Metadata{"seed": NewInt(int64(i))}, 0, nil, CASCond{}); err != nil {
			t.Fatalf("seed insert %d: %v", i, err)
		}
	}
	assertDurableAtAck(t, nc.wal, "named seed inserts")

	// logPayloadOp, once per mutator.
	if err := nc.SetPayload(1, Metadata{"a": NewInt(1)}, nil); err != nil {
		t.Fatalf("named SetPayload: %v", err)
	}
	assertDurableAtAck(t, nc.wal, "named SetPayload")

	if err := nc.OverwritePayload(2, Metadata{"b": NewInt(2)}, nil); err != nil {
		t.Fatalf("named OverwritePayload: %v", err)
	}
	assertDurableAtAck(t, nc.wal, "named OverwritePayload")

	if err := nc.DeletePayloadKeys(3, []string{"seed"}); err != nil {
		t.Fatalf("named DeletePayloadKeys: %v", err)
	}
	assertDurableAtAck(t, nc.wal, "named DeletePayloadKeys")

	if err := nc.ClearPayload(4); err != nil {
		t.Fatalf("named ClearPayload: %v", err)
	}
	assertDurableAtAck(t, nc.wal, "named ClearPayload")

	// DeleteCAS.
	removed, err := nc.Delete(5)
	if err != nil || !removed {
		t.Fatalf("named Delete: removed=%v err=%v", removed, err)
	}
	assertDurableAtAck(t, nc.wal, "named Delete")
}

// TestMVStagedOpsDurableAtAck covers the two MultiVectorIndex chokepoints:
// DeleteCAS and logPayloadOp — neither of which had any test before.
func TestMVStagedOpsDurableAtAck(t *testing.T) {
	cs := newCollectionStore(t)
	if err := cs.CreateMultiVector("mv", mvWALConfig()); err != nil {
		t.Fatal(err)
	}
	idx, ok := cs.GetMultiVector("mv")
	if !ok {
		t.Fatal("MV index missing")
	}
	for i := 1; i <= 6; i++ {
		tokens := [][]float32{{float32(i), 0, 0, 0}}
		if _, err := idx.AddCASKeyTTLSparse(uint64(i), tokens, Metadata{"seed": NewInt(int64(i))}, nil, nil, CASCond{}); err != nil {
			t.Fatalf("seed add %d: %v", i, err)
		}
	}
	assertDurableAtAck(t, idx.wal, "MV seed adds")

	// logPayloadOp, once per mutator.
	if err := idx.SetPayload(1, Metadata{"a": NewInt(1)}, nil); err != nil {
		t.Fatalf("MV SetPayload: %v", err)
	}
	assertDurableAtAck(t, idx.wal, "MV SetPayload")

	if err := idx.OverwritePayload(2, Metadata{"b": NewInt(2)}, nil); err != nil {
		t.Fatalf("MV OverwritePayload: %v", err)
	}
	assertDurableAtAck(t, idx.wal, "MV OverwritePayload")

	if err := idx.DeletePayloadKeys(3, []string{"seed"}); err != nil {
		t.Fatalf("MV DeletePayloadKeys: %v", err)
	}
	assertDurableAtAck(t, idx.wal, "MV DeletePayloadKeys")

	if err := idx.ClearPayload(4); err != nil {
		t.Fatalf("MV ClearPayload: %v", err)
	}
	assertDurableAtAck(t, idx.wal, "MV ClearPayload")

	// DeleteCAS.
	if !idx.Delete(5) {
		t.Fatal("MV Delete(5) reported not-removed")
	}
	assertDurableAtAck(t, idx.wal, "MV Delete")
}
