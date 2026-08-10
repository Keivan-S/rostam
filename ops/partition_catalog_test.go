// SPDX-License-Identifier: Apache-2.0

package ops

import (
	"bytes"
	"encoding/binary"
	"testing"
)

type memCatalogStore struct{ m map[string][]byte }

func newMemCatalogStore() *memCatalogStore { return &memCatalogStore{m: map[string][]byte{}} }
func (s *memCatalogStore) GetCatalog(key []byte) ([]byte, bool) {
	v, ok := s.m[string(key)]
	return v, ok
}
func (s *memCatalogStore) PutCatalog(key, val []byte) error { s.m[string(key)] = val; return nil }

func TestCatalogPutGetCachedAndInvalidate(t *testing.T) {
	store := newMemCatalogStore()
	cat := NewCatalog(store)

	// Unknown collection -> P=1 (single-partition default), ok=false.
	if p, ok := cat.Partitions("docs"); p != 1 || ok {
		t.Fatalf("unknown collection = (%d,%v), want (1,false)", p, ok)
	}
	// Put P=8.
	if err := cat.SetPartitions("docs", 8); err != nil {
		t.Fatal(err)
	}
	if p, ok := cat.Partitions("docs"); p != 8 || !ok {
		t.Fatalf("after set = (%d,%v), want (8,true)", p, ok)
	}
	// A second reader over the same store sees it (record is persisted).
	if p, _ := NewCatalog(store).Partitions("docs"); p != 8 {
		t.Fatalf("fresh reader P=%d, want 8", p)
	}
	// Resplit to 16 bumps the version; the original cat must observe it on next read.
	if err := NewCatalog(store).SetPartitions("docs", 16); err != nil {
		t.Fatal(err)
	}
	if p, _ := cat.Partitions("docs"); p != 16 {
		t.Fatalf("after external resplit P=%d, want 16 (stale cache)", p)
	}
}

func TestCatalogSetPartitionsRejectsNonPositive(t *testing.T) {
	cat := NewCatalog(newMemCatalogStore())
	for _, p := range []int{0, -1, -100} {
		if err := cat.SetPartitions("docs", p); err == nil {
			t.Errorf("SetPartitions(docs,%d) = nil, want error", p)
		}
	}
}

func TestCatalogGenerationRoundTrip(t *testing.T) {
	c := NewCatalog(newMemCatalogStore())
	if err := c.SetPartitionsGen("docs", 8, 3); err != nil {
		t.Fatal(err)
	}
	p, gen, ok := c.PartitionsGen("docs")
	if !ok || p != 8 || gen != 3 {
		t.Fatalf("got (%d,%d,%v), want (8,3,true)", p, gen, ok)
	}
	// Legacy SetPartitions writes gen 0.
	if err := c.SetPartitions("img", 4); err != nil {
		t.Fatal(err)
	}
	if p, gen, ok := c.PartitionsGen("img"); !ok || p != 4 || gen != 0 {
		t.Fatalf("legacy set: got (%d,%d,%v), want (4,0,true)", p, gen, ok)
	}
	// Legacy Partitions still works (ignores gen).
	if p, ok := c.Partitions("docs"); !ok || p != 8 {
		t.Fatalf("legacy read (%d,%v)", p, ok)
	}
	// Unknown collection.
	if _, _, ok := c.PartitionsGen("nope"); ok {
		t.Fatal("unknown should be !ok")
	}
	// Non-positive p rejected.
	if err := c.SetPartitionsGen("x", 0, 1); err == nil {
		t.Fatal("p=0 should error")
	}
}

func TestCatalogRecordBackwardCompat(t *testing.T) {
	// An 8-byte (pre-gen) record decodes with gen 0 (existing on-disk data).
	old := make([]byte, 8)
	binary.LittleEndian.PutUint32(old[0:4], 5) // Partitions
	binary.LittleEndian.PutUint32(old[4:8], 2) // Version
	rec, ok := decodeCatalogRecord(old)
	if !ok || rec.Partitions != 5 || rec.Generation != 0 {
		t.Fatalf("old 8-byte record decoded %+v, want P=5 gen=0", rec)
	}
	// New 12-byte record round-trips gen.
	b := encodeCatalogRecord(catalogRecord{Partitions: 8, Version: 1, Generation: 3})
	rec2, ok := decodeCatalogRecord(b)
	if !ok || rec2.Partitions != 8 || rec2.Generation != 3 || rec2.Version != 1 {
		t.Fatalf("new record round-trip %+v", rec2)
	}
}

// TestCatalogRecordStableEncodesLegacy12B proves a Stable record (Status==0,
// no targets) serializes byte-identically to the legacy 12-byte form, so
// existing on-disk data and readers are unaffected.
func TestCatalogRecordStableEncodesLegacy12B(t *testing.T) {
	rec := catalogRecord{Partitions: 8, Version: 7, Generation: 3} // Status/Target* zero
	got := encodeCatalogRecord(rec)
	if len(got) != 12 {
		t.Fatalf("Stable record encoded %d bytes, want 12", len(got))
	}
	want := make([]byte, 12)
	binary.LittleEndian.PutUint32(want[0:4], 8)
	binary.LittleEndian.PutUint32(want[4:8], 7)
	binary.LittleEndian.PutUint32(want[8:12], 3)
	if !bytes.Equal(got, want) {
		t.Fatalf("Stable encoding %v not byte-identical to legacy %v", got, want)
	}
}

// TestCatalogRecordLegacyDecodesStable proves a legacy 12-byte record decodes
// with Status=0 (Stable) and zero targets.
func TestCatalogRecordLegacyDecodesStable(t *testing.T) {
	legacy := make([]byte, 12)
	binary.LittleEndian.PutUint32(legacy[0:4], 4)  // Partitions
	binary.LittleEndian.PutUint32(legacy[4:8], 2)  // Version
	binary.LittleEndian.PutUint32(legacy[8:12], 1) // Generation
	rec, ok := decodeCatalogRecord(legacy)
	if !ok {
		t.Fatal("decode legacy 12B failed")
	}
	if rec.Status != 0 || rec.TargetP != 0 || rec.TargetGen != 0 {
		t.Fatalf("legacy decode got Status=%d TargetP=%d TargetGen=%d, want all 0", rec.Status, rec.TargetP, rec.TargetGen)
	}
	if rec.Partitions != 4 || rec.Version != 2 || rec.Generation != 1 {
		t.Fatalf("legacy decode core fields %+v", rec)
	}
}

// TestCatalogRecordReshardingRoundTrip proves a Resharding record (Status!=0)
// serializes to 32 bytes and round-trips all eight fields (incl. the Source pin),
// and that a legacy 24-byte resharding record (no Source) decodes Source=0.
func TestCatalogRecordReshardingRoundTrip(t *testing.T) {
	rec := catalogRecord{Partitions: 2, Version: 9, Generation: 0, Status: 1, TargetP: 4, TargetGen: 1, SourceP: 2, SourceGen: 0}
	b := encodeCatalogRecord(rec)
	if len(b) != 32 {
		t.Fatalf("Resharding record encoded %d bytes, want 32", len(b))
	}
	got, ok := decodeCatalogRecord(b)
	if !ok {
		t.Fatal("decode 32B failed")
	}
	if got != rec {
		t.Fatalf("round-trip got %+v, want %+v", got, rec)
	}
	// Backward-compat: a 24-byte resharding record written before the Source fields
	// existed decodes with SourceP=SourceGen=0 (the dualTargets pre-upgrade fallback).
	legacy24 := make([]byte, 24)
	binary.LittleEndian.PutUint32(legacy24[0:4], 2)   // Partitions
	binary.LittleEndian.PutUint32(legacy24[4:8], 9)   // Version
	binary.LittleEndian.PutUint32(legacy24[8:12], 0)  // Generation
	binary.LittleEndian.PutUint32(legacy24[12:16], 1) // Status
	binary.LittleEndian.PutUint32(legacy24[16:20], 4) // TargetP
	binary.LittleEndian.PutUint32(legacy24[20:24], 1) // TargetGen
	rec24, ok := decodeCatalogRecord(legacy24)
	if !ok || rec24.Status != 1 || rec24.TargetP != 4 || rec24.TargetGen != 1 || rec24.SourceP != 0 || rec24.SourceGen != 0 {
		t.Fatalf("legacy 24B decode %+v, want Source=0", rec24)
	}
}

// TestCatalogRecordShortDecodeTolerance proves decode tolerates short records:
// <8B invalid; 8B legacy (gen 0); records between 12 and 24 decode the prefix
// fields present without panicking.
func TestCatalogRecordShortDecodeTolerance(t *testing.T) {
	if _, ok := decodeCatalogRecord(make([]byte, 7)); ok {
		t.Fatal("7-byte record should be invalid")
	}
	// 8-byte legacy: Status/Target* default to zero.
	eight := make([]byte, 8)
	binary.LittleEndian.PutUint32(eight[0:4], 3)
	binary.LittleEndian.PutUint32(eight[4:8], 1)
	rec, ok := decodeCatalogRecord(eight)
	if !ok || rec.Partitions != 3 || rec.Generation != 0 || rec.Status != 0 {
		t.Fatalf("8-byte decode %+v", rec)
	}
	// A 20-byte record (has Status+TargetP but truncated TargetGen) must not
	// panic; TargetGen stays zero.
	twenty := make([]byte, 20)
	binary.LittleEndian.PutUint32(twenty[0:4], 2)
	binary.LittleEndian.PutUint32(twenty[4:8], 5)
	binary.LittleEndian.PutUint32(twenty[8:12], 0)
	binary.LittleEndian.PutUint32(twenty[12:16], 1) // Status
	binary.LittleEndian.PutUint32(twenty[16:20], 4) // TargetP
	rec2, ok := decodeCatalogRecord(twenty)
	if !ok || rec2.Status != 1 || rec2.TargetP != 4 || rec2.TargetGen != 0 {
		t.Fatalf("20-byte decode %+v", rec2)
	}
}

// TestCatalogReshardGenRoundTrip exercises the Catalog-level reshard read/write:
// SetReshardGen records status+targets, ReshardGen reads them back, and clearing
// (Status=0) returns the record to the legacy 12-byte Stable form.
func TestCatalogReshardGenRoundTrip(t *testing.T) {
	store := newMemCatalogStore()
	c := NewCatalog(store)
	if err := c.SetPartitionsGen("docs", 2, 0); err != nil {
		t.Fatal(err)
	}
	// No reshard yet -> Stable.
	if st, tp, tg, sp, sg, ok := c.ReshardGen("docs"); ok || st != 0 || tp != 0 || tg != 0 || sp != 0 || sg != 0 {
		t.Fatalf("fresh reshard = (%d,%d,%d,%d,%d,%v), want all-zero/false", st, tp, tg, sp, sg, ok)
	}
	// Begin resharding 2 -> 4 (gen 0 -> 1), pinning source (old) P=2 gen=0.
	if err := c.SetReshardGen("docs", 1, 4, 1, 2, 0); err != nil {
		t.Fatal(err)
	}
	st, tp, tg, sp, sg, ok := c.ReshardGen("docs")
	if !ok || st != 1 || tp != 4 || tg != 1 || sp != 2 || sg != 0 {
		t.Fatalf("resharding = (%d,%d,%d,%d,%d,%v), want (1,4,1,2,0,true)", st, tp, tg, sp, sg, ok)
	}
	// Live partition count/gen unchanged by reshard begin.
	if p, gen, ok := c.PartitionsGen("docs"); !ok || p != 2 || gen != 0 {
		t.Fatalf("live gen changed: (%d,%d,%v)", p, gen, ok)
	}
	// Clearing -> Stable, and the stored record is back to legacy 12 bytes.
	if err := c.SetReshardGen("docs", 0, 0, 0, 0, 0); err != nil {
		t.Fatal(err)
	}
	if _, _, _, _, _, ok := c.ReshardGen("docs"); ok {
		t.Fatal("after clear reshard should be Stable (!ok)")
	}
	raw, _ := store.GetCatalog(catalogKey("docs"))
	if len(raw) != 12 {
		t.Fatalf("after clear stored record is %d bytes, want legacy 12", len(raw))
	}
}

// TestSetPartitionsGenPreservesReshardState guards the Phase-4 cutover: when the
// live read gen is flipped to the target via SetPartitionsGen while a reshard is
// still in progress, the in-progress reshard fields (Status/TargetP/TargetGen)
// must be PRESERVED, not collapsed back to the legacy 12-byte Stable form.
func TestSetPartitionsGenPreservesReshardState(t *testing.T) {
	store := newMemCatalogStore()
	c := NewCatalog(store)
	// Live gen 0 at P=2.
	if err := c.SetPartitionsGen("docs", 2, 0); err != nil {
		t.Fatal(err)
	}
	// Begin resharding 2 -> 4 (gen 0 -> 1), source pin P=2 gen=0.
	if err := c.SetReshardGen("docs", 1, 4, 1, 2, 0); err != nil {
		t.Fatal(err)
	}
	// Phase-4 cutover: flip the live read gen to the target (P=4, gen=1) while the
	// reshard is still in progress.
	if err := c.SetPartitionsGen("docs", 4, 1); err != nil {
		t.Fatal(err)
	}

	// (a) Live gen updated.
	if p, gen, ok := c.PartitionsGen("docs"); !ok || p != 4 || gen != 1 {
		t.Fatalf("PartitionsGen after cutover = (%d,%d,%v), want (4,1,true)", p, gen, ok)
	}
	// (b) Reshard state preserved across the flip — INCLUDING the source pin, which
	// is what keeps the dual-write hitting the old gen after the live gen flips.
	if st, tp, tg, sp, sg, ok := c.ReshardGen("docs"); !ok || st != 1 || tp != 4 || tg != 1 || sp != 2 || sg != 0 {
		t.Fatalf("ReshardGen after cutover = (%d,%d,%d,%d,%d,%v), want (1,4,1,2,0,true)", st, tp, tg, sp, sg, ok)
	}
	// (c) Stored record kept the full 32-byte reshard form (not collapsed to 12B).
	raw, _ := store.GetCatalog(catalogKey("docs"))
	if len(raw) != 32 {
		t.Fatalf("after cutover stored record is %d bytes, want 32 (reshard preserved)", len(raw))
	}
}
