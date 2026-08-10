// SPDX-License-Identifier: Apache-2.0

package cluster

import (
	"reflect"
	"testing"
)

func TestStateEncodeDecodeRoundtrip(t *testing.T) {
	in := State{
		NumShards: 256,
		Members: []Peer{
			{NodeID: "n1", RaftAddr: "a:1", ServerAddr: "a:2"},
			{NodeID: "n2", RaftAddr: "b:1", ServerAddr: "b:2"},
		},
		Placement: [][]string{{"n1", "n2"}, {"n1", "n2"}},
	}
	b, err := encodeState(in)
	if err != nil {
		t.Fatal(err)
	}
	out, err := decodeState(b)
	if err != nil {
		t.Fatal(err)
	}
	if !reflect.DeepEqual(in, out) {
		t.Errorf("roundtrip mismatch: %+v vs %+v", in, out)
	}
}

func TestLogEntryEncodeDecodeRoundtrip(t *testing.T) {
	in := LogEntry{
		Op:        OpSetMembers,
		Members:   []Peer{{NodeID: "n1", RaftAddr: "a:1", ServerAddr: "a:2"}},
		NumShards: 256,
	}
	b, err := encodeLogEntry(in)
	if err != nil {
		t.Fatal(err)
	}
	out, err := decodeLogEntry(b)
	if err != nil {
		t.Fatal(err)
	}
	if !reflect.DeepEqual(in, out) {
		t.Errorf("roundtrip mismatch: %+v vs %+v", in, out)
	}
}

func TestDecodeStateEmptyError(t *testing.T) {
	if _, err := decodeState(nil); err == nil {
		t.Error("expected error decoding empty state")
	}
}

func TestDecodeLogEntryEmptyError(t *testing.T) {
	if _, err := decodeLogEntry(nil); err == nil {
		t.Error("expected error decoding empty log entry")
	}
}

func TestLogEntryCatalogRoundtrip(t *testing.T) {
	in := LogEntry{Op: OpSetCatalogEntry, Collection: "default/docs", Partitions: 8}
	b, err := encodeLogEntry(in)
	if err != nil {
		t.Fatal(err)
	}
	got, err := decodeLogEntry(b)
	if err != nil {
		t.Fatal(err)
	}
	if got.Op != OpSetCatalogEntry || got.Collection != "default/docs" || got.Partitions != 8 {
		t.Fatalf("round-trip = %+v, want SetCatalogEntry docs/8", got)
	}
}

func TestLogEntryGenerationRoundtrip(t *testing.T) {
	in := LogEntry{Op: OpSetCatalogEntry, Collection: "docs", Partitions: 8, Generation: 3}
	b, err := encodeLogEntry(in)
	if err != nil {
		t.Fatal(err)
	}
	got, err := decodeLogEntry(b)
	if err != nil {
		t.Fatal(err)
	}
	if got.Generation != 3 {
		t.Fatalf("Generation round-trip = %d, want 3", got.Generation)
	}
	if got.Op != OpSetCatalogEntry || got.Collection != "docs" || got.Partitions != 8 {
		t.Fatalf("round-trip = %+v, want SetCatalogEntry docs/8/gen3", got)
	}
}

func TestStateCatalogGenEncodeDecodeRoundtrip(t *testing.T) {
	s := State{
		NumShards:  4,
		Catalog:    map[string]uint32{"default/docs": 8},
		CatalogGen: map[string]uint32{"default/docs": 3},
	}
	b, err := encodeState(s)
	if err != nil {
		t.Fatal(err)
	}
	got, err := decodeState(b)
	if err != nil {
		t.Fatal(err)
	}
	if got.CatalogGen["default/docs"] != 3 {
		t.Fatalf("CatalogGen round-trip = %v, want docs=3", got.CatalogGen)
	}

	// OLD-format state (CatalogGen nil) must decode with CatalogGen nil — gob
	// handles the missing field gracefully (backward compat for old snapshots).
	old := State{NumShards: 4, Catalog: map[string]uint32{"default/docs": 8}}
	ob, err := encodeState(old)
	if err != nil {
		t.Fatal(err)
	}
	og, err := decodeState(ob)
	if err != nil {
		t.Fatal(err)
	}
	if og.CatalogGen != nil {
		t.Fatalf("old-format CatalogGen = %v, want nil", og.CatalogGen)
	}
}

// TestLogEntryReshardRoundtrip mirrors TestLogEntryGenerationRoundtrip for the
// reshard fields: an OpSetCatalogReshard LogEntry must round-trip its reshard
// payload through encode/decode.
func TestLogEntryReshardRoundtrip(t *testing.T) {
	in := LogEntry{
		Op:               OpSetCatalogReshard,
		Collection:       "docs",
		ReshardStatus:    1,
		ReshardTargetP:   4,
		ReshardTargetGen: 1,
		ReshardSourceP:   2,
		ReshardSourceGen: 0,
	}
	b, err := encodeLogEntry(in)
	if err != nil {
		t.Fatal(err)
	}
	got, err := decodeLogEntry(b)
	if err != nil {
		t.Fatal(err)
	}
	if got.Op != OpSetCatalogReshard || got.Collection != "docs" {
		t.Fatalf("round-trip = %+v, want SetCatalogReshard docs", got)
	}
	if got.ReshardStatus != 1 || got.ReshardTargetP != 4 || got.ReshardTargetGen != 1 || got.ReshardSourceP != 2 || got.ReshardSourceGen != 0 {
		t.Fatalf("reshard fields round-trip = (Status=%d,TargetP=%d,TargetGen=%d,SourceP=%d,SourceGen=%d), want (1,4,1,2,0)",
			got.ReshardStatus, got.ReshardTargetP, got.ReshardTargetGen, got.ReshardSourceP, got.ReshardSourceGen)
	}
}

// TestStateCatalogReshardEncodeDecodeRoundtrip mirrors
// TestStateCatalogGenEncodeDecodeRoundtrip for CatalogReshard: a State carrying a
// reshard entry must gob round-trip it, and an old-format State (no CatalogReshard)
// must decode to nil (every collection Stable) for backward compatibility.
func TestStateCatalogReshardEncodeDecodeRoundtrip(t *testing.T) {
	s := State{
		NumShards:  4,
		Catalog:    map[string]uint32{"default/docs": 2},
		CatalogGen: map[string]uint32{"default/docs": 0},
		CatalogReshard: map[string]ReshardEntry{
			"default/docs": {Status: 1, TargetP: 4, TargetGen: 1, SourceP: 2, SourceGen: 0},
		},
	}
	b, err := encodeState(s)
	if err != nil {
		t.Fatal(err)
	}
	got, err := decodeState(b)
	if err != nil {
		t.Fatal(err)
	}
	e := got.CatalogReshard["default/docs"]
	if e.Status != 1 || e.TargetP != 4 || e.TargetGen != 1 || e.SourceP != 2 || e.SourceGen != 0 {
		t.Fatalf("CatalogReshard round-trip = %+v, want (Status=1,TargetP=4,TargetGen=1,SourceP=2,SourceGen=0)", e)
	}

	// OLD-format state (CatalogReshard nil) must decode with CatalogReshard nil —
	// gob handles the missing field gracefully (every collection all-Stable).
	old := State{NumShards: 4, Catalog: map[string]uint32{"default/docs": 8}}
	ob, err := encodeState(old)
	if err != nil {
		t.Fatal(err)
	}
	og, err := decodeState(ob)
	if err != nil {
		t.Fatal(err)
	}
	if og.CatalogReshard != nil {
		t.Fatalf("old-format CatalogReshard = %v, want nil", og.CatalogReshard)
	}
}

// TestLogEntryAliasBatchRoundtrip mirrors TestLogEntryReshardRoundtrip for the
// alias-batch field: an OpSetAliasBatch LogEntry must round-trip its AliasBatch
// payload through encode/decode.
func TestLogEntryAliasBatchRoundtrip(t *testing.T) {
	in := LogEntry{
		Op: OpSetAliasBatch,
		AliasBatch: []AliasAction{
			{Alias: "prod", Delete: true},
			{Alias: "prod", Canonical: "default/coll_v2"},
		},
	}
	b, err := encodeLogEntry(in)
	if err != nil {
		t.Fatal(err)
	}
	got, err := decodeLogEntry(b)
	if err != nil {
		t.Fatal(err)
	}
	if got.Op != OpSetAliasBatch {
		t.Fatalf("round-trip Op = %v, want OpSetAliasBatch", got.Op)
	}
	if !reflect.DeepEqual(got.AliasBatch, in.AliasBatch) {
		t.Fatalf("AliasBatch round-trip = %+v, want %+v", got.AliasBatch, in.AliasBatch)
	}
}

// TestStateAliasesEncodeDecodeRoundtrip mirrors
// TestStateCatalogReshardEncodeDecodeRoundtrip for Aliases: a State carrying
// aliases must gob round-trip them, and an old-format State (no Aliases) must
// decode to nil (no aliases) for backward compatibility.
func TestStateAliasesEncodeDecodeRoundtrip(t *testing.T) {
	s := State{
		NumShards: 4,
		Catalog:   map[string]uint32{"default/coll_v2": 2},
		Aliases: map[string]string{
			"prod":  "default/coll_v2",
			"stage": "default/coll_v1",
		},
	}
	b, err := encodeState(s)
	if err != nil {
		t.Fatal(err)
	}
	got, err := decodeState(b)
	if err != nil {
		t.Fatal(err)
	}
	if !reflect.DeepEqual(got.Aliases, s.Aliases) {
		t.Fatalf("Aliases round-trip = %v, want %v", got.Aliases, s.Aliases)
	}

	// OLD-format state (Aliases nil) must decode with Aliases nil — gob handles
	// the missing field gracefully (no aliases).
	old := State{NumShards: 4, Catalog: map[string]uint32{"default/docs": 8}}
	ob, err := encodeState(old)
	if err != nil {
		t.Fatal(err)
	}
	og, err := decodeState(ob)
	if err != nil {
		t.Fatal(err)
	}
	if og.Aliases != nil {
		t.Fatalf("old-format Aliases = %v, want nil", og.Aliases)
	}
}

func TestStateCatalogEncodeDecodeRoundtrip(t *testing.T) {
	s := State{
		NumShards: 4,
		Members: []Peer{
			{NodeID: "n1", RaftAddr: "a:1", ServerAddr: "a:2"},
		},
		Catalog: map[string]uint32{"default/docs": 8, "default/img": 3},
	}
	b, err := encodeState(s)
	if err != nil {
		t.Fatal(err)
	}
	got, err := decodeState(b)
	if err != nil {
		t.Fatal(err)
	}
	if got.Catalog["default/docs"] != 8 || got.Catalog["default/img"] != 3 {
		t.Fatalf("catalog round-trip = %v, want docs=8 img=3", got.Catalog)
	}
}
