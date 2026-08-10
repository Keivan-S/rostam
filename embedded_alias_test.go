// SPDX-License-Identifier: Apache-2.0

package rostam

import (
	"context"
	"errors"
	"testing"

	"github.com/rostamlabs/rostam/ops"
	"github.com/rostamlabs/rostam/vector"
)

// denseCfg is a small dense config for alias tests. partitions=1 keeps creation
// cheap (alias management is family-agnostic; the resolution chokepoints that
// exercise partitioning are elsewhere).
func denseCfg(partitions int) VectorConfig {
	return VectorConfig{Dim: 4, M: 8, EfConstruction: 50, EfSearch: 32, Seed: 1, Metric: vector.L2, Partitions: partitions}
}

// TestEmbeddedAliasLifecycle drives the full create→list→swap→delete lifecycle
// over the embedded Store (mirrors the reshard embedded e2e tests).
func TestEmbeddedAliasLifecycle(t *testing.T) {
	s := newSingleEmbedded(t)
	waitLeaderEmbedded(t, s)
	ctx := context.Background()

	if err := s.CreateCollection(ctx, "docs", denseCfg(1)); err != nil {
		t.Fatalf("CreateCollection docs: %v", err)
	}
	if err := s.CreateCollection(ctx, "docs2", denseCfg(1)); err != nil {
		t.Fatalf("CreateCollection docs2: %v", err)
	}

	// Create prod→docs.
	if err := s.CreateAlias(ctx, "prod", "docs"); err != nil {
		t.Fatalf("CreateAlias prod→docs: %v", err)
	}
	list, err := s.ListAliases(ctx, "")
	if err != nil {
		t.Fatalf("ListAliases: %v", err)
	}
	if got := list["prod"]; got != "docs" {
		t.Fatalf("after create: prod→%q, want docs (list=%v)", got, list)
	}

	// Atomic swap: {delete prod, create prod→docs2}.
	if err := s.AliasBatch(ctx, []AliasAction{
		{Alias: "prod", Delete: true},
		{Alias: "prod", Canonical: "docs2"},
	}); err != nil {
		t.Fatalf("AliasBatch swap: %v", err)
	}
	list, err = s.ListAliases(ctx, "")
	if err != nil {
		t.Fatalf("ListAliases after swap: %v", err)
	}
	if got := list["prod"]; got != "docs2" {
		t.Fatalf("after swap: prod→%q, want docs2 (list=%v)", got, list)
	}

	// Filtered list: only aliases targeting docs2.
	filtered, err := s.ListAliases(ctx, "docs2")
	if err != nil {
		t.Fatalf("ListAliases(docs2): %v", err)
	}
	if len(filtered) != 1 || filtered["prod"] != "docs2" {
		t.Fatalf("filtered=%v, want {prod:docs2}", filtered)
	}
	if other, err := s.ListAliases(ctx, "docs"); err != nil || len(other) != 0 {
		t.Fatalf("ListAliases(docs)=%v err=%v, want empty", other, err)
	}

	// Delete prod → empty.
	if err := s.DeleteAlias(ctx, "prod"); err != nil {
		t.Fatalf("DeleteAlias prod: %v", err)
	}
	list, err = s.ListAliases(ctx, "")
	if err != nil {
		t.Fatalf("ListAliases after delete: %v", err)
	}
	if len(list) != 0 {
		t.Fatalf("after delete: list=%v, want empty", list)
	}
}

// TestEmbeddedAliasUpsert: creating an existing alias overwrites it (upsert).
func TestEmbeddedAliasUpsert(t *testing.T) {
	s := newSingleEmbedded(t)
	waitLeaderEmbedded(t, s)
	ctx := context.Background()
	for _, c := range []string{"a", "b"} {
		if err := s.CreateCollection(ctx, c, denseCfg(1)); err != nil {
			t.Fatalf("CreateCollection %s: %v", c, err)
		}
	}
	if err := s.CreateAlias(ctx, "prod", "a"); err != nil {
		t.Fatalf("CreateAlias prod→a: %v", err)
	}
	if err := s.CreateAlias(ctx, "prod", "b"); err != nil {
		t.Fatalf("CreateAlias prod→b (upsert): %v", err)
	}
	list, _ := s.ListAliases(ctx, "")
	if list["prod"] != "b" {
		t.Fatalf("upsert: prod→%q, want b", list["prod"])
	}
}

func TestEmbeddedAliasTargetMissing(t *testing.T) {
	s := newSingleEmbedded(t)
	waitLeaderEmbedded(t, s)
	ctx := context.Background()
	err := s.CreateAlias(ctx, "prod", "nope")
	if !errors.Is(err, ErrAliasTargetMissing) {
		t.Fatalf("CreateAlias to missing target: err=%v, want ErrAliasTargetMissing", err)
	}
}

func TestEmbeddedAliasShadowsCollection(t *testing.T) {
	s := newSingleEmbedded(t)
	waitLeaderEmbedded(t, s)
	ctx := context.Background()
	if err := s.CreateCollection(ctx, "docs", denseCfg(1)); err != nil {
		t.Fatalf("CreateCollection docs: %v", err)
	}
	// Alias name == existing real collection "docs".
	err := s.CreateAlias(ctx, "docs", "docs")
	if !errors.Is(err, ErrAliasShadowsCollection) {
		t.Fatalf("CreateAlias shadowing collection: err=%v, want ErrAliasShadowsCollection", err)
	}
}

func TestEmbeddedAliasReservedChar(t *testing.T) {
	s := newSingleEmbedded(t)
	waitLeaderEmbedded(t, s)
	ctx := context.Background()
	if err := s.CreateCollection(ctx, "docs", denseCfg(1)); err != nil {
		t.Fatalf("CreateCollection docs: %v", err)
	}
	err := s.CreateAlias(ctx, "a#b", "docs")
	if !errors.Is(err, ErrAliasReservedChar) {
		t.Fatalf("CreateAlias reserved-char name: err=%v, want ErrAliasReservedChar", err)
	}
}

func TestEmbeddedAliasTargetIsAlias(t *testing.T) {
	s := newSingleEmbedded(t)
	waitLeaderEmbedded(t, s)
	ctx := context.Background()
	if err := s.CreateCollection(ctx, "docs", denseCfg(1)); err != nil {
		t.Fatalf("CreateCollection docs: %v", err)
	}
	if err := s.CreateAlias(ctx, "prod", "docs"); err != nil {
		t.Fatalf("CreateAlias prod→docs: %v", err)
	}
	// prod→docs exists; creating second→prod must reject (target is an alias).
	err := s.CreateAlias(ctx, "second", "prod")
	if !errors.Is(err, ErrAliasTargetIsAlias) {
		t.Fatalf("CreateAlias target-is-alias: err=%v, want ErrAliasTargetIsAlias", err)
	}
}

func TestEmbeddedCreateCollectionRejectsAliasName(t *testing.T) {
	s := newSingleEmbedded(t)
	waitLeaderEmbedded(t, s)
	ctx := context.Background()
	if err := s.CreateCollection(ctx, "docs", denseCfg(1)); err != nil {
		t.Fatalf("CreateCollection docs: %v", err)
	}
	if err := s.CreateAlias(ctx, "prod", "docs"); err != nil {
		t.Fatalf("CreateAlias prod→docs: %v", err)
	}
	// Creating a collection named "prod" (an existing alias) must reject.
	err := s.CreateCollection(ctx, "prod", denseCfg(1))
	if !errors.Is(err, ErrAliasShadowsCollection) {
		t.Fatalf("CreateCollection with alias name: err=%v, want ErrAliasShadowsCollection", err)
	}
}

// TestEmbeddedAliasBatchAllOrNothing: a batch with any invalid create rejects the
// WHOLE batch — nothing is applied.
func TestEmbeddedAliasBatchAllOrNothing(t *testing.T) {
	s := newSingleEmbedded(t)
	waitLeaderEmbedded(t, s)
	ctx := context.Background()
	if err := s.CreateCollection(ctx, "docs", denseCfg(1)); err != nil {
		t.Fatalf("CreateCollection docs: %v", err)
	}
	// First action valid, second targets a missing collection.
	err := s.AliasBatch(ctx, []AliasAction{
		{Alias: "good", Canonical: "docs"},
		{Alias: "bad", Canonical: "nope"},
	})
	if !errors.Is(err, ErrAliasTargetMissing) {
		t.Fatalf("AliasBatch with bad action: err=%v, want ErrAliasTargetMissing", err)
	}
	list, _ := s.ListAliases(ctx, "")
	if len(list) != 0 {
		t.Fatalf("rejected batch applied partially: list=%v, want empty", list)
	}
}

// TestEmbeddedAliasDispatchThrough exercises the fanout-dispatcher virtual ops
// (alias_batch / alias_list) — the coordinator-op path the remote transports use.
func TestEmbeddedAliasDispatchThrough(t *testing.T) {
	s := newSingleEmbedded(t)
	waitLeaderEmbedded(t, s)
	emb := s.(*embedded)
	fan := newFanoutDispatcher(emb, emb.node)
	ctx := context.Background()

	if err := s.CreateCollection(ctx, "docs", denseCfg(1)); err != nil {
		t.Fatalf("CreateCollection docs: %v", err)
	}

	// alias_batch create prod→docs through the dispatcher.
	if _, err := fan.Call("alias_batch", ops.EncodeAliasCreateArgs("prod", "docs")); err != nil {
		t.Fatalf("fan.Call(alias_batch create): %v", err)
	}
	// alias_list through the dispatcher.
	body, err := fan.Call("alias_list", ops.EncodeAliasListArgs(""))
	if err != nil {
		t.Fatalf("fan.Call(alias_list): %v", err)
	}
	entries, err := ops.DecodeAliasListResult(body)
	if err != nil {
		t.Fatalf("DecodeAliasListResult: %v", err)
	}
	if len(entries) != 1 || entries[0].Alias != "prod" || entries[0].Collection != "docs" {
		t.Fatalf("dispatch list = %+v, want [{prod docs}]", entries)
	}

	// alias_batch with a bad target surfaces the op error through the dispatcher.
	_, err = fan.Call("alias_batch", ops.EncodeAliasCreateArgs("bad", "nope"))
	if !errors.Is(err, ErrAliasTargetMissing) {
		t.Fatalf("fan.Call(alias_batch bad) err=%v, want ErrAliasTargetMissing", err)
	}

	// alias_batch delete through the dispatcher.
	if _, err := fan.Call("alias_batch", ops.EncodeAliasDeleteArgs("prod")); err != nil {
		t.Fatalf("fan.Call(alias_batch delete): %v", err)
	}
	body, _ = fan.Call("alias_list", ops.EncodeAliasListArgs(""))
	entries, _ = ops.DecodeAliasListResult(body)
	if len(entries) != 0 {
		t.Fatalf("after dispatch delete: entries=%+v, want empty", entries)
	}
}
