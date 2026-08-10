// SPDX-License-Identifier: Apache-2.0

package rostam

import (
	"context"
	"strings"
	"testing"

	"github.com/rostamlabs/rostam/ops"
	"github.com/rostamlabs/rostam/vector"
)

// A PARTITIONED CREATE IS P+1 CLUSTER OPS WITH NO TRANSACTION AROUND THEM.
//
// When one of them failed part-way, the coordinator returned the error and
// walked away, leaving the logical collection plus partitions 0..k-1 behind with
// NO catalog entry. The caller was then stuck between two answers that were both
// wrong: the collection did not work (nothing recorded that it was partitioned,
// so every read and write resolved to the empty logical marker), and it did not
// not-exist either — repeating the identical create returned "already exists",
// so the obvious recovery was the one thing that could not work.
//
// These tests drive a real mid-loop failure by planting one of the physical
// partition names in advance, which is the cheapest way to make partition k fail
// while k-1 succeed.

// plantedPartitionBlocker creates the physical collection that partition p of
// `name` would occupy, so the coordinator's create loop fails exactly there.
func plantedPartitionBlocker(t *testing.T, e *embedded, name string, p int, cfg VectorConfig) string {
	t.Helper()
	phys := string(ops.PartitionKeyGen(name, 0, p))
	physCfg := cfg
	physCfg.Partitions = 0
	if _, err := e.Call(context.Background(), "vector_create_collection", ops.EncodeCreateCollectionArgs(phys, physCfg)); err != nil {
		t.Fatalf("planting the blocker at partition %d: %v", p, err)
	}
	return phys
}

func TestPartitionedCreateRollsBackOnPartialFailure(t *testing.T) {
	stores := newInmemEmbeddedCluster(t, 1, 4)
	e := stores[0].(*embedded)
	ctx := context.Background()

	const (
		name    = "docs"
		P       = 4
		blockAt = 2
	)
	cfg := VectorConfig{Dim: 4, Metric: vector.L2, M: 8, EfConstruction: 50, EfSearch: 64, Seed: 1, Partitions: P}

	plantedPartitionBlocker(t, e, name, blockAt, cfg)

	err := stores[0].CreateCollection(ctx, name, cfg)
	if err == nil {
		t.Fatal("create succeeded despite a pre-existing physical partition — the blocker did not block")
	}
	if !strings.Contains(err.Error(), "create partition") {
		t.Fatalf("create failed with %v; wanted the partition-create error that triggers rollback", err)
	}

	// The logical marker must be GONE. This is the assertion that matters: while
	// it survives, the collection name is occupied by something the caller can
	// neither use nor recreate.
	if e.collectionExists(ctx, name) {
		t.Error("the logical collection survived a failed partitioned create — the name is now occupied " +
			"by a collection that is not partitioned and cannot be recreated")
	}
	// The partitions created BEFORE the failure must be gone too.
	for p := 0; p < blockAt; p++ {
		phys := string(ops.PartitionKeyGen(name, 0, p))
		if _, cerr := e.Call(ctx, "vector_get_config", ops.EncodeGetConfigArgs(phys)); cerr == nil {
			t.Errorf("partition %d survived the failed create as orphaned storage", p)
		}
	}
	// And no catalog entry may be left claiming the collection is partitioned.
	if _, _, ok := e.catalog.PartitionsGen(name); ok {
		t.Error("a catalog entry survived a create that never completed")
	}
}

// TestPartitionedCreateIsRetryableAfterFailure is the user-visible half: once
// whatever caused the failure is gone, the SAME create must succeed. Before the
// rollback this was impossible — the orphaned logical marker made every retry
// fail with "already exists", and clearing it by hand was awkward because the
// drop path finds partitions through the catalog, which had no entry.
func TestPartitionedCreateIsRetryableAfterFailure(t *testing.T) {
	stores := newInmemEmbeddedCluster(t, 1, 4)
	e := stores[0].(*embedded)
	ctx := context.Background()

	const (
		name    = "docs"
		P       = 4
		blockAt = 2
	)
	cfg := VectorConfig{Dim: 4, Metric: vector.L2, M: 8, EfConstruction: 50, EfSearch: 64, Seed: 1, Partitions: P}

	blocker := plantedPartitionBlocker(t, e, name, blockAt, cfg)
	if err := stores[0].CreateCollection(ctx, name, cfg); err == nil {
		t.Fatal("create unexpectedly succeeded")
	}

	// Clear the obstruction, exactly as an operator would after reading the error.
	if _, err := e.Call(ctx, "vector_drop_collection", ops.EncodeDropCollectionArgs(blocker)); err != nil {
		t.Fatalf("dropping the blocker: %v", err)
	}

	if err := stores[0].CreateCollection(ctx, name, cfg); err != nil {
		t.Fatalf("retrying the create after clearing the cause failed: %v — a failed partitioned create "+
			"must leave the name reusable", err)
	}
	// And it must be a REAL partitioned collection, not a marker that happens to
	// exist: the catalog is the only thing that makes P take effect.
	gotP, gen, ok := e.catalog.PartitionsGen(name)
	if !ok || gotP != P || gen != 0 {
		t.Fatalf("after the successful retry the catalog says (P=%d, gen=%d, ok=%v); want (%d, 0, true)", gotP, gen, ok, P)
	}
	for p := 0; p < P; p++ {
		phys := string(ops.PartitionKeyGen(name, 0, p))
		if _, err := e.Call(ctx, "vector_get_config", ops.EncodeGetConfigArgs(phys)); err != nil {
			t.Errorf("partition %d missing after the successful retry: %v", p, err)
		}
	}

	// The collection must actually WORK end to end — the point of the retry.
	if err := stores[0].VectorUpsert(ctx, name, 1, []float32{1, 2, 3, 4}, "doc-1", VectorInsertOpts{}); err != nil {
		t.Fatalf("upsert into the recreated partitioned collection: %v", err)
	}
}
