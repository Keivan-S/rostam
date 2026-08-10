// SPDX-License-Identifier: Apache-2.0

package main

import (
	"context"
	"testing"
	"time"

	"github.com/rostamlabs/rostam/objstore"
	"github.com/rostamlabs/rostam/vector"
)

// withAWSCreds sets dummy AWS creds for the duration of a test (t.Setenv restores
// them automatically), so buildTierPlan's credential-presence check passes for the
// happy-path cases that actually construct a store. The values are never used to
// sign a real request in these tests (no network).
func withAWSCreds(t *testing.T) {
	t.Helper()
	t.Setenv("AWS_ACCESS_KEY_ID", "AKIATEST")
	t.Setenv("AWS_SECRET_ACCESS_KEY", "secrettestsecret")
}

// clearAWSCreds removes the AWS creds so the fail-loud missing-creds path can be
// exercised deterministically regardless of the host environment.
func clearAWSCreds(t *testing.T) {
	t.Helper()
	t.Setenv("AWS_ACCESS_KEY_ID", "")
	t.Setenv("AWS_SECRET_ACCESS_KEY", "")
}

// TestBuildTierPlanNothingConfigured: the OPT-IN contract — no bucket and no
// cold-tier ⇒ (nil, nil): no objstore constructed, no driver, no admin backend.
func TestBuildTierPlanNothingConfigured(t *testing.T) {
	clearAWSCreds(t)
	plan, err := buildTierPlan(tierFlags{})
	if err != nil {
		t.Fatalf("unset flags must not error, got %v", err)
	}
	if plan != nil {
		t.Fatalf("unset flags must yield a nil plan (no objstore), got %+v", plan)
	}
	// And no admin backend is built for a nil plan.
	if ab := newAdminBackend(plan); ab != nil {
		t.Fatalf("nil plan must yield a nil admin backend, got %v", ab)
	}
}

// TestBuildTierPlanFailLoud: a bucket/cold-tier IS requested but a required piece
// (region or creds) is missing ⇒ a FATAL error from the validate helper (the
// helper returns the error; main turns it into log.Fatalf — the test never calls
// os.Exit, which is exactly why the validation is a returning function).
func TestBuildTierPlanFailLoud(t *testing.T) {
	t.Run("bucket without region", func(t *testing.T) {
		withAWSCreds(t)
		_, err := buildTierPlan(tierFlags{Bucket: "b", Interval: time.Minute})
		if err == nil {
			t.Fatal("bucket set but region missing must be FATAL (non-nil error)")
		}
	})
	t.Run("bucket without creds", func(t *testing.T) {
		clearAWSCreds(t)
		_, err := buildTierPlan(tierFlags{Bucket: "b", Region: "us-east-1", Interval: time.Minute})
		if err == nil {
			t.Fatal("bucket set but AWS creds missing must be FATAL (non-nil error)")
		}
	})
	t.Run("cold-tier without bucket", func(t *testing.T) {
		withAWSCreds(t)
		_, err := buildTierPlan(tierFlags{ColdTierAfter: time.Hour, Region: "us-east-1"})
		if err == nil {
			t.Fatal("cold-tier set but bucket missing must be FATAL (non-nil error)")
		}
	})
}

// TestBuildTierPlanIntervalAloneIsNoOp: -backup-interval alone (no bucket, no
// cold-tier) is NOT an S3 misconfig — it is the trigger for the separate
// filesystem backup path. The S3 plan must treat it as "nothing configured"
// (nil, nil), constructing no S3 objstore.
func TestBuildTierPlanIntervalAloneIsNoOp(t *testing.T) {
	clearAWSCreds(t)
	plan, err := buildTierPlan(tierFlags{Interval: time.Minute})
	if err != nil {
		t.Fatalf("interval alone (FS-backup trigger) must not error the S3 plan, got %v", err)
	}
	if plan != nil {
		t.Fatalf("interval alone must yield a nil S3 plan, got %+v", plan)
	}
}

// TestBuildTierPlanConfigured: a fully-specified bucket+region+creds yields a live
// plan with the shared store and the requested knobs threaded through.
func TestBuildTierPlanConfigured(t *testing.T) {
	withAWSCreds(t)
	plan, err := buildTierPlan(tierFlags{
		Bucket:        "my-bucket",
		Region:        "us-east-1",
		Endpoint:      "http://127.0.0.1:9000",
		Interval:      30 * time.Minute,
		Retention:     7,
		Tenant:        "acme",
		ColdTierAfter: time.Hour,
		PathStyle:     true,
	})
	if err != nil {
		t.Fatalf("fully-configured flags must build a plan, got %v", err)
	}
	if plan == nil {
		t.Fatal("expected a non-nil plan")
	}
	if plan.Store == nil {
		t.Fatal("expected a shared object store on the plan")
	}
	if plan.Tenant != "acme" || plan.Retention != 7 || plan.BackupInterval != 30*time.Minute || plan.ColdTierAfter != time.Hour {
		t.Fatalf("plan knobs not threaded through: %+v", plan)
	}
	// And an admin backend IS built for a live plan (store attached later).
	if ab := newAdminBackend(plan); ab == nil {
		t.Fatal("live plan must yield a non-nil admin backend")
	}
}

func tieringTestConfig() vector.Config {
	return vector.Config{Dim: 2, M: 4, EfConstruction: 10, EfSearch: 10, Seed: 1, Metric: vector.L2}
}

func newTieringStore(t *testing.T) *vector.CollectionStore {
	t.Helper()
	store, err := vector.OpenCollectionStore(t.TempDir())
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() { _ = store.Close() })
	return store
}

func mustCreateTiering(t *testing.T, store *vector.CollectionStore, name string, ids ...uint64) {
	t.Helper()
	if err := store.CreateCollection(name, tieringTestConfig()); err != nil {
		t.Fatalf("create %q: %v", name, err)
	}
	c, ok := store.Acquire(name)
	if !ok {
		t.Fatalf("acquire %q after create", name)
	}
	defer c.Release()
	for _, id := range ids {
		if err := c.Insert(id, []float32{float32(id), 0}, 0, nil, nil); err != nil {
			t.Fatalf("insert %d into %q: %v", id, name, err)
		}
	}
}

// TestRunBackupTick drives ONE backup tick body directly (no ticker/goroutine)
// against an in-memory MemStore and asserts the expected snapshot objects appear.
func TestRunBackupTick(t *testing.T) {
	store := newTieringStore(t)
	mustCreateTiering(t, store, "docs", 1, 2, 3)
	mustCreateTiering(t, store, "logs", 4, 5)

	obj := objstore.NewMemStore()
	plan := &tierPlan{Store: obj, Tenant: "acme", Retention: 0}

	runBackupTick(context.Background(), store, plan)

	infos, err := obj.List(context.Background(), "acme/")
	if err != nil {
		t.Fatal(err)
	}
	// Two collections × (one .snap + one .cfg.json) = 4 objects.
	var snaps int
	for _, in := range infos {
		if len(in.Key) > 5 && in.Key[len(in.Key)-5:] == ".snap" {
			snaps++
		}
	}
	if snaps != 2 {
		t.Fatalf("expected 2 snapshot objects after a backup tick, got %d (all=%d)", snaps, len(infos))
	}
}

// TestRunSweepTick evicts an idle collection on a sweep and leaves a
// recently-accessed one hot. The clock is injected so the sweep is deterministic.
func TestRunSweepTick(t *testing.T) {
	store := newTieringStore(t)

	// Inject a controllable clock for the engine's lastAccess stamping.
	base := time.Date(2026, 6, 23, 12, 0, 0, 0, time.UTC)
	now := base
	store.SetClock(func() time.Time { return now })

	mustCreateTiering(t, store, "idle", 1, 2)
	mustCreateTiering(t, store, "fresh", 3, 4)

	obj := objstore.NewMemStore()
	plan := &tierPlan{Store: obj, Tenant: "acme", ColdTierAfter: time.Hour}

	// First sweep at base: SweepCold seeds lastAccess for never-seen collections
	// (first sight) and evicts nothing.
	now = base
	runSweepTick(base, store, plan)
	if cold := store.ColdNames(); len(cold) != 0 {
		t.Fatalf("first sweep must seed, not evict; cold=%v", cold)
	}

	// Advance past the idle threshold, but touch "fresh" right before the sweep so
	// only "idle" is past the cutoff.
	now = base.Add(2 * time.Hour)
	c, ok := store.Acquire("default/fresh")
	if !ok {
		t.Fatal("acquire fresh")
	}
	c.Release() // stamps fresh's lastAccess at now (2h)

	runSweepTick(base.Add(2*time.Hour), store, plan)

	cold := store.ColdNames()
	if len(cold) != 1 || cold[0] != "default/idle" {
		t.Fatalf("expected only default/idle evicted, got %v", cold)
	}
	if store.IsCold("default/fresh") {
		t.Fatal("recently-accessed default/fresh must NOT be evicted")
	}
}
