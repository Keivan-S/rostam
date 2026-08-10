// SPDX-License-Identifier: Apache-2.0

package vector

import (
	"bytes"
	"errors"
	"testing"
)

// TestRestorePreservesNonStreamConfig pins the copy-then-overwrite invariant in
// readSnapshot.
//
// The snapshot header carries exactly six Config fields (Dim, Metric, M,
// EfConstruction, EfSearch, Seed). readSnapshot used to build a fresh Config
// from those six plus a hand-written list of fields worth back-filling, then
// assign it over h.cfg — so every field nobody remembered to list was ZEROED on
// the restored index. That is silent: Collection.Config() keeps its own copy and
// happily keeps reporting the original, while the write path reads the zeroed
// h.cfg.
//
// Level0FullDegree is the same failure class as the restored-metric bug: it is
// read live by forwardM on every insert, so losing it halves the level-0
// out-degree of everything written after a restore, with no error anywhere.
// MaxVectors/MaxBytes are worse in a different way — the quota simply stops
// existing.
func TestRestorePreservesNonStreamConfig(t *testing.T) {
	const dim = 8
	// Every field here is non-default AND outside the six the stream carries.
	cfg := Config{
		Dim: dim, Metric: L2, M: 8, EfConstruction: 32, EfSearch: 16, Seed: 7,

		Level0FullDegree:      true,
		MaxVectors:            4,
		MaxBytes:              1 << 40,
		MaxEfSearch:           512,
		ExtendCandidates:      true,
		ExtendCandidatesMax:   64,
		FilterFirstThreshold:  12345,
		FilterFirstRelativeBP: 4321,
	}
	if err := cfg.Validate(); err != nil {
		t.Fatalf("test config is invalid: %v", err)
	}

	src := newCollectionStore(t)
	if err := src.CreateCollection("docs", cfg); err != nil {
		t.Fatalf("CreateCollection: %v", err)
	}
	c, ok := src.Get("docs")
	if !ok {
		t.Fatal("source collection missing")
	}
	vec := make([]float32, dim)
	for i := 1; i <= 4; i++ { // exactly MaxVectors points
		vec[i%dim] = float32(i)
		if err := c.Insert(uint64(i), vec, 0, nil, nil); err != nil {
			t.Fatalf("Insert %d: %v", i, err)
		}
	}
	var snap bytes.Buffer
	if err := c.Snapshot(&snap); err != nil {
		t.Fatalf("Snapshot: %v", err)
	}

	// The config-FAITHFUL restore path (what the backup package uses): the target
	// is created from the real Config, then Restore streams the graph on top. The
	// whole point of that path is that the non-stream config survives.
	dst := newCollectionStore(t)
	if err := dst.RestoreCollectionWithConfig("docs", cfg, bytes.NewReader(snap.Bytes())); err != nil {
		t.Fatalf("RestoreCollectionWithConfig: %v", err)
	}
	rc, ok := dst.Get("docs")
	if !ok {
		t.Fatal("restored collection missing")
	}
	h, ok := rc.idx.(*hnsw)
	if !ok {
		t.Fatalf("restored index is %T, want *hnsw", rc.idx)
	}

	// Assert on the INDEX's config, not Collection.Config(): the collection keeps
	// a separate copy that stayed correct even while h.cfg was being zeroed, so
	// checking the collection would have passed against the bug.
	got := h.cfg
	for _, f := range []struct {
		name      string
		got, want any
	}{
		{"Level0FullDegree", got.Level0FullDegree, cfg.Level0FullDegree},
		{"MaxVectors", got.MaxVectors, cfg.MaxVectors},
		{"MaxBytes", got.MaxBytes, cfg.MaxBytes},
		{"MaxEfSearch", got.MaxEfSearch, cfg.MaxEfSearch},
		{"ExtendCandidates", got.ExtendCandidates, cfg.ExtendCandidates},
		{"ExtendCandidatesMax", got.ExtendCandidatesMax, cfg.ExtendCandidatesMax},
		{"FilterFirstThreshold", got.FilterFirstThreshold, cfg.FilterFirstThreshold},
		{"FilterFirstRelativeBP", got.FilterFirstRelativeBP, cfg.FilterFirstRelativeBP},
	} {
		if f.got != f.want {
			t.Errorf("restored h.cfg.%s = %v, want %v — readSnapshot dropped a non-stream Config field", f.name, f.got, f.want)
		}
	}
	// The six stream-carried fields must still come from the STREAM.
	if got.Dim != dim || got.Metric != L2 || got.M != 8 || got.EfConstruction != 32 || got.EfSearch != 16 || got.Seed != 7 {
		t.Errorf("restored stream fields = {Dim:%d Metric:%v M:%d EfC:%d EfS:%d Seed:%d}, want the snapshot's",
			got.Dim, got.Metric, got.M, got.EfConstruction, got.EfSearch, got.Seed)
	}

	// Behavioural checks — the two effects a caller would actually notice.

	// Level0FullDegree drives forwardM, read live on every post-restore insert.
	if fm := h.forwardM(0); fm != 2*h.cfg.M {
		t.Errorf("post-restore forwardM(0) = %d, want %d (2*M) — level-0 out-degree silently halved for every insert after a restore", fm, 2*h.cfg.M)
	}

	// MaxVectors: the index holds exactly MaxVectors points, so the next insert
	// must be refused. With the quota zeroed it silently succeeded.
	err := rc.Insert(99, vec, 0, nil, nil)
	if !errors.Is(err, ErrCollectionFull) {
		t.Errorf("insert past MaxVectors after restore returned %v, want ErrCollectionFull — the quota was silently lifted", err)
	}
}
