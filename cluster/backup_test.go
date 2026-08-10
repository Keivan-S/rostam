// SPDX-License-Identifier: Apache-2.0

package cluster

import (
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"strings"
	"testing"
	"time"

	"github.com/rostamlabs/rostam/objstore"
	"github.com/rostamlabs/rostam/ops"
)

// backupAllNodes runs a per-node backup on every node of tc to obj at the given
// timestamp (so retention pruning is deterministic). It returns the total number
// of shard blobs written across the cluster.
func backupAllNodes(t *testing.T, tc *testCluster, obj objstore.ObjectStore, tenant string, retention int, ts time.Time) {
	t.Helper()
	ctx := context.Background()
	for _, n := range tc.nodes {
		if n == nil {
			continue
		}
		if _, err := n.backupOwnedShardsAt(ctx, obj, tenant, retention, ts); err != nil {
			t.Fatalf("backupOwnedShardsAt(node %s): %v", n.cfg.NodeID, err)
		}
		if _, err := n.backupMetaCatalogAt(ctx, obj, tenant, retention, ts); err != nil {
			t.Fatalf("backupMetaCatalogAt(node %s): %v", n.cfg.NodeID, err)
		}
	}
}

// writeKeys writes n keys (key-<i> -> val-<i>) through the cluster client.
func writeKeys(t *testing.T, tc *testCluster, n int) {
	t.Helper()
	ctx := context.Background()
	for i := 0; i < n; i++ {
		k := []byte(fmt.Sprintf("key-%04d", i))
		v := []byte(fmt.Sprintf("val-%04d", i))
		if _, err := tc.client.Call(ctx, "put", ops.EncodePutArgs(k, v, 0)); err != nil {
			t.Fatalf("put key-%04d: %v", i, err)
		}
	}
}

// keysWithSuffix returns the set of keys under obj whose key ends in suffix.
func keysWithSuffix(t *testing.T, obj objstore.ObjectStore, tenant, suffix string) []string {
	t.Helper()
	infos, err := obj.List(context.Background(), tenant+"/")
	if err != nil {
		t.Fatalf("List: %v", err)
	}
	var out []string
	for _, in := range infos {
		if strings.HasSuffix(in.Key, suffix) {
			out = append(out, in.Key)
		}
	}
	return out
}

// TestClusterBackupArtifacts brings up a multi-node cluster,
// writes keys, backs up every node to a shared MemStore, and asserts the artifact
// layout: exactly one .shard blob (+ its sidecar) per shard — written by that
// shard's leader — and one .meta catalog blob from the meta leader.
func TestClusterBackupArtifacts(t *testing.T) {
	const numShards = 4
	tc := newTestCluster(t, 3, numShards, 3) // RF=3: every node owns every shard
	writeKeys(t, tc, 40)

	mem := objstore.NewMemStore()
	backupAllNodes(t, tc, mem, "acme", 0, time.Now().UTC())

	// One .shard per shard index (the leader wrote it; a RF=3 shard is NOT copied
	// 3 times). Group by shard index.
	shardKeys := keysWithSuffix(t, mem, "acme", shardExt)
	byShard := map[int]int{}
	for _, k := range shardKeys {
		idx, ok := shardIndexFromKey(k)
		if !ok {
			t.Fatalf("shard key %q has no shard-NNNN segment", k)
		}
		byShard[idx]++
	}
	for s := 0; s < numShards; s++ {
		if byShard[s] == 0 {
			t.Errorf("shard %d: no backup blob written", s)
		}
		if byShard[s] > 1 {
			t.Errorf("shard %d: %d blobs written (expected 1 — only the leader backs up)", s, byShard[s])
		}
	}

	// A sidecar per shard blob.
	sidecars := keysWithSuffix(t, mem, "acme", shardSidecarExt)
	if len(sidecars) != len(shardKeys) {
		t.Errorf("sidecars=%d, shard blobs=%d (want equal)", len(sidecars), len(shardKeys))
	}

	// Exactly one meta catalog blob (only the meta leader writes it).
	metaKeys := keysWithSuffix(t, mem, "acme", metaExt)
	if len(metaKeys) != 1 {
		t.Errorf("meta blobs=%d, want 1 (only the meta leader writes it): %v", len(metaKeys), metaKeys)
	}
}

// TestClusterBackupRetentionPrunes asserts the generalized retention prunes each
// (node,shard) prefix and the meta prefix oldest-first over the .shard/.meta
// layout: after several runs with retention=2, no prefix keeps more than 2.
func TestClusterBackupRetentionPrunes(t *testing.T) {
	const numShards = 3
	tc := newTestCluster(t, 3, numShards, 3)
	writeKeys(t, tc, 20)

	mem := objstore.NewMemStore()
	base := time.Now().UTC()
	const runs = 4
	const retention = 2
	for r := 0; r < runs; r++ {
		// Distinct, increasing per-run timestamp so keys are unique and sort in run
		// order (retention keeps the newest N).
		backupAllNodes(t, tc, mem, "acme", retention, base.Add(time.Duration(r)*time.Second))
	}

	// Group .shard keys by their directory prefix (everything up to the basename).
	shardKeys := keysWithSuffix(t, mem, "acme", shardExt)
	perDir := map[string]int{}
	for _, k := range shardKeys {
		dir := k[:strings.LastIndex(k, "/")+1]
		perDir[dir]++
	}
	if len(perDir) == 0 {
		t.Fatal("no shard blobs written")
	}
	maxCount := 0
	for dir, c := range perDir {
		if c > retention {
			t.Errorf("shard dir %q kept %d blobs, want <= %d", dir, c, retention)
		}
		if c > maxCount {
			maxCount = c
		}
	}
	if maxCount != retention {
		t.Errorf("max shard blobs per dir = %d, want %d (proving pruning ran, not just under-filled)", maxCount, retention)
	}

	// Meta prefix likewise capped at retention.
	metaKeys := keysWithSuffix(t, mem, "acme", metaExt)
	if len(metaKeys) > retention {
		t.Errorf("meta blobs=%d, want <= %d", len(metaKeys), retention)
	}
}

// TestClusterBackupRestoreE2E is the acceptance test: back up a running cluster →
// stand up a FRESH cluster with the SAME topology and empty DataDirs (simulating
// wiped nodes) → restore from the artifacts → assert every key is readable and the
// topology (an alias in the meta catalog) recovered.
func TestClusterBackupRestoreE2E(t *testing.T) {
	const numShards = 4
	const nKeys = 60
	tc1 := newTestCluster(t, 3, numShards, 3)
	writeKeys(t, tc1, nKeys)

	// Topology state in the meta catalog: an alias that must survive restore.
	metaLeader := metaLeaderNode(t, tc1)
	if err := metaLeader.SetAliases([]AliasAction{{Alias: "acme/live", Canonical: "acme/docs"}}, 5*time.Second); err != nil {
		t.Fatalf("SetAliases: %v", err)
	}

	mem := objstore.NewMemStore()
	backupAllNodes(t, tc1, mem, "acme", 0, time.Now().UTC())

	// Fresh cluster, SAME topology (node IDs n1..n3, same shard count), brand-new
	// empty DataDirs — a full teardown-and-wipe.
	tc2 := newTestCluster(t, 3, numShards, 3)

	// Sanity: the fresh cluster has none of the keys yet.
	if got, err := tc2.client.Call(context.Background(), "get", ops.EncodeKeyArgs([]byte("key-0000"))); err == nil && len(got) > 0 {
		t.Fatalf("fresh cluster unexpectedly already has key-0000 = %q", got)
	}

	// Restore: call on every node; each restores the shards it leads (Raft streams
	// to followers) and the meta leader restores the catalog.
	for _, n := range tc2.nodes {
		if n == nil {
			continue
		}
		if err := n.RestoreFromBackup(context.Background(), mem, "acme", false); err != nil {
			t.Fatalf("RestoreFromBackup(node %s): %v", n.cfg.NodeID, err)
		}
	}

	// Every key must be readable through the restored cluster.
	ctx := context.Background()
	for i := 0; i < nKeys; i++ {
		k := []byte(fmt.Sprintf("key-%04d", i))
		want := fmt.Sprintf("val-%04d", i)
		got := readKeyWithRetry(t, tc2, ctx, k)
		if string(got) != want {
			t.Fatalf("restored key-%04d = %q, want %q", i, got, want)
		}
	}

	// Topology recovered from the meta catalog: the alias resolves on the restored
	// cluster.
	rl := metaLeaderNode(t, tc2)
	if target, ok := rl.ResolveAlias("acme/live"); !ok || target != "acme/docs" {
		t.Fatalf("restored alias acme/live = %q ok=%v, want acme/docs", target, ok)
	}
}

// backupAllPBNodes runs a per-node backup on every node of a PB cluster.
func backupAllPBNodes(t *testing.T, tc *pbTestCluster, obj objstore.ObjectStore, tenant string, retention int, ts time.Time) {
	t.Helper()
	ctx := context.Background()
	for _, n := range tc.nodes {
		if n == nil {
			continue
		}
		if _, err := n.backupOwnedShardsAt(ctx, obj, tenant, retention, ts); err != nil {
			t.Fatalf("PB backupOwnedShardsAt(node %s): %v", n.cfg.NodeID, err)
		}
		if _, err := n.backupMetaCatalogAt(ctx, obj, tenant, retention, ts); err != nil {
			t.Fatalf("PB backupMetaCatalogAt(node %s): %v", n.cfg.NodeID, err)
		}
	}
}

// pbPrimaryIdx returns the index (into tc.nodes) of shard sh's current primary,
// polling until the control plane names one.
func pbPrimaryIdx(t *testing.T, tc *pbTestCluster, sh int) int {
	t.Helper()
	deadline := time.Now().Add(15 * time.Second)
	for time.Now().Before(deadline) {
		primary := tc.nodes[0].meta.FSM.ShardPrimary(sh)
		if primary != "" {
			for i, p := range tc.peers {
				if p.NodeID == primary {
					return i
				}
			}
		}
		time.Sleep(50 * time.Millisecond)
	}
	t.Fatalf("shard %d: no primary named within 15s", sh)
	return -1
}

// pbPutKey writes key→val to its shard's primary, polling the Put until the
// primary's self-lease is granted (no construction-time retry — see the PB harness).
func pbPutKey(t *testing.T, tc *pbTestCluster, numShards int, key, val []byte) {
	t.Helper()
	sh := shardOf(key, numShards)
	idx := pbPrimaryIdx(t, tc, sh)
	args := ops.EncodePutArgs(key, val, 0)
	deadline := time.Now().Add(15 * time.Second)
	var lastErr error
	for time.Now().Before(deadline) {
		_, err := tc.nodes[idx].Call("put", args)
		if err == nil {
			return
		}
		lastErr = err
		time.Sleep(50 * time.Millisecond)
	}
	t.Fatalf("PB put %q never succeeded: %v", key, lastErr)
}

// TestClusterBackupRestoreE2E_PB is the PB acceptance variant: back up a running
// primary-backup cluster → stand up a FRESH PB cluster (same topology) → restore →
// assert every key is readable on its shard's primary AND that the control plane
// advanced each shard's epoch past the restored floor and re-formed the ISR.
func TestClusterBackupRestoreE2E_PB(t *testing.T) {
	const numShards = 4
	const minISR = 2
	tc1 := newPBTestCluster(t, 3, numShards, minISR)

	// Write a spread of keys (each routed to its shard's primary).
	keys := make([][]byte, 0, 16)
	for i := 0; i < 16; i++ {
		k := []byte(fmt.Sprintf("pbkey-%03d", i))
		v := []byte(fmt.Sprintf("pbval-%03d", i))
		pbPutKey(t, tc1, numShards, k, v)
		keys = append(keys, k)
	}

	mem := objstore.NewMemStore()
	backupAllPBNodes(t, tc1, mem, "pb", 0, time.Now().UTC())

	// Fresh PB cluster, same topology, empty DataDirs.
	tc2 := newPBTestCluster(t, 3, numShards, minISR)

	for _, n := range tc2.nodes {
		if n == nil {
			continue
		}
		if err := n.RestoreFromBackup(context.Background(), mem, "pb", false); err != nil {
			t.Fatalf("PB RestoreFromBackup(node %s): %v", n.cfg.NodeID, err)
		}
	}

	// Every key readable on its shard's primary in the restored cluster.
	for i, k := range keys {
		sh := shardOf(k, numShards)
		idx := pbPrimaryIdx(t, tc2, sh)
		want := fmt.Sprintf("pbval-%03d", i)
		var got []byte
		bd := time.Now().Add(5 * time.Second)
		for time.Now().Before(bd) {
			if v, err := tc2.nodes[idx].getShard(sh).Get(k); err == nil && string(v) == want {
				got = v
				break
			}
			time.Sleep(25 * time.Millisecond)
		}
		if string(got) != want {
			t.Fatalf("restored PB %q = %q, want %q", k, got, want)
		}
	}

	// Epoch advanced past the restored floor (the static seed used epoch 1, so the
	// restore re-seeds at >= 2) and the ISR re-formed to the full owner set.
	for sh := 0; sh < numShards; sh++ {
		var epochOK, isrOK bool
		bd := time.Now().Add(5 * time.Second)
		for time.Now().Before(bd) {
			ep := tc2.nodes[0].meta.FSM.ShardEpoch(sh)
			isr := tc2.nodes[0].meta.FSM.ShardISR(sh)
			epochOK = ep >= 2
			isrOK = len(isr) >= minISR
			if epochOK && isrOK {
				break
			}
			time.Sleep(50 * time.Millisecond)
		}
		if !epochOK {
			t.Errorf("shard %d: epoch did not advance past floor (epoch=%d, want >=2)", sh, tc2.nodes[0].meta.FSM.ShardEpoch(sh))
		}
		if !isrOK {
			t.Errorf("shard %d: ISR did not re-form (size=%d, want >=%d)", sh, len(tc2.nodes[0].meta.FSM.ShardISR(sh)), minISR)
		}
	}
}

// readKeyWithRetry reads k through the cluster client, briefly retrying to absorb
// a post-restore leadership settle.
func readKeyWithRetry(t *testing.T, tc *testCluster, ctx context.Context, k []byte) []byte {
	t.Helper()
	deadline := time.Now().Add(5 * time.Second)
	var last []byte
	for time.Now().Before(deadline) {
		got, err := tc.client.Call(ctx, "get", ops.EncodeKeyArgs(k))
		if err == nil && len(got) > 0 {
			return got
		}
		last = got
		time.Sleep(50 * time.Millisecond)
	}
	return last
}

// TestRestoreMissingArtifactFailsLoud (M1) proves the DR-trust fix: if a shard's
// backup artifact is missing, restore FAILS LOUD (ErrRestoreIncomplete) by default
// — it must never silently bring that shard up empty behind a clean success — and
// succeeds only under the explicit allowMissingShards override, leaving exactly the
// missing shard empty while the rest restore.
func TestRestoreMissingArtifactFailsLoud(t *testing.T) {
	const numShards = 4
	const nKeys = 60
	tc1 := newTestCluster(t, 3, numShards, 3)
	writeKeys(t, tc1, nKeys)

	mem := objstore.NewMemStore()
	backupAllNodes(t, tc1, mem, "acme", 0, time.Now().UTC())

	// Delete shard 0's artifact (blob + sidecar) to simulate a lost object.
	latest, err := latestShardBlobs(context.Background(), mem, "acme")
	if err != nil {
		t.Fatalf("latestShardBlobs: %v", err)
	}
	victimKey, ok := latest[0]
	if !ok {
		t.Fatalf("shard 0 has no artifact to delete (test setup): %v", latest)
	}
	if err := mem.Delete(context.Background(), victimKey); err != nil {
		t.Fatalf("delete blob: %v", err)
	}
	_ = mem.Delete(context.Background(), sidecarKeyFor(victimKey)) // best-effort

	tc2 := newTestCluster(t, 3, numShards, 3)

	// Default: restore MUST fail loud with ErrRestoreIncomplete.
	err = tc2.nodes[0].RestoreFromBackup(context.Background(), mem, "acme", false)
	if !errors.Is(err, ErrRestoreIncomplete) {
		t.Fatalf("default restore with a missing shard: got err=%v, want ErrRestoreIncomplete", err)
	}

	// Override: restore succeeds; shard 0 comes up empty, the rest restore.
	for _, n := range tc2.nodes {
		if n == nil {
			continue
		}
		if err := n.RestoreFromBackup(context.Background(), mem, "acme", true); err != nil {
			t.Fatalf("restore with allowMissingShards=true: %v", err)
		}
	}

	ctx := context.Background()
	var sawShard0Key, checkedOther bool
	for i := 0; i < nKeys; i++ {
		k := []byte(fmt.Sprintf("key-%04d", i))
		if shardOf(k, numShards) == 0 {
			// Deleted shard: key must be ABSENT (came up empty).
			sawShard0Key = true
			got, gerr := tc2.client.Call(ctx, "get", ops.EncodeKeyArgs(k))
			if gerr == nil && len(got) > 0 {
				t.Fatalf("key %s (shard 0, artifact deleted) unexpectedly present = %q", k, got)
			}
		} else {
			// A non-deleted shard's keys must still be readable.
			checkedOther = true
			if got := readKeyWithRetry(t, tc2, ctx, k); string(got) != fmt.Sprintf("val-%04d", i) {
				t.Fatalf("key %s (non-deleted shard) = %q, want present", k, got)
			}
		}
	}
	if !sawShard0Key || !checkedOther {
		t.Fatalf("test did not exercise both cases (shard0=%v other=%v)", sawShard0Key, checkedOther)
	}
}

// writeMetaBlobOnly crafts a MetaFSM catalog blob (numShards + member IDs) directly
// into obj, so a topology-guard test can assert the mismatch WITHOUT standing up a
// second real cluster (the guard fires in pre-flight, before any shard blob is read).
func writeMetaBlobOnly(t *testing.T, obj objstore.ObjectStore, tenant string, numShards int, memberIDs []string) {
	t.Helper()
	members := make([]Peer, 0, len(memberIDs))
	for _, id := range memberIDs {
		members = append(members, Peer{NodeID: id})
	}
	data, err := encodeState(State{NumShards: numShards, Members: members})
	if err != nil {
		t.Fatalf("encodeState: %v", err)
	}
	key := metaBlobKey(tenant, time.Now().UTC())
	if err := obj.Put(context.Background(), key, bytes.NewReader(data), int64(len(data))); err != nil {
		t.Fatalf("put meta blob: %v", err)
	}
}

// writeShardArtifact crafts a .shard blob + a .shard.json sidecar recording
// backupNumShards, so a topology-guard test can drive the AUTHORITATIVE sidecar
// shard-count check without a second real cluster. The blob body is a placeholder
// (the count guard reads only the sidecar).
func writeShardArtifact(t *testing.T, obj objstore.ObjectStore, tenant, nodeID string, shardIndex, backupNumShards int) {
	t.Helper()
	key := shardBlobKey(tenant, nodeID, shardIndex, time.Now().UTC())
	if err := obj.Put(context.Background(), key, bytes.NewReader([]byte("x")), 1); err != nil {
		t.Fatalf("put shard blob: %v", err)
	}
	scData, err := json.Marshal(shardSidecar{ShardIndex: shardIndex, NumShards: backupNumShards, AppliedIndex: 1})
	if err != nil {
		t.Fatalf("marshal sidecar: %v", err)
	}
	if err := obj.Put(context.Background(), sidecarKeyFor(key), bytes.NewReader(scData), int64(len(scData))); err != nil {
		t.Fatalf("put sidecar: %v", err)
	}
}

// TestRestoreTopologyMismatch (M3) proves the layered topology guard is honest about
// which error each direction produces:
//
//   - The AUTHORITATIVE guard is the shard SIDECAR's NumShards (config-sourced): a
//     mismatch in EITHER direction → ErrRestoreTopologyMismatch, INCLUDING the
//     empty-trailing-shard modulus hazard the artifact guards miss.
//   - MORE shards than the cluster (a .shard artifact naming an index >= NumShards),
//     even with no sidecar → ErrRestoreTopologyMismatch via the range guard. Holds
//     even when the best-effort catalog NumShards is zero (bootstrap churn).
//   - A POPULATED catalog Members set with a foreign node ID → ErrRestoreTopologyMismatch,
//     via the advisory cross-check (only enforced when Members is non-empty).
//   - A smaller SIDECAR-LESS backup is INDISTINGUISHABLE from missing artifacts, so it
//     honestly surfaces as ErrRestoreIncomplete (safe fail-loud). The dedicated
//     missing-artifact path is covered by TestRestoreMissingArtifactFailsLoud.
//
// The best-effort catalog NumShards is deliberately advisory (it can be zero in a
// healthy cluster after bootstrap churn); the reliable count comes from the sidecar.
func TestRestoreTopologyMismatch(t *testing.T) {
	const numShards = 4

	t.Run("more-shards-artifact-out-of-range", func(t *testing.T) {
		tc := newTestCluster(t, 3, numShards, 3) // node IDs n1..n3, 4 shards
		mem := objstore.NewMemStore()
		writeMetaBlobOnly(t, mem, "acme", numShards, []string{"n1", "n2", "n3"})
		// A .shard artifact naming shard 5 (>= NumShards=4): the range guard rejects it
		// before any blob content is read, so a placeholder body is fine.
		key := shardBlobKey("acme", tc.nodes[0].cfg.NodeID, numShards+1, time.Now().UTC())
		if err := mem.Put(context.Background(), key, bytes.NewReader([]byte("x")), 1); err != nil {
			t.Fatalf("put out-of-range shard blob: %v", err)
		}
		err := tc.nodes[0].RestoreFromBackup(context.Background(), mem, "acme", false)
		if !errors.Is(err, ErrRestoreTopologyMismatch) {
			t.Fatalf("more-shards: got err=%v, want ErrRestoreTopologyMismatch", err)
		}
	})

	t.Run("node-id-set-differs-populated-members", func(t *testing.T) {
		tc := newTestCluster(t, 3, numShards, 3)
		mem := objstore.NewMemStore()
		// Populated Members with a foreign ID (n9): the advisory cross-check enforces it.
		writeMetaBlobOnly(t, mem, "acme", numShards, []string{"n1", "n2", "n9"})
		err := tc.nodes[0].RestoreFromBackup(context.Background(), mem, "acme", false)
		if !errors.Is(err, ErrRestoreTopologyMismatch) {
			t.Fatalf("node-id-differs: got err=%v, want ErrRestoreTopologyMismatch", err)
		}
	})

	t.Run("fewer-shards-surfaces-as-incomplete", func(t *testing.T) {
		tc := newTestCluster(t, 3, numShards, 3)
		mem := objstore.NewMemStore()
		// A catalog claiming 2 shards (matching node IDs) but no shard artifacts: a
		// smaller backup is indistinguishable from missing shards 0..3, so it correctly
		// fails as Incomplete — NOT a topology mismatch.
		writeMetaBlobOnly(t, mem, "acme", 2, []string{"n1", "n2", "n3"})
		err := tc.nodes[0].RestoreFromBackup(context.Background(), mem, "acme", false)
		if !errors.Is(err, ErrRestoreIncomplete) {
			t.Fatalf("fewer-shards: got err=%v, want ErrRestoreIncomplete", err)
		}
	})

	t.Run("authoritative-sidecar-count-catches-empty-trailing", func(t *testing.T) {
		// The modulus hazard the artifact range/completeness guards MISS: a backup
		// from a 6-shard cluster whose trailing shards were empty (left no blobs)
		// restored onto this 4-shard cluster. The sidecar records NumShards=6 (config-
		// sourced, authoritative), so the count guard fails it loud even though the
		// artifact for shard 0 is perfectly in range and would otherwise pass.
		tc := newTestCluster(t, 3, numShards, 3)
		mem := objstore.NewMemStore()
		writeMetaBlobOnly(t, mem, "acme", 6, []string{"n1", "n2", "n3"})
		writeShardArtifact(t, mem, "acme", tc.nodes[0].cfg.NodeID, 0, 6) // sidecar says 6-shard backup
		err := tc.nodes[0].RestoreFromBackup(context.Background(), mem, "acme", false)
		if !errors.Is(err, ErrRestoreTopologyMismatch) {
			t.Fatalf("empty-trailing via sidecar count: got err=%v, want ErrRestoreTopologyMismatch", err)
		}
	})
}

// TestBackupCompletenessAccounting (M2) covers the run-summary accounting two ways:
// (1) the pure SummarizeBackupResults folds an UNCOVERED (hosted, no leader) shard
// into Uncovered rather than silently counting it done; (2) a real healthy-cluster
// backup reports zero uncovered and accounts for every hosted shard.
func TestBackupCompletenessAccounting(t *testing.T) {
	// (1) Pure accounting: a hosted shard with no known leader is UNCOVERED.
	results := []ShardBackupResult{
		{ShardIndex: 0, Hosted: true, Led: true, Key: "k0"},      // backed up
		{ShardIndex: 1, Hosted: true, Led: true, Skipped: true},  // empty
		{ShardIndex: 2, Hosted: true, NoLeaderKnown: true},       // UNCOVERED
		{ShardIndex: 3, Hosted: true},                            // led elsewhere (benign)
		{ShardIndex: 4, Hosted: true, Led: true, Err: errFake()}, // failed
	}
	s := SummarizeBackupResults(results)
	if s.Hosted != 5 || s.Backed != 1 || s.Empty != 1 || s.Uncovered != 1 || s.Failed != 1 {
		t.Fatalf("summary = %+v, want Hosted=5 Backed=1 Empty=1 Uncovered=1 Failed=1", s)
	}

	// (2) A healthy multinode cluster: every hosted shard is covered (led or empty),
	// zero uncovered — proving the accounting integrates and never false-positives.
	const numShards = 4
	tc := newTestCluster(t, 3, numShards, 3)
	writeKeys(t, tc, 40)
	mem := objstore.NewMemStore()
	for _, n := range tc.nodes {
		if n == nil {
			continue
		}
		res, err := n.backupOwnedShardsAt(context.Background(), mem, "acme", 0, time.Now().UTC())
		if err != nil {
			t.Fatalf("backup(node %s): %v", n.cfg.NodeID, err)
		}
		sum := SummarizeBackupResults(res)
		if sum.Hosted != numShards {
			t.Errorf("node %s: hosted=%d, want %d (RF=3 hosts every shard)", n.cfg.NodeID, sum.Hosted, numShards)
		}
		if sum.Uncovered != 0 {
			t.Errorf("node %s: uncovered=%d, want 0 in a healthy cluster", n.cfg.NodeID, sum.Uncovered)
		}
		// Every hosted shard is accounted for by exactly one bucket.
		if sum.Backed+sum.Empty+sum.Failed+sum.Uncovered+ledElsewhere(res) != sum.Hosted {
			t.Errorf("node %s: buckets do not sum to hosted: %+v", n.cfg.NodeID, sum)
		}
	}
}

// ledElsewhere counts hosted shards this node neither led nor found leaderless — a
// benign "covered by another node's leader" skip.
func ledElsewhere(results []ShardBackupResult) int {
	c := 0
	for _, r := range results {
		if r.Hosted && !r.Led && !r.NoLeaderKnown && r.Err == nil {
			c++
		}
	}
	return c
}

func errFake() error { return errors.New("fake backup failure") }
