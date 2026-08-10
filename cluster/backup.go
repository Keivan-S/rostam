// SPDX-License-Identifier: Apache-2.0

package cluster

import (
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"log/slog"
	"path"
	"sort"
	"strings"
	"time"

	hraft "github.com/hashicorp/raft"

	"github.com/rostamlabs/rostam/objstore"
	"github.com/rostamlabs/rostam/raft"
	"github.com/rostamlabs/rostam/shard"
)

// Per-node cluster backup.
//
// A cluster deployment partitions its state across per-shard replication groups
// (Raft or PB), so it cannot be backed up through the single-node
// vector-collections-only path (Server.VectorStore() is nil in cluster mode).
// Instead each node snapshots the shards it LEADS/PRIMARIES — a per-shard
// point-in-time RSST blob (cache KV + vector collections + applied index, via
// shard.Store.BackupSnapshot) — plus the MetaRaft catalog (placement / aliases /
// partitions / collections / epoch). The DR contract is PER-SHARD (= per-key,
// since the data model has no cross-shard transactions) point-in-time; a global
// consistent cut is deferred.
//
// Object-store key layout (reuses objstore + the FS/S3 stores + generalized
// retention):
//
//	<tenant>/node-<nodeID>/shard-<NNNN>/<RFC3339>.shard        the snapshot blob
//	<tenant>/node-<nodeID>/shard-<NNNN>/<RFC3339>.shard.json   its sidecar
//	<tenant>/meta/<RFC3339>.meta                                the MetaRaft catalog
//
// Only the LEADER (Raft) / PRIMARY (PB) of a shard writes it, so a RF>1 shard is
// backed up once, not RF times; likewise only the MetaRaft leader writes the
// catalog. Restore (RestoreFromBackup) requires the SAME shard count and node IDs
// (placement is reproduced deterministically from config); a differing topology
// fails loud (placement remap is deferred).

const (
	shardExt        = ".shard"
	shardSidecarExt = ".shard.json"
	metaExt         = ".meta"
	// backupTimeLayout versions every object key. It is FIXED-WIDTH to the
	// nanosecond so it stays lexicographically==temporally sortable (retention keeps
	// the newest N by a plain descending sort) AND two backups of the same prefix
	// within one second cannot collide (which time.RFC3339's 1s granularity allowed
	// — the second Put would silently overwrite the first). A fixed 9-digit fraction
	// is used deliberately: time.RFC3339Nano trims trailing zeros, which would break
	// the lexical ordering ('.' < 'Z', so a sub-second stamp would sort BEFORE a
	// whole-second one that is actually earlier). With .UTC() the offset is always
	// "Z", so every key is the same width.
	backupTimeLayout = "2006-01-02T15:04:05.000000000Z07:00"
)

// ErrRestoreTopologyMismatch is returned by RestoreFromBackup when the backup's
// shard ARTIFACT SET does not fit this cluster's topology: a shard artifact names an
// index >= this cluster's NumShards (the backup has MORE shards than the cluster —
// caught by the range guard), or the meta catalog carries a POPULATED node-ID set
// that differs from this cluster's peers. Topology is enforced via the artifact set,
// not the best-effort catalog NumShards/Members fields (those can be zero/empty in a
// healthy cluster after bootstrap churn and are only an advisory cross-check). A
// backup with FEWER shards is indistinguishable from missing artifacts and surfaces
// as ErrRestoreIncomplete, not this. Restore to a different topology (placement
// remap) is deferred; this fails loud rather than silently mis-placing data.
var ErrRestoreTopologyMismatch = errors.New("cluster: restore topology mismatch (a backup artifact names a shard outside this cluster's range, or a populated node-ID set differs)")

// ErrRestoreIncomplete is returned by RestoreFromBackup (by default) when the
// backup is missing an artifact for one or more shards in [0, NumShards): a lost or
// never-written .shard object would otherwise bring those shards up EMPTY behind a
// clean "restore complete", silently losing every key that hashes to them. This also
// covers a backup taken from a SMALLER cluster (fewer shards) — indistinguishable
// from missing shards, and equally unsafe, so it correctly fails here rather than as
// a topology mismatch. Pass allowMissingShards=true to override (those shards then
// come up empty, logged loudly) — an explicit operator decision, never a silent
// default.
var ErrRestoreIncomplete = errors.New("cluster: restore incomplete — one or more shards have no backup artifact (pass allowMissingShards to override)")

// shardSidecar is the JSON metadata written next to each shard snapshot blob. It
// records the applied index the blob was taken at, the control-plane epoch (a
// FLOOR for PB restore — the control plane re-seeds at a strictly higher epoch),
// the replication mode, the shard/node identity, and the backup cluster's shard
// count. NumShards is the AUTHORITATIVE topology source for restore's shard-count
// guard: it is taken from n.cfg.NumShards (deterministic config, always correct),
// NOT from the meta FSM — whose NumShards/Members can be transiently 0/empty under
// the documented bootstrap OpSetMembers-commit-loss fragility (see node.go), which
// would otherwise make the topology check flake.
type shardSidecar struct {
	AppliedIndex uint64 `json:"appliedIndex"`
	Epoch        uint64 `json:"epoch"`
	Mode         string `json:"mode"`
	ShardIndex   int    `json:"shardIndex"`
	NodeID       string `json:"nodeID"`
	NumShards    int    `json:"numShards"`
}

// ShardBackupResult reports the outcome of one HOSTED shard in a backup run. A
// shard this node does not host produces no result (it is not this node's
// responsibility). The states are mutually exclusive per hosted shard:
//   - Led + Key set:            this node led it and wrote the blob.
//   - Led + Skipped:            this node led it but it was empty (nothing applied).
//   - Led + Err:                this node led it and the backup failed.
//   - !Led + NoLeaderKnown:     hosted but no leader/primary is known — so NO node
//     backs it up this cycle (an UNCOVERED shard: the completeness gap M2 surfaces).
//   - !Led + !NoLeaderKnown:    hosted but led elsewhere — covered by the leader's
//     backup (a benign, expected skip).
type ShardBackupResult struct {
	ShardIndex    int
	Key           string // "" if skipped/failed/not-led
	Size          int64
	AppliedIndex  uint64
	Hosted        bool  // this node hosts (owns) the shard
	Led           bool  // this node is the leader/primary (it attempted the backup)
	Skipped       bool  // led, but the shard had nothing to back up (empty)
	NoLeaderKnown bool  // hosted, NOT led, and no leader/primary is known → uncovered
	Err           error // non-nil on failure; does NOT abort the other shards
}

// shardHasLeader reports whether a leader (Raft) / primary (PB) is currently KNOWN
// for shard i from this node's view. Used to distinguish a benign not-led skip (a
// leader exists elsewhere and will back the shard up) from an UNCOVERED shard (no
// leader/primary at all, so no node backs it up this cycle — the M2 gap).
func (n *Node) shardHasLeader(i int, st *shard.Store) bool {
	if n.clusterMode() == ReplicationModePB {
		return n.pbControl != nil && n.pbControl.Primary(i) != ""
	}
	return st.LeaderAddr() != ""
}

// shardDirPrefix is the per-(node,shard) key prefix that scopes both a snapshot's
// key and its retention List. shard-%04d matches the on-disk DataDir layout.
func shardDirPrefix(tenant, nodeID string, shardIndex int) string {
	return path.Join(tenant, "node-"+nodeID, fmt.Sprintf("shard-%04d", shardIndex)) + "/"
}

func shardBlobKey(tenant, nodeID string, shardIndex int, ts time.Time) string {
	return shardDirPrefix(tenant, nodeID, shardIndex) + ts.UTC().Format(backupTimeLayout) + shardExt
}

// sidecarKeyFor maps a snapshot key to its sidecar key by swapping .shard for
// .shard.json (both share the <tenant>/node/shard/<ts> stem).
func sidecarKeyFor(shardKey string) string {
	return strings.TrimSuffix(shardKey, shardExt) + shardSidecarExt
}

func metaDirPrefix(tenant string) string { return path.Join(tenant, "meta") + "/" }

func metaBlobKey(tenant string, ts time.Time) string {
	return metaDirPrefix(tenant) + ts.UTC().Format(backupTimeLayout) + metaExt
}

// clusterMode returns the effective replication mode ("raft" or "pb").
func (n *Node) clusterMode() string {
	if n.cfg.ReplicationMode == ReplicationModePB {
		return ReplicationModePB
	}
	return ReplicationModeRaft
}

// leadsShard reports whether this node is the LEADER (Raft) / PRIMARY (PB) of
// shard i — the single owner that should write the backup so a RF>1 shard is not
// copied RF times.
func (n *Node) leadsShard(i int, st *shard.Store) bool {
	if n.clusterMode() == ReplicationModePB {
		return n.pbControl != nil && n.pbControl.Primary(i) == n.cfg.NodeID
	}
	return st.IsLeader()
}

// BackupOwnedShards snapshots every shard this node LEADS/PRIMARIES to obj under
// the <tenant>/node-<id>/shard-NNNN/ layout, writing a .shard blob + a .shard.json
// sidecar per shard and (when retention > 0) pruning older snapshots oldest-first.
// A shard this node does not lead, or an empty shard (nothing applied), is
// skipped. Per-shard failures are isolated (captured in the result, the run
// continues); the joined error is nil iff every attempted shard succeeded.
func (n *Node) BackupOwnedShards(ctx context.Context, obj objstore.ObjectStore, tenant string, retention int) ([]ShardBackupResult, error) {
	return n.backupOwnedShardsAt(ctx, obj, tenant, retention, time.Now().UTC())
}

// backupOwnedShardsAt is BackupOwnedShards with an explicit, caller-supplied
// timestamp so tests can pin distinct stamps (retention pruning is then fully
// deterministic — the same ts always yields the same key).
func (n *Node) backupOwnedShardsAt(ctx context.Context, obj objstore.ObjectStore, tenant string, retention int, ts time.Time) ([]ShardBackupResult, error) {
	mode := n.clusterMode()
	shards := n.snapshotShards()
	var results []ShardBackupResult
	var errs []error
	for i, st := range shards {
		if st == nil {
			continue // not hosted on this node — not this node's responsibility
		}
		if !n.leadsShard(i, st) {
			// Hosted but not the leader/primary. If a leader/primary is KNOWN, the
			// shard is covered by that node's backup — a benign skip. If NONE is known
			// (mid-election, or PB primary=="" during failover), NO node backs it up
			// this cycle: record it UNCOVERED so the run summary can WARN (M2).
			results = append(results, ShardBackupResult{
				ShardIndex:    i,
				Hosted:        true,
				NoLeaderKnown: !n.shardHasLeader(i, st),
			})
			continue
		}
		res := ShardBackupResult{ShardIndex: i, Hosted: true, Led: true}
		data, appliedIndex, err := st.BackupSnapshot(ctx)
		if errors.Is(err, shard.ErrNothingToBackup) {
			res.Skipped = true
			results = append(results, res)
			continue
		}
		if err != nil {
			res.Err = fmt.Errorf("shard %d: snapshot: %w", i, err)
			results = append(results, res)
			errs = append(errs, res.Err)
			continue
		}
		var epoch uint64
		if n.pbControl != nil {
			epoch = n.pbControl.Epoch(i)
		}
		key := shardBlobKey(tenant, n.cfg.NodeID, i, ts)
		if err := obj.Put(ctx, key, bytes.NewReader(data), int64(len(data))); err != nil {
			res.Err = fmt.Errorf("shard %d: put %q: %w", i, key, err)
			results = append(results, res)
			errs = append(errs, res.Err)
			continue
		}
		res.Key = key
		res.Size = int64(len(data))
		res.AppliedIndex = appliedIndex

		sidecar := shardSidecar{AppliedIndex: appliedIndex, Epoch: epoch, Mode: mode, ShardIndex: i, NodeID: n.cfg.NodeID, NumShards: n.cfg.NumShards}
		scData, err := json.Marshal(sidecar)
		if err != nil {
			res.Err = fmt.Errorf("shard %d: marshal sidecar: %w", i, err)
			results = append(results, res)
			errs = append(errs, res.Err)
			continue
		}
		scKey := sidecarKeyFor(key)
		if err := obj.Put(ctx, scKey, bytes.NewReader(scData), int64(len(scData))); err != nil {
			res.Err = fmt.Errorf("shard %d: put sidecar %q: %w", i, scKey, err)
			results = append(results, res)
			errs = append(errs, res.Err)
			continue
		}

		if retention > 0 {
			if err := pruneBackups(ctx, obj, shardDirPrefix(tenant, n.cfg.NodeID, i), shardExt, shardSidecarExt, retention); err != nil {
				res.Err = fmt.Errorf("shard %d: prune: %w", i, err)
				errs = append(errs, res.Err)
			}
		}
		results = append(results, res)
	}
	return results, errors.Join(errs...)
}

// BackupRunSummary is a completeness accounting of one node's backup run over the
// shards it HOSTS (M2): it makes a chronically un-backed shard visible in the run
// log BEFORE a restore needs it, and distinguishes a legitimately-empty shard from
// one skipped because no leader/primary existed.
type BackupRunSummary struct {
	Hosted    int // shards this node hosts (the denominator)
	Backed    int // led + a blob written
	Empty     int // led, but legitimately empty (nothing applied)
	Uncovered int // hosted, not led, and NO leader/primary known → NOBODY backed it up
	Failed    int // led, but the backup errored
}

// SummarizeBackupResults folds a per-shard result slice into a BackupRunSummary.
func SummarizeBackupResults(results []ShardBackupResult) BackupRunSummary {
	var s BackupRunSummary
	for _, r := range results {
		if r.Hosted {
			s.Hosted++
		}
		switch {
		case r.Err != nil:
			s.Failed++
		case r.NoLeaderKnown:
			s.Uncovered++
		case r.Skipped:
			s.Empty++
		case r.Led:
			s.Backed++
		}
	}
	return s
}

// BackupMetaCatalog writes the MetaRaft catalog (placement / aliases / partitions
// / collections / epoch) to obj at <tenant>/meta/<RFC3339>.meta, then prunes older
// meta snapshots to retention (when > 0). Only the MetaRaft LEADER writes it (the
// catalog is identical on every node — gating avoids RF× copies); a follower, or a
// single-node cluster with no meta group, returns ("", nil) as a benign skip.
func (n *Node) BackupMetaCatalog(ctx context.Context, obj objstore.ObjectStore, tenant string, retention int) (string, error) {
	return n.backupMetaCatalogAt(ctx, obj, tenant, retention, time.Now().UTC())
}

func (n *Node) backupMetaCatalogAt(ctx context.Context, obj objstore.ObjectStore, tenant string, retention int, ts time.Time) (string, error) {
	if n.meta == nil || n.meta.Raft.State() != hraft.Leader {
		return "", nil // not the meta leader (or single-node, no meta group): skip
	}
	data, err := n.meta.FSM.SnapshotBytes()
	if err != nil {
		return "", fmt.Errorf("cluster: meta snapshot bytes: %w", err)
	}
	key := metaBlobKey(tenant, ts)
	if err := obj.Put(ctx, key, bytes.NewReader(data), int64(len(data))); err != nil {
		return "", fmt.Errorf("cluster: put meta %q: %w", key, err)
	}
	if retention > 0 {
		if err := pruneBackups(ctx, obj, metaDirPrefix(tenant), metaExt, "", retention); err != nil {
			return key, fmt.Errorf("cluster: prune meta: %w", err)
		}
	}
	return key, nil
}

// pruneBackups enforces retention over one prefix: it lists the prefix, keeps the
// newest `retention` objects whose key ends in ext (RFC3339 keys sort
// chronologically, so a descending sort is newest-first), and deletes the rest.
// When sidecarExt is non-empty, each pruned object's sibling sidecar (same stem,
// ext→sidecarExt) is deleted alongside it. ErrNotFound is tolerated on delete.
func pruneBackups(ctx context.Context, obj objstore.ObjectStore, prefix, ext, sidecarExt string, retention int) error {
	infos, err := obj.List(ctx, prefix)
	if err != nil {
		return fmt.Errorf("list %q: %w", prefix, err)
	}
	keys := make([]string, 0, len(infos))
	for _, in := range infos {
		if strings.HasSuffix(in.Key, ext) {
			keys = append(keys, in.Key)
		}
	}
	if len(keys) <= retention {
		return nil
	}
	sort.Sort(sort.Reverse(sort.StringSlice(keys))) // newest first
	for _, key := range keys[retention:] {
		if err := obj.Delete(ctx, key); err != nil && !errors.Is(err, objstore.ErrNotFound) {
			return fmt.Errorf("delete %q: %w", key, err)
		}
		if sidecarExt != "" {
			scKey := strings.TrimSuffix(key, ext) + sidecarExt
			if err := obj.Delete(ctx, scKey); err != nil && !errors.Is(err, objstore.ErrNotFound) {
				return fmt.Errorf("delete %q: %w", scKey, err)
			}
		}
	}
	return nil
}

// tsFromKey extracts the RFC3339 basename (minus ext) from an object key so blobs
// under DIFFERENT node-<id> dirs are compared by their TIMESTAMP (the globally
// newest snapshot of a shard wins), not by the full key (whose node segment would
// otherwise dominate the sort).
func tsFromKey(key, ext string) string {
	base := path.Base(key)
	return strings.TrimSuffix(base, ext)
}

// latestShardBlobs scans every backup under <tenant>/ and returns, per shard
// index, the object key of the GLOBALLY-latest .shard snapshot (across all
// node-<id> dirs — leadership may have moved between runs, so a shard's freshest
// blob can live under any node's prefix). Keys are grouped by the shard-NNNN path
// segment and ranked by the RFC3339 basename.
func latestShardBlobs(ctx context.Context, obj objstore.ObjectStore, tenant string) (map[int]string, error) {
	infos, err := obj.List(ctx, tenant+"/")
	if err != nil {
		return nil, fmt.Errorf("cluster: restore list %q: %w", tenant, err)
	}
	latest := make(map[int]string)
	latestTS := make(map[int]string)
	for _, in := range infos {
		if !strings.HasSuffix(in.Key, shardExt) {
			continue
		}
		shardIdx, ok := shardIndexFromKey(in.Key)
		if !ok {
			continue
		}
		ts := tsFromKey(in.Key, shardExt)
		if cur, seen := latestTS[shardIdx]; !seen || ts > cur {
			latestTS[shardIdx] = ts
			latest[shardIdx] = in.Key
		}
	}
	return latest, nil
}

// shardIndexFromKey parses the shard index out of a key's "shard-NNNN" path
// segment. Returns ok=false if no such segment is present.
func shardIndexFromKey(key string) (int, bool) {
	for _, seg := range strings.Split(key, "/") {
		if strings.HasPrefix(seg, "shard-") {
			var idx int
			if _, err := fmt.Sscanf(seg, "shard-%d", &idx); err == nil {
				return idx, true
			}
		}
	}
	return 0, false
}

// latestMetaBlob returns the key of the newest <tenant>/meta/<ts>.meta object, or
// ok=false when no meta backup exists.
func latestMetaBlob(ctx context.Context, obj objstore.ObjectStore, tenant string) (string, bool, error) {
	infos, err := obj.List(ctx, metaDirPrefix(tenant))
	if err != nil {
		return "", false, fmt.Errorf("cluster: restore list meta: %w", err)
	}
	latest := ""
	for _, in := range infos {
		if strings.HasSuffix(in.Key, metaExt) && in.Key > latest {
			latest = in.Key
		}
	}
	if latest == "" {
		return "", false, nil
	}
	return latest, true, nil
}

// fetchBlob reads an object fully into memory.
func fetchBlob(ctx context.Context, obj objstore.ObjectStore, key string) ([]byte, error) {
	rc, err := obj.Get(ctx, key)
	if err != nil {
		return nil, fmt.Errorf("get %q: %w", key, err)
	}
	defer func() { _ = rc.Close() }()
	buf := new(bytes.Buffer)
	if _, err := buf.ReadFrom(rc); err != nil {
		return nil, fmt.Errorf("read %q: %w", key, err)
	}
	return buf.Bytes(), nil
}

// fetchSidecar reads and decodes a shard blob's sidecar. A missing sidecar
// (pre-sidecar backup, or a partial run) yields a zero shardSidecar and ok=false.
func fetchSidecar(ctx context.Context, obj objstore.ObjectStore, shardKey string) (shardSidecar, bool, error) {
	rc, err := obj.Get(ctx, sidecarKeyFor(shardKey))
	if err != nil {
		if errors.Is(err, objstore.ErrNotFound) {
			return shardSidecar{}, false, nil
		}
		return shardSidecar{}, false, err
	}
	defer func() { _ = rc.Close() }()
	buf := new(bytes.Buffer)
	if _, err := buf.ReadFrom(rc); err != nil {
		return shardSidecar{}, false, err
	}
	var sc shardSidecar
	if err := json.Unmarshal(buf.Bytes(), &sc); err != nil {
		return shardSidecar{}, false, fmt.Errorf("decode sidecar: %w", err)
	}
	return sc, true, nil
}

// backupShardCount reports the backup cluster's shard count, read from any shard
// sidecar's NumShards field (recorded from n.cfg.NumShards at backup time — the
// AUTHORITATIVE, config-sourced topology signal, unlike the best-effort meta
// catalog). Sidecars are read in shard-index order for determinism; the first with
// a positive NumShards wins (every sidecar from one backup agrees). ok=false when no
// shard artifact/sidecar exists, or every sidecar predates the NumShards field
// (value 0) — the caller then falls back to the range + completeness guards.
func backupShardCount(ctx context.Context, obj objstore.ObjectStore, latest map[int]string) (int, bool, error) {
	idxs := make([]int, 0, len(latest))
	for i := range latest {
		idxs = append(idxs, i)
	}
	sort.Ints(idxs)
	for _, i := range idxs {
		sc, ok, err := fetchSidecar(ctx, obj, latest[i])
		if err != nil {
			return 0, false, err
		}
		if ok && sc.NumShards > 0 {
			return sc.NumShards, true, nil
		}
	}
	return 0, false, nil
}

// RestoreFromBackup restores this node's slice of a cluster DR backup produced by
// BackupOwnedShards + BackupMetaCatalog. It is designed to run on a FRESH cluster
// brought up with the SAME topology (same shard count AND same node-ID set) and to
// be invoked on EVERY node. It runs a pre-flight that FAILS LOUD before touching any
// state, then does the mode-specific install:
//
//	PRE-FLIGHT (every node — each sees the same object store, so all agree):
//	  - Topology (M3): the AUTHORITATIVE guard is the shard SIDECAR's NumShards
//	    (recorded from config, reliable) — it must EQUAL this cluster's NumShards, and
//	    a mismatch in EITHER direction fails loud (ErrRestoreTopologyMismatch),
//	    including the empty-trailing-shard modulus hazard the artifact guards miss. As
//	    belt-and-suspenders (sidecar-less/legacy backups): the range guard rejects any
//	    artifact naming a shard >= NumShards ("backup has MORE shards" →
//	    ErrRestoreTopologyMismatch), and the M1 completeness check below rejects any
//	    shard in range with no artifact ("backup has FEWER shards / a shard is missing"
//	    → ErrRestoreIncomplete; a smaller sidecar-less backup is indistinguishable from missing
//	    shards, so it honestly fails as Incomplete). As an ADVISORY cross-check, a
//	    POPULATED catalog Members set whose node IDs differ from this cluster's peers
//	    still → ErrRestoreTopologyMismatch; an empty Members set is skipped (logged).
//	  - Completeness (M1): EVERY shard in [0,NumShards) must have a backup artifact.
//	    A missing .shard (lost object, botched prune, S3 lifecycle) would otherwise
//	    bring that shard up EMPTY behind a clean "restore complete" — silent key loss.
//	    ErrRestoreIncomplete by default; allowMissingShards is the explicit override
//	    (missing shards then come up empty, logged loudly per shard).
//
//	INSTALL:
//	  1. META (MetaRaft leader only): install the catalog blob via hashicorp/raft's
//	     Restore (DR primitive), faulting it onto the meta followers.
//	  2. SHARD DATA (per hosted shard): Raft installs on the shard LEADER
//	     (raft.Restore streams to followers; a follower's attempt is a benign
//	     NotLeader skip); PB has no raft to stream, so EVERY owner installs directly.
//	  3. PB EPOCH ADVANCE (MetaRaft leader only): bump each restored shard past its
//	     epoch FLOOR and re-form the ISR, so the control plane re-seeds at a strictly
//	     higher generation than any epoch the restored data was written under.
func (n *Node) RestoreFromBackup(ctx context.Context, obj objstore.ObjectStore, tenant string, allowMissingShards bool) error {
	latest, err := latestShardBlobs(ctx, obj, tenant)
	if err != nil {
		return err
	}

	// ---- Pre-flight: AUTHORITATIVE shard-count guard (M3, BOTH directions) ----
	// The sidecar records the backup cluster's NumShards from n.cfg.NumShards
	// (deterministic config, always correct), so it is the RELIABLE topology signal
	// — unlike the meta catalog's NumShards, which can be transiently 0 under the
	// bootstrap OpSetMembers-commit-loss fragility. A mismatch in EITHER direction is
	// fatal, and this crucially catches the EMPTY-TRAILING-SHARD case the artifact
	// range/completeness guards alone MISS: a backup from a 6-shard cluster whose
	// shards 4-5 were empty (no blobs) restored onto a 4-shard cluster passes range
	// (no out-of-range blob) and completeness (0-3 present), yet silently changes the
	// key→shard modulus (6→4), mis-routing every key. The sidecar count fails it loud.
	if bc, ok, serr := backupShardCount(ctx, obj, latest); serr != nil {
		return serr
	} else if ok && bc != n.cfg.NumShards {
		return fmt.Errorf("%w: backup shard count %d (from sidecar) != cluster shard count %d",
			ErrRestoreTopologyMismatch, bc, n.cfg.NumShards)
	}

	// ---- Pre-flight: advisory catalog cross-check + meta blob fetch ----
	// metaData is reused for the leader's meta restore below (fetched once).
	var metaData []byte
	if n.meta != nil {
		metaKey, ok, ferr := latestMetaBlob(ctx, obj, tenant)
		if ferr != nil {
			return ferr
		}
		if !ok {
			// A multi-node cluster backup ALWAYS writes a meta catalog (the meta leader
			// does). Its absence means an incomplete/foreign backup — refuse rather than
			// come up with a fresh-but-unverified topology.
			return fmt.Errorf("%w: no meta catalog artifact found under %s", ErrRestoreIncomplete, metaDirPrefix(tenant))
		}
		metaData, err = fetchBlob(ctx, obj, metaKey)
		if err != nil {
			return err
		}
		st, derr := decodeState(metaData)
		if derr != nil {
			return fmt.Errorf("cluster: restore decode meta catalog %q: %w", metaKey, derr)
		}
		// Topology cross-check against the catalog is ADVISORY: MetaState.NumShards and
		// Members are BEST-EFFORT fields. The bootstrap OpSetMembers entry can be lost
		// to leadership churn (see meta_fsm.go OpSetPlacement self-heal), leaving
		// NumShards=0 / Members empty in a perfectly good cluster — NumShards then only
		// self-heals incrementally via OpSetPlacement. A hard compare on those fields
		// fires spuriously. The AUTHORITATIVE topology signal is the shard ARTIFACT SET,
		// enforced below and covering BOTH directions: the range guard rejects a backup
		// naming MORE shards than the cluster; the M1 completeness check rejects one with
		// FEWER (a smaller backup is indistinguishable from missing shards — both
		// honestly surface as ErrRestoreIncomplete). So a NumShards disagreement is only
		// logged, never fatal.
		if st.NumShards != n.cfg.NumShards {
			slog.Info("restore NOTE: meta catalog NumShards disagrees with cluster NumShards (best-effort field; relying on artifact set)", "component", "cluster", "catalog_num_shards", st.NumShards, "cluster_num_shards", n.cfg.NumShards)
		}
		// Node-ID set: enforce ONLY when Members is populated (a real signal). An empty
		// Members means the bootstrap OpSetMembers was lost to churn — skip rather than
		// spuriously reject a legitimate same-topology restore. A genuine mismatch with a
		// populated list still fails loud with ErrRestoreTopologyMismatch.
		if len(st.Members) > 0 {
			if !nodeIDSetsMatch(st.Members, n.cfg.Peers) {
				return fmt.Errorf("%w: backup node IDs %v != cluster node IDs %v",
					ErrRestoreTopologyMismatch, sortedMemberIDs(st.Members), sortedPeerIDs(n.cfg.Peers))
			}
		} else {
			slog.Info("restore NOTE: meta catalog Members empty (best-effort field; skipping node-ID cross-check, relying on artifact set)", "component", "cluster")
		}
	}

	// Range guard: the authoritative "backup has MORE shards than the cluster" check
	// (artifact-derived, so it holds even when the best-effort catalog NumShards is
	// zero, and also covers a single-node cluster.Node with no meta catalog). No
	// artifact may name a shard outside [0, NumShards).
	for shardIdx := range latest {
		if shardIdx < 0 || shardIdx >= n.cfg.NumShards {
			return fmt.Errorf("%w: backup has shard %d, cluster has %d shards",
				ErrRestoreTopologyMismatch, shardIdx, n.cfg.NumShards)
		}
	}

	// ---- Pre-flight: completeness (M1) ----
	var missing []int
	for i := 0; i < n.cfg.NumShards; i++ {
		if _, ok := latest[i]; !ok {
			missing = append(missing, i)
		}
	}
	if len(missing) > 0 {
		for _, i := range missing {
			action := "aborting"
			if allowMissingShards {
				action = "restoring EMPTY (allowMissingShards)"
			}
			slog.Warn("restore: NO BACKUP FOUND for shard", "component", "cluster", "shard", i, "action", action)
		}
		if !allowMissingShards {
			return fmt.Errorf("%w: shards %v have no .shard artifact", ErrRestoreIncomplete, missing)
		}
	}

	// ---- 1. Meta restore (meta leader) using the pre-fetched blob ----
	if metaData != nil && n.meta.Raft.State() == hraft.Leader {
		// index: the catalog's own recorded frontier so the restore hole sits above it.
		idx := metaFrontierFromBlob(metaData)
		meta := &hraft.SnapshotMeta{Version: 1, Index: idx, Term: 1, Size: int64(len(metaData))}
		if err := n.meta.Raft.Restore(meta, bytes.NewReader(metaData), 30*time.Second); err != nil {
			return fmt.Errorf("cluster: restore meta catalog: %w", err)
		}
		slog.Info("restored meta catalog", "component", "cluster", "shards", n.cfg.NumShards)
	}

	// ---- 2. Shard data ----
	shards := n.snapshotShards()
	pbMode := n.clusterMode() == ReplicationModePB
	for i, st := range shards {
		if st == nil {
			continue // not hosted on this node
		}
		key, ok := latest[i]
		if !ok {
			continue // missing artifact + allowMissingShards → leave empty (warned above)
		}
		data, err := fetchBlob(ctx, obj, key)
		if err != nil {
			return err
		}
		sc, _, err := fetchSidecar(ctx, obj, key)
		if err != nil {
			return fmt.Errorf("cluster: restore shard %d sidecar: %w", i, err)
		}
		if err := st.RestoreSnapshot(ctx, data, sc.AppliedIndex); err != nil {
			// Raft followers legitimately are-not-leader; the leader installs and
			// replicates to them, so skip rather than fail.
			if !pbMode && errors.Is(err, raft.ErrNotLeader) {
				continue
			}
			return fmt.Errorf("cluster: restore shard %d: %w", i, err)
		}
		slog.Info("restored shard", "component", "cluster", "shard", i, "key", key, "applied_index", sc.AppliedIndex)
	}

	// 3. PB epoch advance past the restored floor + ISR re-form (meta leader only).
	// The restored epoch is only a FLOOR: the control plane re-seeds the shard at a
	// STRICTLY HIGHER generation so no epoch the restored data was written under can
	// ever be re-used (belt-and-suspenders — a full-cluster DR restore has no
	// surviving stragglers, but the monotonic bump keeps the fencing invariant). The
	// re-seed carries the full ISR (all owners) in the SAME entry as the epoch
	// bump — ApplySetShardSeed, not epoch-then-ISR. Two entries would leave an
	// intermediate committed state with a singleton ISR that a restored primary
	// could read and ack against alone (see ApplySetShardSeed / bootstrapPBShardControl).
	if pbMode && n.meta != nil && n.meta.Raft.State() == hraft.Leader {
		for i := range shards {
			key, ok := latest[i]
			if !ok {
				continue
			}
			sc, _, err := fetchSidecar(ctx, obj, key)
			if err != nil {
				return fmt.Errorf("cluster: restore shard %d sidecar (epoch): %w", i, err)
			}
			floor := sc.Epoch
			if metaFloor := n.pbControl.Epoch(i); metaFloor > floor {
				floor = metaFloor
			}
			newEpoch := floor + 1
			primary := n.meta.FSM.ShardPrimary(i)
			if primary == "" {
				primary = n.primaryForShard(i)
			}
			if err := n.meta.ApplySetShardSeed(i, newEpoch, primary, n.shardOwners(i), 5*time.Second); err != nil {
				return fmt.Errorf("cluster: restore shard %d epoch advance + ISR re-form: %w", i, err)
			}
		}
	}
	return nil
}

// shardOwners returns the reproduced placement owner set for shard i (the ISR the
// PB restore re-forms at the advanced epoch).
func (n *Node) shardOwners(i int) []string {
	if i >= 0 && i < len(n.placement) {
		return n.placement[i]
	}
	return nil
}

// primaryForShard picks a default primary for a PB shard from its reproduced
// placement (the first owner), used when the restored meta has no recorded
// primary for the shard.
func (n *Node) primaryForShard(i int) string {
	if i >= 0 && i < len(n.placement) && len(n.placement[i]) > 0 {
		return n.placement[i][0]
	}
	return n.cfg.NodeID
}

// metaFrontierFromBlob decodes the command frontier (State.LastIndex) recorded in
// a MetaFSM catalog blob, so the meta restore leaves its log hole above it. A blob
// that fails to decode (should not happen for a well-formed backup) yields 0,
// which raft maxes against the current index anyway.
func metaFrontierFromBlob(data []byte) uint64 {
	st, err := decodeState(data)
	if err != nil {
		return 0
	}
	return st.LastIndex
}

// nodeIDSetsMatch reports whether the backed-up member set and this cluster's peer
// set name the SAME node IDs (order-independent). It is the node-ID half of the
// same-topology restore guard (M3): a restore onto a cluster with different node
// IDs would mis-place shards (the reproduced placement is keyed by node ID).
func nodeIDSetsMatch(members, peers []Peer) bool {
	if len(members) != len(peers) {
		return false
	}
	set := make(map[string]struct{}, len(members))
	for _, m := range members {
		set[m.NodeID] = struct{}{}
	}
	for _, p := range peers {
		if _, ok := set[p.NodeID]; !ok {
			return false
		}
	}
	return len(set) == len(peers) // guards duplicate IDs on one side
}

// sortedMemberIDs / sortedPeerIDs render a member/peer set as a sorted node-ID
// slice for a deterministic mismatch error message.
func sortedMemberIDs(members []Peer) []string {
	ids := make([]string, 0, len(members))
	for _, m := range members {
		ids = append(ids, m.NodeID)
	}
	sort.Strings(ids)
	return ids
}

func sortedPeerIDs(peers []Peer) []string {
	ids := make([]string, 0, len(peers))
	for _, p := range peers {
		ids = append(ids, p.NodeID)
	}
	sort.Strings(ids)
	return ids
}
