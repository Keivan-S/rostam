// SPDX-License-Identifier: Apache-2.0

package main

import (
	"context"
	"errors"
	"fmt"
	"log/slog"
	"net"
	"net/url"
	"os"
	"sync"
	"time"

	"github.com/rostamlabs/rostam/backup"
	"github.com/rostamlabs/rostam/httpapi"
	"github.com/rostamlabs/rostam/objstore"
	"github.com/rostamlabs/rostam/vector"
)

// S3 cold-tiering / backup configuration assembled from the -backup-*/-cold-tier-*
// /-s3-path-style flags. It is the OPT-IN object-storage tier: a periodic S3
// backup cron AND/OR an idle cold-tier sweeper, both driven from the cmd layer
// (the ONLY place time.Now is read — the backup and engine layers stay
// deterministic, taking the wall clock as an argument).
//
// FAIL-LOUD / OPT-IN contract (mirrors the TLS/RBAC flag validation):
//   - Nothing configured (-backup-bucket empty AND -cold-tier-after 0) ⇒ a nil
//     plan: NO objstore is constructed, NO goroutine is started, behavior is
//     byte-identical to a server built without this feature.
//   - A bucket or cold-tier IS requested but region/creds are missing ⇒ a FATAL
//     error at startup (never a silent degrade). Credentials are NEVER echoed in
//     any error or log line.
type tierFlags struct {
	Bucket        string
	Endpoint      string
	Region        string
	Interval      time.Duration // backup cron period; 0 = backup cron off
	Retention     int
	Tenant        string
	ColdTierAfter time.Duration // idle-evict threshold; 0 = sweeper off
	PathStyle     bool
}

// tierPlan is the validated, ready-to-run object-storage plan. A nil *tierPlan
// means "nothing configured" (the opt-in default — no objstore, no goroutines).
type tierPlan struct {
	// Store is the single objstore.S3Store shared by the backup cron and the
	// cold-tier sweeper (one client, one set of creds).
	Store objstore.ObjectStore
	// Tenant is the object-key namespace prefix for both backup and cold-tier
	// snapshots.
	Tenant string
	// BackupInterval > 0 enables the periodic backup cron.
	BackupInterval time.Duration
	// Retention is the keep-last-N applied by each backup run (0 = keep all).
	Retention int
	// ColdTierAfter > 0 enables the idle cold-tier sweeper.
	ColdTierAfter time.Duration
}

// buildTierPlan validates the object-storage flags and, when configured, builds
// the single shared S3 store. It is the testable construct+validate seam: it
// returns (nil, nil) for the opt-in "nothing configured" case and (nil, err) for
// a fail-loud misconfiguration, so main() can surface the error as log.Fatalf
// WITHOUT this function ever calling os.Exit (the unit tests assert both the
// no-op and the fatal paths here).
//
// Creds are read from the AWS_* environment (AWS_ACCESS_KEY_ID /
// AWS_SECRET_ACCESS_KEY / AWS_SESSION_TOKEN) via objstore's env fallback; this
// function only checks for their PRESENCE (never logs their value).
func buildTierPlan(f tierFlags) (*tierPlan, error) {
	configured := f.Bucket != "" || f.ColdTierAfter > 0
	if !configured {
		// Opt-in: nothing requested ⇒ no objstore, no driver. Byte-identical to a
		// server built without the feature.
		return nil, nil
	}

	// A backup cron requires a bucket to write into: an interval without a bucket
	// is a misconfig (it would have nowhere to put snapshots). Fail loud rather
	// than silently disable a requested backup.
	if f.Interval > 0 && f.Bucket == "" {
		return nil, errors.New("-backup-interval set but -backup-bucket is empty; backup has no destination")
	}
	// Both the backup cron and the cold-tier sweeper need a bucket to evict/back
	// up into. A cold-tier-after without a bucket cannot offload anything.
	if f.Bucket == "" {
		return nil, errors.New("-cold-tier-after set but -backup-bucket is empty; cold tiering has no object store to evict into")
	}
	if f.Region == "" {
		return nil, errors.New("-backup-bucket/-cold-tier-after set but -backup-region is empty; an S3 region is required for request signing")
	}
	// Credentials MUST be present (explicit env). We check the env directly so the
	// failure is loud at startup rather than on the first PUT. NEVER log the value.
	if os.Getenv("AWS_ACCESS_KEY_ID") == "" || os.Getenv("AWS_SECRET_ACCESS_KEY") == "" {
		return nil, errors.New("-backup-bucket/-cold-tier-after set but AWS credentials are missing; set AWS_ACCESS_KEY_ID and AWS_SECRET_ACCESS_KEY in the environment")
	}

	// Warn on a plaintext http:// endpoint pointed at a NON-loopback host: the
	// SigV4 Authorization header and request bodies would then traverse the network
	// in cleartext. http:// to localhost/127.0.0.1/::1 is the documented MinIO/
	// localstack dev path and stays silent; a remote http:// target is a downgrade
	// the operator likely did not intend.
	if warn := insecureEndpointWarning(f.Endpoint); warn != "" {
		slog.Warn(warn)
	}

	store, err := objstore.NewS3Store(objstore.Config{
		Endpoint:  f.Endpoint,
		Region:    f.Region,
		Bucket:    f.Bucket,
		PathStyle: f.PathStyle,
		// Creds left zero ⇒ objstore reads them from the AWS_* env (checked above).
	})
	if err != nil {
		return nil, fmt.Errorf("object store: %w", err)
	}

	return &tierPlan{
		Store:          store,
		Tenant:         f.Tenant,
		BackupInterval: f.Interval,
		Retention:      f.Retention,
		ColdTierAfter:  f.ColdTierAfter,
	}, nil
}

// insecureEndpointWarning returns a non-empty warning message when endpoint is a
// plaintext http:// URL pointed at a non-loopback host (a cleartext-credentials
// downgrade), or "" when the endpoint is empty, https, unparseable, or http:// to
// a loopback host (the documented MinIO/localstack dev path). It never returns an
// error: a malformed endpoint is surfaced later by NewS3Store.
func insecureEndpointWarning(endpoint string) string {
	if endpoint == "" {
		return ""
	}
	u, err := url.Parse(endpoint)
	if err != nil || u.Scheme != "http" {
		return ""
	}
	host := u.Hostname()
	if isLoopbackHost(host) {
		return ""
	}
	return fmt.Sprintf("-backup-endpoint %q is plaintext http:// to a non-loopback host: SigV4-signed credentials and request bodies will traverse the network in cleartext; use https:// for remote object stores (http:// is intended only for local MinIO/localstack)", endpoint)
}

// isLoopbackHost reports whether host is a loopback name or IP (localhost, an IP
// in 127.0.0.0/8, or ::1).
func isLoopbackHost(host string) bool {
	if host == "localhost" {
		return true
	}
	if ip := net.ParseIP(host); ip != nil {
		return ip.IsLoopback()
	}
	return false
}

// runS3BackupLoop ticks every interval and snapshots every collection to the
// shared object store via backup.Backup. The per-run Timestamp is computed HERE
// with time.Now() (the backup package itself never calls time.Now, staying
// deterministic/testable) as a sortable UTC RFC3339 stamp so retention prunes
// oldest-first. A backup failure is LOGGED (never the credentials) and the loop
// CONTINUES — a transient object-store error must never crash the serving
// process. The loop exits when ctx is cancelled (server shutdown) and closes done
// so main can wait for a clean stop.
func runS3BackupLoop(ctx context.Context, vs *vector.CollectionStore, plan *tierPlan, interval time.Duration, done chan<- struct{}) {
	defer close(done)
	ticker := time.NewTicker(interval)
	defer ticker.Stop()
	for {
		select {
		case <-ctx.Done():
			return
		case <-ticker.C:
			runBackupTick(ctx, vs, plan)
		}
	}
}

// runBackupTick runs ONE backup pass with the wall-clock timestamp and logs the
// per-run result. Factored out of the loop so tests can drive a single tick
// directly (no ticker, no goroutine). Errors are logged and swallowed.
func runBackupTick(ctx context.Context, vs *vector.CollectionStore, plan *tierPlan) {
	opts := backup.BackupOpts{
		Tenant:    plan.Tenant,
		Timestamp: time.Now().UTC(),
		Retention: plan.Retention,
	}
	results, err := backup.Backup(ctx, vs, plan.Store, opts)
	if err != nil {
		slog.Warn("backup run had failures (continuing)", "err", err)
	}
	ok := 0
	for _, r := range results {
		if r.Err == nil {
			ok++
		}
	}
	slog.Info("backup run", "ok", ok, "total", len(results), "ts", opts.Timestamp.Format(time.RFC3339))
}

// runColdTierSweeper ticks on a cadence derived from ColdTierAfter and evicts
// every collection idle longer than the threshold via store.SweepCold, supplying
// the wall clock HERE (the engine never calls time.Now). Evictions are logged;
// per-collection errors are logged and the loop CONTINUES. The sweep cadence is a
// fraction of the idle threshold (so an idle collection is caught within roughly
// one threshold of going idle), floored to a sane minimum. Exits on ctx cancel
// and closes done.
func runColdTierSweeper(ctx context.Context, vs *vector.CollectionStore, plan *tierPlan, done chan<- struct{}) {
	defer close(done)
	ticker := time.NewTicker(sweepInterval(plan.ColdTierAfter))
	defer ticker.Stop()
	for {
		select {
		case <-ctx.Done():
			return
		case <-ticker.C:
			// time.Now() is read HERE (the driver) and handed to the tick body, which
			// stays a pure function of the supplied clock so tests can inject one.
			runSweepTick(time.Now(), vs, plan)
		}
	}
}

// runSweepTick runs ONE idle-eviction sweep at the supplied wall clock `now`. The
// driver passes time.Now(); tests pass a fixed clock. It is a pure function of
// `now` (the engine never reads time.Now itself).
func runSweepTick(now time.Time, vs *vector.CollectionStore, plan *tierPlan) {
	evicted, err := vs.SweepCold(now, plan.ColdTierAfter, plan.Store, plan.Tenant)
	if err != nil {
		slog.Warn("cold-tier sweep had failures (continuing)", "err", err)
	}
	if len(evicted) > 0 {
		slog.Info("cold-tier evicted idle collection(s)", "count", len(evicted), "collections", evicted)
	}
}

// adminBackend implements httpapi.AdminBackend over the single-node collection
// store + the shared object store. It backs the admin REST endpoints
// (backup-now / list-backups / evict / restore). time.Now() is read HERE (the
// cmd layer) to stamp the backup/evict timestamps — the backup and engine layers
// stay deterministic.
//
// The collection store is set AFTER rostam.NewServer returns (NewServer builds
// the HTTP handler — and captures cfg.Admin — before the store is reachable from
// the cmd layer), so vs is guarded and a request that races in before SetStore
// gets a clean 412 "not ready" rather than a nil deref.
type adminBackend struct {
	plan *tierPlan

	mu sync.RWMutex
	vs *vector.CollectionStore
}

// newAdminBackend wires the admin surface. Returns nil when plan is nil
// (nothing configured) so the cmd layer leaves cfg.Admin nil and the routes 412.
// The store is attached later via SetStore.
func newAdminBackend(plan *tierPlan) *adminBackend {
	if plan == nil {
		return nil
	}
	return &adminBackend{plan: plan}
}

// SetStore attaches the live collection store once rostam.NewServer has built it.
func (b *adminBackend) SetStore(vs *vector.CollectionStore) {
	b.mu.Lock()
	b.vs = vs
	b.mu.Unlock()
}

// store returns the attached collection store, or (nil, false) if SetStore has
// not run yet (a request racing server startup).
func (b *adminBackend) store() (*vector.CollectionStore, bool) {
	b.mu.RLock()
	vs := b.vs
	b.mu.RUnlock()
	return vs, vs != nil
}

// BackupNow runs one immediate backup of every live collection.
func (b *adminBackend) BackupNow(ctx context.Context) ([]httpapi.BackupReport, error) {
	vs, ok := b.store()
	if !ok {
		return nil, errNotReady
	}
	opts := backup.BackupOpts{
		Tenant:    b.plan.Tenant,
		Timestamp: time.Now().UTC(),
		Retention: b.plan.Retention,
	}
	results, err := backup.Backup(ctx, vs, b.plan.Store, opts)
	reports := make([]httpapi.BackupReport, 0, len(results))
	for _, r := range results {
		rep := httpapi.BackupReport{Collection: r.Collection, Key: r.Key, Size: r.Size}
		if r.Err != nil {
			rep.Error = r.Err.Error()
		}
		reports = append(reports, rep)
	}
	return reports, err
}

// ListBackups lists every snapshot object under the configured tenant prefix.
func (b *adminBackend) ListBackups(ctx context.Context) ([]httpapi.BackupObject, error) {
	infos, err := b.plan.Store.List(ctx, b.plan.Tenant)
	if err != nil {
		return nil, err
	}
	out := make([]httpapi.BackupObject, 0, len(infos))
	for _, in := range infos {
		out = append(out, httpapi.BackupObject{
			Key:          in.Key,
			Size:         in.Size,
			LastModified: in.LastModified.UTC().Format(time.RFC3339),
		})
	}
	return out, nil
}

// EvictCollection cold-tiers the named collection to object storage.
func (b *adminBackend) EvictCollection(ctx context.Context, name string) error {
	vs, ok := b.store()
	if !ok {
		return errNotReady
	}
	return vs.EvictCollection(ctx, name, b.plan.Store, b.plan.Tenant, time.Now().UTC())
}

// RestoreCollection eagerly promotes a cold collection back into memory. It is a
// no-op for an already-hot collection. Acquire transparently triggers the lazy
// restore path for a cold stub, so an Acquire+Release round-trips the promote. A
// failed Acquire on a still-cold collection means the promote failed (object
// store unreachable) — reported as an error; the stub stays recoverable.
func (b *adminBackend) RestoreCollection(ctx context.Context, name string) error {
	vs, ready := b.store()
	if !ready {
		return errNotReady
	}
	c, ok := vs.Acquire(name)
	if ok {
		c.Release()
		return nil
	}
	if vs.IsCold(name) {
		return fmt.Errorf("restore %q: promote from object store failed (collection remains cold)", name)
	}
	// "no collection" matches httpapi statusForError's 404 bucket (the engine uses
	// the same wording for a missing collection); "no such collection" matched
	// neither bucket and fell through to a 500.
	return fmt.Errorf("restore %q: no collection %q", name, name)
}

// errNotReady is returned by an admin method called before SetStore has attached
// the live store (a sub-millisecond request racing server startup; SetStore runs
// synchronously right after rostam.NewServer returns). Surfaced as a transient
// server error so a retry succeeds.
var errNotReady = errors.New("admin backend not ready (server still starting)")

// sweepInterval derives the sweeper's tick cadence from the idle threshold:
// roughly a quarter of the threshold so an idle collection is evicted within ~one
// threshold of going idle, clamped to [10s, 5m] so a tiny threshold doesn't spin
// and a huge one still sweeps periodically.
func sweepInterval(coldAfter time.Duration) time.Duration {
	d := coldAfter / 4
	const min, max = 10 * time.Second, 5 * time.Minute
	if d < min {
		return min
	}
	if d > max {
		return max
	}
	return d
}
