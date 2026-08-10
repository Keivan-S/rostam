// SPDX-License-Identifier: Apache-2.0

package httpapi

import (
	"context"
	"net/http"
)

// AdminBackend is the OPT-IN object-storage admin surface backing the
// /v1/admin/backup, /v1/admin/backups, and /v1/collections/{name}/evict|restore
// REST endpoints. It is implemented in the cmd layer over a *vector.CollectionStore
// + an objstore.ObjectStore (the SAME shared client the backup cron and cold-tier
// sweeper use), and threaded in via Options.Admin.
//
// It is deliberately NOT routed through the op Dispatcher: backup/evict/restore
// need direct access to the collection store and the object store, which the
// Dispatcher abstraction (op-name + binary args) does not expose. The handlers
// therefore authorize via the SAME authz gate (the admin op names below classify
// as admin / fail-closed) and then call this backend directly.
//
// nil (the default, and whenever no bucket/cold-tier is configured) ⇒ the routes
// are still registered but return 412 Precondition Failed ("object storage not
// configured"), so an admin call against an un-tiered server fails loud rather
// than 404-ing or silently no-op'ing.
type AdminBackend interface {
	// BackupNow runs one immediate backup of every live collection and returns one
	// BackupReport per collection (Key empty + Error set on a per-collection
	// failure; the run never aborts on one collection).
	BackupNow(ctx context.Context) ([]BackupReport, error)
	// ListBackups lists every backup snapshot object under the configured tenant
	// prefix.
	ListBackups(ctx context.Context) ([]BackupObject, error)
	// EvictCollection cold-tiers the named collection to the object store (no-op if
	// already cold; error if unknown).
	EvictCollection(ctx context.Context, name string) error
	// RestoreCollection eagerly promotes a cold collection back from the object
	// store (no-op if already hot).
	RestoreCollection(ctx context.Context, name string) error
}

// BackupReport is one collection's outcome from BackupNow.
type BackupReport struct {
	Collection string `json:"collection"`
	Key        string `json:"key,omitempty"`
	Size       int64  `json:"size"`
	Error      string `json:"error,omitempty"`
}

// BackupObject is one snapshot object as listed by ListBackups.
type BackupObject struct {
	Key          string `json:"key"`
	Size         int64  `json:"size"`
	LastModified string `json:"last_modified"`
}

// Admin op names. They are NOT in the ops registry — authz.actionFor classifies
// any op it cannot map as "admin" (fail-closed), so authorizing against these
// names requires an admin-scoped principal exactly like the key-admin ops. Naming
// them explicitly keeps the audit/log line meaningful.
const (
	opAdminBackupNow   = "admin_backup_now"
	opAdminListBackups = "admin_list_backups"
	opAdminEvict       = "admin_evict_collection"
	opAdminRestore     = "admin_restore_collection"
)

// errAdminUnconfigured is the 412 body returned when an admin object-storage
// endpoint is hit but no objstore is configured (no -backup-bucket/-cold-tier).
const errAdminUnconfigured = "object storage not configured (start with -backup-bucket and -backup-region, and AWS_* credentials)"

// adminConfigured authorizes (admin-gated) and then checks the backend is wired.
// Returns false (after writing the response) when auth fails (401) or no objstore
// is configured (412). On true the caller may use a.admin.
func (a *api) adminConfigured(w http.ResponseWriter, r *http.Request, op string) bool {
	if !a.authorize(w, r, op, nil) {
		return false
	}
	if a.admin == nil {
		writeError(w, http.StatusPreconditionFailed, errAdminUnconfigured)
		return false
	}
	return true
}

// backupNow handles POST /v1/admin/backup: trigger an immediate backup of every
// live collection. Admin-scoped.
func (a *api) backupNow(w http.ResponseWriter, r *http.Request) {
	if !a.adminConfigured(w, r, opAdminBackupNow) {
		return
	}
	reports, err := a.admin.BackupNow(r.Context())
	if err != nil {
		// A partial failure still returns the per-collection reports (so the caller
		// sees which succeeded); surface the joined error too.
		writeJSON(w, http.StatusOK, map[string]any{"backups": reports, "error": err.Error()})
		return
	}
	writeJSON(w, http.StatusOK, map[string]any{"backups": reports})
}

// listBackups handles GET /v1/admin/backups: list every snapshot object under the
// configured tenant prefix. Admin-scoped.
func (a *api) listBackups(w http.ResponseWriter, r *http.Request) {
	if !a.adminConfigured(w, r, opAdminListBackups) {
		return
	}
	objs, err := a.admin.ListBackups(r.Context())
	if err != nil {
		writeInternalError(w, r.URL.Path, err)
		return
	}
	writeJSON(w, http.StatusOK, map[string]any{"backups": objs})
}

// evictCollection handles POST /v1/collections/{name}/evict: cold-tier the named
// collection to object storage. Admin-scoped.
func (a *api) evictCollection(w http.ResponseWriter, r *http.Request) {
	if !a.adminConfigured(w, r, opAdminEvict) {
		return
	}
	name := r.PathValue("name")
	if err := a.admin.EvictCollection(r.Context(), name); err != nil {
		writeDispatchError(w, r.URL.Path, err)
		return
	}
	writeJSON(w, http.StatusOK, map[string]bool{"evicted": true})
}

// restoreCollection handles POST /v1/collections/{name}/restore: eagerly promote
// a cold collection back into memory from object storage. Admin-scoped.
func (a *api) restoreCollection(w http.ResponseWriter, r *http.Request) {
	if !a.adminConfigured(w, r, opAdminRestore) {
		return
	}
	name := r.PathValue("name")
	if err := a.admin.RestoreCollection(r.Context(), name); err != nil {
		writeDispatchError(w, r.URL.Path, err)
		return
	}
	writeJSON(w, http.StatusOK, map[string]bool{"restored": true})
}
