// SPDX-License-Identifier: Apache-2.0

package httpapi

import (
	"context"
	"net/http"
	"net/http/httptest"
	"path/filepath"
	"sort"
	"strings"
	"sync"
	"testing"

	"github.com/rostamlabs/rostam/authz"
	"github.com/rostamlabs/rostam/vector"
)

// fakeAdmin is an in-memory httpapi.AdminBackend used to round-trip the admin
// REST endpoints without a real object store or collection engine.
type fakeAdmin struct {
	mu      sync.Mutex
	backups map[string]int64 // key -> size
	cold    map[string]bool  // collection -> cold?
	known   map[string]bool  // collection exists?
}

func newFakeAdmin() *fakeAdmin {
	return &fakeAdmin{
		backups: map[string]int64{},
		cold:    map[string]bool{},
		known:   map[string]bool{"default/docs": true},
	}
}

func (f *fakeAdmin) BackupNow(ctx context.Context) ([]BackupReport, error) {
	f.mu.Lock()
	defer f.mu.Unlock()
	var reports []BackupReport
	for col := range f.known {
		key := "default/" + col + "/2026.snap"
		f.backups[key] = 10
		reports = append(reports, BackupReport{Collection: col, Key: key, Size: 10})
	}
	sort.Slice(reports, func(i, j int) bool { return reports[i].Collection < reports[j].Collection })
	return reports, nil
}

func (f *fakeAdmin) ListBackups(ctx context.Context) ([]BackupObject, error) {
	f.mu.Lock()
	defer f.mu.Unlock()
	var out []BackupObject
	for k, sz := range f.backups {
		out = append(out, BackupObject{Key: k, Size: sz, LastModified: "2026-06-23T00:00:00Z"})
	}
	sort.Slice(out, func(i, j int) bool { return out[i].Key < out[j].Key })
	return out, nil
}

func (f *fakeAdmin) EvictCollection(ctx context.Context, name string) error {
	f.mu.Lock()
	defer f.mu.Unlock()
	if !f.known[name] {
		// Mirror the real engine's missing-collection wording ("no collection ...")
		// so this fake exercises the SAME statusForError bucket the engine hits.
		return &adminTestErr{"no collection " + name}
	}
	f.cold[name] = true
	return nil
}

func (f *fakeAdmin) RestoreCollection(ctx context.Context, name string) error {
	f.mu.Lock()
	defer f.mu.Unlock()
	if !f.known[name] {
		// Mirror the real engine's missing-collection wording ("no collection ...")
		// so this fake exercises the SAME statusForError bucket the engine hits.
		return &adminTestErr{"no collection " + name}
	}
	f.cold[name] = false
	return nil
}

type adminTestErr struct{ msg string }

func (e *adminTestErr) Error() string { return e.msg }

// stubDispatcher satisfies Dispatcher for the admin handler (which never routes
// through it).
type stubDispatcher struct{}

func (stubDispatcher) Call(string, []byte) ([]byte, error) { return nil, nil }
func (stubDispatcher) LeaderAddr() string                  { return "" }

// TestAdminBackupEvictRestoreRoundTrip exercises the four admin endpoints over the
// real router: backup_now → list_backups shows it; evict → restore round-trip.
func TestAdminBackupEvictRestoreRoundTrip(t *testing.T) {
	fa := newFakeAdmin()
	h := Handler(stubDispatcher{}, Options{Admin: fa})

	// backup_now
	rec := do(t, h, "POST", "/v1/admin/backup", "", nil)
	if rec.Code != http.StatusOK {
		t.Fatalf("backup_now = %d (%s)", rec.Code, rec.Body)
	}

	// list_backups shows the snapshot from backup_now.
	var listed struct {
		Backups []BackupObject `json:"backups"`
	}
	rec = do(t, h, "GET", "/v1/admin/backups", "", &listed)
	if rec.Code != http.StatusOK {
		t.Fatalf("list_backups = %d (%s)", rec.Code, rec.Body)
	}
	if len(listed.Backups) != 1 || listed.Backups[0].Key != "default/default/docs/2026.snap" {
		t.Fatalf("list_backups did not reflect backup_now: %+v", listed.Backups)
	}

	// evict
	rec = do(t, h, "POST", "/v1/collections/default%2Fdocs/evict", "", nil)
	if rec.Code != http.StatusOK {
		t.Fatalf("evict = %d (%s)", rec.Code, rec.Body)
	}
	if !fa.cold["default/docs"] {
		t.Fatal("evict did not mark the collection cold")
	}

	// restore
	rec = do(t, h, "POST", "/v1/collections/default%2Fdocs/restore", "", nil)
	if rec.Code != http.StatusOK {
		t.Fatalf("restore = %d (%s)", rec.Code, rec.Body)
	}
	if fa.cold["default/docs"] {
		t.Fatal("restore did not promote the collection back to hot")
	}

	// evict/restore of an UNKNOWN collection must surface 404 (not 500): the engine
	// returns a "no collection" error, which statusForError maps to NotFound. This
	// guards the regression where the engine said "no such collection" (matching
	// neither 404 bucket) and the fake masked it with differently-worded text.
	rec = do(t, h, "POST", "/v1/collections/default%2Fnope/evict", "", nil)
	if rec.Code != http.StatusNotFound {
		t.Fatalf("evict unknown = %d, want 404 (%s)", rec.Code, rec.Body)
	}
	rec = do(t, h, "POST", "/v1/collections/default%2Fnope/restore", "", nil)
	if rec.Code != http.StatusNotFound {
		t.Fatalf("restore unknown = %d, want 404 (%s)", rec.Code, rec.Body)
	}
}

// TestAdminUnconfigured412: when no AdminBackend is wired (no objstore
// configured), the routes exist but return 412 — a clear "not configured" error,
// never a 404 or a silent no-op.
func TestAdminUnconfigured412(t *testing.T) {
	h := Handler(stubDispatcher{}, Options{}) // Admin nil

	for _, tc := range []struct{ method, path string }{
		{"POST", "/v1/admin/backup"},
		{"GET", "/v1/admin/backups"},
		{"POST", "/v1/collections/docs/evict"},
		{"POST", "/v1/collections/docs/restore"},
	} {
		rec := do(t, h, tc.method, tc.path, "", nil)
		if rec.Code != http.StatusPreconditionFailed {
			t.Errorf("%s %s with no objstore = %d, want 412", tc.method, tc.path, rec.Code)
		}
	}
}

// TestAdminRBACGated: the admin endpoints are admin-scoped. A non-admin (read)
// principal is denied (401) BEFORE the backend is touched; an admin principal is
// allowed. The op names are not in the ops registry, so authz.actionFor
// classifies them as admin (fail-closed) — only an admin scope covers them.
func TestAdminRBACGated(t *testing.T) {
	// A real RBAC authenticator over a registry with a read-only key and an admin
	// key. The ops registry is nil — tolerated; the admin op names are not in it,
	// so authz.actionFor classifies them as admin (fail-closed) either way.
	keyReg, err := vector.OpenKeyRegistry(filepath.Join(t.TempDir(), "keys.json"))
	if err != nil {
		t.Fatal(err)
	}
	if err := keyReg.AddKey(vector.APIKey{Token: "reader", Tenant: "*", Scopes: []string{"read:*"}}); err != nil {
		t.Fatal(err)
	}
	if err := keyReg.AddKey(vector.APIKey{Token: "boss", Tenant: "*", Scopes: []string{"*:*"}}); err != nil {
		t.Fatal(err)
	}
	auth := authz.NewRBACAuthenticator(keyReg, nil, "")
	fa := newFakeAdmin()
	h := Handler(stubDispatcher{}, Options{Authenticator: auth, Admin: fa})

	// Read-only principal → 401 on an admin endpoint (denied before the backend).
	if code := adminReq(t, h, "reader", "POST", "/v1/admin/backup"); code != http.StatusUnauthorized {
		t.Fatalf("read-only key on admin backup = %d, want 401", code)
	}
	// Admin principal → allowed.
	if code := adminReq(t, h, "boss", "POST", "/v1/admin/backup"); code != http.StatusOK {
		t.Fatalf("admin key on admin backup = %d, want 200", code)
	}
}

// adminReq issues a bearer-authenticated request against the handler and returns
// the status code.
func adminReq(t *testing.T, h http.Handler, token, method, path string) int {
	t.Helper()
	r := httptest.NewRequest(method, path, strings.NewReader(""))
	if token != "" {
		r.Header.Set("Authorization", "Bearer "+token)
	}
	rec := httptest.NewRecorder()
	h.ServeHTTP(rec, r)
	return rec.Code
}
