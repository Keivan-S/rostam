// SPDX-License-Identifier: Apache-2.0

package dashboard

import (
	"net/http/httptest"
	"strings"
	"testing"
)

// do issues a GET against Handler() and returns the recorder, mirroring the
// httpapi package's own request helper but staying local to this package (the
// embed reads dashboard/dist at compile time, so these assertions must hold
// whether that tree is the checked-in placeholder or a real `make ui` build).
func do(t *testing.T, path string) *httptest.ResponseRecorder {
	t.Helper()
	req := httptest.NewRequest("GET", path, nil)
	rec := httptest.NewRecorder()
	Handler().ServeHTTP(rec, req)
	return rec
}

// TestHandlerServesIndex proves the app shell is served at "/" as 200 text/html.
func TestHandlerServesIndex(t *testing.T) {
	rec := do(t, "/")
	if rec.Code != 200 {
		t.Fatalf("/ = %d, want 200 (%s)", rec.Code, rec.Body)
	}
	if ct := rec.Header().Get("Content-Type"); !strings.HasPrefix(ct, "text/html") {
		t.Fatalf("/ content-type = %q, want text/html", ct)
	}
}

// TestHandlerSPAFallback proves an unknown, non-asset path (a client-side
// route) falls back to the same index.html shell rather than 404ing.
func TestHandlerSPAFallback(t *testing.T) {
	index := do(t, "/")
	rec := do(t, "/collections/docs")
	if rec.Code != 200 {
		t.Fatalf("/collections/docs = %d, want 200 (%s)", rec.Code, rec.Body)
	}
	if ct := rec.Header().Get("Content-Type"); !strings.HasPrefix(ct, "text/html") {
		t.Fatalf("/collections/docs content-type = %q, want text/html", ct)
	}
	if rec.Body.String() != index.Body.String() {
		t.Fatalf("/collections/docs body != index.html body")
	}
}

// TestHandlerTraversalContained proves a "../" path segment cannot escape the
// embedded dist tree: the response is either the SPA shell (200) or a 404,
// never the contents of a real host file like /etc/passwd.
func TestHandlerTraversalContained(t *testing.T) {
	rec := do(t, "/../../etc/passwd")
	if rec.Code != 200 && rec.Code != 404 {
		t.Fatalf("traversal = %d, want 200 or 404 (%s)", rec.Code, rec.Body)
	}
	if strings.Contains(rec.Body.String(), "root:") {
		t.Fatalf("traversal response leaked host file contents: %s", rec.Body)
	}
}

// TestHandlerMissingAssetIs404 proves a request under assets/ for a name that
// does not exist in the embed gets a genuine 404, not the SPA shell — Vite
// asset names are content-hashed, so a miss means the file never existed
// (never the client-side-routing case the SPA fallback exists for).
func TestHandlerMissingAssetIs404(t *testing.T) {
	rec := do(t, "/assets/does-not-exist-0123456789.js")
	if rec.Code != 404 {
		t.Fatalf("missing assets/ file = %d, want 404 (%s)", rec.Code, rec.Body)
	}
}
