// SPDX-License-Identifier: Apache-2.0

// Package dashboard embeds Rostam's web dashboard (a static single-page app) and
// serves it over HTTP. The compiled Vite output lives under dist/ and is baked
// into the binary with go:embed, so a release ships the UI with no runtime asset
// dependency. This package deliberately imports NOTHING from the rest of Rostam:
// the assets are inert HTML/JS/CSS and all data access goes through the authed
// /v1 REST endpoints, so the httpapi package can mount Handler() without forming
// an import cycle back into the engine.
package dashboard

import (
	"embed"
	"io/fs"
	"mime"
	"net/http"
	"path"
	"strings"
)

// dist holds the built SPA. The all: prefix embeds every file including those
// whose names start with '.', so the checked-in dist/.gitkeep (the only tracked
// file — the Vite output is git-ignored and synced in by `make ui`) is enough
// for the embed to resolve and `go build ./...` to compile without a node build.
// When only the .gitkeep is present, Handler serves the unbuiltPage placeholder.
//
//go:embed all:dist
var dist embed.FS

// Handler serves the embedded SPA. It serves a real asset file when the request
// path names one, and otherwise falls back to dist/index.html so client-side
// routing (deep links like /dashboard/collections/docs) resolves to the app
// shell rather than a 404. Content-Type is set from the file extension.
//
// Auth is intentionally NOT enforced here: the assets carry no data (every read
// goes through the authed /v1 endpoints), so gating the static shell would only
// break the login/landing experience without protecting anything.
func Handler() http.Handler {
	// Root the FS at dist/ so request paths map directly onto asset names
	// ("/app.js" -> "app.js") without the embed's dist/ prefix leaking in.
	sub, err := fs.Sub(dist, "dist")
	if err != nil {
		// Unreachable: dist is embedded at build time, so the subtree always exists.
		panic("dashboard: embed dist subtree: " + err.Error())
	}
	return http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		// Normalize to a clean, rooted asset name. A leading "/" is stripped because
		// the embedded FS is not rooted; "" means the index.
		name := strings.TrimPrefix(path.Clean("/"+r.URL.Path), "/")
		if name == "" {
			name = "index.html"
		}
		if serveFile(w, sub, name) {
			return
		}
		// A missing file under assets/ is a genuine 404: those names are
		// content-hashed by Vite, so a miss means the asset never existed or was
		// purged, not a client-side route. Falling back to the SPA shell here
		// would mask that with a misleading 200 (e.g. a stale HTML page still
		// referencing a JS bundle that has since been rebuilt under a new hash).
		if strings.HasPrefix(name, "assets/") {
			http.NotFound(w, r)
			return
		}
		// Unknown non-asset path: hand back the SPA shell so the client router
		// takes over.
		if serveFile(w, sub, "index.html") {
			return
		}
		// No index.html means the Vite output was never synced into dist/ (a bare
		// `go build` without `make ui`). dist/ still embeds via the .gitkeep, so we
		// are here rather than failing the build; serve a friendly placeholder that
		// tells the operator how to build the real UI instead of a bare 500.
		w.Header().Set("Content-Type", "text/html; charset=utf-8")
		w.Header().Set("Cache-Control", "no-cache")
		_, _ = w.Write([]byte(unbuiltPage))
	})
}

// unbuiltPage is served for /dashboard/ when the compiled SPA has not been synced
// into dist/ (i.e. the binary was built without running `make ui`). It is a valid
// standalone HTML page — no external assets — so it renders even with an otherwise
// empty embed.
const unbuiltPage = `<!doctype html>
<html lang="en">
<head><meta charset="utf-8"><title>Rostam dashboard</title>
<style>body{font:15px/1.6 ui-sans-serif,system-ui,sans-serif;max-width:34rem;margin:12vh auto;padding:0 1.5rem;color:#111419;background:#fff}code{background:#f0b429;color:#111419;padding:.1rem .4rem;border-radius:.25rem;font-family:ui-monospace,monospace}</style>
</head>
<body>
<h1>Rostam dashboard</h1>
<p>The web UI has not been built into this binary. Build the assets and rebuild the server:</p>
<p><code>make ui &amp;&amp; make build</code></p>
<p>The REST API is unaffected and available under <code>/v1</code>.</p>
</body>
</html>
`

// serveFile writes name from fsys with a Content-Type derived from its
// extension and a Cache-Control appropriate to the SPA build's naming scheme:
// content-hashed files under assets/ never change under a given name, so they
// are cached for a year as immutable; everything else (index.html, and any
// other top-level file) is mutable and served no-cache. Returns true when the
// file existed and was served. A directory is treated as a miss (there is no
// directory listing), so it falls through to the SPA shell.
func serveFile(w http.ResponseWriter, fsys fs.FS, name string) bool {
	data, err := fs.ReadFile(fsys, name)
	if err != nil {
		return false
	}
	if ct := mime.TypeByExtension(path.Ext(name)); ct != "" {
		w.Header().Set("Content-Type", ct)
	}
	if strings.HasPrefix(name, "assets/") {
		w.Header().Set("Cache-Control", "public, max-age=31536000, immutable")
	} else {
		w.Header().Set("Cache-Control", "no-cache")
	}
	//nolint:gosec // G705: data is read only from the build-time embedded dist FS
	// (see the go:embed above), and name is sanitized by path.Clean + fs.Sub in
	// Handler, so it cannot select anything outside the embedded assets.
	_, _ = w.Write(data)
	return true
}
