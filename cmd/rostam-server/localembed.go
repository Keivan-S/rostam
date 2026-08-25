// SPDX-License-Identifier: Apache-2.0

package main

import (
	"fmt"
	"os"
	"strings"

	"github.com/rostamlabs/rostam/semcache"
	"github.com/rostamlabs/rostam/semcache/local"
	"github.com/rostamlabs/rostam/semcache/localcatalog"
)

// newLocalEmbedder builds an in-process, pure-Go embedder (rembed) for the model
// named by ROSTAM_EMBED_LOCAL. name is either a curated catalog entry
// ("minilm-l6-v2", or "1"/"default"/"" for the default), or — as a passthrough —
// any Hugging Face model id ("org/model") not in the catalog, loaded as-is.
func newLocalEmbedder(name string, lookupEnv func(string) (string, bool)) (semcache.Embedder, error) {
	// rembed caches downloaded models under REMBED_CACHE (default the OS user
	// cache dir), and v0.3.0 exposes no functional cache-dir option — only the
	// env var — so bridging the historical ROSTAM_EMBED_MODELS_DIR to
	// REMBED_CACHE (when the native var is unset) keeps existing deployments'
	// cache location without reconfiguration. This mutates process env via
	// os.Setenv; it is called once at server startup, before any concurrent
	// embedder use, so the global write is safe.
	if root, ok := lookupEnv("ROSTAM_EMBED_MODELS_DIR"); ok && root != "" {
		if _, set := lookupEnv("REMBED_CACHE"); !set {
			_ = os.Setenv("REMBED_CACHE", root)
		}
	}

	if spec, ok := localcatalog.Lookup(name); ok {
		return local.New(spec.HFRepo, spec.Name, spec.Dim)
	}
	// Passthrough: an unrecognized name is treated as a Hugging Face model id.
	// A bare word (no "/") is neither a catalog entry nor a valid Hub id, so
	// fail loud rather than hand rembed something it will reject obscurely.
	if !strings.Contains(name, "/") {
		return nil, fmt.Errorf("rostam-server: unknown local embedding model %q: not a catalog entry (see: rostam-server mcp -list-embed-models) and not a Hugging Face model id (expected \"org/model\")", name)
	}
	return local.New(name, name, 0)
}
