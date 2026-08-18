// SPDX-License-Identifier: Apache-2.0
//go:build localembed

package main

import (
	"context"
	"fmt"
	"net/http"
	"os"
	"path/filepath"
	"time"

	"github.com/rostamlabs/rostam/semcache"
	"github.com/rostamlabs/rostam/semcache/local"
	"github.com/rostamlabs/rostam/semcache/localcatalog"
)

func newLocalEmbedder(name string, lookupEnv func(string) (string, bool)) (semcache.Embedder, error) {
	spec, ok := localcatalog.Lookup(name)
	if !ok {
		return nil, fmt.Errorf("rostam-server: unknown local embedding model %q (see: rostam-server mcp -list-embed-models)", name)
	}
	root, _ := lookupEnv("ROSTAM_EMBED_MODELS_DIR")
	if root == "" {
		home, err := os.UserHomeDir()
		if err != nil {
			return nil, fmt.Errorf("rostam-server: cannot resolve model cache dir; set ROSTAM_EMBED_MODELS_DIR: %w", err)
		}
		root = filepath.Join(home, ".rostam", "models")
	}
	libPath, err := local.ResolveORTLibPath(lookupEnv)
	if err != nil {
		return nil, err
	}
	hc := &http.Client{Timeout: 30 * time.Minute} // large weights on a slow link
	return local.NewEmbedder(context.Background(), spec, root, libPath, hc)
}
