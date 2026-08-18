// SPDX-License-Identifier: Apache-2.0
//go:build !localembed

package main

import (
	"fmt"

	"github.com/rostamlabs/rostam/semcache"
	"github.com/rostamlabs/rostam/semcache/localcatalog"
)

// newLocalEmbedder is the compiled-out stub: without -tags localembed the local
// ONNX embedder is not in the binary. If the model name is a real catalog entry
// we say so specifically; otherwise we still name the missing build tag.
func newLocalEmbedder(name string, _ func(string) (string, bool)) (semcache.Embedder, error) {
	if _, ok := localcatalog.Lookup(name); ok {
		return nil, fmt.Errorf("rostam-server: model %q requires a build with -tags localembed (this binary was built without it)", name)
	}
	return nil, fmt.Errorf("rostam-server: local embedding requires -tags localembed (unknown model %q)", name)
}
