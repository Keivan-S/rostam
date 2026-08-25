// SPDX-License-Identifier: Apache-2.0

// Package localcatalog is a curated allowlist of embedding models the in-process
// local embedder can run. It is pure data so every build can list models and
// emit precise errors. Short catalog names map to Hugging Face model ids; the
// embedder (rembed) derives tokenizer, pooling, and normalization from the model
// itself, so nothing about a model's internals needs to live here — only its
// public identity (name, Hub repo, dimension, license).
package localcatalog

import "sort"

// ModelSpec identifies one selectable model. HFRepo is the Hugging Face id
// passed to rembed.Load; Dim is the model's output dimension (used to validate
// the loaded model and to size the cache collection).
type ModelSpec struct {
	Name    string
	HFRepo  string
	Dim     int
	License string
}

// DefaultModel is chosen when ROSTAM_EMBED_LOCAL is "1", "default", or empty.
const DefaultModel = "minilm-l6-v2"

var catalog = map[string]ModelSpec{
	// --- 384-dim "small" tier --------------------------------------------
	"minilm-l6-v2": {
		Name:    "minilm-l6-v2",
		HFRepo:  "sentence-transformers/all-MiniLM-L6-v2",
		Dim:     384,
		License: "Apache-2.0",
	},
	"bge-small-en-v1.5": {
		Name:    "bge-small-en-v1.5",
		HFRepo:  "BAAI/bge-small-en-v1.5",
		Dim:     384,
		License: "MIT",
	},
	"gte-small": {
		Name:    "gte-small",
		HFRepo:  "thenlper/gte-small",
		Dim:     384,
		License: "MIT",
	},

	// --- 768-dim "base" tier ---------------------------------------------
	"bge-base-en-v1.5": {
		Name:    "bge-base-en-v1.5",
		HFRepo:  "BAAI/bge-base-en-v1.5",
		Dim:     768,
		License: "MIT",
	},
	"gte-base": {
		Name:    "gte-base",
		HFRepo:  "thenlper/gte-base",
		Dim:     768,
		License: "MIT",
	},
	"all-mpnet-base-v2": {
		Name:    "all-mpnet-base-v2",
		HFRepo:  "sentence-transformers/all-mpnet-base-v2",
		Dim:     768,
		License: "Apache-2.0",
	},
}

// Lookup resolves a model name (with "", "1", "default" mapping to DefaultModel).
func Lookup(name string) (ModelSpec, bool) {
	if name == "" || name == "1" || name == "default" {
		name = DefaultModel
	}
	m, ok := catalog[name]
	return m, ok
}

// Names returns the catalog keys, sorted.
func Names() []string {
	out := make([]string, 0, len(catalog))
	for n := range catalog {
		out = append(out, n)
	}
	sort.Strings(out)
	return out
}
