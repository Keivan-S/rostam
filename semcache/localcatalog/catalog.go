// SPDX-License-Identifier: Apache-2.0

// Package localcatalog describes the embedding models the -tags localembed
// build can run locally. It is pure data (no cgo, no build tag) so every build
// can list models and emit precise errors even when the embedder is compiled
// out.
package localcatalog

import "sort"

type Pooling int

const (
	PoolingMean Pooling = iota // masked mean over token embeddings
	PoolingCLS                 // take the [CLS] (index 0) token
)

func (p Pooling) String() string {
	switch p {
	case PoolingMean:
		return "mean"
	case PoolingCLS:
		return "cls"
	default:
		return "unknown"
	}
}

// ModelSpec fully describes one selectable model. Every per-model quirk that
// affects correctness lives here so the embedder code stays generic.
type ModelSpec struct {
	Name      string
	HFRepo    string
	OnnxURL   string
	OnnxSHA   string // lowercase hex SHA-256 of the .onnx artifact
	VocabURL  string
	VocabSHA  string // lowercase hex SHA-256 of vocab.txt
	Dim       int
	Pooling   Pooling
	LowerCase bool
	License   string
	// ClsToken and SepToken name the sequence-framing special tokens used by
	// the tokenizer. Empty means the BERT defaults "[CLS]"/"[SEP]"; models in
	// the RoBERTa/MPNet lineage set them to "<s>"/"</s>" instead. The unknown
	// token is always "[UNK]".
	ClsToken string
	SepToken string
}

// DefaultModel is chosen when ROSTAM_EMBED_LOCAL is "1", "default", or empty.
const DefaultModel = "minilm-l6-v2"

var catalog = map[string]ModelSpec{
	"minilm-l6-v2": {
		Name:      "minilm-l6-v2",
		HFRepo:    "sentence-transformers/all-MiniLM-L6-v2",
		OnnxURL:   "https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2/resolve/main/onnx/model.onnx",
		OnnxSHA:   "6fd5d72fe4589f189f8ebc006442dbb529bb7ce38f8082112682524616046452",
		VocabURL:  "https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2/resolve/main/vocab.txt",
		VocabSHA:  "07eced375cec144d27c900241f3e339478dec958f92fddbc551f295c992038a3",
		Dim:       384,
		Pooling:   PoolingMean,
		LowerCase: true,
		License:   "Apache-2.0",
	},
	"bge-small-en-v1.5": {
		Name:      "bge-small-en-v1.5",
		HFRepo:    "BAAI/bge-small-en-v1.5",
		OnnxURL:   "https://huggingface.co/BAAI/bge-small-en-v1.5/resolve/main/onnx/model.onnx",
		OnnxSHA:   "828e1496d7fabb79cfa4dcd84fa38625c0d3d21da474a00f08db0f559940cf35",
		VocabURL:  "https://huggingface.co/BAAI/bge-small-en-v1.5/resolve/main/vocab.txt",
		VocabSHA:  "07eced375cec144d27c900241f3e339478dec958f92fddbc551f295c992038a3",
		Dim:       384,
		Pooling:   PoolingCLS,
		LowerCase: true,
		License:   "MIT",
	},
	"gte-small": {
		Name:      "gte-small",
		HFRepo:    "thenlper/gte-small",
		OnnxURL:   "https://huggingface.co/thenlper/gte-small/resolve/main/onnx/model.onnx",
		OnnxSHA:   "0b01312b59bec0a2558a626f2937be4cbe4bb16d1511560153f598cec488f1f8",
		VocabURL:  "https://huggingface.co/thenlper/gte-small/resolve/main/vocab.txt",
		VocabSHA:  "07eced375cec144d27c900241f3e339478dec958f92fddbc551f295c992038a3",
		Dim:       384,
		Pooling:   PoolingMean,
		LowerCase: true,
		License:   "MIT",
	},

	// --- 768-dim "base" tier ---------------------------------------------

	"bge-base-en-v1.5": {
		Name:      "bge-base-en-v1.5",
		HFRepo:    "BAAI/bge-base-en-v1.5",
		OnnxURL:   "https://huggingface.co/BAAI/bge-base-en-v1.5/resolve/main/onnx/model.onnx",
		OnnxSHA:   "9bc579acdba21c253c62a9bf866891355a63ffa3442b52c8a37d75b2ccb91848",
		VocabURL:  "https://huggingface.co/BAAI/bge-base-en-v1.5/resolve/main/vocab.txt",
		VocabSHA:  "07eced375cec144d27c900241f3e339478dec958f92fddbc551f295c992038a3",
		Dim:       768,
		Pooling:   PoolingCLS,
		LowerCase: true,
		License:   "MIT",
	},
	"gte-base": {
		Name:      "gte-base",
		HFRepo:    "thenlper/gte-base",
		OnnxURL:   "https://huggingface.co/thenlper/gte-base/resolve/main/onnx/model.onnx",
		OnnxSHA:   "dbfc7a6898c7c95fc53e52aaaf8302b5b2f5e8ec90d0eafa8e3d4acd26abef39",
		VocabURL:  "https://huggingface.co/thenlper/gte-base/resolve/main/vocab.txt",
		VocabSHA:  "07eced375cec144d27c900241f3e339478dec958f92fddbc551f295c992038a3",
		Dim:       768,
		Pooling:   PoolingMean,
		LowerCase: true,
		License:   "MIT",
	},
	// all-mpnet-base-v2 uses "<s>"/"</s>" framing and its ONNX export declares
	// no token_type_ids input; both quirks are handled generically (see the
	// tokenizer's configurable special tokens and the onnx session's declared-
	// input introspection).
	"all-mpnet-base-v2": {
		Name:      "all-mpnet-base-v2",
		HFRepo:    "sentence-transformers/all-mpnet-base-v2",
		OnnxURL:   "https://huggingface.co/sentence-transformers/all-mpnet-base-v2/resolve/main/onnx/model.onnx",
		OnnxSHA:   "74187b16d9c946fea252e120cfd7a12c5779d8b8b86838a2e4c56573c47941bd",
		VocabURL:  "https://huggingface.co/sentence-transformers/all-mpnet-base-v2/resolve/main/vocab.txt",
		VocabSHA:  "dbd90cb94e2247bd4d4ccaecbf616d2290e66691d7d5e5bb81f063c2d0649ada",
		Dim:       768,
		Pooling:   PoolingMean,
		LowerCase: true,
		License:   "Apache-2.0",
		ClsToken:  "<s>",
		SepToken:  "</s>",
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
