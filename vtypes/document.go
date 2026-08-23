// SPDX-License-Identifier: Apache-2.0

package vtypes

import "encoding/json"

// contentField is the reserved metadata key under which document content is
// carried inside a point's metadata map. It mirrors the engine's contentField.
const contentField = "$content"

// Document is a search result enriched with its stored content and (filterable)
// metadata — what a RAG caller actually wants back from a query.
type Document struct {
	ID       uint64   `json:"id"`
	Distance float32  `json:"distance"`
	Score    float32  `json:"score"` // fusion score for hybrid results; 0 for plain KNN
	Content  string   `json:"content"`
	Metadata Metadata `json:"metadata,omitempty"` // user metadata, with the reserved content field removed
}

// RawDocument is Document with its metadata left as the JSON bytes the result
// wire already carries, instead of a decoded map. It exists so a response whose
// only destination is JSON does not need the metadata decoded and re-encoded.
//
// IT IS NOT A SECOND RESULT SHAPE. Its JSON rendering is byte-identical to
// Document's for every value the wire can carry.
//
// LIFETIME: Metadata ALIASES the buffer it was decoded from — it is a window into
// the op result bytes, not a copy. A RawDocument must not outlive that buffer.
type RawDocument struct {
	ID       uint64          `json:"id"`
	Distance float32         `json:"distance"`
	Score    float32         `json:"score"` // fusion score for hybrid results; 0 for plain KNN
	Content  string          `json:"content"`
	Metadata json.RawMessage `json:"metadata,omitempty"` // verbatim wire bytes; aliases the source buffer
}

// RawGroup is Group with raw-metadata hits — the RawDocument counterpart of
// Group. Key is kept as raw JSON too: the group wire carries it as the
// json.Marshal of a Value, so re-emitting those bytes is what marshalling the
// Value would produce.
type RawGroup struct {
	Key  json.RawMessage `json:"key"`
	Hits []RawDocument   `json:"hits"`
}

// WithContent returns a copy of meta carrying document content in the reserved
// content field — for callers (e.g. the networked client) that must embed
// content into metadata before encoding an upsert. The reserved field is
// excluded from filtering and stripped from SearchDocs' returned Metadata. The
// caller's map is never mutated; an empty content returns meta unchanged.
func WithContent(meta Metadata, content string) Metadata {
	if content == "" {
		return meta
	}
	m := make(Metadata, len(meta)+1)
	for k, v := range meta {
		m[k] = v
	}
	m[contentField] = NewString(content)
	return m
}
