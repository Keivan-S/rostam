// SPDX-License-Identifier: Apache-2.0

package ops

import (
	"encoding/json"
	"fmt"
	"io"
	"math/rand"
	"testing"

	"github.com/rostamlabs/rostam/vector"
)

// WHERE THE JSON ACTUALLY COSTS ON THE READ PATH.
//
// These benchmarks exist because the obvious guess is wrong. The payload blob
// decode on bulk ingest LOOKS like the hot JSON site — a 1M-point filter case
// really is 1M json.Unmarshal calls through DecodeBulkStagePayloadArgs — but a
// CPU profile of the whole staged ingest puts encoding/json at 0.37% of samples
// (0.26% Unmarshal + 0.11% Marshal) against 98% in hnsw.BuildConcurrentMeta.
// The graph link phase is so much more expensive per point that the JSON beside
// it is unmeasurable. Anything that makes the ingest JSON faster is invisible.
//
// The read path is the opposite. A search_docs response marshals each hit's
// metadata (EncodeVectorDocs), unmarshals each one straight back
// (decodeVectorDocsN), and then marshals the whole response again in
// writeJSON — three passes over the same payloads, against a search that got
// cheaper every time the index work was optimized. BenchmarkDocsJSONCost
// reports the three passes separately so the split stays visible.
//
// BenchmarkDocsResponseShape is the one that matters for planning. Its
// "elide-round-trip" arm is not a proposed API: it measures what the request
// would cost if the encode/decode pair were skipped when producer and consumer
// are the same process. The op registry's contract is
// Call(name string, args []byte) ([]byte, error), so single-node, direct and
// embedded deployments pay a full marshal + unmarshal round-trip through a
// serialization boundary that has no network on the far side. On a k=100 search
// over rich payloads that round-trip is roughly half the request.
//
// "raw-passthrough" is what SHIPPED, and the distinction is worth keeping
// straight. Eliding needs the producer and consumer to be the same process,
// which means a local path and a remote path that compute the same answer two
// ways — the divergence risk that got a local bypass rejected on the batch-get
// path. The raw passthrough instead keeps ONE path for both: the wire marshal
// stays (a remote peer really does need those bytes), and the response is built
// by splicing each hit's metadata straight out of the result body instead of
// decoding it in order to re-encode it. It therefore captures most of the elide
// arm's win, captures it on the REMOTE path too where eliding is impossible by
// definition, and never introduces a second answer. Measured medians of 11
// rounds, pinned (taskset -c 0-7, -cpu 1), against wire-round-trip: scalar/k10
// 2.10x, scalar/k100 2.29x, richdoc/k10 2.02x, richdoc/k100 2.14x.
//
// Run:
//
//	go test ./ops -run '^$' -bench 'DocsJSONCost|DocsResponseShape' -benchtime 2s

// benchDocsCorpus builds k documents carrying either the one-scalar payload the
// VectorDBBench filter case uses or a realistic multi-field user document.
func benchDocsCorpus(k int, rich bool) []vector.Document {
	r := rand.New(rand.NewSource(7)) //nolint:gosec // deterministic fixture
	docs := make([]vector.Document, k)
	for i := range docs {
		var m vector.Metadata
		if rich {
			m = vector.Metadata{
				"id":       vector.NewInt(int64(i)),
				"title":    vector.NewString(fmt.Sprintf("Document number %d about distributed systems", i)),
				"author":   vector.NewString("A. N. Author"),
				"lang":     vector.NewString("en"),
				"score":    vector.NewFloat(r.Float64()),
				"public":   vector.NewBool(i%2 == 0),
				"tags":     vector.NewStrings([]string{"alpha", "beta", "gamma"}),
				"ts":       vector.NewInt(int64(1700000000 + i)),
				"category": vector.NewString("engineering"),
			}
		} else {
			m = vector.Metadata{"id": vector.NewInt(int64(i))}
		}
		docs[i] = vector.Document{ID: uint64(i), Distance: r.Float32(), Score: r.Float32(), Metadata: m}
	}
	return docs
}

// BenchmarkDocsJSONCost separates the three JSON passes a search_docs response
// makes over the same payloads. ns/doc is the comparable figure across k.
func BenchmarkDocsJSONCost(b *testing.B) {
	for _, rich := range []bool{false, true} {
		shape := "scalar"
		if rich {
			shape = "richdoc"
		}
		for _, k := range []int{10, 100} {
			docs := benchDocsCorpus(k, rich)
			name := fmt.Sprintf("%s/k%d", shape, k)

			// pass 1: EncodeVectorDocs' per-doc metadata marshal (shard side)
			b.Run(name+"/encode-docs-wire", func(b *testing.B) {
				b.ReportAllocs()
				for i := 0; i < b.N; i++ {
					_ = EncodeVectorDocs(docs)
				}
				b.ReportMetric(float64(b.Elapsed().Nanoseconds())/float64(b.N*k), "ns/doc")
			})

			// pass 2: decodeVectorDocsN's per-doc metadata unmarshal (caller side)
			body := EncodeVectorDocs(docs)
			b.Run(name+"/decode-docs-wire", func(b *testing.B) {
				b.ReportAllocs()
				for i := 0; i < b.N; i++ {
					if _, err := DecodeVectorDocs(body); err != nil {
						b.Fatal(err)
					}
				}
				b.ReportMetric(float64(b.Elapsed().Nanoseconds())/float64(b.N*k), "ns/doc")
			})

			// pass 3: writeJSON's single whole-response encode (HTTP side)
			resp := map[string]any{"documents": docs}
			b.Run(name+"/encode-http-response", func(b *testing.B) {
				b.ReportAllocs()
				for i := 0; i < b.N; i++ {
					if err := json.NewEncoder(io.Discard).Encode(resp); err != nil {
						b.Fatal(err)
					}
				}
				b.ReportMetric(float64(b.Elapsed().Nanoseconds())/float64(b.N*k), "ns/doc")
			})
		}
	}
}

// BenchmarkDocsResponseShape contrasts the whole response-production sequence
// as it runs today against the same sequence with the in-process wire
// round-trip removed. The gap is what a single-node deployment currently pays
// to serialize a result to itself.
func BenchmarkDocsResponseShape(b *testing.B) {
	for _, rich := range []bool{false, true} {
		shape := "scalar"
		if rich {
			shape = "richdoc"
		}
		for _, k := range []int{10, 100} {
			docs := benchDocsCorpus(k, rich)
			name := fmt.Sprintf("%s/k%d", shape, k)

			b.Run(name+"/wire-round-trip", func(b *testing.B) {
				b.ReportAllocs()
				for i := 0; i < b.N; i++ {
					out, err := DecodeVectorDocs(EncodeVectorDocs(docs))
					if err != nil {
						b.Fatal(err)
					}
					if err := json.NewEncoder(io.Discard).Encode(map[string]any{"documents": out}); err != nil {
						b.Fatal(err)
					}
				}
				b.ReportMetric(float64(b.Elapsed().Nanoseconds())/float64(b.N*k), "ns/doc")
			})

			b.Run(name+"/elide-round-trip", func(b *testing.B) {
				b.ReportAllocs()
				for i := 0; i < b.N; i++ {
					if err := json.NewEncoder(io.Discard).Encode(map[string]any{"documents": docs}); err != nil {
						b.Fatal(err)
					}
				}
				b.ReportMetric(float64(b.Elapsed().Nanoseconds())/float64(b.N*k), "ns/doc")
			})

			// The arm that SHIPPED. "elide-round-trip" is an upper bound nobody can
			// reach without splitting the local and remote paths in two; this one
			// keeps ONE path by never decoding the metadata it is about to re-encode
			// — the wire marshal stays (a remote peer still needs those bytes), the
			// unmarshal disappears, and the response encode copies the metadata
			// verbatim instead of walking a map. It is what
			// ops.DecodeVectorDocsRaw + writeJSON actually do, so the number here is
			// the number a search_docs request gets.
			b.Run(name+"/raw-passthrough", func(b *testing.B) {
				b.ReportAllocs()
				for i := 0; i < b.N; i++ {
					out, err := DecodeVectorDocsRaw(EncodeVectorDocs(docs))
					if err != nil {
						b.Fatal(err)
					}
					if err := json.NewEncoder(io.Discard).Encode(map[string]any{"documents": out}); err != nil {
						b.Fatal(err)
					}
				}
				b.ReportMetric(float64(b.Elapsed().Nanoseconds())/float64(b.N*k), "ns/doc")
			})
		}
	}
}
