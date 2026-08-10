// SPDX-License-Identifier: Apache-2.0

package ops

import (
	"testing"

	"github.com/cespare/xxhash/v2"
)

// BenchmarkRouteKeyExtract measures the per-routed-op cost the cluster router pays
// before any work happens: extract the collection routing key from the args and
// hash it to a shard index. It compares the two routes cluster.Node.shardIndexFor
// can take — the stored KeyExtractor (still the path for a dynamically registered
// WASM op) against RouteKeyInto over a stack scratch (every built-in) — with the
// extractor reached exactly as production reaches it: the KeyExtractor through a
// registry-supplied FUNCTION VALUE, the Into form through a DIRECT call, which is
// what keeps buf on the stack.
//
// "bare" is the common case (a user names a collection "docs" and it canonicalizes
// to "default/docs"); "qualified" is the already-canonical name the fan-out
// coordinator emits, where the Into form returns a window into args.
func BenchmarkRouteKeyExtract(b *testing.B) {
	r := NewRegistry()
	if err := RegisterBuiltins(r); err != nil {
		b.Fatalf("RegisterBuiltins: %v", err)
	}
	_, ke, layout, ok := r.LookupRouting("vector_get")
	if !ok {
		b.Fatal("vector_get not registered")
	}
	for _, tc := range []struct {
		label string
		col   string
	}{
		{"bare", "docs"},
		{"qualified", "default/docs"},
	} {
		args := append([]byte{byte(len(tc.col))}, tc.col...)
		args = append(args, make([]byte, 8)...) // an id, as a real op carries
		b.Run(tc.label+"/KeyExtractor", func(b *testing.B) {
			b.ReportAllocs()
			var sink uint64
			for i := 0; i < b.N; i++ {
				key, found := ke(args)
				if !found {
					b.Fatal("no key")
				}
				sink += xxhash.Sum64(key)
			}
			_ = sink
		})
		b.Run(tc.label+"/RouteKeyInto", func(b *testing.B) {
			b.ReportAllocs()
			var sink uint64
			for i := 0; i < b.N; i++ {
				var buf [128]byte
				key := RouteKeyInto(layout, args, buf[:0])
				if key == nil {
					b.Fatal("no key")
				}
				sink += xxhash.Sum64(key)
			}
			_ = sink
		})
	}
}
