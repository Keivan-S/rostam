// SPDX-License-Identifier: Apache-2.0

package vector

import (
	"math"
	"math/rand"
	"slices"
	"testing"
)

func TestMinHeapPopsAscending(t *testing.T) {
	var h minHeap
	for _, d := range []float32{3, 1, 4, 1, 5, 9, 2, 6} {
		h.push(packKey(0, d))
	}
	var got []float32
	for h.len() > 0 {
		got = append(got, keyUnpack(h.pop()).dist)
	}
	want := []float32{1, 1, 2, 3, 4, 5, 6, 9}
	for i := range got {
		if got[i] != want[i] {
			t.Fatalf("min-heap pop[%d] = %v, want %v (full: %v)", i, got[i], want[i], got)
		}
	}
}

func TestMaxHeapPopsDescending(t *testing.T) {
	var h maxHeap
	for _, d := range []float32{3, 1, 4, 1, 5, 9, 2, 6} {
		h.push(packKey(0, d))
	}
	var got []float32
	for h.len() > 0 {
		got = append(got, keyUnpack(h.pop()).dist)
	}
	want := []float32{9, 6, 5, 4, 3, 2, 1, 1}
	for i := range got {
		if got[i] != want[i] {
			t.Fatalf("max-heap pop[%d] = %v, want %v (full: %v)", i, got[i], want[i], got)
		}
	}
}

func TestMaxHeapPeekIsTop(t *testing.T) {
	var h maxHeap
	h.push(packKey(1, 2.0))
	h.push(packKey(2, 5.0))
	h.push(packKey(3, 3.0))
	if got := keyUnpack(h.peek()).dist; got != 5.0 {
		t.Errorf("peek = %v, want 5.0", got)
	}
	if h.len() != 3 {
		t.Errorf("peek should not pop; len = %d, want 3", h.len())
	}
}

// TestPackKeyRoundTripsAndOrdersLikeFloat is the load-bearing property of the
// packed-key heaps: the uint64 key's UNSIGNED order must be exactly the
// (dist, slot) order of the pair it encodes, INCLUDING across negative
// distances (pickDist returns -dotProduct for the DotProduct metric, so
// negatives are routine and a naive bit-cast would sort them backwards).
func TestPackKeyRoundTripsAndOrdersLikeFloat(t *testing.T) {
	dists := []float32{
		float32(math.Inf(-1)), -1e30, -3.5, -1, -1e-30, math.Float32frombits(0x80000000), // -0
		0, 1e-30, 1, 3.5, 1e30, float32(math.Inf(1)),
	}
	for _, d := range dists {
		got := keyUnpack(packKey(7, d))
		if got.slot != 7 {
			t.Errorf("packKey(7, %v): slot round-trip = %d, want 7", d, got.slot)
		}
		if math.Float32bits(got.dist) != math.Float32bits(d) {
			t.Errorf("packKey(7, %v): dist round-trip = %v (bits %#x, want %#x)",
				d, got.dist, math.Float32bits(got.dist), math.Float32bits(d))
		}
	}
	// -0 and +0 compare equal as floats and map to distinct order bits; skip the
	// pair-ordering check for them and assert only the strict pairs.
	for i := range dists {
		for j := range dists {
			a, b := dists[i], dists[j]
			if a == b {
				continue
			}
			ka, kb := packKey(0, a), packKey(0, b)
			if (a < b) != (ka < kb) {
				t.Fatalf("order mismatch: (%v < %v) = %v but (key %#x < key %#x) = %v",
					a, b, a < b, ka, kb, ka < kb)
			}
		}
	}
	// Equal distances tie-break by ascending slot.
	if packKey(1, 2.5) >= packKey(2, 2.5) {
		t.Fatal("equal distances must tie-break by ascending slot")
	}
}

// TestOrderBitsSortsNaNLast pins the NaN sentinel. Under the plain total-order
// transform a NEGATIVE NaN maps to 0x003FFFFF — below -Inf, i.e. rank 1 — and
// that is reachable: under DotProduct pickDist returns -dot, so a +NaN dot
// product becomes a -NaN distance, and a NaN component can reach the arena via
// the gRPC ingress or a direct Go API call (arena.Insert checks only dimension).
// Both NaN signs must sort ABOVE every real distance so a garbage point can
// never displace a genuine result.
func TestOrderBitsSortsNaNLast(t *testing.T) {
	posNaN := math.Float32frombits(0x7FC00000)
	negNaN := math.Float32frombits(0xFFC00000)
	// A signalling NaN and a NaN whose mantissa has only its lowest bit set, to
	// confirm the test is on the exponent/mantissa shape and not one payload.
	sigNaN := math.Float32frombits(0x7F800001)
	negSigNaN := math.Float32frombits(0xFF800001)

	finite := []float32{
		float32(math.Inf(-1)), -1e30, -1, 0, 1, 1e30, float32(math.Inf(1)),
	}
	for _, nan := range []float32{posNaN, negNaN, sigNaN, negSigNaN} {
		if !math.IsNaN(float64(nan)) {
			t.Fatalf("fixture: %#x is not NaN", math.Float32bits(nan))
		}
		got := orderBits(nan)
		if got != nanOrder {
			t.Errorf("orderBits(NaN %#x) = %#x, want the sentinel %#x",
				math.Float32bits(nan), got, uint32(nanOrder))
		}
		for _, f := range finite {
			if orderBits(f) >= got {
				t.Errorf("orderBits(%v) = %#x must sort BELOW NaN %#x (got %#x)",
					f, orderBits(f), math.Float32bits(nan), got)
			}
		}
	}
	// -Inf must remain the smallest REAL distance; the old bug put -NaN under it.
	if orderBits(float32(math.Inf(-1))) != 0x007FFFFF {
		t.Errorf("orderBits(-Inf) = %#x, want 0x007fffff", orderBits(float32(math.Inf(-1))))
	}
	// A NaN key still reports as NaN when unpacked, rather than a plausible float.
	if d := keyUnpack(packKey(3, posNaN)).dist; !math.IsNaN(float64(d)) {
		t.Errorf("unpacked NaN distance = %v, want NaN", d)
	}
	if s := keyUnpack(packKey(3, posNaN)).slot; s != 3 {
		t.Errorf("unpacked NaN slot = %d, want 3", s)
	}
}

// TestNaNVectorCannotDisplaceResults is the end-to-end statement of the above:
// a stored vector containing NaN must not be returned ahead of genuine
// neighbors. Under DotProduct (where pickDist negates, turning a +NaN dot into
// a -NaN distance) the naive transform ranked such a point first for every
// query; the old float compare never did, and neither must this.
func TestNaNVectorCannotDisplaceResults(t *testing.T) {
	for _, metric := range []Metric{DotProduct, L2, Cosine} {
		t.Run(metricName(metric), func(t *testing.T) {
			const dim, n = 8, 200
			h, err := newHNSW(Config{Dim: dim, Metric: metric, M: 8, EfConstruction: 64, EfSearch: 32, Seed: 1})
			if err != nil {
				t.Fatal(err)
			}
			defer h.Close()

			// The poisoned point goes in FIRST, which makes it the graph's entry
			// point. That matters: searchLayerCore seeds entry points into the
			// result heap unconditionally, so this guarantees the NaN distance is
			// actually computed and ranked. Inserted last it tends to be
			// unreachable (its own links are garbage), and the test would pass
			// while proving nothing — which is exactly what it did at first.
			const nanID = 9999
			bad := make([]float32, dim)
			for j := range bad {
				bad[j] = float32(math.NaN())
			}
			if _, _, err := h.Insert(nanID, bad, 0, nil, nil, nil, CASCond{}); err != nil {
				t.Fatal(err)
			}

			rng := rand.New(rand.NewSource(21))
			var query []float32
			for i := 1; i <= n; i++ {
				v := make([]float32, dim)
				for j := range v {
					v[j] = float32(rng.NormFloat64())
				}
				if query == nil {
					query = append([]float32(nil), v...)
				}
				if _, _, err := h.Insert(uint64(i), v, 0, nil, nil, nil, CASCond{}); err != nil {
					t.Fatal(err)
				}
			}

			res, err := h.Search(query, 10)
			if err != nil {
				t.Fatal(err)
			}
			if len(res) == 0 {
				t.Fatal("no results")
			}
			for rank, r := range res {
				if r.ID == nanID {
					t.Fatalf("NaN-valued point ranked #%d of %d (distance %v): a point whose "+
						"distance is not a number must never outrank a real neighbor",
						rank+1, len(res), r.Distance)
				}
			}
		})
	}
}

// metricName gives a readable subtest name for a Metric.
func metricName(m Metric) string {
	switch m {
	case L2:
		return "L2"
	case Cosine:
		return "Cosine"
	case DotProduct:
		return "DotProduct"
	}
	return "unknown"
}

// TestReplaceTopEqualsPushThenPop pins the fused bounded-k operation used by
// searchLayerCore: when the incoming key is strictly closer than the root,
// replaceTop must leave EXACTLY the heap that push-then-pop would.
func TestReplaceTopEqualsPushThenPop(t *testing.T) {
	rng := rand.New(rand.NewSource(3))
	for trial := 0; trial < 500; trial++ {
		n := 1 + rng.Intn(32)
		var base maxHeap
		for i := 0; i < n; i++ {
			base.push(packKey(uint32(rng.Intn(1000)), float32(rng.NormFloat64())))
		}
		// An incoming key strictly closer (smaller) than the root.
		root := base.peek()
		if keyDist(root) == 0 {
			continue
		}
		key := root - 1 - uint64(rng.Intn(1000))

		pushPop := append(maxHeap(nil), base...)
		pushPop.push(key)
		pushPop.pop()

		replaced := append(maxHeap(nil), base...)
		replaced.replaceTop(key)

		gotA, gotB := drainMax(pushPop), drainMax(replaced)
		if !slices.Equal(gotA, gotB) {
			t.Fatalf("trial %d: replaceTop != push+pop\n push+pop: %v\n replace : %v", trial, gotA, gotB)
		}
	}
}

// drainMax pops a copy of h into descending order.
func drainMax(h maxHeap) []uint64 {
	out := make([]uint64, 0, h.len())
	for h.len() > 0 {
		out = append(out, h.pop())
	}
	return out
}

// TestHeapsMatchSortedOrderRandomized fuzzes both heaps against a plain sort,
// which is also the invariant the searchLayerCore result drain relies on
// (slices.Sort over the raw heap array yields ascending (dist, slot)).
func TestHeapsMatchSortedOrderRandomized(t *testing.T) {
	rng := rand.New(rand.NewSource(11))
	for trial := 0; trial < 300; trial++ {
		n := rng.Intn(64)
		keys := make([]uint64, 0, n)
		var mn minHeap
		var mx maxHeap
		for i := 0; i < n; i++ {
			k := packKey(uint32(rng.Intn(500)), float32(rng.NormFloat64()))
			keys = append(keys, k)
			mn.push(k)
			mx.push(k)
		}
		slices.Sort(keys)

		for i := 0; i < n; i++ {
			if got := mn.pop(); got != keys[i] {
				t.Fatalf("trial %d: minHeap pop[%d] = %#x, want %#x", trial, i, got, keys[i])
			}
			if got := mx.pop(); got != keys[n-1-i] {
				t.Fatalf("trial %d: maxHeap pop[%d] = %#x, want %#x", trial, i, got, keys[n-1-i])
			}
		}
	}
}
