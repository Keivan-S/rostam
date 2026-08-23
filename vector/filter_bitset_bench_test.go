// SPDX-License-Identifier: Apache-2.0

package vector

import (
	"fmt"
	"math/rand"
	"testing"
)

// gateBenchIndex builds a filtered-search benchmark corpus where `hitFrac` of
// the points carry bucket="hit". FilterFirstThreshold is pinned to 1 so the
// planner always declines filter-first and the GRAPH path — the only path the
// admission gate lives on — is what gets measured. That matches
// BenchmarkSearchFilteredParallel's existing setup.
func gateBenchIndex(b *testing.B, n, dim int, hitFrac float64) *hnsw {
	b.Helper()
	corpus := makeCorpus(n, dim, 42)
	h, err := newHNSW(Config{Dim: dim, M: 16, EfConstruction: 200, EfSearch: 64, Seed: 1, Metric: L2,
		FilterFirstThreshold: 1})
	if err != nil {
		b.Fatalf("newHNSW: %v", err)
	}
	rng := rand.New(rand.NewSource(7))
	for i, v := range corpus {
		bucket := "miss"
		if rng.Float64() < hitFrac {
			bucket = "hit"
		}
		if _, _, err := h.Insert(uint64(i+1), v, 0, Metadata{"bucket": NewString(bucket)}, nil, nil, CASCond{}); err != nil {
			b.Fatalf("insert: %v", err)
		}
	}
	return h
}

// BenchmarkAdmitGateAB is the A/B the gate exists to win: the same index, the
// same filter, the same queries, with the bitset gate off and on.
//
// The selectivity sweep is the point, not a formality. The gate's build cost
// scales with the posting set (selectivity x N) while its saving scales with the
// traversal (ef x 2M, independent of N), so it wins big at high selectivity,
// wins less as the filter loosens, and must LOSE at some point — where the
// "auto" arm is expected to decline and land on the off arm's number. An A/B
// that only measured the favourable selectivity would be measuring the choice of
// benchmark, not the feature.
//
// Arms: off = today's path; auto = production (gateProfitable decides); force =
// the gate armed regardless, which is what shows where the cost model SHOULD say
// no.
func BenchmarkAdmitGateAB(b *testing.B) {
	const dim, n = 128, 10_000
	queries := makeCorpus(1_000, dim, 99)
	filter := Filter{Op: FilterEq, Field: "bucket", Value: NewString("hit")}

	for _, hit := range []float64{0.5, 0.1, 0.01} {
		h := gateBenchIndex(b, n, dim, hit)
		for _, arm := range []struct {
			name string
			mode gateMode
		}{{"off", gateOff}, {"auto", gateAuto}, {"force", gateForce}} {
			b.Run(fmt.Sprintf("hit=%g/%s", hit, arm.name), func(b *testing.B) {
				prev := admitGateMode
				admitGateMode = arm.mode
				defer func() { admitGateMode = prev }()
				dst := make([]Result, 0, 10)
				b.ReportAllocs()
				b.ResetTimer()
				for i := 0; i < b.N; i++ {
					dst, _ = h.SearchInto(dst[:0], queries[i%len(queries)], 10, filter)
				}
			})
		}
	}
}

// BenchmarkAdmitGateABParallel is BenchmarkAdmitGateAB from every core at once,
// mirroring BenchmarkSearchFilteredParallel. The serial form cannot see the
// effect that matters most at scale: the gate replaces a metadata map lookup
// (which pulls a random cache line per candidate) with a bit test over a
// 1-bit-per-slot array, so the win should grow, not shrink, under core pressure.
func BenchmarkAdmitGateABParallel(b *testing.B) {
	const dim, n = 128, 10_000
	queries := makeCorpus(1_000, dim, 99)
	filter := Filter{Op: FilterEq, Field: "bucket", Value: NewString("hit")}

	for _, hit := range []float64{0.1, 0.01} {
		h := gateBenchIndex(b, n, dim, hit)
		for _, arm := range []struct {
			name string
			mode gateMode
		}{{"off", gateOff}, {"auto", gateAuto}} {
			b.Run(fmt.Sprintf("hit=%g/%s", hit, arm.name), func(b *testing.B) {
				prev := admitGateMode
				admitGateMode = arm.mode
				defer func() { admitGateMode = prev }()
				b.ResetTimer()
				b.RunParallel(func(pb *testing.PB) {
					i := 0
					dst := make([]Result, 0, 10)
					for pb.Next() {
						dst, _ = h.SearchInto(dst[:0], queries[i%len(queries)], 10, filter)
						i++
					}
				})
			})
		}
	}
}

// BenchmarkAdmitGateUnits measures the three quantities gateProfitable's cost
// constants encode, so the constants are calibrated rather than guessed. Run it
// and compare: gateWordClearDNS/gatePostingDNS/gateVisitSaveDNS should be within
// a small factor of ns/op x 10 for wordclear/posting/predicate respectively.
//
//   - wordclear: zeroing one bitset word (gateWordClearDNS).
//   - posting:   visiting one posting entry while filling the bitset — a map
//     iteration step plus the bit write (gatePostingDNS).
//   - predicate: what the gate replaces on a rejected candidate — an Eq
//     predicate over a metadata map (gateVisitSaveDNS).
func BenchmarkAdmitGateUnits(b *testing.B) {
	const words = 1 << 14 // 16384 words = 1M slots
	buf := make([]uint64, words)
	b.Run("wordclear", func(b *testing.B) {
		b.ResetTimer()
		for i := 0; i < b.N; i++ {
			clear(buf)
		}
		b.ReportMetric(float64(b.Elapsed().Nanoseconds())/float64(b.N*words)*10, "dns/word")
	})

	const postings = 100_000
	set := make(map[uint32]struct{}, postings)
	rng := rand.New(rand.NewSource(3))
	for len(set) < postings {
		set[uint32(rng.Intn(1<<20))] = struct{}{} //nolint:gosec // bounded by the shift
	}
	b.Run("posting", func(b *testing.B) {
		var g admitGate
		plan := []narrowSet{{set, narrowExact}}
		b.ResetTimer()
		for i := 0; i < b.N; i++ {
			g.build(plan, 1<<20, true)
		}
		b.ReportMetric(float64(b.Elapsed().Nanoseconds())/float64(b.N*postings)*10, "dns/posting")
	})

	// rangekey: the per-DISTINCT-KEY cost of unioning a range out of the payload
	// index — a scalarKey hash probe into the field's key map plus a fresh map
	// iterator over that key's posting set. It is the term the m4 model did not
	// have, and its absence is why a 99%-pass range over a unique-valued field
	// looked affordable when it was not (gateRangeKeyDNS).
	//
	// Measured with ONE posting per key so the per-posting term contributes as
	// little as possible: the reported number is per key, per-posting cost
	// included, which is what the model wants (a range's postings are visited
	// through its keys, never independently).
	b.Run("rangekey", func(b *testing.B) {
		const keys = 50_000
		p := newPayloadIndex()
		for i := 0; i < keys; i++ {
			p.reindex(uint32(i), Metadata{"v": NewInt(int64(i))}) //nolint:gosec // bounded loop index
		}
		b.ResetTimer()
		for i := 0; i < b.N; i++ {
			if _, ok := p.orderingSetCapped("v", FilterGte, NewInt(0), keys+1, keys+1); !ok {
				b.Fatal("orderingSet declined")
			}
		}
		b.ReportMetric(float64(b.Elapsed().Nanoseconds())/float64(b.N*keys)*10, "dns/key")
	})

	// unionposting: the per-POSTING cost of the same union when the keys are FEW
	// and fat — one map insert into a growing result set, without the per-key
	// probe and iterator setup the arm above measures. It is the other half of the
	// complement gate's build cost, and it is NOT the same quantity as
	// gatePostingDNS: that one is a BIT WRITE into an already-materialized plan,
	// this one is a map insert (gateUnionPostingDns).
	b.Run("unionposting", func(b *testing.B) {
		const keys, per = 50, 1_000
		p := newPayloadIndex()
		for i := 0; i < keys*per; i++ {
			p.reindex(uint32(i), Metadata{"v": NewInt(int64(i % keys))}) //nolint:gosec // bounded loop index
		}
		b.ResetTimer()
		for i := 0; i < b.N; i++ {
			if _, ok := p.orderingSetCapped("v", FilterGte, NewInt(0), keys*per+1, keys+1); !ok {
				b.Fatal("orderingSet declined")
			}
		}
		b.ReportMetric(float64(b.Elapsed().Nanoseconds())/float64(b.N*keys*per)*10, "dns/posting")
	})

	b.Run("predicate", func(b *testing.B) {
		pred, err := CompileFilter(Filter{Op: FilterEq, Field: "bucket", Value: NewString("hit")})
		if err != nil {
			b.Fatal(err)
		}
		metas := make([]Metadata, 256)
		for i := range metas {
			v := "miss"
			if i%10 == 0 {
				v = "hit"
			}
			metas[i] = Metadata{"bucket": NewString(v), "other": NewInt(int64(i))}
		}
		var sink bool
		b.ResetTimer()
		for i := 0; i < b.N; i++ {
			sink = pred(metas[i&255])
		}
		_ = sink
		b.ReportMetric(float64(b.Elapsed().Nanoseconds())/float64(b.N)*10, "dns/eval")
	})
}
